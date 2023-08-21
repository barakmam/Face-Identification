import collections

import cv2
import easydict as easydict
from tqdm import tqdm
import torch
import torchvision.transforms
from PIL import Image
import numpy as np
import os
import pickle
from glob import glob
from os.path import join
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch._six import string_classes
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from torch.utils.data.sampler import WeightedRandomSampler
from hydra.utils import get_original_cwd
import random
from sklearn.preprocessing import normalize
from mycode.utils.misc import read_cfg
from mycode.data.multi_epoch_dataloader import MultiEpochsDataLoader
from insightFace.recognition.arcface_torch.dataset import MXFaceDataset
from mycode.data.transform import SimilarityTrans


class BaseYTF(Dataset):
    def __init__(self, cfg, transform, stage, mode, id2label_dict):
        """
        This class for the YTF data that is already in HR
        (not necessarily sharp GT videos, just in image size of the face embedder input).
        """
        super().__init__()
        self.data_splits_path = '/inputs/bionicEye/data'
        self.cfg = cfg
        self.stage = stage
        self.videos_path = os.path.join(cfg.data.path, 'videos')
        self.transform = transform
        self.seed = cfg.system.seed
        self.hr_res = cfg.model.fr.hr_res
        self.sharp_videos_path = self.videos_path.replace(self.videos_path.split('/')[-1], 'sharp')
        with open(os.path.join(self.data_splits_path, cfg.data.name, 'splits', stage, mode + '.pickle'), 'rb') as handle:
            self.dataset = pickle.load(handle)
        self.targets = np.array([id2label_dict[name.split('/')[-1][:-2]] for name in self.dataset])

    def __str__(self):
        return 'ytf'

    def __len__(self):
        return len(self.dataset)


class YoutubeFacesAUX(Dataset):
    def __init__(self, cfg, stage, mode, transform):
        super().__init__()
        self.data_splits_path = '/inputs/bionicEye/data'
        if not os.path.exists(join(self.data_splits_path, str(self))):
            print('THERE ARE NO SPLITS IN: ', join(self.data_splits_path, str(self)))
        with open(os.path.join(self.data_splits_path, cfg.data.name, 'id2label_dict.pickle'), 'rb') as handle:
            self.id2label_dict = pickle.load(handle)

        if stage == 'train' or stage is None:
            self.triplets = YoutubeFacesTriplets(cfg, transform, 'train', 'train', self.id2label_dict)
            return
        if (stage == 'test' or stage is None) and mode == 'hr_input':
            self.data = FaceImage(cfg, transform, 'all_data', 'all_data', self.id2label_dict)
        if (stage == 'test' or stage is None) and mode == 'lr_input':
            self.data = LRYTF(cfg, transform, stage, 'all', self.id2label_dict)

    def __str__(self):
        return 'ytf'


class FaceImage(BaseYTF):
    def __init__(self, cfg, transform, stage, mode, id2label_dict):
        """
            This class for the YTF data that is already cropped face
        """
        super().__init__(cfg, transform, stage, mode, id2label_dict)
        frames_dataset = []
        frames_targets = []
        frames_landmarks = []
        if 'similarity_trans' in cfg.data.augmentations.test.types:
            with open(os.path.join(self.cfg.data.path, 'detection.pickle'), 'rb') as h:
                stats = pickle.load(h)
                landmarks = stats['landmarks']
        for vid_name, vid_id in zip(self.dataset, self.targets):
            frames = glob(os.path.join(self.videos_path, vid_name, '*'))
            frames_dataset.extend(frames)
            frames_targets.extend([vid_id]*len(frames))
            if 'similarity_trans' in cfg.data.augmentations.test.types:
                frames_landmarks.append(landmarks[vid_name])

        self.dataset = np.array(frames_dataset)
        self.targets = np.array(frames_targets)

        if 'similarity_trans' in cfg.data.augmentations.test.types:
            self.landmarks = np.concatenate(frames_landmarks)
            self.sim_trans = SimilarityTrans()

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.dataset[idx]), cv2.COLOR_BGR2RGB)
        if self.transform:
            if 'similarity_trans' in self.cfg.data.augmentations.test.types:
                image = self.sim_trans(image, self.landmarks[idx])
            image = self.transform(image)
        assert image.shape[1] == image.shape[2] == self.hr_res, f"frame {self.dataset[idx]} is not of size 112x112"
        return image, self.targets[idx], idx

    def __str__(self):
        return 'image_ytf'


class LRYTF(BaseYTF):
    def __init__(self, cfg, transform, stage, mode, id2label_dict):
        super().__init__(cfg, transform, stage, mode, id2label_dict)
        self.n_frames_sr = cfg.sr.num_input_frames

    def __getitem__(self, idx):
        """
        insert to the SR transformer several frames neighborhoods. each neighborhood is SR into one frame.
        these SR frames are inserted to the face recognition transformer
        """

        filename = []
        frames_name = np.array(os.listdir(join(self.videos_path, self.dataset[idx])))
        lr_frames = []
        target_frames = []
        for ii in range(self.n_frames_fr + self.n_frames_sr - 1):
            # if the fr transformer inputs is 40 frames,
            # then the sr transformer that it's input is 5 frames for each target (including the target frame)
            # should receive 40 + 4 (44 frames of neighbors and targets in LR)
            image = Image.open(join(self.videos_path, self.dataset[idx], frames_name[ii]))
            if self.transform:
                image = self.transform(image)
            assert image.shape[1] == image.shape[2] == self.hr_res // 4, \
                f"frame {join(self.videos_path, self.dataset[idx], frames_name[ii])} is not of size {self.hr_res // 4}"
            lr_frames.append(image)
            if ii < self.n_frames_fr + self.n_frames_sr - 1 - 2:
                target_frame = Image.open(join(self.sharp_videos_path, self.dataset[idx], frames_name[ii + 2]))
                if self.transform:
                    target_frame = self.transform(target_frame)
                target_frames.append(target_frame)
                filename.append(join(self.sharp_videos_path, self.dataset[idx], frames_name[ii + 2]))

        lr_frames = torch.stack(lr_frames)
        target_frames = torch.stack(target_frames)
        label = self.targets[idx]
        return lr_frames, target_frames, label, filename

    def __str__(self):
        return 'lr_ytf'


class YoutubeFacesTriplets(BaseYTF):
    def __init__(self, cfg, transform, stage, mode, id2label_dict):
        super().__init__(cfg, transform, stage, mode, id2label_dict)

        self.n_triplets = cfg.data.n_triplets
        self.sr_scale = cfg.sr.sr_scale
        self.n_frames_sr = cfg.sr.num_input_frames

        self.train_triplets = self.generate_triplets(self.targets, self.n_triplets)

    @staticmethod
    def generate_triplets(labels, n_triplets):
        print('Creating Triplets: ')
        triplets = []
        for x in tqdm(range(n_triplets)):
            idx = np.random.randint(0, len(labels))
            idx_matches = np.where(labels == labels[idx])[0]
            idx_no_matches = np.where(labels != labels[idx])[0]
            idx_a, idx_p = np.random.choice(idx_matches, 2, replace=False)
            idx_n = np.random.choice(idx_no_matches, 1).item()
            if [idx_a, idx_p, idx_n] not in triplets:
                triplets.append([idx_a, idx_p, idx_n])
        return np.array(triplets)

    @staticmethod
    def generate_hard_triplets(feats, labels, n_triplets):
        """
        sample negative by the similarity value.
        i.e. negative with high similarity will be sampled in probability p=similarity/sum(all similarities)
        """
        print('Creating Hard Triplets: ')
        triplets = []
        feats = normalize(feats.mean(1))
        logits = np.einsum('ae,be->ab', feats, feats)
        logits_exp = np.exp(logits).astype('float64')
        for x in tqdm(range(n_triplets)):
            idx = np.random.randint(0, len(labels))
            idx_matches = np.where(labels == labels[idx])[0]
            idx_a, idx_p = np.random.choice(idx_matches, 2, replace=False)
            idx_no_matches = np.where(labels != labels[idx])[0]
            logits_exp_no_matches = logits_exp[idx_a, idx_no_matches]
            proba_no_matches = logits_exp[idx_a, idx_no_matches] / logits_exp_no_matches.sum()
            idx_n = np.random.choice(idx_no_matches, 1, p=proba_no_matches).item()
            if [idx_a, idx_p, idx_n] not in triplets:
                triplets.append([idx_a, idx_p, idx_n])
        return np.array(triplets)

    def __getitem__(self, idx):
        t = self.train_triplets[idx]
        a, p, n = self.dataset[t[0]], self.dataset[t[1]], self.dataset[t[2]]

        def get_frames(self, video_name):
            frames_name = np.array(os.listdir(join(self.videos_path, video_name)))
            start_frame = torch.randint(len(frames_name) - self.n_frames_fr, (1,)).item()
            gt_frame = Image.open(join(self.sharp_videos_path, video_name, frames_name[start_frame + 2]))
            if self.transform:
                gt_frame = self.transform(gt_frame)

            frames = []
            filename = []
            for frame_num in frames_name[start_frame:(start_frame + self.n_frames_sr)]:
                filename.append(join(self.videos_path, video_name, frame_num))
                image = Image.open(filename[-1])
                if self.transform:
                    image = self.transform(image)
                assert image.shape[1] == image.shape[2] == self.hr_res // self.sr_scale, \
                    f"frame {frame_num} is not of size {self.hr_res // self.sr_scale}"
                frames.append(image)
            frames = torch.stack(frames)
            return frames, gt_frame, filename, start_frame

        frames_a, gt_a, name_a, start_a = get_frames(self, a)
        frames_p, gt_p, name_p, start_p = get_frames(self, p)
        frames_n, gt_n, name_n, start_n = get_frames(self, n)
        gt_frames = torch.stack((gt_a, gt_p, gt_n))

        return torch.stack((frames_a, frames_p, frames_n)), gt_frames, \
               [name_a, name_p, name_n], torch.tensor([start_a, start_p, start_n])

    def __str__(self):
        return 'ytf'

    def __len__(self):
        return len(self.train_triplets)


class YTFForDBVSRCode(Dataset):
    def __init__(self):
        super().__init__()
        self.data_splits_path = '/inputs/bionicEye/data'
        self.videos_path = '/datasets/BionicEye/YouTubeFaces/faces'
        self.n_frames = 5
        self.transform = torchvision.transforms.ToTensor()
        self.q = 112
        self.sr_scale = 4

        with open(os.path.join(self.data_splits_path, str(self), 'id2label_dict.pickle'), 'rb') as handle:
            id2label_dict = pickle.load(handle)
        with open(os.path.join(self.data_splits_path, str(self), 'train', 'train' + '.pickle'), 'rb') as handle:
            self.dataset = pickle.load(handle)
        self.targets = np.array([id2label_dict[name.split('/')[-1][:-2]] for name in self.dataset])

    def __getitem__(self, idx):
        # Take care with sampler of train and test

        def get_frames(self, video_name):
            # Take care with sampler of train and test
            frames = []
            filename = []
            frames_name = np.array(os.listdir(join(self.videos_path, video_name)))
            start_frame = np.random.randint(len(frames_name) - self.n_frames)
            for frame_num in frames_name[start_frame:(start_frame + self.n_frames)]:
                filename.append(join(self.videos_path, video_name, frame_num))
                image = Image.open(filename[-1])
                if self.transform:
                    image = self.transform(image)
                if 'blurdown' in filename[-1]:
                    assert image.shape[1] == image.shape[
                        2] == self.hr_res // self.sr_scale, f"frame {frame_num} is not of size {self.hr_res // self.sr_scale}"
                frames.append(image)
            frames = torch.stack(frames)
            return frames, filename

        lr_frames, filename = get_frames(self, 'blurdown_x4/' + self.dataset[idx])
        gt, _ = get_frames(self, 'sharp/' + self.dataset[idx])
        return lr_frames, gt, filename

    def __str__(self):
        return 'ytf'

    def __len__(self):
        return len(self.dataset)


class MS1M(MXFaceDataset):
    def __init__(self, cfg, transform):
        super().__init__(cfg.data.path, local_rank=0, transform=transform)
        with open(os.path.join(cfg.data.path, 'train.lst')) as fp:
            self.dataset = fp.readlines()
        self.dataset = np.array([f.split('\t')[1] for f in self.dataset])

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, label, idx


def collate_videos_varying_length(batch):

    outputs = []
    out = None
    tensors_batch = [elem[0] for elem in batch]
    tensor_elem = tensors_batch[0]
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in tensors_batch)
        storage = tensor_elem.storage()._new_shared(numel)
        out = tensor_elem.new(storage).resize_(len(tensors_batch), *list(tensor_elem.size()))
    outputs.append(torch.concat(tensors_batch, 0, out=out))

    outputs.append(torch.as_tensor([elem[1] for elem in batch]))  # labels of the videos (numpy)
    outputs.append(np.concatenate([elem[2] for elem in batch]))  # filenames of the videos (list of lists of strings)
    return outputs


class BaseDM(pl.LightningDataModule):

    def __init__(self, cfg, transform_dict):
        super().__init__()
        self.cfg = cfg
        self.num_workers = cfg.data.num_workers
        self.batch_size = cfg.data.batch_size
        self.transform_dict = transform_dict
        self.train = None
        self.test = None
        self.g = torch.Generator().manual_seed(0)
        self.data_name = cfg.data.name

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def train_dataloader(self):
        return MultiEpochsDataLoader(self.train.triplets, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          drop_last=True)
                          # , worker_init_fn=self.seed_worker, generator=self.g)

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        return MultiEpochsDataLoader(self.test.data, batch_size=self.batch_size, num_workers=self.num_workers)

    def make_weights_for_balanced_classes(self):
        _, count_cls = np.unique(self.train.targets, return_counts=True)
        weight_per_class = sum(count_cls)/count_cls
        weight = [0] * len(self.train.targets)
        for idx, val in enumerate(self.train.targets):
            weight[idx] = weight_per_class[val]
        return weight

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass


class ImageDM(BaseDM):
    """
    Select the train data and test data
    """
    def __init__(self, cfg, transform_dict={'train': None, 'test': None}):
        super().__init__(cfg, transform_dict)

    def setup(self, stage=None):
        if stage == 'train' or stage is None:
            # input to the SR transformer and afterward to the face embedder
            self.train = YoutubeFacesAUX(self.cfg, 'train', 'train', self.transform_dict['train'])
            print(f"train num triplets: {len(self.train.triplets)}:")

        # Assign test dataset for use in dataloader
        if stage == 'test' or stage is None:
            if self.data_name == 'ytf':
                if self.train is None:
                    # input to the face embedder
                    self.test = YoutubeFacesAUX(self.cfg, 'test', mode='hr_input', transform=self.transform_dict['test'])
                    print('\nCalculating embeddings for videos in: ', self.cfg.data.path)
                else:
                    # input to the SR transformer and afterward to the face embedder
                    self.test = YoutubeFacesAUX(self.cfg, 'test', mode='lr_input', transform=self.transform_dict['test'])
                    print('\nCalculating SR videos & embeddings for videos in: ', self.cfg.data.path)
            elif self.data_name == 'ms1m':
                self.test = easydict.EasyDict({'data': MS1M(self.cfg, None)})
                # for the MS1M there is a built in transform in the Class


if __name__ == "__main__":
    cfg = read_cfg('configs')
    dataset = YoutubeFacesAUX(cfg, 'train', 'train', torchvision.transforms.ToTensor())
    d = MultiEpochsDataLoader(dataset.train, batch_size=4)
    a = iter(d)
    b = next(a)
