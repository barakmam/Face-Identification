import cv2
import torch
import torchvision.transforms
from PIL import Image
import numpy as np
import os
import pickle
from glob import glob
from os.path import join
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import random
from mycode.v1.utils.misc import read_cfg
from mycode.v1.data.multi_epoch_dataloader import MultiEpochsDataLoader
from mycode.v1.data.transform import SimilarityTrans


class BaseYTF(Dataset):
    def __init__(self, cfg, transform, stage, mode, id2label_dict):
        """
        This class for the YTF data that is already in HR
        (not necessarily sharp GT videos, just in image size of the face embedder input).
        """
        super().__init__()
        self.metadata_path = cfg.data.metadata_path
        self.cfg = cfg
        self.stage = stage
        self.videos_path = os.path.join(cfg.data.path, 'videos')
        self.transform = transform
        self.seed = cfg.system.seed
        self.hr_res = cfg.transformer.fr.hr_res
        self.sharp_videos_path = self.videos_path.replace(self.videos_path.split('/')[-1], 'sharp')
        with open(os.path.join(self.metadata_path, 'splits', stage, mode + '.pickle'), 'rb') as handle:
            self.dataset = pickle.load(handle)
        self.targets = np.array([id2label_dict[name.split('/')[-1][:-2]] for name in self.dataset])

    def __str__(self):
        return 'ytf'

    def __len__(self):
        return len(self.dataset)


class YoutubeFacesAUX(Dataset):
    def __init__(self, cfg, stage, mode, transform):
        super().__init__()
        self.metadata_path = cfg.data.metadata_path
        with open(os.path.join(self.metadata_path, 'id2label_dict.pickle'), 'rb') as handle:
            self.id2label_dict = pickle.load(handle)

        if (stage == 'test' or stage is None) and mode == 'hr_input':
            self.data = FaceImage(cfg, transform, 'all_data', 'all_data', self.id2label_dict)

    def __str__(self):
        return 'ytf'


class FaceImage(BaseYTF):
    def __init__(self, cfg, transform, stage, mode, id2label_dict):
        """
            This class for the YTF data. the output is the detected face in the image (the one that
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
        try:
            image = cv2.cvtColor(cv2.imread(self.dataset[idx]), cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(e)
            raise Exception('Error in getitem: ', self.dataset[idx])
        if self.transform:
            if 'similarity_trans' in self.cfg.data.augmentations.test.types:
                image = self.sim_trans(image, self.landmarks[idx])
            image = self.transform(image)
        assert image.shape[1] == image.shape[2] == self.hr_res, f"frame {self.dataset[idx]} is not of size 112x112"
        return image, self.targets[idx], idx

    def __str__(self):
        return 'hr_ytf'


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


if __name__ == "__main__":
    cfg = read_cfg('configs')
    dataset = YoutubeFacesAUX(cfg, 'train', 'train', torchvision.transforms.ToTensor())
    d = MultiEpochsDataLoader(dataset.train, batch_size=4)
    a = iter(d)
    b = next(a)
