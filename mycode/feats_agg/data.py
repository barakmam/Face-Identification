import os
import pickle
import sys
from itertools import compress

import numpy as np
import torch
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate, DataLoader
from torchvision import transforms
import collections

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mycode.utils.misc import filter_videos, filter_templates

seed = 7


# Transforms

class ShuffleRows:
    def __call__(self, mat):
        return mat[np.random.permutation(mat.shape[0])]


class SampleRows:
    def __init__(self, num_rows):
        self.num_rows = num_rows

    def __call__(self, mat):
        num_rows = self.num_rows if self.num_rows > 0 else np.random.randint(0, mat.shape[0])
        return mat[np.random.permutation(mat.shape[0])[:num_rows]]


class RandomMasking:
    def __call__(self, mat):
        w = np.random.randint(mat.shape[1] // 7)
        x0 = np.random.randint(mat.shape[1] - w)
        h = np.random.randint(mat.shape[0] // 7)
        y0 = np.random.randint(mat.shape[0] - h)
        mat[y0:(y0 + h), x0:(x0 + w)] = 0
        return mat


class UseSingleRow:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, mat):
        if np.random.rand() < self.p:
            mat = np.repeat(mat[[np.random.randint(len(mat))]], len(mat), axis=0)
        return mat


# Datasets:

class SingleTemplate(Dataset):
    def __init__(self, cfg, data_state='train'):
        np.random.seed(seed)
        self.name = cfg.name
        self.data_state = data_state
        self.cfg = cfg

        splits_path = os.path.join(cfg.path, 'splits')
        self.random_gallery = True if data_state == 'train' else False
        self.num_frames = cfg.num_frames

        min_num_images = cfg.min_num_images + 1  # +1 for the gallery image in case of the MS1M dataset
        max_num_images = cfg.max_num_images


        self.feats_g, self.ids_g, self.filename_g = \
            filter_templates(os.path.join(cfg.path, 'embeddings.pickle'), data_state, splits_path, min_num_images, max_num_images)
        self.feats_g = [normalize(f) for f in self.feats_g]

        if cfg.embedding_dir != cfg.path:
            self.feats_q, self.ids_q, self.filename_q = \
                filter_templates(os.path.join(cfg.embedding_dir, 'embeddings.pickle'), data_state, splits_path, min_num_images, max_num_images)
            self.feats_q = [normalize(f) for f in self.feats_q]
        else:
            self.feats_q, self.ids_q, self.filename_q = self.feats_g, self.ids_g, self.filename_g

        self.transform = None

    def __len__(self):
        return len(self.feats_q)

    def __getitem__(self, idx):
        feats_q = self.feats_q[idx]
        if len(feats_q.shape) == 1:
            feats_q = np.expand_dims(feats_q, 0)
        if self.random_gallery:
            g_ind = np.random.randint(len(feats_q))
        else:
            g_ind = -1
        feats_g = normalize(self.feats_g[idx][[g_ind]]).squeeze()
        feats_q = np.delete(feats_q, g_ind, axis=0)
        if self.transform:
            feats_q = self.transform(feats_q)

        feats_q = feats_q[np.random.choice(len(feats_q), self.num_frames, replace=False)]

        raw_sim = feats_q @ feats_g
        return {'feats_g': feats_g, 'feats_q': feats_q, 'ids': self.ids_q[idx], 'raw_sim': raw_sim}


class SingleVideoFrames(Dataset):
    def __init__(self, cfg, data_state='train'):
        np.random.seed(seed)
        self.data_state = data_state
        self.name = cfg.name
        self.num_frames = cfg.num_frames
        self.transform = None
        # if data_state == 'train':
        #     self.transform = transforms.Compose([
        #         SampleRows(),
        #         RandomMasking()
        #     ])
        self.feats_q, self.id_q, self.filenames = \
            filter_videos(os.path.join(cfg.embedding_dir, 'embeddings.pickle'), os.path.join(data_state, 'query'),
                          cfg.detection_q)
        self.feats_q = [normalize(f) for f in self.feats_q]

        if self.num_frames:
            vids_with_min_frames = np.array(
                [ii for ii in range(len(self.feats_q)) if len(self.feats_q[ii]) >= cfg.num_frames])
            self.feats_q = [f for f in self.feats_q if len(f) >= cfg.num_frames]
            self.id_q = self.id_q[vids_with_min_frames]
            self.filenames = [f for f in self.filenames if len(f) >= cfg.num_frames]

            self.infer_frames = [np.random.choice(len(f), self.num_frames, replace=False) for f in self.feats_q]
        # with open(os.path.join(cfg.embedding_dir, 'sim_to_correct_gallery.pickle'), 'rb') as h:
        #     pos_sim =

        head_pose_file = '/datasets/BionicEye/YouTubeFaces/head_pose/head_pose_supplied.pickle'
        self.gallery, self.id_g, self.filenames_g, self.frontal_inds = \
            filter_videos(os.path.join(cfg.path, 'embeddings.pickle'), os.path.join(data_state, 'gallery'),
                          os.path.join(cfg.path, 'detection.pickle'), head_pose_file, 1)
        self.gallery = normalize(self.gallery, axis=(self.gallery.ndim-1))  # axis=-1 is not supported

    def __len__(self):
        return len(self.feats_q)

    def __getitem__(self, idx):
        feats_q = self.feats_q[idx]
        if len(feats_q.shape) == 1:
            feats_q = np.expand_dims(feats_q, 0)
        feats_g = self.gallery[self.id_g == self.id_q[idx]].squeeze()
        if self.transform:
            feats_q = self.transform(feats_q)
            feats_q = feats_q

        selected_frames = np.random.choice(len(feats_q), self.num_frames, replace=False)
        if self.data_state != 'train' and self.num_frames > 0:
            selected_frames = self.infer_frames[idx]
        feats_q = feats_q[selected_frames]
        raw_sim = feats_q @ feats_g
        return {'feats_q': feats_q, 'feats_g': feats_g, 'ids': self.id_q[idx], 'raw_sim': raw_sim}


class QueryGalleryPair(SingleVideoFrames):
    def __init__(self, data_state, state='testing', data_path='/datasets/BionicEye/YouTubeFaces/faces/sharp',
                 splits_path='/inputs/bionicEye/data/ytf/splits', num_frames=40):
        super().__init__(data_state + '_query', data_path, splits_path, num_frames)
        self.gallery = SingleVideoFrames(data_state + '_gallery', data_path, splits_path, num_frames=1)
        self.state = state

        # create all pairs of [single query frame, single gallery]
        query, frame, gallery = np.meshgrid(np.arange(len(self.dataset)), np.arange(num_frames),
                                            np.arange(len(self.gallery)))
        self.pairs = np.hstack(query.flatten(), frame.flatten, gallery.flatten())

        if state == 'training' and data_state == 'train':
            with open(os.path.join(data_path, 'rank_of_correct_gallery_trainset.pickle'), 'rb') as h:
                self.ranks = pickle.load(h)
            # get the ranking for the frontal frames chosen:
            self.ranks = np.vstack([self.ranks[self.filenames[ii][0].split('/')[-2]][self.frontal_inds[ii]] for ii in
                                    range(len(self.ranks))])

    def __getitem__(self, idx):
        t = self.pairs[idx]
        query_id = self.labels[t[0]]
        gallery_id = self.gallery.labels[t[2]]
        query_frame_feats = self.dataset[t[0], t[1]]
        gallery_feats = self.gallery.dataset[t[2]]

        if self.state == 'training' and self.data_state == 'train':
            all_frames_rank = self.ranks[t[0]]
            frame_rank = all_frames_rank[t[1]]
            label = (query_id == gallery_id) and (frame_rank == all_frames_rank.min())
            return {'query': query_frame_feats, 'gallery': gallery_feats, 'label': label}
        else:
            query_vid = t[0]
            return {'query': query_frame_feats, 'gallery': gallery_feats, 'query_vid': query_vid,
                    'query_id': query_id, 'gallery_id': gallery_id}

    def __len__(self):
        return len(self.pairs)


class FrameSelectorData(SingleVideoFrames):
    def __init__(self, data_state, state='testing', data_path='/datasets/BionicEye/YouTubeFaces/faces/sharp',
                 splits_path='/inputs/bionicEye/data/ytf/splits', num_frames=40):
        super().__init__(data_state + '_query', data_path, splits_path, num_frames)
        self.gallery = SingleVideoFrames(data_state + '_gallery', data_path, splits_path, num_frames=1)
        self.state = state
        if state == 'training' and data_state == 'train':
            with open(os.path.join(data_path, 'rank_of_correct_gallery_trainset.pickle'), 'rb') as h:
                self.ranks = pickle.load(h)
            # get the ranking for the frontal frames chosen:
            self.ranks = np.vstack([self.ranks[self.filenames[ii][0].split('/')[-2]][self.frontal_inds[ii]] for ii in
                                    range(len(self.ranks))])

    def __getitem__(self, idx):
        feats, label = super().__getitem__(idx)
        sim = feats @ self.gallery.dataset.T

        if self.state == 'training' and self.data_state == 'train':
            # choose the label to be the index of the frame that has the best ranking.
            # since we might have multiple frames that achieve the best ranking, sample randomly from there indices
            label = np.random.choice(np.where(self.ranks[idx] == np.min(self.ranks[idx]))[0])
        return feats, label, sim


class SimilarityData(Dataset):
    def __init__(self, data_state, state='testing', data_path='/datasets/BionicEye/YouTubeFaces/faces/sharp',
                 splits_path='/inputs/bionicEye/data/ytf/splits', num_frames=40, k_sim=40):
        super(SimilarityData, self).__init__()
        self.data_state = data_state
        self.num_frames = num_frames
        self.k_sim = k_sim
        self.transform = None

        prepared_data_file = f'/inputs/bionicEye/data/feats_agg/similarity_data/num_frames_{num_frames}/{data_state}.pickle'
        if os.path.exists(prepared_data_file):
            print('Loading Similarity Data From: ', prepared_data_file)
            with open(prepared_data_file, 'rb') as h:
                data = pickle.load(h)
            self.query_vid_idx = data['query_vid_idx']
            self.sim = data['sim']
            self.id_g = data['id_g']
            self.id_q = data['id_q']
        else:
            head_pose_file = '/datasets/BionicEye/YouTubeFaces/head_pose/head_pose_supplied.pickle'
            stats_path = os.path.join(data_path, 'embeddings.pickle')
            feats_q, self.id_q, _, _ = filter_videos(stats_path, data_state + '_query', num_frames, splits_path,
                                                     head_pose_file)
            feats_g, self.id_g, _, _ = filter_videos(stats_path, data_state + '_gallery', detection_, splits_path,
                                                     head_pose_file)

            print(f'{data_state}: Preparing Similarity Matrix...')
            self.sim = np.einsum('qfe,ge->qfg', feats_q, feats_g)
            argsort = np.flip(np.argsort(self.sim, axis=-1), axis=-1)
            self.sim = np.take_along_axis(self.sim, argsort, axis=-1)
            self.id_g = self.id_g[argsort]

            self.query_vid_idx = np.repeat(np.arange(len(self.id_q)), num_frames)
            self.sim = np.concatenate(self.sim)
            self.id_g = np.concatenate(self.id_g)
            self.id_q = self.id_q[self.query_vid_idx]
            print(f'{data_state}: Similarity Matrix - DONE!')
            print('Saving Similarity Data in: ', prepared_data_file)
            os.makedirs('/'.join(prepared_data_file.split('/')[:-1]), exist_ok=True)
            with open(prepared_data_file, 'wb') as h:
                pickle.dump(
                    {'query_vid_idx': self.query_vid_idx, 'sim': self.sim, 'id_g': self.id_g, 'id_q': self.id_q}, h)

        self.labels = (self.id_g[:, 0] == self.id_q).astype('int')

    def __getitem__(self, idx):
        sim = self.sim[idx, :self.k_sim]
        query_id = self.id_q[idx]
        gallery_id = self.id_g[idx, :self.k_sim]
        label = self.labels[idx]
        vid_idx = self.query_vid_idx[idx]
        if self.transform:
            sim = self.transform(sim)

        return {'sim': sim, 'label': label, 'vid_idx': vid_idx, 'id_q': query_id, 'id_g': gallery_id}

    def __len__(self):
        # if self.data_state == 'train':
        #     return 16*self.num_frames
        return len(self.sim)


class TripltesEmbeddings(SingleVideoFrames):
    def __init__(self, state, data_path='/datasets/BionicEye/YouTubeFaces/faces/sharp',
                 splits_path='/inputs/bionicEye/data/ytf_new_splits', n_triplets=10000, num_frames=40):
        super().__init__(state, data_path, splits_path, num_frames)
        from mycode.data.data_module import YoutubeFacesTriplets
        self.n_triplets = n_triplets
        self.triplets = YoutubeFacesTriplets.generate_hard_triplets(self.dataset, self.ids, self.n_triplets)
        # self.triplets = YoutubeFacesTriplets.generate_triplets(self.ids, self.n_triplets)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t = self.triplets[idx]
        a, _ = super().__getitem__(t[0])
        p, _ = super().__getitem__(t[1])
        n, _ = super().__getitem__(t[2])
        # take one frame to positive & negative to resamble the Video2Single identification
        p, n = p[0, torch.randint(self.num_frames, (1,))].squeeze(), n[
            0, torch.randint(self.num_frames, (1,))].squeeze()
        return a, p, n




def pad_matrices(matrix_list):
    # Find the maximum number of rows in the matrix list
    max_rows = max(matrix.shape[0] for matrix in matrix_list)

    # Pad each matrix with zeros to the maximum number of rows
    padded_matrices = np.zeros((len(matrix_list), max_rows, matrix_list[0].shape[1]), dtype=matrix_list[0].dtype)
    for i, matrix in enumerate(matrix_list):
        padded_matrices[i, :matrix.shape[0], :] = matrix

    return padded_matrices


def collate_padding(batch):
    query = [d['feats_q'] for d in batch]
    padded_query = pad_matrices(query)
    for i, d in enumerate(batch):
        d['feats_q'] = padded_query[i]
    return default_collate(batch)


def collate_shuffle(batch):
    all_query = np.stack([d['feats_q'] for d in batch])
    frames_len = len(all_query[0])
    for i, d in enumerate(batch):
        num_impostor = frames_len - 3  # np.random.randint(3, frames_len//3 + 1)
        ids_inds_impostor = np.random.choice(np.delete(np.arange(len(all_query)), i), num_impostor)
        frames_to_switch = np.random.choice(frames_len, num_impostor, replace=False)
        d['feats_q'][frames_to_switch] = all_query[ids_inds_impostor, np.random.randint(frames_len, size=num_impostor)]
        d['genuine'] = np.delete(np.arange(frames_len), frames_to_switch)
    return default_collate(batch)


def get_dataloader(cfg, data_state, batch_size):
    if cfg.collate_fn == 'pad':
        collate_fn = collate_padding
    elif cfg.collate_fn == 'shuffle':
        collate_fn = collate_shuffle
    else:
        collate_fn = None

    if cfg.name == 'ms1m':
        return DataLoader(SingleTemplate(cfg, data_state), batch_size=batch_size, shuffle=True if data_state == 'train' else False,
                                     drop_last=True if data_state == 'train' else False, collate_fn=collate_fn)
    elif cfg.name == 'ytf':
        return DataLoader(SingleVideoFrames(cfg, data_state), batch_size=batch_size, shuffle=True if data_state == 'train' else False,
                                     drop_last=True if data_state == 'train' else False, collate_fn=collate_fn)


if __name__ == '__main__':

    # MS1M Baseline test
    ms1m_data_path = '/inputs/bionicEye/data/ms1m-retinaface-t1'
    stats_path = os.path.join(ms1m_data_path, 'embeddings.pickle')
    splits_path = os.path.join(ms1m_data_path, 'splits')
    feats, ids, filename = filter_templates(stats_path, 'test', splits_path, min_num_images=10, max_num_images=40)
    query = np.stack([f[:-1].mean(0) for f in feats])
    gallery = normalize(np.stack([f[-1] for f in feats]))
    logits = query @ gallery.T
    rank1 = (logits.diagonal() == logits.max(1)).mean()
    print('MS1M Test Rank1: ', rank1)
