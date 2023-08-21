import os
import pickle

import pandas as pd
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.preprocessing import normalize
from torch.utils.data.dataset import Dataset
import torch

from glob import glob

from torchvision.transforms import transforms
from tqdm import tqdm

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mycode.data.multi_epoch_dataloader import MultiEpochsDataLoader
from mycode.utils.metrics import get_recall_at_k
from mycode.utils.analysis import plot_recall_at_k


k_vals = [1, 2, 5, 10, 20, 30]


def plot_eyes_distance_hist(ed_q, ed_g):
    plt.hist(ed_q, 50, density=True, label='Query', alpha=0.5)
    plt.hist(ed_g, 50, density=True, label='Gallery', alpha=0.5)
    plt.axline([ed_q.mean().item(), 0], [ed_q.mean().item(), 10], color='darkblue',
               label=f'Query average:\n{ed_q.mean().item():.2f}'),
    plt.axline([ed_g.mean().item(), 0], [ed_g.mean().item(), 10], color='red',
               label=f'Gallery average:\n{ed_g.mean().item():.2f}'),
    plt.legend(), plt.title('Eyes Distance Histogram', fontsize=17), plt.xlabel('Eyes Distance [pixels]',
                                                                                fontsize=17), plt.ylabel('Frames PDF',
                                                                                                         fontsize=17),
    plt.ylim([0, 0.23]), plt.tight_layout(), plt.show()


def plot_bounding_box_hist(bb_q_area, bb_g_area, ylim=[0, 0.0013]):
    plt.hist(bb_q_area, 50, density=True, label='Query', alpha=0.5)
    plt.hist(bb_g_area, 50, density=True, label='Gallery', alpha=0.5)
    plt.axline([bb_q_area.mean().item(), 0], [bb_q_area.mean().item(), 10], color='darkblue',
               label=f'Query average:\n{bb_q_area.mean().item():.2f}'),
    plt.axline([bb_g_area.mean().item(), 0], [bb_g_area.mean().item(), 10], color='red',
               label=f'Gallery average:\n{bb_g_area.mean().item():.2f}'),
    plt.legend(), plt.title('Bounding Box Histogram', fontsize=17), plt.xlabel('BB Area [pixels]',
                                                                                fontsize=17), plt.ylabel('Frames PDF',
                                                                                                         fontsize=17),
    plt.ylim(ylim), plt.tight_layout(), plt.show()


def enlarge_bb(bb, scale=1.2):
    if bb.ndim > 1:
        w = bb[:, [2]] * scale
        h = bb[:, [3]] * scale
        x0 = np.maximum(0, bb[:, [0]] - w * (scale - 1) / 2)
        y0 = np.maximum(0, bb[:, [1]] - h * (scale - 1) / 2)
        return np.concatenate([x0, y0, w, h], axis=1).astype('int')
    else:
        w = bb[2] * scale
        h = bb[3] * scale
        x0 = max(0, bb[0] - w * (scale - 1) / 2)
        y0 = max(0, bb[1] - h * (scale - 1) / 2)
        return np.array([x0, y0, w, h]).astype('int')


def preprocess(unique_vids, detection_dir, eyes_dist_threshold=10):
    # Face Detection and thresholding
    # removes frames that has eyes distance less than eyes_dist_threshold pixels
    print('Face Detection Started!')
    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detection/retinaface'))
    from mycode.detection.retinaface.retinaface import RetinaFace
    from mycode.detection.retinaface.detection import detection
    detector = RetinaFace('/inputs/RUNAI/code/mycode/models/weights/retinaface-R50/R50', 0, 0, 'net3')
    for vid in tqdm(unique_vids):
        tracklet_name = vid.split('.')[0].split('/')[-1]
        bb, lmk, score = detection(video_path_to_frames_dir(vid), detector)
        eyes_dist = np.linalg.norm(lmk[:, 0] - lmk[:, 1], axis=-1)
        small_ed_idx = np.where(eyes_dist < eyes_dist_threshold)
        bb[small_ed_idx], lmk[small_ed_idx], score[small_ed_idx] = -1, -1, 0
        assert len(bb) > 0, 'No Detection Output'
        with open(os.path.join(detection_dir, tracklet_name + '.pickle'), 'wb') as h:
            pickle.dump({'bb': bb, 'lmk': lmk, 'score': score}, h)
    print('Face Detection DONE!')
    print('Saved detections to {}'.format(detection_dir))


def create_identification_splits(ids_vids, detection_dir, detection_threshold):
    ids = np.arange(len(ids_vids))

    vids_g, largest_bb_frame_idx, id_g = [], [], []
    vids_q, id_q = [], []
    for ii in range(len(ids)):
        # for the gallery take the video that has the largest bounding box
        # also save the frame index that is the best in that video (for V2S identification)
        max_eyes_dist_video = ''
        max_eyes_dist_size = 0
        best_vids_frame_idx = 0
        for vid_name in ids_vids[ii]:
            tracklet_name = vid_name.split('.')[0].split('/')[-1]
            with open(os.path.join(detection_dir, tracklet_name + '.pickle'), 'rb') as h:
                detection_info = pickle.load(h)
            score = detection_info['score']
            frames_inds = np.arange(len(score))
            eyes_dist = np.linalg.norm(detection_info['lmk'][score > detection_threshold, 0, :] - detection_info['lmk'][score > detection_threshold, 1, :], axis=1)
            frames_inds = frames_inds[score > detection_threshold]
            score = score[score > detection_threshold]


            if eyes_dist.max() > max_eyes_dist_size:
                max_eyes_dist_video = vid_name
                max_eyes_dist_size = eyes_dist.max()
                best_vids_frame_idx = frames_inds[eyes_dist.argmax()]
        largest_bb_frame_idx.append(best_vids_frame_idx)
        vids_g.append(max_eyes_dist_video)
        id_g.append(ids[ii])

        # all the rest of the videos from that id will go to the query set
        vids_q.append(ids_vids[ii][ids_vids[ii] != max_eyes_dist_video])
        id_q.extend([ids[ii]]*(len(ids_vids[ii]) - 1))

    vids_q = np.concatenate(vids_q)
    query_gallery_split = {'query': {'vids': vids_q, 'ids': np.array(id_q)},
                           'gallery': {'vids': np.array(vids_g), 'largest_bb_frame_idx': np.array(largest_bb_frame_idx),
                                       'ids': np.array(id_g)}}
    return query_gallery_split


def find_connected_components(arr):
    """
    Given a numpy array of shape (N, 2) where each row defines a connection
    between two elements, this function returns a list of lists, where each
    list is a connected component in the input array.

    Parameters
    ----------
    arr : numpy.ndarray
        A 2D array of shape (N, 2) representing connections between elements.

    Returns
    -------
    list
        A list of lists, where each inner list is a connected component in the input array.
    """

    import networkx as nx
    # create a graph from the input array
    graph = nx.from_edgelist(arr)

    # find the connected components using networkx function
    components = list(nx.connected_components(graph))

    # convert the sets to lists and return the result
    result = [list(c) for c in components]
    return result


def video_path_to_frames_dir(vid_path):
    vid_name = '_'.join(vid_path.split('/')[-1].split('.')[0].split('_')[:-1])
    id_num = vid_path.split('/')[-1].split('.')[0].split('_')[-1]
    frames_dir = f'/vis/outputs/gavriel/webcams/abbeyroad_uk_{vid_name}/ids/{id_num}/rgb'
    if not os.path.isdir(frames_dir):
        # in case running the mapping of Gavriel outputs is /outputs_gavriel:
        frames_dir = f'/outputs_gavriel/webcams/abbeyroad_uk_{vid_name}/ids/{id_num}/rgb'
    return frames_dir


class WebCams(Dataset):
    def __init__(self, vids_name, lmk_dir):
        super().__init__()
        self.vids_name = vids_name
        self.lmk_dir = lmk_dir

        frames = []
        vid_idx = []
        frame_idx = []
        lmks = []
        score = []
        bb = []
        for ii in range(len(vids_name)):
            curr_frames = glob(video_path_to_frames_dir(vids_name[ii]) + '/*')
            num_frames = len(curr_frames)
            frames.extend(curr_frames)
            frame_idx.extend(np.arange(num_frames))
            vid_idx.extend([ii]*num_frames)
            tracklet_name = vids_name[ii].split('.')[0].split('/')[-1]
            with open(os.path.join(lmk_dir, tracklet_name + '.pickle'), 'rb') as h:
                detection_info = pickle.load(h)
            lmks.append(detection_info['lmk'])
            bb.append(detection_info['bb'])
            score.append(detection_info['score'])
        lmks = np.concatenate(lmks)
        bb = np.concatenate(bb)
        score = np.concatenate(score)
        frames = np.array(frames)
        frame_idx = np.array(frame_idx)
        vid_idx = np.array(vid_idx)

        # filter the frames that do not have detection. it might be whole video that is filtered out.
        detected = score > 0
        self.frames = frames[detected]
        self.frame_idx = frame_idx[detected]
        self.lmks = lmks[detected]
        self.bb = bb[detected]
        self.vid_idx = vid_idx[detected]

        from mycode.data.transform import SimilarityTrans
        self.sim_trans = SimilarityTrans()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):
        im = cv2.cvtColor(cv2.imread(self.frames[item]), cv2.COLOR_BGR2RGB)
        lmk = self.lmks[item]  # relative to the full image
        bb = self.bb[item]
        vid_idx = self.vid_idx[item]
        frame_idx = self.frame_idx[item]
        bb = enlarge_bb(bb)
        face = im[bb[1]:(bb[1] + bb[3]), bb[0]:(bb[0] + bb[2])]
        lmk = lmk - bb[:2]  # relative to the face crop
        face = self.sim_trans(face, lmk)
        face = self.transform(face)
        # use flip image:
        # face_flip = np.fliplr(face)
        # face_flip = self.transform(face_flip)
        # input = torch.stack([face, face_flip], 0)
        # return input, vid_idx
        return face, vid_idx, frame_idx


def features(vids_name, lmk_dir):
    testset = WebCams(vids_name, lmk_dir)
    test_loader = MultiEpochsDataLoader(testset, batch_size=64, drop_last=False, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from mycode.models.backbones import get_model
    model = get_model('r100', fp16=False).to('cuda')
    prefix = '/inputs/bionicEye/insightFace/models/torch/r100_ms1mv3/backbone.pth'
    state_dict = torch.load(prefix, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_feats = []
    all_vid_idx = []
    all_frame_idx = []

    print('Feature Extraction Started!')
    for x, vid, frame in tqdm(test_loader):
        all_vid_idx.append(vid.numpy())
        all_frame_idx.append(frame.numpy())
        x = x.to(device)
        feats = model(x)
        all_feats.append(feats.detach().cpu().numpy())

    print('Feature Extraction DONE!')

    return np.concatenate(all_feats), vids_name[np.concatenate(all_vid_idx)], np.concatenate(all_frame_idx)


def verification(feats, vids, pairs, labels):
    """
    example:
    # Face Verification:
    # pairs = pd.concat([pos, neg])
    # iloc_detcted = [ii for ii in range(len(pairs))
    #                 if ((pairs.iloc[ii].query in vids) and (pairs.iloc[ii].gallery in vids))]
    # labels = pairs.iloc[iloc_detcted].label.to_numpy()
    # labels[labels == 'Yes'] = 1
    # labels[labels == 'No'] = 0
    # labels = labels.astype('int')
    # pairs = np.array([pairs.iloc[iloc_detcted]['query'].to_numpy(), pairs.iloc[iloc_detcted]['gallery'].to_numpy()]).T
    # verification(feats, vids, pairs, labels)
    """
    from sklearn.preprocessing import normalize
    from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
    feats1 = np.array([feats[vids == p[0]].sum(0) for p in pairs])
    feats2 = np.array([feats[vids == p[1]].sum(0) for p in pairs])
    feats1 = normalize(feats1)
    feats2 = normalize(feats2)
    preds = np.sum(feats1 * feats2, 1)
    auc = roc_auc_score(labels, preds)

    fpr, tpr, thresholds = roc_curve(labels, preds)
    plt.plot(fpr, tpr), plt.title('ROC Curve')
    plt.xlabel('False Positive Rate'), plt.ylabel('True Positive Rate')
    plt.show()

    pre, re, th = precision_recall_curve(labels, preds)
    plt.plot(re, pre), plt.title('Precision Recall Curve')
    plt.xlabel('Recall'), plt.ylabel('Precision')
    plt.show()

    print('Verification Test:')
    print('Negatives: ', sum(labels == 0), 'Positives: ', sum(labels == 1))
    print('AUC: ', auc)


def filter_detections_by_threshold(feats, vids, frame_idx, detection_dir, threshold):
    """
    :param vids: vids array after detection with threshold=0.2 (floor threshold).
    :param frame_idx: the frame indx corresponds to vids array
    :param threshold: the score threshold to omit inds with smaller score.
    :return: inds to take over the arrays
    """

    print('#'*10, 'Filtering out detections with score under ', threshold, '#'*10)
    inds_to_take = []

    def filter_parallel(ii):
        tracklet_name = vids[ii].split('.')[0].split('/')[-1]
        with open(os.path.join(detection_dir, tracklet_name + '.pickle'), 'rb') as h:
            detection_info = pickle.load(h)
        if detection_info['score'][frame_idx[ii]] >= threshold:
            inds_to_take.append(ii)


    Parallel(n_jobs=256, backend='threading')(delayed(filter_parallel)(ii)
                         for ii in tqdm(range(len(vids))))

    inds_to_take = np.sort(inds_to_take)
    return feats[inds_to_take], vids[inds_to_take], frame_idx[inds_to_take]


def v2s_identification_webcams(embedding_file, query_gallery_split, detection_dir, best_k_frames=None, filters=None, plot_tp_n_fp=False, aggregate=None):
    with open(embedding_file, 'rb') as h:
        feats_stats = pickle.load(h)
    feats, vids, frame_idx = feats_stats['feats'], feats_stats['vids'], feats_stats['frame_idx']

    id_q, id_g = query_gallery_split['query']['ids'], query_gallery_split['gallery']['ids']
    feats_g, filename_g, bb_g = [], [], []
    for vid, frame_ind in zip(query_gallery_split['gallery']['vids'], query_gallery_split['gallery']['largest_bb_frame_idx']):
        tracklet_name = vid.split('.')[0].split('/')[-1]
        with open(os.path.join(detection_dir, tracklet_name + '.pickle'), 'rb') as h:
            detection_info = pickle.load(h)
        bb_g.append(enlarge_bb(detection_info['bb'][frame_ind]))
        feats_g.append(feats[(vids == vid) & (frame_idx == frame_ind)])

        all_vid_frames = np.array(glob(video_path_to_frames_dir(vid) + '/*'))
        filename_g.append(all_vid_frames[frame_ind])
    feats_g = normalize(np.vstack(feats_g))


    feats_q, filename_q, bb_q = [], [], []
    for vid in query_gallery_split['query']['vids']:
        tracklet_name = vid.split('.')[0].split('/')[-1]
        with open(os.path.join(detection_dir, tracklet_name + '.pickle'), 'rb') as h:
            detection_info = pickle.load(h)

        score = detection_info['score'][feats_stats['frame_idx'][vids == vid]]
        lmk = detection_info['lmk'][feats_stats['frame_idx'][vids == vid]]
        curr_bb = enlarge_bb(detection_info['bb'][feats_stats['frame_idx'][vids == vid]])

        inds = np.arange(len(curr_bb))
        if best_k_frames:
            filter = []
            if 'bb_size' in filters:
                bb_size = curr_bb[:, 2] * curr_bb[:, 3]  # w*h
                filter.append(bb_size/max(bb_size))
            if 'eyes_dist' in filters:
                eyes_dist = np.linalg.norm(lmk[:, 0] - lmk[:, 1], axis=-1)
                filter.append(eyes_dist/max(eyes_dist))
            if 'feat_norm' in filters:
                feats_norm = np.linalg.norm(feats[vids == vid], axis=-1)
                filter.append(feats_norm/max(feats_norm))
            max_inds_len = min(best_k_frames, len(inds))  # if there are less frames than k_largest_bb, take all frames.
            inds = np.argpartition(np.sum(filter, axis=0), -max_inds_len)[-max_inds_len:]

        bb_q.append(curr_bb[inds])

        if aggregate:
            feats_q.append(aggregate(feats[vids == vid][inds]))
        else:
            feats_q.append(feats[vids == vid][inds].mean(0))

        all_vid_frames = np.array(glob(video_path_to_frames_dir(vid) + '/*'))
        filename_q.append(all_vid_frames[feats_stats['frame_idx'][vids == vid][inds]])


    feats_q = normalize(np.vstack(feats_q))
    print('Webcams Identification')
    print('Query: ', len(id_q))
    print('Gallery: ', len(id_g))

    if plot_tp_n_fp:
        from mycode.utils.plots import plot_identification_samples
        logits = feats_q @ feats_g.T
        plot_identification_samples(logits, id_q, id_g, filename_q, filename_g, bb_q, bb_g)

    rank_at_k = get_recall_at_k(feats_q, feats_g, id_q, id_g, k_vals)
    print(f'CMC @ Rank - V2S Webcams: ', rank_at_k)
    return rank_at_k


def s2s_identification_webcams(embedding_file, query_gallery_split, detection_dir, best_k_frames=None, filters=None, plot_tp_n_fp=False):
    with open(embedding_file, 'rb') as h:
        feats_stats = pickle.load(h)
    feats, vids, frame_idx = feats_stats['feats'], feats_stats['vids'], feats_stats['frame_idx']

    id_q, id_g = query_gallery_split['query']['ids'], query_gallery_split['gallery']['ids']

    feats_g, filename_g, bb_g = [], [], []
    for vid, frame_ind in zip(query_gallery_split['gallery']['vids'],
                              query_gallery_split['gallery']['largest_bb_frame_idx']):
        all_vid_frames = np.array(glob(video_path_to_frames_dir(vid) + '/*'))
        filename_g.append(all_vid_frames[frame_ind])
        tracklet_name = vid.split('.')[0].split('/')[-1]
        with open(os.path.join(detection_dir, tracklet_name + '.pickle'), 'rb') as h:
            detection_info = pickle.load(h)
        bb_g.append(enlarge_bb(detection_info['bb'][frame_ind]))
        feats_g.append(feats[(vids == vid) & (frame_idx == frame_ind)])
    feats_g = normalize(np.vstack(feats_g))

    all_rank_at_k = []
    for trial in range(30):
        feats_q, filename_q, bb_q = [], [], []
        for vid in query_gallery_split['query']['vids']:
            tracklet_name = vid.split('.')[0].split('/')[-1]
            with open(os.path.join(detection_dir, tracklet_name + '.pickle'), 'rb') as h:
                detection_info = pickle.load(h)

            score = detection_info['score'][feats_stats['frame_idx'][vids == vid]]
            lmk = detection_info['lmk'][feats_stats['frame_idx'][vids == vid]]
            curr_bb = enlarge_bb(detection_info['bb'][feats_stats['frame_idx'][vids == vid]])

            inds = np.arange(len(curr_bb))
            if best_k_frames:
                filter = []
                if 'bb_size' in filters:
                    bb_size = curr_bb[:, 2] * curr_bb[:, 3]  # w*h
                    filter.append(bb_size / max(bb_size))
                if 'eyes_dist' in filters:
                    eyes_dist = np.linalg.norm(lmk[:, 0] - lmk[:, 1], axis=-1)
                    filter.append(eyes_dist / max(eyes_dist))
                if 'feat_norm' in filters:
                    feats_norm = np.linalg.norm(feats[vids == vid], axis=-1)
                    filter.append(feats_norm / max(feats_norm))
                max_inds_len = min(best_k_frames,
                                   len(inds))  # if there are less frames than k_largest_bb, take all frames.
                inds = np.argpartition(np.sum(filter, axis=0), -max_inds_len)[-max_inds_len:]

            sample_ind = np.random.choice(inds)
            bb_q.append(curr_bb[sample_ind])
            feats_q.append(feats[vids == vid][sample_ind])

            all_vid_frames = np.array(glob(video_path_to_frames_dir(vid) + '/*'))
            filename_q.append(all_vid_frames[feats_stats['frame_idx'][vids == vid][sample_ind]])

        feats_q = np.vstack(feats_q)
        if plot_tp_n_fp:
            from mycode.utils.plots import plot_identification_samples
            logits = feats_q @ feats_g.T
            plot_identification_samples(logits, query_gallery_split['query']['ids'], query_gallery_split['gallery']['ids'],
                                        filename_q, filename_g, bb_q, bb_g)

        curr_rank_at_k = get_recall_at_k(feats_q, feats_g, id_q, id_g, k_vals)
        all_rank_at_k.append(curr_rank_at_k)

    print('Webcams Identification')
    print('Query: ', len(id_q))
    print('Gallery: ', len(id_g))
    rank_at_k = np.mean(all_rank_at_k, axis=0)
    print(f'CMC @ Rank - S2S Webcams: ', rank_at_k)
    return rank_at_k


def v2s_oracle(embedding_file, query_gallery_split):
    with open(embedding_file, 'rb') as h:
        feats_stats = pickle.load(h)
    feats, vids, frame_idx = feats_stats['feats'], feats_stats['vids'], feats_stats['frame_idx']

    id_q, id_g = query_gallery_split['query']['ids'], query_gallery_split['gallery']['ids']
    feats_g, filename_g, bb_g = [], [], []
    for vid, frame_ind in zip(query_gallery_split['gallery']['vids'], query_gallery_split['gallery']['largest_bb_frame_idx']):
        tracklet_name = vid.split('.')[0].split('/')[-1]
        with open(os.path.join('face_detection', tracklet_name + '.pickle'), 'rb') as h:
            detection_info = pickle.load(h)
        bb_g.append(enlarge_bb(detection_info['bb'][frame_ind]))
        feats_g.append(feats[(vids == vid) & (frame_idx == frame_ind)])

        all_vid_frames = np.array(glob(video_path_to_frames_dir(vid) + '/*'))
        filename_g.append(all_vid_frames[frame_ind])
    feats_g = normalize(feats_g)


    feats_q, filename_q, bb_q = [], [], []
    for ii, vid in enumerate(query_gallery_split['query']['vids']):
        curr_feats = feats[vids == vid]
        correct_gallery_idx = np.where(id_g == id_q[ii])[0].item()
        sim_mat = curr_feats @ feats_g.T
        argsort = np.argsort(sim_mat)[:, ::-1]  # descending order
        where = np.where(argsort == correct_gallery_idx)
        feats_q.append(curr_feats[np.argmin(where[1])])
        all_vid_frames = np.array(glob(video_path_to_frames_dir(vid) + '/*'))
        filename_q.append(all_vid_frames[feats_stats['frame_idx'][vids == vid][np.argmin(where[1])]])
    feats_q = np.vstack(feats_q)

    print('Webcams Identification')
    print('Query: ', len(id_q))
    print('Gallery: ', len(id_g))

    rank_at_k = get_recall_at_k(feats_q, feats_g, id_q, id_g, k_vals)
    print(f'CMC @ Rank - Oracle V2S Webcams: ', rank_at_k)
    return rank_at_k


def calc_eyes_distance(detection_dir, query_gallery_split, vids, frame_idx):
    """
    example: calc_eyes_distance(detection_dir, query_gallery_split, vids, frame_idx)
    """
    eyes_dist_g, id_g = [], []
    for vid, frame_ind, id_ in zip(query_gallery_split['gallery']['vids'],
                              query_gallery_split['gallery']['largest_bb_frame_idx'],
                              query_gallery_split['gallery']['ids']):
        tracklet_name = vid.split('.')[0].split('/')[-1]
        with open(os.path.join(detection_dir, tracklet_name + '.pickle'), 'rb') as h:
            detection_info = pickle.load(h)
        eyes_dist_g.append(
            np.linalg.norm(
                detection_info['lmk'][frame_ind, 0, :] - detection_info['lmk'][frame_ind, 1, :])
        )
        id_g.append(id_)
    eyes_dist_g, id_g = np.array(eyes_dist_g), np.array(id_g)


    eyes_dist_q, id_q = [], []
    for vid, id_ in zip(query_gallery_split['query']['vids'], query_gallery_split['query']['ids']):
        tracklet_name = vid.split('.')[0].split('/')[-1]
        with open(os.path.join(detection_dir, tracklet_name + '.pickle'), 'rb') as h:
            detection_info = pickle.load(h)
        frames_inds = frame_idx[vids == vid]
        eyes_dist_q.append(
            np.linalg.norm(
                detection_info['lmk'][frames_inds, 0, :] - detection_info['lmk'][frames_inds, 1, :], axis=1)
        )
        id_q.append(id_)

    id_q = np.array(id_q)

    mean_q = [arr.mean() for arr in eyes_dist_q]
    std_q = [arr.std() for arr in eyes_dist_q]
    plt.errorbar(id_q, mean_q, std_q, fmt='o', linewidth=2,
                 markersize=3, color='teal', capsize=2, capthick=1, label='Query')
    all_mean_q = np.concatenate(eyes_dist_q).mean()
    plt.axline((0, all_mean_q), slope=0, color='green', linewidth=5)

    plt.scatter(id_g, eyes_dist_g, color='darkmagenta', label='Gallery')
    all_mean_g = np.mean(eyes_dist_g)
    plt.axline((0, all_mean_g), slope=0, color='indigo', linewidth=5)

    plt.xlabel('#Subject', fontsize=20), plt.ylabel('Eyes Distance (Euclidean)', fontsize=20)
    plt.title('Eyes Distance In Each Sample', fontsize=20), plt.grid()
    plt.xticks(fontsize=20), plt.yticks(fontsize=20)
    plt.legend(), plt.tight_layout()
    plt.show()


def compare_results(embedding_file, query_gallery_split, detection_dir):
    iden_k_largest_bb = 20
    r1 = s2s_identification_webcams(embedding_file, query_gallery_split, detection_dir)
    r2 = v2s_identification_webcams(embedding_file, query_gallery_split, detection_dir)
    r3 = v2s_identification_webcams(embedding_file, query_gallery_split, detection_dir, iden_k_largest_bb,
                                    filters=['eyes_dist'])
    r4 = v2s_identification_webcams(embedding_file, query_gallery_split, detection_dir, iden_k_largest_bb,
                                    filters=['bb_size'])
    r5 = v2s_identification_webcams(embedding_file, query_gallery_split, detection_dir, iden_k_largest_bb,
                                    filters=['feat_norm'])
    r6 = v2s_identification_webcams(embedding_file, query_gallery_split, detection_dir, iden_k_largest_bb,
                                    filters=['bb_size', 'eyes_dist', 'feat_norm'])
    r7 = v2s_identification_webcams(embedding_file, query_gallery_split, detection_dir, 1,
                                    filters=['bb_size', 'eyes_dist', 'feat_norm'])
    plot_recall_at_k([r1, r2, r3, r4, r5, r6, r7],
                     [
                         's2s', 'v2s all frames', 'v2s filter: Eyes Distance', 'v2s filter: Bounding Box',
                         'v2s filter: Feature Norm', 'v2s filter: ED + BB + FN', 'v2s max single (ED + BB + FN)',
                      ], 'Webcams', k=k_vals)

    r1 = s2s_identification_webcams(embedding_file, query_gallery_split, detection_dir, iden_k_largest_bb, filters=['bb_size', 'eyes_dist', 'feat_norm'])
    plot_recall_at_k([r3, r1, r6],
                         ['s2s - random',
                         's2s - random best 20', 'v2s - V0'
                      ], 'Webcams', k=k_vals)


def filter_for_identification(positive_pairs, detected_videos, manual_video_score_file):
    ids_vids_lists = find_connected_components(positive_pairs)
    # filtering the videos in the components that don't have any face detection with score over detection_threshold
    ids_vids_lists_detected = [[curr_id_vids for curr_id_vids in id_vids if curr_id_vids in detected_videos] for id_vids in
                               ids_vids_lists]
    # after the filtering, there might be ids with less than 2 videos (so they can't be in the closet-set identification),
    # here we filter them out:
    ids_vids_lists_detected = [np.array(l) for l in ids_vids_lists_detected if len(l) > 1]
    # filter videos without visible face (for detection threshold 0.9):
    with open(manual_video_score_file, 'rb') as h:
        info = pickle.load(h)
    manual_vid_filtered = []
    for id_vids in ids_vids_lists_detected:
        curr_id_filtered = []
        for vid in id_vids:
            if (vid in info['vid']) and (info['manual_score'][info['vid'] == vid] > 0.1):
                curr_id_filtered.append(vid)
        manual_vid_filtered.append(curr_id_filtered)
    ids_vids_lists_detected = [np.array(l) for l in manual_vid_filtered if len(l) > 1]

    return ids_vids_lists_detected


class AggregateTransformer:
    def __init__(self):
        from mycode.feats_agg.models import Aggregator
        from torchvision.transforms import ToTensor
        ckpt = '/outputs/bionicEye/feats_agg/mlruns/771810300347970854/da8c5196574f409185eb5460fe7edb6e/artifacts/checkpoints/epoch=43-val_rank1=0.998.ckpt'
        self.model = Aggregator.load_from_checkpoint(ckpt)
        self.transform = ToTensor()


    def __call__(self, feats):
        feats = self.transform(feats)
        out = self.model(feats.to(self.model.device)).detach().cpu().numpy().squeeze()
        return out


def main():
    base_path = '/inputs/bionicEye/webcams/test'
    detection_dir = os.path.join(base_path, 'face_detection')
    embedding_file = os.path.join(base_path, 'embeddings.pickle')
    pairs = pd.read_csv(os.path.join(base_path, 'annotations/benchmark_pairs_v1.csv'))
    manual_video_score_file = os.path.join(base_path, 'annotations/manual_score.pickle')
    detection_threshold = 0.9
    do_detection = False
    do_feature_extraction = False
    plot_iden_res = False
    iden_k_best = 20
    aggregate_frames = AggregateTransformer()

    # get fixed positive pairs and negative pairs. not necessarily have face detected.
    # with open(os.path.join(base_path, 'annotations/test_idx.pickle'), 'rb') as h:
    #     index = pickle.load(h)
    index = pd.read_pickle(os.path.join(base_path, 'annotations/test_idx.pickle'))
    pos = pairs.loc[index['pos']]
    neg = pairs.loc[index['neg']]

    unique_vids = np.unique(np.concatenate([pos['query'], pos['gallery'], neg['query'], neg['gallery']]))

    if do_detection:
        preprocess(unique_vids, detection_dir)

    # Feature Extraction (the output consists of videos that have at least one frame detected)
    if not do_feature_extraction and os.path.isfile(embedding_file):
        with open(embedding_file, 'rb') as h:
            stats = pickle.load(h)
        feats, vids, frame_idx = stats['feats'], stats['vids'], stats['frame_idx']
    else:
        feats, vids, frame_idx = features(unique_vids, detection_dir)
        with open(embedding_file, 'wb') as h:
            pickle.dump({'feats': feats, 'vids': vids, 'frame_idx': frame_idx}, h)
        print('Saved embeddings in: ', embedding_file)


    # Face Identification
    feats, vids, frame_idx = filter_detections_by_threshold(feats, vids, frame_idx, detection_dir, detection_threshold)
    pos_pairs = np.array([pos['query'], pos['gallery']]).T
    ids_vids_lists_detected = filter_for_identification(pos_pairs, vids, manual_video_score_file)
    query_gallery_split = create_identification_splits(ids_vids_lists_detected, detection_dir, detection_threshold)

    # calc_eyes_distance(detection_dir, query_gallery_split, vids, frame_idx)

    rank_v2s = v2s_identification_webcams(embedding_file, query_gallery_split, detection_dir, iden_k_best,
                                          filters=['bb_size', 'eyes_dist', 'feat_norm'], plot_tp_n_fp=plot_iden_res)
    rank_v2s_agg = v2s_identification_webcams(embedding_file, query_gallery_split, detection_dir, iden_k_best,
                                          filters=['bb_size', 'eyes_dist', 'feat_norm'], plot_tp_n_fp=plot_iden_res, aggregate=aggregate_frames)
    rank_s2s = s2s_identification_webcams(embedding_file, query_gallery_split, detection_dir, iden_k_best,
                                          filters=['bb_size', 'eyes_dist', 'feat_norm'])
    plot_recall_at_k([rank_s2s, rank_v2s, rank_v2s_agg], ['S2S', 'V2S', 'Transformer Aggregation'], 'Webcams\nbest 20 frames filter: ED + BB + FN', k=k_vals)

    # compare_results(embedding_file, query_gallery_split, detection_dir)


if __name__ == "__main__":
    seed = 7
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    main()



