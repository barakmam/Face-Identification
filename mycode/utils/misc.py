import time
from datetime import datetime
import cv2
from omegaconf import DictConfig
import os
from hydra import compose, initialize
import io
from PIL import Image, ImageDraw
import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob
from torchvision import transforms
from hydra.core.global_hydra import GlobalHydra
import hydra
from tqdm import tqdm
import pickle
import scipy.io as sio
from itertools import compress


def count_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_trained_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#Params: ', total_params)
    print('#Trainable Params: ', total_trained_params)
    return


def filter_videos(embedding_path, state, detection_file, head_pose_file=None,  num_frames=None,
                  detection_filters=None, splits_path='/inputs/bionicEye/data/ytf/splits'):
    feats, id_, filename = \
        filter_videos_by_videoname(embedding_path,
                                   filter_videoname_path=os.path.join(splits_path, f'{state}.pickle'))

    if detection_filters:
        with open(detection_file, 'rb') as h:
            detection = pickle.load(h)
        filtered_inds = get_combined_filters_inds(feats, filename, detection, detection_filters, num_frames)
        feats = [feats[ii][filtered_inds[ii]] for ii in range(len(feats))]
        filename = [filename[ii][filtered_inds[ii]] for ii in range(len(filename))]
        return feats, id_, filename

    # if score_filter >= 0:
    with open(detection_file, 'rb') as h:
        detection = pickle.load(h)
    # only filter by score
    filter_inds = get_score_inds(filename, detection['score'], 0.01)
    if head_pose_file:
        assert num_frames is not None, "num_frames must not be None"
        with open(head_pose_file, 'rb') as h:
            head_pose = pickle.load(h)
        frontal_inds = get_frontal_inds(filename, head_pose, num_frames, filter_inds)
        feats = np.stack([feats[ii][frontal_inds[ii]] for ii in range(len(feats))]).squeeze()
        filename = [filename[ii][frontal_inds[ii]] for ii in range(len(filename))]
        return feats, id_, filename, frontal_inds

    feats = [feats[ii][filter_inds[ii]] for ii in range(len(feats)) if len(filter_inds[ii]) > 0]
    filename = [filename[ii][filter_inds[ii]] for ii in range(len(filename)) if len(filter_inds[ii]) > 0]
    id_ = np.array([id_[ii] for ii in range(len(id_)) if len(filter_inds[ii]) > 0])
    return feats, id_, filename

    # if head_pose_file:
    #     assert num_frames is not None, "num_frames must not be None"
    #     with open(head_pose_file, 'rb') as h:
    #         head_pose = pickle.load(h)
    #     frontal_inds = get_frontal_inds(filename, head_pose, num_frames)
    #     feats = np.stack([feats[ii][frontal_inds[ii]] for ii in range(len(feats))]).squeeze()
    #     filename = [filename[ii][frontal_inds[ii]] for ii in range(len(filename))]
    #     return feats, id_, filename, frontal_inds

    return feats, id_, filename


def filter_templates(embedding_path, state, splits_path='/inputs/bionicEye/ms1m-retinaface-t1/splits', min_num_images=None, max_num_images=150):
    feats, id_, filename = \
        filter_templates_by_ids(embedding_path, filter_ids_path=os.path.join(splits_path, state, 'all.pickle'))
    filter_num_images_inds = [min_num_images <= len(template) <= max_num_images for template in feats]
    feats = list(compress(feats, filter_num_images_inds))
    id_ = id_[filter_num_images_inds]
    filename = list(compress(filename, filter_num_images_inds))

    return feats, id_, filename


def filter_templates_by_ids(embeddings_filename, filter_ids_path):
    with open(embeddings_filename, 'rb') as handle:
        embeddings = pickle.load(handle)
    with open(filter_ids_path, 'rb') as handle:
        filter = pickle.load(handle)
    inds = np.in1d(embeddings['id'], filter['ids'])
    return list(compress(embeddings['feats'], inds)), \
           embeddings['id'][inds], \
           list(compress(embeddings['filename'], inds))


def filter_videos_by_videoname(embeddings, filter_videoname_path):
    if isinstance(embeddings, str):
        with open(embeddings, 'rb') as handle:
            embeddings = pickle.load(handle)
    with open(filter_videoname_path, 'rb') as handle:
        videoname = pickle.load(handle)
    inds = np.in1d(np.array([name[0].split('/')[-2] for name in embeddings['filename']]), videoname)
    return list(compress(embeddings['feats'], inds)), \
           embeddings['id'][inds], \
           list(compress(embeddings['filename'], inds))


def get_score_inds(filenames, score, thresh=0.0):
    vid_names = [frames_names[0].split('/')[-2] for frames_names in filenames]
    pos_score_inds = []
    for name in vid_names:
        pos_score_curr_idx = np.where(score[name] > thresh)[0]
        pos_score_inds.append(pos_score_curr_idx)
    return pos_score_inds


def get_combined_filters_inds(feats, filenames, detection_dict, filters, num_frames):
    vid_names = [frames_names[0].split('/')[-2] for frames_names in filenames]
    selected_inds = []
    for ii, name in enumerate(vid_names):
        filters_list = []
        if 'eyes_dist' in filters:
            ed = np.linalg.norm(detection_dict['landmarks'][name][:, 0] - detection_dict['landmarks'][name][:, 1], axis=-1)
            filters_list.append(ed/max(ed))
        if 'bb_size' in filters:
            bb_size = detection_dict['bb'][name][:, 2] * detection_dict['bb'][name][:, 3]
            filters_list.append(bb_size / max(bb_size))
        if 'feat_norm' in filters:
            norm = np.linalg.norm(feats[ii], axis=-1)
            filters_list.append(norm / max(norm))

        filters_list = np.stack(filters_list)

        all_inds = np.arange(len(feats[ii]))
        if 'eyes_dist' or 'bb_size' in filters:
            pos_score_curr_idx = np.where(detection_dict['score'][name] > 0)[0]
            all_inds = all_inds[pos_score_curr_idx]

        max_inds_len = min(num_frames, len(all_inds))  # if there are less frames than k_largest_bb, take all frames.
        inds_of_all_inds = np.argpartition(filters_list.sum(0)[all_inds], -max_inds_len)[-max_inds_len:]
        selected_inds.append(all_inds[inds_of_all_inds])
    return selected_inds


def get_query_feats(embeddings_filename, splits_path='/inputs/bionicEye/data/ytf_new_splits'):
    return filter_templates_by_ids(embeddings_filename, filter_ids_path=os.path.join(splits_path, 'test/query.pickle'))


def get_gallery_feats(embeddings_filename, splits_path='/inputs/bionicEye/data/ytf_new_splits'):
    return filter_templates_by_ids(embeddings_filename,
                                   filter_ids_path=os.path.join(splits_path, 'test/gallery.pickle'))


def get_train_feats(embeddings_filename, splits_path='/inputs/bionicEye/data/ytf_new_splits'):
    return filter_templates_by_ids(embeddings_filename, filter_ids_path=os.path.join(splits_path, 'train/train.pickle'))


def compare_models(model_1, model_2):
    """
    Compares the models weights
    """
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print('Mismatch found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


def print_dict(d, ii=0):
    if ii == 0:
        print('CONFIG:')
    for key, val in d.items():
        if isinstance(val, DictConfig):
            print('\t' * ii, key)
            print_dict(val, ii + 1)
        else:
            print('\t' * ii, key, ':\t', val)


def read_cfg(cfg_dir, cfg_name):
    # GlobalHydra.instance().clear()
    # print('##################################################################')
    # print('TRY')
    # print('##################################################################')
    # print(os.path.relpath(cfg_dir))
    # initialize(config_path=os.path.relpath(cfg_dir))
    # cfg = compose(config_name=cfg_name)
    try:
        GlobalHydra.instance().clear()
        print('##################################################################')
        print('TRY')
        print('##################################################################')
        print(os.path.relpath(cfg_dir))
        initialize(config_path=os.path.relpath(cfg_dir))
        cfg = compose(config_name=cfg_name)
    except hydra.errors.MissingConfigException:
        print('##################################################################')
        print('EXCEPT')
        print('##################################################################')
        GlobalHydra.instance().clear()
        initialize(config_path='../../' + os.path.relpath(cfg_dir))
        cfg = compose(config_name=cfg_name)
    return cfg


def plot_counts_hist(count_majority_vote, majority_vote_equals):
    """
    hist of counts for the target video.
    """
    fig = plt.figure()
    plt.bar(*np.unique(count_majority_vote.cpu().numpy(), return_counts=True), label='Total')
    plt.bar(*np.unique(count_majority_vote[majority_vote_equals].cpu().numpy(), return_counts=True), label='Correct')
    plt.title('Target Majority Vote Count Hist', fontsize=17), plt.xlabel('# Frame Votes', fontsize=17)
    plt.ylabel('# Query Videos', fontsize=17), plt.grid(), plt.legend()
    return fig


def plot_match(query_path, gallery_path, score=None):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(plt.imread(query_path))
    ax[0].set_title(f'Query: {query_path.split("/")[-2]}'), ax[0].axis('off')
    ax[1].imshow(plt.imread(gallery_path))
    ax[1].set_title(f'Gallery Match: {gallery_path.split("/")[-2]}'), ax[1].axis('off')
    if score:
        fig.suptitle(f'Score: {score}')
    # plt.axis('off')


def save_vid_from_images(images, video_out_path, size=None, fps=2):
    """
    :param size: (width, height)
    :return:
    """
    if size is None:
        size = images[0].shape[:2][::-1]
    video = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    for frame in images:
        if frame.dtype != 'uint8':
            frame = (frame.clip(0, 1) * 255).astype(np.uint8)
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), size)
        video.write(frame)
    video.release()
    cv2.destroyAllWindows()
    print('\n*** Video saved in ', video_out_path, '***')


def save_vid_from_folder(vid_folder, video_out_path, fps=2, size=None):
    frames = glob(os.path.join(vid_folder, '*'))
    images = []
    for f in frames:
        images.append(plt.imread(f))
    save_vid_from_images(images, video_out_path, size, fps)


def concat_videos(res=112):
    dbvsr_files = sorted(glob('/datasets/BionicEye/YouTubeFaces/faces/blurred_sharp/videos/Sarah_Michelle_Gellar_4/*'))
    video_inr_files = sorted(
        glob('/datasets/BionicEye/YouTubeFaces/faces/sharp/videos/Sarah_Michelle_Gellar_4/*'))
    # raw_padded = sorted(glob('/inputs/bionicEye/webcams/faces/abbeyroad_uk_01_02_2022_14_41/id_2/padded/*')[2:-3])
    # raw_scaled = sorted(glob('/inputs/bionicEye/webcams/faces/abbeyroad_uk_01_02_2022_14_41/id_2/scaled/*')[2:-3])
    # degraded = sorted(glob('/datasets/BionicEye/YouTubeFaces/faces/blurdown_x4/Alison_Lohman_0/*')[2:-2])

    description = ['Blurred', 'Sharp']#, 'Bicubic', 'Raw']  # , 'Degraded']
    all_methods = [dbvsr_files, video_inr_files]#, raw_scaled, raw_padded]  # , degraded]
    n_frames = min(len(dbvsr_files), len(video_inr_files))#, len(raw_scaled), len(raw_padded))  # , len(degraded))

    dst_path = '/outputs/bionicEye/videos'

    all_frames = []
    for frame_idx in tqdm(range(n_frames)):
        frame = []
        for method_idx in range(len(all_methods)):
            image = plt.imread(all_methods[method_idx][frame_idx])
            if image.shape[2] == 4:
                # in case the image saved with PIL or other libraray that adds mask to the image channels
                image = image[..., :3]
            if image.dtype != 'uint8':
                image = (image.clip(0, 1) * 255).astype(np.uint8)
            if description[method_idx] == 'Bicubic' and (image.shape[0] != res or image.shape[1] != res):
                # create and add the Bicubic frame
                image = np.array(Image.fromarray(image).resize((res, res)))
            if description[method_idx] == 'Degraded':
                padded_face = Image.new('RGB', (res, res))
                padded_face.paste(Image.fromarray(image), ((res - image.shape[0]) // 2, (res - image.shape[1]) // 2))
                image = np.array(padded_face)

            image = Image.fromarray(image).resize((res, res))
            ImageDraw.Draw(image).text(
                (0, 0),  # Coordinates
                description[method_idx],  # Text
                (0, 170, 240)  # color: Cerulean (kind of clue color)
            )
            frame.append(np.array(image))

        all_frames.append(np.concatenate(frame, axis=1))

    os.makedirs(dst_path, exist_ok=True)
    save_vid_from_images(all_frames, os.path.join(dst_path, 'graduate_blur_comparison.avi'), fps=20)


def get_frontal_inds(filenames, head_pose, num_frames, relevant_inds=None):
    # relevant_inds should be in the same order as vid_names. i.e. relevant_inds[ii] relates to vid_names[ii]
    vid_names = [frames_names[0].split('/')[-2] for frames_names in filenames]
    canonical_frame_inds = []
    for ii, name in enumerate(vid_names):
        sorted_frontal_frames_inds = np.argsort(abs(head_pose[name]['yaw'])\
                                                + abs(head_pose[name]['pitch'])
                                                + abs(head_pose[name]['roll']))
        if relevant_inds is not None:
            sorted_frontal_frames_inds = sorted_frontal_frames_inds[np.in1d(sorted_frontal_frames_inds, relevant_inds[ii])]

        canonical_frame_inds.append(sorted_frontal_frames_inds[:num_frames])
    return canonical_frame_inds


def prepare_rep_ytf(embedding_q, head_pose, embedding_g=None):
    """
    from all embeddings we create the data for the REP-YTF test.
    We need to prepare 2 matrices:
    1. features of each video. shape [3425, D]. each row is the video representative feature vector (e.g. mean of all frames features)
    2. features of gallery image from each vide. shape [3425, D]. each row is feature vector of one frame from the video.
     this will be used in video2single scenario and the video is in the gallery (so we need one image from the gallery video)
     there is a list of the frames chosen for the gallery in the config file.

    :param head_pose: head pose of each frame in each video
    :param embedding_q: filename in the server.
    :return:
    """

    if embedding_g is None:
        embedding_g = embedding_q

    with open(embedding_q, 'rb') as h:
        my_stats_q = pickle.load(h)
    with open(embedding_g, 'rb') as h:
        my_stats_g = pickle.load(h)
    my_vid_names = np.array([names[0].split('/')[-2] for names in my_stats_q['filename']])

    frontal_inds = get_frontal_inds(my_stats_q['filename'], head_pose, num_frames=50)
    # frontal_inds has for each video the most frontal inds sorted from the most frontal to the least [num_videos, num of frontal inds]

    # rep_ytf_config = sio.loadmat('/Users/barakm/Code/bionic-eye/REP-YTF/config/ytf/rep_ytf_config.mat')  # run from my pc
    rep_ytf_config = sio.loadmat('/inputs/bionicEye/data/ytf/rep_ytf/config/ytf/rep_ytf_config.mat')  # run from server

    video_descriptors = []
    gallery_most_frontal_desc = []
    for ii in range(len(rep_ytf_config['videoList'])):
        vid_name = rep_ytf_config['videoList'][ii][0][0].replace('/', '_')
        correspond_my_idx = np.where(vid_name == my_vid_names)[0].item()
        # I append the mean of the 50 frontal frames in the curr video.
        video_descriptors.append(my_stats_q['feats'][correspond_my_idx][frontal_inds[correspond_my_idx]].mean(0))

        # get the frame number for gallery from the config
        gallery_frame_idx = np.where([int(rep_ytf_config['imgGalleryList'][ii][0][0].split('/')[-1].split('.')[1]) ==
                                      int(name.split('/')[-1].split('.')[-2]) for name in my_stats_g['filename'][correspond_my_idx]])[0].item()
        gallery_most_frontal_desc.append(my_stats_g['feats'][correspond_my_idx][gallery_frame_idx])

    dir_name = 'exp10-' + datetime.now().strftime("%d_%m_%Y-%H_%M")
    details = 'YTF bbs\n' \
              'avg of 50 frontal frames using all frames\n' \
              'head pose From YTF database\n' \
              'gallery best image from config\n' \
              'Normalized image\n' \
              'LR scale 4: query videos\n' \
              'ArcFace insightFace arch'

    our_dir = f'/outputs/bionicEye/rep-ytf/sharp/{dir_name}'
    os.makedirs(our_dir)
    sio.savemat(os.path.join(our_dir, 'Descriptors.mat'),
                {'Descriptors': np.vstack(video_descriptors)})
    sio.savemat(os.path.join(our_dir, 'DescriptorsImgGallery.mat'),
                {'DescriptorsImgGallery': np.vstack(gallery_most_frontal_desc)})
    with open(os.path.join(our_dir, 'details.txt'), "w") as f:
        f.write(details)


if __name__ == "__main__":
    # concat_videos(res=256)
    with open('/datasets/BionicEye/YouTubeFaces/head_pose/head_pose_supplied.pickle', 'rb') as h:
        head_pose_dict = pickle.load(h)
    prepare_rep_ytf(
        '/outputs/bionicEye/v1/extract-feats/bicdown-face-scale-4-norm-sim-r100_01-04-2023_19-54/embeddings.pickle',
        head_pose_dict,
    embedding_g='/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle')

    # save_vid_from_folder('/datasets/BionicEye/YouTubeFaces/faces/blurred_sharp/videos/Sarah_Michelle_Gellar_4',
    #                      '/outputs/bionicEye/videos/synthetically_blurred_video.avi', fps=10, size=(112, 112))

    # concat_videos()
