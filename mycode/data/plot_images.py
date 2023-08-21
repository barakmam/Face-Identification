import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from os.path import join
from glob import glob
from tqdm import tqdm
import deepface
from deepface.detectors import FaceDetector
import xml.etree.ElementTree as et
import pandas as pd
from PIL import Image
import cv2


def face_crop(img, src_bb, dst_bb_w, dst_bb_h):
    """
    crop face with bb of size dst_bb centered in src_bb center.
    src_bb: [[left_up(col),left_up(row)],
         [left_down(col),left_down(row)],
         [right_up(col),right_up(row)],
         [right_down(col),right_down(row)]]
    dst_bb_w, dst_bb_h: width, height
    out_bb: [left_up(col), left_up(row), w, h]
    """
    face_center = [src_bb[:, 0].mean(), src_bb[:, 1].mean()]  # col, row
    x0 = min(max(0, face_center[0] - dst_bb_w/2), img.shape[1] - dst_bb_w)
    y0 = min(max(0, face_center[1] - dst_bb_h/2), img.shape[0] - dst_bb_h)
    out_bb = np.array([
        x0,
        y0,
        dst_bb_w,
        dst_bb_h
    ]).round().astype(np.uint16)
    crop = img[out_bb[1]:(out_bb[1] + out_bb[3]), out_bb[0]:(out_bb[0] + out_bb[2])]
    return crop


def create_video_from_frames(images, vid_dir, video_name):
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    fps = 24
    video = cv2.VideoWriter(join(vid_dir, video_name), 0, fps, (width, height))

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()


##### Youtube videos frames #####

# plot image:
# src_path = '/Users/barakm/Code/bionic-eye/datasets/youtube-faces/all_videos'
# dst_path = '/Users/barakm/Code/bionic-eye/code_files/DBVSR/dataset/my_data/sharp'
# all_vid_names = os.listdir(src_path)
# vid_name = all_vid_names[0]
# data = np.load(join(src_path, vid_name))
# n_frames = data['boundingBox'].shape[-1]
#
# # for vid_path in all_vid_names:
# #     data = np.load(vid_path)
# #     n_frames = data['boundingBox'].shape[-1]
# #     # frame_idx = 10
# os.makedirs(join(dst_path, vid_name.split('.')[0] + '_full_size'), exist_ok=True)
# os.makedirs(join(dst_path, vid_name.split('.')[0] + '_face'), exist_ok=True)
#
# # arange images in size that is dividable by 4 to insert DBVSR transformer (SR by 4)
# dst_bb_w = np.max(data['boundingBox'][2, 0] - data['boundingBox'][0, 0])
# dst_bb_h = np.max(data['boundingBox'][1, 1] - data['boundingBox'][0, 1])
# dst_bb_w += (4 - dst_bb_w % 4) % 4  # size should be dividable by 4
# dst_bb_h += (4 - dst_bb_w % 4) % 4  # size should be dividable by 4
# original_w = data['colorImages'].shape[1]
# original_h = data['colorImages'].shape[0]
# original_w += (4 - original_w % 4) % 4  # size should be dividable by 4
# original_h += (4 - original_h % 4) % 4  # size should be dividable by 4
# for frame_idx in range(0, n_frames):
#     Image.fromarray(data['colorImages'][:original_h, :original_w, :, frame_idx]).save(join(dst_path, vid_name.split('.')[0] + '_full_size', f'{frame_idx}'.zfill(4) + '.png'))
#     # plt.figure(), plt.imshow(data['colorImages'][:, :, :, frame_idx])
#     # plt.gca().add_patch(Rectangle((data['boundingBox'][0, 0, frame_idx], data['boundingBox'][0, 1, frame_idx]),
#     #                               data['boundingBox'][2, 0, frame_idx] - data['boundingBox'][0, 0, frame_idx],
#     #                               data['boundingBox'][1, 1, frame_idx] - data['boundingBox'][0, 1, frame_idx],
#     #                               linewidth=1, edgecolor='r', facecolor='none'))
# # print(filename'Width: {data["colorImages"].shape[1]}, Height: {data["colorImages"].shape[0]}')
#
# # bb limits:
# #     row_start = int(data['boundingBox'][0, 1, frame_idx])
# #     row_end = int(data['boundingBox'][1, 1, frame_idx])
# #     col_start = int(data['boundingBox'][0, 0, frame_idx])
# #     col_end = int(data['boundingBox'][2, 0, frame_idx])
#     face = face_crop(data['colorImages'][..., frame_idx], data['boundingBox'][..., frame_idx], dst_bb_w, dst_bb_h)
#     Image.fromarray(face).save(join(dst_path, vid_name.split('.')[0] + '_face', f'{frame_idx}'.zfill(4) + '.png'))
#     # plt.figure(), plt.imshow(face), plt.title(filename'face idx: {frame_idx}'), plt.show()
#     # down_sampled = face[::4, ::4]
#     # plt.figure(), plt.imshow(down_sampled), plt.title(filename'down_sampled idx: {frame_idx}'), plt.show()
#
#
# # images = sorted(glob('/Users/barakm/Code/bionic-eye/code_files/DBVSR/dataset/my_data/blurdown_x4/Lisa_Leslie_5_full_size/*'))[2:-2]
# # create_video_from_frames(images, '/Users/barakm/Code/bionic-eye/code_files/DBVSR/dataset/my_data/blurdown_x4', 'Lisa_Leslie_5_full_size.avi')
# # run over images and save faces
# # info = pd.read_csv('/Users/barakm/Code/bionic-eye/datasets/youtube-faces/youtube_faces_with_keypoints_full.csv')
# # info = info[info['averageFaceSize'] > 100]
# # for ii in range(len(info)):
# #     vid = []
# #     all_frames = np.load(os.path.join('/Users/barakm/Code/bionic-eye/datasets/youtube-faces/all_videos',
# #                                       info.iloc[ii]['videoID'] + '.npz'))
# #     for frame_idx in all_frames.shape[-1]:
# #         row_start = int(data['boundingBox'][0, 1, 0])
# #         row_end = int(data['boundingBox'][1, 1, 0])
# #         col_start = int(data['boundingBox'][0, 0, 0])
# #         col_end = int(data['boundingBox'][2, 0, 0])
#
#
#
# # w = []
# # h = []
# # for vid in tqdm(all_vid_names):
# #     curr_vid = np.load(vid)
# #     w.append(curr_vid["colorImages"].shape[1])
# #     h.append(curr_vid["colorImages"].shape[0])
# #


##### ChokePoint frames #####
full = plt.imread('/Users/barakm/Code/bionic-eye/datasets/ChokePoint/P1E_S1/P1E_S1_C1/00000242.jpg')
plt.imshow(full)
detection_model = FaceDetector.build_model('retinaface')
detected = FaceDetector.detect_faces(detection_model, 'retinaface', full, False)
if not isinstance(detected, list):
    detected = [detected]
for face, region in detected:
    plt.gca().add_patch(Rectangle((region[0], region[1]), region[2], region[3], linewidth=1,
                                  edgecolor='r', facecolor='none'))
    # print(filename, 'rect width: {region[2]}, height: {region[3]}')
plt.show()
#
# # crop faces:
# data_dir = '/Users/barakm/Code/bionic-eye/datasets/ChokePoint'
# frames_set = 'P1E_S1'
# camera = 'C1'
#
# gt = et.parse(os.path.join(data_dir, 'groundtruth', frames_set + '_' + camera + '.xml')).getroot()
# for frame in gt:
#     if frame.find('person'):
#         full = plt.imread(os.path.join(data_dir, frames_set, frames_set + '_' + camera,
#                                        frame.attrib['number'] + '.jpg'))
#         detected = FaceDetector.detect_faces(
#             FaceDetector.build_model('retinaface'), 'retinaface', full, False)
#         if isinstance(detected, list):
#             # if there are multiple faces, choose the labeled one
#             for ii in range(len(detected)):
#                 region = detected[ii][1]
#                 right_eye_loc = frame.find('person/rightEye').attrib
#                 if region[0] < right_eye_loc['y'] < region[0] + region[2] and\
#                         region[1] < right_eye_loc['x'] < region[1] + region[3]:
#                     detected = detected[ii]
#                     break
#         face, region = detected
#         id = frame.find('person').attrib['id']
#
