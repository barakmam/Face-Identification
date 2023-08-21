import os
import time

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from glob import glob
import scipy.io as sio
import pickle
from os.path import join

# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def face_crop_parallel_YTF(ytf_dir, dst_dir):
    """
    Use the supplied bounding boxes to crop the videos in YTF dataset.
    Enlarging each bounding box by 20% and then crop,

    creates new folder with all the cropped videos in different format.
    """

    def face_crop(filename):
        with open(join(src_dir, filename)) as fp:
            Lines = fp.readlines()
            if len(Lines) == 0:
                print(filename, ' has no bbs')
                return


            for line in Lines:
                splits = line.split(',')
                w = int(int(splits[4])*1.2)
                h = int(int(splits[5])*1.2)
                x0 = max(0, int(splits[2]) - w // 2)
                y0 = max(0, int(splits[3]) - h // 2)
                splits[0] = splits[0].replace("\\", '/')
                id_name = splits[0].split('/')[0]
                vid_num = splits[0].split('/')[1]
                vid_name = id_name + '_' + vid_num
                frame_num = splits[0].split('.')[1].zfill(5)
                frame = plt.imread(join(src_dir, splits[0]))
                face = frame[y0:(y0+h), x0:(x0+w)]
                os.makedirs(join(dst_dir, 'videos', vid_name), exist_ok=True)
                Image.fromarray(face).save(join(dst_dir, 'videos', vid_name, frame_num + '.png'))

    src_dir = os.path.join(ytf_dir, 'frame_images_DB')
    txt_files = glob(src_dir + '/*.txt')
    start = time.time()
    Parallel(n_jobs=128)(delayed(face_crop)(txtfile)
                         for txtfile in tqdm(sorted(txt_files)))
    # for txtfile in tqdm(sorted(txt_files)):
    #      face_crop(txtfile)

    print("Took: ", time.time() - start)


def downsample_bicubic_by_scale(src_dir, dst_dir, scale):
    """
        Use bicubic to downsample image by scale in each dimension
    """

    dst_dir = dst_dir + f'_scale_{scale}'

    def degrade(vid_full_path):
        vid_name = vid_full_path.split('/')[-1]
        # if os.path.exists(join(dst_dir, 'videos', vid_name)):
        #     return
        os.makedirs(join(dst_dir, 'videos', vid_name), exist_ok=True)
        for idx, frame in enumerate(glob(vid_full_path + '/*')):
            frame_name = frame.split('/')[-1]
            image = Image.open(frame)
            image = image.resize((image.size[0]//scale, image.size[1]//scale), Image.Resampling.BICUBIC)
            image.save(join(dst_dir, 'videos', vid_name, frame_name))

    print('Started function: downsample_bicubic, scale: ', scale, ' on: ', src_dir)
    vid_names = glob(join(src_dir, 'videos', '*'))
    # vid_names = [src_dir]
    start = time.time()
    Parallel(n_jobs=256)(delayed(degrade)(vid_name)
                         for vid_name in tqdm(sorted(vid_names)))
    # for vid_name in tqdm(sorted(vid_names)):
    #     degrade(vid_name)

    print("Took: ", time.time() - start)


def reformat_ytf_headpose_file(ytf_dir, full_output_filename='../ytf/head_pose_supplied.pickle'):
    vids_files = sorted(glob(os.path.join(ytf_dir, 'headpose_DB', '*')))
    head_pose = {}

    print('Starting reformatting head pose supplied...')
    for f in tqdm(vids_files):
        y, r, p = sio.loadmat(f)['headpose']  # I dont know if this is the order of pitch, yaw, roll.
        head_pose[f.split('/')[-1].split('.')[0].split('apirun_')[1]] = \
            {'yaw': p * np.pi/180, 'roll': y * np.pi/180, 'pitch': r * np.pi/180}

    with open(full_output_filename, 'wb') as h:
        pickle.dump(head_pose, h)
    print('Saved: ', full_output_filename)
    print('DONE!')


if __name__ == '__main__':
    ytf_dir_path = '/datasets/BionicEye/YouTubeFaces/raw'
    crop_dir = '/datasets/BionicEye/YouTubeFaces/faces/sharp'
    lr_dir = '/datasets/BionicEye/YouTubeFaces/faces/bicubic_lr'


    # face_crop_parallel_YTF(ytf_dir_path, crop_dir)

    # downsample_bicubic(crop_dir, lr_dir, res=14)

    # reformat_ytf_headpose_file(ytf_dir_path)
