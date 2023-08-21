import pickle

import cv2
import sys

import numpy as np
import os
import glob

from tqdm import tqdm

from retinaface import RetinaFace

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'  # to avoid the process off "Running performance tests to find the best convolution algorithm"

thresh = 0.2

count = 1

gpuid = 0
detector = RetinaFace('/inputs/bionicEye/insightFace/models/detection/retinaface-R50/R50', 0, gpuid, 'net3')


out_dict = {'info': 'extracted by Retina50 from insightFace Git', 'landmarks': {}, 'score': {}}
data_path = '/datasets/BionicEye/YouTubeFaces/faces/bicubic_lr_scale_8'
vids = os.listdir(os.path.join(data_path, 'videos'))

print('Extracting Landmarks for: ', data_path)


def parallel(vid, detector):
    frames = sorted(glob.glob(os.path.join(data_path, 'videos', vid, '*')))
    curr_lmk = []
    curr_score = []
    for f in frames:
        im = cv2.imread(f)
        faces, landmarks = detector.detect(im, thresh)
        # take the one with the best score (which is the first one)
        if len(landmarks) == 0:
            curr_lmk.append(np.ones((5, 2), dtype='float32')*-1)
            curr_score.append(0)
        else:
            curr_lmk.append(landmarks[0])
            curr_score.append(faces[0][-1])
    out_dict['landmarks'][vid] = np.stack(curr_lmk)
    out_dict['score'][vid] = np.array(curr_score)
    print('Done with ', vid)


for vid_name in tqdm(vids):
    parallel(vid_name, detector)

with open(os.path.join(data_path, 'detection.pickle'), 'wb') as h:
    pickle.dump(out_dict, h)
print('######## Saved Detection info in: ', os.path.join(data_path, 'detection.pickle'))
print('DONE!')
