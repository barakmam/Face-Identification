import pickle

import cv2
import sys

import numpy as np
import os
import glob

from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from mycode.v1.detection.retinaface.retinaface import RetinaFace
# from .retinaface import RetinaFace

thresh = 0.3

count = 1

gpuid = 0
detector = RetinaFace('/inputs/bionicEye/insightFace/models/detection/retinaface-R50/R50', 0, gpuid, 'net3')


out_dict = {'info': 'extracted by Retina50 from insightFace Git', 'landmarks': {}, 'score': {}, 'bb': {}}
scale = 4
data_path = f'/datasets/BionicEye/YouTubeFaces/full/dbvsr/bicubic_lr_scale_{scale}'
sharp_bb_path = '/datasets/BionicEye/YouTubeFaces/full/sharp/bb.pickle'
vids = os.listdir(os.path.join(data_path, 'videos'))
vids = ['Mariana_Ohata_3']

print('Extracting Landmarks for: ', data_path)


def non_max_suppression_fast(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # if probs:
    #     return boxes[pick].astype("int"), probs[pick]
    #
    # # return only the bounding boxes that were picked
    # return boxes[pick].astype("int")

    return pick


def overlap_area(bb_arr, bb_known):
    # bb format : [x, y, w, h]

    # Calculate the coordinates of the intersection rectangle
    x1 = np.maximum(bb_arr[:, 0], bb_known[0])
    y1 = np.maximum(bb_arr[:, 1], bb_known[1])
    x2 = np.minimum(bb_arr[:, 0] + bb_arr[:, 2], bb_known[0] + bb_known[2])
    y2 = np.minimum(bb_arr[:, 1] + bb_arr[:, 3], bb_known[1] + bb_known[3])

    # Calculate the area of the intersection rectangle
    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Calculate the area of each bounding box
    bb_area = bb_arr[:, 2] * bb_arr[:, 3]

    # Calculate the overlap area of each bounding box with the known bounding box
    overlap = intersection_area / bb_area

    return overlap


def parallel(vid, detector, sharp_bbs, scale_from_sharp):
    frames = sorted(glob.glob(os.path.join(data_path, 'videos', vid, '*')))
    curr_lmk = []
    curr_score = []
    curr_bb = []
    for ii, f in enumerate(frames):
        try:
            im = cv2.imread(f)
            faces, landmarks = detector.detect(im, thresh)
            # bbs format: [x1, y1, x2, y2]
            # convert to [x, y, w, h]
            faces[:, :-1] = np.concatenate([faces[:, :2],
                                            faces[:, [2]] - faces[:, [0]],
                                            faces[:, [3]] - faces[:, [1]]], axis=1)
            # take the one with the best score (which is the first one)
            if len(landmarks) == 0:
                curr_lmk.append(np.ones((5, 2), dtype='float32')*-1)
                curr_score.append(0)
                curr_bb.append(np.array([-1]*4).astype('int'))
            else:
                inds = non_max_suppression_fast(faces[:, :-1])
                # if len(inds) == 1:
                #     curr_lmk.append(landmarks[inds[0]])
                #     curr_score.append(faces[inds[0], -1])
                #     curr_bb.append(faces[inds[0], :-1].astype('int'))
                #     continue
                # if there are several bbs detected, use the YTF supplied BB to determine which of the bb to use.
                # scale the bb that is known of the sharp data with the scaling that has been to the LR data.
                overlap = overlap_area(faces[inds, :-1], sharp_bbs[ii]/scale_from_sharp)
                if overlap.max() < 0.3:
                    # in case the target id (as given  by the YTF data) is not detected, don't use the b
                    curr_lmk.append(np.ones((5, 2), dtype='float32') * -1)
                    curr_score.append(0)
                    curr_bb.append(np.array([-1] * 4).astype('int'))
                else:
                    selected = inds[overlap.argmax()]
                    curr_lmk.append(landmarks[selected])
                    curr_score.append(faces[selected, -1])
                    curr_bb.append(faces[selected, :-1].astype('int'))
        except Exception as e:
            print('Error in ', f)
            print('The error is: ', e)
            continue

    out_dict['landmarks'][vid] = np.stack(curr_lmk)
    out_dict['score'][vid] = np.array(curr_score)
    out_dict['bb'][vid] = np.stack(curr_bb)
    print('Done with ', vid)


with open(sharp_bb_path, 'rb') as h:
    sharp_bbs = pickle.load(h)

for jj, vid_name in enumerate(vids):
    print('idx = ', jj, ' Processing ', vid_name)
    parallel(vid_name, detector, sharp_bbs[vid_name], scale/4)

with open(os.path.join(data_path, 'detection.pickle'), 'wb') as h:
    pickle.dump(out_dict, h)
print('######## Saved Detection info in: ', os.path.join(data_path, 'detection.pickle'))
print('DONE!')

# if faces is not None:
#     print('find', faces.shape[0], 'faces')
#     for i in range(faces.shape[0]):
#         #print('score', faces[i][4])
#         box = faces[i].astype(np.int)
#         #color = (255,0,0)
#         color = (0, 0, 255)
#         cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
#         if landmarks is not None:
#             landmark5 = landmarks[i].astype(np.int)
#             #print(landmark.shape)
#             for l in range(landmark5.shape[0]):
#                 color = (0, 0, 255)
#                 if l == 0 or l == 3:
#                     color = (0, 255, 0)
#                 cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color,
#                            2)
#
#     filename = './detector_test.jpg'
#     print('writing', filename)
#     cv2.imwrite(filename, img)

