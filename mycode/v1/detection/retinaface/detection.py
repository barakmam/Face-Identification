import cv2
import numpy as np
from glob import glob
from tqdm import tqdm


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


def clean_detection(im, bbs, score=None):
    # if there were several different faces, use the largest face in the image (suppose this is the wanted person)
    if len(bbs) == 1:
        return 0
    bb_size = bbs[:, 2]*bbs[:, 3]
    largest = bb_size.argmax()
    return largest


def detection(images_dir, detector, thresh=0.2):
    images = glob(images_dir + '/*')
    print(f'Face Detection on {images_dir} with {len(images)} images ...')
    curr_lmk = []
    curr_score = []
    curr_bb = []
    for f in images:
        image = cv2.imread(f)
        # bbs format: [x1, y1, x2, y2]
        faces, landmarks = detector.detect(image, thresh)
        if len(landmarks) == 0:
            curr_lmk.append(np.ones((5, 2), dtype='float32')*-1)
            curr_score.append(0)
            curr_bb.append(np.ones(4, dtype='float32')*-1)
        else:
            pick = non_max_suppression_fast(faces[:, :-1], faces[:, -1])
            bbs = faces[pick, :-1]
            score = faces[pick, -1]
            landmarks = landmarks[pick]
            clean_ind = clean_detection(image, bbs, score)
            curr_lmk.append(landmarks[clean_ind])
            curr_score.append(score[clean_ind])
            bb_xywh = np.array([*bbs[clean_ind, :2],
                                bbs[clean_ind, 2] - bbs[clean_ind, 0],
                                bbs[clean_ind, 3] - bbs[clean_ind, 1]])
            curr_bb.append(bb_xywh.astype('int'))

    curr_lmk = np.array(curr_lmk)
    curr_score = np.array(curr_score)
    curr_bb = np.array(curr_bb)
    print(f'DONE: {images_dir}. detections/images = {sum(curr_score > 0)}/{len(images)}')

    return curr_bb, curr_lmk, curr_score
