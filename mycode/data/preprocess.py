import copy
import math
import os
import shutil
import time
import itertools
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import cv2
from glob import glob
import scipy.io as sio
import pickle
import logging
from os.path import join

# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#


def face_crop_parallel_YTF(src_dir, dst_dir, output_face_resolution=112):
    """
    works on the full YTF dataset
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

                if output_face_resolution:
                    face = cv2.resize(frame[y0:(y0+h), x0:(x0+w)], (output_face_resolution, output_face_resolution), interpolation=cv2.INTER_CUBIC)
                else:
                    face = frame[y0:(y0+h), x0:(x0+w)]
                os.makedirs(join(dst_dir, 'videos', vid_name), exist_ok=True)
                Image.fromarray(face).save(join(dst_dir, 'videos', vid_name, frame_num + '.png'))


    txt_files = glob(src_dir + '/*.txt')
    start = time.time()
    Parallel(n_jobs=128)(delayed(face_crop)(txtfile)
                         for txtfile in tqdm(sorted(txt_files)))
    # for txtfile in tqdm(sorted(txt_files)):
    #      face_crop(txtfile)

    print("Took: ", time.time() - start)


def rearange_parallel_YTF(src_dir, dst_dir):
    """
    works on the full YTF dataset and saves the data in another hierarchy
    """

    def parallel(filename):
        with open(join(src_dir, filename)) as fp:
            Lines = fp.readlines()
            if len(Lines) == 0:
                print(filename, ' has no bbs')
                return

            bb = {}
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
                os.makedirs(join(dst_dir, 'videos', vid_name), exist_ok=True)
                Image.fromarray(frame).save(join(dst_dir, 'videos', vid_name, frame_num + '.png'))
                if vid_name in bb.keys():
                    bb[vid_name].append(np.array([x0, y0, w, h]))
                else:
                    bb[vid_name] = [np.array([x0, y0, w, h])]
            os.makedirs(join(dst_dir, 'bb'), exist_ok=True)
            for k in bb.keys():
                bb[k] = np.vstack(bb[k])
            with open(join(dst_dir, 'bb', id_name + '.pickle'), 'wb') as h:
                pickle.dump(bb, h)

    txt_files = glob(src_dir + '/*.txt')
    print('function rearange_parallel_YTF started! with BB saving')
    start = time.time()
    Parallel(n_jobs=128)(delayed(parallel)(txtfile)
                         for txtfile in tqdm(sorted(txt_files)))
    # for txtfile in tqdm(sorted(txt_files)):
    #      face_crop(txtfile)
    bbs_files = glob(join(dst_dir, 'bb', '*'))
    bb_dict = {}
    for f in bbs_files:
        with open(f, 'rb') as h:
            curr_bb = pickle.load(h)
        bb_dict.update(curr_bb)
    with open(join(dst_dir, 'bb.pickle'), 'wb') as h:
        pickle.dump(bb_dict, h)
    print('bb dict saved in:', join(dst_dir, 'bb.pickle'))
    shutil.rmtree(join(dst_dir, 'bb'))

    print("Took: ", time.time() - start)


def crop_faces_with_mediapipe(src_dir, dst_dir):
    """
    src_dir: frames dir
    detecting with mediapipe
    """
    names = os.listdir(src_dir)
    os.makedirs(dst_dir, 'videos', exist_ok=True)

    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    detection_model = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    face_resolution = 112

    def crop_parallel(idx, filename):
        image = plt.imread(os.path.join(src_dir, filename))
        # image = image[300:, 1100:-300]
        # process the image with MediaPipe Face Detection.
        if 'float' in image.dtype.name:
            image = (image * 255).astype('uint8')
        results = detection_model.process(image)

        # Draw face detections of each face.
        if not results.detections:
            print('\n###### NO FACE DETECTED:', idx, filename, '######')
            return
        all_bb = []
        for detection in results.detections:
            all_bb.append([
                mp_drawing._normalized_to_pixel_coordinates(detection.location_data.relative_bounding_box.xmin,
                                                            detection.location_data.relative_bounding_box.ymin,
                                                            image.shape[1], image.shape[0]),
                mp_drawing._normalized_to_pixel_coordinates(detection.location_data.relative_bounding_box.width,
                                                            detection.location_data.relative_bounding_box.height,
                                                            image.shape[1], image.shape[0])
            ])
            (x, y), (w, h) = all_bb[-1]
            face = Image.fromarray(image[y:(y + h), x:(x + w)]).resize((face_resolution, face_resolution))
            face.save(os.path.join(dst_dir, 'videos', filename))


    with tqdm(enumerate(names), unit="images", desc=f"Cropping Faces") as t:
        start = time.time()
        # Parallel(n_jobs=128, backend='threading')(delayed(crop_parallel)(ii, filename)
        #                      for ii, filename in t)
        for ii, filename in t:
            crop_parallel(ii, filename)
        print("Took: ", time.time() - start)


def crop_ytf_with_retinaface(src_dir, dst_dir):
    """
    src_dir: videos dir of the YTF raw data
    detecting with retinaface
    (bb = bounding box)
    """
    txt_files_of_bb = sorted([name for name in os.listdir(src_dir) if name.endswith('.txt')])

    # os.makedirs(os.path.join(dst_dir, 'videos', 'scaled'), exist_ok=True)
    # os.makedirs(os.path.join(dst_dir, 'videos', 'padded'), exist_ok=True)

    from deepface.detectors import FaceDetector
    detection_model = FaceDetector.build_model('retinaface')
    face_resolution = 256
    scaling_size = 64

    logging.basicConfig(filename=dst_dir + '_log.log', level=logging.DEBUG, filemode='a')
    logging.info('Logging of crop_faces_with_retinaface, 11.1.23, crop square without alignment')

    def crop_parallel(idx, id_txt_file):
        with open(join(src_dir, id_txt_file)) as fp:
            Lines = fp.readlines()
            if len(Lines) == 0:
                print('idx ', idx, ' : ', id_txt_file, ' has no bbs')
                logging.info('idx ' + str(idx) + ' : ' + id_txt_file + ': has no bbs')
                return
            for line in Lines:
                splits = line.split(',')
                w = int(splits[4])
                h = int(splits[5])
                w = h = max(w, h)  # take square crop anyway.
                x0 = max(0, int(splits[2]) - w // 2)
                y0 = max(0, int(splits[3]) - h // 2)
                splits[0] = splits[0].replace("\\", '/')
                id_name = splits[0].split('/')[0]
                vid_num = splits[0].split('/')[1]
                vid_name = id_name + '_' + vid_num
                frame_num = splits[0].split('.')[1].zfill(5)
                if os.path.exists(os.path.join(dst_dir, 'videos', vid_name, frame_num + '.png')):
                    continue
                image = plt.imread(join(src_dir, splits[0]))
                # process the image with retinaface Face Detection.
                if 'float' in image.dtype.name:
                    image = (image * 255).astype('uint8')
                results = FaceDetector.detect_faces(detection_model, 'retinaface', image, align=True)

                if not results:
                    print('\n###### NO FACE DETECTED. idx:', idx, 'filename: ',
                          os.path.join(src_dir, splits[0]), '######')
                    logging.info(f'###### NO FACE DETECTED. idx:  {idx}, filename: {os.path.join(src_dir, splits[0])} ######')
                    # using the given bb:
                    cropped_aligned_face = Image.fromarray(image[y0:(y0 + h), x0:(x0 + w)])
                    os.makedirs(join(dst_dir, 'videos', vid_name), exist_ok=True)
                    cropped_aligned_face.save(
                        os.path.join(dst_dir, 'videos', vid_name, frame_num + '.png'))
                    continue

                relevant_bb_idx = 0
                if len(results) > 1:
                    print(f'\n******** MORE THAN ONE FACE DETECTED. idx:  {idx}, filename: , {os.path.join(src_dir, splits[0])}, ********')
                    logging.info(f'******** MORE THAN ONE FACE DETECTED. idx:  {idx}, filename: , {os.path.join(src_dir, splits[0])}, ********')

                    # find bb of the box that is closest to the supplied bb of the YTF dataset.
                    xy_res = []
                    for res in results:
                        xy_res.append(res[1][:2])
                    relevant_bb_idx = np.argmin(np.linalg.norm(np.array(xy_res) - np.array([x0, y0]), axis=1))

                x, y, w, h = results[relevant_bb_idx][1]  # detector bb
                w = h = max(w, h)  # crop square anyway
                x, y = max(0, x - round(0.1 * w)), max(0, y - round(0.1 * h))
                w, h = round(1.2 * w), round(1.2 * h)
                if math.sqrt((x - x0)**2 + (y - y0)**2) > 50:
                    logging.info(f'!!! Different Face Detected: idx {idx}, filename: {os.path.join(src_dir, splits[0])} !!!')

                cropped_square_face = Image.fromarray(image[y:(y+h), x:(x+w)])
                # save the face as it is (probably not square, and different size for each frame is possible)
                # Do the processing before inserting to models
                os.makedirs(join(dst_dir, 'videos', vid_name), exist_ok=True)
                cropped_square_face.save(
                    os.path.join(dst_dir, 'videos', vid_name, frame_num + '.png'))
            print(f'DONE: idx {idx}, file {id_txt_file}')

    # def crop_parallel(idx, id_name):
    #     if os.path.isfile(os.path.join(src_dir, id_name)):
    #         return
    #     vid_numbers = os.listdir(os.path.join(src_dir, id_name))
    #     for vid in vid_numbers:
    #         frames = os.listdir(os.path.join(src_dir, id_name, vid))
    #         if len(frames) == 0:
    #             return
    #         os.makedirs(os.path.join(dst_dir, 'videos', id_name + '_' + vid), exist_ok=True)
    #         for filename in frames:
    #             image = plt.imread(os.path.join(src_dir, id_name, vid, filename))
    #             # process the image with retinaface Face Detection.
    #             if 'float' in image.dtype.name:
    #                 image = (image * 255).astype('uint8')
    #             results = FaceDetector.detect_faces(detection_model, 'retinaface', image, align=True)
    #
    #             # Draw face detections of each face.
    #             if not results:
    #                 print('\n###### NO FACE DETECTED. idx:', idx, 'filename: ',
    #                       os.path.join(src_dir, id_name, vid, filename), '######')
    #                 logging.info('###### NO FACE DETECTED. idx:', idx, 'filename: ',
    #                              os.path.join(src_dir, id_name, vid, filename), '######')
    #                 return
    #             if len(results) > 1:
    #                 print('\n******** MORE THAN ONE FACE DETECTED. idx:', idx, 'filename: ',
    #                       os.path.join(src_dir, id_name, vid, filename), '********')
    #                 logging.info('******** MORE THAN ONE FACE DETECTED (taking the first one). idx:', idx, 'filename: ',
    #                              os.path.join(src_dir, id_name, vid, filename), '********')
    #             cropped_aligned_face = Image.fromarray(results[0][0])
    #             # save the face as it is (probably not square, and different size for each frame is possible)
    #             # Do the processing before inserting to models
    #             cropped_aligned_face.save(os.path.join(dst_dir, 'videos', id_name + '_' + vid, filename.split('.')[1].zfill(5) + '.png'))
    #
    #             # x, y, w, h = results[0][1]
    #             # w = h = max(w, h)  # square bb
    #             # x, y = x - int(0.1*w), y - int(0.1*w)
    #             # w = h = int(w * 1.2)
    #             # face = Image.fromarray(image[y:(y + h), x:(x + w)])
    #             # face.save(os.path.join(dst_dir, 'videos', id_name + '_' + vid, filename.split('.')[1].zfill(5) + '.png'))
    #
    #             # scaled_face = face.resize((scaling_size, scaling_size))
    #             # # pad face centered
    #             # padded_face = Image.new('RGB', (face_resolution, face_resolution))
    #             # padded_face.paste(face, ((face_resolution - h)//2, (face_resolution - w)//2))
    #             #
    #             # scaled_face.save(os.path.join(dst_dir, 'videos', 'scaled', video_name, filename))
    #             # padded_face.save(os.path.join(dst_dir, 'videos', 'padded', video_name, filename))

    with tqdm(enumerate(txt_files_of_bb), unit=" Videos", desc=f"Cropping Faces") as t:
        start = time.time()
        Parallel(n_jobs=64, backend='threading')(delayed(crop_parallel)(ii, txt)
                             for ii, txt in t)
        # for ii, txt in t:
        #     crop_parallel(ii, txt)
        print("Took: ", time.time() - start)
        logging.info(f'Took:  {time.time() - start}')


def retinaface_crop(src_dir, dst_dir):
    """
    crop square from each image using retina face
    :param src_dir: videos base dir
    :param dst_dir:
    :return:
    """
    vids = sorted(os.listdir(src_dir))
    from deepface.detectors import FaceDetector
    detection_model = FaceDetector.build_model('retinaface')

    logging.basicConfig(filename=dst_dir + '/log_tmp.log', level=logging.DEBUG, filemode='a')
    logging.info('Logging of retinaface_crop, 16.1.23, crop square from supplied aligned videos')

    def retinaface_crop_parallel(idx, vid_name):
        if os.path.isfile(os.path.join(src_dir, vid_name)):
            return
        frames = os.listdir(os.path.join(src_dir, vid_name))
        for f_name in frames:
            if os.path.exists(os.path.join(dst_dir, 'videos', vid_name, f_name)):
                continue
            image = plt.imread(join(src_dir, vid_name, f_name))
            # process the image with retinaface Face Detection.
            if 'float' in image.dtype.name:
                image = (image * 255).astype('uint8')
            results = FaceDetector.detect_faces(detection_model, 'retinaface', image, align=False)

            if not results:
                print('\n###### NO FACE DETECTED. idx:', idx, 'filename: ',
                      os.path.join(src_dir, vid_name, f_name), '######')
                logging.info(
                    f'###### NO FACE DETECTED. idx:  {idx}, filename: {os.path.join(src_dir, vid_name, f_name)} ######')
                continue
            if len(results) > 1:
                print(f'\n******** MORE THAN ONE FACE DETECTED. idx:  {idx}, filename: , {os.path.join(src_dir, vid_name, f_name)}, ********')
                logging.info(f'******** MORE THAN ONE FACE DETECTED. idx:  {idx}, filename: , {os.path.join(src_dir, vid_name, f_name)}, ********')
                continue
            x, y, w, h = results[0][1]  # detector bb
            w = h = max(w, h)  # crop square anyway
            x, y = max(0, x - round(0.1 * w)), max(0, y - round(0.1 * h))
            w, h = round(1.2 * w), round(1.2 * h)

            cropped_square_face = Image.fromarray(image[y:(y + h), x:(x + w)])
            # save the face as it is (probably not square, and different size for each frame is possible)
            # Do the processing before inserting to models
            # os.makedirs(join(dst_dir, 'videos', vid_name), exist_ok=True)
            # cropped_square_face.save(
            #     os.path.join(dst_dir, 'videos', vid_name, f_name))
        print(f'DONE: idx {idx}, file {vid_name}')

    print('Started RetinaFace Cropping')
    with tqdm(enumerate(vids[20:]), unit=" Videos", desc=f"Cropping Faces") as t:
        start = time.time()
        # Parallel(n_jobs=16, backend='threading')(delayed(retinaface_crop_parallel)(ii, vid)
        #                                          for ii, vid in t)
        for ii, vid in t:
            retinaface_crop_parallel(ii, vid)
        print("Took: ", time.time() - start)
        logging.info(f'Took:  {time.time() - start}')


def degrade_images_parallel(src_dir, dst_dir):
    """
        Use the DBVSR degradation method
    """
    from DBVSR.code.script_gene_dataset import get_blur_kernel, get_lr_blurdown, kernel2png
    def degrade(vid_full_path):
        vid_name = vid_full_path.split('/')[-1]
        os.makedirs(join(dst_dir, 'videos', 'blurdown_x4', vid_name), exist_ok=True)
        os.makedirs(join(dst_dir, 'kernel_png', vid_name), exist_ok=True)
        os.makedirs(join(dst_dir, 'kernel', vid_name), exist_ok=True)
        kernel = get_blur_kernel(trian=False)
        for frame in glob(vid_full_path + '/*'):
            frame_name = frame.split('/')[-1]
            HR_img = cv2.imread(frame)
            LRx4 = get_lr_blurdown(HR_img, kernel)
            cv2.imwrite(join(dst_dir, 'videos', 'blurdown_x4', vid_name, frame_name), LRx4)
            cv2.imwrite(join(dst_dir, 'kernel_png', vid_name, frame_name), kernel2png(kernel))
            sio.savemat(join(dst_dir, 'kernel', vid_name, "{}.mat".format(frame_name.split('.')[0])), {'Kernel': kernel})


    # vid_names = glob(src_dir + '/*/')
    vid_names = [src_dir]
    start = time.time()
    # Parallel(n_jobs=256)(delayed(degrade)(vid_name)
    #                      for vid_name in tqdm(sorted(vid_names)))
    for vid_name in tqdm(sorted(vid_names)):
        degrade(vid_name)

    print("Took: ", time.time() - start)


def downsample_bicubic(src_dir, dst_dir, res):
    """
    Use bicubic to downsample
    """

    dst_dir = dst_dir + f'_res_{res}'

    def degrade(vid_full_path):
        vid_name = vid_full_path.split('/')[-1]
        if os.path.exists(join(dst_dir, 'videos', vid_name)):
            return
        os.makedirs(join(dst_dir, 'videos', vid_name), exist_ok=True)
        for idx, frame in enumerate(glob(vid_full_path + '/*')):
            frame_name = frame.split('/')[-1]
            image = Image.open(frame).resize((res, res), Image.Resampling.BICUBIC)
            image.save(join(dst_dir, 'videos', vid_name, frame_name))

    print('Started function: downsample_bicubic, res: ', res)
    vid_names = glob(join(src_dir, 'videos', '*'))
    # vid_names = [src_dir]
    start = time.time()
    Parallel(n_jobs=128)(delayed(degrade)(vid_name)
                         for vid_name in tqdm(sorted(vid_names)))
    # for vid_name in tqdm(sorted(vid_names)):
    #     degrade(vid_name)

    print("Took: ", time.time() - start)


def downsample_bilinear(src_dir, dst_dir, res):
    """
    Use bilinear to downsample
    """

    dst_dir = dst_dir + f'_res_{res}'

    def degrade(vid_full_path):
        vid_name = vid_full_path.split('/')[-1]
        if os.path.exists(join(dst_dir, 'videos', vid_name)):
            return
        os.makedirs(join(dst_dir, 'videos', vid_name), exist_ok=True)
        for idx, frame in enumerate(glob(vid_full_path + '/*')):
            frame_name = frame.split('/')[-1]
            image = Image.open(frame).resize((res, res), Image.Resampling.BILINEAR)
            image.save(join(dst_dir, 'videos', vid_name, frame_name))

    print('Started function: downsample_bilinear, res: ', res)
    vid_names = glob(join(src_dir, 'videos', '*'))
    # vid_names = [src_dir]
    start = time.time()
    Parallel(n_jobs=128)(delayed(degrade)(vid_name)
                         for vid_name in tqdm(sorted(vid_names)))
    # for vid_name in tqdm(sorted(vid_names)):
    #     degrade(vid_name)

    print("Took: ", time.time() - start)


def sr_bicubic_parallel(src_dir, dst_dir):
    scale = 4

    def sr_bicubic(vid_full_path):
        vid_name = vid_full_path.split('/')[-1]
        if os.path.isfile(vid_full_path) or os.path.exists(join(dst_dir, 'videos', vid_name)):
            # good practice for other files in this directory or folders that already exists
            return
        os.makedirs(join(dst_dir, 'videos', vid_name))
        for frame in glob(vid_full_path + '/*'):
            frame_name = frame.split('/')[-1]
            LRx4 = cv2.imread(frame)
            sr_bic = cv2.resize(LRx4, (LRx4.shape[0]*scale, LRx4.shape[1]*scale), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(join(dst_dir, 'videos', 'videos', vid_name, frame_name), sr_bic)

    vid_names = glob(src_dir + '/*')
    start = time.time()
    Parallel(n_jobs=256)(delayed(sr_bicubic)(vid_name)
                         for vid_name in tqdm(sorted(vid_names)))
    print("Took: ", time.time() - start)


def downsample_bicubic_adaptive_ytf(src_dir, dst_dir):
    """
    Use bicubic to downsample face images to get specific average distance between eyes.
    """
    from retinaface import RetinaFace
    from deepface.detectors import FaceDetector
    detection_model = FaceDetector.build_model('retinaface')
    os.makedirs(dst_dir, exist_ok=True)
    logging.basicConfig(filename=dst_dir + '/created.log', level=logging.DEBUG, filemode='a')
    logging.info('Logging of downsample_bicubic_adaptive_ytf, 22.2.23,'
                 ' bicubic downsample to get average distance between eyes of 7 in each video')

    min_eyes_dist = 7
    max_eyes_dist = 7
    if os.path.exists(join(src_dir, 'eyes_distance_dict.pickle')):
        with open(join(src_dir, 'eyes_distance_dict.pickle'), 'rb') as h:
            eyes_distance_dict = pickle.load(h)
    else:
        eyes_distance_dict = {}
    scale_dict = {}

    def degrade(vid_full_path):
        vid_name = vid_full_path.split('/')[-1]
        # if os.path.exists(join(dst_dir, 'videos', vid_name)):
        #     return
        eyes_distance = []
        images = []
        frames_names = []
        for idx, frame in enumerate(glob(vid_full_path + '/*')):
            frame_name = frame.split('/')[-1]
            frames_names.append(frame_name)
            image = Image.open(frame)
            images.append(image)

            # it is easier for the transformer to get an image with small face inside than face image.
            # so I pad it.
            if vid_name not in eyes_distance_dict.keys():
                padded_face = Image.new('RGB', (image.size[0] * 2, image.size[1] * 2))
                padded_face.paste(image, (0, 0))
                results = RetinaFace.detect_faces(np.array(padded_face), model=detection_model, threshold=0.8)
                if not results or isinstance(results, tuple):
                    print('\n###### NO FACE DETECTED. idx:', idx, 'filename: ', frame, '######')
                    logging.info(f'###### NO FACE DETECTED. idx:  {idx}, filename: {frame} ######')
                    eyes_distance.append(-1)
                elif len(results) > 1:
                    print(f'\n******** MORE THAN ONE FACE DETECTED. idx:  {idx}, filename: , {frame}, ********')
                    logging.info(f'******** MORE THAN ONE FACE DETECTED. idx:  {idx}, filename: , {frame}, ********')
                    scores = np.array([results[f'face_{ii}']['score']for ii in range(1, len(results) + 1)])
                    relevant_face_num = np.argmax(scores).item() + 1
                    eyes_distance.append(
                        np.linalg.norm(np.array(results[f'face_{relevant_face_num}']["landmarks"]["right_eye"])
                                       - np.array(results[f'face_{relevant_face_num}']["landmarks"]["left_eye"])).item())
                else:
                    eyes_distance.append(
                        np.linalg.norm(np.array(results['face_1']["landmarks"]["right_eye"]) - np.array(results['face_1']["landmarks"]["left_eye"])).item())

        if vid_name not in eyes_distance_dict.keys():
            eyes_distance = np.array(eyes_distance)
            eyes_distance_dict[vid_name] = copy.deepcopy(eyes_distance)
        else:
            eyes_distance = copy.deepcopy(eyes_distance_dict[vid_name])

        if np.all(eyes_distance == -1):
            print(f'ALL FRAMES NOT DETECTED - Video: {vid_full_path}')
            logging.info(f'ALL FRAMES NOT DETECTED - Video: {vid_full_path}')
            eyes_distance = np.ones(eyes_distance.shape)*20

        mean_d = np.mean(eyes_distance[eyes_distance >= 0])
        eyes_distance[eyes_distance == -1] = mean_d

        os.makedirs(join(dst_dir, 'videos', vid_name), exist_ok=True)
        linear_wanted_eyes_dist = np.linspace(min_eyes_dist, max_eyes_dist, len(images))
        scales = linear_wanted_eyes_dist / mean_d
        scales = np.minimum(scales, 1)  # do not upscale frames
        scale_dict[vid_name] = scales
        for idx, image in enumerate(images):
            w, h = image.size
            w = int(w*scales[idx])
            h = int(h*scales[idx])
            image = image.resize((w, h), Image.BICUBIC)
            image.save(join(dst_dir, 'videos', vid_name, frames_names[idx]))

    print('Started function: downsample_bicubic_adaptive_ytf')
    vid_names = glob(join(src_dir, 'videos', '*'))
    start = time.time()
    Parallel(n_jobs=128, backend='threading')(delayed(degrade)(vid_name)
                         for vid_name in tqdm(sorted(vid_names)))
    for vid_name in tqdm(sorted(vid_names)):
        degrade(vid_name)
    print("Took: ", time.time() - start)
    with open(join(dst_dir, 'scale.pickle'), 'wb') as h:
        pickle.dump(scale_dict, h)
    print('Saved ', join(dst_dir, 'scale.pickle'))
    if not os.path.exists(join(src_dir, 'eyes_distance_dict.pickle')):
        with open(join(src_dir, 'eyes_distance_dict.pickle'), 'wb') as h:
            pickle.dump(eyes_distance_dict, h)
        print('Saved ', join(src_dir, 'eyes_distance_dict.pickle'))
    print('DONE!')


def convert_video_to_frames(vid_path, dst_path):
    """
    :param vid_path:
    :param dst_path: path of the dir to insert frames to.
    :return:
    """
    vidcap = cv2.VideoCapture(vid_path)
    os.makedirs(dst_path, exist_ok=True)
    success, image = vidcap.read()
    count = 0
    print(f'Converting video {vid_path} into frames...\n')
    while success:
        cv2.imwrite(os.path.join(dst_path, f"{str(count).zfill(6)}.png"), image)  # save frame as PNG file
        success, image = vidcap.read()
        count += 1
        if count % 50 == 0:
            print(f'{count} Done!')

    print(f'Frames of {vid_path} saved in {dst_path}\n')
    print(f'Total of {count} frames\n')


def splits_ids_train_val_query_gallery(videos_path, dst_dir, train_frac, val_frac):
    """
    splits the videos by the ids. ignore the ids with only one video.
    for the gallery take only one video per id and the rest goes to the gallery.
    """
    print('Running function: splits_ids_train_val_query_gallery')

    videos = np.array(sorted(os.listdir(videos_path)))

    ids = np.array([name[:-2] for name in videos])
    ids_unique, counts = np.unique(ids, return_counts=True)

    # remove ids with 1 video
    ids_one_vid = ids_unique[counts == 1]
    ids_multiple_vid = ids_unique[~np.in1d(ids_unique, ids_one_vid)]

    train_ids = np.random.choice(ids_multiple_vid, round(train_frac*len(ids_multiple_vid)), replace=False)
    train_vids = np.sort(np.concatenate([videos[ids == id_] for id_ in train_ids]))
    remain_ids = ids_multiple_vid[~np.in1d(ids_multiple_vid, train_ids)]
    val_ids = np.random.choice(remain_ids, round(val_frac*len(ids_multiple_vid)), replace=False)
    val_vids = np.sort(np.concatenate([videos[ids == id_] for id_ in val_ids]))

    remain_ids = remain_ids[~np.in1d(remain_ids, val_ids)]
    remain_vids = [videos[ids == id_] for id_ in remain_ids]
    gallery_vids = np.sort([np.random.choice(curr_id_vids) for curr_id_vids in remain_vids])
    remain_vids = np.concatenate(remain_vids)
    query_vids = np.sort(remain_vids[~np.in1d(remain_vids, gallery_vids)])

    os.makedirs(join(dst_dir, 'test'), exist_ok=True)
    os.makedirs(join(dst_dir, 'train'), exist_ok=True)
    os.makedirs(join(dst_dir, 'val'), exist_ok=True)
    with open(os.path.join(dst_dir, 'train', 'train.pickle'), 'wb') as handle:
        pickle.dump(train_vids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dst_dir, 'val', 'val.pickle'), 'wb') as handle:
        pickle.dump(val_vids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dst_dir, 'test', 'gallery.pickle'), 'wb') as handle:
        pickle.dump(gallery_vids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dst_dir, 'test', 'query.pickle'), 'wb') as handle:
        pickle.dump(query_vids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dst_dir, 'all_data.pickle'), 'wb') as handle:
        pickle.dump(videos, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Saved splits in: ', dst_dir)
    print('DONE!')


def calc_head_pose(src_dir, output_filename):
    from sixdrepnet import SixDRepNet
    from sixdrepnet.utils import compute_euler_angles_from_rotation_matrices

    module = SixDRepNet(
        dict_path=
        '/inputs/bionicEye/SixDRepNet_weights/6DRepNet_300W_LP_AFLW2000.pth'
    )
    head_pose = {}

    head_pose_trans = {}
    from transform import SimilarityTrans
    sim_trans = SimilarityTrans()
    with open(os.path.join(src_dir, 'landmarks_n_score.pickle'), 'rb') as h:
        stats = pickle.load(h)
    landmarks = stats['landmarks']
    vids = sorted(os.listdir(os.path.join(src_dir, 'videos')))
    pose_estimator = module.model.to('cuda').eval()

    print('Starting Head Pose Estimation !')
    with torch.no_grad():
        for ii, vid in tqdm(enumerate(vids)):
            if os.path.isfile(os.path.join(src_dir, 'videos', vid)):
                continue
            frames = sorted(glob(os.path.join(src_dir, 'videos', vid, '*')))
            imgs = []
            imgs_trans = []
            for idx, frame in enumerate(frames):
                imgs.append(module.transformations(Image.open(frame)))
                im_trans = Image.fromarray(cv2.cvtColor(sim_trans(cv2.imread(frame), lmk=landmarks[vid][idx]), cv2.COLOR_BGR2RGB))
                imgs_trans.append(module.transformations(im_trans))

            imgs = torch.stack(imgs).to('cuda')
            pred = pose_estimator(imgs)
            euler = compute_euler_angles_from_rotation_matrices(pred)
            p = euler[:, 0].cpu().detach().numpy()
            y = euler[:, 1].cpu().detach().numpy()
            r = euler[:, 2].cpu().detach().numpy()
            head_pose[vid] = {'pitch': p, 'yaw': y, 'roll': r}

            imgs_trans = torch.stack(imgs_trans).to('cuda')
            pred_trans = pose_estimator(imgs_trans)
            euler = compute_euler_angles_from_rotation_matrices(pred_trans)
            p = euler[:, 0].cpu().detach().numpy()
            y = euler[:, 1].cpu().detach().numpy()
            r = euler[:, 2].cpu().detach().numpy()
            head_pose_trans[vid] = {'pitch': p, 'yaw': y, 'roll': r}

            if ii % 50 == 0:
                with open(output_filename + '.pickle', 'wb') as h:
                    pickle.dump(head_pose, h)
                with open(output_filename + '_trans.pickle', 'wb') as h:
                    pickle.dump(head_pose_trans, h)
                print('Saved head pose until video number:', ii)

    with open(output_filename + '.pickle', 'wb') as h:
        pickle.dump(head_pose, h)
    with open(output_filename + '_trans.pickle', 'wb') as h:
        pickle.dump(head_pose_trans, h)
    print('Saved head pose until video number:', ii)
    print("Head Pose Estimation Done!")


def omit_mask_channel(frames_name_list):

    def omit_parallel(frame):
        img = np.array(Image.open(frame))[:, :, :3]
        Image.fromarray(img).save(frame)

    start = time.time()
    print('Starting: omit_mask_channel')
    Parallel(n_jobs=256)(delayed(omit_parallel)(filename)
                         for filename in tqdm(sorted(frames_name_list)))
    print("Took: ", time.time() - start)


def rearanging_aligned_sharp(aligned_dir='/datasets/BionicEye/YouTubeFaces/raw/aligned_images_DB',
                             dst_base_dir='/datasets/BionicEye/YouTubeFaces/aligned_sharp'):
    from distutils.dir_util import copy_tree
    ids = os.listdir(aligned_dir)
    print('function: rearanging_aligned_sharp Started!')
    for curr_id in ids:
        vids_num = os.listdir(os.path.join(aligned_dir, curr_id))
        for vid in vids_num:
            src_dir = os.path.join(aligned_dir, curr_id, vid)
            dst_dir = os.path.join(dst_base_dir, curr_id + '_' + vid)
            os.makedirs(os.path.join(dst_dir, 'videos'))
            copy_tree(src_dir, os.path.join(dst_dir, 'videos'))

    print('function: rearanging_aligned_sharp Finished!')


def reformat_ytf_headpose_file(headpose_dir='/datasets/BionicEye/YouTubeFaces/raw/headpose_DB',
                               full_output_filename='/datasets/BionicEye/YouTubeFaces/head_pose/head_pose_supplied.pickle'):
    vids_files = sorted(glob(os.path.join(headpose_dir, '*')))
    head_pose = {}

    print('Starting reformatting head pose supplied...')
    for f in vids_files:
        p, y, r = sio.loadmat(f)['headpose']  # I dont know if this is the order of pitch, yaw, roll.
        head_pose[f.split('/')[-1].split('.')[0].split('apirun_')[1]] = \
            {'pitch': p * np.pi/180, 'yaw': y * np.pi/180, 'roll': r * np.pi/180}

    with open(full_output_filename, 'wb') as h:
        pickle.dump(head_pose, h)
    print('Saved: ', full_output_filename)
    print('DONE!')


def get_ytf_bb_size(src_dir='/datasets/BionicEye/YouTubeFaces/raw/frame_images_DB', dst_file_path='/datasets/BionicEye/YouTubeFaces/bb_size.pickle'):

    bb_size_dict = {}
    txt_files = glob(src_dir + '/*.txt')

    def get_ytf_bb_size_parallel(txtfile):
        with open(join(src_dir, txtfile)) as fp:
            Lines = fp.readlines()
            if len(Lines) == 0:
                print(txtfile, ' has no bbs')
                return
            for l in Lines:
                splits = l.split(',')
                w = int(splits[4])
                h = int(splits[5])
                splits[0] = splits[0].replace("\\", '/')
                id_name = splits[0].split('/')[0]
                vid_num = splits[0].split('/')[1]
                vid_name = id_name + '_' + vid_num
                if not vid_name in bb_size_dict.keys():
                    bb_size_dict[vid_name] = [[w, h]]
                else:
                    bb_size_dict[vid_name].append([w, h])

            print('Done with id: ', id_name)

    print('Started function: get_ytf_bb_size_parallel')
    start = time.time()

    # Parallel(n_jobs=256)(delayed(get_ytf_bb_size_parallel)(txtfile)
    #                      for txtfile in tqdm(sorted(txt_files)))
    for txtfile in tqdm(sorted(txt_files)):
        get_ytf_bb_size_parallel(txtfile)

    for k in bb_size_dict.keys():
        bb_size_dict[k] = np.vstack(bb_size_dict[k])

    print('Number of videos calculated:', len(bb_size_dict.keys()))

    print('Saving at: ', dst_file_path)
    with open(dst_file_path, 'wb') as h:
        pickle.dump(bb_size_dict, h)

    print("Took: ", time.time() - start)


def create_verification_pairs(videos_names, ids, dst_dir):
    """
    :param videos_names: names of the videos
    :param ids: same len as videos_names. tells the id for each video.
    :param dst_dir: to save the results pairs
    """

    print('Started function: create_verification_pairs')

    pairs = []
    same = []
    for id1 in np.unique(ids):
        # first, insert all n choose 2 different pairs of the same id
        same_id_vids_inds = np.where(ids == id1)[0]
        same_id_vids_names = [videos_names[vid_idx] for vid_idx in same_id_vids_inds]
        l = list(itertools.combinations(same_id_vids_names, 2))
        s = set(l)
        for pair in list(s):
            pairs.append(np.sort(pair))
            same.append(np.ones(1))

        # lets say the are 6 different pairs of the same id (i.e. 4 different videos of this id - 4 choose 2 = 6)
        # now insert 6 pairs: (same id, different id). iterate of the videos with modulo
        # (some videos from the same id will appear more than others)
        for ii in range(len(s)):
            not_same_vid_ind = np.random.choice(np.where(ids != id1)[0])
            not_same_vid_name = videos_names[not_same_vid_ind]
            pair = np.sort([not_same_vid_name, same_id_vids_names[ii % len(same_id_vids_names)]])
            while np.any(np.all(pair == np.vstack(pairs), axis=1)):
                not_same_vid_ind = np.random.choice(np.where(ids != id1)[0])
                not_same_vid_name = videos_names[not_same_vid_ind]
                pair = np.sort([not_same_vid_name, same_id_vids_names[ii % len(same_id_vids_names)]])
            pairs.append(pair)
            same.append(np.zeros(1))

    pairs = np.vstack(pairs)
    same = np.concatenate(same)

    assert len(np.unique(pairs, axis=0)) == len(pairs), "There is at least one pair that show up twice or more"
    assert sum(same == 1) == sum(same == 0), "The pairs are not balanced"

    outfile = os.path.join(dst_dir, 'verification.pickle')
    with open(outfile, 'wb') as h:
        pickle.dump({'pairs': pairs, 'same': same}, h)

    print('Saved verification pairs in: ', outfile)
    print('total pairs (balanced): ', len(pairs))
    print('DONE!')


def create_identification_splits(videos_names, ids, dst_dir):
    """
    use example:
    _, ids, filenames = filter_feats('/datasets/BionicEye/YouTubeFaces/faces/sharp/embeddings.pickle', 'train', splits_path='/inputs/bionicEye/data/ytf_new_splits')
    videos_names = [filenames[ii][0].split('/')[-2] for ii in range(len(filenames))]
    create_identification_splits(videos_names, ids, dst_dir='/inputs/bionicEye/data/ytf_new_splits/train')

    split to gallery\query sets in close-set case.
    """
    print('running function: create_identification_splits')
    videos_names = np.array(videos_names)
    vid_per_id = [videos_names[ids == id_] for id_ in np.unique(ids)]
    gallery_vids = np.sort([np.random.choice(curr_id_vids) for curr_id_vids in vid_per_id])
    query_vids = np.sort(videos_names[~np.in1d(videos_names, gallery_vids)])

    os.makedirs(dst_dir, exist_ok=True)
    with open(os.path.join(dst_dir, 'gallery.pickle'), 'wb') as handle:
        pickle.dump(gallery_vids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dst_dir, 'query.pickle'), 'wb') as handle:
        pickle.dump(query_vids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Saved splits in: ', dst_dir)
    print('DONE!')


def blur_videos(src_dir, dst_dir, min_sigma=0.1, max_sigma=5, kernel_size=(11, 11)):
    """
        use example:
        blur_videos('/datasets/BionicEye/YouTubeFaces/faces/raw_sharp', '/datasets/BionicEye/YouTubeFaces/faces/blurred_raw_sharp')

        Use gaussian kernels to blur the images. NO sampling (image stays in the same size as before).
    """
    sigma_dict = {}

    def blur_video(vid_full_path):
        vid_name = vid_full_path.split('/')[-1]
        # if os.path.exists(join(dst_dir, 'videos', vid_name)):
        #     return
        os.makedirs(join(dst_dir, 'videos', vid_name), exist_ok=True)
        frames = glob(vid_full_path + '/*')
        sigmas = np.linspace(max_sigma, min_sigma, len(frames))
        for idx, frame in enumerate(frames):
            frame_name = frame.split('/')[-1]
            blurred_image = cv2.GaussianBlur(plt.imread(frame), kernel_size, sigmas[idx])
            blurred_image = (255*np.clip(blurred_image, 0, 1)).astype('uint8')
            blurred_image = Image.fromarray(blurred_image)
            blurred_image.save(join(dst_dir, 'videos', vid_name, frame_name))
        sigma_dict[vid_name] = sigmas

    vid_names = sorted(glob(join(src_dir, 'videos', '*')))
    start = time.time()
    Parallel(n_jobs=128, backend='threading')(delayed(blur_video)(vid_name)
                         for vid_name in tqdm(vid_names))
    # for vid_name in tqdm(vid_names):
    #     blur_video(vid_name)

    print("Took: ", time.time() - start)

    print('Saving sigma_dict in ', os.path.join(dst_dir, 'sigma.pickle'))
    with open(os.path.join(dst_dir, 'sigma.pickle'), 'wb') as h:
        pickle.dump(sigma_dict, h)


def calc_blur_cv2(src_dir):
    """
    use the variance of the laplacian to estimate the blurriness of each image
    """

    blur_dict = {}
    vid_names = sorted(os.listdir(os.path.join(src_dir, 'videos')))

    def calc_blur_parallel(vid_name):
        frames_blur = []
        frames = glob(os.path.join(src_dir, 'videos', vid_name, '*'))
        for frame in frames:
            image = plt.imread(frame)
            if image.dtype != 'uint8':
                image = np.uint8(image.clip(0, 1)*255)
            var_of_lap = cv2.Laplacian(image, cv2.CV_64F).var()
            frames_blur.append(var_of_lap)
        blur_dict[vid_name] = np.array(frames_blur)

    print('Started function: calc_blur_parallel')
    Parallel(n_jobs=128, backend='threading')(delayed(calc_blur_parallel)(vid_name)
                         for vid_name in tqdm(vid_names))
    # for vid_name in tqdm(vid_names):
    #     calc_blur_parallel(vid_name)

    print('Saving blur_dict in ', os.path.join(src_dir, 'blur_dict.pickle'))
    with open(os.path.join(src_dir, 'blur_dict.pickle'), 'wb') as h:
        pickle.dump(blur_dict, h)


def split_templates(ids, dst_dir, train_frac, val_frac, min_images_len=None):
    """
    ids: array of ids. ids[ii] is the id of image ii.
        len(id) should be the number of images in the dataset
    split ids to train, val, test
    remove ids with template that includes less images than min_images_len
    """
    ids_unique, counts = np.unique(ids, return_counts=True)
    inds = np.arange(len(ids))

    if min_images_len:
        # remove the small templates
        remove_ids = ids_unique[counts < min_images_len]
        inds = np.where(~np.in1d(ids, remove_ids))[0]
        ids = ids[inds]
        ids_unique = np.unique(ids)

    train_size = round(train_frac * len(ids_unique))
    val_size = round(val_frac * len(ids_unique))


    # for each id, cluster the indices of this id images in the total dataset:
    inds_cluster = [inds[ids == id_] for id_ in ids_unique]
    perm_inds = np.random.permutation(len(ids_unique))
    train_idx = perm_inds[:train_size]
    val_idx = perm_inds[train_size:(train_size+val_size)]
    test_idx = perm_inds[(train_size+val_size):]

    train_ids = ids_unique[train_idx]
    train_images_inds = [inds_cluster[ii] for ii in train_idx]
    val_ids = ids_unique[val_idx]
    val_images_inds = [inds_cluster[ii] for ii in val_idx]
    test_ids = ids_unique[test_idx]
    test_images_inds = [inds_cluster[ii] for ii in test_idx]

    for name, ids, images_inds in zip(['train', 'val', 'test'], [train_ids, val_ids, test_ids],
                                      [train_images_inds, val_images_inds, test_images_inds]):
        os.makedirs(os.path.join(dst_dir, 'splits', name))
        with open(os.path.join(dst_dir, 'splits', name, 'all.pickle'), 'wb') as h:
            pickle.dump({'ids': ids, 'images_inds': images_inds}, h)
        print(f'Saved {name} split. has {len(ids)} ids')

    print('DONE!')


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


def downsample_bilinear_by_scale(src_dir, dst_dir, scale):
    """
        Use biliner to downsample image by scale in each dimension
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
            image = image.resize((image.size[0]//scale, image.size[1]//scale), Image.Resampling.BILINEAR)
            image.save(join(dst_dir, 'videos', vid_name, frame_name))

    print('Started function: downsample_bilinear_by_scale, scale: ', scale, ' on: ', src_dir)
    vid_names = glob(join(src_dir, 'videos', '*'))
    # vid_names = [src_dir]
    start = time.time()
    Parallel(n_jobs=256)(delayed(degrade)(vid_name)
                         for vid_name in tqdm(sorted(vid_names)))
    # for vid_name in tqdm(sorted(vid_names)):
    #     degrade(vid_name)

    print("Took: ", time.time() - start)


def effective_eyes_distance(src_dir):
    with open(os.path.join(src_dir, 'detection.pickle'), 'rb') as h:
        info = pickle.load(h)
        lmk = info['landmarks']
        score = info['score']

    vids = os.listdir(os.path.join(src_dir, 'videos'))

    from transform import SimilarityTrans
    sim_trans = SimilarityTrans()

    curr_eyes_dist = []
    effective_eyes_dist = []

    for vid in tqdm(vids):
        vid_lmk = lmk[vid]
        vid_score = score[vid]
        curr_vid_dists = []
        curr_vid_eff_dists = []
        for f_idx in range(len(vid_lmk)):
            if vid_score[f_idx] > 0:
                curr_vid_dists.append(np.linalg.norm(vid_lmk[f_idx, 0, :] - vid_lmk[f_idx, 1, :]))
                M = sim_trans.estimate_affine(vid_lmk[f_idx])
                new_eyes_coord = (M @ np.append(vid_lmk[f_idx, :2], np.ones((2, 1)), axis=1).T).T
                curr_vid_eff_dists.append(np.linalg.norm(new_eyes_coord[0] - new_eyes_coord[1]))
        curr_eyes_dist.append(curr_vid_dists)
        effective_eyes_dist.append(curr_vid_eff_dists)

    mean_eyes_dist_curr = np.concatenate(curr_eyes_dist).mean()
    mean_eyes_dist_effective = np.nanmean(np.concatenate(effective_eyes_dist))

    print('Mean eyes distance before prerocessing: ', mean_eyes_dist_curr)
    print('Mean eyes distance after prerocessing: ', mean_eyes_dist_effective)


if __name__ == '__main__':
    # face_resolution = 112  # ArcFace input size
    # src_dir = '/datasets/BionicEye/youtubeFaces/raw'
    # dst_dir = '/datasets/BionicEye/youtubeFaces/faces/sharp'
    # start = time.time()
    # Parallel(n_jobs=256)(delayed(crop_and_resize_parallel)(ii, filename, src_dir, dst_dir, face_resolution)
    #                      for ii, filename in enumerate(tqdm(sorted(os.listdir(src_dir)))))
    # print("Took: ", time.time() - start)

    # degrade_images_parallel('/tmp/bionicEye/instantNerf/data/nerf/fox_lr/images', '/tmp/bionicEye/instantNerf/data/nerf/fox_lr')

    # sr_bicubic_parallel('/datasets/BionicEye/YouTubeFaces/faces/bicubic_down_x4', '/datasets/BionicEye/YouTubeFaces/faces/bicubic_sr_bicdown')

    # convert_video_to_frames('/inputs/bionicEye/webcams/videos/abbeyroad_uk_01_02_2022_14_41.mp4', '/inputs/bionicEye/webcams/frames/full/abbeyroad_uk_01_02_2022_14_41')

    # crop_faces_with_retinaface('/datasets/BionicEye/YouTubeFaces/raw/frame_images_DB', '/datasets/BionicEye/YouTubeFaces/retinaface_square/sharp')

    # downsample_bicubic('/datasets/BionicEye/YouTubeFaces/faces/sharp', '/datasets/BionicEye/YouTubeFaces/faces/bicubic_down_x4', res=28)

    # filter_existing_videos()

    # crop_faces('/inputs/bionicEye/videoINR/data/rami/full_size', '/inputs/bionicEye/videoINR/data/rami/face')

    # calc_head_pose('/datasets/BionicEye/YouTubeFaces/faces/sharp', '/datasets/BionicEye/YouTubeFaces/head_pose/sixdrepnet_rad')

    # frames = glob('/datasets/BionicEye/YouTubeFaces/faces/dbvsr_bicdown/*/*.png', recursive=True)
    # omit_mask_channel(frames)

    # rearanging_aligned_sharp()

    # reformat_ytf_headpose_file()

    # retinaface_crop('/datasets/BionicEye/YouTubeFaces/faces/aligned_sharp', '/datasets/BionicEye/YouTubeFaces/faces/aligned_cropped_sharp')

    # get_ytf_bb_size()

    # splits_ids_train_val_query_gallery(videos_path='/datasets/BionicEye/YouTubeFaces/faces/sharp',
    #                                    dst_dir='/inputs/bionicEye/data/ytf_new_splits',
    #                                    train_frac=0.7, val_frac=0.1)

    # downsample_bicubic_adaptive_ytf('/datasets/BionicEye/YouTubeFaces/faces/sharp', '/datasets/BionicEye/YouTubeFaces/faces/sharp_bicdown_adaptive')

    # face_crop_parallel_YTF('/datasets/BionicEye/YouTubeFaces/raw/frame_images_DB', '/datasets/BionicEye/YouTubeFaces/faces/sharp',
    #                        output_face_resolution=None)

    # blur_videos('/datasets/BionicEye/YouTubeFaces/faces/sharp', '/datasets/BionicEye/YouTubeFaces/faces/blurred_raw_sharp')

    # calc_blur_cv2('/datasets/BionicEye/YouTubeFaces/faces/sharp')


    file = '/datasets/MS1M_RetinaFace_t1/ms1m-retinaface-t1/train.lst'
    with open(file) as fp:
        lines = fp.readlines()
    ids = [int(line.split('\t')[-1].split('\n')[0]) for line in lines]
    split_templates(ids, dst_dir='/inputs/bionicEye/data/ms1m-retinaface-t1', train_frac=0.9, val_frac=0.03, min_images_len=None)

    # rearange_parallel_YTF('/datasets/BionicEye/YouTubeFaces/raw/frame_images_DB', '/datasets/BionicEye/YouTubeFaces/full/sharp')

    # downsample_bicubic_by_scale('/datasets/BionicEye/YouTubeFaces/faces/sharp', '/datasets/BionicEye/YouTubeFaces/faces/bicubic_lr', scale=8)

    # downsample_bilinear_by_scale('/datasets/BionicEye/YouTubeFaces/faces/sharp', '/datasets/BionicEye/YouTubeFaces/faces/bilinear_lr', scale=4)

    # lr_dirs = [name for name in os.listdir('/datasets/BionicEye/YouTubeFaces/full') if 'lr' in name]
    # for d in lr_dirs:
    #     print(d)
    #     effective_eyes_distance(f'/datasets/BionicEye/YouTubeFaces/full/{d}')