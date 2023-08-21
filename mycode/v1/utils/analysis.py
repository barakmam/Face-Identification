import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from matplotlib.backends.backend_pdf import PdfPages


def s2s_identification(embedding_path_q, embedding_path_g, splits_path, head_pose_file):
    feats_q, id_q, _ = filter_videos(embedding_path_q, 'test_query', splits_path)
    feats_g, id_g, _, _ = filter_videos(embedding_path_g, 'test_gallery', splits_path, head_pose_file, 1)

    feats_g = normalize(feats_g)
    recalls_at_k = []
    for ii in range(50):
        single_feat_q = np.vstack([feats_q[vid_idx][np.random.randint(0, len(feats_q[vid_idx]))]
                                   for vid_idx in range(len(feats_q))])
        single_feat_q = normalize(single_feat_q)
        recalls_at_k.append(get_recall_at_k(single_feat_q, feats_g, id_q, id_g, k_vals))
    print('Identification test. query: ', len(id_q), ' gallery: ', len(id_g))
    print('Rank@K on Single Frame: ', np.mean(recalls_at_k, axis=0))
    return np.mean(recalls_at_k, axis=0)


def v2s_identification(embedding_path_q, embedding_path_g, splits_path, head_pose_file, detection_file=None, score_filter=-1):
    """
    average of all query video
    gallery most frontal image
    """
    feats_q, id_q, _ = filter_videos(embedding_path_q, 'test_query', splits_path, detection_file=detection_file, score_filter=score_filter)
    feats_g, id_g, _, _ = filter_videos(embedding_path_g, 'test_gallery', splits_path, head_pose_file, 1)

    relevant_queries = np.array([ii for ii in range(len(id_q)) if len(feats_q[ii]) > 0])
    id_q = id_q[relevant_queries]
    feats_q = np.vstack([feats_q[ii].mean(0) for ii in relevant_queries])
    feats_q = normalize(feats_q)
    feats_g = normalize(feats_g)
    recalls_at_k = get_recall_at_k(feats_q, feats_g, id_q, id_g, k_vals)

    print('Identification test. query: ', len(id_q), ' gallery: ', len(id_g))
    print(f'Rank@K on Most Frontal Gallery Frame & Mean of All Query Frames: ', recalls_at_k)
    return recalls_at_k


def v2s_frontal_identification(embedding_path_q, embedding_path_g, splits_path, head_pose_file, detection_file=None, num_query_frames=40, score_filter=-1):
    """
    take num_query_frames most frontals query frames. for the gallery use the most frontal frame.
    :param score_filter:
    :param detection_file:
    """
    feats_q, id_q, _, _ = filter_videos(embedding_path_q, 'test_query', splits_path, head_pose_file, num_query_frames, detection_file, score_filter=score_filter)
    feats_g, id_g, _, _ = filter_videos(embedding_path_g, 'test_gallery', splits_path, head_pose_file, 1)

    relevant_queries = np.array([ii for ii in range(len(id_q)) if len(feats_q[ii]) > 0])
    id_q = id_q[relevant_queries]
    feats_q = np.vstack([feats_q[ii] if num_query_frames == 1 else feats_q[ii].mean(0) for ii in relevant_queries ])

    feats_q = normalize(feats_q)
    feats_g = normalize(feats_g)
    recalls_at_k = get_recall_at_k(feats_q, feats_g, id_q, id_g, k_vals)

    print(f'Rank@K on Most Frontal Gallery Frame & Mean of Most {num_query_frames} Frontal Queries: ', recalls_at_k)
    return recalls_at_k


def v2s_with_filters(embedding_path_q, embedding_path_g, splits_path, head_pose_file, detection_file, filters=None, num_query_frames=40):
    feats_q, id_q, _ = filter_videos(embedding_path_q, 'test_query', splits_path,
                                     detection_file=detection_file, detection_filters=filters, num_frames=num_query_frames)
    feats_g, id_g, _, _ = filter_videos(embedding_path_g, 'test_gallery', splits_path, head_pose_file, 1)

    relevant_queries = np.array([ii for ii in range(len(id_q)) if len(feats_q[ii]) > 0])
    id_q = id_q[relevant_queries]
    feats_q = np.vstack([feats_q[ii].mean(0) for ii in relevant_queries])
    feats_q = normalize(feats_q)
    feats_g = normalize(feats_g)
    recalls_at_k = get_recall_at_k(feats_q, feats_g, id_q, id_g, k_vals)

    print('Identification test. query: ', len(id_q), ' gallery: ', len(id_g))
    print(f'CMC @ Rank on Most Frontal Gallery Frame & Mean of {num_query_frames} Query Frames with filters: {filters}:', recalls_at_k)
    return recalls_at_k


def v2v_identification(embedding_path_q, embedding_path_g):
    feats_q, id_q, _ = filter_videos(embedding_path_q, 'test_query', splits_path)
    feats_g, id_g, _ = filter_videos(embedding_path_g, 'test_gallery', splits_path)

    feats_q = np.vstack([frames_feats.mean(0) for frames_feats in feats_q])
    feats_g = np.vstack([frames_feats.mean(0) for frames_feats in feats_g])

    feats_q = normalize(feats_q)
    feats_g = normalize(feats_g)
    recall_at_k = get_recall_at_k(feats_q, feats_g, id_q, id_g, k_vals)
    print('Rank@K V2V: ', recall_at_k)
    return recall_at_k


def best_frame_identification_oracle(embedding_path_q, embedding_path_g, splits_path, head_pose_file, num_query_frames=40):
    """
    Assuming we know the correct id for each query.
    for each query, we check which of the frames is closet to the correct gallery more than all other galleries.
    this will be the best frame we can use to optimize the rank @ 1 accuracy.
    (in our case this is the Rank @ 1 since each id shows up in the gallery only once.).
    I use the head pose to resemble a case that the gallery images are pretty good.
    """

    feats_q, id_q, filename_q, _ = filter_videos(embedding_path_q, 'test_query', splits_path, head_pose_file, num_query_frames)
    feats_g, id_g, _, _ = filter_videos(embedding_path_g, 'test_gallery', splits_path, head_pose_file, 1)

    query_best_frame = []
    rank_of_correct_gallery = {}  # for each frame in the video, get the rank of the correct gallery
    for ii in range(len(feats_q)):
        correct_gallery_idx = np.where(id_g == id_q[ii])[0].item()
        sim_mat = feats_q[ii] @ feats_g.T
        argsort = np.argsort(sim_mat)[:, ::-1]  # descending order
        where = np.where(argsort == correct_gallery_idx)
        vid_name = filename_q[ii][0].split('/')[-2]
        rank_of_correct_gallery[vid_name] = where[1]
        query_best_frame.append(feats_q[ii][np.argmin(where[1])])

    feats_q = np.vstack(query_best_frame)
    feats_q = normalize(feats_q)
    feats_g = normalize(feats_g)
    recall_at_k = get_recall_at_k(feats_q, feats_g, id_q, id_g, k_vals)
    print('Using Best Query Frame. Average Rank@K: ', recall_at_k)
    return recall_at_k, rank_of_correct_gallery


def plot_recall_at_k(recall_at_k_list, descriptions, title=None, lgd_title=None, k=None, fmt=None, show=True):
    if k is None:
        k = k_vals
    plt.figure(dpi=200)
    for desc_idx in range(len(descriptions)):
        if fmt:
            plt.plot(k, recall_at_k_list[desc_idx], fmt[desc_idx], label=descriptions[desc_idx])
        else:
            plt.plot(k, recall_at_k_list[desc_idx], label=descriptions[desc_idx])
    plt.title(f'CMC Curve {"- " + title if title else ""}', fontsize=17)
    plt.xlabel('Rank', fontsize=17), plt.ylabel('Identification Rate', fontsize=17), plt.grid()
    lgd = plt.legend()
    if lgd_title:
        lgd.set_title(lgd_title)
    if show:
        plt.show()


if __name__ == '__main__':
    from misc import filter_videos
    from metrics import get_recall_at_k, k_vals

    splits_path = '../ytf/splits'
    head_pose_file = '../ytf/head_pose_supplied.pickle'
    # head_pose_file = '/datasets/BionicEye/YouTubeFaces/head_pose/sixdrepnet_rad_trans.pickle'

    # hr_sim_trans_r100
    embedding_path_q = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    embedding_path_g = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    r1, rank_of_correct_gallery = best_frame_identification_oracle(embedding_path_q, embedding_path_g, splits_path, head_pose_file, num_query_frames=40)
    r2 = v2s_identification(embedding_path_q, embedding_path_g, splits_path, head_pose_file)
    r3 = v2s_frontal_identification(embedding_path_q, embedding_path_g, splits_path, head_pose_file, num_query_frames=40)
    r4 = v2s_frontal_identification(embedding_path_q, embedding_path_g, splits_path, head_pose_file, num_query_frames=1)
    plot_recall_at_k([r2, r3, r4],
                     [#'Oracle: Max rank frame from 40 frontals',
                      'Mean of all frames',
                      'V2S query 40 frontals mean',
                      'V2S query 1 frontal',
                      ], show=False)
    pdf_pages = PdfPages('/outputs/bionicEye/thesis/images/hr_sim_trans_r100.pdf')
    pdf_pages.savefig()
    pdf_pages.close()

    # preprocess_results
    r1 = v2s_identification('/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle',
                            '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle', splits_path, head_pose_file)
    r2 = v2s_identification('/outputs/bionicEye/v1/extract-feats/sim-trans-sharp-r100_26-03-2023_11-31/embeddings.pickle',
                            '/outputs/bionicEye/v1/extract-feats/sim-trans-sharp-r100_26-03-2023_11-31/embeddings.pickle', splits_path, head_pose_file)
    r3 = v2s_identification('/outputs/bionicEye/v1/extract-feats/norm-sharp-r100_26-03-2023_09-06/embeddings.pickle',
                            '/outputs/bionicEye/v1/extract-feats/norm-sharp-r100_26-03-2023_09-06/embeddings.pickle', splits_path, head_pose_file)
    r4 = v2s_identification('/outputs/bionicEye/v1/extract-feats/sharp-r100_26-03-2023_13-19/embeddings.pickle',
                            '/outputs/bionicEye/v1/extract-feats/sharp-r100_26-03-2023_13-19/embeddings.pickle', splits_path, head_pose_file)
    plot_recall_at_k([r1, r2, r3, r4],
                     ['Image normalization + similarity transform',
                      'Similarity transform',
                      'Image normalization',
                      'Raw'
                      ], 'V2S - All Frames Mean', show=False)
    pdf_pages = PdfPages('/outputs/bionicEye/thesis/images/preprocess_results.pdf')
    pdf_pages.savefig()
    pdf_pages.close()



    # # scenarios
    embedding_path_q = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    embedding_path_g = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    r1 = v2v_identification(embedding_path_q, embedding_path_g)
    r2 = v2s_identification(embedding_path_q,
                            embedding_path_g, splits_path, head_pose_file)
    r3 = s2s_identification(embedding_path_q,
                            embedding_path_g, splits_path, head_pose_file)
    plot_recall_at_k([r1, r2, r3],
                     ['V2V',
                      'V2S',
                      'S2S'], 'Different Scenarios', show=False)
    pdf_pages = PdfPages('/outputs/bionicEye/thesis/images/scenarios.pdf')
    pdf_pages.savefig()
    pdf_pages.close()

    #
    #
    # face_lr # face lr comparison
    embedding_path_g = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    r1 = v2s_identification('/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle',
                            embedding_path_g, splits_path, head_pose_file)
    r2 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-face-scale-2-norm-sim-r100_01-04-2023_19-31/embeddings.pickle',
                            embedding_path_g, splits_path, head_pose_file)
    r3 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-face-scale-3-norm-sim-r100_01-04-2023_19-54/embeddings.pickle',
                            embedding_path_g, splits_path, head_pose_file)
    r4 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-face-scale-4-norm-sim-r100_01-04-2023_19-54/embeddings.pickle',
                            embedding_path_g, splits_path, head_pose_file)
    r5 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-face-scale-5-norm-sim-r100_01-04-2023_19-55/embeddings.pickle',
                            embedding_path_g, splits_path, head_pose_file)
    r6 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-face-scale-6-norm-sim-r100_01-04-2023_19-55/embeddings.pickle',
                            embedding_path_g, splits_path, head_pose_file)
    r7 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-face-scale-7-norm-sim-r100_01-04-2023_19-55/embeddings.pickle',
                            embedding_path_g, splits_path, head_pose_file)
    r8 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-face-scale-8-norm-sim-r100_01-04-2023_19-56/embeddings.pickle',
                            embedding_path_g, splits_path, head_pose_file)
    plot_recall_at_k([r1, r2, r3, r4, r5, r6, r7, r8],
                     ['1', '2', '3', '4', '5', '6', '7', '8'], 'V2S - Low Resolution Query Faces', show=False)
    pdf_pages = PdfPages('/outputs/bionicEye/thesis/images/face_lr.pdf')
    pdf_pages.savefig()
    pdf_pages.close()


    #
    # # full image (in lr) detection comparison
    embedding_path_g = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    score_detection_th = 0
    r1 = v2s_identification('/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle',
        embedding_path_g, splits_path, head_pose_file,
                            '/datasets/BionicEye/YouTubeFaces/faces/sharp/detection.pickle', score_detection_th)
    print('Detection on scale 2')
    r2 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-2-norm-sim-r100_30-03-2023_10-53/embeddings.pickle',
                            embedding_path_g, splits_path, head_pose_file,
                            '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_2/detection.pickle', score_detection_th)
    print('Detection on scale 3')
    r3 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-3-norm-sim-r100_30-03-2023_11-43/embeddings.pickle',
                            embedding_path_g, splits_path, head_pose_file,
                            '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_3/detection.pickle', score_detection_th)
    print('Detection on scale 4')
    r4 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-4-norm-sim-r100_30-03-2023_11-44/embeddings.pickle',
                            embedding_path_g, splits_path, head_pose_file,
                            '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_4/detection.pickle', score_detection_th)
    print('Detection on scale 5')
    r5 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-5-norm-sim-r100_30-03-2023_11-44/embeddings.pickle',
                            embedding_path_g, splits_path, head_pose_file,
                            '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_5/detection.pickle', score_detection_th)
    print('Detection on scale 6')
    r6 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-6-norm-sim-r100_30-03-2023_11-45/embeddings.pickle',
                            embedding_path_g, splits_path, head_pose_file,
                            '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_6/detection.pickle', score_detection_th)
    print('Detection on scale 7')
    r7 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-7-norm-sim-r100_30-03-2023_11-45/embeddings.pickle',
                            embedding_path_g, splits_path, head_pose_file,
                            '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_7/detection.pickle', score_detection_th)
    print('Detection on scale 8')
    r8 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-8-norm-sim-r100_30-03-2023_11-46/embeddings.pickle',
                            embedding_path_g, splits_path, head_pose_file,
                            '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_8/detection.pickle', score_detection_th)
    plot_recall_at_k([r1, r2, r3, r4, r5, r6, r7, r8],
                     ['1', '2', '3', '4', '5', '6', '7', '8'], 'V2S - Low Resolution Query Full Image', lgd_title='Scale', show=False)
    pdf_pages = PdfPages('/outputs/bionicEye/thesis/images/full_lr.pdf')
    pdf_pages.savefig()
    pdf_pages.close()



    # filtering by factors
    # embedding_path_q = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    # embedding_path_g = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    # r1 = v2s_with_filters(embedding_path_q, embedding_path_g, splits_path, head_pose_file,
    #                       '/datasets/BionicEye/YouTubeFaces/faces/sharp/detection.pickle', filters=['eyes_dist'])
    # r2 = v2s_with_filters(embedding_path_q, embedding_path_g, splits_path, head_pose_file,
    #                       '/datasets/BionicEye/YouTubeFaces/faces/sharp/detection.pickle', filters=['bb_size'])
    # r3 = v2s_with_filters(embedding_path_q, embedding_path_g, splits_path, head_pose_file,
    #                       '/datasets/BionicEye/YouTubeFaces/faces/sharp/detection.pickle', filters=['feat_norm'])
    # r4 = v2s_with_filters(embedding_path_q, embedding_path_g, splits_path, head_pose_file,
    #                       '/datasets/BionicEye/YouTubeFaces/faces/sharp/detection.pickle',
    #                       filters=['eyes_dist', 'feat_norm'])
    # plot_recall_at_k([r1, r3, r4],
    #                  ['1', '3', '4'], 'V2S - Filters', lgd_title='Filters')

    # VSR full image detection comparison
    # embedding_path_g = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    # score_detection_th = 0
    # r1 = v2s_identification('/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle',
    #     embedding_path_g, splits_path, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/faces/sharp/detection.pickle', score_detection_th)
    # r2 = v2s_identification('/outputs/bionicEye/v1/extract-feats/dbvsr-lr-scale-2_23-04-2023_12-09/embeddings.pickle',
    #     embedding_path_g, splits_path, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/dbvsr/bicubic_lr_scale_2/detection.pickle', score_detection_th)
    # r3 = v2s_identification('/outputs/bionicEye/v1/extract-feats/dbvsr-lr-scale-4_23-04-2023_14-03/embeddings.pickle',
    #     embedding_path_g, splits_path, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/dbvsr/bicubic_lr_scale_4/detection.pickle', score_detection_th)
    # r4 = v2s_identification('/outputs/bionicEye/v1/extract-feats/dbvsr-lr-scale-8_23-04-2023_08-33/embeddings.pickle',
    #     embedding_path_g, splits_path, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/dbvsr/bicubic_lr_scale_8/detection.pickle', score_detection_th)
    # plot_recall_at_k([r1, r2, r3, r4],
    #                  ['High resolution', 'VSR on low resolution scale 2', 'VSR on low resolution scale 4', 'VSR on low resolution scale 8'],
    #                  'V2S - Video Super Resolution')

    # vsr_lr_v2s # VSR vs LR v2s
    embedding_path_g = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    score_detection_th = 0
    r0 = v2s_identification('/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle',
        embedding_path_g, splits_path, head_pose_file,
                            '/datasets/BionicEye/YouTubeFaces/faces/sharp/detection.pickle', score_detection_th)
    r1 = v2s_identification('/outputs/bionicEye/v1/extract-feats/dbvsr-lr-scale-2_23-04-2023_12-09/embeddings.pickle',
        embedding_path_g, splits_path, head_pose_file,
                            '/datasets/BionicEye/YouTubeFaces/full/dbvsr/bicubic_lr_scale_2/detection.pickle', score_detection_th)
    r2 = v2s_identification('/outputs/bionicEye/v1/extract-feats/dbvsr-lr-scale-4_23-04-2023_14-03/embeddings.pickle',
        embedding_path_g, splits_path, head_pose_file,
                            '/datasets/BionicEye/YouTubeFaces/full/dbvsr/bicubic_lr_scale_4/detection.pickle', score_detection_th)
    r3 = v2s_identification('/outputs/bionicEye/v1/extract-feats/dbvsr-lr-scale-8_23-04-2023_08-33/embeddings.pickle',
        embedding_path_g, splits_path, head_pose_file,
                            '/datasets/BionicEye/YouTubeFaces/full/dbvsr/bicubic_lr_scale_8/detection.pickle', score_detection_th)
    r11 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-2-norm-sim-r100_30-03-2023_10-53/embeddings.pickle',
                            embedding_path_g, splits_path, head_pose_file,
                            '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_2/detection.pickle', score_detection_th)
    r22 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-4-norm-sim-r100_30-03-2023_11-44/embeddings.pickle',
                            embedding_path_g, splits_path, head_pose_file,
                            '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_4/detection.pickle', score_detection_th)
    r33 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-8-norm-sim-r100_30-03-2023_11-46/embeddings.pickle',
                            embedding_path_g, splits_path, head_pose_file,
                            '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_8/detection.pickle', score_detection_th)
    plot_recall_at_k([r0, r1, r11, r2, r22, r3, r33],
                     ['High resolution',
                      'VSR on low resolution scale 2', 'Low resolution scale 2',
                      'VSR on low resolution scale 4', 'Low resolution scale 4',
                      'VSR on low resolution scale 8', 'Low resolution scale 8'],
                     'V2S \n Video Super Resolution vs Low Resolution',
                     fmt=['k', 'b', 'b--', 'r', 'r--', 'g', 'g--'], show=False)
    pdf_pages = PdfPages('/outputs/bionicEye/thesis/images/vsr_lr_v2s.pdf')
    pdf_pages.savefig()
    pdf_pages.close()

    # VSR vs LR scale 4 v2s most frontal and v2s
    embedding_path_g = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    score_detection_th = 0
    r1 = v2s_frontal_identification('/outputs/bionicEye/v1/extract-feats/dbvsr-lr-scale-4_23-04-2023_14-03/embeddings.pickle',
        embedding_path_g, splits_path, head_pose_file,
                            '/datasets/BionicEye/YouTubeFaces/full/dbvsr/bicubic_lr_scale_4/detection.pickle', 1, score_detection_th)
    r11 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-4-norm-sim-r100_30-03-2023_11-44/embeddings.pickle',
                            embedding_path_g, splits_path, head_pose_file,
                            '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_4/detection.pickle', score_detection_th)
    r2 = v2s_frontal_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-4-norm-sim-r100_30-03-2023_11-44/embeddings.pickle',
        embedding_path_g, splits_path, head_pose_file,
                            '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_4/detection.pickle', 1, score_detection_th)
    plot_recall_at_k([r11, r1, r2],
                     ['LR: Aggregation over all frames', 'VSR: most frontal frame', 'LR: most frontal frame'],
                     'Down-sample Scale 4\nVideo Super Resolution vs Low Resolution',
                     show=False)
    pdf_pages = PdfPages('/outputs/bionicEye/thesis/images/vsr_vs_lr.pdf')
    pdf_pages.savefig()
    pdf_pages.close()
