from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from tqdm import tqdm
import pickle
import scipy as sp


def s2s_identification(embedding_path_q, embedding_path_g, head_pose_file):
    feats_q, id_q, _ = filter_videos(embedding_path_q, 'test/query', detection_, splits_path)
    feats_g, id_g, _, _ = filter_videos(embedding_path_g, 'test/gallery', detection_, head_pose_file, 1)

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


def v2s_identification(embedding_path_q, embedding_path_g, head_pose_file, detection_q, detection_g):
    """
    average of all query video
    gallery most frontal image
    """
    feats_q, id_q, _ = filter_videos(embedding_path_q, 'test/query', detection_q)
    feats_g, id_g, _, _ = filter_videos(embedding_path_g, 'test/gallery', detection_g, head_pose_file, 1)

    relevant_queries = np.array([ii for ii in range(len(id_q)) if len(feats_q[ii]) > 0])
    id_q = id_q[relevant_queries]
    feats_q = np.vstack([feats_q[ii].mean(0) for ii in relevant_queries])
    feats_q = normalize(feats_q)
    feats_g = normalize(feats_g)
    recalls_at_k = get_recall_at_k(feats_q, feats_g, id_q, id_g, k_vals)

    print('Identification test. query: ', len(id_q), ' gallery: ', len(id_g))
    print(f'Rank@K on Most Frontal Gallery Frame & Mean of All Query Frames: ', recalls_at_k)
    return recalls_at_k


def v2s_frontal_identification(embedding_path_q, embedding_path_g, head_pose_file, detection_file=None, num_query_frames=40, score_filter=-1):
    """
    take num_query_frames most frontals query frames. for the gallery use the most frontal frame.
    :param score_filter:
    :param detection_file:
    """
    feats_q, id_q, _, _ = filter_videos(embedding_path_q, 'test/query', detection_file, head_pose_file,
                                        num_query_frames)
    feats_g, id_g, _, _ = filter_videos(embedding_path_g, 'test/gallery', detection_, head_pose_file, 1)

    feats_q = feats_q if num_query_frames == 1 else feats_q.mean(1)
    feats_q = normalize(feats_q)
    feats_g = normalize(feats_g)
    recalls_at_k = get_recall_at_k(feats_q, feats_g, id_q, id_g, k_vals)

    print(f'Rank@K on Most Frontal Gallery Frame & Mean of Most {num_query_frames} Frontal Queries: ', recalls_at_k)
    return recalls_at_k


def v2s_with_filters(embedding_path_q, embedding_path_g, head_pose_file, detection_file, filters=None, num_query_frames=40):
    feats_q, id_q, _ = filter_videos(embedding_path_q, 'test/query', detection_file=detection_file,
                                     head_pose_file=splits_path, num_frames=num_query_frames, detection_filters=filters)
    feats_g, id_g, _, _ = filter_videos(embedding_path_g, 'test/gallery', detection_, head_pose_file, 1)

    relevant_queries = np.array([ii for ii in range(len(id_q)) if len(feats_q[ii]) > 0])
    id_q = id_q[relevant_queries]
    feats_q = np.vstack([feats_q[ii].mean(0) for ii in relevant_queries])
    feats_q = normalize(feats_q)
    feats_g = normalize(feats_g)
    recalls_at_k = get_recall_at_k(feats_q, feats_g, id_q, id_g, k_vals)

    print('Identification test. query: ', len(id_q), ' gallery: ', len(id_g))
    print(f'CMC @ Rank on Most Frontal Gallery Frame & Mean of {num_query_frames} Query Frames with filters: {filters}:', recalls_at_k)
    return recalls_at_k


def v2s_any_hit_identification(embedding_path_q, embedding_path_g, head_pose_file, num_query_frames):
    feats_q, id_q, filename_q = filter_videos(embedding_path_q, 'query', detection_, head_pose_file, num_query_frames)
    feats_g, id_g, filename_g = filter_videos(embedding_path_g, 'gallery', detection_, head_pose_file, 1)

    acc = any_hit_identification(feats_q, feats_g, id_q, id_g)
    print(f'V2S Accuracy of Any-Hit Identification with {num_query_frames} Frontal Frames Query & Most Frontal Gallery: {acc:.3}')
    return acc


def v2s_histogram_identification(embedding_path_q, embedding_path_g, head_pose_file, num_query_frames):
    """
    for each of the query frames do the ranking.
    with the similarity scores make a scores accumulation for each id in their short-lists.
    then, rank the galleries for this query by the accumulations values.
    """
    feats_q, id_q, filename_q = filter_videos(embedding_path_q, 'query', detection_, head_pose_file, num_query_frames)
    feats_g, id_g, filename_g = filter_videos(embedding_path_g, 'gallery', detection_, head_pose_file, 1)

    logits = np.einsum('qfe,ge->qfg', feats_q, feats_g)

    ranking_inds = np.flip(np.argsort(logits, -1), -1)
    logits_rank_sort = np.take_along_axis(logits, ranking_inds, -1)

    rank_at_k = np.zeros(len(k_vals))
    for jj, k in enumerate(k_vals):
        query_rank = []
        logits_rank_sort_k = logits_rank_sort[:, :, :k]
        for ii in range(len(id_q)):
            k_nearest_g_id = id_g[ranking_inds[ii, :, :k]]
            showed_galleries = np.unique(k_nearest_g_id.flatten())
            gallery_accumulated_score = []
            for g in showed_galleries:
                gallery_accumulated_score.append(np.sum(logits_rank_sort_k[ii, k_nearest_g_id == g]))
            sort_per_query = np.flip(np.argsort(gallery_accumulated_score))
            gallery_accumulated_score = np.array(gallery_accumulated_score)[sort_per_query]
            showed_galleries = showed_galleries[sort_per_query]
            query_rank.append([gallery_accumulated_score, showed_galleries])
            rank_at_k[jj] += (id_q[ii] == showed_galleries[0])
        rank_at_k[jj] /= len(id_q)

    print('Accumulated scores ranking @ k:', rank_at_k)

    return rank_at_k


def v2s_identification_with_blur(embedding_path_q, embedding_path_g, head_pose_file, num_query_frames, sigma_dict):
    feats_q, id_q, filename_q, frontal_inds = filter_videos(embedding_path_q, 'query', detection_q, head_pose_file,
                                                            num_query_frames)
    feats_g, id_g, _, _ = filter_videos(embedding_path_g, 'gallery', detection_g, head_pose_file, 1)

    sigmas = np.vstack([sigma_dict[filename_q[ii][0].split('/')[-2]][frontal_inds[ii]] for ii in range(len(feats_q))])
    weights = 1 / sigmas
    weights = np.expand_dims(weights / weights.sum(1, keepdims=True), axis=2)
    feats_q = np.sum(feats_q * weights, axis=1)

    recalls_at_k = get_recall_at_k(feats_q, feats_g, id_q, id_g, k_vals)

    print(f'Recall@K, V2S Using Blur STD, Frontal Gallery, {num_query_frames} Frontal Queries: ', recalls_at_k)
    return recalls_at_k


def v2v_identification(embedding_path_q, embedding_path_g, detection_q, detection_g):
    feats_q, id_q, _ = filter_videos(embedding_path_q, 'test/query', detection_q, splits_path)
    feats_g, id_g, _ = filter_videos(embedding_path_g, 'test/gallery', detection_g, splits_path)

    feats_q = np.vstack([frames_feats.mean(0) for frames_feats in feats_q])
    feats_g = np.vstack([frames_feats.mean(0) for frames_feats in feats_g])

    feats_q = normalize(feats_q)
    feats_g = normalize(feats_g)
    recall_at_k = get_recall_at_k(feats_q, feats_g, id_q, id_g, k_vals)
    print('Rank@K V2V: ', recall_at_k)
    return recall_at_k


def best_frame_identification_oracle(embedding_path_q, embedding_path_g, head_pose_file, detection_q, detection_g):
    """
    Assuming we know the correct id for each query.
    for each query, we check which of the frames is closet to the correct gallery more than all other galleries.
    this will be the best frame we can use to optimize the rank @ 1 accuracy.
    (in our case this is the Rank @ 1 since each id shows up in the gallery only once.).
    I use the head pose to resemble a case that the gallery images are pretty good.
    """

    feats_q, id_q, filename_q = filter_videos(embedding_path_q, 'test/query', detection_q)
    feats_g, id_g, _, _ = filter_videos(embedding_path_g, 'test/gallery', detection_g, head_pose_file, 1)
    feats_g = normalize(feats_g)

    query_best_frame = []
    rank_of_correct_gallery = {}  # for each frame in the video, get the rank of the correct gallery
    for ii in range(len(feats_q)):
        correct_gallery_idx = np.where(id_g == id_q[ii])[0].item()
        sim_mat = feats_q[ii] @ feats_g.T
        argsort = np.argsort(sim_mat)[:, ::-1]  # descending order
        where = np.where(argsort == correct_gallery_idx)
        vid_name = filename_q[ii][0].split('/')[-2]
        rank_of_correct_gallery[vid_name] = where[1] + 1  # rank from 1 to k.
        query_best_frame.append(feats_q[ii][np.argmin(where[1])])

    feats_q = np.vstack(query_best_frame)
    feats_q = normalize(feats_q)
    recall_at_k = get_recall_at_k(feats_q, feats_g, id_q, id_g, k_vals)
    print('Using Best Query Frame. Average Rank@K: ', recall_at_k)
    return recall_at_k, rank_of_correct_gallery


def closest_frame_iden_oracle(embedding_path_q, embedding_path_g, head_pose_file, detection_q, detection_g):
    """
    Assuming we know the correct id for each query.
    for each query, we check which of the frames is closet to the correct gallery more than all other frames in the query.
    I use the head pose to resemble a case that the gallery images are pretty good.
    """

    feats_q, id_q, filename_q = filter_videos(embedding_path_q, 'test/query', detection_q)
    feats_g, id_g, _, _ = filter_videos(embedding_path_g, 'test/gallery', detection_g, head_pose_file, 1)

    query_best_frame = []
    sim_to_correct_gallery = {}  # for each frame in the video, get the rank of the correct gallery
    for ii in range(len(feats_q)):
        correct_gallery_idx = np.where(id_g == id_q[ii])[0].item()
        sim_mat = normalize(feats_q[ii]) @ normalize(feats_g[[correct_gallery_idx]]).T
        vid_name = filename_q[ii][0].split('/')[-2]
        sim_to_correct_gallery[vid_name] = sim_mat
        query_best_frame.append(feats_q[ii][np.argmax(sim_mat)])

    feats_q = np.vstack(query_best_frame)
    feats_q = normalize(feats_q)
    feats_g = normalize(feats_g)
    recall_at_k = get_recall_at_k(feats_q, feats_g, id_q, id_g, k_vals)
    print('Using Closest Query Frame. Average Rank@K: ', recall_at_k)
    return recall_at_k, sim_to_correct_gallery


def weighted_mean_from_oracle(embedding_path_q, embedding_path_g, head_pose_file, detection_q, detection_g):
    """
    Assuming we know the correct id for each query.
    for each query, we weight the frames by the similarity to the correct gallery and sum.
    I use the head pose to resemble a case that the gallery images are pretty good.
    """

    feats_q, id_q, filename_q = filter_videos(embedding_path_q, 'test/query', detection_q)
    feats_g, id_g, _, _ = filter_videos(embedding_path_g, 'test/gallery', detection_g, head_pose_file, 1)
    feats_q = [normalize(f) for f in feats_q]
    # feats_q = [f[np.random.choice(len(f), 10, replace=False)] for f in feats_q]  # sample random 10
    feats_g = normalize(feats_g)
    query_weighted = []
    for ii in range(len(feats_q)):
        correct_gallery_idx = np.where(id_g == id_q[ii])[0].item()
        sim = normalize(feats_q[ii]) @ normalize(feats_g[[correct_gallery_idx]]).T
        # w = (sim - sim.min(0))/(sim.max(0) - sim.min(0))
        weighted = np.sum(feats_q[ii] * sim, axis=0)
        query_weighted.append(weighted)

    feats_q = np.vstack(query_weighted)
    feats_q = normalize(feats_q)
    recall_at_k = get_recall_at_k(feats_q, feats_g, id_q, id_g, k_vals)
    print('Using Oracle Positive Similarity to Weighted Query Frames. Average Rank@K: ', recall_at_k)
    return recall_at_k


def pose_similarity_corr(embedding_path, head_pose_file):
    feats, id_, filename = filter_videos(embedding_path, 'all_data/all_data', detection_, splits_path)
    with open(head_pose_file, 'rb') as h:
        head_pose = pickle.load(h)

    print('Calculating head pose vs similarity correlation')
    similarity = []
    yaw_dist, roll_dist, pitch_dist = [], [], []
    head_pose_dist = []
    for curr_id in tqdm(np.unique(id_)):
        pos = np.where(id_ == curr_id)[0]
        pairs = combinations(pos, 2)
        for pair in pairs:
            vidname1 = filename[pair[0]][0].split('/')[-2]
            vidname2 = filename[pair[1]][0].split('/')[-2]
            feat1 = normalize(feats[pair[0]])
            feat2 = normalize(feats[pair[1]])
            sim = feat1 @ feat2.T
            head_pose1 = np.vstack([head_pose[vidname1]['yaw'], head_pose[vidname1]['pitch'], head_pose[vidname1]['roll']]).T % (2*np.pi)
            head_pose2 = np.vstack([head_pose[vidname2]['yaw'], head_pose[vidname2]['pitch'], head_pose[vidname2]['roll']]).T % (2*np.pi)
            pose_diff_radiance = (np.expand_dims(head_pose1, 1) - np.expand_dims(head_pose2, 0)) % (2*np.pi)
            pose_diff_radiance[pose_diff_radiance > np.pi] = 2*np.pi - pose_diff_radiance[pose_diff_radiance > np.pi]
            yaw_dist.append(pose_diff_radiance[..., 0].flatten())
            pitch_dist.append(pose_diff_radiance[..., 1].flatten())
            roll_dist.append(pose_diff_radiance[..., 2].flatten())
            pose_distance = np.linalg.norm(pose_diff_radiance, axis=-1)
            similarity.append(sim.flatten())
            head_pose_dist.append(pose_distance.flatten())


    # head_pose_dist = np.hstack(head_pose_dist)
    # similarity = np.hstack(similarity)
    # plt.scatter(head_pose_dist, similarity, s=0.5)
    # plt.xlabel('Head Pose Distance (L2)', fontsize=17), plt.ylabel('Cosine Similarity', fontsize=17)
    # plt.title('Similarity vs Head Pose Distance \nSame id Different Videos', fontsize=17), plt.grid(), plt.tight_layout()
    # plt.show()

    yaw_dist = np.hstack(yaw_dist)
    pitch_dist = np.hstack(pitch_dist)
    roll_dist = np.hstack(roll_dist)
    similarity = np.hstack(similarity)
    perm = np.random.permutation(len(yaw_dist))
    inds = perm[:30000]

    plt.scatter(yaw_dist[inds], similarity[inds], s=0.5, label='Yaw')
    plt.scatter(pitch_dist[inds], similarity[inds], s=0.5, label='Pitch')
    plt.scatter(roll_dist[inds], similarity[inds], s=0.5, label='Roll')
    plt.xlabel('Distance (L2)', fontsize=17), plt.ylabel('Cosine Similarity', fontsize=17), plt.legend(),
    plt.title('Similarity vs Head Pose Distance \nSame id Different Videos', fontsize=17), plt.grid(), plt.tight_layout()
    plt.show()


def pose_dist_of_best_frames_to_correct_gallery(embedding_path_q, embedding_path_g,
                                                rank_of_correct_gallery_file, head_pose_file):
    _, id_q, filename_q = filter_videos(embedding_path_q, 'test/query', detection_q)
    _, id_g, filename_g, frontal_inds_g = filter_videos(embedding_path_g, 'test/gallery', detection_g, head_pose_file, 1)

    with open(head_pose_file, 'rb') as h:
        head_pose = pickle.load(h)
    with open(rank_of_correct_gallery_file, 'rb') as h:
        gallery_rank = pickle.load(h)

    avg_dist_rank1 = []
    avg_dist_rank_larger_than_1 = []
    for ii in range(len(filename_q)):
        vid_name_q = filename_q[ii][0].split('/')[-2]
        idx_of_correct_gallery = np.where(id_g == id_q[ii])[0].item()
        vid_name_g = filename_g[idx_of_correct_gallery][0].split('/')[-2]
        pose_q = head_pose[vid_name_q]
        pose_g = head_pose[vid_name_g]
        pose_g = np.array([*pose_g.values()]).T[frontal_inds_g[idx_of_correct_gallery]]
        rank = gallery_rank[vid_name_q]
        best_rank_inds = rank == np.min(rank)
        pose_q = np.vstack([*pose_q.values()]).T

        angles_diff = (pose_q - pose_g) % 2*np.pi
        angles_diff[angles_diff > np.pi] = 2 * np.pi - angles_diff[angles_diff > np.pi]
        dist_pose = np.linalg.norm(angles_diff, axis=-1)

        pose_dist_best = dist_pose[best_rank_inds]
        pose_dist_poor = dist_pose[~best_rank_inds]

        avg_dist_rank1.append(pose_dist_best.mean())
        avg_dist_rank_larger_than_1.append(pose_dist_poor.mean())

    plt.scatter(range(len(avg_dist_rank1)), avg_dist_rank1, label='Rank 1 Frames', color='blue')
    plt.axline((0, np.mean(avg_dist_rank1)), slope=0, color='deepskyblue')
    plt.scatter(range(len(avg_dist_rank_larger_than_1)), avg_dist_rank_larger_than_1, label='Rank > 1 Frames', color='red')
    plt.axline((0, np.nanmean(avg_dist_rank_larger_than_1)), slope=0, color='salmon')
    # plt.ylim([0, 2])

    plt.xlabel('#Query'), plt.ylabel('Mean Pose Distance (L2)'), plt.legend()
    plt.grid(), plt.title('Pose Distance of Frame to Correct Gallery')


def plot_recall_at_k(recall_at_k_list, descriptions, title=None, lgd_title=None, k=None, fmt=None):
    if k is None:
        k = k_vals
    # plt.figure(dpi=200)
    for desc_idx in range(len(descriptions)):
        if fmt:
            plt.plot(k, recall_at_k_list[desc_idx], fmt[desc_idx], label=descriptions[desc_idx])
        else:
            plt.plot(k, recall_at_k_list[desc_idx], label=descriptions[desc_idx])
    plt.title(f'CMC Curve {"- " + title if title else ""}', fontsize=17)
    plt.xlabel('Rank', fontsize=17), plt.ylabel('Identification Rate', fontsize=17), plt.grid()
    lgd = plt.legend(), plt.tight_layout()
    if lgd_title:
        lgd.set_title(lgd_title)
    plt.show()


def hist_head_pose(embedding_path_q, embedding_path_g, head_pose_file, title=None):

    _, _, filename_q, frontal_inds_q = filter_videos(embedding_path_q, 'test/query', detection_, head_pose_file, 40)
    _, _, filename_g, frontal_inds_g = filter_videos(embedding_path_g, 'test/gallery', detection_, head_pose_file, 1)

    with open(head_pose_file, 'rb') as h:
        head_pose = pickle.load(h)

    yaw_q, pitch_q, roll_q = [], [], []
    yaw_g, pitch_g, roll_g = [], [], []

    for ii in range(len(filename_q)):
        vid_name = filename_q[ii][0].split('/')[-2]
        yaw_q.extend(head_pose[vid_name]['yaw'][frontal_inds_q[ii]])
        pitch_q.extend(head_pose[vid_name]['pitch'][frontal_inds_q[ii]])
        roll_q.extend(head_pose[vid_name]['roll'][frontal_inds_q[ii]])

    for ii in range(len(filename_g)):
        vid_name = filename_g[ii][0].split('/')[-2]
        yaw_g.extend(head_pose[vid_name]['yaw'][frontal_inds_g[ii]])
        pitch_g.extend(head_pose[vid_name]['pitch'][frontal_inds_g[ii]])
        roll_g.extend(head_pose[vid_name]['roll'][frontal_inds_g[ii]])

    font = {'family': 'normal',
            'size': 17}
    import matplotlib
    matplotlib.rc('font', **font)

    plt.hist(yaw_q, label='query', density=True, alpha=0.5), plt.hist(yaw_g, label='gallery', density=True, alpha=0.5)
    plt.xlabel('Yaw [rad]'), plt.ylabel('Frames Density')
    plt.xlim([-1, 1])
    plt.legend(), plt.grid()
    plt.title(f'Yaw Histogram {"- " + title if title else ""}', fontsize=17), plt.tight_layout(), plt.show()

    plt.hist(pitch_q, label='query', density=True, alpha=0.5), plt.hist(pitch_g, label='gallery', density=True, alpha=0.5)
    plt.xlabel('Pitch [rad]'), plt.ylabel('Frames Density')
    plt.xlim([-0.6, 0.6])
    plt.legend(), plt.grid()
    plt.title(f'Pitch Histogram {"- " + title if title else ""}', fontsize=17), plt.tight_layout(), plt.show()
    #
    plt.hist(roll_q, label='query', density=True, alpha=0.5), plt.hist(roll_g, label='gallery', density=True, alpha=0.5)
    plt.xlabel('Roll [rad]'), plt.ylabel('Frames Density')
    plt.xlim([-0.4, 0.4])
    plt.legend(), plt.grid()
    plt.title(f'Roll Histogram {"- " + title if title else ""}', fontsize=17), plt.tight_layout(), plt.show()


def feature_norm_vs_scale(lr_dirs, scale):
    import os
    fn = []
    for d in tqdm(lr_dirs):
        with open(os.path.join(d, 'embeddings.pickle'), 'rb') as h:
            feats = pickle.load(h)['feats']

        curr_fn = np.concatenate([np.linalg.norm(f, axis=-1) for f in feats]).mean()
        fn.append(curr_fn)

    plt.plot(scale, fn)
    plt.title('Feature Norm on Different LR Scale', fontsize=17)
    plt.grid()
    plt.xlabel('Scale', fontsize=17)
    plt.ylabel('Average Feature Norm (L2)', fontsize=17)
    plt.show()


def plot_oracle_frames(embedding_path_q, embedding_path_g, head_pose_file, oracle_dict, detection_q, detection_g):
    """
    plot frames that has low, medium and high similarity to the correct gallery
    """
    feats_q, ids_q, filename_q = filter_videos(embedding_path_q, 'test/query', detection_file=detection_q)
    feats_g, ids_g, filename_g, inds_g = filter_videos(embedding_path_g, 'test/gallery', detection_g, head_pose_file,
                                                 num_frames=1)

    with open(detection_q, 'rb') as h:
        detect_dict = pickle.load(h)
        lmk_q, score_q = detect_dict['landmarks'], detect_dict['score']
    with open(detection_g, 'rb') as h:
        detect_dict = pickle.load(h)
        lmk_g, score_g = detect_dict['landmarks'], detect_dict['score']


    filename_g = np.array(filename_g)
    num_figures = 5
    # vids_inds = np.random.choice(len(filename_q), size=num_figures, replace=False)
    vids_inds = range(len(filename_q))
    done = 0
    for i in vids_inds:
        if len(filename_q[i]) < 9:
            continue
        vid_q = filename_q[i][0].split('/')[-2]
        sim = oracle_dict[vid_q].squeeze()
        argsort = np.argsort(sim)
        curr_filename_q = filename_q[i]
        low_sim_inds = argsort[:3]
        med_sim_inds = argsort[(len(argsort)//2 - 1):(len(argsort)//2 + 2)]
        high_sim_inds = argsort[-3:]


        if sim[high_sim_inds[0]] - 0.3 > sim[low_sim_inds[-1]]:
            # plot only videos with wide range of similarity
            curr_lmk_q = lmk_q[vid_q][score_q[vid_q] > 0.01]

            norms = [np.linalg.norm(feats_q[i][low_sim_inds], axis=-1),
                     np.linalg.norm(feats_q[i][med_sim_inds], axis=-1),
                     np.linalg.norm(feats_q[i][high_sim_inds], axis=-1)]
            titles = [[f'sim: {s:.3}, norm: {n:.3}' for s, n in zip(sim_, norm)] for sim_, norm in
                      zip([sim[low_sim_inds], sim[med_sim_inds], sim[high_sim_inds]], norms)]

            vid_g_ind = np.where(ids_g == ids_q[i])[0].item()
            curr_filename_g = filename_g[vid_g_ind]
            vid_g = curr_filename_g[0].split('/')[-2]
            curr_lmk_g = lmk_g[vid_g][inds_g[vid_g_ind]][0]

            plot_image_grid([curr_filename_q[low_sim_inds], curr_filename_q[med_sim_inds], curr_filename_q[high_sim_inds]],
                            titles,
                            [curr_lmk_q[low_sim_inds], curr_lmk_q[med_sim_inds], curr_lmk_q[high_sim_inds]],
                            ['Low', 'Medium', 'High'],
                            curr_filename_g, curr_lmk_g, 'Frames to Gallery Similarity ')
            done += 1

        if done == num_figures:
            break






if __name__ == '__main__':
    from misc import filter_videos
    from metrics import get_recall_at_k, k_vals, any_hit_identification
    from plots import plot_image_grid

    splits_path = '/inputs/bionicEye/data/ytf/splits'
    # head_pose_file = '/inputs/bionicEye/data/ytf/head_pose_supplied.pickle'
    # head_pose_file = '/datasets/BionicEye/YouTubeFaces/head_pose/sixdrepnet_rad_trans.pickle'
    head_pose_file = '/datasets/BionicEye/YouTubeFaces/head_pose/head_pose_supplied.pickle'


    embedding_path_q = '/outputs/bionicEye/extract-feats/feats-bilinear-face-scale-4_28-05-2023_10-10/embeddings.pickle'
    embedding_path_g = '/inputs/bionicEye/data/ytf/embeddings.pickle'
    detection_q = '/datasets/BionicEye/YouTubeFaces/faces/bilinear_lr_scale_4/detection.pickle'
    detection_g = '/datasets/BionicEye/YouTubeFaces/faces/sharp/detection.pickle'
    r2, sim_to_correct_gallery = closest_frame_iden_oracle(embedding_path_q, embedding_path_g, head_pose_file, detection_q, detection_g)
    plot_oracle_frames(embedding_path_q, embedding_path_g, head_pose_file, sim_to_correct_gallery, detection_q, detection_g)

    # pose_similarity_corr('/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle', head_pose_file)

    # embedding_path_q = '/outputs/bionicEye/extract-feats/feats-bilinear-face-scale-4_28-05-2023_10-10/embeddings.pickle'
    # embedding_path_q = '/outputs/bionicEye/v1/extract-feats/bicdown-face-scale-4-norm-sim-r100_01-04-2023_19-54/embeddings.pickle'
    # embedding_path_g = '/inputs/bionicEye/data/ytf/embeddings.pickle'
    # detection_q = '/datasets/BionicEye/YouTubeFaces/faces/bicubic_lr_scale_4/detection.pickle'
    # detection_g = '/datasets/BionicEye/YouTubeFaces/faces/sharp/detection.pickle'
    # r1, rank_of_correct_gallery = best_frame_identification_oracle(embedding_path_q, embedding_path_g, head_pose_file, detection_q, detection_g)
    # r2, sim_to_correct_gallery = closest_frame_iden_oracle(embedding_path_q, embedding_path_g, head_pose_file, detection_q, detection_g)
    # r3 = weighted_mean_from_oracle(embedding_path_q, embedding_path_g, head_pose_file, detection_q, detection_g)
    # r4 = v2s_identification(embedding_path_q, embedding_path_g, head_pose_file, detection_q, detection_g)
    # plot_recall_at_k([r1, r2, r3, r4],
    #                  ['Oracle: Max rank frame from all frames',
    #                   'Oracle: closest frame than all frames',
    #                   'Oracle: similarity weighted frames, all frames',
    #                   'Baseline: mean of all frames'
    #                   ], title='YTF - Query LRx4')
    # with open(embedding_path_q.split('/embeddings')[0] + '/rank_of_correct_gallery.pickle', 'wb') as h:
    #     pickle.dump(rank_of_correct_gallery, h)
    # with open(embedding_path_q.split('/embeddings')[0] + '/sim_to_correct_gallery.pickle', 'wb') as h:
    #     pickle.dump(sim_to_correct_gallery, h)


    # head_pose_file = '/datasets/BionicEye/YouTubeFaces/head_pose/sixdrepnet_rad_trans.pickle'

    # hr_sim_trans_r100
    # embedding_path_q = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    # embedding_path_g = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    # r1, rank_of_correct_gallery = best_frame_identification_oracle(embedding_path_q, embedding_path_g, head_pose_file, num_query_frames=40)
    # r2 = v2s_identification(embedding_path_q, embedding_path_g, head_pose_file)
    # r3 = v2s_frontal_identification(embedding_path_q, embedding_path_g, head_pose_file, num_query_frames=40)
    # r4 = v2s_frontal_identification(embedding_path_q, embedding_path_g, head_pose_file, num_query_frames=1)
    # plot_recall_at_k([r1, r2, r3, r4],
    #                  ['Oracle: Max rank frame from 40 frontals',
    #                   'Mean of all frames',
    #                   'V2S query 40 frontals mean',
    #                   'V2S query 1 frontal',
    #                   ])


    # preprocess_results
    # r1 = v2s_identification('/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle',
    #                         '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle', head_pose_file)
    # r2 = v2s_identification('/outputs/bionicEye/v1/extract-feats/sim-trans-sharp-r100_26-03-2023_11-31/embeddings.pickle',
    #                         '/outputs/bionicEye/v1/extract-feats/sim-trans-sharp-r100_26-03-2023_11-31/embeddings.pickle', head_pose_file)
    # r3 = v2s_identification('/outputs/bionicEye/v1/extract-feats/norm-sharp-r100_26-03-2023_09-06/embeddings.pickle',
    #                         '/outputs/bionicEye/v1/extract-feats/norm-sharp-r100_26-03-2023_09-06/embeddings.pickle', head_pose_file)
    # r4 = v2s_identification('/outputs/bionicEye/v1/extract-feats/sharp-r100_26-03-2023_13-19/embeddings.pickle',
    #                         '/outputs/bionicEye/v1/extract-feats/sharp-r100_26-03-2023_13-19/embeddings.pickle', head_pose_file)
    # plot_recall_at_k([r1, r2, r3, r4],
    #                  ['Image normalization + similarity transform',
    #                   'Similarity transform',
    #                   'Image normalization',
    #                   'Raw'
    #                   ], 'V2S - All Frames Mean')

    #
    # # scenarios
    # embedding_path_q = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    # embedding_path_g = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    # r1 = v2v_identification(embedding_path_q, embedding_path_g)
    # r2 = v2s_identification(embedding_path_q,
    #                         embedding_path_g, head_pose_file)
    # r3 = s2s_identification(embedding_path_q,
    #                         embedding_path_g, head_pose_file)
    # plot_recall_at_k([r1, r2, r3],
    #                  ['V2V',
    #                   'V2S',
    #                   'S2S'], 'Different Scenarios')
    #
    #
    # # face lr comparison
    # embedding_path_g = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    # r1 = v2s_identification('/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle',
    #                         embedding_path_g, head_pose_file)
    # r2 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-face-scale-2-norm-sim-r100_01-04-2023_19-31/embeddings.pickle',
    #                         embedding_path_g, head_pose_file)
    # r3 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-face-scale-3-norm-sim-r100_01-04-2023_19-54/embeddings.pickle',
    #                         embedding_path_g, head_pose_file)
    # r4 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-face-scale-4-norm-sim-r100_01-04-2023_19-54/embeddings.pickle',
    #                         embedding_path_g, head_pose_file)
    # r5 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-face-scale-5-norm-sim-r100_01-04-2023_19-55/embeddings.pickle',
    #                         embedding_path_g, head_pose_file)
    # r6 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-face-scale-6-norm-sim-r100_01-04-2023_19-55/embeddings.pickle',
    #                         embedding_path_g, head_pose_file)
    # r7 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-face-scale-7-norm-sim-r100_01-04-2023_19-55/embeddings.pickle',
    #                         embedding_path_g, head_pose_file)
    # r8 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-face-scale-8-norm-sim-r100_01-04-2023_19-56/embeddings.pickle',
    #                         embedding_path_g, head_pose_file)
    # plot_recall_at_k([r1, r2, r3, r4, r5, r6, r7, r8],
    #                  ['1', '2', '3', '4', '5', '6', '7', '8'], 'V2S - Low Resolution Query Faces')
    #
    #
    # # full image (in lr) detection comparison
    # embedding_path_g = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    # score_detection_th = 0
    # r1 = v2s_identification('/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle',
    #     embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/faces/sharp/detection.pickle', score_detection_th)
    # print('Detection on scale 2')
    # r2 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-2-norm-sim-r100_30-03-2023_10-53/embeddings.pickle',
    #                         embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_2/detection.pickle', score_detection_th)
    # print('Detection on scale 3')
    # r3 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-3-norm-sim-r100_30-03-2023_11-43/embeddings.pickle',
    #                         embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_3/detection.pickle', score_detection_th)
    # print('Detection on scale 4')
    # r4 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-4-norm-sim-r100_30-03-2023_11-44/embeddings.pickle',
    #                         embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_4/detection.pickle', score_detection_th)
    # print('Detection on scale 5')
    # r5 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-5-norm-sim-r100_30-03-2023_11-44/embeddings.pickle',
    #                         embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_5/detection.pickle', score_detection_th)
    # print('Detection on scale 6')
    # r6 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-6-norm-sim-r100_30-03-2023_11-45/embeddings.pickle',
    #                         embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_6/detection.pickle', score_detection_th)
    # print('Detection on scale 7')
    # r7 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-7-norm-sim-r100_30-03-2023_11-45/embeddings.pickle',
    #                         embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_7/detection.pickle', score_detection_th)
    # print('Detection on scale 8')
    # r8 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-8-norm-sim-r100_30-03-2023_11-46/embeddings.pickle',
    #                         embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_8/detection.pickle', score_detection_th)
    # plot_recall_at_k([r1, r2, r3, r4, r5, r6, r7, r8],
    #                  ['1', '2', '3', '4', '5', '6', '7', '8'], 'V2S - Low Resolution Query Full Image', lgd_title='Scale')



    # filtering by factors
    # embedding_path_q = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    # embedding_path_g = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    # r1 = v2s_with_filters(embedding_path_q, embedding_path_g, head_pose_file,
    #                       '/datasets/BionicEye/YouTubeFaces/faces/sharp/detection.pickle', filters=['eyes_dist'])
    # r2 = v2s_with_filters(embedding_path_q, embedding_path_g, head_pose_file,
    #                       '/datasets/BionicEye/YouTubeFaces/faces/sharp/detection.pickle', filters=['bb_size'])
    # r3 = v2s_with_filters(embedding_path_q, embedding_path_g, head_pose_file,
    #                       '/datasets/BionicEye/YouTubeFaces/faces/sharp/detection.pickle', filters=['feat_norm'])
    # r4 = v2s_with_filters(embedding_path_q, embedding_path_g, head_pose_file,
    #                       '/datasets/BionicEye/YouTubeFaces/faces/sharp/detection.pickle',
    #                       filters=['eyes_dist', 'feat_norm'])
    # plot_recall_at_k([r1, r3, r4],
    #                  ['1', '3', '4'], 'V2S - Filters', lgd_title='Filters')

    # VSR full image detection comparison
    # embedding_path_g = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    # score_detection_th = 0
    # r1 = v2s_identification('/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle',
    #     embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/faces/sharp/detection.pickle', score_detection_th)
    # r2 = v2s_identification('/outputs/bionicEye/v1/extract-feats/dbvsr-lr-scale-2_23-04-2023_12-09/embeddings.pickle',
    #     embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/dbvsr/bicubic_lr_scale_2/detection.pickle', score_detection_th)
    # r3 = v2s_identification('/outputs/bionicEye/v1/extract-feats/dbvsr-lr-scale-4_23-04-2023_14-03/embeddings.pickle',
    #     embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/dbvsr/bicubic_lr_scale_4/detection.pickle', score_detection_th)
    # r4 = v2s_identification('/outputs/bionicEye/v1/extract-feats/dbvsr-lr-scale-8_23-04-2023_08-33/embeddings.pickle',
    #     embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/dbvsr/bicubic_lr_scale_8/detection.pickle', score_detection_th)
    # plot_recall_at_k([r1, r2, r3, r4],
    #                  ['High resolution', 'VSR on low resolution scale 2', 'VSR on low resolution scale 4', 'VSR on low resolution scale 8'],
    #                  'V2S - Video Super Resolution')

    # VSR vs LR v2s
    # embedding_path_g = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    # score_detection_th = 0
    # r0 = v2s_identification('/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle',
    #     embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/faces/sharp/detection.pickle', score_detection_th)
    # r1 = v2s_identification('/outputs/bionicEye/v1/extract-feats/dbvsr-lr-scale-2_23-04-2023_12-09/embeddings.pickle',
    #     embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/dbvsr/bicubic_lr_scale_2/detection.pickle', score_detection_th)
    # r2 = v2s_identification('/outputs/bionicEye/v1/extract-feats/dbvsr-lr-scale-4_23-04-2023_14-03/embeddings.pickle',
    #     embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/dbvsr/bicubic_lr_scale_4/detection.pickle', score_detection_th)
    # r3 = v2s_identification('/outputs/bionicEye/v1/extract-feats/dbvsr-lr-scale-8_23-04-2023_08-33/embeddings.pickle',
    #     embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/dbvsr/bicubic_lr_scale_8/detection.pickle', score_detection_th)
    # r11 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-2-norm-sim-r100_30-03-2023_10-53/embeddings.pickle',
    #                         embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_2/detection.pickle', score_detection_th)
    # r22 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-4-norm-sim-r100_30-03-2023_11-44/embeddings.pickle',
    #                         embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_4/detection.pickle', score_detection_th)
    # r33 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-8-norm-sim-r100_30-03-2023_11-46/embeddings.pickle',
    #                         embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_8/detection.pickle', score_detection_th)
    # plot_recall_at_k([r0, r1, r11, r2, r22, r3, r33],
    #                  ['High resolution',
    #                   'VSR on low resolution scale 2', 'Low resolution scale 2',
    #                   'VSR on low resolution scale 4', 'Low resolution scale 4',
    #                   'VSR on low resolution scale 8', 'Low resolution scale 8'],
    #                  'V2S \n Video Super Resolution vs Low Resolution',
    #                  fmt=['k', 'b', 'b--', 'r', 'r--', 'g', 'g--'])

    # VSR vs LR scale 4 v2s most frontal and v2s
    # embedding_path_g = '/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23/embeddings.pickle'
    # score_detection_th = 0
    # r1 = v2s_frontal_identification('/outputs/bionicEye/v1/extract-feats/dbvsr-lr-scale-4_23-04-2023_14-03/embeddings.pickle',
    #     embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/dbvsr/bicubic_lr_scale_4/detection.pickle', 1, score_detection_th)
    # r11 = v2s_identification('/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-4-norm-sim-r100_30-03-2023_11-44/embeddings.pickle',
    #                         embedding_path_g, head_pose_file,
    #                         '/datasets/BionicEye/YouTubeFaces/full/bicubic_lr_scale_4/detection.pickle', score_detection_th)
    # plot_recall_at_k([r1, r11],
    #                  ['VSR most frontal frame', 'LR all frames'],
    #                  'Down-sample Scale 4\nVideo Super Resolution vs Low Resolution',
    #                  )

    # lr_dirs = ['/outputs/bionicEye/v1/extract-feats/sharp-norm-sim-r100_28-03-2023_14-23',
    # '/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-2-norm-sim-r100_30-03-2023_10-53',
    # '/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-3-norm-sim-r100_30-03-2023_11-43',
    # '/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-4-norm-sim-r100_30-03-2023_11-44',
    # '/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-5-norm-sim-r100_30-03-2023_11-44',
    # '/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-6-norm-sim-r100_30-03-2023_11-45',
    # '/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-7-norm-sim-r100_30-03-2023_11-45',
    # '/outputs/bionicEye/v1/extract-feats/bicdown-full-scale-8-norm-sim-r100_30-03-2023_11-46']
    # scalse = [1, 2, 3, 4, 5, 6, 7, 8]
    # feature_norm_vs_scale(lr_dirs, scalse)
