import os
import pickle
from collections import Counter
from mycode.utils.misc import plot_counts_hist, get_query_feats, get_gallery_feats
from mycode.utils.metrics import get_recall_at_k, get_tar_at_far, mean_reciprocal_rank
from sklearn import metrics
import matplotlib.pyplot as plt
from ArcFace_paulpias.config import get_config
import torch
import numpy as np
from mycode.utils.misc import read_cfg
import time
from mycode.utils.misc import compare_models
from hydra.core.global_hydra import GlobalHydra
from os.path import join
from torch.utils.data import DataLoader
from mycode.data.multi_epoch_dataloader import MultiEpochsDataLoader


def identification_test(cfg=None, run_dir='.', logger=None):
    if cfg is None:
        cfg = read_cfg(join(run_dir, '.hydra'), 'config.yaml')

    query_feats, query_id, query_filename = get_query_feats(os.path.join(cfg.model.fr.embeddings_path, 'embeddings.pickle'))
    gallery_feats, gallery_id, gallery_filename = get_gallery_feats(
        '/datasets/BionicEye/YouTubeFaces/faces/sharp/embeddings.pickle', '/inputs/bionicEye/data/ytf_new_splits')

    query_feats = np.vstack([frames_feats.mean(0) for frames_feats in query_feats])
    gallery_feats = np.vstack([frames_feats[len(frames_feats)//2] for frames_feats in gallery_feats])

    logits = query_feats @ gallery_feats.T
    # logits = np.einsum("vfe,ie->vif", query_feats, gallery_feats[:, [0], :].squeeze())
    # logits: [batch video idx, gallery video idx, batch frame idx, gallery frame idx]

    if not cfg.data.distractors:
        # all gallery ids exist in query
        not_distractors = torch.isin(gallery_id, query_id)
        logits = logits[:, not_distractors]
        gallery_id = gallery_id[not_distractors]
        gallery_filename = gallery_filename[not_distractors]
    if not cfg.data.nobody:
        # all query ids exist in gallery
        exist_in_gallery = torch.isin(query_id, gallery_id)
        logits = logits[exist_in_gallery]
        query_id = query_id[exist_in_gallery]
        query_filename = query_filename[exist_in_gallery.cpu()]

    pred_score, argmax_gallery = logits.max(1)  # max over gallery images
    pred_id = gallery_id[argmax_gallery].cpu().numpy()
    pred_filename = gallery_filename[argmax_gallery.cpu()]

    k_vals = [1, 5, 10, 20, 50, 100]
    recall_at_k = get_recall_at_k(query_feats, gallery_feats, query_id, gallery_id, k_vals)
    mrr = mean_reciprocal_rank(logits, query_id, gallery_id)

    target_majority_vote = torch.zeros(len(query_id)).to(query_id.device)
    count_majority_vote = torch.zeros(len(query_id)).to(query_id.device)
    for ii in range(len(query_id)):
        most_common = Counter(pred_id[ii]).most_common(1)[0]
        target_majority_vote[ii] = most_common[0]
        count_majority_vote[ii] = most_common[1]

    majority_vote_equals = target_majority_vote.eq(query_id)
    majority_vote_acc = torch.mean(majority_vote_equals.float())

    if logger is not None:
        for k, rec in zip(k_vals, recall_at_k):
            logger.experiment.add_scalar('Recall @ K - ' + cfg.data.test, rec, k)
        logger.experiment.add_scalar('Majority Vote Acc - ' + cfg.data.test, majority_vote_acc)
        logger.experiment.add_scalar('MRR - ' + cfg.data.test, mrr)

        fig = plot_counts_hist(count_majority_vote, majority_vote_equals)

        logger.experiment.add_figure('Target Majority Vote Count Hist - ' + cfg.data.test, fig)

    if cfg.model.fr.output.save:
        save_d = {"pred_id": pred_id, "pred_score": pred_score.cpu(), "pred_filename": pred_filename}
        with open('output_preds.pickle', 'wb') as handle:
            pickle.dump(save_d, handle, protocol=pickle.HIGHEST_PROTOCOL)
        save_d = {"majority_vote_acc": majority_vote_acc.cpu(), "recall_at_k": [k_vals, recall_at_k], "mrr": mrr}
        with open('identification_results.pickle', 'wb') as handle:
            pickle.dump(save_d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('\nTest Majority Vote Acc: {:.4f}'.format(majority_vote_acc))
    print(f'\nTest Recall @ K:\n\t {k_vals}\n\t {recall_at_k}')
    print('\nTest MRR: {:.4f}\n'.format(mrr))


def verification(cfg=None, run_dir='.', logger=None):
    if cfg is None:
        cfg = read_cfg(join(run_dir, '.hydra'), 'config.yaml')

    embeddings_file = join(cfg.model.fr.embeddings_path, 'embeddings.pickle')
    with open(embeddings_file, 'rb') as handle:
        embeddings = pickle.load(handle)

    if cfg is None:
        cfg = read_cfg(join(run_dir, '.hydra'), 'config.yaml')

    feats = embeddings['feats']
    filename = embeddings['filename'][:, 0]
    filename = np.array([name.split('/')[-2] for name in filename])
    score = []
    label = []
    filename_1 = []
    filename_2 = []

    data = YoutubeFacesVerification()
    pairs_loader = MultiEpochsDataLoader(data, batch_size=256, num_workers=0)
    for name1, name2, is_same in pairs_loader:
        name1 = np.array(name1)
        name2 = np.array(name2)
        feat1 = feats[np.where(name1.reshape(-1, 1) == filename)[1]]
        feat2 = feats[np.where(name2.reshape(-1, 1) == filename)[1]]
        score.append((feat1.mean(1) * feat2.mean(1)).sum(1))
        # the score is the corresponding frames with minimal distance.
        # can be changed to minimum over all possible frames coupling between the two videos.
        label.append(is_same)
        filename_1.append(name1)
        filename_2.append(name2)

    filename_1 = np.concatenate(filename_1)
    filename_2 = np.concatenate(filename_2)
    score = torch.concat(score).cpu().numpy()
    label = torch.concat(label).float()

    ap = metrics.average_precision_score(label, score)
    far, tar, thresholds = metrics.roc_curve(label, score)
    frr = 1 - tar
    auc = metrics.roc_auc_score(label, score)
    eer = far[np.nanargmin(np.absolute((frr - far)))]
    tar_at_far_01 = get_tar_at_far(tar, far, 0.1)
    tar_at_far_1 = get_tar_at_far(tar, far, 1)

    if logger is not None:
        logger.experiment.add_scalar(f'TAR @ FAR 0.1 % {cfg.data.test}', tar_at_far_01)
        logger.experiment.add_scalar(f'TAR @ FAR 1 % {cfg.data.test}', tar_at_far_1)
        logger.experiment.add_scalar(f'EER {cfg.data.test}', eer)
        logger.experiment.add_scalar(f'AP {cfg.data.test}', ap)
        logger.experiment.add_scalar(f'AUC {cfg.data.test}', auc)

        fig = plt.figure()
        plt.plot(far, tar), plt.xlabel('FAR', fontsize=17), plt.ylabel('TAR', fontsize=17)
        plt.title(f'ROC Curve with AUC: {auc:.3f}', fontsize=17)
        logger.experiment.add_figure(f'ROC Curve - {cfg.data.test}', fig)

        fig = plt.figure()
        plt.hist(score[label == 1], label='Same', alpha=0.5), plt.hist(score[label == 0], label='Different', alpha=0.5)
        plt.xlabel('Score', fontsize=17), plt.ylabel('# pairs', fontsize=17), plt.title(f'Scores Hist', fontsize=17)
        logger.experiment.add_figure(f'Scores Hist - {cfg.data.test}', fig)

    print(f'\nTAR @ FAR 0.1 % {cfg.data.test}: {tar_at_far_01:.3f}')
    print(f'TAR @ FAR 1 % {cfg.data.test}: {tar_at_far_1:.3f}')
    print(f'EER {cfg.data.test}: {eer:.3f}')
    print(f'AP {cfg.data.test}: {ap:.3f}')
    print(f'AUC {cfg.data.test}: {auc:.3f}')

    if cfg.model.fr.output.save:
        save_d = {"filename_1": filename_1, "filename_2": filename_2,
                  "label": label, "score": score}
        with open('verification_preds.pickle', 'wb') as handle:
            pickle.dump(save_d, handle, protocol=pickle.HIGHEST_PROTOCOL)
        save_d = {"tar_at_far_01": tar_at_far_01, "tar_at_far_1": tar_at_far_1,
                  "eer": eer, "ap": ap, "far": far, "tar": tar, 'auc': auc}
        with open('verification_results.pickle', 'wb') as handle:
            pickle.dump(save_d, handle, protocol=pickle.HIGHEST_PROTOCOL)


def verification_metrics(run_dir):
    output_file = join(run_dir, 'verification_preds.pickle')
    with open(output_file, 'rb') as handle:
        output = pickle.load(handle)

    label = output['label']
    score = output['score']
    ap = metrics.average_precision_score(label, score)
    far, tar, thresholds = metrics.roc_curve(label, score)
    frr = 1 - tar
    auc = metrics.roc_auc_score(label, score)
    eer = far[np.nanargmin(np.absolute((frr - far)))]
    tar_at_far_01 = get_tar_at_far(tar, far, 0.1)
    tar_at_far_1 = get_tar_at_far(tar, far, 1)

    save_d = {"tar_at_far_01": tar_at_far_01, "tar_at_far_1": tar_at_far_1,
              "eer": eer, "ap": ap, "far": far, "tar": tar, 'auc': auc}
    with open(join(run_dir, 'verification_results.pickle'), 'wb') as handle:
        pickle.dump(save_d, handle, protocol=pickle.HIGHEST_PROTOCOL)


# def identification_metrics(run_dir):
#     output_file = join(run_dir, 'faceRec_out.pickle')
#     with open(output_file, 'rb') as handle:
#         output = pickle.load(handle)
#     logits, query_feats, query_id, query_filename, gallery_feats, gallery_id, gallery_filename = output.values()
#
#     k_vals = [1, 5, 10, 20, 50, 100]
#     recall_at_k = get_recall_at_k(logits, query_id, gallery_id, k_vals)
#     mrr = mean_reciprocal_rank(logits, query_id, gallery_id)
#
#     with open(join(run_dir, 'output_preds.pickle'), 'rb') as handle:
#         preds = pickle.load(handle)
#
#     pred_id = preds['pred_id']
#     target_majority_vote = torch.zeros(len(query_id)).to(query_id.device)
#     count_majority_vote = torch.zeros(len(query_id)).to(query_id.device)
#     for ii in range(len(query_id)):
#         most_common = Counter(pred_id[ii]).most_common(1)[0]
#         target_majority_vote[ii] = most_common[0]
#         count_majority_vote[ii] = most_common[1]
#
#     majority_vote_equals = target_majority_vote.eq(query_id)
#     majority_vote_acc = torch.mean(majority_vote_equals.float())
#
#     save_d = {"majority_vote_acc": majority_vote_acc.cpu(), "recall_at_k": [k_vals, recall_at_k], "mrr": mrr}
#     with open(join(run_dir, 'identification_results.pickle'), 'wb') as handle:
#         pickle.dump(save_d, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # identification_test(run_dir="/outputs/bionicEye/singlerun/bic_down/bicubic_sr_bicdown_30-10-2022_15-03")
    verification(run_dir="/outputs/bionicEye/singlerun/sharp/sharp_11-10-2022_14-44")


