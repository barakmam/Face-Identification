import torch
import numpy as np
from scipy.interpolate import interp1d

k_vals = [1, 2, 5, 10, 20, 40, 70, 100]  # np.arange(1, 200)  # [1, 2, 5, 10, 20, 40, 70, 100]  # For recall@K


def get_recall_at_k(feat_q, feat_g, query_id, gallery_id, k_vals, logits=None, reduction='mean'):
    """
    feat_q: matrix that each row is a feature vector of a different query.
    feat_g: matrix that each row is a feature vector of a different gallery.
    logits: similarity scores of shape (num_query, num_gallery)
    recall at k:
    fraction of relevant items in all recommendations that are in the k first recommendations
    """
    if logits is None:
        logits = feat_q @ feat_g.T
    if k_vals is None:
        k_vals = np.arange(len(gallery_id))

    argsort = np.argsort(-logits, 1)  # descending
    recall_at_k = []
    equals_tot = (query_id.reshape(-1, 1) == gallery_id)
    for k in k_vals:
        preds = gallery_id[argsort[:, :k]]
        equals_in_k = (query_id.reshape(-1, 1) == preds)
        curr_recall_at_k = np.sum(equals_in_k, 1) / np.sum(equals_tot, 1)
        recall_at_k.append(curr_recall_at_k)

    recall_at_k = np.stack(recall_at_k)
    if reduction == 'none':
        return recall_at_k
    return np.mean(recall_at_k, 1)


def get_tar_at_far(tar, far, percentage):
    fraction = percentage / 100
    under = np.where(far < fraction)[0].max()
    above = np.where(far > fraction)[0].min()
    return interp1d([far[under], far[above]], [tar[under], tar[above]])(fraction)


def mean_reciprocal_rank(logits, labels, gallery_id):
    logits2 = logits.amax(2)
    _, argsort = torch.sort(logits2, 1, descending=True)
    targets = gallery_id[argsort]
    mrr = (1/(1 + torch.where(labels.reshape(-1, 1) == targets)[1])).mean()
    return mrr


def any_hit_identification(feat_q, feat_g, labels, gallery_id):
    """
    for each query, if there is at least one frame from the query video that
    rank the correct gallery at rank 1 we say it is a hit.

    feat_q: shape [num_queries, num_frames, embedding_size]. normalized embeddings with norm 2
    feat_g: shape [num_galleries, embedding_size]. normalized embeddings with norm 2
    labels: id of the queries
    gallery_id: id of galleries
    """
    logits = np.einsum('qfe,ge->qfg', feat_q, feat_g)  # similarity

    # the gallery id that is most closer to each frame in each query:
    max_gallery_id = gallery_id[logits.argmax(-1)]

    # if the label of the query is in any of the frame, it is a hit
    acc = np.mean(np.any(max_gallery_id == labels.reshape(-1, 1), axis=-1))
    return acc

