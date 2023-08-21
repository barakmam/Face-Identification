import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from glob import glob
import pickle
from mycode.utils.misc import plot_table


def plot_roc(dirs, desc):
    plt.figure()
    for curr_dir, curr_desc in zip(dirs, desc):
        file = glob(curr_dir + '/**/verification_results.pickle', recursive=True)[0]
        with open(file, 'rb') as handle:
            res = pickle.load(handle)
        plt.plot(res['far'], res['tar'], label=curr_desc + f' ({res["auc"]:.3})')

    plt.grid()
    plt.title('ROC Curve', fontsize=17), plt.xlabel('FAR', fontsize=17), plt.ylabel('TAR', fontsize=17)
    plt.legend().set_title('Method (AUC)', prop={'size': 14, 'weight': 'heavy'})
    plt.show()


def plot_scalars_results(dirs, desc):
    """
    plot table of metrics:
    AP, AUC, EER, TAR @ FAR 1%, TAR @ FAR 0.1%, Majority Vote Acc, MRR
    """
    data = []
    metrics = ['AP', 'AUC', 'EER', 'TAR @ FAR 0.1%', 'TAR @ FAR 1%', 'Majority Vote Acc', 'MRR']
    for curr_dir in dirs:
        ver_res = glob(curr_dir + '/**/verification_results.pickle', recursive=True)[0]
        with open(ver_res, 'rb') as handle:
            ver_res = pickle.load(handle)
        iden_res = glob(curr_dir + '/**/identification_results.pickle', recursive=True)[0]
        with open(iden_res, 'rb') as handle:
            iden_res = pickle.load(handle)
        data.append([ver_res['ap'], ver_res['auc'], ver_res['eer'], ver_res['tar_at_far_01'],
                     ver_res['tar_at_far_1'], iden_res['majority_vote_acc'], iden_res['mrr']])

    plot_table(metrics, desc, data, 'Recognition Results')


def plot_recall_at_k(dirs, desc):
    plt.figure()
    for curr_dir, curr_desc in zip(dirs, desc):
        file = glob(curr_dir + '/**/identification_results.pickle', recursive=True)[0]
        with open(file, 'rb') as handle:
            res = pickle.load(handle)
        plt.plot(res['recall_at_k'][0], res['recall_at_k'][1], '--o', label=curr_desc)

    plt.grid(), plt.legend()
    plt.title('Recall @ K', fontsize=17), plt.xlabel('K', fontsize=17), plt.ylabel('Recall', fontsize=17)
    plt.show()


def plot_verification_scores(scores, labels):
    plt.hist(scores[labels == 1], label='Same', fontsize=17, alpha=0.5)
    plt.hist(scores[labels == 0], label='Different', fontsize=17, alpha=0.5)
    plt.title('Verification Scores Hist', fontsize=17), plt.grid(), plt.legend()
    plt.xlabel('Similarity Score', fontsize=17), plt.ylabel('# pairs', fontsize=17)
    plt.show()


if __name__ == '__main__':
    run_dirs = ['/outputs/bionicEye/singlerun/sharp/sharp_11-10-2022_14-44',
                '/outputs/bionicEye/singlerun/bic_down/dbvsr_bicdown_11-10-2022_15-23',
                '/outputs/bionicEye/singlerun/bic_down/video_inr_bic_down_30-10-2022_14-38',
                '/outputs/bionicEye/singlerun/bic_down/bicubic_sr_bicdown_30-10-2022_15-03']
    description = ['Sharp', 'DBVSR',  'VideoINR', 'Bicubic SR']

    plot_scalars_results(run_dirs, description)
    plot_roc(run_dirs, description)
    plot_recall_at_k(run_dirs, description)




