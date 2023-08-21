import os
import pickle
# import distinctipy
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


def plot_identification_samples(logits, id_q, id_g, filename_q, filename_g, bb_q, bb_g, save=False):
    """
        take TP cases with low\medium\high confidence
        and for each of them plot the gallery image and several frames from the query video.
        Do the same for FP.
    """
    id_g_rank1_idx = np.array([logits[row].argmax() for row in range(len(id_q))])
    tp_idx = np.array([row for row in range(len(id_q)) if id_q[row] == id_g[logits[row].argmax()]])
    tp_vals = logits[tp_idx].max(1)
    sorted_tp_idx = tp_idx[tp_vals.argsort()]
    low_tp_idx = sorted_tp_idx[1]
    med_tp_idx = sorted_tp_idx[len(sorted_tp_idx)//2]
    high_tp_idx = sorted_tp_idx[-2]

    fp_idx = np.array([row for row in range(len(id_q)) if id_q[row] != id_g[logits[row].argmax()]])
    fp_vals = logits[fp_idx].max(1)
    sorted_fp_idx = fp_idx[fp_vals.argsort()]
    low_fp_idx = sorted_fp_idx[1]
    med_fp_idx = sorted_fp_idx[len(sorted_fp_idx) // 2]
    high_fp_idx = sorted_fp_idx[-2]

    path_to_save = '/outputs/webcams_test/plot_identification_samples'
    os.makedirs(path_to_save, exist_ok=True)
    for ii, (q_idx, title) in enumerate(zip([low_tp_idx, med_tp_idx, high_tp_idx, low_fp_idx, med_fp_idx, high_fp_idx],
                            ['TP - Low', 'TP - Medium', 'TP - High', 'FP - Low', 'FP - Medium', 'FP - High'])):
        q_inds = np.random.choice(len(filename_q[q_idx]), 3, replace=False)
        q_frames_to_plot = filename_q[q_idx][q_inds]
        face_bb_q = bb_q[q_idx][q_inds]

        g_img = [filename_g[id_g_rank1_idx[q_idx]]]
        face_bb_g = bb_g[id_g_rank1_idx[q_idx]].reshape(1, -1)
        correct_gallery_idx = np.where(id_g == id_q[q_idx])[0].item()
        if correct_gallery_idx != id_g_rank1_idx[q_idx]:
            g_img.append(filename_g[correct_gallery_idx])
            face_bb_g = np.vstack([face_bb_g, bb_g[correct_gallery_idx]])
        plot_rank1_res(q_frames_to_plot, g_img, face_bb_q, face_bb_g, title + ' Similarity', f'{path_to_save}/{ii}.png',
                       save)


def plot_rank1_res(query_frames, gallery_image, query_bb, gallery_bb, fig_title, dst_save_path, save):
    # Create a new figure
    fig = plt.figure(figsize=(12, 6))

    # Add a title to the figure
    fig.suptitle(fig_title)

    # Create a grid of 2 rows and 4 columns
    cols_num = len(query_frames) + len(gallery_image)
    grid = plt.GridSpec(2, cols_num, width_ratios=[1]*cols_num)

    # Plot the query frames
    query_title = 'Query Frames'
    query_title_ax = fig.add_subplot(grid[0, :3])
    query_title_ax.set_title(query_title, fontsize=14, fontweight='bold', y=1)
    query_title_ax.axis('off')
    for i in range(len(query_frames)):
        ax = fig.add_subplot(grid[0, i])
        img = plt.imread(query_frames[i])
        ax.imshow(img)
        ax.axis('off')

        bb = query_bb[i].astype('int')
        ax = fig.add_subplot(grid[1, i])
        img = plt.imread(query_frames[i])
        face_image = img[bb[1]:(bb[1]+bb[3]), bb[0]:(bb[0]+bb[2])]
        ax.imshow(face_image)
        ax.axis('off')

    # Plot the gallery image
    grid_idx = len(query_frames)
    gallery_title = 'Rank 1 Gallery'
    gallery_title_ax = fig.add_subplot(grid[0, grid_idx])
    gallery_title_ax.set_title(gallery_title, fontsize=14, fontweight='bold', y=1)
    gallery_title_ax.axis('off')
    ax = fig.add_subplot(grid[0, grid_idx])
    img = plt.imread(gallery_image[0])
    ax.imshow(img)
    ax.axis('off')

    ax = fig.add_subplot(grid[1, grid_idx])
    img = plt.imread(gallery_image[0])
    bb = gallery_bb[0].astype('int')
    face_image = img[bb[1]:(bb[1] + bb[3]), bb[0]:(bb[0] + bb[2])]
    ax.imshow(face_image)
    ax.axis('off')

    if len(gallery_image) == 2:
        # if there are 2 gallery images, then the first one is the rank 1 gallery
        # and the second one is the correct gallery image (for FP case)
        grid_idx = len(query_frames) + 1
        gallery_title = 'Correct Gallery'
        gallery_title_ax = fig.add_subplot(grid[0, grid_idx])
        gallery_title_ax.set_title(gallery_title, fontsize=14, fontweight='bold', y=1)
        gallery_title_ax.axis('off')
        ax = fig.add_subplot(grid[0, grid_idx])
        img = plt.imread(gallery_image[1])
        ax.imshow(img)
        ax.axis('off')

        ax = fig.add_subplot(grid[1, grid_idx])
        img = plt.imread(gallery_image[1])
        bb = gallery_bb[1].astype('int')
        face_image = img[bb[1]:(bb[1] + bb[3]), bb[0]:(bb[0] + bb[2])]
        ax.imshow(face_image)
        ax.axis('off')

    if save:
        plt.savefig(dst_save_path)
        print('fig saved in: ', dst_save_path)
    # Show the figure
    plt.show()


def plot_full_images_with_bb(ytf_dir):
    src_dir = os.path.join(ytf_dir, 'frame_images_DB')
    txt_files = glob(src_dir + '/*.txt')
    images_to_plot = []
    num_images = 10
    txt_files = np.random.permutation(txt_files)
    for ii in range(len(txt_files)):
        filename = txt_files[ii]
        with open(filename) as fp:
            Lines = fp.readlines()
            if len(Lines) == 0:
                print(filename, ' has no bbs')
                continue

            line = Lines[np.random.randint(len(Lines))]
            splits = line.split(',')
            w = int(splits[4])
            h = int(splits[5])
            x0 = max(0, int(splits[2]) - w // 2)
            y0 = max(0, int(splits[3]) - h // 2)
            splits[0] = splits[0].replace("\\", '/')
            frame = plt.imread(os.path.join(src_dir, splits[0]))
            face = frame[y0:(y0 + h), x0:(x0 + w)]
            images_to_plot.append(face)
        if len(images_to_plot) == num_images:
            break

    fig, axes = plt.subplots(nrows=2, ncols=num_images // 2, figsize=(10, 5))
    axes = axes.flatten()

    for i, img in enumerate(images_to_plot):
        axes[i].imshow(img)
        axes[i].axis('off')

    fig.suptitle('YouTubeFaces Samples', fontsize=17)
    plt.tight_layout()
    plt.savefig('')
    plt.show()
    # pdf_pages = PdfPages('/outputs/bionicEye/thesis/images/ytf_samples.pdf')
    # pdf_pages.savefig()
    # pdf_pages.close()


def plot_low_resolution(src_dir):
    lr_dirs = sorted([name for name in os.listdir(src_dir) if 'lr_scale' in name])
    hr_dir = os.path.join(src_dir, 'sharp')
    all_dirs = [hr_dir] + lr_dirs
    scale = [1] + [int(dir_name[-1]) for dir_name in lr_dirs]
    num_vids = 5
    vids_names = np.random.choice(os.listdir(os.path.join(hr_dir, 'videos')), num_vids, replace=False)
    grid = []
    for vid in vids_names:
        images = []
        frame = np.random.choice(os.listdir(os.path.join(src_dir, hr_dir, 'videos', vid)))
        for s, d in zip(scale, all_dirs):
            im = plt.imread(os.path.join(src_dir, d, 'videos', vid, frame))
            images.append(im)
        grid.append(images)

    fig, axs = plt.subplots(nrows=len(grid), ncols=len(grid[0]))
    fig.subplots_adjust(hspace=0, wspace=0)

    for i, row in enumerate(grid):
        for j, img in enumerate(row):
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
            axs[i, j].set_xticklabels([])
            axs[i, j].set_yticklabels([])
            if i == 0:
                axs[i, j].set_title(f'Scale: {scale[j]}', fontsize=10)

    fig.suptitle('Low Resolution Images Comparison', fontsize=17)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_full_images_with_bb('/datasets/BionicEye/YouTubeFaces/raw')
    # plot_low_resolution('/datasets/BionicEye/YouTubeFaces/faces')

