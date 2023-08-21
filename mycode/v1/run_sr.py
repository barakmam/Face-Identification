import glob
import os
import matplotlib.pyplot as plt
import torch
import hydra
from joblib import delayed, Parallel
from omegaconf import DictConfig
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.misc import print_dict, read_cfg
import time
from mycode.data.transform import get_transforms_dict
from DBVSR.code.model.pwc_recons import PWC_Recons
from tqdm import tqdm
from PIL import Image
import pickle

torch.backends.cudnn.benchmark = False


def get_model(device):
    # by the params in DBVSR/code/inference.py
    model = PWC_Recons(
        n_colors=3, n_sequence=5, extra_RBS=1, recons_RBS=3, n_feat=128, scale=4, device=device
    )
    model_path = '/inputs/bionicEye/DBVSR/pretrain_models/gaussian_e1r3.pt'
    model.load_state_dict(torch.load(model_path), strict=True)
    print('\nPretrained DBVSR transformer weights found in {}'.format(model_path))
    if device == 'cuda':
        model = model.cuda()
    return model


def run_DBVSR(cfg):
    device = 'cuda'
    model = get_model(device).to(device)
    transform = get_transforms_dict(cfg)['test']
    vids = os.listdir(os.path.join(cfg.data.path, 'videos'))
    # with open('/inputs/bionicEye/data/ytf/splits/test/query.pickle', 'rb') as h:
    #     vid_test_query = pickle.load(h)
    # with open('/inputs/bionicEye/data/ytf/splits/test/gallery.pickle', 'rb') as h:
    #     vid_test_gallery = pickle.load(h)
    # vids = list(vid_test_query) + list(vid_test_gallery)

    def parallel(vid, model):
        print('Start video {}'.format(vid))
        frames = glob.glob(os.path.join(cfg.data.path, 'videos', vid, '*'))
        out_dir = os.path.join(cfg.data.dst_dir, 'videos', vid)
        os.makedirs(out_dir, exist_ok=True)
        for ii in range(len(frames) - 5):  # DBVSR receive 5 frames as input, the target frame is the middle one
            try:
                target_frame_name = frames[ii + 2].split('/')[-1].split('.')[0]
                if os.path.isfile(os.path.join(out_dir, target_frame_name.zfill(5) + '.png')):
                    continue
                inputs = []
                for jj in range(ii, ii + 5):
                    inputs.append(transform(plt.imread(frames[jj])))
                inputs = torch.stack(inputs).to(device).unsqueeze(0)
                out = model({"x": inputs})[0]["recons"].detach().clamp(0, 1).cpu().numpy().squeeze().transpose(1, 2, 0)
                out = (out * 255).astype('uint8')
                Image.fromarray(out).save(os.path.join(out_dir, target_frame_name.zfill(5) + '.png'))
            except Exception as e:
                print('Error in video {}'.format(vid), ' in ii = {}'.format(ii), 'error:', e)
                continue
        print('Finish video {}'.format(vid))
        sys.stdout.flush()

    for vid in tqdm(vids):
        parallel(vid, model)


@hydra.main(config_path="./configs", config_name="run_sr")
def main(cfg: DictConfig) -> None:
    print_dict(cfg)
    run_DBVSR(cfg)


if __name__ == "__main__":
    main()