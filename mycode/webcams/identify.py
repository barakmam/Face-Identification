import cv2
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import hydra
from glob import glob
from omegaconf import DictConfig
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.backbones import get_model
from data.transform import SimilarityTrans
from detection.retinaface.detection import detection
from detection.retinaface.retinaface import RetinaFace


class SingleImage(Dataset):
    def __init__(self, fullname, lmk):
        super().__init__()
        self.landmarks = lmk
        self.dataset = fullname
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        self.sim_trans = SimilarityTrans()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.dataset[idx]), cv2.COLOR_BGR2RGB)
        image = self.sim_trans(image, self.landmarks[idx])
        image = self.transform(image)
        return image


def extract_feats(images_names, lmk, model, batch_size, device):
    print('Extracting Features...')
    dataloader = DataLoader(SingleImage(images_names, lmk), drop_last=False, shuffle=False, batch_size=batch_size)
    feats = []
    for x in dataloader:
        feats.append(model(x.to(device)).detach().cpu().numpy())
    feats = np.vstack(feats)
    return feats


@hydra.main(config_path="configs", config_name="identify")
def main(cfg: DictConfig) -> None:
    device = 'cuda' if torch.cuda.is_available() and (cfg.system.num_gpus > 0) else 'cpu'

    # get the pretrained detection transformer. this retinaface is mx module (not torch)
    detection_model = RetinaFace(cfg.model.detection.weights_path_prefix, 0, 0, 'net3')

    # get the pretrained face embedder transformer (torch ResNet-100 transformer)
    embedder_model = get_model('r100', fp16=False).to(device)
    state_dict = torch.load(cfg.model.fr.weights_path, map_location=device)
    embedder_model.load_state_dict(state_dict)
    embedder_model.eval()

    # Query - from frames to features: perform face and landmarks detection -> then extract features.
    frames_q = glob(cfg.data.query_video_dir + '/*')
    bb, lmk, score_q = detection(cfg.data.query_video_dir, detection_model, cfg.model.detection.threshold)
    feats_q = extract_feats(frames_q, lmk, embedder_model, cfg.data.batch_size, device)
    feats_q = normalize(feats_q.mean(0, keepdims=True))

    # Gallery - from frames to features: perform face and landmarks detection -> then extract features.
    images_g = glob(cfg.data.gallery_images_dir + '/*')
    bb, lmk, score = detection(cfg.data.gallery_images_dir, detection_model, cfg.model.detection.threshold)
    feats_g = extract_feats(images_g, lmk, embedder_model, cfg.data.batch_size, device)
    feats_g = normalize(feats_g)

    # report the ID of the query by the name of the closest gallery subject.
    logits = (feats_q @ feats_g.T).squeeze()
    max_ind = logits.argmax().item()
    pred = images_g[max_ind]
    confidence = logits[max_ind]
    confidence = 100 * (confidence + 1) / 2  # from [-1, 1] to percentages
    print('#'*20)
    print(f'The prediction for the query: {cfg.data.query_video_dir} \nis the gallery: {pred} \nwith confidence: {confidence:.0f}%')
    print('#'*20)


if __name__ == "__main__":
    main()
