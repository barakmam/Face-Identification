import os
import numpy as np
from ArcFace_paulpias.model import Backbone
import pytorch_lightning as pl
import torch
from torch import nn
import matplotlib.pyplot as plt
import pickle
from torchmetrics import PeakSignalNoiseRatio as PSNR
from insightFace.recognition.arcface_torch.backbones.mobilefacenet import get_mbf
import torch.nn.functional as F
from mycode.utils.metrics import get_recall_at_k



class DBVSR(pl.LightningModule):

    def __init__(self, cfg, cuda='cuda'):
        super().__init__()
        """
        Inputs:
            cfg with all the necessary hyper-params to use
        """
        # log hyper-parameters
        self.sr_scale = cfg.sr_scale
        self.cfg = cfg
        self.lr = cfg.lr
        self.weight_decay = cfg.loss.weight_decay
        self.num_input_frames = cfg.num_input_frames
        self.is_cuda = cuda
        self.l2_reg = cfg.loss.l2_reg
        self.model = self.get_model(device=cuda)
        self.losses, self.losses_names = self.choose_loss(cfg)
        self.save_hyperparameters(logger=True)

    def choose_loss(self, cfg):
        losses = []
        if 'l1' in cfg.loss.names:
            losses.append(nn.L1Loss())
        return losses, cfg.loss.names

    def forward(self, x):
        x = self.model({"x": x})
        return x

    def training_step(self, batch, batch_idx):
        return self.step(batch, stage='train')

    def training_epoch_end(self, outputs):
        AP, avg_loss = self.epoch_end(outputs, 'train', self.trainer.datamodule.train_data)
        self.log(f"train_{self.trainer.datamodule.test_data}_loss", avg_loss)
        self.logger.output.train_AP.append(AP)
        self.logger.output.train_loss.append(avg_loss.item())

    def test_step(self, batch, batch_idx):
        x, y, filename = batch
        feats = self(x.view(x.shape[0]*x.shape[1], *x.shape[2:])).view(x.shape[0], x.shape[1], -1)

        batch_dictionary = {
            # REQUIRED: It is required for us to return "loss"
            # "loss": loss,
            "feats": feats,
            "id": y,
            "filename": np.array(filename)
        }
        return batch_dictionary

    @staticmethod
    def get_model(device='cuda'):
        from DBVSR.code.model.pwc_recons import PWC_Recons
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


class FaceRecModule(pl.LightningModule):

    def __init__(self, cfg, cuda=False):
        super().__init__()
        """
        Inputs:
            cfg with all the necessary hyper-params to use
        """
        # log hyper-parameters
        self.cfg = cfg
        self.is_cuda = cuda
        self.model = self.get_model(arch=cfg.arch, cuda=cuda)
        self.save_hyperparameters(logger=True)

    def forward(self, x):
        x = self.model(x)
        return x

    def test_step(self, batch, batch_idx):
        x, y, filename_ind = batch
        # feats = self(x.view(x.shape[0]*x.shape[1], *x.shape[2:])).view(x.shape[0], x.shape[1], -1)
        feats = self(x)  # for videos of different length

        batch_dictionary = {
            "feats": feats,
            "id": y,
            "filename_ind": filename_ind
        }
        return batch_dictionary

    def test_epoch_end(self, outputs):
        # for several processes in ddp mode:
        # if self.trainer.is_global_zero:
        #     # print('\n\n ******** Rank: ', torch.distributed.get_rank(), ' Exited on Return!')
        #     # return
        #     print('\n\n ******** Rank: ', torch.distributed.get_rank(), ' Got Here! -- 1')
        #     if torch.distributed.get_world_size() > 1:
        #         # get all processes outputs
        #         all_rank_outputs = [None for _ in range(torch.distributed.get_world_size())]
        #         torch.distributed.all_gather_object(all_rank_outputs, outputs)
        #         outputs = [x for y in all_rank_outputs for x in y]
        #     print('\n\n ******** Rank: ', torch.distributed.get_rank(), ' Got Here! -- 2')

        feats = np.concatenate([x['feats'].cpu().numpy() for x in outputs])
        id_ = np.concatenate([x['id'].cpu().numpy() for x in outputs])
        filename_inds = np.concatenate([x['filename_ind'].cpu().numpy() for x in outputs])

        filename = self.trainer.datamodule.test.data.dataset[filename_inds]
        vids_name_all = np.array([name.split('/')[-2] for name in filename])
        vids_name_unique = np.unique(vids_name_all)

        feats_list = []
        id_list = []
        filename_list = []
        for vid_name in vids_name_unique:
            inds_of_vid = vid_name == vids_name_all
            feats_list.append(feats[inds_of_vid])
            id_list.append(id_[inds_of_vid][0])
            filename_list.append(filename[inds_of_vid])


        if self.cfg.embeddings_path:
            save_d = {"feats": feats_list, "id": np.array(id_list), "filename": filename_list}
            with open(os.path.join(self.cfg.embeddings_path, 'embeddings.pickle'), 'wb') as handle:
                pickle.dump(save_d, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('\n\nembeddings saved in: ', os.path.join(self.cfg.embeddings_path, 'embeddings.pickle'))

    @staticmethod
    def get_model(arch='resnet', cuda=True):
        if arch == 'mbf':
            model = get_mbf(fp16=True, num_features=512)
            model_path = '/outputs/bionicEye/ms1m-retinaface/pretrain-mbf_18-01-2023_13-40/Code/insightFace/recognition/arcface_torch/train_outputs/transformer.pt'
            model.load_state_dict(torch.load(model_path))
            print('\nUsing Model Architecture: MobileFaceNet')
        elif arch == 'r100':
            from insightFace.recognition.arcface_torch.backbones import get_model as get_model_insightface
            model = get_model_insightface('r100', fp16=False)
            model_path = '/inputs/bionicEye/insightFace/models/torch/r100_ms1mv3/backbone.pth'
            model.load_state_dict(torch.load(model_path))
            print('\nUsing Model Architecture: ResNet-100')
        else:
            # by the params in ArcFace_paulpias.config.get_config
            net_depth = 50
            drop_ratio = 0.6
            net_mode = 'ir_se'
            model = Backbone(net_depth, drop_ratio, net_mode)
            model_path = '/inputs/RUNAI/code/ArcFace_paulpias/work_space/save/model_final.pth'
            model.load_state_dict(torch.load(model_path))
            print('\nUsing Model Architecture: ResNet-50')

        print('\nPretrained FaceRec transformer weights found in {}'.format(model_path))
        if cuda:
            model = model.cuda()
        return model

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


class SRFR(pl.LightningModule):
    """
    Super-resolution transformer concatenated to face recognition transformer
    """

    def __init__(self, cfg, cuda=False):
        super().__init__()
        """
        Inputs:
            cfg with all the necessary hyper-params to use
        """
        # log hyper-parameters
        self.sr_model = DBVSR(cfg.transformer.sr, 'cuda' if cuda else 'cpu')
        self.fr_model = FaceRecModule(cfg.transformer.fr, cuda)

        self.cfg = cfg
        self.lr = cfg.transformer.lr
        self.l2_reg = cfg.transformer.loss.l2_reg
        self.is_cuda = cuda
        self.losses, self.losses_names = self.choose_loss(cfg)
        self.psnr = PSNR(data_range=1)
        self.save_hyperparameters(logger=True)

    def choose_loss(self, cfg):
        losses = []
        if 'triplet_loss' in cfg.transformer.loss.names:
            losses.append(nn.TripletMarginLoss())
        if 'l1' in cfg.transformer.loss.names:
            losses.append(nn.L1Loss())
        return losses, cfg.transformer.loss.names

    def forward(self, x):
        # x: [Batch, Video, Frame, C, H, W]
        shape = x.shape
        x = x.view(shape[0] * shape[1], *shape[2:])
        x = self.sr_model(x)
        x = x[0]["recons"]
        features = self.fr_model(x)
        x = x.view(shape[0], shape[1], shape[3],
                   shape[4] * self.sr_model.sr_scale, shape[5] * self.sr_model.sr_scale)
        features = features.view(shape[0], shape[1], -1)
        return x, features

    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch
        x_sr, feats = self(x)
        loss = 0
        for idx, loss_func in enumerate(self.losses):
            if type(loss_func) == nn.TripletMarginLoss:
                anchor = feats[:, 0]
                pos = feats[:, 1]
                nag = feats[:, 2]
                l = loss_func(anchor, pos, nag)
            elif type(loss_func) == nn.L1Loss:
                l = loss_func(x_sr, y)
            else:
                print("Loss not found")
                quit()
            loss += l

        if self.l2_reg > 0:
            l2_loss = sum(p.pow(2.0).sum() for p in self.model.parameters())
            loss += self.l2_reg * l2_loss

        psnr = self.psnr(y, x_sr)


        batch_dictionary = {
            # REQUIRED: It is required for us to return "loss"
            "loss": loss,
            "psnr": psnr.detach()
        }

        return batch_dictionary

    def training_epoch_end(self, outputs):
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_psnr = torch.stack([x['psnr'] for x in outputs]).mean()
        self.logger.experiment.add_scalar(f'train_triplet_loss YTF vs Epoch',
                                          avg_loss, self.trainer.current_epoch + 1)
        self.logger.experiment.add_scalar(f'train_PSNR YTF vs Epoch',
                                          avg_psnr, self.trainer.current_epoch + 1)
        self.log(f"train_{self.trainer.datamodule.train_data}_loss", avg_loss)

    def test_step(self, batch, batch_idx):
        x, y, id_, filename = batch
        input = []
        for ii in range(self.cfg.transformer.sr.num_input_frames):
            input.append(x[:, ii:(self.cfg.transformer.fr.num_input_frames + ii)])
        input = torch.stack(input, dim=2)
        x_sr, feats = self(input)

        batch_dictionary = {
            "feats": feats,
            "id": id_,
            "filename": np.array(filename)
        }
        return batch_dictionary

    def test_epoch_end(self, outputs):
        self.fr_model.test_epoch_end(outputs)

    def super_resolve(self, vids_batch):
        out = self.sr_model(vids_batch)
        out = out[0]["recons"]
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.sr_model.parameters(), lr=self.sr_model.lr, weight_decay=self.sr_model.weight_decay)
        return optimizer


class Aggregator(pl.LightningModule):
    def __init__(self, cfg, cuda=False):
        super().__init__()
        """
        Inputs:
            cfg with all the necessary hyper-params to use
        """
        # log hyper-parameters
        self.transformer = nn.TransformerEncoderLayer(d_model=cfg.transformer.d_model, nhead=cfg.transformer.nhead)
        self.softmax = nn.Softmax(dim=-2)
        self.cfg = cfg
        self.lr = cfg.transformer.lr
        self.temperature = cfg.transformer.temperature
        self.l2_reg = cfg.transformer.loss.l2_reg
        self.is_cuda = cuda
        self.losses, self.losses_names = self.choose_loss(cfg)
        self.save_hyperparameters(logger=True)

        # self.gallery_dataloaders =
        self.test_out = {}

    def forward(self, x):
        # x: [Batch, Frame, embedding]
        out = self.transformer(x)
        x = x * self.softmax(out)
        x = torch.sum(x, -2)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        aggregated_feats = self(x)

        logits = (aggregated_feats @ y.T) / self.temperature
        images_similarity = y @ y.T
        feats_similarity = aggregated_feats @ aggregated_feats.T
        targets = F.softmax(
            (images_similarity + feats_similarity) / 2 * self.temperature, dim=-1
        )
        aggregated_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + aggregated_loss) / 2.0  # shape: (batch_size)
        loss = loss.mean()

        if self.l2_reg > 0:
            l2_loss = sum(p.pow(2.0).sum() for p in self.model.parameters())
            loss += self.l2_reg * l2_loss

        batch_dictionary = {
            # REQUIRED: It is required for us to return "loss"
            "loss": loss,
        }
        return batch_dictionary

    def training_epoch_end(self, outputs):
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar(f'train/loss',
                                          avg_loss, self.trainer.current_epoch + 1)
        self.log(f"train/loss", avg_loss)

    def test_step(self, batch, batch_idx):
        x, id_ = batch
        feats = self(x)

        batch_dictionary = {
            "feats": feats,
            "id": id_,
        }
        return batch_dictionary

    def test_epoch_end(self, outputs):
        feats = np.concatenate([x['feats'].cpu().numpy() for x in outputs])
        id_ = np.concatenate([x['id'].cpu().numpy() for x in outputs])
        self.test_out = {'feats': feats, 'id': id_}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.optim.lr, weight_decay=self.cfg.optim.weight_decay)
        return optimizer


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()