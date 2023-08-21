import torch
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger
from torch import nn
import pytorch_lightning as pl

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import l2_norm


class LightModel(nn.Module):
    def __init__(self, input_dim, d_model=1):
        super().__init__()
        self.fc = nn.Linear(input_dim, d_model)
        # self.bn = nn.BatchNorm1d(d_model)
        # self.classifier = nn.Linear(d_model, 1)
        # self.ln = nn.LayerNorm(input_dim)

    def forward(self, x):
        num_frames = torch.count_nonzero(x, dim=1)[:, [0]]
        # x = self.bn(x)
        weights = self.fc(x.view(-1, x.shape[-1])).view(x.shape[0], x.shape[1], -1)
        # weights = self.ln(weights.squeeze()).unsqueeze(2)
        x = (x * weights).sum(1) / weights.sum(1)
        return x

    def get_first_embedding_len(self):
        tmp = torch.ones((1, *self.input_size))
        tmp = self.convolutions_out_vector(tmp)
        return tmp.shape[-1]


class Attention(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        # enlarge the values of the frames vectors since they are too small
        # yielding numeric issues in the attention
        self.norm_mul = 100
        self.attention = nn.MultiheadAttention(input_size[-1], 1, batch_first=True)
        self.avg_params = nn.Parameter(torch.ones(input_size) / input_size[1])

        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def convolutions_out_vector(self, x):
        b = x.shape[0]

        # x = self.block1(x)
        # x = self.block2(x)
        # x = self.block3(x)
        # x = x.view(b, -1)

        x1 = self.row_conv(x)
        x2 = self.block_conv(x)
        x3 = self.col_conv(x)
        x = torch.concat([x1.view(b, -1), x2.view(b, -1), x3.view(b, -1)], dim=1)
        return x

    def features(self, x):
        # x = self.convolutions_out_vector(x)
        x = x.squeeze()
        x = x * self.norm_mul
        emphasize, _ = self.attention(x, x, x)
        emphasize = emphasize.softmax(dim=-2)
        x = torch.sum(x * emphasize, -2).squeeze()
        x = l2_norm(x)
        return x

    def get_first_embedding_len(self):
        tmp = torch.ones((1, *self.input_size))
        tmp = self.convolutions_out_vector(tmp)
        return tmp.shape[-1]


class Transformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.cls = nn.Parameter(torch.randn((1, input_dim)))
        self.cls_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim)
        )
        self.transformer = nn.TransformerEncoderLayer(input_dim, 8, dim_feedforward=256, batch_first=True,
                                                      dropout=0.4)

    def forward(self, x):
        x = self.features(x)
        return x

    def features(self, x):
        cls_token = l2_norm(self.cls).repeat(x.shape[0], 1, 1).to(x.device)
        x = torch.concat([cls_token, x], dim=1)
        x = self.transformer(x)
        x = x[:, 0]
        x = self.cls_mlp(x)
        x = l2_norm(x)
        return x

    def get_first_embedding_len(self):
        tmp = torch.ones((1, *self.input_size))
        tmp = self.convolutions_out_vector(tmp)
        return tmp.shape[-1]


class Aggregator(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.model.input_dim
        from mycode.models.backbones.vit import FeatureTransformer
        self.transformer = FeatureTransformer(cfg.data.num_frames, depth=cfg.model.depth, agg_method=cfg.model.agg_method,
                                              use_mlp=cfg.model.use_mlp)
        # self.transformer.requires_grad_(False)
        # self.transformer.feature._modules['2'].requires_grad_(True)

        self.temperature = cfg.loss.temperature
        self.lr = cfg.optim.lr
        self.gamma = cfg.optim.gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', weight=cfg.loss.weight)
        self.bce_loss = nn.BCELoss(reduction='none', weight=cfg.loss.weight)
        self.l2_loss = nn.MSELoss(reduction='none')
        self.weight_decay = cfg.optim.weight_decay
        self.save_hyperparameters()


        self.training_step_outputs = []
        self.infer_step_outputs = []

    def get_similarity(self, query, target, ids):
        id_g_unique, inverse_indices = torch.unique(ids, return_inverse=True)
        query = l2_norm(query)
        if len(id_g_unique) != len(ids):
            unique_inds = torch.zeros_like(id_g_unique)
            unique_inds[inverse_indices] = torch.arange(len(inverse_indices), device=inverse_indices.device)
            target = target[unique_inds]
            sim = query @ target.T
            same_id = torch.where(ids.view(-1, 1) == id_g_unique.view(1, -1))
            return sim, same_id
        else:
            # when we have unique gallery ids in the first place (like ms1m)
            # pred[i] is the same id as target[i], thus the label is the identity matrix
            sim = query @ target.T
            same_id = (torch.arange(len(sim), device=sim.device), )*2
            return sim, same_id

    def training_step(self, batch):
        pred = self(batch['feats_q'])
        sim, same_id = self.get_similarity(pred, batch['feats_g'], batch['ids'])
        avg_pos_sim = sim[same_id].mean()

        logits = sim / self.temperature
        loss = 0
        if 'ce' in self.cfg.loss.types:
            ce_loss = self.ce_loss(logits, same_id[1]).mean()
            self.trainer.logger.log_metrics({f'ce_loss/train': ce_loss.item()})
            loss += ce_loss
        if 'oracle' in self.cfg.loss.types:
            oracle_loss = -((l2_norm(pred) * batch['feats_q'][range(len(batch['feats_q'])), batch['raw_sim'].argmax(1)]).sum(1).sigmoid().log())
            oracle_loss = oracle_loss.mean()
            self.trainer.logger.log_metrics({f'oracle_loss/train': oracle_loss.item()})
            loss += oracle_loss
        if 'l2' in self.cfg.loss.types and self.cfg.model.agg_method == 'cls':
            cls_attn = self.transformer.blocks[-1].attn.attn_map[:, :, 0, 1:]
            l2_loss = 8 * self.l2_loss(cls_attn, batch['raw_sim'].unsqueeze(1).repeat(1, cls_attn.shape[1], 1)).mean()
            self.trainer.logger.log_metrics({f'l2_loss/train': l2_loss.item()})
            loss += l2_loss

        acc = (sim[same_id] == sim.max(dim=1).values).float().mean()
        batch_dictionary = {
            "loss": loss,
            "acc": acc,
            "sim": avg_pos_sim
        }
        self.training_step_outputs.append(batch_dictionary)
        # REQUIRED: It is required for us to return "loss"
        return loss

    def on_train_epoch_end(self) -> None:
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in self.training_step_outputs]).mean()

        if isinstance(self.trainer.logger, (MLFlowLogger, WandbLogger)):
            self.trainer.logger.log_metrics({f'{self.trainer.train_dataloader.dataset.name}_loss/train': avg_loss.item(),
                                             f'{self.trainer.train_dataloader.dataset.name}_acc/train': avg_acc.item()},
                                            self.trainer.current_epoch)
        else:
            self.logger.experiment.add_scalar(f'{self.trainer.train_dataloader.dataset.name}_loss/train',
                                              avg_loss.item(), self.trainer.current_epoch)
            self.logger.experiment.add_scalar(f'{self.trainer.train_dataloader.dataset.name}_acc/train',
                                              avg_acc.item(), self.trainer.current_epoch)

        if 'sim' in self.training_step_outputs[0].keys():
            avg_sim = torch.stack([x['sim'] for x in self.training_step_outputs]).mean()

            if isinstance(self.trainer.logger, (MLFlowLogger, WandbLogger)):
                self.trainer.logger.log_metrics({f'{self.trainer.train_dataloader.dataset.name}_similarity/train': avg_sim.item()},
                                                self.trainer.current_epoch)
            else:
                self.logger.experiment.add_scalar(f'{self.trainer.train_dataloader.dataset.name}_similarity/train',
                                              avg_sim.item(), self.trainer.current_epoch)

        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        return self.infer_step(batch, batch_idx, data_state='val')

    def on_validation_epoch_end(self):
        self.infer_epoch_end(data_state='val')

    def test_step(self, batch, batch_idx):
        return self.infer_step(batch, batch_idx, data_state='test')

    def on_test_epoch_end(self) -> None:
        self.infer_epoch_end(data_state='test')

    def infer_step(self, batch, batch_idx, data_state):
        query, target = batch['feats_q'], batch['feats_g']
        pred = self(query)
        sim, same_id = self.get_similarity(pred, target, batch['ids'])
        logits = sim / self.temperature

        loss = 0
        if 'ce' in self.cfg.loss.types:
            ce_loss = self.ce_loss(logits, same_id[1]).mean()
            self.trainer.logger.log_metrics({f'ce_loss/{data_state}': ce_loss.item()})
            loss += ce_loss
        if 'oracle' in self.cfg.loss.types:
            oracle_loss = -(
                (l2_norm(pred) * batch['feats_q'][range(len(batch['feats_q'])), batch['raw_sim'].argmax(1)]).sum(
                    1).sigmoid().log())
            oracle_loss = oracle_loss.mean()
            self.trainer.logger.log_metrics({f'oracle_loss/{data_state}': oracle_loss.item()})
            loss += oracle_loss
        if 'l2' in self.cfg.loss.types and self.cfg.model.agg_method == 'cls':
            cls_attn = self.transformer.blocks[-1].attn.attn_map[:, :, 0, 1:]
            l2_loss = 8 * self.l2_loss(cls_attn, batch['raw_sim'].unsqueeze(1).repeat(1, cls_attn.shape[1], 1)).mean()
            self.trainer.logger.log_metrics({f'l2_loss/{data_state}': l2_loss.item()})
            loss += l2_loss

        acc = (sim[same_id] == sim.max(dim=1).values).float().sum()
        batch_dictionary = {
            "loss": loss,
            "acc": acc,
            "pred": pred,
            "target": target,
            "ids": batch['ids'],
        }

        self.infer_step_outputs.append(batch_dictionary)
        return loss

    def infer_epoch_end(self, data_state):
        # calculating average loss
        num_samples, sum_loss, pred, target, ids = 0, [], [], [], []
        for x in self.infer_step_outputs:
            num_samples += len(x['pred'])
            # sum_loss.append(x['loss'])
            pred.append(x['pred'])
            target.append(x['target'])
            ids.append(x['ids'])

        # avg_loss = (torch.stack(sum_loss).sum() / num_samples).item()
        pred = torch.concat(pred)
        target = torch.concat(target)
        ids = torch.concat(ids)

        sim, same_id = self.get_similarity(pred, target, ids)
        avg_pos_sim = sim[same_id].mean().item()
        logits = sim / self.temperature
        loss = self.ce_loss(logits, same_id[1]).sum()
        avg_loss = (loss.sum() / num_samples).item()

        rank1 = torch.mean((sim[same_id] == sim.max(dim=1).values).float()).item()

        name = self.trainer.test_dataloaders.dataset.name if data_state == 'test' \
            else self.trainer.val_dataloaders.dataset.name

        if isinstance(self.trainer.logger, (MLFlowLogger, WandbLogger)):
            self.trainer.logger.log_metrics({f'{name}_loss/{data_state}': avg_loss,
                                             f'{name}_rank1/{data_state}': rank1,
                                             f'{name}_similarity/{data_state}': avg_pos_sim}, self.trainer.current_epoch)
        else:
            self.logger.experiment.add_scalar(f'{name}_loss/{data_state}',
                                              avg_loss, self.trainer.current_epoch)
            self.logger.experiment.add_scalar(f'{name}_Rank1/{data_state}',
                                              rank1, self.trainer.current_epoch)
            self.logger.experiment.add_scalar(f'{name}_similarity/{data_state}',
                                              avg_pos_sim, self.trainer.current_epoch)
        self.log(f'{data_state}_loss', avg_loss)
        self.log(f'{data_state}_rank1', rank1)

        self.infer_step_outputs.clear()

    def forward(self, x):
        out = self.transformer(x)

        # cls_attn = self.transformer.blocks[-1].attn.attn_map[:, :, 0, 1:].mean(1)
        # out = (x * cls_attn.unsqueeze(2)).mean(1)
        return out

    def features(self, x):
        x = self(x)
        return x

    def configure_optimizers(self):
        if self.cfg.optim.name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.cfg.optim.name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError("Unknown optimizer")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.optim.milestones, gamma=self.cfg.optim.gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def identification_rank1(query, target):
        # query[i] is the same id as target[i]
        logits = query @ target.T
        rank1_accuracy = torch.mean((logits.diag() == logits.max(dim=1).values).float())
        return rank1_accuracy

