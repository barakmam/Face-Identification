from pytorch_lightning.callbacks import Callback, EarlyStopping
import torch
import numpy as np
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger
from sklearn.preprocessing import normalize
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mycode.utils.metrics import get_recall_at_k, k_vals


class MyEarlyStopping(EarlyStopping):
    """
    Start check for early-stopping after min_epochs (variable in trainer)
    """
    def __init__(self, monitor, patience, mode, min_delta=0.02):
        super(MyEarlyStopping, self).__init__(monitor=monitor, patience=patience, mode=mode, min_delta=min_delta, check_on_train_epoch_end=True)

    # def on_validation_end(self, trainer, pl_module):
    #     super(MyEarlyStopping, self).on_validation_end(trainer, pl_module)
    #     if trainer.current_epoch < trainer.min_epochs:
    #         self.wait_count = 0
    #         return

    def on_train_epoch_end(self, trainer, pl_module):
        super(MyEarlyStopping, self).on_train_epoch_end(trainer, pl_module)
        if trainer.current_epoch < trainer.min_epochs:
            self.wait_count = 0
            return


def run_identification(model, loader_q, loader_g, device):
    model.eval()
    with torch.no_grad():
        out_q = []
        id_q = []
        for feat_q, id_ in loader_q:
            out_q.append(model.features(feat_q.to(device)).cpu().numpy())
            id_q.append(id_)
        out_q = np.vstack(out_q)
        id_q = np.concatenate(id_q)

        out_g = []
        id_g = []
        for feat_g, id_ in loader_g:
            out_g.append(feat_g.squeeze())
            id_g.append(id_)
        out_g = np.vstack(out_g)
        id_g = np.concatenate(id_g)

        out_g = normalize(out_g)
        recall_at_k = get_recall_at_k(out_q, out_g, id_q, id_g, k_vals)
    return recall_at_k


def run_identification_rank1(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        out_q = []
        out_g = []
        for batch in test_loader:
            out_q.append(model.features(batch['feats_q'].to(device)).cpu().numpy())
            out_g.append(batch['feats_g'].numpy())
        out_q = np.vstack(out_q)
        out_g = np.vstack(out_g).squeeze()
        logits = out_q @ out_g.T
        rank1 = (logits.diagonal() == logits.max(1)).mean()
        return rank1


def run_identification_rank1_avg(test_loader):
    out_q = []
    out_g = []
    for batch in test_loader:
        out_q.append(batch['feats_q'].numpy())
        out_g.append(batch['feats_g'].numpy())
    out_q = np.vstack(out_q)
    out_g = np.vstack(out_g).squeeze()
    sim = out_q.mean(1) @ out_g.T
    rank1 = (sim.diagonal() == sim.max(1)).mean()
    return rank1, sim.diagonal().mean()


class TestYTF(Callback):
    def __init__(self, cfg):
        super().__init__()
        from mycode.feats_agg.data import get_dataloader
        self.test_loader = get_dataloader(cfg, 'test', batch_size=32)

    def on_train_epoch_end(self, trainer, pl_module):
        rank1 = run_identification_rank1(pl_module, self.test_loader, pl_module.device)
        # print(f'Epoch {trainer.current_epoch}: YTF CMC = {ytf_cmc}')
        if isinstance(trainer.logger, (MLFlowLogger, WandbLogger)):
            trainer.logger.log_metrics({'ytf_rank1/test': rank1}, trainer.current_epoch)
        else:
            trainer.logger.experiment.add_scalar('ytf_rank1/test', rank1, global_step=trainer.current_epoch + 1)

        # rank1, sim = run_identification_rank1_avg(self.test_loader)
        # trainer.logger.experiment.add_scalar('ytf_baseline_rank1', rank1, global_step=trainer.current_epoch + 1)
        # trainer.logger.experiment.add_scalar('ytf_baseline_sim', sim, global_step=trainer.current_epoch + 1)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        rank1 = run_identification_rank1(pl_module, self.test_loader, pl_module.device)
        if isinstance(trainer.logger, (MLFlowLogger, WandbLogger)):
            trainer.logger.log_metrics({'ytf_rank1/test': rank1}, step=-1)
        else:
            trainer.logger.experiment.add_scalar('ytf_rank1/test', rank1, global_step=0)

        rank1, sim = run_identification_rank1_avg(self.test_loader)
        if isinstance(trainer.logger, (MLFlowLogger, WandbLogger)):
            trainer.logger.log_metrics({'ytf_baseline_rank1': rank1, 'ytf_baseline_sim': sim}, step=0)
        else:
            trainer.logger.experiment.add_scalar('ytf_baseline_rank1', rank1, global_step=0)
            trainer.logger.experiment.add_scalar('ytf_baseline_sim', sim, global_step=0)
        print(f'YTF Test Rank 1 - Baseline:', rank1)


class TestMS1M_AVG(Callback):
    def __init__(self, cfg):
        super().__init__()
        from mycode.feats_agg.data import get_dataloader
        cfg.name = 'ms1m'
        self.test_loader = get_dataloader(cfg, 'test', batch_size=32)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        rank1, _ = run_identification_rank1_avg(self.test_loader)
        print(f'MS1M Test Rank 1 - Baseline:', rank1)
        pass
