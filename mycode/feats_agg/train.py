
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger
# from clearml import Task

import os
import sys

import torch
import numpy as np
import pytorch_lightning as pl
from glob import glob
import hydra

from myCallbacks import MyEarlyStopping, TestYTF, TestMS1M_AVG
from data import get_dataloader
from models import Aggregator
from utils import LitProgressBar


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print('#'*10 + ' sys path append: ', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

seed = 7
os.environ['HYDRA_FULL_ERROR'] = '1'


@hydra.main(config_path="./configs", config_name="train", version_base='1.1')
def train(cfg: DictConfig):
    torch.manual_seed(cfg.system.seed)
    np.random.seed(cfg.system.seed)
    torch.set_float32_matmul_precision('high')

    # logger = TensorBoardLogger('./', name=None, default_hp_metric=False)
    # ckpt_dir = os.path.join(logger.log_dir, 'checkpoints')

    # logger = MLFlowLogger(experiment_name=cfg.exp_name, run_name=cfg.run_name, tracking_uri=cfg.mlflow_path)
    # ckpt_dir = os.path.join(cfg.mlflow_path, logger.experiment_id, logger.run_id, 'artifacts/checkpoints')
    # logger.log_hyperparams(cfg)

    logger = WandbLogger(name=cfg.run_name, project=cfg.exp_name, config=cfg)
    ckpt_dir = os.path.join(logger.save_dir, 'checkpoints')

    # Task.init(project_name=cfg.exp_name, task_name=cfg.run_name)
    # ckpt_dir = os.path.join('./checkpoints')
    # logger = None

    train_loader = get_dataloader(cfg.data, 'train', cfg.model.batch_size.train)
    val_loader = get_dataloader(cfg.data, 'val', cfg.model.batch_size.train)
    test_loader = get_dataloader(cfg.data, 'test', cfg.model.batch_size.test)

    model = Aggregator(cfg)
    # count_model_params(model)

    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=1,  # set the logging frequency
        accelerator="gpu",
        devices=cfg.system.num_gpus,
        max_epochs=cfg.training.max_epochs,
        min_epochs=cfg.training.min_epochs,
        # auto_lr_find=True,
        check_val_every_n_epoch=1,
        callbacks=[
            ModelCheckpoint(dirpath=ckpt_dir, monitor=f'val_loss', mode='min',
                            filename='{epoch}-{val_rank1:.3f}', save_top_k=1),
            MyEarlyStopping(monitor=f'val_loss', patience=cfg.training.patience, mode='min', min_delta=0),
            TestYTF(cfg.ytf.data),
            # TestMS1M_AVG(cfg.data),
            LitProgressBar()
        ],
        num_sanity_val_steps=0
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader,
                 ckpt_path=glob(os.path.join(ckpt_dir, '*'))[0])

    print('Checkpoint saved in: ', glob(os.path.join(os.getcwd(), ckpt_dir, '*'))[0])


    if isinstance(logger, MLFlowLogger):
        logger.experiment.log_artifact(logger.run_id, os.getcwd())
    if isinstance(logger, WandbLogger):
        logger.experiment.log_artifact(os.getcwd())


if __name__ == '__main__':
    np.set_printoptions(precision=3)  # print numpy arrays up to 3 decimal digits
    train()


