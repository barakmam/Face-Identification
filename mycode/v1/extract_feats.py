import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import random

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mycode.v1.data.data_module import ImageDM
from mycode.v1.data.transform import get_transforms_dict
from mycode.v1.utils.misc import print_dict
from mycode.v1.models.models import FaceRecModule


def run(cfg):
    dm = ImageDM(cfg, transform_dict=get_transforms_dict(cfg))
    dm.setup('test')

    # Init our transformer
    model = FaceRecModule(cfg=cfg.transformer.fr, cuda=torch.cuda.is_available())
    logger = TensorBoardLogger('./', name=None, default_hp_metric=False)

    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=1,  # set the logging frequency
        # gpus=cfg.system.gpus,  # assign -1 to run over all gpus
        # auto_lr_find=True,
        accelerator="gpu",
        devices=cfg.system.num_gpus,
        callbacks=[
            ModelCheckpoint(dirpath=os.path.join(logger.log_dir, 'checkpoints'))
        ],
        strategy='dp'
    )

    trainer.test(model, datamodule=dm)

    # identification_test(cfg=cfg, logger=logger)
    # verification(cfg=cfg, logger=logger)


@hydra.main(config_path="./configs", config_name="extract_feats")
def main(cfg: DictConfig) -> None:
    print_dict(cfg)
    # Reproducibility
    torch.manual_seed(cfg.system.seed)
    np.random.seed(cfg.system.seed)
    random.seed(cfg.system.seed)

    run(cfg)


if __name__ == "__main__":
    main()
