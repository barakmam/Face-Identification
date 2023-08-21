import os
import matplotlib.pyplot as plt
import torch
from glob import glob
import numpy as np
from os.path import join
import hydra
from omegaconf import DictConfig
from mycode.utils.misc import print_dict
import time
from mycode.data.data_module import ImageDM
from mycode.data.transform import get_transforms_dict
from mycode.models.models import FaceRecModule, SRFR
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import pytorch_lightning as pl
from mycode.inference import identification_test, verification
from pytorch_lightning.callbacks import ModelCheckpoint
import random
from mycode.utils.misc import compare_models, read_cfg
from mycode.utils.myCallbacks import ImageLogger
import cProfile


def run(cfg):
    dm = ImageDM(cfg, transform_dict=get_transforms_dict(cfg))
    dm.setup()

    # Init our transformer
    if cfg.transformer.ckpt:
        model = SRFR.load_from_checkpoint(checkpoint_path=cfg.transformer.ckpt)
        logger = TensorBoardLogger('./', version=0, name=None, default_hp_metric=False)
    else:
        model = SRFR(cfg, cuda=torch.cuda.is_available())
        logger = TensorBoardLogger('./', name=None, default_hp_metric=False)
    # logger = WandbLogger()
    # logger.watch(transformer, log_graph=False)
    # wandb.init(project='bionicEye', config=cfg)

    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=1,  # set the logging frequency
        # gpus=cfg.system.gpus,  # assign -1 to run over all gpus
        accelerator="gpu",
        devices=cfg.system.num_gpus,
        max_epochs=cfg.training.max_epochs,
        min_epochs=cfg.training.min_epochs,
        # auto_lr_find=True,
        check_val_every_n_epoch=2,
        callbacks=[
            ModelCheckpoint(dirpath=join(logger.log_dir, 'checkpoints')),
            ImageLogger(cfg)
        #     MyEarlyStopping(monitor=f'train_{cfg.data.train}_loss', patience=3, mode='min', min_delta=0.006)
        ]
    )


    if cfg.stage.train:
        start = time.time()
        trainer.fit(model, datamodule=dm)
        print(f'Training took: {(time.time() - start)/60} minutes')
        trainer.save_checkpoint(join(logger.log_dir, f'checkpoints/{type(model).__name__}.ckpt'))


    if cfg.stage.test:
        trainer.test(model, datamodule=dm)
        identification_test(cfg, logger=logger)
        verification(cfg, logger=logger)


@hydra.main(config_path="./configs", config_name="train")
def main(cfg: DictConfig) -> None:
    print_dict(cfg)
    # Reproducibility
    torch.manual_seed(cfg.system.seed)
    np.random.seed(cfg.system.seed)
    random.seed(cfg.system.seed)
    # cProfile.run('foo()')
    run(cfg)


if __name__ == "__main__":
    main()



