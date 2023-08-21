from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from mycode.data.data_module import YoutubeFacesAUX
from mycode.data.transform import get_transforms_dict
from mycode.utils.misc import save_vid_from_images


class ImageLogger(Callback):
    """
    show SR images from videos
    """

    def __init__(self, cfg, num_samples=9):
        super().__init__()

        self.cfg = cfg
        self.num_samples = num_samples
        self.test = YoutubeFacesAUX(cfg, 'test', mode='lr_input', transform=get_transforms_dict(cfg)['test'])
        self.test_vids, self.target_frames, _, _ = next(iter(torch.utils.data.DataLoader(self.test.data, batch_size=num_samples)))
        self.test_vids = self.test_vids[:, 0]
        self.target_frames = self.target_frames[:, 0]
        self.progress_images = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Bring the tensors to same device as the pl_module
        test_vids = self.test_vids.to(device=pl_module.device)
        self.target_frames = self.target_frames.to(device=pl_module.device)
        # Get transformer prediction
        test_sr = pl_module.super_resolve(test_vids)
        grid_input = torch.concat([torch.stack([sr, gt]) for sr, gt in zip(test_sr, self.target_frames)])
        grid = make_grid(grid_input, nrow=6).cpu().numpy().transpose(1, 2, 0)

        self.progress_images.append(grid)
        fig = plt.figure()
        plt.imshow(grid)
        plt.xticks([])
        plt.yticks([])
        trainer.logger.experiment.add_figure('Super Resolved [SR, GT]', fig, global_step=trainer.current_epoch)

    def image_grid(self, images, labels, preds):
        figure = plt.figure(figsize=(12, 8))

        num_imgs_to_plot = 16
        for ii, x, y, pred in zip(range(num_imgs_to_plot), images[:num_imgs_to_plot],
                                  labels[:num_imgs_to_plot], preds[:num_imgs_to_plot]):
            plt.subplot(int(np.sqrt(num_imgs_to_plot)), int(np.sqrt(num_imgs_to_plot)), ii + 1)
            plt.title('label: ' + y + '. pred: ' + pred)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            image = (x.cpu().numpy().transpose(1, 2, 0) + 1) / 2  # convert to range [0,1] from [-1, 1]
            plt.imshow(image)

        return figure

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.cfg.training.save_sr_progress:
            save_vid_from_images(self.progress_images, 'SR_progress.avi')


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
