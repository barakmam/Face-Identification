from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm import tqdm
import sys
from torch import Tensor
import torch


class LitProgressBar(TQDMProgressBar):

    # def init_train_tqdm(self) -> tqdm:
    #     """ Override this to customize the tqdm bar for training. """
    #     bar = tqdm(
    #         desc='Training',
    #         initial=self.train_batch_idx,
    #         position=(2 * self.process_position),
    #         disable=self.is_disabled,
    #         leave=True,
    #         dynamic_ncols=False,  # This two lines are only for pycharm
    #         ncols=100,
    #         file=sys.stdout,
    #         smoothing=0,
    #     )
    #     return bar

    def init_validation_tqdm(self) -> tqdm:
        bar = tqdm(
            disable=True,
        )
        return bar
        # """ Override this to customize the tqdm bar for validation. """
        # # The main progress bar doesn't exist in `trainer.validate()`
        # has_main_bar = self.main_progress_bar is not None
        # bar = tqdm(
        #     desc='Validating',
        #     position=(2 * self.process_position + has_main_bar),
        #     disable=self.is_disabled,
        #     leave=False,
        #     dynamic_ncols=False,
        #     ncols=100,
        #     file=sys.stdout
        # )
        # return bar

    def init_test_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for testing. """
        bar = tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=False,
            ncols=100,
            file=sys.stdout
        )
        return bar


class CosineDistance:
    @staticmethod
    def __call__(x: Tensor, y: Tensor) -> Tensor:
        return 1 - x @ y.T


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

