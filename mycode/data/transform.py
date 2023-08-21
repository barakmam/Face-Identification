import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms.functional as F
from skimage import transform as sk_trans
import cv2


class SimilarityTrans:
    """similarity transform by 5 landmarks"""

    def __init__(self, out_size=(112, 112), src=None):
        if src is None:
            # from insightFace, for images of size (112, 112):
            src = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]], dtype=np.float32)
            src[:, 0] += 8.0
        self.src = src
        self.out_size = out_size

        self.sim_trans = sk_trans.SimilarityTransform()

    def __call__(self, img, lmk):
        self.sim_trans.estimate(lmk, self.src)
        M = self.sim_trans.params[0:2, :]
        img = cv2.warpAffine(img, M, self.out_size, borderValue=0.0)
        return img

    def estimate_affine(self, lmk):
        self.sim_trans.estimate(lmk, self.src)
        M = self.sim_trans.params[0:2, :]
        return M


class RandomJpegCompression(object):
    def __init__(self, qf=None):
        self.qf = qf

    def __call__(self, image):
        curr_qf = self.qf if self.qf is not None else np.random.randint(20, 100)
        outputIoStream = BytesIO()
        image.save(outputIoStream, "JPEG", quality=curr_qf, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)

    def __repr__(self):
        return self.__class__.__name__ + '(qf=)'.format(self.qf)


class AddClippedGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        tensor += torch.randn(tensor.size()) * self.std + self.mean
        tensor[tensor > 1] = 1
        tensor[tensor < 0] = 0
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')


class RatioCenterCrop:
    def __init__(self, ratio):
        """
        use the ratio to get the new size of the image:
        for example: new image width = image width / ratio
        :param ratio:
        """
        self.ratio = ratio

    def __call__(self, im):
        width, height = im.size  # Get dimensions
        new_width = width // self.ratio
        new_height = height // self.ratio

        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2

        # Crop the center of the image
        return im.crop((left, top, right, bottom))


def get_transforms(info):
    trans = []
    if info.types is None:
        info.types = []

    if 'to_pil' in info.types:
        trans.append(transforms.ToPILImage())
    if 'ratio_center_crop' in info.types:
        trans.append(RatioCenterCrop(info.params.ratio))
    if 'blur' in info.types:
        trans.append(transforms.GaussianBlur(3, sigma=[0.5, 1.5]))
    if 'JPEG' in info.types:
        trans.append(RandomJpegCompression(info.params.qf))
    if 'square_pad' in info.types:
        trans.append(SquarePad())
    if 'resize' in info.types:
        trans.append(transforms.Resize((info.params.input_size, info.params.input_size),
                                       transforms.InterpolationMode.BICUBIC))


    trans.append(transforms.ToTensor())

    if 'normalize' in info.types:
        trans.append(transforms.Normalize(info.params.norm.mean, info.params.norm.std))

    if 'horizontal_flip' in info.types:
        trans.append(transforms.RandomHorizontalFlip(p=info.params.horizontal_flip_probability))
    if 'noise' in info.types:
        trans.append(AddClippedGaussianNoise(0., info.params.std))

    trans = transforms.Compose(trans)
    return trans


def get_transforms_dict(cfg):
    test_trans = get_transforms(cfg.data.augmentations.test)
    train_trans = get_transforms(cfg.data.augmentations.train)
    transforms_dict = {'train': train_trans, 'val': test_trans, 'test': test_trans}
    return transforms_dict


# if __name__ == '__main__':










