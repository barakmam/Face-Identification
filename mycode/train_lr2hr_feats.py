from torch.utils.data import Dataset
import torch
from torchvision import transforms
import pickle
import numpy as np
import os
from mycode.utils.misc import get_train_feats, get_query_feats, get_gallery_feats
from mycode.utils.metrics import get_recall_at_k

from glob import glob
import matplotlib.pyplot as plt
from mycode.data.multi_epoch_dataloader import MultiEpochsDataLoader
from ArcFace_paulpias.model import l2_norm
from torch import nn, Tensor
from typing import Callable, Optional


class TrainData(Dataset):
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor()])
        np.random.seed(7)
        self.dataset, self.ids, self.filenames = \
            get_train_feats('/datasets/BionicEye/YouTubeFaces/faces/bicubic_sr_bicdown/embeddings.pickle',
                            '/inputs/bionicEye/data/ytf_new_splits')

        self.targets, _, _ = get_train_feats('/datasets/BionicEye/YouTubeFaces/faces/sharp/embeddings.pickle',
                                             '/inputs/bionicEye/data/ytf_new_splits')

        # with open(os.path.join('/inputs/bionicEye/data', 'ytf', 'id2label_dict.pickle'), 'rb') as handle:
        #     self.name2id_dict = pickle.load(handle)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        lr_feats = self.dataset[idx]
        hr_feats = self.targets[idx]
        return lr_feats, hr_feats


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FeatsSR(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # self.conv1 = BasicBlock(1, 8)
        self.input_size = input_size

        # self.block1 = BasicBlock(1, 16)
        # self.block2 = BasicBlock(16, 16)
        # self.block3 = BasicBlock(16, 16)


        # self.row_conv = nn.Conv2d(1, 4, (2, 5))
        # self.block_conv = nn.Conv2d(1, 4, (3, 7))
        # self.col_conv = nn.Conv2d(1, 4, (37, 3))
        # self.block_conv = nn.Conv2d(1, 4, (7, 7))


        # self.block1 = nn.Sequential(
        #     nn.Conv2d(1, 4, (3, 14)),
        #     nn.BatchNorm2d(4),
        # )
        # self.block2 = nn.Sequential(
        #     nn.Conv2d(4, 4, (3, 7)),
        #     nn.BatchNorm2d(4),
        # )
        # self.block3 = nn.Sequential(
        #     nn.Conv2d(4, 4, (3, 7)),
        #     nn.BatchNorm2d(4),
        # )

        self.attention = nn.MultiheadAttention(8, 8, batch_first=True)
        # self.embedding_len = self.get_embedding_len()

        # self.fc1 = nn.Linear(self.embedding_len, 40*512)

    def forward(self, x):
        # x = self.features(x)
        #
        # x = self.fc1(x)
        # x = x.view(x.shape[0], 40, 512)

        x = self.attention(x)
        return l2_norm(x)

    def features(self, x):
        b = x.shape[0]
        # x = self.block1(x)
        # x = self.block2(x)
        # x = self.block3(x)

        x1 = self.row_conv(x)
        x2 = self.block_conv(x)
        x3 = self.col_conv(x)
        x = torch.concat([x1.view(b, -1), x2.view(b, -1), x3.view(b, -1)], dim=1)

        # x = x.view(b, -1)
        return x

    def get_embedding_len(self):
        tmp = torch.ones((1, *self.input_size))
        tmp = self.features(tmp)
        return tmp.shape[-1]


class CosineDistance:
    @staticmethod
    def __call__(x: Tensor, y: Tensor) -> Tensor:
        return 1 - x @ y.T


device = 'cuda'
epochs = 100
batch_size = 64
train_dataset = TrainData()
train_loader = MultiEpochsDataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
query_feats, query_ids, query_filename = get_query_feats('/datasets/BionicEye/YouTubeFaces/faces/bicubic_sr_bicdown/embeddings.pickle')
gallery_feats, gallery_ids, gallery_filename = get_gallery_feats(
    '/datasets/BionicEye/YouTubeFaces/faces/sharp/embeddings.pickle', '/inputs/bionicEye/data/ytf_new_splits')
loader_q = MultiEpochsDataLoader(query_feats, batch_size=128, shuffle=False, drop_last=False)
loader_g = MultiEpochsDataLoader(gallery_feats, batch_size=128, shuffle=False, drop_last=False)
k_vals = [1, 5, 10, 20, 50, 100]  # for Recall@K

model = FeatsSR(input_size=(1, 40, 512)).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()


for epoch in range(epochs):
    # Identification:
    model.eval()
    with torch.no_grad():
        out_q = []
        for feat_q in loader_q:
            out_q.append(model(feat_q.unsqueeze(1).to(device)))
        out_q = torch.vstack(out_q)

        out_g = []
        for feat_g in loader_g:
            out_g.append(model(feat_g.unsqueeze(1).to(device)))
        out_g = torch.vstack(out_g)

        recall_at_k = get_recall_at_k(out_q.mean(1), gallery_feats.mean(1), query_ids, gallery_ids, k_vals)
    print('Recall @ K: ', recall_at_k)

    loss_arr = []
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optim.step()
        loss_arr.append(loss.item())

    print(f'Mean Loss: {np.mean(loss_arr)},\t Epoch: {epoch}')




