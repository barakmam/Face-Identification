import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import Optional, Callable


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: Optional[None] = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        
        with torch.cuda.amp.autocast(True):
            batch_size, num_token, embed_dim = x.shape
            #qkv is [3,batch_size,num_heads,num_token, embed_dim//num_heads]
            qkv = self.qkv(x).reshape(
                batch_size, num_token, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        with torch.cuda.amp.autocast(False):
            q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
            # q, k = qkv[0].float(), qkv[1].float()
            # v = x
            # self.attn_map = (q @ k.transpose(-2, -1)) * self.scale
            # self.attn_map = self.attn_map.softmax(dim=-1)
            # attn_drop = self.attn_drop(self.attn_map)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn_drop = self.attn_drop(attn)
            x = (attn_drop @ v).transpose(1, 2).reshape(batch_size, num_token, embed_dim)
        with torch.cuda.amp.autocast(True):
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 num_patches: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: Optional[None] = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: Callable = nn.ReLU6):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        with torch.cuda.amp.autocast(True):
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=108, patch_size=9, in_channels=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        assert height == self.img_size[0] and width == self.img_size[1], \
            f"Input image size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class EmbedFeatures(nn.Module):
    def __init__(self, num_features, embed_dim=512):
        super().__init__()
        self.num_patches = num_features
        self.proj = nn.Identity()

    def forward(self, x):
        return x


class FeatureTransformer(nn.Module):
    """
    Feature Transformer receiving features in rows stack
    """

    def __init__(self,
                 num_features,
                 embed_dim: int = 512,
                 depth: int = 24,
                 agg_method: str = 'concat',
                 use_mlp: bool = False,
                 num_heads: int = 8,
                 attn_mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: Optional[None] = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.05,
                 mask_ratio = 0.05,
                 using_checkpoint = False,
                 ):
        super().__init__()
        # num_features for consistency with other models

        self.num_features = num_features
        self.embed_dim = embed_dim

        self.mask_ratio = mask_ratio
        self.using_checkpoint = using_checkpoint

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim, num_heads=num_heads, num_patches=num_features, mlp_ratio=attn_mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
                for i in range(depth)]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.agg_method = agg_method
        if agg_method == "concat":
            self.agg = lambda x: x.reshape(x.shape[0], -1)
            in_features = embed_dim * num_features
        elif agg_method == "mean":
            self.agg = lambda x: x.mean(1)
            in_features = embed_dim
        elif agg_method == "cls":
            self.cls = nn.Parameter(torch.randn(1, 1, embed_dim)*0.02)
            self.agg = lambda x: x[:, 0]
            in_features = embed_dim
        else:
            raise NotImplemented(f'Unknown agg_method: {agg_method}')

        if use_mlp:
            self.feature = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=embed_dim, bias=False),
                    nn.BatchNorm1d(num_features=embed_dim, eps=2e-5),
                    nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False),
                    nn.BatchNorm1d(num_features=embed_dim, eps=2e-5)
                )
        else:
            self.feature = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio=0.1):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.size()  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_features(self, x):
        x = self.pos_drop(x)

        for i, func in enumerate(self.blocks):
            if self.using_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(func, x)
            else:
                x = func(x)
        # x = self.norm(x.float())

        return x

    def forward(self, x):
        if self.agg_method == 'cls':
            x = torch.cat([self.cls.repeat(x.shape[0], 1, 1), x], dim=1)
        x = self.forward_features(x)
        x = self.agg(x)
        x = self.feature(x)
        return x
