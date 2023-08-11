import numpy as np
import math
import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from models import misc
from models import persistence
from models.basic_module import FullyConnectedLayer, Conv2dLayer, MappingNet, MinibatchStdLayer, DisFromRGB, DisBlock, StyleConv, ToRGB, get_style_code, ToSeg
from models.utils import ToSEG
from functools import partial

@misc.profiled_function
def nf(stage, channel_base=32768, channel_decay=1.0, channel_max=512):
    NF = {512: 64, 256: 128, 128: 256, 64: 512, 32: 512, 16: 512, 8: 512, 4: 512}
    return NF[2 ** stage]


# 二分类任务做边缘修正
class confidence_map(nn.Module):
    def __init__(self):
        super(confidence_map, self).__init__()

    def forward(self, x):
        x = x[:, 1:, :, :]
        x = torch.exp(-((x-0.5)*(x-0.5))/(2*torch.std(x)*torch.std(x)))
        return 1-x

class edge_region_inpainting(nn.Module):
    def __init__(self, max_iter=5):
        super(edge_region_inpainting, self).__init__()
        self.ccm1 = confidence_map()
        self.softmax = nn.Softmax(dim=1)
        self.conv_r = nn.Conv2d(180, 180, 3, 1, 1)
        self.conv_o = nn.Conv2d(180, 2, 1)
        self.relu = partial(F.relu, inplace=True)
        self.max_iter = max_iter
        self.bn = nn.BatchNorm2d(180)
        self.threshold = 20

    def forward(self, feature, map):
        for i in range(self.max_iter):
            ori = feature
            x = feature * map
            feature = self.conv_r(x) / (F.avg_pool2d(map, 3, 1, padding=1))
            feature = self.bn(feature)
            feature = self.relu(feature)
            feature = feature + ori
            new_map = self.ccm1(self.softmax(self.conv_o(feature)))
            if torch.norm(1-new_map) < self.threshold or torch.norm(map-new_map) < 1:
                break
            else:
                map = new_map

        return feature, self.conv_o(feature)

# 多分类任务做边缘修正
class confidence_map_multiple(nn.Module):
    def __init__(self):
        super(confidence_map_multiple, self).__init__()

    def  forward(self, x):
        x0 = x[:, 0:1, :, :]
        x0 = 1 - torch.exp(-((x0 - 0.5)*(x0 - 0.5)) / (2 * torch.std(x0) * torch.std(x0)))
        x1 = x[:, 1:2, :, :]
        x1 = 1 - torch.exp(-((x1 - 0.5) * (x1 - 0.5)) / (2 * torch.std(x1) * torch.std(x1)))
        x2 = x[:, 2:3, :, :]
        x2 = 1 - torch.exp(-((x2 - 0.5) * (x2 - 0.5)) / (2 * torch.std(x2) * torch.std(x2)))
        x3 = x[:, 3:, :, :]
        x3 = 1 - torch.exp(-((x3 - 0.5) * (x3 - 0.5)) / (2 * torch.std(x3) * torch.std(x3)))

        res = torch.cat([x0, x1, x2, x3], dim=1)

        return res

class edge_region_inpainting_multiple(nn.Module):
    def __init__(self, max_iter=5):
        super(edge_region_inpainting_multiple, self).__init__()
        self.ccm1 = confidence_map_multiple()
        self.softmax = nn.Softmax(dim=1)
        self.conv_r = nn.Conv2d(180 * 4, 180, 3, 1, 1)
        self.conv_o = nn.Conv2d(180, 4, 1)
        self.relu = partial(F.relu, inplace=True)
        self.max_iter = max_iter
        self.bn = nn.BatchNorm2d(180)
        self.threshold = 20

    def forward(self, feature, map):
        for i in range(self.max_iter):
            ori = feature
            x = torch.cat([feature * map[:, 0:1, ...], feature * map[:, 1:2, ...], feature * map[:, 2:3, ...], feature * map[:, 3:4, ...]], dim=1)
            feature = self.conv_r(x)
            # feature = self.conv_r(x) / (F.avg_pool2d(map, 3, 1, padding=1) * 9)
            # feature = self.bn(feature)
            feature = self.relu(feature)
            feature = feature + ori
            new_map = self.ccm1(self.softmax(self.conv_o(feature)))
            if torch.norm(1 - new_map) < self.threshold or torch.norm(map - new_map) < 1:
                break
            else:
                map = new_map

        # return self.softmax(self.conv_o(feature))
        return feature, self.conv_o(feature)



@persistence.persistent_class
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = FullyConnectedLayer(in_features=in_features, out_features=hidden_features, activation='lrelu')
        self.fc2 = FullyConnectedLayer(in_features=hidden_features, out_features=out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


@misc.profiled_function
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


@misc.profiled_function
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def attention_reverse(affinity, window_size, H, W):
    """
    Args:
        affinity: (B, -1, H * W, self.window_size * self.window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, num_heads, H*W, H*W)
    """
    device = affinity.device
    B = affinity.shape[0]
    x = affinity.detach().cpu().repeat(1, 1, 1, H // window_size, W // window_size).contiguous()
    x = x.reshape(B, -1, H * W, H * W).to(device)
    return x

@persistence.persistent_class
class Conv2dLayerPartial(nn.Module):
    def __init__(self,
                 in_channels,                    # Number of input channels.
                 out_channels,                   # Number of output channels.
                 kernel_size,                    # Width and height of the convolution kernel.
                 bias            = True,         # Apply additive bias before the activation function?
                 activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
                 up              = 1,            # Integer upsampling factor.
                 down            = 1,            # Integer downsampling factor.
                 resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
                 conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
                 trainable       = True,         # Update the weights of this layer during training?
                 ):
        super().__init__()
        self.conv = Conv2dLayer(in_channels, out_channels, kernel_size, bias, activation, up, down, resample_filter,
                                conv_clamp, trainable)

        self.weight_maskUpdater = torch.ones(1, 1, kernel_size, kernel_size)
        self.slide_winsize = kernel_size ** 2
        self.stride = down
        self.padding = kernel_size // 2 if kernel_size % 2 == 1 else 0

    def forward(self, x, mask=None):
        if mask is not None:
            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)
                update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding)
                mask_ratio = self.slide_winsize / (update_mask + 1e-8)
                update_mask = torch.clamp(update_mask, 0, 1)  # 0 or 1
                mask_ratio = torch.mul(mask_ratio, update_mask)
            x = self.conv(x)
            x = torch.mul(x, mask_ratio)
            return x, update_mask
        else:
            x = self.conv(x)
            return x, None


@persistence.persistent_class
class WindowAttention(nn.Module):      # 提出的MCA，对含有大量hole的mask的图像时做transformer
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, down_ratio=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.k = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.v = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.proj = FullyConnectedLayer(in_features=dim, out_features=dim)

        self.softmax = nn.Softmax(dim=-1)
        # self.attn_proj = nn.Dropout(attn_drop)

    def forward(self, x, mask_windows=None, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        norm_x = F.normalize(x, p=2.0, dim=-1)
        q = self.q(norm_x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(norm_x).view(B_, -1, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
        v = self.v(x).view(B_, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn_ = (q @ k) * self.scale

        if mask is not None:
            nW = mask.shape[0]
            attn_ = attn_.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn_ = attn_.view(-1, self.num_heads, N, N)

        if mask_windows is not None:
            attn_mask_windows = mask_windows.squeeze(-1).unsqueeze(1).unsqueeze(1)
            attn_ = attn_ + attn_mask_windows.masked_fill(attn_mask_windows == 0, float(-100.0)).masked_fill(
                attn_mask_windows == 1, float(0.0))
            with torch.no_grad():
                mask_windows = torch.clamp(torch.sum(mask_windows, dim=1, keepdim=True), 0, 1).repeat(1, N, 1)

        attn = self.softmax(attn_)
        # 之前mat是没有加Drop层
        # attn = self.attn_proj(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        # capture affinity from attention of transformer
        # attn_copy = attn.clone().reshape(B_, self.num_heads, self.window_size[0], self.window_size[1], attn.shape[-1])
        # attn_copy = attn_copy.reshape(-1, self.num_heads, attn.shape[-1], attn.shape[-1])

        return x, mask_windows


@persistence.persistent_class
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, down_ratio=1, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        if self.shift_size > 0:
            down_ratio = 1
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                    down_ratio=down_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                    proj_drop=drop)

        self.fuse = FullyConnectedLayer(in_features=dim * 2, out_features=dim, activation='lrelu')

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size, return_attn_pred=False, mask=None):
        # H, W = self.input_resolution
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)
        if mask is not None:
            mask = mask.view(B, H, W, 1)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if mask is not None:
                shifted_mask = torch.roll(mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            if mask is not None:
                shifted_mask = mask

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        if mask is not None:
            mask_windows = window_partition(shifted_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size, 1)
        else:
            mask_windows = None

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows, mask_windows = self.attn(x_windows, mask_windows=mask_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C; attn_(nW*B, nH, window_size*window_size, window_size*window_size)
        else:
            attn_windows, mask_windows = self.attn(x_windows, mask_windows=mask_windows, mask=self.calculate_mask(x_size).to(x.device))  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        if mask is not None:
            mask_windows = mask_windows.view(-1, self.window_size, self.window_size, 1)
            shifted_mask = window_reverse(mask_windows, self.window_size, H, W)
        # reverse attention map
        # if attn_pred is not None:
        #     attn_reverse_pred = attn_pred.reshape(B, -1, H * W, self.window_size, self.window_size)
        #     shifted_attn_pred = attention_reverse(attn_reverse_pred, self.window_size, H, W)
        # attn_reverse_map = attn_maps.reshape(B, -1, H * W, self.window_size, self.window_size)
        # shifted_attn_maps = attention_reverse(attn_reverse_map, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            if mask is not None:
                mask = torch.roll(shifted_mask, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            # if attn_pred is not None:
            #     attn_preds = torch.roll(shifted_attn_pred, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            # attn_map = torch.roll(attn_maps, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            if mask is not None:
                mask = shifted_mask
            # if attn_pred is not None:
            #     attn_preds = shifted_attn_pred
            # attn_map = attn_maps


        # 对attn_map进行点积操作（降采样的过程) 然后计算各个window之间的相关性，最后上采样到指定的长宽
        # B_, hW = attn_map.shape[0], attn_map.shape[1]     # B_=B*num_Window
        # attn_map = attn_map.view(B_, hW, 1, -1)
        # attn_map_t = attn_map.permute(0, 1, 3, 2)
        # attn_map_fusion = (attn_map @ attn_map_t)   #  shape (B_, hW, 1, 1)

        # 在各个window之间计算self-attention

        x = x.view(B, H * W, C)
        if mask is not None:
            mask = mask.view(B, H * W, 1)

        # FFN
        x = self.fuse(torch.cat([shortcut, x], dim=-1))
        x = self.mlp(x)

        # if return_attn_pred:
        #     return x, mask, attn_preds, attn_map
        # else:
        #     return x, mask, None, attn_map
        return x, mask


@persistence.persistent_class
class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, down=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       activation='lrelu',
                                       down=down,
                                       )
        self.down = down

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.down != 1:
            ratio = 1 / self.down
            x_size = (int(x_size[0] * ratio), int(x_size[1] * ratio))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask


@persistence.persistent_class
class PatchUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, up=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       activation='lrelu',
                                       up=up,
                                       )
        self.up = up

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.up != 1:
            x_size = (int(x_size[0] * self.up), int(x_size[1] * self.up))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask



@persistence.persistent_class
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, down_ratio=1,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, sr_ratio=1):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        if downsample is not None:
            # self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            self.downsample = downsample
        else:
            self.downsample = None

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, down_ratio=down_ratio, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        self.conv = Conv2dLayerPartial(in_channels=dim, out_channels=dim, kernel_size=3, activation='lrelu')

    def forward(self, x, x_size, mask=None):
        # attns_pred = []
        # attns_map = []
        if self.downsample is not None:
            x, x_size, mask = self.downsample(x, x_size, mask)
        identity = x
        for blk in self.blocks:
            if self.use_checkpoint:
                x, mask  = checkpoint.checkpoint(blk, x, x_size, mask)
                # attns_pred.append(attn_pred)
                # attns_map.append(attn_map)
            else:
                x, mask = blk(x, x_size, mask)
                # attns_pred.append(attn_pred)
                # attns_map.append(attn_map)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(token2feature(x, x_size), mask)
        x = feature2token(x) + identity
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask


@persistence.persistent_class
class ToToken(nn.Module):
    def __init__(self, in_channels=3, dim=128, kernel_size=5, stride=1):
        super().__init__()

        self.proj = Conv2dLayerPartial(in_channels=in_channels, out_channels=dim, kernel_size=kernel_size, activation='lrelu')

    def forward(self, x, mask):
        x, mask = self.proj(x, mask)

        return x, mask

#----------------------------------------------------------------------------

@persistence.persistent_class
class EncFromRGB(nn.Module):
    def __init__(self, in_channels, out_channels, activation):  # res = 2, ..., resolution_log2
        super().__init__()
        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                activation=activation,
                                )
        self.conv1 = Conv2dLayer(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                activation=activation,
                                )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        return x

@persistence.persistent_class
class ConvBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, activation):  # res = 2, ..., resolution_log
        super().__init__()

        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 down=2,
                                 )
        self.conv1 = Conv2dLayer(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        return x


def token2feature(x, x_size):
    B, N, C = x.shape
    h, w = x_size
    x = x.permute(0, 2, 1).reshape(B, C, h, w)
    return x


def feature2token(x):
    B, C, H, W = x.shape
    x = x.view(B, C, -1).transpose(1, 2)
    return x


@persistence.persistent_class
class Encoder(nn.Module):
    def __init__(self, res_log2, img_channels, activation, patch_size=5, channels=16, drop_path_rate=0.1):
        super().__init__()

        self.resolution = []

        for idx, i in enumerate(range(res_log2, 3, -1)):  # from input size to 16x16
            res = 2 ** i
            self.resolution.append(res)
            if i == res_log2:
                block = EncFromRGB(img_channels * 2 + 1, nf(i), activation)
            else:
                block = ConvBlockDown(nf(i+1), nf(i), activation)
            setattr(self, 'EncConv_Block_%dx%d' % (res, res), block)

    def forward(self, x):
        out = {}
        for res in self.resolution:
            res_log2 = int(np.log2(res))
            x = getattr(self, 'EncConv_Block_%dx%d' % (res, res))(x)
            out[res_log2] = x

        return out


@persistence.persistent_class
class ToStyle(nn.Module):
    def __init__(self, in_channels, out_channels, activation, drop_rate):
        super().__init__()
        self.conv = nn.Sequential(
                Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, activation=activation, down=2),
                Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, activation=activation, down=2),
                Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, activation=activation, down=2),
                )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = FullyConnectedLayer(in_features=in_channels,
                                      out_features=out_channels,
                                      activation=activation)
        # self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x.flatten(start_dim=1))
        # x = self.dropout(x)

        return x


@persistence.persistent_class
class DecBlockFirstV2(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):
        super().__init__()
        self.res = res

        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=3,
                                activation=activation,
                                )
        # self.conv1 = StyleConv(in_channels=in_channels,
        #                       out_channels=out_channels,
        #                       style_dim=style_dim,
        #                       resolution=2**res,
        #                       kernel_size=3,
        #                       use_noise=use_noise,
        #                       activation=activation,
        #                       demodulate=demodulate,
        #                       )
        self.conv1 = Conv2dLayer(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            activation=activation,
        )
        # self.toRGB = ToRGB(in_channels=out_channels,
        #                    out_channels=img_channels,
        #                    style_dim=style_dim,
        #                    kernel_size=1,
        #                    demodulate=False,
        #                    )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            kernel_size=1,
        )

    def forward(self, x, ws, gs, E_features, noise_mode='random'):
        # x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = self.conv0(x)
        x = x + E_features[self.res]
        # style = get_style_code(ws[:, 0], gs)
        # x = self.conv1(x, style, noise_mode=noise_mode)
        x = self.conv1(x)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img


# ----------------------------------------------------------------------------
@persistence.persistent_class
class DecSegBlock(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, img_channels):
        super().__init__()
        self.res = res

        self.conv0 = Conv2dLayer(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               up=2,
                               activation=activation,
                               )
        self.conv1 = Conv2dLayer(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               activation=activation,
                               )
        self.toRGB = ToSeg(in_channels=out_channels,
                           out_channels=img_channels,
                           kernel_size=1,
                           )

    def forward(self, x, img, skip):
        x = self.conv0(x)
        x = x + skip
        x = self.conv1(x)
        img = self.toRGB(x, skip=img)

        return x, img


#----------------------------------------------------------------------------

@persistence.persistent_class
class DecBlock(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):  # res = 4, ..., resolution_log2
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               up=2,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.conv1 = StyleConv(in_channels=out_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )

    def forward(self, x, img, ws, gs, E_features, noise_mode='random'):
        style = get_style_code(ws[:, self.res * 2 - 9], gs)
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, self.res * 2 - 8], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, self.res * 2 - 7], gs)
        img = self.toRGB(x, style, skip=img)

        return x, img


@persistence.persistent_class
class Decoder(nn.Module):
    def __init__(self, res_log2, activation, style_dim, use_noise, demodulate, img_channels):
        super().__init__()
        self.Dec_16x16 = DecBlockFirstV2(4, nf(4), nf(4), activation, style_dim, use_noise, demodulate, img_channels)
        for res in range(5, res_log2 + 1):
            setattr(self, 'Dec_%dx%d' % (2 ** res, 2 ** res),
                    DecBlock(res, nf(res - 1), nf(res), activation, style_dim, use_noise, demodulate, img_channels))
        self.res_log2 = res_log2

    def forward(self, x, ws, gs, E_features, noise_mode='random'):
        x, img = self.Dec_16x16(x, ws, gs, E_features, noise_mode=noise_mode)
        for res in range(5, self.res_log2 + 1):
            block = getattr(self, 'Dec_%dx%d' % (2 ** res, 2 ** res))
            x, img = block(x, img, ws, gs, E_features, noise_mode=noise_mode)

        return img

from .basic_module import UpSampling2d
@persistence.persistent_class
class DecStyleBlock(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):
    # def __init__(self, res, in_channels, out_channels, activation, img_channels, num_class):
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               up=2,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        # self.conv0 = Conv2dLayer(
        #                         in_channels=in_channels,
        #                         out_channels=out_channels,
        #                         kernel_size=3,
        #                         up=2,
        #                         activation=activation,
        # )
        # 利用fixed convs使得插值没有棋盘伪影
        # self.conv0 = UpSampling2d(in_channel=in_channels, out_channel=out_channels, kernel_size=3, stride=2, padding=1, hold_mode='hold_last', bias_mode='bias_last'
        # )
        self.conv1 = StyleConv(in_channels=out_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        # self.conv1 = Conv2dLayer(
        #     in_channels=out_channels,
        #     out_channels=out_channels,
        #     kernel_size=3,
        #     activation=activation,
        # )
        # self.conv1 = nn.Conv2d(
        #     in_channels=out_channels,
        #     out_channels=out_channels,
        #     kernel_size=3,
        #     padding=1,
        # )
        # self.activation = nn.LeakyReLU()
        # self.conv1_fusion_channel = nn.Conv2d(in_channels=308, out_channels=180,
        #                              kernel_size=1, stride=1, padding=0)

        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )
        # self.to_seg = ToSEG(in_channel=out_channels, out_channel=num_class, style_dim=style_dim)
        # self.toRGB = ToRGB(
        #                 in_channels=out_channels,
        #                 out_channels=img_channels,
        #                 kernel_size=1
        # )
        # self.to_seg = ToSEG(in_channel=out_channels, out_channel=num_class)

    # def forward(self, x, img_rgb, img_seg, skip):
    def forward(self, x, img_rgb, style, skip, noise_mode='random'):
        # x = self.conv0(x)
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + skip
        # x = self.conv1(x)
        # x = self.activation(x)
        x = self.conv1(x, style, noise_mode=noise_mode)
        # img_rgb = self.toRGB(x, skip=img_rgb)
        # img_seg = self.to_seg(x, skip=img_seg)
        img_rgb = self.toRGB(x, style, skip=img_rgb)
        # img_seg = self.to_seg(x, skip=img_seg)
        # if conv_out is not None:
        #     # 融合一个conv_out，代表noise直接经过多个conv_block之后的特征向量, 这样在第一阶段的encoder的特征有两部分，然后再经过decoder，再经过第二阶段的refinement
        #     x = torch.cat([x, conv_out], dim=1)
        #     x = self.conv1_fusion_channel(x)
        #     x = self.conv1_fusion(x, style, noise_mode=noise_mode)
        #     img_rgb = self.toRGB(x, style, skip=img_rgb)
        #     img_seg = self.to_seg(x, style, skip=img_seg)
        # else:
        #     x = self.conv1(x, style, noise_mode=noise_mode)
        #     img_rgb = self.toRGB(x, style, skip=img_rgb)
        #     img_seg = self.to_seg(x, style, skip=img_seg)

        return x, img_rgb

@persistence.persistent_class
class FirstStage_encoder(nn.Module):
    def __init__(self, img_channels=3, img_resolution=256, dim=180, activation='lrelu'):
        super().__init__()
        res = 64

        self.conv_first = Conv2dLayerPartial(in_channels=img_channels+1, out_channels=dim, kernel_size=3, activation=activation)
        self.enc_conv = nn.ModuleList()
        down_time = int(np.log2(img_resolution // res))
        for i in range(down_time):  # from input size to 64:  4 layers
            self.enc_conv.append(
                Conv2dLayerPartial(in_channels=dim, out_channels=dim, kernel_size=3, down=2, activation=activation)
            )

            # from 64 -> 16 -> 64
            depths = [2, 3, 4, 3, 2]
            ratios = [1, 1 / 2, 1 / 2, 2, 2]
            num_heads = 6  # original=6
            window_sizes = [8, 16, 16, 16, 8]
            drop_path_rate = 0.1
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

            self.tran = nn.ModuleList()
            for i, depth in enumerate(depths):  # 5 stages transformer block
                res = int(res * ratios[i])
                if ratios[i] < 1:
                    merge = PatchMerging(dim, dim, down=int(1 / ratios[i]))
                elif ratios[i] > 1:
                    merge = PatchUpsampling(dim, dim, up=ratios[i])
                else:
                    merge = None
                self.tran.append(
                    BasicLayer(dim=dim, input_resolution=[res, res], depth=depth, num_heads=num_heads,
                               window_size=window_sizes[i], drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                               downsample=merge)
                )

    def forward(self, images_in, masks_in):
        feature_dict = {}
        x = torch.cat([masks_in - 0.5, images_in * masks_in], dim=1)

        skips = []
        x, mask = self.conv_first(x, masks_in)  # input size
        skips.append(x)
        for i, block in enumerate(self.enc_conv):  # input size to 64
            x, mask= block(x, mask)
            if i != len(self.enc_conv) - 1:
                skips.append(x)
        feature_dict.update({'conv_list': x})

        x_size = x.size()[-2:]
        x = feature2token(x)
        mask = feature2token(mask)
        mid = len(self.tran) // 2
        for i, block in enumerate(self.tran):  # 64 to 16
            if i < mid:
                x, x_size, mask  = block(x, x_size, mask)
                # for m in range(len(attn_low_pred)):
                #     # attns_preds.append(attn_low_pred[m])
                #     attns_maps.append(attn_low_map[m])
                skips.append(x)
                feature_dict.update({'trans_{}'.format(i): x})
            elif i > mid:
                x, x_size, mask = block(x, x_size, None)
                x = x + skips[mid - i]
                # for m in range(len(attn_high_pred)):
                #     if i == len(self.tran) - 1 :
                #         attns_preds.append(attn_high_pred[m])
                #     attns_maps.append(attn_high_map[m])
            else:
                x, x_size, mask = block(x, x_size, None)
                # for m in range(len(attn_mid_pred)):
                #     # attns_preds.append(attn_mid_pred[m])
                #     attns_maps.append(attn_mid_map[m])
                feature_dict.update({'trans_{}'.format(i): x})

        x = token2feature(x, x_size).contiguous()

        return feature_dict



@persistence.persistent_class
class FirstStage(nn.Module):
    # def __init__(self, img_channels, num_class, img_resolution=256, dim=180, w_dim=512, activation='lrelu'):
    def __init__(self, img_channels, num_class, img_resolution=256, dim=180, w_dim=512, use_noise=False, demodulate=True, activation='lrelu'):
        super().__init__()
        res = 64

        self.conv_first = Conv2dLayerPartial(in_channels=img_channels+1, out_channels=dim, kernel_size=3, activation=activation)
        self.enc_conv = nn.ModuleList()
        down_time = int(np.log2(img_resolution // res))
        for i in range(down_time):  # from input size to 64:  4 layers
            self.enc_conv.append(
                Conv2dLayerPartial(in_channels=dim, out_channels=dim, kernel_size=3, down=2, activation=activation)
            )

        # from 64 -> 16 -> 64
        depths = [2, 3, 4, 3, 2]
        ratios = [1, 1/2, 1/2, 2, 2]
        num_heads = 6   # original=6
        window_sizes = [8, 16, 16, 16, 8]
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.tran = nn.ModuleList()
        for i, depth in enumerate(depths):    # 5 stages transformer block
            res = int(res * ratios[i])
            if ratios[i] < 1:
                merge = PatchMerging(dim, dim, down=int(1/ratios[i]))
            elif ratios[i] > 1:
                merge = PatchUpsampling(dim, dim, up=ratios[i])
            else:
                merge = None
            self.tran.append(
                BasicLayer(dim=dim, input_resolution=[res, res], depth=depth, num_heads=num_heads,
                           window_size=window_sizes[i], drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                           downsample=merge)
            )

        # global style
        down_conv = []
        for i in range(int(np.log2(16))):   # 4
            down_conv.append(Conv2dLayer(in_channels=dim, out_channels=dim, kernel_size=3, down=2, activation=activation))
        down_conv.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.down_conv = nn.Sequential(*down_conv)
        self.to_style = FullyConnectedLayer(in_features=dim, out_features=dim*2, activation=activation)

        self.ws_style = FullyConnectedLayer(in_features=w_dim, out_features=dim, activation=activation)
        self.to_square = FullyConnectedLayer(in_features=dim, out_features=16*16, activation=activation)
        style_dim = dim * 3
        self.dec_conv1 = nn.ModuleList()
        for i in range(down_time):  # from 64 to input size
            res = res * 2
            self.dec_conv1.append(DecStyleBlock(res, dim, dim, activation, style_dim, use_noise, demodulate, img_channels, num_class))
            # self.dec_conv.append(
            #     DecStyleBlock(res, dim, dim, activation, img_channels, num_class))

        # self.softmax = nn.Softmax(dim=1)
        # self.ccm = confidence_map()
        # self.eri_1 = edge_region_inpainting(max_iter=1)
        # self.eri_2 = edge_region_inpainting(max_iter=1)
        # self.eri_3 = edge_region_inpainting(max_iter=1)
        self.feature_dict = {}

    # def forward(self, images_in, masks_in):
    def forward(self, images_in, masks_in, ws, noise_mode='random'):     #
    # def forward(self, images_in, masks_in, ws, conv_out, noise_mode='random'):
    #     attns_preds = []
    #     attns_maps = []
        x = torch.cat([masks_in - 0.5, images_in * masks_in], dim=1)

        skips = []
        x, mask = self.conv_first(x, masks_in)  # input size
        skips.append(x)
        for i, block in enumerate(self.enc_conv):  # input size to 64
            x, mask= block(x, mask)
            if i != len(self.enc_conv) - 1:
                skips.append(x)

        x_size = x.size()[-2:]
        x = feature2token(x)
        mask = feature2token(mask)
        mid = len(self.tran) // 2
        for i, block in enumerate(self.tran):  # 64 to 16
            if i < mid:
                x, x_size, mask  = block(x, x_size, mask)
                # for m in range(len(attn_low_pred)):
                #     # attns_preds.append(attn_low_pred[m])
                #     attns_maps.append(attn_low_map[m])
                skips.append(x)
            elif i > mid:
                x, x_size, mask = block(x, x_size, None)
                x = x + skips[mid - i]
                # for m in range(len(attn_high_pred)):
                #     if i == len(self.tran) - 1 :
                #         attns_preds.append(attn_high_pred[m])
                #     attns_maps.append(attn_high_map[m])
            else:
                x, x_size, mask = block(x, x_size, None)
                # for m in range(len(attn_mid_pred)):
                #     # attns_preds.append(attn_mid_pred[m])
                #     attns_maps.append(attn_mid_map[m])

                # MAT提出的SSM模块，加入各种style，促使生成的图像的多种风格， 是在i=2的阶段（第三个transformer的stage）去融合noise
                # if noise_mode == 'random':
                # mul_map是随机生成的二值掩码，即B，
                # mul_map = torch.ones_like(x) * 0.5
                # mul_map = F.dropout(mul_map, training=True)   # 在 training 模式下，基于伯努利分布抽样，以概率 p 对张量 input 的值随机置0
                # noise经过mapping network后的noise-unconditional style，即s_u
                ws = self.ws_style(ws[:, -1])
                # 对style noise做一个映射，经过E的映射
                # add_n = self.to_square(ws).unsqueeze(1)
                # 用来上采样或下采样，可以给定size或者scale_factor来进行上下采样
                # add_n = F.interpolate(add_n, size=x.size(1), mode='linear', align_corners=False).squeeze(1).unsqueeze(-1)
                # 将引入的noise s_u加入到重建的图像中，并新建二值掩码B作用到原图中以增加原图的多样性
                # x = x * mul_map + add_n * (1 - mul_map)
                # 这里提取新融合Noise的图像的风格特征，即s_c
                gs = self.to_style(self.down_conv(token2feature(x, x_size)).flatten(start_dim=1))
                # 这里将s_u和s_c融合形成新的style 特征
                style = torch.cat([gs, ws], dim=1)

        x = token2feature(x, x_size).contiguous()
        img_rgb = None
        img_seg = None
        for i, block in enumerate(self.dec_conv1):
                x, img_rgb, img_seg = block(x, img_rgb, img_seg, style, skips[len(self.dec_conv1) - i - 1], noise_mode=noise_mode)
                # x, img_rgb, img_seg = block(x, img_rgb, img_seg, skips[len(self.dec_conv1) - i - 1])

            # if i == len(self.dec_conv) - 1:
            #     x, img_rgb, img_seg = block(x, img_rgb, img_seg, style, conv_out, skips[len(self.dec_conv) - i - 1], noise_mode=noise_mode)
            # else:
            #     conv_out_fusion = None
            #     x, img_rgb, img_seg = block(x, img_rgb, img_seg, style, conv_out_fusion, skips[len(self.dec_conv) - i - 1], noise_mode=noise_mode)

        # ensemble
        img_rgb = img_rgb * (1 - masks_in) + images_in * masks_in

        # refine segmentation at the second stage for seg
        # img_seg_score_1 = self.softmax(img_seg)
        # seg_confidence_1 = self.ccm(img_seg_score_1)
        # feature_1, img_seg_refine_1 = self.eri_1(x, seg_confidence_1)
        #
        # img_seg_score_2 = self.softmax(img_seg_refine_1)
        # seg_confidence_2 = self.ccm(img_seg_score_2)
        # feature_2, img_seg_refine_2 = self.eri_2(feature_1, seg_confidence_2)
        #
        # img_seg_score_3 = self.softmax(img_seg_refine_2)
        # seg_confidence_3 = self.ccm(img_seg_score_3)
        # feature_3, img_seg_refine_3 = self.eri_3(feature_2, seg_confidence_3)

        # return img_rgb, img_seg
        return x, img_rgb, img_seg


@persistence.persistent_class
class SynthesisNet(nn.Module):
    def __init__(self,
                 w_dim,                     # Intermediate latent (W) dimensionality.
                 img_resolution,            # Output image resolution.
                 num_class,                 # 分割的种类
                 img_channels   = 3,        # Number of color channels.
                 channel_base   = 32768,    # Overall multiplier for the number of channels.
                 channel_decay  = 1.0,
                 channel_max    = 512,      # Maximum number of channels in any layer.
                 activation     = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
                 drop_rate      = 0.5,
                 use_noise      = True,
                 demodulate     = True,
                 ):
        super().__init__()
        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2 ** resolution_log2 and img_resolution >= 4

        self.num_layers = resolution_log2 * 2 - 3 * 2
        self.img_resolution = img_resolution
        self.resolution_log2 = resolution_log2

        # first stage
        self.first_stage = FirstStage(img_channels, num_class, img_resolution=img_resolution, w_dim=w_dim, use_noise=False, demodulate=demodulate)
        # self.first_stage = FirstStage(img_channels, num_class, img_resolution=img_resolution, w_dim=w_dim)

        # second stage
        # self.enc = Encoder(resolution_log2, img_channels, activation, patch_size=5, channels=16)
        # self.to_square = FullyConnectedLayer(in_features=w_dim, out_features=16*16, activation=activation)
        # self.to_style = ToStyle(in_channels=nf(4), out_channels=nf(2) * 2, activation=activation, drop_rate=drop_rate)
        # style_dim = w_dim + nf(2) * 2
        # self.dec = Decoder(resolution_log2, activation, style_dim, use_noise, demodulate, img_channels)

    # def forward(self, images_in, masks_in):
    def forward(self, images_in, masks_in, ws, noise_mode='random'):
        x_feature, out_stg1_rgb, out_stg1_seg = self.first_stage(images_in, masks_in, ws, noise_mode=noise_mode)
        # x_feature, out_stg1_rgb, out_stg1_seg = self.first_stage(images_in, masks_in)

    # def forward(self, images_in, masks_in, ws, conv_out, noise_mode='random', return_stg1=False):
        # out_stg1_rgb, out_stg1_seg = self.first_stage(images_in, masks_in, ws, conv_out, noise_mode=noise_mode)

        # encoder   第二阶段：refine生成的被随机掩码的图像rgb，
        # x_rgb = images_in * masks_in + out_stg1_rgb * (1 - masks_in)
        # x_rgb = torch.cat([masks_in - 0.5, x_rgb, images_in * masks_in], dim=1)
        # E_features_rgb = self.enc(x_rgb)
        #
        # fea_16_rgb = E_features_rgb[4]    # 代表的是feature map是16*16的
        # mul_map_rgb = torch.ones_like(fea_16_rgb) * 0.5
        # mul_map_rgb = F.dropout(mul_map_rgb, training=True)
        # add_n_rgb = self.to_square(ws[:, 0]).view(-1, 16, 16).unsqueeze(1)
        # add_n_rgb = F.interpolate(add_n_rgb, size=fea_16_rgb.size()[-2:], mode='bilinear', align_corners=False)
        # fea_16_rgb = fea_16_rgb * mul_map_rgb + add_n_rgb * (1 - mul_map_rgb)
        # fea_16_rgb = fea_16_rgb * masks_in + add_n_rgb * (1 - masks_in)
        # E_features_rgb[4] = fea_16_rgb
        #
        # style
        # gs_rgb = self.to_style(fea_16_rgb)
        #
        # # decoder
        # img_rgb = self.dec(fea_16_rgb, ws, gs_rgb, E_features_rgb, noise_mode=noise_mode)
        #
        # # ensemble
        # img_rgb = img_rgb * (1 - masks_in) + images_in * masks_in

        # if not return_stg1:
        #     return img_rgb
        # else:
        #     return img_rgb, out_stg1_rgb
        return x_feature, out_stg1_rgb, out_stg1_seg


@persistence.persistent_class
class Generator(nn.Module):
    def __init__(self,
                 z_dim,                  # Input latent (Z) dimensionality, 0 = no latent.
                 c_dim,                  # Conditioning label (C) dimensionality, 0 = no label.
                 w_dim,                  # Intermediate latent (W) dimensionality.
                 num_class,              # 分割的种类
                 img_resolution,         # resolution of generated image
                 img_channels,           # Number of input color channels.
                 synthesis_kwargs = {},  # Arguments for SynthesisNetwork.
                 mapping_kwargs   = {},  # Arguments for MappingNetwork.
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.synthesis = SynthesisNet(w_dim=w_dim,
                                      img_resolution=img_resolution,
                                      num_class=num_class,
                                      img_channels=img_channels,
                                      **synthesis_kwargs)

        self.mapping = MappingNet(z_dim=z_dim,
                                  c_dim=c_dim,
                                  w_dim=w_dim,
                                  num_ws=self.synthesis.num_layers,
                                  **mapping_kwargs)

        # self.attn_proj = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=1, bias=True)
        # self.softmax = nn.Softmax(dim=-1)

    # def forward(self, images_in, masks_in):
    def forward(self, images_in, masks_in, ws, noise_mode='random'):
    # def forward(self, images_in, masks_in, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False,
    #             noise_mode='random'):
    #     ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
    #                       skip_w_avg_update=skip_w_avg_update)
        x_feature, img_rgb, img_seg = self.synthesis(images_in, masks_in, ws, noise_mode=noise_mode)
        # x_feature, img_rgb, img_seg = self.synthesis(images_in, masks_in)

        # attn_maps = _attns_maps
        # attns_avg = []
        #
        # # 对每个block的所有注意力头得到的affinity matrix求平均
        # # 第1个depths: 2层basiclayer
        # attns_avg.append(torch.cat(attn_maps[0:2], dim=1))
        # # 第2个depths: 3层basiclayer
        # attns_avg.append(torch.cat(attn_maps[2:5], dim=1))
        # # 第3个depths: 4层basiclayer  每层得到的feature map shape(1, 6, 256, 256)
        # attns_avg.append(torch.cat(attn_maps[5:9], dim=1))
        # # 第4个depths: 3层basiclayer
        # attns_avg.append(torch.cat(attn_maps[9:12], dim=1))
        # # 第5个depths: 2层basiclayer
        # attns_avg.append(torch.cat(attn_maps[12:], dim=1))
        #
        # # 对每个block做平均
        # attns_avg_layer = attns_avg
        # for i in range(len(attns_avg_layer)):
        #     attns_avg_layer[i] = attns_avg_layer[i].mean(1).unsqueeze(1)
        #     if attns_avg_layer[i].shape[1:] != attns_avg_layer[0].shape[1:]:
        #         attns_avg_layer[i] =F.interpolate(attns_avg_layer[i], size=attns_avg[0].shape[2:], mode='bilinear', align_corners=False)
        #
        # attn_map_cat = torch.cat(attns_avg_layer, dim=1)
        #
        # attn_map_cat_avg = attn_map_cat.mean(1)
        #
        # attn_map_cat_avg = torch.sigmoid(attn_map_cat_avg)
        #
        # # 计算affinity矩阵, 利用最后一层的transformer block的attention作为affinity矩阵
        # attn_pred = _attns_preds[-1]
        # # attn_cat = torch.cat(_attns_preds[-2:], dim=1)
        # attn_affinity = attn_pred + attn_pred.permute(0, 1, 3, 2)
        # attn_pred = self.attn_proj(attn_affinity)
        # attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]
        #
        # attn_total = torch.stack((attn_pred, attn_map_cat_avg), 0)

        return x_feature, img_rgb, img_seg


    # def forward(self, images_in, masks_in, ws, conv_out, noise_mode='random', return_stg1=False):
    #
    #     if not return_stg1:
    #         img_rgb = self.synthesis(images_in, masks_in, ws, conv_out, noise_mode=noise_mode)
    #         return img_rgb
    #     else:
    #         img_rgb, out_stg1 = self.synthesis(images_in, masks_in, ws, conv_out, noise_mode=noise_mode, return_stg1=True)
    #         return img_rgb, out_stg1


@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
                 c_dim,                        # Conditioning label (C) dimensionality.
                 img_resolution,               # Input resolution.
                 img_channels,                 # Number of input color channels.
                 channel_base       = 32768,    # Overall multiplier for the number of channels.
                 channel_max        = 512,      # Maximum number of channels in any layer.
                 channel_decay      = 1,
                 cmap_dim           = None,     # Dimensionality of mapped conditioning label, None = default.
                 activation         = 'lrelu',
                 mbstd_group_size   = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
                 mbstd_num_channels = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
                 ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2 ** resolution_log2 and img_resolution >= 4
        self.resolution_log2 = resolution_log2

        if cmap_dim == None:
            cmap_dim = nf(2)
        if c_dim == 0:
            cmap_dim = 0
        self.cmap_dim = cmap_dim

        if c_dim > 0:
            self.mapping = MappingNet(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None)

        Dis = [DisFromRGB(img_channels+1, nf(resolution_log2), activation)]
        for res in range(resolution_log2, 2, -1):
            Dis.append(DisBlock(nf(res), nf(res-1), activation))

        if mbstd_num_channels > 0:
            Dis.append(MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels))
        Dis.append(Conv2dLayer(nf(2) + mbstd_num_channels, nf(2), kernel_size=3, activation=activation))
        self.Dis = nn.Sequential(*Dis)

        self.fc0 = FullyConnectedLayer(nf(2)*4**2, nf(2), activation=activation)
        self.fc1 = FullyConnectedLayer(nf(2), 1 if cmap_dim == 0 else cmap_dim)

        # for 64x64
        Dis_stg1 = [DisFromRGB(img_channels+1, nf(resolution_log2) // 2, activation)]
        for res in range(resolution_log2, 2, -1):
            Dis_stg1.append(DisBlock(nf(res) // 2, nf(res - 1) // 2, activation))

        if mbstd_num_channels > 0:
            Dis_stg1.append(MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels))
        Dis_stg1.append(Conv2dLayer(nf(2) // 2 + mbstd_num_channels, nf(2) // 2, kernel_size=3, activation=activation))
        self.Dis_stg1 = nn.Sequential(*Dis_stg1)

        self.fc0_stg1 = FullyConnectedLayer(nf(2) // 2 * 4 ** 2, nf(2) // 2, activation=activation)
        self.fc1_stg1 = FullyConnectedLayer(nf(2) // 2, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, images_in, masks_in, images_stg1, c):
        x = self.Dis(torch.cat([masks_in - 0.5, images_in], dim=1))
        x = self.fc1(self.fc0(x.flatten(start_dim=1)))

        x_stg1 = self.Dis_stg1(torch.cat([masks_in - 0.5, images_stg1], dim=1))
        x_stg1 = self.fc1_stg1(self.fc0_stg1(x_stg1.flatten(start_dim=1)))

        if self.c_dim > 0:
            cmap = self.mapping(None, c)

        if self.cmap_dim > 0:
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
            x_stg1 = (x_stg1 * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return x, x_stg1


if __name__ == '__main__':
    device = torch.device('cuda:0')
    batch = 1
    res = 512
    G = Generator(z_dim=512, c_dim=0, w_dim=512, num_class=2, img_resolution=512, img_channels=3).to(device)
    D = Discriminator(c_dim=0, img_resolution=res, img_channels=3).to(device)
    img = torch.randn(batch, 3, res, res).to(device)
    mask = torch.randn(batch, 1, res, res).to(device)
    z = torch.randn(batch, 512).to(device)
    G.eval()

    # def count(block):
    #     return sum(p.numel() for p in block.parameters()) / 10 ** 6
    # print('Generator', count(G))
    # print('discriminator', count(D))

    with torch.no_grad():
        img, img_stg1 = G(img, mask, z, None, return_stg1=True)
    print('output of G:', img.shape, img_stg1.shape)
    score, score_stg1 = D(img, mask, img_stg1, None)
    print('output of D:', score.shape, score_stg1.shape)
