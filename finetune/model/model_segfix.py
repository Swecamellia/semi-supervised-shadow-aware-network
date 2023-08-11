# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html
from pyexpat import features
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.utils import misc, persistence
from models.mat import Conv2dLayerPartial, PatchMerging, PatchUpsampling, BasicLayer, FullyConnectedLayer, DecStyleBlock, token2feature, feature2token
from models.basic_module import Conv2dLayer, MappingNet
from functools import partial
from segmentation_models_pytorch.base import modules as md
from utils.offset import DTOffsetConfig, DTOffsetHelper

# We use this code from https://github.com/openseg-group/openseg.pytorch

class OffsetBlock(nn.Module):
    '''
    This module takes relative offset as input and outputs feature at each position (coordinate + offset)
    '''
    def __init__(self):
        super(OffsetBlock, self).__init__()
        self.coord_map = None
        self.norm_factor = None
    
    def _gen_coord_map(self, H, W):
        coord_vecs = [torch.arange(length, dtype=torch.float).cuda() for length in (H, W)]
        coord_h, coord_w = torch.meshgrid(coord_vecs)
        return coord_h, coord_w
    
    def forward(self, x, offset_map):
        n, c, h, w = x.size()
        
        if self.coord_map is None or self.coord_map[0].size() != offset_map.size()[2:]:
            self.coord_map = self._gen_coord_map(h, w)
            self.norm_factor = torch.cuda.FloatTensor([(w-1) / 2, (h-1) / 2])
        
        # offset to absolute coordinate
        grid_h = offset_map[:, 0] + self.coord_map[0]                               # (N, H, W)
        grid_w = offset_map[:, 1] + self.coord_map[1]                               # (N, H, W)

        # scale to [-1, 1], order of grid: [x, y] (i.e., [w, h])
        grid = torch.stack([grid_w, grid_h], dim=-1) / self.norm_factor - 1.        # (N, H, W, 2)

        # use grid to obtain output feature
        feats = F.grid_sample(x, grid, padding_mode='border', mode='bilinear', align_corners=True)     
        feats = torch.round(feats)                  # (N, C, H, W)
        
        return feats

class OffsetModule(nn.Module):
    def __init__(self):
        super(OffsetModule, self).__init__()
        self.offset_block = OffsetBlock()
    
    def forward(self, x, offset):
        # sample
        x_out = self.offset_block(x, offset)
        return x_out


class boundary_refine(nn.Module):
    def __init__(self, in_channels):
        super(boundary_refine, self).__init__()
        num_directions = DTOffsetConfig.num_classes
        num_masks = 2
        self.num = 1

        mid_channels = 128

        self.dir_head = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels,
                      num_directions,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False))
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels,
                      num_masks,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False))
    

    def forward(self, x):
        _, _, h, w = x[-1].size()
        # feats = x[0]
        for i in range(len(x)-1):
            x[i] = F.interpolate(x[i],
                                 size=(h, w),
                                 mode='bilinear',
                                 align_corners=True)

        feats = torch.cat(x, 1)
        # feats = x
        mask_map = self.mask_head(feats)
        dir_map = self.dir_head(feats)
        return mask_map, dir_map
        

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPSeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            SeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, separable=False):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv

        modules.append(ASPPConvModule(in_channels, out_channels, rate1))
        modules.append(ASPPConvModule(in_channels, out_channels, rate2))
        modules.append(ASPPConvModule(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class SeparableConv2d(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.num = 11
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, scale_factor=2, mode="nearest")

            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)

class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]

        self.blocks = nn.ModuleList(blocks)


    def forward(self, *features):

        head = features[0]
        skips = features[1][::-1]

        x = self.center(head)
        features = [head]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            # features = x
            x = decoder_block(x, skip)
            
            features.append(x)
        return x, features

class SwinEncoder(nn.Module):
    def __init__(self, img_channels, output_channels, img_resolution=256, res=64, dim=128, w_dim=512, use_noise=False, demodulate=True, activation='lrelu'):
        super().__init__()

        self.conv_first = Conv2dLayerPartial(in_channels=img_channels, out_channels=dim, kernel_size=3, activation=activation)
        self.enc_conv = nn.ModuleList()
        down_time = int(np.log2(img_resolution // res))
        self.num_layers = int(np.log2(img_resolution)) * 2 - 3 * 2
        for i in range(down_time):  # from input size to 64
            self.enc_conv.append(
                Conv2dLayerPartial(in_channels=dim, out_channels=dim, kernel_size=3, down=2, activation=activation)
            )
        
        # from 64 -> 16 -> 64
        depths = [2, 3, 4, 3, 2]
        ratios = [1, 1/2, 1/2, 2, 2]
        num_heads = 8
        window_sizes = [8, 16, 16, 16, 8]
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.tran = nn.ModuleList()
        for i, depth in enumerate(depths):
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
        for i in range(int(np.log2(16))):
            down_conv.append(Conv2dLayer(in_channels=dim, out_channels=dim, kernel_size=3, down=2, activation=activation))
        down_conv.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.down_conv = nn.Sequential(*down_conv)

        for p in self.parameters():
            p.requires_grad = False

        self.decoder = UnetDecoder([128, 128, 128, 128, 128], [128, 128, 128, 128, output_channels])
        # self.psp = PSPDecoder([128, 128, 128, 128, 128], out_channels=128)
        self.aspp = nn.Sequential(
            ASPP(128,128, [12, 24, 36], separable=True),
            SeparableConv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        style_dim = dim * 3
        self.dec_conv = nn.ModuleList()
        self.dec_conv1 = nn.ModuleList()

        for i in range(down_time):  # from 64 to input size
            res = res * 2
            self.dec_conv.append(DecStyleBlock(res, dim, dim, activation, style_dim, use_noise, demodulate, output_channels))
        self.eri = boundary_refine(256)
        

    def forward(self, x, mask,  ws, noise_mode='random'):

        skips = []
        x, mask = self.conv_first(x, mask)  # input size
        skips.append(x)
        for i, block in enumerate(self.enc_conv):  # input size to 64
            x, mask = block(x, mask)
            if i != len(self.enc_conv) - 1:
                skips.append(x)

        x_size = x.size()[-2:]
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        mid = len(self.tran) // 2
        for i, block in enumerate(self.tran):  # 64 to 16
            if i < mid:
                x, x_size, mask = block(x, x_size, mask)
                skips.append(token2feature(x, x_size=x_size))
            elif i > mid:
                break
                x, x_size, mask = block(x, x_size, None)
                x = x + skips[mid - i]
            else:
                x, x_size, mask = block(x, x_size, None)

        x = token2feature(x, x_size).contiguous()
        x = self.aspp(x)
        x, features = self.decoder(x, skips)

        boundary, direction = self.eri(features[3:-1])
        return x, boundary, direction
        # return x


class GeneratorSeg(nn.Module):
    def __init__(self, img_channels, output_channels, img_resolution=256, res = 64, dim=128, w_dim=512, use_noise=False, demodulate=True, activation='lrelu'):
        super().__init__()
        
        self.encoder = SwinEncoder(img_channels, output_channels, img_resolution, res, dim, w_dim, use_noise, demodulate, activation)
        self.mapping = MappingNet(z_dim=w_dim,
                                  c_dim=0,
                                  w_dim=w_dim,
                                  num_ws=self.encoder.num_layers) 
       
        self.refiner = OffsetModule()


    def get_offset(self, mask_logits, dir_logits):

        edge_mask = mask_logits[:, 1] > 0.5
        dir_logits = torch.softmax(dir_logits, dim=1)
        n, _, h, w = dir_logits.shape

        keep_mask = edge_mask

        dir_label = torch.argmax(dir_logits, dim=1).float()
        offset = DTOffsetHelper.label_to_vector(dir_label)
        offset = offset.permute(0, 2, 3, 1)
        offset[~keep_mask, :] = 0
        return offset.permute(0, 3, 1, 2)

    def forward(self, img, mask, return_latents=False):
        batch = img.shape[0]

        noise_generate = torch.from_numpy(np.random.randn(batch, 512)).to(img.device)
        labels = torch.zeros([noise_generate.shape[0], 0], device=img.device)
        ws = self.mapping(noise_generate, labels)
        noise_mode = 'random'
        logists = self.encoder(img, mask, ws, noise_mode)
        return logists
