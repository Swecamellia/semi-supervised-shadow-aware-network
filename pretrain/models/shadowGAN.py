# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unittest2 import skip
from models import misc, persistence
from models.mat import Conv2dLayerPartial, PatchMerging, PatchUpsampling, BasicLayer, FullyConnectedLayer, DecSegBlock, token2feature, feature2token
from models.basic_module import Conv2dLayer, MappingNet

class SwinEncoder(nn.Module):
    def __init__(self, img_channels, output_channels, img_resolution=256, res=64, dim=128, w_dim=512, use_noise=False, demodulate=True, activation='lrelu'):
        super().__init__()

        resolution_log2 = int(np.log2(img_resolution))
        self.num_ws = resolution_log2 * 2 - 3 * 2
        self.mapping = MappingNet(z_dim=512,
                                  c_dim=0,
                                  w_dim=512,
                                  num_ws=self.num_ws,
                                  )

        self.conv_first = Conv2dLayerPartial(in_channels=img_channels, out_channels=dim, kernel_size=3, activation=activation)
        self.enc_conv = nn.ModuleList()
        down_time = int(np.log2(img_resolution // res))
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
        # self.to_style = FullyConnectedLayer(in_features=dim, out_features=dim*2, activation=activation)
        self.ws_style = FullyConnectedLayer(in_features=w_dim, out_features=dim, activation=activation)
        self.to_square = FullyConnectedLayer(in_features=dim, out_features=16*16, activation=activation)
        
        style_dim = dim * 3
        self.dec_conv = nn.ModuleList()
        for i in range(down_time):  # from 64 to input size
            res = res * 2
            self.dec_conv.append(DecSegBlock(res, dim, dim, activation, output_channels))

        # self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.out1 = nn.Conv2d(dim, dim//2, 3, 1, 1)
        # self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.out2 = nn.Conv2d(dim//2, dim//4, 3, 1, 1)
        # self.out3 = nn.Conv2d(dim//4, 2, 1)

    def forward(self, images_in, masks_in, return_latents=False, return_latent_output=False, phase='train'):

        batch = images_in.shape[0]
        if phase == 'train' or phase == 'test':
            noise_generate = torch.from_numpy(np.random.randn(batch, 512)).to(images_in.device)
            labels = torch.zeros([noise_generate.shape[0], 0], device=images_in.device)
            ws = self.mapping(noise_generate, labels)
            noise_mode = 'random'
        else:
            ws = torch.ones((batch, 10, 512), device=images_in.device)
            noise_mode = 'none'

        skips = []
        x, mask = self.conv_first(images_in, masks_in)  # input size
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
                skips.append(x)
            elif i > mid:
                x, x_size, mask = block(x, x_size, None)
                x = x + skips[mid - i]
            else:
                x, x_size, mask = block(x, x_size, None)

                # mul_map = torch.ones_like(x) * 0.5
                # mul_map = F.dropout(mul_map, training=True)
                ws = self.ws_style(ws[:, -1])
                # add_n = self.to_square(ws).unsqueeze(1)
                # add_n = F.interpolate(add_n, size=x.size(1), mode='linear', align_corners=False).squeeze(1).unsqueeze(-1)
                # x = x * mul_map + add_n * (1 - mul_map)
                gs = self.to_style(self.down_conv(token2feature(x, x_size)).flatten(start_dim=1))
                style = torch.cat([gs, ws], dim=1)

        x = token2feature(x, x_size).contiguous()
        img_rgb = None
        img_seg = None
        for i, block in enumerate(self.dec_conv):
            x, img_rgb, img_seg = block(x, img_rgb, img_seg, style, skips[len(self.dec_conv1) - i - 1], noise_mode=noise_mode)

        # ensemble
        img_rgb = img_rgb * (1 - masks_in) + images_in * masks_in

        if return_latents:
            return img_rgb, ws, x
        elif return_latent_output:
            return img_rgb, img_seg, x
        else:
            return img_rgb, img_seg

# class SegDecoder(nn.Module):
#     def __init__(self, img_channels, img_resolution=256, res = 64, dim=180, w_dim=512, use_noise=False, demodulate=True, activation='lrelu'):
#         super().__init__()
#         down_time = int(np.log2(img_resolution // res))
#         style_dim = dim * 3
#         self.dec_conv = nn.ModuleList()
#         for i in range(down_time):  # from 64 to input size
#             res = res * 2
#             self.dec_conv.append(DecStyleBlock(res, dim, dim, activation, style_dim, use_noise, demodulate, img_channels))

#     def forward(self, x, mask):
#         img = None
#         for i, block in enumerate(self.dec_conv):
#             x, img = block(x, img, style, skips[len(self.dec_conv)-i-1], noise_mode='none')
#         return img

class GeneratorSeg(nn.Module):
    def __init__(self, img_channels, output_channels, img_resolution=256, res = 64, dim=128, w_dim=512, use_noise=False, demodulate=True, activation='lrelu'):
        super().__init__()
        
        self.encoder = SwinEncoder(img_channels, output_channels, img_resolution, res, dim, w_dim, use_noise, demodulate, activation)
        # self.decoder = SegDecoder(img_channels, img_resolution, res, dim, w_dim, use_noise, demodulate, activation)

    # def make_noise(self):
    #     device = self.input.input.device

    #     noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

    #     for i in range(3, self.log_size + 1):
    #         for _ in range(2):
    #             noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))
    #     return noises

    # def mean_latent(self, n_latent):
    #     latent_in = torch.randn(
    #         n_latent, self.style_dim, device=self.input.input.device
    #     )
    #     latent = self.style(latent_in).mean(0, keepdim=True)
    #     return latent

    # def get_latent(self, input):
    #     return self.encoder(input)

    # def sample(self, num, latent_space_type='Z'):
    #     """Samples latent codes randomly.
    #     Args:
    #     num: Number of latent codes to sample. Should be positive.
    #     latent_space_type: Type of latent space from which to sample latent code.
    #         Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)
    #     Returns:
    #     A `numpy.ndarray` as sampled latend codes.
    #     Raises:
    #     ValueError: If the given `latent_space_type` is not supported.
    #     """
    #     latent_space_type = latent_space_type.upper()
    #     if latent_space_type == 'Z':
    #         latent_codes = np.random.randn(num, self.style_dim)
    #     elif latent_space_type == 'W':
    #         latent_codes = np.random.randn(num, self.style_dim)
    #     elif latent_space_type == 'WP':
    #         latent_codes = np.random.randn(num, self.n_latent, self.style_dim)
    #     else:
    #         raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

    #     return latent_codes.astype(np.float32)

    def forward(self, img, mask, return_latents=False):

        logists = self.encoder(img, mask)
        return torch.softmax(logists, dim=1)
        # return logists
        # seg_image, rec_image = self.decoder(feature, mask)

        # if return_latents:
        #     return seg_image, rec_image, feature
        # else:
        #     return seg_image, rec_image