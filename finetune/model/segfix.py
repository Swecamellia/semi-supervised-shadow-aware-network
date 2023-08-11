from json import decoder
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union, List
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead, ClassificationHead
from segmentation_models_pytorch.base import initialization as init
from utils.offset import DTOffsetConfig, DTOffsetHelper

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md


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
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
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

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

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

    def forward(self, features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        decoder_features = []
        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            decoder_features.append(x)

        return x, decoder_features



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
        feats = F.grid_sample(x, grid, padding_mode='border', align_corners=True)                       # (N, C, H, W)
        
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
        mask_map = self.mask_head(feats)
        dir_map = self.dir_head(feats)
        return mask_map, dir_map

class segfix(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder1 = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head1 = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )


        self.boundary_head = boundary_refine(sum(decoder_channels))

        self.refiner = OffsetModule()

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder1)
        init.initialize_head(self.segmentation_head1)

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

    def forward(self, x, noise):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output, features = self.decoder1(features)

        masks = self.segmentation_head1(decoder_output)

        bdr, dir = self.boundary_head(features)

        return masks, bdr, dir
