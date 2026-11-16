"""
RRDBNet - the CNN architecture used by Real-ESRGAN (and ESRGAN).

This is a clean, dependency-free PyTorch re-implementation of the generator
network. Layer names match the official pretrained checkpoints
(RealESRGAN_x2plus.pth / RealESRGAN_x4plus.pth), so the official weights load
directly with `load_state_dict(..., strict=True)`.

High-level data flow
--------------------
    low-res image
        -> conv_first                    (shallow feature extraction)
        -> 23 x RRDB blocks              (deep feature extraction: residual-in-residual dense blocks)
        -> conv_body + skip connection   (global residual connection)
        -> upsample x2 (nearest) + conv  (two upsampling stages,
        -> upsample x2 (nearest) + conv   = 4x in total)
        -> conv_hr -> conv_last          (reconstruction to RGB)
        -> high-res image

For the 2x model the input is first rearranged with pixel-unshuffle
(space-to-depth: 3 channels -> 12 channels, half the width/height). The same
network then runs and its fixed 4x upsampling gives a net 2x result.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """5 densely-connected conv layers; every layer sees ALL previous feature maps."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Residual scaling (0.2) stabilises training of very deep networks.
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block = 3 residual dense blocks + a skip connection."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Generator network of ESRGAN / Real-ESRGAN.

    Args:
        num_in_ch:   input channels (3 for RGB)
        num_out_ch:  output channels (3 for RGB)
        scale:       2 or 4 (see module docstring)
        num_feat:    channel width of the feature maps
        num_block:   number of RRDB blocks (23 for the 'plus' models)
        num_grow_ch: growth channels inside each dense block
    """

    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, scale: int = 4,
                 num_feat: int = 64, num_block: int = 23, num_grow_ch: int = 32):
        super().__init__()
        if scale not in (2, 4):
            raise ValueError("RRDBNet here supports scale 2 or 4 only.")
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4  # pixel-unshuffle(2) turns 3 channels into 12

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsampling stages
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            # input height/width must be even (the caller pads if needed)
            feat = F.pixel_unshuffle(x, downscale_factor=2)
        else:
            feat = x
        feat = self.conv_first(feat)                 # shallow features
        body_feat = self.conv_body(self.body(feat))  # deep features
        feat = feat + body_feat                      # global residual connection
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))
