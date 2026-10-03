"""Image-feature pyramid and multi-scale fusion decoder, adapted from FILM (Reda et al., ECCV 2022).

The feature extractor is used frozen with FILM's pretrained weights (`extract.pt`).
"""
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from . import softsplat


def conv(in_channels, out_channels, size, activation="relu"):
    _conv = nn.Conv2d(in_channels, out_channels, kernel_size=size, padding="same")
    if activation is None:
        return _conv
    return nn.Sequential(_conv, nn.LeakyReLU(0.2))


def build_image_pyramid(image: torch.Tensor, levels: int) -> List[torch.Tensor]:
    """Finest level first; each level is a 2x2 average pool of the previous one."""
    pyramid = [image]
    for _ in range(levels - 1):
        image = F.avg_pool2d(image, 2, 2)
        pyramid.append(image)
    return pyramid


def concatenate_pyramids(*pyramids: List[torch.Tensor]) -> List[torch.Tensor]:
    """Channel-concatenates pyramids level by level (truncated to the shortest)."""
    return [torch.cat(levels, dim=1) for levels in zip(*pyramids)]


def splat_pyramid(pyramid: List[torch.Tensor], flows: List[torch.Tensor],
                  metrics: List[torch.Tensor]) -> List[torch.Tensor]:
    """Softmax-splats every pyramid level with its flow (pixels) and splatting metric."""
    return [softsplat.softsplat(tenIn=x, tenFlow=f, tenMetric=m, strMode="soft")
            for x, f, m in zip(pyramid, flows, metrics)]


class SubTreeExtractor(nn.Module):
    def __init__(self, in_channels=3, channels=64, n_layers=4):
        super().__init__()
        convs = []
        for i in range(n_layers):
            convs.append(nn.Sequential(conv(in_channels, channels << i, 3),
                                       conv(channels << i, channels << i, 3)))
            in_channels = channels << i
        self.convs = nn.ModuleList(convs)

    def forward(self, image, n):
        head, pyramid = image, []
        for i, layer in enumerate(self.convs):
            head = layer(head)
            pyramid.append(head)
            if i < n - 1:
                head = F.avg_pool2d(head, kernel_size=2, stride=2)
        return pyramid


class FeatureExtractor(nn.Module):
    """FILM's cascaded feature pyramid: level i concatenates sub-tree features of levels <= i."""

    def __init__(self, in_channels=3, channels=64, sub_levels=4):
        super().__init__()
        self.extract_sublevels = SubTreeExtractor(in_channels, channels, sub_levels)
        self.sub_levels = sub_levels

    def forward(self, image_pyramid):
        sub_pyramids = [self.extract_sublevels(img, min(len(image_pyramid) - i, self.sub_levels))
                        for i, img in enumerate(image_pyramid)]
        feature_pyramid = []
        for i in range(len(image_pyramid)):
            features = sub_pyramids[i][0]
            for j in range(1, min(i, self.sub_levels - 1) + 1):
                features = torch.cat([features, sub_pyramids[i - j][j]], dim=1)
            feature_pyramid.append(features)
        return feature_pyramid


def _channels_at_level(level, filters, extra_channels):
    return sum(filters << i for i in range(level)) + extra_channels


class Fusion(nn.Module):
    """U-Net-decoder-style fusion of a feature pyramid into an RGB image (coarse to fine).

    `filters` must match the per-level channel count of the input pyramid: 64 for one splatted
    FILM feature pyramid, 128 for two (the event-flow and latent-flow splats).
    """

    def __init__(self, n_layers=4, specialized_layers=3, filters=64, extra_channels=0):
        super().__init__()
        self.output_conv = nn.Conv2d(filters, 3, kernel_size=1)
        self.convs = nn.ModuleList()
        in_channels = _channels_at_level(n_layers, filters, extra_channels)
        increase = 0
        for i in reversed(range(n_layers)):
            num_filters = (filters << i) if i < specialized_layers else (filters << specialized_layers)
            self.convs.append(nn.ModuleList([
                conv(in_channels, num_filters, size=2, activation=None),
                conv(in_channels + (increase or num_filters), num_filters, size=3),
                conv(num_filters, num_filters, size=3)]))
            in_channels = num_filters
            increase = _channels_at_level(i, filters, extra_channels) - num_filters // 2

    def forward(self, pyramid):
        net = pyramid[-1]
        for k, layers in enumerate(self.convs):
            i = len(self.convs) - 1 - k
            net = F.interpolate(net, size=pyramid[i].shape[2:4], mode="nearest")
            net = layers[0](net)
            net = torch.cat([pyramid[i], net], dim=1)
            net = layers[2](layers[1](net))
        return self.output_conv(net)
