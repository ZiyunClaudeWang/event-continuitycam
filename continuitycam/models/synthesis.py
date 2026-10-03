"""Event-based tri-plane neural synthesis branch (Sec. 3.2, Fig. 9 of the supplement).

A U-Net maps the event volumes + initial frame to multi-scale features. At every scale the x-y plane is read
directly from the U-Net output, and the x-t and y-t planes are obtained by projecting further U-Net features
along y (for x-t) or x (for y-t) onto an explicit time axis of `time_res` samples. A frame at time t is decoded by
bilinearly sampling the three planes at (x, y, t), multiplying the samples (Hadamard product), concatenating
across scales, and running a small CNN.
"""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet


def sample_planes(planes, pts):
    """planes: per scale, [xy, xt, yt] feature maps of shape [B, D, Hs, Ws] (time along the rows of xt / yt).
    pts: [B, N, 3] coordinates (x, y, t) in [-1, 1].
    Returns [B*N, sum_s D]: per scale the product of the three plane samples, scales concatenated.
    """
    feats = []
    for scale_planes in planes:
        prod = 1.
        for plane, axes in zip(scale_planes, itertools.combinations(range(3), 2)):
            B, D = plane.shape[:2]
            coords = pts[..., axes].view(B, 1, -1, 2)
            s = F.grid_sample(plane, coords, mode="bilinear", padding_mode="border", align_corners=True)
            prod = prod * s.view(B, D, -1).transpose(-1, -2).contiguous().view(-1, D)
        feats.append(prod)
    return torch.cat(feats, dim=-1)


class AxisProjection(nn.Module):
    """Projects a feature map [B, M, h, w] along one spatial axis onto a (time, other axis) plane.

    Attention-pools the features along `pool_dim` (2: over y, giving an x-t plane; 3: over x, giving a y-t plane),
    mixes neighbouring positions along the remaining axis, and maps the channels -- where the encoder carries the
    event timing -- to `plane_dim` features at `time_res` time samples. Output [B, plane_dim, time_res, L]: rows
    are time, columns the remaining spatial axis (the layout sample_planes reads).
    """

    def __init__(self, in_channels, plane_dim, time_res, pool_dim):
        super().__init__()
        self.plane_dim, self.time_res, self.pool_dim = plane_dim, time_res, pool_dim
        self.attention = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.mix = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
                                 nn.Conv1d(in_channels, plane_dim * time_res, kernel_size=1))

    def forward(self, f):
        weights = torch.softmax(self.attention(f).float(), dim=self.pool_dim).to(f.dtype)
        pooled = (weights * f).sum(self.pool_dim)                       # [B, M, L]
        out = self.mix(pooled)
        return out.view(out.shape[0], self.plane_dim, self.time_res, out.shape[-1])


class TriPlaneSynthesisNet(nn.Module):
    """The U-Net predicts, per scale, the x-y plane (`plane_dim` channels) plus `proj_channels` features from which
    the AxisProjection heads build the x-t and y-t planes (`time_res` time samples each)."""

    def __init__(self, num_event_channels=60, plane_dim=8, num_scales=4, base_channels=128, time_res=24,
                 proj_channels=32):
        super().__init__()
        self.plane_dim = plane_dim
        self.encoder = UNet(num_event_channels + 3, plane_dim + proj_channels, num_encoders=num_scales,
                            base_num_channels=base_channels, num_residual_blocks=2)
        self.proj_xt = nn.ModuleList(AxisProjection(proj_channels, plane_dim, time_res, pool_dim=2)
                                     for _ in range(num_scales))
        self.proj_yt = nn.ModuleList(AxisProjection(proj_channels, plane_dim, time_res, pool_dim=3)
                                     for _ in range(num_scales))
        self.color_decoder = nn.Sequential(
            nn.Conv2d(plane_dim * num_scales, 64, kernel_size=3, stride=1, padding="same"), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same"), nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding="same"))

    def encode(self, volumes, prev_image):
        """volumes: [B, 60, H, W] polarity-separated event volumes; prev_image: [B, 3, H, W] in [-1, 1].
        Returns per scale [xy, xt, yt]."""
        out = self.encoder(torch.cat((volumes, prev_image), dim=1).float())
        d = self.plane_dim
        return [[p[:, :d], self.proj_xt[i](p[:, d:]), self.proj_yt[i](p[:, d:])] for i, p in enumerate(out)]

    def decode(self, planes, t, size):
        """t: [B] in [0, 1] (fraction of the window); size: output (H, W)."""
        B, (H, W) = t.shape[0], size
        dev = t.device
        x = (torch.arange(W, device=dev)[None, None, :].repeat(B, H, 1) * 1. / (W - 1)) * 2 - 1
        y = (torch.arange(H, device=dev)[None, :, None].repeat(B, 1, W) * 1. / (H - 1)) * 2 - 1
        tt = (t.float() * 2 - 1)[:, None, None].repeat(1, H, W)
        pts = torch.stack([x, y, tt], dim=-1).view(B, -1, 3)
        feats = sample_planes(planes, pts).view(B, H, W, -1).permute(0, 3, 1, 2)
        return self.color_decoder(feats)

    def forward(self, volumes, prev_image, t):
        return self.decode(self.encode(volumes, prev_image), t, prev_image.shape[-2:])
