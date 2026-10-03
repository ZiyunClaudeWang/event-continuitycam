"""Continuous event-based trajectory field (Sec. 3.1).

A U-Net regresses, per pixel and scale, K motion coefficients for x and y plus a softmax-splatting
weight. A small MLP (shared across all inputs) defines K learned basis functions g_k(t), so the
displacement at any time t is sum_k w_k(u) g_k(t). The coefficients are refined coarse to fine: at each
scale they are the U-Net's prediction plus the upsampled coefficients of the coarser scale.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet


class MotionNet(nn.Module):
    def __init__(self, num_event_channels=60, num_basis=5, base_channels=64):
        super().__init__()
        self.num_basis = num_basis
        # zero-initialised heads: training starts from exactly zero flow
        self.unet = UNet(num_event_channels, num_basis * 2 + 1, num_encoders=4, base_num_channels=base_channels,
                         num_residual_blocks=2, zero_init_heads=True)
        self.basis = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                   nn.Linear(64, num_basis))
        # scale of the softmax-splatting metric: metric = sigmoid(.) * alpha
        self.alpha = nn.Parameter(torch.ones(1))

    def encode(self, volumes):
        """Time-independent part: per-scale coefficients + splatting-metric channel, and the event mask."""
        volumes = volumes.float()
        with_events = volumes.abs().sum(dim=1, keepdim=True) > 1e-4
        preds = self.unet(volumes)
        K2 = 2 * self.num_basis
        for i in range(1, len(preds)):                      # coarse-to-fine coefficients
            up = F.interpolate(preds[i - 1][:, :K2], preds[i].shape[-2:], mode="bilinear", align_corners=False)
            preds[i] = torch.cat((preds[i][:, :K2] + up, preds[i][:, K2:]), dim=1)
        return {"preds": preds, "with_events": with_events}

    def basis_at(self, t):
        """[B, 1, K] basis values at times t ([B] in [0, 1])."""
        return self.basis(t.float()[:, None, None])

    def flows_at(self, enc, t, mask_flow_with_events=False):
        """Flows (coarse to fine, displacement as a fraction of the image size) and splatting metrics at the
        same scales, at time t ([B] in [0, 1])."""
        basis = self.basis_at(t)  # [B, 1, K]
        K = self.num_basis
        flows, metrics = [], []
        for p in enc["preds"]:
            h, w = p.shape[-2:]
            metrics.append(torch.sigmoid(p[:, 2 * K:2 * K + 1]) * self.alpha)
            coeffs = p[:, :2 * K].view(p.shape[0], 2, K, h, w)
            flow = torch.cat([
                torch.sum(coeffs[:, 0][:, None] * basis[:, :, :, None, None], dim=2, keepdim=True),
                torch.sum(coeffs[:, 1][:, None] * basis[:, :, :, None, None], dim=2, keepdim=True),
            ], dim=2)[:, 0]
            if mask_flow_with_events:
                flow = (F.interpolate(enc["with_events"].float(), size=(h, w), mode="nearest") > 0.5) * flow
            flows.append(flow)
        return {"flows": flows, "metrics": metrics, "with_events": enc["with_events"]}

    def forward(self, volumes, t, mask_flow_with_events=False):
        """volumes: [B, 60, H, W] normalized event volumes; t: [B] in [0, 1]."""
        return self.flows_at(self.encode(volumes), t, mask_flow_with_events)
