"""U-Net backbone of both the motion and the synthesis branch (Fig. 8 of the supplement): 4 encoders,
2 residual bottleneck blocks and 4 decoders with skip connections, with a prediction at every decoder scale.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels):
    return nn.GroupNorm(32 if channels % 32 == 0 else 8, channels)


class PreActResBlock(nn.Module):
    """GN-SiLU-conv3x3-GN-SiLU-conv3x3 plus a (1x1 if needed) shortcut: the standard U-Net double conv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.shortcut(x) + h


class UNet(nn.Module):
    """`forward` returns `num_encoders` predictions, coarsest first, at 1/2**(n-1), ..., 1/2, 1 of the input size.

    A full-resolution stem whose features are skipped to the finest decoder level; one residual double-conv block
    per level on both sides; downsampling by stride-2 conv; bilinear upsampling before concatenating the
    same-resolution skip; GroupNorm (independent of the batch size, so the full model trains at batch 1 and
    evaluates identically); channels capped at 4 x base. The input is padded to a multiple of 2**num_encoders and
    every prediction is cropped back to its exact scale, so any input size works.

    zero_init_heads: start every prediction at exactly 0 (e.g. zero initial flow).
    """

    def __init__(self, num_input_channels, num_output_channels, num_encoders=4,
                 base_num_channels=64, num_residual_blocks=2, zero_init_heads=False):
        super().__init__()
        self.num_encoders = n = num_encoders
        c = base_num_channels
        widths = [c // 2] + [min(c * 2 ** i, 4 * c) for i in range(n)]   # level 0 = full res, level k = 1/2**k
        self.stem = nn.Sequential(nn.Conv2d(num_input_channels, widths[0], 3, padding=1),
                                  PreActResBlock(widths[0], widths[0]))
        self.down = nn.ModuleList(nn.Conv2d(widths[k], widths[k + 1], 3, stride=2, padding=1) for k in range(n))
        self.enc_blocks = nn.ModuleList(PreActResBlock(widths[k + 1], widths[k + 1]) for k in range(n))
        self.mid = nn.ModuleList(PreActResBlock(widths[n], widths[n]) for _ in range(num_residual_blocks))
        # decoder step i goes from level n-i to level n-1-i
        self.dec_blocks = nn.ModuleList(PreActResBlock(widths[n - i] + widths[n - 1 - i], widths[n - 1 - i])
                                        for i in range(n))
        self.heads = nn.ModuleList(nn.Sequential(_group_norm(widths[n - 1 - i]), nn.SiLU(),
                                                 nn.Conv2d(widths[n - 1 - i], num_output_channels, 1))
                                   for i in range(n))
        if zero_init_heads:
            for head in self.heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def forward(self, x):
        H, W = x.shape[-2:]
        n, m = self.num_encoders, 2 ** self.num_encoders
        x = F.pad(x, (0, -W % m, 0, -H % m))
        x = self.stem(x)
        skips = [x]
        for down, block in zip(self.down, self.enc_blocks):
            x = block(down(x))
            skips.append(x)
        for block in self.mid:
            x = block(x)
        preds = []
        for i, (block, head) in enumerate(zip(self.dec_blocks, self.heads)):
            skip = skips[n - 1 - i]
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
            s = 2 ** (n - 1 - i)
            preds.append(head(x)[..., :-(-H // s), :-(-W // s)])
        return preds
