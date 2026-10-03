"""Turns a collated batch of raw windows (on the GPU) into model inputs.

Steps: images to [-1, 1]; event volumes at full sensor resolution; bilinear resize to the working resolution.
"""
import torch
import torch.nn.functional as F

from .voxel import motion_volumes, synthesis_volumes


def _resize(x, size):
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def event_volumes(raw, H, W, size):
    """Builds each sample's volumes at sensor resolution and resizes them to `size` right away."""
    B = raw["prev_image"].shape[0]
    # collate concatenates events sample by sample, so each sample is a contiguous slice
    counts = raw["ev_counts"].tolist()
    ev = {k: raw["ev_" + k].long().split(counts) for k in ("x", "y", "t", "p", "interval")}
    synth_out, motion_out = [], []
    for b in range(B):
        e = {k: v[b] for k, v in ev.items()}
        zero = torch.zeros_like(e["x"])
        synth_out.append(_resize(synthesis_volumes(e["x"], e["y"], e["t"], e["p"], e["interval"], zero, 1, H, W), size))
        motion_out.append(_resize(motion_volumes(e["x"], e["y"], e["t"], e["interval"], zero, 1, H, W), size))
    return torch.cat(synth_out), torch.cat(motion_out)


@torch.no_grad()
def prepare_batch(raw, size=None):
    """raw: output of bsergb.collate moved to the GPU. size: (H, W) to run the model at (None = sensor resolution).
    Returns prev_image, next_images ([B, K, 3, H, W], the ground truth), synth_volumes, flow_volumes, k (target frame
    offsets) and ts (target times in [0, 1], [B, K]); images in [-1, 1]."""
    prev = raw["prev_image"].float() / 127.5 - 1.0
    nxt = raw["next_images"].float() / 127.5 - 1.0          # [B, K, 3, H, W]
    B, K, _, H, W = nxt.shape
    size = tuple(size or (H, W))
    synth, motion = event_volumes(raw, H, W, size)
    return {"prev_image": _resize(prev, size), "next_images": _resize(nxt.flatten(0, 1), size).view(B, K, 3, *size),
            "synth_volumes": synth, "flow_volumes": motion, "k": raw["k"], "ts": raw["t"].to(prev.device)}
