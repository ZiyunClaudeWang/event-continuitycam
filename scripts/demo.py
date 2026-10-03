"""Continuous video from one key frame and the events that follow it, with a pretrained model.

    python scripts/demo.py --ckpt continuitycam_bsergb.ckpt --seq 1_TEST/acquarium_08 --start 20 --frames 61 --out demo
    python scripts/demo.py --dataset e2d2 --ckpt continuitycam_e2d2.ckpt --seq 1_TEST/231030_160305_crossroad --start 0 --out demo_e2d2

Renders frames at t = 0 .. 1 over the window after the key frame (BS-ERGB: 6 frame intervals, so 61 frames = 10 per
interval; E2D2: 0.25 s) and writes them as PNGs and an MP4 (prediction | ground truth at the nearest earlier frame
time). For BS-ERGB the window can be squeezed with --window r (r frame intervals mapped onto t in [0, 1]).
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from continuitycam.data.batch import prepare_batch  # noqa: E402
from continuitycam.data.bsergb import BSERGBWindows, collate  # noqa: E402
from continuitycam.data.bsergb_files import SPLITS  # noqa: E402
from continuitycam.data.e2d2 import E2D2Windows  # noqa: E402
from continuitycam.models.continuitycam import load_model  # noqa: E402


def to_bgr(x):
    """[3, H, W] in [-1, 1] (RGB) -> uint8 [H, W, 3] BGR."""
    return ((x.clamp(-1, 1).permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).astype(np.uint8)[..., ::-1]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="model weights (continuitycam_bsergb.ckpt)")
    ap.add_argument("--extract", default="extract.pt", help="FILM feature-extractor weights")
    ap.add_argument("--dataset", default="bsergb", choices=["bsergb", "e2d2"])
    ap.add_argument("--root", default=None, help="dataset root (default: datasets/bs_ergb or datasets/e2d2)")
    ap.add_argument("--seq", required=True, help="e.g. 1_TEST/acquarium_08 (BS-ERGB), 1_TEST/231030_160305_crossroad (E2D2)")
    ap.add_argument("--start", type=int, default=0, help="key frame index")
    ap.add_argument("--frames", type=int, default=61, help="number of output frames over t in [0, 1]")
    ap.add_argument("--window", type=float, default=6, help="BS-ERGB: window length in frame intervals")
    ap.add_argument("--size", type=int, nargs=2, default=None, help="inference size H W (default: sensor resolution)")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--out", default="demo")
    a = ap.parse_args()

    model = load_model(os.path.expanduser(a.ckpt), os.path.expanduser(a.extract))
    split = {v: k for k, v in SPLITS.items()}[a.seq.split("/")[0]]
    root = os.path.expanduser(a.root or f"datasets/{'bs_ergb' if a.dataset == 'bsergb' else 'e2d2'}")
    if a.dataset == "e2d2":
        ds = E2D2Windows(root, split)
        ds.items = [(a.seq, a.start, list(range(1, ds.meta[a.seq][1])))]   # all frames of the window
    else:
        ds = BSERGBWindows(root, split, skip=int(min(6, a.window)), window=a.window)
        ds.items = [(a.seq, a.start)]
    raw = {k: v.cuda() if torch.is_tensor(v) else v for k, v in collate([ds[0]]).items()}
    batch = prepare_batch(raw, size=tuple(a.size) if a.size else None)

    os.makedirs(a.out, exist_ok=True)
    ts = np.linspace(0, 1, a.frames)
    gt = [to_bgr(batch["prev_image"][0])] + [to_bgr(batch["next_images"][0, j]) for j in range(batch["next_images"].shape[1])]
    gt_t = [0.0] + batch["ts"][0].tolist()
    video = None
    with torch.cuda.amp.autocast():
        enc = model.encode(batch["prev_image"], batch["flow_volumes"], batch["synth_volumes"])
        for i, t in enumerate(ts):
            pred = to_bgr(model.decode(enc, torch.full((1,), float(t), device="cuda"))["frame"][0])
            ref = gt[max(j for j, g in enumerate(gt_t) if g <= t + 1e-6)]
            frame = np.concatenate([pred, ref], axis=1)
            frame = np.pad(frame, ((0, frame.shape[0] % 2), (0, frame.shape[1] % 2), (0, 0)))  # even size for MP4
            cv2.imwrite(os.path.join(a.out, f"{i:04d}.png"), pred)
            if video is None:
                video = cv2.VideoWriter(os.path.join(a.out, "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                                        (frame.shape[1], frame.shape[0]))
            video.write(frame)
    video.release()
    print(f"wrote {a.frames} frames and demo.mp4 (prediction | ground truth at the last frame time <= t) to {a.out}")


if __name__ == "__main__":
    main()
