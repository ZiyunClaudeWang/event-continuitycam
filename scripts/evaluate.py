"""Evaluate a checkpoint on BS-ERGB with the paper's protocol (TimeLens-style key frames).

Key frames are `skip+1` frames apart; from each key frame (and the events) the model predicts the
`skip` frames in between. Metrics follow the paper's evaluation:
  - predictions clipped to [-1, 1] and converted with ((x + 1) * 127.5).astype(uint8) (truncation),
    compared against the raw BS-ERGB PNGs, both in cv2's BGR order;
  - PSNR / SSIM: skimage with data_range = gt.max() - gt.min(), SSIM with gaussian weights;
  - LPIPS: AlexNet, inputs (bgr - 128) / 128.
Reported: the mean over sequences of per-sequence means (the paper's numbers), and the mean over all frames.
Also "psnr_moving": PSNR over moving pixels only (mean over channels of |gt - key frame| > 0.1 on the [-1, 1] scale, i.e.
> 12.75 in uint8), per frame (frames with < 0.5% moving pixels are skipped), and the same for copying the key frame.

    python scripts/evaluate.py --ckpt continuitycam_bsergb.ckpt --skip 3 [--window r] [--out results.json]
    python scripts/evaluate.py --dataset e2d2 --ckpt continuitycam_e2d2.ckpt [--out results.json]

E2D2: key frames every 0.25 s, all frames in between (data/e2d2.py); the paper reports its E2D2 numbers as the mean over
all frames ("psnr_all_frames", ...).

Inference runs at the sensor resolution unless --size H W is given; predictions are always compared at full
resolution. --window r squeezes the events of r frame intervals into the model's 6 slices, so the frames in between
are queried at t = k/r instead of k/6.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from continuitycam.data.batch import prepare_batch  # noqa: E402
from continuitycam.data.bsergb import BSERGBWindows, collate  # noqa: E402
from continuitycam.data.e2d2 import E2D2Windows  # noqa: E402
from continuitycam.models.continuitycam import load_model  # noqa: E402


def to_uint8_bgr(x):
    """[3, H, W] in [-1, 1] (RGB) -> uint8 [H, W, 3] BGR (8-bit predictions, as written to PNG)."""
    img = np.clip(x.permute(1, 2, 0).float().cpu().numpy(), -1, 1)
    return ((img + 1) * 127.5).astype(np.uint8)[..., ::-1]


def psnr_ssim(args):
    gt, pred = args
    rng = gt.max() - gt.min()
    return (peak_signal_noise_ratio(gt, pred, data_range=rng),
            structural_similarity(gt, pred, data_range=rng, gaussian_weights=True, channel_axis=2))


def psnr_moving(gt, pred, key):
    """PSNR (uint8 range) over pixels where gt differs from the key frame by > 12.75 on average over channels."""
    mask = np.abs(gt.astype(np.float32) - key.astype(np.float32)).mean(-1) > 12.75
    if mask.mean() < 0.005:
        return None, None
    def p(a):
        mse = ((a.astype(np.float32) - gt.astype(np.float32)) ** 2)[mask].mean()
        return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-10)))
    return p(pred), p(key)


def lpips_alex(fn, gt, pred):
    t = lambda a: (torch.from_numpy(np.ascontiguousarray(a)).cuda().permute(2, 0, 1)[None].float() - 128) / 128
    return fn(t(gt), t(pred)).item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="model weights (continuitycam_bsergb.ckpt)")
    ap.add_argument("--extract", default="extract.pt", help="FILM feature-extractor weights")
    ap.add_argument("--dataset", default="bsergb", choices=["bsergb", "e2d2"])
    ap.add_argument("--root", default=None, help="dataset root (default: datasets/bs_ergb or datasets/e2d2)")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--window", type=float, default=6,
                    help="window length r in frame intervals (6: the paper's protocol); the frames in between sit at t = k/r")
    ap.add_argument("--size", type=int, nargs=2, default=None, help="inference size H W (default: sensor resolution)")
    ap.add_argument("--out", default=None, help="write per-frame metrics + summary JSON here")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N windows (debug)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    model = load_model(os.path.expanduser(a.ckpt), os.path.expanduser(a.extract))
    root = os.path.expanduser(a.root or f"datasets/{'bs_ergb' if a.dataset == 'bsergb' else 'e2d2'}")
    if a.dataset == "e2d2":
        ds = E2D2Windows(root, a.split)
    else:
        ds = BSERGBWindows(root, a.split, skip=a.skip, window=a.window)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=6, collate_fn=collate)
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    size = tuple(a.size) if a.size else None

    frames = []
    pending = []
    pool = ProcessPoolExecutor(a.workers)
    for i, raw in enumerate(tqdm(loader, desc=f"{a.dataset} {a.split}")):
        if a.limit is not None and i >= a.limit:
            break
        raw = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in raw.items()}
        gt_images = raw["next_images"][0]                               # [K, 3, H, W] uint8 RGB
        batch = prepare_batch(raw, size=size)
        ks = raw["k"][0].tolist()
        with torch.cuda.amp.autocast():                                 # the encoders run once per window
            enc = model.encode(batch["prev_image"], batch["flow_volumes"], batch["synth_volumes"])
            preds = [model.decode(enc, batch["ts"][:, j])["frame"] for j in range(len(ks))]
        for j, (k, pred) in enumerate(zip(ks, preds)):
            pred = pred.float()
            if pred.shape[-2:] != gt_images.shape[-2:]:
                pred = F.interpolate(pred, gt_images.shape[-2:], mode="bilinear", align_corners=False)
            gt = gt_images[j].permute(1, 2, 0).cpu().numpy()[..., ::-1].copy()
            pr = to_uint8_bgr(pred[0])
            key = raw["prev_image"][0].permute(1, 2, 0).cpu().numpy()[..., ::-1].copy()
            mov, mov_copy = psnr_moving(gt, pr, key)
            frames.append({"seq": raw["seq"][0], "frame": int(raw["start"][0]) + k, "t": float(batch["ts"][0, j]),
                           "lpips": lpips_alex(lpips_fn, gt, pr), "psnr_moving": mov, "psnr_moving_copy": mov_copy})
            pending.append(pool.submit(psnr_ssim, (gt, pr)))
    for f, fut in zip(frames, tqdm(pending, desc="psnr/ssim")):
        f["psnr"], f["ssim"] = fut.result()

    summary = {"dataset": a.dataset, "split": a.split, "skip": a.skip, "window": a.window, "ckpt": a.ckpt, "size": size,
               "num_frames": len(frames)}
    for m in ("psnr", "ssim", "lpips", "psnr_moving", "psnr_moving_copy"):
        per_seq = {}
        for f in frames:
            if f[m] is not None:
                per_seq.setdefault(f["seq"], []).append(f[m])
        summary[m] = float(np.mean([np.mean(v) for v in per_seq.values()]))
        summary[m + "_all_frames"] = float(np.mean([v for vs in per_seq.values() for v in vs]))
    print(json.dumps(summary, indent=1))
    k = "_all_frames" if a.dataset == "e2d2" else ""                # the paper's convention per dataset
    print(f"{a.dataset} {a.split}, as in the paper: PSNR {summary['psnr' + k]:.2f}  SSIM {summary['ssim' + k]:.3f}  "
          f"LPIPS {summary['lpips' + k]:.3f}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump({"summary": summary, "frames": frames}, fh)


if __name__ == "__main__":
    main()
