"""E2D2 evaluation windows (the paper's beamsplitter dataset).

Layout: <root>/meta_penn_1_TEST.txt and <root>/1_TEST/<seq>/seq.h5 with images [N, 480, 640, 3] (RGB, aligned with the
event camera), img_ts [N] (us), events x, y, t (us), p, and img2event [N] (index of the last event before each frame).
The frame rate differs between recordings (10-66 Hz), so a window spans a fixed duration: W = int(clip_time / frame
interval) + 1 frame intervals (clip_time = 0.25 s). Key frames are W frames apart; from each key frame s and the window's
events (cut into the model's 6 slices by time) the model predicts frames s+1 .. s+W-1 at t = k / W.

Samples have the layout of bsergb.BSERGBWindows (raw events; volumes are built on the GPU).
"""
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

SLICES = 6
SPLITS = {"test": "1_TEST", "val": "2_VALIDATION", "train": "3_TRAINING"}


class E2D2Windows(Dataset):
    def __init__(self, root, split="test", clip_time=0.25):
        self.root = root
        self.items, self.meta, self._h5 = [], {}, {}
        with open(os.path.join(root, f"meta_penn_{SPLITS[split]}.txt")) as f:
            seqs = [line.split()[0] for line in f if line.strip()]
        for seq in seqs:
            with h5py.File(os.path.join(root, seq, "seq.h5"), "r") as h:
                ts = h["img_ts"][:]
            n = len(ts)
            W = int(clip_time * 1e6 / ((ts[-1] - ts[0]) / (n - 1))) + 1      # frames per window
            seq_end = np.searchsorted(ts, ts[-1] - clip_time * 1e6) - 1         # last usable key frame
            self.meta[seq] = (ts, W)
            for key in range(0, seq_end, W):
                ks = [k for k in range(1, W) if key + k < seq_end]
                if ks:
                    self.items.append((seq, key, ks))

    def __len__(self):
        return len(self.items)

    def h5(self, seq):
        if seq not in self._h5:                                  # one handle per worker process
            self._h5[seq] = h5py.File(os.path.join(self.root, seq, "seq.h5"), "r")
        return self._h5[seq]

    def __getitem__(self, idx):
        seq, s, ks = self.items[idx]
        ts, W = self.meta[seq]
        h = self.h5(seq)
        r = float(W)
        # events of frames s .. s + W, fractional frame index u relative to s, cut into 6 slices by time
        e0, e1 = int(h["img2event"][s]) + 1, int(h["img2event"][min(s + W, len(ts) - 1)]) + 1
        t = h["t"][e0:e1].astype(np.int64)
        i = np.clip(np.searchsorted(ts, t, side="right") - 1, s, s + W - 1)
        u = (i - s) + np.clip((t - ts[i]) / (ts[i + 1] - ts[i]), 0.0, np.nextafter(1.0, 0.0))
        keep = (u >= 0) & (u < r)
        x = h["x"][e0:e1][keep].astype(np.int16)
        y = h["y"][e0:e1][keep].astype(np.int16)
        p = h["p"][e0:e1][keep].astype(np.int8)
        t, u = t[keep], u[keep]
        sl = np.minimum(np.floor(u * SLICES / r), SLICES - 1).astype(np.int8)
        t0 = t[0] if len(t) else 0
        images = h["images"]
        return {
            "prev_image": torch.from_numpy(images[s]).permute(2, 0, 1),
            "next_images": torch.stack([torch.from_numpy(images[s + k]).permute(2, 0, 1) for k in ks]),
            "k": torch.tensor(ks),
            "t": torch.tensor([k / r for k in ks]).float(),
            "ev_x": torch.from_numpy(x), "ev_y": torch.from_numpy(y),
            "ev_t": torch.from_numpy((t - t0).astype(np.int32)), "ev_p": torch.from_numpy(p),
            "ev_interval": torch.from_numpy(sl),
            "seq": seq, "start": s,
        }
