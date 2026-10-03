"""BS-ERGB evaluation windows (the TimeLens protocol).

Key frames are skip+1 frames apart. A window starts at key frame s and holds the events of the next 6 frame
intervals, one per input slice; the model predicts frames s+1 .. s+skip at t = k/6.

With window r (r frame intervals, possibly fractional), the events of r intervals are cut into the 6 slices by time
(each slice r/6 frame intervals long), so frame s+k sits at t = k/r. Time is measured in frame intervals, linear
between frame timestamps; r = 6 is the standard window.

Samples hold raw events; event volumes are built on the GPU (see voxel.py / batch.py).
"""
import math

import numpy as np
import torch
from torch.utils.data import Dataset

from .bsergb_files import SPLITS, WINDOW, list_sequences, num_frames, num_windows, read_events, read_image, read_timestamps

SLICES = WINDOW - 1


class BSERGBWindows(Dataset):
    def __init__(self, root, split="test", skip=3, window=6):
        """split: "test", "val" or "train" (BS-ERGB's 1_TEST, 2_VALIDATION, 3_TRAINING); skip: frames between key frames;
        window: window length r in frame intervals (skip <= r <= 9)."""
        assert split in SPLITS and 1 <= skip <= min(window, SLICES) and window <= WINDOW + 2, (split, skip, window)
        self.root, self.skip = root, skip
        self.window = None if window == SLICES else float(window)
        self.items, self.timestamps, self.image_size = [], {}, None
        for seq in list_sequences(root, split):
            if self.window is not None:
                self.timestamps[seq] = read_timestamps(root, seq)
            self.items += [(seq, s) for s in range(num_windows(num_frames(root, seq)))[::skip + 1]]
            if self.image_size is None:
                self.image_size = read_image(root, seq, 0).shape[:2]

    def __len__(self):
        return len(self.items)

    def read_window(self, seq, s, H, W):
        """Events of the window as x, y, t, p, slice, and the time origin t0 (the first event). Standard window:
        slice = event file (frame interval). Window r: each event's fractional frame index u (relative to s, linear
        between frame times) gives slice = floor(6 u / r); events with u >= r are dropped."""
        r = self.window
        xs, ys, ts, ps, iv = [], [], [], [], []
        for i in range(SLICES if r is None else math.ceil(r)):
            x, y, t, p = read_events(self.root, seq, s + i, H, W)
            if r is None:
                sl = np.full(len(x), i, np.int8)
            else:
                frame_t = self.timestamps[seq]
                frac = (t.astype(np.int64) - frame_t[s + i]) / float(frame_t[s + i + 1] - frame_t[s + i])
                u = i + np.clip(frac, 0.0, np.nextafter(1.0, 0.0))     # in [i, i+1): events/i.npz spans [ts_i, ts_i+1)
                keep = u < r
                x, y, t, p, u = x[keep], y[keep], t[keep], p[keep], u[keep]
                sl = np.minimum(np.floor(u * SLICES / r), SLICES - 1).astype(np.int8)
            xs.append(x), ys.append(y), ts.append(t), ps.append(p), iv.append(sl)
        t0 = ts[0][0] if len(ts[0]) else np.uint32(0)
        x, y, t, p, sl = (np.concatenate(a) for a in (xs, ys, ts, ps, iv))
        return x, y, t, p, sl, t0

    def __getitem__(self, idx):
        seq, s = self.items[idx]
        H, W = self.image_size
        ks = list(range(1, self.skip + 1))
        if self.window is None:
            times = torch.tensor(ks).float() / SLICES
        else:                                                   # frame s+k is at u = k, i.e. t = k / r
            times = torch.tensor([k / self.window for k in ks]).float()
        x, y, t, p, sl, t0 = self.read_window(seq, s, H, W)
        return {
            "prev_image": torch.from_numpy(np.ascontiguousarray(read_image(self.root, seq, s))).permute(2, 0, 1),
            "next_images": torch.stack([torch.from_numpy(np.ascontiguousarray(read_image(self.root, seq, s + k)))
                                        .permute(2, 0, 1) for k in ks]),
            "k": torch.tensor(ks),
            "t": times,                                         # target times in [0, 1]
            # int16/int32/int8 keep a window at ~10 bytes per event through the DataLoader
            "ev_x": torch.from_numpy(x.view(np.int16)),
            "ev_y": torch.from_numpy(y.view(np.int16)),
            "ev_t": torch.from_numpy((t - t0).view(np.int32)),
            "ev_p": torch.from_numpy(p.view(np.int8)),
            "ev_interval": torch.from_numpy(sl),
            "seq": seq,
            "start": s,
        }


def collate(samples):
    """Stacks images; concatenates events and records which sample each event belongs to."""
    out = {"seq": [s["seq"] for s in samples], "start": torch.tensor([s["start"] for s in samples])}
    for key in ("prev_image", "next_images", "k", "t"):
        out[key] = torch.stack([s[key] for s in samples])
    for key in ("ev_x", "ev_y", "ev_t", "ev_p", "ev_interval"):
        out[key] = torch.cat([s[key] for s in samples])
    out["ev_counts"] = torch.tensor([len(s["ev_x"]) for s in samples])
    return out
