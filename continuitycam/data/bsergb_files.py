"""Reading the raw BS-ERGB release (TimeLens++, Tulyakov et al. CVPR 2022).

Layout: <root>/<split>/<seq>/images/NNNNNN.png (+ timestamp.txt) and <seq>/events/NNNNNN.npz, where
events/i.npz holds the events between images i and i+1: x, y (uint16, in 1/32 pixel), timestamp
(uint32, us), polarity (uint8). Coordinates are absolute in the image frame; a few junk events lie
outside it (e.g. x ~ 65535) and are dropped.
"""
import os

import cv2
import numpy as np

SPLITS = {"test": "1_TEST", "val": "2_VALIDATION", "train": "3_TRAINING"}
COORD_SCALE = 32
WINDOW = 7  # frames per training/eval window: the initial frame + 6 intervals of events


def num_windows(n_frames):
    """Window starts are 0 .. n-10; evaluation windows are every test_skip+1-th start (the paper's protocol)."""
    return max(n_frames - WINDOW - 2, 0)


def list_sequences(root, split):
    with open(os.path.join(root, f"meta_timelens_{SPLITS[split]}.txt")) as f:
        return [line.split()[0] for line in f if line.strip()]


def num_frames(root, seq):
    return len([f for f in os.listdir(os.path.join(root, seq, "images")) if f.endswith(".png")])


def read_timestamps(root, seq):
    """Frame timestamps (us, same clock as the events); events/i.npz spans [ts[i], ts[i+1])."""
    return np.loadtxt(os.path.join(root, seq, "images", "timestamp.txt")).astype(np.int64)


def read_image(root, seq, idx):
    """RGB uint8 [H, W, 3]."""
    img = cv2.imread(os.path.join(root, seq, "images", f"{idx:06d}.png"), cv2.IMREAD_COLOR)
    return img[:, :, ::-1]


def read_events(root, seq, interval, height, width):
    """Events between images `interval` and `interval+1`: x_px, y_px (uint16, truncated to whole
    pixels), t (uint32, us) and p (uint8, 0/1), in the dataset's own (frame-aligned) coordinates."""
    ev = np.load(os.path.join(root, seq, "events", f"{interval:06d}.npz"))
    x, y, t, p = ev["x"], ev["y"], ev["timestamp"], ev["polarity"]
    # integer-only equivalent of 0 <= raw/32 < size and floor(raw/32)
    keep = (x < width * COORD_SCALE) & (y < height * COORD_SCALE)
    return x[keep] >> 5, y[keep] >> 5, t[keep], p[keep]
