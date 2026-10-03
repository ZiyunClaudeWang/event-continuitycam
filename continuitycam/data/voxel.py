"""GPU event voxelization for a batch of windows (6 intervals each).

Two volumes per interval (Sec. 7.3 of the supplement):

  synthesis volume: 5 time bins of positive events (>= 0) followed by 5 bins of negative events
      (<= 0), bilinear in time, not normalized. Formerly e2vid's `events_to_voxel_grid`, applied to
      each polarity separately; time runs from the first to the last event of that polarity.
  motion volume: 10 time bins of unsigned event counts, bilinear in time, then scaled so the 98th
      percentile of non-zero values is 1 and clipped. Formerly `gen_discretized_event_volume_no_polarity`
      + `normalize_event_volume`.

Channel layout of both outputs: interval-major, i.e. channel = interval * 10 + bin.

Time bins: each interval's bins span its first to last event -- per polarity for the synthesis
volume. Pixel coordinates are truncated to integers (no spatial
interpolation).
"""
import torch

NUM_INTERVALS = 6
POL_BINS = 5
FLOW_BINS = 10


def _group_min_max(t, group, n_groups):
    """Per-group min and max of timestamps (float64, exact for microseconds). Empty groups get
    (inf, -inf). Uses one stable sort instead of scatter_reduce, whose atomics serialize badly when
    millions of events share a handful of groups."""
    t = t.double()
    if t.numel() == 0:
        return (torch.full((n_groups,), float("inf"), device=t.device, dtype=t.dtype),
                torch.full((n_groups,), float("-inf"), device=t.device, dtype=t.dtype))
    order = torch.argsort(group * (t.max() - t.min() + 1) + (t - t.min()))
    g_sorted, t_sorted = group[order], t[order]
    ids = torch.arange(n_groups, device=t.device)
    starts = torch.searchsorted(g_sorted, ids)
    ends = torch.searchsorted(g_sorted, ids, right=True)
    nonempty = ends > starts
    last = t.numel() - 1
    lo = torch.where(nonempty, t_sorted[starts.clamp(max=last)], torch.full_like(t_sorted[:1], float("inf")))
    hi = torch.where(nonempty, t_sorted[(ends - 1).clamp(min=0)], torch.full_like(t_sorted[:1], float("-inf")))
    return lo, hi


def _upper_percentile(values, q=0.98):
    """The normalization bound: the k-th smallest non-zero value, k = max(int(q * n), 1). Sorting is much faster on the GPU than kthvalue on one
    long 1-D tensor."""
    nz = values[values != 0]
    if nz.numel() == 0:
        return None
    k = max(int(q * nz.numel()), 1)
    return torch.sort(nz)[0][k - 1]


def synthesis_volumes(x, y, t, p, interval, sample, B, H, W):
    """x, y, t (us), p (0/1), interval (0..5), sample (0..B-1): 1-D int64 tensors on one device.
    Returns [B, 60, H, W] float32."""
    group = (sample * NUM_INTERVALS + interval) * 2 + p
    t_first, t_last = _group_min_max(t, group, B * NUM_INTERVALS * 2)
    start, dt = t_first[group], (t_last - t_first)[group].double()
    dt[dt == 0] = 1.0
    ts = ((POL_BINS - 1) * (t.double() - start) / dt).clamp(0, POL_BINS - 1)
    ti = ts.floor().long()
    frac = ts - ti
    pol = p.double() * 2 - 1
    channel = interval * 10 + (1 - p) * POL_BINS            # positives first, then negatives
    base = ((sample * NUM_INTERVALS * 10 + channel) * H + y) * W + x
    vol = torch.zeros(B * NUM_INTERVALS * 10 * H * W, device=x.device, dtype=torch.float32)
    for bins, val in ((ti, pol * (1.0 - frac)), (ti + 1, pol * frac)):
        ok = bins < POL_BINS
        vol.index_put_((base[ok] + bins[ok] * H * W,), val[ok].float(), accumulate=True)
    return vol.view(B, NUM_INTERVALS * 10, H, W)


def motion_volumes(x, y, t, interval, sample, B, H, W):
    """Unsigned 10-bin volumes, normalized per interval. Returns [B, 60, H, W] float32.

    Time is scaled as (t - t_min) * 9 / (t_max - t_min + 1e-6) with t in units of 1e9 us, i.e. with an epsilon of
    1 ms (~3% of a BS-ERGB interval).
    """
    group = sample * NUM_INTERVALS + interval
    t_min, t_max = _group_min_max(t, group, B * NUM_INTERVALS)
    tt = t.double() / 1e9
    lo, hi = t_min[group] / 1e9, t_max[group] / 1e9
    ts = (tt - lo) * ((FLOW_BINS - 1) / (hi - lo + 1e-6))
    # bilinear in time
    t_fl = torch.floor(ts)
    frac = ts - t_fl
    t_fl = t_fl.long()
    base = ((sample * NUM_INTERVALS * FLOW_BINS + interval * FLOW_BINS) * H + y) * W + x
    vol = torch.zeros(B * NUM_INTERVALS * FLOW_BINS * H * W, device=x.device, dtype=torch.float64)
    vol.index_put_((base + t_fl * H * W,), 1.0 - frac, accumulate=True)
    ok = t_fl + 1 < FLOW_BINS
    vol.index_put_((base[ok] + (t_fl[ok] + 1) * H * W,), frac[ok], accumulate=True)
    vol = vol.view(B * NUM_INTERVALS, FLOW_BINS * H * W)
    for v in vol:  # normalize each interval volume (in place)
        # m = max(|2nd percentile|, 98th percentile) of the non-zero values; the volume is non-negative,
        # so m is the 98th percentile.
        m = _upper_percentile(v)
        if m is not None:
            v.clamp_(-m, m).div_(m + 1e-5)
    return vol.view(B, NUM_INTERVALS * FLOW_BINS, H, W).float()
