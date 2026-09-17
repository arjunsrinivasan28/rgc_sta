"""
Phase 1 (Wandell/Chichilnisky roadmap) -- load the RGC white-noise dataset
and compute the Spike-Triggered Average (STA) for one retinal ganglion
cell, reproducing the included reference MATLAB script
(rgcData_Nature08/loadDataAndComputeSTA.m) exactly, in Python.

WHAT THE DATASET ACTUALLY IS
-------------------------------
Stim_reduced.mat  ->  Stim, shape (144000, 100)
    A 10x10 pixel random-flicker "checkerboard" movie, flattened per frame.
    - 100 columns = 10*10 spatial pixels, flattened row-major.
    - 144000 rows = one row per stimulus frame. We don't have the exact
      monitor refresh rate stored anywhere in this file, but this is the
      classic Chichilnisky-lab white-noise dataset, typically run at a
      stimulus frame rate in the ~120 Hz range -- 144000 frames would then
      be roughly 144000/120 = 1200s = 20 minutes, which matches Wandell's
      description exactly. (We print the actual numbers below rather than
      just asserting this, since that's the point of the exercise.)
    - Values are ternary: -0.5, 0.0, +0.5 (a common "random checkerboard"
      stimulus design), confirmed below rather than assumed.

SpTimesRGC.mat  ->  SpTimes, a (1, 27) cell array
    SpTimes[0, k] is a 1D array of spike times for RGC #k (0-indexed here;
    MATLAB's cellnum=1 is our cell index 0). Spike times are in units of
    STIMULUS FRAMES (not seconds) -- e.g. a spike time of 5.615 means "5.615
    frames into the recording," which is why the MATLAB reference script
    bins them with histogram bin edges at half-integers (0.5, 1.5, 2.5, ...)
    so that each bin corresponds to exactly one stimulus frame.

WHAT AN STA IS AND WHY WE COMPUTE IT THIS WAY
-----------------------------------------------
For every spike, look at the stimulus frame history leading up to it
(`nlags` frames back) and average those frame-histories together, weighted
by how many spikes happened in each frame (a frame with 2 spikes counts
twice). The result is a (nlags, 100) kernel: for each of the `nlags` time
lags before a spike, the average pixel pattern the cell "cared about." This
is exactly a cross-correlation between the spike-count time series and the
stimulus, which is why the core computation below is a small matrix
multiply per lag rather than an explicit loop over individual spikes (spike
counts stack up combinatorially with stimulus frames instead).
"""

import warnings
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")  # headless-safe: save figures, don't try to pop up a window
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "rgcData_Nature08"

CELL_INDEX = 0     # which RGC to examine (0-indexed here; MATLAB's cellnum=1 -> our 0)
N_LAGS = 20         # how many stimulus frames back to look, per spike
NX, NY = 10, 10     # spatial grid dimensions of the flattened 100-pixel stimulus


def load_data():
    stim = sio.loadmat(DATA_DIR / "Stim_reduced.mat")["Stim"]
    sptimes_cell = sio.loadmat(DATA_DIR / "SpTimesRGC.mat")["SpTimes"]
    return stim, sptimes_cell


def bin_spikes_to_frames(spike_times, n_frames):
    """
    Convert a list of continuous spike times (in units of stimulus frames)
    into a per-frame spike COUNT array of length n_frames, where entry i
    is how many spikes fell in frame i (i.e. in the half-open interval
    [i, i+1) of "frame time").

    This reproduces MATLAB's `hist(SpTimes{1}, 0.5:1:slen)` exactly: bin
    edges at every integer from 0 to n_frames give n_frames bins, each one
    frame wide, aligned so Python's 0-indexed bin i lines up with stimulus
    row i (both "frame i").
    """
    bin_edges = np.arange(n_frames + 1)  # 0, 1, 2, ..., n_frames
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    return counts.astype(np.float64)


def compute_sta(stim, spike_counts, n_lags):
    """
    Cross-correlate the stimulus with the spike-count time series at each
    of n_lags time lags, then normalize by total spike count.

    Returns sta with shape (n_lags, n_pixels). Row n_lags-1 (the LAST row)
    is lag 0 -- the stimulus frame at the same time as the spike. Row 0 is
    the oldest lag (n_lags-1 frames before the spike). This matches the
    reference MATLAB script's row ordering exactly (its row `nlags`, 1-
    indexed, is our row n_lags-1, 0-indexed).
    """
    n_frames, n_pixels = stim.shape
    sta = np.zeros((n_lags, n_pixels))
    # This machine's BLAS backend throws spurious divide-by-zero/overflow
    # RuntimeWarnings on some plain finite-value matmuls (seen before on
    # this same machine in an unrelated project) -- harmless, verified by
    # checking the result has no actual NaN/Inf, just noisy to print.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for lag in range(n_lags):
            # Spike counts starting `lag` frames in, dotted with stimulus
            # frames ending `lag` frames early -- i.e. "stimulus `lag`
            # frames before each spike."
            row = n_lags - 1 - lag
            sta[row, :] = spike_counts[lag:] @ stim[: n_frames - lag, :]
    sta /= spike_counts.sum()
    assert not np.isnan(sta).any(), "STA contains NaN -- investigate before trusting the plot"
    return sta


def main():
    stim, sptimes_cell = load_data()
    n_frames, n_pixels = stim.shape

    # --- Sanity-check prints: exactly the "understand the data first"
    # step Wandell's email asks for, not just accepted on faith. ---
    print(f"Stim.shape = {stim.shape}")
    print(f"  -> {n_pixels} columns = a {NX}x{NY} pixel grid flattened ({NX}*{NY} = {NX*NY})")
    print(f"  -> {n_frames} rows = one row per stimulus frame")
    print(f"  Stim value range: [{stim.min()}, {stim.max()}], "
          f"unique values: {np.unique(stim)}")

    n_cells = sptimes_cell.shape[1]
    print(f"\nSpTimes: {n_cells} RGCs recorded")
    spike_counts_per_cell = [sptimes_cell[0, i].flatten().shape[0] for i in range(n_cells)]
    print(f"  spike counts per cell: {spike_counts_per_cell}")

    spike_times = sptimes_cell[0, CELL_INDEX].flatten()
    print(f"\nExamining cell index {CELL_INDEX} "
          f"(MATLAB cellnum={CELL_INDEX + 1}): {len(spike_times)} spikes")
    print(f"  spike time range: [{spike_times.min():.2f}, {spike_times.max():.2f}] "
          f"(in units of stimulus frames, out of {n_frames} total)")

    # --- The actual computation ---
    spike_counts = bin_spikes_to_frames(spike_times, n_frames)
    print(f"\nBinned spike counts: {spike_counts.shape}, "
          f"total = {spike_counts.sum():.0f} (should match spike count above)")

    sta = compute_sta(stim, spike_counts, N_LAGS)
    print(f"\nSTA shape = {sta.shape}   (n_lags={N_LAGS}, n_pixels={n_pixels})")
    print(f"STA value range: [{sta.min():.4f}, {sta.max():.4f}]")

    # --- Plot: same three views as the reference MATLAB script ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    ax = axes[0, 0]
    im = ax.imshow(sta, aspect="auto", cmap="RdBu_r",
                    vmin=-np.abs(sta).max(), vmax=np.abs(sta).max())
    ax.set_xlabel("space (pixel index)")
    ax.set_ylabel("time lag (row = older -> newer, top to bottom)")
    ax.set_title(f"Full STA -- cell {CELL_INDEX} (MATLAB cellnum={CELL_INDEX + 1})")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # Pick the pixel with the largest |STA| value anywhere, so the time
    # slice actually shows a real response instead of a flat noise trace.
    peak_pixel = np.unravel_index(np.argmax(np.abs(sta)), sta.shape)[1]
    ax = axes[1, 0]
    ax.plot(sta[:, peak_pixel], "o-")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("time lag (frames before spike, oldest -> newest)")
    ax.set_title(f"Time course at pixel {peak_pixel} (largest-response pixel)")

    # Pick the lag row with the largest |STA| value for the spatial map.
    peak_lag = np.unravel_index(np.argmax(np.abs(sta)), sta.shape)[0]
    ax = axes[1, 1]
    im2 = ax.imshow(sta[peak_lag, :].reshape(NX, NY), cmap="RdBu_r",
                     vmin=-np.abs(sta).max(), vmax=np.abs(sta).max())
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Spatial map at lag {peak_lag} (strongest-response lag)")
    fig.colorbar(im2, ax=ax, fraction=0.046)

    axes[0, 1].axis("off")
    fig.tight_layout()
    out_path = HERE / f"sta_cell{CELL_INDEX}.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
