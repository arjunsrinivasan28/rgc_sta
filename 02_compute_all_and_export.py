"""
Phase 1, all cells -- compute the STA for every one of the 27 recorded
RGCs (not just cell 0), then export everything the web viewer needs
(rgc_sta_data.json) so the browser can display any cell's receptive field
interactively without needing to redo the computation client-side.

Run this after 01_compute_sta.py (which explains the algorithm in detail
with comments -- this script reuses the exact same math, just looped
over every cell instead of one).
"""

import json
import warnings
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "rgcData_Nature08"
PLOTS_DIR = HERE / "plots"
WEBAPP_DIR = HERE / "webapp"

N_LAGS = 20
NX, NY = 10, 10


def bin_spikes_to_frames(spike_times, n_frames):
    bin_edges = np.arange(n_frames + 1)
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    return counts.astype(np.float64)


def compute_sta(stim, spike_counts, n_lags):
    n_frames, n_pixels = stim.shape
    sta = np.zeros((n_lags, n_pixels))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for lag in range(n_lags):
            row = n_lags - 1 - lag
            sta[row, :] = spike_counts[lag:] @ stim[: n_frames - lag, :]
    sta /= spike_counts.sum()
    assert not np.isnan(sta).any()
    return sta


def main():
    stim = sio.loadmat(DATA_DIR / "Stim_reduced.mat")["Stim"]
    sptimes_cell = sio.loadmat(DATA_DIR / "SpTimesRGC.mat")["SpTimes"]
    n_frames, n_pixels = stim.shape
    n_cells = sptimes_cell.shape[1]

    PLOTS_DIR.mkdir(exist_ok=True)
    WEBAPP_DIR.mkdir(exist_ok=True)

    all_cells = []
    print(f"Computing STA for all {n_cells} cells...")
    for i in range(n_cells):
        spike_times = sptimes_cell[0, i].flatten()
        spike_counts = bin_spikes_to_frames(spike_times, n_frames)
        sta = compute_sta(stim, spike_counts, N_LAGS)

        peak_flat = np.argmax(np.abs(sta))
        peak_lag, peak_pixel = np.unravel_index(peak_flat, sta.shape)
        peak_value = sta[peak_lag, peak_pixel]
        polarity = "OFF-center" if peak_value < 0 else "ON-center"

        all_cells.append({
            "cell_index": i,
            "matlab_cellnum": i + 1,
            "n_spikes": int(len(spike_times)),
            "sta": np.round(sta, 6).tolist(),  # (n_lags, n_pixels)
            "peak_lag": int(peak_lag),
            "peak_pixel": int(peak_pixel),
            "peak_value": float(peak_value),
            "polarity": polarity,
        })
        print(f"  cell {i:2d} (MATLAB #{i+1:2d}): {len(spike_times):6d} spikes, "
              f"{polarity}, peak={peak_value:+.4f} at lag={peak_lag}, pixel={peak_pixel}")

    n_off = sum(1 for c in all_cells if c["polarity"] == "OFF-center")
    n_on = n_cells - n_off
    print(f"\n{n_on} ON-center, {n_off} OFF-center (of {n_cells} total)")

    # --- Export everything the web viewer needs ---
    export = {
        "n_cells": n_cells,
        "n_lags": N_LAGS,
        "nx": NX,
        "ny": NY,
        "n_frames": int(n_frames),
        "n_pixels": int(n_pixels),
        "cells": all_cells,
    }
    out_path = WEBAPP_DIR / "sta_data.json"
    with open(out_path, "w") as f:
        json.dump(export, f)
    print(f"\nSaved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

    # --- Also save one static summary figure: every cell's peak spatial map ---
    fig, axes = plt.subplots(4, 7, figsize=(16, 10))
    for i, ax in enumerate(axes.flat):
        if i >= n_cells:
            ax.axis("off")
            continue
        c = all_cells[i]
        sta = np.array(c["sta"])
        vmax = np.abs(sta).max()
        ax.imshow(sta[c["peak_lag"], :].reshape(NX, NY), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"#{c['matlab_cellnum']} ({c['polarity'][:3]})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Peak-lag spatial receptive field, all 27 RGCs")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "all_cells_grid.png", dpi=130)
    print(f"Saved {PLOTS_DIR / 'all_cells_grid.png'}")


if __name__ == "__main__":
    main()
