# RGC Receptive Fields -- Spike-Triggered Average (Phase 1)

Phase 1 of the Wandell/Chichilnisky retina project roadmap: load the
Chichilnisky/Pillow retinal ganglion cell white-noise dataset and compute
the **Spike-Triggered Average (STA)** for each recorded cell -- the
classical, simplest linear-filter model of a neuron's receptive field, and
the baseline every fancier model in later phases has to beat.

**Live interactive result:** https://retina-rgc-surrogate.vercel.app

## The dataset

`rgcData_Nature08/` (from the Dropbox link in Wandell's roadmap email):

- **`Stim_reduced.mat`** -> `Stim`, shape `(144000, 100)`. A 10x10 pixel
  random-flicker checkerboard movie, one row per stimulus frame, flattened
  spatially into 100 columns. Values are ternary (-0.5, 0, +0.5).
- **`SpTimesRGC.mat`** -> `SpTimes`, a `(1, 27)` cell array. `SpTimes[0, k]`
  is the list of spike times (in units of *stimulus frames*, not seconds)
  for RGC number k.
- **`Stim_reducedRpt.mat`** / **`MtspRGCrpt.mat`** -- the held-out 10-second
  repeat stimulus and its spike data, for Phase 3 validation (not used yet).
- **`StimCoords.mat`** -- per-cell spatial ROI metadata, not needed for
  this baseline.
- **`loadDataAndComputeSTA.m`** -- the original MATLAB reference script
  included in the dataset. Our Python code reproduces its algorithm
  exactly (verified line by line), not an approximation of it.

## Pipeline

| Script | What it does |
|---|---|
| `01_compute_sta.py` | Computes and plots the STA for one cell (default: cell 0 / MATLAB `cellnum=1`), heavily commented to explain the data format and the STA algorithm. Start here to actually understand what's happening. |
| `02_compute_all_and_export.py` | Runs the same computation for all 27 cells, saves a summary grid plot, and exports `webapp/sta_data.json` for the interactive viewer. |

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib
python3 01_compute_sta.py            # one cell, explained
python3 02_compute_all_and_export.py # all cells + web export
```

## Result

16 of 27 cells are **OFF-center** (spiking is preceded by a *decrease* in
light at one specific pixel), 11 are **ON-center** (preceded by an
*increase*) -- a normal mix for a real retinal patch. Neighboring cells'
receptive-field centers tile across the stimulus grid in a spatially
organized way, exactly like a real retinal mosaic, which is a good sanity
check that the computation is correct and not an artifact.

## The web viewer (`webapp/`)

A dependency-free static site (no build step) that lets you click through
all 27 cells and see, for each one: the spatial receptive field at any
time lag (with a slider/play button to scrub through and watch it appear
and fade), the time course at the cell's peak-response pixel, and the full
STA image. All data is precomputed by `02_compute_all_and_export.py` --
there's no live model inference here (unlike the sibling `4T_pixel_ml`
project's characterizer), since an STA is a fixed property of a recorded
cell, not something with free parameters to explore live.

Deploy: same pattern as `4T_pixel_ml/webapp` -- `cd webapp && vercel --prod`.

## What's next (from Wandell's roadmap, not started yet)

- **Phase 2:** build a predictive model on top of the STA baseline --
  either an LTSpice circuit analog of the retina (resistor grid for
  spatial filtering, RC circuits for temporal delay, an integrate-and-fire
  spike generator) or a CNN, inspired by Baccus/Ganguli's DeepRetina work.
- **Phase 3:** validate whichever model against the held-out 10-second
  repeat stimulus, comparing predicted spikes to the real PSTH
  (peri-stimulus time histogram) averaged over the 600 repeats.
