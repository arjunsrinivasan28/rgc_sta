# RGC Receptive Fields -- Wandell/Chichilnisky Roadmap, Phases 1-2

**Phase 1:** load the Chichilnisky/Pillow retinal ganglion cell white-noise
dataset and compute the **Spike-Triggered Average (STA)** for each recorded
cell -- the classical, simplest linear-filter model of a neuron's
receptive field, and the baseline every fancier model has to beat.

**Phase 2:** both "choose your own adventure" paths from Wandell's email,
not just one -- **Path A** (an ngspice analog circuit: resistor grid for
spatial filtering, RC stage for temporal delay, integrate-and-fire spike
generator) and **Path B** (a CNN, DeepRetina-style), compared against the
Phase 1 STA baseline on identical held-out data.

**Live interactive result (Phase 1):** https://retina-rgc-surrogate.vercel.app

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
| `03_cnn_path_b.py` | Phase 2, Path B: trains a small CNN to predict cell 0's spike rate from a stimulus history window, and compares it against the STA baseline on identical held-out data (proper train/val/test temporal split -- test set never touched during model selection). |
| `04_path_a_ngspice_circuit.py` | Phase 2, Path A: builds and simulates a real 10x10 resistive-grid + RC + integrate-and-fire ngspice circuit driven by real stimulus data, and compares its spike output against the same real cell. |

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib torch
python3 01_compute_sta.py               # Phase 1: one cell, explained
python3 02_compute_all_and_export.py    # Phase 1: all cells + web export
python3 03_cnn_path_b.py                # Phase 2, Path B: CNN vs STA
python3 04_path_a_ngspice_circuit.py    # Phase 2, Path A: circuit vs real cell
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

## Phase 2, Path B: CNN

A small CNN (2 conv layers over the 10x10 grid, history frames as
channels, matching the Phase 1 STA's 20-frame window exactly so both
models see identical information) trained with a Poisson loss to predict
cell 0's spike count. Compared against the STA baseline on an *identical*
held-out temporal test split, with model selection done on a separate
validation block (never the test set):

| Model | Test-set correlation (predicted vs. actual spike count) |
|---|---|
| STA (Phase 1 baseline) | 0.570 |
| CNN (Path B) | **0.687** |

A real, honest improvement (+0.12) -- not by a huge margin, which makes
sense: this is still a single cell with a fairly clean, close-to-linear
receptive field, so a linear filter was never going to be far off. See
`plots/path_b_cnn_vs_sta.png`. One real bug worth knowing about: an
earlier version of this CNN used global average pooling before the dense
layers, which collapses the entire spatial map and destroys exactly the
"where in space" information a receptive field is defined by -- it scored
*worse* than the STA (0.14) until that was fixed to flatten the full
spatial map instead.

## Phase 2, Path A: ngspice circuit

A real 3-stage retina circuit -- resistive-grid spatial (center-surround)
filtering, an RC temporal low-pass, and a smooth-comparator
integrate-and-fire spike generator -- built and simulated in ngspice,
driven by the same real stimulus data as Path B, and compared against the
same real cell's recorded spikes. See `04_path_a_ngspice_circuit.py`'s
module docstring and inline comments for the full design.

To keep the simulation tractable, the resistive grid is built only in a
5x5 window around the cell's real receptive-field center (from Phase 1's
STA), not the full 10x10 sensor -- a real center-surround response is
spatially local and a resistor grid's influence falls off with distance,
so this is a physically justified cut. The window is simulated for 100
frames (the window still contains 25 real spikes to compare against).

**Result:** 45 circuit-generated spikes vs. 25 real spikes in the same
window, with rough temporal correspondence between the two rasters (see
`plots/path_a_circuit_result.png`) -- not a precise match (this is a
hand-tuned toy circuit with no fitting to the real cell, unlike Path B),
but a genuinely functioning 3-stage circuit rather than noise. Some
bursty/chattery clustering remains when the membrane hovers right at
threshold; a real fix would need proper refractory-period tuning, which
was out of scope here.

Several real, non-obvious bugs surfaced and got fixed along the way
(documented in the script's comments, not hidden):

1. This ngspice build silently fails to evaluate the transient waveform of
   whichever element is listed *first* in the netlist -- fixed with a
   harmless dummy element first.
2. `PWL FILE=` (external file) breaks with more than one such source in
   the same netlist -- fixed by embedding every pixel's stimulus as inline
   PWL breakpoints instead.
3. A steep (gain-200) `tanh` comparator made the local-truncation-error
   estimator collapse the timestep near every threshold approach --
   measured directly at 525,632 internal steps for what should have been
   ~80. Lowering the gain to 20 fixed this (15x faster) while staying a
   sharp, effectively-binary comparator.
4. The reset switch's threshold was calibrated against the wrong voltage
   scale (`V_THRESH/2`, meant for the ~0.1V membrane, applied to a signal
   that swings to 3.3V) -- this clamped the membrane below the real
   threshold and produced zero spikes, fixed by calibrating against
   `V_SPIKE/2` instead.
5. After fixing (4), the membrane's own RC recovery time constant (~100us)
   was far shorter than a real inter-spike interval (~30ms), so it
   re-crossed threshold within microseconds of every reset and chattered
   at its maximum rate (159,982 "spikes" in one run). Raising the membrane
   capacitance so the natural recovery time constant is tens of
   milliseconds fixed this.
6. Even then, the center-surround gain (20x) amplified a small
   finite-sample DC bias in a 100-frame random window (the photoreceptor's
   own sample mean isn't exactly 0) into a permanent multi-volt offset
   that pinned the membrane above threshold almost continuously. Lowering
   the gain to 4x brought the circuit into its intended regime.

## What's next (Phase 3, from Wandell's roadmap, not started yet)

Validate whichever Phase 2 model(s) against the held-out 10-second repeat
stimulus, comparing predicted spikes to the real PSTH (peri-stimulus time
histogram) averaged over the 600 repeats -- the actual generalization test
neither Path A nor Path B has been through yet (both were only checked
against held-out *frames* of the same recording, not the separate repeat
dataset Wandell's email specifically calls out for this purpose).
