"""
Phase 2, Path B (deep learning) -- a small CNN that maps a short history of
the 10x10 stimulus to a predicted spike rate for one RGC, trained on the
REAL recorded spikes, and compared honestly against the Phase 1 STA
(linear-filter) baseline on held-out data.

WHY A CNN HERE, CONCRETELY
----------------------------
The STA (Phase 1) is a single fixed linear kernel: predicted response =
(stimulus history) . (STA weights). That's a strong, interpretable
baseline, but it can only capture a LINEAR relationship between stimulus
and spike rate. A CNN with a nonlinearity after the convolution can, in
principle, capture things a pure linear filter structurally cannot:
contrast normalization, non-monotonic tuning, and interactions between
space and time that don't factor into one fixed kernel. This is exactly
the "Path B" DeepRetina-style approach Wandell's email describes.

INPUT / OUTPUT
-----------------
Input:  the last N_HIST stimulus frames (each 10x10), i.e. a
        (N_HIST, 10, 10) tensor -- the same time window the STA looked at
        (N_HIST = 20, matching Phase 1's n_lags exactly, so the two
        models see literally the same information).
Output: predicted spike COUNT in the current frame (a Poisson rate).

TRAIN/TEST SPLIT
-------------------
A temporal split (first 80% of frames for training, last 20% held out) --
NOT a random split. Spikes in adjacent frames are highly correlated (a
neuron's own recent spiking predicts its near-future spiking to some
degree, and slow drifts in overall excitability exist), so a random
frame-by-frame split would leak information between train and test in a
way a temporal split doesn't.
"""

import warnings
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "rgcData_Nature08"
PLOTS_DIR = HERE / "plots"

CELL_INDEX = 0       # same cell as Phase 1's worked example, for direct comparison
N_HIST = 20           # frames of stimulus history per prediction, matches Phase 1's n_lags
NX, NY = 10, 10
TRAIN_FRACTION = 0.7  # temporal blocks: 70% train, 10% validation, 20% test
VAL_FRACTION = 0.1
BATCH_SIZE = 256
N_EPOCHS = 30
LEARNING_RATE = 1e-3

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def bin_spikes_to_frames(spike_times, n_frames):
    bin_edges = np.arange(n_frames + 1)
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    return counts.astype(np.float32)


def build_dataset(stim, spike_counts, n_hist):
    """
    Turn the (n_frames, 100) stimulus and (n_frames,) spike-count arrays
    into (n_frames - n_hist, n_hist, NX, NY) input windows and
    (n_frames - n_hist,) target spike counts -- one training example per
    frame (after the first n_hist frames, which don't have enough history).
    """
    n_frames = stim.shape[0]
    stim_grid = stim.reshape(n_frames, NX, NY)
    n_examples = n_frames - n_hist
    X = np.zeros((n_examples, n_hist, NX, NY), dtype=np.float32)
    for i in range(n_examples):
        X[i] = stim_grid[i : i + n_hist]
    y = spike_counts[n_hist:]
    return X, y


class SpikeCNN(nn.Module):
    """
    Small 3D-ish CNN: a couple of spatial conv layers applied per history
    frame (via treating history as channels, since our spatial grid is
    tiny -- 10x10 -- so full 3D convolution would be overkill), then a
    small dense head predicting one scalar (log spike rate) per example.
    """

    def __init__(self, n_hist, nx=10, ny=10):
        super().__init__()
        self.conv1 = nn.Conv2d(n_hist, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        # NOTE: earlier version used AdaptiveAvgPool2d(1) here, which
        # collapses the entire spatial map to one value per channel before
        # the dense layers. That structurally throws away WHERE in space
        # the response is -- exactly the information a receptive field is
        # defined by -- which is why that version badly underperformed the
        # STA (0.14 vs 0.57 correlation). Flattening the full spatial map
        # instead lets the network learn a location, the same information
        # the STA has direct access to.
        self.fc1 = nn.Linear(8 * nx * ny, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = x.flatten(1)
        x = self.act(self.fc1(x))
        log_rate = self.fc2(x).squeeze(-1)
        return log_rate  # log of predicted Poisson rate -- see PoissonNLLLoss(log_input=True)


def sta_baseline_predictions(X_train, y_train, X_test, n_hist):
    """
    The Phase 1 STA, reimplemented here in the SAME train/test split as
    the CNN, so the two models are compared honestly on identical held-out
    data -- not against the Phase 1 script's different (full-dataset) STA.
    """
    n_pixels = NX * NY
    X_train_flat = X_train.reshape(X_train.shape[0], n_hist, n_pixels)
    # STA: for each lag, (spike count) . (stimulus) summed over training examples.
    # Equivalent to correlating the RAW stimulus/spike-count sequences,
    # restricted to the training portion -- reproducing 01_compute_sta.py's
    # algorithm exactly, just on a prefix of the data instead of all of it.
    sta = np.einsum("n,nlp->lp", y_train, X_train_flat) / max(y_train.sum(), 1e-9)

    X_test_flat = X_test.reshape(X_test.shape[0], n_hist, n_pixels)
    pred = np.einsum("nlp,lp->n", X_test_flat, sta)
    return pred, sta


def main():
    print(f"Using device: {DEVICE}")

    stim = sio.loadmat(DATA_DIR / "Stim_reduced.mat")["Stim"]
    sptimes_cell = sio.loadmat(DATA_DIR / "SpTimesRGC.mat")["SpTimes"]
    n_frames = stim.shape[0]
    spike_times = sptimes_cell[0, CELL_INDEX].flatten()
    spike_counts = bin_spikes_to_frames(spike_times, n_frames)

    X, y = build_dataset(stim, spike_counts, N_HIST)
    print(f"Dataset: X={X.shape}, y={y.shape} (mean spike count/frame = {y.mean():.4f})")

    n_train = int(len(X) * TRAIN_FRACTION)
    n_val = int(len(X) * VAL_FRACTION)
    X_train, X_val, X_test = X[:n_train], X[n_train : n_train + n_val], X[n_train + n_val :]
    y_train, y_val, y_test = y[:n_train], y[n_train : n_train + n_val], y[n_train + n_val :]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} examples "
          f"(temporal blocks, in that order)")

    # --- STA baseline, same split ---
    sta_pred, sta_kernel = sta_baseline_predictions(X_train, y_train, X_test, N_HIST)
    sta_corr = np.corrcoef(sta_pred, y_test)[0, 1]
    print(f"\nSTA baseline: test-set correlation (predicted vs actual spike count) = {sta_corr:.4f}")

    # --- CNN ---
    X_train_t = torch.from_numpy(X_train).to(DEVICE)
    y_train_t = torch.from_numpy(y_train).to(DEVICE)
    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    X_test_t = torch.from_numpy(X_test).to(DEVICE)

    model = SpikeCNN(N_HIST).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.PoissonNLLLoss(log_input=True)

    n_train_examples = X_train_t.shape[0]
    best_val_corr = -np.inf
    best_state = None
    print(f"\nTraining CNN for up to {N_EPOCHS} epochs "
          f"(checkpointing on VALIDATION correlation, test set untouched until the end)...")
    for epoch in range(N_EPOCHS):
        model.train()
        perm = torch.randperm(n_train_examples, device=DEVICE)
        total_loss = 0.0
        for start in range(0, n_train_examples, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            log_rate = model(xb)
            loss = loss_fn(log_rate, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        avg_loss = total_loss / n_train_examples

        model.eval()
        with torch.no_grad():
            val_pred = torch.exp(model(X_val_t)).cpu().numpy()
        val_corr = np.corrcoef(val_pred, y_val)[0, 1]
        is_best = val_corr > best_val_corr
        if is_best:
            best_val_corr = val_corr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0 or is_best:
            print(f"  epoch {epoch+1:3d}: train Poisson NLL={avg_loss:.4f}, "
                  f"val correlation={val_corr:.4f}{'  <- best so far' if is_best else ''}")

    print(f"\nBest validation correlation: {best_val_corr:.4f} -- "
          f"restoring that checkpoint before touching the test set.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        cnn_test_pred = torch.exp(model(X_test_t)).cpu().numpy()
    cnn_corr = np.corrcoef(cnn_test_pred, y_test)[0, 1]

    print(f"\n=== FINAL COMPARISON (identical held-out test set, {len(y_test)} frames) ===")
    print(f"  STA baseline (linear filter): correlation = {sta_corr:.4f}")
    print(f"  CNN (Path B):                 correlation = {cnn_corr:.4f}")
    if cnn_corr > sta_corr:
        print(f"  -> CNN improves on the linear baseline by {cnn_corr - sta_corr:+.4f}")
    else:
        print(f"  -> CNN does NOT beat the linear baseline here "
              f"({cnn_corr - sta_corr:+.4f}) -- a real, reportable result, not a failure to hide.")

    # --- Plot: predicted vs actual spike counts over a test-set window ---
    PLOTS_DIR.mkdir(exist_ok=True)
    window = slice(0, 600)  # first 600 test frames, readable on one plot
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(y_test[window], label="actual spike count", color="black", linewidth=1)
    axes[0].plot(sta_pred[window], label="STA baseline prediction", color="tab:blue", alpha=0.8)
    axes[0].set_title(f"Cell {CELL_INDEX}: STA baseline vs actual (test set), corr={sta_corr:.3f}")
    axes[0].legend(fontsize=8)

    axes[1].plot(y_test[window], label="actual spike count", color="black", linewidth=1)
    axes[1].plot(cnn_test_pred[window], label="CNN prediction", color="tab:red", alpha=0.8)
    axes[1].set_title(f"Cell {CELL_INDEX}: CNN (Path B) vs actual (test set), corr={cnn_corr:.3f}")
    axes[1].set_xlabel("test-set frame index")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out_path = PLOTS_DIR / "path_b_cnn_vs_sta.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved comparison plot to {out_path}")

    torch.save(model.state_dict(), HERE / "path_b_cnn_model.pt")
    print(f"Saved trained model to {HERE / 'path_b_cnn_model.pt'}")


if __name__ == "__main__":
    main()
