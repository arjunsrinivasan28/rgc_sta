"""
5-fold cross-validation for Path B (CNN) vs the STA baseline, cell 0.

The recording is cut into 5 contiguous time blocks. Each fold holds one block
out as the test set, uses half of the next block as validation (for picking the
CNN checkpoint), and trains on the rest. Blocks are contiguous, not shuffled,
and N_HIST examples on each side of the test and validation blocks are dropped
from training so overlapping stimulus windows can't leak across the boundary.
"""

import importlib.util
import json
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("path_b", HERE / "03_cnn_path_b.py")
pb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pb)

N_FOLDS = 5
SEED = 0


def run_fold(k, X, y, bounds):
    n = len(X)
    t0, t1 = bounds[k], bounds[k + 1]
    nk0, nk1 = bounds[(k + 1) % N_FOLDS], bounds[(k + 1) % N_FOLDS + 1]
    v0, v1 = nk0, (nk0 + nk1) // 2

    idx = np.arange(n)
    test = (idx >= t0) & (idx < t1)
    val = (idx >= v0) & (idx < v1)
    gap = pb.N_HIST
    near = np.zeros(n, dtype=bool)
    for a, b in [(t0, t1), (v0, v1)]:
        near |= (idx >= a - gap) & (idx < b + gap)
    train = ~near

    X_tr, y_tr = X[train], y[train]
    X_va, y_va = X[val], y[val]
    X_te, y_te = X[test], y[test]

    sta_pred, _ = pb.sta_baseline_predictions(X_tr, y_tr, X_te, pb.N_HIST)
    sta_corr = np.corrcoef(sta_pred, y_te)[0, 1]

    torch.manual_seed(SEED + k)
    dev = pb.DEVICE
    X_tr_t = torch.from_numpy(X_tr).to(dev)
    y_tr_t = torch.from_numpy(y_tr).to(dev)
    X_va_t = torch.from_numpy(X_va).to(dev)
    X_te_t = torch.from_numpy(X_te).to(dev)

    model = pb.SpikeCNN(pb.N_HIST).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=pb.LEARNING_RATE)
    loss_fn = nn.PoissonNLLLoss(log_input=True)
    best_val, best_state = -np.inf, None
    for epoch in range(pb.N_EPOCHS):
        model.train()
        perm = torch.randperm(X_tr_t.shape[0], device=dev)
        for s in range(0, X_tr_t.shape[0], pb.BATCH_SIZE):
            b = perm[s : s + pb.BATCH_SIZE]
            opt.zero_grad()
            loss_fn(model(X_tr_t[b]), y_tr_t[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = torch.exp(model(X_va_t)).cpu().numpy()
        vc = np.corrcoef(vp, y_va)[0, 1]
        if vc > best_val:
            best_val = vc
            best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        cnn_pred = torch.exp(model(X_te_t)).cpu().numpy()
    cnn_corr = np.corrcoef(cnn_pred, y_te)[0, 1]
    print(f"fold {k + 1}/{N_FOLDS}: test={t1 - t0} frames, "
          f"STA={sta_corr:.4f}, CNN={cnn_corr:.4f} (best val {best_val:.4f})", flush=True)
    return float(sta_corr), float(cnn_corr)


def main():
    print(f"device: {pb.DEVICE}")
    stim = sio.loadmat(pb.DATA_DIR / "Stim_reduced.mat")["Stim"]
    sp = sio.loadmat(pb.DATA_DIR / "SpTimesRGC.mat")["SpTimes"][0, pb.CELL_INDEX].flatten()
    counts = pb.bin_spikes_to_frames(sp, stim.shape[0])
    X, y = pb.build_dataset(stim, counts, pb.N_HIST)
    bounds = np.linspace(0, len(X), N_FOLDS + 1).astype(int)

    sta, cnn = [], []
    for k in range(N_FOLDS):
        s, c = run_fold(k, X, y, bounds)
        sta.append(s)
        cnn.append(c)

    sta, cnn = np.array(sta), np.array(cnn)
    print("\n=== 5-fold cross-validation, cell 0 ===")
    print(f"STA: {sta.mean():.3f} +/- {sta.std(ddof=1):.3f}   folds: {np.round(sta, 3).tolist()}")
    print(f"CNN: {cnn.mean():.3f} +/- {cnn.std(ddof=1):.3f}   folds: {np.round(cnn, 3).tolist()}")
    print(f"CNN beats STA in {(cnn > sta).sum()}/{N_FOLDS} folds; mean gain {np.mean(cnn - sta):+.3f}")

    (HERE / "plots").mkdir(exist_ok=True)
    with open(HERE / "cv_results.json", "w") as f:
        json.dump({"sta": sta.tolist(), "cnn": cnn.tolist()}, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(N_FOLDS)
    w = 0.38
    ax.bar(x - w / 2, sta, w, label=f"STA (mean {sta.mean():.3f})", color="#94A3B8")
    ax.bar(x + w / 2, cnn, w, label=f"CNN (mean {cnn.mean():.3f})", color="#14B8A6")
    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {i + 1}" for i in x])
    ax.set_ylabel("test correlation")
    ax.set_title("Cell 0, 5-fold cross-validation (contiguous time blocks)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "plots" / "path_b_cv.png", dpi=150)
    print("saved plots/path_b_cv.png and cv_results.json")


if __name__ == "__main__":
    main()
