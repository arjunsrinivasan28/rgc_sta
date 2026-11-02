"""
Phase 3: validate the monkey models on the held-out 10-second repeat stimulus.

The repeat movie (Stim_reducedRpt.mat, 1200 frames) was shown 600 times and is NOT part of the long recording
the models are trained on. Averaging the 600 spike trains gives a PSTH, a clean estimate of the cell's true
firing rate over time. Each model predicts a rate for the repeat movie, and we compare it to that PSTH.

Models (cell 0, trained on the long recording only): STA, STA clipped at 0, small CNN.
The noise ceiling is the correlation between PSTHs from two random halves of the repeats (Spearman-Brown
corrected), i.e. how well a perfect model could do given only 600 noisy repeats.
Saves phase3_psth_results.json and the predictions used by the circuit comparison (phase3_predictions.npz).
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

CELL = pb.CELL_INDEX
N_HIST = pb.N_HIST


def psth(trains, n_frames=1200, idx=None):
    cols = range(trains.shape[1]) if idx is None else idx
    tot = np.zeros(n_frames)
    for j in cols:
        t = trains[:, j]
        t = t[t > 0]
        tot += np.bincount(np.floor(t).astype(int).clip(0, n_frames - 1), minlength=n_frames)
    return tot / len(list(cols))


def fit_scale(pred, y):
    """Least-squares gain and offset mapping a prediction onto spike counts per frame."""
    A = np.c_[pred, np.ones_like(pred)]
    return np.linalg.lstsq(A, y, rcond=None)[0]


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    stim = sio.loadmat(pb.DATA_DIR / "Stim_reduced.mat")["Stim"]
    sp = sio.loadmat(pb.DATA_DIR / "SpTimesRGC.mat")["SpTimes"][0, CELL].flatten()
    rpt_stim = sio.loadmat(pb.DATA_DIR / "Stim_reducedRpt.mat")["Stim"]
    rpt = sio.loadmat(pb.DATA_DIR / "MtspRGCrpt.mat")["MtspRGCrpt"][0, CELL]
    print(f"repeat spike matrix {rpt.shape} (max spikes per repeat x 600 repeats), repeat movie {rpt_stim.shape}")

    counts = pb.bin_spikes_to_frames(sp, stim.shape[0])
    X, y = pb.build_dataset(stim, counts, N_HIST)
    n_tr = int(len(X) * 0.9)
    Xtr, ytr, Xva, yva = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]

    Xr, _ = pb.build_dataset(rpt_stim, np.zeros(rpt_stim.shape[0], dtype=np.float32), N_HIST)
    target = psth(rpt)[N_HIST:]
    rng = np.random.RandomState(0)
    halves = []
    for _ in range(50):
        p = rng.permutation(600)
        a, b = psth(rpt, idx=p[:300])[N_HIST:], psth(rpt, idx=p[300:])[N_HIST:]
        r = np.corrcoef(a, b)[0, 1]
        halves.append(2 * r / (1 + r))
    ceiling = float(np.mean(halves))
    print(f"PSTH mean {target.mean():.3f} spikes/frame; noise ceiling (split-half, SB-corrected) r = {ceiling:.3f}")

    # --- STA (trained on 90% of the long recording) ---
    sta_tr, kernel = pb.sta_baseline_predictions(Xtr, ytr, Xtr, N_HIST)
    sta_rep = np.einsum("nlp,lp->n", Xr.reshape(len(Xr), N_HIST, -1), kernel)
    clip_tr, clip_rep = np.maximum(sta_tr, 0), np.maximum(sta_rep, 0)
    preds = {"STA": (sta_tr, sta_rep), "STA clipped": (clip_tr, clip_rep)}

    # --- CNN, checkpoint chosen on the validation tail of the long recording ---
    torch.manual_seed(0)
    dev = pb.DEVICE
    Xt, yt, Xv = (torch.from_numpy(a).to(dev) for a in (Xtr, ytr, Xva))
    model = pb.SpikeCNN(N_HIST).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=pb.LEARNING_RATE)
    loss_fn = nn.PoissonNLLLoss(log_input=True)
    best, state = -np.inf, None
    for ep in range(pb.N_EPOCHS):
        model.train()
        perm = torch.randperm(Xt.shape[0], device=dev)
        for s in range(0, Xt.shape[0], pb.BATCH_SIZE):
            b = perm[s:s + pb.BATCH_SIZE]
            opt.zero_grad(); loss_fn(model(Xt[b]), yt[b]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vc = np.corrcoef(torch.exp(model(Xv)).cpu().numpy(), yva)[0, 1]
        if vc > best:
            best, state = vc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        cnn_tr = torch.exp(model(Xt)).cpu().numpy()
        cnn_rep = torch.exp(model(torch.from_numpy(Xr).to(dev))).cpu().numpy()
    preds["CNN"] = (cnn_tr, cnn_rep)
    print(f"CNN best validation correlation {best:.3f}")

    out, scaled = {}, {}
    for name, (ptr, prep) in preds.items():
        if name == "CNN":                        # already a Poisson rate, use as is
            sc = prep
        else:
            g, c = fit_scale(ptr, ytr)           # gain/offset fit on the long recording only
            sc = np.maximum(g * prep + c, 0)
        scaled[name] = sc
        out[name] = {"corr_psth": float(np.corrcoef(prep, target)[0, 1]), "r2_psth": r2(sc, target)}
        out[name]["frac_of_ceiling"] = out[name]["corr_psth"] / ceiling
        print(f"{name:12s} corr with PSTH {out[name]['corr_psth']:.3f}  ({out[name]['frac_of_ceiling']*100:.0f}% of ceiling)  R^2 {out[name]['r2_psth']:.3f}")
    out["noise_ceiling_corr"] = ceiling
    out["n_repeats"] = 600
    json.dump(out, open(HERE / "phase3_psth_results.json", "w"), indent=1)
    np.savez(HERE / "phase3_predictions.npz", target=target, **{k.replace(" ", "_"): v for k, v in scaled.items()})

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
    t = np.arange(N_HIST, 1200)
    ax = axes[0]
    ax.plot(t, target, color="k", lw=1.2, label="real PSTH (600 repeats)")
    ax.plot(t, scaled["STA clipped"], color="#94A3B8", lw=1, label=f"STA clipped (r={out['STA clipped']['corr_psth']:.2f})")
    ax.plot(t, scaled["CNN"], color="#14B8A6", lw=1, label=f"CNN (r={out['CNN']['corr_psth']:.2f})")
    ax.set_xlabel("frame of the repeat movie"); ax.set_ylabel("spikes per frame"); ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.set_title(f"Cell 0 on the held-out repeat movie (noise ceiling r = {ceiling:.2f})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    names = ["STA", "STA clipped", "CNN"]
    ax.bar(names, [out[n]["corr_psth"] for n in names], color=["#CBD5E1", "#94A3B8", "#14B8A6"])
    ax.axhline(ceiling, color="k", ls="--", lw=1); ax.text(2.45, ceiling + .01, "noise ceiling", ha="right", fontsize=8)
    ax.set_ylabel("correlation with PSTH"); ax.set_ylim(0, 1); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(HERE / "plots" / "phase3_psth.png", dpi=150)
    print("saved plots/phase3_psth.png")


if __name__ == "__main__":
    main()
