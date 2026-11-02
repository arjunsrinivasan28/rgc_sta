"""
Does clipping (rectifying) the STA prediction at zero close the gap to the CNN?

Wandell's point: spike counts can't be negative, but the STA prediction (a weighted sum of
the stimulus) can be. A CNN with an exponential output never goes negative, so part of its
"win" over the STA might just be that. Variants, all scored on the same 5 contiguous folds
used for the CNN cross-validation (05_cv_path_b.py), cell 0:

  STA            raw linear prediction
  STA clipped    max(prediction, 0)
  STA clip+thr   max(prediction - c, 0), c picked on the validation block (a linear-nonlinear model)
  CNN            numbers from cv_results.json (same folds)

Scaling a prediction does not change correlation, but it changes the fit (R^2), so R^2 is also
reported after a single scale factor fit on the training block.
"""

import importlib.util
import json
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("path_b", HERE / "03_cnn_path_b.py")
pb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pb)
N_FOLDS = 5


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 else 0.0


def r2_scaled(pred_tr, y_tr, pred_te, y_te):
    a = (pred_tr @ y_tr) / max(pred_tr @ pred_tr, 1e-12)
    resid = y_te - a * pred_te
    return float(1 - (resid @ resid) / ((y_te - y_te.mean()) @ (y_te - y_te.mean())))


def main():
    stim = sio.loadmat(pb.DATA_DIR / "Stim_reduced.mat")["Stim"]
    sp = sio.loadmat(pb.DATA_DIR / "SpTimesRGC.mat")["SpTimes"][0, pb.CELL_INDEX].flatten()
    counts = pb.bin_spikes_to_frames(sp, stim.shape[0])
    X, y = pb.build_dataset(stim, counts, pb.N_HIST)
    n = len(X)
    Xf = X.reshape(n, -1)
    bounds = np.linspace(0, n, N_FOLDS + 1).astype(int)
    idx = np.arange(n)
    cnn = json.load(open(HERE / "cv_results.json"))["cnn"]

    res = {k: {"corr": [], "r2": []} for k in ["STA", "STA clipped", "STA clip+thr"]}
    for k in range(N_FOLDS):
        t0, t1 = bounds[k], bounds[k + 1]
        nk0, nk1 = bounds[(k + 1) % N_FOLDS], bounds[(k + 1) % N_FOLDS + 1]
        v0, v1 = nk0, (nk0 + nk1) // 2
        near = np.zeros(n, dtype=bool)
        for a, b in [(t0, t1), (v0, v1)]:
            near |= (idx >= a - pb.N_HIST) & (idx < b + pb.N_HIST)
        tr = ~near
        te = (idx >= t0) & (idx < t1)
        va = (idx >= v0) & (idx < v1)

        _, sta = pb.sta_baseline_predictions(X[tr], y[tr], X[te], pb.N_HIST)
        pred = Xf @ sta.ravel().astype(np.float32)  # linear prediction for every example
        p_tr, p_va, p_te = pred[tr], pred[va], pred[te]
        y_tr, y_va, y_te = y[tr], y[va], y[te]

        res["STA"]["corr"].append(corr(p_te, y_te))
        res["STA"]["r2"].append(r2_scaled(p_tr, y_tr, p_te, y_te))
        res["STA clipped"]["corr"].append(corr(np.maximum(p_te, 0), y_te))
        res["STA clipped"]["r2"].append(r2_scaled(np.maximum(p_tr, 0), y_tr, np.maximum(p_te, 0), y_te))
        cs = np.quantile(p_va, np.linspace(0.0, 0.98, 50))
        c = cs[int(np.argmax([corr(np.maximum(p_va - c_, 0), y_va) for c_ in cs]))]
        res["STA clip+thr"]["corr"].append(corr(np.maximum(p_te - c, 0), y_te))
        res["STA clip+thr"]["r2"].append(r2_scaled(np.maximum(p_tr - c, 0), y_tr, np.maximum(p_te - c, 0), y_te))
        print(f"fold {k+1}: STA {res['STA']['corr'][-1]:.3f}  clipped {res['STA clipped']['corr'][-1]:.3f}  "
              f"clip+thr {res['STA clip+thr']['corr'][-1]:.3f}  CNN {cnn[k]:.3f}", flush=True)

    print("\n=== cell 0, 5-fold CV: mean +/- sd across folds ===")
    summary = {}
    for k, v in res.items():
        c, r = np.array(v["corr"]), np.array(v["r2"])
        summary[k] = {"corr_mean": c.mean(), "corr_sd": c.std(ddof=1), "r2_mean": r.mean(), "r2_sd": r.std(ddof=1), "folds": c.tolist()}
        print(f"{k:13s} corr {c.mean():.3f} +/- {c.std(ddof=1):.3f}   scaled R^2 {r.mean():.3f} +/- {r.std(ddof=1):.3f}")
    c = np.array(cnn)
    summary["CNN"] = {"corr_mean": c.mean(), "corr_sd": c.std(ddof=1), "folds": c.tolist()}
    print(f"{'CNN':13s} corr {c.mean():.3f} +/- {c.std(ddof=1):.3f}")
    json.dump(summary, open(HERE / "rectified_sta_results.json", "w"), indent=2)

    names = ["STA", "STA clipped", "STA clip+thr", "CNN"]
    cols = ["#8C8F94", "#B6B1A9", "#2D716F", "#8C1515"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, nm in enumerate(names):
        ax.bar(i, summary[nm]["corr_mean"], yerr=summary[nm]["corr_sd"], color=cols[i], capsize=4)
        ax.text(i, summary[nm]["corr_mean"] + 0.02, f"{summary[nm]['corr_mean']:.3f}", ha="center", fontsize=10)
    ax.set_xticks(range(4)); ax.set_xticklabels(names)
    ax.set_ylabel("test correlation (5-fold, cell 0)"); ax.set_ylim(0, 0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(HERE / "plots" / "rectified_sta.png", dpi=150)
    print("saved plots/rectified_sta.png, rectified_sta_results.json")


if __name__ == "__main__":
    main()
