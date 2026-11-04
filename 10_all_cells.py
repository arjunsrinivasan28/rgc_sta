"""
All 27 monkey cells: STA vs STA clipped at 0 vs CNN, one 70/10/20 temporal split per cell (same recipe as
03_cnn_path_b.py, which was cell 0 only). Tests Wandell's idea that the CNN's lead over the STA is the same
'never predict below zero' effect in every cell. Saves all_cells_results.json and plots/all_cells.png.
"""
import importlib.util, json
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pb", HERE / "03_cnn_path_b.py")
pb = importlib.util.module_from_spec(spec); spec.loader.exec_module(pb)


def r2s(pred, y):
    A = np.c_[pred, np.ones_like(pred)]
    g = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(1 - ((y - A @ g) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def run_cell(stim, sp, k):
    counts = pb.bin_spikes_to_frames(sp, stim.shape[0])
    X, y = pb.build_dataset(stim, counts, pb.N_HIST)
    ntr, nva = int(len(X) * .7), int(len(X) * .1)
    Xtr, Xva, Xte = X[:ntr], X[ntr:ntr + nva], X[ntr + nva:]
    ytr, yva, yte = y[:ntr], y[ntr:ntr + nva], y[ntr + nva:]
    sta, _ = pb.sta_baseline_predictions(Xtr, ytr, Xte, pb.N_HIST)
    clip = np.maximum(sta, 0)
    torch.manual_seed(k); dev = pb.DEVICE
    Xt, yt, Xv, Xe = (torch.from_numpy(a).to(dev) for a in (Xtr, ytr, Xva, Xte))
    m = pb.SpikeCNN(pb.N_HIST).to(dev); opt = torch.optim.Adam(m.parameters(), lr=pb.LEARNING_RATE)
    lf = nn.PoissonNLLLoss(log_input=True); best, st = -9, None
    for ep in range(pb.N_EPOCHS):
        m.train(); perm = torch.randperm(Xt.shape[0], device=dev)
        for s in range(0, Xt.shape[0], pb.BATCH_SIZE):
            b = perm[s:s + pb.BATCH_SIZE]; opt.zero_grad(); lf(m(Xt[b]), yt[b]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vc = np.nan_to_num(np.corrcoef(torch.exp(m(Xv)).cpu().numpy(), yva)[0, 1], nan=-1)
        if vc > best: best, st = vc, {a: v.clone() for a, v in m.state_dict().items()}
    m.load_state_dict(st); m.eval()
    with torch.no_grad(): cnn = torch.exp(m(Xe)).cpu().numpy()
    c = lambda p: float(np.corrcoef(p, yte)[0, 1]) if p.std() > 0 else 0.0
    return {"spikes": int(counts.sum()), "sta": c(sta), "clip": c(clip), "cnn": c(cnn),
            "sta_r2": r2s(sta, yte), "clip_r2": r2s(clip, yte), "cnn_r2": r2s(cnn, yte)}


def main():
    stim = sio.loadmat(pb.DATA_DIR / "Stim_reduced.mat")["Stim"]
    sps = sio.loadmat(pb.DATA_DIR / "SpTimesRGC.mat")["SpTimes"][0]
    res = []
    for k in range(27):
        r = run_cell(stim, sps[k].flatten(), k); r["cell"] = k; res.append(r)
        print(f"cell {k:2d}: STA {r['sta']:.3f}  clipped {r['clip']:.3f}  CNN {r['cnn']:.3f}  (spikes {r['spikes']})", flush=True)
        json.dump(res, open(HERE / "all_cells_results.json", "w"), indent=1)
    a = {k: np.array([r[k] for r in res]) for k in ("sta", "clip", "cnn", "sta_r2", "clip_r2", "cnn_r2")}
    gap = a["cnn"] - a["sta"]; closed = (a["clip"] - a["sta"]) / np.where(gap > 0.005, gap, np.nan)
    print(f"mean corr: STA {a['sta'].mean():.3f}, clipped {a['clip'].mean():.3f}, CNN {a['cnn'].mean():.3f}")
    print(f"mean R2: STA {a['sta_r2'].mean():.3f}, clipped {a['clip_r2'].mean():.3f}, CNN {a['cnn_r2'].mean():.3f}")
    print(f"CNN > STA in {(a['cnn']>a['sta']).sum()}/27; clipped > STA in {(a['clip']>a['sta']).sum()}/27; CNN > clipped in {(a['cnn']>a['clip']).sum()}/27")
    print(f"median fraction of gap closed by clipping (cells with gap>0.005): {np.nanmedian(closed):.2f}")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter(a["sta"], a["cnn"], c="#94A3B8", s=22, label="STA"); ax[0].scatter(a["clip"], a["cnn"], c="#8C1515", s=22, label="STA clipped")
    ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0].set_xlabel("linear model correlation"); ax[0].set_ylabel("CNN correlation"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("Each dot = one of 27 cells", fontsize=10)
    x = np.arange(27); o = np.argsort(a["cnn"])
    for i, (k, c, n) in enumerate((("sta", "#CBD5E1", "STA"), ("clip", "#8C1515", "clipped"), ("cnn", "#14B8A6", "CNN"))):
        ax[1].bar(x + (i - 1) * .27, a[k][o], .27, color=c, label=n)
    ax[1].set_xticks([]); ax[1].set_xlabel("cells, sorted by CNN correlation"); ax[1].set_ylabel("test correlation"); ax[1].legend(frameon=False, fontsize=8)
    for q in ax: q.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(HERE / "plots" / "all_cells.png", dpi=150)


if __name__ == "__main__":
    main()
