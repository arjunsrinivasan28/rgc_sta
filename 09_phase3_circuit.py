"""
Phase 3 for the circuit: run the ngspice circuit (hand-tuned and fitted, both with the OFF-cell sign) on the full
1200-frame held-out repeat movie and compare to the real PSTH, next to the STA and CNN predictions from
07_phase3_psth.py. The circuit emits spikes, so everything is compared in 5-frame bins (about 42 ms).
Needs circuit_fit_results.json (08_circuit_fit.py) and phase3_predictions.npz (07_phase3_psth.py).
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
spec = importlib.util.spec_from_file_location("fit", HERE / "08_circuit_fit.py")
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)

N_HIST, BIN = 20, 5


def binsum(x, b=BIN):
    n = len(x) // b * b
    return x[:n].reshape(-1, b).sum(1)


def main():
    rpt = sio.loadmat(fit.DATA / "Stim_reducedRpt.mat")["Stim"]
    pred = np.load(HERE / "phase3_predictions.npz")
    target = pred["target"]                                  # frames N_HIST..1199
    full_psth = np.zeros(1200)
    trains = sio.loadmat(fit.DATA / "MtspRGCrpt.mat")["MtspRGCrpt"][0, 0]
    for j in range(trains.shape[1]):
        t = trains[:, j]; t = t[t > 0]
        full_psth += np.bincount(np.floor(t).astype(int).clip(0, 1199), minlength=1200)
    full_psth /= trains.shape[1]
    res = json.load(open(HERE / "circuit_fit_results.json"))
    out, series = {}, {}
    for name in ("hand", "fitted"):
        spikes = fit.simulate(res[name]["params"], rpt)
        counts = np.bincount(np.floor(spikes).astype(int).clip(0, 1199), minlength=1200).astype(float)
        a, b = binsum(counts[N_HIST:]), binsum(target)
        out[f"circuit {name}"] = {"corr_psth_5frame": float(np.corrcoef(a, b)[0, 1]),
                                  "spikes": int(len(spikes)), "psth_spikes": float(full_psth[N_HIST:].sum()),
                                  "mean_rate_ratio": float(len(spikes) / full_psth.sum())}
        series[f"circuit {name}"] = a
    ref = binsum(target)
    for k, key in (("STA clipped", "STA_clipped"), ("CNN", "CNN"), ("STA", "STA")):
        p = pred[key]
        out[k] = {"corr_psth_5frame": float(np.corrcoef(binsum(p), ref)[0, 1])}
        series[k] = binsum(p)
    for k, v in out.items():
        print(k, {a: round(b, 3) for a, b in v.items()})
    json.dump(out, open(HERE / "phase3_circuit_results.json", "w"), indent=1)

    x = np.arange(len(ref)) * BIN + N_HIST
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5))
    ax[0].plot(x, ref, color="k", lw=1.3, label="real PSTH (600 repeats)")
    ax[0].plot(x, series["CNN"], color="#14B8A6", lw=1, label=f"CNN (r={out['CNN']['corr_psth_5frame']:.2f})")
    ax[0].plot(x, series["circuit fitted"], color="#8C1515", lw=1, label=f"fitted circuit (r={out['circuit fitted']['corr_psth_5frame']:.2f})")
    ax[0].plot(x, series["circuit hand"], color="#C9A227", lw=1, label=f"hand-tuned circuit (r={out['circuit hand']['corr_psth_5frame']:.2f})")
    ax[0].set_xlabel("frame of the repeat movie"); ax[0].set_ylabel("spikes per 5 frames"); ax[0].legend(frameon=False, fontsize=8, ncol=2)
    ax[0].set_title("Held-out repeat movie, 5-frame bins", fontsize=10); ax[0].spines[["top", "right"]].set_visible(False)
    names = ["STA", "STA clipped", "CNN", "circuit hand", "circuit fitted"]
    cols = ["#CBD5E1", "#94A3B8", "#14B8A6", "#C9A227", "#8C1515"]
    ax[1].bar(names, [out[n]["corr_psth_5frame"] for n in names], color=cols)
    ax[1].set_ylabel("correlation with PSTH"); ax[1].set_ylim(0, 1); ax[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(HERE / "plots" / "phase3_circuit.png", dpi=150)
    print("saved plots/phase3_circuit.png")


if __name__ == "__main__":
    main()
