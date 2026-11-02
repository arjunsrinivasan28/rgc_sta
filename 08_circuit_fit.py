"""
Fit the ngspice retina circuit to the real cell (cell 0), with refractory period and spike-frequency adaptation.

Wandell's suggestions from the third meeting: add a refractory period, adjust the threshold, fit the circuit
parameters (gain, threshold, time constants) to the real spikes, try spike-frequency adaptation, and test on
something longer than 100 frames. ngspice runs a 200-frame window in about a second, so the fit runs the real
circuit directly (random search, then local refinement) on training windows from the first 70% of the recording.
It is then tested on windows from the last 30% and on the held-out repeat movie (Phase 3), against the PSTH.
Everything uses the OFF-cell sign (CIRCUIT_SIGN = -1).
"""

import importlib.util
import json
import shutil
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import scipy.io as sio

HERE = Path(__file__).resolve().parent
TRAIN_STARTS = [3000, 20000, 41000, 58000, 75000, 90000]   # inside the first 70% (frames < 100800)
TRAIN_LEN = 200
TEST_STARTS = [105000, 113000, 121000, 129000, 137000]      # last 30%, never used in the fit
TEST_LEN = 400
BIN = 5

# name: (low, high, log-scale?)
SPACE = {
    "CENTER_SURROUND_GAIN": (1.0, 12.0, True),
    "V_THRESH": (0.05, 0.35, False),
    "tau_bip": (0.008, 0.08, True),
    "C_MEM": (1e-7, 1e-6, True),
    "REF_K": (0.0, 0.8, False),
    "REF_TAU": (0.002, 0.03, True),
    "ADA_K": (0.0, 0.6, False),
    "ADA_TAU": (0.05, 0.4, True),
}
HAND = {"CENTER_SURROUND_GAIN": 4.0, "V_THRESH": 0.15, "tau_bip": 0.03, "C_MEM": 3e-7,
        "REF_K": 0.0, "REF_TAU": 0.005, "ADA_K": 0.0, "ADA_TAU": 0.1}


def _load_pa():
    spec = importlib.util.spec_from_file_location("pa", HERE / "04_path_a_ngspice_circuit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PA = _load_pa()
DATA = HERE / "rgcData_Nature08"
STIM = sio.loadmat(DATA / "Stim_reduced.mat")["Stim"]
REAL = sio.loadmat(DATA / "SpTimesRGC.mat")["SpTimes"][0, 0].flatten()
TARGET = (3, 5)


def simulate(params, stim_slice):
    """Run the circuit on stim_slice with params; return spike times in frames from the slice start."""
    PA.CIRCUIT_SIGN = -1.0
    PA.CENTER_SURROUND_GAIN = params["CENTER_SURROUND_GAIN"]
    PA.V_THRESH = params["V_THRESH"]
    PA.R_TAU = params["tau_bip"] / PA.C_TAU
    PA.C_MEM = params["C_MEM"]
    PA.REF_K, PA.REF_TAU = params["REF_K"], params["REF_TAU"]
    PA.ADA_K, PA.ADA_TAU = params["ADA_K"], params["ADA_TAU"]
    d = Path(tempfile.mkdtemp(prefix="ckt_"))
    try:
        net, _ = PA.build_circuit(stim_slice, TARGET, outdir=d)
        PA.run_ngspice(net)
        raw = np.loadtxt(d / "circuit_out.txt")
        t, spike = raw[:, 6], raw[:, 7]
        return PA.detect_spike_times(t, spike, PA.V_SPIKE / 2) / PA.FRAME_DT
    except Exception:
        return None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def real_in(start, n):
    s = REAL[(REAL >= start) & (REAL < start + n)] - start
    return s


def binned(times, n, b=BIN):
    return np.histogram(times, bins=np.arange(0, n + b, b))[0]


def score_window(params, start, n):
    c = simulate(params, STIM[start:start + n])
    r = real_in(start, n)
    if c is None or len(c) == 0:
        return -1.0, 0, len(r)
    a, b = binned(c, n), binned(r, n)
    corr = np.corrcoef(a, b)[0, 1] if a.std() > 0 else 0.0
    pen = 0.7 * abs(np.log((len(c) + 1) / (len(r) + 1)))
    return float(corr - pen), len(c), len(r)


def objective(params):
    return float(np.mean([score_window(params, s, TRAIN_LEN)[0] for s in TRAIN_STARTS]))


def to_params(u):
    out = {}
    for (k, (lo, hi, lg)), x in zip(SPACE.items(), u):
        out[k] = float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo)))) if lg else float(lo + x * (hi - lo))
    return out


def _eval(u):
    return objective(to_params(u))


def main():
    rng = np.random.RandomState(0)
    names = list(SPACE)
    with Pool(6) as pool:
        base = objective(HAND)
        print(f"hand-tuned circuit (flipped sign), training objective {base:.3f}", flush=True)
        U = rng.rand(96, len(names))
        S = pool.map(_eval, list(U))
        order = np.argsort(S)[::-1]
        best_u, best_s = U[order[0]].copy(), S[order[0]]
        print(f"random search of {len(U)}: best objective {best_s:.3f}", flush=True)
        history = [float(best_s)]
        scale = 0.25
        for rnd in range(5):
            cand = np.clip(best_u + rng.randn(36, len(names)) * scale, 0, 1)
            sc = pool.map(_eval, list(cand))
            j = int(np.argmax(sc))
            if sc[j] > best_s:
                best_s, best_u = sc[j], cand[j].copy()
            history.append(float(best_s))
            print(f"refine round {rnd + 1}: best {best_s:.3f} (step {scale:.2f})", flush=True)
            scale *= 0.65
    fit = to_params(best_u)
    print("fitted parameters:", {k: round(v, 5) for k, v in fit.items()}, flush=True)

    def evaluate(params, label):
        rows = []
        for s in TEST_STARTS:
            sc, nc, nr = score_window(params, s, TEST_LEN)
            c = simulate(params, STIM[s:s + TEST_LEN])
            a, b = binned(c, TEST_LEN), binned(real_in(s, TEST_LEN), TEST_LEN)
            rows.append({"start": s, "circuit_spikes": int(nc), "real_spikes": int(nr), "corr": float(np.corrcoef(a, b)[0, 1])})
        print(f"{label}: held-out windows ({TEST_LEN} frames each): mean r {np.mean([r['corr'] for r in rows]):.3f}, "
              f"spikes circuit {sum(r['circuit_spikes'] for r in rows)} vs real {sum(r['real_spikes'] for r in rows)}", flush=True)
        return rows

    res = {"hand": {"params": HAND, "train_objective": base, "heldout": evaluate(HAND, "hand-tuned")},
           "fitted": {"params": fit, "train_objective": float(best_s), "heldout": evaluate(fit, "fitted")},
           "history": history,
           "train_starts": TRAIN_STARTS, "train_len": TRAIN_LEN, "test_starts": TEST_STARTS, "test_len": TEST_LEN}
    json.dump(res, open(HERE / "circuit_fit_results.json", "w"), indent=1)
    print("saved circuit_fit_results.json")


if __name__ == "__main__":
    main()
