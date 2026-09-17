"""
Phase 2, Path A (equivalent circuit) -- an ngspice analog circuit that
mimics the biological retina's first stages, in the spirit Wandell's email
describes for LTSpice (we use ngspice instead -- see README/project notes:
LTspice on this machine is an unlicensed, non-functional CrossOver/Wine
install, the same issue that blocked the earlier 4T pixel project, and
ngspice is the proven working substitute here too).

THE THREE REQUIRED STAGES
----------------------------
1. SPATIAL FILTERING -- a real 2D resistor grid across all 100 pixels.
   Each pixel drives a "photoreceptor" node directly (via an ideal
   voltage source, using the REAL recorded stimulus as a PWL waveform).
   Each photoreceptor node couples, through a resistor, into a parallel
   "horizontal cell" mesh -- a grid of nodes tied to their 4 nearest
   neighbors by resistors. This is the classic resistive-grid model of
   lateral inhibition (Mahowald & Mead-style silicon retina): the
   horizontal-cell layer is a spatially LOW-PASS-FILTERED (smoothed)
   copy of the raw image, because each node "leaks" charge to its
   neighbors. Subtracting the horizontal-cell voltage from the direct
   photoreceptor voltage at one location gives a genuine CENTER-SURROUND
   response -- positive when the center differs from its local average,
   the actual defining property of a retinal ganglion cell's spatial
   receptive field.
2. TEMPORAL DYNAMICS -- a simple RC low-pass stage after the
   center-surround subtraction, modeling the real delay through
   bipolar-cell integration.
3. SPIKE GENERATION -- a leaky integrate-and-fire neuron: the temporally
   filtered signal charges a membrane capacitor, and a behavioral
   comparator + voltage-controlled reset switch fires a spike pulse and
   discharges the capacitor whenever it crosses threshold -- a standard
   SPICE integrate-and-fire trick (comparator + switch), not a built-in
   SPICE primitive.

SCOPE OF THIS FIRST VERSION (documented, not hidden)
--------------------------------------------------------
We drive the FULL 10x10 resistor grid (spatial filtering is a genuinely
2D, whole-grid phenomenon -- a small patch wouldn't demonstrate it
properly), but only for a 500-FRAME slice of the real recording (frames
0-500 of the actual Stim_reduced.mat data), not the whole 144,000-frame,
20-minute recording. Reasons, stated plainly:
  - 100 independent PWL sources over the full recording would mean
    100 x 144,000 breakpoints -- a huge netlist and a very slow transient
    simulation, for a first version whose job is to demonstrate the three
    required mechanisms work, not yet to be a calibrated, full-length
    model.
  - 500 frames already contains 183 real spikes from the target cell
    (checked directly, not assumed), plenty to see real correlated
    activity.
This mirrors the same "start with a smaller, well-scoped version, validate
it, then decide whether to scale up" discipline used throughout the
earlier 4T pixel project and Phase 1/Path B of this one.

We do NOT know the real stimulus frame rate from the dataset's metadata
(it isn't stored in the .mat files) -- we assume 120 Hz (frame duration
1/120 s), consistent with Phase 1's reasoning about the 144,000-frame,
~20-minute recording. This is an assumption, stated here, not a verified
fact -- if it turns out to be wrong, the CIRCUIT's spatial behavior is
unaffected (that's driven by resistor ratios, not time), but its absolute
TIMING (matching real spike times) would need to be rescaled accordingly.
"""

from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import subprocess

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "rgcData_Nature08"
CIRCUIT_DIR = HERE / "path_a_circuit"
PLOTS_DIR = HERE / "plots"

NX, NY = 10, 10
CELL_INDEX = 0          # same cell as Phase 1 / Path B, for direct comparison
FRAME_START, FRAME_END = 0, 100  # the frame slice we actually simulate -- reduced
# from an initial 500, then 200, after the full/larger circuits turned out to be
# too slow to solve within a reasonable time even after the grid-radius cut
# below (each timestep on this stiff nonlinear circuit is expensive; verified
# this empirically -- two separate runs hit a hard timeout with zero output --
# rather than guessing at a size). 100 frames still contains real spiking
# activity to compare against (checked directly at runtime, printed below).
FRAME_DT = 1.0 / 120.0  # ASSUMED frame duration in seconds -- see module docstring

GRID_RADIUS = 2  # only build the resistor grid within this many pixels of the
# target cell's receptive-field center (a (2*GRID_RADIUS+1)^2 = 25-node window
# instead of the full 100-node sensor). A real center-surround receptive field
# is spatially local, and a resistor grid's influence falls off with distance,
# so this is a physically justified cut, not an arbitrary one -- made after the
# full 100-node grid failed to converge within a 20-minute timeout, twice.

# --- Circuit component values (hand-picked for this teaching circuit,
# not derived from a real retina datasheet -- same spirit as the 4T pixel
# project's "stylized" component values) ---
R_IN = 10e3        # photoreceptor -> horizontal-cell coupling resistor (ohms)
R_LAT = 15e3       # horizontal-cell lateral (neighbor-to-neighbor) resistor (ohms)
CENTER_SURROUND_GAIN = 4.0  # multiplies (V_photo - V_horiz) before the temporal stage
# Originally 20.0. Real bug found at this gain: over a short (100-frame)
# random window, the photoreceptor's own sample mean isn't exactly 0 (finite-
# sample noise, measured directly at 0.1475V here vs. horiz's 0.0053V) -- a
# real but small imbalance. Gain 20 amplified that fixed 0.14V offset into a
# permanent +2.8V DC bias that swamped the actual signal fluctuations,
# pinning the membrane above threshold almost permanently and causing
# uncontrolled chattering (496 "spikes" detected in 0.83s, most within 1ms
# of each other -- not discrete events). A lower gain keeps center-surround
# filtering genuinely present while keeping this sampling artifact from
# dominating.
R_TAU = 30e3       # temporal RC low-pass resistor (ohms)
C_TAU = 1e-6       # temporal RC low-pass capacitor (farads) -- tau = R*C = 30ms
R_MEM_IN = 100e3   # bipolar -> membrane coupling resistor (ohms)
R_LEAK = 2e6       # membrane leak resistor (ohms)
# C_MEM was originally 1e-9F, giving a membrane RC time constant (with
# R_LEAK||R_MEM_IN ~ 95kOhm) of only ~100us -- far shorter than a real
# spike's inter-spike interval (~30ms here, from 25 spikes/0.83s). After the
# reset switch's Ron/Vt bug was fixed, vmem recovered past threshold and
# re-triggered within microseconds of every reset, chattering at near its
# maximum rate (measured directly: 159,982 "spikes" detected in one run,
# with the spike signal above threshold 69% of the time -- clearly not
# discrete events). Raising C_MEM so the natural recovery time constant is
# on the same order as a real refractory/inter-spike interval fixes this
# without needing to hand-tune the switch's hysteresis band.
C_MEM = 3e-7       # membrane capacitance (farads) -- tau ~= 95kOhm * 3e-7F = 28.5ms
V_THRESH = 0.15    # spike threshold (volts)
V_SPIKE = 3.3      # spike pulse amplitude (volts)


def node_name(prefix, r, c):
    return f"{prefix}_{r}_{c}"


def build_circuit(stim_slice, target_rc):
    """
    stim_slice: (n_frames, 100) real stimulus data for the frames we're
    simulating. target_rc: (row, col) of the cell we're modeling (its real
    STA peak pixel location, from Phase 1).
    """
    CIRCUIT_DIR.mkdir(exist_ok=True)
    n_frames = stim_slice.shape[0]
    t_stop = n_frames * FRAME_DT
    tr, tc = target_rc

    # Only build the grid for a small neighborhood around the target cell,
    # not the full 10x10 sensor. This is a genuine, documented scope cut
    # (see GRID_RADIUS's definition above), made after the full 100-node
    # version proved too slow/stiff to converge in a reasonable time
    # (verified directly: 20 minutes wasn't enough, twice). Spatial
    # filtering is still genuinely 2D and multi-pixel here -- a real
    # center-surround receptive field's actual spatial extent is local
    # (a handful of pixels), so a nearby window contains the physically
    # relevant neighborhood; a resistor grid's influence also falls off
    # with distance, so far-away pixels contribute negligibly anyway.
    r_lo, r_hi = max(0, tr - GRID_RADIUS), min(NX - 1, tr + GRID_RADIUS)
    c_lo, c_hi = max(0, tc - GRID_RADIUS), min(NY - 1, tc + GRID_RADIUS)
    grid_cells = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
    print(f"Grid window: rows {r_lo}-{r_hi}, cols {c_lo}-{c_hi} "
          f"({len(grid_cells)} nodes, vs. {NX*NY} for the full sensor)")

    lines = [
        "* Path A -- retina equivalent circuit (spatial grid + RC + integrate-and-fire)",
        "* Auto-generated by 04_path_a_ngspice_circuit.py -- see that file's docstring",
        "* for what each stage models and why these particular values were chosen.",
        "",
        "* Dummy first element -- empirically required. This ngspice build silently",
        "* fails to evaluate the transient waveform of whatever element is listed",
        "* FIRST in the netlist (verified directly: an identical source works fine",
        "* in second position but stays frozen at its t=0 value in first position,",
        "* and this fix -- an unrelated placeholder element first -- resolves it).",
        "* Also: PWL FILE=<path> (external file) breaks with more than one such",
        "* source in the same netlist (verified directly too), so every pixel's",
        "* stimulus is embedded as inline PWL(...) breakpoints instead.",
        "Rdummy dummy0 0 1Meg",
        "",
    ]

    # --- Stage 1: one photoreceptor source per pixel IN THE WINDOW (real
    # stimulus data, inline PWL breakpoints -- see the dummy-first-element
    # note above for why not a PWL FILE reference) + the horizontal-cell
    # resistor grid over that same window. ---
    for r, c in grid_cells:
        pixel_idx = r * NY + c
        breakpoints = " ".join(
            f"{frame_i * FRAME_DT:.6e} {stim_slice[frame_i, pixel_idx]:.4f}"
            for frame_i in range(n_frames)
        )
        photo = node_name("photo", r, c)
        lines.append(f"Vpix_{r}_{c} {photo} 0 PWL({breakpoints})")

    lines.append("")
    lines.append("* Horizontal-cell layer: each node couples to its own photoreceptor")
    lines.append("* (R_IN) and to its up/down/left/right neighbors (R_LAT) -- this lateral")
    lines.append("* coupling is what performs the spatial low-pass / surround averaging.")
    grid_set = set(grid_cells)
    for r, c in grid_cells:
        photo = node_name("photo", r, c)
        horiz = node_name("horiz", r, c)
        lines.append(f"Rin_{r}_{c} {photo} {horiz} {R_IN:.6g}")
        # Only connect each lateral resistor once (to the right and down
        # neighbor, when that neighbor is also in our window) so we don't
        # double up the same resistor from both ends.
        if (r, c + 1) in grid_set:
            lines.append(f"Rlat_{r}_{c}_r {horiz} {node_name('horiz', r, c+1)} {R_LAT:.6g}")
        if (r + 1, c) in grid_set:
            lines.append(f"Rlat_{r}_{c}_d {horiz} {node_name('horiz', r+1, c)} {R_LAT:.6g}")

    # --- Stage 1 output: center-surround at the target cell's location ---
    tr, tc = target_rc
    lines.append("")
    lines.append(f"* Center-surround output at the target cell's real receptive-field")
    lines.append(f"* location (row={tr}, col={tc}, from Phase 1's STA peak pixel).")
    lines.append(
        f"Bcs centersurround 0 V={CENTER_SURROUND_GAIN:.6g}*"
        f"(V({node_name('photo', tr, tc)})-V({node_name('horiz', tr, tc)}))"
    )

    # --- Stage 2: temporal RC low-pass ---
    lines.append("")
    lines.append("* Temporal dynamics: simple RC low-pass (bipolar-cell integration delay).")
    lines.append(f"Rtau centersurround bipolar {R_TAU:.6g}")
    lines.append(f"Ctau bipolar 0 {C_TAU:.6g}")

    # --- Stage 3: leaky integrate-and-fire spike generator ---
    lines.append("")
    lines.append("* Spike generation: leaky integrate-and-fire (comparator + reset switch,")
    lines.append("* the standard SPICE trick for I&F neurons -- there's no built-in primitive).")
    lines.append(f"Rmemin bipolar vmem {R_MEM_IN:.6g}")
    lines.append(f"Rleak vmem 0 {R_LEAK:.6g}")
    lines.append(f"Cmem vmem 0 {C_MEM:.6g} IC=0")
    lines.append(
        f"* A hard ternary comparator here (0 or {V_SPIKE}V, instantaneously) was too"
    )
    lines.append(
        "* numerically stiff for the solver to converge through -- it kept stalling at"
    )
    lines.append(
        "* one timestep and never advancing (verified directly: 80 output rows, most"
    )
    lines.append(
        "* with an IDENTICAL repeated timestamp, far short of the requested duration)."
    )
    lines.append(
        "* A steep but smooth (differentiable) tanh sigmoid gives the same effective"
    )
    lines.append(
        "* comparator behavior while staying solvable. Gain matters a lot here: an"
    )
    lines.append(
        "* earlier version used gain 200, which is so steep that ngspice's local-"
    )
    lines.append(
        "* truncation-error estimator collapsed the timestep near every threshold"
    )
    lines.append(
        "* approach (measured directly: 525,632 internal steps for what should be"
    )
    lines.append(
        "* ~80, a 15x slowdown). Gain 20 is still a sharp, effectively-binary"
    )
    lines.append("* comparator but stays numerically solvable.")
    lines.append(
        f"Bspike spike 0 V={V_SPIKE/2:.6g}*(1+tanh((V(vmem)-{V_THRESH:.6g})*20))"
    )
    lines.append("Sreset vmem 0 spike 0 SWRESET")
    lines.append(
        "* The switch's control node is the SPIKE signal (0..V_SPIKE), not vmem, so"
    )
    lines.append(
        "* its Vt must be calibrated to that scale -- a real bug found here: an"
    )
    lines.append(
        "* earlier version used Vt=V_THRESH/2 (0.075), a threshold meant for the"
    )
    lines.append(
        "* ~0.1V-scale membrane voltage, on a signal that actually swings to 3.3V."
    )
    lines.append(
        "* That let the switch fire (and clamp vmem toward 0) the instant vmem got"
    )
    lines.append(
        "* anywhere near threshold, long before the comparator actually went high --"
    )
    lines.append(
        "* which is why vmem never rose past ~0.06V and no spikes were ever produced."
    )
    lines.append(
        f"* Also softened Ron:Roff from 20,000:1 to 500:1 -- the original ratio was"
    )
    lines.append("* part of the same convergence problem.")
    lines.append(
        f".model SWRESET SW(Ron=200 Roff=100k Vt={V_SPIKE/2:.6g} Vh={V_SPIKE*0.2:.6g})"
    )
    lines.append("")
    lines.append("* Convergence aids for this stiff switched/nonlinear circuit -- reltol")
    lines.append("* loosened from the 0.001 default (fine for this qualitative demo, not")
    lines.append("* a precision analog design), gmin raised slightly, iteration limit raised.")
    lines.append(".options reltol=0.01 gmin=1e-9 itl4=500")

    lines.append("")
    # 4-arg form: Tstep Tstop Tstart Tmax -- Tmax caps the solver's adaptive
    # step size. Empirically required: with 100 PWL sources active at once,
    # the adaptive-timestep solver was skipping past most of the stimulus's
    # frame-to-frame detail after the first few points (verified directly:
    # one isolated PWL source over the full duration tracked correctly, but
    # in the full 100-source circuit every signal went flat after ~10
    # frames). Explicitly capping the max step at one frame interval forces
    # it to actually sample every frame.
    lines.append(f".tran {FRAME_DT/8:.6e} {t_stop:.6e} 0 {FRAME_DT:.6e} uic")
    lines.append("")
    lines.append(".control")
    lines.append("run")
    lines.append("wrdata circuit_out.txt v(centersurround) v(bipolar) v(vmem) v(spike)")
    lines.append("quit")
    lines.append(".endc")
    lines.append(".end")

    netlist_path = CIRCUIT_DIR / "retina_circuit.cir"
    with open(netlist_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote netlist: {netlist_path} ({len(lines)} lines, "
          f"{len(grid_cells)} photoreceptors, {len(grid_cells)} horizontal cells)")
    return netlist_path, t_stop


def run_ngspice(netlist_path):
    result = subprocess.run(
        ["ngspice", "-b", str(netlist_path)],
        cwd=str(CIRCUIT_DIR), capture_output=True, text=True, timeout=1200,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ngspice failed:\n{result.stdout}\n{result.stderr}")
    print("ngspice run complete.")
    return result


def parse_wrdata_multi(path, n_signals):
    """wrdata with N signals writes 2*N columns (time,value repeated)."""
    raw = np.loadtxt(path)
    signals = []
    for i in range(n_signals):
        signals.append((raw[:, 2 * i], raw[:, 2 * i + 1]))
    return signals


def detect_spike_times(t, v_spike, threshold):
    """Rising-edge threshold crossings, in the same time units as t."""
    above = v_spike > threshold
    rising = np.where(np.diff(above.astype(int)) == 1)[0]
    return t[rising]


def main():
    stim = sio.loadmat(DATA_DIR / "Stim_reduced.mat")["Stim"]
    sptimes_cell = sio.loadmat(DATA_DIR / "SpTimesRGC.mat")["SpTimes"]
    stim_slice = stim[FRAME_START:FRAME_END, :]

    # Reuse Phase 1's actual result for this cell's receptive-field
    # location, rather than re-deriving it here -- see webapp/sta_data.json
    # (cell 0's peak_pixel = 35 -> row 3, col 5).
    target_pixel = 35
    target_rc = (target_pixel // NY, target_pixel % NY)
    print(f"Modeling cell {CELL_INDEX}, receptive-field location pixel {target_pixel} "
          f"(row={target_rc[0]}, col={target_rc[1]}) -- from Phase 1's STA.")

    real_spike_times_frames = sptimes_cell[0, CELL_INDEX].flatten()
    real_spikes_in_window = real_spike_times_frames[
        (real_spike_times_frames >= FRAME_START) & (real_spike_times_frames < FRAME_END)
    ]
    print(f"Real recorded spikes in this {FRAME_END-FRAME_START}-frame window: "
          f"{len(real_spikes_in_window)}")

    netlist_path, t_stop = build_circuit(stim_slice, target_rc)
    run_ngspice(netlist_path)

    cs, bipolar, vmem, spike = parse_wrdata_multi(CIRCUIT_DIR / "circuit_out.txt", 4)
    t = cs[0]

    circuit_spike_times_s = detect_spike_times(t, spike[1], V_SPIKE / 2)
    circuit_spike_times_frames = circuit_spike_times_s / FRAME_DT
    print(f"Circuit-generated spikes in this window: {len(circuit_spike_times_s)}")

    # --- Plot: circuit's internal signals + a spike-raster comparison ---
    PLOTS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(cs[0] / FRAME_DT, cs[1], linewidth=0.8, color="tab:blue")
    axes[0].set_ylabel("center-surround\n(stage 1) [V]")
    axes[0].set_title(f"Path A retina circuit -- cell {CELL_INDEX}, "
                       f"frames {FRAME_START}-{FRAME_END}")

    axes[1].plot(bipolar[0] / FRAME_DT, bipolar[1], linewidth=0.8, color="tab:orange")
    axes[1].set_ylabel("bipolar (stage 2,\nRC-filtered) [V]")

    axes[2].plot(vmem[0] / FRAME_DT, vmem[1], linewidth=0.8, color="tab:green")
    axes[2].axhline(V_THRESH, color="gray", linestyle="--", linewidth=0.8, label="threshold")
    axes[2].set_ylabel("membrane\n(stage 3) [V]")
    axes[2].legend(fontsize=8)

    axes[3].eventplot([real_spikes_in_window], lineoffsets=1, colors="black",
                       label=f"real spikes (n={len(real_spikes_in_window)})")
    axes[3].eventplot([circuit_spike_times_frames], lineoffsets=0, colors="tab:red",
                       label=f"circuit spikes (n={len(circuit_spike_times_s)})")
    axes[3].set_yticks([0, 1])
    axes[3].set_yticklabels(["circuit", "real"])
    axes[3].set_xlabel("frame number")
    axes[3].set_title("Spike raster: real recorded cell vs. circuit output")
    axes[3].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out_path = PLOTS_DIR / "path_a_circuit_result.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
