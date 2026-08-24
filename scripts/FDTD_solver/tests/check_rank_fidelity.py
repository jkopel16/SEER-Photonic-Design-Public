"""
check_rank_fidelity.py -- T6: are within-sigma E orderings real physics?

The audit measured +-2.5% staircase label jitter on the ORDERED cell and a
geometry-dependent res-60 systematic. Within-sigma E differences are
~1-2%. If orderings scramble under grid refinement, within-sigma ranking
(the project's central novelty claim) is unresolvable at res 60; if they
hold, it is sound. This measures it directly.

Method: take 5 banked jitter sigma=0.10 samples spanning the E range,
re-solve each at res 90 with BOTH polarizations (matching production
labels), and compare all 10 pairwise orderings res-60 vs res-90.

Run with PC_RESOLUTION=90 in the environment (asserted below).

SUPERSEDED (audit, final 2026-07-26): the ~0.5% floor this script
measured (N=5) was replaced by Test 7 (0.8-1%, res-90 referee, N=15) and
finally by Test 9 / rank_fidelity_r120.py (0.30%, res-120 referee,
N=15/105 pairs).  Res 90 was later convicted as an unconverged referee
(ladder_referee.py).  Keep this file as the record of the first
measurement; do not act on its verdict.
"""

from __future__ import annotations

import glob
import os
import sys
import numpy as np


# solver modules live one directory up from tests/
_SOLVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SOLVER_DIR)

import config as C
from logutil import tee
from materials_gpu import fit_all
from optics_core import solar_weight, planar_reference_stack
import fdtd_torch as F

assert C.RESOLUTION == 90, "run with PC_RESOLUTION=90"
SIGMA_TARGET = float(os.environ.get("PC_RANK_SIGMA", "0.10"))
N_PICK = int(os.environ.get("PC_RANK_N", "5"))
BANK = os.environ.get("PC_BANK",
                      os.path.join(_SOLVER_DIR, "data_production"))
# Diagnostics land in tests/gpu_out_diag regardless of CWD (PC_DIAG overrides).
DIAG_DIR = os.environ.get("PC_DIAG") or os.path.join(
    _SOLVER_DIR, "tests", "gpu_out_diag")
NORM_DIR = os.path.join(DIAG_DIR, "norm_cache_rank90")


def main():
    tee("check_rank_fidelity", DIAG_DIR)
    device = F.resolve_device(os.environ.get("PC_DEVICE", "auto"))
    print(C.describe())
    print(f"[device] {device}   [bank] {BANK}   sigma={SIGMA_TARGET}\n")

    # ---- pick banked samples spanning the E range ------------------------
    cand = []
    for p in sorted(glob.glob(os.path.join(BANK, "samples",
                                           "sample_*.npz"))):
        z = np.load(p, allow_pickle=False)
        if (str(z["disorder_class"]) == "jitter"
                and abs(float(z["sigma"]) - SIGMA_TARGET) < 1e-9):
            cand.append((float(z["E"]), int(z["sample_id"]), p))
    if len(cand) < N_PICK:
        raise SystemExit(f"only {len(cand)} banked jitter sigma="
                         f"{SIGMA_TARGET} samples; need {N_PICK}")
    cand.sort()
    idx = np.linspace(0, len(cand) - 1, N_PICK).round().astype(int)
    picks = [cand[i] for i in idx]
    print(f"[picked {N_PICK} of {len(cand)} banked samples, "
          f"spanning E {picks[0][0]:.4f} .. {picks[-1][0]:.4f}]")
    for E60, sid, _ in picks:
        print(f"    sample {sid:6d}  E(res60) = {E60:.4f}")

    # ---- re-solve each at res 90, both polarizations ---------------------
    fits, ad, mats = fit_all()
    wl = C.raw_wavelength_grid()
    w = solar_weight(wl)
    ref = planar_reference_stack(ad["si"], ad["zno"], ad["ag"], wl,
                                 C.THICKNESS_NM, C.BUFFER_NM)
    eta_p = float(np.sum(w * np.asarray(ref["A_si"], float)))

    print(f"\n{'sid':>6} {'E res60':>9} {'E res90':>9} {'shift%':>8} "
          f"{'cap':>5} {'runtime':>9}")
    print("-" * 54)
    rows = []
    for E60, sid, path in picks:
        z = np.load(path, allow_pickle=False)
        holes = [tuple(h) for h in np.asarray(z["holes_xyr_nm"])]
        a_sup = float(z["a_super_nm"])
        A_si, A_par, info = F.broadband_absorption_many(
            [holes], a_sup, C.THICKNESS_NM, wl, fits, C.BUFFER_NM,
            C.RESOLUTION, C.DECAY_TOL, C.MAX_TIME, NORM_DIR,
            device=device, n_cells_tag=f"rank{sid}")
        A = np.asarray(A_si[0], float)
        E90 = float(np.sum(w * np.clip(A, 0, 1))) / eta_p
        cap = bool(np.any(info["hit_time_cap"]))
        rt = float(np.sum(info["runtime_s"]))
        rows.append((sid, E60, E90))
        print(f"{sid:6d} {E60:9.4f} {E90:9.4f} "
              f"{100*(E90/E60-1):8.2f} {str(cap):>5} {rt:8.0f}s",
              flush=True)

    # ---- pairwise ordering analysis ---------------------------------------
    print("\n[pairwise orderings: does res-90 agree with res-60?]")
    print(f"{'pair':>15} {'dE60%':>7} {'dE90%':>7}  ordering")
    print("-" * 48)
    n_ok, n_tot = 0, 0
    kept, flipped = [], []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            s1, a60, a90 = rows[i]
            s2, b60, b90 = rows[j]
            d60 = 100 * (b60 / a60 - 1)
            d90 = 100 * (b90 / a90 - 1)
            same = (d60 > 0) == (d90 > 0)
            n_tot += 1
            n_ok += same
            (kept if same else flipped).append(abs(d60))
            print(f"{s1:6d} vs {s2:5d} {d60:7.2f} {d90:7.2f}  "
                  f"{'KEPT' if same else 'FLIPPED'}")

    shifts = [100 * (E90 / E60 - 1) for _, E60, E90 in rows]
    print(f"\n[summary]")
    print(f"  orderings preserved : {n_ok}/{n_tot}")
    print(f"  res60->90 shift     : mean {np.mean(shifts):+.2f}%  "
          f"spread {np.std(shifts):.2f}%   (ordered cell was -1.8%)")
    if flipped:
        print(f"  flips occurred at |dE60| = "
              f"{', '.join(f'{x:.2f}%' for x in sorted(flipped))}")
        floor = max(flipped)
        print(f"\n  VERDICT: PARTIAL. Orderings below ~{floor:.1f}% "
              f"separation are staircase\n  lottery; above it they hold. "
              "Within-sigma ranking is REAL for pairs\n  separated by more "
              f"than ~{floor:.1f}%.  NOTE: this was the FIRST\n  "
              "measurement (N=5, res-90 referee) and is SUPERSEDED by\n  "
              "rank_fidelity_r120.py (Test 9, N=15, res-120 referee,\n  "
              "floor 0.30%). Gate the GA on 0.30%, not on this number.")
    else:
        print("\n  VERDICT: ALL ORDERINGS PRESERVED. Within-sigma ranking "
              "at res 60 is\n  real physics down to the smallest gap "
              "tested. Question 2 is sound;\n  the central novelty claim "
              "stands. Report the shift spread above as\n  the disordered-"
              "label jitter (expect it to be well under the ordered\n  "
              "cell's +-2.5%).")


if __name__ == "__main__":
    main()
