"""
check_num_res.py -- does the NUMERATOR carry the blue-band grid error?

The planar ladder (check_res_production.py) showed eta_FDTD(planar) falls
8.2% from res 60 -> 120 (0.15177 -> 0.13960 vs TMM 0.14034), concentrated
entirely in band 1.  But E = eta_FDTD(pattern) / eta_TMM(planar) is
FDTD/TMM -- there is NO cancellation -- yet the E-ladder is flat and the
RCWA anchor agrees to 1.58%.  Those facts are in tension.

Resolution: measure eta_FDTD(ordered unit cell) per band across the same
ladder.  Cheap (1x1 cell).

  eta_num FLAT while eta_planar collapses -> the patterned structure's blue
      response is broad/saturated and grid-insensitive; the planar stack's
      sharp Fabry-Perot fringes are what res 60 mangles.  E at res 60 is
      SOUND; the flat E-ladder and the RCWA anchor are explained, not lucky.

  eta_num COLLAPSES ~8% too -> E should have moved and didn't.  Something is
      masking it.  Do NOT bank 1551 samples until that is understood.
"""

from __future__ import annotations

import os
import sys
import numpy as np


# solver modules live one directory up from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from logutil import tee
from materials_gpu import fit_all, BANDS
from optics_core import solar_weight, planar_reference_stack
import fdtd_torch as F

RES_LADDER = [int(r) for r in
              os.environ.get("PC_RES_LADDER", "60,90,120").split(",")]
NORM_DIR = os.path.join(C.OUT_DIR, "norm_cache_numcheck")


def band_masks(wl):
    out = []
    for b, (lo, hi) in enumerate(BANDS):
        m = (wl >= lo) & (wl < hi) if b < len(BANDS) - 1 else \
            (wl >= lo) & (wl <= hi)
        out.append(m)
    return out


def main():
    tee("check_num_res", C.OUT_DIR)
    device = F.resolve_device(os.environ.get("PC_DEVICE", "auto"))
    print(C.describe())
    print(f"[device] {device}\n")
    print("METRIC: AM1.5G-weighted eta of the ORDERED 1x1 unit cell "
          "(the E NUMERATOR),\n        per band, across a resolution "
          "ladder.  Compare its trend to the\n        planar ladder: "
          "planar band1 went +1.178 -> +0.115 -> -0.068 pp\n        "
          "(eta 0.15177 -> 0.14125 -> 0.13960 vs TMM 0.14034).\n")

    fits, adapters, mats = fit_all()
    wl = C.raw_wavelength_grid()
    w = solar_weight(wl)
    masks = band_masks(wl)

    ref = planar_reference_stack(adapters["si"], adapters["zno"],
                                 adapters["ag"], wl, C.THICKNESS_NM,
                                 C.BUFFER_NM)
    eta_planar_tmm = float(np.sum(w * np.asarray(ref["A_si"], float)))
    print(f"[denominator] eta_TMM(planar) = {eta_planar_tmm:.5f}  "
          f"(analytic, resolution-independent)\n")

    print(f"{'res':>5} {'b1 eta':>9} {'b2 eta':>9} {'b3 eta':>9} "
          f"{'eta_num':>9} {'E':>8} {'dE vs res60':>12} {'runtime':>9}")
    print("-" * 78)
    rows = []
    for res in RES_LADDER:
        A_si, A_par, info = F.broadband_absorption_many(
            [F.ordered_square_holes(C.A_NM, 1, C.R_OVER_A * C.A_NM)[0]],
            C.A_NM, C.THICKNESS_NM, wl, fits, C.BUFFER_NM, res,
            C.DECAY_TOL, C.MAX_TIME, NORM_DIR, device=device,
            pols=("x",), n_cells_tag=f"numchk{res}")
        A = np.asarray(A_si[0], float)
        per_band = [float(np.sum(w[m] * A[m])) for m in masks]
        eta_num = float(np.sum(w * A))
        E = eta_num / eta_planar_tmm
        rows.append((res, per_band, eta_num, E))
        dE = "" if len(rows) == 1 else \
            f"{100*(E/rows[0][3] - 1):+11.2f}%"
        print(f"{res:5d} {per_band[0]:9.5f} {per_band[1]:9.5f} "
              f"{per_band[2]:9.5f} {eta_num:9.5f} {E:8.4f} {dE:>12} "
              f"{info['runtime_s'][0]:8.1f}s")

    print("\n[interpretation]")
    b1 = [r[1][0] for r in rows]
    drift_b1 = 100 * (b1[-1] / b1[0] - 1)
    drift_E = 100 * (rows[-1][3] / rows[0][3] - 1)
    print(f"  band-1 numerator eta drift, res {RES_LADDER[0]} -> "
          f"{RES_LADDER[-1]} : {drift_b1:+.2f}%")
    print(f"  E drift over the same range               : {drift_E:+.2f}%")
    print(f"  (planar band-1 eta drifted -8.2% over this range)")
    if abs(drift_b1) < 3.0:
        print("\n  VERDICT: NUMERATOR IS GRID-INSENSITIVE. The patterned "
              "structure's blue-band\n  absorption is broad and near-"
              "saturated -- unlike the planar stack's sharp\n  Fabry-Perot "
              "fringes, which is what res 60 mangles. E at res 60 is SOUND.\n"
              "  The flat E-ladder and the RCWA anchor are now EXPLAINED, "
              "not coincidental.\n  Launch the FULL campaign.")
    elif abs(drift_b1 - (-8.2)) < 3.0:
        print("\n  VERDICT: NUMERATOR CARRIES THE SAME ERROR. Then E should "
              "have drifted ~-8%\n  across the ladder and it did not "
              "(observed 2.528/2.522/2.568/2.587).\n  Those two facts cannot "
              "both be true -- something is masking it. STOP and\n  "
              "reconcile before banking 1551 samples.")
    else:
        print(f"\n  VERDICT: PARTIAL ({drift_b1:+.2f}%). The numerator "
              "carries some blue-band error\n  but less than the planar "
              "stack. E inherits roughly this much. If |drift| is\n  within "
              "the 2.5% ladder uncertainty you already quote, document and "
              "proceed;\n  if larger, raise production resolution.")


if __name__ == "__main__":
    main()
