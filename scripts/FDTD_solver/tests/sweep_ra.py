"""
sweep_ra.py -- is r/a = 0.35 optimal on THIS engine (torch-FDTD), not
just on the retired RCWA stage?

Solves the ordered 1x1 unit cell at r/a = 0.30 ... 0.40 at production
numerics, computes E for each with the analytic TMM denominator, and
says whether 0.35 is the argmax within label jitter.

CAVEAT baked in: the ordered cell is the WORST case for staircase jitter
(audit section 3), so each point carries ~+-2-3%. A flat-ish curve
across 0.325-0.375 means "broad optimum, 0.35 is inside it" -- that is a
valid answer, not a failed sweep.

OUTCOME (audit Test 4): confirmed -- a broad plateau over r/a =
0.325-0.35 on torch-FDTD (7-11% falloff on the flanks); the frozen 0.35
design sits inside it, and the plateau ordering reverses between res 60
and 90.  The accurate claim is "inside a plateau", not "the argmax".
"""

from __future__ import annotations

import os
import sys
import numpy as np


# solver modules live one directory up from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from logutil import tee
from materials_gpu import fit_all
from optics_core import solar_weight, planar_reference_stack
import fdtd_torch as F

RA_LIST = [float(x) for x in
           os.environ.get("PC_RA_LIST", "0.30,0.325,0.35,0.375,0.40")
           .split(",")]
NORM_DIR = os.path.join(C.OUT_DIR, "norm_cache_rasweep")


def main():
    tee("sweep_ra", C.OUT_DIR)
    device = F.resolve_device(os.environ.get("PC_DEVICE", "auto"))
    print(C.describe())
    print(f"[device] {device}")
    print(f"[sweep]  ordered 1x1 unit cell, res={C.RESOLUTION}, "
          f"r/a = {RA_LIST}\n")

    fits, ad, mats = fit_all()
    wl = C.raw_wavelength_grid()
    w = solar_weight(wl)
    ref = planar_reference_stack(ad["si"], ad["zno"], ad["ag"], wl,
                                 C.THICKNESS_NM, C.BUFFER_NM)
    eta_p = float(np.sum(w * np.asarray(ref["A_si"], float)))
    print(f"[denominator] eta_TMM(planar) = {eta_p:.5f}\n")

    print(f"{'r/a':>7} {'r (nm)':>8} {'eta_num':>9} {'E':>8} "
          f"{'cap?':>5} {'runtime':>9}")
    print("-" * 52)
    results = []
    for ra in RA_LIST:
        holes, a_sup = F.ordered_square_holes(C.A_NM, 1, ra * C.A_NM)
        A_si, A_par, info = F.broadband_absorption_many(
            [holes], a_sup, C.THICKNESS_NM, wl, fits, C.BUFFER_NM,
            C.RESOLUTION, C.DECAY_TOL, C.MAX_TIME, NORM_DIR,
            device=device, pols=("x",),
            n_cells_tag=f"ra{int(round(1000*ra))}")
        A = np.asarray(A_si[0], float)
        eta = float(np.sum(w * A))
        E = eta / eta_p
        results.append((ra, E))
        print(f"{ra:7.3f} {ra*C.A_NM:8.1f} {eta:9.5f} {E:8.4f} "
              f"{str(bool(info['hit_time_cap'][0])):>5} "
              f"{info['runtime_s'][0]:8.1f}s")

    # ---- verdict ---------------------------------------------------------
    ras = np.array([r for r, _ in results])
    Es = np.array([e for _, e in results])
    best = ras[np.argmax(Es)]
    E035 = float(Es[np.argmin(np.abs(ras - 0.35))])
    Emax = float(Es.max())
    gap = 100 * (Emax / E035 - 1)
    JITTER = 2.5  # percent, the staircase label jitter on ordered cells

    print(f"\n[verdict]")
    print(f"  argmax r/a          : {best:.3f}   (E = {Emax:.4f})")
    print(f"  frozen  r/a = 0.350 : E = {E035:.4f}")
    print(f"  gap best vs 0.35    : {gap:+.2f}%   "
          f"(ordered-cell label jitter ~ +-{JITTER}%)")
    if best == 0.35 or gap <= JITTER:
        print("\n  KEEP r/a = 0.35. Either it is the argmax outright or "
              "the gap to the\n  best point is inside the staircase "
              "jitter -- indistinguishable at\n  production numerics. "
              "The RCWA-chosen design is confirmed on this\n  engine; "
              "cite this sweep in the writeup and launch the campaign.")
    else:
        print(f"\n  RCWA's 0.35 is NOT the argmax here (best {best:.3f}, "
              f"{gap:+.1f}% better,\n  outside jitter). DECISION NEEDED "
              "before the FULL campaign: re-freeze\n  the design at the "
              "new r/a (invalidates the mini-dataset; fresh PC_OUT)\n  "
              "or keep 0.35 and soften the 'optimized periodic baseline' "
              "claim to\n  'near-optimal'. Do not bank 1551 samples "
              "before choosing.")

    # ---- figure ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(ras, Es, yerr=Es * JITTER / 100, fmt="o-", lw=1.5,
                capsize=4, color="#1f5fa8")
    ax.axvline(0.35, color="grey", ls=":", lw=1.2,
               label="frozen design (RCWA sweep; plateau confirmed on torch-FDTD)")
    ax.set_xlabel("r/a")
    ax.set_ylabel("E (ordered 1x1, torch-FDTD, res %d)" % C.RESOLUTION)
    ax.set_title("r/a re-sweep on the production engine\n"
                 "(bars = +-2.5% ordered-cell staircase jitter)")
    ax.legend()
    fig.tight_layout()
    figp = os.path.join(C.OUT_DIR, "figs", "fig_ra_resweep.png")
    os.makedirs(os.path.dirname(figp), exist_ok=True)
    fig.savefig(figp, dpi=160)
    print(f"\n  figure -> {figp}")


if __name__ == "__main__":
    main()
