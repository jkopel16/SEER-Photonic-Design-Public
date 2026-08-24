"""
diag_radius_class.py -- is the radius class broken, or only sigma=0.020?

radius/sigma=0.020 is 100% quarantined (9/9, A_si -> 1e5..1e10, all
cap=True).  The campaign never reached radius sigma >= 0.04, so we do not
yet know whether this is ONE dead grid cell or a THIRD of the factorial
design.  This solves one radius sample per sigma at production numerics
and reports which survive.

Answers the only question that changes campaign scope.
"""

from __future__ import annotations

import os
import sys
import numpy as np


# solver modules live one directory up from tests/
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS_DIR))

import config as C
from logutil import tee
from materials_gpu import fit_all
from optics_core import solar_weight, planar_reference_stack
import fdtd_torch as F
from run_dataset import build_plan, make_record

SIGMAS_TO_TEST = [float(s) for s in
                  os.environ.get("PC_DIAG_SIGMAS",
                                 "0.02,0.04,0.10,0.20,0.30").split(",")]
# Diagnostics land in tests/gpu_out_diag regardless of CWD (PC_DIAG overrides).
DIAG_DIR = os.environ.get("PC_DIAG") or os.path.join(_TESTS_DIR, "gpu_out_diag")
NORM_DIR = os.path.join(DIAG_DIR, "norm_cache_r60")


def main():
    os.makedirs(DIAG_DIR, exist_ok=True)
    tee("diag_radius_class", DIAG_DIR)
    device = F.resolve_device(os.environ.get("PC_DEVICE", "auto"))
    print(C.describe())
    print(f"[device] {device}\n")

    plan = build_plan()
    fits, ad, mats = fit_all()
    wl = C.raw_wavelength_grid()
    w = solar_weight(wl)
    ref = planar_reference_stack(ad["si"], ad["zno"], ad["ag"], wl,
                                 C.THICKNESS_NM, C.BUFFER_NM)
    eta_p = float(np.sum(w * np.asarray(ref["A_si"], float)))

    print(f"{'sid':>6} {'sigma':>6} {'A_si max':>11} {'cap':>6} "
          f"{'E':>8} {'runtime':>9}  verdict")
    print("-" * 72)
    results = []
    for sig in SIGMAS_TO_TEST:
        row = next((p for p in plan if p["class"] == "radius"
                    and abs(p["sigma"] - sig) < 1e-9), None)
        if row is None:
            print(f"{'--':>6} {sig:6.3f}   (not in manifest)")
            continue
        rec = make_record(row)
        holes, a_sup = rec["holes"], rec["a_super_nm"]
        A_si, A_par, info = F.broadband_absorption_many(
            [holes], a_sup, C.THICKNESS_NM, wl, fits, C.BUFFER_NM,
            C.RESOLUTION, C.DECAY_TOL, C.MAX_TIME, NORM_DIR,
            device=device, pols=("x",),
            n_cells_tag=f"diagr{int(round(1000*sig))}")
        A = np.asarray(A_si[0], float)
        amax = float(A.max())
        cap = bool(info["hit_time_cap"][0])
        E = float(np.sum(w * np.clip(A, 0, 1))) / eta_p
        ok = amax < 1.05
        results.append((sig, ok, amax))
        print(f"{row['sample_id']:6d} {sig:6.3f} {amax:11.4g} "
              f"{str(cap):>6} {E:8.4f} {info['runtime_s'][0]:8.0f}s  "
              f"{'OK' if ok else 'EXPLODED'}")

    bad = [s for s, ok, _ in results if not ok]
    good = [s for s, ok, _ in results if ok]
    print(f"\n[verdict]")
    print(f"  survived : {good}")
    print(f"  exploded : {bad}")
    if bad == [0.02] or (bad and max(bad) <= 0.04):
        print("\n  ONE DEAD CELL (low-sigma radius only). The radius class "
              "is usable.\n  Options: (a) drop sigma=0.02 from the radius "
              "class -- at res 60 a 2%\n  radius perturbation is +-4.5 nm "
              "on a 16.7 nm grid, i.e. SUB-PIXEL, so it\n  is arguably not "
              "a meaningful experimental condition anyway; (b) run just\n  "
              "that cell at higher res. Either is defensible; (a) is free.")
    elif not bad:
        print("\n  NOTHING EXPLODED. The sigma=0.02 failures were "
              "seed-specific, not\n  systematic. Re-run those 9 samples "
              "and move on.")
    else:
        print("\n  THE RADIUS CLASS IS BROADLY UNSTABLE at res 60. This is "
              "a scope decision:\n  raise resolution for the whole class "
              "(cost) or drop the class from the\n  factorial design "
              "(halves your class count -- jitter + random only).\n  Do NOT "
              "launch the full campaign before deciding.")


if __name__ == "__main__":
    main()
