"""
check_res_production.py -- resolve the section-8 open question.

Measures the AM1.5G-photon-flux-WEIGHTED eta error (not pointwise max, not
the ratio E) of the PLANAR production stack, engine vs analytic fitted-TMM,
at a resolution ladder, via production's own pseudo-1D code path
(broadband_absorption_many on an empty hole list / tiny uniform cell).
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
NORM_DIR = os.path.join(C.OUT_DIR, "norm_cache_rescheck")


def weighted_eta_err(res, fits, adapters, wl, device):
    """Per-band weighted-eta gap, FDTD vs analytic TMM, same materials."""
    a_tiny = 4000.0 / res                      # 4 in-plane cells (1D-exact)
    A_si, A_par, _ = F.broadband_absorption_many(
        [[]], a_tiny, C.THICKNESS_NM, wl, fits, C.BUFFER_NM, res,
        1e-6, 800.0, NORM_DIR, device=device, bits=32, pols=("x",),
        n_cells_tag=f"rescheck{res}")
    A_fdtd = np.asarray(A_si[0], float)

    ref = planar_reference_stack(adapters["si"], adapters["zno"],
                                 adapters["ag"], wl, C.THICKNESS_NM,
                                 C.BUFFER_NM)
    A_tmm = np.asarray(ref["A_si"], float)

    w = solar_weight(wl)
    per_band, eta_f, eta_t = [], 0.0, 0.0
    for b, (lo, hi) in enumerate(BANDS):
        m = (wl >= lo) & (wl < hi) if b < len(BANDS) - 1 else \
            (wl >= lo) & (wl <= hi)
        if not np.any(m):
            per_band.append(0.0)
            continue
        ef = float(np.sum(w[m] * A_fdtd[m]))
        et = float(np.sum(w[m] * A_tmm[m]))
        per_band.append(ef - et)
    eta_f = float(np.sum(w * A_fdtd))
    eta_t = float(np.sum(w * A_tmm))
    return per_band, eta_f, eta_t


def main():
    tee("check_res_production", C.OUT_DIR)
    device = F.resolve_device(os.environ.get("PC_DEVICE", "auto"))
    print(C.describe())
    print(f"[device] {device}")
    print("\nMETRIC: AM1.5G-photon-flux-weighted eta of the PLANAR "
          "production stack,\n        FDTD pseudo-1D vs analytic fitted-TMM. "
          "Both use the SAME fitted\n        materials, so any gap is PURE "
          "grid error.  decay_tol=1e-6 (tight)\n        so ring-down "
          "truncation cannot contaminate the measurement.\n")

    fits, adapters, mats = fit_all()
    wl = C.raw_wavelength_grid()
    print(f"[grid] {len(wl)} wavelengths, {wl[0]:.0f}-{wl[-1]:.0f} nm\n")

    print(f"{'res':>5} {'cells/lam@400':>14} {'band1(pp)':>11} "
          f"{'band2(pp)':>11} {'band3(pp)':>11} {'total(pp)':>11} "
          f"{'eta_fdtd':>10} {'eta_tmm':>9}")
    print("-" * 88)
    rows = []
    for res in RES_LADDER:
        n400 = float(np.atleast_1d(adapters["si"].n(400.0))[0])
        cells = (400.0 / n400) / (1000.0 / res)
        pb, ef, et = weighted_eta_err(res, fits, adapters, wl, device)
        tot = sum(abs(x) for x in pb)
        rows.append((res, tot))
        print(f"{res:5d} {cells:14.1f} {100*pb[0]:11.3f} {100*pb[1]:11.3f} "
              f"{100*pb[2]:11.3f} {100*tot:11.3f} {ef:10.5f} {et:9.5f}")

    print("\n[interpretation]")
    d = dict(rows)
    e60 = d.get(60)
    finest = min(v for _, v in rows)
    if e60 is None:
        print("  (res 60 not in the ladder -- rerun with it to judge "
              "production.)")
        return
    print(f"  res-60 total weighted error : {100*e60:.3f} pp")
    print(f"  finest-res total            : {100*finest:.3f} pp")
    if 100 * e60 < 0.5:
        print("\n  VERDICT: CASE 1 -- res 60 resolves the absorber's "
              "blue-band index at the\n  level that matters for the weighted "
              "number. Absolute absorption %\n  claims stand as-is. Document "
              "this table in the numerics section and\n  launch the FULL "
              "campaign.")
    elif 100 * e60 < 1.5:
        print("\n  VERDICT: CASE 2 (mild) -- error is real and convergent. "
              "E, the E-vs-sigma\n  shape and sigma* are SAFE (grid error "
              "correlates between numerator and\n  denominator and largely "
              "cancels in the ratio). But every ABSOLUTE\n  absorption % you "
              f"quote inherits ~{100*e60:.1f} pp. Either quote them with that "
              "error\n  bar, or raise production to the first res where this "
              "drops below ~0.5 pp.")
    else:
        print("\n  VERDICT: CASE 2 (serious) -- res 60 is materially "
              "under-resolving Si in the\n  blue. Raise RESOLUTION before the "
              "FULL campaign (and re-run the E anchor\n  at the new res; a "
              "flat E-ladder does NOT clear this).")
    print("\n  NOTE: this measures the numerator and denominator "
          "SEPARATELY -- exactly\n  the point. The E-ladder was blind here "
          "because grid error cancels in\n  the ratio.")


if __name__ == "__main__":
    main()
