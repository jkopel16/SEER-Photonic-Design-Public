"""Spectral-window robustness: does the blue band drive ANY conclusion?

Motivation: at production numerics the planar gate shows large POINTWISE
|dA| in the blue (fringe shift from grid dispersion).  The project's
claims survive because the error is common-mode across samples -- but
that argument deserves a direct test.  Every banked sample and every
res-120 cache entry stores its full A_si(lambda) spectrum, so E can be
recomputed under any spectral weighting WITHOUT re-solving anything.

This script recomputes E for the N=15 rank-fidelity panel under four
windows -- full (400-1100), no-blue (700-1100), fringe-free NIR only
(950-1100), and blue-only (400-700, diagnostic) -- at BOTH res 60 (bank
spectra) and res 120 (verification cache spectra), and answers:

  Q1  Sanity: does the full-window reconstruction reproduce the banked
      E labels exactly?  (validates the arithmetic)
  Q2  Do within-class RANKINGS depend on the window?  Spearman between
      full-window and restricted-window E at fixed resolution.
  Q3  Does the res-60 -> res-120 ranking-stability verdict (the 0.30 %
      floor) hold under each window?  Spearman(60,120), flips, largest
      flipped gap, common-mode shift -- per window.

If rankings and the floor are window-independent, the blue-band
pointwise error demonstrably drives no conclusion (one-line robustness
statement for the poster).  If they move, we learn it now, from cache,
for free.

Usage (CPU is fine, runs in seconds; bank defaults to
scripts/FDTD_solver/data_production):
    python reweight_robustness.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS_DIR))   # solver modules
sys.path.insert(0, _TESTS_DIR)                    # sibling test modules

from ladder_referee import cache_path, spearman, CACHE_DIR, DIAG_DIR  # noqa: E402
import config as C                                                    # noqa: E402
from optics_core import planar_reference_stack, solar_weight          # noqa: E402
from run_dataset import get_materials, _load_samples, OUT_DIR         # noqa: E402
from rank_fidelity_r120 import PANEL_SIDS, REFEREE_RES                # noqa: E402

WINDOWS = [("full_400-1100", 0.0, np.inf),
           ("no_blue_700-1100", 700.0, np.inf),
           ("nir_only_950-1100", 950.0, np.inf),
           ("blue_only_400-700", 0.0, 700.0)]


def window_E(wl, A, A_ref, w, lo, hi):
    m = (wl >= lo) & (wl < hi)
    if not m.any():
        return float("nan")
    return float(np.sum(w[m] * A[m]) / np.sum(w[m] * A_ref[m]))


def pair_stats(sids, Ea, Eb):
    """(spearman, n_flips, largest_flipped_gap_pct on basis A)."""
    flips, largest = 0, 0.0
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            da = (Ea[i] - Ea[j]) / Ea[j] * 100
            db = (Eb[i] - Eb[j]) / Eb[j] * 100
            if da * db < 0:
                flips += 1
                largest = max(largest, abs(da))
    return spearman(Ea, Eb), flips, largest


def main():
    print(f"[reweight] bank={OUT_DIR}  panel N={len(PANEL_SIDS)}  "
          f"windows={[w[0] for w in WINDOWS]}")
    m = get_materials()
    ad = m["adapters"]

    samples = {int(x["sample_id"]): x for x in _load_samples()}
    ref_cache = {}

    def reference(wl):
        key = (len(wl), float(wl[0]), float(wl[-1]))
        if key not in ref_cache:
            ref = planar_reference_stack(
                ad["si"], ad["zno"], ad["ag"], wl,
                C.THICKNESS_NM, C.BUFFER_NM)
            ref_cache[key] = (np.asarray(ref["A_si"], float),
                              solar_weight(wl))
        return ref_cache[key]

    # ---- collect spectra --------------------------------------------
    rows = []          # (sid, wl60, A60, wl120|None, A120|None, E_banked)
    for sid in PANEL_SIDS:
        x = samples.get(sid)
        if x is None:
            print(f"  [warn] sid {sid} not in bank; skipped")
            continue
        wl60 = np.asarray(x["wavelengths_nm"], float)
        A60 = np.asarray(x["A_si"], float)
        wl120 = A120 = None
        cp = cache_path(sid, REFEREE_RES)
        if os.path.exists(cp):
            z = np.load(cp)
            if "A_si" in z.files and "wavelengths_nm" in z.files:
                wl120 = np.asarray(z["wavelengths_nm"], float)
                A120 = np.asarray(z["A_si"], float)
            else:
                print(f"  [warn] res-{REFEREE_RES} cache for sid {sid} "
                      "lacks a stored spectrum; window analysis at 120 "
                      "skipped for it")
        rows.append((sid, wl60, A60, wl120, A120, float(x["E"])))

    # ---- per-window E at both resolutions ---------------------------
    E = {name: {"60": {}, "120": {}} for name, _, _ in WINDOWS}
    for sid, wl60, A60, wl120, A120, _ in rows:
        Aref60, w60 = reference(wl60)
        for name, lo, hi in WINDOWS:
            E[name]["60"][sid] = window_E(wl60, A60, Aref60, w60, lo, hi)
        if wl120 is not None:
            Aref120, w120 = reference(wl120)
            for name, lo, hi in WINDOWS:
                E[name]["120"][sid] = window_E(wl120, A120, Aref120,
                                               w120, lo, hi)

    sids = [r[0] for r in rows]

    # ---- Q1: sanity -------------------------------------------------
    recon = np.array([E["full_400-1100"]["60"][s] for s in sids])
    banked = np.array([r[5] for r in rows])
    print(f"\nQ1  full-window reconstruction vs banked E: "
          f"max |diff| = {np.max(np.abs(recon - banked)):.2e}  "
          f"(should be ~1e-6; if large, the label definition differs "
          f"from this reconstruction -- STOP and reconcile)")

    # ---- Q2: window-dependence of rankings at fixed res -------------
    print("\nQ2  ranking agreement vs full window (res 60):")
    full60 = np.array([E["full_400-1100"]["60"][s] for s in sids])
    for name, _, _ in WINDOWS[1:]:
        e = np.array([E[name]["60"][s] for s in sids])
        rho, flips, largest = pair_stats(sids, full60, e)
        print(f"    {name:<20} Spearman {rho:+.3f}   {flips}/"
              f"{len(sids) * (len(sids) - 1) // 2} flips   "
              f"largest flipped gap {largest:.2f}%")

    # ---- Q3: the floor verdict per window ---------------------------
    print(f"\nQ3  res-60 vs res-{REFEREE_RES} ranking stability, "
          "per window:")
    have120 = [s for s in sids if s in E["full_400-1100"]["120"]]
    if len(have120) < 5:
        print("    (fewer than 5 res-120 spectra cached -- rerun the "
              "res-120 panel with the current scripts, which store "
              "A_si, or accept Q2 as the available evidence)")
    else:
        print(f"    n = {len(have120)} samples with cached "
              f"res-{REFEREE_RES} spectra")
        for name, _, _ in WINDOWS:
            e60 = np.array([E[name]["60"][s] for s in have120])
            e120 = np.array([E[name]["120"][s] for s in have120])
            rho, flips, largest = pair_stats(have120, e60, e120)
            shift = (e120 / e60 - 1) * 100
            print(f"    {name:<20} Spearman {rho:+.3f}   {flips} flips  "
                  f" largest flipped gap {largest:.2f}%   "
                  f"shift {shift.mean():+.2f}% +/- {shift.std():.3f}%")

    # ---- outputs -----------------------------------------------------
    os.makedirs(DIAG_DIR, exist_ok=True)
    out = os.path.join(DIAG_DIR, "reweight_robustness.csv")
    with open(out, "w") as f:
        f.write("sid," + ",".join(f"E60_{n}" for n, _, _ in WINDOWS)
                + "," + ",".join(f"E120_{n}" for n, _, _ in WINDOWS)
                + "\n")
        for s in sids:
            f.write(str(s))
            for n, _, _ in WINDOWS:
                f.write(f",{E[n]['60'][s]:.6f}")
            for n, _, _ in WINDOWS:
                v = E[n]["120"].get(s, float("nan"))
                f.write(f",{v:.6f}")
            f.write("\n")
    print(f"\n[out] wrote {out}")
    print("\nReading the verdict: if Q2 restricted-window Spearman is "
          "~1 with flips only among near-ties, and Q3's flip/floor "
          "numbers match the full-window Test-9 result, the blue-band "
          "pointwise error drives no conclusion.  The blue_only row is "
          "diagnostic -- it isolates where fringe error lives and is "
          "EXPECTED to be the noisiest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
