"""Ordered lattice at res 120: close the last hole in the grid-shift story.

Context (audit Tests 2/4/8/9): every rung's ABSOLUTE E is grid-dependent
and V-shaped -- disordered samples drop ~-4.8 % at res 90 then rebound to
-1.66 % net at res 120 (Test 9, n = 15).  The ordered sigma = 0 cell,
however, has only ever been solved at res 60 (2.4569) and res 90
(2.4122, twice, independent runs).  Its res-120 value is UNKNOWN, so the
ordered-vs-disordered refinement asymmetry -- and with it, how much of
the Fig-3 cross-class gap (2.457 -> 2.668, +8.6 % at res 60) is grid
artifact -- has never been measured at the rung the project trusts.

This script solves that one sample at res 120 and prints the verdict:

  * ordered 60->120 shift, next to the disordered -1.66 % (Test 9)
  * the asymmetry = difference of those two shifts
  * the implied res-120-corrected cross-class gap, using the champion's
    class shift for the numerator and the ordered shift for the
    denominator (an estimate: the champion itself has not been re-solved;
    that is the separate res-120 elite verification task)

Numerics are IDENTICAL to ladder_referee.py / rank_fidelity_r120.py
(same solve_rung: campaign wavelength grid, decay_tol, cap x1.5 at
res 120, same norm-cache tag) -- so the res-120 norm cache from those
runs is reused and the numbers are directly comparable to the Test-8/9
tables.

Cost: ONE sample at res 120, ~2 GPU-h.  Kill/resume safe (result is
cached exactly like the ladder solves; rerun to pick up the cache).
Run inside tmux on a GPU node:

    cd scripts/FDTD_solver/tests
    python ordered_r120.py
    python ordered_r120.py --also-90   # optional: re-derive 2.4122 too

Caveat to carry into any writeup (same as Test 8): res 120 is the best
referee we have, not proven truth -- absolute E still oscillates across
rungs.  This measurement bounds the ordered-vs-disordered asymmetry at
the referee rung; it does not mint a converged absolute E.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS_DIR))   # solver modules
sys.path.insert(0, _TESTS_DIR)                    # sibling test modules

from ladder_referee import (cache_path, solve_rung, CACHE_DIR,      # noqa: E402
                            DIAG_DIR)
import config as C                                                  # noqa: E402
from logutil import tee                                             # noqa: E402
from optics_core import (planar_reference_stack, solar_weight)      # noqa: E402
from run_dataset import get_materials, OUT_DIR                      # noqa: E402

# Logged constants this test is judged against (audit sections in braces)
E90_ORDERED = 2.4122          # Tests 2 & 4, two independent runs
DISORDERED_60_120 = -1.66     # % -- Test 9 common-mode shift, n = 15
DISORDERED_60_90 = -4.79      # % -- Test 8 common-mode shift, n = 5
CHAMPION_E60 = 2.6675         # jitter s=0.15 champion (runs/inverse_v2)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python ordered_r120.py",
        description="Re-solve the sigma = 0 ordered lattice at res 120.")
    ap.add_argument("--also-90", action="store_true", default=False,
                    help="also re-solve at res 90 (sanity vs the logged "
                         "2.4122; ~40 min extra).")
    ap.add_argument("--device", default=None)
    return ap.parse_args(argv)


def find_ordered():
    """The banked sigma = 0 sample, located by class (not sample id)."""
    labels = os.path.join(OUT_DIR, "labels.csv")
    hit = next((r for r in csv.DictReader(open(labels))
                if r["class"] == "ordered"), None)
    if hit is None:
        raise SystemExit("no 'ordered' row in labels.csv")
    sid = int(hit["sample_id"])
    z = np.load(os.path.join(OUT_DIR, "samples", f"sample_{sid:06d}.npz"),
                allow_pickle=False)
    return sid, z


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(DIAG_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tee("ordered_r120", DIAG_DIR)     # mirrors all stdout to a log file
    log = print

    sid, x = find_ordered()
    E60 = float(x["E"])
    log(f"[panel] ordered lattice sid={sid}  banked E60={E60:.4f}")
    if abs(E60 - 2.4570) > 2e-3:
        log(f"[warn] banked E60 {E60:.4f} != audit's 2.4570 -- "
            "check you are on the production bank")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        log("[warn] no GPU -- a res-120 solve on CPU is impractical")

    m = get_materials()
    ad = m["adapters"]
    wl = C.raw_wavelength_grid()
    weights = solar_weight(wl)
    ref = planar_reference_stack(ad["si"], ad["zno"], ad["ag"], wl,
                                 C.THICKNESS_NM, C.BUFFER_NM)

    rungs = ([90] if args.also_90 else []) + [120]
    E = {60: E60, 90: E90_ORDERED}
    for res in rungs:
        cp = cache_path(sid, res)
        if os.path.exists(cp):
            E[res] = float(np.load(cp)["E"])
            log(f"[cache] res {res}: E = {E[res]:.4f} (reused {cp})")
            continue
        log(f"[solve] res {res} ... (cap x{1.5 if res > 90 else 1.0}, "
            f"expect ~{2.0 * (res / 120) ** 4:.1f} h)")
        e, secs, A = solve_rung(x, res, device, m, wl, weights, ref)
        np.savez_compressed(cp, E=e, A_si=A, res=res, sid=sid)
        E[res] = e
        log(f"[solve] res {res}: E = {e:.4f}  ({secs / 3600:.2f} h)")

    # ---- verdict ---------------------------------------------------------
    d6090 = 100 * (E[90] / E[60] - 1)
    d60120 = 100 * (E[120] / E[60] - 1)
    log("\n=== ordered lattice across rungs ===")
    log(f"  E60 = {E[60]:.4f}   E90 = {E[90]:.4f}   E120 = {E[120]:.4f}")
    log(f"  60->90  : {d6090:+.2f} %   (disordered common mode "
        f"{DISORDERED_60_90:+.2f} %, Test 8)")
    log(f"  60->120 : {d60120:+.2f} %   (disordered common mode "
        f"{DISORDERED_60_120:+.2f} %, Test 9)")

    asym = d60120 - DISORDERED_60_120
    log(f"\n  ordered-vs-disordered 60->120 asymmetry: {asym:+.2f} pts")

    # implied correction to the res-60 cross-class gap (estimate only:
    # champion not itself re-solved -- that is the elite-verification task)
    gap60 = 100 * (CHAMPION_E60 / E[60] - 1)
    champ120 = CHAMPION_E60 * (1 + DISORDERED_60_120 / 100)
    gap120 = 100 * (champ120 / E[120] - 1)
    log(f"  cross-class gap: {gap60:+.1f} % at res 60 -> ~{gap120:+.1f} % "
        f"implied at res 120")
    log("  (champion scaled by the Test-9 class shift; solve the champion "
        "itself at res 120 -- --res120-top -- before quoting this)")

    if abs(asym) <= 0.5:
        log("\nVERDICT: shifts are symmetric within ~sampling spread -- the "
            "res-60 cross-class gap is essentially real; the res-90 "
            "asymmetry was that rung's own oscillation.")
    else:
        log(f"\nVERDICT: a {asym:+.2f}-pt asymmetry survives at the referee "
            "rung -- quote the cross-class gap with the res-120 correction, "
            "never the res-60 (or res-90) number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
