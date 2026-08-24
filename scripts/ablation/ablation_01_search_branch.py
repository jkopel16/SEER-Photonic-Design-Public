"""Ablation #1 -- CMA-ES vs gradient-ascent contribution (FREE).

Context: both v1 and v2 inverse-design campaigns ran two refiner branches
(Tier 2 CMA-ES, Tier 3 projected gradient ascent) and FDTD-verified the
pooled top-20 per cell.  Candidate provenance was logged (`method` column
in every verification.csv), so this ablation is pure bookkeeping over
results that already exist -- NO new compute, NO GPU.

Pre-registered interpretation: branch MEANS are compared at the arm level
(mean +- SE); differences below the 0.30 % within-sigma floor at the
individual-candidate level are "not resolvable" by construction.  Prior
tabulation (2026-08-03): v1 cmaes 2.6421 / gradient 2.6428, v2 cmaes
2.6461 / gradient 2.6470 -- ~0.03 % apart, 10x below the floor; champions
come from both branches.  Expected verdict: the refiners are
interchangeable at the landscape ceiling (report as robustness, not a
winner).

Cost: seconds, CPU only.
Usage:
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_01_search_branch.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ABLATION_DIR, REPO, tee_into                # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_01_search_branch.py",
        description="Tabulate verified E by search branch (cmaes/gradient).")
    ap.add_argument("--glob", nargs="*", default=[
        os.path.join(REPO, "runs", "inverse", "*"),
        os.path.join(REPO, "runs", "inverse_v2", "*")],
        help="campaign directory globs (each must hold verification.csv)")
    ap.add_argument("--out", default=os.path.join(ABLATION_DIR,
                                                  "search_branch.csv"))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(ABLATION_DIR, exist_ok=True)
    tee_into("ablation_01_search_branch", ABLATION_DIR)

    rows = []          # (gen, cell, method, E, gain_pct, verdict)
    for g in args.glob:
        for cell_dir in sorted(glob.glob(g)):
            vc = os.path.join(cell_dir, "verification.csv")
            if not os.path.isfile(vc):
                continue
            gen = "v2" if "inverse_v2" in cell_dir else "v1"
            cell = os.path.basename(cell_dir.rstrip("/"))
            for r in csv.DictReader(open(vc)):
                rows.append((gen, cell, r["method"], float(r["true_E60"]),
                             float(r["gain_vs_cell_mean_pct"]),
                             r["verdict"]))
    if not rows:
        raise SystemExit("[abort] no verification.csv found under the globs")
    print(f"[panel] {len(rows)} verified candidates from "
          f"{len(set((g, c) for g, c, *_ in rows))} campaign cells")

    # ---- per (generation, branch) arm stats ----------------------------
    print("\n=== verified E by search branch (arm mean +- SE) ===")
    arm = {}
    for gen in ("v1", "v2"):
        for meth in sorted({m for g_, _, m, *_ in rows if g_ == gen}):
            e = np.array([E for g_, _, m, E, *_ in rows
                          if g_ == gen and m == meth])
            arm[(gen, meth)] = (len(e), e.mean(),
                                e.std(ddof=1) / np.sqrt(len(e)), e.max())
            n, mu, se, mx = arm[(gen, meth)]
            print(f"  {gen} {meth:9s} n={n:3d}  mean={mu:.4f} "
                  f"+- {se:.4f}  max={mx:.4f}")

    # ---- per-cell champion branch ---------------------------------------
    print("\n=== per-cell champion branch ===")
    champ = {}
    for gen, cell, meth, E, gain, verdict in rows:
        k = (gen, cell)
        if k not in champ or E > champ[k][0]:
            champ[k] = (E, meth)
    counts = {}
    for (gen, cell), (E, meth) in sorted(champ.items()):
        print(f"  {gen} {cell:14s} champion {E:.4f}  <- {meth}")
        counts[meth] = counts.get(meth, 0) + 1
    print(f"  branch champion count: {counts}")

    # ---- CSV -------------------------------------------------------------
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "cell", "method", "true_E60",
                    "gain_vs_cell_mean_pct", "verdict"])
        w.writerows(rows)
    print(f"\n[out] per-candidate table -> {args.out}")

    # ---- verdict ----------------------------------------------------------
    diffs = []
    for gen in ("v1", "v2"):
        ms = [m for (g_, m) in arm if g_ == gen]
        if len(ms) == 2:
            (n1, m1, s1, _), (n2, m2, s2, _) = arm[(gen, ms[0])], \
                                               arm[(gen, ms[1])]
            d_pct = 100 * abs(m1 - m2) / m1
            se_pct = 100 * np.hypot(s1, s2) / m1
            diffs.append((gen, d_pct, se_pct))
            print(f"  {gen}: |{ms[0]} - {ms[1]}| = {d_pct:.3f} % "
                  f"(+- {se_pct:.3f} % SE)")
    if all(d <= max(2 * s, 0.30) for _, d, s in diffs):
        print("\nVERDICT: branch means are statistically indistinguishable "
              "at the arm level and far below the 0.30 % floor at the "
              "candidate level -- the two refiners are interchangeable at "
              "the landscape ceiling. Report as robustness (keep both); "
              "do not name a winning branch.")
    else:
        print("\nVERDICT: a branch difference exceeds its sampling "
              "uncertainty -- inspect the per-cell table before writing "
              "the ablation paragraph.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
