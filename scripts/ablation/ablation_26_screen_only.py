"""Ablation #26 -- screen-only search: does the CMA-ES tier earn its keep?

Context: the production search stacks four tiers (random baseline,
50k surrogate screen, CMA-ES, gradient).  The ends are justified:
random loses by ~1.7% verified E (#6) and the gradient tier is null.
The middle attribution is missing: does CMA-ES find anything the cheap
50k screen's best candidates do not?  This arm re-runs the champion
cell (jitter sigma = 0.15) with tiers baseline+screen ONLY (identical
budgets, kappa = 0.2, same bundle and export rules), then
FDTD-verifies the top 20.  Screen-only matching the production arm
(2.6472 +- 0.0025) means a cheap screen suffices and the pipeline
simplifies; a gap means the optimizer tier has a measured verified-E
value.  Arm means +- SE; per-candidate differences below the 0.30%
floor are not resolvable.

Numerics: production flags minus the cmaes/gradient tiers;
--no-calibration for the same reason as #5 (the calibration file
postdates the campaigns and would silently change the LCB).

Cost: ~30 min GPU design + ~7 h verification (20 solves, resumable).
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_26_screen_only.py --dry-run
    python3 scripts/ablation/ablation_26_screen_only.py           # design
    python3 scripts/ablation/ablation_26_screen_only.py --verify  # +FDTD
Re-run anytime to (re)print the comparison from whatever exists.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, CHAMPION, DEPLOYED_BUNDLE,   # noqa: E402
                    ENV_PY, REPO, cell_tag, fdtd_arm, run_cmd,
                    run_verify, tee_into)

PROD_DIR = os.path.join(REPO, "runs", "inverse_v2", cell_tag(*CHAMPION))

# production budgets for the two retained tiers (run_v2_campaigns.sh)
SCREEN_FLAGS = [
    "--tiers", "baseline", "screen",
    "--n-baseline", "5000",
    "--n-screen", "50000", "--screen-keep-frac", "0.2",
    "--export-top", "20",
]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_26_screen_only.py",
        description="Inverse design without the CMA-ES/gradient tiers.")
    ap.add_argument("--disorder-class", default=CHAMPION[0])
    ap.add_argument("--sigma", type=float, default=CHAMPION[1])
    ap.add_argument("--bundle", default=DEPLOYED_BUNDLE)
    ap.add_argument("--verify", action="store_true", default=False,
                    help="run FDTD verification after (or instead of, if "
                         "candidates already exist) the design step.")
    ap.add_argument("--dry-run", action="store_true", default=False)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tag = cell_tag(args.disorder_class, args.sigma)
    out_dir = os.path.join(ABLATION_DIR, f"screen_only_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    tee_into("ablation_26_screen_only", out_dir)

    # ---- design (skipped if already exported) ---------------------------
    manifest = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest):
        print(f"[skip] {manifest} exists -- design already done")
    else:
        cmd = [ENV_PY, "-u", "-m", "models.inverse_design",
               "--bundle", args.bundle,
               "--disorder-class", args.disorder_class,
               "--sigma", str(args.sigma),
               "--export-dir", out_dir,
               "--kappa", "0.2",
               "--no-calibration"] + SCREEN_FLAGS
        rc = run_cmd(cmd, dry_run=args.dry_run)
        if rc != 0:
            print("[abort] design step failed")
            return rc

    # ---- verify ----------------------------------------------------------
    vcsv = os.path.join(out_dir, "verification.csv")
    if args.verify and not os.path.exists(vcsv):
        rc = run_verify(out_dir, dry_run=args.dry_run)
        if rc != 0:
            print("[abort] verification failed")
            return rc

    # ---- comparison (whatever exists so far) -----------------------------
    prod_csv = os.path.join(PROD_DIR, "verification.csv")
    if os.path.exists(vcsv):
        abl = fdtd_arm(vcsv, label=f"screen-only ({tag})")
        prod = fdtd_arm(prod_csv, label="production 4-tier")
        print("\n=== screen-only ablation, verified arms (mean +- SE) ===")
        for a in (prod, abl):
            s = a["arm"]
            print(f"  {a['label']:28s} n={s['n']:2d}  E = {s['mean']:.4f} "
                  f"+- {s['se']:.4f}  max {s['max']:.4f}  "
                  f"claimable {s['n_claimable']}/{s['n']}")
        d = abl["arm"]["mean"] - prod["arm"]["mean"]
        se = (abl["arm"]["se"] ** 2 + prod["arm"]["se"] ** 2) ** 0.5
        print(f"\n  arm difference (screen-only - production): {d:+.4f} "
              f"+- {se:.4f} E")
        if abs(d) <= 2 * se:
            print("\nVERDICT: NULL at the arm level -- the 50k screen's "
                  "best candidates verify as well as the full 4-tier "
                  "pipeline in this cell; CMA-ES adds nothing measurable "
                  "here (scope: one cell, like #5/#7).")
        else:
            print("\nVERDICT: the arms differ beyond 2 SE -- the CMA-ES "
                  "tier has a measurable verified-E effect; cite this "
                  "arm table.")
    else:
        print("\n[note] no verification.csv yet -- run with --verify on a "
              "GPU node; predicted-only results are never quotable (the "
              "surrogate proposes, FDTD disposes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
