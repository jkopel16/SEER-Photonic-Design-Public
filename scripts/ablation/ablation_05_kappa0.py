"""Ablation #5 -- LCB safeguard: kappa = 0 vs the production kappa = 0.2.

Context: the production inverse design maximizes the lower confidence
bound LCB = mean - 0.2*s, a safeguard against the optimizer exploiting
surrogate error (winner's curse).  This ablation re-runs the search with
plain ensemble mean as fitness (kappa = 0) in the champion cell (jitter
sigma = 0.15), all other production flags identical, then FDTD-verifies
the top 20.  The figure is the per-candidate PREDICTED vs VERIFIED gap:
if kappa = 0 exploits model error, predicted E rises while verified E
falls relative to the production arm (runs/inverse_v2/jitter_s015).

PRE-REGISTERED INTERPRETATION (decided before results, with #7):
`surrogate_bias_on_champions` is already NEGATIVE in every production
cell at kappa = 0.2, so a null result -- kappa = 0 also verifying clean
-- is live.  If BOTH #5 and #7 come back null, the abstract phrase
"with safeguards against exploitation of surrogate error" softens to
"designed to prevent exploitation of surrogate error" (a description of
the method, not a demonstrated necessity).  If either shows exploitation,
the original phrasing stands with this ablation as evidence.  Arm means
are reported +- SE; per-candidate differences below the 0.30 % floor are
"not resolvable".

Numerics: identical to the production campaign (same bundle, seed 0,
tiers, budgets; --no-calibration because uq/calibration.json postdates
the campaigns and would silently auto-load and change the LCB).

Cost: ~1-2 h GPU design + ~5 h verification (20 solves, resumable).
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_05_kappa0.py --dry-run   # inspect
    python3 scripts/ablation/ablation_05_kappa0.py             # design
    python3 scripts/ablation/ablation_05_kappa0.py --verify    # +FDTD
Re-run anytime to (re)print the comparison from whatever exists.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, CHAMPION, DEPLOYED_BUNDLE,   # noqa: E402
                    REPO, cell_tag, fdtd_arm, run_inverse,
                    run_verify, tee_into)

PROD_DIR = os.path.join(REPO, "runs", "inverse_v2", cell_tag(*CHAMPION))
KAPPA_PROD = 0.2


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_05_kappa0.py",
        description="Inverse design with kappa = 0 (no LCB safeguard).")
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
    out_dir = os.path.join(ABLATION_DIR, f"kappa0_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    tee_into("ablation_05_kappa0", out_dir)

    # ---- design (skipped if already exported) ---------------------------
    manifest = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest):
        print(f"[skip] {manifest} exists -- design already done")
    else:
        rc = run_inverse(out_dir, args.disorder_class, args.sigma,
                         args.bundle, kappa=0.0, dry_run=args.dry_run)
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
        abl = fdtd_arm(vcsv, label=f"kappa=0 ({tag})")
        prod = fdtd_arm(prod_csv, label=f"kappa={KAPPA_PROD} (production)")
        print("\n=== kappa ablation, verified arms (mean +- SE) ===")
        for a in (prod, abl):
            s = a["arm"]
            print(f"  {a['label']:28s} n={s['n']:2d}  E = {s['mean']:.4f} "
                  f"+- {s['se']:.4f}  max {s['max']:.4f}  "
                  f"claimable {s['n_claimable']}/{s['n']}")
        print("\n  per-candidate predicted-vs-verified table: "
              f"{vcsv} (the pred/true gap is the exploitation figure)")
        d = abl["arm"]["mean"] - prod["arm"]["mean"]
        se = (abl["arm"]["se"] ** 2 + prod["arm"]["se"] ** 2) ** 0.5
        print(f"\n  arm difference (kappa0 - production): {d:+.4f} "
              f"+- {se:.4f} E")
        if abs(d) <= 2 * se:
            print("\nVERDICT: NULL at the arm level -- kappa = 0 verified "
                  "as well as the production LCB run. Apply the "
                  "pre-registered fallback TOGETHER with #7's outcome "
                  "(see module docstring); do not rationalize post hoc.")
        else:
            print("\nVERDICT: the arms differ beyond 2 SE -- the LCB "
                  "safeguard has a measurable verified-E effect; the "
                  "abstract's 'safeguards' phrasing stands, cite this arm "
                  "table.")
    else:
        print("\n[note] no verification.csv yet -- run with --verify on a "
              "GPU node to produce the verified arm; predicted-only "
              "results are never quotable (the surrogate proposes, FDTD "
              "disposes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
