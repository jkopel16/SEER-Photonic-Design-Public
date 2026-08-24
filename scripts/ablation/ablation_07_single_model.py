"""Ablation #7 -- ensemble vs single-model SEARCH.

Context: the production optimizer searches against the 5-member v2
ensemble.  Ablations #8/#9/#13 measure what ensembling buys for
PREDICTION; this one isolates what it buys for SEARCH: the optimizer
runs against a single member (a sliced 1-member bundle) and the top 20
are FDTD-verified in the champion cell.

Design choice (stated up front): the search uses kappa = 0.  A 1-member
v2 bundle still emits an aleatoric s, but that is not the production
epistemic mixture -- keeping any s-term would confound "no ensemble"
with "different uncertainty penalty".  kappa = 0 removes the s-term from
the fitness entirely, so the comparison is plain single-model mean vs
the production ensemble LCB, with ablation #5 (ensemble mean, kappa = 0)
as the bridge arm separating the two effects.

PRE-REGISTERED INTERPRETATION: shares #5's fallback rule -- if BOTH #5
and #7 verify clean (null), the abstract's "safeguards against
exploitation" softens to "designed to prevent exploitation"; if either
shows exploitation, the phrasing stands.  Arm means +- SE; sub-floor
per-candidate differences are "not resolvable".

Cost: slice free; ~1-2 h GPU design + ~5 h verification (20 solves).
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_07_single_model.py --slice-only  # smoke
    python3 scripts/ablation/ablation_07_single_model.py              # design
    python3 scripts/ablation/ablation_07_single_model.py --verify     # +FDTD
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, CHAMPION, DEPLOYED_BUNDLE,   # noqa: E402
                    REPO, cell_tag, fdtd_arm, run_inverse,
                    run_verify, slice_bundle, tee_into)

PROD_DIR = os.path.join(REPO, "runs", "inverse_v2", cell_tag(*CHAMPION))
SLICED = os.path.join(ABLATION_DIR, "single_member_bundle.pt")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_07_single_model.py",
        description="Inverse design against a single ensemble member.")
    ap.add_argument("--disorder-class", default=CHAMPION[0])
    ap.add_argument("--sigma", type=float, default=CHAMPION[1])
    ap.add_argument("--slice-only", action="store_true", default=False,
                    help="write + sanity-load the 1-member bundle, then "
                         "stop (smoke test; no design launched).")
    ap.add_argument("--verify", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tag = cell_tag(args.disorder_class, args.sigma)
    out_dir = os.path.join(ABLATION_DIR, f"single_member_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    tee_into("ablation_07_single_model", out_dir)

    # ---- 1-member bundle --------------------------------------------------
    if not os.path.exists(SLICED):
        slice_bundle(DEPLOYED_BUNDLE, SLICED, keep=1)
    else:
        print(f"[skip] {SLICED} exists")
    # sanity-load through the production scorer (prints "1-model ensemble").
    # ON CPU deliberately: this parent process stays alive while the design
    # subprocess runs, and on exclusive-process GPUs a parent-held CUDA
    # context makes the child fail with "device busy" (observed 2026-08-04).
    import torch                                               # noqa: E402
    from models.inverse_design import SurrogateScorer          # noqa: E402
    SurrogateScorer(SLICED, torch.device("cpu"), use_tta=True, kappa=0.0,
                    calibration=None)
    if args.slice_only:
        print("\nVERDICT: slice written and loads through SurrogateScorer "
              "-- smoke test passed, no design launched.")
        return 0

    # ---- design ------------------------------------------------------------
    manifest = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest):
        print(f"[skip] {manifest} exists -- design already done")
    else:
        rc = run_inverse(out_dir, args.disorder_class, args.sigma,
                         SLICED, kappa=0.0, dry_run=args.dry_run)
        if rc != 0:
            print("[abort] design step failed")
            return rc

    # ---- verify + comparison -----------------------------------------------
    vcsv = os.path.join(out_dir, "verification.csv")
    if args.verify and not os.path.exists(vcsv):
        rc = run_verify(out_dir, dry_run=args.dry_run)
        if rc != 0:
            return rc

    if os.path.exists(vcsv):
        single = fdtd_arm(vcsv, label=f"single member, kappa=0 ({tag})")
        prod = fdtd_arm(os.path.join(PROD_DIR, "verification.csv"),
                        label="5-member LCB (production)")
        arms = [prod, single]
        k0 = os.path.join(ABLATION_DIR, f"kappa0_{tag}", "verification.csv")
        if os.path.exists(k0):
            arms.insert(1, fdtd_arm(k0, label="5-member mean, kappa=0 (#5)"))
        print("\n=== search ablation, verified arms (mean +- SE) ===")
        for a in arms:
            s = a["arm"]
            print(f"  {a['label']:34s} n={s['n']:2d}  E = {s['mean']:.4f} "
                  f"+- {s['se']:.4f}  max {s['max']:.4f}  "
                  f"claimable {s['n_claimable']}/{s['n']}")
        print("\n  read DOWN the three arms: production -> #5 isolates the "
              "s-term, #5 -> #7 isolates the ensemble mean itself.")
        print("\nVERDICT: apply the pre-registered rule jointly with #5 "
              "(module docstring) -- decide the abstract wording from the "
              "two nulls/non-nulls, not from this arm alone.")
    else:
        print("\n[note] no verification.csv yet -- run with --verify; "
              "predicted-only results are never quotable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
