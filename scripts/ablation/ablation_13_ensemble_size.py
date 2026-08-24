"""Ablation #13 -- ensemble size sweep (1, 3, 5 members), prediction only.

Context: how do accuracy, ranking and the UQ columns scale with ensemble
size?  No FDTD involved (the search-side value of the ensemble is
ablation #7's question).

CONFOUND AVOIDED (pre-registered): the sweep uses SHARED splits, not the
deployed k-fold rotation -- under k-fold, each member trains on
(1 - 1/k) of the pool, so k = 1/3/5 would change per-member training
data (0/67/80 % of pool) at the same time as ensemble size.  (k = 1 also
trips model.py's ensemble >= 2 gate.)  The table caption must therefore
say: this sweep is INDICATIVE of ensemble-size scaling, not a
measurement of the k-fold production model; the k=5 shared row is
ablation #8's run, not deployed v2.

The k = 1 row exercises the evaluator's single-member branch: epistemic
variance identically 0, s = aleatoric head only.

Cost: 1 + 3 = 4 member trainings (~0.8x one production run); the k = 5
row is REUSED from ablation #8 (same configuration -- NLL + shared +
5 members) rather than retrained.
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_13_ensemble_size.py --dry-run
    python3 scripts/ablation/ablation_13_ensemble_size.py
    python3 scripts/ablation/ablation_13_ensemble_size.py --skip-train
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, evaluate_bundle,             # noqa: E402
                    run_training, standard_train_cli, tee_into)

KFOLD_OFF_DIR = os.path.join(ABLATION_DIR, "kfold_off")   # = k5 shared row


def out_dir_for(k):
    return (KFOLD_OFF_DIR if k == 5
            else os.path.join(ABLATION_DIR, f"ens_k{k}"))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_13_ensemble_size.py",
        description="Ensemble size sweep, shared splits, NLL head.")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--skip-train", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(ABLATION_DIR, exist_ok=True)
    tee_into("ablation_13_ensemble_size", ABLATION_DIR)

    # phase 1: ALL trainings first (subprocesses). Parent stays CUDA-free
    # until no more children will be launched -- on exclusive-process GPUs
    # a parent-held CUDA context (from an evaluation) makes the next
    # training subprocess fail with "device busy" (observed 2026-08-04).
    if not args.skip_train:
        for k in args.sizes:
            if k == 5:
                continue        # k=5 shared row is ablation #8's training
            out_dir = out_dir_for(k)
            print(f"\n################ train ensemble k = {k} ############")
            rc = run_training(
                standard_train_cli(out_dir, kfold=False, ensemble=k),
                out_dir, dry_run=args.dry_run)
            if rc != 0:
                print(f"[warn] training failed for k={k}; continuing")

    # phase 2: evaluations (parent may take the GPU now)
    rows = {}
    for k in args.sizes:
        if args.dry_run:
            continue
        bundle = os.path.join(out_dir_for(k), "surrogate_bundle.pt")
        print(f"\n################ ensemble k = {k} ################")
        if not os.path.exists(bundle):
            print(f"[warn] no bundle at {bundle}"
                  + (" (train it via ablation_08_kfold_off.py)"
                     if k == 5 else "") + "; skipping eval")
            continue
        out = evaluate_bundle(bundle, label=f"ens_k{k}_shared")
        rows[k] = out["rows"]["test/tta_on"]

    if rows:
        print("\n=== ensemble size sweep (shared splits, TTA on) ===")
        print(f"  {'k':>2s} {'MAE':>10s} {'RMSE':>10s} {'rho':>7s} "
              f"{'RMS(s)/RMSE':>12s} {'PICP1':>7s} {'s_epi':>9s}")
        for k in sorted(rows):
            r = rows[k]
            epi = r.get("rms_s_epi")
            print(f"  {k:2d} {r['mae']:10.6f} {r['rmse']:10.6f} "
                  f"{r['rho_pooled']:+7.3f} {r['rms_s_over_rmse']:12.3f} "
                  f"{r['picp_1s']:7.3f} "
                  f"{epi if epi is not None else 0.0:9.6f}")
        print("\nVERDICT: caption MUST carry the pre-registered note -- "
              "shared-split sweep, indicative of size scaling only; the "
              "deployed model is k-fold-rotated and is NOT on this curve "
              "(its row lives in ablation #4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
