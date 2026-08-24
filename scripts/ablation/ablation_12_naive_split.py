"""Ablation #12 -- grouped sigma-stratified split vs NAIVE random split.

Context: every reported surrogate metric stands on the grouped,
sigma-stratified split (model.py stratified_group_split).  This ablation
trains the identical v2 recipe on a PLAIN SHUFFLED 80/10/10 split -- no
grouping, no stratification, and likewise naive member folds -- to
measure how much a naive evaluation would have flattered the metrics.

DELIBERATE ASYMMETRY (pre-registered, must reach the table caption):
this row is evaluated on ITS OWN naive test set, not the production test
set.  The two rows are intentionally on different footing -- the
comparison measures the OPTIMISM OF NAIVE EVALUATION, i.e. what a paper
that split naively would have reported.  It is not a sample-for-sample
model comparison.  (On samples_128.npz, sample_id is unique, so grouping
per se is a no-op today; the active ingredients are sigma stratification
and the fold construction.  Say so honestly if the delta is small.)

Implementation: no naive-split code path exists in models/ (and models/
is not edited), so this script trains in-process via the factored member
loop (train_one imported from models.model; member seeds seed + i*137;
byte-identical hyperparameters via the best_params overlay), saves a
standard bundle + a split.json sidecar, and evaluates through the shared
evaluator (the sidecar tells it which split to use).

Cost: ~1 training run (5 members, in-process).
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    # smoke test (~minutes): exercises loop + bundle + sidecar evaluator
    /project/rise-batteries/photonics-fdtd/bin/python3 -u \
        scripts/ablation/ablation_12_naive_split.py --epochs 3 --ensemble 2
    # real run:
    /project/rise-batteries/photonics-fdtd/bin/python3 -u \
        scripts/ablation/ablation_12_naive_split.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, DATA_128, evaluate_bundle,   # noqa: E402
                    load_args, load_dataset, naive_split,
                    save_bundle, tee_into, train_members)

OUT_DIR = os.path.join(ABLATION_DIR, "naive_split")
NOTE = ("naive plain-shuffle split (no grouping, no sigma stratification, "
        "naive member folds); evaluated on its OWN test set -- measures "
        "the optimism of naive evaluation, deliberately not "
        "sample-for-sample comparable with grouped-split rows")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_12_naive_split.py",
        description="v2 recipe on a naive random split (import-loop).")
    ap.add_argument("--epochs", type=int, default=None,
                    help="debug override (smoke tests only)")
    ap.add_argument("--ensemble", type=int, default=5,
                    help="debug override (smoke tests only)")
    ap.add_argument("--seed", type=int, default=137)
    return ap.parse_args(argv)


def main(argv=None):
    args_cli = parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    tee_into("ablation_12_naive_split", OUT_DIR)

    bundle_path = os.path.join(OUT_DIR, "surrogate_bundle.pt")
    if os.path.exists(bundle_path):
        print(f"[skip] {bundle_path} exists -- evaluating only")
    else:
        # deployed v2 recipe args + best_params overlay
        args = load_args(["-i", DATA_128, "-o", OUT_DIR,
                          "--raster-only", "--fft-channel", "--nll-head",
                          "--kfold-members",
                          "--ensemble", str(args_cli.ensemble),
                          "--seed", str(args_cli.seed), "--no-wandb"])
        if args_cli.epochs is not None:
            args.epochs = args_cli.epochs
            print(f"[debug] epochs override -> {args.epochs} "
                  "(SMOKE TEST ONLY -- not a table row)")

        from models.model import normalize                    # noqa: E402
        data, X, y, recipe, groups = load_dataset(args)
        train_idx, val_idx, test_idx = naive_split(len(y), seed=args.seed)
        print(f"[split] NAIVE train={len(train_idx)} val={len(val_idx)} "
              f"test={len(test_idx)} (plain shuffle, seed {args.seed})")

        X_norm, y_norm, xm, xs, ymn, ysd = normalize(X, y, train_idx)
        members, best_vals = train_members(
            args, X_norm, y_norm, np.asarray(data["sigma"]), groups,
            train_idx, val_idx, kfold=True, naive_folds=True)

        save_bundle(OUT_DIR, members, best_vals, args,
                    (xm, xs, ymn, ysd), recipe, X.shape[-1])
        with open(os.path.join(OUT_DIR, "split.json"), "w") as f:
            json.dump({"note": NOTE,
                       "train": train_idx.tolist(),
                       "val": val_idx.tolist(),
                       "test": test_idx.tolist()}, f)
        print("[out] split.json sidecar written (evaluator uses it)")

    out = evaluate_bundle(bundle_path, label="naive_split")
    row = out["rows"]["test/tta_on"]
    print(f"\nVERDICT: naive-split (own test set) MAE {row['mae']:.6f}, "
          f"rho {row['rho_pooled']:+.3f}, PICP1 {row['picp_1s']:.3f} vs "
          "deployed grouped-split row (#4). If the naive numbers look "
          "BETTER, that gap is the leakage/selection optimism the grouped "
          "protocol prevents -- the caption must carry the own-test-set "
          "note verbatim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
