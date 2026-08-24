"""Ablation #15 -- leave-one-sigma-cell-out generalization.

Context: the inverse design searches WITHIN disorder cells the surrogate
was trained on; the never-exercised active-learning loop would matter
most if the model merely memorized cells.  This ablation trains the v2
recipe with one whole (class, sigma) cell excluded and evaluates BOTH on
the usual in-distribution test split (of the remaining data) AND on the
held-out cell: can the surrogate interpolate across disorder strength?

Default holdout: jitter sigma = 0.125 (an interior cell -- both
neighbours, 0.10 and 0.15, stay in training, so this tests
INTERpolation; hold out 0.15 or an edge cell to probe extrapolation).

Reading the result (pre-registered): the holdout row's within-sigma rho
IS the OOD generalization number; its MAE also absorbs any cell-mean
offset.  Compare against the same cell's rho in the deployed model's
per-cell table (#4's JSON) -- deployed-trained-on-it vs held-out is the
memorize-vs-interpolate gap.  n per cell is ~155, so single-cell rho
carries real sampling noise (rho std ~ 0.08 under the null); read the
magnitude, not the second decimal.

Implementation: custom split -> in-process training via the factored
member loop (same as #12): stratified_group_split on the REMAINDER
(indices mapped back), k-fold member rotation as deployed, best_params
hyperparameters, bundle + split.json sidecar with the holdout as an
extra eval set.

Cost: ~1 training run (5 members, in-process).
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    # smoke test (~minutes):
    /project/rise-batteries/photonics-fdtd/bin/python3 -u \
        scripts/ablation/ablation_15_cell_holdout.py --epochs 3 --ensemble 2
    # real run (default holdout jitter s=0.125):
    /project/rise-batteries/photonics-fdtd/bin/python3 -u \
        scripts/ablation/ablation_15_cell_holdout.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, DATA_128, cell_tag,          # noqa: E402
                    evaluate_bundle, load_args, load_dataset,
                    save_bundle, tee_into, train_members)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_15_cell_holdout.py",
        description="Train v2 recipe with one disorder cell held out.")
    ap.add_argument("--holdout-class", default="jitter",
                    choices=["jitter", "radius"])
    ap.add_argument("--holdout-sigma", type=float, default=0.125)
    ap.add_argument("--epochs", type=int, default=None,
                    help="debug override (smoke tests only)")
    ap.add_argument("--ensemble", type=int, default=5,
                    help="debug override (smoke tests only)")
    ap.add_argument("--seed", type=int, default=137)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    tag = cell_tag(a.holdout_class, a.holdout_sigma)
    out_dir = os.path.join(ABLATION_DIR, f"holdout_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    tee_into("ablation_15_cell_holdout", out_dir)
    holdout_key = f"holdout_{tag}"

    bundle_path = os.path.join(out_dir, "surrogate_bundle.pt")
    if os.path.exists(bundle_path):
        print(f"[skip] {bundle_path} exists -- evaluating only")
    else:
        args = load_args(["-i", DATA_128, "-o", out_dir,
                          "--raster-only", "--fft-channel", "--nll-head",
                          "--kfold-members",
                          "--ensemble", str(a.ensemble),
                          "--seed", str(a.seed), "--no-wandb"])
        if a.epochs is not None:
            args.epochs = a.epochs
            print(f"[debug] epochs override -> {args.epochs} "
                  "(SMOKE TEST ONLY -- not a table row)")

        from models.model import (normalize,                   # noqa: E402
                                  stratified_group_split)
        data, X, y, recipe, groups = load_dataset(args)
        sig = np.asarray(data["sigma"], dtype=float)
        cls = np.asarray(data["disorder_class"])
        hold = np.flatnonzero((cls == a.holdout_class)
                              & np.isclose(sig, a.holdout_sigma))
        if len(hold) == 0:
            raise SystemExit(f"[abort] no samples in cell {tag}")
        rest = np.setdiff1d(np.arange(len(y)), hold)
        print(f"[holdout] {tag}: {len(hold)} samples held out entirely; "
              f"{len(rest)} remain for the standard split")

        # grouped sigma-stratified split of the remainder, mapped back
        g_rest = (np.asarray(groups)[rest] if groups is not None else None)
        tr_r, va_r, te_r = stratified_group_split(
            sig[rest], groups=g_rest, seed=args.seed)
        train_idx, val_idx, test_idx = rest[tr_r], rest[va_r], rest[te_r]
        print(f"[split] train={len(train_idx)} val={len(val_idx)} "
              f"test={len(test_idx)} (+{len(hold)} holdout)")

        X_norm, y_norm, xm, xs, ymn, ysd = normalize(X, y, train_idx)
        members, best_vals = train_members(
            args, X_norm, y_norm, sig, groups, train_idx, val_idx,
            kfold=True, naive_folds=False)

        save_bundle(out_dir, members, best_vals, args,
                    (xm, xs, ymn, ysd), recipe, X.shape[-1])
        with open(os.path.join(out_dir, "split.json"), "w") as f:
            json.dump({"note": f"cell holdout: {tag} excluded from "
                               "train/val/test; extra set = the held-out "
                               "cell (OOD generalization)",
                       "train": train_idx.tolist(),
                       "val": val_idx.tolist(),
                       "test": test_idx.tolist(),
                       "extra": {holdout_key: hold.tolist()}}, f)
        print("[out] split.json sidecar written (holdout as extra set)")

    out = evaluate_bundle(bundle_path, label=f"holdout_{tag}")
    tst = out["rows"]["test/tta_on"]
    ood = out["rows"][f"{holdout_key}/tta_on"]
    print(f"\nVERDICT: in-distribution test rho {tst['rho_pooled']:+.3f} "
          f"(MAE {tst['mae']:.6f}); HELD-OUT cell {tag} rho "
          f"{ood['rho_pooled']:+.3f} (MAE {ood['mae']:.6f}, PICP1 "
          f"{ood['picp_1s']:.3f}). Compare the holdout rho against the "
          "deployed model's per-cell rho for the same cell (#4 JSON): "
          "parity = the surrogate interpolates across disorder strength; "
          "collapse = cell memorization, and within-cell-only search "
          "claims must say so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
