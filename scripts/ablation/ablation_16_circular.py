"""Ablation #16 -- circular conv padding + cyclic-shift augmentation.

Context: the physics is exactly invariant under cyclic translations of the
periodic supercell, but the deployed model uses zero-padded convolutions
and shows a measured prediction spread of ~0.013 in E under random integer
rolls (runs/interpretability/validation/validation.json, step 5), versus a
D4 per-view spread of ~0.002 and a within-cell width of ~0.04.  This arm
tests the identified remedy: padding_mode="circular" on every conv
(matching the torus topology) plus training-time cyclic-shift augmentation
(raster channel rolled; the |FFT| channel is mathematically shift-invariant
and is left untouched, labels reused exactly).

Everything else is the deployed v2 recipe with the same best_params.json;
NO new hyperparameter sweep, so any delta is attributable to the padding
and augmentation alone.  This arm is a side experiment: the deployed
bundle stays the production model regardless of outcome.

Cost: ~1 training run (5 members) + minutes of evaluation.
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_16_circular.py --dry-run
    python3 scripts/ablation/ablation_16_circular.py
    python3 scripts/ablation/ablation_16_circular.py --skip-train
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, REPO, ENV_PY, evaluate_bundle,  # noqa: E402
                    run_cmd, run_training, standard_train_cli, tee_into)

OUT_DIR = os.path.join(ABLATION_DIR, "circular_shift")
DEPLOYED_SHIFT_JSON = os.path.join(
    REPO, "runs", "interpretability", "validation", "validation.json")
VALIDATE = os.path.join(REPO, "scripts", "interpretability",
                        "validate_saliency.py")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_16_circular.py",
        description="Retrain v2 recipe with circular padding + shift aug.")
    ap.add_argument("--skip-train", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--device", default="cuda",
                    help="device for the shift-spread test (default cuda)")
    ap.add_argument("--wandb", action="store_true", default=False,
                    help="log the training run to Weights & Biases "
                         "(ablation convention is offline; opt-in)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    tee_into("ablation_16_circular", OUT_DIR)

    if not args.skip_train:
        cli = standard_train_cli(OUT_DIR,
                                 extra=["--circular-padding", "--shift-aug"])
        if args.wandb:
            cli = [f for f in cli if f != "--no-wandb"]
        rc = run_training(cli, OUT_DIR, dry_run=args.dry_run)
        if rc != 0 or args.dry_run:
            return rc

    bundle = os.path.join(OUT_DIR, "surrogate_bundle.pt")

    # shift-spread test FIRST (subprocess): on exclusive-process GPUs it
    # cannot get a CUDA context once this parent claims the device in
    # evaluate_bundle below.
    shift_dir = os.path.join(OUT_DIR, "shift_test")
    rc = run_cmd([ENV_PY, VALIDATE, "--steps", "5",
                  "--bundle", bundle, "--val-dir", shift_dir,
                  "--device", args.device])
    if rc != 0:
        return rc

    out = evaluate_bundle(bundle, label="circular_shift")
    on = out["rows"]["test/tta_on"]
    new = json.load(open(os.path.join(shift_dir, "validation.json")))
    new_spread = new["step5"]["mean_spread"]
    ref_spread = None
    if os.path.exists(DEPLOYED_SHIFT_JSON):
        ref = json.load(open(DEPLOYED_SHIFT_JSON))
        ref_spread = ref.get("step5", {}).get("mean_spread")

    print(f"\nVERDICT: circular+shift arm -- test tta_on MAE "
          f"{on['mae']:.6f} rho {on['rho_pooled']:+.3f} "
          f"RMS(s)/RMSE {on.get('rms_s_over_rmse', float('nan')):.3f} "
          "(deployed reference: MAE 0.005440 rho +0.701 ratio 1.005). "
          f"Circular-shift mean spread {new_spread:.4f} in E"
          + (f" (deployed: {ref_spread:.4f})." if ref_spread else ".")
          + " Deployed bundle remains the production model; this arm "
          "answers the invariance remedy question only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
