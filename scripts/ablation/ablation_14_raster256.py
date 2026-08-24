"""Ablation #14 -- input raster resolution: 128 px vs 256 px.

Context: the deployed pipeline rasterizes each layout at 128 px
(dx ~ 35.5 nm/px over the 4.55 um supercell); hole-radius perturbations
at low sigma are only a few px.  This run rebuilds the dataset at 256 px
and retrains the identical v2 recipe to test whether raster quantization
limits accuracy.  (Carried as an optional item in the audit/handoff
since 08-02; folded into the ablation suite here.)

Recipe identity at 256 px: training uses --raster-only --fft-channel, so
the FFT channel is computed ON THE FLY from whatever raster it gets
(model.py resolve_input) -- byte-identical pipeline, just finer pixels.
The dataset is therefore built with --no-fft-channel (no baked channel
to drop; halves the npz).  The CNN is fully convolutional + GAP, so no
architecture change.

NOTE: step (i) writes data/samples_256.npz -- the ONE ablation output
outside runs/ablation/ (it is a dataset, and reusable).  Expect ~4x the
128 px training cost per epoch; if the GPU OOMs pass e.g.
--batch-size 16 (best_params batch is 32).

Cost: dataset build (CPU, ~minutes) + ~4x one training run.
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_14_raster256.py --dry-run
    python3 scripts/ablation/ablation_14_raster256.py
    python3 scripts/ablation/ablation_14_raster256.py --skip-train
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, ENV_PY, REPO,                # noqa: E402
                    evaluate_bundle, run_cmd, run_training,
                    standard_train_cli, tee_into)

OUT_DIR = os.path.join(ABLATION_DIR, "raster256")
DATA_256 = os.path.join(REPO, "data", "samples_256.npz")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_14_raster256.py",
        description="256 px raster ablation: rebuild dataset + retrain.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override best_params batch size (OOM relief).")
    ap.add_argument("--skip-train", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    tee_into("ablation_14_raster256", OUT_DIR)

    # ---- step (i): dataset build ------------------------------------------
    if os.path.exists(DATA_256):
        print(f"[skip] {DATA_256} exists")
    else:
        rc = run_cmd([ENV_PY, "-u", "-m", "models.build_dataset",
                      "-i", os.path.join(REPO, "data", "samples"),
                      "--img-size", "256", "--no-fft-channel",
                      "-o", DATA_256], dry_run=args.dry_run)
        if rc != 0:
            print("[abort] dataset build failed")
            return rc

    # ---- step (ii): retrain -------------------------------------------------
    # NB a plain CLI --batch-size would be clobbered by the --use-best
    # overlay (model.py:1181-1185), so the override patches the per-run
    # copy of best_params.json instead.
    if not args.skip_train:
        rc = run_training(standard_train_cli(OUT_DIR, data=DATA_256),
                          OUT_DIR, dry_run=args.dry_run,
                          patch_params=({"batch_size": args.batch_size}
                                        if args.batch_size else None))
        if rc != 0 or args.dry_run:
            return rc
    out = evaluate_bundle(os.path.join(OUT_DIR, "surrogate_bundle.pt"),
                          data_path=DATA_256, label="raster256")
    row = out["rows"]["test/tta_on"]
    print(f"\nVERDICT: 256 px MAE {row['mae']:.6f}, rho "
          f"{row['rho_pooled']:+.3f} vs deployed 128 px (0.005447, "
          "+0.701). Parity = 128 px raster quantization is not the "
          "accuracy bottleneck (labels/data are); improvement = raster "
          "resolution was leaving accuracy on the table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
