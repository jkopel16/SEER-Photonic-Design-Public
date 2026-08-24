"""Ablation #23 -- structure factor ONLY (`--fft-only`).

Context: the input 2x2 is incomplete.  We have raster + |FFT| (deployed,
rho 0.701) and raster only (#11, rho 0.564), but never the mirror: can
the structure factor ALONE carry the prediction?  Physically, light
couples into guided modes through the pattern's spatial frequencies, so
if E is (approximately) a functional of |FFT| alone, this arm should
approach the deployed score; the gap it leaves measures what real-space
information (phase, local geometry) contributes beyond the power
spectrum.  Either outcome sharpens the paper's central claim about WHY
the physics-informed channel works.

The input is fft_onfly_v1 derived from the raster (the raster channel
itself is dropped, so input_shape=1).  Everything else is the deployed
v2 recipe (NLL head, k-fold, 5 members, seed 137, best_params).

Cost: ~1 training run (5 members).
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_23_fft_only.py --dry-run
    python3 scripts/ablation/ablation_23_fft_only.py
    python3 scripts/ablation/ablation_23_fft_only.py --skip-train
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, evaluate_bundle,             # noqa: E402
                    run_training, standard_train_cli, tee_into)

OUT_DIR = os.path.join(ABLATION_DIR, "fft_only")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_23_fft_only.py",
        description="Retrain v2 recipe on the structure factor alone.")
    ap.add_argument("--skip-train", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--wandb", action="store_true", default=False,
                    help="log the training run to Weights & Biases "
                         "(ablation convention is offline; opt-in)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    tee_into("ablation_23_fft_only", OUT_DIR)

    # deployed recipe, but the input is |FFT| alone: --fft-only replaces
    # --fft-channel (mutually exclusive; --raster-only still strips the
    # baked channel first so fft_onfly_v1 derives from the raster)
    cli = standard_train_cli(OUT_DIR, extra=["--fft-only"])
    cli = [f for f in cli if f != "--fft-channel"]
    if args.wandb:
        cli = [f for f in cli if f != "--no-wandb"]
    if not args.skip_train:
        rc = run_training(cli, OUT_DIR, dry_run=args.dry_run)
        if rc != 0 or args.dry_run:
            return rc
    out = evaluate_bundle(os.path.join(OUT_DIR, "surrogate_bundle.pt"),
                          label="fft_only")
    row = out["rows"]["test/tta_on"]
    print(f"\nVERDICT: fft-only MAE {row['mae']:.6f}, rho "
          f"{row['rho_pooled']:+.3f}, PICP1 {row.get('picp_1s', float('nan')):.3f} "
          "-- compare to deployed both-channels (MAE 0.005447, rho 0.701) "
          "and #11 raster-only (MAE 0.006286, rho 0.564). Near-deployed = "
          "E is essentially a functional of the power spectrum; a gap = "
          "real-space information contributes beyond |FFT|.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
