"""Ablation #21 -- multi-head spatial self-attention (`--attention sa`).

Context: SE is a channel-only gate -- it re-weights WHAT, never WHERE.
This variant swaps it for multi-head self-attention over the spatial
tokens (residual + LayerNorm, 4 heads) at every residual block.  It is
BY FAR the heaviest attention candidate: pooling happens AFTER each
stage, so the per-stage token counts are 128x128 = 16,384 -> 4,096 ->
1,024 -> 256, and stage 1's quadratic attention dominates everything
(it only fits in memory via the fused SDPA kernel).  If training OOMs,
lower the batch size via run_training's patch_params (a plain
--batch-size flag would be clobbered by the --use-best overlay).  The
receptive field is as wide as the whole feature map -- the strongest
possible test of 'does WHERE matter for E'.  Two disclosed confounds
versus se/none/cbam/eca: this module REPLACES the block output with a
LayerNorm'd attention residual (a normalization change, on top of the
attention change) rather than gating multiplicatively.

Everything else is the deployed v2 recipe (NLL head, k-fold, 5 members,
seed 137, best_params).

Cost: ~1 training run (5 members), MUCH slower than SE (attention is
quadratic in spatial dim; ~+20% params at the deployed width 128).
Run LAST in any batch.
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_21_attn_sa.py --dry-run
    python3 scripts/ablation/ablation_21_attn_sa.py
    python3 scripts/ablation/ablation_21_attn_sa.py --skip-train
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, evaluate_bundle,             # noqa: E402
                    run_training, standard_train_cli, tee_into)

OUT_DIR = os.path.join(ABLATION_DIR, "attn_sa")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_21_attn_sa.py",
        description="Retrain v2 recipe with spatial self-attention.")
    ap.add_argument("--skip-train", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--wandb", action="store_true", default=False,
                    help="log the training run to Weights & Biases "
                         "(ablation convention is offline; opt-in)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    tee_into("ablation_21_attn_sa", OUT_DIR)

    cli = standard_train_cli(OUT_DIR) + ["--attention", "sa"]
    if args.wandb:
        cli = [f for f in cli if f != "--no-wandb"]
    if not args.skip_train:
        rc = run_training(cli, OUT_DIR, dry_run=args.dry_run)
        if rc != 0 or args.dry_run:
            return rc
    out = evaluate_bundle(os.path.join(OUT_DIR, "surrogate_bundle.pt"),
                          label="attn_sa")
    row = out["rows"]["test/tta_on"]
    print(f"\nVERDICT: self-attn MAE {row['mae']:.6f}, rho "
          f"{row['rho_pooled']:+.3f}, PICP1 {row.get('picp_1s', float('nan')):.3f} "
          "-- compare to deployed v2 (MAE 0.005447, rho 0.701, PICP1 0.651). "
          "Unusual for a CNN surrogate; either a gap (spatial mixing "
          "matters beyond channel gating) or parity at this data size "
          "is honest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())