"""Ablation #24 -- self-attention at stage 4 only (`--attention sa4`).

Context: arm #21 (sa at every block) was the worst row in the table
(rho 0.488), but it is confounded twice: stage 1 attends over 16,384
tokens (a scale the module was never meant for), and the module swaps
the multiplicative gate for a LayerNorm residual at every stage.  This
arm is the fair test the module's docstring intended: SE stays at
stages 1-3, self-attention runs only at stage 4 (16x16 = 256 tokens,
where the receptive field question actually lives).  Cheap, and
isolates 'does long-range spatial mixing help where it can matter'
from #21's cost and normalization confounds.

Everything else is the deployed v2 recipe (NLL head, k-fold, 5 members,
seed 137, best_params).

Cost: ~1 training run (5 members), barely slower than SE.
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_24_attn_sa4.py --dry-run
    python3 scripts/ablation/ablation_24_attn_sa4.py
    python3 scripts/ablation/ablation_24_attn_sa4.py --skip-train
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, evaluate_bundle,             # noqa: E402
                    run_training, standard_train_cli, tee_into)

OUT_DIR = os.path.join(ABLATION_DIR, "attn_sa4")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_24_attn_sa4.py",
        description="Retrain v2 with self-attention at stage 4 only.")
    ap.add_argument("--skip-train", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--wandb", action="store_true", default=False,
                    help="log the training run to Weights & Biases "
                         "(ablation convention is offline; opt-in)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    tee_into("ablation_24_attn_sa4", OUT_DIR)

    cli = standard_train_cli(OUT_DIR) + ["--attention", "sa4"]
    if args.wandb:
        cli = [f for f in cli if f != "--no-wandb"]
    if not args.skip_train:
        rc = run_training(cli, OUT_DIR, dry_run=args.dry_run)
        if rc != 0 or args.dry_run:
            return rc
    out = evaluate_bundle(os.path.join(OUT_DIR, "surrogate_bundle.pt"),
                          label="attn_sa4")
    row = out["rows"]["test/tta_on"]
    print(f"\nVERDICT: stage-4 self-attn MAE {row['mae']:.6f}, rho "
          f"{row['rho_pooled']:+.3f}, PICP1 {row.get('picp_1s', float('nan')):.3f} "
          "-- compare to deployed SE (MAE 0.005447, rho 0.701) and #21 "
          "all-blocks sa (MAE 0.006693, rho 0.488). Beats SE = long-range "
          "mixing helps where the receptive field matters; parity/worse = "
          "the SE result stands unconfounded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
