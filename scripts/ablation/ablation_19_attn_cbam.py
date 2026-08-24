"""Ablation #19 -- CBAM attention (`--attention cbam`, Woo et al. 2018).

Context: the deployed v2 surrogate gates channels with SE (global pool
-> bottleneck MLP -> sigmoid).  CBAM keeps that channel gate and adds a
spatial gate -- a 7x7 conv over the stacked {avgpool, maxpool} along the
channel axis -> sigmoid -> per-pixel rescale -- testing whether adding a
WHERE to SE's WHAT shifts accuracy or calibration on this dataset.

Everything else is the deployed v2 recipe (NLL head, k-fold, 5 members,
seed 137, best_params).  Parameters: +392 over SE (1,218,378 vs
1,217,986 at the deployed width 128), i.e. +0.03%.

Cost: ~1 training run (5 members).  Same order as #8/#9/#10/#11.
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_19_attn_cbam.py --dry-run
    python3 scripts/ablation/ablation_19_attn_cbam.py
    python3 scripts/ablation/ablation_19_attn_cbam.py --skip-train
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, evaluate_bundle,             # noqa: E402
                    run_training, standard_train_cli, tee_into)

OUT_DIR = os.path.join(ABLATION_DIR, "attn_cbam")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_19_attn_cbam.py",
        description="Retrain v2 recipe with CBAM attention.")
    ap.add_argument("--skip-train", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--wandb", action="store_true", default=False,
                    help="log the training run to Weights & Biases "
                         "(ablation convention is offline; opt-in)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    tee_into("ablation_19_attn_cbam", OUT_DIR)

    cli = standard_train_cli(OUT_DIR) + ["--attention", "cbam"]
    if args.wandb:
        cli = [f for f in cli if f != "--no-wandb"]
    if not args.skip_train:
        rc = run_training(cli, OUT_DIR, dry_run=args.dry_run)
        if rc != 0 or args.dry_run:
            return rc
    out = evaluate_bundle(os.path.join(OUT_DIR, "surrogate_bundle.pt"),
                          label="attn_cbam")
    row = out["rows"]["test/tta_on"]
    print(f"\nVERDICT: CBAM MAE {row['mae']:.6f}, rho "
          f"{row['rho_pooled']:+.3f}, PICP1 {row.get('picp_1s', float('nan')):.3f} "
          "-- compare to deployed v2 (MAE 0.005447, rho 0.701, PICP1 0.651). "
          "Beat SE = spatial gating earns its paragraph; parity = the "
          "channel gate carries the attention value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())