"""Ablation #18 -- squeeze-excitation attention OFF (`--attention none`).

Context: the deployed v2 surrogate is a residual CNN whose per-block
attention is the SE gate (global pool -> bottleneck MLP -> sigmoid
channel rescale) -- the only attention mechanism in the model.  This
run replaces SE with an identity no-op in every residual block
(ResidualBlock keeps its equational form; only the gate goes away) to
measure what the channel attention contributes to accuracy and
calibration.  Either outcome is publishable: "SE buys X" or "conv body
alone suffices at this dataset size".

Everything else is the deployed v2 recipe (NLL head, k-fold, 5 members,
seed 137, best_params).

Cost: ~1 training run (5 members).  Same order as #8/#9/#10/#11.
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_18_attn_none.py --dry-run
    python3 scripts/ablation/ablation_18_attn_none.py
    python3 scripts/ablation/ablation_18_attn_none.py --skip-train
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, evaluate_bundle,             # noqa: E402
                    run_training, standard_train_cli, tee_into)

OUT_DIR = os.path.join(ABLATION_DIR, "attn_none")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_18_attn_none.py",
        description="Retrain v2 recipe with the SE gate removed.")
    ap.add_argument("--skip-train", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--wandb", action="store_true", default=False,
                    help="log the training run to Weights & Biases "
                         "(ablation convention is offline; opt-in)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    tee_into("ablation_18_attn_none", OUT_DIR)

    # standard cli is the deployed recipe; the delta is the attention
    cli = standard_train_cli(OUT_DIR) + ["--attention", "none"]
    if args.wandb:
        cli = [f for f in cli if f != "--no-wandb"]
    if not args.skip_train:
        rc = run_training(cli, OUT_DIR, dry_run=args.dry_run)
        if rc != 0 or args.dry_run:
            return rc
    out = evaluate_bundle(os.path.join(OUT_DIR, "surrogate_bundle.pt"),
                          label="attn_none")
    row = out["rows"]["test/tta_on"]
    print(f"\nVERDICT: no-attention MAE {row['mae']:.6f}, rho "
          f"{row['rho_pooled']:+.3f}, PICP1 {row.get('picp_1s', float('nan')):.3f} "
          "-- compare to deployed v2 (MAE 0.005447, rho 0.701, PICP1 0.651). "
          "A material gap = the SE gate earns its Methods paragraph; "
          "parity = the residual conv body alone suffices at this size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())