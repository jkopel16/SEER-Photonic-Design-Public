"""Ablation #20 -- ECA attention (`--attention eca`, Wang et al. 2020).

Context: the deployed v2 surrogate gates channels with SE -- a global
pool -> bottleneck MLP (reduction 8) -> sigmoid.  ECA replaces the MLP
with a 1D conv over the channel descriptor whose kernel size follows
the published heuristic (k=5 at the deployed width 128), no reduction:
5 params per block versus SE's 4,240 at width 128.  Tests whether the
bottleneck/reduction in SE matters, or whether a shuffling-only conv
suffices (or wins) on this dataset.

Everything else is the deployed v2 recipe (NLL head, k-fold, 5 members,
seed 137, best_params).

Cost: ~1 training run (5 members).  Same order as #8/#9/#10/#11.
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_20_attn_eca.py --dry-run
    python3 scripts/ablation/ablation_20_attn_eca.py
    python3 scripts/ablation/ablation_20_attn_eca.py --skip-train
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, evaluate_bundle,             # noqa: E402
                    run_training, standard_train_cli, tee_into)

OUT_DIR = os.path.join(ABLATION_DIR, "attn_eca")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_20_attn_eca.py",
        description="Retrain v2 recipe with ECA attention.")
    ap.add_argument("--skip-train", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--wandb", action="store_true", default=False,
                    help="log the training run to Weights & Biases "
                         "(ablation convention is offline; opt-in)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    tee_into("ablation_20_attn_eca", OUT_DIR)

    cli = standard_train_cli(OUT_DIR) + ["--attention", "eca"]
    if args.wandb:
        cli = [f for f in cli if f != "--no-wandb"]
    if not args.skip_train:
        rc = run_training(cli, OUT_DIR, dry_run=args.dry_run)
        if rc != 0 or args.dry_run:
            return rc
    out = evaluate_bundle(os.path.join(OUT_DIR, "surrogate_bundle.pt"),
                          label="attn_eca")
    row = out["rows"]["test/tta_on"]
    print(f"\nVERDICT: ECA MAE {row['mae']:.6f}, rho "
          f"{row['rho_pooled']:+.3f}, PICP1 {row.get('picp_1s', float('nan')):.3f} "
          "-- compare to deployed v2 (MAE 0.005447, rho 0.701, PICP1 0.651). "
          "ECA ~ SE = the MLP bottleneck is unnecessary; ECA < SE = the "
          "bottleneck caps overfitting and earns its keep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())