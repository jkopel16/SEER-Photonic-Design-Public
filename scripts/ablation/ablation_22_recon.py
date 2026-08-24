"""Ablation #22 -- multi-task reconstruction head (`--recon-head`).

Context: the deployed v2 surrogate is single-task; with only 2,724
unique layouts, nothing stops the trunk from collapsing its bottleneck
into a class-level summary that suffices for E regression alone.  This
arm adds a light decoder from the pre-GAP stage-4 feature map back to
the 128x128 raster channel and a second loss term,
recon_lambda * MSE(recon, normalized channel 0), so the bottleneck must
keep the actual layout geometry recoverable (multi-task
regularization).  The regression head, forward(), validation loss and
early stopping are all UNCHANGED, so the arm stays comparable with
every other row: the reconstruction task acts on training gradients
only, and the saved bundle scores identically through every existing
consumer (the decoder rides along in the state_dict, rebuilt from the
arch's recon_head flag).

recon_lambda is fixed at the model.py default (0.1) for this arm; if
the arm looks promising, a 3-point lambda check (0.03/0.1/0.3) is the
cheap follow-up before any full sweep.

Everything else is the deployed v2 recipe (NLL head, k-fold, 5 members,
seed 137, best_params).

Cost: ~1 training run (5 members; decoder adds a few % step time).
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_22_recon.py --dry-run
    python3 scripts/ablation/ablation_22_recon.py
    python3 scripts/ablation/ablation_22_recon.py --skip-train
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, evaluate_bundle,             # noqa: E402
                    run_training, standard_train_cli, tee_into)

OUT_DIR = os.path.join(ABLATION_DIR, "recon_mtl")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_22_recon.py",
        description="Retrain v2 recipe with the multi-task recon head.")
    ap.add_argument("--skip-train", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--wandb", action="store_true", default=False,
                    help="log the training run to Weights & Biases "
                         "(ablation convention is offline; opt-in)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    tee_into("ablation_22_recon", OUT_DIR)

    # standard cli is the deployed recipe; the delta is the recon head
    cli = standard_train_cli(OUT_DIR) + ["--recon-head"]
    if args.wandb:
        cli = [f for f in cli if f != "--no-wandb"]
    if not args.skip_train:
        rc = run_training(cli, OUT_DIR, dry_run=args.dry_run)
        if rc != 0 or args.dry_run:
            return rc
    out = evaluate_bundle(os.path.join(OUT_DIR, "surrogate_bundle.pt"),
                          label="recon_mtl")
    row = out["rows"]["test/tta_on"]
    print(f"\nVERDICT: recon multi-task MAE {row['mae']:.6f}, rho "
          f"{row['rho_pooled']:+.3f}, PICP1 {row.get('picp_1s', float('nan')):.3f} "
          "-- compare to deployed v2 (MAE 0.005447, rho 0.701, PICP1 0.651). "
          "Better = the reconstruction task keeps useful geometry in the "
          "bottleneck (regularization pays); parity/worse = E regression "
          "already retains what it needs at this dataset size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
