"""Ablation #9 -- loss head: SmoothL1 point-estimate vs beta-NLL.

Context: the paper's heteroscedastic beta-NLL loss is nonstandard for a
regression surrogate and buys the calibrated self-predicted error s.
This ablation trains the remaining two corners of the 2x2
{loss head} x {member data policy} (see ablation #8 for the grid):

    SmoothL1 + k-fold  -> runs/ablation/smoothl1_kfold/
    SmoothL1 + shared  -> runs/ablation/smoothl1_shared/  (~ the v1 recipe
                          under the v2 hyperparameters/seed)

Honesty notes, pre-registered:
  * The point-estimate loss is SmoothL1 (Huber, beta from best_params),
    NOT plain MSE -- the table row is named "SmoothL1", not "MSE".
  * A SmoothL1 bundle has no learned s; its s column is ensemble member
    DISAGREEMENT only (exactly what v1 historically used), flagged by the
    evaluator in `notes`.  PICP from disagreement-only s is expected to
    be badly under-covered -- that expected failure IS the result.
  * --kfold-members is not gated on the loss head (model.py:1232 checks
    only ensemble >= 2), so both corners run through the stock trainer.
  * Fairness: identical early-stopping PROTOCOL (patience 20), not
    identical epoch counts -- the two losses converge on different
    schedules.  var-warmup is NLL-arm-only machinery and is simply
    inactive here.

Cost: 1-2 training runs (5 members each).
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_09_loss_head.py --dry-run
    python3 scripts/ablation/ablation_09_loss_head.py                # both
    python3 scripts/ablation/ablation_09_loss_head.py --corner kfold
    python3 scripts/ablation/ablation_09_loss_head.py --skip-train   # eval
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, evaluate_bundle,             # noqa: E402
                    run_training, standard_train_cli, tee_into)

CORNERS = {"kfold": os.path.join(ABLATION_DIR, "smoothl1_kfold"),
           "shared": os.path.join(ABLATION_DIR, "smoothl1_shared")}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_09_loss_head.py",
        description="SmoothL1 corners of the loss x member-policy 2x2.")
    ap.add_argument("--corner", choices=["kfold", "shared", "both"],
                    default="both")
    ap.add_argument("--skip-train", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(ABLATION_DIR, exist_ok=True)
    tee_into("ablation_09_loss_head", ABLATION_DIR)
    corners = (["kfold", "shared"] if args.corner == "both"
               else [args.corner])

    # phase 1: ALL trainings first (subprocesses). The parent must stay
    # CUDA-free until no more children will be launched: on exclusive-
    # process GPUs a parent-held CUDA context (from an evaluation) makes
    # the next training subprocess fail with "device busy" (2026-08-04).
    if not args.skip_train:
        for c in corners:
            out_dir = CORNERS[c]
            print(f"\n################ train SmoothL1 + {c} ################")
            rc = run_training(
                standard_train_cli(out_dir, nll=False, kfold=(c == "kfold")),
                out_dir, dry_run=args.dry_run)
            if rc != 0:
                print(f"[warn] training failed for corner {c}; continuing")

    # phase 2: evaluations (parent may take the GPU now)
    for c in corners:
        if args.dry_run:
            continue
        bundle = os.path.join(CORNERS[c], "surrogate_bundle.pt")
        if not os.path.exists(bundle):
            print(f"[warn] no bundle for corner {c}; skipping eval")
            continue
        out = evaluate_bundle(bundle, label=f"smoothl1_{c}")
        row = out["rows"]["test/tta_on"]
        print(f"[corner {c}] MAE {row['mae']:.6f}  rho "
              f"{row['rho_pooled']:+.3f}  PICP1(s=disagreement) "
              f"{row.get('picp_1s', float('nan')):.3f}")

    if not args.dry_run:
        print("\nVERDICT: place these corners in the 2x2 with deployed v2 "
              "(#4) and NLL+shared (#8). Accuracy column answers 'does "
              "the NLL head cost accuracy?'; the PICP column (s = "
              "disagreement only here, per evaluator notes) answers "
              "'what does the learned s buy over ensemble spread?'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
