"""Ablation #8 -- k-fold member rotation OFF (NLL head + shared splits).

Context: v1 -> v2 changed TWO things at once -- the heteroscedastic
beta-NLL head AND k-fold member rotation (v1's members shared one train
split; that data-sharing is the documented cause of the 4.5x
overconfident v1 intervals, coverage 0.21 -> 0.65 at +-1s).  This run is
the missing corner of the 2x2 {loss head} x {member data policy}:

    NLL + k-fold   = deployed v2            (reference row, ablation #4)
    NLL + shared   = THIS RUN
    SmoothL1 + k-fold / + shared = ablation #9

Together the four rows attribute the coverage fix: if THIS run's PICP
collapses toward v1's, decorrelated members bought the calibration; if
it stays ~0.65/0.96/0.996, the NLL head did.  This is the
highest-information retrain in the suite.

Protocol: identical to deployed v2 except --kfold-members is dropped
(same data, seed 137, best_params hyperparams, early stopping).  Note
members now train on the 80 % master train split instead of ~4/5 of the
90 % train+val pool -- a small data-size delta inherent to the policy
being ablated.

Cost: ~1 training run (5 members) on one GPU.
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_08_kfold_off.py --dry-run
    python3 scripts/ablation/ablation_08_kfold_off.py
    python3 scripts/ablation/ablation_08_kfold_off.py --skip-train  # eval only
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, evaluate_bundle,             # noqa: E402
                    run_training, standard_train_cli, tee_into)

OUT_DIR = os.path.join(ABLATION_DIR, "kfold_off")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_08_kfold_off.py",
        description="Retrain v2 recipe without k-fold member rotation.")
    ap.add_argument("--skip-train", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    tee_into("ablation_08_kfold_off", OUT_DIR)

    if not args.skip_train:
        rc = run_training(standard_train_cli(OUT_DIR, kfold=False),
                          OUT_DIR, dry_run=args.dry_run)
        if rc != 0 or args.dry_run:
            return rc
    out = evaluate_bundle(os.path.join(OUT_DIR, "surrogate_bundle.pt"),
                          label="nll_shared_split")
    row = out["rows"]["test/tta_on"]
    print(f"\nVERDICT: NLL + shared-split PICP(1/2/3s) = "
          f"{row['picp_1s']:.3f}/{row['picp_2s']:.3f}/{row['picp_3s']:.3f} "
          "-- compare against deployed v2 (0.651/0.960/0.996) and v1's "
          "shared-split history (0.21/0.37/0.48) to attribute the "
          "coverage fix; combine with ablation #9's two corners for the "
          "full 2x2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
