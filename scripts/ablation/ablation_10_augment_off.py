"""Ablation #10 -- D4 training augmentation OFF.

Context: the physics is exactly D4-symmetric (E is invariant under the 8
dihedral ops of the square supercell; verified numerically in
tests/check_d4_symmetry.py), so training-time augmentation is
effectively an 8x data multiplier at zero label cost.  This run
quantifies that win: identical v2 recipe with --no-augment.

TTA disambiguation (pre-registered): training augmentation and test-time
augmentation are SEPARATE knobs.  This script changes only the training
knob; the shared evaluator always emits BOTH TTA arms for every bundle,
so the four rows

    #4  deployed(aug on)/tta_on  #4  deployed(aug on)/tta_off
    #10 aug_off/tta_on           #10 aug_off/tta_off

form the complete train-aug x TTA 2x2 with no extra runs.

Cost: ~1 training run (5 members).
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_10_augment_off.py --dry-run
    python3 scripts/ablation/ablation_10_augment_off.py
    python3 scripts/ablation/ablation_10_augment_off.py --skip-train
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, evaluate_bundle,             # noqa: E402
                    run_training, standard_train_cli, tee_into)

OUT_DIR = os.path.join(ABLATION_DIR, "augment_off")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_10_augment_off.py",
        description="Retrain v2 recipe without D4 training augmentation.")
    ap.add_argument("--skip-train", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    tee_into("ablation_10_augment_off", OUT_DIR)

    if not args.skip_train:
        rc = run_training(standard_train_cli(OUT_DIR, augment=False),
                          OUT_DIR, dry_run=args.dry_run)
        if rc != 0 or args.dry_run:
            return rc
    out = evaluate_bundle(os.path.join(OUT_DIR, "surrogate_bundle.pt"),
                          label="augment_off")
    on = out["rows"]["test/tta_on"]
    off = out["rows"]["test/tta_off"]
    print(f"\nVERDICT: aug-off rows -- tta_on MAE {on['mae']:.6f} rho "
          f"{on['rho_pooled']:+.3f}; tta_off MAE {off['mae']:.6f} rho "
          f"{off['rho_pooled']:+.3f}. Combine with ablation #4's two rows "
          "for the train-aug x TTA 2x2; the aug-on-minus-aug-off delta is "
          "the data-efficiency win of exploiting the exact D4 symmetry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
