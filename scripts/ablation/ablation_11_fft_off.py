"""Ablation #11 -- structure-factor (FFT) input channel OFF.

Context: the deployed v2 model sees 2 channels -- the geometry raster
plus log1p|FFT| of it (the structure factor up to phase; absorption
enhancement is governed by how the pattern scatters into guided modes,
which lives in reciprocal space).  This run trains on the raster alone
(--raster-only WITHOUT --fft-channel -> recipe ["raster"], in_ch = 1) to
test whether the hand-engineered physics channel actually contributes or
the CNN recovers it unaided.  Either outcome is publishable: "channel
adds X" or "CNN learns it from geometry".

Everything else is the deployed recipe (NLL head, k-fold, 5 members,
seed 137, best_params).  Note the FFT channel is deterministic in the
raster, so this measures inductive-bias value, not information content.

Cost: ~1 training run (5 members; slightly faster, 1 input channel).
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_11_fft_off.py --dry-run
    python3 scripts/ablation/ablation_11_fft_off.py
    python3 scripts/ablation/ablation_11_fft_off.py --skip-train
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, evaluate_bundle,             # noqa: E402
                    run_training, standard_train_cli, tee_into)

OUT_DIR = os.path.join(ABLATION_DIR, "fft_off")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_11_fft_off.py",
        description="Retrain v2 recipe with the raster channel only.")
    ap.add_argument("--skip-train", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    tee_into("ablation_11_fft_off", OUT_DIR)

    # standard cli always has --raster-only; the delta is: no --fft-channel
    cli = standard_train_cli(OUT_DIR)
    cli.remove("--fft-channel")
    if not args.skip_train:
        rc = run_training(cli, OUT_DIR, dry_run=args.dry_run)
        if rc != 0 or args.dry_run:
            return rc
    out = evaluate_bundle(os.path.join(OUT_DIR, "surrogate_bundle.pt"),
                          label="fft_channel_off")
    row = out["rows"]["test/tta_on"]
    print(f"\nVERDICT: raster-only MAE {row['mae']:.6f}, rho "
          f"{row['rho_pooled']:+.3f} -- compare to deployed v2 (MAE "
          "0.005447, rho 0.701). A material gap = the structure-factor "
          "channel earns its Methods paragraph; parity = the CNN recovers "
          "reciprocal-space structure unaided (report either honestly).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
