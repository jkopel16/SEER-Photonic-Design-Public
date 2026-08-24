"""Rasterize banked samples -> (X, y, ...) npz. Optional FFT channel.

Thin wrapper around data_augmentation.build_xy_from_samples that skips Stage B
(augment_rotations_flip). Output schema is identical to samples_aug.npz so
models/model.py trains unchanged.

With --fft-channel, channel 1 = log(1 + |FFT(Si)|²) normalized to [0, 1],
giving the model explicit information about the structure factor (diffraction
periodicity, Bloch-mode content).  Channel 0 remains the Si occupancy.

CHANNEL PROVENANCE (added 2026-07-24): the output npz now carries a
`channel_recipe` field naming every channel ("raster", "fft_baked_v1")
so the trainer and inverse design never have to guess what the input is
-- the missing metadata behind the 4-channel double-FFT ablation
incident.  The fft_baked_v1 transform below is FROZEN byte-for-byte
(log1p power, per-sample min-max, unshifted, no mean removal) because
samples_128.npz was built with it; changing it would silently invalidate
comparability with every model trained on that file.  Known quirks,
accepted for compatibility: the DC peak is included in the min-max (it
compresses off-center features somewhat) and the unshifted layout makes
D4 augmentation of the channel a ~1-bin approximation of the true FFT of
the augmented raster.  Any improved transform must be a NEW named recipe
(e.g. "fft_baked_v2"), never an edit of v1.

Usage:
    # defaults: data/samples -> data/samples_128.npz (128 px, FFT channel on)
    python -m models.build_dataset

    # legacy 64 px raster-only output
    python -m models.build_dataset --img-size 64 --no-fft-channel \\
        -o data/samples_64.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)

# Allow running as script or as module (-m models.build_dataset)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, _REPO_ROOT)

from models.data_augmentation import build_xy_from_samples, save_augmented


def compute_fft_channel(X):
    """Compute log(1 + |FFT(Si)|²) normalized to [0, 1] per sample.

    X: (N, 1, H, W) float32 Si occupancy in [0, 1]
    Returns: (N, 1, H, W) float32 structure factor channel in [0, 1]

    The log scale compresses the huge dynamic range of |FFT|² so the DC
    peak doesn't swamp the off-center diffraction features that matter
    for light trapping.  Per-sample normalization ensures each layout's
    structure factor is equally visible to the CNN.
    """
    X_t = np.fft.fft2(X.astype(np.float64), axes=(-2, -1))
    power = np.abs(X_t) ** 2  # |FFT|²
    log_power = np.log1p(power)  # log(1 + |FFT|²)
    # per-sample min-max normalize to [0, 1]
    mins = log_power.min(axis=(-2, -1), keepdims=True)
    maxs = log_power.max(axis=(-2, -1), keepdims=True)
    denom = maxs - mins
    denom[denom < 1e-12] = 1.0
    fft_channel = ((log_power - mins) / denom).astype(np.float32)
    return fft_channel


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m models.build_dataset",
        description="Rasterize banked samples to a non-augmented .npz "
                    "(no rotations/flips).",
    )
    ap.add_argument("-i", "--in", dest="inp",
                    default=os.path.join(_REPO_ROOT, "data", "samples"),
                    help="samples/ directory of sample_XXXXXX.npz archives "
                         "(default: <repo>/data/samples).")
    ap.add_argument("-o", "--out", default=None,
                    help="Output .npz path. Default: <in>_<img-size>.npz sibling.")
    ap.add_argument("--img-size", type=int, default=128,
                    help="CNN raster edge in px (default 128, the v2 "
                         "reference raster; 64 = legacy).")
    ap.add_argument("--supersample", type=int, default=4,
                    help="Supersampling factor for the anti-aliased raster "
                         "(default 4, matches scripts/fdtd_torch.py).")
    ap.add_argument("--top", type=int, default=None,
                    help="Use only the first N samples (testing).")
    ap.add_argument("--fft-channel", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Add structure-factor channel: log(1 + |FFT(Si)|²) "
                         "normalized to [0,1].  Output X becomes (N, 2, H, W). "
                         "On by default (matches data/samples_128.npz); "
                         "disable with --no-fft-channel.")
    args = ap.parse_args(argv)

    bundle = build_xy_from_samples(
        args.inp, img_size=args.img_size, supersample=args.supersample,
        top=args.top,
    )

    if args.fft_channel:
        X = bundle["X"]
        if X.ndim == 3:
            X = X[:, None, :, :]  # (N, H, W) -> (N, 1, H, W)
        fft_ch = compute_fft_channel(X)
        # X becomes (N, 2, H, W): channel 0 = Si occupancy, channel 1 = |FFT|²
        bundle["X"] = np.concatenate([X, fft_ch], axis=1).astype(np.float32)
        print(f"[fft-channel] added structure factor channel -> "
              f"X.shape = {bundle['X'].shape}")
        bundle["channel_recipe"] = np.array(["raster", "fft_baked_v1"])
    else:
        bundle["channel_recipe"] = np.array(["raster"])
    print(f"[channels] recipe = {list(bundle['channel_recipe'])}")

    if args.out is None:
        args.out = args.inp.rstrip("/") + f"_{args.img_size}.npz"
    save_augmented(args.out, bundle)

    X, y = bundle["X"], bundle["y"]
    print(f"{X.shape[0]} samples -> X={X.shape}, y={y.shape}, "
          f"saved to {args.out}")
    if "fill_achieved" in bundle:
        delta = float(abs((1.0 - X[:, 0].mean()) - bundle["fill_achieved"].mean()))
        print(f"    fill-vs-raster delta mean = {delta:.4f} "
              f"(check_nn_sample gate is 0.02)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
