"""
data_augmentation.py -- build the (X, y) ML pair from banked samples
and expand it 8x via 0/90/180/270-degree rotations of the original
geometry image plus 0/90/180/270-degree rotations of its horizontal
flip, carrying every view's label and identifiers to the output .npz.

Pipeline
--------
Stage A  rasterize.  The mini_samples/ tree stores the raw hole list and
         parameters but produces no rasterized network-input image.  We
         rebuild the exact [1, 128, 128] float32 Si-occupancy image from
         holes_xyr_nm + a_super_nm, byte-identical to what the campaign
         would have written -- the same recipe used by
         scripts/check_nn_sample.py:52-56 against scripts/fdtd_torch.py
         rasterize_mask.
Stage B  augment.    8 views per sample (4 rotations of the original
         + 4 rotations of the horizontally-flipped original).  Every
         derived view inherits the source sample's scalar label E and
         identifiers (sample_id, disorder_class, sigma, seed,
         fill_achieved) and its full A_si spectrum.

Usage
-----
    # rasterize-and-augment from the default bank copy (data/samples)
    python -m models.data_augmentation          # -> data/samples_aug.npz

    # augment a pre-built X/y npz (skip Stage A)
    python -m models.data_augmentation -i path/to/pair.npz -o aug.npz

Optional flags: --top N (only first N samples), --dedupe (drop exact
duplicate views), --img-size (default 128), --supersample (default 4).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch


# ==========================================================================
# Stage A -- layout -> rasterized [1, 128, 128] image
# ==========================================================================

def rasterize_mask(holes, a_super_nm, Nx, Ny, supersample=4):
    """Anti-aliased Si occupancy image (1 = Si, 0 = air), minimum-image.

    Verbatim mirror of scripts/fdtd_torch.rasterize_mask (canonical
    implementation lives at scripts/fdtd_torch.py:123).  Inlined here so
    this module has no sys.path dependency on the non-package scripts/
    directory; keep the two in lockstep if either is edited.
    """
    ss = supersample
    NX, NY = Nx * ss, Ny * ss
    px = a_super_nm / NX
    py = a_super_nm / NY
    xs = (np.arange(NX) + 0.5) * px
    ys = (np.arange(NY) + 0.5) * py
    inside = np.zeros((NX, NY), dtype=bool)
    for (hx, hy, hr) in holes:
        i0 = int(np.floor((hx - hr) / px)) - 1
        i1 = int(np.ceil((hx + hr) / px)) + 1
        j0 = int(np.floor((hy - hr) / py)) - 1
        j1 = int(np.ceil((hy + hr) / py)) + 1
        ii = np.arange(i0, i1 + 1) % NX
        jj = np.arange(j0, j1 + 1) % NY
        dx = xs[ii] - hx
        dx -= a_super_nm * np.round(dx / a_super_nm)
        dy = ys[jj] - hy
        dy -= a_super_nm * np.round(dy / a_super_nm)
        disk = (dx[:, None] ** 2 + dy[None, :] ** 2) <= hr * hr
        inside[np.ix_(ii, jj)] |= disk
    frac_si = 1.0 - inside.astype(float)
    return frac_si.reshape(Nx, ss, Ny, ss).mean(axis=(1, 3))


SAMPLE_KEYS = [
    "holes_xyr_nm", "a_super_nm", "E", "wavelengths_nm", "A_si",
    "sample_id", "disorder_class", "sigma", "seed", "fill_achieved",
]


def _load_record_from_npz(npz_path):
    """Read one sample_XXXXXX.npz archive written by scripts/run_dataset.py.

    Mirrors scripts/check_nn_sample.py:46 -- scalar and array fields are
    stored 0-d / 1-d inside the archive; we lift them verbatim.
    """
    rec = {}
    with np.load(npz_path, allow_pickle=False) as z:
        for k in SAMPLE_KEYS:
            if k not in z.files:
                raise KeyError(f"{npz_path} missing field {k}")
            rec[k] = z[k]
    return rec


def _load_record_from_dir(sample_dir):
    """Read the per-sample .npy files written by scripts/run_dataset.py.

    Each sample lives in a sample_XXXXXX/ subdirectory holding one .npy
    per field.  Older ad-hoc export form; kept for backward compat.
    """
    out = {}
    for k in SAMPLE_KEYS:
        p = os.path.join(sample_dir, k + ".npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing {p}")
        out[k] = np.load(p, allow_pickle=True)
    return out


def _split_records_and_build(recs, img_size, supersample):
    """Common Stage-A tail: rasterize each record and stack into ML dict."""
    Xs, ys, sids, dcs, sigs, seeds, fills, A_sis = [], [], [], [], [], [], [], []
    wl = None
    for rec in recs:
        holes = [tuple(h) for h in np.asarray(rec["holes_xyr_nm"])]
        img = rasterize_mask(
            holes, float(rec["a_super_nm"]),
            img_size, img_size, supersample=supersample,
        ).astype(np.float32)
        Xs.append(img[None, :, :])
        ys.append(np.asarray(rec["E"], dtype=np.float32))
        sids.append(np.asarray(rec["sample_id"], dtype=np.int64))
        dcs.append(np.asarray(rec["disorder_class"]))
        sigs.append(np.asarray(rec["sigma"], dtype=np.float64))
        seeds.append(np.asarray(rec["seed"], dtype=np.int64))
        fills.append(np.asarray(rec["fill_achieved"], dtype=np.float64))
        A = np.asarray(rec["A_si"], dtype=np.float32).ravel()
        A_sis.append(A)
        wl = np.asarray(rec["wavelengths_nm"], dtype=np.float64).ravel()

    return {
        "X": np.stack(Xs, axis=0),
        "y": np.stack(ys, axis=0).reshape(-1),
        "sample_id": np.stack(sids, axis=0).reshape(-1),
        "disorder_class": np.stack(dcs, axis=0).reshape(-1),
        "sigma": np.stack(sigs, axis=0).reshape(-1),
        "seed": np.stack(seeds, axis=0).reshape(-1),
        "fill_achieved": np.stack(fills, axis=0).reshape(-1),
        "A_si": np.stack(A_sis, axis=0),
        "wavelengths_nm": wl,
    }


def build_xy_from_samples(samples_dir, *, img_size=128, supersample=4,
                          top=None):
    """Walk samples_dir/ and build the rasterized ML dict.

    Prefers sample_XXXXXX.npz archives (the canonical campaign output
    from scripts/FDTD_solver/run_dataset.py); falls back to sample_XXXXXX/ subdirs
    of individual .npy files for the older ad-hoc export.

    Returns a dict with stacked numpy arrays:
        X              (N, 1, img_size, img_size) float32 in [0, 1]
        y              (N,)                      float32  (scalar E)
        sample_id      (N,)                      int64
        disorder_class (N,)                      <U16
        sigma          (N,)                      float64
        seed           (N,)                      int64
        fill_achieved  (N,)                      float64
        A_si           (N, W)                    float32
        wavelengths_nm (W,)                      float64
    """
    npzs = sorted(glob.glob(os.path.join(samples_dir, "sample_*.npz")))
    dirs = sorted(d for d in
                  glob.glob(os.path.join(samples_dir, "sample_*"))
                  if os.path.isdir(d))
    if not npzs and not dirs:
        raise SystemExit(f"no sample_*.npz or sample_*/ under {samples_dir}")
    if top is not None:
        npzs = npzs[:top]
        dirs = dirs[:top]

    recs = []
    if npzs:
        for p in npzs:
            recs.append(_load_record_from_npz(p))
    else:
        for d in dirs:
            recs.append(_load_record_from_dir(d))
    return _split_records_and_build(recs, img_size, supersample)


# ==========================================================================
# Stage B -- 8x rotation + horizontal-flip augmentation
# ==========================================================================

ROT_K = (0, 1, 2, 3)  # 0, 90, 180, 270 degrees


def _replicate(per_sample, n_views, n_src):
    """Tile a (n_src,) per-sample array into (n_views * n_src,)."""
    if per_sample is None:
        return None
    a = np.asarray(per_sample)
    return np.repeat(a, n_views, axis=0)


def augment_rotations_flip(X, y, *, sample_id=None, disorder_class=None,
                           sigma=None, seed=None, fill_achieved=None,
                           A_si=None, wavelengths_nm=None, dedupe=False):
    """Expand every sample into 8 views (4 rotations x 2 flip-state).

    Views, in order, for sample i:
        0..3: rot90(X[i], k=0..3)              -- original rotations
        4..7: rot90(flip(X[i], -1), k=0..3)     -- flipped then rotated

    Each view inherits y[i] and every supplied per-sample identifier /
    spectrum.  Rotations use torch.rot90 with dims=(-2, -1) so they act
    on the (H, W) plane of a (1, H, W) image; no resampling, so values
    stay literal float32 in [0, 1].
    """
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    n = X_np.shape[0]
    assert X_np.ndim == 4, f"X must be (N,1,H,W); got {X_np.shape}"
    assert X_np.shape[1] == 1, f"X must have 1 channel; got {X_np.shape}"
    assert y_np.shape[0] == n, f"len(y)={y_np.shape[0]} != N={n}"

    X_t = torch.from_numpy(np.ascontiguousarray(X_np)).float()
    views, labels = [], []
    idx_map = []  # (i, k, flipped)
    for i in range(n):
        base = X_t[i]
        for k in ROT_K:
            v = torch.rot90(base, k, dims=(-2, -1))
            views.append(v)
            labels.append(y_np[i])
            idx_map.append((i, k, False))
        flipped = torch.flip(base, dims=(-1,))
        for k in ROT_K:
            v = torch.rot90(flipped, k, dims=(-2, -1))
            views.append(v)
            labels.append(y_np[i])
            idx_map.append((i, k, True))

    X_aug = torch.stack(views, dim=0).to(torch.float32).numpy()
    y_aug = np.asarray(labels, dtype=y_np.dtype)
    n_views_per = len(ROT_K) * 2

    aug = {
        "X": X_aug,
        "y": y_aug,
        "sample_id":      _replicate(sample_id,      n_views_per, n),
        "disorder_class": _replicate(disorder_class, n_views_per, n),
        "sigma":          _replicate(sigma,          n_views_per, n),
        "seed":           _replicate(seed,           n_views_per, n),
        "fill_achieved":  _replicate(fill_achieved,  n_views_per, n),
        "A_si":           _replicate(A_si,            n_views_per, n),
        "wavelengths_nm": None if wavelengths_nm is None
        else np.asarray(wavelengths_nm),
    }
    aug = {k: v for k, v in aug.items() if v is not None}

    if dedupe:
        keep = _dedupe_indices(X_aug)
        aug = {k: (v[keep] if v is not None and v.shape and v.shape[0] == len(idx_map)
                   else v) for k, v in aug.items()}
    return aug


def _dedupe_indices(X_aug, decimals=6):
    """Indices of unique views in X_aug, preserving first-occurrence order."""
    rounded = np.round(X_aug, decimals)
    seen = {}
    keep = []
    for i, view in enumerate(rounded):
        key = view.tobytes()
        if key not in seen:
            seen[key] = i
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)


# ==========================================================================
# IO
# ==========================================================================

def save_augmented(out_path, aug):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, **aug)


def load_augmented(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ==========================================================================
# CLI
# ==========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m models.data_augmentation",
        description="Rasterize banked samples and/or augment via 4 rotations "
                    "x (original + horizontal flip).",
    )
    ap.add_argument("-i", "--in", dest="inp",
                    default=os.path.join(
                        os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))),
                        "data", "samples"),
                    help="Either a samples/ directory of sample_XXXXXX.npz "
                         "archives, or an .npz with X and y "
                         "(default: <repo>/data/samples).")
    ap.add_argument("-o", "--out", default=None,
                    help="Output .npz path. Default: <in>_aug.npz sibling "
                         "to the input.")
    ap.add_argument("--top", type=int, default=None,
                    help="Use only the first N samples (testing).")
    ap.add_argument("--dedupe", action="store_true",
                    help="Drop exact-duplicate views (off by default).")
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--supersample", type=int, default=4)
    args = ap.parse_args(argv)

    if os.path.isdir(args.inp):
        bundle = build_xy_from_samples(
            args.inp, img_size=args.img_size, supersample=args.supersample,
            top=args.top,
        )
        src_label = args.inp
    else:
        bundle = load_augmented(args.inp)
        if "X" not in bundle or "y" not in bundle:
            raise SystemExit(f"{args.inp} has no X/y arrays")
        src_label = args.inp

    n_src = bundle["X"].shape[0]
    aug = augment_rotations_flip(
        bundle["X"], bundle["y"],
        sample_id=bundle.get("sample_id"),
        disorder_class=bundle.get("disorder_class"),
        sigma=bundle.get("sigma"),
        seed=bundle.get("seed"),
        fill_achieved=bundle.get("fill_achieved"),
        A_si=bundle.get("A_si"),
        wavelengths_nm=bundle.get("wavelengths_nm"),
        dedupe=args.dedupe,
    )

    if args.out is None:
        stem = args.inp.rstrip("/")
        args.out = (stem if os.path.isdir(args.inp)
                    else os.path.splitext(args.inp)[0]) + "_aug.npz"
    save_augmented(args.out, aug)

    print(f"{n_src} -> {aug['X'].shape[0]} views "
          f"(X={aug['X'].shape}, y={aug['y'].shape}), saved to {args.out}")
    if "fill_achieved" in aug and "X" in aug:
        delta = float(
            abs((1.0 - aug["X"].mean()) - aug["fill_achieved"].mean()))
        print(f"    fill-vs-raster delta mean = {delta:.4f} "
              f"(check_nn_sample gate is 0.02)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
