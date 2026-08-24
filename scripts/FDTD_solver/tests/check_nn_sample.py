"""
check_nn_sample.py -- prove one banked sample is ML-ready.

Run AFTER at least one sample is banked:

    python -u check_nn_sample.py            # uses the first banked sample
    python -u check_nn_sample.py 17         # or a specific sample_id

What it does (the exact transformation the CNN training code will do):
  1. load samples/sample_XXXXXX.npz
  2. rasterize the hole list -> the [1, 128, 128] float32 occupancy image
     (silicon = 1, air = 0, anti-aliased edges), the ONLY network input
  3. pull the scalar label E (and the auxiliary spectrum)
  4. assert shapes / dtypes / value ranges are exactly as specified
  5. save figs/nn_sample_check.png (image + spectrum + label) and
     nn_sample_pair.npz (X, y) -- the literal training pair

If this passes, the dataset -> network handoff works; generate freely.
"""

from __future__ import annotations

import glob
import os
import sys
import numpy as np


# solver modules live one directory up from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from fdtd_torch import rasterize_mask

SAMPLES = os.path.join(C.OUT_DIR, "samples")


def main():
    files = sorted(glob.glob(os.path.join(SAMPLES, "sample_*.npz")))
    if not files:
        raise SystemExit(f"no banked samples in {SAMPLES} -- run "
                         "'python -u run_dataset.py generate --limit 1' "
                         "first.")
    if len(sys.argv) > 1:
        want = int(sys.argv[1])
        files = [f for f in files
                 if int(os.path.basename(f)[7:13]) == want]
        if not files:
            raise SystemExit(f"sample {want} is not banked.")
    z = np.load(files[0], allow_pickle=False)
    sid = int(z["sample_id"])
    print(f"[sample {sid}]  class={z['disorder_class']}  "
          f"sigma={float(z['sigma']):.3g}  seed={int(z['seed'])}")

    # ---- 1. the network INPUT: geometry image only ----------------------
    holes = [tuple(h) for h in np.asarray(z["holes_xyr_nm"])]
    img = rasterize_mask(holes, float(z["a_super_nm"]),
                         C.IMG_SIZE, C.IMG_SIZE,
                         supersample=C.SUPERSAMPLE).astype(np.float32)
    X = img[None, :, :]                       # [1, 128, 128]

    # ---- 2. the network TARGETS ------------------------------------------
    y = np.float32(z["E"])                    # primary scalar label
    wl = np.asarray(z["wavelengths_nm"], float)
    A = np.asarray(z["A_si"], np.float32)     # auxiliary spectral head

    # ---- 3. contract checks ----------------------------------------------
    checks = [
        ("X shape", X.shape == (1, C.IMG_SIZE, C.IMG_SIZE)),
        ("X dtype float32", X.dtype == np.float32),
        ("X in [0, 1]", 0.0 <= float(X.min()) and float(X.max()) <= 1.0),
        ("X has both phases", float(X.min()) < 0.5 < float(X.max())),
        ("anti-aliased edges", bool(np.any((X > 0.01) & (X < 0.99)))),
        ("fill fraction sane",
         abs((1.0 - float(X.mean())) - float(z["fill_achieved"])) < 0.02),
        ("y finite scalar", np.isfinite(y) and y.shape == ()),
        ("y plausible (0.2-8)", 0.2 < float(y) < 8.0),
        ("spectrum shape matches grid", A.shape == wl.shape),
        ("spectrum in [0, 1]",
         float(np.nanmin(A)) > -1e-3 and float(np.nanmax(A)) <= 1.0),
        ("no NaN in spectrum", not np.isnan(A).any()),
    ]
    ok = True
    for name, passed in checks:
        ok &= bool(passed)
        print(f"    {'OK ' if passed else 'FAIL'}  {name}")

    # ---- 4. artifacts ------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2),
                                 gridspec_kw={"width_ratios": [1, 1.5]})
    a1.imshow(X[0].T, origin="lower", cmap="gray", vmin=0, vmax=1)
    a1.set_title(f"network input X  [1,{C.IMG_SIZE},{C.IMG_SIZE}]\n"
                 "(Si=1 white, air=0 black)")
    a1.set_xticks([]); a1.set_yticks([])
    a2.plot(wl, A, lw=1.0, color="#1f5fa8")
    a2.set_xlabel("wavelength (nm)")
    a2.set_ylabel(r"$A_\mathrm{Si}$")
    a2.set_title(f"label  E = {float(y):.4f}   "
                 f"(aux spectrum, {len(wl)} pts)")
    fig.suptitle(f"NN-readiness check: sample {sid} "
                 f"({z['disorder_class']}, sigma={float(z['sigma']):.3g})")
    fig.tight_layout()
    figp = os.path.join(C.OUT_DIR, "figs", "nn_sample_check.png")
    os.makedirs(os.path.dirname(figp), exist_ok=True)
    fig.savefig(figp, dpi=150)
    pairp = os.path.join(C.OUT_DIR, "nn_sample_pair.npz")
    np.savez_compressed(pairp, X=X, y=y, A_spectrum=A, wavelengths=wl,
                        sample_id=sid)
    print(f"\n    preview -> {figp}\n    training pair -> {pairp}")
    print("\nVERDICT:", "READY -- the dataset->network handoff works."
          if ok else "NOT READY -- fix the FAILs above before generating "
          "in bulk.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
