"""Plot a random augmented view from samples_aug.npz.

Run from anywhere with the project venv:
    python -m models.plot_aug
or pass an explicit augmented npz path:
    python -m models.plot_aug path/to/samples_aug.npz
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_AUG = os.path.join(_REPO_ROOT, "data", "samples_aug.npz")
OUT_FIG = os.path.join(_REPO_ROOT, "data", "aug_preview.png")


def main(argv=None):
    path = argv[1] if argv and len(argv) > 1 else DEFAULT_AUG
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run data_augmentation first")

    z = dict(np.load(path, allow_pickle=False))
    n = z["X"].shape[0]
    i = np.random.default_rng().integers(0, n)

    X = z["X"][i, 0]
    y = float(z["y"][i])
    sid = int(z["sample_id"][i])
    cls = str(z["disorder_class"][i])
    sigma = float(z["sigma"][i])
    wl = z["wavelengths_nm"]
    A = z["A_si"][i]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2),
                                 gridspec_kw={"width_ratios": [1, 1.5]})
    a1.imshow(X.T, origin="lower", cmap="gray", vmin=0, vmax=1)
    a1.set_title(f"view {i} / {n}    (Si=1 white, air=0 black)\n"
                 f"sample_id={sid}  {cls}  sigma={sigma:.3g}")
    a1.set_xticks([]); a1.set_yticks([])
    a2.plot(wl, A, lw=1.0, color="#1f5fa8")
    a2.set_xlabel("wavelength (nm)")
    a2.set_ylabel(r"$A_\mathrm{Si}$")
    a2.set_title(f"label  E = {y:.4f}")
    fig.suptitle(f"augmented view {i} of {n}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(OUT_FIG)), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    print(f"view {i}/{n}  sid={sid} {cls} sigma={sigma:.3g} E={y:.4f}")
    print(f"fig -> {OUT_FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
