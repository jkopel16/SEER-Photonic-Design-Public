"""Paper figure: k-space power of the signed saliency map, trained vs
weight-randomized, with the first-reciprocal-lattice band circled.

Reproduces validate_saliency.py step 1 exactly (same layout, seeds, and
SmoothGrad recipe), then plots only the spectra. Output:
figures/attribution_band.pdf (+ .png preview).
"""
import os
import sys

import numpy as np

REPO = "/project/rise-batteries/Photonics_RISE"
sys.path.insert(0, os.path.join(REPO, "scripts", "interpretability"))

import validate_saliency as vs  # noqa: E402
import torch  # noqa: E402
from models.model import PhotonicCNN  # noqa: E402
from saliency import IMG, DEFAULT_LAYOUTS, load_layout, raster_tensor  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    if len(sys.argv) > 1:
        vs.DEV = torch.device(sys.argv[1])  # e.g. cuda

    cache = os.path.join(REPO, "figures", "attribution_band_maps.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        s_tr, s_rd = d["s_tr"], d["s_rd"]
        print("loaded cached maps from", cache)
    else:
        bundle, models, x_mean, x_std, recipe = vs.load_bundle_models()
        torch.manual_seed(0)
        rec = load_layout(DEFAULT_LAYOUTS[1][1])  # champ_v2, as in step 1
        c1 = raster_tensor(rec["holes"], rec["a_super_nm"], vs.DEV)
        rands = [PhotonicCNN(**bundle["arch"]).to(vs.DEV).eval()
                 for _ in range(5)]
        s_tr, _ = vs.smooth_maps(models, c1, x_mean, x_std, recipe, seed=0)
        s_rd, _ = vs.smooth_maps(rands, c1, x_mean, x_std, recipe, seed=0)
        np.savez(cache, s_tr=s_tr, s_rd=s_rd)

    _, conc_tr, _, _ = vs.spectral_stats(s_tr)
    _, conc_rd, _, _ = vs.spectral_stats(s_rd)
    print(f"in-band: trained {conc_tr:.4f}  random {conc_rd:.4f}")

    half = 24  # px crop half-width around the k-space origin
    cy = cx = IMG // 2
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.35), dpi=300)
    for ax, gmap, conc, title in (
            (axes[0], s_tr, conc_tr, "trained SEER"),
            (axes[1], s_rd, conc_rd, "weights randomized")):
        F = np.fft.fftshift(np.abs(np.fft.fft2(gmap - gmap.mean())) ** 2)
        F[cy, cx] = 0.0
        F = F / F.sum()
        crop = F[cy - half:cy + half + 1, cx - half:cx + half + 1]
        logp = np.log10(crop + 1e-12)
        ax.imshow(logp.T, origin="lower", cmap="magma",
                  vmin=-7.0, vmax=logp.max(),
                  extent=[-half, half, -half, half])
        for r in vs.ANNULUS:
            ax.add_patch(plt.Circle((0, 0), r, fill=False, color="w",
                                    lw=0.9, ls=(0, (3, 2))))
        ax.text(0.5, 0.035, f"{100 * conc:.1f}% of power in band",
                transform=ax.transAxes, ha="center", va="bottom",
                color="w", fontsize=8,
                bbox=dict(facecolor="black", alpha=0.65,
                          edgecolor="none", boxstyle="round,pad=0.25"))
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.4)
    out = os.path.join(REPO, "figures", "attribution_band")
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out + ".png", bbox_inches="tight", pad_inches=0.03)
    print("wrote", out + ".pdf")


if __name__ == "__main__":
    main()
