#!/usr/bin/env python3
"""Five validation steps for the real-space saliency findings, in one script.

  1  Weight-randomization control (Adebayo): are the champion fringes
     LEARNED, or an artifact of pulling gradients back through the FFT
     channel?  Gatekeeper for every fringe claim.
  2  Signed-gradient FFT duality: the k-space saliency peaks at 7-10 px;
     if the real-space fringes are its dual, the spectrum of the SIGNED
     gradient map must peak in the same annulus.  (The saved |grad| maps
     cannot be used: rectification doubles spatial frequency.)
  3  SmoothGrad + shared-color-scale preview: denoised maps, one global
     scale so champion-vs-bank magnitudes are comparable.
  4  Rotation equivariance (approximate by construction: raster-frame
     rotation misregisters the fftshifted C2 spectrum by one pixel).
  5  Circular-shift invariance: E is exactly invariant under cyclic
     translations; the zero-padded CNN is not.  Measures how much the
     prediction depends on absolute position.

Outputs under runs/interpretability/validation/: step PNGs, smoothgrad.npz,
and validation.json holding every number.  Run:

  LD_LIBRARY_PATH=/project/rise-batteries/photonics-fdtd/lib \\
  /project/rise-batteries/photonics-fdtd/bin/python3 validate_saliency.py
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import saliency  # noqa: E402  (wires repo + solver sys.path via common)
from saliency import (IMG, LATTICE_PX, OUT_ROOT, DEFAULT_LAYOUTS,  # noqa: E402
                      load_layout, raster_tensor, d4_op)
import common  # noqa: E402

import torch  # noqa: E402
from models.model import PhotonicCNN, build_input_channels  # noqa: E402
from models.inverse_design import SurrogateScorer  # noqa: E402

VAL_DIR = os.path.join(OUT_ROOT, "validation")
ANNULUS = (5, 12)          # px; k-space saliency peaked at 7-10, +-tolerance
DEV = torch.device("cpu")  # overridden by --device in main()


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
BUNDLE_PATH = common.DEPLOYED_BUNDLE   # overridden by --bundle in main()


def load_bundle_models():
    bundle = torch.load(BUNDLE_PATH, map_location="cpu",
                        weights_only=False)
    models = []
    for sd in bundle["state_dicts"]:
        m = PhotonicCNN(**bundle["arch"])
        m.load_state_dict(sd)
        m.to(DEV).eval()
        models.append(m)
    norm = bundle["norm"]
    x_mean = torch.as_tensor(norm["x_mean"],
                             dtype=torch.float32).reshape(1, -1, 1, 1).to(DEV)
    x_std = torch.as_tensor(norm["x_std"],
                            dtype=torch.float32).reshape(1, -1, 1, 1).to(DEV)
    recipe = bundle["channel_recipe"]
    return bundle, models, x_mean, x_std, recipe


def signed_grad(model, c1, x_mean, x_std, recipe, noise_std=0.0, rng=None):
    """Signed dE_hat/dC1 (normalized-units output head, identity view).

    noise_std > 0 adds Gaussian noise to the raster (SmoothGrad draw)."""
    base = c1
    if noise_std > 0:
        base = c1 + noise_std * torch.from_numpy(
            rng.standard_normal(c1.shape).astype(np.float32)).to(c1.device)
    leaf = base.clone().requires_grad_(True)
    x = build_input_channels(leaf, recipe)
    xn = (x - x_mean) / x_std
    model(xn)[..., 0].sum().backward()
    return leaf.grad[0, 0].detach().cpu().numpy().astype(np.float64)


def smooth_maps(models_list, c1, x_mean, x_std, recipe, n_noise=50,
                sigma=0.10, seed=0):
    """SmoothGrad-averaged maps over noise draws x models.

    Returns (signed_mean, mean_abs): the signed mean detects coherent
    first-moment fringes; mean_abs mirrors the original sal_c1 preview
    (rectified, so any spectral peak appears at DOUBLE the underlying
    spatial frequency)."""
    rng = np.random.default_rng(seed)
    acc_s = np.zeros((IMG, IMG))
    acc_a = np.zeros((IMG, IMG))
    n = 0
    with torch.enable_grad():
        for m in models_list:
            for _ in range(n_noise):
                g = signed_grad(m, c1, x_mean, x_std, recipe,
                                noise_std=sigma, rng=rng)
                acc_s += g
                acc_a += np.abs(g)
                n += 1
    return acc_s / n, acc_a / n


def spectral_stats(gmap, check_signed=True):
    """Radial power profile + annulus concentration + angular anisotropy
    of a map's |FFT|^2 (mean removed).  check_signed guards the duality
    analysis against accidentally feeding a rectified |grad| map."""
    if check_signed:
        assert gmap.min() < 0 < gmap.max(), "expected a signed map"
    F = np.fft.fftshift(np.abs(np.fft.fft2(gmap - gmap.mean())) ** 2)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    cy = cx = IMG // 2
    rad = np.hypot(yy - cy, xx - cx)
    F[cy, cx] = 0.0
    total = F.sum()
    prof = np.zeros(IMG // 2)
    for b in range(IMG // 2):
        m = (rad >= b) & (rad < b + 1)
        prof[b] = F[m].sum()
    ann = (rad >= ANNULUS[0]) & (rad < ANNULUS[1])
    conc = float(F[ann].sum() / total)
    ang = np.arctan2(yy - cy, xx - cx)[ann]
    pw = F[ann]
    hist, _ = np.histogram(ang, bins=24, weights=pw)
    aniso = float(hist.max() / (hist.mean() + 1e-30))
    peak = int(np.argmax(prof[1:]) + 1)   # exclude the 0-1 px bin
    return prof / total, conc, aniso, peak


# ---------------------------------------------------------------------------
def step1(models, x_mean, x_std, recipe, bundle, results):
    print("\n[step 1] weight-randomization control (champ_v2, "
          "SmoothGrad-denoised for BOTH nets)")
    torch.manual_seed(0)
    rec = load_layout(DEFAULT_LAYOUTS[1][1])          # champ_v2
    c1 = raster_tensor(rec["holes"], rec["a_super_nm"], DEV)
    rands = [PhotonicCNN(**bundle["arch"]).to(DEV).eval() for _ in range(5)]
    s_tr, a_tr = smooth_maps(models, c1, x_mean, x_std, recipe, seed=0)
    s_rd, a_rd = smooth_maps(rands, c1, x_mean, x_std, recipe, seed=0)
    out = {}
    for name, s, a in (("trained", s_tr, a_tr), ("random", s_rd, a_rd)):
        _, conc_s, aniso_s, peak_s = spectral_stats(s)
        _, conc2, aniso2, peak2 = spectral_stats(a, check_signed=False)
        out[name] = {"signed": {"annulus_conc": conc_s,
                                "anisotropy": aniso_s, "peak_px": peak_s},
                     "rectified": {"doubled_annulus_conc_note":
                                   "peak expected at 2x underlying freq",
                                   "anisotropy": aniso2, "peak_px": peak2}}
        print(f"  {name}: signed peak {peak_s}px conc {conc_s:.3f} "
              f"aniso {aniso_s:.1f} | rectified peak {peak2}px "
              f"aniso {aniso2:.1f}")
    results["step1"] = out
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=150)
    v_s = max(np.abs(s_tr).max(), np.abs(s_rd).max())
    for c, (s, a, t) in enumerate(((s_tr, a_tr, "trained (5 members)"),
                                   (s_rd, a_rd, "random (5 inits)"))):
        axes[0, c].imshow(s.T, origin="lower", cmap="RdBu_r",
                          vmin=-v_s, vmax=v_s)
        axes[0, c].set_title(f"{t}: signed SmoothGrad", fontsize=9)
        axes[1, c].imshow(a.T, origin="lower", cmap="magma")
        axes[1, c].set_title(f"{t}: mean $|$grad$|$", fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(VAL_DIR, "step1_randomization.png"))
    plt.close(fig)
    tr, rd = out["trained"]["signed"], out["random"]["signed"]
    learned = (tr["annulus_conc"] > 1.5 * rd["annulus_conc"]
               or tr["anisotropy"] > 1.5 * rd["anisotropy"])
    print(f"  verdict (signed maps): "
          f"{'LEARNED structure' if learned else 'comparable to random'}")
    results["step1"]["verdict_learned_signed"] = bool(learned)


def step2(models, x_mean, x_std, recipe, results):
    print("\n[step 2] signed-gradient FFT duality (all layouts)")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
    out = {}
    for tag, spec in DEFAULT_LAYOUTS:
        rec = load_layout(spec)
        c1 = raster_tensor(rec["holes"], rec["a_super_nm"], DEV)
        s, _ = smooth_maps(models, c1, x_mean, x_std, recipe,
                           n_noise=30, seed=3)
        prof, conc, _, peak = spectral_stats(s)
        inside = ANNULUS[0] <= peak < ANNULUS[1]
        out[tag] = {"peak_px": peak, "annulus_conc": conc,
                    "peak_in_annulus": bool(inside)}
        ax.plot(prof[:40], label=f"{tag} (peak {peak}px)", lw=1.2)
        print(f"  {tag}: spectral peak {peak}px, {100*conc:.0f}% of power in "
              f"{ANNULUS[0]}-{ANNULUS[1]}px annulus -> "
              f"{'inside' if inside else 'OUTSIDE'} predicted band")
    ax.axvspan(*ANNULUS, alpha=0.15, color="grey",
               label=f"k-space saliency band")
    ax.axvline(LATTICE_PX, color="k", lw=0.6, ls=(0, (3, 2)))
    ax.set_xlabel("spatial frequency (px)")
    ax.set_ylabel("fraction of |FFT(signed grad)|$^2$")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(VAL_DIR, "step2_duality.png"))
    plt.close(fig)
    results["step2"] = out


def step3(models, x_mean, x_std, recipe, results, n_noise=50, sigma=0.10):
    print(f"\n[step 3] SmoothGrad ({n_noise} draws, sigma={sigma}) "
          "+ shared-scale preview")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    rng = np.random.default_rng(0)
    maps, rasters = {}, {}
    for tag, spec in DEFAULT_LAYOUTS:
        rec = load_layout(spec)
        c1 = raster_tensor(rec["holes"], rec["a_super_nm"], DEV)
        acc = np.zeros((IMG, IMG))
        with torch.enable_grad():
            for m in models:
                for _ in range(n_noise):
                    acc += signed_grad(m, c1, x_mean, x_std, recipe,
                                       noise_std=sigma, rng=rng)
        maps[tag] = acc / (len(models) * n_noise)      # signed mean
        rasters[tag] = c1[0, 0].cpu().numpy()
        print(f"  {tag}: |SmoothGrad| max {np.abs(maps[tag]).max():.2e}")
    np.savez(os.path.join(VAL_DIR, "smoothgrad.npz"),
             **{f"sg_{t}": m for t, m in maps.items()})
    # global scale from the pooled 99th percentile of |maps|
    allabs = np.concatenate([np.abs(m).ravel() for m in maps.values()])
    vmax = float(np.percentile(allabs, 99))
    overlay_cmap = LinearSegmentedColormap.from_list(
        "sal", [(1, 1, 1, 0), (0.84, 0.37, 0.0, 0.9)])
    fig, axes = plt.subplots(4, 3, figsize=(9.5, 12.5), dpi=150)
    for r, (tag, _) in enumerate(DEFAULT_LAYOUTS):
        g, ras = maps[tag], rasters[tag]
        a = np.clip(np.abs(g), 0, vmax) / vmax
        axes[r, 0].imshow(ras.T, origin="lower", cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(f"{tag}\nmax={np.abs(g).max():.1e}",
                              fontsize=8)
        axes[r, 1].imshow(g.T, origin="lower", cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax)
        axes[r, 2].imshow(ras.T, origin="lower", cmap="gray", vmin=0, vmax=1)
        axes[r, 2].imshow(a.T, origin="lower", cmap=overlay_cmap,
                          vmin=0, vmax=1)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    for c, t in enumerate(["raster $C_1$",
                           "SmoothGrad (signed, one global scale)",
                           "overlay $|$SmoothGrad$|$"]):
        axes[0, c].set_title(t, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(VAL_DIR, "preview_v2.png"))
    plt.close(fig)
    results["step3"] = {t: {"absmax": float(np.abs(m).max())}
                        for t, m in maps.items()}
    results["step3"]["global_vmax_p99"] = vmax
    results["step3"]["caption_notes"] = (
        "Interior-pixel gradients are model diagnostics; only edge-adjacent "
        "gradient corresponds to a realizable perturbation. Per-hole "
        "occlusion (Sec 5.7) is the in-distribution attribution.")


def step4(models, x_mean, x_std, recipe, results, n_noise=10):
    print("\n[step 4] rotation equivariance (approximate by construction)")
    rng = np.random.default_rng(1)
    rec = load_layout(DEFAULT_LAYOUTS[1][1])
    c1 = raster_tensor(rec["holes"], rec["a_super_nm"], DEV)
    c1r = torch.rot90(c1, 1, dims=[2, 3])

    def sg(c, seed):
        r = np.random.default_rng(seed)
        acc = np.zeros((IMG, IMG))
        with torch.enable_grad():
            for _ in range(n_noise):
                acc += signed_grad(models[0], c, x_mean, x_std, recipe,
                                   noise_std=0.10, rng=r)
        return acc / n_noise

    g = sg(c1, 2)
    gr = sg(c1r, 2)                     # same noise seed for a fair pair
    want = np.rot90(g, 1)               # numpy rot90 on [x,y]-indexed map
    # torch rot90 over dims [2,3] equals np.rot90 on the [x,y] array
    denom = np.abs(want).max() + 1e-30
    dev_max = float(np.abs(gr - want).max() / denom)
    corr = float(np.corrcoef(gr.ravel(), want.ravel())[0, 1])
    results["step4"] = {"rel_max_dev": dev_max, "pearson_corr": corr}
    print(f"  rotate-layout vs rotate-map: corr {corr:.3f}, "
          f"rel max dev {dev_max:.2f}")
    print("  (approximate agreement expected: raster-frame rotation "
          "misregisters the fftshifted C2 spectrum by 1 px)")


def step5(scorer, results, n_extra=16, n_rolls=16):
    print("\n[step 5] circular-shift invariance of the prediction")
    rng = np.random.default_rng(7)
    layouts = [(t, load_layout(s)) for t, s in DEFAULT_LAYOUTS]
    d = np.load(common.DATA_128, mmap_mode="r")
    m = (d["disorder_class"] == "jitter") & np.isclose(d["sigma"], 0.15)
    idx = rng.choice(np.where(m)[0], size=n_extra, replace=False)
    for i in idx:
        sid = int(d["sample_id"][i])
        p = os.path.join(common.REPO, "data", "samples",
                         f"sample_{sid:06d}.npz")
        layouts.append((f"bank{sid}", load_layout(p)))
    spreads, stds = {}, {}
    for tag, rec in layouts:
        ras = raster_tensor(rec["holes"], rec["a_super_nm"],
                            DEV)[0, 0].cpu().numpy()
        # roll round-trip sanity (once)
        if tag == layouts[0][0]:
            r1 = np.roll(ras, (5, 9), axis=(0, 1))
            assert np.array_equal(np.roll(r1, (-5, -9), axis=(0, 1)), ras)
        versions = [ras]
        for _ in range(n_rolls):
            dx, dy = rng.integers(1, IMG, size=2)
            versions.append(np.roll(ras, (int(dx), int(dy)), axis=(0, 1)))
        X = torch.from_numpy(
            np.stack(versions).astype(np.float32)).unsqueeze(1).to(DEV)
        with torch.no_grad():
            mean, _ = scorer._forward_images(X)
        e = mean.cpu().numpy()
        spreads[tag] = float(e.max() - e.min())
        stds[tag] = float(e.std())
    sp = np.array(list(spreads.values()))
    results["step5"] = {
        "per_layout_spread": spreads,
        "mean_spread": float(sp.mean()), "max_spread": float(sp.max()),
        "reference_d4_view_spread": 0.002,
        "reference_cell_width": 0.04}
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=150)
    ax.hist(sp, bins=12, color="#0072B2", alpha=0.85)
    ax.axvline(0.002, color="#D55E00", lw=1.2, ls=(0, (4, 2)),
               label="D4 per-view spread")
    ax.axvline(0.04, color="k", lw=1.2, ls=(0, (1, 1)),
               label="within-cell width")
    ax.set_xlabel("prediction spread over 16 circular shifts (E units)")
    ax.set_ylabel("layouts")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(VAL_DIR, "step5_shift_spread.png"))
    plt.close(fig)
    print(f"  {len(layouts)} layouts x {n_rolls} rolls: mean spread "
          f"{sp.mean():.4f}, max {sp.max():.4f} in E units")
    print(f"  references: D4 per-view spread ~0.002, cell width ~0.04")


# ---------------------------------------------------------------------------
def main():
    global DEV, VAL_DIR, BUNDLE_PATH
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", default="1,2,3,4,5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bundle", default=common.DEPLOYED_BUNDLE)
    ap.add_argument("--val-dir", default=None,
                    help="output dir (default runs/interpretability/"
                         "validation; set when testing another bundle)")
    args = ap.parse_args()
    DEV = torch.device(args.device)
    BUNDLE_PATH = args.bundle
    if args.val_dir:
        VAL_DIR = args.val_dir
    steps = {int(s) for s in args.steps.split(",")}
    os.makedirs(VAL_DIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")

    bundle, models, x_mean, x_std, recipe = load_bundle_models()
    results = {}
    jpath = os.path.join(VAL_DIR, "validation.json")
    if os.path.exists(jpath):
        results = json.load(open(jpath))

    if 1 in steps:
        step1(models, x_mean, x_std, recipe, bundle, results)
    if 2 in steps:
        step2(models, x_mean, x_std, recipe, results)
    if 3 in steps:
        step3(models, x_mean, x_std, recipe, results)
    if 4 in steps:
        step4(models, x_mean, x_std, recipe, results)
    if 5 in steps:
        scorer = SurrogateScorer(BUNDLE_PATH, DEV, use_tta=True,
                                 kappa=0.2, batch_size=64, calibration=None)
        step5(scorer, results)

    with open(jpath, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nall requested steps done -> {jpath}")


if __name__ == "__main__":
    main()
