"""Poster figure: test residual + ranking fidelity vs training-set size.

Replots a finished learning_curve.json (models/learning_curve.py) with the
y-axis in PHYSICAL E units -- the mean absolute residual
|E_pred - E_true| -- rather than normalised SmoothL1 loss, so the curve can
be read against two physical references:

  * the 0.30 % within-sigma ranking-resolvability floor (audit Test 9,
    res-120 referee), drawn as a grey band from the axis;
  * the ensemble's own self-predicted error (mean member disagreement,
    recorded per curve point when the run used --ensemble > 1), drawn as a
    filled band.  The gap between that band and the residual curve IS the
    overconfidence that models/calibrate_uq.py corrects.

v2 runs (learning_curve.py --nll-head) carry the heteroscedastic model's
raw self-predicted sigma per curve point; those JSONs are auto-detected
and rendered as a three-panel figure instead:

  A. residual vs N with the RAW +-1 sigma band (no calibration) + rho;
  B. the sigma decomposed into aleatoric (learned, plateaus at the noise
     floor) vs epistemic (member disagreement, falls with N) -- the
     "enough data" argument in E units;
  C. predictive entropy 0.5*ln(2*pi*e*sigma^2) vs N (same sigma as the
     band, in log units).

Member disagreement below ENS_STD_MIN_N samples is dominated by training
instability rather than epistemic uncertainty (see the N=200 spike in
seed137_ens), so UQ series start there -- same convention as
models/learning_curve.py's own plot.

Usage:
    python -m models.plot_learning_curve runs/learning_curve_seed137_v2
    python -m models.plot_learning_curve <dir> --calibrated
    python -m models.plot_learning_curve <dir> --out fig5.png --dpi 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from models.learning_curve import ENS_STD_MIN_N  # noqa: E402

Y_MEAN = 2.5875              # seed-137 train-split mean E
FLOOR_PCT = 0.30             # within-sigma resolvability floor, % of E
NOISE_PCT = 0.09             # per-label engine noise, % of E (audit Test 9)

C_RESID = "#d62728"
C_UQ = "#7b3fa0"
C_RHO = "#1f77b4"
C_ALEA = "#2a9d8f"           # aleatoric (learned sigma)
C_EPI = "#e8871a"            # epistemic (member disagreement)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m models.plot_learning_curve",
        description="Poster figure: residual + ranking fidelity vs N.")
    ap.add_argument("run_dir",
                    help="directory holding learning_curve.json")
    ap.add_argument("--out", default=None,
                    help="default <run_dir>/learning_curve_residual.png")
    ap.add_argument("--calibrated", action="store_true",
                    help="also draw the band after applying the post-hoc "
                         "calibration map from calibrate_uq.py. NOTE: that "
                         "map was fit on the FULL-dataset ensemble, so only "
                         "its right-hand end is strictly justified; the "
                         "small-N end is an extrapolation.")
    ap.add_argument("--calibration", default=os.path.join(
        _REPO_ROOT, "runs", "surrogate_128_fft_nll_sweep", "uq",
        "calibration.json"))
    ap.add_argument("--single", action="store_true", default=False,
                    help="v2 runs: emit ONLY the top panel (measured vs "
                         "predicted error vs N) as a standalone poster "
                         "figure, skipping the decomposition and third "
                         "panels.")
    ap.add_argument("--entropy", action="store_true", default=False,
                    help="v2 runs: show predictive entropy in the third "
                         "panel instead of the calibration ratio. NOTE "
                         "0.5*ln(2*pi*e*sigma^2) is the GAUSSIAN entropy "
                         "-- for a non-Gaussian predictive distribution "
                         "with the same variance it is an upper bound, so "
                         "the default panel avoids it.")
    ap.add_argument("--dpi", type=int, default=150)
    return ap.parse_args(argv)


def _log_xticks(ax, ns):
    """Log x-axis with readable sample-size ticks (shared across panels)."""
    ax.set_xscale("log", base=10)
    ax.set_xlim(min(ns) * 0.9, max(ns) * 1.1)
    ticks = [ns[0]]
    for n in ns[1:-1]:
        if np.log10(n / ticks[-1]) >= 0.15:
            ticks.append(n)
    if np.log10(ns[-1] / ticks[-1]) < 0.08:
        ticks.pop()
    ticks.append(ns[-1])
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot_v2(ns, res, args):
    """Three-panel figure for --nll-head runs.

    DISTRIBUTION-FREE by construction: the model's predicted error scale
    (RMS sigma) is compared directly against the realised error scale
    (test RMSE).  Both are second moments, so "predicted = realised" is a
    variance-matching statement valid for ANY residual distribution -- no
    Gaussian, no 68 % interval, no coverage claim.  (Comparing sigma with
    the MEAN absolute error would not be: E|r| = sigma*sqrt(2/pi) only
    holds for a normal.)
    """
    def col(key, floor_n=ENS_STD_MIN_N):
        return np.array([res[str(n)].get(key, float("nan"))
                         if n >= floor_n else float("nan")
                         for n in ns], float)

    mae = np.array([res[str(n)]["test_mae"] for n in ns], float)
    rmse = np.array([res[str(n)]["test_rmse"] for n in ns], float)
    rho = np.array([res[str(n)]["test_within_sigma_spearman"] for n in ns],
                   float)

    # prefer the RMS aggregates (exact variance matching); older runs only
    # carry mean-of-sigma, which sits BELOW the RMS by Jensen -- so the
    # ratio computed from it is a lower bound on the true one.
    # No small-N gate here (floor_n=0): unlike v1's member-disagreement
    # band, the v2 forecast is dominated by the learned head and is
    # meaningful at every N -- data-starved models over-estimate their
    # error, which is worth SHOWING, not masking.
    sig = col("rms_pred_sigma_total", floor_n=0)
    exact = bool(np.isfinite(sig).any())
    if not exact:
        sig = col("mean_pred_sigma_total", floor_n=0)
    alea = col("rms_aleatoric_std" if exact else "mean_aleatoric_std",
               floor_n=0)
    epi = col("rms_epistemic_std" if exact else "mean_epistemic_std",
              floor_n=0)
    ent = col("mean_entropy", floor_n=0)

    floor_e = FLOOR_PCT / 100.0 * Y_MEAN
    noise_e = NOISE_PCT / 100.0 * Y_MEAN
    nsa = np.asarray(ns, float)
    n_mem = res[str(ns[-1])].get("n_members", 1)
    ratio = sig / rmse

    print("test RMSE per N: "
          + ", ".join(f"{n}:{m:.4f}" for n, m in zip(ns, rmse)))
    print("predicted sigma per N: "
          + ", ".join(f"{n}:{s:.4f}" for n, s in zip(ns, sig)
                      if np.isfinite(s)))
    print(f"resolvability floor ({FLOOR_PCT:g}% of E) = {floor_e:.4f}")
    print(f"sigma aggregate: {'RMS (exact)' if exact else 'mean (lower bound)'}")

    if args.single:
        fig, axA = plt.subplots(figsize=(8.6, 5.4))
        axB = axC = None
    else:
        fig = plt.figure(figsize=(10, 7.6))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0],
                              hspace=0.36, wspace=0.30)
        axA = fig.add_subplot(gs[0, :])
        axB = fig.add_subplot(gs[1, 0])
        axC = fig.add_subplot(gs[1, 1])

    # ---- Panel A: realised error vs predicted error, as two lines -------
    # the vertical gap between them IS the miscalibration; no band, so the
    # error curve itself stays legible
    axA.set_xlabel("Number of training samples")
    axA.set_ylabel("Error scale, RMS (E units)", color=C_RESID)
    axA.axhspan(0, floor_e, color="#bbbbbb", alpha=0.35, lw=0, zorder=0)
    axA.axhline(floor_e, color="#777777", ls="--", lw=1.2, zorder=1)
    axA.annotate(f"{FLOOR_PCT:.2f} % ranking-resolvability floor",
                 (0.015, 0.02), xycoords="axes fraction", ha="left",
                 va="bottom", fontsize=8.5, color="#555555")

    m = np.isfinite(sig)
    axA.fill_between(nsa[m], rmse[m], sig[m], color=C_UQ, alpha=0.18, lw=0,
                     zorder=2, label="gap = miscalibration")
    n_test = res[str(ns[-1])].get("n_test")
    axA.plot(ns, rmse, "o-", color=C_RESID, lw=2.0, ms=6, zorder=6,
             label=("actual error  (test RMSE"
                    + (f", n={n_test}" if n_test else "") + ")"))
    # no sigma symbol here: on the poster, sigma is the DISORDER strength;
    # the uncertainty is written s (or spelled out) to avoid the collision
    axA.plot(nsa[m], sig[m], "D--", color=C_UQ, lw=2.0, ms=6, zorder=6,
             label="predicted error  (model's own forecast, RMS)")

    if np.isfinite(ratio[-1]):
        # the tracking claim goes in the poster commentary, not on the
        # image -- print it for the caption writer
        dev = np.abs(ratio - 1.0)
        ok = np.isfinite(dev) & (dev <= 0.10)
        k = len(ns)
        while k > 0 and ok[k - 1]:
            k -= 1
        print(f"caption: forecast within "
              f"{float(np.nanmax(dev[k:])) * 100:.0f}% of actual at every "
              f"N >= {int(nsa[k])}"
              + ("; over-estimates (cautious) below that"
                 if k > 0 and np.nanmin(ratio[:k]) > 1.0 else ""))

    axA.tick_params(axis="y", labelcolor=C_RESID)
    axA.grid(True, alpha=0.3)
    # tight y-range: start just below the lowest curve point instead of 0
    # (the floor band would otherwise fill ~40% of the panel with gray)
    y_all = np.concatenate([rmse, sig[np.isfinite(sig)]])
    axA.set_ylim(float(y_all.min()) * 0.86, float(y_all.max()) * 1.06)
    _log_xticks(axA, ns)

    axA2 = axA.twinx()
    axA2.set_ylabel("Test within-sigma Spearman rho", color=C_RHO)
    axA2.plot(ns, rho, "s:", color=C_RHO, lw=1.6, ms=5, alpha=0.85,
              label=f"test Spearman rho ({n_mem}-ens mean)"
                    if n_mem > 1 else "test Spearman rho (single model)")
    axA2.tick_params(axis="y", labelcolor=C_RHO)
    axA2.set_ylim(bottom=0)
    h1, l1 = axA.get_legend_handles_labels()
    h2, l2 = axA2.get_legend_handles_labels()
    axA.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, fontsize=8.5,
               framealpha=0.9)
    axA.set_title("Does the model know how wrong it is?  "
                  "(second moments only -- no distributional assumption)",
                  fontsize=9.5, color="#555555", pad=6)

    # ---- Panel B: aleatoric vs epistemic (the "enough data" argument) ---
    if args.single:
        fig.suptitle("Learning curve: the model's own error forecast "
                     "tracks its measured error", fontsize=12)
        memn = [res[str(n)].get("n_member_train") for n in ns]
        if all(m is not None for m in memn) and any(
                abs(m - n) > 0.1 * n for m, n in zip(memn, ns)):
            fig.text(0.5, 0.005,
                     f"k-fold members: each trains on {memn[0]}-{memn[-1]} "
                     f"samples across N={ns[0]}-{ns[-1]}",
                     ha="center", va="bottom", fontsize=7.5,
                     color="#666666")
        fig.tight_layout()
        out = args.out or os.path.join(args.run_dir,
                                       "learning_curve_residual.png")
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"calibration ratio (predicted sigma / actual RMSE) at "
              f"N={ns[-1]}: {ratio[-1]:.3f}")
        print(f"[fig] -> {out}")
        return 0

    mB = np.isfinite(alea) & np.isfinite(epi)
    axB.plot(nsa[mB], alea[mB], "o-", color=C_ALEA, lw=1.8, ms=6,
             label="aleatoric (learned $\\sigma$)")
    axB.plot(nsa[mB], epi[mB], "^--", color=C_EPI, lw=1.8, ms=6,
             label="epistemic (member spread)")
    axB.axhline(noise_e, color="#777777", ls=":", lw=1.2)
    axB.annotate(f"label-noise floor ({NOISE_PCT:g} % of E)",
                 (0.03, noise_e), xycoords=("axes fraction", "data"),
                 textcoords="offset points", xytext=(0, 3),
                 ha="left", va="bottom", fontsize=8, color="#555555")
    axB.set_xlabel("Number of training samples")
    axB.set_ylabel("Predicted $\\sigma$ component (E units)")
    axB.set_yscale("log")
    axB.grid(True, alpha=0.3, which="both")
    _log_xticks(axB, [n for n in ns if n >= ENS_STD_MIN_N])
    axB.legend(fontsize=8, framealpha=0.9)
    axB.set_title("Uncertainty decomposition (law of total variance)",
                  fontsize=9.5)

    # ---- Panel C: calibration ratio (or entropy, opt-in) ----------------
    if args.entropy:
        mC = np.isfinite(ent)
        axC.plot(nsa[mC], ent[mC], "o:", color=C_UQ, lw=1.8, ms=6)
        axC.set_ylabel("Predictive entropy (nats)")
        axC.annotate(r"$H = \frac{1}{2}\ln(2\pi e\,\sigma^2)$"
                     "\n(Gaussian max-entropy bound\nfor this variance)",
                     (0.97, 0.95), xycoords="axes fraction", ha="right",
                     va="top", fontsize=8, color="#555555")
        axC.set_title("Predictive entropy", fontsize=9.5)
    else:
        mC = np.isfinite(ratio)
        axC.axhspan(0.8, 1.25, color=C_UQ, alpha=0.10, lw=0)
        axC.axhline(1.0, color="#555555", ls="--", lw=1.3)
        axC.annotate("perfectly calibrated", (0.03, 1.0),
                     xycoords=("axes fraction", "data"),
                     textcoords="offset points", xytext=(0, 4),
                     ha="left", va="bottom", fontsize=8, color="#555555")
        axC.plot(nsa[mC], ratio[mC], "D-", color=C_UQ, lw=1.8, ms=6)
        axC.set_ylabel(r"predicted $\sigma$ / actual RMSE")
        axC.set_ylim(0.0, max(1.6, float(np.nanmax(ratio[mC])) * 1.15))
        axC.annotate("under-confident\n(intervals too wide)", (0.97, 0.97),
                     xycoords="axes fraction", ha="right", va="top",
                     fontsize=7.5, color="#888888")
        axC.annotate("over-confident", (0.97, 0.03),
                     xycoords="axes fraction", ha="right", va="bottom",
                     fontsize=7.5, color="#888888")
        axC.set_title("Calibration ratio", fontsize=9.5)
    axC.set_xlabel("Number of training samples")
    axC.grid(True, alpha=0.3)
    _log_xticks(axC, [n for n in ns if n >= ENS_STD_MIN_N])

    fig.suptitle("Learning curve: error below the claimability floor, "
                 "with an honest error forecast (heteroscedastic ensemble)",
                 fontsize=12)
    if not exact:
        fig.text(0.5, 0.028,
                 "predicted $\\sigma$ aggregated as mean-of-$\\sigma$; the "
                 "RMS aggregate is larger, so the plotted ratio is a lower "
                 "bound", ha="center", va="bottom", fontsize=7.5,
                 color="#888888")

    # k-fold members train on (subset + val) minus their own rotated-out
    # fold, so the effective per-member size drifts from the x label --
    # say so on the figure rather than in a caption nobody carries around
    memn = [res[str(n)].get("n_member_train") for n in ns]
    if all(m is not None for m in memn) and any(
            abs(m - n) > 0.1 * n for m, n in zip(memn, ns)):
        fig.text(0.5, 0.005,
                 f"k-fold members: each trains on {memn[0]}-{memn[-1]} "
                 f"samples across N={ns[0]}-{ns[-1]} (subset + val fold, "
                 "minus its own rotated-out fold)",
                 ha="center", va="bottom", fontsize=7.5, color="#666666")

    out = args.out or os.path.join(args.run_dir,
                                   "learning_curve_residual.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    if np.isfinite(ratio[-1]):
        print(f"calibration ratio (predicted sigma / actual RMSE) at "
              f"N={ns[-1]}: {ratio[-1]:.3f}")
    print(f"[fig] -> {out}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    jpath = os.path.join(args.run_dir, "learning_curve.json")
    if not os.path.exists(jpath):
        raise SystemExit(f"no learning_curve.json in {args.run_dir}")
    d = json.load(open(jpath))
    ns = d["progression"]
    res = d["results"]

    # v2 (--nll-head) runs carry the model's raw self-predicted sigma:
    # render the three-panel figure instead of the v1 band-vs-residual one
    if any(np.isfinite(res[str(n)].get("mean_pred_sigma_total",
                                       float("nan"))) for n in ns):
        if args.calibrated:
            print("[note] --calibrated ignored: v2 runs are plotted with "
                  "their RAW sigma (that honesty is the point)")
        return plot_v2(ns, res, args)
    mae = np.array([res[str(n)]["test_mae"] for n in ns], float)
    rho = np.array([res[str(n)]["test_within_sigma_spearman"] for n in ns],
                   float)
    uq = np.array([res[str(n)].get("mean_ensemble_std", float("nan"))
                   if n >= ENS_STD_MIN_N else float("nan")
                   for n in ns], float)
    floor_e = FLOOR_PCT / 100.0 * Y_MEAN
    have_uq = bool(np.isfinite(uq).any())

    print("test MAE per N: "
          + ", ".join(f"{n}:{m:.4f}" for n, m in zip(ns, mae)))
    print(f"resolvability floor ({FLOOR_PCT:g}% of E) = {floor_e:.4f}")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Number of training samples")
    ax1.set_ylabel(
        r"Test residual  $|E_{\mathrm{pred}} - E_{\mathrm{true}}|$  (mean)",
        color=C_RESID)

    # --- physical floor -------------------------------------------------
    ax1.axhspan(0, floor_e, color="#bbbbbb", alpha=0.35, lw=0, zorder=0)
    ax1.axhline(floor_e, color="#777777", ls="--", lw=1.2, zorder=1)
    ax1.annotate(f"{FLOOR_PCT:.2f} % ranking-resolvability floor",
                 (0.015, 0.02), xycoords="axes fraction", ha="left",
                 va="bottom", fontsize=8.5, color="#555555")

    # --- the ensemble's self-predicted error: a +-1 sigma band AROUND
    #     the residual curve.  If the UQ were calibrated the band would be
    #     about as wide as the residual itself; a skin-tight band IS the
    #     overconfidence, read directly off the figure.
    nsa = np.asarray(ns, float)
    if have_uq:
        m = np.isfinite(uq)
        ax1.fill_between(nsa[m], (mae - uq)[m], (mae + uq)[m], color=C_UQ,
                         alpha=0.30, lw=0, zorder=2,
                         label=r"ensemble's self-predicted error"
                               r" ($\pm1\sigma$ band)")
        xf, uf = ns[-1], float(uq[-1])
    else:
        xf, uf = ns[-1], 0.00156      # deployed 5-ens, from ensemble_uq
        ax1.errorbar([xf], [mae[-1]], yerr=[[uf], [uf]], color=C_UQ,
                     lw=1.6, capsize=4, zorder=5,
                     label=r"ensemble's self-predicted error ($\pm1\sigma$)")

    # --- optional: the band width after post-hoc calibration ------------
    if args.calibrated and have_uq:
        if not os.path.exists(args.calibration):
            raise SystemExit(f"calibration not found: {args.calibration}")
        cal = json.load(open(args.calibration))
        a, b = float(cal["a"]), float(cal["b"])
        cal_band = np.sqrt(a ** 2 + (b * uq) ** 2)
        m = np.isfinite(cal_band)
        for sgn in (+1, -1):
            # residuals are magnitudes, so the lower envelope floors at 0
            env = np.clip(mae + sgn * cal_band, 0.0, None)
            ax1.plot(nsa[m], env[m], color=C_UQ,
                     lw=1.4, ls=(0, (4, 2)), alpha=0.85, zorder=4,
                     label=("after post-hoc calibration"
                            if sgn > 0 else None))
        print(f"calibrated sigma at N={ns[-1]}: {cal_band[-1]:.5f} "
              f"(raw {uq[-1]:.5f}, residual {mae[-1]:.5f})")

    ax1.plot(ns, mae, "o-", color=C_RESID, lw=1.8, ms=6, zorder=6,
             label="test residual (mean)")

    # --- name the overconfidence where the band is tightest -------------
    if have_uq and uf > 0:
        ax1.annotate(
            f"band is {mae[-1] / uf:.1f}" + r"$\times$ too narrow"
            "\n(overconfident UQ)",
            (xf, mae[-1] + uf), textcoords="offset points",
            xytext=(-6, 10), ha="right", va="bottom", fontsize=8.5,
            color=C_UQ, zorder=8)

    ax1.tick_params(axis="y", labelcolor=C_RESID)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, float(mae.max()) * 1.15)
    ax1.set_xscale("log", base=10)
    ax1.set_xlim(min(ns) * 0.9, max(ns) * 1.1)
    ticks = [ns[0]]
    for n in ns[1:-1]:
        if np.log10(n / ticks[-1]) >= 0.15:
            ticks.append(n)
    if np.log10(ns[-1] / ticks[-1]) < 0.08:
        ticks.pop()
    ticks.append(ns[-1])
    ax1.set_xticks(ticks)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.xaxis.set_minor_formatter(NullFormatter())

    ax2 = ax1.twinx()
    ax2.set_ylabel("Test within-sigma Spearman rho", color=C_RHO)
    n_mem = res[str(ns[-1])].get("n_members", 1)
    ax2.plot(ns, rho, "s--", color=C_RHO, lw=1.8, ms=6,
             label=f"test Spearman rho ({n_mem}-ens mean)"
                   if n_mem > 1 else "test Spearman rho (single model)")
    ax2.tick_params(axis="y", labelcolor=C_RHO)
    ax2.set_ylim(bottom=0)

    fig.suptitle("Learning curve: prediction residual and ranking fidelity "
                 "vs dataset size", fontsize=12)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, fontsize=8.5,
               framealpha=0.9)
    fig.tight_layout()

    out = args.out or os.path.join(args.run_dir,
                                   "learning_curve_residual.png")
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"[fig] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
