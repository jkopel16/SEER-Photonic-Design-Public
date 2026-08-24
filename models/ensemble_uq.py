"""Deep-ensemble uncertainty quantification for a surrogate bundle.

Loads an existing surrogate_bundle.pt (already trained with --ensemble N ->
N independently-seeded members) and runs dedicated UQ diagnostics on the
bundle's exact held-out test split (rebuilt via the same seed + grouping
as training, so indices match the original run).  No retraining, no
changes to the bundle; pure evaluation.

Both bundle formats are supported:
  v1 (homoscedastic): members output a single E prediction; epistemic =
      std over member D4-TTA-means, aleatoric = mean per-member TTA std.
  v2 (heteroscedastic NLL head, output_shape=2): each member predicts a
      Gaussian (mu, sigma) with variance incl. the D4-view spread; the
      ensemble is combined as a Gaussian mixture -- epistemic = member-
      mean spread, aleatoric = mean member variance, total^2 = their sum
      (identical to evaluate_and_report's mixture, model.py:1355-1367).

Decomposes per-sample predictive uncertainty (all de-normalised to
physical E units via the bundle's y_std):

  epistemic_std  = std over the N per-member D4-TTA-means
                   (deep-ensemble disagreement; the existing
                   ``ensemble_std`` in models.model.evaluate_and_report)
  aleatoric_std  = mean of per-member TTA std
                   (D4-view disagreement -- view/aleatoric sensitivity)
  total_std      = sqrt(epistemic^2 + aleatoric^2)
                   (independent-components assumption, standard deep
                   ensemble convention)
  mu             = mean over the N per-member D4-TTA-means

Headline scores (physical E units, ASCII-safe names):
  nll          Gaussian negative log-likelihood assuming N(mu, total^2)
  crps_gaussian  continuous ranked probability score (closed form,
                scipy-free, exact for a Gaussian predictive)
  picp_{1,2,3}sigma   Predictive Interval Coverage Probability at
                     +-1/2/3 sigma total_std (calibration: should be
                     ~0.683 / 0.954 / 0.997 for a well-calibrated
                     ensemble; >1.0 = conservative, <1.0 = overconfident)
  mpiw_{1,2}sigma   Mean Predictive Interval Width (sharpness)

Discrimination (uncertainty should track error):
  error_total_rho      Spearman(|y-mu|, total_std)
  error_epistemic_rho  Spearman(|y-mu|, epistemic_std)
  error_aleatoric_rho  Spearman(|y-mu|, aleatoric_std)
  err_low_std / err_high_std / oracle_ratio   mean |y-mu| in the
                                               low/high total_std half;
                                               ratio should be >1

Per-cell: reuses models.model.within_sigma_spearman cells, augmenting
each with per-cell PICP_1sigma and mean total_std (sample-size weighted
in the pooled scalars).

Outputs (in --out-dir, default <bundle dir>/uq/):
  uq_metrics.json         scalars + per_cell_uq
  uq_predictions.csv      per-sample bands
  uq_calibration.png     reliability diagram + coverage curve
  uq_sharpness.png        epistemic/aleatoric/total std histograms
  uq_discrimination.png   binned |error| vs total_std
  uq_entropy.png          differential-entropy histogram + |error| vs entropy
  uq_pred_vs_actual.png   pred vs actual with +-1 sigma error bars,
                          colored by total_std

Usage:
  python -m models.ensemble_uq \
      --bundle runs/surrogate_128_fft_nll_sweep/surrogate_bundle.pt

  # data + out-dir override; wandb off (default is on iff installed + key)
  python -m models.ensemble_uq --bundle <bundle> --data <npz> \
      --out-dir <dir> --no-wandb
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402

try:
    import wandb
except ImportError:
    wandb = None

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from models.model import (  # noqa: E402
    device,
    PhotonicCNN,
    PhotonicDataset,
    D4_TTA_OPS,
    build_input_channels,
    infer_bundle_recipe,
    stratified_group_split,
    regression_metrics,
    within_sigma_spearman,
    spearman_rho,
    predict_gaussian,
)


# ==========================================================================
# CLI
# ==========================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m models.ensemble_uq",
        description="Deep-ensemble uncertainty quantification for a "
                    "surrogate bundle (no retraining).",
    )
    ap.add_argument(
        "--bundle",
        default=os.path.join(_REPO_ROOT, "runs",
                             "surrogate_128_fft_nll_sweep",
                             "surrogate_bundle.pt"),
        help="surrogate_bundle.pt from models/model.py.",
    )
    ap.add_argument(
        "-i", "--data",
        default=os.path.join(_REPO_ROOT, "data", "samples_128.npz"),
        help="Dataset .npz that the bundle was trained on.",
    )
    ap.add_argument(
        "-o", "--out-dir", default=None,
        help="Where to write UQ artifacts. None = <bundle dir>/uq/.",
    )
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument(
        "--seed", type=int, default=None,
        help="Split seed. None = use bundle's training seed so the test "
             "split matches the original run exactly (recommended).")
    ap.add_argument(
        "--use-wandb", dest="use_wandb", action="store_true", default=True)
    ap.add_argument(
        "--no-wandb", dest="use_wandb", action="store_false")
    ap.add_argument("--project", default="solar-cell-absorption")
    return ap.parse_args(argv)


# ==========================================================================
# Per-sample UQ decomposition (the core new work)
# ==========================================================================
@torch.inference_mode()
def ensemble_uq_predictions(members, loader):
    """Return per-sample mu, epistemic_std, aleatoric_std (normalised units).

    members: list of eval-mode PhotonicCNN instances on `device`.
    loader:  batched PhotonicDataset over the test split (normalised X,y).

    For each test sample:
      mu            = mean over members of (member's D4-TTA-mean prediction)
      epistemic_std = std over members of the per-member D4-TTA-means
      aleatoric_std = mean over members of the per-member D4-TTA std

    Predictions and stds are in *normalised* y units; the caller
    de-normalises by y_std to physical E.
    """
    mu_chunks, epi_chunks, ale_chunks = [], [], []
    for batch in loader:
        X_b = batch["X"].to(device, non_blocking=True)
        member_means, member_stds = [], []  # each (B,) per member
        for m in members:
            views = torch.stack(
                [m(op(X_b)).squeeze(-1) for op in D4_TTA_OPS])  # (8, B)
            member_means.append(views.mean(dim=0))             # (B,)
            member_stds.append(views.std(dim=0, unbiased=False))
        M = torch.stack(member_means)  # (N, B)
        S = torch.stack(member_stds)   # (N, B)
        mu_chunks.append(M.mean(dim=0).float().cpu())
        epi_chunks.append(M.std(dim=0, unbiased=False).float().cpu())
        ale_chunks.append(S.mean(dim=0).float().cpu())
    return (torch.cat(mu_chunks), torch.cat(epi_chunks),
            torch.cat(ale_chunks))


@torch.inference_mode()
def ensemble_uq_predictions_hetero(members, loader):
    """v2 (NLL-head) bundles: per-sample mu, epistemic_std, aleatoric_std.

    Each member emits a Gaussian (mu_i, sigma_i) per sample, its variance
    already including the D4-view spread (law of total variance over the
    orbit, see models.model.predict_gaussian).  The ensemble is combined
    as a Gaussian mixture -- identical to evaluate_and_report's v2 branch
    (model.py:1355-1367) -- but split into components so the v1
    decomposition plots/scalars keep their meaning:

      mu           = mean_i mu_i
      aleatoric^2  = mean_i var_i          (mean member variance)
      epistemic^2  = var_i(mu_i)           (member-mean spread)
      total^2      = aleatoric^2 + epistemic^2
                   = mean_i(var_i + mu_i^2) - mu^2   (mixture variance)

    Normalised units; the caller de-normalises by y_std.
    """
    mem_mu, mem_var = [], []  # each (n_test,) per member
    for m in members:
        mu_i, sig_i, _ = predict_gaussian(m, loader, use_tta=True)
        mem_mu.append(mu_i)
        mem_var.append(sig_i ** 2)
    MU = torch.stack(mem_mu)     # (N, n_test)
    S2 = torch.stack(mem_var)    # (N, n_test)
    mu = MU.mean(dim=0)
    ale = S2.mean(dim=0).sqrt()
    epi = MU.std(dim=0, unbiased=False)
    return mu.float().cpu(), epi.float().cpu(), ale.float().cpu()


def gaussian_nll(y, mu, sigma):
    """Mean Gaussian negative log-likelihood (lower is better)."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float, copy=True)
    sigma = np.clip(sigma, 1e-10, None)
    return float(np.mean(
        0.5 * np.log(2 * np.pi * sigma ** 2) + ((y - mu) ** 2) / (2 * sigma ** 2)))


def crps_gaussian(y, mu, sigma):
    """Mean CRPS for a Gaussian predictive (closed form, scipy-free).

    CRPS = sigma * [ z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
    where z = (y - mu)/sigma and phi/Phi use erf -- exact for the
    Gaussian forecast, proper, negatively oriented (lower is better).
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float, copy=True)
    sigma = np.clip(sigma, 1e-10, None)
    z = (y - mu) / sigma
    sqrt2 = np.sqrt(2.0)
    Phi = 0.5 * (1.0 + _erf(z / sqrt2))
    phi = np.exp(-(z ** 2) / 2.0) / np.sqrt(2.0 * np.pi)
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def _erf(z):
    """Vectorised erf via Abramowitz-Stegun 7.1.26 rational approximation.

    scipy-free (numpy-only): |err| < 1.5e-7, well below the engine noise
    floor that bounds the UQ conclusions. requirements does pin scipy,
    but keeping this dependency-light makes the metric self-contained.
    """
    z = np.asarray(z, dtype=float)
    sign = np.sign(z)
    x = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * x)
    a1, a2, a3, a4, a5 = (0.254829592, -0.284496736, 1.421413741,
                          -1.453152027, 1.061405429)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t \
        * np.exp(-x * x)
    return sign * y


def gaussian_coverage(k):
    """2*Phi(k) - 1 = 2*erf(k/sqrt2) - 1, the +-k sigma tail cumulative."""
    return 2.0 * _erf(np.asarray(k, dtype=float) / np.sqrt(2.0)) - 1.0


def picp(y, mu, sigma, k):
    """Predictive Interval Coverage Probability at +-k sigma."""
    lo = np.asarray(mu) - k * np.asarray(sigma)
    hi = np.asarray(mu) + k * np.asarray(sigma)
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


def mpiw(sigma, k):
    """Mean Predictive Interval Width at +-k sigma."""
    return float(np.mean(2.0 * k * np.asarray(sigma)))


def differential_entropy_gaussian(sigma):
    """Per-sample differential entropy of N(mu, sigma^2) in nats.

    H = 0.5 * ln(2*pi*e*sigma^2).  Constant offset 0.5*ln(2*pi*e) is kept
    so absolute values are interpretable; the metric is monotonic in
    sigma.  Clipped to avoid log(0).
    """
    sigma = np.clip(np.asarray(sigma, dtype=float), 1e-10, None)
    return 0.5 * np.log(2.0 * np.pi * np.e * sigma ** 2)


def reliability_diagram(y, mu, sigma, n_bins=10):
    """Reliability diagram bins: predicted CDF quantile at y_true.

    Returns (bin_centers, empirical_freq, n_per_bin) where bin_centers are
    the predicted CDF quantiles binned into n_bins deciles of [0,1] and
    empirical_freq is the observed fraction of samples whose y_true falls in
    that decile.  Perfect calibration = identity.  Used purely for the
    figure (the scalars live in uq_metrics.json)."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.clip(np.asarray(sigma, dtype=float), 1e-10, None)
    # predicted CDF of y_true under N(mu, sigma^2) via erf (same _erf)
    z = (y - mu) / sigma
    cdf = 0.5 * (1.0 + _erf(np.clip(z, -8.0, 8.0) / np.sqrt(2.0)))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    empirical = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        m = (cdf >= bin_edges[i]) & (cdf < bin_edges[i + 1])
        if i == n_bins - 1:
            m = (cdf >= bin_edges[i]) & (cdf <= bin_edges[i + 1])
        counts[i] = int(m.sum())
        empirical[i] = counts[i] / max(len(cdf), 1)
    return centers, empirical, counts


# ==========================================================================
# Figures
# ==========================================================================
def plot_calibration(y, mu, sigma):
    """2-panel: reliability diagram + coverage curve at 1/2/3 sigma."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    centers, emp, n = reliability_diagram(y, mu, sigma, n_bins=10)
    ax = axes[0]
    ax.bar(centers, emp, width=centers[1] - centers[0] if len(centers) > 1
           else 0.1, alpha=0.55, edgecolor="black", linewidth=0.5,
           label="observed")
    ax.plot([0, 1], [0, 1] if emp.max() < 1 else [0, emp.max()],
            "k--", linewidth=1, label="perfect")
    ax.set_xlabel("predicted CDF decile of y_true")
    ax.set_ylabel("fraction of samples in bin")
    ax.set_title("Reliability diagram\n(predicted vs observed CDF)")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)

    ax = axes[1]
    ks = np.array([1.0, 2.0, 3.0])
    nominal = gaussian_coverage(ks)
    empirical = np.array([picp(y, mu, sigma, k) for k in ks])
    x = np.arange(len(ks))
    ax.bar(x - 0.2, nominal, width=0.4, alpha=0.55, edgecolor="black",
           linewidth=0.5, label="nominal")
    ax.bar(x + 0.2, empirical, width=0.4, alpha=0.55, edgecolor="black",
           linewidth=0.5, label="empirical")
    for i, (nm, em) in enumerate(zip(nominal, empirical)):
        ax.text(i - 0.2, nm + 0.01, f"{nm:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.2, em + 0.01, f"{em:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"+-{k:.0f} sigma" for k in ks])
    ax.set_ylabel("coverage")
    ax.set_ylim(0, 1.15)
    ax.set_title("Coverage: nominal vs empirical PICP")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_sharpness(epi, ale, total):
    """Histograms of epistemic / aleatoric / total std."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = 30
    ax.hist(epi, bins=bins, alpha=0.55, label=f"epistemic "
            f"(mean={epi.mean():.5f})", edgecolor="black", linewidth=0.4)
    ax.hist(ale, bins=bins, alpha=0.55, label=f"aleatoric "
            f"(mean={ale.mean():.5f})", edgecolor="black", linewidth=0.4)
    ax.hist(total, bins=bins, alpha=0.4, label=f"total "
            f"(mean={total.mean():.5f})", edgecolor="black", linewidth=0.4,
            color="gray", histtype="stepfilled")
    ax.set_xlabel("predictive std (physical E units)")
    ax.set_ylabel("count")
    ax.set_title("Sharpness: predictive uncertainty decomposition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_discrimination(y, mu, sigma):
    """Binned |y-mu| vs sigma (10 total_std quantile bins)."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    abs_err = np.abs(y - mu)
    n_bins = 10
    # 10 quantile bins of sigma
    edges = np.quantile(sigma, np.linspace(0, 1, n_bins + 1))
    edges[-1] = edges[-1] * (1.0 + 1e-9)  # include the max
    centers, bin_mean_err, bin_mean_std, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (sigma >= lo) & (sigma < hi) if i < n_bins - 1 \
            else (sigma >= lo) & (sigma <= hi)
        if m.sum() < 1:
            continue
        centers.append((lo + hi) / 2.0)
        bin_mean_err.append(float(abs_err[m].mean()))
        bin_mean_std.append(float(sigma[m].mean()))
        counts.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bin_mean_std, bin_mean_err, "o-", markersize=6,
            linewidth=1.5)
    for x, yc, n in zip(bin_mean_std, bin_mean_err, counts):
        ax.text(x, yc, f" n={n}", fontsize=7, va="bottom")
    ax.set_xlabel("mean total_std in quantile bin")
    ax.set_ylabel("mean |y - mu| in bin")
    rho = spearman_rho(abs_err, sigma)
    ax.set_title(f"Discrimination: |error| vs total_std\n"
                 f"Spearman rho(|err|, total_std) = {rho:+.3f} "
                 f"(>0 = useful uncertainty)")
    ax.axhline(abs_err.mean(), color="gray", linestyle=":", linewidth=0.7,
              label=f"mean |err|={abs_err.mean():.5f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_entropy(y, mu, sigma):
    """2-panel: differential-entropy histogram + binned |error| vs entropy."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    entropy = differential_entropy_gaussian(sigma)
    abs_err = np.abs(y - mu)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: entropy histogram
    ax = axes[0]
    ax.hist(entropy, bins=30, alpha=0.55, edgecolor="black", linewidth=0.4,
            label=f"mean H={entropy.mean():.4f} nats")
    ax.axvline(entropy.mean(), color="red", linestyle="--", linewidth=1,
               label=f"mean={entropy.mean():.4f}")
    ax.axvline(np.median(entropy), color="gray", linestyle=":",
               linewidth=1, label=f"median={np.median(entropy):.4f}")
    ax.set_xlabel("predictive differential entropy (nats)")
    ax.set_ylabel("count")
    ax.set_title("Predictive entropy distribution\n"
                 "(Gaussian differential entropy, N(mu, total^2))")
    ax.legend(fontsize=8)

    # Panel B: |error| binned by entropy quantiles (pattern: plot_discrimination)
    ax = axes[1]
    n_bins = 10
    edges = np.quantile(entropy, np.linspace(0, 1, n_bins + 1))
    edges[-1] = edges[-1] * (1.0 + 1e-9)
    bin_mean_err, bin_mean_h, counts = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (entropy >= lo) & (entropy < hi) if i < n_bins - 1 \
            else (entropy >= lo) & (entropy <= hi)
        if m.sum() < 1:
            continue
        bin_mean_err.append(float(abs_err[m].mean()))
        bin_mean_h.append(float(entropy[m].mean()))
        counts.append(int(m.sum()))
    ax.plot(bin_mean_h, bin_mean_err, "o-", markersize=6, linewidth=1.5)
    for x, yc, n in zip(bin_mean_h, bin_mean_err, counts):
        ax.text(x, yc, f" n={n}", fontsize=7, va="bottom")
    rho = spearman_rho(abs_err, entropy)
    ax.set_xlabel("mean predictive entropy in quantile bin (nats)")
    ax.set_ylabel("mean |y - mu| in bin")
    ax.set_title(f"Entropy discrimination: |error| vs H\n"
                 f"Spearman rho(|err|, H) = {rho:+.3f} "
                 f"(>0 = useful uncertainty)")
    ax.axhline(abs_err.mean(), color="gray", linestyle=":", linewidth=0.7,
               label=f"mean |err|={abs_err.mean():.5f}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_pred_vs_actual(y, mu, sigma, cls=None, sig=None):
    """pred vs actual scatter with +-1 sigma error bars, colored by std."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 7))
    lims = [min(y.min(), mu.min()), max(y.max(), mu.max())]
    sc = ax.scatter(y, mu, c=sigma, cmap="viridis", s=22, alpha=0.85,
                    edgecolor="none", zorder=3)
    # subsample error bars for legibility (too many drown the line)
    n = len(y)
    idx = (np.arange(min(n, 80)) * max(1, n // 80))
    idx = np.clip(idx, 0, n - 1)
    ax.errorbar(y[idx], mu[idx], yerr=sigma[idx], fmt="none",
                ecolor="gray", alpha=0.6, elinewidth=0.7, capsize=2,
                zorder=2, label="+-1 sigma")
    ax.plot(lims, lims, "r--", linewidth=1, zorder=1)
    fig.colorbar(sc, ax=ax, label="total_std")
    ax.set_xlabel("Actual E")
    ax.set_ylabel("Predicted E (mu)")
    ax.set_title("Predicted vs actual with +-1 sigma predictive bands\n"
                 "(color = total_std)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def sparsification_curves(y, mu, sigma):
    """Sparsification: drop the most-uncertain points first, watch the
    error of what remains.  DISTRIBUTION-FREE -- uses only the RANKING of
    sigma, no interval, no Gaussian (Ilg et al., ECCV 2018).

    Returns (frac_removed, rmse_by_sigma, rmse_oracle, ause) where the
    oracle removes by true |error| (best possible ranking) and
    ause = area between the two curves, each normalised by the full-set
    RMSE.  0 = sigma ranks errors perfectly; larger = worse.
    """
    y = np.asarray(y, float)
    err2 = (y - np.asarray(mu, float)) ** 2
    n = len(err2)
    fracs = np.arange(n) / n

    def curve(order):
        e2 = err2[order]                          # removal order
        tail = np.cumsum(e2[::-1])[::-1]          # sum of e2 from k..end
        return np.sqrt(tail / np.arange(n, 0, -1))

    c_uq = curve(np.argsort(-np.asarray(sigma, float)))
    c_or = curve(np.argsort(-err2))
    r0 = float(c_uq[0])                           # full-set RMSE
    ause = float(np.trapezoid((c_uq - c_or) / r0, fracs)
                 if hasattr(np, "trapezoid")
                 else np.trapz((c_uq - c_or) / r0, fracs))
    return fracs, c_uq, c_or, ause


def plot_error_calibration(y, mu, sigma, n_bins=6):
    """Error-based calibration (Levi et al. 2022; Tran et al. 2020): the
    model's error FORECAST vs the MEASURED error, binned.

    Sort the test set by predicted sigma, cut into equal-count bins; for
    each bin plot (RMS predicted sigma, RMS observed error).  Points on
    the identity line mean the forecast matches the measurement.  Second
    moments only: no interval, no coverage level, no assumption -- or
    mention -- of any error distribution.
    """
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    err2 = (y - mu) ** 2
    order = np.argsort(sigma)
    bins = np.array_split(order, n_bins)

    px, py, pe = [], [], []
    for b in bins:
        px.append(float(np.sqrt(np.mean(sigma[b] ** 2))))
        rm = float(np.sqrt(np.mean(err2[b])))
        py.append(rm)
        # sampling noise of an RMS over n_b points ~ rm/sqrt(2 n_b)
        pe.append(rm / np.sqrt(2 * len(b)))

    abs_err = np.sqrt(err2)
    lim = 1.05 * max(1.1 * max(max(px), max(py) + max(pe)),
                     float(abs_err.max()), float(sigma.max()))
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, lim], [0, lim], "--", color="#555555", lw=1.4,
            label="forecast = measurement")
    # individual layouts, background context only: each dot is ONE
    # residual, and a single draw scatters widely around its own typical
    # size even under a perfect forecast -- the binned RMS points are the
    # quantity the model actually forecasts
    ax.scatter(sigma, abs_err, s=9, color="#bbbbbb", alpha=0.45, lw=0,
               zorder=2, label="individual layouts ($|$error$|$)")
    ax.errorbar(px, py, yerr=pe, fmt="o", color="#7b3fa0", ms=8,
                lw=1.6, capsize=3, zorder=5,
                label=f"test set in {n_bins} bins by predicted error")
    ov_p = float(np.sqrt(np.mean(sigma ** 2)))
    ov_m = float(np.sqrt(np.mean(err2)))
    ax.plot([ov_p], [ov_m], "*", color="#d62728", ms=16, zorder=6,
            label=f"whole test set (n={len(y)}, {ov_p / ov_m:.2f}x)")
    ax.set_xlabel("Model's predicted error, RMS (E units)")
    ax.set_ylabel("Measured error, RMS (E units)")
    ax.set_title("The model forecasts its own error\n"
                 "(forecast vs measurement -- no distributional "
                 "assumption)", fontsize=10)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, framealpha=0.9, loc="upper left")
    fig.tight_layout()
    return fig


def plot_sparsification(y, mu, sigma):
    fracs, c_uq, c_or, ause = sparsification_curves(y, mu, sigma)
    r0 = c_uq[0]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.axhline(r0, color="#999999", ls=":", lw=1.4,
               label="random removal (no skill)")
    ax.plot(fracs, c_uq, "-", color="#7b3fa0", lw=2.0,
            label="remove by predicted $\\sigma$ (largest first)")
    ax.plot(fracs, c_or, "--", color="#555555", lw=1.6,
            label="oracle: remove by true $|$error$|$")
    ax.fill_between(fracs, c_or, c_uq, color="#7b3fa0", alpha=0.15, lw=0)
    ax.annotate(f"AUSE = {ause:.3f}\n(0 = perfect error ranking)",
                (0.03, 0.06), xycoords="axes fraction", ha="left",
                va="bottom", fontsize=9, color="#7b3fa0")
    ax.set_xlabel("Fraction of test set removed (most uncertain first)")
    ax.set_ylabel("RMSE of remaining samples (E units)")
    ax.set_title("Sparsification: does $\\sigma$ rank the errors?  "
                 "(rank-based, no distributional assumption)", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, framealpha=0.9, loc="upper right")
    fig.tight_layout()
    return fig


# ==========================================================================
# Main
# ==========================================================================
def main(argv=None):
    args = parse_args(argv)
    if not os.path.exists(args.bundle):
        raise FileNotFoundError(f"bundle not found: {args.bundle}")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"dataset not found: {args.data}")

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(args.bundle)),
                                    "uq")
    os.makedirs(args.out_dir, exist_ok=True)

    # --- load bundle ---
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    fmt = bundle.get("format")
    if fmt not in ("photonic-surrogate-bundle-v1",
                   "photonic-surrogate-bundle-v2"):
        raise SystemExit(f"unrecognized bundle format in {args.bundle}: {fmt}")
    hetero = bool(bundle.get("heteroscedastic", False)) or \
        int(bundle.get("arch", {}).get("output_shape", 1)) == 2
    n_members = len(bundle["state_dicts"])
    if n_members < 2:
        raise SystemExit(
            f"bundle has only {n_members} member(s); deep-ensemble UQ "
            "needs >= 2. Retrain with --ensemble N (N>=2).")
    arch = bundle["arch"]
    n = bundle["norm"]
    y_mean, y_std = float(n["y_mean"]), float(n["y_std"])
    img_size = int(bundle["img_size"])
    recipe = infer_bundle_recipe(bundle)
    print(f"[uq] bundle: {args.bundle}")
    print(f"[uq] {n_members} members, arch={arch}, img={img_size}")
    if hetero:
        print("[uq] v2 heteroscedastic NLL bundle: mixture-combined "
              "Gaussian members (epistemic = member-mean spread, "
              "aleatoric = mean member variance)")
    print(f"[uq] channel recipe: {recipe}")

    # --- load dataset and rebuild the bundle's exact test split ---
    data = np.load(args.data, allow_pickle=False)
    X = torch.from_numpy(data["X"]).float()
    y = torch.from_numpy(data["y"]).float()
    if X.dim() == 3:
        X = X.unsqueeze(1)
    if int(X.shape[-1]) != img_size:
        raise SystemExit(
            f"dataset raster {int(X.shape[-1])}px != bundle img_size "
            f"{img_size}px -- wrong dataset?")

    b_recipe = recipe
    ds_recipe = ([str(r) for r in data["channel_recipe"]]
                 if "channel_recipe" in data.files
                 else (["raster"] if X.shape[1] == 1
                       else ["raster", "fft_baked_v1"]))
    if ds_recipe == b_recipe:
        pass
    else:
        print(f"[uq] rebuilding channels {b_recipe} from the raster "
              f"(dataset carries {ds_recipe})")
        X = build_input_channels(X[:, :1], b_recipe)

    split_seed = (args.seed if args.seed is not None
                  else bundle["train_config"].get("seed", 137))
    if args.seed is None:
        print(f"[uq] using bundle training seed {split_seed} for the "
              "test split (matches the original run).")
    else:
        print(f"[uq] using override --seed {split_seed}")
    groups = data["sample_id"] if "sample_id" in data.files else None
    _, _, test_idx = stratified_group_split(
        data["sigma"], groups=groups, seed=split_seed)
    print(f"[uq] test n={len(test_idx)} "
          f"(grouped={'yes' if groups is not None else 'no'})")

    xm = torch.as_tensor(n["x_mean"], dtype=torch.float32).reshape(1, -1, 1, 1)
    xs = torch.as_tensor(n["x_std"], dtype=torch.float32).reshape(1, -1, 1, 1)
    X_norm = (X - xm) / xs
    # data['y'] is in PHYSICAL E units (matches the CSV's y_true); the
    # bundle's y_mean/y_std are normalisation stats.  The loader needs
    # NORMALISED targets (matching the model's output space), and the
    # de-normalisation mu/std * y_std + y_mean happens downstream -- mirror
    # eval_bundle_main (model.py:1265) exactly.  Earlier versions fed raw y
    # in and were de-normalising TWICE (-> 12x inflated targets / sanity
    # check failure).
    y_norm = (y - y_mean) / y_std
    test_loader = DataLoader(
        PhotonicDataset(X_norm[test_idx], y_norm[test_idx]),
        batch_size=args.batch_size, shuffle=False)

    members = []
    for sd in bundle["state_dicts"]:
        m = PhotonicCNN(**arch)
        m.load_state_dict(sd)
        m.to(device).eval()
        members.append(m)

    # --- per-sample UQ decomposition ---
    if hetero:
        mu_n, epi_n, ale_n = ensemble_uq_predictions_hetero(
            members, test_loader)
    else:
        mu_n, epi_n, ale_n = ensemble_uq_predictions(members, test_loader)
    mu = (mu_n * y_std + y_mean).numpy()                       # physical E
    epi = (epi_n * y_std).numpy()                             # physical E
    ale = (ale_n * y_std).numpy()                             # physical E
    total = np.sqrt(epi ** 2 + ale ** 2)
    # data['y'] is already physical E, and y_norm = (y - y_mean)/y_std was
    # fed to the loader; de-normalising the loader's targets via
    # y_norm[te]*y_std + y_mean is identity with y[te].  Use raw y[te].
    targets = y[test_idx].numpy()

    # --- sanity: ensemble-mean MAE must match the bundle's stored test MAE ---
    stored = bundle.get("test_metrics", {})
    metrics = regression_metrics(targets, mu)
    if "mae" in stored:
        d = abs(metrics["mae"] - float(stored["mae"]))
        if d > 1e-3:
            raise SystemExit(
                f"sanity check failed: re-eval MAE {metrics['mae']:.6f} "
                f"!= stored {stored['mae']:.6f} (delta {d:.6f}). The test "
                "split is not identical to the training run -- check "
                "--data / --seed vs. the bundle's train_config.")
        print(f"[uq] sanity OK: re-eval MAE {metrics['mae']:.6f} matches "
              f"stored {stored['mae']:.6f}")
    if hetero and "nll_picp_1sigma" in stored:
        p1 = picp(targets, mu, total, 1.0)
        if abs(p1 - float(stored["nll_picp_1sigma"])) > 0.02:
            raise SystemExit(
                f"sanity check failed: hetero re-eval PICP1 {p1:.4f} != "
                f"stored {stored['nll_picp_1sigma']:.4f}. The mixture "
                "combination does not reproduce the training run -- "
                "check --data / --seed vs. the bundle's train_config.")
        print(f"[uq] sanity OK: hetero PICP1 {p1:.4f} matches stored "
              f"{stored['nll_picp_1sigma']:.4f}")

    sigma_test = np.asarray(data["sigma"], dtype=float)[test_idx]
    cls_test = (np.asarray(data["disorder_class"])[test_idx]
                if "disorder_class" in data.files else None)
    sid_test = (np.asarray(data["sample_id"])[test_idx]
                if "sample_id" in data.files else np.asarray(test_idx))

    # --- scalar UQ metrics ---
    abs_err = np.abs(targets - mu)
    metrics.update({
        "n_members": n_members,
        "n_test": int(len(targets)),
        "split_seed": int(split_seed),
        # proper scores
        "nll": gaussian_nll(targets, mu, total),
        "crps_gaussian": crps_gaussian(targets, mu, total),
        # calibration
        "picp_1sigma": picp(targets, mu, total, 1.0),
        "picp_2sigma": picp(targets, mu, total, 2.0),
        "picp_3sigma": picp(targets, mu, total, 3.0),
        # sharpness
        "mpiw_1sigma": mpiw(total, 1.0),
        "mpiw_2sigma": mpiw(total, 2.0),
        "mean_epistemic_std": float(epi.mean()),
        "mean_aleatoric_std": float(ale.mean()),
        "mean_total_std": float(total.mean()),
        "median_total_std": float(np.median(total)),
        # discrimination
        "error_total_rho": spearman_rho(abs_err, total),
        "error_epistemic_rho": spearman_rho(abs_err, epi),
        "error_aleatoric_rho": spearman_rho(abs_err, ale),
    })
    # oracle ratio: high-std half should have higher mean error
    mid = np.median(total)
    lo = abs_err[total <= mid]
    hi = abs_err[total > mid]
    metrics["err_low_std"] = float(lo.mean()) if len(lo) else float("nan")
    metrics["err_high_std"] = float(hi.mean()) if len(hi) else float("nan")
    metrics["oracle_ratio"] = (metrics["err_high_std"] / metrics["err_low_std"]
                               if metrics["err_low_std"] > 0 else float("nan"))
    # area under the sparsification error curve (rank-based, see
    # sparsification_curves): 0 = sigma orders the errors perfectly
    metrics["ause_rmse"] = sparsification_curves(targets, mu, total)[3]
    # predictive entropy (Gaussian differential entropy, nats)
    pred_entropy = differential_entropy_gaussian(total)
    metrics["mean_predictive_entropy"] = float(pred_entropy.mean())
    metrics["median_predictive_entropy"] = float(np.median(pred_entropy))
    metrics["error_entropy_rho"] = spearman_rho(abs_err, pred_entropy)

    print("\n=== deep-ensemble UQ metrics ===")
    for k in ["mae", "rmse", "r2", "nll", "crps_gaussian",
              "picp_1sigma", "picp_2sigma", "picp_3sigma",
              "mpiw_1sigma", "mean_epistemic_std", "mean_aleatoric_std",
              "mean_total_std", "mean_predictive_entropy",
              "median_predictive_entropy", "error_total_rho",
              "error_entropy_rho", "error_epistemic_rho",
              "error_aleatoric_rho", "err_low_std", "err_high_std",
              "oracle_ratio"]:
        v = metrics.get(k, float("nan"))
        if isinstance(v, float):
            print(f"  {k:24s} {v:.6f}")
        else:
            print(f"  {k:24s} {v}")

    # --- per-cell UQ ---
    rho_pooled, rho_cells = within_sigma_spearman(
        targets, mu, sigma_test, cls_test)
    metrics["within_sigma_spearman"] = rho_pooled
    per_cell_uq = {}
    # rho_cells keys are STRINGS built as f"{class}/sigma={sigma:g}" with
    # NaN sigma mapped to inf (model.py:752,769) -- rebuild the identical
    # per-sample key and mask by string equality
    sig_key = np.where(np.isnan(sigma_test), np.inf, sigma_test)
    cls_key = (cls_test if cls_test is not None
               else np.full(len(targets), None))
    sample_cell = np.array([f"{c}/sigma={s:g}"
                            for c, s in zip(cls_key, sig_key)])
    for cell, d in rho_cells.items():
        key = sample_cell == cell
        if key.sum() < 1:
            per_cell_uq[cell] = {**d,
                                 "picp_1sigma": float("nan"),
                                 "mean_total_std": float("nan")}
            continue
        per_cell_uq[cell] = {
            **d,
            "picp_1sigma": picp(targets[key], mu[key], total[key], 1.0),
            "mean_total_std": float(total[key].mean()),
        }
    metrics["per_cell_uq"] = per_cell_uq
    print("\n=== per-cell UQ ===")
    for cell, d in sorted(per_cell_uq.items()):
        print(f"  {cell:28s} rho={d['rho']:+.3f}  "
              f"picp1={d['picp_1sigma']:.3f}  "
              f"std={d['mean_total_std']:.5f}  n={d['n']}")

    # --- write uq_metrics.json ---
    json_path = os.path.join(args.out_dir, "uq_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[uq] wrote {json_path}")

    # --- write uq_predictions.csv ---
    csv_path = os.path.join(args.out_dir, "uq_predictions.csv")
    cls_arr = (cls_test if cls_test is not None
               else np.full(len(targets), "?"))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "disorder_class", "sigma",
                    "y_true", "mu",
                    "epistemic_std", "aleatoric_std", "total_std",
                    "abs_err", "in_1sigma"])
        for sid, dc, sg, yt, m, ep, al, tot, ae in zip(
                sid_test, cls_arr, sigma_test, targets, mu,
                epi, ale, total, abs_err):
            w.writerow([sid, dc, sg, f"{yt:.6f}", f"{m:.6f}",
                        f"{ep:.6f}", f"{al:.6f}", f"{tot:.6f}",
                        f"{ae:.6f}", int(abs(ae) <= tot)])
    print(f"[uq] wrote {csv_path}")

    # --- figures ---
    figs = {
        "uq_calibration.png": plot_calibration(targets, mu, total),
        "uq_sharpness.png": plot_sharpness(epi, ale, total),
        "uq_discrimination.png": plot_discrimination(targets, mu, total),
        "uq_sparsification.png": plot_sparsification(targets, mu, total),
        "uq_error_calibration.png": plot_error_calibration(
            targets, mu, total),
        "uq_entropy.png": plot_entropy(targets, mu, total),
        "uq_pred_vs_actual.png": plot_pred_vs_actual(
            targets, mu, total, cls_test, sigma_test),
    }
    for name, fig in figs.items():
        p = os.path.join(args.out_dir, name)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[uq] wrote {p}")

    # --- wandb (mirror eval_bundle_main) ---
    run = None
    if args.use_wandb and wandb is not None:
        run = wandb.init(
            project=args.project,
            name=f"uq-{os.path.basename(os.path.dirname(args.bundle))}",
            config={"bundle": args.bundle, "data": args.data,
                    "out_dir": args.out_dir, "split_seed": split_seed},
            job_type="ensemble_uq")
    if run is not None:
        wandb.summary.update(metrics)
        wandb.log({name.replace(".png", ""): wandb.Image(p)
                   for name, p in [(k, f"{args.out_dir}/{k}")
                                   for k in figs]})
        # save artifacts Files tab (paths, not Figure objects)
        for p in [json_path, csv_path,
                  *[os.path.join(args.out_dir, k) for k in figs]]:
            wandb.save(p, base_path=args.out_dir)
        # per-cell table
        table = wandb.Table(columns=["cell", "rho", "picp_1sigma",
                                     "mean_total_std", "n"])
        for cell, d in sorted(per_cell_uq.items()):
            table.add_data(cell, d["rho"], d["picp_1sigma"],
                           d["mean_total_std"], d["n"])
        wandb.log({"per_cell_uq": table})
        wandb.finish()

    print(f"\n[uq] picp1={metrics['picp_1sigma']:.3f} "
          f"picp2={metrics['picp_2sigma']:.3f} "
          f"picp3={metrics['picp_3sigma']:.3f} "
          f"nll={metrics['nll']:.6f} "
          f"err_tot_rho={metrics['error_total_rho']:+.3f} "
          f"oracle_ratio={metrics['oracle_ratio']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
