"""Post-hoc recalibration of the deep-ensemble uncertainty (no retraining).

The deployed ensemble is ~4.5x overconfident (test PICP at +-1/2/3 sigma:
0.21/0.37/0.48 vs nominal 0.683/0.954/0.997) and its residuals are
heavy-tailed, so no single scale factor can fix every level.  This script
fits a two-stage post-hoc calibration on the VALIDATION split and reports
before/after on the untouched TEST split:

  Stage 1 (parametric).  Quadrature-affine variance map fit by Gaussian
  NLL on val:
      sigma_cal(x)^2 = a^2 + (b * sigma_raw(x))^2        a, b >= 0
  `a` (physical E units) absorbs the homoscedastic error floor -- label
  noise plus model error that member disagreement never saw; `b` rescales
  whatever per-point signal sigma_raw carries.  Because the raw std
  barely discriminates error (Spearman(|err|, sigma) ~ 0.065), expect
  `a` to dominate; that is the honest outcome, not a failure.  Fit by a
  deterministic two-round grid search (numpy-only, no optimizer state).

  Stage 2 (distribution-free).  Split-conformal intervals with the
  recalibrated std as the conformal scale: on val, scores
      s_i = |y_i - mu_i| / sigma_cal_i
  and for a nominal central level p the conformal factor is the
  ceil((n+1)p)-th smallest score:
      interval(p) = mu +- q_hat(p) * sigma_cal
  This carries the standard finite-sample marginal guarantee
  coverage >= p for p <= n/(n+1) (~0.996 at n=272), including the heavy
  tails that defeat any Gaussian scalar.  For levels above that (the
  3-sigma nominal 0.9973) q_hat falls back to the max val score and the
  guarantee is capped -- stated, not hidden.

Contract: calibration parameters are fit on VAL ONLY (the split the
bundle's training run used for early stopping -- mildly optimistic, and
negligible against the 4.5x miscalibration); every reported number is
computed on TEST, which no fitting step ever touches.

Downstream: models/inverse_design.py can consume calibration.json and
form LCB = mu - kappa * sqrt(a^2 + (b*sigma_epi)^2).  A pure rescale is
mathematically a kappa change (ranking-invariant), so no deployed,
FDTD-verified campaign needs re-running.

Why not retrain: weights, hyperparameters, splits, and the SmoothL1
objective are untouched; mu -- hence MAE/R^2 and every ranking metric --
is bit-identical before/after.  A retrain-with-NLL-head alternative
(2-output head, beta-NLL loss, lr/beta re-tune) is the escalation path
only if homoscedastic-but-honest intervals prove insufficient.

Outputs (in --out-dir, default <bundle dir>/uq/):
  calibration.json          a, b, conformal level->q_hat table, sorted
                            val scores, before/after metrics, sanity gate
  uq_recalibration.png/pdf  paper figure: coverage curve + PICP bars

Usage:
  python -m models.calibrate_uq            # deployed bundle, defaults
  python -m models.calibrate_uq --no-conformal   # Stage-1-only ablation
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
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
    build_input_channels,
    infer_bundle_recipe,
    stratified_group_split,
    regression_metrics,
)
from models.ensemble_uq import (  # noqa: E402
    ensemble_uq_predictions,
    ensemble_uq_predictions_hetero,
    gaussian_nll,
    crps_gaussian,
    picp,
    mpiw,
    _erf,
)

# the exact central coverages of +-1/2/3 sigma for a Gaussian; the paper
# quotes calibration at these levels alongside the sweep grid
SIGMA_LEVELS = (0.6826894921370859, 0.9544997361036416, 0.9973002039367398)

COLORS = {"before": "#c0392b", "gauss": "#7b3fa0", "conformal": "#1f5fa8"}


# ==========================================================================
# CLI
# ==========================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m models.calibrate_uq",
        description="Fit post-hoc UQ calibration on the val split; report "
                    "before/after coverage on the untouched test split.")
    ap.add_argument(
        "--bundle",
        default=os.path.join(_REPO_ROOT, "runs",
                             "surrogate_128_fft_nll_sweep",
                             "surrogate_bundle.pt"))
    ap.add_argument(
        "-i", "--data",
        default=os.path.join(_REPO_ROOT, "data", "samples_128.npz"))
    ap.add_argument(
        "-o", "--out-dir", default=None,
        help="Where to write calibration.json + figure. "
             "None = <bundle dir>/uq/.")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument(
        "--seed", type=int, default=None,
        help="Split seed. None = bundle's training seed (recommended; "
             "reproduces the exact train/val/test partition).")
    ap.add_argument(
        "--levels", default="0.05:0.99:0.02",
        help="Coverage-curve grid as start:stop:step (nominal central "
             "levels; the +-1/2/3 sigma Gaussian levels are always added).")
    ap.add_argument(
        "--no-conformal", dest="conformal", action="store_false",
        default=True,
        help="Ablation: Stage-1 (Gaussian recalibration) only.")
    ap.add_argument(
        "--use-wandb", dest="use_wandb", action="store_true", default=True)
    ap.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    ap.add_argument("--project", default="solar-cell-absorption")
    return ap.parse_args(argv)


def parse_levels(spec):
    start, stop, step = (float(x) for x in spec.split(":"))
    grid = np.arange(start, stop + 1e-12, step)
    levels = sorted(set(np.round(grid, 6)) | set(np.round(SIGMA_LEVELS, 6)))
    return [float(p) for p in levels if 0.0 < p < 1.0]


# ==========================================================================
# Prediction reconstruction (mirrors ensemble_uq main; val AND test)
# ==========================================================================
def reconstruct_predictions(args):
    """Bundle + dataset -> per-sample (mu, total_std, y) for val AND test.

    Mirrors models/ensemble_uq.py's load/split/predict block exactly (same
    channel rebuild, same normalisation, same split seed) but keeps the
    middle return of stratified_group_split -- the validation indices --
    instead of discarding it.
    """
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    fmt = bundle.get("format")
    if fmt not in ("photonic-surrogate-bundle-v1",
                   "photonic-surrogate-bundle-v2"):
        raise SystemExit(f"unrecognized bundle format in {args.bundle}: {fmt}")
    hetero = bool(bundle.get("heteroscedastic", False)) or \
        int(bundle.get("arch", {}).get("output_shape", 1)) == 2
    if len(bundle["state_dicts"]) < 2:
        raise SystemExit("bundle has < 2 members; ensemble UQ undefined.")
    arch, norm = bundle["arch"], bundle["norm"]
    y_mean, y_std = float(norm["y_mean"]), float(norm["y_std"])
    img_size = int(bundle["img_size"])
    recipe = infer_bundle_recipe(bundle)
    print(f"[cal] bundle: {args.bundle}")
    print(f"[cal] {len(bundle['state_dicts'])} members, recipe={recipe}")
    if hetero:
        print("[cal] v2 heteroscedastic bundle: sigma = Gaussian-mixture "
              "total std (expect a near-identity Stage-1 map)")

    data = np.load(args.data, allow_pickle=False)
    X = torch.from_numpy(data["X"]).float()
    y = torch.from_numpy(data["y"]).float()
    if X.dim() == 3:
        X = X.unsqueeze(1)
    if int(X.shape[-1]) != img_size:
        raise SystemExit(f"dataset raster {int(X.shape[-1])}px != bundle "
                         f"img_size {img_size}px -- wrong dataset?")
    ds_recipe = ([str(r) for r in data["channel_recipe"]]
                 if "channel_recipe" in data.files
                 else (["raster"] if X.shape[1] == 1
                       else ["raster", "fft_baked_v1"]))
    if ds_recipe != recipe:
        print(f"[cal] rebuilding channels {recipe} from the raster "
              f"(dataset carries {ds_recipe})")
        X = build_input_channels(X[:, :1], recipe)

    split_seed = (args.seed if args.seed is not None
                  else bundle["train_config"].get("seed", 137))
    groups = data["sample_id"] if "sample_id" in data.files else None
    _, val_idx, test_idx = stratified_group_split(
        data["sigma"], groups=groups, seed=split_seed)
    if set(val_idx) & set(test_idx):
        raise SystemExit("val/test index overlap -- split reconstruction "
                         "is broken; do not fit.")
    print(f"[cal] split seed {split_seed}: n_val={len(val_idx)} "
          f"n_test={len(test_idx)} (disjoint)")

    xm = torch.as_tensor(norm["x_mean"],
                         dtype=torch.float32).reshape(1, -1, 1, 1)
    xs = torch.as_tensor(norm["x_std"],
                         dtype=torch.float32).reshape(1, -1, 1, 1)
    X_norm = (X - xm) / xs
    y_norm = (y - y_mean) / y_std

    members = []
    for sd in bundle["state_dicts"]:
        m = PhotonicCNN(**arch)
        m.load_state_dict(sd)
        m.to(device).eval()
        members.append(m)

    out = {}
    for name, idx in (("val", val_idx), ("test", test_idx)):
        loader = DataLoader(PhotonicDataset(X_norm[idx], y_norm[idx]),
                            batch_size=args.batch_size, shuffle=False)
        predict_fn = (ensemble_uq_predictions_hetero if hetero
                      else ensemble_uq_predictions)
        mu_n, epi_n, ale_n = predict_fn(members, loader)
        mu = (mu_n * y_std + y_mean).numpy()
        epi = (epi_n * y_std).numpy()
        ale = (ale_n * y_std).numpy()
        out[name] = {
            "mu": mu,
            "sigma": np.sqrt(epi ** 2 + ale ** 2),   # raw total std
            "y": y[idx].numpy(),
        }

    # sanity gate: the reconstructed TEST split must reproduce the
    # bundle's stored test MAE, else every conclusion below is invalid
    stored = bundle.get("test_metrics", {})
    metrics = regression_metrics(out["test"]["y"], out["test"]["mu"])
    if "mae" in stored:
        d = abs(metrics["mae"] - float(stored["mae"]))
        if d > 1e-3:
            raise SystemExit(
                f"sanity check failed: re-eval MAE {metrics['mae']:.6f} != "
                f"stored {stored['mae']:.6f} (delta {d:.6f}); the split "
                "does not match the training run.")
        print(f"[cal] sanity OK: test MAE {metrics['mae']:.6f} matches "
              f"stored {stored['mae']:.6f}")
    return out, bundle, int(split_seed), metrics


# ==========================================================================
# Stage 1: quadrature-affine variance map, Gaussian NLL on val
# ==========================================================================
def fit_quadrature_affine(resid, sigma):
    """argmin_{a,b>=0} mean NLL of N(0, a^2 + b^2 sigma^2) for resid.

    Deterministic two-round coarse-to-fine grid search (no optimizer
    state, so calibration.json is byte-reproducible).
    """
    r2, s2 = resid.astype(float) ** 2, sigma.astype(float) ** 2

    def nll(a, b):
        v = a * a + b * b * s2
        return float(np.mean(0.5 * np.log(2 * np.pi * v) + r2 / (2 * v)))

    a_grid = np.logspace(-4, -1.5, 200)
    b_grid = np.linspace(0.0, 20.0, 200)
    best = (float("inf"), a_grid[0], b_grid[0])
    for _round in range(3):
        for a in a_grid:
            for b in b_grid:
                v = nll(a, b)
                if v < best[0]:
                    best = (v, float(a), float(b))
        _, a0, b0 = best
        a_grid = np.linspace(max(a0 * 0.5, 1e-6), a0 * 1.5, 60)
        db = max((b_grid[1] - b_grid[0]) * 2, 1e-3)
        b_grid = np.linspace(max(b0 - db, 0.0), b0 + db, 60)
    return best[1], best[2], best[0]


def sigma_cal_of(sigma, a, b):
    return np.sqrt(a * a + (b * np.asarray(sigma, dtype=float)) ** 2)


# ==========================================================================
# Stage 2: split-conformal quantile table on standardized scores
# ==========================================================================
def fit_conformal(resid, sigma_cal, levels):
    """Order-statistic conformal factors q_hat(p) on the val scores."""
    scores = np.sort(np.abs(resid) / sigma_cal)
    n = len(scores)
    q_hat, capped = {}, {}
    for p in levels:
        k = int(np.ceil((n + 1) * p))
        if k > n:
            q_hat[p] = float(scores[-1])   # max score: guarantee capped
            capped[p] = True
        else:
            q_hat[p] = float(scores[k - 1])
            capped[p] = False
    return q_hat, capped, scores


def z_of(p):
    """Gaussian central-coverage p -> z: p = 2*Phi(z)-1, via erfinv."""
    try:
        from scipy.special import erfinv
        return float(np.sqrt(2.0) * erfinv(p))
    except ImportError:
        lo, hi = 0.0, 10.0                  # bisection on the local erf
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if 2.0 * _erf(mid / np.sqrt(2.0)) - 1.0 < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def coverage(resid, half_width):
    return float(np.mean(np.abs(resid) <= half_width))


# ==========================================================================
# Evaluation
# ==========================================================================
def evaluate_split(y, mu, sigma_raw, a, b, q_hat, levels):
    resid = y - mu
    s_cal = sigma_cal_of(sigma_raw, a, b)
    out = {
        "before": {
            "nll": gaussian_nll(y, mu, sigma_raw),
            "crps": crps_gaussian(y, mu, sigma_raw),
            "picp_1sigma": picp(y, mu, sigma_raw, 1.0),
            "picp_2sigma": picp(y, mu, sigma_raw, 2.0),
            "picp_3sigma": picp(y, mu, sigma_raw, 3.0),
            "mpiw_1sigma": mpiw(sigma_raw, 1.0),
            "mean_sigma": float(np.mean(sigma_raw)),
        },
        "after_gaussian": {
            "nll": gaussian_nll(y, mu, s_cal),
            "crps": crps_gaussian(y, mu, s_cal),
            "picp_1sigma": picp(y, mu, s_cal, 1.0),
            "picp_2sigma": picp(y, mu, s_cal, 2.0),
            "picp_3sigma": picp(y, mu, s_cal, 3.0),
            "mpiw_1sigma": mpiw(s_cal, 1.0),
            "mean_sigma": float(np.mean(s_cal)),
        },
    }
    curves = {"levels": levels,
              "before": [coverage(resid, z_of(p) * sigma_raw)
                         for p in levels],
              "after_gaussian": [coverage(resid, z_of(p) * s_cal)
                                 for p in levels]}
    if q_hat is not None:
        curves["after_conformal"] = [
            coverage(resid, q_hat[p] * s_cal) for p in levels]
        out["after_conformal"] = {
            "picp_1sigma": coverage(resid, q_hat[_near(levels, SIGMA_LEVELS[0])] * s_cal),
            "picp_2sigma": coverage(resid, q_hat[_near(levels, SIGMA_LEVELS[1])] * s_cal),
            "picp_3sigma": coverage(resid, q_hat[_near(levels, SIGMA_LEVELS[2])] * s_cal),
            "mean_halfwidth_68": float(np.mean(
                q_hat[_near(levels, SIGMA_LEVELS[0])] * s_cal)),
        }
    return out, curves


def _near(levels, p):
    """The grid level closest to p (grid always contains SIGMA_LEVELS)."""
    return min(levels, key=lambda q: abs(q - p))


# ==========================================================================
# Paper figure
# ==========================================================================
def make_figure(curves, metrics, n_test, conformal, path_base):
    levels = np.asarray(curves["levels"])
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(11.2, 4.6),
        gridspec_kw={"width_ratios": [1.55, 1.0]})

    # ---- Panel A: coverage curve on TEST -------------------------------
    band = 2.0 * np.sqrt(levels * (1.0 - levels) / n_test)
    axA.fill_between(levels, levels - band, levels + band,
                     color="#bbbbbb", alpha=0.35, lw=0, zorder=1,
                     label=f"perfect $\\pm$ sampling noise (n={n_test})")
    axA.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#555555", zorder=2)
    axA.plot(levels, curves["before"], "o-", ms=3.5, lw=1.8,
             color=COLORS["before"], zorder=3,
             label="before (raw ensemble)")
    axA.plot(levels, curves["after_gaussian"], "^--", ms=3.5, lw=1.6,
             color=COLORS["gauss"], zorder=4,
             label="after: Gaussian recalibration")
    if conformal:
        axA.plot(levels, curves["after_conformal"], "D-", ms=3.5, lw=2.2,
                 color=COLORS["conformal"], zorder=5,
                 label="after: + split-conformal")
    for p in SIGMA_LEVELS[:2]:
        i = int(np.argmin(np.abs(levels - p)))
        axA.annotate(f"{curves['before'][i]:.2f}",
                     (levels[i], curves["before"][i]),
                     textcoords="offset points", xytext=(6, -11),
                     fontsize=8.5, color=COLORS["before"])
        which = "after_conformal" if conformal else "after_gaussian"
        axA.annotate(f"{curves[which][i]:.2f}",
                     (levels[i], curves[which][i]),
                     textcoords="offset points", xytext=(-4, 8),
                     fontsize=8.5,
                     color=COLORS["conformal" if conformal else "gauss"])
    axA.set_xlabel("Nominal central coverage", fontsize=11)
    axA.set_ylabel("Empirical coverage (held-out test)", fontsize=11)
    axA.set_xlim(0, 1.0)
    axA.set_ylim(0, 1.05)
    axA.grid(True, alpha=0.25)
    axA.set_axisbelow(True)
    axA.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    axA.set_title("Coverage curve", fontsize=11.5)

    # ---- Panel B: PICP bars at the +-1/2/3 sigma levels ----------------
    x = np.arange(3)
    nominal = list(SIGMA_LEVELS)
    before = [metrics["test_before"][f"picp_{k}sigma"] for k in (1, 2, 3)]
    axB.bar(x - 0.22, nominal, width=0.2, color="#ffffff",
            edgecolor="#555555", lw=1.2, label="nominal", zorder=2)
    axB.bar(x, before, width=0.2, color=COLORS["before"], zorder=2,
            label="before")
    if conformal:
        after = [metrics["test_after_conformal"][f"picp_{k}sigma"]
                 for k in (1, 2, 3)]
        lab = "after (conformal)"
        col = COLORS["conformal"]
    else:
        after = [metrics["test_after_gaussian"][f"picp_{k}sigma"]
                 for k in (1, 2, 3)]
        lab = "after (Gaussian)"
        col = COLORS["gauss"]
    axB.bar(x + 0.22, after, width=0.2, color=col, zorder=2, label=lab)
    for xi, v in zip(x, before):
        axB.annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom",
                     fontsize=8.5, color="#333333",
                     textcoords="offset points", xytext=(0, 1.5))
    for xi, v in zip(x, after):
        axB.annotate(f"{v:.2f}", (xi + 0.22, v), ha="center", va="bottom",
                     fontsize=8.5, color="#333333",
                     textcoords="offset points", xytext=(0, 1.5))
    hw68 = metrics.get("test_after_conformal", metrics[
        "test_after_gaussian"]).get("mean_halfwidth_68",
                                    metrics["test_after_gaussian"]
                                    ["mpiw_1sigma"] / 2)
    hw68_b = metrics["test_before"]["mpiw_1sigma"] / 2
    axB.annotate(
        f"68% half-width: {hw68_b:.4f} $\\rightarrow$ {hw68:.4f} E\n"
        "(the price of honest coverage)",
        xy=(0.5, 0.04), xycoords="axes fraction", ha="center",
        fontsize=8.5, color="#333333",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"})
    axB.set_xticks(x)
    axB.set_xticklabels(["$\\pm 1\\sigma$\n(68.3%)", "$\\pm 2\\sigma$\n"
                         "(95.4%)", "$\\pm 3\\sigma$\n(99.7%)"])
    axB.set_ylabel("Coverage", fontsize=11)
    axB.set_ylim(0, 1.13)
    axB.grid(True, axis="y", alpha=0.25)
    axB.set_axisbelow(True)
    axB.legend(loc="upper left", fontsize=8.5, framealpha=0.95, ncol=1)
    axB.set_title("Coverage at the Gaussian levels", fontsize=11.5)

    fig.suptitle("Uncertainty recalibration: fit on validation, "
                 "evaluated on held-out test", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_base}.{ext}",
                    dpi=300 if ext == "png" else None)
        print(f"[cal] wrote {path_base}.{ext}")
    plt.close(fig)


# ==========================================================================
def main(argv=None):
    args = parse_args(argv)
    if not os.path.exists(args.bundle):
        raise FileNotFoundError(f"bundle not found: {args.bundle}")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"dataset not found: {args.data}")
    if args.out_dir is None:
        args.out_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.bundle)), "uq")
    os.makedirs(args.out_dir, exist_ok=True)
    levels = parse_levels(args.levels)

    preds, bundle, split_seed, test_metrics = reconstruct_predictions(args)
    v, t = preds["val"], preds["test"]
    resid_val = v["y"] - v["mu"]
    val_rmse = float(np.sqrt(np.mean(resid_val ** 2)))
    print(f"[cal] val MAE {np.mean(np.abs(resid_val)):.6f}  "
          f"RMSE {val_rmse:.6f}")

    # ---- Stage 1 fit (VAL ONLY) ----------------------------------------
    a, b, nll_val = fit_quadrature_affine(resid_val, v["sigma"])
    print(f"[cal] quadrature-affine fit: a={a:.6f}  b={b:.4f}  "
          f"(val NLL {gaussian_nll(v['y'], v['mu'], v['sigma']):.3f} -> "
          f"{nll_val:.3f})")
    if not (0.3 * val_rmse <= a <= 2.0 * val_rmse):
        print(f"[cal] WARNING: a={a:.5f} far from val RMSE {val_rmse:.5f} "
              "-- inspect the fit before trusting downstream use.")

    # ---- Stage 2 fit (VAL ONLY) ----------------------------------------
    q_hat = capped = scores = None
    if args.conformal:
        s_cal_val = sigma_cal_of(v["sigma"], a, b)
        q_hat, capped, scores = fit_conformal(resid_val, s_cal_val, levels)
        qs = [q_hat[p] for p in levels]
        assert all(q2 >= q1 - 1e-12 for q1, q2 in zip(qs, qs[1:])), \
            "conformal q_hat not monotone in p"
        # by-construction check: val coverage within order-statistic slack.
        # Compare the SCORES against q_hat directly -- re-multiplying
        # q_hat * sigma_cal can drop the boundary sample by one ulp.
        n_val = len(resid_val)
        sval = np.abs(resid_val) / s_cal_val
        for p in levels:
            cov = float(np.mean(sval <= q_hat[p] * (1 + 1e-12)))
            assert cov >= p - 1e-9 and cov - p <= 2.0 / n_val, \
                f"val conformal coverage {cov:.4f} violates bound at p={p}"
        print(f"[cal] conformal table fit on n_val={n_val} "
              f"(guarantee holds for p <= {n_val / (n_val + 1):.4f}; "
              f"{sum(capped.values())} level(s) capped at max score)")

    # ---- evaluate before/after on VAL (echo) and TEST (report) ---------
    metrics = {}
    for name, d in (("val", v), ("test", t)):
        ev, curves = evaluate_split(d["y"], d["mu"], d["sigma"], a, b,
                                    q_hat, levels)
        for k, m in ev.items():
            metrics[f"{name}_{k}"] = m
        if name == "test":
            test_curves = curves

    print("\n=== test coverage (before -> after) ===")
    for k in (1, 2, 3):
        aft = (metrics.get("test_after_conformal")
               or metrics["test_after_gaussian"])[f"picp_{k}sigma"]
        print(f"  +-{k} sigma: nominal {SIGMA_LEVELS[k-1]:.3f}   "
              f"{metrics['test_before'][f'picp_{k}sigma']:.3f} -> {aft:.3f}")
    print(f"  NLL  {metrics['test_before']['nll']:.3f} -> "
          f"{metrics['test_after_gaussian']['nll']:.3f}")
    print(f"  CRPS {metrics['test_before']['crps']:.5f} -> "
          f"{metrics['test_after_gaussian']['crps']:.5f}")
    print(f"  mean sigma {metrics['test_before']['mean_sigma']:.5f} -> "
          f"{metrics['test_after_gaussian']['mean_sigma']:.5f} "
          "(label-noise floor 0.00233)")

    # ---- persist calibration.json --------------------------------------
    payload = {
        "format": "photonic-uq-calibration-v1",
        "method": ("quadrature_affine_nll"
                   + ("+split_conformal_std_scaled" if args.conformal
                      else "")),
        "bundle": os.path.abspath(args.bundle),
        "bundle_format": bundle.get("format"),
        "data": os.path.abspath(args.data),
        "split_seed": split_seed,
        "fit_split": "val",
        "n_val": int(len(v["y"])),
        "n_test": int(len(t["y"])),
        "a": float(a),
        "b": float(b),
        "conformal": (None if not args.conformal else {
            "score": "abs_residual_over_sigma_cal",
            "levels": [float(p) for p in levels],
            "q_hat": [float(q_hat[p]) for p in levels],
            "capped": [bool(capped[p]) for p in levels],
            "sorted_val_scores": [float(s) for s in scores],
        }),
        "metrics": metrics,
        "test_coverage_curves": {k: [float(x) for x in vv]
                                 for k, vv in test_curves.items()
                                 if k != "levels"},
        "sanity": {"test_mae": float(test_metrics["mae"]),
                   "stored_mae": float(bundle["test_metrics"]["mae"]),
                   "val_rmse": val_rmse},
    }
    json_path = os.path.join(args.out_dir, "calibration.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[cal] wrote {json_path}")

    # ---- figure ---------------------------------------------------------
    make_figure(test_curves, metrics, len(t["y"]), args.conformal,
                os.path.join(args.out_dir, "uq_recalibration"))

    # ---- wandb ----------------------------------------------------------
    if args.use_wandb and wandb is not None:
        run = wandb.init(
            project=args.project,
            name=f"calibrate-{os.path.basename(os.path.dirname(args.bundle))}",
            config={"bundle": args.bundle, "a": a, "b": b,
                    "conformal": args.conformal,
                    "split_seed": split_seed},
            job_type="calibrate_uq")
        flat = {}
        for grp, m in metrics.items():
            for k, x in m.items():
                flat[f"{grp}/{k}"] = x
        wandb.summary.update(flat)
        png = os.path.join(args.out_dir, "uq_recalibration.png")
        wandb.log({"uq_recalibration": wandb.Image(png)})
        for p in (json_path, png):
            wandb.save(p, base_path=args.out_dir)
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
