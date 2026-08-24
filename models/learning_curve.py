"""Learning curve: evaluate the surrogate's loss vs number of training samples.

Trains PhotonicCNN on progressively larger subsets of the dataset using
the same hyperparameters (optionally loaded from a prior W&B sweep via
--best-params) and plots the validation loss + test within-sigma Spearman
rho against training-set size.

Usage:
    # defaults: progression 50 100 200 500 1000 1500 2000 2724
    python -m models.learning_curve

    # use best-known hyperparams from a sweep (point at the sweep's
    # best_params.json -- e.g. runs/surrogate_128_fft_nll_sweep/best_params.json)
    python -m models.learning_curve --best-params runs/surrogate_128_fft_nll_sweep/best_params.json

    # faster: fewer epochs, smaller progression
    python -m models.learning_curve --epochs 60 --progression 100 500 2000

Outputs (in --out-dir):
    learning_curve.png      dual-axis plot
    learning_curve.json     per-size metrics as JSON
    learning_curve.csv      same data as CSV
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from models.model import (device, PhotonicCNN, PhotonicDataset,
                          normalize, resolve_input, stratified_group_split,
                          stratified_group_folds, train_one, predict,
                          predict_gaussian, regression_metrics,
                          within_sigma_spearman)
from models.model import _REPO_ROOT as _MODEL_REPO_ROOT

DEFAULT_PROGRESSION = [50, 100, 200, 500, 1000, 1500, 2000, 2724]

# below this training-set size member disagreement is dominated by training
# noise (see the N=200 spike in seed137_ens), not epistemic uncertainty --
# don't plot it
ENS_STD_MIN_N = 500


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m models.learning_curve",
        description="Learning curve: loss vs number of training samples.",
    )
    ap.add_argument("-i", "--data",
                    default=os.path.join(_REPO_ROOT, "data",
                                         "samples_128.npz"))
    ap.add_argument("-o", "--out-dir",
                    default=os.path.join(_REPO_ROOT, "runs",
                                         "learning_curve"))
    ap.add_argument("--progression", type=int, nargs="+",
                    default=DEFAULT_PROGRESSION,
                    help="Training set sizes to evaluate.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10,
                    help="Validation holdout fraction (default 0.10). "
                         "Shrink to push the largest curve point closer "
                         "to the full dataset; val loss gets noisier.")
    ap.add_argument("--test-frac", type=float, default=0.10,
                    help="Test holdout fraction (default 0.10).")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden-units", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--stochastic-depth", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--smoothl1-beta", type=float, default=1.0)
    ap.add_argument("--nll-head", action="store_true", default=False,
                    help="v2 recipe: heteroscedastic (mu, log var) head "
                         "trained by beta-NLL (see models/model.py). Each "
                         "curve point then also records the model's raw "
                         "self-predicted sigma (total/aleatoric/epistemic) "
                         "and predictive entropy on the test split.")
    ap.add_argument("--beta-nll", type=float, default=0.5,
                    help="beta of beta-NLL (overridden by --best-params).")
    ap.add_argument("--var-warmup", type=int, default=10,
                    help="epochs of mu-only training before the variance "
                         "head switches on (overridden by --best-params).")
    ap.add_argument("--kfold-members", action="store_true", default=False,
                    help="rotate a different held-out val fold per member "
                         "over (subset + val), mirroring the deployed v2 "
                         "bundle's k-fold recipe; needs --ensemble >= 2.")
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--tta", action="store_true", default=True)
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.add_argument("--ensemble", type=int, default=1,
                    help="members trained per curve point (1 = single "
                         "model). >1 also records the ensemble's mean "
                         "self-predicted error (member disagreement) per "
                         "size; multiplies runtime accordingly.")
    ap.add_argument("--xscale", choices=("log", "linear"), default="log",
                    help="x-axis scale for the plot (default log: training "
                         "sizes span ~50x, which linear squashes).")
    ap.add_argument("--best-params", default=None,
                    help="Path to best_params.json from a W&B sweep. "
                         "Overrides hyperparameters (hidden_units, lr, "
                         "batch_size, etc.) with the sweep's best values.")
    return ap.parse_args(argv)


def plot_curve(results, args, loss_floor, out_dir):
    """Render the learning-curve figure from a results dict (N -> metrics).

    Standalone so an existing run's learning_curve.json can be replotted
    without retraining.  Draws the ensemble's self-predicted error
    (mean member disagreement, physical E units) as a purple series on a
    third axis whenever the results carry finite mean_ensemble_std.
    """
    ns = sorted(results.keys())
    val_losses = [results[n]["best_val_loss"] for n in ns]
    spearmans = [results[n]["test_within_sigma_spearman"] for n in ns]
    ens_stds = np.array([results[n].get("mean_ensemble_std", float("nan"))
                         if n >= ENS_STD_MIN_N else float("nan")
                         for n in ns], dtype=float)
    have_ens = bool(np.isfinite(ens_stds).any())

    # wider canvas when the third (offset) axis is present
    nll = bool(getattr(args, "nll_head", False))
    fig, ax1 = plt.subplots(figsize=(8.8 if have_ens else 8, 5))
    ax1.set_xlabel("Number of training samples")
    ax1.set_ylabel("Validation loss (Gaussian NLL)" if nll
                   else "Validation loss (SmoothL1)", color="#d62728")
    ax1.plot(ns, val_losses, "o-", color="#d62728", linewidth=1.8,
             markersize=6, label="val loss")
    ax1.axhline(loss_floor, color="#d62728", linewidth=1.2, linestyle="--",
                alpha=0.6, label="label-noise floor")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(True, alpha=0.3)
    if not nll:
        # pad below the noise-floor line so it separates from the spine
        # (NLL floors are negative -- autoscale handles those)
        ax1.set_ylim(bottom=-2 * loss_floor)

    # x scale: log is the learning-curve convention (sizes span ~50x here,
    # so linear squashes the early points); --xscale linear if preferred.
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    if args.xscale == "log":
        ax1.set_xscale("log", base=10)
        ax1.set_xlim(min(ns) * 0.9, max(ns) * 1.1)
        ticks = [ns[0]]
        for n in ns[1:-1]:
            if np.log10(n / ticks[-1]) >= 0.15:
                ticks.append(n)
        if np.log10(ns[-1] / ticks[-1]) < 0.08:
            ticks.pop()
        ticks.append(ns[-1])
        ax1.xaxis.set_minor_formatter(NullFormatter())
    else:
        span = max(ns) - min(ns)
        ax1.set_xlim(min(ns) - 0.03 * span, max(ns) + 0.03 * span)
        ticks = [ns[0]]
        for n in ns[1:]:
            if (n - ticks[-1]) >= 0.06 * span:
                ticks.append(n)
        if ns[-1] not in ticks:
            if ticks and (ns[-1] - ticks[-1]) < 0.06 * span:
                ticks.pop()
            ticks.append(ns[-1])
    ax1.set_xticks(ticks)
    ax1.xaxis.set_major_formatter(ScalarFormatter())

    ax2 = ax1.twinx()
    ax2.set_ylabel("Test within-sigma Spearman rho", color="#1f77b4")
    ax2.plot(ns, spearmans, "s--", color="#1f77b4", linewidth=1.8,
             markersize=6, label="test Spearman rho")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    ax3 = None
    if have_ens:
        # self-predicted error is in physical E units (not loss units),
        # so it gets its own offset axis rather than sharing ax1
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.13))
        ax3.set_ylabel("Ensemble self-predicted error (E units)",
                       color="tab:purple")
        ax3.plot(ns, ens_stds, "o:", color="tab:purple", linewidth=1.5,
                 markersize=6, label="self-predicted error")
        ax3.tick_params(axis="y", labelcolor="tab:purple")
        ax3.set_ylim(bottom=0)

    fig.suptitle("Learning curve: loss and ranking fidelity vs dataset size",
                 fontsize=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if ax3 is not None:
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines1, labels1 = lines1 + lines3, labels1 + labels3
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center",
               ncol=4 if ax3 is not None else 3, fontsize=8, framealpha=0.9)
    fig.tight_layout()

    fig_path = os.path.join(out_dir, "learning_curve.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] -> {fig_path}")
    return fig_path


def main(argv=None):
    args = parse_args(argv)
    if args.kfold_members and args.ensemble < 2:
        raise SystemExit("--kfold-members needs --ensemble >= 2")
    os.makedirs(args.out_dir, exist_ok=True)
    progression = sorted(args.progression)

    if args.best_params is not None:
        if not os.path.exists(args.best_params):
            raise FileNotFoundError(
                f"--best-params {args.best_params} not found")
        with open(args.best_params) as f:
            bp = json.load(f)
        for k, v in bp.items():
            # skip unknown keys and None-default args (type(None)(v) raises)
            if hasattr(args, k) and getattr(args, k) is not None:
                setattr(args, k, type(getattr(args, k))(v))
        print(f"[args] loaded best params from {args.best_params}",
              flush=True)

    if device.type == "cpu":
        print("[warn] no GPU detected -- at default settings each curve "
              "point trains for a long time with only periodic epoch "
              "output. This is a many-hour CPU job; run on a GPU node.",
              flush=True)

    # ---- data loading (same as model.py) -------------------------------
    data = np.load(args.data, allow_pickle=False)
    X = torch.from_numpy(data["X"]).float()
    y = torch.from_numpy(data["y"]).float()
    if X.dim() == 3:
        X = X.unsqueeze(1)
    elif X.dim() != 4:
        raise ValueError(f"Expected X with 3 or 4 dims, got {tuple(X.shape)}")
    ds_recipe = (data["channel_recipe"] if "channel_recipe" in data.files
                 else None)
    X, recipe = resolve_input(X, ds_recipe, args)
    in_ch = int(X.shape[1])
    n_total = len(y)

    groups = data["sample_id"] if "sample_id" in data.files else None
    train_idx, val_idx, test_idx = stratified_group_split(
        data["sigma"], groups=groups, val_frac=args.val_frac,
        test_frac=args.test_frac, seed=args.seed)
    print(f"[split] train={len(train_idx)} val={len(val_idx)} "
          f"test={len(test_idx)} "
          f"(grouped={'yes' if groups is not None else 'no'})", flush=True)

    # Sizes can only be drawn from the TRAIN split, not the full dataset:
    # clamp and dedupe so oversized requests don't silently train the same
    # subset twice (results are keyed by actual n_train).
    clamped = sorted({min(n, len(train_idx)) for n in progression})
    if clamped != progression:
        print(f"[progression] requested {progression} -> train split has "
              f"{len(train_idx)} samples; using {clamped}", flush=True)
        progression = clamped

    X_norm, y_norm, x_mean, x_std, y_mean, y_std = normalize(X, y, train_idx)

    pin = device.type == "cuda"
    val_loader = DataLoader(
        PhotonicDataset(X_norm[val_idx], y_norm[val_idx]),
        batch_size=args.batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(
        PhotonicDataset(X_norm[test_idx], y_norm[test_idx]),
        batch_size=args.batch_size, shuffle=False, pin_memory=pin)

    sigma_test = np.asarray(data["sigma"])[test_idx]
    cls_test = (np.asarray(data["disorder_class"])[test_idx]
                if "disorder_class" in data.files else None)

    # ---- train for each sample size ------------------------------------
    def _epoch_log(d):
        if d["epoch"] % 10 == 0 or d["epoch"] == args.epochs - 1:
            print(f"    epoch {d['epoch']:3d}  train {d['train_loss']:.5f}  "
                  f"val {d['val_loss']:.5f}", flush=True)

    results = {}  # N -> dict of metrics
    for i, n in enumerate(progression):
        seed_i = args.seed + i * 137
        n_train = min(n, len(train_idx))
        idx_n = train_idx[:n_train]
        print(f"\n{'='*60}")
        print(f"  N={n_train}  (seed={seed_i})")
        print(f"{'='*60}", flush=True)

        train_loader_n = DataLoader(
            PhotonicDataset(X_norm[idx_n], y_norm[idx_n]),
            batch_size=args.batch_size, shuffle=True, pin_memory=pin)

        # per-member loaders: shared (subset, val) by default; with
        # --kfold-members, rotated val folds over (subset + val) -- the
        # deployed v2 bundle's recipe scaled down to N (test untouched,
        # normalisation stats stay from the master train split)
        member_loaders = [(train_loader_n, val_loader)] * args.ensemble
        n_member_train = n_train
        if args.kfold_members:
            pool_n = np.concatenate([idx_n, val_idx])
            folds = stratified_group_folds(
                np.asarray(data["sigma"])[pool_n],
                groups=(np.asarray(groups)[pool_n] if groups is not None
                        else None),
                k=args.ensemble, seed=seed_i)
            member_loaders = []
            for f in folds:
                vi = pool_n[f]
                ti = np.setdiff1d(pool_n, vi)
                member_loaders.append((
                    DataLoader(PhotonicDataset(X_norm[ti], y_norm[ti]),
                               batch_size=args.batch_size, shuffle=True,
                               pin_memory=pin),
                    DataLoader(PhotonicDataset(X_norm[vi], y_norm[vi]),
                               batch_size=args.batch_size, shuffle=False,
                               pin_memory=pin)))
            n_member_train = int(len(pool_n) - len(folds[0]))
            print(f"  [kfold] rotated val folds over subset+val "
                  f"(pool={len(pool_n)}); ~{n_member_train} train "
                  f"samples per member", flush=True)

        # train an ensemble per size (--ensemble 1 = the old single model);
        # member m gets seed seed_i + 1000*m, mirroring model.py's scheme
        member_preds, member_sigmas, best_val_loss = [], [], float("inf")
        targets = None
        for mem in range(args.ensemble):
            if args.ensemble > 1:
                print(f"  -- member {mem + 1}/{args.ensemble}", flush=True)
            m, vloss = train_one(
                args, member_loaders[mem], seed_i + 1000 * mem,
                in_ch=in_ch, log=_epoch_log)
            best_val_loss = min(best_val_loss, float(vloss))
            if args.nll_head:
                preds_n, sig_n, targets_n = predict_gaussian(
                    m, test_loader, use_tta=args.tta)
                member_sigmas.append((sig_n * y_std).numpy())
            else:
                preds_n, targets_n = predict(m, test_loader,
                                             use_tta=args.tta)
            member_preds.append((preds_n * y_std + y_mean).numpy())
            targets = (targets_n * y_std + y_mean).numpy()

        stack = np.stack(member_preds)          # (members, n_test)
        preds = stack.mean(axis=0)
        # the ensemble's self-predicted error: mean member disagreement
        # (epistemic std, physical E units); NaN for a single model
        ens_std = (float(stack.std(axis=0, ddof=1).mean())
                   if args.ensemble > 1 else float("nan"))

        # v2 UQ summary (E units): per test point the ensemble is a
        # Gaussian mixture, so var_total = mean member variance
        # (aleatoric) + population variance of member means (epistemic);
        # predictive entropy = 0.5*ln(2*pi*e*var_total)
        sig_total = sig_alea = sig_epi = entropy = float("nan")
        rms_total = rms_alea = rms_epi = float("nan")
        if args.nll_head:
            var_alea = (np.stack(member_sigmas) ** 2).mean(axis=0)
            var_epi = stack.var(axis=0)
            var_total = var_alea + var_epi
            sig_total = float(np.sqrt(var_total).mean())
            sig_alea = float(np.sqrt(var_alea).mean())
            sig_epi = float(np.sqrt(var_epi).mean())
            # RMS aggregates are the ones that compare like-for-like with
            # the test RMSE (variance matching, no distributional
            # assumption); mean-of-sigma sits below them by Jensen
            rms_total = float(np.sqrt(var_total.mean()))
            rms_alea = float(np.sqrt(var_alea.mean()))
            rms_epi = float(np.sqrt(var_epi.mean()))
            entropy = float(np.mean(
                0.5 * np.log(2.0 * np.pi * np.e * var_total)))

        rmetrics = regression_metrics(targets, preds)
        rho_pooled, rho_cells = within_sigma_spearman(
            targets, preds, sigma_test, cls_test)

        entry = {
            "n_train": int(n_train),
            "n_member_train": int(n_member_train),
            "n_test": int(len(test_idx)),
            "best_val_loss": float(best_val_loss),
            "test_mae": rmetrics["mae"],
            "test_rmse": rmetrics["rmse"],
            "test_r2": rmetrics["r2"],
            "test_pct_error": rmetrics["pct_error"],
            "test_within_sigma_spearman": float(rho_pooled),
            "n_members": int(args.ensemble),
            "mean_ensemble_std": ens_std,
            "mean_pred_sigma_total": sig_total,
            "mean_aleatoric_std": sig_alea,
            "mean_epistemic_std": sig_epi,
            "mean_entropy": entropy,
            "rms_pred_sigma_total": rms_total,
            "rms_aleatoric_std": rms_alea,
            "rms_epistemic_std": rms_epi,
        }
        results[n_train] = entry
        print(f"  val_loss={best_val_loss:.6f}  test_mae={rmetrics['mae']:.6f}"
              f"  test_spearman={rho_pooled:.4f}  test_r2={rmetrics['r2']:.4f}"
              + (f"  ens_std={ens_std:.6f}" if args.ensemble > 1 else "")
              + (f"  sig_total={sig_total:.6f}" if args.nll_head else ""),
              flush=True)

    # ---- plot ----------------------------------------------------------
    # Loss floor for a PERFECT model given per-label engine noise
    # (audit Test 9: sigma_n ~ 0.09% of E).  Labels are z-normalized by
    # y_std, and torch's SmoothL1 is r^2 / (2*beta) inside the elbow --
    # residuals here are ~0.26 vs beta ~4.3, so the quadratic branch is the
    # one that applies.  Dropping the 1/beta factor inflates the floor.
    _resid_norm = (0.0009 * float(y_mean)) / float(y_std)
    if args.nll_head:
        # perfect-model NLL floor: residual at the label-noise limit with
        # sigma matching it exactly -> 0.5*ln(2*pi*e*sigma_n^2) (negative
        # here since sigma_n << 1 in normalised units)
        loss_floor = float(0.5 * np.log(2.0 * np.pi * np.e
                                        * _resid_norm ** 2))
    else:
        loss_floor = _resid_norm ** 2 / (2.0 * float(args.smoothl1_beta))

    ns = sorted(results.keys())
    plot_curve(results, args, loss_floor, args.out_dir)

    # ---- export ---------------------------------------------------------
    json_path = os.path.join(args.out_dir, "learning_curve.json")
    with open(json_path, "w") as f:
        json.dump({"progression": ns, "results": results,
                    "config": vars(args)}, f, indent=2)
    print(f"[json] -> {json_path}")

    csv_path = os.path.join(args.out_dir, "learning_curve.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[ns[0]].keys()))
        w.writeheader()
        for n in ns:
            w.writerow(results[n])
    print(f"[csv]  -> {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
