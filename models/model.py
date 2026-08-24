"""CNN surrogate training for absorption enhancement E from photonic-crystal layouts.

v4 -- upgrades over the previous minimal script, in line with the audit
(AUDIT_resolution_accuracy.md) and the inverse-design stage that consumes
this model (models/inverse_design.py):

  1. GROUP-AWARE STRATIFIED SPLIT.  If the dataset carries `sample_id`
     (the augmented .npz does: 8 D4 views per physical sample), all views
     of one sample now land in the same split.  The old index-level split
     leaked rotated/flipped copies of test-set layouts into training,
     inflating test metrics.  Falls back to plain sigma-stratified split
     when sample_id is absent (e.g. non-augmented pilot data).
  2. RESIDUAL + SQUEEZE-EXCITATION BACKBONE with stochastic depth --
     replaces the plain 4-block CNN; same `--hidden-units` knob.
  3. EMA WEIGHTS.  Validation/test use an exponential moving average of
     the weights, which is consistently a free accuracy win for small
     regression CNNs and stabilises early stopping.
  4. AMP + GRADIENT CLIPPING on GPU (carried over from the previous
     version of this script, formerly named base_model_v3.py).
  5. D4 TEST-TIME AUGMENTATION at eval: E is invariant under the 8
     dihedral ops of the square supercell, so averaging the 8 predictions
     is variance reduction with zero bias.
  6. ENSEMBLE TRAINING (`--ensemble N`) exporting ONE self-contained
     `surrogate_bundle.pt` (state dicts + x/y normalisation + config).
     This bundle is the exact artifact inverse_design.py loads, so the
     robust ensemble+TTA fitness there needs no retraining glue.
  7. WITHIN-SIGMA SPEARMAN on the test split -- the decisive metric per
     audit Tests 6/9 (within-sigma ranking fidelity): reported per
     (class, sigma) cell and pooled, alongside MAE/RMSE/R^2.
  8. `--no-wandb` for offline SCC runs (wandb mode="disabled").
  9. `--fft-channel`: appends the log-magnitude 2D FFT (the structure
     factor, up to phase) as a second input channel, computed on the fly
     from the 1-channel raster -- the dataset file never changes.
     Bundle records input_shape + per-channel norm; inverse_design.py
     detects and recomputes the channel automatically (differentiably).
  10. W&B sweep support (--sweep/--sweep-count/--use-best), merged from
     the previous version of this script.

CLI is backward compatible with the previous script; new flags only add.
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")  # headless-safe: no display needed to render/log figures
import matplotlib.pyplot as plt  # noqa: E402

try:
    import wandb
except ImportError:  # allow fully offline use
    wandb = None


device = torch.device(
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available() else "cpu")
print(f"[model] device = {device}")

# Anchor path defaults to the repo root so they work from any CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m models.model",
        description="Train the PhotonicCNN surrogate on photonic-crystal layouts.",
    )
    ap.add_argument("-i", "--data",
                    default=os.path.join(_REPO_ROOT, "data",
                                         "samples_128.npz"))
    ap.add_argument("-o", "--out-dir",
                    default=os.path.join(_REPO_ROOT, "runs",
                                         "surrogate_new"),
                    help="Directory for checkpoints + the surrogate bundle. "
                         "Deliberately NOT the deployed dir "
                         "(runs/surrogate_128_fft_nll_sweep): "
                         "surrogate_bundle.pt, test_metrics.json and the "
                         "figures are overwritten in place -- never reuse "
                         "an out-dir for a new run (audit sec. 11.3).")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden-units", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--stochastic-depth", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=20,
                    help="Early-stop after this many epochs w/o val improvement.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoothl1-beta", type=float, default=1.0)
    ap.add_argument("--nll-head", action="store_true", default=False,
                    help="Heteroscedastic head: the network outputs "
                         "(mu, log sigma^2) and trains with beta-NLL "
                         "instead of SmoothL1. Produces a per-point "
                         "aleatoric sigma the plain ensemble lacks. "
                         "Bundle is written as format v2.")
    ap.add_argument("--beta-nll", type=float, default=0.5,
                    help="beta-NLL weighting (Seitzer et al.): 0 = plain "
                         "Gaussian NLL (unstable), 1 = MSE-like gradient "
                         "weighting. 0.5 is the standard compromise.")
    ap.add_argument("--var-warmup", type=int, default=10,
                    help="epochs of mu-only (SmoothL1) training before "
                         "the variance head switches on, so sigma "
                         "calibrates against a sensible mean. Best-model "
                         "tracking starts after warm-up.")
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="EMA decay for eval weights (0 disables EMA).")
    ap.add_argument("--kfold-members", action="store_true", default=False,
                    help="rotate a different validation fold (k = ensemble "
                         "size) out of the train+val pool for each member, "
                         "so members train on different data subsets "
                         "instead of identical data + different init. "
                         "Decorrelates the ensemble (better epistemic "
                         "sigma). Test split untouched.")
    ap.add_argument("--ensemble", type=int, default=1,
                    help="Train N independently-seeded models; export all in "
                         "one surrogate bundle and report ensemble metrics.")
    ap.add_argument("--augment", action="store_true", default=True,
                    help="D4 dihedral augmentation on training batches.")
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--tta", action="store_true", default=True,
                    help="D4 test-time augmentation for val/test predictions.")
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.add_argument("--circular-padding", action="store_true", default=False,
                    help="Circular (toroidal) conv padding, matching the "
                         "supercell's periodic boundaries (ablation #16).")
    ap.add_argument("--shift-aug", action="store_true", default=False,
                    help="Cyclic-shift augmentation on training batches "
                         "(rolls the raster channel only; ablation #16).")
    ap.add_argument("--wandb", dest="use_wandb", action="store_true",
                    default=True)
    ap.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    ap.add_argument("--raster-only", action="store_true", default=False,
                    help="Use only channel 0 (the occupancy raster) of a "
                         "multi-channel dataset -- for attribution "
                         "ablations (e.g. quantifying what the baked FFT "
                         "channel contributes).")
    ap.add_argument("--fft-only", action="store_true", default=False,
                    help="Ablation #23: train on the structure factor "
                         "ALONE (fft_onfly_v1 derived from the raster; "
                         "the raster channel itself is dropped). "
                         "Mutually exclusive with --fft-channel.")
    ap.add_argument("--fft-channel", action="store_true", default=False,
                    help="Append log-magnitude 2D FFT of the layout as a "
                         "second input channel (the structure factor, up "
                         "to phase). Physically: absorption enhancement "
                         "is governed by how the hole pattern scatters "
                         "into guided modes, which lives in reciprocal "
                         "space; feeding |FFT| turns a hard global-"
                         "interference inference into local pattern "
                         "recognition.")
    ap.add_argument("--project", default="solar-cell-absorption")
    ap.add_argument("--eval-bundle", default=None,
                    help="Skip training: load this surrogate_bundle.pt and "
                         "run the full test diagnostics (sigma-colored "
                         "plots, per-cell Spearman, CSV/JSON, wandb) on "
                         "the dataset's test split. The split is rebuilt "
                         "with the bundle's training seed, so it matches "
                         "the original run exactly.")
    ap.add_argument("--sweep", action="store_true", default=False,
                    help="Run a W&B sweep instead of a single training run.")
    ap.add_argument("--sweep-count", type=int, default=30,
                    help="Number of sweep trials to run.")
    ap.add_argument("--use-best", action="store_true", default=False,
                    help="Load best hyperparameters from best_params.json.")
    ap.add_argument("--combo-lambda", type=float, default=0.1,
                    help="Loss penalty weight in the combined sweep score: "
                         "score = rho - lambda * (loss / 0.01). "
                         "Higher lambda penalizes high loss more. "
                         "0 = pure rho (default in earlier runs).")
    ap.add_argument("--attention", default="se",
                    choices=sorted(ATTENTION_REGISTRY),
                    help="Per-block attention module: "
                         "se = SqueezeExcite (deployed v2 default), "
                         "none = no attention (ablation #18), "
                         "cbam = channel+spatial (Woo et al. 2018, #19), "
                         "eca = 1D-conv channel attn (Wang et al. 2020, #20), "
                         "sa = multi-head spatial self-attn at every block "
                         "(#21; cost grows with feature map size), "
                         "sa4 = SE at stages 1-3 + self-attn at stage 4 "
                         "only (#24; the unconfounded, cheap SA test).")
    ap.add_argument("--recon-head", action="store_true", default=False,
                    help="Multi-task ablation #22: add a decoder that "
                         "reconstructs the (normalized) raster channel "
                         "from the pre-GAP feature map; train loss gains "
                         "recon_lambda * MSE(recon, channel 0). Inference "
                         "paths are unaffected (forward() is unchanged).")
    ap.add_argument("--recon-lambda", type=float, default=0.1,
                    help="Weight of the reconstruction MSE term "
                         "(only with --recon-head).")
    ap.add_argument("--member-seed-offset", type=int, default=0,
                    help="Ablation #25: shift member training seeds "
                         "(init + batch order) WITHOUT changing the "
                         "split, to measure run-to-run training noise "
                         "on an identical test set.")
    return ap.parse_args(argv)


sweep_config = {
    "method": "bayes",
    "metric": {"name": "combo_score", "goal": "maximize"},
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 5,
        "eta": 3,
    },
    "parameters": {
        "hidden_units": {"values": [32, 48, 64, 96, 128]},
        "dropout": {"min": 0.05, "max": 0.5},
        "lr": {"min": 1e-5, "max": 3e-3, "distribution": "log_uniform_values"},
        "weight_decay": {"min": 1e-6, "max": 1e-2, "distribution": "log_uniform_values"},
        "batch_size": {"values": [32, 64, 128, 256]},
        "smoothl1_beta": {"min": 0.05, "max": 5.0, "distribution": "log_uniform_values"},
        "warmup_epochs": {"values": [0, 3, 5, 8]},
        "stochastic_depth": {"min": 0.0, "max": 0.3},
        "ema_decay": {"values": [0.0, 0.999, 0.998]},
    },
}

# --nll-head --sweep: only the loss-adjacent knobs float; architecture,
# schedule and regularisation stay at whatever the CLI passes in (i.e.
# the deployed Huber sweep's winners).  combo_score in this mode is
# rho + 0.5 * error~sigma rho -- ranking is the constraint, per-point
# discrimination is the goal.
nll_sweep_config = {
    "method": "bayes",
    "metric": {"name": "combo_score", "goal": "maximize"},
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 5,
        "eta": 3,
    },
    "parameters": {
        "lr": {"min": 1e-5, "max": 3e-3,
               "distribution": "log_uniform_values"},
        "weight_decay": {"min": 1e-6, "max": 1e-2,
                         "distribution": "log_uniform_values"},
        "beta_nll": {"values": [0.0, 0.25, 0.5, 1.0]},
        "var_warmup": {"values": [5, 10, 20]},
    },
}


def sweep_agent(base_args):
    """Run one sweep trial. Hyperparameters come from wandb.config."""
    run = wandb.init(project=base_args.project, group="sweep", job_type="train")
    cfg = wandb.config

    args = copy.copy(base_args)
    if getattr(base_args, "nll_head", False):
        # NLL sweep: only the loss-adjacent knobs come from the sweep;
        # everything else keeps the CLI values (= the Huber sweep's best)
        args.lr = cfg.lr
        args.weight_decay = cfg.weight_decay
        args.beta_nll = cfg.beta_nll
        args.var_warmup = cfg.var_warmup
    else:
        args.hidden_units = cfg.hidden_units
        args.dropout = cfg.dropout
        args.lr = cfg.lr
        args.weight_decay = cfg.weight_decay
        args.batch_size = cfg.batch_size
        args.smoothl1_beta = cfg.smoothl1_beta
        args.warmup_epochs = cfg.warmup_epochs
        args.stochastic_depth = cfg.stochastic_depth
        args.ema_decay = cfg.ema_decay
    args.epochs = 120
    args.patience = 20
    args.augment = True
    args.tta = True
    args.ensemble = 1
    args.grad_clip = 1.0

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    data = np.load(args.data, allow_pickle=False)
    X = torch.from_numpy(data["X"]).float()
    y = torch.from_numpy(data["y"]).float()
    if X.dim() == 3:
        X = X.unsqueeze(1)
    ds_recipe = (data["channel_recipe"] if "channel_recipe" in data.files
                 else None)
    X, recipe = resolve_input(X, ds_recipe, args)
    in_ch = int(X.shape[1])

    groups = data["sample_id"] if "sample_id" in data.files else None
    train_idx, val_idx, test_idx = stratified_group_split(
        data["sigma"], groups=groups, seed=args.seed)

    X_norm, y_norm, x_mean, x_std, y_mean, y_std = normalize(X, y, train_idx)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        PhotonicDataset(X_norm[train_idx], y_norm[train_idx]),
        batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(
        PhotonicDataset(X_norm[val_idx], y_norm[val_idx]),
        batch_size=args.batch_size, shuffle=False, pin_memory=pin)

    def log_fn(d):
        wandb.log(d)

    m, best_val = train_one(args, (train_loader, val_loader), args.seed,
                            in_ch=in_ch, log=log_fn)

    # evaluate Spearman rho on validation set for sweep optimisation
    preds, targets, tta_std = predict_with_tta_stats(m, val_loader)
    preds_np = (preds * y_std + y_mean).numpy()
    targets_np = (targets * y_std + y_mean).numpy()
    tta_std_np = (tta_std * y_std).numpy()   # de-normalised std
    sigma_val = np.asarray(data["sigma"])[val_idx]
    cls_val = (np.asarray(data["disorder_class"])[val_idx]
               if "disorder_class" in data.files else None)
    rho_pooled, rho_cells = within_sigma_spearman(targets_np, preds_np,
                                                   sigma_val, cls_val)
    # relative uncertainty (coefficient of variation)
    rel_uncert = float(tta_std_np.mean() / max(abs(preds_np.mean()), 1e-8))

    # error-uncertainty correlation: high std should = high error
    abs_errors = np.abs(preds_np - targets_np)
    if getattr(args, "nll_head", False):
        # discrimination via the LEARNED sigma, and a combo that rewards
        # it: ranking is the constraint, discrimination is the goal.
        # (best_val is an NLL here -- can be negative -- so the Huber
        # combo's loss term does not transfer.)
        _, sig_n, _ = predict_gaussian(m, val_loader, use_tta=True)
        err_uncert_rho = spearman_rho(abs_errors, (sig_n * y_std).numpy())
        combo = rho_pooled + 0.5 * err_uncert_rho
    else:
        err_uncert_rho = spearman_rho(abs_errors, tta_std_np)
        # combined score: reward rank accuracy, penalize high loss
        combo = rho_pooled - args.combo_lambda * (best_val / 0.01)

    wandb.summary["within_sigma_spearman"] = rho_pooled
    wandb.summary["best_val_loss"] = best_val
    wandb.summary["combo_score"] = combo
    wandb.summary["tta_std_mean"] = float(tta_std_np.mean())
    wandb.summary["tta_std_median"] = float(np.median(tta_std_np))
    wandb.summary["relative_uncertainty"] = rel_uncert
    wandb.summary["error_uncertainty_rho"] = err_uncert_rho
    wandb.log({"within_sigma_spearman": rho_pooled, "best_val_loss": best_val,
               "combo_score": combo, "tta_std_mean": float(tta_std_np.mean()),
               "relative_uncertainty": rel_uncert,
               "error_uncertainty_rho": err_uncert_rho})
    for cell, d in rho_cells.items():
        safe = cell.replace("/", "_").replace("=", "_")
        wandb.summary[f"rho/{safe}"] = d["rho"]
    print(f"  -> val Spearman rho = {rho_pooled:.4f}, "
          f"val_loss = {best_val:.6f}, combo = {combo:.4f}, "
          f"TTA std = {tta_std_np.mean():.6f}, "
          f"rel_uncert = {rel_uncert:.4f}, "
          f"err~uncert rho = {err_uncert_rho:+.3f}")

    best_path = os.path.join(args.out_dir, "best_params.json")
    prev_best = -float("inf")
    if os.path.exists(best_path):
        with open(best_path) as f:
            prev_best = json.load(f).get("combo_score", -float("inf"))
    if combo > prev_best:
        active_cfg = (nll_sweep_config if getattr(args, "nll_head", False)
                      else sweep_config)
        params = {k: v for k, v in cfg.items()
                  if k in active_cfg["parameters"]}
        params["combo_score"] = combo
        params["within_sigma_spearman"] = rho_pooled
        params["best_val_loss"] = best_val
        with open(best_path, "w") as f:
            json.dump(params, f, indent=2)
        print(f"  -> new best: combo={combo:.4f} saved to {best_path}")

    wandb.finish()
    return combo


# ==========================================================================
# Architecture
# ==========================================================================
class SqueezeExcite(nn.Module):
    """Channel attention: global pool -> bottleneck MLP -> sigmoid gate.

    The deployed v2 default attention (`--attention se`).  Mirrors the
    SE block originally introduced in SENet (Hu et al. 2018)."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, hidden), nn.GELU(),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)[:, :, None, None]


class NoAttention(nn.Module):
    """Identity placeholder for the `--attention none` ablation: replaces
    SE with a no-op so ResidualBlock's equational form is unchanged.
    Lets #18 measure what the SE gate's gated-rescaling contributes to
    accuracy and calibration, with every other knob pinned."""
    def forward(self, x):
        return x


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al. 2018):
    channel gate followed by spatial gate.  The channel gate is SE's
    twin (avg+max pool -> shared bottleneck MLP -> sigmoid); the spatial
    gate is a 7x7 conv over [avgpool, maxpool] along channel axis ->
    sigmoid.  Tests whether adding a WHERE to SE's WHAT shifts the
    surrogate.  `reduction` matches SE's default so the channel side is
    capacity-matched."""
    def __init__(self, channels, reduction=8, spatial_kernel=7):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(),
            nn.Linear(hidden, channels))
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, spatial_kernel, padding=spatial_kernel // 2,
                      bias=False),
            nn.Sigmoid())

    def forward(self, x):
        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        ca = torch.sigmoid(self.channel_mlp(avg) + self.channel_mlp(mx))
        x = x * ca[:, :, None, None]
        avg_sp = x.mean(dim=1, keepdim=True)
        mx_sp = x.amax(dim=1, keepdim=True)
        sa = self.spatial(torch.cat([avg_sp, mx_sp], dim=1))
        return x * sa


class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al. 2020): a 1D conv over
    the channel descriptor, no MLP bottleneck, no reduction.  The kernel
    size follows the paper's heuristic psi(C) with gamma=2, b=1:
    t = (log2(C) + 1) / 2, k = t rounded up to the nearest odd
    (k=5 at C=128, k=3 at C=32).  Tests whether the bottleneck/reduction
    in SE matters, or whether a shuffling-only conv is enough (or
    better) for this dataset."""
    def __init__(self, channels, reduction=None):
        super().__init__()
        c = max(channels, 2)
        t = int((np.log2(c) + 1) / 2)
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2,
                              bias=False)

    def forward(self, x):
        y = self.avg(x).squeeze(-1).transpose(1, 2)   # (B, 1, C)
        y = torch.sigmoid(self.conv(y)).transpose(1, 2).unsqueeze(-1)
        return x * y


class SpatialSelfAttention(nn.Module):
    """Multi-head spatial self-attention (Wang et al. 2018 style,
    residual + LayerNorm).  With `--attention sa` this is applied at
    EVERY residual block; pooling happens after each stage, so stage 1
    attends over the full 128x128 = 16,384 tokens (then 4,096 / 1,024 /
    256).  Stage 1 dominates the cost, and the score matrix only fits
    in memory via the fused SDPA kernel -- expect to lower the batch
    size if training OOMs.  Note this module REPLACES the block's
    output with a LayerNorm'd attention residual rather than gating it
    multiplicatively like se/cbam/eca, so arm #21 changes normalization
    as well as attention (a deliberate, disclosed confound).  Tests
    whether long-range token mixing changes anything versus a local SE
    gate."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads,
                                          batch_first=True)
        self.out_norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)             # (B, HW, C)
        a, _ = self.attn(self.norm(seq), self.norm(seq),
                         self.norm(seq), need_weights=False)
        seq = self.out_norm(seq + a)
        return seq.transpose(1, 2).reshape(B, C, H, W)


# Registry consumed by ResidualBlock + PhotonicCNN.  All variants,
# including `sa`, are applied at every stage; `sa` at stage 1 (16,384
# tokens) is expensive by design -- see SpatialSelfAttention's docstring.
def build_attention(name, channels, img_size=None):
    """name -> attention module.  `img_size` is reserved for variants
    that need the spatial dim (currently none besides `sa`, which gets
    it from the feature tensor at forward time)."""
    name = (name or "se").lower()
    if name == "se":
        return SqueezeExcite(channels)
    if name == "none":
        return NoAttention()
    if name == "cbam":
        return CBAM(channels)
    if name == "eca":
        return ECA(channels)
    if name == "sa":
        return SpatialSelfAttention(channels)
    raise ValueError(f"unknown attention '{name}': expected one of "
                     f"{sorted(ATTENTION_REGISTRY)}")


# "sa4" is stage-wise (SE at stages 1-3, sa at stage 4) and is resolved
# by PhotonicCNN.__init__, never passed to build_attention directly.
ATTENTION_REGISTRY = {"se", "none", "cbam", "eca", "sa", "sa4"}


class ResidualBlock(nn.Module):
    """Conv-GN-GELU x2 + attention, residual, optional stochastic depth.

    The block's attention is selected by the `attention` string
    (`se`, `none`, `cbam`, `eca`, `sa`); default `se` reproduces the
    deployed v2 model byte-for-byte.  See build_attention()."""

    def __init__(self, in_c, out_c, dilation=1, drop_path=0.0,
                 padding_mode="zeros", attention="se"):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=dilation, dilation=dilation,
                      bias=False, padding_mode=padding_mode),
            nn.GroupNorm(8, out_c), nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, padding=dilation, dilation=dilation,
                      bias=False, padding_mode=padding_mode),
            nn.GroupNorm(8, out_c),
            build_attention(attention, out_c),
        )
        self.skip = (nn.Identity() if in_c == out_c
                     else nn.Conv2d(in_c, out_c, 1, bias=False))
        self.act = nn.GELU()
        self.drop_path = drop_path

    def forward(self, x):
        y = self.body(x)
        if self.training and self.drop_path > 0.0:
            keep = 1.0 - self.drop_path
            mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep
            y = y * mask / keep
        return self.act(self.skip(x) + y)


class PhotonicCNN(nn.Module):
    """Residual-attention CNN -> GAP -> MLP head -> scalar E.

    4 stages at constant width `hidden_units` with MaxPool downsampling
    between stages; the final stage uses dilation-2 convs for a wider
    receptive field at 8x-downsampled resolution (matches the previous
    model's design intent; supercell-scale correlations matter for E).

    `attention` selects the per-block attention module ({se, none, cbam,
    eca, sa}; default `se` = deployed v2).  `img_size` is the input
    raster size and is recorded in the bundle for parity with future
    variants that size themselves off the stage-4 spatial dim.

    `recon_head=True` (ablation #22, `--recon-head`) additionally builds
    a light decoder from the pre-GAP stage-4 feature map back to a
    1-channel raster at input resolution (3x bilinear-upsample+conv,
    width h//2), exposed via forward_with_recon().  forward() is
    UNCHANGED either way: every inference consumer (predict paths,
    SurrogateScorer, saliency) sees the (B, output_shape) head only.
    The reconstruction task forces the trunk to keep layout geometry
    recoverable instead of collapsing to a class-level summary
    (multi-task regularization for the small dataset).
    """

    def __init__(self, input_shape, hidden_units, output_shape,
                 dropout=0.15, stochastic_depth=0.1, padding_mode="zeros",
                 attention="se", img_size=None, recon_head=False):
        super().__init__()
        if hidden_units % 8 != 0:
            raise ValueError(
                f"hidden_units={hidden_units} must be divisible by 8")
        h = hidden_units
        dp = stochastic_depth
        pm = padding_mode
        # "sa4" (#24) is stage-wise: SE where the maps are large,
        # self-attention only at stage 4 (256 tokens) -- the fair SA
        # test, free of #21's stage-1 cost and normalization confound
        # at stages 1-3.
        attn = "se" if attention == "sa4" else attention
        attn4 = "sa" if attention == "sa4" else attention
        self.stem = nn.Sequential(
            nn.Conv2d(input_shape, h, 3, padding=1, bias=False,
                      padding_mode=pm),
            nn.GroupNorm(8, h), nn.GELU(),
        )
        self.stage1 = ResidualBlock(h, h, drop_path=0.00,
                                    padding_mode=pm, attention=attn)
        self.stage2 = ResidualBlock(h, h, drop_path=dp / 3,
                                    padding_mode=pm, attention=attn)
        self.stage3 = ResidualBlock(h, h, drop_path=2 * dp / 3,
                                    padding_mode=pm, attention=attn)
        self.stage4 = ResidualBlock(h, h, dilation=2, drop_path=dp,
                                    padding_mode=pm, attention=attn4)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(h, h), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, output_shape),
        )
        self.decoder = None
        if recon_head:
            # 16x16 stage-4 map -> 128x128 raster: 3x (upsample + conv).
            # h//2 stays divisible by 8 for every sweepable width.
            hd = h // 2

            def up(c_in, c_out):
                return [nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=False),
                        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False,
                                  padding_mode=pm),
                        nn.GroupNorm(8, c_out), nn.GELU()]

            self.decoder = nn.Sequential(
                *up(h, hd), *up(hd, hd), *up(hd, hd),
                nn.Conv2d(hd, 1, 3, padding=1, padding_mode=pm))

    def _features(self, x):
        x = self.stem(x)
        x = self.pool(self.stage1(x))
        x = self.pool(self.stage2(x))
        x = self.pool(self.stage3(x))
        return self.stage4(x)

    def forward(self, x):
        return self.head(self.gap(self._features(x)))

    def forward_with_recon(self, x):
        """(head output, reconstructed channel-0 raster).  Training-only
        path for the multi-task objective; requires recon_head=True."""
        f = self._features(x)
        return self.head(self.gap(f)), self.decoder(f)


# ==========================================================================
# Data
# ==========================================================================
def fft_onfly_v1(raster):
    """On-the-fly structure-factor feature (convention "fft_onfly_v1").

    raster: (B, 1, H, W) Si occupancy in [0, 1], torch tensor.
    Returns the FEATURE ONLY, (B, 1, H, W): log1p |FFT| of the
    mean-removed raster, fftshift-centered.  Differentiable.
    Distinct from the dataset-baked convention below -- the two are NOT
    interchangeable; a model must be scored with the convention it was
    trained on (hence the channel_recipe bookkeeping).
    """
    x = raster - raster.mean(dim=(-2, -1), keepdim=True)
    F = torch.fft.fftshift(torch.abs(torch.fft.fft2(x)), dim=(-2, -1))
    return torch.log1p(F)


def fft_baked_v1(raster):
    """Torch mirror of build_dataset.compute_fft_channel ("fft_baked_v1").

    raster: (B, 1, H, W) in [0, 1].  Returns the FEATURE ONLY:
    log1p(|FFT|^2), per-sample min-max normalized to [0, 1], UNSHIFTED,
    no mean removal -- byte-level convention frozen to match the channel
    baked into samples_128.npz (float32-vs-float64 differences are the
    only deviation, ~1e-7).  Differentiable a.e. (min/max subgradients),
    so the Tier-3 inverse-design path works.
    """
    F = torch.fft.fft2(raster.double())
    log_power = torch.log1p(F.abs() ** 2)
    mins = log_power.amin(dim=(-2, -1), keepdim=True)
    maxs = log_power.amax(dim=(-2, -1), keepdim=True)
    return ((log_power - mins) / (maxs - mins).clamp(min=1e-12)).float()


CHANNEL_BUILDERS = {"fft_onfly_v1": fft_onfly_v1,
                    "fft_baked_v1": fft_baked_v1}


def build_input_channels(raster, recipe):
    """(B,1,H,W) raster + recipe -> (B,C,H,W) model input.

    recipe: "raster" (allowed at position 0 only) and/or entries in
    CHANNEL_BUILDERS; every derived channel is computed FROM the raster,
    so a recipe without "raster" (e.g. the #23 fft-only ablation's
    ["fft_onfly_v1"]) is valid and simply omits the raster itself.
    This is THE single definition of the input pipeline; training,
    eval-bundle, and inverse_design all call it, so a bundle's recorded
    channel_recipe fully determines its input.
    """
    if not recipe:
        raise ValueError("empty channel recipe")
    chans = []
    for i, name in enumerate(recipe):
        if name == "raster":
            if i != 0:
                raise ValueError(f"'raster' only allowed at position 0, "
                                 f"got {recipe}")
            chans.append(raster)
        elif name in CHANNEL_BUILDERS:
            chans.append(CHANNEL_BUILDERS[name](raster))
        else:
            raise ValueError(f"unknown channel builder '{name}' "
                             f"(known: {sorted(CHANNEL_BUILDERS)})")
    return torch.cat(chans, dim=1)


def add_fft_channel(X):
    """Back-compat wrapper: append fft_onfly_v1 of CHANNEL 0 ONLY.

    Fixed from the earlier version that FFT'd every input channel --
    which, on a dataset that already carried a baked FFT channel,
    produced 4-channel input including an FFT-of-an-FFT (the invalid
    2026-07-24 ablation).  Prefer build_input_channels + recipes.
    """
    return torch.cat([X, fft_onfly_v1(X[:, :1])], dim=1)


def resolve_input(X, dataset_recipe, args):
    """Apply --raster-only / --fft-channel to the loaded X; return
    (X, recipe) where recipe is the authoritative channel list recorded
    in the bundle.

    dataset_recipe: from the npz's channel_recipe field when present,
    else None -> inferred from channel count (1 -> raster; 2 -> raster +
    fft_baked_v1, the samples_128.npz convention, printed loudly since
    it is an assumption; other -> error).
    """
    n_ch = int(X.shape[1])
    if dataset_recipe is not None:
        recipe = [str(r) for r in dataset_recipe]
        if len(recipe) != n_ch:
            raise SystemExit(f"dataset channel_recipe {recipe} has "
                             f"{len(recipe)} entries but X has {n_ch} "
                             "channels -- corrupt dataset?")
    elif n_ch == 1:
        recipe = ["raster"]
    elif n_ch == 2:
        recipe = ["raster", "fft_baked_v1"]
        print("[input] ASSUMING 2-channel dataset = raster + baked FFT "
              "(build_dataset --fft-channel convention). Rebuild the "
              "dataset with the updated build_dataset.py to embed "
              "channel_recipe and silence this assumption.")
    else:
        raise SystemExit(f"dataset has {n_ch} channels and no "
                         "channel_recipe field -- cannot determine what "
                         "they are. Rebuild with updated build_dataset.py.")

    if getattr(args, "raster_only", False) and n_ch > 1:
        print(f"[input] --raster-only: keeping channel 0 of {n_ch} "
              f"(dropping {recipe[1:]})")
        X = X[:, :1]
        recipe = ["raster"]

    if getattr(args, "fft_only", False):
        if getattr(args, "fft_channel", False):
            raise SystemExit("--fft-only and --fft-channel are mutually "
                             "exclusive (fft-only IS the fft channel).")
        if recipe[0] != "raster":
            raise SystemExit(f"--fft-only needs channel 0 = raster to "
                             f"derive from, got {recipe}")
        X = fft_onfly_v1(X[:, :1])
        recipe = ["fft_onfly_v1"]
        print(f"[input] --fft-only: structure factor only -> X "
              f"{tuple(X.shape)}")

    if getattr(args, "fft_channel", False):
        if any(r.startswith("fft") for r in recipe):
            raise SystemExit(
                "--fft-channel requested but the input already contains "
                f"an FFT channel ({recipe}). Use --raster-only together "
                "with --fft-channel to replace the baked channel with "
                "the on-the-fly one, or drop --fft-channel.")
        X = torch.cat([X, fft_onfly_v1(X[:, :1])], dim=1)
        recipe = recipe + ["fft_onfly_v1"]
        print(f"[input] appended fft_onfly_v1 -> X {tuple(X.shape)}")

    print(f"[input] channel recipe: {recipe}")
    return X, recipe


def infer_bundle_recipe(bundle):
    """Channel recipe for OLD bundles that predate the recipe field.

    input_shape 1 -> raster only.
    input_shape 2 + train_config.fft_channel -> raster + fft_onfly_v1.
    input_shape 2 otherwise -> raster + fft_baked_v1 (the samples_128
    2-channel dataset; printed as an assumption).
    input_shape 4 -> the invalid double-FFT ablation bundle: refuse.
    """
    if "channel_recipe" in bundle:
        return [str(r) for r in bundle["channel_recipe"]]
    n = int(bundle["arch"].get("input_shape", 1))
    tc = bundle.get("train_config", {})
    if n == 1:
        return ["raster"]
    if n == 2 and tc.get("fft_channel", False):
        return ["raster", "fft_onfly_v1"]
    if n == 2:
        print("[bundle] ASSUMING legacy 2-channel bundle = raster + "
              "baked FFT (samples_128.npz convention).")
        return ["raster", "fft_baked_v1"]
    raise SystemExit(
        f"bundle has input_shape={n} and no channel_recipe -- if this is "
        "the 2026-07-24 double-FFT ablation bundle, it is invalid "
        "(FFT-of-FFT input); retrain from the fixed script.")


class PhotonicDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"X": self.X[idx], "y": self.y[idx]}


def normalize(X, y, train_idx):
    """Train-split-only statistics (no test leakage), per input channel."""
    x_mean = X[train_idx].mean(dim=(0, 2, 3), keepdim=True)
    x_std = X[train_idx].std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-8)
    y_mean = y[train_idx].mean()
    y_std = y[train_idx].std().clamp_min(1e-8)
    return (X - x_mean) / x_std, (y - y_mean) / y_std, \
        x_mean, x_std, y_mean, y_std


def augment_batch_d4(X):
    """Vectorised in-batch D4 augmentation: one of 8 dihedral ops per image.

    Replaces the previous per-image Python loop -- groups the batch by
    sampled op and applies each op once, so the cost is 8 tensor ops
    regardless of batch size.
    """
    k = torch.randint(0, 4, (X.shape[0],), device=X.device)
    flip = torch.rand(X.shape[0], device=X.device) > 0.5
    out = X.clone()
    for kk in range(4):
        for ff in (False, True):
            m = (k == kk) & (flip == ff)
            if not m.any():
                continue
            v = X[m]
            if ff:
                v = torch.flip(v, dims=[3])
            if kk:
                v = torch.rot90(v, kk, dims=[2, 3])
            out[m] = v
    return out


def augment_batch_shift(X):
    """In-batch cyclic-shift augmentation: random integer roll per image.

    Valid because E is exactly invariant under cyclic translations of the
    periodic supercell (the raster wraps).  ONLY channel 0 (the raster) is
    rolled: the structure-factor channel |FFT| is mathematically invariant
    under cyclic shifts of C1, so it is left untouched (and any further
    channels transform like C2).  Labels are reused unchanged.
    """
    out = X.clone()
    H, W = X.shape[2], X.shape[3]
    dx = torch.randint(0, H, (X.shape[0],))
    dy = torch.randint(0, W, (X.shape[0],))
    for i in range(X.shape[0]):
        out[i, 0] = torch.roll(X[i, 0],
                               shifts=(int(dx[i]), int(dy[i])),
                               dims=(0, 1))
    return out


D4_TTA_OPS = [lambda x, k=k, f=f: torch.rot90(
    torch.flip(x, dims=[3]) if f else x, k, dims=[2, 3])
    for f in (False, True) for k in range(4)]


# ==========================================================================
# Splitting -- group-aware and sigma-stratified
# ==========================================================================
def stratified_group_split(sigma, groups=None, val_frac=0.10, test_frac=0.10,
                           seed=42):
    """Split into train/val/test, stratified by sigma, grouped by sample.

    `groups` (e.g. sample_id) ensures every view of a physical sample lands
    in exactly one split -- REQUIRED for augmented datasets, where the old
    per-index split let rotated copies of a test layout appear in training
    (silent leakage; test metrics were optimistic).  With groups=None this
    reduces to the previous per-index stratified split.
    """
    rng = np.random.default_rng(seed)
    sigma = np.asarray(sigma, dtype=float)
    sigma_fill = np.where(np.isnan(sigma), np.inf, sigma)

    if groups is None:
        groups = np.arange(len(sigma))
    groups = np.asarray(groups)

    # one row per group, carrying its sigma (views share sigma by construction)
    uniq_groups, first_idx = np.unique(groups, return_index=True)
    group_sigma = sigma_fill[first_idx]

    train_g, val_g, test_g = [], [], []
    for s in np.unique(group_sigma):
        g = uniq_groups[group_sigma == s].copy()
        rng.shuffle(g)
        n_g = len(g)
        if n_g == 1:
            train_g.append(g[0])
            continue
        n_test = max(1, int(round(n_g * test_frac)))
        n_val = max(1, int(round(n_g * val_frac)))
        while n_test + n_val >= n_g:
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
        test_g.extend(g[:n_test].tolist())
        val_g.extend(g[n_test:n_test + n_val].tolist())
        train_g.extend(g[n_test + n_val:].tolist())

    def expand(gl):
        mask = np.isin(groups, np.asarray(gl))
        return np.where(mask)[0].astype(np.int64)

    return expand(train_g), expand(val_g), expand(test_g)


def stratified_group_folds(sigma, groups, k, seed=42):
    """Partition indices 0..len(sigma)-1 into k sigma-stratified,
    group-disjoint folds (same construction idea as
    stratified_group_split: shuffle each stratum's groups, deal them
    round-robin).  Used by --kfold-members to give each ensemble member
    its own rotated-out validation fold, so members train on genuinely
    different subsets instead of identical data + different init."""
    sigma = np.asarray(sigma, dtype=float)
    key = np.where(np.isnan(sigma), np.inf, sigma)
    groups = (np.arange(len(sigma)) if groups is None
              else np.asarray(groups))
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for s in np.unique(key):
        m = key == s
        gs = np.unique(groups[m])
        rng.shuffle(gs)
        for j, g in enumerate(gs):
            folds[j % k].append(np.flatnonzero(m & (groups == g)))
    return [np.sort(np.concatenate(f).astype(np.int64)) for f in folds]


# ==========================================================================
# Optim helpers
# ==========================================================================
def build_param_groups(model, weight_decay):
    """Exclude norm/bias params from weight decay (standard practice)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "norm" in name.lower() or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


class ModelEMA:
    """Exponential moving average of model weights, evaluated in fp32."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ema_p, p in zip(self.module.state_dict().values(),
                            model.state_dict().values()):
            if ema_p.dtype.is_floating_point:
                ema_p.mul_(d).add_(p.detach(), alpha=1.0 - d)
            else:
                ema_p.copy_(p)


# ==========================================================================
# Evaluation
# ==========================================================================
_LOGVAR_CLAMP = (-12.0, 4.0)   # normalised units; keeps exp() finite


def _mu_of(out):
    """Scalar prediction from either head: (B,1) -> (B), (B,2) -> mu."""
    return out[..., 0] if out.shape[-1] == 2 else out.squeeze(-1)


def beta_nll_loss(mu, logvar, y, beta):
    """beta-NLL (Seitzer et al. 2022): Gaussian NLL with each point's
    gradient re-weighted by var^beta (detached).  beta=0 is plain NLL --
    known to be unstable because the model can inflate sigma to mute hard
    samples; beta=1 recovers MSE-like mu gradients; 0.5 is the standard
    compromise."""
    logvar = logvar.clamp(*_LOGVAR_CLAMP)
    var = logvar.exp()
    nll = 0.5 * (logvar + (y - mu) ** 2 / var)
    if beta > 0:
        nll = nll * var.detach() ** beta
    return nll.mean()


@torch.inference_mode()
def predict(model, loader, use_tta=False):
    """Return (preds, targets) in normalised units, optionally D4-TTA-averaged."""
    model.eval()
    preds, targets = [], []
    for batch in loader:
        X_b = batch["X"].to(device, non_blocking=True)
        if use_tta:
            p = torch.stack([_mu_of(model(op(X_b)))
                             for op in D4_TTA_OPS]).mean(dim=0)
        else:
            p = _mu_of(model(X_b))
        preds.append(p.float().cpu())
        targets.append(batch["y"])
    return torch.cat(preds), torch.cat(targets)


@torch.inference_mode()
def predict_with_tta_stats(model, loader):
    """Return (preds, targets, tta_std) — TTA std per sample."""
    model.eval()
    preds, targets, tta_stds = [], [], []
    for batch in loader:
        X_b = batch["X"].to(device, non_blocking=True)
        views = torch.stack([_mu_of(model(op(X_b)))
                             for op in D4_TTA_OPS])           # (8, B)
        tta_stds.append(views.std(dim=0, unbiased=False).float().cpu())
        preds.append(views.mean(dim=0).float().cpu())
        targets.append(batch["y"])
    return torch.cat(preds), torch.cat(targets), torch.cat(tta_stds)


@torch.inference_mode()
def predict_gaussian(model, loader, use_tta=False):
    """NLL-head models only: (mu, sigma, targets) in normalised units.

    With TTA, mu is the view mean and var is the mean per-view variance
    plus the view spread of mu (law of total variance over the D4 orbit).
    """
    model.eval()
    mus, sigmas, targets = [], [], []
    ops = D4_TTA_OPS if use_tta else [lambda x: x]
    for batch in loader:
        X_b = batch["X"].to(device, non_blocking=True)
        mu_v, var_v = [], []
        for op in ops:
            out = model(op(X_b))
            mu_v.append(out[..., 0])
            var_v.append(out[..., 1].clamp(*_LOGVAR_CLAMP).exp())
        MU = torch.stack(mu_v)
        var = (torch.stack(var_v).mean(dim=0)
               + MU.var(dim=0, unbiased=False))
        mus.append(MU.mean(dim=0).float().cpu())
        sigmas.append(var.sqrt().float().cpu())
        targets.append(batch["y"])
    return torch.cat(mus), torch.cat(sigmas), torch.cat(targets)


def spearman_rho(a, b):
    """Spearman rank correlation, average-rank ties, no scipy dependency."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3:
        return float("nan")

    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(1, len(v) + 1)
        # average ties
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def within_sigma_spearman(y_true, y_pred, sigma, disorder_class=None,
                          label_noise_pct=0.09):
    """The decisive metric (audit Tests 6/9): can the surrogate rank
    layouts *within* a (class, sigma) cell?  Pooled aggregate is the
    sample-size weighted mean rho over cells with >= 3 members.

    Per cell we also report the noise-imposed CEILING on rho.  Labels
    carry engine noise (audit Test 9, 60->120 referee: differential
    spread 0.126% => per-label ~0.126/sqrt(2) ~ 0.09% of E, the default
    label_noise_pct; the earlier 0.163 constant came from the superseded
    60->90 differential of +/-0.23% and was conservative -- ceilings
    computed with it were too low, efficiencies too high); even a
    PERFECT model's correlation with noisy labels is attenuated to
        rho_max ~ sigma_s / sqrt(sigma_s^2 + sigma_n^2)
    where sigma_s is the cell's true label spread.  We report
        efficiency = rho / rho_max
    Efficiency ~ 1: the model is at the physical ceiling, nothing left to
    learn from these labels.  Efficiency ~ 0 with a healthy ceiling: a
    genuine model blind spot.  (A binary spread-vs-floor flag is too
    blunt: cells with spread below the 0.30% claimability floor can still
    be ranked well above chance, because Spearman aggregates over pairs.)
    NOTE: with n ~ 10 per cell, a single rho has sampling std ~ 0.33
    under the null; read the pattern, not individual cells.
    """
    sigma = np.asarray(sigma, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    key = np.where(np.isnan(sigma), np.inf, sigma)
    if disorder_class is not None:
        cells = list(zip(np.asarray(disorder_class), key))
    else:
        cells = [(None, s) for s in key]
    uniq = sorted(set(cells))
    per_cell, ws, rhos = {}, [], []
    for c in uniq:
        m = np.array([cc == c for cc in cells])
        if m.sum() < 3:
            continue
        rho = spearman_rho(y_true[m], y_pred[m])
        spread_pct = float(y_true[m].std() / max(abs(y_true[m].mean()),
                                                 1e-12) * 100)
        ceiling = spread_pct / np.sqrt(spread_pct ** 2
                                       + label_noise_pct ** 2)
        eff = rho / ceiling if ceiling > 0 else float("nan")
        per_cell[f"{c[0]}/sigma={c[1]:g}"] = {
            "rho": rho, "n": int(m.sum()),
            "true_spread_pct": round(spread_pct, 4),
            "rho_ceiling": round(float(ceiling), 4),
            "efficiency": round(float(eff), 4),
        }
        if np.isfinite(rho):
            rhos.append(rho)
            ws.append(m.sum())
    pooled = (float(np.average(rhos, weights=ws)) if rhos else float("nan"))
    return pooled, per_cell


def regression_metrics(y_true, y_pred):
    resid = y_true - y_pred
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "r2": float(1.0 - np.sum(resid ** 2) / max(ss_tot, 1e-12)),
        "pct_error": float(np.mean(np.abs(resid) /
                                   np.clip(np.abs(y_true), 1e-8, None)) * 100),
    }


def plot_residuals(y_true, y_pred):
    """3-panel diagnostic in de-normalised units."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=15, edgecolor="none")
    axes[0].plot(lims, lims, "r--", linewidth=1)
    axes[0].set_xlabel("Actual E")
    axes[0].set_ylabel("Predicted E")
    axes[0].set_title("Predicted vs Actual")
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=15, edgecolor="none")
    axes[1].axhline(0, color="r", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Predicted E")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs Predicted")
    axes[2].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    axes[2].axvline(0, color="r", linestyle="--", linewidth=1)
    axes[2].set_title(f"Residuals (mean={residuals.mean():.4g}, "
                      f"std={residuals.std():.4g})")
    fig.tight_layout()
    return fig


def plot_pred_by_sigma(y_true, y_pred, sigma, disorder_class=None):
    """Predicted-vs-actual colored by sigma: the plateau diagnostic.

    If the high-E shelf (flat predictions at the top of the range) is
    dominated by one color band -- the low-sigma cells, where jitter is
    sub-pixel at the legacy 64 px raster (audit: +/-65 nm ~ 1 px at 64^2
    on the 7x7 / 4550 nm supercell; at the current 128 px default 1 px ~
    35 nm, so the same jitter spans ~1.8 px) -- the ceiling is a raster
    resolution problem, not a capacity problem.  Random-class samples
    (sigma = NaN) are drawn as gray crosses.
    """
    sigma = np.asarray(sigma, dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    finite = np.isfinite(sigma)
    if (~finite).any():
        ax.scatter(y_true[~finite], y_pred[~finite], marker="x", s=30,
                   c="gray", alpha=0.7, label="random (sigma=NaN)")
    markers = {"jitter": "o", "radius": "^"}
    if disorder_class is not None and finite.any():
        dc = np.asarray(disorder_class)
        sc = None
        for cls, mk in markers.items():
            m = finite & (dc == cls)
            if not m.any():
                continue
            sc = ax.scatter(y_true[m], y_pred[m], c=sigma[m], marker=mk,
                            cmap="viridis", vmin=np.nanmin(sigma),
                            vmax=np.nanmax(sigma[finite]), s=28,
                            alpha=0.85, edgecolor="none",
                            label=f"{cls} ({mk})")
        # any finite-sigma class not in markers (e.g. ordered)
        rest = finite & ~np.isin(dc, list(markers))
        if rest.any():
            sc = ax.scatter(y_true[rest], y_pred[rest], c=sigma[rest],
                            marker="s", cmap="viridis", s=28, alpha=0.85,
                            edgecolor="k", linewidth=0.3, label="other")
    else:
        sc = ax.scatter(y_true[finite], y_pred[finite], c=sigma[finite],
                        cmap="viridis", s=28, alpha=0.85, edgecolor="none")
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="sigma")
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual E")
    ax.set_ylabel("Predicted E")
    ax.set_title("Predicted vs Actual, colored by sigma")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_spearman_by_cell(rho_cells, pooled):
    """Bar chart of within-sigma Spearman per (class, sigma) cell.

    Each bar (achieved rho) is drawn against a black tick at that cell's
    noise-imposed CEILING (max rho a perfect model could reach given the
    engine noise in the labels).  Annotation: n and efficiency =
    rho/ceiling.  Bar at the tick => nothing left to learn from these
    labels; bar far below a high tick => genuine model blind spot."""
    if not rho_cells:
        return None
    keys = sorted(rho_cells.keys())
    rhos = [rho_cells[k]["rho"] for k in keys]
    ns = [rho_cells[k]["n"] for k in keys]
    ceils = [rho_cells[k]["rho_ceiling"] for k in keys]
    effs = [rho_cells[k]["efficiency"] for k in keys]
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(keys)), 5))
    colors = ["#2a9d8f" if np.isfinite(e) and e > 0.7
              else "#e9c46a" if np.isfinite(e) and e > 0.35
              else "#e76f51" for e in effs]
    bars = ax.bar(range(len(keys)), rhos, color=colors, edgecolor="black",
                  linewidth=0.5)
    for i, (b, n, ce, ef) in enumerate(zip(bars, ns, ceils, effs)):
        ax.plot([i - 0.4, i + 0.4], [ce, ce], color="k", linewidth=1.6)
        ax.text(i, max(b.get_height(), 0) + 0.03,
                f"n={n}\neff {100 * ef:.0f}%", ha="center", fontsize=7)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axhline(pooled, color="b", linestyle=":", linewidth=1,
               label=f"pooled rho = {pooled:.3f}")
    ax.plot([], [], color="k", linewidth=1.6,
            label="noise ceiling (max achievable rho)")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("within-cell Spearman rho")
    ax.set_ylim(min(-0.25, min([r for r in rhos if np.isfinite(r)],
                               default=0) - 0.1), 1.25)
    ax.set_title("Within-sigma ranking fidelity by (class, sigma) cell\n"
                 "(bar = achieved rho, tick = noise ceiling, "
                 "color = efficiency)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ==========================================================================
# Training
# ==========================================================================
def train_one(args, loaders, seed, in_ch=1, log=None):
    torch.manual_seed(seed)
    train_loader, val_loader = loaders

    nll = bool(getattr(args, "nll_head", False))
    out_dim = 2 if nll else 1
    var_warmup = int(getattr(args, "var_warmup", 0)) if nll else 0
    pad_mode = ("circular" if getattr(args, "circular_padding", False)
                 else "zeros")
    attn = getattr(args, "attention", "se")
    recon = bool(getattr(args, "recon_head", False))
    recon_lambda = float(getattr(args, "recon_lambda", 0.1))
    model = PhotonicCNN(in_ch, args.hidden_units, out_dim,
                        dropout=args.dropout,
                        stochastic_depth=args.stochastic_depth,
                        padding_mode=pad_mode,
                        attention=attn,
                        recon_head=recon).to(device)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None

    loss_fn = nn.SmoothL1Loss(beta=args.smoothl1_beta)
    optimizer = torch.optim.AdamW(
        build_param_groups(model, args.weight_decay), lr=args.lr)

    warmup_epochs = min(args.warmup_epochs, args.epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - warmup_epochs, 1))
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = cosine

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    best_state = None
    since_improve = 0

    for epoch in range(args.epochs):
        model.train()
        loss_sum, n_b = 0.0, 0
        recon_sum = 0.0
        for batch in train_loader:
            X_b = batch["X"].to(device, non_blocking=True)
            y_b = batch["y"].to(device, non_blocking=True)
            if args.augment:
                X_b = augment_batch_d4(X_b)
            if getattr(args, "shift_aug", False):
                X_b = augment_batch_shift(X_b)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if recon:
                    out, recon_hat = model.forward_with_recon(X_b)
                else:
                    out = model(X_b)
                if nll:
                    mu_b, lv_b = out[..., 0], out[..., 1]
                    loss = (loss_fn(mu_b, y_b) if epoch < var_warmup
                            else beta_nll_loss(mu_b, lv_b, y_b,
                                               args.beta_nll))
                else:
                    loss = loss_fn(out.squeeze(-1), y_b)
                if recon:
                    # target = the (normalized, already-augmented) raster
                    # channel of this very batch: self-consistent with D4
                    # and shift augmentation by construction.
                    recon_loss = nn.functional.mse_loss(recon_hat,
                                                        X_b[:, :1])
                    loss = loss + recon_lambda * recon_loss
                    recon_sum += recon_loss.item()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            loss_sum += loss.item()
            n_b += 1
        scheduler.step()
        if recon and epoch % 20 == 0:
            print(f"    epoch {epoch}: train recon MSE "
                  f"{recon_sum / max(n_b, 1):.4f}")

        eval_model = ema.module if ema is not None else model
        if nll:
            vp, vsig, vt = predict_gaussian(eval_model, val_loader)
            if epoch < var_warmup:
                val_loss = loss_fn(vp, vt).item()
            else:                       # plain Gaussian NLL (the objective)
                v = vsig ** 2
                val_loss = float((0.5 * torch.log(2 * torch.pi * v)
                                  + (vt - vp) ** 2 / (2 * v)).mean())
        else:
            vp, vt = predict(eval_model, val_loader, use_tta=False)
            val_loss = loss_fn(vp, vt).item()
        val_mae = (vp - vt).abs().mean().item()

        if log is not None:
            d = {"epoch": epoch, "train_loss": loss_sum / max(n_b, 1),
                 "val_loss": val_loss, "val_mae": val_mae,
                 "lr": scheduler.get_last_lr()[0]}
            if recon:
                d["train_recon_mse"] = recon_sum / max(n_b, 1)
            log(d)

        # during variance warm-up the val metric is on a different scale
        # (SmoothL1 vs NLL) -- start best-model tracking after warm-up
        if nll and epoch < var_warmup:
            continue

        if val_loss < best_val:
            best_val = val_loss
            since_improve = 0
            best_state = copy.deepcopy(eval_model.state_dict())
        else:
            since_improve += 1
            if since_improve >= args.patience:
                print(f"  early stop at epoch {epoch} "
                      f"(no improvement in {args.patience})")
                break

    if best_state is None:      # e.g. epochs <= var_warmup: take the last
        best_state = copy.deepcopy(
            (ema.module if ema is not None else model).state_dict())
    final = PhotonicCNN(in_ch, args.hidden_units, out_dim,
                        dropout=args.dropout,
                        stochastic_depth=args.stochastic_depth,
                        padding_mode=pad_mode,
                        attention=attn,
                        recon_head=recon).to(device)
    final.load_state_dict(best_state)
    final.eval()
    return final, best_val


def main(argv=None):
    args = parse_args(argv)
    if args.eval_bundle:
        return eval_bundle_main(args)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.sweep:
        if wandb is None:
            raise ImportError("wandb is required for sweeps: pip install wandb")
        cfg = nll_sweep_config if args.nll_head else sweep_config
        sweep_id = wandb.sweep(cfg, project=args.project)
        print(f"\nSweep: https://wandb.ai/{wandb.api.default_entity}/{args.project}/sweeps/{sweep_id}\n")
        wandb.agent(sweep_id,
                    function=lambda: sweep_agent(args),
                    count=args.sweep_count)
        return 0

    if args.use_best:
        best_path = os.path.join(args.out_dir, "best_params.json")
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"No best_params.json at {best_path}")
        with open(best_path) as f:
            bp = json.load(f)
        for k, v in bp.items():
            if k == "best_val_loss":
                continue
            if hasattr(args, k):
                setattr(args, k, type(getattr(args, k))(v))
        print(f"Loaded best params from {best_path}")

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

    groups = data["sample_id"] if "sample_id" in data.files else None
    if groups is None:
        print("[split] no sample_id field: per-index stratified split "
              "(fine for non-augmented data; DO NOT train augmented data "
              "without sample_id -- views would leak across splits).")
    train_idx, val_idx, test_idx = stratified_group_split(
        data["sigma"], groups=groups, seed=args.seed)
    print(f"[split] train={len(train_idx)} val={len(val_idx)} "
          f"test={len(test_idx)} (grouped={'yes' if groups is not None else 'no'})")

    X_norm, y_norm, x_mean, x_std, y_mean, y_std = normalize(X, y, train_idx)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        PhotonicDataset(X_norm[train_idx], y_norm[train_idx]),
        batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(
        PhotonicDataset(X_norm[val_idx], y_norm[val_idx]),
        batch_size=args.batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(
        PhotonicDataset(X_norm[test_idx], y_norm[test_idx]),
        batch_size=args.batch_size, shuffle=False, pin_memory=pin)

    run = None
    if args.use_wandb and wandb is not None:
        run = wandb.init(project=args.project,
                         name=f"v4-seed{args.seed}-ens{args.ensemble}",
                         config=vars(args))
    log = (lambda d: wandb.log(d)) if run is not None else None

    # ------------------------- train ensemble -----------------------------
    fold_loaders = None
    if args.kfold_members:
        if args.ensemble < 2:
            raise SystemExit("--kfold-members needs --ensemble >= 2")
        # each member trains on the train+val pool minus its own rotated
        # val fold; the test split stays outside every fold, and the
        # normalisation stats stay from the master train split (computed
        # above, never from any member's val fold or from test)
        pool = np.concatenate([train_idx, val_idx])
        folds = stratified_group_folds(
            np.asarray(data["sigma"])[pool],
            groups=(np.asarray(groups)[pool] if groups is not None
                    else None),
            k=args.ensemble, seed=args.seed)
        print(f"[kfold] {args.ensemble} members, rotated val folds over "
              f"the train+val pool (n={len(pool)}); fold sizes "
              f"{[len(f) for f in folds]}; test split untouched")
        fold_loaders = []
        for f in folds:
            vi = pool[f]
            ti = np.setdiff1d(pool, vi)
            fold_loaders.append((
                DataLoader(PhotonicDataset(X_norm[ti], y_norm[ti]),
                           batch_size=args.batch_size, shuffle=True,
                           pin_memory=pin),
                DataLoader(PhotonicDataset(X_norm[vi], y_norm[vi]),
                           batch_size=args.batch_size, shuffle=False,
                           pin_memory=pin)))

    members, best_vals = [], []
    for i in range(args.ensemble):
        # --member-seed-offset shifts ONLY the member training seeds
        # (init + batch order); the split/folds above use args.seed
        # unchanged, so replicates (#25) share the exact test set.
        seed_i = args.seed + int(getattr(args, "member_seed_offset", 0)) \
            + i * 137
        loaders_i = (fold_loaders[i] if fold_loaders is not None
                     else (train_loader, val_loader))
        tag = f", val fold {i}" if fold_loaders is not None else ""
        print(f"\n=== member {i + 1}/{args.ensemble} "
              f"(seed={seed_i}{tag}) ===")
        m, bv = train_one(args, loaders_i, seed_i,
                          in_ch=in_ch, log=log if i == 0 else None)
        members.append(m)
        best_vals.append(bv)

    # ------------------------- test evaluation ----------------------------
    metrics, rho_cells = evaluate_and_report(
        args, members, test_loader, y_mean, y_std,
        sigma_test=np.asarray(data["sigma"])[test_idx],
        cls_test=(np.asarray(data["disorder_class"])[test_idx]
                  if "disorder_class" in data.files else None),
        sid_test=(np.asarray(data["sample_id"])[test_idx]
                  if "sample_id" in data.files else test_idx),
        run=run)

    # -------------------- export the surrogate bundle ---------------------
    # This single file is the contract with models/inverse_design.py.
    bundle = {
        # v2 = heteroscedastic (mu, log sigma^2) head; v1 consumers
        # (inverse_design, ensemble_uq) refuse it loudly rather than
        # silently mis-reading the 2-channel output
        "format": ("photonic-surrogate-bundle-v2"
                   if args.nll_head else "photonic-surrogate-bundle-v1"),
        "heteroscedastic": bool(args.nll_head),
        "state_dicts": [m.state_dict() for m in members],
        "arch": {"input_shape": in_ch, "hidden_units": args.hidden_units,
                 "output_shape": 2 if args.nll_head else 1,
                 "dropout": args.dropout,
                 "stochastic_depth": args.stochastic_depth,
                 "padding_mode": ("circular"
                                  if getattr(args, "circular_padding", False)
                                  else "zeros"),
                 "attention": getattr(args, "attention", "se"),
                 "img_size": int(X.shape[-1]),
                 "recon_head": bool(getattr(args, "recon_head", False))},
        "norm": {"x_mean": x_mean.flatten().tolist(),
                 "x_std": x_std.flatten().tolist(),
                 "y_mean": float(y_mean), "y_std": float(y_std)},
        "img_size": int(X.shape[-1]),
        "channel_recipe": recipe,
        "train_config": vars(args),
        "test_metrics": metrics,
        "best_val_losses": best_vals,
    }
    bundle_path = os.path.join(args.out_dir, "surrogate_bundle.pt")
    torch.save(bundle, bundle_path)
    print(f"\nsaved bundle -> {bundle_path}")

    if run is not None:
        wandb.save(bundle_path, base_path=args.out_dir)
        wandb.finish()
    return 0


def evaluate_and_report(args, members, test_loader, y_mean, y_std,
                        sigma_test, cls_test, sid_test, run=None):
    """Full test-split diagnostics: metrics, per-cell Spearman, all plots,
    JSON + per-sample CSV -- saved to out_dir AND pushed to wandb.

    Everything the plateau investigation needs lands in one place:
      residual_plots.png       the 3-panel classic
      pred_by_sigma.png        pred-vs-actual colored by sigma (shelf check)
      spearman_by_cell.png     ranking fidelity per (class, sigma) cell
      test_metrics.json        scalars + per-cell rho
      test_predictions.csv     per-sample sample_id, class, sigma, y, yhat
    In wandb: same metrics as images, metrics in run.summary, the per-cell
    table as a wandb.Table, and all files via wandb.save (Files tab).
    """
    # per-member predictions with TTA std
    member_preds, member_tta_stds = [], []
    for m in members:
        p, t, s = predict_with_tta_stats(m, test_loader)
        member_preds.append(p)
        member_tta_stds.append(s)
    preds_n = torch.stack(member_preds).mean(dim=0)       # (M, B) -> (B)
    tta_std_n = torch.stack(member_tta_stds).mean(dim=0)  # mean TTA std across ensemble
    ensemble_std_n = torch.stack(member_preds).std(dim=0, unbiased=False)  # ensemble disagreement
    preds = (preds_n * y_std + y_mean).numpy()
    targets = (t * y_std + y_mean).numpy()
    tta_std = (tta_std_n * y_std).numpy()
    ensemble_std = (ensemble_std_n * y_std).numpy()

    metrics = regression_metrics(targets, preds)
    rho_pooled, rho_cells = within_sigma_spearman(
        targets, preds, sigma_test, cls_test)
    metrics["within_sigma_spearman"] = rho_pooled
    metrics["tta_std_mean"] = float(tta_std.mean())
    metrics["ensemble_std_mean"] = float(ensemble_std.mean())
    metrics["relative_uncertainty"] = float(
        tta_std.mean() / max(abs(preds.mean()), 1e-8))
    abs_errors = np.abs(preds - targets)
    metrics["error_uncertainty_rho"] = spearman_rho(abs_errors, tta_std)

    if getattr(args, "nll_head", False):
        # heteroscedastic ensemble = Gaussian mixture:
        #   mu = mean_i mu_i,  var = mean_i(var_i + mu_i^2) - mu^2
        mem_mu, mem_var = [], []
        for m in members:
            mu_i, sig_i, _ = predict_gaussian(m, test_loader,
                                              use_tta=args.tta)
            mem_mu.append(mu_i)
            mem_var.append(sig_i ** 2)
        MU = torch.stack(mem_mu)
        mu_mix = MU.mean(dim=0)
        var_mix = (torch.stack(mem_var) + MU ** 2).mean(dim=0) - mu_mix ** 2
        sigma = (var_mix.clamp(min=0).sqrt() * y_std).numpy()
        r = np.abs(targets - (mu_mix * y_std + y_mean).numpy())
        metrics["nll_sigma_mean"] = float(sigma.mean())
        metrics["nll_picp_1sigma"] = float(np.mean(r <= sigma))
        metrics["nll_picp_2sigma"] = float(np.mean(r <= 2 * sigma))
        metrics["nll_picp_3sigma"] = float(np.mean(r <= 3 * sigma))
        # the Phase-2 acceptance metric: does sigma RANK the errors?
        metrics["error_sigma_rho"] = spearman_rho(r, sigma)

    print("\n=== test metrics (ensemble mean"
          f"{', TTA' if args.tta else ''}) ===")
    for k, v in metrics.items():
        print(f"  {k:24s} {v:.6f}")
    for cell, d in sorted(rho_cells.items()):
        print(f"  rho[{cell:24s}] = {d['rho']:+.3f}  "
              f"(n={d['n']}, ceiling {d['rho_ceiling']:.2f}, "
              f"efficiency {100 * d['efficiency']:.0f}%)")

    # ---- figures ----
    figs = {"test_residual_plots": plot_residuals(targets, preds),
            "test_pred_by_sigma": plot_pred_by_sigma(
                targets, preds, sigma_test, cls_test),
            "test_spearman_by_cell": plot_spearman_by_cell(
                rho_cells, rho_pooled)}
    fig_paths = {}
    for name, fig in figs.items():
        if fig is None:
            continue
        p = os.path.join(args.out_dir, name.replace("test_", "") + ".png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        fig_paths[name] = p

    # ---- files ----
    json_path = os.path.join(args.out_dir, "test_metrics.json")
    with open(json_path, "w") as f:
        json.dump({**metrics, "per_cell_spearman": rho_cells}, f, indent=2)
    csv_path = os.path.join(args.out_dir, "test_predictions.csv")
    cls_arr = (np.asarray(cls_test) if cls_test is not None
               else np.full(len(targets), "?"))
    with open(csv_path, "w") as f:
        f.write("sample_id,disorder_class,sigma,y_true,y_pred,abs_err\n")
        for sid, dc, sg, yt, yp in zip(sid_test, cls_arr, sigma_test,
                                       targets, preds):
            f.write(f"{sid},{dc},{sg},{yt:.6f},{yp:.6f},{abs(yt-yp):.6f}\n")
    print(f"[report] wrote {json_path}, {csv_path}, "
          f"{len(fig_paths)} figures -> {args.out_dir}")

    # ---- wandb ----
    if run is not None:
        wandb.summary.update(metrics)
        wandb.log({name: wandb.Image(p) for name, p in fig_paths.items()})
        table = wandb.Table(columns=["cell", "spearman_rho", "n",
                                     "true_spread_pct", "rho_ceiling",
                                     "efficiency"])
        for cell, d in sorted(rho_cells.items()):
            table.add_data(cell, d["rho"], d["n"], d["true_spread_pct"],
                           d["rho_ceiling"], d["efficiency"])
            safe_cell = cell.replace("/", "_").replace("=", "_")
            wandb.summary[f"rho/{safe_cell}"] = d["rho"]
            wandb.summary[f"n/{safe_cell}"] = d["n"]
            wandb.summary[f"spread/{safe_cell}"] = d["true_spread_pct"]
            wandb.summary[f"efficiency/{safe_cell}"] = d["efficiency"]
        wandb.log({"within_sigma_spearman_by_cell": table})
        for p in [json_path, csv_path, *fig_paths.values()]:
            wandb.save(p, base_path=args.out_dir)

    return metrics, rho_cells


def eval_bundle_main(args):
    """--eval-bundle: re-run the full test diagnostics on an EXISTING
    bundle, no retraining.  Rebuilds the identical test split from the
    dataset (same --seed and grouping as training => same indices) and
    uses the bundle's own normalisation stats, so numbers match the
    training run exactly.  Use this to get the sigma-colored plots and
    per-cell Spearman out of a model that has already finished (or a
    best.pth-era run re-wrapped in a bundle)."""
    os.makedirs(args.out_dir, exist_ok=True)
    bundle = torch.load(args.eval_bundle, map_location="cpu",
                        weights_only=False)
    if bundle.get("format") not in ("photonic-surrogate-bundle-v1",
                                    "photonic-surrogate-bundle-v2"):
        raise SystemExit(f"unrecognized bundle format in {args.eval_bundle}")
    # evaluate with the head the bundle was trained with, not the CLI default
    args.nll_head = bool(bundle.get("heteroscedastic", False))

    data = np.load(args.data, allow_pickle=False)
    X = torch.from_numpy(data["X"]).float()
    y = torch.from_numpy(data["y"]).float()
    if X.dim() == 3:
        X = X.unsqueeze(1)
    if int(X.shape[-1]) != int(bundle["img_size"]):
        raise SystemExit(
            f"dataset raster {int(X.shape[-1])}px != bundle "
            f"img_size {int(bundle['img_size'])}px -- wrong dataset?")
    # Rebuild the bundle's exact input from the dataset's raster channel.
    b_recipe = infer_bundle_recipe(bundle)
    ds_recipe = ([str(r) for r in data["channel_recipe"]]
                 if "channel_recipe" in data.files
                 else (["raster"] if X.shape[1] == 1
                       else ["raster", "fft_baked_v1"]))
    if ds_recipe == b_recipe:
        pass                                  # dataset already matches
    else:
        print(f"[eval] rebuilding channels {b_recipe} from the raster "
              f"(dataset carries {ds_recipe})")
        X = build_input_channels(X[:, :1], b_recipe)
    print(f"[eval] channel recipe: {b_recipe}")

    split_seed = bundle["train_config"].get("seed", args.seed)
    if split_seed != args.seed:
        print(f"[eval] using the bundle's training seed {split_seed} for "
              f"the split (not --seed {args.seed}) so the test set matches.")
    groups = data["sample_id"] if "sample_id" in data.files else None
    _, _, test_idx = stratified_group_split(
        data["sigma"], groups=groups, seed=split_seed)
    print(f"[eval] test n={len(test_idx)} "
          f"(grouped={'yes' if groups is not None else 'no'})")

    n = bundle["norm"]
    xm = torch.as_tensor(n["x_mean"], dtype=torch.float32).reshape(1, -1, 1, 1)
    xs = torch.as_tensor(n["x_std"], dtype=torch.float32).reshape(1, -1, 1, 1)
    X_norm = (X - xm) / xs
    y_norm = (y - n["y_mean"]) / n["y_std"]
    test_loader = DataLoader(
        PhotonicDataset(X_norm[test_idx], y_norm[test_idx]),
        batch_size=args.batch_size, shuffle=False)

    members = []
    for sd in bundle["state_dicts"]:
        m = PhotonicCNN(**bundle["arch"])
        m.load_state_dict(sd)
        m.to(device).eval()
        members.append(m)
    print(f"[eval] {len(members)}-model ensemble from {args.eval_bundle}")

    run = None
    if args.use_wandb and wandb is not None:
        run = wandb.init(project=args.project,
                         name=f"eval-{os.path.basename(args.eval_bundle)}",
                         config={"eval_bundle": args.eval_bundle,
                                 "data": args.data}, job_type="eval")

    evaluate_and_report(
        args, members, test_loader, n["y_mean"], n["y_std"],
        sigma_test=np.asarray(data["sigma"])[test_idx],
        cls_test=(np.asarray(data["disorder_class"])[test_idx]
                  if "disorder_class" in data.files else None),
        sid_test=(np.asarray(data["sample_id"])[test_idx]
                  if "sample_id" in data.files else test_idx),
        run=run)
    if run is not None:
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())