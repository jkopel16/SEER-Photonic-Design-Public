"""Ablation #17 -- circular padding retuned with its own two-stage sweep.

Context: ablation #16 retrained the deployed v2 recipe with circular conv
padding + cyclic-shift augmentation under the DEPLOYED hyperparameters and
found the shift spread collapses (0.013 -> 0.0014 in E) at a ranking cost
(rho 0.701 -> 0.656, MAE 0.00544 -> 0.00570).  Open question: is that cost
intrinsic to the circular architecture, or hyperparameter mismatch (the
deployed knobs were tuned for zero padding)?  This arm answers it by
replicating the deployed two-stage sweep protocol for the circular net:

  stage A: Huber sweep (bayes, hyperband) -- architecture + schedule knobs
           (hidden_units, dropout, lr, wd, batch, smoothl1_beta, warmup,
           stochastic depth, EMA), single-member trials
  stage B: NLL sweep -- loss-adjacent knobs (lr, wd, beta_nll, var_warmup)
           with stage A's winners pinned on the CLI
  stage C: final 5-member k-fold NLL ensemble with the merged winners,
           then evaluate_bundle + circular-shift spread test + VERDICT

Both sweeps and the final train pass --circular-padding --shift-aug, so
every trial optimizes the actual architecture under test.  W&B is required
for the sweeps (bayes + hyperband live server-side); the sweep URLs print
at stage start.  Deployed bundle stays the production model regardless.

Stages are resumable: a stage is skipped when its output artifact already
exists (best_params.json for A/B via --skip detection, surrogate_bundle.pt
for C via run_training's own guard).

Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_17_circular_retune.py --dry-run
    python3 scripts/ablation/ablation_17_circular_retune.py --wandb-train
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, DATA_128, ENV_PY, REPO,      # noqa: E402
                    evaluate_bundle, run_cmd, run_training,
                    standard_train_cli, tee_into)

OUT_DIR = os.path.join(ABLATION_DIR, "circular_retune")
HUBER_DIR = os.path.join(OUT_DIR, "huber_sweep")
NLL_DIR = os.path.join(OUT_DIR, "nll_sweep")
FINAL_DIR = os.path.join(OUT_DIR, "final")
VALIDATE = os.path.join(REPO, "scripts", "interpretability",
                        "validate_saliency.py")

CIRC_FLAGS = ["--circular-padding", "--shift-aug"]
BASE = ["-i", DATA_128, "--raster-only", "--fft-channel", "--seed", "137"]

# stage A knobs (models.model sweep_config parameters) -> CLI flags
HUBER_KEYS = {
    "hidden_units": "--hidden-units",
    "dropout": "--dropout",
    "lr": "--lr",
    "weight_decay": "--weight-decay",
    "batch_size": "--batch-size",
    "smoothl1_beta": "--smoothl1-beta",
    "warmup_epochs": "--warmup-epochs",
    "stochastic_depth": "--stochastic-depth",
    "ema_decay": "--ema-decay",
}
# stage B knobs (nll_sweep_config parameters); these overwrite stage A's
# lr / weight_decay in the final merge, matching the deployed protocol
NLL_KEYS = ["lr", "weight_decay", "beta_nll", "var_warmup"]

# references for the VERDICT line
DEPLOYED_REF = {"mae": 0.005440, "rho": 0.701, "ratio": 1.005}
UNTUNED_DIR = os.path.join(ABLATION_DIR, "circular_shift")


def load_params(path):
    """best_params.json minus bookkeeping keys."""
    with open(path) as f:
        bp = json.load(f)
    for k in ("combo_score", "within_sigma_spearman", "best_val_loss"):
        bp.pop(k, None)
    return bp


def sweep_cmd(out_dir, count, nll, pin=None):
    cmd = [ENV_PY, "-u", "-m", "models.model"] + BASE + [
        "-o", out_dir, "--sweep", "--sweep-count", str(count)] + CIRC_FLAGS
    if nll:
        cmd.append("--nll-head")
    for k, v in (pin or {}).items():
        cmd.extend([HUBER_KEYS[k], str(v)])
    return cmd


def fmt(v, spec):
    return format(v, spec) if v is not None else "n/a"


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_17_circular_retune.py",
        description="Two-stage sweep + final ensemble for the circular net.")
    ap.add_argument("--n-huber", type=int, default=40,
                    help="stage A trial count (default 40)")
    ap.add_argument("--n-nll", type=int, default=30,
                    help="stage B trial count (default 30)")
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--device", default="cuda",
                    help="device for the shift-spread test (default cuda)")
    ap.add_argument("--wandb-train", action="store_true", default=False,
                    help="also stream the final 5-member training run to "
                         "W&B (the sweeps always stream)")
    args = ap.parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    tee_into("ablation_17_circular_retune", OUT_DIR)

    # ---- stage A: Huber sweep (architecture + schedule) -------------------
    huber_bp = os.path.join(HUBER_DIR, "best_params.json")
    if os.path.exists(huber_bp):
        print(f"[stage A] {huber_bp} exists -- skipping the Huber sweep "
              "(delete the dir to redo)")
    else:
        rc = run_cmd(sweep_cmd(HUBER_DIR, args.n_huber, nll=False),
                     dry_run=args.dry_run)
        if rc != 0:
            return rc
    huber = (load_params(huber_bp) if os.path.exists(huber_bp)
             else {})                                  # empty only in dry-run
    if huber:
        print(f"[stage A] winners: {huber}")
    elif not args.dry_run:
        print("[stage A] ERROR: sweep finished but wrote no best_params.json")
        return 1

    # ---- stage B: NLL sweep (loss knobs, stage A winners pinned) ----------
    nll_bp = os.path.join(NLL_DIR, "best_params.json")
    if os.path.exists(nll_bp):
        print(f"[stage B] {nll_bp} exists -- skipping the NLL sweep "
              "(delete the dir to redo)")
    else:
        pin = {k: huber[k] for k in HUBER_KEYS if k in huber}
        if args.dry_run and not pin:
            print("[dry] stage B pins stage A's winners on the CLI "
                  "(unknown until stage A runs)")
        rc = run_cmd(sweep_cmd(NLL_DIR, args.n_nll, nll=True, pin=pin),
                     dry_run=args.dry_run)
        if rc != 0:
            return rc
    nll = load_params(nll_bp) if os.path.exists(nll_bp) else {}
    if nll:
        print(f"[stage B] winners: {nll}")
    elif not args.dry_run:
        print("[stage B] ERROR: sweep finished but wrote no best_params.json")
        return 1

    # ---- stage C: final 5-member ensemble with merged winners -------------
    merged = dict(huber)
    merged.update({k: nll[k] for k in NLL_KEYS if k in nll})
    print(f"[stage C] merged hyperparameters: {merged if merged else '(dry)'}")
    cli = standard_train_cli(FINAL_DIR, extra=CIRC_FLAGS)
    if args.wandb_train:
        cli = [f for f in cli if f != "--no-wandb"]
    rc = run_training(cli, FINAL_DIR, dry_run=args.dry_run,
                      patch_params=merged or None)
    if rc != 0 or args.dry_run:
        return rc

    bundle = os.path.join(FINAL_DIR, "surrogate_bundle.pt")

    # shift-spread test FIRST (subprocess): on exclusive-process GPUs it
    # cannot get a CUDA context once this parent claims the device in
    # evaluate_bundle below.
    shift_dir = os.path.join(FINAL_DIR, "shift_test")
    rc = run_cmd([ENV_PY, VALIDATE, "--steps", "5",
                  "--bundle", bundle, "--val-dir", shift_dir,
                  "--device", args.device])
    if rc != 0:
        return rc
    spread = json.load(open(os.path.join(
        shift_dir, "validation.json")))["step5"]["mean_spread"]

    out = evaluate_bundle(bundle, label="circular_retuned")
    on = out["rows"]["test/tta_on"]

    # untuned circular arm (#16), read from disk when available
    ut_mae = ut_rho = ut_spread = None
    p = os.path.join(UNTUNED_DIR, "ablation_metrics.json")
    if os.path.exists(p):
        r = json.load(open(p))["rows"]["test/tta_on"]
        ut_mae, ut_rho = r["mae"], r["rho_pooled"]
    p = os.path.join(UNTUNED_DIR, "shift_test", "validation.json")
    if os.path.exists(p):
        ut_spread = json.load(open(p))["step5"]["mean_spread"]

    print(f"\nVERDICT: circular retuned -- test tta_on MAE {on['mae']:.6f} "
          f"rho {on['rho_pooled']:+.3f} RMS(s)/RMSE "
          f"{on.get('rms_s_over_rmse', float('nan')):.3f}, "
          f"shift spread {spread:.4f} in E.\n"
          f"  deployed (zeros, tuned):    MAE {DEPLOYED_REF['mae']:.6f} "
          f"rho +{DEPLOYED_REF['rho']:.3f} ratio {DEPLOYED_REF['ratio']:.3f} "
          "spread 0.0129\n"
          f"  circular, deployed knobs:   MAE {fmt(ut_mae, '.6f')} "
          f"rho {fmt(ut_rho, '+.3f')} spread {fmt(ut_spread, '.4f')}\n"
          "  If MAE/rho now match deployed at ~0.001-0.002 spread, the #16 "
          "gap was hyperparameter mismatch; if the gap persists, it is "
          "intrinsic and the deployed zero-padding choice is sweep-backed. "
          "Deployed bundle remains the production model either way.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
