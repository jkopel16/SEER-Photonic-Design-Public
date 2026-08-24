"""Ablation #25 -- training-noise replicates of the deployed recipe.

Context: several table rows are separated by ~0.02 in rho (SE gate
+0.020, recon parity, fft-only vs ECA), but the table has no measure of
run-to-run training noise: how much do MAE and rho move when ONLY the
member initialization seeds change, on the identical split and test
set?  This arm trains two replicates of the exact deployed recipe with
`--member-seed-offset 1000 / 2000` (init + batch order shift; the
seed-137 split, folds and normalization are untouched) and reports the
spread across {deployed, rep1, rep2}.  That spread is the error bar
every close comparison in Table~ablations inherits.

Both replicates are trained (subprocesses) BEFORE any in-process
evaluation: on exclusive-process GPUs a subprocess cannot get a CUDA
context once this parent claims the device.

Cost: ~2 training runs (5 members each).
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_25_seed_replicates.py --dry-run
    python3 scripts/ablation/ablation_25_seed_replicates.py
    python3 scripts/ablation/ablation_25_seed_replicates.py --skip-train
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, evaluate_bundle,             # noqa: E402
                    run_training, standard_train_cli, tee_into)

OUT_DIR = os.path.join(ABLATION_DIR, "seed_replicates")
REPS = [("seed_rep1", 1000), ("seed_rep2", 2000)]
DEPLOYED_JSON = os.path.join(ABLATION_DIR, "tta_deployed",
                             "ablation_metrics.json")
DEPLOYED_REF = {"mae": 0.005442, "rho_pooled": 0.701}   # fallback


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_25_seed_replicates.py",
        description="Deployed recipe, member seeds shifted: noise bar.")
    ap.add_argument("--skip-train", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--wandb", action="store_true", default=False,
                    help="log the training runs to Weights & Biases "
                         "(ablation convention is offline; opt-in)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    tee_into("ablation_25_seed_replicates", OUT_DIR)

    # ---- train ALL replicates first (subprocesses; see docstring) --------
    if not args.skip_train:
        for name, off in REPS:
            rep_dir = os.path.join(ABLATION_DIR, name)
            cli = standard_train_cli(
                rep_dir, extra=["--member-seed-offset", str(off)])
            if args.wandb:
                cli = [f for f in cli if f != "--no-wandb"]
            rc = run_training(cli, rep_dir, dry_run=args.dry_run)
            if rc != 0:
                return rc
        if args.dry_run:
            return 0

    # ---- evaluate (parent may claim the GPU now) --------------------------
    rows = {}
    if os.path.exists(DEPLOYED_JSON):
        rows["deployed"] = json.load(
            open(DEPLOYED_JSON))["rows"]["test/tta_on"]
    else:
        rows["deployed"] = dict(DEPLOYED_REF)
    for name, _ in REPS:
        bundle = os.path.join(ABLATION_DIR, name, "surrogate_bundle.pt")
        rows[name] = evaluate_bundle(bundle, label=name)["rows"]["test/tta_on"]

    print("\n=== training-noise replicates (identical split/test set) ===")
    for k, r in rows.items():
        print(f"  {k:10s} MAE {r['mae']:.6f}  rho {r['rho_pooled']:+.4f}")
    maes = [r["mae"] for r in rows.values()]
    rhos = [r["rho_pooled"] for r in rows.values()]
    n = len(rhos)
    rho_mean = sum(rhos) / n
    rho_sd = (sum((x - rho_mean) ** 2 for x in rhos) / (n - 1)) ** 0.5
    print(f"\nVERDICT: over {n} runs of the deployed recipe, rho spans "
          f"{min(rhos):+.4f} to {max(rhos):+.4f} "
          f"(range {max(rhos) - min(rhos):.4f}, sd {rho_sd:.4f}); "
          f"MAE spans {min(maes):.6f} to {max(maes):.6f}. "
          "Table deltas comfortably above this range are real; deltas "
          "within it are not resolvable at n=1 per arm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
