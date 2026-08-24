"""Ablation #4 -- D4 test-time augmentation on/off (inference only).

Context: the deployed v2 surrogate averages every prediction over the 8
D4 symmetry views (TTA) and folds the view spread into its variance (law
of total variance over the orbit).  This ablation quantifies what TTA
buys -- in accuracy AND in the UQ columns -- with NO retraining: both
arms are inference passes over the same bundle.

Why this script exists at all: models/model.py's --no-tta is a partial
no-op (evaluate_and_report's mean prediction is always TTA-averaged,
model.py:1332-1335), and ensemble_uq.py has no TTA switch.  The honest
path is predict_gaussian(use_tta=...), which common.evaluate_bundle uses.

THIS IS ALSO THE SUITE'S GATE: the tta_on/test row must reproduce the
bundle's stored test_metrics (MAE 0.005447, rho 0.7011, PICP
0.651/0.960/0.996) -- that agreement proves the shared evaluator's split
reconstruction and mixture math, on which EVERY retraining ablation row
depends.  If the gate fails, fix the evaluator before launching any
Phase-1 retrain.

Cost: GPU minutes.  Usage (GPU node):
    cd /project/rise-batteries/Photonics_RISE
    /project/rise-batteries/photonics-fdtd/bin/python3 -u \
        scripts/ablation/ablation_04_tta.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, DEPLOYED_BUNDLE,             # noqa: E402
                    evaluate_bundle, tee_into)

# Deployed v2 stored test metrics (bundle test_metrics; audit S20)
GATE = {"mae": 0.005447, "within_sigma_spearman": 0.7011,
        "nll_picp_1sigma": 0.6506, "nll_picp_2sigma": 0.9599,
        "nll_picp_3sigma": 0.9963}
GATE_TOL = 5e-4          # generous vs 1e-4 float noise; catches real drift


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_04_tta.py",
        description="TTA on/off ablation + shared-evaluator gate test.")
    ap.add_argument("--bundle", default=DEPLOYED_BUNDLE)
    ap.add_argument("--out-dir", default=os.path.join(ABLATION_DIR,
                                                      "tta_deployed"))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    tee_into("ablation_04_tta", args.out_dir)

    out = evaluate_bundle(
        args.bundle, label="deployed_v2",
        out_json=os.path.join(args.out_dir, "ablation_metrics.json"))

    on, off = out["rows"]["test/tta_on"], out["rows"]["test/tta_off"]

    # ---- gate: reproduce the bundle's stored metrics ---------------------
    stored = out.get("stored_test_metrics", {})
    print("\n=== evaluator gate (tta_on vs bundle test_metrics) ===")
    checks = [("mae", on["mae"]), ("within_sigma_spearman", on["rho_pooled"]),
              ("nll_picp_1sigma", on.get("picp_1s")),
              ("nll_picp_2sigma", on.get("picp_2s")),
              ("nll_picp_3sigma", on.get("picp_3s"))]
    ok = True
    for key, mine in checks:
        ref = stored.get(key, GATE[key])
        d = abs(mine - ref)
        flag = "OK " if d <= GATE_TOL else "FAIL"
        ok &= d <= GATE_TOL
        print(f"  {flag} {key:24s} evaluator {mine:.6f}  stored {ref:.6f}"
              f"  (|d| = {d:.2e})")
    if not ok:
        print("\nVERDICT: GATE FAILED -- the shared evaluator does not "
              "reproduce the deployed bundle's metrics. Do NOT launch any "
              "retraining ablation until this is fixed; every row would "
              "be on a broken metrics path.")
        return 1

    # ---- the ablation result ---------------------------------------------
    print("\n=== TTA ablation (test split, n = %d) ===" % on["n"])
    print(f"  {'':14s} {'TTA on':>12s} {'TTA off':>12s}")
    for k, name in [("mae", "MAE"), ("rmse", "RMSE"),
                    ("rho_pooled", "rho (pooled)"),
                    ("rms_s_over_rmse", "RMS(s)/RMSE"),
                    ("picp_1s", "PICP +-1s"), ("picp_2s", "PICP +-2s")]:
        print(f"  {name:14s} {on[k]:12.4f} {off[k]:12.4f}")
    print("\nVERDICT: gate passed -- the shared evaluator is trustworthy; "
          "the two rows above are the TTA ablation for the paper table "
          "(and the deployed-reference row every retrain compares to). "
          "Together with ablation_10's two rows they form the "
          "train-aug x TTA 2x2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
