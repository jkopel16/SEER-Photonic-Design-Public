"""Ablation suite collector -- ONE grid for the paper (FREE, rerunnable).

Scans runs/ablation/*/ablation_metrics.json (surrogate rows, all produced
by the same shared evaluator, so they are comparable by construction) and
the FDTD arms (kappa0_*, single_member_*, random_baseline/*, plus the
production runs/inverse_v2/* as reference arms) and emits:

    runs/ablation/ablation_table.csv   -- machine-readable grid
    runs/ablation/ablation_table.md    -- the paper table + footnotes

Missing inputs render as 'pending' rows, never a crash -- rerun anytime
as results land.

Pre-registered footnotes are emitted with the table so they cannot be
lost between the run and the manuscript:
  a. naive-split row is evaluated on its OWN test set (deliberate;
     measures the optimism of naive evaluation).
  b. ensemble-size sweep uses shared splits -- indicative of size
     scaling, NOT a measurement of the k-fold production model.
  c. SmoothL1 rows have no learned s; their s columns use ensemble
     member disagreement only (v1 convention).
  d. FDTD arms: means +- SE at the arm level; individual-candidate
     differences below the 0.30 % floor are not resolvable.
  e. All calibration columns are second-moment (RMS-vs-RMS) or coverage;
     no error distribution is assumed.

Usage:
    cd /project/rise-batteries/Photonics_RISE
    python3 scripts/ablation/ablation_table.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, REPO, cell_tag, CELLS,       # noqa: E402
                    fdtd_arm)

# display order + which are expected (pending if absent)
SURROGATE_ROWS = [
    ("tta_deployed", "deployed v2 (reference; #4 TTA)"),
    ("kfold_off", "#8 NLL + shared splits"),
    ("smoothl1_kfold", "#9 SmoothL1 + k-fold"),
    ("smoothl1_shared", "#9 SmoothL1 + shared"),
    ("augment_off", "#10 D4 train-aug off"),
    ("fft_off", "#11 FFT channel off"),
    ("naive_split", "#12 naive split (own test set)"),
    ("ens_k1", "#13 ensemble k=1 (shared)"),
    ("ens_k3", "#13 ensemble k=3 (shared)"),
    ("raster256", "#14 raster 256 px"),
    ("circular_shift", "#16 circular pad + shift aug"),
    ("circular_retune/final", "#16b circular, retuned (260-trial sweep)"),
    ("attn_none", "#18 no attention"),
    ("attn_cbam", "#19 CBAM"),
    ("attn_eca", "#20 ECA"),
    ("attn_sa", "#21 self-attn (all blocks)"),
    ("recon_mtl", "#22 recon multi-task"),
    ("fft_only", "#23 structure factor only"),
    ("attn_sa4", "#24 self-attn (stage 4 only)"),
    ("seed_rep1", "#25 replicate (init seeds +1000)"),
    ("seed_rep2", "#25 replicate (init seeds +2000)"),
]
COLS = ["mae", "rmse", "rho_pooled", "rms_s_over_rmse",
        "picp_1s", "picp_2s", "picp_3s"]

FOOTNOTES = """\
Footnotes (pre-registered):
  a. naive_split is evaluated on its OWN naive test set -- deliberately
     not sample-for-sample comparable; it measures the optimism of naive
     evaluation.
  b. ensemble-size rows use shared splits: indicative of size scaling
     only; the deployed model is k-fold-rotated and lives on the
     reference row, not this curve. (k=5 shared == the #8 row.)
  c. SmoothL1 rows have no learned s: s columns are ensemble member
     disagreement only (the v1 convention); expected under-coverage IS
     the result.
  d. FDTD arms report mean +- SE (arm level) and per-candidate tables
     separately; per-candidate differences below the 0.30 % within-sigma
     floor are 'not resolvable'. Random-baseline vs optimizer is
     equal-budget only over all 8 cells (160 vs 160 solves).
  e. All s-vs-error columns are RMS-vs-RMS or coverage counts -- no
     error distribution assumed (never s vs mean |error|).
  f. holdout_* rows: 'test' = in-distribution split of the remaining
     cells; the holdout row is the held-out cell (OOD generalization).
  g. attention rows replace the SE gate in every residual block with
     `--attention {none,cbam,eca,sa}`; everything else is the deployed
     v2 recipe.  none = identity (SE removed); cbam = channel + spatial
     (Woo 2018); eca = 1D-conv channel attn, no bottleneck (Wang 2020);
     sa = multi-head spatial self-attention at every block (stage-1
     cost dominates; also swaps the gate for a LayerNorm residual, a
     disclosed confound).  Rows share the deployed test split.
  h. #16 retrains with circular conv padding (torus topology) + cyclic
     shift augmentation; #22 adds a decoder reconstructing the raster
     channel from the pre-GAP features (loss + 0.1 * recon MSE) --
     regression head, validation loss and early stopping unchanged.
     Both use the deployed best_params (no re-sweep), test split shared.
"""


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_table.py",
        description="Collect every ablation result into one grid.")
    ap.add_argument("--out-csv", default=os.path.join(ABLATION_DIR,
                                                      "ablation_table.csv"))
    ap.add_argument("--out-md", default=os.path.join(ABLATION_DIR,
                                                     "ablation_table.md"))
    return ap.parse_args(argv)


def fmt(v, nd=4):
    if v is None:
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def collect_surrogate():
    """-> list of dicts: one per (dir, eval set, tta arm)."""
    found = {os.path.basename(os.path.dirname(p)): p for p in glob.glob(
        os.path.join(ABLATION_DIR, "*", "ablation_metrics.json"))}
    # nested result dirs (e.g. circular_retune/final) keyed by relative path
    found.update({os.path.relpath(os.path.dirname(p), ABLATION_DIR): p
                  for p in glob.glob(os.path.join(
                      ABLATION_DIR, "*", "*", "ablation_metrics.json"))})
    # holdout dirs are discovered dynamically (name carries the cell)
    dynamic = [(d, f"#15 cell holdout ({d.split('holdout_', 1)[1]})")
               for d in sorted(found) if d.startswith("holdout_")]
    out = []
    for dirname, label in SURROGATE_ROWS + dynamic:
        if dirname not in found:
            out.append({"row": label, "dir": dirname, "set": "-",
                        "tta": "-", "status": "pending"})
            continue
        with open(found[dirname]) as f:
            m = json.load(f)
        for key, row in sorted(m["rows"].items()):
            set_name, tta = key.rsplit("/tta_", 1)
            rec = {"row": label, "dir": dirname, "set": set_name,
                   "tta": tta, "status": "done", "n": row.get("n")}
            for c in COLS:
                rec[c] = row.get(c)
            out.append(rec)
    return out


def collect_fdtd():
    """-> list of (label, stats-dict-or-None)."""
    arms = []

    def add(label, csv_path, **kw):
        if os.path.isfile(csv_path):
            arms.append((label, fdtd_arm(csv_path, label=label, **kw)))
        else:
            arms.append((label, None))

    champ = cell_tag("jitter", 0.15)
    add(f"production LCB k=0.2 ({champ})",
        os.path.join(REPO, "runs", "inverse_v2", champ, "verification.csv"))
    add(f"#5 kappa=0 ({champ})",
        os.path.join(ABLATION_DIR, f"kappa0_{champ}", "verification.csv"))
    add(f"#26 screen-only ({champ})",
        os.path.join(ABLATION_DIR, f"screen_only_{champ}",
                     "verification.csv"))
    add(f"#7 single member ({champ})",
        os.path.join(ABLATION_DIR, f"single_member_{champ}",
                     "verification.csv"))
    for cls, s in CELLS:
        tag = cell_tag(cls, s)
        add(f"#6 random ({tag})",
            os.path.join(ABLATION_DIR, "random_baseline", tag,
                         "verification.csv"))
        add(f"    optimizer ref ({tag})",
            os.path.join(REPO, "runs", "inverse_v2", tag,
                         "verification.csv"))
    return arms


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(ABLATION_DIR, exist_ok=True)
    sur = collect_surrogate()
    fdtd = collect_fdtd()

    # ------------------------------- CSV ---------------------------------
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "row", "dir", "set", "tta", "status", "n"]
                   + COLS + ["mean_E", "se_E", "max_E", "n_claimable",
                             "true_col"])
        for r in sur:
            w.writerow(["surrogate", r["row"], r["dir"], r["set"], r["tta"],
                        r["status"], r.get("n")]
                       + [r.get(c) for c in COLS]
                       + ["", "", "", "", ""])
        for label, a in fdtd:
            if a is None:
                w.writerow(["fdtd", label, "", "", "", "pending", ""]
                           + [""] * len(COLS) + ["", "", "", "", ""])
            else:
                s = a["arm"]
                w.writerow(["fdtd", label, "", "", "", "done", s.get("n")]
                           + [""] * len(COLS)
                           + [s.get("mean"), s.get("se"), s.get("max"),
                              s.get("n_claimable"), a["true_col"]])
    print(f"[out] {args.out_csv}")

    # ------------------------------- MD ----------------------------------
    lines = ["# Ablation grid", "",
             "## Surrogate rows (shared evaluator; test split, "
             "grouped sigma-stratified unless footnoted)", "",
             "| row | set | TTA | n | MAE | RMSE | rho | RMS(s)/RMSE | "
             "PICP 1s | 2s | 3s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sur:
        if r["status"] == "pending":
            lines.append(f"| {r['row']} | pending |  |  |  |  |  |  |  |  "
                         "|  |")
            continue
        lines.append(
            f"| {r['row']} | {r['set']} | {r['tta']} | {r.get('n', '-')} | "
            f"{fmt(r.get('mae'), 6)} | {fmt(r.get('rmse'), 6)} | "
            f"{fmt(r.get('rho_pooled'), 3)} | "
            f"{fmt(r.get('rms_s_over_rmse'), 3)} | "
            f"{fmt(r.get('picp_1s'), 3)} | {fmt(r.get('picp_2s'), 3)} | "
            f"{fmt(r.get('picp_3s'), 3)} |")
    lines += ["", "## FDTD arms (verified E; mean +- SE at arm level)", "",
              "| arm | n | mean E | SE | max E | claimable | col |",
              "|---|---|---|---|---|---|---|"]
    for label, a in fdtd:
        if a is None:
            lines.append(f"| {label} | pending |  |  |  |  |  |")
        else:
            s = a["arm"]
            lines.append(f"| {label} | {s.get('n')} | {fmt(s.get('mean'))} "
                         f"| {fmt(s.get('se'))} | {fmt(s.get('max'))} | "
                         f"{s.get('n_claimable', '-')} | {a['true_col']} |")
    lines += ["", "```", FOOTNOTES, "```", ""]
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"[out] {args.out_md}")

    n_done = sum(1 for r in sur if r["status"] == "done")
    n_fdtd = sum(1 for _, a in fdtd if a is not None)
    print(f"\nVERDICT: {n_done} surrogate rows and {n_fdtd}/{len(fdtd)} "
          "FDTD arms collected; 'pending' rows fill in as runs land -- "
          "rerun this script anytime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
