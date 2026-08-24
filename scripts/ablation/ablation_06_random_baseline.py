"""Ablation #6 -- equal-budget random baseline, matched PER CELL (option a).

Context: the headline design claim is that verified optimizer candidates
beat the dataset.  The fair skeptic's question is "why not spend the same
FDTD verification budget on random draws?".  This ablation answers it at
matched budget: the production campaigns verified top-20 per cell x 8
cells = 160 solves, so the control arm is 20 FRESH random draws per cell
x 8 cells = 160 solves, best-of-20 vs top-20 WITHIN each cell.  (The
single-cell variant was explicitly rejected -- best-of-20 total across 8
cells would be a strawman.)

Draws use the production generator (disorder.make_layout, FULL geometry,
W_MIN etch rule -- manufacturable by construction) with seeds disjoint
from every banked sample (BANK_BASE_SEED + 4e6 offset).  Each draw is
scored by the deployed v2 surrogate so the pred-vs-true bookkeeping in
the verifier (surrogate bias, Spearman) stays meaningful; predictions
play NO role in selecting the draws.

PRE-REGISTERED INTERPRETATION: compare arm means +- SE and the per-cell
best-of-20 vs the production top-20; per-candidate differences below the
0.30 % floor are "not resolvable".  The bank cell means (n = 155/cell)
say random ~ cell mean, so the expected result is optimizer >> random --
but the point is to MEASURE it at equal budget.

Cost: generation + scoring minutes; verification ~160 res-60 solves
~ 30 GPU-h total, sequential per cell, fully resumable (per-cell
verify_cache + cells with a finished verification.csv are skipped).
Usage (GPU node, tmux):
    cd /project/rise-batteries/Photonics_RISE
    # smoke test (writes 2 candidates in one cell, checks schema, no FDTD):
    python3 scripts/ablation/ablation_06_random_baseline.py \
        --cells jitter_s015 --n-per-cell 2 --check-schema
    # full generation + scoring (all 8 cells):
    python3 scripts/ablation/ablation_06_random_baseline.py
    # verification (many hours; rerun to resume):
    python3 scripts/ablation/ablation_06_random_baseline.py --verify
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ABLATION_DIR, ABLATION_SEED_OFFSET,        # noqa: E402
                    A_SUPER_NM, BANK_BASE_SEED, CELLS,
                    DEPLOYED_BUNDLE, GEOM, REPO, cell_tag,
                    fdtd_arm, run_verify, tee_into,
                    write_candidate_dir)

BASE_DIR = os.path.join(ABLATION_DIR, "random_baseline")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python scripts/ablation/ablation_06_random_baseline.py",
        description="Equal-budget random control arm, 20 draws per cell.")
    ap.add_argument("--cells", default="all",
                    help="'all' or comma list of tags, e.g. "
                         "jitter_s015,radius_s020")
    ap.add_argument("--n-per-cell", type=int, default=20)
    ap.add_argument("--check-schema", action="store_true", default=False,
                    help="re-open written npz/manifest and assert every "
                         "field verify_candidates.py requires.")
    ap.add_argument("--verify", action="store_true", default=False,
                    help="FDTD-verify each cell sequentially (skips cells "
                         "whose verification.csv already exists).")
    ap.add_argument("--dry-run", action="store_true", default=False)
    return ap.parse_args(argv)


def pick_cells(spec):
    if spec == "all":
        return CELLS
    want = set(spec.split(","))
    cells = [(c, s) for c, s in CELLS if cell_tag(c, s) in want]
    if len(cells) != len(want):
        raise SystemExit(f"[abort] unknown cell tag(s) in {spec}; known: "
                         f"{[cell_tag(c, s) for c, s in CELLS]}")
    return cells


def generate_cell(cls, sigma, c_idx, n, scorer):
    """n manufacturable random draws + surrogate scores for one cell."""
    from disorder import make_layout                           # noqa: E402
    layouts = []
    for k in range(n):
        seed = BANK_BASE_SEED + ABLATION_SEED_OFFSET \
            + (c_idx + 1) * 1000 + k
        layouts.append(make_layout(cls, sigma, seed, **GEOM))
    pred = scorer.score_holes([r["holes"] for r in layouts], A_SUPER_NM)
    return layouts, pred


def check_schema(cell_dir):
    """Assert exactly what verify_candidates.load_candidates reads."""
    with open(os.path.join(cell_dir, "manifest.json")) as f:
        man = json.load(f)
    for key in ("disorder_class", "sigma", "baseline", "candidates"):
        assert key in man, f"manifest missing {key}"
    assert "pred_E_mean" in man["baseline"]
    for entry in man["candidates"]:
        assert "file" in entry and "pred_E_mean" in entry
        z = np.load(os.path.join(cell_dir, entry["file"]),
                    allow_pickle=False)
        holes = z["holes_xyr_nm"]
        assert holes.ndim == 2 and holes.shape[1] == 3
        assert float(z["a_super_nm"]) == A_SUPER_NM
    print(f"[schema] OK: {len(man['candidates'])} candidates in "
          f"{cell_dir} carry every field the verifier reads")


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(BASE_DIR, exist_ok=True)
    tee_into("ablation_06_random_baseline", BASE_DIR)
    cells = pick_cells(args.cells)

    # ---- generation + scoring --------------------------------------------
    todo = [(c, s) for c, s in cells if not os.path.exists(
        os.path.join(BASE_DIR, cell_tag(c, s), "manifest.json"))]
    if todo and not args.dry_run:
        # Scorer ON CPU deliberately: this parent stays alive through the
        # verify subprocesses, and on exclusive-process GPUs a parent-held
        # CUDA context makes every child fail with "device busy" (observed
        # 2026-08-04). 160 CPU scorings cost ~minutes; selection does not
        # depend on them anyway.
        import torch                                           # noqa: E402
        from models.inverse_design import SurrogateScorer      # noqa: E402
        scorer = SurrogateScorer(DEPLOYED_BUNDLE, torch.device("cpu"),
                                 use_tta=True, kappa=0.0, calibration=None)
    for cls, sigma in cells:
        tag = cell_tag(cls, sigma)
        cell_dir = os.path.join(BASE_DIR, tag)
        if os.path.exists(os.path.join(cell_dir, "manifest.json")):
            print(f"[skip] {tag}: candidates already written")
        elif args.dry_run:
            print(f"[dry] would draw {args.n_per_cell} layouts for {tag}")
            continue
        else:
            c_idx = CELLS.index((cls, sigma))
            layouts, pred = generate_cell(cls, sigma, c_idx,
                                          args.n_per_cell, scorer)
            # baseline pred mean: production 5000-draw estimate when the
            # cell manifest exists, else this arm's own mean (noted)
            prod_man = os.path.join(REPO, "runs", "inverse_v2", tag,
                                    "manifest.json")
            if os.path.exists(prod_man):
                with open(prod_man) as f:
                    base = json.load(f)["baseline"]["pred_E_mean"]
                note = "pred_E_mean cloned from production manifest " \
                       "(5000-draw surrogate estimate)"
            else:
                base = float(np.mean(pred["mean"]))
                note = f"own {args.n_per_cell}-draw surrogate mean " \
                       "(no production manifest found)"
            write_candidate_dir(cell_dir, cls, sigma, layouts, pred,
                                baseline_pred_mean=base,
                                baseline_note=note)
            print(f"[gen] {tag}: {args.n_per_cell} draws, surrogate "
                  f"pred mean {np.mean(pred['mean']):.4f} "
                  f"max {np.max(pred['mean']):.4f}")
        if args.check_schema:
            check_schema(cell_dir)

    # ---- verification (sequential, resumable) -----------------------------
    if args.verify:
        for cls, sigma in cells:
            tag = cell_tag(cls, sigma)
            cell_dir = os.path.join(BASE_DIR, tag)
            if os.path.exists(os.path.join(cell_dir, "verification.csv")):
                print(f"[skip] {tag}: verification.csv exists")
                continue
            print(f"\n################ verifying {tag} ################")
            rc = run_verify(cell_dir, dry_run=args.dry_run)
            if rc != 0:
                print(f"[warn] verification failed for {tag}; continuing "
                      "(rerun --verify to resume)")

    # ---- comparison table ---------------------------------------------------
    done = [(c, s) for c, s in cells if os.path.exists(
        os.path.join(BASE_DIR, cell_tag(c, s), "verification.csv"))]
    if done:
        print("\n=== random arm vs production optimizer arm (per cell) ===")
        print(f"  {'cell':14s} {'rand mean+-SE':>18s} {'rand best':>10s} "
              f"{'opt mean+-SE':>18s} {'opt best':>10s}")
        for cls, sigma in done:
            tag = cell_tag(cls, sigma)
            rnd = fdtd_arm(os.path.join(BASE_DIR, tag,
                                        "verification.csv"))["arm"]
            pcsv = os.path.join(REPO, "runs", "inverse_v2", tag,
                                "verification.csv")
            opt = fdtd_arm(pcsv)["arm"] if os.path.exists(pcsv) else None
            o = (f"{opt['mean']:.4f}+-{opt['se']:.4f}", f"{opt['max']:.4f}") \
                if opt else ("pending", "-")
            print(f"  {tag:14s} {rnd['mean']:.4f}+-{rnd['se']:.4f}"
                  f"{'':>4s} {rnd['max']:10.4f} {o[0]:>18s} {o[1]:>10s}")
        if len(done) == len(CELLS):
            print("\nVERDICT: all 8 cells verified -- this is the "
                  "equal-budget (160 vs 160 solves) matched-per-cell "
                  "baseline; quote best-of-20 vs top-20 within cells and "
                  "arm means +- SE, never per-candidate sub-floor gaps.")
        else:
            print(f"\n[note] {len(done)}/{len(CELLS)} cells verified -- "
                  "rerun --verify to continue; the equal-budget claim "
                  "needs all 8.")
    else:
        print("\n[note] nothing verified yet; the claim table appears "
              "after --verify runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
