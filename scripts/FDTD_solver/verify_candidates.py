"""Verify inverse-design candidates with the REAL FDTD engine.

Takes the export directory written by models/inverse_design.py
(candidate_XXXX.npz + manifest.json), re-solves each layout with the
production torch-FDTD stack at res 60 (bit-identical numerics to the
training bank: campaign wavelength grid, decay_tol, cap, norm-cache
tag), and reports TRUE E against three honest reference points:

  1. the banked (class, sigma) cell's true-E distribution (mean / p95 /
     max of the campaign labels) -- the real "average random layout"
     comparison, not the surrogate's predicted baseline;
  2. the surrogate's own predictions -- winner's-curse check (does the
     champion's predicted edge survive contact with the solver?) and
     rank agreement Spearman(pred, true);
  3. the Test-9 resolvability floor (0.30 % conservative): candidates
     are classified CLAIMABLE (true gain vs cell mean > floor),
     ELITE-VERIFY (positive but sub-floor -- needs the res-120 rung to
     say anything), or BELOW (no true edge).

Optionally re-solves the top K candidates AND 2 banked controls (best +
median of the cell) at res 120 -- the elite-verification rung from the
resolution ladder (never res 90) -- so champion-vs-field comparisons are
same-referee.

Cost: res 60 ~ 7-8 min/sample (15 candidates ~ 2 h); res 120 ~ 2
h/sample (top-3 + 2 controls ~ 10 h).  Per-(layout, res) checkpoint
cache: kill/resume safe; for res-120 batches on a shared GPU, run
one-candidate-per-process if the allocator OOM from the rank-fidelity
runs reappears.

Verified layouts are also exported as bank-schema sample records
(sample_9XXXXX.npz) so they can feed straight back into
build_dataset.py -> retraining (active learning).

Setup: none needed by default -- inverse_design.py exports straight into
candidates/ next to this script (scripts/FDTD_solver/candidates/), which
is also the default --in-dir here.  The bank reference comes from the
config default (scripts/FDTD_solver/data_production; override with
PC_OUT).  All outputs (cache, CSV, verdict JSON, verified_samples/) are
written into the --in-dir folder.

Usage:
    PC_COMPILE=1 PC_MODE=FULL python verify_candidates.py
    # elite rung for the top 3 (+2 banked controls):
    ... python verify_candidates.py --res120-top 3
    # re-print analysis from cache:
    ... python verify_candidates.py --analyze-only
    # or point at an archived export explicitly:
    ... python verify_candidates.py --in-dir runs/inverse/jitter_s010
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "tests"))   # ladder_referee lives there

from ladder_referee import solve_rung, spearman            # noqa: E402
import config as C                                          # noqa: E402
import fdtd_torch as F                                      # noqa: E402
from logutil import tee                                     # noqa: E402
from optics_core import planar_reference_stack, solar_weight  # noqa: E402
from run_dataset import get_materials, _load_samples, OUT_DIR  # noqa: E402
from disorder import violating_holes, air_fraction          # noqa: E402

FLOOR_PCT = 0.30      # Test 9 (res-120 referee): zero flips above 0.30 %
ELITE_RES = 120


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python verify_candidates.py",
        description="FDTD-verify inverse-design candidates.")
    ap.add_argument("--in-dir",
                    default=os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "candidates"),
                    help="Folder with the inverse_design.py output "
                         "(manifest.json + candidate_XXXX.npz). Defaults "
                         "to ./candidates next to this script.")
    ap.add_argument("--res120-top", type=int, default=0,
                    help="Also solve the top K candidates (by true res-60 "
                         "E) at res 120, plus --n-controls banked "
                         "controls, for elite verification.")
    ap.add_argument("--n-controls", type=int, default=2,
                    help="Banked same-cell controls to solve at res 120 "
                         "(best + evenly spaced).")
    ap.add_argument("--floor", type=float, default=FLOOR_PCT,
                    help="Claimability floor in %% of E (default "
                         "%(default)s: audit Test 9, res-120 referee -- "
                         "zero flips above 0.30 %%).")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--device", default=os.environ.get("PC_DEVICE", "auto"))
    return ap.parse_args(argv)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_candidates(in_dir):
    man_path = os.path.join(in_dir, "manifest.json")
    if not os.path.exists(man_path):
        raise SystemExit(
            f"no manifest.json in {in_dir} -- create the folder and copy "
            "the ENTIRE inverse_design export into it (manifest.json + "
            "all candidate_XXXX.npz), or pass --in-dir.")
    with open(man_path) as f:
        manifest = json.load(f)
    cands = []
    for entry in manifest["candidates"]:
        p = os.path.join(in_dir, entry["file"])
        with np.load(p, allow_pickle=False) as z:
            holes = np.asarray(z["holes_xyr_nm"], dtype=float)
            a_super = float(z["a_super_nm"])
        cands.append({"name": os.path.splitext(entry["file"])[0],
                      "holes_xyr_nm": holes, "a_super_nm": a_super,
                      "method": entry.get("method", "?"),
                      "pred_E_mean": float(entry["pred_E_mean"]),
                      "pred_E_std": float(entry.get("pred_E_std", 0.0)),
                      "pred_E_lcb": float(entry.get("pred_E_lcb",
                                                    entry["pred_E_mean"]))})
    return manifest, cands


def cell_bank_stats(cls, sigma):
    """True-E distribution of the banked (class, sigma) cell."""
    rows = []
    for x in _load_samples():
        x_cls = str(np.asarray(x["disorder_class"]).item()) \
            if np.asarray(x["disorder_class"]).shape == () else str(x["disorder_class"])
        x_sig = float(np.asarray(x["sigma"]))
        if x_cls == cls and np.isclose(x_sig, sigma, atol=1e-9):
            rows.append(x)
    if not rows:
        return None, []
    E = np.array([float(x["E"]) for x in rows])
    order = np.argsort(-E)
    rows = [rows[i] for i in order]
    E = E[order]
    return {"n": len(E), "mean": float(E.mean()), "std": float(E.std()),
            "p95": float(np.percentile(E, 95)), "max": float(E.max()),
            "min": float(E.min())}, rows


# --------------------------------------------------------------------------
# Solving with checkpoint cache
# --------------------------------------------------------------------------
def cache_path(cache_dir, name, res):
    return os.path.join(cache_dir, f"{name}_res{res}.npz")


def solve_cached(entry, res, cache_dir, device, ctx, analyze_only):
    cp = cache_path(cache_dir, entry["name"], res)
    if os.path.exists(cp):
        return float(np.load(cp)["E"])
    if analyze_only:
        return None
    # 37 min/sample measured at res 90 (ladder_referee); cost scales ~res^4
    est = 37 * (res / 90) ** 4
    print(f"[solve] {entry['name']} @ res {res}  (~{est:.0f} min)...",
          flush=True)
    x = {"holes_xyr_nm": entry["holes_xyr_nm"],
         "a_super_nm": entry["a_super_nm"]}
    Ev, secs, spec = solve_rung(x, res, device, ctx["m"], ctx["wl"],
                                ctx["weights"], ctx["ref"])
    np.savez(cp, E=Ev, runtime_s=secs, res=res, name=entry["name"],
             A_si=spec, wavelengths_nm=ctx["wl"], decay_tol=C.DECAY_TOL,
             cap=C.MAX_TIME * (1.5 if res > 90 else 1.0))
    print(f"        E(res{res}) = {Ev:.4f}   ({secs:.0f}s)", flush=True)
    return float(Ev)


def export_bank_record(out_dir, entry, E, ctx, sigma, cls, idx):
    """Write a bank-schema record so build_dataset can ingest the
    verified layout (active learning).  sample_id 900000+ marks
    GA-derived samples; keep them out of test splits if mixing."""
    os.makedirs(out_dir, exist_ok=True)
    sid = 900000 + idx
    cp = cache_path(os.path.join(os.path.dirname(out_dir), "verify_cache"),
                    entry["name"], 60)
    A_si = np.load(cp)["A_si"] if os.path.exists(cp) else np.zeros(1)
    np.savez(os.path.join(out_dir, f"sample_{sid:06d}.npz"),
             holes_xyr_nm=entry["holes_xyr_nm"],
             a_super_nm=entry["a_super_nm"], E=E,
             wavelengths_nm=ctx["wl"], A_si=A_si, sample_id=sid,
             disorder_class=cls, sigma=sigma, seed=-1,
             fill_achieved=air_fraction(entry["holes_xyr_nm"][:, 2],
                                        entry["a_super_nm"]))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    cache_dir = os.path.join(args.in_dir, "verify_cache")
    os.makedirs(cache_dir, exist_ok=True)
    tee("verify_candidates", args.in_dir)
    device = F.resolve_device(args.device)

    manifest, cands = load_candidates(args.in_dir)
    cls = manifest["disorder_class"]
    sigma = float(manifest["sigma"])
    pred_base = manifest["baseline"]["pred_E_mean"]
    print(f"[verify] {len(cands)} candidates, {cls} sigma={sigma}, "
          f"bank={OUT_DIR}, device={device}")
    print(f"[verify] numerics: campaign wl grid, decay_tol={C.DECAY_TOL:g}"
          f", cap={C.MAX_TIME:g} (res60) -- identical to the bank")
    print(f"[verify] gates: floor {args.floor:.2f} % (Test 9, res-120 "
          f"referee); elite rung res {ELITE_RES}")

    # ---- constraint re-check before spending GPU ----------------------
    for e in cands:
        h = e["holes_xyr_nm"]
        bad = violating_holes(h[:, :2], h[:, 2], e["a_super_nm"],
                              manifest.get("w_min_nm", 50.0))
        if bad:
            print(f"  [WARN] {e['name']}: wall violations at holes {bad} "
                  "-- solving anyway, but flag for the record")

    # ---- banked cell reference ----------------------------------------
    bank_stats, bank_rows = cell_bank_stats(cls, sigma)
    if bank_stats is None:
        print(f"  [WARN] no banked samples in cell ({cls}, {sigma}) -- "
              "gains vs cell mean unavailable; only pred-vs-true "
              "comparisons will be reported.")
    else:
        print(f"[bank cell] n={bank_stats['n']}  "
              f"mean={bank_stats['mean']:.4f}  p95={bank_stats['p95']:.4f}"
              f"  max={bank_stats['max']:.4f}")

    # ---- solver context (lazy) ----------------------------------------
    solver_ctx = {}

    def ctx():
        if not solver_ctx:
            m = get_materials()
            ad = m["adapters"]
            wl = C.raw_wavelength_grid()
            solver_ctx.update(
                m=m, wl=wl, weights=solar_weight(wl),
                ref=planar_reference_stack(ad["si"], ad["zno"], ad["ag"],
                                           wl, C.THICKNESS_NM,
                                           C.BUFFER_NM))
        return solver_ctx

    # ---- res 60: every candidate --------------------------------------
    for e in cands:
        e["E60"] = solve_cached(e, 60, cache_dir, device, ctx(),
                                args.analyze_only) \
            if not args.analyze_only or os.path.exists(
                cache_path(cache_dir, e["name"], 60)) \
            else None
    solved = [e for e in cands if e.get("E60") is not None]
    if len(solved) < len(cands):
        print(f"[note] {len(cands) - len(solved)} candidate(s) not yet "
              "solved (analyze-only or interrupted); rerun to continue.")
    if not solved:
        print("[abort] nothing solved yet.")
        return 0

    # ---- report --------------------------------------------------------
    print("\n" + "=" * 76)
    print("VERIFICATION TABLE (res 60 -- same numerics as the bank)")
    print("=" * 76)
    hdr = (f"  {'candidate':<16} {'method':<8} {'pred E':>8} {'true E60':>9}"
           f" {'pred-true':>9}")
    if bank_stats:
        hdr += f" {'gain vs cell mean':>17}  verdict"
    print(hdr)
    for e in sorted(solved, key=lambda e: -e["E60"]):
        row = (f"  {e['name']:<16} {e['method']:<8} "
               f"{e['pred_E_mean']:>8.4f} {e['E60']:>9.4f} "
               f"{e['pred_E_mean'] - e['E60']:>+9.4f}")
        if bank_stats:
            gain = (e["E60"] / bank_stats["mean"] - 1) * 100
            e["gain_pct"] = gain
            verdict = ("CLAIMABLE" if gain > args.floor
                       else "ELITE-VERIFY" if gain > 0 else "below mean")
            e["verdict"] = verdict
            row += f" {gain:>+16.2f}%  {verdict}"
        print(row)

    predv = np.array([e["pred_E_mean"] for e in solved])
    truev = np.array([e["E60"] for e in solved])
    bias = float((predv - truev).mean())
    rho = spearman(predv, truev) if len(solved) >= 3 else float("nan")
    print(f"\n  surrogate bias on champions (pred - true): {bias:+.4f} "
          f"({100 * bias / truev.mean():+.2f} %)  <- winner's-curse check")
    print(f"  rank agreement Spearman(pred, true): {rho:+.3f}  "
          f"(n={len(solved)})")
    if bank_stats:
        best = max(solved, key=lambda e: e["E60"])
        print(f"\n  CHAMPION: {best['name']}  true E60 = {best['E60']:.4f}")
        print(f"    vs cell mean {bank_stats['mean']:.4f}: "
              f"{(best['E60'] / bank_stats['mean'] - 1) * 100:+.2f} % "
              f"(floor {args.floor:.2f} %)")
        print(f"    vs cell p95  {bank_stats['p95']:.4f}: "
              f"{(best['E60'] / bank_stats['p95'] - 1) * 100:+.2f} %")
        print(f"    vs cell MAX  {bank_stats['max']:.4f}: "
              f"{(best['E60'] / bank_stats['max'] - 1) * 100:+.2f} %  "
              "<- beats every random draw in the bank?")
        n_beat = sum(e["E60"] > bank_stats["max"] for e in solved)
        print(f"    candidates above the banked max: {n_beat}/{len(solved)}")

    # ---- res 120 elite rung -------------------------------------------
    if args.res120_top > 0:
        print("\n" + "=" * 76)
        print(f"ELITE VERIFICATION (res {ELITE_RES})")
        print("=" * 76)
        top = sorted(solved, key=lambda e: -e["E60"])[:args.res120_top]
        controls = []
        if bank_rows:
            picks = [0] + [round(i * (len(bank_rows) - 1) /
                                 max(args.n_controls - 1, 1))
                           for i in range(1, args.n_controls)]
            for k in sorted(set(picks))[:args.n_controls]:
                x = bank_rows[k]
                controls.append({
                    "name": f"control_sid{int(np.asarray(x['sample_id'])):06d}",
                    "holes_xyr_nm": np.asarray(x["holes_xyr_nm"], float),
                    "a_super_nm": float(np.asarray(x["a_super_nm"])),
                    "E60": float(x["E"]), "is_control": True})
        for e in top + controls:
            e["E120"] = solve_cached(e, ELITE_RES, cache_dir, device,
                                     ctx(), args.analyze_only)
        done = [e for e in top + controls if e.get("E120") is not None]
        if done:
            print(f"\n  {'layout':<22} {'E60':>8} {'E120':>9} "
                  f"{'shift %':>8}")
            for e in sorted(done, key=lambda e: -e["E120"]):
                tag = "  [banked control]" if e.get("is_control") else ""
                print(f"  {e['name']:<22} {e['E60']:>8.4f} "
                      f"{e['E120']:>9.4f} "
                      f"{(e['E120'] / e['E60'] - 1) * 100:>+8.2f}{tag}")
            ctrl = [e for e in done if e.get("is_control")]
            champ = [e for e in done if not e.get("is_control")]
            if ctrl and champ:
                cm = max(ctrl, key=lambda e: e["E120"])
                bm = max(champ, key=lambda e: e["E120"])
                print(f"\n  same-referee margin, best champion vs best "
                      f"control: "
                      f"{(bm['E120'] / cm['E120'] - 1) * 100:+.2f} %  "
                      f"(floor {args.floor:.2f} %)")

    # ---- outputs -------------------------------------------------------
    csv_path = os.path.join(args.in_dir, "verification.csv")
    with open(csv_path, "w") as f:
        f.write("name,method,pred_E_mean,pred_E_lcb,true_E60,"
                "gain_vs_cell_mean_pct,verdict,true_E120\n")
        for e in solved:
            f.write(f"{e['name']},{e['method']},{e['pred_E_mean']:.5f},"
                    f"{e['pred_E_lcb']:.5f},{e['E60']:.5f},"
                    f"{e.get('gain_pct', float('nan')):.3f},"
                    f"{e.get('verdict', '')},"
                    f"{e.get('E120', float('nan'))}\n")
    verdict_json = {
        "cell": {"class": cls, "sigma": sigma}, "floor_pct": args.floor,
        "bank_cell": bank_stats, "pred_baseline_mean": pred_base,
        "surrogate_bias_on_champions": bias,
        "spearman_pred_true": rho,
        "n_claimable": sum(e.get("verdict") == "CLAIMABLE"
                           for e in solved),
        "champion": max(solved, key=lambda e: e["E60"])["name"],
        "numerics": {"decay_tol": C.DECAY_TOL, "cap60": C.MAX_TIME,
                     "cap120": C.MAX_TIME * 1.5,
                     "wavelength_grid": "campaign", "mode": C.MODE},
    }
    with open(os.path.join(args.in_dir, "verification_verdict.json"),
              "w") as f:
        json.dump(verdict_json, f, indent=2)
    # active-learning export
    exp_dir = os.path.join(args.in_dir, "verified_samples")
    for i, e in enumerate(sorted(solved, key=lambda e: -e["E60"])):
        export_bank_record(exp_dir, e, e["E60"], ctx() if solver_ctx
                           else {"wl": np.load(cache_path(
                               cache_dir, e["name"], 60))["wavelengths_nm"]},
                           sigma, cls, i)
    print(f"\n[out] verification.csv + verdict JSON -> {args.in_dir}")
    print(f"[out] {len(solved)} bank-schema records -> {exp_dir} "
          "(active learning; sample_id 900000+)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
