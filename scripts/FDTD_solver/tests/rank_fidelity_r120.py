"""Rank fidelity, round three: the full N=15 panel judged by res 120.

Context: check_rank_fidelity_v2 (N=15, res-90 referee) softened the
within-sigma floor to ~0.8-1% -- 24/105 pairs flipped, overlap instead of
clean separation.  The 60/90/120 ladder (ladder_referee.py, 5 samples)
then showed res 90 was NOT a converged referee: res 60 and res 120 agree
perfectly on all 10 pairs, the two most damaging N=15 "flips"
(1929/1892 vs 310) flip BACK at res 120, and per-sample deviations
anti-correlate across transitions (per-rung noise, not geometry).

This script settles it: re-solve the REMAINING 10 of the N=15 panel at
res 120 (the ladder's 5 are already cached and reused), then redo the
full 105-pair analysis with res 120 as referee:

  * res60 vs res120  -- the headline: the honest re-measurement of the
    within-sigma ranking floor with a better referee
  * res90 vs res120  -- how much of the N=15 damage was res-90's own
    per-sample error (uses the logged v2 res-90 values)
  * flip-rate-by-separation bins + overlap analysis in the same style as
    the v2 log, so the two measurements are directly comparable
  * revised conservative floor = largest flipped gap under the 120
    referee, with the soft-transition bins reported honestly

Numerics are IDENTICAL to ladder_referee.py (same solve_rung: campaign
wavelength grid, decay_tol, cap x1.5 at res 120, same norm-cache tag), so
the 5 cached ladder solves are bit-valid here and are picked up
automatically from gpu_out_diag/ladder_cache/.

Cost: 10 remaining samples x ~2 h each ~ 20 GPU-hours sequential.
Kill/resume safe -- every finished sample is checkpointed; rerun the same
command to continue.  Run inside tmux.

Usage (single GPU node, one command; bank defaults to
scripts/FDTD_solver/data_production, PC_OUT overrides):
    python rank_fidelity_r120.py
    python rank_fidelity_r120.py --analyze-only

Caveat to carry into any writeup: res 120 is a better referee than res 90
(it agrees with res 60 where res 90 dissents, and sits past the band-1
sign crossing), but it is not proven truth -- absolute E still oscillates
across rungs.  The claim this measurement supports is about RANKING
stability under grid refinement, which is the quantity the project uses.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS_DIR))   # solver modules
sys.path.insert(0, _TESTS_DIR)                    # sibling test modules

from ladder_referee import (cache_path, solve_rung, LOGGED_RES90,   # noqa: E402
                            spearman, bootstrap_std_ci,
                            geometry_forensics, CACHE_DIR, DIAG_DIR)
import config as C                                                  # noqa: E402
import fdtd_torch as F                                              # noqa: E402
from logutil import tee                                             # noqa: E402
from optics_core import (planar_reference_stack, solar_weight)      # noqa: E402
from run_dataset import get_materials, _load_samples, OUT_DIR       # noqa: E402

# The N=15 panel from check_rank_fidelity_v2 (bank data_production --
# formerly gpu_out_n8r60 -- sigma = 0.10).  E60 is read from the bank at
# runtime; sids listed here.
PANEL_SIDS = [1881, 315, 321, 1929, 1892, 1879, 1946, 1923, 363, 344,
              361, 1919, 329, 310, 1906]

REFEREE_RES = 120
SEP_BINS = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, np.inf)]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python rank_fidelity_r120.py",
        description="N=15 within-sigma rank fidelity with res 120 referee.")
    ap.add_argument("--sids", type=int, nargs="+", default=PANEL_SIDS)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--device", default=os.environ.get("PC_DEVICE", "auto"))
    return ap.parse_args(argv)


# --------------------------------------------------------------------------
# Solving (reuses ladder cache + numerics)
# --------------------------------------------------------------------------
def gather(args, samples_by_sid, device):
    solver_ctx = {}

    def ctx():
        if not solver_ctx:
            m = get_materials()
            ad = m["adapters"]
            wl = C.raw_wavelength_grid()
            weights = solar_weight(wl)
            ref = planar_reference_stack(ad["si"], ad["zno"], ad["ag"], wl,
                                         C.THICKNESS_NM, C.BUFFER_NM)
            solver_ctx.update(m=m, wl=wl, weights=weights, ref=ref)
        return solver_ctx

    os.makedirs(CACHE_DIR, exist_ok=True)
    E60, E90, E120 = {}, {}, {}
    todo = []
    for sid in args.sids:
        E60[sid] = float(samples_by_sid[sid]["E"])
        if sid in LOGGED_RES90:
            E90[sid] = LOGGED_RES90[sid]
        cp = cache_path(sid, REFEREE_RES)
        if os.path.exists(cp):
            E120[sid] = float(np.load(cp)["E"])
        else:
            todo.append(sid)

    if todo:
        est_h = len(todo) * 2.0
        print(f"[jobs] {len(todo)} sample(s) to solve at res {REFEREE_RES} "
              f"(~{est_h:.0f} h sequential); "
              f"{len(args.sids) - len(todo)} already cached")
    if args.analyze_only:
        todo = []

    for k, sid in enumerate(todo):
        print(f"[solve {k + 1}/{len(todo)}] sid {sid} @ res "
              f"{REFEREE_RES}...", flush=True)
        c = ctx()
        Ev, secs, spec = solve_rung(samples_by_sid[sid], REFEREE_RES,
                                    device, c["m"], c["wl"], c["weights"],
                                    c["ref"])
        np.savez(cache_path(sid, REFEREE_RES), E=Ev, runtime_s=secs,
                 res=REFEREE_RES, sid=sid, A_si=spec,
                 wavelengths_nm=c["wl"], decay_tol=C.DECAY_TOL,
                 cap=C.MAX_TIME * 1.5)
        E120[sid] = Ev
        print(f"        E(res{REFEREE_RES}) = {Ev:.4f}   ({secs:.0f}s)",
              flush=True)
    return E60, E90, E120


# --------------------------------------------------------------------------
# Pairwise analysis
# --------------------------------------------------------------------------
def pairwise(sids, Ea, Eb):
    """All pairs: (i, j, gap_a_pct, gap_b_pct, flipped)."""
    out = []
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            da = (Ea[i] - Ea[j]) / Ea[j] * 100
            db = (Eb[i] - Eb[j]) / Eb[j] * 100
            out.append((sids[i], sids[j], da, db, da * db < 0))
    return out


def report_pairs(name, pairs):
    flips = [p for p in pairs if p[4]]
    kept = [p for p in pairs if not p[4]]
    kept_gaps = sorted(abs(p[2]) for p in kept)
    print(f"\n[{name}]  {len(pairs)} pairs; {len(kept)} kept, "
          f"{len(flips)} flipped")
    if flips:
        for s1, s2, da, db, _ in sorted(flips, key=lambda p: -abs(p[2])):
            print(f"    {s1} vs {s2}: dA {da:+.2f}%  dB {db:+.2f}%")
        largest_flip = max(abs(p[2]) for p in flips)
        below = [g for g in kept_gaps if g < largest_flip]
        print(f"  largest FLIPPED gap : {largest_flip:.2f}%")
        print(f"  smallest KEPT gap   : {kept_gaps[0]:.2f}%"
              if kept_gaps else "  (no kept pairs)")
        print(f"  kept pairs below largest flipped gap: {len(below)}"
              + ("  -- OVERLAPPING (soft floor)" if below else
                 "  -- CLEAN SEPARATION"))
    else:
        largest_flip = 0.0
        print("  no flips -- perfect ranking agreement")
    print("  flip rate by |separation| bin (referee gap basis):")
    for lo, hi in SEP_BINS:
        binp = [p for p in pairs if lo <= abs(p[2]) < hi]
        if not binp:
            continue
        fr = sum(p[4] for p in binp) / len(binp)
        hi_s = f"{hi:g}" if np.isfinite(hi) else "inf"
        print(f"    {lo:.1f}-{hi_s}%: {sum(p[4] for p in binp)}/"
              f"{len(binp)} flipped ({100 * fr:.0f}%)")
    return flips, kept_gaps, largest_flip


def main(argv=None):
    args = parse_args(argv)
    tee("rank_fidelity_r120", DIAG_DIR)
    device = F.resolve_device(args.device)
    print(f"[r120] bank={OUT_DIR}  panel N={len(args.sids)}  "
          f"referee=res {REFEREE_RES}  device={device}")
    print(f"[r120] numerics: decay_tol={C.DECAY_TOL:g}, "
          f"cap={C.MAX_TIME * 1.5:g} at res {REFEREE_RES}, campaign "
          f"wavelength grid -- identical to ladder_referee (cache shared)")

    samples = _load_samples()
    samples_by_sid = {int(x["sample_id"]): x for x in samples}
    missing = [s for s in args.sids if s not in samples_by_sid]
    if missing:
        raise SystemExit(f"sids not found in bank {OUT_DIR}: {missing}")

    E60d, E90d, E120d = gather(args, samples_by_sid, device)

    sids = [s for s in args.sids if s in E120d]
    if len(sids) < len(args.sids):
        print(f"[warn] res-{REFEREE_RES} incomplete: analyzing {len(sids)}"
              f"/{len(args.sids)} (rerun to continue solving)")
    if len(sids) < 5:
        print("[abort] need >= 5 solved samples for a meaningful panel.")
        return 0

    E60 = np.array([E60d[s] for s in sids])
    E120 = np.array([E120d[s] for s in sids])
    have90 = [s for s in sids if s in E90d]
    E90 = np.array([E90d[s] for s in have90])

    # ---------------- table ----------------
    print("\n" + "=" * 74)
    print(f"PANEL TABLE (N={len(sids)})")
    print("=" * 74)
    s60120 = (E120 / E60 - 1) * 100
    print(f"  {'sid':>6} {'E60':>8} {'E90':>8} {'E120':>8} {'60->120%':>9}"
          f"   min wall")
    for k, s in enumerate(sids):
        e90s = f"{E90d[s]:8.4f}" if s in E90d else "       -"
        g = geometry_forensics(samples_by_sid[s])
        print(f"  {s:>6} {E60[k]:8.4f} {e90s} {E120[k]:8.4f} "
              f"{s60120[k]:+9.2f}   {g['min_wall_nm']:5.1f} nm")

    lo, hi = bootstrap_std_ci(s60120)
    print(f"\n  60->120 shift: mean {s60120.mean():+.2f}%  "
          f"spread {s60120.std():.3f}%  CI [{lo:.2f}%, {hi:.2f}%]")
    print(f"  (v2 reference, res-90 referee: 60->90 mean -4.85%, "
          f"spread 0.23%)")

    # ---------------- the headline: res60 judged by res120 -------------
    pairs_60_120 = pairwise(sids, E60, E120)
    rho_60_120 = spearman(E60, E120)
    print("\n" + "=" * 74)
    print(f"HEADLINE: res-60 rankings judged by res-{REFEREE_RES}  "
          f"(Spearman {rho_60_120:+.3f})")
    print("=" * 74)
    flips, kept_gaps, largest_flip = report_pairs(
        f"res60 vs res{REFEREE_RES}", pairs_60_120)

    # ---------------- how bad was the old referee ----------------------
    rho_90_120 = float("nan")
    n_flips_90 = None
    if len(have90) >= 5:
        E60_90 = np.array([E60d[s] for s in have90])
        pairs_90_120 = pairwise(have90, E90, np.array(
            [E120d[s] for s in have90]))
        pairs_60_90 = pairwise(have90, E60_90, E90)
        rho_90_120 = spearman(E90, [E120d[s] for s in have90])
        print("\n" + "=" * 74)
        print("OLD REFEREE ON TRIAL: res-90 judged by res-120  "
              f"(Spearman {rho_90_120:+.3f})")
        print("=" * 74)
        f90, _, _ = report_pairs("res90 vs res120", pairs_90_120)
        f6090, _, _ = report_pairs("res60 vs res90 (v2 replication)",
                                   pairs_60_90)
        n_flips_90 = len(f90)
        print(f"\n  interpretation: of the v2 res60-vs-res90 flips, those "
              f"that res-120 sides with res-60 on were res-90's error, "
              f"not res-60's.")

    # ---------------- verdict ------------------------------------------
    print("\n" + "=" * 74)
    print(f"VERDICT (N={len(sids)}, {len(pairs_60_120)} pairs)")
    print("=" * 74)
    verdict = {"spearman_60_120": rho_60_120,
               "spearman_90_120": rho_90_120,
               "n_flips_60_120": len(flips),
               "largest_flipped_gap_pct": largest_flip}
    if not flips:
        print(f"  res-60 rankings agree PERFECTLY with res-{REFEREE_RES} "
              f"across all {len(pairs_60_120)} pairs.  The v2 floor of "
              "0.8-1% was a res-90 artifact; the conservative floor under "
              "this referee is bounded by the smallest tested gap "
              f"({min(kept_gaps):.2f}%).")
        verdict["floor_pct"] = f"< {min(kept_gaps):.2f} (no flips observed)"
    else:
        print(f"  conservative floor (largest flipped gap under the "
              f"res-{REFEREE_RES} referee): {largest_flip:.2f}%")
        print("  quote the flip-rate bins above for the soft transition; "
              "gate the GA at the conservative number.")
        verdict["floor_pct"] = largest_flip

    # ---------------- outputs ------------------------------------------
    os.makedirs(DIAG_DIR, exist_ok=True)
    csv_path = os.path.join(DIAG_DIR, "rank_fidelity_r120.csv")
    with open(csv_path, "w") as f:
        f.write("sid,E60,E90,E120,shift60120_pct,min_wall_nm\n")
        for k, s in enumerate(sids):
            g = geometry_forensics(samples_by_sid[s])
            e90s = f"{E90d[s]:.6f}" if s in E90d else ""
            f.write(f"{s},{E60[k]:.6f},{e90s},{E120[k]:.6f},"
                    f"{s60120[k]:.4f},{g['min_wall_nm']:.2f}\n")
    with open(os.path.join(DIAG_DIR, "rank_fidelity_r120_verdict.json"),
              "w") as f:
        json.dump({**verdict, "sids": sids,
                   "shift_60_120_mean_pct": float(s60120.mean()),
                   "shift_60_120_spread_pct": float(s60120.std()),
                   "shift_spread_ci_pct": [lo, hi],
                   "numerics": {"decay_tol": C.DECAY_TOL,
                                "cap": C.MAX_TIME * 1.5,
                                "referee_res": REFEREE_RES,
                                "wavelength_grid": "campaign",
                                "mode": C.MODE}}, f, indent=2)
    print(f"\n[out] wrote {csv_path} and rank_fidelity_r120_verdict.json "
          f"in {DIAG_DIR}")

    # ---------------- figure -------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        r60 = np.argsort(np.argsort(E60)) + 1
        r120 = np.argsort(np.argsort(E120)) + 1
        axes[0].scatter(r60, r120, s=45)
        for k, s in enumerate(sids):
            axes[0].annotate(str(s), (r60[k], r120[k]),
                             textcoords="offset points", xytext=(4, 4),
                             fontsize=7)
        lim = [0.5, len(sids) + 0.5]
        axes[0].plot(lim, lim, "r--", lw=1)
        axes[0].set_xlabel("rank at res 60")
        axes[0].set_ylabel(f"rank at res {REFEREE_RES}")
        axes[0].set_title(f"Rank agreement (Spearman {rho_60_120:+.3f})")
        gaps = [abs(p[2]) for p in pairs_60_120]
        flipped = [p[4] for p in pairs_60_120]
        axes[1].hist([ [g for g, fl in zip(gaps, flipped) if not fl],
                       [g for g, fl in zip(gaps, flipped) if fl] ],
                     bins=15, stacked=True, label=["kept", "flipped"],
                     color=["#2a9d8f", "#e76f51"], edgecolor="black",
                     linewidth=0.4)
        axes[1].set_xlabel("|pairwise gap at res 60| (%)")
        axes[1].set_ylabel("pair count")
        axes[1].set_title(f"Flips vs separation (res-{REFEREE_RES} referee)")
        axes[1].legend()
        fig.tight_layout()
        p = os.path.join(DIAG_DIR, "rank_fidelity_r120.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[out] wrote {p}")
    except Exception as e:                              # noqa: BLE001
        print(f"[warn] figure skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
