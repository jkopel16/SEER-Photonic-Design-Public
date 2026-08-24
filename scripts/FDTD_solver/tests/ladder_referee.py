"""Resolution ladder 60/90/120: is res-90 a trustworthy ranking referee?

Motivation (check_rank_fidelity_v2, N=15, 2026-07-23): res-60 vs res-90
rankings at sigma=0.10 disagreed on 24/105 pairs with flips up to 0.75%
separation, softening the within-sigma resolvability floor to ~0.8-1%.
But that conclusion silently assumes res-90 orderings are themselves
converged -- and the resolution audit flagged res 90 as a lucky band-1
error crossing, not a convergence floor.  This script adds the missing
rung: re-solve a small set of banked samples at res 120 and ask two
questions the N=15 run could not answer:

  Q1 (referee stability)   Do res-90 rankings survive res 120?  If
     Spearman(90,120) ~ 1 with no flips, res 90 was a valid referee and
     the ~0.8-1% floor stands.  If 90->120 flips look like 60->90 flips,
     part of the 24 flips were res-90's fault and the floor at res 60 is
     better than measured (and unknown until a converged referee exists).
  Q2 (elite-verification floor)   Does the per-sample differential shift
     (the +/-0.23% jitter) SHRINK from 60->90 to 90->120?  If yes, the
     escape hatch is real: champions verified at res 120 (NEVER res 90 --
     see Q1's outcome below) can claim margins finer than the res-60
     floor, even though the res-60 training labels cannot.  The spread
     ratio is that floor's scale factor.

OUTCOME (2026-07-25, audit Test 8): Q1 returned res90-UNRELIABLE -- res
60 and res 120 agree on all 10 pairs (Spearman +1.000) while res 90
dissents against both neighbours.  Followed by rank_fidelity_r120.py
(Test 9): floor 0.30%, 60->120 spread 0.126%.

Also computed per sample, for free: geometry forensics (min wall gap,
radius stats) to test whether the anomalous common-mode outliers
(1929/1892/310 -- 18 of the 24 flips) share a geometric signature.

Design choices (all deliberate, to isolate the resolution effect):
  * SAME wavelength grid at every rung (the campaign grid) -- the planar
    TMM reference is analytic and resolution-independent, so any E shift
    is purely the FDTD A_si.
  * decay_tol and cap match the campaign/rank-fidelity numerics at res
    90 (so its rung reproduces the referee run); res 120 gets cap x1.5,
    the same elevation cmd_verify uses.
  * res-60 E comes from the bank (no re-solve); res-90 E can be imported
    from the logged check_rank_fidelity_v2 table (--use-logged-res90,
    default ON for the 15 known sids) or re-solved.
  * Per-(sample, res) checkpoint cache: kill/resume safe -- rerun the
    same command after any interruption and it continues.

Cost estimate: res 120 ~ (120/90)^4 ~ 3.2x the ~37 min/sample of res 90
=> ~2 h/sample on an A40/A100; the default 5-sample panel is ~10 h
sequential on one GPU.

Usage -- single GPU node, one command:
    python ladder_referee.py
(bank defaults to scripts/FDTD_solver/data_production; PC_OUT overrides)
Solves everything missing, then runs the full analysis in the same
invocation.  Kill/resume safe: every finished (sample, rung) is cached,
so rerunning the same command continues where it stopped.  Optional:
    --sids 1929 1892 310        # smaller/other panel
    --analyze-only              # just re-analyze the cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS_DIR))   # solver modules

import config as C                                   # noqa: E402
import fdtd_torch as F                               # noqa: E402
from logutil import tee                              # noqa: E402
from optics_core import (planar_reference_stack, solar_weight,  # noqa: E402
                         enhancement)
from disorder import pair_gaps                       # noqa: E402
from run_dataset import (get_materials, _load_samples,          # noqa: E402
                         OUT_DIR)
from fdtd_torch import broadband_absorption_many     # noqa: E402

# Diagnostics land in tests/gpu_out_diag regardless of CWD (PC_DIAG overrides).
DIAG_DIR = os.environ.get("PC_DIAG") or os.path.join(_TESTS_DIR, "gpu_out_diag")
CACHE_DIR = os.path.join(DIAG_DIR, "ladder_cache")

# Default picks: the three anomalous common-mode outliers from the N=15
# run (18 of 24 flips between them) + the two E-range extremes as
# common-mode-typical controls.
DEFAULT_SIDS = [1929, 1892, 310, 1881, 1906]

# E(res90) transcribed from the check_rank_fidelity_v2 log of 2026-07-23
# (bank data_production -- formerly gpu_out_n8r60 -- sigma=0.10,
# campaign wavelength grid, decay_tol
# 3e-4, cap 2500).  Used only when --use-logged-res90 is on AND the
# requested sid appears here; anything else is solved fresh.
# These are the res-90 values later shown to be the unconverged referee;
# kept ONLY so Test 9 (rank_fidelity_r120.py) can put that referee on trial.
LOGGED_RES90 = {
    1881: 2.4801, 315: 2.4836, 321: 2.4811, 1929: 2.4955, 1892: 2.4975,
    1879: 2.4914, 1946: 2.4916, 1923: 2.4904, 363: 2.4960, 344: 2.5006,
    361: 2.4907, 1919: 2.5032, 329: 2.4988, 310: 2.4940, 1906: 2.5044,
}

RUNGS = (60, 90, 120)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python ladder_referee.py",
        description="60/90/120 resolution ladder: referee validation.")
    ap.add_argument("--sids", type=int, nargs="+", default=DEFAULT_SIDS)
    ap.add_argument("--use-logged-res90", action="store_true", default=True)
    ap.add_argument("--no-use-logged-res90", dest="use_logged_res90",
                    action="store_false",
                    help="Re-solve res 90 instead of importing the logged "
                         "check_rank_fidelity_v2 values.")
    ap.add_argument("--analyze-only", action="store_true",
                    help="Skip solving; analyze whatever is cached "
                         "(missing rungs are reported, not solved).")
    ap.add_argument("--device", default=os.environ.get("PC_DEVICE", "auto"))
    return ap.parse_args(argv)


# --------------------------------------------------------------------------
# Solving one (sample, resolution) rung with checkpointing
# --------------------------------------------------------------------------
def cache_path(sid, res):
    return os.path.join(CACHE_DIR, f"sid{sid:06d}_res{res}.npz")


def solve_rung(x, res, device, m, wl, weights, ref):
    """One FDTD solve at the given resolution; returns (E, seconds)."""
    holes = [tuple(h) for h in np.asarray(x["holes_xyr_nm"])]
    cap = C.MAX_TIME * (1.5 if res > 90 else 1.0)
    t0 = time.time()
    A_si, _, _ = broadband_absorption_many(
        [holes], float(x["a_super_nm"]), C.THICKNESS_NM, wl,
        m["fits"], C.BUFFER_NM, res, C.DECAY_TOL, cap,
        os.path.join(OUT_DIR, "norm_cache"), device=device,
        n_cells_tag=f"lad{C.N_CELLS}r{res}")
    E = enhancement(np.nan_to_num(A_si[0]), ref["A_si"], weights)[0]
    return float(E), time.time() - t0, np.nan_to_num(A_si[0])


def gather_E(args, samples_by_sid, device):
    """Fill the (sid, rung) -> E table, solving what's missing.

    Materials/reference are initialized lazily -- analyze-only runs (and
    runs where everything is already cached) never touch the solver."""
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

    # build the full to-solve list up front
    jobs = []
    E = {sid: {} for sid in args.sids}
    for sid in args.sids:
        x = samples_by_sid[sid]
        E[sid][60] = float(x["E"])                       # banked label
        for res in (90, 120):
            if res == 90 and args.use_logged_res90 and sid in LOGGED_RES90:
                E[sid][90] = LOGGED_RES90[sid]
                continue
            cp = cache_path(sid, res)
            if os.path.exists(cp):
                E[sid][res] = float(np.load(cp)["E"])
                continue
            jobs.append((sid, res))

    my_jobs = [] if args.analyze_only else jobs
    if jobs:
        est_h = sum(37 * (r / 90) ** 4 for _, r in jobs) / 60
        print(f"[jobs] {len(jobs)} rung(s) to solve (~{est_h:.1f} h "
              f"total on this GPU)")

    for sid, res in my_jobs:
        est = 37 * (res / 90) ** 4
        print(f"[solve] sid {sid} @ res {res}  (~{est:.0f} min)...",
              flush=True)
        c = ctx()
        Ev, secs, spec = solve_rung(samples_by_sid[sid], res, device,
                                    c["m"], c["wl"], c["weights"], c["ref"])
        np.savez(cache_path(sid, res), E=Ev, runtime_s=secs, res=res,
                 sid=sid, A_si=spec, wavelengths_nm=c["wl"],
                 decay_tol=C.DECAY_TOL,
                 cap=C.MAX_TIME * (1.5 if res > 90 else 1.0))
        E[sid][res] = Ev
        print(f"        E(res{res}) = {Ev:.4f}   ({secs:.0f}s)", flush=True)

    # pick up anything a previous (killed) run already cached
    for sid in args.sids:
        for res in (90, 120):
            if res not in E[sid] and os.path.exists(cache_path(sid, res)):
                E[sid][res] = float(np.load(cache_path(sid, res))["E"])
    return E


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def pair_flips(sids, Ea, Eb, la, lb):
    """All pairs where rung a and rung b disagree on the ordering."""
    flips, kept_gaps = [], []
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            da = (Ea[i] - Ea[j]) / Ea[j] * 100
            db = (Eb[i] - Eb[j]) / Eb[j] * 100
            if da * db < 0:
                flips.append((sids[i], sids[j], da, db))
            else:
                kept_gaps.append(abs(da))
    return flips, kept_gaps


def bootstrap_std_ci(v, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    boots = np.array([v[rng.integers(0, len(v), len(v))].std()
                      for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def geometry_forensics(x):
    holes = np.asarray(x["holes_xyr_nm"], float)
    L = float(x["a_super_nm"])
    _, _, gaps = pair_gaps(holes[:, :2], holes[:, 2], L)
    r = holes[:, 2]
    return {"min_wall_nm": float(gaps.min()),
            "p5_wall_nm": float(np.percentile(gaps, 5)),
            "r_min_nm": float(r.min()), "r_max_nm": float(r.max()),
            "r_std_nm": float(r.std())}


def analyze(args, E, samples_by_sid):
    sids = [s for s in args.sids if all(r in E[s] for r in RUNGS)]
    missing = [s for s in args.sids if s not in sids]
    if missing:
        print(f"[warn] incomplete rungs for sids {missing}; analyzing "
              f"{len(sids)} complete samples")
    if len(sids) < 3:
        print("[abort] need >= 3 complete samples to analyze rankings.")
        return

    E60 = np.array([E[s][60] for s in sids])
    E90 = np.array([E[s][90] for s in sids])
    E120 = np.array([E[s][120] for s in sids])
    s6090 = (E90 / E60 - 1) * 100
    s90120 = (E120 / E90 - 1) * 100

    print("\n" + "=" * 72)
    print("LADDER TABLE")
    print("=" * 72)
    print(f"  {'sid':>6} {'E60':>8} {'E90':>8} {'E120':>8} "
          f"{'60->90%':>9} {'90->120%':>9}   geometry (min wall / r spread)")
    for k, s in enumerate(sids):
        g = geometry_forensics(samples_by_sid[s])
        tag = " <- anomalous" if s in (1929, 1892, 310) else ""
        print(f"  {s:>6} {E60[k]:8.4f} {E90[k]:8.4f} {E120[k]:8.4f} "
              f"{s6090[k]:+9.2f} {s90120[k]:+9.2f}   "
              f"{g['min_wall_nm']:5.1f} nm / {g['r_std_nm']:4.1f} nm{tag}")

    # -- common mode + differential, both transitions --------------------
    dev6090 = s6090 - s6090.mean()
    dev90120 = s90120 - s90120.mean()
    lo1, hi1 = bootstrap_std_ci(s6090)
    lo2, hi2 = bootstrap_std_ci(s90120)
    ratio = (s90120.std() / s6090.std()) if s6090.std() > 0 else np.nan
    print("\nCOMMON MODE + DIFFERENTIAL JITTER")
    print(f"  60->90 : mean {s6090.mean():+.2f}%  spread {s6090.std():.3f}%"
          f"  CI [{lo1:.2f}%, {hi1:.2f}%]")
    print(f"  90->120: mean {s90120.mean():+.2f}%  spread {s90120.std():.3f}%"
          f"  CI [{lo2:.2f}%, {hi2:.2f}%]")
    print(f"  spread ratio (90->120)/(60->90) = {ratio:.2f}"
          f"   [<1: jitter shrinks at higher res -> elite-verification "
          f"floor scales down accordingly]")

    # -- rankings across rungs -------------------------------------------
    print("\nRANKING STABILITY")
    for (la, Ea), (lb, Eb) in [((60, E60), (90, E90)),
                               ((90, E90), (120, E120)),
                               ((60, E60), (120, E120))]:
        flips, kept = pair_flips(sids, Ea, Eb, la, lb)
        rho = spearman(Ea, Eb)
        n_pairs = len(sids) * (len(sids) - 1) // 2
        print(f"  res{la} vs res{lb}: Spearman {rho:+.3f}, "
              f"{len(flips)}/{n_pairs} pairs flipped"
              + (f", largest flipped gap "
                 f"{max(abs(f[2]) for f in flips):.2f}%" if flips else ""))
        for f in flips:
            print(f"      {f[0]} vs {f[1]}: d{la} {f[2]:+.2f}%  "
                  f"d{lb} {f[3]:+.2f}%")

    # -- verdict ----------------------------------------------------------
    flips90120, _ = pair_flips(sids, E90, E120, 90, 120)
    flips6090, _ = pair_flips(sids, E60, E90, 60, 90)
    rho90120 = spearman(E90, E120)
    verdict = {}
    print("\n" + "=" * 72)
    print("VERDICT (n={} -- treat as directional, not definitive)"
          .format(len(sids)))
    print("=" * 72)
    if rho90120 >= 0.9 and len(flips90120) == 0:
        # [NOT the observed outcome -- the 2026-07-25 run returned
        #  res90-unreliable; see the docstring OUTCOME stamp.]
        verdict["referee"] = "res90-stable"
        print("  Q1: res-90 rankings SURVIVE res 120 on this set -> res 90 "
              "was a valid referee; the ~0.8-1% floor from the N=15 run "
              "stands as measured.")
    elif len(flips90120) >= max(1, len(flips6090) // 2):
        verdict["referee"] = "res90-unreliable"
        print("  Q1: res-90 rankings CHURN at res 120 at a rate comparable "
              "to 60->90 -> res 90 was NOT a converged referee; part of "
              "the 24 N=15 flips are res-90's fault, the true res-60 "
              "floor is better than 0.8-1% but unknown until a converged "
              "referee exists (consider a res-150 spot check).")
    else:
        verdict["referee"] = "res90-marginal"
        print("  Q1: some 90->120 churn but less than 60->90 -> res 90 is "
              "a partially converged referee; the floor is between the "
              "measured 0.8-1% and something finer.")
    if np.isfinite(ratio) and ratio < 0.6:
        verdict["elite_hatch"] = "open"
        print(f"  Q2: differential jitter SHRINKS x{ratio:.2f} at the "
              "higher rung -> the elite-verification escape hatch is "
              "open: champions re-solved at res 120 can claim margins "
              f"~{ratio:.2f}x the res-60 floor.")
    elif np.isfinite(ratio) and ratio < 1.0:
        verdict["elite_hatch"] = "partial"
        print(f"  Q2: jitter shrinks modestly (x{ratio:.2f}) -> elevated "
              "verification helps but does not dissolve the floor.")
    else:
        verdict["elite_hatch"] = "closed"
        print(f"  Q2: jitter does NOT shrink (x{ratio:.2f}) -> the "
              "differential noise is not a grid artifact that refinement "
              "removes at this scale; margins under the floor stay "
              "unclaimable even for verified elites.")

    # -- outputs ----------------------------------------------------------
    os.makedirs(DIAG_DIR, exist_ok=True)
    csv_path = os.path.join(DIAG_DIR, "ladder.csv")
    with open(csv_path, "w") as f:
        f.write("sid,E60,E90,E120,shift6090_pct,shift90120_pct,"
                "dev6090_pct,dev90120_pct,min_wall_nm,r_std_nm,anomalous\n")
        for k, s in enumerate(sids):
            g = geometry_forensics(samples_by_sid[s])
            f.write(f"{s},{E60[k]:.6f},{E90[k]:.6f},{E120[k]:.6f},"
                    f"{s6090[k]:.4f},{s90120[k]:.4f},{dev6090[k]:.4f},"
                    f"{dev90120[k]:.4f},{g['min_wall_nm']:.2f},"
                    f"{g['r_std_nm']:.2f},{int(s in (1929, 1892, 310))}\n")
    with open(os.path.join(DIAG_DIR, "ladder_verdict.json"), "w") as f:
        json.dump({"sids": sids, "spearman_60_90": spearman(E60, E90),
                   "spearman_90_120": rho90120,
                   "spearman_60_120": spearman(E60, E120),
                   "spread_6090_pct": float(s6090.std()),
                   "spread_90120_pct": float(s90120.std()),
                   "spread_ratio": float(ratio),
                   "flips_60_90": len(flips6090),
                   "flips_90_120": len(flips90120),
                   "verdict": verdict,
                   "numerics": {"decay_tol": C.DECAY_TOL,
                                "cap60_90": C.MAX_TIME,
                                "cap120": C.MAX_TIME * 1.5,
                                "wavelength_grid": "campaign",
                                "mode": C.MODE}}, f, indent=2)
    print(f"\n[out] wrote {csv_path} and ladder_verdict.json in {DIAG_DIR}")

    # -- figure -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for k, s in enumerate(sids):
            style = dict(marker="o", linewidth=1.5)
            if s in (1929, 1892, 310):
                style.update(linestyle="--")
            axes[0].plot(RUNGS, [E60[k], E90[k], E120[k]],
                         label=f"sid {s}", **style)
        axes[0].set_xticks(RUNGS)
        axes[0].set_xlabel("resolution (px/um)")
        axes[0].set_ylabel("E")
        axes[0].set_title("Ladder: E per rung (dashed = anomalous trio)")
        axes[0].legend(fontsize=7)
        axes[1].scatter(dev6090, dev90120, s=45)
        for k, s in enumerate(sids):
            axes[1].annotate(str(s), (dev6090[k], dev90120[k]),
                             textcoords="offset points", xytext=(4, 4),
                             fontsize=8)
        lim = max(np.abs(np.r_[dev6090, dev90120]).max() * 1.2, 0.1)
        axes[1].axhline(0, color="k", lw=0.6)
        axes[1].axvline(0, color="k", lw=0.6)
        axes[1].set_xlim(-lim, lim)
        axes[1].set_ylim(-lim, lim)
        axes[1].set_xlabel("deviation from common mode, 60->90 (%)")
        axes[1].set_ylabel("deviation from common mode, 90->120 (%)")
        axes[1].set_title("Per-sample differential: does it persist "
                          "or shrink?")
        fig.tight_layout()
        p = os.path.join(DIAG_DIR, "ladder.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[out] wrote {p}")
    except Exception as e:                            # noqa: BLE001
        print(f"[warn] figure skipped: {e}")


def main(argv=None):
    args = parse_args(argv)
    tee("ladder_referee", DIAG_DIR)
    device = F.resolve_device(args.device)
    print(f"[ladder] bank={OUT_DIR}  rungs={RUNGS}  sids={args.sids}  "
          f"device={device}")
    print(f"[ladder] numerics: decay_tol={C.DECAY_TOL:g}, "
          f"cap={C.MAX_TIME:g} (res<=90) / {C.MAX_TIME * 1.5:g} (res 120), "
          f"campaign wavelength grid (fixed across rungs)")

    samples = _load_samples()
    samples_by_sid = {int(x["sample_id"]): x for x in samples}
    missing = [s for s in args.sids if s not in samples_by_sid]
    if missing:
        raise SystemExit(f"sids not found in bank {OUT_DIR}: {missing}")

    E = gather_E(args, samples_by_sid, device)
    analyze(args, E, samples_by_sid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
