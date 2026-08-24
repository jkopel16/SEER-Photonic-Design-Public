"""
check_rank_fidelity_v2.py -- T6 extended: within-sigma ranking floor at N~15.

The original Test 6 (check_rank_fidelity.py) measured +-0.23% disordered
label jitter and a ~0.5% ranking floor from 5 samples / 10 pairs. Five
samples is a thin base for a spread estimate that downstream claims
(GA champion gate, tier count) now lean on. This re-runs the measurement
at N=15 (105 pairs), with:

  1. PER-SAMPLE CHECKPOINTING. Each res-90 solve is cached to
     gpu_out_diag/rank90_cache/rank90_sid<sid>.npz immediately after it
     finishes, with the numerics settings stored as guard keys. A killed
     or preempted run resumes from the cache -- re-launching the script
     skips every already-solved sample. At ~80 min/sample this run is
     ~20 h; assume it WILL be interrupted.
  2. Upgraded statistics for the larger N: Spearman rho between res-60
     and res-90 labels, bootstrap CI on the shift spread, and an
     explicit kept/flipped separation check (is there clean daylight
     between the largest flipped gap and the smallest kept gap, as in
     the N=5 run, or do they now overlap?).

Method is otherwise identical to Test 6: banked jitter sigma=0.10
samples picked evenly across the class E range, re-solved at res 90
with both polarizations, all pairwise orderings compared.

Usage (on scc, inside tmux -- see launch commands):
    PC_RESOLUTION=90 PC_RANK_N=15 python -u check_rank_fidelity_v2.py

Env knobs (all optional):
    PC_RANK_SIGMA   sigma class to test          (default 0.10)
    PC_RANK_N       number of samples            (default 15)
    PC_BANK         banked-sample directory      (default <solver dir>/data_production)
    PC_RANK_SEED    bootstrap RNG seed           (default 0)

SUPERSEDED (audit, final 2026-07-26): this run's 0.8-1% floor was a
res-90 artifact.  ladder_referee.py (Test 8) showed res 90 sits near a
band-1 grid-dispersion error sign crossing and is not a converged
referee; rank_fidelity_r120.py (Test 9) re-judged the same N=15 panel at
res 120 and found ZERO flips above 0.30%.  The current floor is 0.30%.
The res-90 E values logged by this script survive only as
ladder_referee.LOGGED_RES90 (kept to put the old referee on trial).
"""

from __future__ import annotations

import glob
import os
import sys
import time
import numpy as np


# solver modules live one directory up from tests/
_SOLVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SOLVER_DIR)

import config as C
from logutil import tee
from materials_gpu import fit_all
from optics_core import solar_weight, planar_reference_stack
import fdtd_torch as F

assert C.RESOLUTION == 90, "run with PC_RESOLUTION=90"

SIGMA_TARGET = float(os.environ.get("PC_RANK_SIGMA", "0.10"))
N_PICK = int(os.environ.get("PC_RANK_N", "15"))
BANK = os.environ.get("PC_BANK",
                      os.path.join(_SOLVER_DIR, "data_production"))
BOOT_SEED = int(os.environ.get("PC_RANK_SEED", "0"))

# Diagnostics land in tests/gpu_out_diag regardless of CWD (PC_DIAG overrides).
DIAG_DIR = os.environ.get("PC_DIAG") or os.path.join(
    _SOLVER_DIR, "tests", "gpu_out_diag")
NORM_DIR = os.path.join(DIAG_DIR, "norm_cache_rank90")
CACHE_DIR = os.path.join(DIAG_DIR, "rank90_cache")

# Guard keys: a cached solve is only reused if it was produced with the
# numerics we are running now. Anything else is silently stale physics.
GUARD = dict(resolution=C.RESOLUTION, decay_tol=C.DECAY_TOL,
             max_time=C.MAX_TIME, thickness=C.THICKNESS_NM,
             buffer=C.BUFFER_NM)


def cache_path(sid: int) -> str:
    return os.path.join(CACHE_DIR, f"rank90_sid{sid}.npz")


def cache_load(sid: int):
    """Return (E90, cap, runtime) from cache, or None if absent/stale."""
    p = cache_path(sid)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    for k, v in GUARD.items():
        if k not in z.files or not np.isclose(float(z[k]), float(v)):
            print(f"    [cache] sid {sid}: stale ({k} mismatch) -- "
                  f"re-solving")
            return None
    return float(z["E90"]), bool(z["cap"]), float(z["runtime_s"])


def cache_save(sid: int, E90: float, cap: bool, runtime_s: float,
               A_si: np.ndarray, wl: np.ndarray) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = cache_path(sid) + ".tmp.npz"
    np.savez(tmp, E90=E90, cap=cap, runtime_s=runtime_s,
             A_si=A_si, wavelength_nm=wl,
             **{k: float(v) for k, v in GUARD.items()})
    os.replace(tmp, cache_path(sid))   # atomic: no half-written caches


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() /
                 np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def main():
    tee("check_rank_fidelity_v2", DIAG_DIR)
    device = F.resolve_device(os.environ.get("PC_DEVICE", "auto"))
    print(C.describe())
    print(f"[device] {device}   [bank] {BANK}   sigma={SIGMA_TARGET}   "
          f"N={N_PICK}")
    print(f"[cache] {CACHE_DIR}  (per-sample checkpointing: kill/resume "
          f"is safe)\n")

    # ---- pick banked samples spanning the E range ------------------------
    cand = []
    for p in sorted(glob.glob(os.path.join(BANK, "samples",
                                           "sample_*.npz"))):
        z = np.load(p, allow_pickle=False)
        if (str(z["disorder_class"]) == "jitter"
                and abs(float(z["sigma"]) - SIGMA_TARGET) < 1e-9):
            cand.append((float(z["E"]), int(z["sample_id"]), p))
    if len(cand) < N_PICK:
        raise SystemExit(f"only {len(cand)} banked jitter sigma="
                         f"{SIGMA_TARGET} samples; need {N_PICK}")
    cand.sort()
    idx = np.unique(np.linspace(0, len(cand) - 1,
                                N_PICK).round().astype(int))
    picks = [cand[i] for i in idx]
    if len(picks) < N_PICK:
        print(f"[note] class only supports {len(picks)} distinct "
              f"evenly-spaced picks")
    print(f"[picked {len(picks)} of {len(cand)} banked samples, "
          f"spanning E {picks[0][0]:.4f} .. {picks[-1][0]:.4f} "
          f"({100*(picks[-1][0]/picks[0][0]-1):.2f}% spread)]")
    for E60, sid, _ in picks:
        print(f"    sample {sid:6d}  E(res60) = {E60:.4f}")

    # ---- planar denominator (analytic, exact) ----------------------------
    fits, ad, mats = fit_all()
    wl = C.raw_wavelength_grid()
    w = solar_weight(wl)
    ref = planar_reference_stack(ad["si"], ad["zno"], ad["ag"], wl,
                                 C.THICKNESS_NM, C.BUFFER_NM)
    eta_p = float(np.sum(w * np.asarray(ref["A_si"], float)))

    # ---- solve (or load) each sample at res 90 ---------------------------
    n_cached = sum(cache_load(sid) is not None for _, sid, _ in picks)
    print(f"\n[{n_cached}/{len(picks)} already cached; "
          f"{len(picks)-n_cached} to solve, ~80 min each at res 90]")
    print(f"\n{'sid':>6} {'E res60':>9} {'E res90':>9} {'shift%':>8} "
          f"{'cap':>5} {'runtime':>9}  src")
    print("-" * 62)

    rows = []
    for E60, sid, path in picks:
        hit = cache_load(sid)
        if hit is not None:
            E90, cap, rt = hit
            src = "cache"
        else:
            z = np.load(path, allow_pickle=False)
            holes = [tuple(h) for h in np.asarray(z["holes_xyr_nm"])]
            a_sup = float(z["a_super_nm"])
            t0 = time.time()
            A_si, A_par, info = F.broadband_absorption_many(
                [holes], a_sup, C.THICKNESS_NM, wl, fits, C.BUFFER_NM,
                C.RESOLUTION, C.DECAY_TOL, C.MAX_TIME, NORM_DIR,
                device=device, n_cells_tag=f"rank{sid}")
            A = np.asarray(A_si[0], float)
            E90 = float(np.sum(w * np.clip(A, 0, 1))) / eta_p
            cap = bool(np.any(info["hit_time_cap"]))
            rt = float(np.sum(info["runtime_s"]))
            cache_save(sid, E90, cap, rt, A, wl)
            src = f"solved ({time.time()-t0:.0f}s wall)"
        rows.append((sid, E60, E90, cap))
        print(f"{sid:6d} {E60:9.4f} {E90:9.4f} "
              f"{100*(E90/E60-1):8.2f} {str(cap):>5} {rt:8.0f}s  {src}",
              flush=True)

    capped = [sid for sid, _, _, cap in rows if cap]
    if capped:
        print(f"\n[WARNING] time-capped samples: {capped} -- their E90 "
              f"is truncated;\n  interpret their pairs with suspicion or "
              f"re-run with a higher MAX_TIME.")
    rows = [(sid, E60, E90) for sid, E60, E90, _ in rows]

    # ---- pairwise ordering analysis --------------------------------------
    print("\n[pairwise orderings: does res-90 agree with res-60?]")
    kept, flipped = [], []
    flip_lines = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            s1, a60, a90 = rows[i]
            s2, b60, b90 = rows[j]
            d60 = 100 * (b60 / a60 - 1)
            d90 = 100 * (b90 / a90 - 1)
            same = (d60 > 0) == (d90 > 0)
            (kept if same else flipped).append(abs(d60))
            if not same:
                flip_lines.append(f"    {s1} vs {s2}: dE60 {d60:+.2f}%  "
                                  f"dE90 {d90:+.2f}%")
    n_tot = len(kept) + len(flipped)
    print(f"  {n_tot} pairs total; {len(kept)} kept, "
          f"{len(flipped)} flipped")
    if flip_lines:
        print("  flipped pairs:")
        print("\n".join(flip_lines))

    # ---- statistics ------------------------------------------------------
    E60s = np.array([r[1] for r in rows])
    E90s = np.array([r[2] for r in rows])
    shifts = 100 * (E90s / E60s - 1)
    rho = spearman(E60s, E90s)

    rng = np.random.default_rng(BOOT_SEED)
    boot = np.array([shifts[rng.integers(0, len(shifts),
                                         len(shifts))].std()
                     for _ in range(10_000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"\n[summary]  N={len(rows)} samples, {n_tot} pairs")
    print(f"  res60->90 shift    : mean {shifts.mean():+.2f}%   "
          f"spread {shifts.std():.2f}%")
    print(f"  spread 95% CI      : [{lo:.2f}%, {hi:.2f}%]  "
          f"(bootstrap, 10k resamples)")
    print(f"  Spearman(E60, E90) : {rho:.3f}")
    print(f"  N=5 reference      : mean -5.00%, spread 0.23%, "
          f"floor ~0.5%")

    if flipped:
        f_max, k_min = max(flipped), (min(kept) if kept else float("nan"))
        print(f"\n  largest FLIPPED gap : {f_max:.2f}%")
        print(f"  smallest KEPT gap   : {k_min:.2f}%")
        if kept and k_min > f_max:
            print(f"  separation          : CLEAN (daylight of "
                  f"{k_min - f_max:.2f}pp between them)")
            print(f"\n  VERDICT: ranking floor ~{f_max:.1f}%. Orderings "
                  f"above it are real\n  physics; below it, staircase "
                  f"lottery. NOTE: this referee is res 90,\n  later "
                  f"convicted as unconverged (Test 8). Do NOT gate the GA "
                  f"on this\n  number -- see rank_fidelity_r120.py "
                  f"(floor 0.30%).")
        else:
            over = sorted(x for x in kept if x < f_max)
            print(f"  separation          : OVERLAPPING -- "
                  f"{len(over)} kept pair(s) sit below the\n"
                  f"    largest flipped gap ({', '.join(f'{x:.2f}%' for x in over)})")
            print(f"\n  VERDICT: the floor is a soft transition, not a "
                  f"sharp edge -- the N=5\n  'perfect separation' was "
                  f"partly small-sample luck. SUPERSEDED:\n  "
                  f"rank_fidelity_r120.py re-judged this same panel at "
                  f"res 120 and found\n  zero flips above 0.30%; most of "
                  f"these 'flips' were the res-90\n  referee's own error. "
                  f"Quote 0.30%, not {f_max:.1f}%.")
    else:
        print(f"\n  VERDICT: ALL {n_tot} ORDERINGS PRESERVED down to the "
              f"smallest gap\n  tested ({min(kept):.2f}%). The floor is "
              f"at or below that gap;\n  the N=5 floor of ~0.5% was "
              f"conservative.")


if __name__ == "__main__":
    main()
