"""
run_dataset.py
--------------
FULL ML-DATASET GENERATION on GPU FDTD instead of Meep. Same deterministic
manifest (sample_id -> class, sigma, seed), same wavelength grid, same
fitted materials, same per-sample npz schema, same figures.  Differences
are purely operational: the solver is fdtd_torch (one GPU, sequential),
each sample is banked to disk the moment it is solved (maximal
resumability for preemptable SCC windows), and sharding is by GPU
(SGE task arrays auto-detected) instead of by CPU process pool.

Subcommands
-----------
  generate   solve everything in the manifest that is not yet banked
             (default subcommand; safe to re-run / resume at any time)
                --shard i/N     solve sample_id %% N == i (auto from
                                SGE_TASK_ID when inside a task array)
                --limit K       stop this invocation after K samples
                --skip-validation   (only inside arrays whose task 0
                                     already gated this device)
  plan       print the manifest summary and exit
  analyze    assemble dataset.npz + labels.csv + figs 5-12
  verify     re-solve the top layouts at elevated numerics -> fig 13
                --top K (default 6)

Scoping the campaign: SEEDS_PER_SIGMA below (or env PC_SEEDS_PER_SIGMA)
is THE cost lever -- set it from the measured ETA table in
timing_report.txt before launching.

Everything persists under PC_OUT: samples/, quarantine/, norm_cache/,
figs/, logs/, dataset.npz, labels.csv, verified.csv, meta.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import numpy as np

import config as C
from logutil import tee
from materials_gpu import fit_all
from optics_core import planar_reference_stack, solar_weight, enhancement
from disorder import make_layout, holes_array
import fdtd_torch as F
from fdtd_torch import broadband_absorption_many, rasterize_mask, \
    run_all_validations

# --------------------------------------------------------------------------
# The experimental grid (identical to the Meep stage's run_campaign.py)
# --------------------------------------------------------------------------
# PC_SIGMAS / PC_CLASSES override the grid for EXTENSION campaigns into a
# fresh PC_OUT directory. Never change the grid of an existing bank: the
# manifest numbers ids class-by-sigma-by-seed, so a different grid
# renumbers every subsequent id and corrupts the id -> cell mapping.
SIGMAS = tuple(float(s) for s in os.environ["PC_SIGMAS"].split(",")) \
    if os.environ.get("PC_SIGMAS") else \
    (0.02, 0.04, 0.06, 0.08, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30)
CLASSES = tuple(os.environ["PC_CLASSES"].split(",")) \
    if os.environ.get("PC_CLASSES") else ("jitter", "radius")
SEEDS_PER_SIGMA = int(os.environ.get("PC_SEEDS_PER_SIGMA", "75"))
N_RANDOM = int(os.environ.get("PC_N_RANDOM", "50"))
# extension block: extra seeds per (class, sigma) cell APPENDED to the
# manifest after the random block. Original ids 0..1550 never move, so
# banked samples stay valid. Locked by meta.json once set.
EXTRA_SEEDS = int(os.environ.get("PC_EXTRA_SEEDS", "0"))

OUT_DIR = C.OUT_DIR
SAMPLES_DIR = os.path.join(OUT_DIR, "samples")
QUAR_DIR = os.path.join(OUT_DIR, "quarantine")
FIGS_DIR = os.path.join(OUT_DIR, "figs")
NORM_DIR = os.path.join(OUT_DIR, "norm_cache")

CLASS_COLORS = {"jitter": "#1f5fa8", "radius": "#c0392b",
                "random": "#e67e22", "ordered": "#27632a"}

_MATS = {}


def get_materials():
    if not _MATS:
        fits, adapters, (si, zno, ag) = fit_all()
        _MATS.update(si=si, zno=zno, ag=ag, fits=fits, adapters=adapters)
    return _MATS


# --------------------------------------------------------------------------
# Shared plumbing (ported unchanged where possible)
# --------------------------------------------------------------------------
def campaign_grid():
    return C.raw_wavelength_grid()


def band_eta(A, wl, weights, lo=700.0, hi=1100.0):
    m = (wl >= lo) & (wl <= hi)
    w = weights[m]
    return float(np.sum(w * A[m]) / np.sum(w))


def numerics_dict(wl):
    return {"mode": C.MODE, "n_cells": C.N_CELLS, "a_nm": C.A_NM,
            "thickness_nm": C.THICKNESS_NM, "r_over_a": C.R_OVER_A,
            "w_min_nm": C.W_MIN_NM, "buffer_nm": C.BUFFER_NM,
            "solver": "torch-fdtd-gpu", "engine": F.ENGINE_VERSION,
            "resolution": C.RESOLUTION,
            "decay_tol": C.DECAY_TOL, "max_time": C.MAX_TIME,
            "bands": 3,
            "n_wl": int(len(wl)), "wl_sum": float(np.sum(wl)),
            "vis_step": C.VIS_STEP_NM, "nir_step": C.NIR_STEP_NM,
            "base_seed": C.BASE_SEED,
            "seeds_per_sigma": SEEDS_PER_SIGMA, "n_random": N_RANDOM,
            "extra_seeds": EXTRA_SEEDS,
            "sigmas": list(SIGMAS), "classes": list(CLASSES)}


def check_meta(wl):
    """Create-or-verify the campaign meta record (drifting-numerics
    guard): any mismatch aborts rather than silently mixing datasets."""
    os.makedirs(OUT_DIR, exist_ok=True)
    meta_path = os.path.join(OUT_DIR, "meta.json")
    now = numerics_dict(wl)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            old = json.load(f)
        # keys absent from an older meta.json (sigmas/classes were added
        # after the production bank was built) are grandfathered, not diffs
        diffs = {k: (old[k], v) for k, v in now.items()
                 if k in old and old[k] != v}
        if diffs:
            raise SystemExit(
                f"ABORT: {OUT_DIR}/meta.json was written with different "
                f"numerics: {diffs}.\nEither restore the original settings "
                f"(PC_MODE / config.py / PC_SEEDS_PER_SIGMA) or point "
                f"PC_OUT at a fresh directory.")
    else:
        with open(meta_path, "w") as f:
            json.dump(now, f, indent=1)
    return now


def build_plan():
    """Deterministic manifest: sample_id -> (class, sigma, seed).
    radius/sigma=0.02 is excluded (sub-pixel at res 60)
    but its ids are still counted so no other id moves.
    EXTRA_SEEDS appends seeds k = SEEDS_PER_SIGMA.. per cell AFTER the
    random block: original ids 0..(base manifest) never move."""
    plan = [{"sample_id": 0, "class": "ordered", "sigma": 0.0,
             "seed": C.BASE_SEED}]
    sid = 1
    for c_idx, cls in enumerate(CLASSES):
        for s_idx, sigma in enumerate(SIGMAS):
            for k in range(SEEDS_PER_SIGMA):
                seed = C.BASE_SEED + (c_idx + 1) * 1_000_000 \
                    + s_idx * 10_000 + k
                if not (cls == "radius" and abs(sigma - 0.02) < 1e-9):
                    plan.append({"sample_id": sid, "class": cls,
                                 "sigma": float(sigma), "seed": seed})
                sid += 1
    for k in range(N_RANDOM):
        plan.append({"sample_id": sid, "class": "random",
                     "sigma": float("nan"),
                     "seed": C.BASE_SEED + 3_000_000 + k})
        sid += 1
    # ---- extension block (ids continue past the random block) ----------
    for c_idx, cls in enumerate(CLASSES):
        for s_idx, sigma in enumerate(SIGMAS):
            for k in range(SEEDS_PER_SIGMA, SEEDS_PER_SIGMA + EXTRA_SEEDS):
                seed = C.BASE_SEED + (c_idx + 1) * 1_000_000 \
                    + s_idx * 10_000 + k
                if not (cls == "radius" and abs(sigma - 0.02) < 1e-9):
                    plan.append({"sample_id": sid, "class": cls,
                                 "sigma": float(sigma), "seed": seed})
                sid += 1
    return plan


def sample_path(sid):
    return os.path.join(SAMPLES_DIR, f"sample_{sid:06d}.npz")


def make_record(row):
    return make_layout(row["class"], row["sigma"], row["seed"],
                       a_nm=C.A_NM, n_cells=C.N_CELLS,
                       r_nm=C.R_OVER_A * C.A_NM, w_min_nm=C.W_MIN_NM)


def gate_sample(A_si, A_par):
    """Automated sanity gates (methodology Sec 1.3 step 4).

    TOL note: a few-tenths-of-a-percent negative dip in A is FDTD
    discretization noise at sharp reflection nulls (finite-grid R rounding
    just above 1), not unphysical absorption -- it does not affect the
    solar-weighted E label.  The bound tolerance is 1e-2 (1 %; raised
    from 5e-3 per audit Test 5, Group A -- the stricter gate quarantined
    three healthy samples on sub-tolerance noise dips); genuine
    failures (numerical blow-up, real energy-conservation violation)
    overshoot this by orders of magnitude, so they are still caught.
    Callers clamp the stored spectra to [0,1] after this passes."""
    TOL = 1e-2
    if np.isnan(A_si).any() or np.isnan(A_par).any():
        return f"{int(np.isnan(A_si).sum())} NaN wavelengths"
    if A_si.min() < -TOL or A_si.max() > 1.0 + TOL:
        return f"A_si out of [0,1]: [{A_si.min():.3g},{A_si.max():.3g}]"
    if A_par.min() < -TOL or A_par.max() > 1.0 + TOL:
        return f"A_par out of [0,1]: [{A_par.min():.3g},{A_par.max():.3g}]"
    if (A_si + A_par).max() > 1.0 + TOL:
        return "A_si + A_par exceeds 1 (R < 0 somewhere)"
    return None


def solve_one(rec, wl, device, resolution=None, decay_tol=None,
              max_time=None):
    m = get_materials()
    holes = [tuple(h) for h in holes_array(rec)]
    return broadband_absorption_many(
        [holes], rec.get("a_super_nm", C.A_SUPER_NM), C.THICKNESS_NM, wl,
        m["fits"], C.BUFFER_NM, resolution or C.RESOLUTION,
        decay_tol or C.DECAY_TOL, max_time or C.MAX_TIME, NORM_DIR,
        device=device, n_cells_tag=f"sc{C.N_CELLS}")


def _detect_shard(args):
    """--shard i/N wins; else SGE task arrays are auto-detected
    (SGE_TASK_ID is 1-indexed -> shard task_id-1 of SGE_TASK_LAST)."""
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        return i, n
    tid = os.environ.get("SGE_TASK_ID", "")
    last = os.environ.get("SGE_TASK_LAST", "")
    if tid.isdigit() and last.isdigit() and int(last) >= 1:
        return int(tid) - 1, int(last)
    return 0, 1


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------
def cmd_generate(args):
    shard_i, shard_n = _detect_shard(args)
    tee(f"dataset_shard{shard_i}", OUT_DIR)
    print("=" * 64)
    print(f"DATASET GENERATION (GPU FDTD)  [{C.describe()}]")
    print(f"grid: {len(CLASSES)} classes x {len(SIGMAS)} sigmas x "
          f"{SEEDS_PER_SIGMA} seeds + {N_RANDOM} random + ordered")
    print("=" * 64)
    device = F.resolve_device(os.environ.get("PC_DEVICE", "auto"))
    print(f"[device] {device}")
    if device != "cuda" and not F.FAKE:
        print("  WARNING: running on CPU -- fine for FAKE/SMOKE plumbing "
              "tests, hopeless for FULL.")
    m = get_materials()
    wl = campaign_grid()
    check_meta(wl)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(QUAR_DIR, exist_ok=True)

    if not args.skip_validation:
        print("\n[pre-flight validation]")
        marker = os.path.join(OUT_DIR, ".validated_3d")
        ok = run_all_validations(
            m["fits"], m["adapters"], (m["si"], m["zno"], m["ag"]),
            C.THICKNESS_NM, C.BUFFER_NM, C.RESOLUTION_UNIT, C.DECAY_TOL,
            C.MAX_TIME, NORM_DIR, a_super_3d_nm=C.A_NM, device=device)
        if not ok:
            raise SystemExit("  ABORT: validation failed -- do not bank "
                             "labels from an engine in this state.")
        with open(marker, "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M UTC\n", time.gmtime()))

    weights = solar_weight(wl)
    ad = m["adapters"]
    # planar reference from the FITTED materials: numerator and
    # denominator of E share identical optics.
    ref = planar_reference_stack(ad["si"], ad["zno"], ad["ag"], wl,
                                 C.THICKNESS_NM, C.BUFFER_NM)

    plan = build_plan()
    quarantined = {int(os.path.basename(p)[7:13]) for p in
                   glob.glob(os.path.join(QUAR_DIR, "sample_*.npz"))}
    todo = [p for p in plan
            if p["sample_id"] % shard_n == shard_i
            and p["sample_id"] not in quarantined
            and not os.path.exists(sample_path(p["sample_id"]))]
    n_skip_q = sum(1 for p in plan
                   if p["sample_id"] % shard_n == shard_i
                   and p["sample_id"] in quarantined)
    if n_skip_q:
        print(f"  NOTE: skipping {n_skip_q} previously QUARANTINED samples"
              f" in this shard;\n  inspect {QUAR_DIR} and delete their "
              f"files to force a re-solve.")
    if args.limit:
        todo = todo[:args.limit]
    print(f"\nshard {shard_i}/{shard_n}: {len(todo)} samples to solve, "
          f"one at a time on {device}")
    if not todo:
        print("nothing to do -- shard complete.")
        return

    numerics_str = json.dumps(numerics_dict(wl))
    t0 = time.time()
    for i, row in enumerate(todo):
        rec = make_record(row)
        A_si, A_par, info = solve_one(rec, wl, device)
        a_si, a_par = A_si[0], A_par[0]
        # clamp sub-tolerance discretization noise (negative dips at sharp
        # reflection nulls) into [0,1]; genuine failures are far larger and
        # still fail the gate below on the pre-clamp magnitude
        a_si = np.clip(np.nan_to_num(a_si), 0.0, 1.0)
        a_par = np.clip(np.nan_to_num(a_par), 0.0, 1.0)
        sid = row["sample_id"]
        reason = gate_sample(A_si[0], A_par[0])
        E, eta, _ = enhancement(np.nan_to_num(a_si), ref["A_si"], weights)
        payload = dict(
            sample_id=sid, disorder_class=rec["class"],
            sigma=rec["sigma"], seed=rec["seed"],
            holes_xyr_nm=holes_array(rec),
            a_super_nm=rec["a_super_nm"],
            thickness_nm=C.THICKNESS_NM, buffer_nm=C.BUFFER_NM,
            wavelengths_nm=wl,
            A_si=a_si.astype(np.float32),
            A_par=a_par.astype(np.float32),
            E=E, eta=eta,
            E_nir=(band_eta(np.nan_to_num(a_si), wl, weights)
                   / band_eta(ref["A_si"], wl, weights)),
            parasitic_frac=float(np.sum(weights * np.nan_to_num(a_par))),
            fill_achieved=rec["fill_achieved"],
            radius_scale=rec["radius_scale"],
            n_redraws=rec["n_redraws"], n_restarts=rec["n_restarts"],
            generator=rec["generator"], numerics=numerics_str,
            resolution=C.RESOLUTION, decay_tol=C.DECAY_TOL,
            device=str(info.get("device", device)),
            runtime_s=float(info["runtime_s"][0]),
            hit_time_cap=bool(info["hit_time_cap"][0]))
        if bool(info["hit_time_cap"][0]) and reason is None:
            reason = ("hit the ring-down time cap (spectra possibly "
                      "Q-truncated) -- raise config.MAX_TIME")
        if reason is None:
            np.savez_compressed(sample_path(sid), **payload)
        else:
            payload["quarantine_reason"] = reason
            np.savez_compressed(
                os.path.join(QUAR_DIR, f"sample_{sid:06d}.npz"), **payload)
            print(f"    QUARANTINED sample {sid} "
                  f"({rec['class']}, sigma={rec['sigma']}): {reason}")
        rate = (time.time() - t0) / (i + 1)
        eta_h = (len(todo) - i - 1) * rate / 3600.0
        print(f"  [{i + 1:5d}/{len(todo)}] sample {sid:6d} "
              f"({rec['class']:7s} s={rec['sigma']:.3g}) "
              f"E={E:.3f}  {info['runtime_s'][0]:6.1f}s  "
              f"->  shard ETA {eta_h:.1f} h", flush=True)
    print("shard complete.")


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------
def cmd_plan(args):
    plan = build_plan()
    n_done = len(glob.glob(os.path.join(SAMPLES_DIR, "sample_*.npz")))
    n_quar = len(glob.glob(os.path.join(QUAR_DIR, "sample_*.npz")))
    print(f"manifest: {len(plan)} samples  "
          f"({len(CLASSES)} classes x {len(SIGMAS)} sigmas x "
          f"{SEEDS_PER_SIGMA} seeds + {N_RANDOM} random + 1 ordered)")
    print(f"banked:   {n_done}   quarantined: {n_quar}   "
          f"remaining: {len(plan) - n_done - n_quar}")
    print(f"output:   {os.path.abspath(OUT_DIR)}")


# --------------------------------------------------------------------------
# analyze  (figures ported from the Meep stage nearly verbatim)
# --------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _load_samples():
    files = sorted(glob.glob(os.path.join(SAMPLES_DIR, "sample_*.npz")))
    if not files:
        raise SystemExit(f"no banked samples in {SAMPLES_DIR} -- run "
                         "generate first.")
    out = []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            out.append({k: z[k] for k in z.files})
    return out


def _sigma_stats(samples, cls):
    stats = {}
    for s in SIGMAS:
        Es = np.array([float(x["E"]) for x in samples
                       if str(x["disorder_class"]) == cls
                       and abs(float(x["sigma"]) - s) < 1e-9])
        if len(Es):
            stats[s] = {"mean": Es.mean(), "std": Es.std(ddof=1)
                        if len(Es) > 1 else 0.0,
                        "sem": (Es.std(ddof=1) / np.sqrt(len(Es)))
                        if len(Es) > 1 else 0.0, "n": len(Es)}
    return stats


def fig_E_vs_sigma(samples, E_ord, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    # zoom the view to the disorder curves around the ordered-lattice
    # baseline: collect every drawn extent (means +- 1 std bands, random
    # band, E_ord) and set explicit y-limits from those, so the planar
    # E = 1 reference far below no longer squashes the interesting range.
    yvals = [E_ord]
    for cls in CLASSES:
        st = _sigma_stats(samples, cls)
        if not st:
            continue
        sig = np.array([0.0] + [s for s in SIGMAS if s in st])
        mean = np.array([E_ord] + [st[s]["mean"] for s in SIGMAS
                                   if s in st])
        sem = np.array([0.0] + [st[s]["sem"] for s in SIGMAS if s in st])
        std = np.array([0.0] + [st[s]["std"] for s in SIGMAS if s in st])
        c = CLASS_COLORS[cls]
        ax.errorbar(sig, mean, yerr=sem, fmt="o-", color=c, lw=2, ms=5,
                    capsize=3,
                    label=f"{cls} disorder (mean $\\pm$ SEM, band "
                          f"$\\pm$1 std)")
        # the +-1 std band is context, not the message: very light, no
        # edge, and behind every line so it never competes with the means
        ax.fill_between(sig, mean - std, mean + std, color=c, alpha=0.055,
                        lw=0, zorder=0)
        yvals += [float((mean - std).min()), float((mean + std).max())]
        i_star = int(np.argmax(mean))
        ax.annotate(f"$\\sigma^*\\approx${sig[i_star]:.2f}",
                    (sig[i_star], mean[i_star]),
                    textcoords="offset points", xytext=(0, 10),
                    color=c, fontsize=9, ha="center")
    E_rand = [float(x["E"]) for x in samples
              if str(x["disorder_class"]) == "random"]
    if E_rand:
        mn, sd = np.mean(E_rand), np.std(E_rand)
        ax.axhline(mn, color=CLASS_COLORS["random"], ls="-.", lw=1.5,
                   label=f"fully random (mean $\\pm$1 std, n={len(E_rand)})")
        ax.axhspan(mn - sd, mn + sd, color=CLASS_COLORS["random"],
                   alpha=0.05, lw=0, zorder=0)
        yvals += [mn - sd, mn + sd]
    ax.axhline(E_ord, color=CLASS_COLORS["ordered"], ls="--", lw=1.5,
               label=f"ordered lattice (E = {E_ord:.3f})")
    lo, hi = min(yvals), max(yvals)
    pad = 0.05 * (hi - lo + 1e-9)
    lo, hi = lo - pad, hi + 3 * pad
    ax.set_ylim(lo, hi)
    in_view = lo <= 1.0 <= hi
    ax.axhline(1.0, color="grey", ls=":", lw=1,
               label="planar film (E = 1)" if in_view
               else "planar film (E = 1, below view)")
    ax.annotate(f"ordered lattice  E = {E_ord:.3f}",
                (1.0, E_ord), xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(-4, 4),
                color=CLASS_COLORS["ordered"], fontsize=9, ha="right")
    ax.set_xlabel("Disorder strength  $\\sigma$")
    ax.set_ylabel("Broadband enhancement factor  E")
    ax.set_title("Enhancement vs disorder strength "
                 f"({C.N_CELLS}x{C.N_CELLS} supercell, reflector stack, "
                 "normal incidence, GPU FDTD)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_scatter(samples, E_ord, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    rng = np.random.default_rng(0)
    for cls in CLASSES + ("random",):
        xs, ys = [], []
        for x in samples:
            if str(x["disorder_class"]) != cls:
                continue
            s = float(x["sigma"])
            xs.append((SIGMAS[-1] + 0.06) if cls == "random" else s)
            ys.append(float(x["E"]))
        if xs:
            xs = np.array(xs) + rng.uniform(-0.004, 0.004, len(xs))
            ax.plot(xs, ys, "o", ms=4, alpha=0.45,
                    color=CLASS_COLORS[cls], label=cls)
    ax.axhline(E_ord, color=CLASS_COLORS["ordered"], ls="--", lw=1.5,
               label="ordered lattice")
    ax.axvline(SIGMAS[-1] + 0.03, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("Disorder strength  $\\sigma$")
    ax.set_ylabel("Enhancement factor  E  (individual realizations)")
    ax.set_title("Same $\\sigma$, different layouts, different E\n"
                 "the within-$\\sigma$ spread is what the CNN will resolve")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_spectra_showcase(samples, ord_sample, ref, wl, path):
    plt = _plt()
    dis = [x for x in samples if str(x["disorder_class"]) != "ordered"]
    dis.sort(key=lambda x: float(x["E"]))
    picks = [("champion", dis[-1]), ("median", dis[len(dis) // 2]),
             ("worst", dis[0])]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    for ax, full in zip(axes, (True, False)):
        m = np.ones_like(wl, bool) if full else (wl >= C.NIR_SPLIT)
        ax.plot(wl[m], np.asarray(ref["A_si"])[m], ls="--",
                color="#7f8c8d", lw=1.5, label="planar reference")
        ax.plot(wl[m], np.asarray(ord_sample["A_si"])[m], lw=1.4,
                color=CLASS_COLORS["ordered"],
                label=f"ordered (E={float(ord_sample['E']):.3f})")
        for tag, x in picks:
            lab = (f"{tag}: {x['disorder_class']} "
                   f"$\\sigma$={float(x['sigma']):.3g} "
                   f"(E={float(x['E']):.3f})")
            ax.plot(wl[m], np.asarray(x["A_si"])[m], lw=1.1, alpha=0.9,
                    label=lab)
        ax.set_ylabel("A$_{Si}$")
        ax.legend(fontsize=8)
    axes[0].set_title("Where the disorder gain/loss lives in the spectrum")
    axes[1].set_title("NIR zoom (the light-trapping band)")
    axes[1].set_xlabel("Wavelength (nm)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return picks


def fig_gallery(samples, ord_sample, picks, path):
    plt = _plt()
    shown = [("ordered", ord_sample)] + picks
    rand = [x for x in samples if str(x["disorder_class"]) == "random"]
    if rand:
        shown.append(("random (best)",
                      max(rand, key=lambda x: float(x["E"]))))
    fig, axes = plt.subplots(1, len(shown),
                             figsize=(3.2 * len(shown), 3.6))
    for ax, (tag, x) in zip(np.atleast_1d(axes), shown):
        holes = [tuple(h) for h in np.asarray(x["holes_xyr_nm"])]
        img = rasterize_mask(holes, float(x["a_super_nm"]), 256, 256, 2)
        ax.imshow(img.T, origin="lower", cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{tag}\nE={float(x['E']):.3f}", fontsize=9)
    fig.suptitle("Layout gallery (silicon = white, holes = black)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_nir_and_parasitic(samples, ord_sample, path_nir, path_par):
    plt = _plt()
    for key, ylab, path, ref_val in (
            ("E_nir", "NIR-band enhancement  "
             "$\\eta_{700-1100}/\\eta^{planar}_{700-1100}$", path_nir,
             float(ord_sample["E_nir"])),
            ("parasitic_frac", "AM1.5G-weighted parasitic Ag loss",
             path_par, float(ord_sample["parasitic_frac"]))):
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for cls in CLASSES:
            xs, ms, sds = [], [], []
            for s in SIGMAS:
                v = [float(x[key]) for x in samples
                     if str(x["disorder_class"]) == cls
                     and abs(float(x["sigma"]) - s) < 1e-9]
                if v:
                    xs.append(s)
                    ms.append(np.mean(v))
                    sds.append(np.std(v))
            ax.errorbar(xs, ms, yerr=sds, fmt="o-", ms=4, capsize=3,
                        color=CLASS_COLORS[cls], label=cls)
        ax.axhline(ref_val, color=CLASS_COLORS["ordered"], ls="--",
                   label="ordered")
        ax.set_xlabel("Disorder strength  $\\sigma$")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def fig_histograms(samples, E_ord, path):
    plt = _plt()
    # production showcase picks; an extension grid (PC_SIGMAS) falls back
    # to its own first four cells so the figure never has zero columns
    show_sigmas = ([s for s in (0.04, 0.10, 0.20, 0.30) if s in SIGMAS]
                   or list(SIGMAS)[:4])
    fig, axes = plt.subplots(1, len(show_sigmas),
                             figsize=(3.4 * len(show_sigmas), 3.6),
                             sharey=True)
    for ax, s in zip(np.atleast_1d(axes), show_sigmas):
        for cls in CLASSES:
            Es = [float(x["E"]) for x in samples
                  if str(x["disorder_class"]) == cls
                  and abs(float(x["sigma"]) - s) < 1e-9]
            if Es:
                ax.hist(Es, bins=24, alpha=0.55,
                        color=CLASS_COLORS[cls], label=cls)
        ax.axvline(E_ord, color=CLASS_COLORS["ordered"], ls="--", lw=1.2)
        ax.set_title(f"$\\sigma$ = {s}")
        ax.set_xlabel("E")
    np.atleast_1d(axes)[0].set_ylabel("count")
    if np.atleast_1d(axes)[0].get_legend_handles_labels()[0]:
        np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Within-$\\sigma$ E distributions (dashed: ordered)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_generator_stats(samples, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for cls in CLASSES:
        xs, ms = [], []
        for s in SIGMAS:
            v = [float(x["n_redraws"]) for x in samples
                 if str(x["disorder_class"]) == cls
                 and abs(float(x["sigma"]) - s) < 1e-9]
            if v:
                xs.append(s)
                ms.append(np.mean(v))
        ax.plot(xs, ms, "o-", color=CLASS_COLORS[cls], label=cls)
    ax.set_xlabel("Disorder strength  $\\sigma$")
    ax.set_ylabel("mean constraint redraws per layout")
    ax.set_title("Where 'disorder' collides with 'manufacturable' "
                 f"(w_min = {C.W_MIN_NM:.0f} nm)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cmd_analyze(args):
    tee("analyze", OUT_DIR)
    print("=" * 64)
    print("CAMPAIGN ANALYSIS")
    print("=" * 64)
    os.makedirs(FIGS_DIR, exist_ok=True)
    m = get_materials()
    ad = m["adapters"]
    samples = _load_samples()
    wl = np.asarray(samples[0]["wavelengths_nm"], dtype=float)
    weights = solar_weight(wl)
    ref = planar_reference_stack(ad["si"], ad["zno"], ad["ag"], wl,
                                 C.THICKNESS_NM, C.BUFFER_NM)
    n_quar = len(glob.glob(os.path.join(QUAR_DIR, "*.npz")))
    print(f"loaded {len(samples)} banked samples "
          f"({n_quar} quarantined, inspected separately)")

    ords = [x for x in samples if str(x["disorder_class"]) == "ordered"]
    if not ords:
        raise SystemExit("the ordered reference sample (id 0) is not "
                         "banked yet -- generate it first.")
    ord_sample = ords[0]
    E_ord = float(ord_sample["E"])

    # ---- assemble the labeled dataset -----------------------------------
    n_holes = C.N_CELLS * C.N_CELLS
    keep = [x for x in samples
            if np.asarray(x["holes_xyr_nm"]).shape == (n_holes, 3)]
    ds_path = os.path.join(OUT_DIR, "dataset.npz")
    np.savez_compressed(
        ds_path,
        holes_xyr_nm=np.stack([np.asarray(x["holes_xyr_nm"])
                               for x in keep]),
        a_super_nm=C.A_SUPER_NM, thickness_nm=C.THICKNESS_NM,
        buffer_nm=C.BUFFER_NM, wavelengths_nm=wl, solar_weights=weights,
        A_planar=np.asarray(ref["A_si"], dtype=np.float32),
        A_spectra=np.stack([np.asarray(x["A_si"], dtype=np.float32)
                            for x in keep]),
        A_parasitic=np.stack([np.asarray(x["A_par"], dtype=np.float32)
                              for x in keep]),
        E=np.array([float(x["E"]) for x in keep]),
        eta=np.array([float(x["eta"]) for x in keep]),
        E_nir=np.array([float(x["E_nir"]) for x in keep]),
        parasitic_frac=np.array([float(x["parasitic_frac"])
                                 for x in keep]),
        disorder_class=np.array([str(x["disorder_class"]) for x in keep]),
        sigma=np.array([float(x["sigma"]) for x in keep]),
        seed=np.array([int(x["seed"]) for x in keep]),
        sample_id=np.array([int(x["sample_id"]) for x in keep]),
        numerics=str(samples[0]["numerics"]))
    lb_path = os.path.join(OUT_DIR, "labels.csv")
    with open(lb_path, "w") as f:
        f.write("sample_id,class,sigma,seed,E,eta,E_nir,parasitic_frac,"
                "fill_achieved,radius_scale,n_redraws,n_restarts\n")
        for x in keep:
            f.write(f"{int(x['sample_id'])},{x['disorder_class']},"
                    f"{float(x['sigma']):.4f},{int(x['seed'])},"
                    f"{float(x['E']):.6f},{float(x['eta']):.6f},"
                    f"{float(x['E_nir']):.6f},"
                    f"{float(x['parasitic_frac']):.6f},"
                    f"{float(x['fill_achieved']):.6f},"
                    f"{float(x['radius_scale']):.6f},"
                    f"{int(x['n_redraws'])},{int(x['n_restarts'])}\n")
    print(f"dataset -> {ds_path}  ({len(keep)} samples)\n"
          f"labels  -> {lb_path}")

    # ---- figures ---------------------------------------------------------
    dis = [x for x in samples if str(x["disorder_class"]) != "ordered"]
    fp = lambda name: os.path.join(FIGS_DIR, name)
    fig_E_vs_sigma(samples, E_ord, fp("fig5_E_vs_sigma.png"))
    fig_scatter(samples, E_ord, fp("fig6_realization_scatter.png"))
    if dis:
        picks = fig_spectra_showcase(samples, ord_sample, ref, wl,
                                     fp("fig7_spectra_showcase.png"))
        fig_gallery(samples, ord_sample, picks,
                    fp("fig8_layout_gallery.png"))
    fig_nir_and_parasitic(samples, ord_sample,
                          fp("fig9_nir_band_vs_sigma.png"),
                          fp("fig10_parasitic_vs_sigma.png"))
    fig_histograms(samples, E_ord, fp("fig11_E_histograms.png"))
    fig_generator_stats(samples, fp("fig12_generator_stats.png"))
    print(f"figures -> {FIGS_DIR}/fig5..fig12")

    # ---- console summary --------------------------------------------------
    print("\n" + "-" * 64)
    print("SUMMARY")
    print(f"  ordered reference     E = {E_ord:.4f}   "
          f"(NIR band x{float(ord_sample['E_nir']):.2f}, parasitic "
          f"{100 * float(ord_sample['parasitic_frac']):.2f}%)")
    if not dis:
        print("  (no disordered samples banked yet -- re-run analyze once "
              "generate has progressed)")
        print("-" * 64)
        return
    champ = max(dis, key=lambda x: float(x["E"]))
    n_beat = sum(float(x["E"]) > E_ord for x in dis)
    for cls in CLASSES:
        st = _sigma_stats(samples, cls)
        if st:
            s_star = max(st, key=lambda s: st[s]["mean"])
            print(f"  {cls:6s}: best mean E = {st[s_star]['mean']:.4f} "
                  f"+- {st[s_star]['sem']:.4f} at sigma* = {s_star} "
                  f"(n = {st[s_star]['n']})")
    print(f"  champion realization  E = {float(champ['E']):.4f}  "
          f"({champ['disorder_class']}, sigma={float(champ['sigma']):.3g},"
          f" seed={int(champ['seed'])})")
    print(f"  realizations beating the ordered lattice: "
          f"{n_beat}/{len(dis)}")
    print(f"  next: python run_dataset.py verify --top 6")
    print("-" * 64)


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------
def cmd_verify(args):
    tee("verify", OUT_DIR)
    print("=" * 64)
    print("ELEVATED-FIDELITY VERIFICATION of the champions")
    print("=" * 64)
    device = F.resolve_device(os.environ.get("PC_DEVICE", "auto"))
    m = get_materials()
    ad = m["adapters"]
    samples = _load_samples()
    check_meta(np.asarray(samples[0]["wavelengths_nm"], dtype=float))
    os.makedirs(FIGS_DIR, exist_ok=True)

    dis = [x for x in samples if str(x["disorder_class"]) != "ordered"]
    dis.sort(key=lambda x: -float(x["E"]))
    ords = [x for x in samples if str(x["disorder_class"]) == "ordered"]
    if not ords:
        raise SystemExit("the ordered reference sample (id 0) is not "
                         "banked -- generate it before verify.")
    picks = ords[:1] + dis[:args.top]

    res_v = int(round(C.RESOLUTION * 1.25))
    tol_v = C.DECAY_TOL / 10.0
    wl_v = C.raw_wavelength_grid(nir_step_nm=C.NIR_STEP_NM / 2)
    w_v = solar_weight(wl_v)
    ref_v = planar_reference_stack(ad["si"], ad["zno"], ad["ag"], wl_v,
                                   C.THICKNESS_NM, C.BUFFER_NM)
    print(f"verification numerics: res={res_v}/um, decay_tol={tol_v:g}, "
          f"NIR step {C.NIR_STEP_NM / 2:g} nm ({len(wl_v)} wavelengths) "
          f"-- vs campaign res={C.RESOLUTION}, tol={C.DECAY_TOL:g}")

    E_ver = []
    for x in picks:
        holes = [tuple(h) for h in np.asarray(x["holes_xyr_nm"])]
        A_si, _, _ = broadband_absorption_many(
            [holes], float(x["a_super_nm"]), C.THICKNESS_NM, wl_v,
            m["fits"], C.BUFFER_NM, res_v, tol_v, C.MAX_TIME * 1.5,
            NORM_DIR, device=device, n_cells_tag=f"vsc{C.N_CELLS}")
        E_ver.append(enhancement(np.nan_to_num(A_si[0]), ref_v["A_si"],
                                 w_v)[0])
        print(f"    verified {str(x['disorder_class'])} "
              f"sample {int(x['sample_id'])}: E={E_ver[-1]:.4f}",
              flush=True)
    E_ord_v = E_ver[0]

    labels, rows = [], []
    print(f"\n  {'layout':28s} {'E campaign':>11s} {'E verified':>11s} "
          f"{'vs ordered':>11s}")
    for x, Ev in zip(picks, E_ver):
        cls = str(x["disorder_class"])
        tag = ("ordered" if cls == "ordered" else
               f"{cls} s={float(x['sigma']):.3g} seed {int(x['seed'])}")
        labels.append(tag)
        rows.append((tag, float(x["E"]), Ev, Ev / E_ord_v - 1))
        print(f"  {tag:28s} {float(x['E']):11.4f} {Ev:11.4f} "
              f"{100 * (Ev / E_ord_v - 1):+10.2f}%")

    with open(os.path.join(OUT_DIR, "verified.csv"), "w") as f:
        f.write("layout,E_campaign,E_verified,rel_vs_ordered\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f}\n")

    plt = _plt()
    fig, ax = plt.subplots(figsize=(1.7 * len(picks) + 3, 5.2))
    colors = [CLASS_COLORS["ordered"]] + [
        CLASS_COLORS[str(x["disorder_class"])] for x in picks[1:]]
    ax.bar(range(len(picks)), E_ver, color=colors, alpha=0.85)
    ax.axhline(E_ord_v, color=CLASS_COLORS["ordered"], ls="--", lw=1.5)
    for i, (Ev, r) in enumerate(zip(E_ver, rows)):
        note = "" if i == 0 else f"  ({100 * r[3]:+.2f}%)"
        ax.annotate(f"{Ev:.4f}{note}", (i, Ev),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8)
    ax.set_xticks(range(len(picks)),
                  [l.replace(" seed", "\nseed") for l in labels],
                  fontsize=8)
    lo, hi = min(E_ver), max(E_ver)
    pad = 0.15 * (hi - lo + 1e-3)
    ax.set_ylim(lo - pad, hi + 3 * pad)
    ax.set_ylabel("Broadband E (elevated-fidelity verification labels)")
    ax.set_title("VERIFIED LEDGER: champions vs ordered at elevated "
                 "numerics")
    fig.tight_layout()
    p = os.path.join(FIGS_DIR, "fig13_verified_ledger.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\n  wrote {p} and verified.csv")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    for name in ("generate", "plan", "analyze", "verify"):
        p = sub.add_parser(name)
        if name == "generate":
            p.add_argument("--shard", type=str, default=None,
                           help="'i/N': solve samples with id %% N == i "
                                "(auto-detected inside SGE task arrays)")
            p.add_argument("--limit", type=int, default=None)
            p.add_argument("--skip-validation", action="store_true")
        if name == "verify":
            p.add_argument("--top", type=int, default=6)
    args = ap.parse_args()
    cmd = args.cmd or "generate"
    if cmd == "generate" and not hasattr(args, "shard"):
        args.shard, args.limit, args.skip_validation = None, None, False
    {"generate": cmd_generate, "plan": cmd_plan,
     "analyze": cmd_analyze, "verify": cmd_verify}[cmd](args)


if __name__ == "__main__":
    main()