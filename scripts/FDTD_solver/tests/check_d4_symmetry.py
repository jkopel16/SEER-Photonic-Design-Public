"""
check_d4_symmetry.py
--------------------
D4 AUGMENTATION VALIDITY CHECK: apply all 8 square-symmetry operations
(identity, 3 rotations, 4 mirrors) to ONE layout's hole list, re-solve
every copy with the production torch-FDTD engine, and verify the E labels
agree.

Why this settles the augmentation debate:
  * The physical claim: at normal incidence, with isotropic materials, a
    square supercell and polarization-summed labels, every D4 image of a
    layout is the SAME boundary-value problem -> identical E exactly.
  * The numerical caveat: the FDTD grid is binary-staircased (no subpixel
    averaging), so a rotated hole list re-rasterizes with a different
    staircase.  Agreement is therefore expected to within the known
    staircase jitter, not necessarily to four decimals.
  * The augmentation consequence: if this check passes, the correct way
    to augment is np.rot90/np.flip on the CNN raster REUSING the original
    E -- never re-simulating rotated copies (that would only sample
    discretization noise).

The script also runs two zero-cost exact checks BEFORE any FDTD:
  1. Isometry invariants: fill fraction and the full pair-gap spectrum
     must be bit-identical across all 8 ops (D4 ops are isometries of the
     torus; any drift here is a bug in the ops themselves).
  2. Raster commutation: re-rasterizing the transformed hole list must
     equal a pure numpy array op (rot90/flip/transpose) applied to the
     base raster.  The printed table of {geometry op -> numpy op} is the
     lookup the CNN augmentation code should use, given rasterize_mask's
     [x, y] ("ij") indexing.

Usage (from the project folder, conda env photonics-fdtd):
    # geometry + raster checks only (seconds, no GPU, no torch):
    python check_d4_symmetry.py --geometry-only

    # plumbing test of the full path (FAST numerics, small GPU cost):
    PC_MODE=FAST python check_d4_symmetry.py

    # the real verdict (production numerics; ~8x one campaign sample,
    # so run it in tmux like any overnight job):
    python check_d4_symmetry.py
    # (all output auto-mirrors to PC_OUT/logs/d4_check_<timestamp>.log
    #  via logutil.tee -- no shell pipe needed)

Defaults to the manifest's jitter sigma=0.10 k=0 sample (the optimum
region -- a layout with no residual symmetry of its own).  Override with
--dclass/--sigma/--seed.  Everything persists under PC_OUT/d4_check/.
The campaign's samples/ directory is never touched; the norm cache IS
shared (same n_cells_tag), so vacuum normalizations are reused, not
recomputed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np


# solver modules live one directory up from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from logutil import tee
from disorder import make_layout, holes_array, pair_gaps, air_fraction

# --------------------------------------------------------------------------
# The 8 elements of D4, acting on hole lists on the [0, L)^2 torus.
# All ops are about the supercell center; radii are untouched (isometries).
# --------------------------------------------------------------------------
def d4_ops(L):
    return [
        ("identity", lambda x, y: (x, y)),
        ("rot90",    lambda x, y: ((L - y) % L, x)),
        ("rot180",   lambda x, y: ((L - x) % L, (L - y) % L)),
        ("rot270",   lambda x, y: (y, (L - x) % L)),
        ("mirror_x", lambda x, y: ((L - x) % L, y)),
        ("mirror_y", lambda x, y: (x, (L - y) % L)),
        ("diag",     lambda x, y: (y, x)),
        ("antidiag", lambda x, y: ((L - y) % L, (L - x) % L)),
    ]


def transform_holes(holes, op):
    return [(*op(hx, hy), hr) for (hx, hy, hr) in holes]


# Candidate pure-array ops the raster of a transformed layout could equal
# (for a [x, y]-indexed image); the script FINDS the match empirically so
# the printed table is ground truth, not a hand-derived convention.
def array_op_candidates():
    return [
        ("as-is",              lambda M: M),
        ("np.rot90(M, 1)",     lambda M: np.rot90(M, 1)),
        ("np.rot90(M, 2)",     lambda M: np.rot90(M, 2)),
        ("np.rot90(M, 3)",     lambda M: np.rot90(M, 3)),
        ("np.flipud(M)",       lambda M: np.flipud(M)),
        ("np.fliplr(M)",       lambda M: np.fliplr(M)),
        ("M.T",                lambda M: M.T),
        ("np.rot90(M, 2).T",   lambda M: np.rot90(M, 2).T),
    ]


# --------------------------------------------------------------------------
# Stage 1: exact geometric invariants (free; must be ~machine-exact)
# --------------------------------------------------------------------------
def check_invariants(base_holes, L, w_min):
    ok = True
    h0 = np.asarray(base_holes, float)
    fill0 = air_fraction(h0[:, 2], L)
    _, _, gaps0 = pair_gaps(h0[:, :2], h0[:, 2], L)
    gaps0 = np.sort(gaps0)
    print("\n[stage 1: isometry invariants across all 8 ops]")
    print(f"    base: fill={fill0:.12f}  min wall="
          f"{gaps0.min():.3f} nm (constraint {w_min:g} nm)")
    for name, op in d4_ops(L):
        ht = np.asarray(transform_holes(base_holes, op), float)
        fill = air_fraction(ht[:, 2], L)
        _, _, gaps = pair_gaps(ht[:, :2], ht[:, 2], L)
        d_fill = abs(fill - fill0)
        d_gap = float(np.max(np.abs(np.sort(gaps) - gaps0)))
        flag = "OK" if (d_fill < 1e-9 and d_gap < 1e-6) else "FAIL"
        ok &= flag == "OK"
        print(f"    {name:9s}: |dfill|={d_fill:.2e}  "
              f"max|d gap spectrum|={d_gap:.2e} nm  -> {flag}")
    return ok


# --------------------------------------------------------------------------
# Stage 2: raster commutation (free; validates the CNN augmentation recipe)
# --------------------------------------------------------------------------
def check_raster_commutation(base_holes, L):
    from fdtd_torch import rasterize_mask
    n = C.IMG_SIZE
    M0 = rasterize_mask(base_holes, L, n, n, supersample=C.SUPERSAMPLE)
    print("\n[stage 2: raster commutation at IMG_SIZE="
          f"{n} (the CNN input)]")
    print("    geometry op -> matching numpy array op on the base raster")
    ok = True
    table = {}
    for name, op in d4_ops(L):
        Mt = rasterize_mask(transform_holes(base_holes, op), L, n, n,
                            supersample=C.SUPERSAMPLE)
        best, best_err = None, np.inf
        for aname, aop in array_op_candidates():
            err = float(np.max(np.abs(aop(M0) - Mt)))
            if err < best_err:
                best, best_err = aname, err
        # one supersample flipping at a disk edge changes a pixel by
        # 1/SUPERSAMPLE^2; allow a couple of such ulp-ties
        tol = 2.5 / C.SUPERSAMPLE ** 2
        flag = "OK" if best_err <= tol else "FAIL"
        ok &= flag == "OK"
        table[name] = best
        print(f"    {name:9s} -> {best:18s} (max|dpixel|={best_err:.2e})"
              f"  -> {flag}")
    print("    (this table is the augmentation lookup for the ML stage,\n"
          "     for [x, y]-indexed rasters as rasterize_mask emits them)")
    return ok, table


# --------------------------------------------------------------------------
# Stage 3: FDTD -- the E labels of all 8 copies
# --------------------------------------------------------------------------
def gate(A_si, A_par):
    """Same sanity gate as run_dataset.gate_sample (inlined so this script
    has no run_dataset import side effects)."""
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


def run_fdtd_stage(base_holes, L, out_dir):
    import fdtd_torch as F
    from fdtd_torch import broadband_absorption_many
    from materials_gpu import fit_all
    from optics_core import planar_reference_stack, solar_weight, \
        enhancement

    device = F.resolve_device(os.environ.get("PC_DEVICE", "auto"))
    fits, adapters, _ = fit_all()
    wl = C.raw_wavelength_grid()
    w = solar_weight(wl)
    ref = planar_reference_stack(adapters["si"], adapters["zno"],
                                 adapters["ag"], wl, C.THICKNESS_NM,
                                 C.BUFFER_NM)

    ops = d4_ops(L)
    holes_list = [transform_holes(base_holes, op) for _, op in ops]
    names = [n for n, _ in ops]
    norm_dir = os.path.join(C.OUT_DIR, "norm_cache")   # SHARED with the
    # campaign (same n_cells_tag below), so vacuum norms are reused.

    print(f"\n[stage 3: FDTD on all 8 copies]  device={device}")
    print(f"    numerics: {C.describe()}")
    print(f"    cost: 8 structures x {3 * 2} band/pol runs each, "
          f"sequential -- expect ~8x one campaign sample.")
    t0 = time.time()

    def progress(done, total):
        el = time.time() - t0
        print(f"      run {done:2d}/{total}  elapsed {el / 60:6.1f} min  "
              f"ETA {(total - done) * el / done / 60:6.1f} min",
              flush=True)

    A_si, A_par, info = broadband_absorption_many(
        holes_list, L, C.THICKNESS_NM, wl, fits, C.BUFFER_NM,
        C.RESOLUTION, C.DECAY_TOL, C.MAX_TIME, norm_dir, device=device,
        n_cells_tag=f"sc{C.N_CELLS}", progress=progress)

    rows = []
    for g, name in enumerate(names):
        reason = gate(A_si[g], A_par[g])
        if bool(info["hit_time_cap"][g]) and reason is None:
            reason = "hit ring-down time cap"
        a = np.clip(np.nan_to_num(A_si[g]), 0.0, 1.0)
        E, eta, _ = enhancement(a, ref["A_si"], w)
        rows.append({"op": name, "E": E, "eta": eta,
                     "gate": reason or "clean",
                     "runtime_s": float(info["runtime_s"][g])})

    # ---- persist everything ---------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "d4_spectra.npz"),
        wavelengths_nm=wl, ops=np.array(names),
        A_si=A_si.astype(np.float32), A_par=A_par.astype(np.float32))
    with open(os.path.join(out_dir, "d4_results.csv"), "w") as f:
        f.write("op,E,eta,gate,runtime_s\n")
        for r in rows:
            f.write(f"{r['op']},{r['E']:.6f},{r['eta']:.6f},"
                    f"{r['gate']},{r['runtime_s']:.1f}\n")
    return rows


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------
def verdict(rows):
    clean = [r for r in rows if r["gate"] == "clean"]
    print("\n" + "=" * 64)
    print("D4 SYMMETRY LEDGER")
    print("=" * 64)
    E0 = rows[0]["E"]
    print(f"    {'op':9s} {'E':>9s} {'dE vs identity':>15s} "
          f"{'rel':>9s}  gate")
    for r in rows:
        print(f"    {r['op']:9s} {r['E']:9.5f} {r['E'] - E0:+15.5f} "
              f"{100 * (r['E'] / E0 - 1):+8.3f}%  {r['gate']}")
    if len(clean) < len(rows):
        print(f"\n    WARNING: {len(rows) - len(clean)} copies failed the "
              "sanity gate -- verdict below uses clean copies only.")
    Es = np.array([r["E"] for r in clean])
    spread = float(Es.max() - Es.min())
    rel = spread / float(Es.mean())
    print(f"\n    clean copies: {len(clean)}/8   "
          f"max spread dE = {spread:.5f}  ({100 * rel:.3f}% of mean)")

    print("\nVERDICT")
    if len(clean) < 8:
        print("    INVESTIGATE: gate failures on symmetry copies of a "
              "clean layout\n    are themselves a red flag (same physics, "
              "different staircase\n    should not blow up).  Inspect "
              "d4_spectra.npz per band first.")
    elif rel < 1e-3:
        print("    PASS (discretely exact).  The engine's staircase + Yee\n"
              "    discretization commutes with D4 to <0.1%.  D4 "
              "augmentation is\n    unconditionally valid; reuse the "
              "original E for all 8 copies.")
    elif rel < 0.01:
        print("    PASS (physically exact).  Spread is within the known\n"
              "    staircase jitter; the underlying label is identical by\n"
              "    symmetry.  D4 augmentation is valid -- augment the "
              "raster with\n    the stage-2 numpy ops and REUSE the "
              "original E.  Do NOT\n    re-simulate rotated copies; the "
              "spread you just measured is\n    discretization noise, "
              "not signal.")
    elif rel < 0.025:
        print("    BORDERLINE.  Spread exceeds 1% but sits inside the "
              "+-2.5%\n    staircase budget measured on high-Q structures."
              "  Most likely\n    this layout has a needle resonance "
              "moving with the staircase.\n    Recommended: re-run this "
              "script once with PC_RESOLUTION=120 (the\n    elite-"
              "verification rung; res 90 was convicted as an unconverged\n"
              "    referee, audit Test 8) -- if the spread shrinks, it is "
              "grid noise\n    and augmentation stands.")
    else:
        print("    FAIL.  Spread exceeds the staircase budget -- a real\n"
              "    asymmetry.  Check, in order: (1) both polarizations "
              "actually\n    ran (pols=('x','y'))  (2) the raster "
              "commutation table above\n    (3) per-band spectra in "
              "d4_spectra.npz to localize the band.\n    Do not augment "
              "until this is understood.")
    print("=" * 64)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dclass", default="jitter",
                    choices=("jitter", "radius", "random"))
    ap.add_argument("--sigma", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=None,
                    help="default: the manifest seed for (dclass, sigma), "
                         "k=0 -- i.e. a layout the campaign has banked")
    ap.add_argument("--geometry-only", action="store_true",
                    help="stages 1-2 only: no torch, no GPU, seconds")
    args = ap.parse_args()

    # manifest-compatible default seed (same formula as run_dataset)
    if args.seed is None:
        SIGMAS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.125, 0.15, 0.20,
                  0.25, 0.30)
        CLASSES = ("jitter", "radius")
        if args.dclass == "random":
            args.seed = C.BASE_SEED + 3_000_000
        else:
            c_idx = CLASSES.index(args.dclass)
            s_idx = min(range(len(SIGMAS)),
                        key=lambda i: abs(SIGMAS[i] - args.sigma))
            args.seed = (C.BASE_SEED + (c_idx + 1) * 1_000_000
                         + s_idx * 10_000)

    tee("d4_check", C.OUT_DIR)
    out_dir = os.path.join(C.OUT_DIR, "d4_check")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 64)
    print("D4 AUGMENTATION VALIDITY CHECK")
    print("=" * 64)
    print(C.describe())
    rec = make_layout(args.dclass, args.sigma, args.seed, a_nm=C.A_NM,
                      n_cells=C.N_CELLS, r_nm=C.R_OVER_A * C.A_NM,
                      w_min_nm=C.W_MIN_NM)
    L = rec["a_super_nm"]
    base_holes = [tuple(h) for h in holes_array(rec)]
    print(f"layout: class={rec['class']} sigma={rec['sigma']} "
          f"seed={rec['seed']}  ({len(base_holes)} holes, "
          f"fill={rec['fill_achieved']:.4f})")

    ok1 = check_invariants(base_holes, L, C.W_MIN_NM)
    ok2, table = check_raster_commutation(base_holes, L)
    with open(os.path.join(out_dir, "augmentation_table.json"), "w") as f:
        json.dump(table, f, indent=1)
    if not (ok1 and ok2):
        raise SystemExit("\nABORT: the free geometric checks failed -- "
                         "fix the ops before burning GPU time.")
    print("\nstages 1-2 PASS: the D4 ops are exact isometries and the "
          "raster\ncommutes with pure numpy array ops "
          f"(table -> {out_dir}/augmentation_table.json).")

    if args.geometry_only:
        print("\n--geometry-only: stopping before FDTD.  Re-run without "
              "the flag\n(in tmux, FULL mode) for the E-label verdict.")
        return

    rows = run_fdtd_stage(base_holes, L, out_dir)
    verdict(rows)
    print(f"\nartifacts: {out_dir}/d4_results.csv, d4_spectra.npz, "
          f"augmentation_table.json")


if __name__ == "__main__":
    main()
