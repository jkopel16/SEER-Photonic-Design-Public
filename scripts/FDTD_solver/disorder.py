"""
disorder.py
-----------
Seeded, constraint-respecting disordered-layout generator for the
photonic-crystal light-trapping project.

The generator is a PURE FUNCTION:

    (disorder class, strength sigma, random seed)  ->  valid hole list + metadata

so every sample is exactly regenerable from its metadata alone.  The output
hole list -- (x, y, r) triples in nm within the supercell -- is precisely the
interface pc_solver.py's solver already consumes, so nothing in the physics
stack changes.

Disorder classes
---------------------------------------------
  "ordered" : the nominal N x N square lattice (the sigma = 0 reference;
              identical to pc_solver.ordered_square_holes()).
  "jitter"  : positional disorder.  Each hole is displaced from its lattice
              site by an independent vector whose x and y components are drawn
              uniformly from [-sigma*a, +sigma*a].  sigma is therefore the
              per-component half-width of the displacement in units of the
              lattice constant a (the "% jitter" of the pseudo-disorder
              literature).
  "radius"  : size disorder.  Each radius is r_nom * (1 + u) with
              u ~ Uniform(-sigma, +sigma); centres stay on the lattice.
  "random"  : the maximal-disorder end-member.  The lattice is discarded and
              N^2 holes are placed by sequential random insertion subject to
              the same wall-thickness constraint.  sigma is meaningless here
              and is recorded as NaN.

Hard constraints enforced on every emitted layout
-------------------------------------------------
  * PERIODIC WRAP-AROUND: hole centres live on the torus [0, L)^2, and ALL
    inter-hole distances use the minimum-image convention -- the supercell
    tiles the plane, so the solver sees exactly what the constraint checker
    sees (no artificial seams).
  * ETCHABILITY: the edge-to-edge silicon wall between every pair of holes
    (minimum-image) must be at least w_min (default 50 nm, the mid-range of
    the ~40-60 nm etch limit from the fabrication literature).
    Violations are resolved by per-hole rejection-resampling with a bounded
    retry budget; if the budget exhausts, the whole layout restarts from a
    derived fresh sub-seed (bounded restarts), and the stats are logged in
    the returned record -- a rising redraw/restart rate vs sigma is itself
    a finding.
  * FIXED FILL FRACTION: after constraints are satisfied, all radii are
    rescaled by one common factor so every sample removes an IDENTICAL air
    fraction (that of the nominal lattice).  Without this, "more disorder"
    secretly co-varies with "different amount of absorber" and the
    disorder-absorption claim is confounded.  Constraints
    are re-verified after rescaling.

Determinism
-----------
All randomness flows from numpy's default_rng seeded with the integer pair
[seed, restart_index]; the accepted restart index is stored in the record, so
regeneration is bit-for-bit exact.

This module needs only numpy -- no solver imports -- so it is importable in
the ML stage without dragging grcwa along.
"""

from __future__ import annotations

import numpy as np

GEN_VERSION = "disorder-gen-1.0"

DISORDER_CLASSES = ("jitter", "radius", "random")   # "ordered" also accepted


# ==========================================================================
# Minimum-image geometry helpers
# ==========================================================================
def min_image(d, L):
    """Wrap separations d onto (-L/2, L/2]: the minimum-image convention."""
    return d - L * np.round(d / L)


def pair_gaps(centers, radii, L):
    """Edge-to-edge silicon wall thickness for every hole pair, using
    minimum-image distances on the L x L torus.

    Returns (i_idx, j_idx, gaps) over the upper triangle (i < j)."""
    c = np.asarray(centers, dtype=float)
    r = np.asarray(radii, dtype=float)
    dx = min_image(c[:, 0][:, None] - c[:, 0][None, :], L)
    dy = min_image(c[:, 1][:, None] - c[:, 1][None, :], L)
    dist = np.hypot(dx, dy)
    gaps = dist - (r[:, None] + r[None, :])
    iu, ju = np.triu_indices(len(c), k=1)
    return iu, ju, gaps[iu, ju]


def violating_holes(centers, radii, L, w_min):
    """Set of hole indices participating in at least one wall-thickness
    violation (gap < w_min, minimum-image)."""
    iu, ju, gaps = pair_gaps(centers, radii, L)
    bad = gaps < w_min
    return set(iu[bad].tolist()) | set(ju[bad].tolist())


def air_fraction(radii, L):
    """Air fill fraction of the supercell.  On the torus each hole contributes
    its full disk area exactly once (wrap-around, no clipping), and the
    non-overlap constraint guarantees disks are disjoint."""
    r = np.asarray(radii, dtype=float)
    return float(np.pi * np.sum(r ** 2) / (L * L))


# ==========================================================================
# The generator
# ==========================================================================
def _nominal_lattice(a_nm, n_cells, r_nm):
    """Nominal N x N square lattice: identical arithmetic to
    pc_solver.ordered_square_holes(), returned as arrays."""
    ij = np.arange(n_cells)
    xx, yy = np.meshgrid((ij + 0.5) * a_nm, (ij + 0.5) * a_nm, indexing="ij")
    centers = np.column_stack([xx.ravel(), yy.ravel()])
    radii = np.full(n_cells * n_cells, float(r_nm))
    return centers, radii


def make_layout(disorder_class, sigma, seed, a_nm=650.0, n_cells=4,
                r_nm=None, w_min_nm=50.0, r_min_nm=50.0, r_max_frac=0.45,
                max_rounds=200, max_restarts=50):
    """Generate one valid layout record.

    Parameters
    ----------
    disorder_class : "ordered" | "jitter" | "radius" | "random"
    sigma          : disorder strength (see module docstring for the exact
                     per-class meaning; ignored for "ordered"/"random").
    seed           : integer seed; together with the class/sigma metadata it
                     regenerates the layout exactly.
    a_nm, n_cells, r_nm : lattice constant, supercell size, nominal radius
                     (r_nm defaults to 0.30 * a_nm as a generic fallback;
                     the PROJECT nominal is config.R_OVER_A = 0.35 * a_nm,
                     which every production call site passes explicitly).
    w_min_nm       : minimum silicon wall between holes (etch limit).
    r_min_nm, r_max_frac : sanity bounds on any single radius
                     (r in [r_min_nm, r_max_frac * a_nm]).
    max_rounds     : per-restart budget of constraint-repair sweeps.
    max_restarts   : whole-layout restart budget before raising.

    Returns
    -------
    record : dict with keys
        holes        list of (x, y, r) nm triples  (the solver interface)
        a_super_nm   supercell edge length
        class, sigma, seed, n_cells, a_nm, r_nom_nm, w_min_nm
        fill_target, fill_achieved, radius_scale
        n_redraws, n_restarts, generator
    """
    if r_nm is None:
        r_nm = 0.30 * a_nm
    L = n_cells * a_nm
    n_holes = n_cells * n_cells
    centers0, radii0 = _nominal_lattice(a_nm, n_cells, r_nm)
    fill_target = air_fraction(radii0, L)

    # ---------------- trivial case: the ordered reference -----------------
    if disorder_class == "ordered" or (disorder_class in ("jitter", "radius")
                                       and sigma == 0.0):
        holes = [(float(x), float(y), float(r))
                 for (x, y), r in zip(centers0, radii0)]
        return {"holes": holes, "a_super_nm": L, "class": "ordered",
                "sigma": 0.0, "seed": int(seed), "n_cells": n_cells,
                "a_nm": a_nm, "r_nom_nm": float(r_nm),
                "w_min_nm": w_min_nm, "fill_target": fill_target,
                "fill_achieved": fill_target, "radius_scale": 1.0,
                "n_redraws": 0, "n_restarts": 0, "generator": GEN_VERSION}

    if disorder_class not in DISORDER_CLASSES:
        raise ValueError(f"unknown disorder class '{disorder_class}'")

    r_max_nm = r_max_frac * a_nm

    for restart in range(max_restarts):
        rng = np.random.default_rng([int(seed), restart])
        n_redraws = 0

        # ------------- per-class initial draw + per-hole redraw -----------
        if disorder_class == "jitter":
            centers = (centers0 +
                       rng.uniform(-sigma * a_nm, sigma * a_nm,
                                   size=(n_holes, 2))) % L
            radii = radii0.copy()

            def redraw(m):
                centers[m] = (centers0[m] + rng.uniform(-sigma * a_nm,
                                                        sigma * a_nm,
                                                        size=2)) % L

        elif disorder_class == "radius":
            centers = centers0.copy()
            radii = np.clip(radii0 * (1.0 + rng.uniform(-sigma, sigma,
                                                        size=n_holes)),
                            r_min_nm, r_max_nm)

            def redraw(m):
                radii[m] = np.clip(radii0[m] * (1.0 + rng.uniform(-sigma,
                                                                  sigma)),
                                   r_min_nm, r_max_nm)

        else:  # "random": sequential random insertion (Sec 1.2 step 2c/4)
            radii = radii0.copy()
            centers = np.zeros((n_holes, 2))
            placed = 0
            for m in range(n_holes):
                for _try in range(max_rounds):
                    cand = rng.uniform(0.0, L, size=2)
                    if placed == 0:
                        centers[m] = cand
                        placed += 1
                        break
                    dx = min_image(centers[:placed, 0] - cand[0], L)
                    dy = min_image(centers[:placed, 1] - cand[1], L)
                    gap = (np.hypot(dx, dy)
                           - radii[:placed] - radii[m])
                    if gap.min() >= w_min_nm:
                        centers[m] = cand
                        placed += 1
                        break
                    n_redraws += 1
                else:
                    break                       # insertion failed
            if placed < n_holes:
                continue                        # restart whole layout

            def redraw(m):
                centers[m] = rng.uniform(0.0, L, size=2)

        # ------------- constraint repair by rejection-resampling ----------
        ok = _repair(centers, radii, L, w_min_nm, redraw, rng, max_rounds)
        n_redraws += ok[1]
        if not ok[0]:
            continue                            # restart whole layout

        # ------------- fixed-fill renormalisation + re-verify -------------
        total_scale = 1.0
        settled = False
        for _pass in range(30):
            scale = np.sqrt(fill_target / air_fraction(radii, L))
            radii *= scale
            total_scale *= scale
            if radii.min() < r_min_nm or radii.max() > r_max_nm:
                break                           # unphysical -> restart
            vio = violating_holes(centers, radii, L, w_min_nm)
            if not vio:
                if abs(scale - 1.0) < 1e-12:
                    settled = True
                    break                       # converged, constraints hold
                continue                        # re-check with scale -> 1
            # rescale broke a wall: redraw offenders, then renormalise again
            okr = _repair(centers, radii, L, w_min_nm, redraw, rng,
                          max_rounds)
            n_redraws += okr[1]
            if not okr[0]:
                break                           # -> restart
        if not settled:
            continue                            # restart whole layout

        holes = [(float(x), float(y), float(r))
                 for (x, y), r in zip(centers, radii)]
        return {"holes": holes, "a_super_nm": L, "class": disorder_class,
                "sigma": (float("nan") if disorder_class == "random"
                          else float(sigma)),
                "seed": int(seed), "n_cells": n_cells, "a_nm": a_nm,
                "r_nom_nm": float(r_nm), "w_min_nm": w_min_nm,
                "fill_target": fill_target,
                "fill_achieved": air_fraction(radii, L),
                "radius_scale": total_scale,
                "n_redraws": int(n_redraws), "n_restarts": restart,
                "generator": GEN_VERSION}

    raise RuntimeError(
        f"layout generation failed after {max_restarts} restarts "
        f"(class={disorder_class}, sigma={sigma}, seed={seed}) -- disorder "
        f"has collided with manufacturability at this strength/fill.")


def _repair(centers, radii, L, w_min, redraw, rng, max_rounds):
    """Rejection-resampling sweeps: redraw every currently violating hole in
    random order, repeat until clean or the round budget exhausts.
    Returns (success, n_redraws)."""
    n_redraws = 0
    for _round in range(max_rounds):
        vio = violating_holes(centers, radii, L, w_min)
        if not vio:
            return True, n_redraws
        for m in rng.permutation(sorted(vio)):
            redraw(int(m))
            n_redraws += 1
    return False, n_redraws


# ==========================================================================
# Small conveniences for drivers / the ML stage
# ==========================================================================
def holes_array(record):
    """(n_holes, 3) float array of the record's (x, y, r) triples."""
    return np.asarray(record["holes"], dtype=float)


def generate_ensemble(disorder_class, sigmas, seeds_per_sigma, base_seed,
                      **layout_kwargs):
    """Full factorial {sigmas} x {seeds} for one class, with unique,
    reproducible integer seeds derived from base_seed.  For class "random",
    pass sigmas=[nan] (or anything of length 1).  Returns a flat list of
    records."""
    records = []
    for s_idx, sigma in enumerate(np.atleast_1d(sigmas)):
        for k in range(seeds_per_sigma):
            seed = int(base_seed + 100 * s_idx + k)
            records.append(make_layout(disorder_class, sigma, seed,
                                       **layout_kwargs))
    return records


# ==========================================================================
# Self-test (no solver needed)
# ==========================================================================
if __name__ == "__main__":
    print(f"[{GEN_VERSION} self-test]")
    A, N, R = 650.0, 4, 0.30 * 650.0
    L = N * A

    # 1) sigma = 0 must reproduce the ordered lattice exactly
    rec0 = make_layout("jitter", 0.0, seed=1, a_nm=A, n_cells=N, r_nm=R)
    c0, r0 = _nominal_lattice(A, N, R)
    ref = [(float(x), float(y), float(r)) for (x, y), r in zip(c0, r0)]
    assert rec0["holes"] == ref, "sigma=0 does not reproduce ordered lattice"
    print("  sigma=0 == ordered lattice          OK")

    # 2) determinism: same (class, sigma, seed) -> identical layout
    a1 = make_layout("jitter", 0.25, 42, a_nm=A, n_cells=N, r_nm=R)
    a2 = make_layout("jitter", 0.25, 42, a_nm=A, n_cells=N, r_nm=R)
    assert a1["holes"] == a2["holes"], "generator is not deterministic"
    print("  determinism (seed round-trip)       OK")

    # 3) constraints + fixed fill across classes and strengths
    for cls, sig in [("jitter", 0.10), ("jitter", 0.40), ("radius", 0.10),
                     ("radius", 0.40), ("random", float("nan"))]:
        for seed in range(5):
            rec = make_layout(cls, sig, 1000 + seed, a_nm=A, n_cells=N,
                              r_nm=R)
            h = holes_array(rec)
            vio = violating_holes(h[:, :2], h[:, 2], L, rec["w_min_nm"])
            assert not vio, f"wall violation in {cls}, sigma={sig}"
            assert abs(rec["fill_achieved"] - rec["fill_target"]) < 1e-12, \
                f"fill not renormalised in {cls}"
        print(f"  {cls:6s} sigma={sig!s:5s}: walls + fixed fill    OK "
              f"(last: {rec['n_redraws']} redraws, "
              f"{rec['n_restarts']} restarts)")
    print("all self-tests passed.")
