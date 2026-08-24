"""
materials_gpu.py
----------------
DISPERSIVE MATERIAL MODELS FOR THE GPU-FDTD STAGE.

This module does NOT refit anything: it loads the FROZEN per-band Drude-Lorentz
fits from material_fits.json (produced and gated by a Meep engine) so the
GPU engine uses literally identical material models -- one of the two
things (with the shared wavelength grid) that keeps GPU labels
apples-to-apples with every earlier stage.

Conventions (identical to the Meep stage; f in 1/lambda_um, exp(-i omega t),
Im eps >= 0 for passive media):

    eps(f) = eps_inf + sum_j  sigma_j f_j^2 / (f_j^2 - f^2 - i gamma_j f)
                     + [Drude]  sigma_D f_D^2 / (-f^2 - i gamma_D f)

Sub-bands (the c-Si band-edge problem forces piecewise fits; every
structure is solved once per band with that band's materials):

    BANDS = (400-700), (700-950), (950-1100) nm

validate_fits() reruns the same hard gates as the Meep stage (n-accuracy
per band + the two LABEL gates: planar broadband eta and the synthetic
strong-trapping eta) -- these run at the start of every driver.
"""

from __future__ import annotations

import hashlib
import json
import os
import numpy as np
from dataclasses import dataclass

from optics_core import (Material, MATERIALS_DIR, load_silicon, load_silver,
                         load_zno, solar_weight, planar_reference_stack)

HERE = os.path.dirname(os.path.abspath(__file__))

BANDS = ((400.0, 700.0), (700.0, 950.0), (950.0, 1100.0))
FIT_MARGIN_NM = 25.0


def band_of(wl_nm: float) -> int:
    """Index of the band that owns wavelength wl_nm (half-open [lo, hi),
    last band closed).  Identical logic to the Meep stage."""
    for i, (lo, hi) in enumerate(BANDS):
        if (wl_nm < hi) or (i == len(BANDS) - 1):
            if wl_nm >= lo or i == 0:
                return i
    return len(BANDS) - 1


# --------------------------------------------------------------------------
# Susceptibility model (analytic side -- must match the engine's ADE)
# --------------------------------------------------------------------------
@dataclass
class Pole:
    kind: str          # "lorentz" | "drude"
    sigma: float
    f0: float
    gamma: float

    def eps_contrib(self, f):
        f = np.asarray(f, dtype=float)
        if self.kind == "lorentz":
            return (self.sigma * self.f0 ** 2
                    / (self.f0 ** 2 - f ** 2 - 1j * self.gamma * f))
        # Drude: sigma f0^2 / (-f^2 - i gamma f)
        return self.sigma * self.f0 ** 2 / (-(f ** 2) - 1j * self.gamma * f)


@dataclass
class BandFit:
    """A fitted eps model valid over one wavelength band."""
    band: tuple
    eps_inf: float
    poles: list
    max_dn: float = 0.0
    max_dk_abs: float = 0.0
    max_dk_rel: float = 0.0

    def eps(self, wl_nm):
        f = 1000.0 / np.asarray(wl_nm, dtype=float)
        out = np.full(f.shape, self.eps_inf, dtype=complex)
        for p in self.poles:
            out = out + p.eps_contrib(f)
        return out

    def n_complex(self, wl_nm):
        e = self.eps(wl_nm)
        nt = np.sqrt(e)
        nt = np.where(nt.imag < 0, -nt, nt)     # passive branch
        return nt

    def ade_pole_table(self):
        """(n_poles, 3) array of the ADE parameters the engine consumes:
        columns (omega0, gamma_omega, K) in ANGULAR units (rad / (um/c)):

            p_ddot + gamma_omega * p_dot + omega0^2 * p = K * E

        Lorentz: omega0 = 2 pi f0,  K = sigma * (2 pi f0)^2
        Drude:   omega0 = 0,        K = sigma * (2 pi f0)^2
        both with gamma_omega = 2 pi gamma, which reproduces eps_contrib()
        exactly in the continuum limit (verified by the planar gate)."""
        rows = []
        for p in self.poles:
            w0 = 0.0 if p.kind == "drude" else 2.0 * np.pi * p.f0
            K = p.sigma * (2.0 * np.pi * p.f0) ** 2
            rows.append((w0, 2.0 * np.pi * p.gamma, K))
        return np.asarray(rows, dtype=float)


# --------------------------------------------------------------------------
# TMM adapter: a Material whose optical constants ARE the fits
# --------------------------------------------------------------------------
class FitAsMaterial(Material):
    """Duck-types optics_core.Material, evaluating n_complex from the
    per-band fits (piecewise across BANDS).  Passing this to
    planar_reference_stack gives the planar reference AS THE FDTD SEES IT
    -- the internally consistent denominator for E, and the correct target
    for the FDTD-vs-TMM validation gates."""

    def __init__(self, band_fits, name):
        # deliberately NOT calling Material.__init__ (no CSV behind this)
        self.name = name
        self.band_fits = band_fits
        self.wl_min = BANDS[0][0] - FIT_MARGIN_NM
        self.wl_max = BANDS[-1][1] + FIT_MARGIN_NM
        self.wl_nm = np.linspace(self.wl_min, self.wl_max, 400)
        nt = self.n_complex(self.wl_nm)
        self.n_tab, self.k_tab = nt.real, nt.imag

    def n_complex(self, wl_nm):
        wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
        out = np.empty(wl.shape, dtype=complex)
        for i, w in enumerate(wl):
            out[i] = self.band_fits[band_of(w)].n_complex(w)
        if np.isscalar(wl_nm) or np.ndim(wl_nm) == 0:
            return out.item()
        return out

    def n(self, wl_nm):
        return np.atleast_1d(self.n_complex(wl_nm)).real

    def k(self, wl_nm):
        return np.atleast_1d(self.n_complex(wl_nm)).imag

    def eps(self, wl_nm):
        nt = self.n_complex(wl_nm)
        return nt ** 2


# --------------------------------------------------------------------------
# Loading the frozen fits
# --------------------------------------------------------------------------
def _fits_from_json(d):
    out = {}
    for name, lst in d.items():
        bfs = []
        for rec in lst:
            bf = BandFit(tuple(rec["band"]), rec["eps_inf"],
                         [Pole(k, s, f0, g)
                          for (k, s, f0, g) in rec["poles"]])
            bf.max_dn = rec["max_dn"]
            bf.max_dk_abs = rec["max_dk_abs"]
            bf.max_dk_rel = rec["max_dk_rel"]
            bfs.append(bf)
        out[name] = bfs
    return out


def _csv_key():
    h = hashlib.sha256()
    for fn in ("silicon_permittivity.csv", "zno_permittivity.csv",
               "silver_permittivity.csv"):
        path = os.path.join(MATERIALS_DIR, fn)
        try:
            with open(path, "rb") as f:
                h.update(f.read())
        except OSError as e:
            raise SystemExit(
                f"material CSV missing/unreadable: {path} ({e}). "
                "The frozen-fit guard cannot be evaluated without it.")
    return h.hexdigest()[:16] + f"|bands={BANDS}"


def fit_all(cache_path=None):
    """Load the FROZEN fits; returns (fits, adapters, (si, zno, ag)) with
    the same shape as the Meep engine's materials_fdtd.fit_all().

    Refuses to run if material_fits.json is missing or was produced from
    different CSVs -- a dataset with drifting materials is worse than no
    dataset.  (To regenerate the fits, run the CPU project's
    materials_fdtd.py and copy the fresh material_fits.json here and the
    CSVs into data/materials/.)
    """
    cache_path = cache_path or os.path.join(HERE, "material_fits.json")
    si, zno, ag = load_silicon(), load_zno(), load_silver()
    if not os.path.exists(cache_path):
        raise SystemExit(
            f"material_fits.json not found at {cache_path}.\n"
            "Copy it (and the three material CSVs it was fitted from) from "
            "the CPU/Meep project folder -- the GPU stage never refits.")
    with open(cache_path) as f:
        d = json.load(f)
    key = _csv_key()
    if d.get("key") != key:
        raise SystemExit(
            "material_fits.json was fitted from DIFFERENT material CSVs "
            f"than the ones in data/materials/\n  (json key {d.get('key')!r} "
            f"vs local {key!r}).\nCopy the matching json+CSV set from the "
            "CPU project; the GPU stage never refits.")
    fits = _fits_from_json(d["fits"])
    adapters = {k: FitAsMaterial(v, name=f"{k}-fit")
                for k, v in fits.items()}
    return fits, adapters, (si, zno, ag)


# --------------------------------------------------------------------------
# Validation: same ledger + label gates as the Meep stage
# --------------------------------------------------------------------------
def validate_fits(fits, adapters, mats, thickness_nm=300.0, buffer_nm=80.0,
                  verbose=True):
    si, zno, ag = mats
    ok = True
    if verbose:
        print("[Drude-Lorentz fit ledger  (per band, fit vs table)]")
    n_gates = {"si": 1.6e-2, "zno": 1e-3, "ag": 1e-2}
    for name, bfs in fits.items():
        for bf in bfs:
            flag = "OK" if bf.max_dn < n_gates[name] else "CHECK"
            ok &= flag == "OK"
            if verbose:
                print(f"    {name:3s} {bf.band[0]:5.0f}-{bf.band[1]:4.0f} "
                      f"nm: max|dn|={bf.max_dn:.2e}  "
                      f"max|dk|={bf.max_dk_abs:.2e}  "
                      f"(rel k err where k>1e-3: "
                      f"{100 * bf.max_dk_rel:.2f}%)  -> {flag}")

    # LABEL GATE 1: planar broadband eta, fitted vs tabulated materials
    wl = np.concatenate([np.arange(400, 700, 6.0),
                         np.arange(700, 1100 + 1e-9, 2.0)])
    w = solar_weight(wl)
    ref_tab = planar_reference_stack(si, zno, ag, wl, thickness_nm,
                                     buffer_nm)
    ref_fit = planar_reference_stack(adapters["si"], adapters["zno"],
                                     adapters["ag"], wl, thickness_nm,
                                     buffer_nm)
    eta_tab = float(np.sum(w * ref_tab["A_si"]))
    eta_fit = float(np.sum(w * ref_fit["A_si"]))
    d_planar = eta_fit / eta_tab - 1
    dmax = float(np.max(np.abs(ref_fit["A_si"] - ref_tab["A_si"])))
    ok &= abs(d_planar) < 0.01

    # LABEL GATE 2: strong-trapping regime (synthetic 4n^2-path absorber)
    L_eff_nm = 15000.0
    k_tab = np.atleast_1d(si.n_complex(wl)).imag
    k_fit = np.atleast_1d(adapters["si"].n_complex(wl)).imag
    A_t = 1.0 - np.exp(-4 * np.pi * k_tab * L_eff_nm / wl)
    A_f = 1.0 - np.exp(-4 * np.pi * np.maximum(k_fit, 0) * L_eff_nm / wl)
    eta_t = float(np.sum(w * A_t))
    eta_f = float(np.sum(w * A_f))
    d_trap = eta_f / eta_t - 1
    ok &= abs(d_trap) < 0.025
    if verbose:
        print(f"    LABEL GATE planar: eta(table)={eta_tab:.5f}  "
              f"eta(fit)={eta_fit:.5f}  ({100 * d_planar:+.2f}%, gate 1%)"
              f"   max|dA_si|={dmax:.3e}")
        print(f"    LABEL GATE trapping (L_eff=15um): "
              f"eta(table)={eta_t:.5f}  eta(fit)={eta_f:.5f}  "
              f"({100 * d_trap:+.2f}%, gate 2.5%)")
    return ok


if __name__ == "__main__":
    fits, adapters, mats = fit_all()
    ok = validate_fits(fits, adapters, mats)
    for name, bfs in fits.items():
        for bf in bfs:
            tab = bf.ade_pole_table()
            print(f"  {name} {bf.band}: eps_inf={bf.eps_inf:.4f}, "
                  f"{len(bf.poles)} poles, max omega_char = "
                  f"{np.sqrt(np.maximum(tab[:, 0] ** 2, tab[:, 2])).max():.1f}"
                  f" rad/(um/c)")
    print("\nmaterials_gpu self-test:", "PASS" if ok else
          "CHECK FAILED -- do not run FDTD on these fits")
