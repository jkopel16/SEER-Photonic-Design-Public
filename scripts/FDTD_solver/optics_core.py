"""
optics_core.py
--------------
Material optical constants, solar-spectrum weighting, and the ANALYTIC
planar-reference physics for the disordered-photonic-crystal light-trapping
project -- REFLECTOR STAGE.

The layer stack is now (see config.py for the full history):

    air / patterned c-Si slab / ZnO buffer / Ag back-reflector (semi-inf)

so the planar reference is no longer the Airy membrane but a general
COHERENT TRANSFER-MATRIX (TMM) multilayer, tmm_unpolarized(), evaluated for
the unpatterned stack.  Because the Ag half-space is opaque, its "T" is the
parasitic metal absorption, and (with the buffer lossless)

    A_Si = 1 - R - T      -- absorption in the silicon film only,

which is the quantity the enhancement factor E is built from.

Materials bundled with the project (CSV + provenance sidecars, in
data/materials/ at the repo root):
    silicon_permittivity.csv  c-Si, Green 2008 tabulation
    silver_permittivity.csv   Ag, McPeak et al. 2015 (measured thin film)
    zno_permittivity.csv      ZnO, Bond 1965 dispersion (k = 0)

This module implements the "pre-requisite data collection" pieces of the
methodology plus the analytic reference and its self-checks.  It has no 
solver imports and is cheap to import anywhere.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# --------------------------------------------------------------------------
# Physical constants (SI)
# --------------------------------------------------------------------------
H_PLANCK = 6.626_070_15e-34   # J s
C_LIGHT = 2.997_924_58e8      # m / s
Q_E = 1.602_176_634e-19       # C

HERE = os.path.dirname(os.path.abspath(__file__))
# Material CSVs live at the repo root: <root>/data/materials
MATERIALS_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "data", "materials"))
SILICON_CSV = os.path.join(MATERIALS_DIR, "silicon_permittivity.csv")
SILVER_CSV = os.path.join(MATERIALS_DIR, "silver_permittivity.csv")
ZNO_CSV = os.path.join(MATERIALS_DIR, "zno_permittivity.csv")
AM15G_CSV = os.path.join(MATERIALS_DIR, "am15g.csv")


# ==========================================================================
# 1. Materials
# ==========================================================================
class Material:
    """Complex refractive index of a dispersive material, from a CSV of (nm, n, k).

    Provides cubic-spline interpolators for n(lambda) and k(lambda) and a
    convenience method for the complex relative permittivity eps = (n + i k)^2.

    NOTE ON SIGN CONVENTION
    -----------------------
    We use the physics/optics time convention exp(-i omega t), for which a
    passive (lossy) medium has a *positive* imaginary refractive index,
    n_tilde = n + i k, and eps = n_tilde^2 = (n^2 - k^2) + i (2 n k), with
    Im(eps) > 0.  grcwa expects this same convention, so we pass eps with a
    positive imaginary part.  (For silver this means Re(eps) < 0 and
    Im(eps) > 0 across the band.)  The Fresnel self-check pins the
    convention down numerically -- see validate_fresnel().
    """

    def __init__(self, csv_path: str, name: str = "material"):
        self.name = name
        # Single clean table with a '#'-commented provenance header and one
        # column-name row:  wavelength_nm, n, k   (wavelength already in nm).
        df = pd.read_csv(csv_path, comment="#")
        df = df.sort_values("wavelength_nm").reset_index(drop=True)
        self.wl_nm = df["wavelength_nm"].to_numpy(dtype=float)
        self.n_tab = df["n"].to_numpy(dtype=float)
        self.k_tab = df["k"].to_numpy(dtype=float)
        self.wl_min, self.wl_max = self.wl_nm.min(), self.wl_nm.max()

        # Cubic interpolators, no extrapolation (we clamp queries to range).
        self._n_spline = CubicSpline(self.wl_nm, self.n_tab, extrapolate=False)
        self._k_spline = CubicSpline(self.wl_nm, self.k_tab, extrapolate=False)

    def _check_range(self, wl_nm):
        wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
        if wl.min() < self.wl_min - 1e-9 or wl.max() > self.wl_max + 1e-9:
            raise ValueError(
                f"[{self.name}] wavelength query [{wl.min():.1f},"
                f"{wl.max():.1f}] nm outside tabulated range "
                f"[{self.wl_min:.1f},{self.wl_max:.1f}] nm."
            )
        return wl

    def n(self, wl_nm):
        wl = self._check_range(wl_nm)
        return self._n_spline(wl)

    def k(self, wl_nm):
        wl = self._check_range(wl_nm)
        return self._k_spline(wl)

    def n_complex(self, wl_nm):
        """Complex refractive index n_tilde = n + i k (exp(-i omega t) convention)."""
        wl = self._check_range(wl_nm)
        out = self._n_spline(wl) + 1j * self._k_spline(wl)
        return out.item() if np.isscalar(wl_nm) or np.ndim(wl_nm) == 0 else out

    def eps(self, wl_nm):
        """Complex relative permittivity, eps = n_tilde^2 with Im(eps) >= 0."""
        nt = self.n_complex(wl_nm)
        return nt ** 2


def load_silicon() -> Material:
    """c-Si (Green 2008 tabulation) -- the absorber."""
    return Material(SILICON_CSV, name="c-Si (Green 2008)")


def load_silver() -> Material:
    """Ag (McPeak 2015 measured thin film) -- the back reflector."""
    return Material(SILVER_CSV, name="Ag (McPeak 2015)")


def load_zno() -> Material:
    """ZnO (Bond 1965 dispersion, k=0) -- the lossless TCO buffer."""
    return Material(ZNO_CSV, name="ZnO (Bond 1965)")


# ==========================================================================
# 2. Solar spectrum: AM1.5G, real ASTM G-173 photon flux
# ==========================================================================
# The AM1.5G photon-flux table is loaded once from am15g.csv (derived from the
# official NREL ASTM G-173-03 file: the 'Global tilt' irradiance column,
# converted to photon flux via Phi = E_irradiance * lambda / (h c)).  We build
# a module-level interpolator so repeated calls are cheap.
_AM15G_WL = None      # nm grid of the tabulated spectrum
_AM15G_PHI = None     # photon flux on that grid


def _load_am15g():
    """Lazily load and cache the AM1.5G photon-flux table from am15g.csv."""
    global _AM15G_WL, _AM15G_PHI
    if _AM15G_WL is None:
        df = pd.read_csv(AM15G_CSV, comment="#")
        df = df.sort_values("wavelength_nm").reset_index(drop=True)
        _AM15G_WL = df["wavelength_nm"].to_numpy(dtype=float)
        _AM15G_PHI = df["photon_flux"].to_numpy(dtype=float)
    return _AM15G_WL, _AM15G_PHI


def am15g_photon_flux(wl_nm: np.ndarray) -> np.ndarray:
    """AM1.5G photon flux at the requested wavelength(s), interpolated from
    the official ASTM G-173-03 'Global tilt' spectrum.

    The methodology requires weighting absorption by the *photon* flux, not
    the irradiance, because one absorbed photon yields at most one electron.
    Linear interpolation onto the query grid; the tabulated spectrum is dense
    (0.5-1 nm spacing) so linear is more than adequate and avoids the ringing
    a cubic spline can introduce at the sharp atmospheric absorption dips.
    """
    wl = np.asarray(wl_nm, dtype=float)
    tab_wl, tab_phi = _load_am15g()
    if wl.min() < tab_wl.min() - 1e-9 or wl.max() > tab_wl.max() + 1e-9:
        raise ValueError(
            f"Wavelength query [{wl.min():.1f},{wl.max():.1f}] nm outside the "
            f"AM1.5G table range [{tab_wl.min():.1f},{tab_wl.max():.1f}] nm."
        )
    return np.interp(wl, tab_wl, tab_phi)


def solar_weight(wl_nm: np.ndarray) -> np.ndarray:
    """Normalised solar photon-flux weights on the given wavelength grid.

    Returns weights that sum to 1 via trapezoidal quadrature on wl_nm, so that
    a weighted average  sum(w_i * A_i)  approximates
    integral(A Phi dlambda) / integral(Phi dlambda).  Works on the
    non-uniform (coarse-visible + dense-NIR) grids this stage uses.
    """
    wl = np.asarray(wl_nm, dtype=float)
    phi = am15g_photon_flux(wl)
    w = np.zeros_like(wl)
    dwl = np.diff(wl)
    w[:-1] += dwl / 2.0
    w[1:] += dwl / 2.0
    w = w * phi
    return w / w.sum()


def enhancement(A_pattern, A_reference, weights):
    """Scalar broadband enhancement factor and the two weighted absorptions:
        E = eta_pattern / eta_reference,  eta = sum(w * A).
    Returns (E, eta_pattern, eta_reference)."""
    eta_p = float(np.sum(weights * A_pattern))
    eta_r = float(np.sum(weights * A_reference))
    return eta_p / eta_r, eta_p, eta_r


# ==========================================================================
# 3. Analytic planar physics
# ==========================================================================
def fresnel_R_normal(n_tilde: complex, n_incident: float = 1.0) -> float:
    """Normal-incidence single-interface reflectance from medium n_incident
    into a semi-infinite medium of complex index n_tilde:
        R = |(n_i - n_tilde) / (n_i + n_tilde)|^2
    """
    r = (n_incident - n_tilde) / (n_incident + n_tilde)
    return float(np.abs(r) ** 2)


def _cos_theta_transmitted(nt, theta_i_rad):
    """Complex refraction cosine inside a medium of complex index nt for
    incidence angle theta_i from air (Snell with the branch chosen so the
    field DECAYS into an absorbing medium: Im(nt * cos_t) >= 0)."""
    nt = np.asarray(nt, dtype=complex)
    sin_t = np.sin(theta_i_rad) / nt
    cos_t = np.sqrt(1.0 - sin_t ** 2)
    flip = np.imag(nt * cos_t) < 0
    return np.where(flip, -cos_t, cos_t)


def tmm_polarized(finite_layers, exit_material, wl_nm, theta_deg, pol):
    """Coherent transfer-matrix R, T, A of the stack

        air / finite_layers[0] / finite_layers[1] / ... / exit (semi-infinite)

    for ONE linear polarisation, vectorised over wavelength.

    Parameters
    ----------
    finite_layers : list of (material, thickness_nm) with material either a
        Material instance or a (constant) complex refractive index.  Layers
        with thickness 0 are skipped.
    exit_material : Material or complex index of the semi-infinite exit
        half-space (Ag here; may be strongly absorbing).
    wl_nm : array of wavelengths (nm).
    theta_deg : incidence angle from air, degrees.
    pol : "s" (TE) or "p" (TM).

    Returns
    -------
    (R, T, A) arrays.  T is the net Poynting flux crossing the final
    interface into the exit half-space, normalised to the incident flux --
    for an absorbing exit medium this IS the power it dissipates.
    A = 1 - R - T is the total absorption in the finite layers.

    Method: standard optical-admittance characteristic matrices (Macleod).
    For each layer  eta = n_tilde cos(theta_t)  (s)  or  n_tilde/cos(theta_t)
    (p),  delta = 2 pi n_tilde cos(theta_t) t / lambda,
        M = [[cos d, i sin d / eta], [i eta sin d, cos d]],
        [B, C]^T = M_1 M_2 ... M_N [1, eta_exit]^T,
        r = (eta_0 B - C)/(eta_0 B + C),        R = |r|^2,
        T = 4 eta_0 Re(eta_exit) / |eta_0 B + C|^2.
    The T expression is the exact interface flux and remains valid for an
    absorbing substrate (potential-transmittance formulation)."""
    wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
    th = np.deg2rad(theta_deg)

    def index_of(m):
        if isinstance(m, Material):
            return np.atleast_1d(m.n_complex(wl))
        return np.full(wl.shape, complex(m))

    def admittance(nt):
        cost = _cos_theta_transmitted(nt, th)
        eta = nt * cost if pol == "s" else nt / cost
        return eta, cost

    eta0 = np.cos(th) if pol == "s" else 1.0 / np.cos(th)   # air, n = 1
    nt_exit = index_of(exit_material)
    eta_exit, _ = admittance(nt_exit)

    B = np.ones(wl.shape, dtype=complex)
    C = eta_exit.astype(complex).copy()
    # Multiply characteristic matrices from the layer nearest the exit upward.
    for mat, t_nm in reversed(list(finite_layers)):
        if t_nm <= 0.0:
            continue
        nt, = (index_of(mat),)
        eta, cost = admittance(nt)
        # SIGN NOTE: the textbook (Macleod) characteristic matrix is written
        # for the engineering convention n_tilde = n - i k.  This project uses
        # the physics convention n_tilde = n + i k (exp(-i omega t)), so the
        # phase thickness carries a minus sign; with it, absorbing layers
        # DECAY (validated against the independent Airy closed form and the
        # bare-interface Fresnel formula in validate_tmm).
        delta = -2.0 * np.pi * nt * cost * t_nm / wl
        cd, sd = np.cos(delta), np.sin(delta)
        B, C = cd * B + 1j * sd / eta * C, 1j * eta * sd * B + cd * C

    denom = eta0 * B + C
    r = (eta0 * B - C) / denom
    R = np.abs(r) ** 2
    T = 4.0 * eta0 * np.real(eta_exit) / np.abs(denom) ** 2
    A = 1.0 - R - T
    return R, T, A


def tmm_unpolarized(finite_layers, exit_material, wl_nm, theta_deg=0.0):
    """Unpolarised (s/p average) coherent TMM of air / finite layers / exit.
    Returns dict with keys R, T, A (arrays over wl_nm); see tmm_polarized."""
    Rs, Ts, As = tmm_polarized(finite_layers, exit_material, wl_nm,
                               theta_deg, "s")
    Rp, Tp, Ap = tmm_polarized(finite_layers, exit_material, wl_nm,
                               theta_deg, "p")
    return {"R": 0.5 * (Rs + Rp), "T": 0.5 * (Ts + Tp),
            "A": 0.5 * (As + Ap)}


def planar_reference_stack(si: Material, zno: Material, ag: Material,
                           wl_nm, thickness_nm, buffer_nm,
                           theta_deg=0.0):
    """THE planar reference of the reflector stage: the UNPATTERNED stack
        air / c-Si slab (thickness_nm) / ZnO buffer (buffer_nm) / Ag,
    coherent, unpolarised, at incidence theta_deg.

    Returns dict with
        A_si  : absorption in the silicon slab (= 1 - R - T, buffer lossless)
        A_par : parasitic absorption in the Ag mirror (= T into the metal)
        R     : reflectance back into air.
    RCWA of a UNIFORM slab on the same stack must reproduce these to
    numerical precision -- that identity is the project's central end-to-end
    physics gate (pc_solver.validate_uniform_slab)."""
    layers = [(si, thickness_nm)]
    if buffer_nm > 0.0:
        layers.append((zno, buffer_nm))
    out = tmm_unpolarized(layers, ag, wl_nm, theta_deg)
    return {"A_si": out["A"], "A_par": out["T"], "R": out["R"]}


def planar_absorption_coherent(material: Material, wl_nm: np.ndarray,
                               thickness_nm: float,
                               theta_deg: float = 0.0) -> np.ndarray:
    """COHERENT absorption A = 1 - R - T of a free-standing (air/film/air)
    MEMBRANE via the closed-form Airy summation, unpolarised.

    Kept from the membrane stage PURELY AS A VALIDATION REFERENCE: the
    general TMM above must reproduce this closed form exactly for the
    air/Si/air special case (validate_tmm), which pins the TMM implementation
    to an independently derived formula before it is trusted as the
    reflector-stack reference."""
    wl = np.asarray(wl_nm, dtype=float)
    nt = np.atleast_1d(material.n_complex(wl))
    th = np.deg2rad(theta_deg)
    cos0 = np.cos(th)
    cost = _cos_theta_transmitted(nt, th)
    beta = 2.0 * np.pi * nt * cost * thickness_nm / wl
    e1 = np.exp(1j * beta)
    e2 = e1 * e1
    A_pol = []
    for pol in ("s", "p"):
        if pol == "s":
            r01 = (cos0 - nt * cost) / (cos0 + nt * cost)
        else:
            r01 = (nt * cos0 - cost) / (nt * cos0 + cost)
        denom = 1.0 - r01 ** 2 * e2
        r_tot = r01 * (1.0 - e2) / denom
        t_tot = (1.0 - r01 ** 2) * e1 / denom
        A_pol.append(1.0 - np.abs(r_tot) ** 2 - np.abs(t_tot) ** 2)
    A = 0.5 * (A_pol[0] + A_pol[1])
    return A if np.ndim(wl_nm) else float(A[0])


# ==========================================================================
# 4. Validations (methodology Sec 1.1 A)
# ==========================================================================
def validate_fresnel(material: Material, wl_probe_nm: float = 600.0,
                     verbose: bool = True) -> bool:
    """Cross-check that the permittivity/sign convention reproduces the
    analytic bare-interface Fresnel reflectance -- the cheapest guard against
    the most common silent physics bug (a sign-convention mismatch)."""
    nt = material.n_complex(wl_probe_nm)
    eps = material.eps(wl_probe_nm)
    nt_from_eps = np.sqrt(eps)
    if nt_from_eps.imag < 0:                    # sqrt branch: Im >= 0
        nt_from_eps = -nt_from_eps
    err = abs(nt_from_eps - nt)
    R = fresnel_R_normal(nt)
    ok = err < 1e-9 and eps.imag >= 0
    if verbose:
        print(f"[Fresnel check @ {wl_probe_nm:.0f} nm, {material.name}]")
        print(f"    n_tilde (direct)      = {nt.real:.4f} + {nt.imag:.4f} i")
        print(f"    n_tilde (from eps^0.5)= {nt_from_eps.real:.4f} + "
              f"{nt_from_eps.imag:.4f} i")
        print(f"    Im(eps) = {eps.imag:+.4f}  (must be >= 0, passive medium)")
        print(f"    round-trip error      = {err:.2e}   "
              f"-> {'OK' if ok else 'FAIL'}")
        print(f"    normal-incidence R    = {R:.4f}")
    return ok


def validate_tmm(si: Material = None, verbose: bool = True) -> bool:
    """The TMM implementation must reproduce two independent closed forms:

      (1) air/Si/air membrane == the Airy summation (independent algebra),
          at normal incidence AND 30 deg oblique;
      (2) air -> semi-infinite Ag == the single-interface Fresnel formula
          (a 'stack' with zero finite layers).

    Passing both pins the admittance bookkeeping (both polarisations, complex
    angles, absorbing exit) before the TMM is trusted as the reflector-stack
    planar reference."""
    if si is None:
        si = load_silicon()
    ag = load_silver()
    wl = np.linspace(420.0, 1080.0, 7)
    ok_all = True
    if verbose:
        print("[TMM cross-checks]")
    for th in (0.0, 30.0):
        out = tmm_unpolarized([(si, 300.0)], 1.0 + 0.0j, wl, th)
        A_airy = planar_absorption_coherent(si, wl, 300.0, theta_deg=th)
        err = float(np.max(np.abs(out["A"] - A_airy)))
        ok = err < 1e-12
        ok_all &= ok
        if verbose:
            print(f"    membrane vs Airy, theta={th:4.0f} deg: max|dA| = "
                  f"{err:.2e}   -> {'OK' if ok else 'FAIL'}")
    out = tmm_unpolarized([], ag, wl, 0.0)
    R_fres = np.array([fresnel_R_normal(n) for n in ag.n_complex(wl)])
    err = float(np.max(np.abs(out["R"] - R_fres)))
    ok = err < 1e-12
    ok_all &= ok
    if verbose:
        print(f"    bare-Ag interface vs Fresnel: max|dR| = {err:.2e}   "
              f"-> {'OK' if ok else 'FAIL'}")
        print(f"    (Ag NIR reflectivity sanity: R(1000 nm) = "
              f"{float(np.interp(1000.0, wl, R_fres)):.4f}, expect ~0.98+)")
    return ok_all


if __name__ == "__main__":
    si, ag, zno = load_silicon(), load_silver(), load_zno()
    for m in (si, ag, zno):
        print(f"Loaded material: {m.name}  "
              f"({m.wl_min:.0f}-{m.wl_max:.0f} nm, {len(m.wl_nm)} points)")
    print()
    ok = validate_fresnel(si, 600.0) and validate_fresnel(ag, 800.0)
    print()
    ok &= validate_tmm(si)
    wl = np.linspace(400, 1100, 8)
    ref = planar_reference_stack(si, zno, ag, wl, 300.0, 80.0)
    print("\n[planar reference stack, air/Si(300)/ZnO(80)/Ag]")
    for w, a, p, r in zip(wl, ref["A_si"], ref["A_par"], ref["R"]):
        print(f"    {w:6.0f} nm  A_si={a:.4f}  A_par={p:.4f}  R={r:.4f}  "
              f"sum={a + p + r:.6f}")
    print("\nall optics_core self-checks:", "PASS" if ok else "FAIL")
