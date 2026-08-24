"""
fdtd_torch.py
-------------
Full-wave FDTD absorption of a photonic-crystal silicon thin film on the
reflector stack, on ONE CUDA GPU (PyTorch):

    air / patterned Si slab (t) / ZnO buffer / Ag mirror (opaque, on PEC)

WHY A CUSTOM ENGINE.  Meep does not support GPUs (its FAQ is explicit), so
"run the Meep scripts on the GPU node" is not a thing that exists.  This
module implements the minimum FDTD this project needs -- normal-incidence
plane wave, in-plane periodic supercell, layered z-stack, per-band
Drude-Lorentz media, top CPML, two DFT flux monitors with a cached
normalization (vacuum) run for incident-field subtraction, and adaptive
DFT-decay stopping -- as ~10 large tensor kernels per timestep, which is
exactly the workload GPUs are good at.

VALIDATION CHAIN (run before ANY physics is banked; run_all_validations):
  1. material-fit label gates            (materials_gpu.validate_fits)
  2. pseudo-1D planar stack vs fitted-material TMM analytic reference
     (tiny in-plane cell, high z-resolution: isolates the engine's ADE +
     CPML + source + monitor plumbing from geometry staircasing)
  3. 3D uniform slab at PRODUCTION supercell numerics vs the same TMM
     (bounds the discretization error of the production grid)
plus, in run_timing_test.py, the CROSS-ENGINE ANCHOR: the unit-cell
ordered lattice must land near the RCWA value E = 2.547.

DISCRETIZATION NOTES:
  * binary staircased geometry (no subpixel averaging in dispersive
    media); RESOLUTION is the convergence knob, exactly as before.
  * dt = 0.5 * min(dx, dz) by default (Courant margin), further capped so
    every ADE pole recurrence is stable (omega_char * dt <= 1.5).  The
    SAME dt is used for a band's normalization and structure runs (the
    DFT subtraction requires it); it is stored in the norm cache and
    asserted on load.
  * fields are float32 by default (PC_PRECISION=64 available); the DFT
    accumulators are float32 pairs updated with torch.addr_ (outer-
    product accumulate), which avoids large temporaries.

FAKE MODE: FDTD_FAKE=1 swaps every solver call for the same fast analytic
stand-in the Meep stage used (planar fitted-TMM + seeded resonance bumps):
full-pipeline plumbing tests with zero physics cost and no torch/GPU
needed.  Never bank FAKE spectra.
"""

from __future__ import annotations

import math
import os
import time
import json
import numpy as np
from types import SimpleNamespace

from optics_core import planar_reference_stack
from materials_gpu import BANDS, band_of, FitAsMaterial

FAKE = os.environ.get("FDTD_FAKE", "0") == "1"

ENGINE_VERSION = "torch-fdtd-1.0"

# --------------------------------------------------------------------------
# z-layout of the FDTD cell (um), bottom-up: PEC wall / Ag / ZnO / Si / air
# / top CPML.  Only the TOP boundary carries PML in structure runs and it 
# touches only air; the Ag is opaque (>= 12 skin depths above the PEC wall), 
# so nothing returns from below.
# --------------------------------------------------------------------------
DPML_UM = 0.40           # CPML thickness (top; also bottom in norm runs)
SRC_GAP_UM = 0.10        # PML -> source plane
MON_GAP_UM = 0.16        # source -> reflection monitor
AIR_GAP_UM = 0.16        # reflection monitor -> Si top surface
AG_UM = 0.35             # silver thickness down to the PEC bottom wall

COURANT = 0.5            # dt = COURANT * min(dx, dz), then pole-capped
# (pole stability is now computed exactly per band -- see band_dt)
DFT_SAMPLES_PER_PERIOD = 12.0   # subsampling target for the DFT stream
CHECK_INTERVAL_UM = 10.0        # decay checks every ~this much sim time
CPML_M = 3               # polynomial grading order
CPML_ALPHA = 0.05        # CFS alpha (frequency shift), constant
CPML_SIGMA_SCALE = 0.8   # sigma_max = scale * (m+1) / dz  (eta0 = 1 units)


def _zgeom(thickness_nm, buffer_nm):
    """Physical z coordinates, z = 0 at the PEC bottom wall."""
    t_si = thickness_nm / 1000.0
    t_zno = buffer_nm / 1000.0
    z_zno0 = AG_UM
    z_si0 = z_zno0 + t_zno
    z_si1 = z_si0 + t_si
    z_apar = z_zno0 + t_zno / 2.0 if t_zno > 0 else z_zno0
    z_refl = z_si1 + AIR_GAP_UM
    z_src = z_refl + MON_GAP_UM
    cz = z_src + SRC_GAP_UM + DPML_UM
    return SimpleNamespace(cz=cz, t_si=t_si, t_zno=t_zno, z_zno0=z_zno0,
                           z_si0=z_si0, z_si1=z_si1, z_apar=z_apar,
                           z_refl=z_refl, z_src=z_src)


# --------------------------------------------------------------------------
# Wavelength-grid <-> band bookkeeping
# --------------------------------------------------------------------------
def split_grid_by_band(wl_nm):
    wl = np.asarray(wl_nm, dtype=float)
    return [np.where([band_of(w) == b for w in wl])[0]
            for b in range(len(BANDS))]


# --------------------------------------------------------------------------
# Geometry helpers (rasterizer)
# --------------------------------------------------------------------------
def ordered_square_holes(a_nm, N, r_nm):
    a_super = N * a_nm
    holes = []
    for i in range(N):
        for j in range(N):
            holes.append(((i + 0.5) * a_nm, (j + 0.5) * a_nm, r_nm))
    return holes, a_super


def rasterize_mask(holes, a_super_nm, Nx, Ny, supersample=4):
    """Anti-aliased Si occupancy image (1 = Si, 0 = air), minimum-image.
    Used for the ML dataset image and figures; NOT for the FDTD grid."""
    ss = supersample
    NX, NY = Nx * ss, Ny * ss
    px = a_super_nm / NX
    py = a_super_nm / NY
    xs = (np.arange(NX) + 0.5) * px
    ys = (np.arange(NY) + 0.5) * py
    inside = np.zeros((NX, NY), dtype=bool)
    for (hx, hy, hr) in holes:
        i0 = int(np.floor((hx - hr) / px)) - 1
        i1 = int(np.ceil((hx + hr) / px)) + 1
        j0 = int(np.floor((hy - hr) / py)) - 1
        j1 = int(np.ceil((hy + hr) / py)) + 1
        ii = np.arange(i0, i1 + 1) % NX
        jj = np.arange(j0, j1 + 1) % NY
        dx = xs[ii] - hx
        dx -= a_super_nm * np.round(dx / a_super_nm)
        dy = ys[jj] - hy
        dy -= a_super_nm * np.round(dy / a_super_nm)
        disk = (dx[:, None] ** 2 + dy[None, :] ** 2) <= hr * hr
        inside[np.ix_(ii, jj)] |= disk
    frac_si = 1.0 - inside.astype(float)
    return frac_si.reshape(Nx, ss, Ny, ss).mean(axis=(1, 3))


def _si_mask_np(holes_nm, L_nm, xs_nm, ys_nm):
    """Binary Si mask (True = silicon) sampled at the staggered in-plane
    coordinates xs x ys, minimum-image on the L x L torus -- the FDTD
    analog of Meep's cylinders-with-periodic-images (staircased)."""
    X = np.asarray(xs_nm, float)[:, None]
    Y = np.asarray(ys_nm, float)[None, :]
    inside = np.zeros((len(xs_nm), len(ys_nm)), dtype=bool)
    for (hx, hy, hr) in holes_nm:
        dx = X - hx
        dx -= L_nm * np.round(dx / L_nm)
        dy = Y - hy
        dy -= L_nm * np.round(dy / L_nm)
        inside |= (dx * dx + dy * dy) <= hr * hr
    return ~inside


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------
def _torch():
    import torch
    return torch


def resolve_device(pref="auto"):
    torch = _torch()
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def _spectral_radius_step(dt, Omega, eps_inf, table):
    """Spectral radius of the one-step amplification matrix of the
    COUPLED Yee + ADE update in a uniform medium, at the worst spatial
    mode (curl-curl eigenvalue Omega).  State: (E, H, p_j, q_j=p_j^-)."""
    J = len(table)
    n = 2 + 2 * J
    M = np.zeros((n, n), dtype=complex)
    iOd = 1j * Omega * dt
    den = 1.0 + table[:, 1] * dt / 2.0
    c1 = (2.0 - table[:, 0] ** 2 * dt ** 2) / den
    c2 = -(1.0 - table[:, 1] * dt / 2.0) / den
    c3 = table[:, 2] * dt ** 2 / den
    # H' = H + iOd E
    M[1, 1] = 1.0
    M[1, 0] = iOd
    # p_j' = c1 p_j + c2 q_j + c3 E ;  q_j' = p_j
    for j in range(J):
        M[2 + j, 2 + j] = c1[j]
        M[2 + j, 2 + J + j] = c2[j]
        M[2 + j, 0] = c3[j]
        M[2 + J + j, 2 + j] = 1.0
    # E' = E + (iOd H' - sum_j (p_j' - p_j)) / eps_inf
    M[0, 0] = 1.0 + (iOd * iOd - np.sum(c3)) / eps_inf
    M[0, 1] = iOd / eps_inf
    for j in range(J):
        M[0, 2 + j] = -(c1[j] - 1.0) / eps_inf
        M[0, 2 + J + j] = -c2[j] / eps_inf
    return float(np.max(np.abs(np.linalg.eigvals(M))))


_DT_CACHE = {}


def band_dt(resolution, fits, band_idx, cz_res=None):
    """The timestep for one band's runs, shared exactly between the
    normalization and structure runs (the DFT subtraction requires it;
    it is stored in the norm cache and asserted on load).

    dt = min(COURANT * dz,  0.95 * exact stability limit of the coupled
    Yee+ADE system) -- the limit is computed by von Neumann analysis of
    the one-step amplification matrix per material at the worst spatial
    mode (bisection on the spectral radius).  A fixed omega_char cap is
    NOT safe: measured, the visible-band Si pole set goes unstable near
    omega_char*dt ~ 1.0 while the 700-950 set is stable beyond 1.5, so
    only the exact per-band bound is simultaneously safe and fast."""
    key = (resolution, band_idx)
    if key in _DT_CACHE:
        return _DT_CACHE[key]
    dx = 1.0 / resolution
    dt_c = COURANT * dx
    Omega = 2.0 * math.sqrt(3.0) / dx        # 3D worst mode, dx=dy=dz
    dt = dt_c
    for name in ("si", "zno", "ag"):
        bf = fits[name][band_idx]
        table = bf.ade_pole_table()
        if not len(table):
            continue
        # Marginal (lossless / Drude-drift) modes sit at radius EXACTLY 1
        # in exact arithmetic, but eigvals of their near-defective pairs
        # carries O(sqrt(machine-eps)) ~ 1e-8 rounding scatter that
        # differs between BLAS builds (measured: an SCC numpy rounded
        # them to 1+2e-9, over-capping band-1 dt by 6x while a container
        # numpy rounded below).  Genuine coupled instabilities appear as
        # radius >= 1 + 1e-4 per step, so 1e-6 cleanly separates the two.
        stable = lambda d: _spectral_radius_step(
            d, Omega, bf.eps_inf, table) <= 1.0 + 1e-6
        if stable(dt_c):
            continue
        lo, hi = 0.0, dt_c
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if stable(mid):
                lo = mid
            else:
                hi = mid
        dt = min(dt, 0.95 * lo)
    _DT_CACHE[key] = dt
    return dt


class _DFTPlane:
    """DFT accumulator for (Ex, Ey, Hx, Hy) on one integer-k z-plane.
    Real/imag float pairs updated with addr_ (outer-product accumulate)."""

    def __init__(self, torch, k, freqs, Nx, Ny, device, dtype):
        self.k = int(k)
        self.freqs = np.asarray(freqs, float)
        self.w = 2.0 * np.pi * self.freqs          # angular, um/c units
        self.nf = len(freqs)
        self.Np = Nx * Ny
        self.Nx, self.Ny = Nx, Ny
        z = lambda: torch.zeros(self.nf, self.Np, device=device,
                                dtype=dtype)
        self.acc = {c: (z(), z()) for c in ("Ex", "Ey", "Hx", "Hy")}
        # phases in float64: omega*t reaches ~1e4-1e5 rad late in a run,
        # where float32 phase error would corrupt the spectra
        self._wt = torch.as_tensor(self.w, device=device,
                                   dtype=torch.float64)
        self._dtype = dtype

    def _phase(self, torch, t, dt_eff):
        ph = self._wt * t
        return (torch.cos(ph).to(self._dtype) * dt_eff,
                torch.sin(ph).to(self._dtype) * dt_eff)

    def accumulate_E(self, torch, Ex, Ey, t, dt_eff):
        c, s = self._phase(torch, t, dt_eff)
        ex = Ex[:, :, self.k].reshape(-1)
        ey = Ey[:, :, self.k].reshape(-1)
        self.acc["Ex"][0].addr_(c, ex)
        self.acc["Ex"][1].addr_(s, ex)
        self.acc["Ey"][0].addr_(c, ey)
        self.acc["Ey"][1].addr_(s, ey)

    def accumulate_H(self, torch, Hx, Hy, t, dt_eff):
        c, s = self._phase(torch, t, dt_eff)
        k = self.k
        hx = (0.5 * (Hx[:, :, k - 1] + Hx[:, :, k])).reshape(-1)
        hy = (0.5 * (Hy[:, :, k - 1] + Hy[:, :, k])).reshape(-1)
        self.acc["Hx"][0].addr_(c, hx)
        self.acc["Hx"][1].addr_(s, hx)
        self.acc["Hy"][0].addr_(c, hy)
        self.acc["Hy"][1].addr_(s, hy)

    def fields_np(self):
        """Complex (nf, Np) arrays per component (numpy, cpu)."""
        out = {}
        for c, (re, im) in self.acc.items():
            out[c] = (re.detach().cpu().numpy().astype(np.float64)
                      + 1j * im.detach().cpu().numpy().astype(np.float64))
        return out

    def power_probe(self):
        """Per-frequency plane-summed |acc|^2 of Ex+Ey (decay monitor)."""
        q = (self.acc["Ex"][0] ** 2).sum(dim=1) \
            + (self.acc["Ex"][1] ** 2).sum(dim=1) \
            + (self.acc["Ey"][0] ** 2).sum(dim=1) \
            + (self.acc["Ey"][1] ** 2).sum(dim=1)
        return q.detach().cpu().numpy().astype(np.float64)


def _flux_np(fields, dA):
    """Net +z Poynting flux per frequency from a plane's complex DFT
    fields: S(f) = 0.5 * sum_pix Re(Ex Hy* - Ey Hx*) * dA."""
    S = 0.5 * np.sum(
        (fields["Ex"] * np.conj(fields["Hy"])).real
        - (fields["Ey"] * np.conj(fields["Hx"])).real, axis=1) * dA
    return S


class _CPML:
    """One-sided CPML in z for the four z-derivative terms.  side='top'
    covers k in [k0, Nz); side='bottom' covers [0, k1)."""

    def __init__(self, torch, Nz, dz, dt, npml, side, Nx, Ny, device,
                 dtype):
        self.side = side
        self.npml = npml
        # grading coordinate: 0 at the inner PML edge -> 1 at the wall,
        # evaluated at integer and half-integer positions
        kk = np.arange(Nz, dtype=float)
        if side == "top":
            u_int = (kk - (Nz - npml)) / npml
            u_half = (kk + 0.5 - (Nz - npml)) / npml
            self.k0, self.k1 = Nz - npml, Nz
        else:
            u_int = ((npml - 1) - kk) / npml          # deepest at k = 0
            u_half = ((npml - 1) - (kk + 0.5)) / npml
            self.k0, self.k1 = 0, npml
        s_max = CPML_SIGMA_SCALE * (CPML_M + 1) / dz

        def coef(u):
            u = np.clip(u, 0.0, 1.0)
            sig = s_max * u ** CPML_M
            b = np.exp(-(sig + CPML_ALPHA) * dt)
            a = np.where(sig > 0, sig / (sig + CPML_ALPHA) * (b - 1.0), 0.0)
            return b, a

        b_i, a_i = coef(u_int)
        b_h, a_h = coef(u_half)
        sl = slice(self.k0, self.k1)
        as_t = lambda v: torch.as_tensor(v[sl], device=device, dtype=dtype)
        # E z-derivatives are used in H updates at HALF-z positions;
        # H z-derivatives are used in E updates at INTEGER-z positions.
        self.bE, self.aE = as_t(b_h), as_t(a_h)
        self.bH, self.aH = as_t(b_i), as_t(a_i)
        z = lambda: torch.zeros(Nx, Ny, self.k1 - self.k0, device=device,
                                dtype=dtype)
        self.psi_Ex = z()   # for dEx/dz  (in Hy update)
        self.psi_Ey = z()   # for dEy/dz  (in Hx update)
        self.psi_Hx = z()   # for dHx/dz  (in Ey update)
        self.psi_Hy = z()   # for dHy/dz  (in Ex update)

    def correct_E(self, dEx_dz, dEy_dz):
        sl = slice(self.k0, self.k1)
        self.psi_Ex.mul_(self.bE).add_(dEx_dz[:, :, sl] * self.aE)
        self.psi_Ey.mul_(self.bE).add_(dEy_dz[:, :, sl] * self.aE)
        dEx_dz[:, :, sl] += self.psi_Ex
        dEy_dz[:, :, sl] += self.psi_Ey

    def correct_H(self, dHx_dz, dHy_dz):
        sl = slice(self.k0, self.k1)
        self.psi_Hx.mul_(self.bH).add_(dHx_dz[:, :, sl] * self.aH)
        self.psi_Hy.mul_(self.bH).add_(dHy_dz[:, :, sl] * self.aH)
        dHx_dz[:, :, sl] += self.psi_Hx
        dHy_dz[:, :, sl] += self.psi_Hy


class _PoleSet:
    """ADE polarization state for ONE material layer and ONE E component:
        p^{n+1} = c1 p^n + c2 p^{n-1} + c3 wz mask E^n      (per pole)
    restricted to the z-span where the material's fill fraction wz > 0.
    wz is the exact fraction of each Yee cell occupied by this layer
    (z-partial-fill homogenization); mask carves the in-plane holes
    (Si only, binary -- Meep-stage staircase).

    PRECISION NOTE (hard-won): the pole STATE and COEFFICIENTS are kept
    in float64 even when the fields are float32.  In float32, rounding
    of (c1, c2) at small dt perturbs the recurrence's near-marginal
    double root at z = 1 by O(sqrt(eps)) ~ 3e-4 per step, which is a
    slow exponential blow-up that NaNs long runs (measured: band-3
    res-500 planar run, fields dead, max|p| growing 50x per 20 um/c,
    NaN at t ~ 370).  float64 pushes the root splitting to O(1e-8) --
    harmless over any realistic run length.  Cost: the pole tensors
    double in size (~1.6 GB at FULL) and the pole segment of the step
    slows modestly; correctness is not negotiable."""

    def __init__(self, torch, table, dt, k0, k1, wz, mask2d, Nx, Ny,
                 device, dtype):
        self.k0, self.k1 = int(k0), int(k1)
        self.out_dtype = dtype
        pdt = torch.float64
        npoles = len(table)
        w0 = torch.as_tensor(table[:, 0], device=device, dtype=pdt)
        gm = torch.as_tensor(table[:, 1], device=device, dtype=pdt)
        K = torch.as_tensor(table[:, 2], device=device, dtype=pdt)
        den = 1.0 + gm * dt / 2.0
        sh = (npoles, 1, 1, 1)
        self.c1 = ((2.0 - w0 ** 2 * dt ** 2) / den).reshape(sh)
        self.c2 = (-(1.0 - gm * dt / 2.0) / den).reshape(sh)
        self.c3 = (K * dt ** 2 / den).reshape(sh)
        nz = self.k1 - self.k0
        self.wz = torch.as_tensor(
            np.asarray(wz, float), device=device,
            dtype=pdt).reshape(1, 1, nz)
        self.p = torch.zeros(npoles, Nx, Ny, nz, device=device, dtype=pdt)
        self.pm = torch.zeros_like(self.p)
        self.mask = (None if mask2d is None else
                     torch.as_tensor(mask2d, device=device,
                                     dtype=pdt).unsqueeze(-1))

    def step_and_dP(self, E):
        """Advance one step using E^n; return sum_poles (p^{n+1} - p^n)
        on the slab (shape Nx, Ny, nz), cast to the field dtype."""
        Es = E[:, :, self.k0:self.k1].to(self.p.dtype) * self.wz
        if self.mask is not None:
            Es = Es * self.mask
        p_new = self.c1 * self.p + self.c2 * self.pm + self.c3 * Es
        dP = (p_new - self.p).sum(dim=0).to(self.out_dtype)
        self.pm = self.p
        self.p = p_new
        return dP


class _Sim:
    """One FDTD run: fixed band materials, fixed polarization source."""

    def __init__(self, holes_nm, a_super_nm, thickness_nm, buffer_nm,
                 resolution, band_idx, pol, freqs, fits, device, dtype_bits,
                 vacuum=False):
        torch = _torch()
        self.torch = torch
        self.device = device
        self.dtype = torch.float64 if dtype_bits == 64 else torch.float32
        self.vacuum = vacuum
        self.pol = pol

        L = a_super_nm / 1000.0
        self.L = L
        zg = _zgeom(thickness_nm, buffer_nm)
        self.zg = zg
        Nx = max(4, int(round(L * resolution)))
        Ny = Nx
        dx = L / Nx
        dz = 1.0 / resolution
        Nz = int(math.ceil(zg.cz / dz)) + 1
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dz = dx, dz
        self.dA = dx * dx

        self.dt = band_dt(resolution, fits, band_idx)
        self.freqs = np.asarray(freqs, float)

        # ---- z plane indices (integer planes) -----------------------------
        self.k_refl = int(round(zg.z_refl / dz))
        self.k_src = int(round(zg.z_src / dz))
        self.k_apar = int(round(zg.z_apar / dz))
        npml = max(6, int(round(DPML_UM / dz)))
        self.npml = npml
        assert self.k_src < Nz - npml - 1, "source inside top PML"
        assert self.k_refl < self.k_src, "monitor above source"

        # ---- fields --------------------------------------------------------
        zeros = lambda: torch.zeros(Nx, Ny, Nz, device=device,
                                    dtype=self.dtype)
        self.Ex, self.Ey, self.Ez = zeros(), zeros(), zeros()
        self.Hx, self.Hy, self.Hz = zeros(), zeros(), zeros()

        # ---- materials -----------------------------------------------------
        # Exact z-partial-fill layer occupancy: for each E component's Yee
        # cell (Ex,Ey own [k-1/2, k+1/2] dz; Ez owns [k, k+1] dz), the
        # fraction of the cell inside each layer.  This removes the layer-
        # interface staircase (which measurably shifts the Fabry-Perot
        # resonances at production resolutions); arithmetic eps mixing is
        # the correct first-order homogenization for the tangential E that
        # carries normal-incidence light.  In-plane hole boundaries remain
        # BINARY (staircase), matching the Meep-stage approximation.
        bounds = ([] if vacuum else
                  [("ag", 0.0, zg.z_zno0),
                   ("zno", zg.z_zno0, zg.z_si0),
                   ("si", zg.z_si0, zg.z_si1)])

        def fill_profile(name, centers):
            lo = hi = None
            for n, l, h in bounds:
                if n == name:
                    lo, hi = l, h
            c = np.asarray(centers, float)
            a, b = c - dz / 2.0, c + dz / 2.0
            return np.clip((np.minimum(b, hi) - np.maximum(a, lo)) / dz,
                           0.0, 1.0)

        z_int = np.arange(Nz) * dz          # Ex, Ey positions
        z_half = (np.arange(Nz) + 0.5) * dz  # Ez positions
        fills = {}
        if not vacuum:
            for nm, lo_, hi_ in bounds:
                fills[(nm, "int")] = fill_profile(nm, z_int)
                fills[(nm, "half")] = fill_profile(nm, z_half)

        eps_inf = {nm: fits[nm][band_idx].eps_inf
                   for nm in ("si", "zno", "ag")} if not vacuum else {}

        # staggered in-plane sample coordinates (nm) for the Si mask
        L_nm = a_super_nm
        xi = np.arange(Nx) * dx * 1000.0
        xh = (np.arange(Nx) + 0.5) * dx * 1000.0
        yi = np.arange(Ny) * dx * 1000.0
        yh = (np.arange(Ny) + 0.5) * dx * 1000.0
        masks = {}
        if not vacuum:
            masks["Ex"] = _si_mask_np(holes_nm, L_nm, xh, yi)  # (i+1/2, j)
            masks["Ey"] = _si_mask_np(holes_nm, L_nm, xi, yh)  # (i, j+1/2)
            masks["Ez"] = _si_mask_np(holes_nm, L_nm, xi, yi)  # (i, j)

        # inverse eps_inf arrays per E component: air (=1) plus the
        # fill-weighted eps_inf excess of each layer; the Si excess is
        # carved by the in-plane hole mask.
        def build_inv_eps(comp, kind):
            arr = np.ones((Nx, Ny, Nz), dtype=float)
            if not vacuum:
                for nm in ("ag", "zno", "si"):
                    f = fills[(nm, kind)]
                    if nm == "si":
                        add = np.where(masks[comp], 1.0, 0.0)[:, :, None] \
                            * (f * (eps_inf["si"] - 1.0))[None, None, :]
                        # inside holes the Si layer's volume is AIR (=1):
                        # no excess to add there.
                        arr = arr + add
                    else:
                        arr = arr + (f * (eps_inf[nm] - 1.0)
                                     )[None, None, :]
            return torch.as_tensor(1.0 / arr, device=device,
                                   dtype=self.dtype)

        self.inv_eps_x = build_inv_eps("Ex", "int")
        self.inv_eps_y = build_inv_eps("Ey", "int")
        self.inv_eps_z = build_inv_eps("Ez", "half")

        # ADE pole sets per (material, component)
        self.poles = []
        if not vacuum:
            for nm in ("si", "zno", "ag"):
                table = fits[nm][band_idx].ade_pole_table()
                if not len(table):
                    continue
                for comp, kind in (("Ex", "int"), ("Ey", "int"),
                                   ("Ez", "half")):
                    f = fills[(nm, kind)]
                    ks = np.where(f > 0)[0]
                    if not len(ks):
                        continue
                    k0, k1 = int(ks[0]), int(ks[-1]) + 1
                    mask2d = masks[comp] if nm == "si" else None
                    self.poles.append((comp, _PoleSet(
                        torch, table, self.dt, k0, k1, f[k0:k1], mask2d,
                        Nx, Ny, device, self.dtype)))

        # ---- CPML ----------------------------------------------------------
        self.pml = [_CPML(torch, Nz, dz, self.dt, npml, "top", Nx, Ny,
                          device, self.dtype)]
        if vacuum:
            self.pml.append(_CPML(torch, Nz, dz, self.dt, npml, "bottom",
                                  Nx, Ny, device, self.dtype))

        # ---- source --------------------------------------------------------
        fcen = 0.5 * (self.freqs.min() + self.freqs.max())
        bw = max(self.freqs.max() - self.freqs.min(), 0.05)
        self.df = 1.25 * bw
        self.fcen = fcen
        self.src_w = 1.0 / self.df
        self.t0 = 5.0 * self.src_w
        self.t_src_end = 2.0 * self.t0
        self.src_comp = "Ex" if pol == "x" else "Ey"

        # ---- DFT monitors --------------------------------------------------
        T_min = 1.0 / self.freqs.max()
        self.sub = max(1, int(math.floor(
            T_min / (DFT_SAMPLES_PER_PERIOD * self.dt))))
        self.dt_eff = self.sub * self.dt
        self.mon_refl = _DFTPlane(torch, self.k_refl, self.freqs, Nx, Ny,
                                  device, self.dtype)
        self.mon_apar = None
        if not vacuum:
            self.mon_apar = _DFTPlane(torch, self.k_apar, self.freqs, Nx,
                                      Ny, device, self.dtype)

        # ---- source-amplitude carrier + optional compiled cores ----------
        self._g = torch.zeros((), device=device, dtype=self.dtype)
        self._core_h_c = None
        self._core_e_c = None
        if os.environ.get("PC_COMPILE", "0") == "1":
            try:
                # Different sims (norm vs structure, different bands)
                # legitimately need different graphs; the default
                # recompile budget (8) is exhausted by the validation
                # gates alone, after which dynamo silently falls back
                # to eager -- measured as losing the whole 3.6x.
                for attr in ("recompile_limit", "cache_size_limit"):
                    if hasattr(torch._dynamo.config, attr):
                        setattr(torch._dynamo.config, attr, 128)
                self._core_h_c = torch.compile(self._core_h)
                self._core_e_c = torch.compile(self._core_e)
            except Exception as e:          # pragma: no cover
                print(f"    (torch.compile unavailable: {e}; eager)")

    # ---- one leapfrog step --------------------------------------------------
    # Split into two BRANCH-FREE cores so torch.compile can capture each
    # as one fused graph (the step-dependent Python branches -- source
    # window, DFT subsampling -- live in the eager orchestrator _step;
    # a branch on the step count inside the traced region is what made
    # naive compile fail with guard errors).  The source amplitude enters
    # as a 0-dim tensor so its per-step change never retriggers tracing.

    def _core_h(self):
        torch = self.torch
        Ex, Ey, Ez = self.Ex, self.Ey, self.Ez
        dt, dx, dz = self.dt, self.dx, self.dz
        # forward differences; x,y periodic via roll, z zero-padded (PEC)
        dEz_dy = (torch.roll(Ez, -1, 1) - Ez) / dx
        dEy_dz = (torch.cat([Ey[:, :, 1:],
                             torch.zeros_like(Ey[:, :, :1])], 2) - Ey) / dz
        dEx_dz = (torch.cat([Ex[:, :, 1:],
                             torch.zeros_like(Ex[:, :, :1])], 2) - Ex) / dz
        dEz_dx = (torch.roll(Ez, -1, 0) - Ez) / dx
        dEy_dx = (torch.roll(Ey, -1, 0) - Ey) / dx
        dEx_dy = (torch.roll(Ex, -1, 1) - Ex) / dx
        for p in self.pml:
            p.correct_E(dEx_dz, dEy_dz)
        self.Hx.sub_(dt * (dEz_dy - dEy_dz))
        self.Hy.sub_(dt * (dEx_dz - dEz_dx))
        self.Hz.sub_(dt * (dEy_dx - dEx_dy))

    def _core_e(self, g):
        """E-side update; g is a 0-dim tensor holding the soft-source
        amplitude for this step (exactly 0.0 outside the source window,
        so the fused graph is identical before/after the source ends)."""
        torch = self.torch
        Ex, Ey, Ez = self.Ex, self.Ey, self.Ez
        Hx, Hy, Hz = self.Hx, self.Hy, self.Hz
        dt, dx, dz = self.dt, self.dx, self.dz
        dHz_dy = (Hz - torch.roll(Hz, 1, 1)) / dx
        dHy_dz = (Hy - torch.cat([torch.zeros_like(Hy[:, :, :1]),
                                  Hy[:, :, :-1]], 2)) / dz
        dHx_dz = (Hx - torch.cat([torch.zeros_like(Hx[:, :, :1]),
                                  Hx[:, :, :-1]], 2)) / dz
        dHz_dx = (Hz - torch.roll(Hz, 1, 0)) / dx
        dHy_dx = (Hy - torch.roll(Hy, 1, 0)) / dx
        dHx_dy = (Hx - torch.roll(Hx, 1, 1)) / dx
        for p in self.pml:
            p.correct_H(dHx_dz, dHy_dz)

        # ADE polarization currents (before E changes: they use E^n)
        E_of = {"Ex": Ex, "Ey": Ey, "Ez": Ez}
        inv_of = {"Ex": self.inv_eps_x, "Ey": self.inv_eps_y,
                  "Ez": self.inv_eps_z}
        dPs = [ps.step_and_dP(E_of[comp]) for comp, ps in self.poles]

        Ex.add_(self.inv_eps_x * (dt * (dHz_dy - dHy_dz)))
        Ey.add_(self.inv_eps_y * (dt * (dHx_dz - dHz_dx)))
        Ez.add_(self.inv_eps_z * (dt * (dHy_dx - dHx_dy)))
        for (comp, ps), d in zip(self.poles, dPs):
            E_of[comp][:, :, ps.k0:ps.k1].sub_(
                inv_of[comp][:, :, ps.k0:ps.k1] * d)

        # soft current-sheet source (uniform plane wave)
        tgt = Ex if self.src_comp == "Ex" else Ey
        tgt[:, :, self.k_src] -= dt * g

        # PEC walls (tangential E = 0 at both z ends)
        Ex[:, :, 0] = 0.0
        Ey[:, :, 0] = 0.0
        Ex[:, :, -1] = 0.0
        Ey[:, :, -1] = 0.0

    def _source_amp(self, n):
        t_J = (n + 0.5) * self.dt
        if t_J >= self.t_src_end:
            return 0.0
        return math.exp(-((t_J - self.t0) / self.src_w) ** 2 / 2.0) \
            * math.cos(2.0 * math.pi * self.fcen * (t_J - self.t0))

    def _step(self, n):
        torch = self.torch
        (self._core_h_c or self._core_h)()
        dft_now = (n + 1) % self.sub == 0
        if dft_now:
            t_H = (n + 0.5) * self.dt
            self.mon_refl.accumulate_H(torch, self.Hx, self.Hy, t_H,
                                       self.dt_eff)
            if self.mon_apar is not None:
                self.mon_apar.accumulate_H(torch, self.Hx, self.Hy,
                                           t_H, self.dt_eff)
        self._g.fill_(self._source_amp(n))
        (self._core_e_c or self._core_e)(self._g)
        if dft_now:
            t_E = (n + 1) * self.dt
            self.mon_refl.accumulate_E(torch, self.Ex, self.Ey, t_E,
                                       self.dt_eff)
            if self.mon_apar is not None:
                self.mon_apar.accumulate_E(torch, self.Ex, self.Ey,
                                           t_E, self.dt_eff)

    # ---- run to DFT convergence --------------------------------------------
    def run(self, decay_tol, max_time, min_extra_time=20.0):
        """Advance until converged or max_time.  TWO stop criteria, either
        one suffices (both only testable after the source has ended):
          (a) field ring-down: summed E^2 on the monitor planes falls
              below decay_tol^2 x its running peak (decay_tol is an
              AMPLITUDE ratio, Meep stop_when_fields_decayed semantics);
          (b) relative change of the accumulated DFT power < decay_tol
              between checks (Meep stop_when_dft_decayed semantics).
        (a) exists because the float32 DFT accumulators have a rounding
        noise floor (~1e-6 relative change per check) that criterion (b)
        alone can never cross for tight tolerances, even after the fields
        are numerically dead -- measured in the pseudo-1D NIR runs."""
        torch = self.torch
        dt = self.dt
        n_max = int(math.ceil(max_time / dt))
        n_min = int(math.ceil((self.t_src_end + min_extra_time) / dt))
        check_every = max(1, int(round(CHECK_INTERVAL_UM / dt)))
        q_prev = None
        u_peak = 0.0
        hit_cap = True
        planes = [self.mon_refl] + ([self.mon_apar]
                                    if self.mon_apar is not None else [])
        n = 0
        while n < n_max:
            self._step(n)
            n += 1
            if n % check_every == 0:
                u = 0.0
                for m in planes:
                    u += float((self.Ex[:, :, m.k] ** 2).sum()
                               + (self.Ey[:, :, m.k] ** 2).sum())
                u_peak = max(u_peak, u)
                if not math.isfinite(u):
                    raise FloatingPointError(
                        f"NaN/Inf in fields at t={n * dt:.1f}")
                if n < n_min:
                    continue
                if u_peak > 0 and u <= (decay_tol ** 2) * u_peak:
                    hit_cap = False
                    break
                q = np.concatenate([m.power_probe() for m in planes])
                if not np.all(np.isfinite(q)):
                    raise FloatingPointError(
                        f"NaN/Inf in DFT accumulators at t={n * dt:.1f}")
                if q_prev is not None:
                    scale = max(float(q.max()), 1e-300)
                    change = float(np.max(np.abs(q - q_prev))) / scale
                    if change < decay_tol:
                        hit_cap = False
                        break
                q_prev = q
        if self.device == "cuda":
            torch.cuda.synchronize()
        return SimpleNamespace(n_steps=n, t_final=n * dt, hit_cap=hit_cap)


# --------------------------------------------------------------------------
# Normalization (vacuum) runs, cached to disk exactly like the Meep stage
# --------------------------------------------------------------------------
def _norm_tag(n_cells_tag, res, band_idx, pol):
    return f"norm_{n_cells_tag}_res{res}_b{band_idx}_{pol}"


def _norm_meta(sim, wl_band):
    return dict(engine=ENGINE_VERSION, Nx=sim.Nx, Nz=sim.Nz,
                dt=round(sim.dt, 12), sub=sim.sub, k_refl=sim.k_refl,
                n_wl=len(wl_band), wl_sum=float(np.sum(wl_band)))


def _ensure_norm(path, a_super_nm, thickness_nm, buffer_nm, resolution,
                 band_idx, pol, wl_band, fits, device, bits, max_time):
    if os.path.exists(path):
        return
    freqs = 1000.0 / np.asarray(wl_band, float)
    sim = _Sim([], a_super_nm, thickness_nm, buffer_nm, resolution,
               band_idx, pol, freqs, fits, device, bits, vacuum=True)
    sim.run(decay_tol=1e-7, max_time=max_time)
    F = sim.mon_refl.fields_np()
    inc = _flux_np(F, sim.dA)         # negative: incident travels -z
    # atomic write: concurrent shards may build the same (deterministic)
    # cache entry; tmp+rename means readers never see a partial file.
    tmp = path[:-4] + f".tmp{os.getpid()}.npz"
    np.savez_compressed(
        tmp, incident=inc, meta=json.dumps(_norm_meta(sim, wl_band)),
        **{f"{c}_re": F[c].real.astype(np.float32) for c in F},
        **{f"{c}_im": F[c].imag.astype(np.float32) for c in F})
    os.replace(tmp, path)


def _load_norm(path, sim, wl_band):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    want = _norm_meta(sim, wl_band)
    if meta != want:
        raise SystemExit(
            f"norm cache {path} was written with different numerics\n"
            f"  cached: {meta}\n  wanted: {want}\n"
            "Delete the norm_cache directory (or point PC_OUT elsewhere).")
    F = {c: z[f"{c}_re"].astype(np.float64)
         + 1j * z[f"{c}_im"].astype(np.float64)
         for c in ("Ex", "Ey", "Hx", "Hy")}
    return np.asarray(z["incident"], float), F


# --------------------------------------------------------------------------
# One (geometry, band, polarization) run -> (R, A_par) on the band's grid
# --------------------------------------------------------------------------
def _fake_spectra(holes_nm, wl_band, thickness_nm, buffer_nm, fits):
    """Deterministic analytic stand-in (verbatim behavior of the Meep
    stage's fake): planar fitted-TMM + seed-dependent Lorentzian bumps.
    PLUMBING TESTS ONLY."""
    wl = np.asarray(wl_band, float)
    si_a = FitAsMaterial(fits["si"], "si-fit")
    zno_a = FitAsMaterial(fits["zno"], "zno-fit")
    ag_a = FitAsMaterial(fits["ag"], "ag-fit")
    ref = planar_reference_stack(si_a, zno_a, ag_a, wl, thickness_nm,
                                 buffer_nm)
    h = np.asarray(holes_nm) if len(holes_nm) else np.zeros((1, 3))
    seed = int(np.abs(h * 977.0).sum()) % (2 ** 31)
    rng = np.random.default_rng(seed)
    A = np.array(ref["A_si"], dtype=float)
    for _ in range(6):
        l0 = rng.uniform(700, 1090)
        wdt = rng.uniform(3, 12)
        amp = rng.uniform(0.05, 0.5)
        A = A + amp * (wdt / 2) ** 2 / ((wl - l0) ** 2 + (wdt / 2) ** 2)
    A = np.clip(A, 0.0, 0.98)
    A_par = np.clip(np.array(ref["A_par"]) * rng.uniform(0.8, 2.0), 0, 0.2)
    R = np.clip(1.0 - A - A_par, 0.0, 1.0)
    return R, A_par


def run_single(holes_nm, a_super_nm, thickness_nm, buffer_nm, resolution,
               band_idx, pol, wl_band, fits, norm_dir, device, bits,
               decay_tol, max_time, n_cells_tag):
    """One FDTD run.  Returns (R, A_par, runtime_s, hit_cap, n_steps)."""
    t0 = time.time()
    if FAKE:
        R, A_par = _fake_spectra(holes_nm, wl_band, thickness_nm,
                                 buffer_nm, fits)
        return R, A_par, time.time() - t0, False, 0

    os.makedirs(norm_dir, exist_ok=True)
    npath = os.path.join(norm_dir, _norm_tag(n_cells_tag, resolution,
                                             band_idx, pol) + ".npz")
    _ensure_norm(npath, a_super_nm, thickness_nm, buffer_nm, resolution,
                 band_idx, pol, wl_band, fits, device, bits, max_time)

    freqs = 1000.0 / np.asarray(wl_band, float)
    sim = _Sim(holes_nm, a_super_nm, thickness_nm, buffer_nm, resolution,
               band_idx, pol, freqs, fits, device, bits, vacuum=False)
    inc, F_inc = _load_norm(npath, sim, wl_band)
    res = sim.run(decay_tol=decay_tol, max_time=max_time)

    F_tot = sim.mon_refl.fields_np()
    F_scat = {c: F_tot[c] - F_inc[c] for c in F_tot}
    flux_refl = _flux_np(F_scat, sim.dA)           # +z (up): reflected
    F_apar = sim.mon_apar.fields_np()
    flux_apar = _flux_np(F_apar, sim.dA)           # net; negative = down

    R = -flux_refl / inc                           # inc < 0 (down)
    A_par = flux_apar / inc
    return (np.asarray(R, float), np.asarray(A_par, float),
            time.time() - t0, bool(res.hit_cap), res.n_steps)


# --------------------------------------------------------------------------
# Batch interface (same signature spirit as the Meep stage)
# --------------------------------------------------------------------------
_COMPILE_CALLS = [0]


def _maybe_reset_dynamo(every=25):
    """Clear dynamo's graph cache every `every` solve batches: same-shape
    sims reuse cached graphs, but over hundreds of samples the cache can
    only grow, and host memory on a multi-day shard is not free.  A reset
    costs one fresh compile (~30 s) per `every` samples -- noise."""
    if os.environ.get("PC_COMPILE", "0") != "1":
        return
    _COMPILE_CALLS[0] += 1
    if _COMPILE_CALLS[0] % every == 0:
        try:
            import torch
            torch._dynamo.reset()
        except Exception:
            pass


def broadband_absorption_many(holes_list, a_super_nm, thickness_nm,
                              wavelengths_nm, fits, buffer_nm,
                              resolution, decay_tol, max_time,
                              norm_dir, device=None, bits=32,
                              pols=("x", "y"), n_cells_tag=None,
                              progress=None):
    """(A_si, A_par, info) spectra for a LIST of geometries on the shared
    wavelength grid.  Shapes match every earlier stage: (n_geom, n_wl).

    Runs len(BANDS) x len(pols) FDTD solves per geometry, SEQUENTIALLY on
    one device (the GPU is the parallelism; shard across GPUs with the
    driver's --shard flag for more).  info carries per-geometry runtime,
    time-cap flags and step counts."""
    _maybe_reset_dynamo()
    wl = np.asarray(wavelengths_nm, dtype=float)
    device = device or resolve_device(os.environ.get("PC_DEVICE", "auto"))
    band_idx_lists = split_grid_by_band(wl)
    n_cells_tag = n_cells_tag or "sc"

    nG, nW = len(holes_list), len(wl)
    R = np.full((nG, nW), np.nan)
    A_par = np.full((nG, nW), np.nan)
    counts = np.zeros((nG, nW))
    runtime = np.zeros(nG)
    hit_cap = np.zeros(nG, dtype=bool)
    steps = np.zeros(nG)

    tasks = [(g, b, p) for g in range(nG)
             for b in range(len(BANDS)) for p in pols
             if len(band_idx_lists[b])]
    for ti, (g, b, p) in enumerate(tasks):
        idx = band_idx_lists[b]
        Rb, Ab, dt_s, cap, ns = run_single(
            holes_list[g], a_super_nm, thickness_nm, buffer_nm, resolution,
            b, p, wl[idx], fits, norm_dir, device, bits, decay_tol,
            max_time, n_cells_tag)
        R[g, idx] = np.where(np.isnan(R[g, idx]), 0.0, R[g, idx]) + Rb
        A_par[g, idx] = np.where(np.isnan(A_par[g, idx]), 0.0,
                                 A_par[g, idx]) + Ab
        counts[g, idx] += 1
        runtime[g] += dt_s
        hit_cap[g] |= cap
        steps[g] += ns
        if progress:
            progress(ti + 1, len(tasks))
    with np.errstate(invalid="ignore"):
        R = R / counts
        A_par = A_par / counts
    A_si = 1.0 - R - A_par
    info = {"runtime_s": runtime, "hit_time_cap": hit_cap,
            "n_steps": steps, "device": device}
    return A_si, A_par, info


# --------------------------------------------------------------------------
# Validations
# --------------------------------------------------------------------------
def validate_planar_pseudo1d(fits, adapters, thickness_nm, buffer_nm,
                             norm_dir, resolution=500, tol=8e-3,
                             device=None, bits=32, verbose=True, n_wl=7):
    """THE central engine physics gate: the UNPATTERNED stack, solved by
    the full 3D engine on a tiny uniform in-plane cell (physically exactly
    1D), must reproduce the analytic transfer-matrix result computed with
    the SAME FITTED materials -- this exercises the ADE poles, the CPML,
    the source, the norm-run subtraction and both flux monitors, isolated
    from geometry staircasing.  All three channels, every band.

    n_wl is the per-band sample count; the GATE uses the historical 7
    because tol is calibrated against that grid (a denser grid resolves
    the sharp visible-band Fabry-Perot peaks and reports a legitimately
    larger sup-norm -- see config.GATE_PLANAR_TOL).  Pass tol=None for a
    report-only run (errors printed, nothing gated), which is how the
    dense figure grid is produced."""
    if FAKE:
        print("    [FDTD_FAKE=1] pseudo-1D planar gate SKIPPED -- "
              "plumbing mode only.")
        return True, []
    device = device or resolve_device(os.environ.get("PC_DEVICE", "auto"))
    a_tiny = 4000.0 / resolution      # 4 in-plane cells
    ok_all = True
    if verbose:
        print(f"[pseudo-1D planar stack: torch-FDTD vs fitted-material "
              f"TMM, res={resolution}/um, device={device}]")
    results = []
    # the norm cache is keyed on the wavelength grid; keep the historical
    # tag for the default point count so existing caches stay valid
    tag = "v1d" if n_wl == 7 else f"v1dw{n_wl}"
    for b, (lo, hi) in enumerate(BANDS):
        wl_b = np.linspace(lo + 5, hi - 5, n_wl)
        A_si, A_par, _ = broadband_absorption_many(
            [[]], a_tiny, thickness_nm, wl_b, fits, buffer_nm, resolution,
            1e-6, 800.0, norm_dir, device=device, bits=bits, pols=("x",),
            n_cells_tag=tag)
        ref = planar_reference_stack(adapters["si"], adapters["zno"],
                                     adapters["ag"], wl_b, thickness_nm,
                                     buffer_nm)
        R = 1.0 - A_si[0] - A_par[0]
        err = float(np.max(np.abs(A_si[0] - ref["A_si"])))
        results.append((wl_b, A_si[0], A_par[0], ref))
        if tol is None:
            verdict = "(report only, not gated)"
        else:
            ok = err < tol
            ok_all &= ok
            verdict = f"-> {'OK' if ok else 'CHECK'}"
        if verbose:
            print(f"    band {lo:.0f}-{hi:.0f} nm: max|dA_si| = {err:.2e}"
                  f"  (R err {np.max(np.abs(R - ref['R'])):.2e}, "
                  f"A_par err "
                  f"{np.max(np.abs(A_par[0] - ref['A_par'])):.2e})"
                  f"  {verdict}")
    return ok_all, results


def validate_uniform_3d(fits, adapters, a_super_nm, thickness_nm,
                        buffer_nm, resolution, decay_tol, max_time,
                        norm_dir, tol_bands=(8e-2, 2e-2, 1e-2),
                        device=None, bits=32, verbose=True,
                        n_cells_tag="unif", n_wl=4):
    """3D production-code-path check: a hole-free supercell must
    reproduce the fitted-TMM planar stack.  Gates are PER BAND because
    the error is axial grid dispersion inside high-index Si, largest in
    the visible where n reaches 5.6 (measured on this engine at res 120:
    6.1e-2 / 2.9e-3 / 1.7e-4 per band; identical mechanism and order in
    Meep at equal resolution).  Pointwise spectral error at sharp
    Fabry-Perot features, NOT label error -- the broadband ratio label E
    is certified separately by the resolution ladder + ordered-lattice
    anchor in run_timing_test.  tol_bands=None -> report-only (errors
    printed, nothing gated).  Returns (ok, per-band results) exactly like
    validate_planar_pseudo1d."""
    if FAKE:
        print("    [FDTD_FAKE=1] 3D uniform gate SKIPPED.")
        return True, []
    ok = True
    if verbose:
        print(f"[3D uniform slab @ res={resolution}, "
              f"cell={a_super_nm:.0f} nm]")
    results = []
    for b, (lo, hi) in enumerate(BANDS):
        wl = np.linspace(lo + 8, hi - 8, n_wl)
        A_si, A_par, _ = broadband_absorption_many(
            [[]], a_super_nm, thickness_nm, wl, fits, buffer_nm,
            resolution, decay_tol, max_time, norm_dir, device=device,
            bits=bits, pols=("x",), n_cells_tag=n_cells_tag)
        ref = planar_reference_stack(adapters["si"], adapters["zno"],
                                     adapters["ag"], wl, thickness_nm,
                                     buffer_nm)
        err = float(np.max(np.abs(A_si[0] - ref["A_si"])))
        results.append((wl, A_si[0], A_par[0], ref))
        if tol_bands is None:
            if verbose:
                print(f"    band {lo:.0f}-{hi:.0f}: max|dA_si| = "
                      f"{err:.2e}  (report only, not gated)")
            continue
        okb = err < tol_bands[b]
        ok &= okb
        if verbose:
            print(f"    band {lo:.0f}-{hi:.0f}: max|dA_si| = {err:.2e} "
                  f"(gate {tol_bands[b]:.0e})  -> "
                  f"{'OK' if okb else 'CHECK (raise resolution)'}")
    return ok, results


def run_all_validations(fits, adapters, mats, thickness_nm, buffer_nm,
                        resolution_3d, decay_tol, max_time, norm_dir,
                        a_super_3d_nm=650.0, device=None, bits=32):
    """Standard pre-flight block: material fits (label-impact gated),
    pseudo-1D engine-vs-TMM, and the 3D uniform-slab check.  Gate
    numerics come from config (mode-scaled: SMOKE is plumbing-grade)."""
    import config as _C
    from materials_gpu import validate_fits
    ok = validate_fits(fits, adapters, mats, thickness_nm, buffer_nm)
    ok1d, _ = validate_planar_pseudo1d(fits, adapters, thickness_nm,
                                       buffer_nm, norm_dir,
                                       resolution=_C.GATE_PLANAR_RES,
                                       tol=_C.GATE_PLANAR_TOL,
                                       device=device, bits=bits)
    ok &= ok1d
    ok3d, _ = validate_uniform_3d(fits, adapters, a_super_3d_nm,
                                  thickness_nm, buffer_nm, resolution_3d,
                                  decay_tol, max_time, norm_dir,
                                  tol_bands=_C.GATE_3D_TOLS,
                                  device=device, bits=bits)
    ok &= ok3d
    return ok


if __name__ == "__main__":
    # geometry self-tests (pure numpy, no torch needed)
    holes = [(10.0, 10.0, 100.0), (3200.0, 3200.0, 100.0)]
    m = rasterize_mask(holes, 3900.0, 128, 128, 2)
    print(f"rasterizer: fill fraction = {1 - m.mean():.5f} "
          f"(expect ~{2 * np.pi * 100 ** 2 / 3900 ** 2:.5f})")
    xs = (np.arange(312) + 0.5) * 12.5
    mk = _si_mask_np(holes, 3900.0, xs, xs)
    print(f"staggered mask: fill = {1 - mk.mean():.5f} (same target)")
    print("FAKE mode:", FAKE)