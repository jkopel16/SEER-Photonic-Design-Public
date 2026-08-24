"""
config.py   (GPU-FDTD stage)
---------
SINGLE SOURCE OF TRUTH for the frozen design, the layer stack, and the
numerical settings shared by every driver in this folder
(run_timing_test.py, run_dataset.py).  Import from here; never re-declare
these constants in a driver.

The unit-cell ordered lattice (Calculated previously via RCWA as E = 2.547) is the
cross-engine anchor that run_timing_test.py checks explicitly.

LAYER STACK (unchanged):   air / patterned c-Si (300 nm) / ZnO (80 nm) /
Ag (opaque).  

ABSORPTION BOOKKEEPING (unchanged, THE label):

    A_Si(lambda) = 1 - R - A_par        (absorption in the silicon ONLY)

with R from a reflection monitor whose incident fields are subtracted via a
cached normalization (vacuum) run, and A_par the net flux entering the
lossless ZnO buffer (everything below it dies in the Ag).  E divides the
AM1.5G-photon-flux-weighted A_Si of the pattern by that of the unpatterned
slab on the same stack, both evaluated with the SAME FITTED materials.

FDTD NUMERICS (same knobs, same meanings as the Meep stage):
  RESOLUTION   grid resolution in pixels/um -- THE convergence knob
               (hole-edge staircasing; binary geometry, no subpixel
               averaging in dispersive media, exactly like the Meep runs).
  DECAY_TOL    ring-down stop: the run ends when the accumulated DFT
               spectra at the monitors stop changing (relative change per
               check interval < DECAY_TOL), so high-Q structures
               automatically ring down longer.
  MAX_TIME     hard cap (in um/c time units, same scale as Meep time);
               hitting it is recorded per sample and quarantine-checked.
  BANDS        3 sub-band material fits x 2 polarizations = 6 runs per
               structure, stitched on the shared wavelength grid.

GPU KNOBS (new in this stage; all via environment variables):
  PC_DEVICE      "cuda" (default if available), "cpu" (slow -- pathfinding
                 and container smoke tests only).
  PC_PRECISION   "32" (default; standard for FDTD, 2x memory bandwidth) or
                 "64" (paranoia runs).
  PC_COMPILE     "1" to wrap the inner timestep in torch.compile (can give
                 a further speedup on the cluster; default "0" = eager,
                 which is the fully-tested path).
  FDTD_FAKE      "1" replaces every solver call with the fast analytic
                 stand-in (planar fitted-TMM + seeded resonance bumps):
                 full-pipeline plumbing tests with zero physics cost and
                 no GPU needed.  Never bank FAKE spectra.

MODES: set the environment variable PC_MODE --
    FULL  (default) : production numerics for the cluster GPU.
    FAST            : 6x6, reduced resolution -- pathfinding.  Curve
                      SHAPES meaningful, absolute E not converged.
    SMOKE           : tiny everything (3x3) -- plumbing tests only.
"""

from __future__ import annotations

import os
import numpy as np

# ==========================================================================
# Frozen DESIGN
# ==========================================================================
A_NM = 650.0            # lattice constant (nm)
THICKNESS_NM = 300.0    # patterned c-Si slab thickness (nm)
R_OVER_A = 0.35         # nominal hole radius / lattice constant
                        # Audit Test 4
                        # re-swept r/a on the production torch-FDTD engine
                        # (tests/sweep_ra.py): BROAD PLATEAU over
                        # 0.325-0.35, frozen design inside it, 7-11%
                        # falloff on the flanks.  The honest claim is
                        # "inside a plateau", not "the argmax".
W_MIN_NM = 50.0         # minimum etchable silicon wall
BUFFER_NM = 80.0        # ZnO buffer thickness (nm); 0.0 -> bare Si/Ag

# Wavelength window (c-Si PV band, bounded by the tabulated Si data)
WL_MIN, WL_MAX = 400.0, 1100.0
NIR_SPLIT = 700.0       # boundary of the coarse-visible / dense-NIR grid

# ==========================================================================
# Numerics presets
# ==========================================================================
MODE = os.environ.get("PC_MODE", "FULL").upper()

if MODE == "FULL":
    N_CELLS = 7        # supercell = N x N unit cells (Oskooi scale)
    RESOLUTION = 60     # px/um (12.5 nm/px); the Meep-stage starting point
    DECAY_TOL = 3e-4
    MAX_TIME = 2500.0   # hard ring-down cap (um/c)
    VIS_STEP_NM = 12.0  # 400-700 nm grid step (broad features)
    NIR_STEP_NM = 2.5   # 700-1100 nm step (narrow-resonance band)
elif MODE == "FAST":
    N_CELLS = 6
    RESOLUTION = 50
    DECAY_TOL = 1e-4
    MAX_TIME = 600.0
    VIS_STEP_NM = 20.0
    NIR_STEP_NM = 6.0
elif MODE == "SMOKE":
    N_CELLS = 3
    RESOLUTION = 24
    DECAY_TOL = 1e-3
    MAX_TIME = 120.0
    VIS_STEP_NM = 75.0
    NIR_STEP_NM = 40.0
else:
    raise ValueError(f"PC_MODE must be FULL, FAST or SMOKE, got {MODE!r}")

# ---- experiment-scoping overrides (set BEFORE launching; the meta.json
# guard keys on the resulting numerics, so point PC_OUT at a fresh
# directory whenever these change) ------------------------------------
#   PC_N_CELLS      supercell size N (e.g. 8 -> 8x8)
#   PC_RESOLUTION   grid resolution in px/um (e.g. 60)
if os.environ.get("PC_N_CELLS"):
    N_CELLS = int(os.environ["PC_N_CELLS"])
if os.environ.get("PC_RESOLUTION"):
    RESOLUTION = int(os.environ["PC_RESOLUTION"])

A_SUPER_NM = N_CELLS * A_NM

# Unit-cell numerics for the ordered-lattice anchor check (the ordered
# lattice is exactly periodic with period a, so it is solved on ONE unit
# cell -- tiny, so it affords a finer grid than production).
RESOLUTION_UNIT = {"FULL": 120, "FAST": 80, "SMOKE": 32}[MODE]

# Rasterization of layout images for figures / the ML dataset (pure numpy;
# unrelated to the FDTD grid)
SUPERSAMPLE = 4
IMG_SIZE = 128          # the CNN input raster

BASE_SEED = 20260709    # everything derives deterministically from this

# ==========================================================================
# GPU / engine knobs
# ==========================================================================
DEVICE = os.environ.get("PC_DEVICE", "auto")      # auto|cuda|cpu
PRECISION = int(os.environ.get("PC_PRECISION", "32"))
COMPILE = os.environ.get("PC_COMPILE", "0") == "1"

# Where all persistent output goes (data, figures, logs, norm cache).
# Default: the production bank next to this file, so every stage --
# run_dataset subcommands, verify_candidates, tests/ -- finds the same
# bank from any CWD.  Override with PC_OUT for scratch/smoke runs.
OUT_DIR = os.environ.get("PC_OUT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data_production")


# ==========================================================================
# Wavelength grid builder (coarse visible + dense NIR) -- byte-identical
# to the Meep stage (FDTD has no Rayleigh-anomaly singularities).
# ==========================================================================
def raw_wavelength_grid(vis_step_nm: float = None,
                        nir_step_nm: float = None) -> np.ndarray:
    """Non-uniform sweep grid: coarse 400-700 nm + dense 700-1100 nm."""
    vs = VIS_STEP_NM if vis_step_nm is None else vis_step_nm
    ns = NIR_STEP_NM if nir_step_nm is None else nir_step_nm
    vis = np.arange(WL_MIN, NIR_SPLIT, vs)
    nir = np.arange(NIR_SPLIT, WL_MAX + 1e-9, ns)
    return np.concatenate([vis, nir])


def describe() -> str:
    fake = "  [FDTD_FAKE]" if os.environ.get("FDTD_FAKE", "0") == "1" \
        else ""
    return (f"mode={MODE}  solver=torch-FDTD(GPU)  "
            f"supercell={N_CELLS}x{N_CELLS} (a_super={A_SUPER_NM:.0f} nm)  "
            f"res={RESOLUTION}/um  decay_tol={DECAY_TOL:g}  "
            f"cap={MAX_TIME:g}  grid={VIS_STEP_NM:g}/{NIR_STEP_NM:g} nm  "
            f"stack=air/Si({THICKNESS_NM:.0f})/ZnO({BUFFER_NM:.0f})/Ag  "
            f"device={DEVICE} fp{PRECISION}{fake}")


# --------------------------------------------------------------------------
# Validation-gate numerics (mode-scaled: SMOKE is a plumbing mode and its
# coarse grids cannot meet production accuracy; gates loosen accordingly).
# Measured on this engine (pseudo-1D vs fitted-TMM, worst band = visible):
#   res 200 ~ 2.8e-2, res 300 ~ 1.3e-2, res 500 ~ 3.2e-3
# and 3D-uniform per band at RESOLUTION_UNIT:
#   res 120 -> 6.1e-2 / 2.9e-3 / 1.7e-4;  res 80 -> 1.3e-1 / ...;
#   res 32 (SMOKE) -> visible band ~4e-1 (numerical dispersion in Si).
# --------------------------------------------------------------------------
GATE_PLANAR_RES = {"FULL": 500, "FAST": 500, "SMOKE": 200}[MODE]
GATE_PLANAR_TOL = {"FULL": 8e-3, "FAST": 8e-3, "SMOKE": 6e-2}[MODE]
GATE_3D_TOLS = {"FULL": (8e-2, 2e-2, 1e-2),
                "FAST": (2e-1, 5e-2, 3e-2),
                "SMOKE": (6e-1, 3e-1, 2e-1)}[MODE]