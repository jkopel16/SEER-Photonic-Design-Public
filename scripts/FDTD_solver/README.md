# GPU FDTD stage (stage v5) -- PyTorch engine for the dataset campaign

## The verdict on "FDTD on GPU"

**Meep has no GPU support.** Its own FAQ is explicit: pymeep runs on
CPUs/MPI only, and there is no CUDA build to enable. So "run the Meep
scripts on an SCC GPU node" is not a thing -- a GPU run needs a different
engine. This folder is that engine: a purpose-built PyTorch FDTD solver
(`fdtd_torch.py`) that reproduces the Meep stage's physics, bookkeeping,
file formats and public API, so everything downstream (labels, manifest,
figures, the ML phase) is unchanged.

Alternatives considered and rejected for this project: `fdtdx` (JAX GPU
FDTD, actively developed, has the needed ADE dispersion + periodic
boundaries -- a legitimate option, but fixed-length JAX loops fight our
adaptive ring-down stopping and it could not be pre-validated as
thoroughly as in-house code); Tidy3D (excellent but a paid cloud
service). If the custom engine ever becomes a maintenance burden, fdtdx
is the fallback to evaluate first.

## What is identical to the Meep stage (by construction)

- `material_fits.json` (byte-identical; sha256-guarded against drift of
  the material CSVs in `data/materials/` at the repo root)
- the wavelength grids, AM1.5G weighting, and the label
  `E = eta(pattern)/eta(planar fitted-TMM reference)`
- the layout generator (`disorder.py`, verbatim) and the deterministic
  manifest: same `sample_id -> (class, sigma, seed)` formulas
- the per-sample npz schema, quarantine gates, `meta.json` numerics
  guard, `dataset.npz` / `labels.csv` / figs 5-13
- the z-stack geometry constants (PML gap budget, Ag-on-PEC termination,
  norm runs with a bottom PML)

## What was validated IN ADVANCE (in-container, CPU torch)

1. **Material gates** -- identical ledger numbers to the CPU stage;
   planar label gate -0.82% (1% gate), trapping gate -0.48% (2.5%).
2. **Pseudo-1D planar stack vs analytic fitted-material TMM** (the
   engine-physics gate: ADE poles, CPML, source, norm subtraction, both
   flux monitors): max|dA_si| = 3.2e-3 at res 500 (gate 8e-3); per-band
   at res 300: 1.3e-2 / 6.3e-4 / 1.2e-4.
3. **3D uniform slab** on the production code path: error identical to
   the pseudo-1D value at equal resolution (verified to 4 digits at res
   80), i.e. purely axial numerics, nothing added by the 3D machinery.
4. **Patterned sanity anchor** (ordered unit cell, res 40, single pol,
   coarse grid): E = 2.79 vs the RCWA anchor 2.547 (+9.5% at
   plumbing-grade numerics), spectra physical, energy bookkeeping clean.
   The decisive full-numerics anchor runs in `tests/run_timing_test.py`.
5. **Stability**: the timestep per band is set by an exact von Neumann
   analysis of the coupled Yee+ADE update (bisected spectral radius),
   after a heuristic cap was demonstrated to be unsafe at coarse
   resolution and needlessly slow otherwise.
6. **End-to-end plumbing** (FAKE mode): generate -> resume -> shard
   autodetect -> analyze -> verify, all outputs and logs persisted.

Known accuracy note, stated plainly: at production resolutions the
POINTWISE spectral error in the 400-700 nm band is dominated by axial
grid dispersion inside high-index Si ((k dz)^2/24; n reaches 5.6). This
is second-order FDTD physics, identical in kind and magnitude in Meep at
equal resolution, largely dt-independent (measured), and it rigidly
shifts sharp features rather than corrupting broadband integrals. The
LABEL is certified by the resolution ladder + anchor in the timing test,
not by pointwise spectra.

## Interpretation limits (audit, final 2026-07-26)

- **Within-σ ranking floor: 0.30 %** (Test 9: N=15, 105 pairs, res-120
  referee — zero flips above 0.30 %). Gains below it are not claimable.
- Differential label jitter 60→120: **0.126 % spread** ⇒ per-label
  engine noise **≈ 0.09 %** of E. (The older ±0.23 % was the 60→90
  measurement through the unconverged res-90 rung.)
- **Elite verification at res 120, never res 90** — res 90 sits near a
  band-1 grid-dispersion error sign crossing and is not a converged
  referee (Test 8). Absolute E is V-shaped in resolution; no single
  rung's absolute value is converged — all claims are relative.
- `radius / σ = 0.020` is dropped from the manifest (sub-pixel at
  res 60, 9/10 divergent — Test 5); its sample ids are reserved so no
  other id moves.
- r/a = 0.35 is frozen inside a **broad 0.325–0.35 plateau**, confirmed
  on torch-FDTD (Test 4) — claim "inside a plateau", not "the argmax".

## Install (SCC)

```bash
module load miniconda
conda activate /project/rise-batteries/photonics-fdtd
pip install torch --index-url https://download.pytorch.org/whl/cu121
# (or check `module avail pytorch`; numpy+matplotlib assumed present)
python -c "import torch; print(torch.cuda.is_available())"
```

Copy this folder to the project space. No Meep needed.

## Workflow

```bash
# 0. local smoke of the plumbing (no GPU, no physics cost):
FDTD_FAKE=1 PC_MODE=SMOKE PC_OUT=/tmp/fake python -u run_dataset.py generate --limit 5

# 1. TIMING TEST on one GPU (gates + anchor + ladder + measured ETA):
#    qsub your submit script, or run tests/run_timing_test.py
#    interactively with PC_MODE=FULL
cat data_production/timing_report.txt

# 2. scope the campaign from the measured ETA, then launch the array
#    (PC_SEEDS_PER_SIGMA is the cost lever)

# 3. after the array drains (safe to re-qsub anytime -- it resumes):
python -u run_dataset.py plan          # progress check
python -u run_dataset.py analyze       # dataset.npz + labels.csv + figs
python -u run_dataset.py verify --top 6
```

Everything lands under `PC_OUT` (default: `data_production/` next to
this README, resolved from any CWD):

```
data_production/
  logs/                 mirrored terminal output of every run (survives
                        node wipes -- check here first, always)
  timing_report.txt/.json
  norm_cache/           shared normalization-run DFT fields (~2-3 GB at
                        FULL; keyed+guarded by numerics, auto-rebuilt)
  samples/sample_XXXXXX.npz     one file per solved sample (resumable)
  quarantine/           gated-out samples with their failure reason
  meta.json             numerics guard (aborts on drift)
  dataset.npz  labels.csv  verified.csv
  figs/                 gate/anchor figures + figs 5-13
```

## Knobs (environment)

| var | default | meaning |
|---|---|---|
| `PC_MODE` | `FULL` | FULL / FAST / SMOKE numerics |
| `PC_OUT` | `<this dir>/data_production` | output root / sample bank |
| `PC_DEVICE` | `auto` | `cuda` / `cpu` |
| `PC_SEEDS_PER_SIGMA` | 75 | THE cost lever (1551 samples at default) |
| `PC_N_RANDOM` | 50 | random-class seeds |
| `PC_PRECISION` | 32 | 64 for float64 fields (4x slower, debug only) |
| `FDTD_FAKE` | 0 | 1 = analytic stand-in, plumbing tests only |
| `PC_COMPILE` | 0 | 1 = try torch.compile (experimental) |

## Caveats, honestly

- **This is a from-scratch engine.** The validation chain above is
  strong, but the FULL-numerics anchor on the GPU node is the go/no-go:
  if `tests/run_timing_test.py` reports the anchor outside its gate, stop and
  send me the log.
- Normal incidence only. The oblique scan (`run_oblique_scan.py`) stays
  on the Meep/CPU stage.
- Geometry is binary-staircased in-plane (like the Meep runs); layer
  interfaces use exact z-partial-fill averaging (slightly better than
  the Meep runs).
- `decay_tol` semantics: amplitude ring-down on the monitor planes OR
  relative DFT-power change, whichever converges first (the float32 DFT
  accumulators have a ~2e-6 noise floor; the amplitude criterion is
  what allows tight tolerances).
- SMOKE mode is plumbing-grade: at res 24 the band-3 resonances are
  numerically garbage and samples will (correctly) be quarantined.
  That is the gates working, not a bug.
- One GPU solves one sample at a time; parallelism = shards (task
  array). Batching multiple geometries through one GPU kernel is the
  obvious future speed lever if the ETA disappoints.
