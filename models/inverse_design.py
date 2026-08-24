"""Surrogate-based inverse design of disordered photonic-crystal layouts.

Goal (methodology extension, inverse-design stage): at a FIXED disorder
class and strength sigma, search layout space for realizations whose
predicted absorption enhancement E beats the average of random draws at
that same sigma -- then hand the winners to the real FDTD engine for
ground-truth validation.

Pipeline tiers (run any subset; each builds on the last):

  Tier 0  baseline    Score a large batch of ordinary random realizations
                      at (class, sigma) with the surrogate.  Their mean /
                      distribution is the target to beat.  Cheap: surrogate
                      only, no physics.
  Tier 1  screen      Monte-Carlo screening: sample many more random
                      realizations, keep the top fraction by predicted E.
                      Measures the headroom available from lucky draws
                      alone, and seeds Tier 2/3 starts.
  Tier 2  cmaes       Derivative-free CMA-ES over the disorder parameters
                      themselves (per-hole jitter offsets or radius
                      offsets), with the disorder strength PINNED so the
                      optimizer cannot cheat by reducing disorder.
  Tier 3  gradient    Projected gradient ascent through a differentiable
                      soft rasterizer with the surrogate frozen -- fastest
                      convergence, same strength pinning.

Anti-gaming safeguards (the critical caveat):
  * The fitness is a LOWER CONFIDENCE BOUND over the ensemble:
        fitness = mean_over_models(TTA-avg prediction) - kappa * std
    A layout that fools one member into a high E is punished by ensemble
    disagreement; kappa (default 1.0) trades exploration vs robustness.
  * D4 test-time augmentation (8 dihedral views) is averaged into every
    member's prediction -- physically E is D4-invariant, so any view
    spread is model error, not signal.
  * Strength pinning: offsets are projected to a fixed RMS matching the
    random-disorder distribution at sigma (and clamped to its support),
    so every candidate has the SAME disorder strength as the baseline.
  * Feasibility: candidates must satisfy the same etchability wall
    (w_min) and fixed-fill constraints as the training generator
    (scripts/FDTD_solver/disorder.py); violations are penalized during search and
    hard-filtered at output.
  * NOTHING here is a result until the FDTD engine confirms it.  Use
    --export-dir to write the top candidates as sample records the
    physics stack can consume; feed (layout, true E) pairs back into the
    training set (active learning) and re-run with the sharper model.

Interpretation guardrails (AUDIT_resolution_accuracy.md):
  * Within-sigma ranking resolvability floor is 0.30% (audit Test 9:
    N=15, 105 pairs, res-120 referee -- zero flips above 0.30%).  The
    60->120 differential label jitter is 0.126% spread, i.e. per-label
    engine noise ~0.09% of E.  (Superseded: the ~0.5% N=5 floor of
    Test 6 and the 0.8-1% interim floor of Test 7's res-90 referee.)
    Predicted gains below the floor are not claimable even if the
    surrogate is confident.  The report prints every gain next to this
    floor.

Usage
-----
    # everything, with defaults: bundle from runs/surrogate_128_fft_nll_sweep/,
    # top 12 candidates exported to scripts/FDTD_solver/candidates/
    # (where verify_candidates.py picks them up)
    python -m models.inverse_design --disorder-class jitter --sigma 0.10

    # archive an export elsewhere instead of the live candidates/ folder
    python -m models.inverse_design --sigma 0.10 \
        --export-dir runs/inverse/jitter_s010

    # quick CPU smoke test
    python -m models.inverse_design --tiers baseline screen \
        --n-baseline 200 --n-screen 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

# Path setup MUST precede the bare-module imports below (this script is
# run both as `python -m models.inverse_design` and as a plain script).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "FDTD_solver"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "models"))

from model import (PhotonicCNN, D4_TTA_OPS,
                   build_input_channels, infer_bundle_recipe)

try:
    from data_augmentation import rasterize_mask
except ImportError:
    rasterize_mask = None

import disorder  # noqa: E402

import config as PC  # noqa: E402  (solver config; fail loud if missing)


# ==========================================================================
# CLI
# ==========================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m models.inverse_design",
        description="Surrogate-driven inverse design at fixed disorder "
                    "class + strength.")
    ap.add_argument("--bundle",
                    default=os.path.join(_REPO_ROOT, "runs",
                                         "surrogate_128_fft_nll_sweep",
                                         "surrogate_bundle.pt"),
                    help="surrogate_bundle.pt from models/model.py "
                         "(default: <repo>/runs/surrogate_128_fft_nll_sweep/"
                         "surrogate_bundle.pt, the deployed v2).")
    ap.add_argument("--disorder-class", choices=["jitter", "radius"],
                    default="jitter")
    ap.add_argument("--sigma", type=float, default=0.10)
    ap.add_argument("--tiers", nargs="+", default=["baseline", "screen",
                                                   "cmaes", "gradient"],
                    choices=["baseline", "screen", "cmaes", "gradient"])
    # lattice -- defaults come from scripts/config.py (PC_MODE-aware) so
    # the search geometry ALWAYS matches the campaign the surrogate was
    # trained on.  Overriding these is almost always a mistake: a CNN
    # trained on 7x7-supercell rasters scored on 4x4 rasters is out-of-
    # distribution and returns a near-constant (silently useless) E.
    ap.add_argument("--a-nm", type=float, default=PC.A_NM)
    ap.add_argument("--n-cells", type=int, default=PC.N_CELLS)
    ap.add_argument("--r-over-a", type=float, default=PC.R_OVER_A)
    ap.add_argument("--w-min-nm", type=float, default=PC.W_MIN_NM)
    ap.add_argument("--r-min-nm", type=float, default=50.0)
    ap.add_argument("--r-max-frac", type=float, default=0.45)
    # scoring
    ap.add_argument("--kappa", type=float, default=1.0,
                    help="LCB penalty: fitness = mean - kappa*std.")
    ap.add_argument("--calibration", default=None,
                    help="calibration.json from models/calibrate_uq.py. "
                         "None = auto-load <bundle dir>/uq/calibration.json "
                         "if it exists.")
    ap.add_argument("--no-calibration", dest="use_calibration",
                    action="store_false", default=True,
                    help="Use the raw (uncalibrated) ensemble std in the "
                         "LCB, as pre-calibration runs did.")
    ap.add_argument("--no-tta", dest="tta", action="store_false",
                    default=True)
    ap.add_argument("--batch-size", type=int, default=256)
    # tier sizes
    ap.add_argument("--n-baseline", type=int, default=2000)
    ap.add_argument("--n-screen", type=int, default=20000)
    ap.add_argument("--screen-keep-frac", type=float, default=0.02)
    ap.add_argument("--cmaes-restarts", type=int, default=4)
    ap.add_argument("--cmaes-iters", type=int, default=60)
    ap.add_argument("--cmaes-popsize", type=int, default=0,
                    help="0 = CMA default 4+floor(3 ln d).")
    ap.add_argument("--grad-starts", type=int, default=8)
    ap.add_argument("--grad-steps", type=int, default=200)
    ap.add_argument("--grad-lr", type=float, default=8.0,
                    help="Adam LR in nm on the offset parameters.")
    ap.add_argument("--soft-tau-px", type=float, default=0.35,
                    help="Soft-edge width of the differentiable raster, in "
                         "pixels. 0.35 px matches the exact AA raster to "
                         "corr 0.992 (binarized agreement 0.997) while "
                         "keeping usable gradients; smaller = more exact "
                         "but weaker gradient signal.")
    # output
    ap.add_argument("--export-dir",
                    default=os.path.join(_REPO_ROOT, "scripts",
                                         "FDTD_solver", "candidates"),
                    help="Write top candidates as FDTD-ready sample records "
                         "(default: <repo>/scripts/FDTD_solver/candidates, "
                         "where verify_candidates.py reads them). Use a "
                         "FRESH/empty directory: stale candidate_*.npz from "
                         "an earlier run are not cleaned up and "
                         "verify_candidates.py globs the whole folder "
                         "(audit sec. 11.3: never reuse an out-dir).")
    ap.add_argument("--export-top", type=int, default=12)
    ap.add_argument("--force", action="store_true",
                    help="Proceed even if the surrogate looks blind to the "
                         "baseline (see the OOD guard).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    return ap.parse_args(argv)


# ==========================================================================
# Surrogate scorer (ensemble + TTA + LCB)
# ==========================================================================
class SurrogateScorer:
    """Loads a surrogate bundle and scores batches of layouts.

    Two entry points:
      score_holes(list_of_hole_lists)  exact anti-aliased raster (the same
                                       rasterize_mask the training data used)
      score_images(tensor)             pre-built (B,1,H,W) raster in [0,1] --
                                       used by the differentiable Tier 3 path
                                       (keeps gradients).
    """

    def __init__(self, bundle_path, device, use_tta=True, kappa=1.0,
                 batch_size=256, calibration=None):
        bundle = torch.load(bundle_path, map_location="cpu",
                            weights_only=False)
        fmt = bundle.get("format")
        if fmt not in ("photonic-surrogate-bundle-v1",
                       "photonic-surrogate-bundle-v2"):
            raise ValueError(f"unrecognized bundle format in {bundle_path}")
        # v2 = heteroscedastic NLL head: members output (mu, log sigma^2)
        # and the scorer's std is the mixture total (aleatoric + members'
        # disagreement) instead of disagreement alone
        self.hetero = bool(bundle.get("heteroscedastic",
                                      fmt.endswith("v2")))
        self.device = device
        self.use_tta = use_tta
        self.kappa = kappa
        self.batch_size = batch_size
        self.img_size = int(bundle["img_size"])
        n = bundle["norm"]
        self.x_mean = torch.as_tensor(
            n["x_mean"], dtype=torch.float32).reshape(1, -1, 1, 1).to(device)
        self.x_std = torch.as_tensor(
            n["x_std"], dtype=torch.float32).reshape(1, -1, 1, 1).to(device)
        self.y_mean, self.y_std = n["y_mean"], n["y_std"]
        self.in_ch = int(bundle["arch"].get("input_shape", 1))
        self.recipe = infer_bundle_recipe(bundle)
        if len(self.recipe) != self.in_ch:
            raise ValueError(f"bundle recipe {self.recipe} does not match "
                             f"input_shape {self.in_ch}")
        self.models = []
        for sd in bundle["state_dicts"]:
            m = PhotonicCNN(**bundle["arch"])
            m.load_state_dict(sd)
            m.to(device).eval()
            self.models.append(m)
        # post-hoc UQ calibration (models/calibrate_uq.py): the LCB uses
        # sigma_cal = sqrt(a^2 + (b*std)^2) instead of the raw member
        # disagreement, which is ~4.5x overconfident.  None = raw std.
        self.cal = calibration
        print(f"[scorer] channel recipe: {self.recipe}")
        print(f"[scorer] {len(self.models)}-model ensemble "
              f"({'heteroscedastic v2' if self.hetero else 'v1'}), "
              f"img={self.img_size}px, TTA={'on' if use_tta else 'off'}, "
              f"kappa={kappa}")
        if self.cal:
            print(f"[scorer] UQ calibration: a={self.cal['a']:.6f} "
                  f"b={self.cal['b']:.4f}  ({self.cal['path']})")
        else:
            print("[scorer] UQ calibration: none (raw ensemble std)")

    def _calibrated(self, std):
        """Raw ensemble std -> calibrated sigma (tensor- and numpy-safe)."""
        if not self.cal:
            return std
        a, b = self.cal["a"], self.cal["b"]
        return (a * a + (b * std) ** 2) ** 0.5

    def _forward_images(self, X):
        """(B,1,H,W) in [0,1] -> (mean_E, std_E) de-normalised. Keeps grad.

        If the bundle was trained with --fft-channel, the structure-factor
        channel is recomputed here (torch.fft is differentiable, so the
        Tier-3 gradient path works unchanged)."""
        if len(self.recipe) > 1:
            X = build_input_channels(X, self.recipe)
        Xn = (X - self.x_mean) / self.x_std
        ops = D4_TTA_OPS if self.use_tta else [lambda x: x]
        if self.hetero:
            # per member: TTA-mean mu and law-of-total-variance var; then
            # ensemble as a Gaussian mixture (all smooth ops -> Tier-3
            # gradients flow unchanged)
            mem_mu, mem_var = [], []
            for m in self.models:
                MU, VAR = [], []
                for op in ops:
                    out = m(op(Xn))
                    MU.append(out[..., 0])
                    VAR.append(out[..., 1].clamp(-12.0, 4.0).exp())
                MU = torch.stack(MU)
                mem_mu.append(MU.mean(dim=0))
                mem_var.append(torch.stack(VAR).mean(dim=0)
                               + MU.var(dim=0, unbiased=False))
            MUs = torch.stack(mem_mu)                   # (M, B) normalised
            mean_n = MUs.mean(dim=0)
            var_n = ((torch.stack(mem_var) + MUs ** 2).mean(dim=0)
                     - mean_n ** 2)
            mean = mean_n * self.y_std + self.y_mean
            std = var_n.clamp(min=0).sqrt() * self.y_std
            return mean, std
        per_model = []
        for m in self.models:
            p = torch.stack([m(op(Xn)).squeeze(-1) for op in ops]).mean(dim=0)
            per_model.append(p)
        P = torch.stack(per_model)                      # (M, B) normalised
        P = P * self.y_std + self.y_mean
        mean = P.mean(dim=0)
        std = P.std(dim=0, unbiased=False) if len(self.models) > 1 \
            else torch.zeros_like(mean)
        return mean, std

    @torch.inference_mode()
    def score_holes(self, holes_list, a_super_nm):
        """Exact raster + score. Returns dict of numpy arrays."""
        means, stds = [], []
        for i in range(0, len(holes_list), self.batch_size):
            chunk = holes_list[i:i + self.batch_size]
            imgs = np.stack([
                rasterize_mask(h, a_super_nm, self.img_size, self.img_size,
                               supersample=4).astype(np.float32)
                for h in chunk])
            X = torch.from_numpy(imgs).unsqueeze(1).to(self.device)
            m, s = self._forward_images(X)
            means.append(m.cpu().numpy())
            stds.append(s.cpu().numpy())
        mean = np.concatenate(means)
        std_raw = np.concatenate(stds)
        std = self._calibrated(std_raw)
        return {"mean": mean, "std": std, "std_raw": std_raw,
                "lcb": mean - self.kappa * std}

    def score_images_grad(self, X):
        """Differentiable scoring for Tier 3. Returns (mean, std, lcb).

        The calibration map sqrt(a^2 + (b*s)^2) is smooth in s, so the
        Tier-3 gradient path works unchanged."""
        m, s = self._forward_images(X)
        s = self._calibrated(s)
        return m, s, m - self.kappa * s


# ==========================================================================
# Fixed-strength disorder parameterization
# ==========================================================================
class DisorderSpace:
    """Maps a flat parameter vector <-> a constrained layout at pinned sigma.

    jitter: params are per-hole (dx, dy) in nm, dim = 2*n_holes.
            Random draws use dx,dy ~ U(-sigma*a, sigma*a), whose
            per-component RMS is sigma*a/sqrt(3).  We pin candidates to
            that SAME per-component RMS (projection) and clamp each
            component to the distribution's support [-sigma*a, sigma*a],
            so no candidate is less (or differently) disordered than the
            baseline population, and none leaves the surrogate's training
            support.
    radius: params are per-hole dr in nm, dim = n_holes, pinned to RMS
            sigma*r_nom/sqrt(3), clamped to +/- sigma*r_nom, then the
            generator's own fixed-fill common rescale is applied so the
            absorber volume is identical to every training sample.
    """

    def __init__(self, disorder_class, sigma, a_nm, n_cells, r_over_a,
                 w_min_nm, r_min_nm, r_max_frac):
        self.cls = disorder_class
        self.sigma = float(sigma)
        self.a = float(a_nm)
        self.n_cells = int(n_cells)
        self.r_nom = float(r_over_a * a_nm)
        self.L = self.a * self.n_cells
        self.n_holes = self.n_cells ** 2
        self.w_min = float(w_min_nm)
        self.r_min = float(r_min_nm)
        self.r_max = float(r_max_frac * a_nm)
        ij = np.arange(self.n_cells)
        xx, yy = np.meshgrid((ij + 0.5) * self.a, (ij + 0.5) * self.a,
                             indexing="ij")
        self.centers0 = np.column_stack([xx.ravel(), yy.ravel()])
        self.radii0 = np.full(self.n_holes, self.r_nom)
        self.fill_target = disorder.air_fraction(self.radii0, self.L)
        if self.cls == "jitter":
            self.dim = 2 * self.n_holes
            self.comp_max = self.sigma * self.a           # support half-width
        elif self.cls == "radius":
            self.dim = self.n_holes
            self.comp_max = self.sigma * self.r_nom
        else:
            raise ValueError("inverse design supports jitter/radius only "
                             "(random has no fixed-strength notion)")
        self.rms_target = self.comp_max / np.sqrt(3.0)

    # ---- strength pinning --------------------------------------------
    def project(self, v):
        """Clamp to the disorder support, then rescale to the pinned RMS.

        Alternating projection (3 rounds) onto box /\\ sphere; if heavy
        clamping makes the exact RMS unreachable inside the box, the final
        vector sits on the box boundary at the max reachable RMS -- i.e.
        never MORE than the target, and in practice within <1% of it.
        """
        v = np.asarray(v, dtype=float).copy()
        for _ in range(3):
            v = np.clip(v, -self.comp_max, self.comp_max)
            rms = np.sqrt(np.mean(v ** 2))
            if rms < 1e-12:
                v = np.random.default_rng(0).uniform(
                    -self.comp_max, self.comp_max, size=self.dim)
                continue
            v *= self.rms_target / rms
        return np.clip(v, -self.comp_max, self.comp_max)

    def project_torch(self, v):
        """Same projection, differentiable-safe (used between grad steps)."""
        with torch.no_grad():
            v.clamp_(-self.comp_max, self.comp_max)
            rms = v.pow(2).mean().sqrt().clamp_min(1e-12)
            v.mul_(self.rms_target / rms)
            v.clamp_(-self.comp_max, self.comp_max)
        return v

    # ---- params -> geometry ------------------------------------------
    def layout(self, v):
        """Parameter vector -> (centers, radii) with fixed-fill rescale."""
        v = np.asarray(v, dtype=float)
        if self.cls == "jitter":
            offsets = v.reshape(self.n_holes, 2)
            centers = (self.centers0 + offsets) % self.L
            radii = self.radii0.copy()      # fill already exact
        else:
            centers = self.centers0.copy()
            radii = np.clip(self.radii0 + v, self.r_min, self.r_max)
            scale = np.sqrt(self.fill_target /
                            disorder.air_fraction(radii, self.L))
            radii = radii * scale           # generator's fixed-fill step
        return centers, radii

    def holes(self, v):
        c, r = self.layout(v)
        return [(float(x), float(y), float(rr))
                for (x, y), rr in zip(c, r)]

    # ---- feasibility --------------------------------------------------
    def violation(self, v):
        """Scalar constraint violation in nm (0 = feasible).

        Sum of wall-thickness deficits (minimum-image, same check as
        disorder.violating_holes) + radius-bound deficits after the
        fixed-fill rescale."""
        c, r = self.layout(v)
        _, _, gaps = disorder.pair_gaps(c, r, self.L)
        wall_def = np.clip(self.w_min - gaps, 0.0, None).sum()
        rad_def = (np.clip(self.r_min - r, 0.0, None).sum() +
                   np.clip(r - self.r_max, 0.0, None).sum())
        return float(wall_def + rad_def)

    def random_params(self, rng):
        """One draw from the SAME distribution the training generator uses
        (pre-repair): the honest random-disorder comparison point."""
        return rng.uniform(-self.comp_max, self.comp_max, size=self.dim)


# ==========================================================================
# Compact CMA-ES (maximization, no external dependency)
# ==========================================================================
def cma_es_maximize(f, x0, step0, iters, popsize, seed, project):
    """(mu/mu_w, lambda)-CMA-ES, Hansen's standard parameterization.

    f       : params -> scalar fitness (maximized)
    project : applied to every candidate before evaluation (strength pin)
    Returns (best_x, best_f, history).
    """
    rng = np.random.default_rng(seed)
    d = len(x0)
    lam = popsize if popsize > 0 else 4 + int(3 * np.log(d))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
    cs = (mueff + 2) / (d + mueff + 5)
    c1 = 2 / ((d + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
    chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

    xmean = np.asarray(x0, dtype=float).copy()
    sigma = step0
    pc = np.zeros(d)
    ps = np.zeros(d)
    C = np.eye(d)
    best_x, best_f = xmean.copy(), -np.inf
    hist = []

    for it in range(iters):
        D2, B = np.linalg.eigh(C)
        D = np.sqrt(np.clip(D2, 1e-20, None))
        z = rng.standard_normal((lam, d))
        y = z @ np.diag(D) @ B.T
        xs = xmean + sigma * y
        xs = np.stack([project(x) for x in xs])
        fs = np.array([f(x) for x in xs])

        order = np.argsort(-fs)
        if fs[order[0]] > best_f:
            best_f = float(fs[order[0]])
            best_x = xs[order[0]].copy()
        hist.append(best_f)

        xold = xmean
        # recombination in x-space (projection makes y != (x-m)/sigma exact,
        # so recompute y from the projected points)
        xsel = xs[order[:mu]]
        xmean = w @ xsel
        ysel = (xsel - xold) / sigma
        yw = w @ ysel

        Cinv_sqrt = B @ np.diag(1.0 / D) @ B.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Cinv_sqrt @ yw)
        hsig = (np.linalg.norm(ps) /
                np.sqrt(1 - (1 - cs) ** (2 * (it + 1))) / chiN
                < 1.4 + 2 / (d + 1))
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * yw
        C = ((1 - c1 - cmu) * C
             + c1 * (np.outer(pc, pc) + (not hsig) * cc * (2 - cc) * C)
             + cmu * (ysel.T * w) @ ysel)
        C = (C + C.T) / 2
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = float(np.clip(sigma, 1e-8, 1e8))

    return best_x, best_f, hist


# ==========================================================================
# Tier 3: differentiable soft rasterizer
# ==========================================================================
class SoftRasterizer:
    """Differentiable (B,1,H,W) Si-occupancy raster from hole parameters.

    Soft edges: per-hole occupancy sigmoid((r - dist)/tau); holes combine
    by probabilistic union; Si fraction = product of (1 - occupancy).
    tau ~ 0.7 px approximates the anti-aliased training raster; the final
    candidates are always re-scored through the EXACT rasterize_mask
    before reporting, so soft-vs-exact mismatch cannot leak into results.
    """

    def __init__(self, space: DisorderSpace, img_size, tau_px, device):
        self.space = space
        self.img = img_size
        self.device = device
        L = space.L
        px = L / img_size
        self.tau = tau_px * px
        coords = (torch.arange(img_size, dtype=torch.float32,
                               device=device) + 0.5) * px
        self.xs = coords            # (H,) -- axis 0 in rasterize_mask
        self.ys = coords            # (W,)
        self.L_t = torch.tensor(L, dtype=torch.float32, device=device)
        self.centers0 = torch.tensor(space.centers0, dtype=torch.float32,
                                     device=device)
        self.radii0 = torch.tensor(space.radii0, dtype=torch.float32,
                                   device=device)

    def _min_image(self, d):
        return d - self.L_t * torch.round(d / self.L_t)

    def forward(self, params):
        """params: (B, dim) tensor with grad -> (B,1,H,W) Si image."""
        B = params.shape[0]
        sp = self.space
        if sp.cls == "jitter":
            offsets = params.view(B, sp.n_holes, 2)
            cx = self.centers0[None, :, 0] + offsets[:, :, 0]
            cy = self.centers0[None, :, 1] + offsets[:, :, 1]
            radii = self.radii0[None, :].expand(B, -1)
        else:
            cx = self.centers0[None, :, 0].expand(B, -1)
            cy = self.centers0[None, :, 1].expand(B, -1)
            radii = (self.radii0[None, :] + params).clamp(sp.r_min, sp.r_max)
            fill = np.pi * (radii ** 2).sum(dim=1) / (sp.L ** 2)
            scale = torch.sqrt(
                torch.tensor(sp.fill_target, device=params.device) / fill)
            radii = radii * scale[:, None]
        # (B, n, H) and (B, n, W) min-image displacements
        dx = self._min_image(self.xs[None, None, :] - cx[:, :, None])
        dy = self._min_image(self.ys[None, None, :] - cy[:, :, None])
        d2 = dx[:, :, :, None] ** 2 + dy[:, :, None, :] ** 2   # (B,n,H,W)
        dist = torch.sqrt(d2 + 1e-6)
        occ = torch.sigmoid((radii[:, :, None, None] - dist) / self.tau)
        si = torch.clamp(1.0 - occ, 1e-6, 1.0).log().sum(dim=1).exp()
        return si.unsqueeze(1)                                  # (B,1,H,W)


def tier3_gradient(space, scorer, args, starts, device):
    """Projected Adam ascent on LCB fitness through the soft rasterizer."""
    raster = SoftRasterizer(space, scorer.img_size, args.soft_tau_px, device)
    params = torch.tensor(np.stack([space.project(s) for s in starts]),
                          dtype=torch.float32, device=device,
                          requires_grad=True)
    opt = torch.optim.Adam([params], lr=args.grad_lr)
    for step in range(args.grad_steps):
        opt.zero_grad(set_to_none=True)
        X = raster.forward(params)
        _, _, lcb = scorer.score_images_grad(X)
        loss = -lcb.sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for i in range(params.shape[0]):
                space.project_torch(params[i])
        if step % 50 == 0 or step == args.grad_steps - 1:
            print(f"    [grad] step {step:4d}  best soft-LCB "
                  f"{lcb.max().item():.4f}  mean {lcb.mean().item():.4f}")
    return [space.project(p) for p in
            params.detach().cpu().numpy()]


# ==========================================================================
# Reporting / export
# ==========================================================================
def summarize(name, scores):
    m = scores["mean"]
    print(f"  {name}: n={len(m)}  mean={m.mean():.4f}  std={m.std():.4f}  "
          f"p95={np.percentile(m, 95):.4f}  max={m.max():.4f}")


def export_candidates(out_dir, space, entries, baseline_stats, args,
                      calibration=None):
    """Write FDTD-ready sample records + manifest for the validation run.

    Each candidate_XXXX.npz mirrors the campaign sample schema closely
    enough for the physics stack (holes_xyr_nm + a_super_nm is the solver
    interface); predicted stats ride along for the active-learning ledger.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for i, e in enumerate(entries):
        holes = np.asarray(space.holes(e["params"]), dtype=float)
        path = os.path.join(out_dir, f"candidate_{i:04d}.npz")
        np.savez(path,
                 holes_xyr_nm=holes,
                 a_super_nm=space.L,
                 disorder_class=space.cls,
                 sigma=space.sigma,
                 method=e["method"],
                 pred_E_mean=e["mean"],
                 pred_E_std=e["std"],
                 pred_E_lcb=e["lcb"],
                 fill_achieved=disorder.air_fraction(holes[:, 2], space.L),
                 params=np.asarray(e["params"], dtype=float))
        manifest.append({"file": os.path.basename(path), "method": e["method"],
                         "pred_E_mean": round(float(e["mean"]), 5),
                         "pred_E_std": round(float(e["std"]), 5),
                         "pred_E_lcb": round(float(e["lcb"]), 5)})
    meta = {
        "disorder_class": space.cls, "sigma": space.sigma,
        "a_nm": space.a, "n_cells": space.n_cells,
        "r_nom_nm": space.r_nom, "w_min_nm": space.w_min,
        "kappa": args.kappa,
        "screen_keep_frac": args.screen_keep_frac,
        "bundle": os.path.abspath(args.bundle),
        "calibration": calibration,
        "baseline": baseline_stats,
        "note": ("Predicted values only. Validate every candidate with the "
                 "FDTD engine before claiming anything; gains below the "
                 "0.30% within-sigma resolvability floor (audit Test 9, "
                 "res-120 referee) are not claimable. Feed (layout, true E) "
                 "pairs back into training (active learning) and re-run "
                 "the search."),
        "candidates": manifest,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] {len(entries)} candidates + manifest -> {out_dir}")


# ==========================================================================
# Main
# ==========================================================================
def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device) if args.device else torch.device(
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    calibration = None
    if args.use_calibration:
        cal_path = args.calibration or os.path.join(
            os.path.dirname(os.path.abspath(args.bundle)), "uq",
            "calibration.json")
        if os.path.exists(cal_path):
            with open(cal_path) as f:
                cal = json.load(f)
            if cal.get("format") != "photonic-uq-calibration-v1":
                raise SystemExit(f"unrecognized calibration format in "
                                 f"{cal_path}")
            # a calibration is fit to ONE bundle's raw std -- never apply
            # it to a different bundle (e.g. v1-fitted a,b onto the
            # natively-calibrated v2 sigma)
            fit_for = os.path.abspath(cal.get("bundle", ""))
            if fit_for != os.path.abspath(args.bundle):
                if args.calibration:
                    raise SystemExit(
                        f"calibration {cal_path} was fit for bundle "
                        f"{fit_for}, not {args.bundle}")
                print(f"[cal] skipping {cal_path}: fit for a different "
                      "bundle (raw std will be used)")
            else:
                calibration = {"a": float(cal["a"]), "b": float(cal["b"]),
                               "path": os.path.abspath(cal_path)}
        elif args.calibration:
            raise SystemExit(f"calibration file not found: {cal_path}")

    scorer = SurrogateScorer(args.bundle, device, use_tta=args.tta,
                             kappa=args.kappa, batch_size=args.batch_size,
                             calibration=calibration)
    space = DisorderSpace(args.disorder_class, args.sigma, args.a_nm,
                          args.n_cells, args.r_over_a, args.w_min_nm,
                          args.r_min_nm, args.r_max_frac)
    print(f"[space] {space.cls} sigma={space.sigma}  dim={space.dim}  "
          f"pinned per-component RMS={space.rms_target:.2f} nm  "
          f"support +/-{space.comp_max:.2f} nm")
    print(f"[space] geometry: {args.n_cells}x{args.n_cells} supercell, "
          f"a={args.a_nm:g} nm (a_super={space.L:g} nm), "
          f"r_nom={space.r_nom:g} nm  [defaults from config mode "
          f"{PC.MODE}]")

    floor_note = ("(audit Test 9: within-sigma resolvability floor 0.30%, "
                  "per-label engine noise ~0.09%)")
    all_candidates = []   # dicts: params/method/mean/std/lcb
    baseline_stats = None

    # -------------------- Tier 0: baseline ---------------------------
    if "baseline" in args.tiers:
        print(f"\n== Tier 0: baseline ({args.n_baseline} generator draws) ==")
        recs = []
        for k in range(args.n_baseline):
            recs.append(disorder.make_layout(
                space.cls, space.sigma, seed=int(args.seed * 10 ** 6 + k),
                a_nm=space.a, n_cells=space.n_cells, r_nm=space.r_nom,
                w_min_nm=space.w_min, r_min_nm=space.r_min,
                r_max_frac=args.r_max_frac))
        holes = [r["holes"] for r in recs]
        base = scorer.score_holes(holes, space.L)
        summarize("baseline", base)
        baseline_stats = {
            "n": args.n_baseline,
            "pred_E_mean": float(base["mean"].mean()),
            "pred_E_std": float(base["mean"].std()),
            "pred_E_p95": float(np.percentile(base["mean"], 95)),
            "pred_E_max": float(base["mean"].max()),
        }
        # ---- OOD blindness guard -------------------------------------
        # In-distribution, the surrogate's predicted spread across random
        # draws at fixed sigma tracks the real within-sigma spread
        # (~0.2-0.3% of E).  A predicted spread far below that means the
        # model is returning a near-constant -- the classic symptom of
        # out-of-distribution inputs (e.g. search geometry != training
        # campaign geometry).  Optimizing a flat landscape produces
        # confident garbage, so stop rather than warn.
        rel_spread = (baseline_stats["pred_E_std"]
                      / max(abs(baseline_stats["pred_E_mean"]), 1e-9))
        if rel_spread < 5e-4 and not args.force:
            raise SystemExit(
                f"\n[OOD GUARD] predicted spread across {args.n_baseline} "
                f"random baseline layouts is {100 * rel_spread:.3f}% of "
                "the mean -- the surrogate is effectively blind to these "
                "inputs (in-distribution it should be ~0.2-0.3%).\n"
                "Most likely cause: search geometry does not match the "
                "training campaign.  Currently searching "
                f"n_cells={args.n_cells}, a={args.a_nm:g} nm, "
                f"r/a={args.r_over_a:g} (config mode {PC.MODE}); confirm "
                "these match the bank the surrogate was trained on.\n"
                "Re-run with --force only if you are certain this is "
                "intended.")
    else:
        raise SystemExit("Tier 0 (baseline) is required: it defines the "
                         "average you are trying to beat.")
    base_mean = baseline_stats["pred_E_mean"]

    def register(params_list, method):
        params_list = [p for p in params_list if space.violation(p) == 0.0]
        if not params_list:
            print(f"  [{method}] no feasible candidates")
            return
        sc = scorer.score_holes([space.holes(p) for p in params_list],
                                space.L)
        for p, m, s, l in zip(params_list, sc["mean"], sc["std"], sc["lcb"]):
            all_candidates.append({"params": p, "method": method,
                                   "mean": float(m), "std": float(s),
                                   "lcb": float(l)})
        i = int(np.argmax(sc["lcb"]))
        gain = (sc["mean"][i] - base_mean) / base_mean * 100
        print(f"  [{method}] best LCB={sc['lcb'][i]:.4f} "
              f"mean={sc['mean'][i]:.4f} (+{gain:.2f}% vs baseline mean) "
              f"{floor_note}")

    # -------------------- Tier 1: MC screening ------------------------
    screen_top_params = []
    if "screen" in args.tiers:
        print(f"\n== Tier 1: MC screening ({args.n_screen} draws) ==")
        params = [space.random_params(rng) for _ in range(args.n_screen)]
        feasible = [p for p in params if space.violation(p) == 0.0]
        print(f"  feasible: {len(feasible)}/{len(params)}")
        sc = scorer.score_holes([space.holes(p) for p in feasible], space.L)
        keep = max(1, int(len(feasible) * args.screen_keep_frac))
        order = np.argsort(-sc["lcb"])[:keep]
        screen_top_params = [feasible[i] for i in order]
        register(screen_top_params[:keep], "screen")

    # -------------------- Tier 2: CMA-ES ------------------------------
    if "cmaes" in args.tiers:
        print(f"\n== Tier 2: CMA-ES ({args.cmaes_restarts} restarts x "
              f"{args.cmaes_iters} iters) ==")

        def fitness(v):
            viol = space.violation(v)
            if viol > 0:
                return -1e3 - viol          # graded infeasibility penalty
            return float(scorer.score_holes([space.holes(v)],
                                            space.L)["lcb"][0])

        cmaes_best = []
        for r in range(args.cmaes_restarts):
            if screen_top_params and r < len(screen_top_params):
                x0 = screen_top_params[r]       # warm start from screening
            else:
                x0 = space.project(space.random_params(rng))
            bx, bf, _ = cma_es_maximize(
                fitness, x0, step0=0.3 * space.rms_target,
                iters=args.cmaes_iters, popsize=args.cmaes_popsize,
                seed=args.seed + 1000 + r, project=space.project)
            print(f"  restart {r}: best fitness {bf:.4f}")
            cmaes_best.append(bx)
        register(cmaes_best, "cmaes")

    # -------------------- Tier 3: gradient ascent ----------------------
    if "gradient" in args.tiers:
        print(f"\n== Tier 3: gradient ascent ({args.grad_starts} starts x "
              f"{args.grad_steps} steps) ==")
        starts = []
        pool = ([c["params"] for c in
                 sorted(all_candidates, key=lambda c: -c["lcb"])]
                or [space.random_params(rng) for _ in range(args.grad_starts)])
        for i in range(args.grad_starts):
            starts.append(pool[i] if i < len(pool)
                          else space.random_params(rng))
        finals = tier3_gradient(space, scorer, args, starts, device)
        register(finals, "gradient")

    # -------------------- report + export ------------------------------
    if not all_candidates:
        print("\nno candidates produced (baseline only).")
        return 0

    all_candidates.sort(key=lambda c: -c["lcb"])
    print("\n== Top candidates (by ensemble LCB; predicted, unvalidated) ==")
    print(f"  baseline mean = {base_mean:.4f}, "
          f"baseline p95 = {baseline_stats['pred_E_p95']:.4f}")
    for i, c in enumerate(all_candidates[:args.export_top]):
        gain = (c["mean"] - base_mean) / base_mean * 100
        print(f"  #{i:2d} [{c['method']:8s}] mean={c['mean']:.4f} "
              f"std={c['std']:.4f} lcb={c['lcb']:.4f}  "
              f"gain vs baseline mean = +{gain:.2f}%")
    print(f"  {floor_note}")
    print("  NEXT STEP: run these through the FDTD engine; only "
          "ground-truth E counts. Feed results back into training.")

    if args.export_dir:
        export_candidates(args.export_dir, space,
                          all_candidates[:args.export_top],
                          baseline_stats, args, calibration=calibration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
