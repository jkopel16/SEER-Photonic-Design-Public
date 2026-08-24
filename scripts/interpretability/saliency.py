#!/usr/bin/env python3
"""Interpretability for the deployed surrogate: Grad-CAM, input-gradient
saliency (real-space and k-space), and two hole-occlusion probes.

Subcommands (each idempotent; skips existing outputs unless --force):

  maps    Grad-CAM on stage4 (mu and log-var heads) + |dE/dC1| (real space)
          + |dE/dC2| (k-space) with radial profile.  Averaged over the 5
          ensemble members and the 8 D4 views (CAMs are inverse-transformed
          back to the input frame before averaging; input gradients land in
          the input frame automatically because the scorer applies the D4 op
          downstream of the input).
  shift   Per-hole positional sensitivity by central finite differences at
          +-delta along x and y.  Radii unchanged -> fill fraction exactly
          preserved (in-distribution probe).  Reports per-hole gradient
          vectors, magnitudes, wall-feasibility flags, and a linearity check
          (delta vs delta/2) on the top holes.
  remove  Per-hole removal importance: fill hole j with Si, re-score.
          Changes fill fraction -- an out-of-generator-distribution probe,
          labeled as such everywhere.
  export  Write the top attributed perturbations + the unperturbed reference
          as a verify_candidates.py-compatible candidate dir, for the FDTD
          spot-check of the attributions (differences of same-grid solves,
          so common-mode grid error cancels).
  figure  Multi-panel figure per layout; annotates verified deltas when
          verify_export/verification.csv exists.

All predictions use the deployed bundle exactly as production does
(5 members, 8-view TTA, ensemble mixture).  Attribution maps explain the
MODEL, not the physics; treat them as consistency checks (Adebayo et al.
2018).  Run with the project env:

  LD_LIBRARY_PATH=/project/rise-batteries/photonics-fdtd/lib \\
  /project/rise-batteries/photonics-fdtd/bin/python3 saliency.py maps
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "ablation"))
import common  # noqa: E402  (inserts REPO and SOLVER_DIR on sys.path)

import torch  # noqa: E402

from models.model import PhotonicCNN, build_input_channels  # noqa: E402
from models.inverse_design import SurrogateScorer  # noqa: E402
from models.data_augmentation import rasterize_mask  # noqa: E402
import disorder  # noqa: E402

REPO = common.REPO
OUT_ROOT = os.path.join(REPO, "runs", "interpretability")
IMG = 128
L_NM = common.A_SUPER_NM                      # 4550.0
W_MIN = common.GEOM["w_min_nm"]               # 50.0
LATTICE_PX = common.GEOM["n_cells"]           # reciprocal-lattice spacing in
                                              # FFT pixels: 1/a = n_cells/L

# (flip, k) grid matching models.model.D4_TTA_OPS ordering.
_D4_FK = [(f, k) for f in (False, True) for k in range(4)]


def d4_op(x, f, k):
    if f:
        x = torch.flip(x, dims=[3])
    return torch.rot90(x, k, dims=[2, 3])


def d4_inv(x, f, k):
    x = torch.rot90(x, -k, dims=[2, 3])
    if f:
        x = torch.flip(x, dims=[3])
    return x


# ---------------------------------------------------------------------------
# Layout loading
# ---------------------------------------------------------------------------
DEFAULT_LAYOUTS = [
    ("champ_v1", os.path.join(REPO, "runs", "inverse", "jitter_s015",
                              "candidate_0003.npz")),
    ("champ_v2", os.path.join(REPO, "runs", "inverse_v2", "jitter_s015",
                              "candidate_0002.npz")),
    ("bank_best", "bank:best"),
    ("bank_median", "bank:median"),
]


def _bank_pick(which):
    """Pick a jitter sigma=0.15 bank layout by label E: 'best' or 'median'."""
    d = np.load(common.DATA_128, mmap_mode="r")
    m = (d["disorder_class"] == "jitter") & np.isclose(d["sigma"], 0.15)
    idx = np.where(m)[0]
    ys = np.asarray(d["y"][idx], dtype=float)
    if which == "best":
        pick = idx[int(np.argmax(ys))]
    elif which == "median":
        pick = idx[int(np.argsort(ys)[len(ys) // 2])]
    else:
        raise SystemExit(f"unknown bank spec 'bank:{which}'")
    sid = int(d["sample_id"][pick])
    path = os.path.join(REPO, "data", "samples", f"sample_{sid:06d}.npz")
    return path, {"sample_id": sid, "E_label": float(d["y"][pick])}


def load_layout(spec):
    """spec: npz path with holes_xyr_nm, or 'bank:best'/'bank:median'.

    Returns dict(holes (49,3) nm, a_super_nm, disorder_class, sigma,
                 source, meta)."""
    meta = {}
    if spec.startswith("bank:"):
        path, meta = _bank_pick(spec.split(":", 1)[1])
    else:
        path = spec
    z = np.load(path, allow_pickle=False)
    if "holes_xyr_nm" not in z:
        raise SystemExit(f"{path}: no holes_xyr_nm key")
    rec = {
        "holes": np.asarray(z["holes_xyr_nm"], dtype=float),
        "a_super_nm": float(z["a_super_nm"]),
        "disorder_class": str(z["disorder_class"]) if "disorder_class" in z
        else "jitter",
        "sigma": float(z["sigma"]) if "sigma" in z else float("nan"),
        "source": path,
        "meta": meta,
    }
    if "pred_E_mean" in z:
        rec["meta"]["pred_E_mean_stored"] = float(z["pred_E_mean"])
    if "E" in z:
        rec["meta"]["E_label"] = float(z["E"])
    return rec


def resolve_targets(args):
    """[(tag, layout_record), ...] from --layout/--tag or the default set."""
    if args.layout:
        tags = args.tag or []
        if tags and len(tags) != len(args.layout):
            raise SystemExit("--tag count must match --layout count")
        out = []
        for i, spec in enumerate(args.layout):
            tag = tags[i] if tags else (
                spec.replace("bank:", "bank_") if spec.startswith("bank:")
                else os.path.basename(os.path.dirname(spec)) + "_"
                + os.path.splitext(os.path.basename(spec))[0])
            out.append((tag, load_layout(spec)))
        return out
    return [(tag, load_layout(spec)) for tag, spec in DEFAULT_LAYOUTS]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def make_scorer(args):
    dev = torch.device(args.device)
    return SurrogateScorer(args.bundle, dev, use_tta=True, kappa=args.kappa,
                           batch_size=args.batch_size, calibration=None)


def raster_tensor(holes, a_super_nm, device):
    r = rasterize_mask(holes, a_super_nm, IMG, IMG, supersample=4)
    return torch.from_numpy(r.astype(np.float32))[None, None].to(device)


def score_base(scorer, rec):
    p = scorer.score_holes([rec["holes"]], rec["a_super_nm"])
    return float(p["mean"][0]), float(p["std"][0])


def _print_base_check(rec, e_base):
    stored = rec["meta"].get("pred_E_mean_stored")
    if stored is not None:
        print(f"  [check] script E_hat={e_base:.6f} vs stored "
              f"pred_E_mean={stored:.6f} (|diff|={abs(e_base - stored):.2e})")
    label = rec["meta"].get("E_label")
    if label is not None:
        print(f"  [info] FDTD label E={label:.6f} (model E_hat={e_base:.6f})")


# ---------------------------------------------------------------------------
# maps
# ---------------------------------------------------------------------------
def gradcam_one(model, xn_view, head):
    """Grad-CAM for one member on one already-transformed input view.

    head: 0 -> mu, 1 -> log sigma^2.  Returns (1,1,16,16)->upsampled 128 map
    in the VIEW frame (caller inverse-transforms)."""
    store = {}

    def fwd_hook(_m, _i, out):
        store["A"] = out
        out.register_hook(lambda g: store.__setitem__("G", g))

    h = model.stage4.register_forward_hook(fwd_hook)
    try:
        out = model(xn_view)
        out[..., head].sum().backward()
    finally:
        h.remove()
    A, G = store["A"], store["G"]                       # (1,C,16,16)
    w = G.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * A).sum(dim=1, keepdim=True)).detach()  # (1,1,16,16)
    return torch.nn.functional.interpolate(
        cam, size=(IMG, IMG), mode="bilinear", align_corners=False)


def member_maps(model, scorer, c1, do_saliency=True):
    """View-averaged maps (input frame) for ONE member.

    Returns dict(cam_mu, cam_s, sal_c1, sal_c2) as (IMG, IMG) float64."""
    out = {k: np.zeros((IMG, IMG), dtype=np.float64)
           for k in ("cam_mu", "cam_s", "sal_c1", "sal_c2")}
    for f, k in _D4_FK:
        with torch.enable_grad():
            # --- Grad-CAM (mu and s heads), view frame -> input frame
            x = build_input_channels(c1, scorer.recipe)
            xn = (x - scorer.x_mean) / scorer.x_std
            xv = d4_op(xn, f, k).detach()
            for head, key in ((0, "cam_mu"), (1, "cam_s")):
                cam = gradcam_one(model, xv, head)
                out[key] += d4_inv(cam, f, k)[0, 0].cpu().numpy() / 8
            if not do_saliency:
                continue
            # --- real-space saliency: C1 leaf, C2 differentiable from it
            c1_leaf = c1.clone().requires_grad_(True)
            x = build_input_channels(c1_leaf, scorer.recipe)
            xn = (x - scorer.x_mean) / scorer.x_std
            model(d4_op(xn, f, k))[..., 0].sum().backward()
            out["sal_c1"] += c1_leaf.grad.abs()[0, 0].cpu().numpy() / 8
            # --- k-space saliency: C2 as its own leaf
            c2 = build_input_channels(
                c1, scorer.recipe)[:, 1:2].detach().requires_grad_(True)
            x = torch.cat([c1, c2], dim=1)
            xn = (x - scorer.x_mean) / scorer.x_std
            model(d4_op(xn, f, k))[..., 0].sum().backward()
            out["sal_c2"] += c2.grad.abs()[0, 0].cpu().numpy() / 8
    return out


def run_maps(args):
    scorer = make_scorer(args)
    dev = scorer.device
    for tag, rec in resolve_targets(args):
        out_dir = os.path.join(OUT_ROOT, tag)
        os.makedirs(out_dir, exist_ok=True)
        out_npz = os.path.join(out_dir, "maps.npz")
        if os.path.exists(out_npz) and not args.force:
            print(f"[maps] {tag}: exists, skipping (--force to redo)")
            continue
        print(f"[maps] {tag} <- {rec['source']}")
        e_base, s_base = score_base(scorer, rec)
        _print_base_check(rec, e_base)

        c1 = raster_tensor(rec["holes"], rec["a_super_nm"], dev)
        n_mem = len(scorer.models)
        cam_mu = np.zeros((n_mem, IMG, IMG), dtype=np.float64)
        cam_s = np.zeros((n_mem, IMG, IMG), dtype=np.float64)
        sal_c1 = np.zeros((n_mem, IMG, IMG), dtype=np.float64)
        sal_c2 = np.zeros((n_mem, IMG, IMG), dtype=np.float64)
        for mi, model in enumerate(scorer.models):
            m = member_maps(model, scorer, c1)
            cam_mu[mi], cam_s[mi] = m["cam_mu"], m["cam_s"]
            sal_c1[mi], sal_c2[mi] = m["sal_c1"], m["sal_c2"]

        mean_cam_mu = cam_mu.mean(axis=0)
        mean_sal_c2 = sal_c2.mean(axis=0)

        # Sanity 1: d4_inv exactly inverts d4_op (validates the CAM
        # inverse-transform bookkeeping).  NOTE deliberately NOT a
        # raster-frame equivariance test: for even-size FFTs, rot90 of the
        # raster misregisters the fftshifted C2 spectrum by one pixel
        # (array center 63.5 vs spectrum center bin 64), so C2(rot(C1)) !=
        # rot(C2(C1)).  Production TTA rotates POST-build in both training
        # and inference, so the pipeline is self-consistent.
        t = torch.randn(1, 1, IMG, IMG)
        rt = max(float((d4_inv(d4_op(t, f, k), f, k) - t).abs().max())
                 for f, k in _D4_FK)
        print(f"  [check] d4 op/inv round-trip: max|err|={rt:.1e} "
              "(expect 0)")
        # Sanity 2: per-view prediction spread (how D4-invariant the model
        # actually is; TTA averages this away in production).
        with torch.no_grad():
            x = build_input_channels(c1, scorer.recipe)
            xn = (x - scorer.x_mean) / scorer.x_std
            ev = [float(scorer.models[0](d4_op(xn, f, k))[..., 0])
                  * scorer.y_std + scorer.y_mean for f, k in _D4_FK]
        print(f"  [check] member-0 per-view E_hat spread: "
              f"{max(ev) - min(ev):.5f} (TTA averages this away)")
        sym_dev = rt

        # radial profile of k-space saliency (fftshift center = IMG//2)
        yy, xx = np.mgrid[0:IMG, 0:IMG]
        rad = np.hypot(yy - IMG // 2, xx - IMG // 2)
        nbin = IMG // 2
        prof = np.zeros(nbin)
        for b in range(nbin):
            m = (rad >= b) & (rad < b + 1)
            prof[b] = mean_sal_c2[m].mean() if m.any() else 0.0

        np.savez(out_npz,
                 raster=c1[0, 0].cpu().numpy(),
                 c2=build_input_channels(
                     c1, scorer.recipe)[0, 1].detach().cpu().numpy(),
                 cam_mu=mean_cam_mu, cam_mu_members=cam_mu,
                 cam_s=cam_s.mean(axis=0),
                 sal_c1=sal_c1.mean(axis=0), sal_c1_members=sal_c1,
                 sal_c2=mean_sal_c2, sal_c2_members=sal_c2,
                 sal_c2_radial=prof,
                 e_base=e_base, s_base=s_base,
                 sym_dev=sym_dev,
                 source=rec["source"])
        # where does k-space sensitivity live, relative to lattice harmonics?
        peaks = np.argsort(prof)[::-1][:5]
        print(f"  [info] k-space radial-profile top bins (px): "
              f"{sorted(peaks.tolist())}  "
              f"(lattice harmonics at {LATTICE_PX}, {2 * LATTICE_PX}, "
              f"{3 * LATTICE_PX}, ...)")
        print(f"  -> {out_npz}")


# ---------------------------------------------------------------------------
# shift
# ---------------------------------------------------------------------------
_DIRS4 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)


def shifted_holes(holes, j, dvec, delta, L):
    h = holes.copy()
    h[j, 0] = (h[j, 0] + dvec[0] * delta) % L
    h[j, 1] = (h[j, 1] + dvec[1] * delta) % L
    return h


def _feasible(holes, L):
    return not disorder.violating_holes(holes[:, :2], holes[:, 2], L, W_MIN)


def _shift_pass(scorer, rec, delta, dirs):
    """Score all per-hole shifts at one delta. Returns (E array (n,d), feas)."""
    holes, L = rec["holes"], rec["a_super_nm"]
    n = len(holes)
    batch, feas = [], np.zeros((n, len(dirs)), dtype=bool)
    for j in range(n):
        for di, dvec in enumerate(dirs):
            h = shifted_holes(holes, j, dvec, delta, L)
            feas[j, di] = _feasible(h, L)
            batch.append(h)
    p = scorer.score_holes(batch, L)
    return p["mean"].reshape(n, len(dirs)), feas


def run_shift(args):
    scorer = make_scorer(args)
    dirs = _DIRS4
    for tag, rec in resolve_targets(args):
        out_dir = os.path.join(OUT_ROOT, tag)
        os.makedirs(out_dir, exist_ok=True)
        out_npz = os.path.join(out_dir, "shift.npz")
        if os.path.exists(out_npz) and not args.force:
            print(f"[shift] {tag}: exists, skipping (--force to redo)")
            continue
        print(f"[shift] {tag} <- {rec['source']}  (delta={args.delta_nm} nm)")
        e_base, _ = score_base(scorer, rec)
        _print_base_check(rec, e_base)
        if not _feasible(rec["holes"], rec["a_super_nm"]):
            print("  [warn] base layout itself violates the wall rule")

        E, feas = _shift_pass(scorer, rec, args.delta_nm, dirs)
        n = len(rec["holes"])
        # central differences: dirs ordered (+x,-x,+y,-y)
        gx = (E[:, 0] - E[:, 1]) / (2 * args.delta_nm)
        gy = (E[:, 2] - E[:, 3]) / (2 * args.delta_nm)
        gmag = np.hypot(gx, gy)                       # dE per nm of shift

        # linearity check at delta/2 on the top holes
        top = np.argsort(gmag)[::-1][:args.linearity_top]
        lin = []
        batch = []
        L = rec["a_super_nm"]
        for j in top:
            for dvec in dirs:
                batch.append(shifted_holes(rec["holes"], j, dvec,
                                           args.delta_nm / 2, L))
        p = scorer.score_holes(batch, L)["mean"].reshape(len(top), len(dirs))
        for r_i, j in enumerate(top):
            gx2 = (p[r_i, 0] - p[r_i, 1]) / args.delta_nm
            gy2 = (p[r_i, 2] - p[r_i, 3]) / args.delta_nm
            g2 = float(np.hypot(gx2, gy2))
            lin.append(g2 / (gmag[j] + 1e-15))
        lin = np.array(lin)
        print(f"  [check] linearity g(delta/2)/g(delta) on top "
              f"{len(top)} holes: median {np.median(lin):.2f} "
              f"(1.0 = linear regime)")

        np.savez(out_npz, holes=rec["holes"], e_base=e_base,
                 delta_nm=args.delta_nm, E_shift=E, feasible=feas,
                 gx=gx, gy=gy, gmag=gmag,
                 linearity_top_idx=top, linearity_ratio=lin,
                 source=rec["source"])
        j = int(np.argmax(gmag))
        print(f"  [info] most position-sensitive hole: #{j} "
              f"|g|={gmag[j]:.2e}/nm -> {args.delta_nm} nm shift moves "
              f"E_hat by ~{gmag[j] * args.delta_nm:.5f}")
        print(f"  -> {out_npz}")


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------
def run_remove(args):
    scorer = make_scorer(args)
    for tag, rec in resolve_targets(args):
        out_dir = os.path.join(OUT_ROOT, tag)
        os.makedirs(out_dir, exist_ok=True)
        out_npz = os.path.join(out_dir, "remove.npz")
        if os.path.exists(out_npz) and not args.force:
            print(f"[remove] {tag}: exists, skipping (--force to redo)")
            continue
        print(f"[remove] {tag} <- {rec['source']}  "
              "(fill-changing probe: out-of-generator-distribution)")
        e_base, _ = score_base(scorer, rec)
        _print_base_check(rec, e_base)
        holes, L = rec["holes"], rec["a_super_nm"]
        n = len(holes)
        batch = [np.delete(holes, j, axis=0) for j in range(n)]
        p = scorer.score_holes(batch, L)
        dE = p["mean"] - e_base                        # E(-j) - E(full)
        dfill = -np.pi * holes[:, 2] ** 2 / L ** 2     # fill change per hole
        np.savez(out_npz, holes=holes, e_base=e_base, dE=dE,
                 dfill=dfill, source=rec["source"])
        j_hurt, j_help = int(np.argmin(dE)), int(np.argmax(dE))
        print(f"  [info] removing hole #{j_hurt} hurts most "
              f"(dE_hat={dE[j_hurt]:+.5f}); removing #{j_help} "
              f"helps most ({dE[j_help]:+.5f})")
        print(f"  -> {out_npz}")


# ---------------------------------------------------------------------------
# export (verifier candidate dir)
# ---------------------------------------------------------------------------
def _fill(holes, L):
    return float(np.pi * np.sum(holes[:, 2] ** 2) / L ** 2)


def run_export(args):
    scorer = make_scorer(args)
    for tag, rec in resolve_targets(args):
        out_dir = os.path.join(OUT_ROOT, tag)
        exp_dir = os.path.join(out_dir, "verify_export")
        man_path = os.path.join(exp_dir, "manifest.json")
        if os.path.exists(man_path) and not args.force:
            print(f"[export] {tag}: exists, skipping (--force to redo)")
            continue
        sh_path = os.path.join(out_dir, "shift.npz")
        rm_path = os.path.join(out_dir, "remove.npz")
        if not (os.path.exists(sh_path) and os.path.exists(rm_path)):
            print(f"[export] {tag}: run shift+remove first, skipping")
            continue
        sh, rm = np.load(sh_path), np.load(rm_path)
        holes, L = rec["holes"], rec["a_super_nm"]
        delta = float(sh["delta_nm"])

        entries = [("reference", holes.copy(),
                    "unperturbed layout (reference solve)")]
        gx, gy, gmag = sh["gx"], sh["gy"], sh["gmag"]
        # COMPOUND shift: every hole moved delta along its own +gradient
        # (one step of real-space gradient ascent).  Single-hole shifts at
        # delta move E_hat by ~1e-4 -- below the per-label solver noise --
        # so the resolvable probe is the whole gradient field at once.
        # Holes whose move breaks the wall rule get their move zeroed.
        h = holes.copy()
        unit = np.stack([gx, gy], axis=1) / (gmag[:, None] + 1e-15)
        h[:, 0] = (h[:, 0] + unit[:, 0] * delta) % L
        h[:, 1] = (h[:, 1] + unit[:, 1] * delta) % L
        frozen = set()
        for _ in range(10):
            bad = disorder.violating_holes(h[:, :2], h[:, 2], L, W_MIN)
            if not bad:
                break
            for j in bad:
                h[j, :2] = holes[j, :2]
                frozen.add(int(j))
        if _feasible(h, L):
            entries.append((
                "occ_shift_all", h,
                f"all holes moved {delta:g} nm along their +grad "
                f"({len(frozen)} frozen for wall feasibility)"))
        else:
            print(f"  [warn] compound shift infeasible even after "
                  f"freezing {len(frozen)} holes; skipped")
        # optional single-hole shifts (default 0: sub-noise individually)
        picked = 0
        for j in np.argsort(gmag)[::-1]:
            if picked >= args.top_shift:
                break
            dvec = np.array([gx[j], gy[j]]) / (gmag[j] + 1e-15)
            hh = shifted_holes(holes, int(j), dvec, delta, L)
            if not _feasible(hh, L):
                continue
            entries.append((
                "occ_shift",
                hh, f"hole {int(j)} shifted {delta:g} nm along +grad "
                    f"(|g|={gmag[j]:.2e}/nm)"))
            picked += 1
        # top removals by |dE|
        for j in np.argsort(np.abs(rm["dE"]))[::-1][:args.top_remove]:
            entries.append((
                "occ_remove",
                np.delete(holes, int(j), axis=0),
                f"hole {int(j)} removed (pred dE={rm['dE'][j]:+.5f}; "
                "fill-changing probe)"))

        p = scorer.score_holes([h for _, h, _ in entries], L)
        os.makedirs(exp_dir, exist_ok=True)
        cands = []
        for i, (method, h, note) in enumerate(entries):
            fn = f"candidate_{i:04d}.npz"
            np.savez(os.path.join(exp_dir, fn),
                     holes_xyr_nm=h, a_super_nm=L,
                     disorder_class=rec["disorder_class"],
                     sigma=rec["sigma"], method=method,
                     pred_E_mean=float(p["mean"][i]),
                     pred_E_std=float(p["std"][i]),
                     pred_E_lcb=float(p["lcb"][i]),
                     fill_achieved=_fill(h, L))
            cands.append({"file": fn, "method": method,
                          "pred_E_mean": float(p["mean"][i]),
                          "pred_E_std": float(p["std"][i]),
                          "pred_E_lcb": float(p["lcb"][i]),
                          "note": note})
        manifest = {
            "disorder_class": rec["disorder_class"],
            "sigma": rec["sigma"],
            "a_nm": common.GEOM["a_nm"], "n_cells": common.GEOM["n_cells"],
            "r_nom_nm": common.GEOM["r_nm"], "w_min_nm": W_MIN,
            "kappa": args.kappa, "screen_keep_frac": None,
            "bundle": os.path.abspath(args.bundle), "calibration": None,
            "baseline": {
                "pred_E_mean": float(p["mean"][0]),
                "note": "baseline = unperturbed layout (candidate_0000); "
                        "compare each perturbed true_E to the reference "
                        "true_E, same grid, so common-mode error cancels"},
            "note": f"interpretability occlusion spot-check for {tag} "
                    f"({rec['source']}); attribution deltas, not designs",
            "candidates": cands,
        }
        with open(man_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"[export] {tag}: {len(entries)} candidates -> {exp_dir}")
        print(f"  verify with:\n  LD_LIBRARY_PATH={common.ENV_LIB} "
              f"{common.ENV_PY} {common.SOLVER_DIR}/verify_candidates.py "
              f"--in-dir {exp_dir} --n-controls 0")


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
BLUE, VERM, MUT = "#0072B2", "#D55E00", "#666666"


def _load_verified(exp_dir):
    """{candidate index -> true_E60} from verification.csv, plus methods."""
    csv = os.path.join(exp_dir, "verification.csv")
    if not os.path.exists(csv):
        return None
    import csv as _csv
    rows = list(_csv.DictReader(open(csv)))
    out = {}
    for r in rows:
        try:
            out[r["name"]] = (r["method"], float(r["true_E60"]))
        except (KeyError, ValueError):
            pass
    return out or None


def run_figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap_cam = LinearSegmentedColormap.from_list(
        "cam", [(1, 1, 1, 0), (0.84, 0.37, 0.0, 0.75)])       # -> vermillion
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})

    for tag, rec in resolve_targets(args):
        out_dir = os.path.join(OUT_ROOT, tag)
        need = [os.path.join(out_dir, f) for f in
                ("maps.npz", "shift.npz", "remove.npz")]
        if not all(os.path.exists(p) for p in need):
            print(f"[figure] {tag}: missing maps/shift/remove, skipping")
            continue
        maps, sh, rm = (np.load(p) for p in need)
        verified = _load_verified(os.path.join(out_dir, "verify_export"))
        holes, L = sh["holes"], rec["a_super_nm"]

        fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.1), dpi=200)
        (ax_a, ax_b, ax_c, ax_d) = axes

        # (a) raster + Grad-CAM(mu)
        ax_a.imshow(maps["raster"].T, origin="lower", cmap="gray",
                    vmin=0, vmax=1, extent=[0, L, 0, L])
        cam = maps["cam_mu"]
        ax_a.imshow((cam / (cam.max() + 1e-12)).T, origin="lower",
                    cmap=cmap_cam, vmin=0, vmax=1, extent=[0, L, 0, L])
        ax_a.set_title("Grad-CAM ($\\hat{E}$ head)", fontsize=8.5)

        # (b) k-space saliency + lattice-harmonic rings
        s2 = maps["sal_c2"]
        ax_b.imshow(s2.T, origin="lower", cmap="magma")
        for mlt in (1, 2, 3):
            ax_b.add_patch(plt.Circle((IMG // 2, IMG // 2),
                                      mlt * LATTICE_PX, fill=False,
                                      color="w", lw=0.7,
                                      ls=(0, (3, 2)), alpha=0.8))
        ax_b.set_title("k-space saliency $|\\partial\\hat{E}/"
                       "\\partial C_2|$", fontsize=8.5)
        ins = ax_b.inset_axes([0.58, 0.62, 0.4, 0.36])
        prof = maps["sal_c2_radial"]
        ins.plot(prof, color=VERM, lw=1.0)
        for mlt in (1, 2, 3):
            ins.axvline(mlt * LATTICE_PX, color=MUT, lw=0.5,
                        ls=(0, (2, 2)))
        ins.set_xticks([]); ins.set_yticks([])
        ins.patch.set_alpha(0.7)

        # (c) shift quiver
        ax_c.set_facecolor("#f5f5f5")
        for (hx, hy, hr) in holes:
            ax_c.add_patch(plt.Circle((hx, hy), hr, fill=False,
                                      color=MUT, lw=0.6))
        gmag = sh["gmag"]
        q = ax_c.quiver(holes[:, 0], holes[:, 1], sh["gx"], sh["gy"], gmag,
                        cmap="viridis", angles="xy", pivot="mid",
                        width=0.008)
        ax_c.set_title("positional sensitivity (fill-preserving)",
                       fontsize=8.5)

        # (d) removal importance
        ax_d.set_facecolor("#f5f5f5")
        dE = rm["dE"]
        vmax = float(np.abs(dE).max()) + 1e-12
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
        smap = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
        for j, (hx, hy, hr) in enumerate(holes):
            ax_d.add_patch(plt.Circle((hx, hy), hr,
                                      color=smap.to_rgba(dE[j]),
                                      ec=MUT, lw=0.4))
        fig.colorbar(smap, ax=ax_d, fraction=0.046, pad=0.02,
                     label="$\\Delta\\hat{E}$ on removal")
        ax_d.set_title("removal importance (fill-changing)", fontsize=8.5)

        for ax in (ax_c, ax_d):
            ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])

        note = (f"{tag}  |  $\\hat{{E}}$={float(maps['e_base']):.4f}  |  "
                "ensemble+D4-averaged; model attributions "
                "(consistency check, not physics evidence)")
        if verified:
            pairs = [f"{m}:{e:.4f}" for m, e in verified.values()]
            note += "  |  FDTD-verified: " + ", ".join(pairs)
        fig.suptitle(note, fontsize=7.5, y=0.02, va="bottom")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"figure.{ext}"))
        plt.close(fig)
        print(f"[figure] {tag} -> {out_dir}/figure.png")


# ---------------------------------------------------------------------------
def main():
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--bundle", default=common.DEPLOYED_BUNDLE)
    shared.add_argument("--device", default="cpu",
                        help="cpu (default; avoids exclusive-GPU conflicts)")
    shared.add_argument("--kappa", type=float, default=0.2)
    shared.add_argument("--batch-size", type=int, default=256)
    shared.add_argument("--layout", action="append", default=None,
                        help="npz path with holes_xyr_nm, or bank:best / "
                             "bank:median; repeatable (default: champion set)")
    shared.add_argument("--tag", action="append", default=None)
    shared.add_argument("--force", action="store_true")
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("maps", parents=[shared])
    p = sub.add_parser("shift", parents=[shared])
    p.add_argument("--delta-nm", type=float, default=13.0)
    p.add_argument("--linearity-top", type=int, default=8)
    sub.add_parser("remove", parents=[shared])
    p = sub.add_parser("export", parents=[shared])
    p.add_argument("--top-shift", type=int, default=0,
                   help="extra single-hole shift exports (default 0: "
                        "individually below solver noise; the compound "
                        "all-holes shift is always exported)")
    p.add_argument("--top-remove", type=int, default=2)
    sub.add_parser("figure", parents=[shared])
    args = ap.parse_args()
    {"maps": run_maps, "shift": run_shift, "remove": run_remove,
     "export": run_export, "figure": run_figure}[args.cmd](args)


if __name__ == "__main__":
    main()
