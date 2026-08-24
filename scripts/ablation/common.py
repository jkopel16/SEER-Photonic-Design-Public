"""Shared infrastructure for the ablation suite (scripts/ablation/).

One import point for everything the per-ablation drivers need:

  * the committed protocol constants (dataset, seed 137, best_params,
    production inverse-design flags, the 8 cells, geometry)
  * `evaluate_bundle` -- THE single metrics path every table row goes
    through (deployed reference included), so rows are comparable by
    construction.  It refuses to run if it cannot reproduce the bundle's
    training-time normalisation statistics (proof the split matched).
  * subprocess wrappers for `models.model`, `models.inverse_design` and
    `verify_candidates.py` with the production environment baked in
    (LD_LIBRARY_PATH, PC_MODE=FULL PC_COMPILE=1, --no-calibration).
  * the factored-out member-training loop (model.py main's semantics:
    member seeds seed + i*137, k-fold over train+val, master-split norm
    stats) for the two ablations that need custom splits (#12, #15).

Nothing here edits models/ or scripts/FDTD_solver/ -- gaps in those CLIs
are worked around HERE (see evaluate_bundle's docstring for why the
existing eval paths could not be reused).

Torch and models.* are imported lazily so the torch-free scripts
(ablation_01, ablation_table) run on any node.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Paths / protocol constants
# ---------------------------------------------------------------------------
_ABL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_ABL_DIR))          # .../Photonics_RISE
SOLVER_DIR = os.path.join(REPO, "scripts", "FDTD_solver")
sys.path.insert(0, REPO)          # models.*
sys.path.insert(0, SOLVER_DIR)    # disorder, logutil, config

ENV_PY = "/project/rise-batteries/photonics-fdtd/bin/python3"
ENV_LIB = "/project/rise-batteries/photonics-fdtd/lib"

DATA_128 = os.path.join(REPO, "data", "samples_128.npz")
DEPLOYED_BUNDLE = os.path.join(
    REPO, "runs", "surrogate_128_fft_nll_sweep", "surrogate_bundle.pt")
BEST_PARAMS = os.path.join(
    REPO, "runs", "surrogate_128_fft_nll_sweep", "best_params.json")
ABLATION_DIR = os.path.join(REPO, "runs", "ablation")

# The 8 production cells (== runs/inverse_v2/ campaign grid)
CELLS = ([("jitter", s) for s in (0.06, 0.08, 0.10, 0.125, 0.15)]
         + [("radius", s) for s in (0.15, 0.20, 0.25)])
CHAMPION = ("jitter", 0.15)      # champion cell (E = 2.6675 at res 60)

# FULL-mode geometry -- make_layout's own defaults are generic fallbacks
# (n_cells=4, r=0.30a); every production call passes these explicitly.
GEOM = dict(a_nm=650.0, n_cells=7, r_nm=0.35 * 650.0, w_min_nm=50.0)
A_SUPER_NM = 4550.0              # 7 * 650

# Bank seeds are BASE_SEED + (c_idx+1)*1e6 + s_idx*1e4 + k with c_idx < 8
# (run_dataset.build_plan); the +4e6 offset keeps ablation draws disjoint
# from every banked seed.
BANK_BASE_SEED = 20260709
ABLATION_SEED_OFFSET = 4_000_000

# Production v2 campaign flags (scripts/run_v2_campaigns.sh:38-47 +
# runs/inverse_v2/*/manifest.json), minus --kappa which each ablation sets.
PROD_INVERSE_FLAGS = [
    "--tiers", "baseline", "screen", "cmaes", "gradient",
    "--n-baseline", "5000",
    "--n-screen", "50000", "--screen-keep-frac", "0.2",
    "--cmaes-restarts", "12", "--cmaes-iters", "600", "--cmaes-popsize", "24",
    "--grad-starts", "12", "--grad-steps", "500", "--grad-lr", "0.015",
    "--export-top", "20",
]

FLOOR_PCT = 0.30                 # within-sigma resolvability floor (audit)


def cell_tag(cls, sigma):
    """'jitter', 0.15 -> 'jitter_s015' (matches runs/inverse_v2 naming:
    s006 s008 s010 s0125 s015 s020 s025)."""
    digits = f"{sigma:g}".split(".")[1]
    if len(digits) == 1:            # 0.1 -> '10', 0.2 -> '20'
        digits += "0"
    return f"{cls}_s0{digits}"


def tee_into(name, out_dir):
    """Mirror stdout/stderr to <out_dir>/logs/ (house logutil.tee)."""
    os.makedirs(out_dir, exist_ok=True)
    from logutil import tee                                   # noqa: E402
    return tee(name, out_dir)


# ---------------------------------------------------------------------------
# Subprocess plumbing
# ---------------------------------------------------------------------------
def _env(extra=None):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ENV_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    if extra:
        env.update(extra)
    return env


def run_cmd(cmd, extra_env=None, dry_run=False):
    """Run a command from the repo root, streaming output. Returns rc."""
    pretty = " ".join(f"{k}={v}" for k, v in (extra_env or {}).items())
    print(f"[cmd] {pretty + ' ' if pretty else ''}{' '.join(cmd)}")
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=REPO, env=_env(extra_env))


def standard_train_cli(out_dir, *, data=DATA_128, nll=True, kfold=True,
                       augment=True, ensemble=5, seed=137, extra=None):
    """CLI for `python -m models.model` = deployed v2 recipe +/- deltas.

    Deployed v2: -i samples_128 --raster-only --fft-channel --nll-head
    --kfold-members --ensemble 5 --seed 137 --use-best (hyperparams from
    best_params.json, which run_training copies into out_dir).
    """
    cli = ["-i", data, "-o", out_dir,
           "--raster-only", "--fft-channel",
           "--ensemble", str(ensemble), "--seed", str(seed),
           "--use-best", "--no-wandb"]
    if nll:
        cli.append("--nll-head")
    if kfold:
        cli.append("--kfold-members")
    if not augment:
        cli.append("--no-augment")
    if extra:
        cli.extend(extra)
    return cli


def run_training(cli, out_dir, dry_run=False, patch_params=None):
    """Copy best_params.json into out_dir (required by --use-best, which
    reads <out-dir>/best_params.json only) and launch models.model.

    patch_params: optional {key: value} overrides applied to the COPIED
    best_params.json (a plain CLI flag would be clobbered by the
    --use-best overlay, model.py:1181-1185)."""
    bundle = os.path.join(out_dir, "surrogate_bundle.pt")
    if os.path.exists(bundle):
        print(f"[skip] {bundle} already exists -- refusing to retrain "
              "(delete the dir to redo; out-dirs are never reused, "
              "audit sec. 11.3)")
        return 0
    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, "best_params.json")
        shutil.copy(BEST_PARAMS, dst)
        if patch_params:
            with open(dst) as f:
                bp = json.load(f)
            bp.update(patch_params)
            with open(dst, "w") as f:
                json.dump(bp, f, indent=2)
            print(f"[params] patched {dst}: {patch_params}")
    else:
        print(f"[dry] would copy {BEST_PARAMS} -> {out_dir}/best_params.json"
              + (f" and patch {patch_params}" if patch_params else ""))
    return run_cmd([ENV_PY, "-u", "-m", "models.model"] + cli,
                   dry_run=dry_run)


def run_inverse(export_dir, cls, sigma, bundle, kappa, dry_run=False):
    """Production inverse-design invocation for one cell.

    ALWAYS passes --no-calibration: the production campaigns ran before
    uq/calibration.json existed (manifests record calibration: null), but
    the file exists now and would silently auto-load and change the LCB.
    """
    cmd = [ENV_PY, "-u", "-m", "models.inverse_design",
           "--bundle", bundle,
           "--disorder-class", cls, "--sigma", str(sigma),
           "--export-dir", export_dir,
           "--kappa", str(kappa),
           "--no-calibration"] + PROD_INVERSE_FLAGS
    return run_cmd(cmd, dry_run=dry_run)


def run_verify(in_dir, dry_run=False):
    """Production FDTD verification of a candidate dir (resumable via
    <in_dir>/verify_cache/)."""
    cmd = [ENV_PY, "-u",
           os.path.join("scripts", "FDTD_solver", "verify_candidates.py"),
           "--in-dir", in_dir]
    return run_cmd(cmd, extra_env={"PC_MODE": "FULL", "PC_COMPILE": "1"},
                   dry_run=dry_run)


# ---------------------------------------------------------------------------
# best-params overlay (import-loop scripts train byte-identical to --use-best)
# ---------------------------------------------------------------------------
def load_args(cli):
    """models.model.parse_args(cli) + best_params.json overlay.

    Same semantics as model.py's --use-best loop (skip best_val_loss and
    non-arg keys, coerce to the default's type) so #12/#15 train with the
    exact deployed hyperparameters."""
    from models.model import parse_args                       # noqa: E402
    args = parse_args(cli)
    with open(BEST_PARAMS) as f:
        bp = json.load(f)
    for k, v in bp.items():
        if k == "best_val_loss":
            continue
        if hasattr(args, k) and getattr(args, k) is not None:
            setattr(args, k, type(getattr(args, k))(v))
    return args


# ---------------------------------------------------------------------------
# Import-loop training (for ablations that need a custom split: #12, #15)
# ---------------------------------------------------------------------------
def naive_split(n, seed=137, val_frac=0.10, test_frac=0.10):
    """Plain shuffled split: no grouping, no sigma stratification (#12)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    return (np.sort(perm[n_test + n_val:]),      # train
            np.sort(perm[n_test:n_test + n_val]),  # val
            np.sort(perm[:n_test]))              # test


def load_dataset(args):
    """np.load + channel resolution, exactly as model.py main (:1188-1198).

    Returns (data, X, y, recipe, groups)."""
    import torch                                              # noqa: F401
    from models.model import resolve_input                    # noqa: E402
    data = np.load(args.data, allow_pickle=False)
    X = torch.from_numpy(data["X"]).float()
    y = torch.from_numpy(data["y"]).float()
    if X.dim() == 3:
        X = X.unsqueeze(1)
    ds_recipe = (data["channel_recipe"] if "channel_recipe" in data.files
                 else None)
    X, recipe = resolve_input(X, ds_recipe, args)
    groups = data["sample_id"] if "sample_id" in data.files else None
    return data, X, y, recipe, groups


def train_members(args, X_norm, y_norm, sigma, groups, train_idx, val_idx,
                  kfold=True, naive_folds=False):
    """model.py main's member loop (:1231-1271), factored out.

    Member seeds are seed + i*137 (model.py:1262 -- NOT learning_curve's
    seed_i + 1000*m).  With kfold, each member's val fold rotates out of
    the train+val pool; the test split stays outside every fold and the
    normalisation stats stay from the master train split (the caller
    computed them BEFORE calling this).  naive_folds replaces the
    sigma-stratified group-aware fold cut with a plain shuffled partition
    (#12: split naivety end-to-end).
    Returns (members, best_vals)."""
    from torch.utils.data import DataLoader                   # noqa: E402
    from models.model import (PhotonicDataset, train_one,     # noqa: E402
                              stratified_group_folds, device)

    pin = device.type == "cuda"

    def mk(idx, shuffle):
        return DataLoader(PhotonicDataset(X_norm[idx], y_norm[idx]),
                          batch_size=args.batch_size, shuffle=shuffle,
                          pin_memory=pin)

    fold_loaders = None
    if kfold:
        if args.ensemble < 2:
            raise SystemExit("kfold member rotation needs ensemble >= 2")
        pool = np.concatenate([train_idx, val_idx])
        if naive_folds:
            rng = np.random.default_rng(args.seed)
            perm = rng.permutation(len(pool))
            folds = [np.sort(perm[j::args.ensemble])
                     for j in range(args.ensemble)]
        else:
            folds = stratified_group_folds(
                np.asarray(sigma)[pool],
                groups=(np.asarray(groups)[pool] if groups is not None
                        else None),
                k=args.ensemble, seed=args.seed)
        print(f"[kfold] {args.ensemble} members over pool n={len(pool)}; "
              f"fold sizes {[len(f) for f in folds]}"
              f"{' (naive folds)' if naive_folds else ''}")
        fold_loaders = []
        for f in folds:
            vi = pool[f]
            ti = np.setdiff1d(pool, vi)
            fold_loaders.append((mk(ti, True), mk(vi, False)))
    else:
        shared = (mk(train_idx, True), mk(val_idx, False))

    members, best_vals = [], []
    in_ch = int(X_norm.shape[1])
    for i in range(args.ensemble):
        seed_i = args.seed + i * 137
        loaders_i = (fold_loaders[i] if fold_loaders is not None else shared)
        tag = f", val fold {i}" if fold_loaders is not None else ""
        print(f"\n=== member {i + 1}/{args.ensemble} (seed={seed_i}{tag}) ===")
        m, bv = train_one(args, loaders_i, seed_i, in_ch=in_ch)
        members.append(m)
        best_vals.append(bv)
    return members, best_vals


def save_bundle(out_dir, members, best_vals, args, norm, recipe, img_size,
                metrics=None):
    """Write a surrogate bundle with EXACTLY model.py's schema (:1285-1307)
    so SurrogateScorer and evaluate_bundle both accept it."""
    import torch                                              # noqa: E402
    x_mean, x_std, y_mean, y_std = norm
    in_ch = int(len(x_mean.flatten()))      # one norm stat per input channel
    bundle = {
        "format": ("photonic-surrogate-bundle-v2"
                   if args.nll_head else "photonic-surrogate-bundle-v1"),
        "heteroscedastic": bool(args.nll_head),
        "state_dicts": [m.state_dict() for m in members],
        "arch": {"input_shape": in_ch,
                 "hidden_units": args.hidden_units,
                 "output_shape": 2 if args.nll_head else 1,
                 "dropout": args.dropout,
                 "stochastic_depth": args.stochastic_depth,
                 "padding_mode": ("circular"
                                  if getattr(args, "circular_padding", False)
                                  else "zeros"),
                 "attention": getattr(args, "attention", "se"),
                 "img_size": int(img_size),
                 "recon_head": getattr(args, "recon_head", False)},
        "norm": {"x_mean": x_mean.flatten().tolist(),
                 "x_std": x_std.flatten().tolist(),
                 "y_mean": float(y_mean), "y_std": float(y_std)},
        "img_size": int(img_size),
        "channel_recipe": list(recipe),
        "train_config": vars(args),
        "test_metrics": metrics or {},
        "best_val_losses": best_vals,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "surrogate_bundle.pt")
    torch.save(bundle, path)
    print(f"[bundle] saved -> {path}")
    return path


def slice_bundle(src, dst, keep=1):
    """Derived bundle with only the first `keep` member(s) (#7).  The
    stored test_metrics are the FULL ensemble's -- dropped so nothing
    downstream mistakes them for the slice's."""
    import torch                                              # noqa: E402
    bundle = torch.load(src, map_location="cpu", weights_only=False)
    bundle["state_dicts"] = bundle["state_dicts"][:keep]
    bundle.pop("test_metrics", None)
    bundle["best_val_losses"] = bundle.get("best_val_losses", [])[:keep]
    bundle["note"] = f"ablation slice: first {keep} member(s) of {src}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save(bundle, dst)
    print(f"[slice] {len(bundle['state_dicts'])} member(s) -> {dst}")
    return dst


# ---------------------------------------------------------------------------
# THE metrics path (every table row, deployed reference included)
# ---------------------------------------------------------------------------
def evaluate_bundle(bundle_path, data_path=None, out_json=None, label=None,
                    batch_size=128):
    """Uniform evaluation of any surrogate bundle -> ablation_metrics.json.

    Why not reuse the existing eval paths: model.py --eval-bundle rejects
    v2 bundles outright (model.py:1448); ensemble_uq/calibrate_uq refuse
    <2-member bundles, force TTA on, and hard-abort on their own sanity
    gates.  And evaluate_and_report's mean prediction is ALWAYS
    TTA-averaged regardless of --no-tta (model.py:1332-1335), so an
    honest TTA ablation needs predict()/predict_gaussian()'s use_tta.

    For each eval set and each TTA arm (on/off), emits: n, MAE, RMSE, R2,
    pooled + per-cell within-sigma Spearman rho; for bundles with an
    uncertainty output additionally RMS(s) (total/aleatoric/epistemic),
    RMS(s)/RMSE and PICP at +-1/2/3 s.  UQ conventions match the audit:
    second-moment (RMS-vs-RMS) comparisons only, never s vs mean |error|.

    s per bundle type:
      v2 (NLL head): Gaussian-mixture total, var = mean_i(sigma_i^2)
        + var_i(mu_i) -- identical to evaluate_and_report / SurrogateScorer.
      v1 (>= 2 members): member disagreement only (no aleatoric head) --
        flagged in `notes`; this is what v1 historically used, so the
        loss-head 2x2 stays comparable.
      single member v1: no s at all.

    Split reconstruction: <bundle_dir>/split.json sidecar when present
    (written by the custom-split ablations #12/#15, may carry extra eval
    sets), else stratified_group_split with the bundle's training seed.
    Either way the recomputed train-split normalisation statistics MUST
    match the bundle's stored ones to 1e-5 -- that agreement is the proof
    the reconstructed split is the training split; mismatch hard-fails.
    """
    import torch                                              # noqa: E402
    from torch.utils.data import DataLoader                   # noqa: E402
    from models.model import (PhotonicCNN, PhotonicDataset,   # noqa: E402
                              infer_bundle_recipe, normalize,
                              predict, predict_gaussian,
                              regression_metrics, resolve_input,
                              stratified_group_split,
                              within_sigma_spearman, device)
    from types import SimpleNamespace                          # noqa: E402

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    fmt = bundle.get("format", "?")
    tc = dict(bundle.get("train_config", {}))
    hetero = bool(bundle.get("heteroscedastic", fmt.endswith("v2")))
    notes = []

    # ---- dataset + channels (exactly as trained) ----
    dp = data_path or tc.get("data", DATA_128)
    if not os.path.isabs(dp):
        dp = os.path.join(REPO, dp)
    data = np.load(dp, allow_pickle=False)
    X = torch.from_numpy(data["X"]).float()
    y = torch.from_numpy(data["y"]).float()
    if X.dim() == 3:
        X = X.unsqueeze(1)
    ds_recipe = (data["channel_recipe"] if "channel_recipe" in data.files
                 else None)
    ns = SimpleNamespace(raster_only=tc.get("raster_only", False),
                         fft_channel=tc.get("fft_channel", False),
                         fft_only=tc.get("fft_only", False))
    X, recipe = resolve_input(X, ds_recipe, ns)
    want = infer_bundle_recipe(bundle)
    if list(recipe) != list(want):
        raise SystemExit(f"[eval] channel recipe mismatch: dataset gives "
                         f"{recipe}, bundle wants {want}")
    if int(X.shape[-1]) != int(bundle["img_size"]):
        raise SystemExit(f"[eval] raster {X.shape[-1]}px != bundle "
                         f"img_size {bundle['img_size']}px")

    # ---- split reconstruction ----
    sidecar = os.path.join(os.path.dirname(os.path.abspath(bundle_path)),
                           "split.json")
    eval_sets = {}
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            sp = json.load(f)
        train_idx = np.asarray(sp["train"], dtype=np.int64)
        val_idx = np.asarray(sp["val"], dtype=np.int64)
        eval_sets["test"] = np.asarray(sp["test"], dtype=np.int64)
        for k, v in sp.get("extra", {}).items():
            eval_sets[k] = np.asarray(v, dtype=np.int64)
        notes.append(f"split from sidecar {os.path.basename(sidecar)}: "
                     + (sp.get("note") or "custom split"))
        print(f"[eval] split from sidecar ({sp.get('note', 'custom')})")
    else:
        groups = data["sample_id"] if "sample_id" in data.files else None
        train_idx, val_idx, test_idx = stratified_group_split(
            data["sigma"], groups=groups, seed=int(tc["seed"]))
        eval_sets["test"] = test_idx

    # ---- normalisation cross-check (proof the split matched training) ----
    X_norm, y_norm, xm, xs, ym, ys = normalize(X, y, train_idx)
    bn = bundle["norm"]
    got = np.concatenate([xm.flatten().numpy(), xs.flatten().numpy(),
                          [float(ym), float(ys)]])
    exp = np.concatenate([np.asarray(bn["x_mean"], dtype=float),
                          np.asarray(bn["x_std"], dtype=float),
                          [bn["y_mean"], bn["y_std"]]])
    if not np.allclose(got, exp, atol=1e-5):
        raise SystemExit(
            "[eval] normalisation statistics do NOT match the bundle's "
            f"(max diff {np.abs(got - exp).max():.2e}) -- the reconstructed "
            "split is not the training split. Refusing to emit metrics.")
    print("[eval] norm stats reproduce the bundle's (split verified)")

    # ---- members ----
    # Defensive back-compat for bundles written before --attention / the
    # img_size arch field existed (the deployed v2 + every v1 bundle in
    # runs/): they default to SE attention.  PhotonicCNN.__init__ has the
    # same defaults, so this is belt-and-suspenders.
    bundle["arch"].setdefault("attention", "se")
    bundle["arch"].setdefault("img_size", None)
    bundle["arch"].setdefault("recon_head", False)
    members = []
    for sd in bundle["state_dicts"]:
        m = PhotonicCNN(**bundle["arch"])
        m.load_state_dict(sd)
        m.to(device).eval()
        members.append(m)
    n_mem = len(members)
    print(f"[eval] {n_mem}-member {fmt}"
          f" ({'heteroscedastic' if hetero else 'point-estimate'})")
    if hetero and n_mem == 1:
        notes.append("single member: epistemic variance is identically 0; "
                     "s is aleatoric only")
    if not hetero and n_mem >= 2:
        notes.append("v1 bundle: s = ensemble member disagreement only "
                     "(no aleatoric head)")

    y_mean, y_std = float(bn["y_mean"]), float(bn["y_std"])
    pin = device.type == "cuda"
    rows = {}
    for set_name, idx in eval_sets.items():
        loader = DataLoader(PhotonicDataset(X_norm[idx], y_norm[idx]),
                            batch_size=batch_size, shuffle=False,
                            pin_memory=pin)
        sig_cell = np.asarray(data["sigma"])[idx]
        cls_cell = (np.asarray(data["disorder_class"])[idx]
                    if "disorder_class" in data.files else None)
        for use_tta in (True, False):
            if hetero:
                mem_mu, mem_var = [], []
                for m in members:
                    mu_i, sig_i, tgt = predict_gaussian(m, loader,
                                                        use_tta=use_tta)
                    mem_mu.append(mu_i.numpy())
                    mem_var.append((sig_i.numpy()) ** 2)
                MU = np.stack(mem_mu)
                mu_n = MU.mean(axis=0)
                var_alea = np.stack(mem_var).mean(axis=0)
                var_epi = MU.var(axis=0)
                s_total = np.sqrt(var_alea + var_epi) * y_std
                s_alea = np.sqrt(var_alea) * y_std
                s_epi = np.sqrt(var_epi) * y_std
            else:
                mem_p = []
                for m in members:
                    p_i, tgt = predict(m, loader, use_tta=use_tta)
                    mem_p.append(p_i.numpy())
                P = np.stack(mem_p)
                mu_n = P.mean(axis=0)
                s_epi = (P.std(axis=0) * y_std) if n_mem > 1 else None
                s_total, s_alea = s_epi, None
            mu = mu_n * y_std + y_mean
            yt = tgt.numpy() * y_std + y_mean

            row = regression_metrics(yt, mu)
            rho, per_cell = within_sigma_spearman(yt, mu, sig_cell, cls_cell)
            row.update({"n": int(len(idx)), "rho_pooled": rho,
                        "per_cell_rho": per_cell})
            if s_total is not None:
                resid = np.abs(yt - mu)
                rms_s = float(np.sqrt(np.mean(s_total ** 2)))
                row.update({
                    "rms_s_total": rms_s,
                    "rms_s_alea": (float(np.sqrt(np.mean(s_alea ** 2)))
                                   if s_alea is not None else None),
                    "rms_s_epi": (float(np.sqrt(np.mean(s_epi ** 2)))
                                  if s_epi is not None else None),
                    "rms_s_over_rmse": rms_s / row["rmse"],
                    "picp_1s": float(np.mean(resid <= s_total)),
                    "picp_2s": float(np.mean(resid <= 2 * s_total)),
                    "picp_3s": float(np.mean(resid <= 3 * s_total)),
                })
            key = f"{set_name}/tta_{'on' if use_tta else 'off'}"
            rows[key] = row
            print(f"[eval] {key:18s} MAE {row['mae']:.6f}  "
                  f"RMSE {row['rmse']:.6f}  rho {rho:+.3f}"
                  + (f"  RMS(s)/RMSE {row['rms_s_over_rmse']:.3f}  "
                     f"PICP1 {row['picp_1s']:.3f}"
                     if s_total is not None else ""))

    out = {"label": label or os.path.basename(os.path.dirname(
               os.path.abspath(bundle_path))),
           "bundle": os.path.abspath(bundle_path),
           "format": fmt, "n_members": n_mem, "data": dp,
           "stored_test_metrics": bundle.get("test_metrics", {}),
           "rows": rows, "notes": notes}
    out_json = out_json or os.path.join(
        os.path.dirname(os.path.abspath(bundle_path)),
        "ablation_metrics.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] wrote {out_json}")
    return out


# ---------------------------------------------------------------------------
# FDTD-side helpers
# ---------------------------------------------------------------------------
def write_candidate_dir(out_dir, cls, sigma, layouts, pred,
                        baseline_pred_mean, bundle=DEPLOYED_BUNDLE,
                        method="random", baseline_note=None):
    """Write layouts in the verify_candidates schema (candidate npz +
    manifest, mirroring inverse_design.export_candidates:612-659) so the
    production verifier consumes them UNMODIFIED.

    layouts: list of disorder.make_layout records.
    pred:    dict of arrays from SurrogateScorer.score_holes (mean/std/lcb)
             aligned with layouts -- keeps pred-vs-true meaningful.
    """
    from disorder import air_fraction                         # noqa: E402
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for i, rec in enumerate(layouts):
        holes = np.asarray(rec["holes"], dtype=float)
        path = os.path.join(out_dir, f"candidate_{i:04d}.npz")
        np.savez(path,
                 holes_xyr_nm=holes,
                 a_super_nm=rec["a_super_nm"],
                 disorder_class=cls,
                 sigma=float(sigma),
                 method=method,
                 pred_E_mean=float(pred["mean"][i]),
                 pred_E_std=float(pred["std"][i]),
                 pred_E_lcb=float(pred["lcb"][i]),
                 fill_achieved=air_fraction(holes[:, 2], rec["a_super_nm"]),
                 seed=int(rec.get("seed", -1)))
        manifest.append({"file": os.path.basename(path), "method": method,
                         "pred_E_mean": round(float(pred["mean"][i]), 5),
                         "pred_E_std": round(float(pred["std"][i]), 5),
                         "pred_E_lcb": round(float(pred["lcb"][i]), 5)})
    meta = {
        "disorder_class": cls, "sigma": float(sigma),
        "a_nm": GEOM["a_nm"], "n_cells": GEOM["n_cells"],
        "r_nom_nm": GEOM["r_nm"], "w_min_nm": GEOM["w_min_nm"],
        "kappa": 0.0,
        "screen_keep_frac": None,
        "bundle": os.path.abspath(bundle),
        "calibration": None,
        "baseline": {"pred_E_mean": float(baseline_pred_mean),
                     "note": baseline_note or ""},
        "note": ("Ablation control arm -- layouts drawn by "
                 "disorder.make_layout, not optimized. Predicted values "
                 "from the deployed v2 surrogate for pred-vs-true "
                 "bookkeeping only."),
        "candidates": manifest,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] {len(layouts)} candidates + manifest -> {out_dir}")


def fdtd_arm(csv_path, label=None, by_method=False):
    """Aggregate one verification.csv into arm stats.

    Two-level resolution rule (pre-registered): arm MEANS carry SE ~
    std/sqrt(n) and can resolve shifts somewhat below the 0.30 % pairwise
    floor; INDIVIDUAL candidate comparisons cannot.  Keep the levels
    separate when reporting."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    def finite(r, k):
        try:
            v = float(r.get(k, "nan"))
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    # An arm reports at res 120 only when EVERY row has it: a partial
    # column (e.g. a single champion re-verified at res 120) must not
    # switch the arm away from the resolution its comparators use.
    key = ("true_E120" if rows and all(finite(r, "true_E120") is not None
                                       for r in rows) else "true_E60")

    def stats(rs):
        e = np.array([finite(r, key) for r in rs
                      if finite(r, key) is not None])
        if len(e) == 0:
            return {"n": 0}
        return {"n": int(len(e)), "mean": float(e.mean()),
                "se": (float(e.std(ddof=1) / np.sqrt(len(e)))
                       if len(e) > 1 else float("nan")),
                "max": float(e.max()),
                "n_claimable": sum(1 for r in rs
                                   if r.get("verdict") == "CLAIMABLE")}

    out = {"label": label or csv_path, "csv": os.path.abspath(csv_path),
           "true_col": key, "arm": stats(rows)}
    if by_method:
        methods = sorted({r.get("method", "?") for r in rows})
        out["by_method"] = {m: stats([r for r in rows
                                      if r.get("method") == m])
                            for m in methods}
    out["rows"] = rows
    return out
