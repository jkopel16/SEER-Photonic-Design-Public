"""
run_timing_test.py
------------------
FIRST THING TO RUN ON THE GPU NODE.  One command answers, with receipts:

  1. does the engine's physics hold on THIS hardware?    (validation gates)
  2. does it reproduce the ordered-lattice cross-engine anchor E ~ 2.547?
  3. how long does ONE production dataset sample take, and therefore what
     does the WHOLE campaign cost on 1/2/4/8 GPUs?

Everything is persisted under PC_OUT (default gpu_out/): a mirrored .log
of all terminal output, timing_report.txt + timing_report.json, and the
gate/anchor figures.  Nothing is lost when the terminal session dies.

Usage on the SCC (see submit_timing.sh, or interactively):

    export PC_MODE=FULL PC_OUT=/projectnb/<proj>/gpu_out
    python -u run_timing_test.py

Flags:
    --quick               skip anchor + ladder; gates + timing probes only
    --skip-anchor         skip the unit-cell anchor
    --resolution-ladder   also solve the anchor at res 60/80/100/120 and
                          report the drift of E (label-convergence
                          evidence; DEFAULT ON for the anchor cell)
    --no-ladder           disable the ladder
    --probe-sigma S       disorder strength of the jitter probe (0.15)

Exit code is nonzero if any gate fails -- safe to chain in job scripts:
    python -u run_timing_test.py && qsub submit_dataset.sh
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
from materials_gpu import fit_all, validate_fits
from optics_core import planar_reference_stack, solar_weight
import fdtd_torch as F
from disorder import make_layout, holes_array

E_RCWA_ANCHOR = 2.547        # ordered unit cell, r/a = 0.35 (RCWA stage)
ANCHOR_TOL_REL = {"FULL": 0.06, "FAST": 0.15, "SMOKE": 0.60}[C.MODE]
# |E/E_anchor - 1| gate; engines differ at finite resolution and the
# coarse modes exist for plumbing, so the gate scales with mode.  The
# resolution ladder shows the convergence trend that justifies FULL.


def wl_grid():
    return C.raw_wavelength_grid()


def _fig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _draw_planar_fig(path, panels, suptitle, fdtd_label):
    """Render the 1x3 band-panel figure from precomputed plot data --
    pure matplotlib, no FDTD, no materials.  `panels` is a list of dicts
    with wl_d/tmm_d (dense analytic line), wl_b/a_f (engine samples) and
    err (per-band sup-norm).  Kept separate from _planar_gate_fig so
    replot_gate_figs.py can restyle saved figures without a GPU."""
    plt = _fig(path)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for axi, p in zip(axes, panels):
        axi.plot(p["wl_d"], p["tmm_d"], "k-", lw=1.5, label="TMM (fits)")
        axi.plot(p["wl_b"], p["a_f"], "o", ms=4, color="#1f5fa8",
                 label=fdtd_label)
        axi.text(0.03, 0.06,
                 r"max$|\Delta A_\mathrm{Si}|$" + f" = {p['err']:.1e}",
                 fontsize=7, transform=axi.transAxes,
                 bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
        axi.set_xlabel("wavelength (nm)")
    axes[0].set_ylabel(r"$A_\mathrm{Si}$")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _planar_gate_fig(path, results, adapters, suptitle, fdtd_label):
    """One panel per band: dense analytic-TMM curve (cheap, so drawn as a
    smooth 200-point line) with the engine's spectral samples on top.
    `results` is the per-band (wl, A_si, A_par, ref) list returned by the
    validators in fdtd_torch.  Everything drawn is also persisted to a
    sidecar .npz next to the figure, so styling can be iterated later
    with replot_gate_figs.py without re-running any FDTD."""
    panels = []
    for (wl_b, a_f, ap_f, ref) in results:
        wl_d = np.linspace(wl_b[0], wl_b[-1], 200)
        ref_d = planar_reference_stack(adapters["si"], adapters["zno"],
                                       adapters["ag"], wl_d,
                                       C.THICKNESS_NM, C.BUFFER_NM)
        panels.append({"wl_d": wl_d,
                       "tmm_d": np.asarray(ref_d["A_si"], float),
                       "wl_b": np.asarray(wl_b, float),
                       "a_f": np.asarray(a_f, float),
                       "err": float(np.max(np.abs(
                           np.asarray(a_f) - np.asarray(ref["A_si"]))))})
    _draw_planar_fig(path, panels, suptitle, fdtd_label)
    np.savez_compressed(
        path[:-4] + ".npz", suptitle=suptitle, fdtd_label=fdtd_label,
        n_bands=len(panels),
        **{f"{k}{i}": p[k] for i, p in enumerate(panels) for k in p})


def solve_unit_cell(fits, resolution, wl, norm_dir, device, pols=("x", "y"),
                    decay_tol=None, max_time=None):
    holes, a_sup = F.ordered_square_holes(C.A_NM, 1, C.R_OVER_A * C.A_NM)
    A_si, A_par, info = F.broadband_absorption_many(
        [holes], a_sup, C.THICKNESS_NM, wl, fits, C.BUFFER_NM, resolution,
        decay_tol or C.DECAY_TOL, max_time or C.MAX_TIME, norm_dir,
        device=device, pols=pols, n_cells_tag="unit1")
    return A_si[0], A_par[0], info


def label_E(wl, A_si, adapters):
    w = solar_weight(wl)
    ref = planar_reference_stack(adapters["si"], adapters["zno"],
                                 adapters["ag"], wl, C.THICKNESS_NM,
                                 C.BUFFER_NM)
    eta_p = float(np.sum(w * np.asarray(ref["A_si"])))
    return float(np.sum(w * A_si)) / eta_p, eta_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-anchor", action="store_true")
    ap.add_argument("--resolution-ladder", dest="ladder",
                    action="store_true", default=True)
    ap.add_argument("--no-ladder", dest="ladder", action="store_false")
    ap.add_argument("--probe-sigma", type=float, default=0.15)
    args = ap.parse_args()

    out = C.OUT_DIR
    os.makedirs(out, exist_ok=True)
    tee("timing_test", out)
    norm_dir = os.path.join(out, "norm_cache")
    figs = os.path.join(out, "figs")

    print(C.describe())
    device = F.resolve_device(os.environ.get("PC_DEVICE", "auto"))
    print(f"[device] {device}", end="")
    if device == "cuda":
        import torch
        print(f"  ({torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f}"
              f" GB)", end="")
    elif not F.FAKE:
        print("\n  WARNING: no CUDA device found -- this will run on CPU "
              "and be extremely slow.\n  On the SCC make sure the job "
              "requested a GPU (see submit_timing.sh).", end="")
    print()

    report = {"mode": C.MODE, "device": device, "fake": F.FAKE,
              "engine": F.ENGINE_VERSION, "gates": {}, "anchor": {},
              "ladder": [], "probes": {}, "eta": {}}
    t_all = time.time()
    fits, adapters, mats = fit_all()

    # ---------------- gates -------------------------------------------------
    print("\n=== GATE 1/3: material fits (label impact) ===")
    ok = validate_fits(fits, adapters, mats, C.THICKNESS_NM, C.BUFFER_NM)
    report["gates"]["materials"] = bool(ok)

    print("\n=== GATE 2/3: pseudo-1D planar stack vs fitted-material TMM "
          "===")
    ok1d, res1d = F.validate_planar_pseudo1d(
        fits, adapters, C.THICKNESS_NM, C.BUFFER_NM, norm_dir,
        resolution=C.GATE_PLANAR_RES, tol=C.GATE_PLANAR_TOL,
        device=device)
    ok &= ok1d
    report["gates"]["planar_pseudo1d"] = bool(ok1d)

    # The figure wants a denser spectral grid than the GATE does.  They are
    # deliberately separate runs: GATE_PLANAR_TOL is calibrated on the
    # 7-point grid, and a denser grid resolves the sharp visible-band
    # Fabry-Perot peaks and so reports a larger (more honest, but
    # differently-calibrated) sup-norm.  Gating on the dense grid would
    # trip a gate that no engine change caused, so the dense run is
    # report-only.
    res_fig = res1d
    _, res_dense = F.validate_planar_pseudo1d(
        fits, adapters, C.THICKNESS_NM, C.BUFFER_NM, norm_dir,
        resolution=C.GATE_PLANAR_RES, tol=None, device=device,
        n_wl=25, verbose=False)
    if res_dense:
        res_fig = res_dense
        errs = [float(np.max(np.abs(a_f - np.asarray(ref["A_si"]))))
                for (wl_b, a_f, ap_f, ref) in res_dense]
        report["planar_pseudo1d_dense_err"] = errs
        print("    dense figure grid (25 pts/band, report only): "
              "max|dA_si| = "
              + ", ".join(f"{e:.2e}" for e in errs))
    if res_fig:
        _planar_gate_fig(
            os.path.join(figs, "fig_gate_planar_1D.png"), res_fig, adapters,
            f"pseudo-1D gate: engine vs analytic TMM (same fitted "
            f"materials, res={C.GATE_PLANAR_RES}/um)", "torch-FDTD")
        print(f"    figure -> figs/fig_gate_planar_1D.png "
              f"({len(res_fig[0][0])} pts/band)")

    print("\n=== GATE 3/3: 3D uniform slab at unit-cell production "
          "numerics ===")
    ok3d, _ = F.validate_uniform_3d(
        fits, adapters, C.A_NM, C.THICKNESS_NM, C.BUFFER_NM,
        C.RESOLUTION_UNIT, C.DECAY_TOL, C.MAX_TIME, norm_dir,
        tol_bands=C.GATE_3D_TOLS, device=device)
    ok &= ok3d
    report["gates"]["uniform_3d"] = bool(ok3d)

    # figure-only companion: the SAME hole-free stack, but at the actual
    # dataset production numerics (N x N supercell, RESOLUTION) -- i.e.
    # the numerics the manufactured samples are solved with.  Report-only:
    # GATE_3D_TOLS were measured at RESOLUTION_UNIT and do not apply.
    print("\n=== 3D uniform slab at PRODUCTION numerics "
          "(figure, report-only) ===")
    _, res3p = F.validate_uniform_3d(
        fits, adapters, C.A_SUPER_NM, C.THICKNESS_NM, C.BUFFER_NM,
        C.RESOLUTION, C.DECAY_TOL, C.MAX_TIME, norm_dir,
        tol_bands=None, device=device,
        n_cells_tag=f"unifsc{C.N_CELLS}", n_wl=25)
    if res3p:
        report["uniform_3d_production_err"] = [
            float(np.max(np.abs(a_f - np.asarray(ref["A_si"]))))
            for (wl_b, a_f, ap_f, ref) in res3p]
        _planar_gate_fig(
            os.path.join(figs, "fig_gate_planar_3D.png"), res3p,
            adapters,
            f"3D uniform slab at production numerics "
            f"({C.N_CELLS}x{C.N_CELLS} supercell, "
            f"res={C.RESOLUTION}/um) vs analytic TMM",
            "torch-FDTD (3D)")
        print(f"    figure -> figs/fig_gate_planar_3D.png")

    if not ok:
        print("\n*** A GATE FAILED -- do not run the dataset until this "
              "is understood. ***")

    wl = wl_grid()
    print(f"\n[wavelength grid] {len(wl)} points "
          f"({wl[0]:.0f}-{wl[-1]:.0f} nm; VIS step {C.VIS_STEP_NM}, "
          f"NIR step {C.NIR_STEP_NM})")

    # ---------------- cross-engine anchor ----------------------------------
    if not (args.quick or args.skip_anchor):
        print("\n=== CROSS-ENGINE ANCHOR: ordered unit cell, both pols, "
              f"res={C.RESOLUTION_UNIT} ===")
        t0 = time.time()
        A_si, A_par, info = solve_unit_cell(fits, C.RESOLUTION_UNIT, wl,
                                            norm_dir, device)
        E, eta_p = label_E(wl, A_si, adapters)
        dE = E / E_RCWA_ANCHOR - 1
        okA = abs(dE) < ANCHOR_TOL_REL
        if F.FAKE:
            print("    [FDTD_FAKE=1] anchor computed on FAKE spectra -- "
                  "reported, NOT gated.")
        else:
            ok &= okA
        print(f"    E(unit cell) = {E:.4f}   RCWA anchor {E_RCWA_ANCHOR}"
              f"   ({100*dE:+.2f}%, gate {100*ANCHOR_TOL_REL:.0f}%)  "
              f"-> {'OK' if okA else 'CHECK'}   [{time.time()-t0:.0f}s, "
              f"cap={bool(info['hit_time_cap'][0])}]")
        report["anchor"] = {"E": E, "rel_err": dE, "ok": bool(okA),
                            "runtime_s": time.time() - t0}
        plt = _fig(os.path.join(figs, "fig_anchor_spectrum.png"))
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
        ref = planar_reference_stack(adapters["si"], adapters["zno"],
                                     adapters["ag"], wl, C.THICKNESS_NM,
                                     C.BUFFER_NM)
        for a in (a1, a2):
            a.plot(wl, np.asarray(ref["A_si"]), "k--", lw=1,
                   label="planar (TMM)")
            a.plot(wl, A_si, color="#27632a", lw=1.4,
                   label=f"ordered PC (E={E:.3f})")
            a.plot(wl, A_par, color="#888", lw=0.9, label=r"$A_{par}$")
            a.set_xlabel("wavelength (nm)")
        a2.set_xlim(700, 1100)
        a1.set_ylabel("absorption")
        a1.legend(frameon=False, fontsize=8)
        fig.suptitle("unit-cell anchor spectrum (torch-FDTD GPU engine)")
        fig.tight_layout()
        fig.savefig(os.path.join(figs, "fig_anchor_spectrum.png"), dpi=160)
        plt.close(fig)
        print("    figure -> figs/fig_anchor_spectrum.png")

        if args.ladder and not F.FAKE and C.MODE == "FULL":
            print("\n=== RESOLUTION LADDER (unit cell): E drift vs "
                  "resolution ===")
            for res in (60, 80, 100, 120):
                if res == C.RESOLUTION_UNIT:
                    E_r = E
                    dt_s = report["anchor"]["runtime_s"]
                else:
                    t0 = time.time()
                    A_r, _, _ = solve_unit_cell(fits, res, wl, norm_dir,
                                                device)
                    E_r, _ = label_E(wl, A_r, adapters)
                    dt_s = time.time() - t0
                report["ladder"].append({"res": res, "E": E_r,
                                         "runtime_s": dt_s})
                print(f"    res={res:4d}:  E = {E_r:.4f}   "
                      f"[{dt_s:.0f}s]")
            Es = [r["E"] for r in report["ladder"]]
            drift = max(Es) - min(Es)
            print(f"    ladder spread max-min = {drift:.4f} "
                  f"({100*drift/Es[-1]:.2f}% of E)  "
                  "<- label discretization uncertainty")

    # ---------------- production timing probes ------------------------------
    print(f"\n=== TIMING PROBES at PRODUCTION numerics "
          f"(N={C.N_CELLS}x{C.N_CELLS}, res={C.RESOLUTION}, "
          f"decay={C.DECAY_TOL:g}) ===")
    probes = {
        "ordered": make_layout("ordered", 0.0, C.BASE_SEED, C.A_NM,
                               C.N_CELLS, C.R_OVER_A * C.A_NM,
                               C.W_MIN_NM),
        f"jitter_s{args.probe_sigma:g}": make_layout(
            "jitter", args.probe_sigma, C.BASE_SEED + 1_010_000, C.A_NM,
            C.N_CELLS, C.R_OVER_A * C.A_NM, C.W_MIN_NM),
    }
    per_sample = []
    for name, rec in probes.items():
        holes = [tuple(h) for h in holes_array(rec)]
        t0 = time.time()
        n_runs = 3 * 2   # bands x pols
        print(f"    {name}: starting ({n_runs} FDTD runs; heartbeat "
              f"after each)...", flush=True)

        def _hb(i, n, _t0=t0, _nm=name):
            el = time.time() - _t0
            print(f"      [{time.strftime('%H:%M:%S')}] {_nm}: run "
                  f"{i}/{n} done, {el/60:.0f} min elapsed, ~"
                  f"{el/i*(n-i)/60:.0f} min left", flush=True)

        A_si, A_par, info = F.broadband_absorption_many(
            [holes], rec["a_super_nm"], C.THICKNESS_NM, wl, fits,
            C.BUFFER_NM, C.RESOLUTION, C.DECAY_TOL, C.MAX_TIME, norm_dir,
            device=device, n_cells_tag=f"sc{C.N_CELLS}", progress=_hb)
        dt_s = time.time() - t0
        E, _ = label_E(wl, A_si[0], adapters)
        per_sample.append(dt_s)
        gpu_mem = ""
        if device == "cuda":
            import torch
            gpu_mem = (f", peak GPU mem "
                       f"{torch.cuda.max_memory_allocated()/2**30:.1f} GB")
            torch.cuda.reset_peak_memory_stats()
        print(f"    {name:16s}: {dt_s:7.1f} s   E={E:.3f}  "
              f"cap={bool(info['hit_time_cap'][0])}  "
              f"steps={info['n_steps'][0]:.0f}{gpu_mem}")
        report["probes"][name] = {"runtime_s": dt_s, "E": E,
                                  "hit_cap": bool(info["hit_time_cap"][0]),
                                  "n_steps": float(info["n_steps"][0])}
        if bool(info["hit_time_cap"][0]):
            print("      WARNING: hit MAX_TIME cap -- production samples "
                  "like this would be quarantined; consider a larger "
                  "PC MAX_TIME or looser decay_tol.")

    # note: the norm runs for the supercell were built during the first
    # probe and are REUSED by every dataset sample, so the second probe's
    # runtime is the honest per-sample cost.
    t_probe = per_sample[-1]

    # ---------------- ETA table ---------------------------------------------
    from run_dataset import build_plan
    plan = build_plan()
    n_samples = len(plan)
    print(f"\n=== DATASET ETA (plan: {n_samples} samples; probe "
          f"{t_probe:.0f} s/sample) ===")
    print(f"    {'GPUs':>5s} {'wall time':>14s} {'samples/12h':>12s}")
    for g in (1, 2, 4, 8):
        wall_h = n_samples * t_probe / 3600.0 / g
        per12 = int(12 * 3600 / t_probe) * g
        print(f"    {g:5d} {wall_h:11.1f} h  {per12:12d}")
    report["eta"] = {"n_samples": n_samples,
                     "s_per_sample": t_probe,
                     "wall_h_1gpu": n_samples * t_probe / 3600.0}
    print("\n    (samples/12h = how many fit in one 12 h SCC window; the "
          "array job\n     resumes automatically, so multiple windows "
          "just continue the manifest)")

    # ---------------- persist ------------------------------------------------
    report["total_runtime_s"] = time.time() - t_all
    report["all_gates_ok"] = bool(ok)
    jpath = os.path.join(out, "timing_report.json")
    with open(jpath, "w") as f:
        json.dump(report, f, indent=2)
    tpath = os.path.join(out, "timing_report.txt")
    with open(tpath, "w") as f:
        f.write(f"GPU-FDTD timing test  ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})\n")
        f.write(f"mode={C.MODE} device={device} fake={F.FAKE}\n")
        f.write(f"gates: {report['gates']}\n")
        if report["anchor"]:
            f.write(f"anchor: E={report['anchor']['E']:.4f} "
                    f"({100*report['anchor']['rel_err']:+.2f}% vs "
                    f"{E_RCWA_ANCHOR})\n")
        for r in report["ladder"]:
            f.write(f"ladder res={r['res']}: E={r['E']:.4f}\n")
        for k, v in report["probes"].items():
            f.write(f"probe {k}: {v['runtime_s']:.1f} s  E={v['E']:.3f} "
                    f"cap={v['hit_cap']}\n")
        f.write(f"plan {report['eta'].get('n_samples','?')} samples, "
                f"{report['eta'].get('s_per_sample',0):.0f} s/sample, "
                f"{report['eta'].get('wall_h_1gpu',0):.1f} h on 1 GPU\n")
        f.write(f"ALL GATES {'PASS' if ok else 'FAIL'}\n")
    print(f"\n[persisted] {tpath}\n[persisted] {jpath}")

    print("\n=== VERDICT ===")
    if ok:
        print("All gates passed on this device.  Interpretation guide:")
        print("  * anchor within a few % of 2.547 -> engine agrees with "
              "the RCWA/Meep\n    lineage at unit-cell scale; ladder "
              "spread is the discretization\n    uncertainty of the "
              "label.")
        print("  * the ETA table is measured, not estimated: scope "
              "SEEDS_PER_SIGMA in\n    run_dataset.py (env "
              "PC_SEEDS_PER_SIGMA) to fit your GPU budget.")
        print("  * next: qsub submit_dataset.sh")
    else:
        print("At least one gate FAILED -- fix before generating data. "
              "The gate\nprintouts above say which physics check broke; "
              "do not bank labels\nfrom an engine in this state.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
