"""Head-to-head cost benchmark: one production FDTD label vs one SEER
evaluation, on this machine's GPU. Produces the exact numbers behind the
papers' cost sentence ("~X minutes per FDTD label versus Y--Z ms per
surrogate evaluation").

Both sides score the SAME jitter layouts:
  FDTD side       broadband_absorption_many at production numerics
                  (PC_MODE=FULL: N=7 supercell, 60 px/um, both pols),
                  the same call run_dataset.py makes per sample.
  Surrogate side  SurrogateScorer.score_holes on the deployed bundle
                  (5-member ensemble, D4 TTA, exact anti-aliased raster),
                  the same call inverse_design.py makes per candidate.

The first FDTD solve builds/loads the norm cache, so its time is reported
but excluded from the headline (matches run_timing_test.py's reasoning:
norm runs are reused by every dataset sample).

Usage (H200/A100/any CUDA node; ~30-40 min for the default 3 FDTD solves):

    cd /project/rise-batteries/Photonics_RISE
    PC_MODE=FULL python -u scripts/time_fdtd_vs_surrogate.py --device cuda

Flags: --n-fdtd 3, --sigma 0.10, --repeats 20 (single-eval latency),
--batch 256 (throughput), --skip-fdtd / --skip-surrogate, --json PATH.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("PC_MODE", "FULL")   # production numerics, before config import

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts", "FDTD_solver"))

import numpy as np
import torch

import config as C                                    # noqa: E402
import fdtd_torch as F                                # noqa: E402
from disorder import make_layout, holes_array         # noqa: E402
from materials_gpu import fit_all                     # noqa: E402
from optics_core import planar_reference_stack, solar_weight  # noqa: E402
from models.inverse_design import SurrogateScorer     # noqa: E402

DEPLOYED = os.path.join(_REPO, "runs", "surrogate_128_fft_nll_sweep",
                        "surrogate_bundle.pt")


def label_E(wl, A_si, adapters):
    w = solar_weight(wl)
    ref = planar_reference_stack(adapters["si"], adapters["zno"],
                                 adapters["ag"], wl, C.THICKNESS_NM,
                                 C.BUFFER_NM)
    eta_p = float(np.sum(w * np.asarray(ref["A_si"])))
    return float(np.sum(w * A_si)) / eta_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=os.environ.get("PC_DEVICE", "auto"))
    ap.add_argument("--bundle", default=DEPLOYED)
    ap.add_argument("--sigma", type=float, default=0.10)
    ap.add_argument("--n-fdtd", type=int, default=3,
                    help="FDTD solves (first is warmup, excluded)")
    ap.add_argument("--repeats", type=int, default=20,
                    help="repeats for single-layout surrogate latency")
    ap.add_argument("--batch", type=int, default=256,
                    help="batch size for surrogate throughput")
    ap.add_argument("--skip-fdtd", action="store_true")
    ap.add_argument("--skip-surrogate", action="store_true")
    ap.add_argument("--json", default=os.path.join(
        _REPO, "runs", "timing_fdtd_vs_surrogate.json"))
    args = ap.parse_args()

    device = F.resolve_device(args.device)
    gpu = (torch.cuda.get_device_name(0) if device == "cuda" else "CPU")
    print(f"[device] {device} ({gpu})")
    print(f"[mode]   {C.MODE}  N={C.N_CELLS}x{C.N_CELLS}  "
          f"res={C.RESOLUTION} px/um  decay={C.DECAY_TOL:g}")
    if C.MODE != "FULL":
        raise SystemExit("refusing to time non-production numerics; "
                         "run with PC_MODE=FULL")

    # same-layout probes for both sides
    recs = [make_layout("jitter", args.sigma, C.BASE_SEED + 2_020_000 + i,
                        C.A_NM, C.N_CELLS, C.R_OVER_A * C.A_NM, C.W_MIN_NM)
            for i in range(max(args.n_fdtd, 1))]
    holes = [[tuple(h) for h in holes_array(r)] for r in recs]
    a_sup = recs[0]["a_super_nm"]
    report = {"device": device, "gpu": gpu, "mode": C.MODE,
              "sigma": args.sigma, "fdtd": {}, "surrogate": {}}

    # ---------------- FDTD: one production label ---------------------------
    if not args.skip_fdtd:
        wl = C.raw_wavelength_grid()
        fits, adapters, _ = fit_all()
        norm_dir = os.path.join(C.OUT_DIR, "norm_cache")
        times, Es = [], []
        for i, h in enumerate(holes):
            t0 = time.time()
            A_si, _, info = F.broadband_absorption_many(
                [h], a_sup, C.THICKNESS_NM, wl, fits, C.BUFFER_NM,
                C.RESOLUTION, C.DECAY_TOL, C.MAX_TIME, norm_dir,
                device=device, n_cells_tag=f"sc{C.N_CELLS}")
            dt = time.time() - t0
            E = label_E(wl, A_si[0], adapters)
            times.append(dt)
            Es.append(E)
            tag = "warmup, excluded" if i == 0 and len(holes) > 1 else "counted"
            print(f"  [fdtd] solve {i + 1}/{len(holes)}: {dt:7.1f} s "
                  f"({dt / 60:.2f} min)  E={E:.4f}  "
                  f"cap={bool(info['hit_time_cap'][0])}  ({tag})")
        counted = times[1:] if len(times) > 1 else times
        med = float(np.median(counted))
        report["fdtd"] = {"per_solve_s": times, "E": Es,
                          "median_s": med, "median_min": med / 60.0}
        print(f"  [fdtd] median per label: {med:.1f} s = {med / 60:.2f} min")

    # ---------------- SEER: one evaluation ---------------------------------
    if not args.skip_surrogate:
        scorer = SurrogateScorer(args.bundle, device, use_tta=True,
                                 kappa=0.2, batch_size=args.batch)
        scorer.score_holes(holes[:1], a_sup)          # warmup / JIT / alloc
        if device == "cuda":
            torch.cuda.synchronize()

        lat = []
        for _ in range(args.repeats):                 # deployment latency
            t0 = time.time()
            scorer.score_holes(holes[:1], a_sup)      # raster + 5 members x 8 TTA
            lat.append(time.time() - t0)
        lat_ms = float(np.median(lat) * 1e3)

        big = [holes[i % len(holes)] for i in range(args.batch)]
        t0 = time.time()                              # search-mode throughput
        scorer.score_holes(big, a_sup)
        thr_ms = float((time.time() - t0) / args.batch * 1e3)

        report["surrogate"] = {
            "single_eval_ms": lat_ms, "batched_ms_per_layout": thr_ms,
            "batch": args.batch, "tta": True, "members": len(scorer.models)}
        print(f"  [seer] single evaluation (batch 1):   {lat_ms:8.1f} ms")
        print(f"  [seer] batched (batch {args.batch}), per layout: "
              f"{thr_ms:8.2f} ms")

    # ---------------- paper-ready sentence ----------------------------------
    if report["fdtd"] and report["surrogate"]:
        f_min = report["fdtd"]["median_min"]
        lo = report["surrogate"]["batched_ms_per_layout"]
        hi = report["surrogate"]["single_eval_ms"]
        r = f_min * 60e3 / hi
        print(f"\n  paper line ({gpu}): ~{f_min:.0f} minutes per FDTD label "
              f"versus {lo:.1f}--{hi:.0f} ms per surrogate evaluation "
              f"(>= {r:,.0f}x)")

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()
