"""
bench_step.py -- measure the raw FDTD step rate at production numerics.

Answers in ~5 minutes what a silent multi-hour probe cannot: how many
milliseconds one timestep costs on THIS GPU, per band, and therefore
what a run / a sample / the campaign actually costs -- plus whether
torch.compile buys anything.

    python -u bench_step.py

No files written; prints everything.  Safe to run alongside nothing else
on the GPU (kill other runs first or numbers will lie).
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np


# solver modules live one directory up from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from materials_gpu import fit_all
import fdtd_torch as F


def bench(sim, n_warm=30, n_meas=150):
    torch = sim.torch
    for i in range(n_warm):
        sim._step(i)
    if sim.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for i in range(n_warm, n_warm + n_meas):
        sim._step(i)
    if sim.device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / n_meas


def main():
    device = F.resolve_device(os.environ.get("PC_DEVICE", "auto"))
    print(C.describe())
    print(f"[device] {device}", end="")
    if device == "cuda":
        import torch
        print(f"  ({torch.cuda.get_device_name(0)})")
    else:
        print("\n  WARNING: benchmarking on CPU -- GPU numbers needed.")
    fits, adapters, mats = fit_all()

    holes, a_sup = F.ordered_square_holes(C.A_NM, C.N_CELLS,
                                          C.R_OVER_A * C.A_NM)
    wl = C.raw_wavelength_grid()
    idx = F.split_grid_by_band(wl)

    print(f"\nsupercell {C.N_CELLS}x{C.N_CELLS}  res={C.RESOLUTION} "
          f"-> grid {int(round(a_sup/1000*C.RESOLUTION))}^2 x "
          f"~{int(np.ceil(1.55*C.RESOLUTION))+1}")
    total_run_s_cap = 0.0
    for b in range(3):
        freqs = 1000.0 / wl[idx[b]]
        sim = F._Sim(holes, a_sup, C.THICKNESS_NM, C.BUFFER_NM,
                     C.RESOLUTION, b, "x", freqs, fits, device, 32,
                     vacuum=False)
        ms = bench(sim) * 1e3
        n_cap = int(np.ceil(C.MAX_TIME / sim.dt))
        run_cap_h = ms / 1e3 * n_cap / 3600
        total_run_s_cap += ms / 1e3 * n_cap
        mem = ""
        if device == "cuda":
            import torch
            mem = (f"  peakmem "
                   f"{torch.cuda.max_memory_allocated()/2**30:.1f} GB")
            torch.cuda.reset_peak_memory_stats()
        print(f"  band {b+1} ({F.BANDS[b][0]:.0f}-{F.BANDS[b][1]:.0f}): "
              f"{ms:7.2f} ms/step  dt={sim.dt:.5f}  "
              f"steps@cap={n_cap:,}  run@cap={run_cap_h:5.2f} h{mem}")
        del sim
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()

    per_sample_cap_h = 2 * total_run_s_cap / 3600   # 2 polarizations
    print(f"\nWORST CASE (every run rings to the cap):")
    print(f"  per sample : {per_sample_cap_h:6.1f} h")
    print(f"  1551 samples on 8 GPUs: "
          f"{1551*per_sample_cap_h/8/24:6.1f} days")
    print("  (disordered samples ring down faster than this ordered "
          "worst case;\n   the real number needs one full jitter-sample "
          "probe, but if the\n   worst case is absurd, the config must "
          "shrink first.)")

    # ---- torch.compile probe -------------------------------------------
    print("\n[torch.compile probe on band 2 (the long-ring band)]")
    try:
        import torch
        freqs = 1000.0 / wl[idx[1]]
        sim = F._Sim(holes, a_sup, C.THICKNESS_NM, C.BUFFER_NM,
                     C.RESOLUTION, 1, "x", freqs, fits, device, 32,
                     vacuum=False)
        ms_eager = bench(sim, 10, 60) * 1e3
        del sim
        if device == "cuda":
            torch.cuda.empty_cache()
        os.environ["PC_COMPILE"] = "1"
        t0 = time.time()
        simc = F._Sim(holes, a_sup, C.THICKNESS_NM, C.BUFFER_NM,
                      C.RESOLUTION, 1, "x", freqs, fits, device, 32,
                      vacuum=False)
        simc._step(0)                    # compile happens on first step
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"  compile latency: {time.time()-t0:.0f} s")
        ms_comp = bench(simc, 30, 120) * 1e3
        os.environ["PC_COMPILE"] = "0"
        print(f"  eager {ms_eager:.2f} ms/step  ->  compiled "
              f"{ms_comp:.2f} ms/step  ({ms_eager/ms_comp:.2f}x)")
        if ms_comp < 0.8 * ms_eager:
            print("  -> torch.compile HELPS: set PC_COMPILE=1 in the "
                  "submit scripts\n     (and module-load the modern gcc "
                  "there too), then RE-RUN the\n     gates once under "
                  "PC_COMPILE=1 before banking any labels.")
        else:
            print("  -> not a meaningful win here.")
    except Exception as e:
        os.environ["PC_COMPILE"] = "0"
        print(f"  torch.compile probe failed ({type(e).__name__}: {e}) "
              "-- not available on this stack; ignore.")

    # ---- what-if table ---------------------------------------------------
    print("\n[scaling estimates from measured band-2 eager rate]")
    print("  (per-step scales ~ with cell count; steps ~ res * sim-time)")
    base_cells = (C.N_CELLS * C.A_NM / 1000 * C.RESOLUTION) ** 2 \
        * (1.55 * C.RESOLUTION)
    for (n, res, cap) in ((10, 80, 1500), (10, 80, 600), (10, 60, 600),
                          (8, 60, 600), (6, 60, 600)):
        cells = (n * C.A_NM / 1000 * res) ** 2 * (1.55 * res)
        steps = cap / (0.5 / res)
        rel = (cells / base_cells) * (steps / (C.MAX_TIME
                                               / (0.5 / C.RESOLUTION)))
        print(f"  N={n:2d} res={res:3d} cap={cap:5d}: worst-case sample "
              f"~ {per_sample_cap_h*rel:6.1f} h  -> 8 GPUs, 1551 samples:"
              f" {1551*per_sample_cap_h*rel/8/24:6.1f} d")
    print("\nSend this whole printout back for the rescoping decision.")


if __name__ == "__main__":
    main()