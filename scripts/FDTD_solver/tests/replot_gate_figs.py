"""
replot_gate_figs.py
-------------------
(Re)build fig_gate_planar_1D.png / fig_gate_planar_3D.png WITHOUT
re-running the full timing test.

Two paths per figure:
  * a sidecar .npz (written by run_timing_test.py next to each figure)
    exists -> replot from it: pure numpy + matplotlib, no GPU, seconds.
    This is the styling-iteration loop.
  * no sidecar (figure predates the sidecar mechanism, or fresh PC_OUT)
    -> solve ONLY the figure data on the GPU (dense pseudo-1D + hole-free
    supercell at production numerics; vacuum norms come from the warm
    norm_cache, so ~10 min total on an A100) and write both the figure
    and its sidecar.  Needs a GPU node, but not the gates/probes/anchor.

Usage:
    PC_OUT=gpu_out python -u replot_gate_figs.py            # both figures
    PC_OUT=gpu_out python -u replot_gate_figs.py --replot-only   # never solve
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np


# solver modules live one directory up from tests/
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TESTS_DIR))   # solver modules
sys.path.insert(0, _TESTS_DIR)                    # sibling test modules

import config as C
from run_timing_test import _draw_planar_fig, _planar_gate_fig

FIG_NAMES = ("fig_gate_planar_1D", "fig_gate_planar_3D")


def _load_panels(data_path):
    z = np.load(data_path, allow_pickle=False)
    panels = [{k: z[f"{k}{i}"] for k in
               ("wl_d", "tmm_d", "wl_b", "a_f", "err")}
              for i in range(int(z["n_bands"]))]
    return panels, str(z["suptitle"]), str(z["fdtd_label"])


def _compute(name, figs):
    """Solve just this figure's FDTD data (mirrors the run_timing_test
    figure blocks; suptitles kept in sync with the driver by hand)."""
    import fdtd_torch as F
    from materials_gpu import fit_all
    fits, adapters, _ = fit_all()
    norm_dir = os.path.join(C.OUT_DIR, "norm_cache")
    device = F.resolve_device(os.environ.get("PC_DEVICE", "auto"))
    if name == "fig_gate_planar_1D":
        _, res = F.validate_planar_pseudo1d(
            fits, adapters, C.THICKNESS_NM, C.BUFFER_NM, norm_dir,
            resolution=C.GATE_PLANAR_RES, tol=None, device=device,
            n_wl=25)
        suptitle = (f"pseudo-1D gate: engine vs analytic TMM (same fitted "
                    f"materials, res={C.GATE_PLANAR_RES}/um)")
        label = "torch-FDTD"
    else:
        _, res = F.validate_uniform_3d(
            fits, adapters, C.A_SUPER_NM, C.THICKNESS_NM, C.BUFFER_NM,
            C.RESOLUTION, C.DECAY_TOL, C.MAX_TIME, norm_dir,
            tol_bands=None, device=device,
            n_cells_tag=f"unifsc{C.N_CELLS}", n_wl=25)
        suptitle = (f"3D uniform slab at production numerics "
                    f"({C.N_CELLS}x{C.N_CELLS} supercell, "
                    f"res={C.RESOLUTION}/um) vs analytic TMM")
        label = "torch-FDTD (3D)"
    if not res:
        raise SystemExit(f"{name}: solver returned no data "
                         "(FDTD_FAKE=1?) -- nothing to plot.")
    _planar_gate_fig(os.path.join(figs, name + ".png"), res, adapters,
                     suptitle, label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot-only", action="store_true",
                    help="never solve; fail if a sidecar .npz is missing")
    args = ap.parse_args()

    figs = os.path.join(C.OUT_DIR, "figs")
    for name in FIG_NAMES:
        png = os.path.join(figs, name + ".png")
        data = os.path.join(figs, name + ".npz")
        if os.path.exists(data):
            panels, suptitle, label = _load_panels(data)
            _draw_planar_fig(png, panels, suptitle, label)
            print(f"replotted from sidecar -> {png}")
        elif args.replot_only:
            raise SystemExit(
                f"{name}: no sidecar data at {data} -- run without "
                "--replot-only on a GPU node once to create it.")
        else:
            print(f"{name}: no sidecar data, solving figure data "
                  "(GPU, warm norm cache)...")
            _compute(name, figs)
            print(f"solved + figure + sidecar -> {png}")


if __name__ == "__main__":
    main()
