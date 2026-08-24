"""E vs sigma for the generated dataset, with a choice of spread encoding.

The dataset figure on its own (no inverse-design overlay -- that lives in
models/plot_inverse_results.py).  Per (class, sigma) cell the mean curve is
always drawn; how the *spread* of the ~155 layouts in that cell is shown is
selectable:

  --spread candlestick  (default) wick over the robust range, body over the
                        interquartile range, bar at the mean.  Crisp and
                        quantitative: four readable numbers per cell.
  --spread gradient     a kernel-density ramp, opaque at the mode and fading
                        to transparent by the robust extent.  Shows the shape
                        of each distribution rather than summary statistics.
  --spread band         the classic +-1 std ribbon joined across sigma.

"Robust extent" is a percentile pair (--pct, default 1 99) so a lone outlier
never stretches a cell's column.

Usage:
    python -m models.plot_e_vs_sigma                        # candlestick
    python -m models.plot_e_vs_sigma --spread gradient
    python -m models.plot_e_vs_sigma --spread band --pct 0 100
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BANK = os.path.join(_REPO_ROOT, "scripts", "FDTD_solver", "data_production")

# same palette as scripts/FDTD_solver/run_dataset.py
COLORS = {"jitter": "#1f5fa8", "radius": "#c0392b",
          "random": "#e67e22", "ordered": "#27632a"}
DISORDER_CLASSES = ("jitter", "radius")


# ==========================================================================
# Loading
# ==========================================================================
def load_bank(labels_csvs):
    """labels.csv file(s) -> ({(class, sigma): [E]}, ordered E, random E).

    Accepts one path or a list; extension campaigns (e.g. a high-sigma
    PC_SIGMAS run in its own directory) merge into the main bank's cells.
    The FIRST file's ordered row wins.
    """
    if isinstance(labels_csvs, (str, os.PathLike)):
        labels_csvs = [labels_csvs]
    rows = []
    for p in labels_csvs:
        rows.extend(csv.DictReader(open(p)))
    cells = {}
    for r in rows:
        if r["class"] in DISORDER_CLASSES:
            cells.setdefault((r["class"], float(r["sigma"])), []).append(
                float(r["E"]))
    E_ord = next((float(r["E"]) for r in rows if r["class"] == "ordered"),
                 None)
    E_rand = [float(r["E"]) for r in rows if r["class"] == "random"]
    return cells, E_ord, E_rand


# ==========================================================================
# Spread encodings -- each draws ONE cell and returns its drawn (lo, hi)
# ==========================================================================
def draw_gradient_ribbon(ax, xs, mean, lo, hi, color, fade=1.3, amax=0.80,
                         nx=600, ny=420):
    """A ribbon like the +-1 std band, but shaded by distance from the mean.

    Most intense along the mean curve, fading to fully transparent at the
    robust envelope (lo/hi), so the ribbon has soft edges instead of a hard
    boundary.  Drawn once per class as a single RGBA image, so overlapping
    classes blend the way translucent bands do.
    """
    xf = np.linspace(float(min(xs)), float(max(xs)), nx)
    mf = np.interp(xf, xs, mean)
    lf = np.interp(xf, xs, lo)
    hf = np.interp(xf, xs, hi)
    yf = np.linspace(float(lf.min()), float(hf.max()), ny)

    Y = yf[:, None]
    up = Y >= mf[None, :]
    # normalised distance from the mean: 0 at the mean, 1 at the envelope
    t = np.where(up,
                 (Y - mf[None, :]) / np.maximum(hf - mf, 1e-12)[None, :],
                 (mf[None, :] - Y) / np.maximum(mf - lf, 1e-12)[None, :])
    a = np.clip(1.0 - t, 0.0, 1.0) ** fade
    a[(Y < lf[None, :]) | (Y > hf[None, :])] = 0.0

    rgba = np.zeros((ny, nx, 4))
    rgba[..., :3] = mcolors.to_rgb(color)
    rgba[..., 3] = a * amax
    ax.imshow(rgba, extent=[xf[0], xf[-1], yf[0], yf[-1]], origin="lower",
              aspect="auto", interpolation="bilinear", zorder=1)


def draw_candle(ax, x, E, color, hw, plo, phi):
    """Wick = robust range, body = interquartile range, bar = mean."""
    lo, hi = np.percentile(E, [plo, phi])
    q1, q3 = np.percentile(E, [25, 75])
    ax.plot([x, x], [lo, hi], color=color, lw=1.1, alpha=0.85, zorder=2,
            solid_capstyle="round")
    ax.add_patch(Rectangle((x - hw, q1), 2 * hw, q3 - q1, facecolor=color,
                           alpha=0.35, edgecolor=color, lw=1.1, zorder=3))
    ax.plot([x - hw, x + hw], [float(np.mean(E))] * 2, color=color, lw=2.0,
            zorder=4, solid_capstyle="butt")
    return lo, hi


# ==========================================================================
# Figure
# ==========================================================================
def make_figure(cells, E_ord, E_rand, args):
    import matplotlib.patheffects as pe
    plo, phi = args.pct
    rng = np.random.default_rng(0)          # deterministic dot wobble
    # paper mode: compact fonts sized for a ~2 in column render, no title
    fs = ({"lab": 8, "ann": 7, "rand": 6, "title": 0.0, "leg": 6,
           "dot": 3, "lw": 1.4, "ms": 3.0}
          if args.paper else
          {"lab": 12, "ann": 10, "rand": 9, "title": 11.5, "leg": 9,
           "dot": 6, "lw": 2.2, "ms": 4.5})
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    if args.paper:
        ax.tick_params(labelsize=7)
    yvals = [E_ord] if E_ord is not None else []

    # candlesticks are columns, so the two classes get nudged apart; the
    # gradient/band ribbons overlay like translucent fills and need no dodge.
    # Either way the mean curves and sigma* labels use the TRUE sigma.
    shown = [c for c in DISORDER_CLASSES if c in args.classes]
    dodge = args.dodge if args.spread == "candlestick" else 0.0
    off = {c: (0.0 if len(shown) == 1 else (-dodge, dodge)[i])
           for i, c in enumerate(shown)}

    for cls in shown:
        col = COLORS[cls]
        sigmas = sorted(s for c, s in cells if c == cls)
        if not sigmas:
            continue
        means, stds, plos, phis = [], [], [], []
        for s in sigmas:
            E = np.asarray(cells[(cls, s)], float)
            means.append(float(E.mean()))
            stds.append(float(E.std(ddof=1)) if len(E) > 1 else 0.0)
            a, b = np.percentile(E, [plo, phi])
            plos.append(float(a))
            phis.append(float(b))
            if args.spread == "candlestick":
                lo, hi = draw_candle(ax, s + off[cls], E, col, args.width,
                                     plo, phi)
                yvals += [float(lo), float(hi)]
            elif args.spread == "scatter":
                # every layout as a dot, both classes on the SAME sigma so
                # the clouds overlap; the wobble is only within the column
                xj = s + rng.uniform(-args.jitter, args.jitter, size=len(E))
                ax.scatter(xj, E, s=fs["dot"], color=col,
                           alpha=args.alpha, lw=0, zorder=2)
                yvals += [float(E.min()), float(E.max())]

        # mean curve, launched from the sigma = 0 ordered anchor
        xs = ([0.0] + list(sigmas)) if E_ord is not None else list(sigmas)
        ys = ([E_ord] + means) if E_ord is not None else means
        sd = ([0.0] + stds) if E_ord is not None else stds
        rlo = ([E_ord] + plos) if E_ord is not None else plos
        rhi = ([E_ord] + phis) if E_ord is not None else phis

        if args.spread == "band":
            ax.fill_between(xs, [m - s for m, s in zip(ys, sd)],
                            [m + s for m, s in zip(ys, sd)], color=col,
                            alpha=0.15, lw=0, zorder=1)
            yvals += [min(m - s for m, s in zip(ys, sd)),
                      max(m + s for m, s in zip(ys, sd))]
        elif args.spread == "gradient":
            draw_gradient_ribbon(ax, xs, ys, rlo, rhi, col,
                                 fade=args.fade)
            yvals += [min(rlo), max(rhi)]
        halo = ([pe.withStroke(linewidth=4.5, foreground="white")]
                if args.spread == "scatter" else None)
        ax.plot(xs, ys, "-", color=col, lw=fs["lw"], marker="o",
                ms=fs["ms"], zorder=6,
                path_effects=halo,
                label=f"{cls} disorder (cell mean)")

        # clear the cell's dot cloud, not just the mean marker, so the
        # label never sits inside the scatter
        i = int(np.argmax(ys))
        top = ys[i]
        key = (cls, xs[i])
        if key in cells:
            top = max(top, float(np.max(cells[key])))
        if args.paper:
            continue
        ax.annotate(f"$\\sigma^*\\approx${xs[i]:.2f}", (xs[i], top),
                    textcoords="offset points", xytext=(0, 11), color=col,
                    fontsize=fs["ann"], ha="center", fontweight="bold",
                    zorder=9,
                    bbox={"facecolor": "white", "alpha": 0.85,
                          "edgecolor": "none", "pad": 1.8})

    if E_rand and args.random_line:
        mn = float(np.mean(E_rand))
        if args.spread == "scatter" and len(E_rand) > 1:
            # dots stay reserved for the disorder classes; the random
            # reference reads as a horizontal mean +- 1 std band instead
            sd = float(np.std(E_rand, ddof=1))
            ax.axhspan(mn - sd, mn + sd, color=COLORS["random"], alpha=0.14,
                       lw=0, zorder=1)
            ax.axhline(mn, color=COLORS["random"], ls="-.", lw=1.4, zorder=5,
                       label=f"fully random (mean $\\pm$1 std, "
                             f"n={len(E_rand)})")
            # jitter displacements wrap mod L, so at large sigma the class
            # converges exactly to this measured uniform-placement ensemble
            import matplotlib.transforms as mtransforms
            if args.paper:
                tr = None  # paper mode: caption identifies the random band
            else:
                tr = mtransforms.blended_transform_factory(ax.transAxes,
                                                       ax.transData)
            if tr is not None:
                ax.text(0.985, mn, "$\\sigma\\!\\to\\!\\infty$ limit of "
                    "position disorder", transform=tr, ha="right",
                    va="center", fontsize=fs["rand"],
                    color=COLORS["random"],
                    fontstyle="italic", zorder=8,
                    bbox={"facecolor": "white", "alpha": 0.85,
                          "edgecolor": "none", "pad": 1.5})
            yvals += [mn - sd, mn + sd]
        else:
            ax.axhline(mn, color=COLORS["random"], ls="-.", lw=1.4, zorder=5,
                       label=f"fully random (mean, n={len(E_rand)})")
        yvals.append(mn)
    if E_ord is not None and args.ordered_line:
        ax.axhline(E_ord, color=COLORS["ordered"], ls="--", lw=1.5, zorder=5,
                   label=f"ordered lattice (E = {E_ord:.3f})")

    lo, hi = min(yvals), max(yvals)
    pad = 0.05 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 2.2 * pad)
    ax.set_xlim(-0.014, max(s for _, s in cells) + 0.022)
    ax.set_xlabel("Disorder strength  $\\sigma$", fontsize=fs["lab"])
    ax.set_ylabel("Enhancement  $E$" if args.paper else
                  "Broadband enhancement factor  E", fontsize=fs["lab"])

    what = {
        "candlestick": (f"candles: wick = {plo:g}-{phi:g} percentile, "
                        "body = interquartile range, bar = mean"),
        "gradient": ("ribbon shaded by distance from the mean, fading out "
                     f"to the {plo:g}-{phi:g} percentile envelope"),
        "band": "band = $\\pm$1 std of the layouts in each cell",
        "scatter": ("one dot per simulated layout; both classes plotted at "
                    "the same $\\sigma$"),
    }[args.spread]
    if not args.paper:
        ax.set_title("Enhancement vs disorder strength (7x7 supercell, "
                     f"reflector stack, normal incidence, GPU FDTD)\n{what}",
                     fontsize=fs["title"])
    ax.grid(True, alpha=0.22)
    ax.set_axisbelow(True)
    if not args.paper:
        ax.legend(loc="lower right", fontsize=fs["leg"], framealpha=0.95)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    n = sum(len(v) for v in cells.values())
    print(f"[bank] {len(cells)} cells, {n} disordered layouts")
    print(f"[fig]  spread={args.spread} -> {args.out}")


# ==========================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m models.plot_e_vs_sigma",
        description="E vs sigma for the generated dataset, with a "
                    "selectable spread encoding.")
    ap.add_argument("--spread",
                    choices=("candlestick", "gradient", "band", "scatter"),
                    default="candlestick",
                    help="how to show each cell's spread (default "
                         "candlestick). scatter = every layout as a dot, "
                         "both classes on the same sigma.")
    ap.add_argument("--pct", nargs=2, type=float, default=[1.0, 99.0],
                    metavar=("LO", "HI"),
                    help="robust extent percentiles, so one outlier cannot "
                         "stretch a cell (default 1 99; use 0 100 for the "
                         "true min/max).")
    ap.add_argument("--classes", nargs="+", default=list(DISORDER_CLASSES),
                    choices=list(DISORDER_CLASSES))
    ap.add_argument("--labels", nargs="+",
                    default=[os.path.join(_BANK, "labels.csv")],
                    help="one or more labels.csv files; extra files (e.g. "
                         "a high-sigma extension campaign) merge into the "
                         "same axes.")
    ap.add_argument("--figsize", nargs=2, type=float,
                    default=[10.0, 5.9], metavar=("W", "H"),
                    help="figure size in inches (default 10.0 5.9)")
    ap.add_argument("--paper", action="store_true", default=False,
                    help="compact paper styling: no title, small fonts "
                         "sized for a ~2 in column render")
    ap.add_argument("--out", default=None,
                    help="default: <bank>/figs/fig_e_vs_sigma_<spread>.png")
    ap.add_argument("--width", type=float, default=0.0056,
                    help="half-width of a cell's column in sigma units.")
    ap.add_argument("--dodge", type=float, default=0.0072,
                    help="candlestick only: sigma offset between the two "
                         "classes' columns (0 = same sigma).")
    ap.add_argument("--jitter", type=float, default=0.0055,
                    help="scatter only: half-width of the horizontal wobble "
                         "inside a column, in sigma units (0 = a hard "
                         "vertical line of dots).")
    ap.add_argument("--alpha", type=float, default=0.30,
                    help="scatter only: dot opacity.")
    ap.add_argument("--fade", type=float, default=1.3,
                    help="gradient only: fade exponent. <1 holds colour "
                         "further out, >1 concentrates it near the mean.")
    ap.add_argument("--no-ordered-line", dest="ordered_line",
                    action="store_false", default=True)
    ap.add_argument("--no-random-line", dest="random_line",
                    action="store_false", default=True)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = os.path.join(_BANK, "figs",
                                f"fig_e_vs_sigma_{args.spread}.png")
    return args


def main(argv=None):
    args = parse_args(argv)
    missing = [p for p in args.labels if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"labels.csv not found: {missing} -- run "
                         "run_dataset.py analyze first.")
    cells, E_ord, E_rand = load_bank(args.labels)
    if not cells:
        raise SystemExit("no jitter/radius cells in labels.csv")
    make_figure(cells, E_ord, E_rand, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
