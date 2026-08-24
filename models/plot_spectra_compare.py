"""Ideal ordered photonic crystal vs the best layout found, side by side.

Two SEPARATE panels sharing one y-axis -- never overlaid, so the peak
heights are directly comparable without curves crossing each other.  The
point of the figure is the mechanism behind the enhancement:

  * the ordered ("ideal") crystal concentrates its near-infrared
    absorption into a few tall, narrow resonances with dead gaps between
    them;
  * the best disordered layout has LOWER peaks but spreads absorption
    across the whole band, integrating to substantially more.

The champion is discovered, not hardcoded: the highest FDTD-computed E
across BOTH the generated dataset and every verified inverse-design run.

Usage:
    python -m models.plot_spectra_compare                 # 700-1100 nm
    python -m models.plot_spectra_compare --range 800 950 # the zoom
    python -m models.plot_spectra_compare --range 400 1100
    python -m models.plot_spectra_compare --champion path/to/sample.npz
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BANK = os.path.join(_REPO_ROOT, "scripts", "FDTD_solver", "data_production")

COLORS = {"jitter": "#1f5fa8", "radius": "#c0392b",
          "random": "#e67e22", "ordered": "#27632a"}


# ==========================================================================
# Loading
# ==========================================================================
def load_spectrum(path):
    """Any bank-schema sample or verify_cache npz -> (wl, A_si, E, label)."""
    z = np.load(path, allow_pickle=False)
    wl = np.asarray(z["wavelengths_nm"], float)
    A = np.asarray(z["A_si"], float)
    E = float(z["E"])
    cls = str(z["disorder_class"]) if "disorder_class" in z.files else None
    sigma = float(z["sigma"]) if "sigma" in z.files else None
    return {"wl": wl, "A": A, "E": E, "cls": cls, "sigma": sigma,
            "path": path}


def find_ordered(labels_csv):
    """The sigma = 0 ordered lattice -- located by class, not by sample id."""
    rows = list(csv.DictReader(open(labels_csv)))
    hit = next((r for r in rows if r["class"] == "ordered"), None)
    if hit is None:
        raise SystemExit("no 'ordered' row in labels.csv -- the ideal "
                         "reference lattice is not banked")
    sid = int(hit["sample_id"])
    path = os.path.join(os.path.dirname(labels_csv), "samples",
                        f"sample_{sid:06d}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"ordered sample record missing: {path}")
    return load_spectrum(path)


def best_in_dataset(labels_csv):
    """Highest-E disordered sample in the bank (one file opened, not 2,724)."""
    rows = [r for r in csv.DictReader(open(labels_csv))
            if r["class"] != "ordered"]
    if not rows:
        return None
    hit = max(rows, key=lambda r: float(r["E"]))
    path = os.path.join(os.path.dirname(labels_csv), "samples",
                        f"sample_{int(hit['sample_id']):06d}.npz")
    if not os.path.exists(path):
        return None
    s = load_spectrum(path)
    s["source"] = "generated dataset"
    return s


def best_in_inverse_runs():
    """Highest-E verified layout across every inverse-design campaign.

    Prefers verified_samples/ (bank schema); falls back to a run's
    verify_cache/ entries, which carry E + A_si + wavelengths_nm too.
    """
    # every generation of inverse-design campaign, not just the first --
    # runs/inverse_v2/ holds the retrained model's re-search
    roots = [os.path.join(_REPO_ROOT, "runs", "inverse", "*"),
             os.path.join(_REPO_ROOT, "runs", "inverse_v2", "*"),
             os.path.join(_REPO_ROOT, "scripts", "FDTD_solver",
                          "candidates", "*")]
    best = None
    for pat in roots:
        for d in sorted(glob.glob(pat)):
            if not os.path.isdir(d):
                continue
            files = sorted(glob.glob(os.path.join(d, "verified_samples",
                                                 "sample_*.npz")))
            if not files:            # fallback: the raw solve cache
                files = sorted(glob.glob(os.path.join(d, "verify_cache",
                                                      "*_res60.npz")))
            for f in files:
                try:
                    s = load_spectrum(f)
                except (KeyError, OSError):
                    continue
                if best is None or s["E"] > best["E"]:
                    s["source"] = f"inverse design ({os.path.basename(d)})"
                    best = s
    return best


def band_stats(s, lo, hi):
    """Mean A_si over [lo, hi] (trapezoid / width) plus the tallest peak."""
    m = (s["wl"] >= lo) & (s["wl"] <= hi)
    if m.sum() < 2:
        raise SystemExit(f"window {lo}-{hi} nm holds < 2 grid points")
    wl, A = s["wl"][m], s["A"][m]
    # numpy >= 2 renamed trapz -> trapezoid; don't touch the missing one
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return {"wl": wl, "A": A,
            "mean": float(trapz(A, wl) / (wl.max() - wl.min())),
            "peak": float(A.max()),
            "peak_wl": float(wl[int(np.argmax(A))])}


# ==========================================================================
# Figure
# ==========================================================================
def draw_layout(ax, s):
    """The hole pattern itself, from the same npz the spectrum came from."""
    z = np.load(s["path"], allow_pickle=False)
    if "holes_xyr_nm" not in z.files or "a_super_nm" not in z.files:
        ax.axis("off")
        return
    holes = np.asarray(z["holes_xyr_nm"], float)
    L = float(z["a_super_nm"])
    for x, y, r in holes:
        # periodic images so holes clipped by the supercell edge still read
        # as circles rather than bites taken out of the pattern
        for dx in (-L, 0.0, L):
            for dy in (-L, 0.0, L):
                if abs(x + dx) > L + r or abs(y + dy) > L + r:
                    continue
                ax.add_patch(plt.Circle((x + dx, y + dy), r, color="black",
                                        lw=0))
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    # square data in a non-square box: hug the spectra instead of
    # centring, so the equal-aspect slack does not read as a gutter
    ax.set_aspect("equal", anchor="W")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#999999")


def make_figure(ordered, champ, lo, hi, args):
    so = band_stats(ordered, lo, hi)
    sc = band_stats(champ, lo, hi)

    # stacked: one wavelength axis for both spectra (shared, labelled once),
    # each with its hole pattern beside it
    fig = plt.figure(figsize=(9.8, 5.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.9, 1.0],
                          hspace=0.09, wspace=0.03)
    axl = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[1, 0], sharex=axl, sharey=axl)
    ax_lay_t = fig.add_subplot(gs[0, 1])
    ax_lay_b = fig.add_subplot(gs[1, 1])
    ymax = max(so["peak"], sc["peak"]) * 1.16

    panels = [
        (axl, ordered, so, COLORS["ordered"],
         f"Ideal ordered crystal  ($\\sigma$ = 0)"),
        (axr, champ, sc, COLORS.get(champ["cls"], "#c0392b"),
         f"Best layout found"
         + (f"  ({champ['cls']}, $\\sigma$ = {champ['sigma']:g})"
            if champ["cls"] else "")),
    ]
    for ax, s, bs, col, title in panels:
        ax.fill_between(bs["wl"], 0, bs["A"], color=col, alpha=0.18, lw=0)
        ax.plot(bs["wl"], bs["A"], color=col, lw=1.5)
        ax.axhline(bs["mean"], color=col, ls="--", lw=1.6, alpha=0.9)
        # park the label over the stretch of band where the spectrum leaves
        # the most headroom, so tall narrow resonances never cut through it
        win = 0.30 * (hi - lo)
        centres = np.linspace(lo + win / 2, hi - win / 2, 40)

        def clearance(xc, _bs=bs, _w=win):
            m2 = (_bs["wl"] >= xc - _w / 2) & (_bs["wl"] <= xc + _w / 2)
            return float(_bs["A"][m2].max()) if m2.any() else np.inf

        xc = float(min(centres, key=clearance))
        ax.annotate(f"band average  $A_{{\\mathrm{{Si}}}}$ = {bs['mean']:.3f}",
                    (xc, bs["mean"]), textcoords="offset points",
                    xytext=(0, 6), ha="center", va="bottom", fontsize=10,
                    color=col, fontweight="bold", zorder=8,
                    bbox={"facecolor": "white", "alpha": 0.85,
                          "edgecolor": "none", "pad": 1.5})
        ax.annotate(f"tallest peak {bs['peak']:.3f}",
                    (bs["peak_wl"], bs["peak"]), textcoords="offset points",
                    xytext=(6, 2), ha="left", va="bottom", fontsize=9,
                    color="#444444")
        # row label inside the axes (top-right, where both spectra have
        # headroom) instead of a title bar -- keeps the two rows tight
        ax.annotate(f"{title}\nE = {s['E']:.4f}", (0.985, 0.93),
                    xycoords="axes fraction", ha="right", va="top",
                    fontsize=11, color=col, fontweight="bold", zorder=9,
                    bbox={"facecolor": "white", "alpha": 0.85,
                          "edgecolor": "none", "pad": 2.5})
        ax.set_xlim(lo, hi)
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)

    # one wavelength axis, labelled once at the bottom
    axl.tick_params(labelbottom=False)
    axr.set_xlabel("Wavelength  (nm)", fontsize=11.5)
    axl.set_ylim(0, ymax)
    fig.supylabel("Silicon absorption  $A_{\\mathrm{Si}}$", fontsize=11.5,
                  x=0.035)

    draw_layout(ax_lay_t, ordered)
    draw_layout(ax_lay_b, champ)

    gain = 100 * (sc["mean"] / so["mean"] - 1) if so["mean"] else float("nan")
    fig.suptitle(
        "Disorder flattens the resonances: lower peaks, more total "
        "absorption\n"
        f"{lo:g}-{hi:g} nm band average {so['mean']:.3f} $\\rightarrow$ "
        f"{sc['mean']:.3f} ({gain:+.0f} %) while the tallest peak falls "
        f"{so['peak']:.3f} $\\rightarrow$ {sc['peak']:.3f}",
        fontsize=12, y=0.995)
    fig.tight_layout(rect=(0.02, 0, 1, 0.93))

    # match each hole-pattern panel to its spectrum's height, and make it
    # square in INCHES (equal aspect would otherwise shrink it to fit the
    # narrower gridspec column, leaving it stubby beside a tall spectrum)
    fig.canvas.draw()
    fw, fh = fig.get_size_inches()
    for ax_lay, ax_spec in ((ax_lay_t, axl), (ax_lay_b, axr)):
        p, q = ax_spec.get_position(), ax_lay.get_position()
        ax_lay.set_aspect("auto")          # the box is square already
        ax_lay.set_position([q.x0, p.y0, p.height * fh / fw, p.height])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # bbox_inches="tight" crops whatever slack the squared-off panels left
    # at the right edge, so the figure ends where the content does
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[{lo:g}-{hi:g} nm]  band average A_si: "
          f"ordered {so['mean']:.4f} -> champion {sc['mean']:.4f} "
          f"({gain:+.1f} %)")
    print(f"                  tallest peak:     "
          f"ordered {so['peak']:.4f} -> champion {sc['peak']:.4f}")
    print(f"[fig] -> {args.out}")


# ==========================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m models.plot_spectra_compare",
        description="Ideal ordered crystal vs the best FDTD-verified "
                    "layout, as two side-by-side spectra.")
    ap.add_argument("--labels", default=os.path.join(_BANK, "labels.csv"))
    ap.add_argument("--range", nargs=2, type=float, default=[700.0, 1100.0],
                    metavar=("LO", "HI"),
                    help="wavelength window in nm (default 700 1100).")
    ap.add_argument("--champion", default=None,
                    help="override auto-discovery with a specific sample "
                         "or verify_cache npz.")
    ap.add_argument("--out", default=os.path.join(
        _BANK, "figs", "fig_spectra_ideal_vs_champion.png"))
    ap.add_argument("--dpi", type=int, default=150)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.exists(args.labels):
        raise SystemExit(f"labels.csv not found: {args.labels}")

    ordered = find_ordered(args.labels)
    print(f"[ideal]    ordered lattice  E = {ordered['E']:.4f}  "
          f"({os.path.relpath(ordered['path'], _REPO_ROOT)})")

    if args.champion:
        champ = load_spectrum(args.champion)
        champ["source"] = "supplied via --champion"
    else:
        pool = [s for s in (best_in_dataset(args.labels),
                            best_in_inverse_runs()) if s]
        if not pool:
            raise SystemExit("found no candidate spectra in the dataset or "
                             "the inverse-design runs")
        pool.sort(key=lambda s: -s["E"])
        champ = pool[0]
        for s in pool[1:]:
            print(f"[runner-up] {s['source']:<38s} E = {s['E']:.4f}")
    print(f"[champion] {champ.get('source', '?'):<38s} E = {champ['E']:.4f}  "
          f"({os.path.relpath(champ['path'], _REPO_ROOT)})")

    make_figure(ordered, champ, float(args.range[0]), float(args.range[1]),
                args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
