"""Inverse-design results plotted over the dataset's E-vs-sigma curve.

One figure, two layers:

  * BASE (the dataset): per-(class, sigma) mean +- SEM of the banked random
    draws with +-1 std bands, launched from the ordered-lattice anchor at
    sigma = 0, plus the fully-random and ordered reference lines.  This is
    the population every search was run against.
  * OVERLAY (the searches): for each FDTD-verified inverse-design campaign,
    a star at the champion's true E, a circle at the mean of its verified
    candidates, and a thin bar spanning their min-max.  A dashed line marks
    the best single layout in the whole bank, so "beats every layout we
    ever simulated" is visible at a glance.

Campaigns are DISCOVERED, not listed: every directory under runs/inverse/
(and the solver's candidates/ staging folder) that contains a
verification_verdict.json is picked up, and its class/sigma are read from
that json rather than from the folder name -- so future runs at any sigma,
under any naming scheme, appear with no edit to this file.  Directories
that have not been verified yet are reported and skipped.

Usage:
    python -m models.plot_inverse_results                  # everything
    python -m models.plot_inverse_results --classes radius # one class
    python -m models.plot_inverse_results path/to/other_run

Outputs (next to the figure): fig_inverse_vs_dataset.png and the matching
.csv of every number in the printed table.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BANK = os.path.join(_REPO_ROOT, "scripts", "FDTD_solver", "data_production")

# same palette as scripts/FDTD_solver/run_dataset.py
COLORS = {"jitter": "#1f5fa8", "radius": "#c0392b",
          "random": "#e67e22", "ordered": "#27632a"}
DISORDER_CLASSES = ("jitter", "radius")


# ==========================================================================
# Loading
# ==========================================================================
def load_bank(labels_csv):
    """labels.csv -> (per-cell stats, ordered E, random E list, n_disordered).

    Per-cell stats: {(class, sigma): {mean, std, sem, max, n}} for the
    disorder classes only.
    """
    rows = list(csv.DictReader(open(labels_csv)))
    cells = {}
    by_cell = {}
    for r in rows:
        cls = r["class"]
        if cls not in DISORDER_CLASSES:
            continue
        by_cell.setdefault((cls, float(r["sigma"])), []).append(float(r["E"]))
    for k, v in by_cell.items():
        cells[k] = {
            "mean": st.mean(v), "max": max(v), "n": len(v), "E": v,
            "std": st.stdev(v) if len(v) > 1 else 0.0,
            "sem": (st.stdev(v) / len(v) ** 0.5) if len(v) > 1 else 0.0,
        }
    E_ord = next((float(r["E"]) for r in rows if r["class"] == "ordered"),
                 None)
    E_rand = [float(r["E"]) for r in rows if r["class"] == "random"]
    # "disordered" = every class except the sigma=0 ordered reference, i.e.
    # the fully-random class counts too (this is the 2,723 the paper quotes).
    E_dis = [float(r["E"]) for r in rows if r["class"] != "ordered"]
    return cells, E_ord, E_rand, len(E_dis), max(E_dis)


def discover_campaigns(extra_dirs, discover=True):
    """Every verified inverse-design directory, newest naming irrelevant.

    Returns (campaigns, skipped) where a campaign is a dict carrying the
    verdict json, the verified E values, and the source directory.

    discover=False skips the default roots so an explicit directory list
    (e.g. a single model generation's campaigns) stands alone instead of
    being merged with them -- otherwise the same cell appears twice.
    """
    roots = [os.path.join(_REPO_ROOT, "runs", "inverse", "*"),
             os.path.join(_REPO_ROOT, "scripts", "FDTD_solver",
                          "candidates", "*")]
    dirs = ([d for pat in roots for d in sorted(glob.glob(pat))
             if os.path.isdir(d)] if discover else [])
    dirs += [os.path.abspath(d) for d in extra_dirs]

    campaigns, skipped = [], []
    seen = set()
    for d in dirs:
        real = os.path.realpath(d)
        if real in seen:
            continue
        seen.add(real)
        vp = os.path.join(d, "verification_verdict.json")
        cp = os.path.join(d, "verification.csv")
        if not os.path.exists(vp) or not os.path.exists(cp):
            # a bare export (or not an export at all)
            if os.path.exists(os.path.join(d, "manifest.json")):
                skipped.append(d)
            continue
        with open(vp) as f:
            verdict = json.load(f)
        rows = list(csv.DictReader(open(cp)))
        E = [float(r["true_E60"]) for r in rows]
        if not E:
            skipped.append(d)
            continue
        campaigns.append({
            "dir": d,
            "name": os.path.basename(d.rstrip(os.sep)),
            "cls": verdict["cell"]["class"],
            "sigma": float(verdict["cell"]["sigma"]),
            "E": E,
            "rows": rows,
            "verdict": verdict,
        })
    campaigns.sort(key=lambda c: (c["cls"], c["sigma"]))
    return campaigns, skipped


# ==========================================================================
# Reporting
# ==========================================================================
def campaign_stats(c, global_max):
    """Every quotable number for one campaign."""
    v, E = c["verdict"], c["E"]
    bank = v.get("bank_cell", {})
    bmean, bmax = bank.get("mean"), bank.get("max")
    champ = max(E)
    gains = [float(r["gain_vs_cell_mean_pct"]) for r in c["rows"]]
    return {
        "campaign": c["name"], "class": c["cls"], "sigma": c["sigma"],
        "n": len(E), "n_claimable": v.get("n_claimable"),
        "bank_n": bank.get("n"),
        "bank_mean": bmean, "bank_max": bmax,
        "champion_E": champ, "cand_mean_E": st.mean(E),
        "cand_min_E": min(E), "cand_max_E": champ,
        "mean_gain_pct": st.mean(gains), "champ_gain_pct": max(gains),
        "mean_gain_sem_pct": (st.stdev(gains) / len(gains) ** 0.5
                              if len(gains) > 1 else 0.0),
        "best_draw_gain_pct": (100 * (bmax / bmean - 1)
                               if bmean and bmax else float("nan")),
        "champ_vs_cell_max_pct": (100 * (champ / bmax - 1)
                                  if bmax else float("nan")),
        "champ_vs_global_max_pct": 100 * (champ / global_max - 1),
        "n_above_cell_max": (sum(1 for e in E if bmax and e > bmax)
                             if bmax else 0),
        "n_above_global_max": sum(1 for e in E if e > global_max),
        "spearman_pred_true": v.get("spearman_pred_true"),
        "surrogate_bias": v.get("surrogate_bias_on_champions"),
    }


def print_table(stats, global_max, n_dis):
    hdr = (f"{'campaign':22s} {'class':6s} {'sigma':>6s} {'n':>3s} "
           f"{'clm':>4s} {'bank mean':>9s} {'champion':>9s} "
           f"{'mean+':>7s} {'champ+':>7s} {'>cell':>6s} {'>glob':>6s} "
           f"{'rho':>6s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for s in stats:
        print(f"{s['campaign']:22s} {s['class']:6s} {s['sigma']:6.3f} "
              f"{s['n']:3d} {s['n_claimable'] or 0:4d} "
              f"{s['bank_mean']:9.4f} {s['champion_E']:9.4f} "
              f"{s['mean_gain_pct']:+6.2f}% {s['champ_gain_pct']:+6.2f}% "
              f"{s['n_above_cell_max']:3d}/{s['n']:<2d} "
              f"{s['n_above_global_max']:3d}/{s['n']:<2d} "
              f"{s['spearman_pred_true']:+6.2f}")
    tot = sum(s["n"] for s in stats)
    clm = sum(s["n_claimable"] or 0 for s in stats)
    above = sum(s["n_above_global_max"] for s in stats)
    best = max(stats, key=lambda s: s["champion_E"])
    print("-" * len(hdr))
    print(f"  {len(stats)} campaigns, {tot} verified candidates, "
          f"{clm} claimable (> 0.30 % floor)")
    print(f"  best single layout in the bank: E = {global_max:.4f} "
          f"(of {n_dis} disordered samples)")
    print(f"  candidates beating it: {above}/{tot} "
          f"({100 * above / tot:.0f} %)")
    print(f"  best designed layout: E = {best['champion_E']:.4f} "
          f"({best['class']} sigma={best['sigma']:g}), "
          f"{best['champ_vs_global_max_pct']:+.2f} % vs the bank's best")
    print("  '>cell' / '>glob' = candidates above their own cell's best "
          "banked draw / above the bank's global best")


# ==========================================================================
# Figure
# ==========================================================================
def make_figure_scatter(cells, campaigns, global_max, n_dis, args):
    """Every layout as a dot, in real E, one column pair per disorder cell.

    Dots (unlike bars) carry no proportion-from-zero implication, so the
    y-axis can be zoomed to where the physics actually lives instead of
    being padded down to 0.  Left column = the banked random layouts of
    that cell, right column = the FDTD-verified designed ones.
    """
    import random as _random
    import matplotlib.patheffects as pe
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    camps = [c for c in campaigns if c["cls"] in args.classes]
    if not camps:
        raise SystemExit("no campaigns to plot for the selected classes")

    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    rng = _random.Random(0)             # deterministic jitter
    GREY = "#8c8c8c"
    HALO = [pe.withStroke(linewidth=5.5, foreground="white")]
    yvals = []

    for i, c in enumerate(camps):
        col = COLORS.get(c["cls"], "#444444")
        cell = cells.get((c["cls"], c["sigma"]))

        # both populations share ONE column per cell: the vertical gap
        # between the clouds is the whole point, so nothing is split
        if cell:
            hi = cell["mean"] * (1 + args.floor / 100.0)
            ax.fill_between([i - 0.42, i + 0.42], cell["mean"], hi,
                            color="#999999", alpha=0.30, lw=0, zorder=1)
            xs = [i + rng.uniform(-0.165, 0.165) for _ in cell["E"]]
            ax.scatter(xs, cell["E"], s=7, color=GREY, alpha=0.30,
                       lw=0, zorder=2)
            ax.plot([i - 0.36, i + 0.36], [cell["mean"]] * 2,
                    color="#2b2b2b", lw=3.4, zorder=4,
                    solid_capstyle="round", path_effects=HALO)
            yvals += [min(cell["E"]), max(cell["E"])]

        xs = [i + rng.uniform(-0.125, 0.125) for _ in c["E"]]
        ax.scatter(xs, c["E"], s=32, color=col, alpha=0.85, lw=0.6,
                   edgecolor="white", zorder=5)
        ax.plot([i - 0.36, i + 0.36], [st.mean(c["E"])] * 2, color=col,
                lw=3.4, zorder=6, solid_capstyle="round", path_effects=HALO)
        yvals += [min(c["E"]), max(c["E"])]

    if args.global_line:
        # labelled in the legend, not in-plot: this line necessarily runs
        # through the cloud that contains the record layout
        ax.axhline(global_max, color="#333333", ls=(0, (6, 3)), lw=1.3,
                   zorder=3)
        yvals.append(global_max)

    lo, hi = min(yvals), max(yvals)
    # a lone far-tail dataset point otherwise steals ~15 % of the canvas:
    # clip the view at N std below the lowest cell mean and say how many
    # points that hides (they are still in the table and the CSV).
    n_hidden = 0
    if args.clip_sigma > 0:
        used = [cells[(c["cls"], c["sigma"])] for c in camps
                if (c["cls"], c["sigma"]) in cells]
        if used:
            lo_clip = min(u["mean"] - args.clip_sigma * u["std"]
                          for u in used)
            if lo_clip > lo:
                n_hidden = sum(1 for u in used for e in u["E"]
                               if e < lo_clip)
                lo = lo_clip
    pad = 0.04 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 2.0 * pad)
    ax.set_xlim(-0.65, len(camps) - 0.35)
    if n_hidden:                        # reported to the console, not drawn
        print(f"[clip] {n_hidden} dataset layout(s) beyond "
              f"{args.clip_sigma:g} std below their cell mean are outside "
              "the plotted range")
    ax.set_xticks(range(len(camps)))
    ax.set_xticklabels([f"{c['cls']}\n$\\sigma$ = {c['sigma']:g}"
                        for c in camps], fontsize=10.5)

    # class colours appear in the swatches themselves, so the legend needs
    # no separate "blue = jitter" entry
    shown = [c for c in DISORDER_CLASSES if c in args.classes]
    dots = tuple(Line2D([], [], marker="o", ms=8, ls="none",
                        color=COLORS[c], mec="white") for c in shown)
    means = tuple([Line2D([], [], color="#2b2b2b", lw=3.4)]
                  + [Line2D([], [], color=COLORS[c], lw=3.4)
                     for c in shown])
    handles = [
        Line2D([], [], marker="o", ms=6, ls="none", color=GREY, alpha=0.6),
        dots,
        means,
        Patch(facecolor="#999999", alpha=0.30),
        Line2D([], [], color="#333333", ls=(0, (6, 3)), lw=1.3),
    ]
    labels = [
        "generated dataset: one dot per random layout",
        "inverse design: one dot per layout",
        "population mean (dark = dataset, coloured = design)",
        f"within {args.floor:.2f} % of the dataset mean (not claimable)",
        f"best of all {n_dis:,} layouts in the dataset (E = "
        f"{global_max:.3f})",
    ]
    if not args.global_line:
        handles, labels = handles[:-1], labels[:-1]
    ax.legend(handles, labels, loc="lower left", fontsize=9, ncol=2,
              framealpha=0.96, borderpad=0.6, columnspacing=1.4,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)})

    ax.set_ylabel("Broadband enhancement factor  E", fontsize=12)
    ax.set_title("Every FDTD-solved layout, by disorder cell: the generated "
                 "dataset vs surrogate-driven inverse design\n"
                 "(production numerics, 7x7 supercell, res 60)",
                 fontsize=12)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"\n[fig] -> {args.out}")


def make_figure_bars(stats, args):
    """Improvement over the random-disorder population, per campaign.

    Everything is a percentage of that cell's random-draw mean, so the bars
    start at a true zero (no truncated-axis distortion) and the whole
    comparison is on one consistent baseline:
      bar         = the average designed layout (mean of 20 +- SEM)
      diamond     = the best designed layout
      black tick  = the best of that cell's ~155 random draws
      grey zone   = below the 0.30 % claimability floor from the audit
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    w = 0.62

    # the sub-floor zone: bars have to climb out of it to mean anything
    ax.axhspan(0, args.floor, color="#bbbbbb", alpha=0.35, lw=0, zorder=0)
    ax.axhline(args.floor, color="#777777", ls="--", lw=1.3, zorder=1)

    for i, s in enumerate(stats):
        c = COLORS.get(s["class"], "#444444")
        ax.bar(i, s["mean_gain_pct"], width=w, color=c, alpha=0.85,
               edgecolor="white", lw=0.8, zorder=3,
               yerr=s["mean_gain_sem_pct"], capsize=3,
               error_kw={"lw": 1.1, "ecolor": "#333333", "zorder": 6})
        br = s["best_draw_gain_pct"]
        if br == br:                                    # not NaN
            ax.plot([i - w / 2, i + w / 2], [br, br], color="#111111",
                    lw=2.2, zorder=5, solid_capstyle="butt")
        # champion: a DIFFERENT SHAPE, not a different shade
        ax.plot([i], [s["champ_gain_pct"]], marker="D", ms=9, color="white",
                mec=c, mew=2.0, zorder=6)
        ax.annotate(f"{s['mean_gain_pct']:.2f}",
                    (i, s["mean_gain_pct"] / 2), ha="center", va="center",
                    fontsize=9.5, color="white", fontweight="bold",
                    zorder=7)

    for i in range(1, len(stats)):
        if stats[i]["class"] != stats[i - 1]["class"]:
            ax.axvline(i - 0.5, color="#cccccc", lw=1.0, zorder=0)

    ax.legend(handles=[
        Patch(facecolor="#6d6d6d", edgecolor="white",
              label="typical designed layout  (mean of 20 $\\pm$ SEM)"),
        Line2D([], [], color="white", marker="D", ms=9, mec="#6d6d6d",
               mew=2.0, ls="none", label="best designed layout"),
        Line2D([], [], color="#111111", lw=2.2,
               label="best of the cell's ~155 random layouts"),
        Patch(facecolor="#bbbbbb", alpha=0.35,
              label=f"below the {args.floor:.2f} % floor: not claimable"),
    ], loc="upper left", fontsize=10, framealpha=0.96, borderpad=0.6)

    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels([f"{s['class']}\n$\\sigma$ = {s['sigma']:g}"
                        for s in stats], fontsize=10.5)
    ax.set_xlim(-0.65, len(stats) - 0.35)
    ax.set_ylim(0, max(s["champ_gain_pct"] for s in stats) * 1.42)
    ax.set_ylabel("Enhancement gain over the random-disorder mean  (%)",
                  fontsize=11.5)
    ax.set_title("What surrogate-driven inverse design buys, per disorder "
                 "cell\n(every value FDTD-verified at production numerics: "
                 "7x7 supercell, res 60)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"\n[fig] -> {args.out}")


def make_figure(cells, E_ord, E_rand, global_max, n_dis, campaigns, args):
    """Direct-labelled, tightly-scaled: the comparison fills the frame.

    The ordered-lattice anchor sits ~8 % below everything else, so drawing
    it squashes the actual comparison into the top third of the axes.  By
    default it (and the fully-random mean) become a footnote and the y-axis
    is scaled to the bands + designed points; --full-range restores them.
    """
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    yvals = []

    for cls in (c for c in DISORDER_CLASSES if c in args.classes):
        c = COLORS[cls]

        # ---- population 1: the banked random draws ----------------------
        sig = sorted(s for cc, s in cells if cc == cls)
        if sig:
            mean = [cells[(cls, s)]["mean"] for s in sig]
            std = [cells[(cls, s)]["std"] for s in sig]
            lo = [m - s for m, s in zip(mean, std)]
            hi = [m + s for m, s in zip(mean, std)]
            ax.fill_between(sig, lo, hi, color=c, alpha=0.12, lw=0, zorder=0)
            ax.plot(sig, mean, "-", color=c, lw=2.0, marker="o", ms=4.0,
                    zorder=3, label=f"{cls}: random draws (n$\\approx$155/cell)")
            yvals += [min(lo), max(hi)]

        # ---- population 2: the verified inverse-design layouts ----------
        camp = sorted((c_ for c_ in campaigns if c_["cls"] == cls),
                      key=lambda c_: c_["sigma"])
        if not camp:
            continue
        dsig = [c_["sigma"] for c_ in camp]
        dmean = [st.mean(c_["E"]) for c_ in camp]
        dstd = [st.stdev(c_["E"]) if len(c_["E"]) > 1 else 0.0
                for c_ in camp]
        dlo = [m - s for m, s in zip(dmean, dstd)]
        dhi = [m + s for m, s in zip(dmean, dstd)]
        ax.fill_between(dsig, dlo, dhi, color=c, alpha=0.30, lw=0, zorder=4)
        ax.plot(dsig, dmean, "--", color=c, lw=2.6, marker="D", ms=7,
                mec="white", mew=1.2, zorder=6,
                label=f"{cls}: inverse design (n=20/cell)")
        yvals += [min(dlo), max(dhi)]

        # shade the improvement: banked mean -> designed mean
        if sig:
            base = [cells[(cls, s)]["mean"] if (cls, s) in cells else None
                    for s in dsig]
            if all(b is not None for b in base):
                ax.fill_between(dsig, base, dmean, color=c, alpha=0.10,
                                lw=0, hatch="///", edgecolor=c, zorder=1)

    # ---- the "beat everything" reference, labelled in place -------------
    if args.global_line:
        ax.axhline(global_max, color="#333333", ls=(0, (6, 3)), lw=1.4,
                   zorder=2)
        yvals.append(global_max)

    # ---- framing --------------------------------------------------------
    lo, hi = min(yvals), max(yvals)
    pad = 0.06 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 1.5 * pad)
    sig_all = [s for _, s in cells] + [c["sigma"] for c in campaigns]
    xhi = max(sig_all) * 1.06
    ax.set_xlim(-0.008 if not args.full_range else -0.014, xhi)
    ax.xaxis.set_major_formatter(ScalarFormatter())

    if args.global_line:
        # label at the LEFT end of the line: the designed curves all sit to
        # the right, so this is the only reliably empty spot on that level
        ax.annotate(f"best of {n_dis:,} banked layouts  (E = "
                    f"{global_max:.3f})", (ax.get_xlim()[0], global_max),
                    textcoords="offset points", xytext=(8, 5),
                    ha="left", va="bottom", fontsize=9, color="#333333",
                    zorder=8)

    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95,
              borderpad=0.6)
    ax.annotate("bands = $\\pm$1 std of each population; hatching = the "
                "improvement won by the search",
                (0.010, 0.012), xycoords="axes fraction", fontsize=8.5,
                color="#777777", va="bottom")

    ax.set_xlabel("Disorder strength  $\\sigma$", fontsize=12)
    ax.set_ylabel("Broadband enhancement factor  E", fontsize=12)
    ax.set_title("Surrogate-driven inverse design vs the random-disorder "
                 "population it searched\n(every value FDTD-verified at "
                 "production numerics: 7x7 supercell, res 60)", fontsize=12)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"\n[fig] -> {args.out}")


# ==========================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m models.plot_inverse_results",
        description="Plot FDTD-verified inverse-design results over the "
                    "dataset's E-vs-sigma curve.")
    ap.add_argument("dirs", nargs="*", default=[],
                    help="Extra campaign directories (auto-discovery "
                         "already covers runs/inverse/ and the solver's "
                         "candidates/ folder).")
    ap.add_argument("--no-discover", dest="discover", action="store_false",
                    default=True,
                    help="plot ONLY the directories given as arguments, "
                         "skipping auto-discovery of runs/inverse/ -- use "
                         "this to plot one model generation on its own "
                         "(e.g. runs/inverse_v2/*), since merging both "
                         "would draw each cell twice.")
    ap.add_argument("--labels",
                    default=os.path.join(_BANK, "labels.csv"),
                    help="Bank labels.csv (default: the production bank).")
    ap.add_argument("--out",
                    default=os.path.join(_BANK, "figs",
                                         "fig_inverse_vs_dataset.png"))
    ap.add_argument("--classes", nargs="+", default=list(DISORDER_CLASSES),
                    choices=list(DISORDER_CLASSES))
    ap.add_argument("--no-global-line", dest="global_line",
                    action="store_false", default=True,
                    help="Omit the best-banked-layout reference line.")
    ap.add_argument("--full-range", action="store_true", default=False,
                    help="curves style only: include the ordered-lattice "
                         "anchor and the fully-random line as drawn lines "
                         "(widens the y-axis ~4x).")
    ap.add_argument("--style", choices=("scatter", "bars", "curves"),
                    default="scatter",
                    help="scatter: every layout as a dot in real E, per "
                         "cell (default). bars: gain over the random-draw "
                         "mean. curves: E vs sigma as mean + spread bands.")
    ap.add_argument("--clip-sigma", type=float, default=3.0,
                    help="scatter style: hide dataset points further than "
                         "this many std below their cell mean so a lone "
                         "outlier does not stretch the axis (0 = show all).")
    ap.add_argument("--floor", type=float, default=0.30,
                    help="Claimability floor in %% (audit Test 9).")
    ap.add_argument("--dpi", type=int, default=150)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.exists(args.labels):
        raise SystemExit(f"labels.csv not found: {args.labels} -- run "
                         "run_dataset.py analyze first.")
    cells, E_ord, E_rand, n_dis, global_max = load_bank(args.labels)
    print(f"[bank] {n_dis} disordered samples, {len(cells)} cells, "
          f"ordered anchor E = {E_ord:.4f}, best banked E = {global_max:.4f}")

    campaigns, skipped = discover_campaigns(args.dirs, discover=args.discover)
    for d in skipped:
        print(f"[skip] {os.path.relpath(d, _REPO_ROOT)} -- exported but not "
              "verified (no verification.csv)")
    if not campaigns:
        raise SystemExit("no verified inverse-design campaigns found -- run "
                         "verify_candidates.py on an export first.")
    for c in campaigns:
        print(f"[found] {c['name']:22s} {c['cls']:6s} sigma={c['sigma']:<6g} "
              f"{len(c['E'])} verified candidates")

    dup = {}
    for c in campaigns:
        dup.setdefault((c["cls"], c["sigma"]), []).append(c["name"])
    for k, names in dup.items():
        if len(names) > 1:
            print(f"[warn] {len(names)} campaigns at {k[0]} sigma={k[1]:g} "
                  f"({', '.join(names)}) -- plotted with an x-offset")

    stats = [campaign_stats(c, global_max) for c in campaigns
             if c["cls"] in args.classes]
    print_table(stats, global_max, n_dis)

    csv_path = os.path.splitext(args.out)[0] + ".csv"
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader()
        w.writerows(stats)
    print(f"[csv] -> {csv_path}")

    if args.style == "scatter":
        make_figure_scatter(cells, campaigns, global_max, n_dis, args)
    elif args.style == "bars":
        make_figure_bars(stats, args)
    else:
        make_figure(cells, E_ord, E_rand, global_max, n_dis, campaigns, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
