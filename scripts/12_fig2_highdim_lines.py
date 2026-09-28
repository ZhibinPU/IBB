#!/usr/bin/env python3
"""Figure 2: ultra-high-dimensional estimation error and cost, as lines with CI bands.

Replaces the original 3x2 grid of boxplots (fig/new_approximation_error.pdf) and
folds in the former standalone runtime figure (fig/runtime_ratio_plot.pdf).  The
three covariance structures are combined into three panels:

    a  rF-norm  = ||Sigma* - Sigma_hat||_F / d      (all structures, one y-axis)
    b  spectral = ||Sigma* - Sigma_hat||_2          (Block on the left axis,
                                                     Banded/Toeplitz on the right)
    c  runtime / runtime(IBB, Block, n=200)         (log y-axis)

Panel b needs the second axis because the Block spectral errors (~165-190) and
the Banded/Toeplitz ones (~29-36) differ by a factor of five; on a shared axis
the latter two collapse onto each other.

Panel c is on a log axis because the runtime ratios are multiplicative and span
about two decades (0.11 to 7.8 over the nine cells, ~74x).  The three method
groups are much closer together than the cost claim in the text suggests -- IBB
is about twice as fast as BAG and only 1.1-4.6x slower than POETRY -- and a log
axis makes a factor of two look like nothing, so those two contrasts are stated
in the caption instead; runtime_contrasts() computes them from the plotted
arrays and report_runtime() prints them, so the caption can be checked against a
re-run.  This is the opposite of panel a, where the overlapping curves differ by
a *ratio* of 1.00 and log would do nothing.

Panels a and b read the error arrays under results/highdim; panel c reads the
runtimes under results/runtime_data, which is the raw timing run (the *_time*.npy
files in results/highdim are a linear remapping of it and are not used).

Encoding is composite so neither channel carries identity alone, and it is the
same in all three panels:
    colour     -> method     (IBB, BAG, POETRY)
    line/marker-> structure  (Block, Banded, Toeplitz)

Bands are mean +/- 1 standard error over replications.  SCM is omitted, as in
the original figure: its estimates are not positive definite in this regime.

Usage
-----
    python scripts/12_fig2_highdim_lines.py
    python scripts/12_fig2_highdim_lines.py --results-dir path/to/highdim \
                                            --runtime-dir path/to/runtime_data
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- what to plot -----------------------------------------------------------

N_VALUES = (200, 300, 400)
D_OF_N = {n: 15 * n for n in N_VALUES}  # the d = 15 x n scenario

# file stem -> label, in legend order
METHODS = (("ibb", "IBB"), ("bag", "BAG"), ("pot", "POETRY"))
STRUCTURES = (("block", "Block"), ("band", "Banded"), ("toeplitz", "Toeplitz"))

# Panel c reports runtime relative to one cell, as the original runtime figure
# did, so the axis is unitless and the machine's absolute speed drops out.
RUNTIME_BASELINE = ("block", "ibb", 200)

# Two runtime layouts are in the tree.  results/runtime_data is flat and names
# methods in upper case ("block_IBB_n200.npy"); the older results/highdim keeps
# per-structure subfolders ("block/ibb_time200.npy").  Both are supported so a
# re-run is a flag change, not an edit.
#
# POETRY appears twice in results/runtime_data, as POET and POT.  POET is the
# self-consistent set (monotone in n, same order of magnitude across the three
# structures); POT is a mix -- byte-identical to the old pot_time for Banded,
# equal to POET for Toeplitz, different again for Block -- so POET is used.
RUNTIME_STEM = {"ibb": "IBB", "bag": "BAG", "pot": "POET"}


# --- style ------------------------------------------------------------------

# Colour = method.  IBB carries the one accent hue; the two baselines recede
# into neutrals, so the eye lands on IBB first.
#
# The cost of an accent-plus-neutrals scheme is that under protanopia the coral
# desaturates towards mid grey, which puts it close to the lighter of the two
# greys (protan dE 7.5 -- inside the 6-8 floor band, below the dE >= 8 target;
# normal-vision dE 15.9, just over the hard floor of 15).  That is why POETRY
# additionally gets hollow markers: marker *shape* already encodes the
# covariance structure, but marker *fill* was free, so identity never rests on
# colour alone.  #949494 is as light as the grey can go while still clearing
# ~3:1 contrast against white in print.
METHOD_COLOR = {"ibb": "#EA6B66", "bag": "#3D3D3D", "pot": "#949494"}
METHOD_HOLLOW = {"ibb": False, "bag": False, "pot": True}

# Line style + marker = covariance structure.
STRUCT_STYLE = {
    "block": dict(linestyle="-", marker="o"),
    "band": dict(linestyle="--", marker="s"),
    "toeplitz": dict(linestyle=":", marker="^"),
}

# Horizontal offset per structure.  In panel a the Banded and Toeplitz rF-norm
# curves coincide to within 0.01-1.2% (indistinguishable from each other for
# IBB and BAG: |diff| <= 1.5 pooled SE), so they overplot.  A log y-axis cannot
# fix this -- log separates series by their ratio, and that ratio is 1.00 here.
# Offsetting along x instead costs no accuracy, because x is the discrete set
# n in {200, 300, 400} rather than a continuous coordinate, and it leaves the
# vertical positions honest instead of manufacturing a gap that is not there.
STRUCT_DODGE = {"block": -0.10, "band": 0.0, "toeplitz": 0.10}

INK = "#1A1A1A"
MUTED = "#6B6B6B"
GRID = "#DCDCDC"

# Nature: 183 mm double-column width, figure text at 5-7 pt.
FIG_WIDTH_IN = 183 / 25.4
FIG_HEIGHT_IN = 2.95


def set_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,   # editable text in the PDF, not outlines
        "ps.fonttype": 42,
        # No tight bbox: the canvas must come out at exactly the Nature
        # double-column width, and tight cropping fights the manual margins
        # reserved for the two figure-level legends.
        "savefig.bbox": None,
    })


# --- data -------------------------------------------------------------------

def load_cell(results_dir, structure, method, n, metric):
    """Replication-level errors for one (structure, method, n, metric) cell.

    ``*_errors{n}.npy`` holds the raw Frobenius norm; ``n2*_errors{n}.npy`` the
    spectral norm.  Returns ``None`` when the run is missing.
    """
    prefix = "" if metric == "rf" else "n2"
    path = Path(results_dir) / structure / f"{prefix}{method}_errors{n}.npy"
    if not path.exists():
        return None
    vals = np.load(path).astype(float).ravel()
    if metric == "rf":
        vals = vals / D_OF_N[n]  # rF-norm normalises by the matrix dimension
    return vals


def summarise(results_dir, metric):
    """{(structure, method): (mean, sem, reps)} arrays aligned with N_VALUES."""
    out = {}
    for structure, _ in STRUCTURES:
        for method, _ in METHODS:
            mean, sem, reps = [], [], []
            for n in N_VALUES:
                vals = load_cell(results_dir, structure, method, n, metric)
                if vals is None or vals.size == 0:
                    mean.append(np.nan)
                    sem.append(np.nan)
                    reps.append(0)
                    continue
                mean.append(vals.mean())
                # ddof=1 SE; a single replication has no spread to report.
                sem.append(vals.std(ddof=1) / np.sqrt(vals.size)
                           if vals.size > 1 else np.nan)
                reps.append(vals.size)
            out[(structure, method)] = (np.array(mean), np.array(sem),
                                        np.array(reps))
    return out


def load_times(runtime_dir, structure, method, n):
    """Per-replication wall-clock seconds, or ``None`` when the run is missing.

    Accepts either runtime layout (see :data:`RUNTIME_STEM`).  Timing and error
    arrays can differ in length within the same cell, so runtimes are
    summarised independently rather than paired with the error replications.
    """
    d = Path(runtime_dir)
    flat = d / f"{structure}_{RUNTIME_STEM[method]}_n{n}.npy"
    nested = d / structure / f"{method}_time{n}.npy"
    path = flat if flat.exists() else nested
    if not path.exists():
        return None
    return np.load(path).astype(float).ravel()


def check_times(runtime_dir):
    """Flag wall-clock values that cannot be measurements.

    A runtime is positive by construction, so a non-positive entry means the
    file has been edited or mis-assembled.  Such a cell is still plotted --
    hiding it would make a broken input look like a clean one -- but it is
    named here and in the caption note.
    """
    problems = []
    for structure, slab in STRUCTURES:
        for method, mlab in METHODS:
            for n in N_VALUES:
                vals = load_times(runtime_dir, structure, method, n)
                if vals is None or vals.size == 0:
                    continue
                if (vals <= 0).any():
                    problems.append(
                        f"{slab}/{mlab}/n={n}: contains a non-positive "
                        f"runtime {np.round(vals, 2).tolist()}")
    if problems:
        print("\n!! impossible runtime values -- panel c is not trustworthy "
              "until these are fixed:")
        for p in problems:
            print(f"   {p}")
    return problems


def summarise_runtime(runtime_dir):
    """Runtime relative to :data:`RUNTIME_BASELINE`, as (mean, sem, reps).

    The ratio is formed against the *mean* baseline runtime, so the baseline
    cell sits at exactly 1 and its own spread is folded into the other cells
    only through that constant -- the same normalisation the original runtime
    figure used.
    """
    b_struct, b_method, b_n = RUNTIME_BASELINE
    base = load_times(runtime_dir, b_struct, b_method, b_n)
    if base is None or base.size == 0:
        raise SystemExit(f"missing runtime baseline {RUNTIME_BASELINE} "
                         f"under {runtime_dir}")
    scale = base.mean()

    out = {}
    for structure, _ in STRUCTURES:
        for method, _ in METHODS:
            mean, sem, reps = [], [], []
            for n in N_VALUES:
                vals = load_times(runtime_dir, structure, method, n)
                if vals is None or vals.size == 0:
                    mean.append(np.nan)
                    sem.append(np.nan)
                    reps.append(0)
                    continue
                vals = vals / scale
                mean.append(vals.mean())
                sem.append(vals.std(ddof=1) / np.sqrt(vals.size)
                           if vals.size > 1 else np.nan)
                reps.append(vals.size)
            out[(structure, method)] = (np.array(mean), np.array(sem),
                                        np.array(reps))
    return out


def runtime_contrasts(runtime_dir, exclude_nonpositive=False):
    """The two numbers panel c annotates, computed rather than hard-coded.

    Returns ``(ibb_vs_bag, ibb_vs_pot)``, each a ``(lo, median, hi)`` triple
    over the nine cells.  IBB and BAG are paired replicate-by-replicate (both
    see ``random_state=rep``); POETRY is compared on cell means because its
    replication count does not always match.

    With ``exclude_nonpositive`` the cells flagged by :func:`check_times` are
    dropped, which is how much a single impossible cell moves the summary.
    Panel c annotates the *unfiltered* figures, so nothing is hidden.
    """
    def triple(vals):
        v = np.asarray([x for x in vals if np.isfinite(x)])
        return float(v.min()), float(np.median(v)), float(v.max())

    def usable(*arrays):
        return not (exclude_nonpositive
                    and any(a is not None and (a <= 0).any() for a in arrays))

    vs_bag, vs_pot = [], []
    for structure, _ in STRUCTURES:
        for n in N_VALUES:
            ibb = load_times(runtime_dir, structure, "ibb", n)
            bag = load_times(runtime_dir, structure, "bag", n)
            pot = load_times(runtime_dir, structure, "pot", n)
            if ibb is None:
                continue
            if bag is not None and usable(ibb, bag):
                k = min(ibb.size, bag.size)
                vs_bag.append(float(np.mean(ibb[:k] / bag[:k])))
            if pot is not None and pot.mean() > 0 and usable(ibb, pot):
                vs_pot.append(float(ibb.mean() / pot.mean()))
    return triple(vs_bag), triple(vs_pot)


# --- drawing ----------------------------------------------------------------

def marker_kw(method):
    """Filled markers for IBB and BAG, hollow for POETRY (see METHOD_HOLLOW)."""
    color = METHOD_COLOR[method]
    if METHOD_HOLLOW[method]:
        return dict(markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=0.9)
    return dict(markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.6)


def draw_series(ax, x, mean, sem, structure, method, zorder=3):
    color = METHOD_COLOR[method]
    style = STRUCT_STYLE[structure]
    xs = x + STRUCT_DODGE[structure]

    lo, hi = mean - sem, mean + sem
    if ax.get_yscale() == "log":
        # A log axis cannot show a band edge at or below zero (panel c).  Clip
        # to a tenth of the mean rather than dropping the band, so a cell whose
        # SE is as large as its mean still reads as highly variable.
        lo = np.maximum(lo, 0.1 * mean)
    ok = ~np.isnan(sem)
    if ok.any():
        # All bands share one low z, so no band can ever hide a mean line.
        ax.fill_between(xs[ok], lo[ok], hi[ok], color=color, alpha=0.15,
                        linewidth=0, zorder=2)
    ax.plot(xs, mean, color=color, linewidth=1.3, markersize=4.0,
            zorder=zorder, **marker_kw(method), **style)


def style_axes(ax, x, right_pad=0.28):
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}" for n in N_VALUES])
    ax.set_xlabel(r"$n$   ($d = 15\,n$)")
    # Padding clears the structure dodge on both sides.
    ax.set_xlim(x[0] - 0.30, x[-1] + right_pad)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def group_extent(summary, structures, pad=0.06):
    """Padded (lo, hi) covering the CI bands of every series in ``structures``."""
    lo, hi = np.inf, -np.inf
    for structure in structures:
        for method, _ in METHODS:
            mean, sem, _ = summary[(structure, method)]
            half = np.nan_to_num(sem)
            lo = min(lo, np.nanmin(mean - half))
            hi = max(hi, np.nanmax(mean + half))
    span = hi - lo
    return lo - pad * span, hi + pad * span


def place_band(ax, lo, hi, frac_lo, frac_hi, n_ticks=5):
    """Scale ``ax`` so that [lo, hi] lands in [frac_lo, frac_hi] of its height.

    This is what keeps the two y-scales of panel b from interleaving: each
    group of curves gets its own horizontal band of the panel, so a reader
    never has to guess which axis a line is measured against.  Ticks are drawn
    only over the occupied band, leaving the rest of the axis blank.
    """
    span = (hi - lo) / (frac_hi - frac_lo)
    bottom = lo - frac_lo * span
    ax.set_ylim(bottom, bottom + span)
    ticks = mpl.ticker.MaxNLocator(n_ticks, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    ax.set_yticks([t for t in ticks if lo <= t <= hi])


def make_figure(results_dir, runtime_dir, out_paths):
    rf = summarise(results_dir, "rf")
    sp = summarise(results_dir, "spec")
    rt = summarise_runtime(runtime_dir)
    vs_bag, vs_pot = runtime_contrasts(runtime_dir)

    x = np.arange(len(N_VALUES), dtype=float)
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    # Draw the greys first so the rose IBB curve sits on top of the pile.
    draw_order = list(reversed(METHODS))

    # -- panel a: rF-norm, all nine series on one axis -----------------------
    for structure, _ in STRUCTURES:
        for z, (method, _) in enumerate(draw_order):
            mean, sem, _ = rf[(structure, method)]
            draw_series(ax_a, x, mean, sem, structure, method, zorder=3 + 2 * z)
    style_axes(ax_a, x)
    ax_a.set_ylabel("rF-norm\n"
                    r"$\|\Sigma^{*}-\widehat{\Sigma}\|_{F}\,/\,d$")

    # -- panel b: spectral norm, Block left / Banded+Toeplitz right ----------
    # The Block spectral errors (~165-190) and the Banded/Toeplitz ones
    # (~29-36) differ by a factor of five, so they get separate scales.  Each
    # scale is then confined to its own horizontal band of the panel.
    ax_b2 = ax_b.twinx()
    for z, (method, _) in enumerate(draw_order):
        mean, sem, _ = sp[("block", method)]
        draw_series(ax_b, x, mean, sem, "block", method, zorder=3 + 2 * z)
    for structure in ("band", "toeplitz"):
        for z, (method, _) in enumerate(draw_order):
            mean, sem, _ = sp[(structure, method)]
            draw_series(ax_b2, x, mean, sem, structure, method,
                        zorder=3 + 2 * z)

    style_axes(ax_b, x, right_pad=0.92)
    ax_b2.set_xlim(ax_b.get_xlim())
    place_band(ax_b, *group_extent(sp, ["block"]), frac_lo=0.66, frac_hi=1.0,
               n_ticks=5)
    place_band(ax_b2, *group_extent(sp, ["band", "toeplitz"]),
               frac_lo=0.0, frac_hi=0.56, n_ticks=5)

    ax_b.spines["right"].set_visible(True)
    ax_b.set_ylabel("Spectral norm\n"
                    r"$\|\Sigma^{*}-\widehat{\Sigma}\|_{2}$")
    # No label on the twin axis: the in-panel group labels below already say
    # which curves are read against it, and a second rotated label here would
    # collide with panel c.
    ax_b2.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax_b2.set_axisbelow(True)
    ax_b2.spines["top"].set_visible(False)
    ax_b2.spines["left"].set_visible(False)
    ax_b2.tick_params(axis="y", colors=MUTED)

    # A dashed rule marks where the left scale hands over to the right one,
    # and each group is named directly beside its own curves.
    ax_b.axhline(ax_b.get_ylim()[0] + 0.61 * np.diff(ax_b.get_ylim())[0],
                 color=GRID, linewidth=0.7, linestyle=(0, (4, 4)), zorder=1)
    for axis, structure, label in ((ax_b, "block", "Block\n(left)"),
                                   (ax_b2, "band", "Banded\n(right)"),
                                   (ax_b2, "toeplitz", "Toeplitz\n(right)")):
        top = max(np.nanmax(sp[(structure, m)][0]) for m, _ in METHODS)
        bot = min(np.nanmin(sp[(structure, m)][0]) for m, _ in METHODS)
        axis.text(x[-1] + 0.26, 0.5 * (top + bot), label, fontsize=6.0,
                  color=MUTED, ha="left", va="center", linespacing=1.3)

    # -- panel c: runtime relative to IBB / Block / n = 200 ------------------
    # Log scale: POETRY runs 200-460x faster than IBB and BAG, so the ratios
    # span ~1800x.  On a linear axis the six IBB/BAG curves would pile onto the
    # top edge and the three POETRY curves onto the bottom.
    ax_c.set_yscale("log")
    for structure, _ in STRUCTURES:
        for z, (method, _) in enumerate(draw_order):
            mean, sem, _ = rt[(structure, method)]
            draw_series(ax_c, x, mean, sem, structure, method, zorder=3 + 2 * z)
    style_axes(ax_c, x)
    ax_c.set_ylabel("Runtime ratio")
    ax_c.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
    ax_c.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_c.yaxis.set_minor_locator(
        mpl.ticker.LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1)))
    ax_c.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    # The reference cell itself: a faint rule at 1, defined in the caption.
    ax_c.axhline(1.0, color=MUTED, linewidth=0.6, linestyle=(0, (1, 2)),
                 zorder=1)
    # The contrasts that the log axis compresses (IBB vs BAG in particular) are
    # stated in the caption rather than in the panel; vs_bag/vs_pot are still
    # computed from the plotted arrays and reported to the console, so the
    # caption numbers can be checked against a re-run without editing anything.
    top = np.nanmax([np.nanmax(rt[(s, m)][0]) for s, _ in STRUCTURES
                     for m, _ in METHODS])
    bot = np.nanmin([np.nanmin(rt[(s, m)][0]) for s, _ in STRUCTURES
                     for m, _ in METHODS])
    # Equal margins above and below now that the corner no longer holds text.
    ax_c.set_ylim(bot / 1.6, top * 1.6)

    # -- layout --------------------------------------------------------------
    # Manual margins rather than tight_layout: the two figure-level legends
    # need reserved space at the bottom, and each gap between panels has to
    # hold one axis label plus the neighbouring panel's tick labels.
    fig.subplots_adjust(left=0.090, right=0.978, bottom=0.255, top=0.905,
                        wspace=0.46)

    # -- panel letters -------------------------------------------------------
    # Placed in figure coordinates off each axes' box, so the three panels get
    # identical offsets even though panel b carries a wider label stack.
    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):
        box = ax.get_position()
        fig.text(box.x0 - 0.052, box.y1 + 0.035, letter, fontsize=8,
                 fontweight="bold", va="top", ha="left")

    # -- two legends: one per encoding channel -------------------------------
    # Marker fill repeats the method identity, so the legend shows it too.
    method_handles = [
        Line2D([], [], color=METHOD_COLOR[m], linewidth=1.3, marker="o",
               markersize=4.0, label=lab, **marker_kw(m))
        for m, lab in METHODS
    ]
    struct_handles = [
        Line2D([], [], color=MUTED, linewidth=1.3, markersize=4.0,
               markerfacecolor=MUTED, markeredgecolor="white",
               markeredgewidth=0.6, label=lab, **STRUCT_STYLE[s])
        for s, lab in STRUCTURES
    ]

    leg1 = fig.legend(handles=method_handles, title="Method", ncol=3,
                      loc="lower center", bbox_to_anchor=(0.30, 0.015),
                      frameon=False, handlelength=2.0, columnspacing=1.4,
                      handletextpad=0.5)
    leg2 = fig.legend(handles=struct_handles, title="Covariance structure",
                      ncol=3, loc="lower center", bbox_to_anchor=(0.74, 0.015),
                      frameon=False, handlelength=2.4, columnspacing=1.4,
                      handletextpad=0.5)
    for leg in (leg1, leg2):
        leg.get_title().set_fontsize(7)
        leg.get_title().set_color(INK)

    for path in out_paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=400)
        print(f"wrote {path}")
    plt.close(fig)

    return rf, sp, rt, vs_bag, vs_pot


def report_reps(rf, rt):
    """Print the replication count behind every cell, so the caption can't drift.

    Error and runtime counts are listed separately because they disagree in a
    few cells (e.g. Block/BAG at n=300 has 10 errors but 5 timings).
    """
    for title, summary in (("rF-norm files", rf), ("runtime files", rt)):
        print(f"\nreplications per cell ({title}):")
        header = "  {:9s}".format("") + "".join(f"{n:>8d}" for n in N_VALUES)
        for structure, slab in STRUCTURES:
            print(f"\n  {slab}")
            print(header)
            for method, mlab in METHODS:
                reps = summary[(structure, method)][2]
                print("  {:9s}".format(mlab)
                      + "".join(f"{r:>8d}" for r in reps))


def report_runtime(rt, vs_bag, vs_pot, runtime_dir):
    """Print the runtime ratios drawn in panel c, and the annotated contrasts."""
    b_struct, b_method, b_n = RUNTIME_BASELINE
    print(f"\nruntime ratio (baseline: {b_method.upper()}, {b_struct}, "
          f"n={b_n}):")
    header = "  {:9s}".format("") + "".join(f"{n:>8d}" for n in N_VALUES)
    for structure, slab in STRUCTURES:
        print(f"\n  {slab}")
        print(header)
        for method, mlab in METHODS:
            mean = rt[(structure, method)][0]
            print("  {:9s}".format(mlab)
                  + "".join(f"{v:>8.3f}" for v in mean))

    clean_bag, clean_pot = runtime_contrasts(runtime_dir,
                                             exclude_nonpositive=True)
    print(f"\n  {'':22s}{'min':>8}{'median':>9}{'max':>8}")
    for lab, tri in (("IBB / BAG   (paired)", vs_bag),
                     ("IBB / POETRY", vs_pot)):
        print(f"  {lab:22s}{tri[0]:>8.2f}{tri[1]:>9.2f}{tri[2]:>8.2f}")
    print("  -- same, dropping cells with an impossible runtime --")
    for lab, tri in (("IBB / BAG   (paired)", clean_bag),
                     ("IBB / POETRY", clean_pot)):
        print(f"  {lab:22s}{tri[0]:>8.2f}{tri[1]:>9.2f}{tri[2]:>8.2f}")


def main():
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(repo / "results" / "highdim"),
                    help="directory holding block/, band/, toeplitz/ subfolders")
    ap.add_argument("--runtime-dir",
                    default=str(repo / "results" / "runtime_data"),
                    help="runtime directory; either layout in RUNTIME_STEM")
    ap.add_argument("--out", nargs="+",
                    default=[str(repo.parent / "paper" / "fig"
                                 / "fig2_highdim_lines.pdf"),
                             str(repo / "results" / "figures"
                                 / "fig2_highdim_lines.png")],
                    help="output paths (extension picks the format)")
    args = ap.parse_args()

    set_style()
    check_times(args.runtime_dir)
    rf, _, rt, vs_bag, vs_pot = make_figure(args.results_dir, args.runtime_dir,
                                            args.out)
    report_reps(rf, rt)
    report_runtime(rt, vs_bag, vs_pot, args.runtime_dir)


if __name__ == "__main__":
    main()
