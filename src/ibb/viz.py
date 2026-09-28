"""Shared plotting style and matrix heatmaps.

The diverging colormap here is the one used for every figure in the paper, so
panels stay comparable; it is deliberately quantised (``N=8``) to make
correlation bands readable rather than smooth.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform

from .stats import cov_to_corr

#: Blue (negative) - beige (zero) - red (positive).
DIVERGING_COLORS = [
    "#2A5CA0",  # deep blue    (strong negative)
    "#5E83BA",  # medium blue
    "#9DBBD8",  # light blue
    "#D6E3F0",  # very light blue
    "#F5F5DC",  # beige        (neutral)
    "#F0D8C9",  # light salmon
    "#E8A798",  # medium salmon
    "#D96C5E",  # coral
    "#C83B3B",  # red          (strong positive)
]

#: The project-wide correlation colormap.
IBB_CMAP = LinearSegmentedColormap.from_list(
    "custom_diverging", DIVERGING_COLORS, N=8
)

#: Yeo-7 network abbreviations, in atlas label order 1..7.
YEO7_LABELS = ["VIS", "SMN", "DAN", "VAN", "LIM", "FPN", "DMN"]

#: Yeo-7 node colours for connectome plots.
YEO7_COLORS = [
    "#7b3294",  # VIS
    "#1f78b4",  # SMN
    "#33a02c",  # DAN
    "#66c2a5",  # VAN
    "#ffd92f",  # LIM
    "#fc8d62",  # FPN
    "#b2182b",  # DMN
]


def matrix_heatmap(
    M,
    vmin=-0.1,
    vmax=0.1,
    figsize=(5, 4),
    labels=None,
    annot=False,
    title=None,
    output_file=None,
    dpi=300,
):
    """Heatmap of a (d, d) matrix in the project style.

    With ``labels`` given (e.g. :data:`YEO7_LABELS`) ticks are shown and the
    plot is squared — the layout used for the 7x7 network-block figures.
    """
    plt.figure(figsize=figsize)

    kwargs = dict(cmap=IBB_CMAP, center=0, vmin=vmin, vmax=vmax, annot=annot)
    if labels is None:
        kwargs.update(xticklabels=False, yticklabels=False)
    else:
        kwargs.update(
            xticklabels=labels,
            yticklabels=labels,
            square=True,
            linecolor="white",
            linewidths=0.1,
        )

    ax = sns.heatmap(M, **kwargs)

    if labels is not None:
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    return ax


def clustered_corr_heatmap(
    Sigma,
    title=None,
    vmin=-0.25,
    vmax=0.25,
    method="average",
    output_file=None,
    dpi=300,
):
    """Reorder a covariance matrix by hierarchical clustering, then plot it.

    Rows/columns are ordered by average linkage on ``1 - |corr|``, which groups
    strongly coupled ROIs (of either sign) together and makes block structure
    visible without imposing an atlas.

    Returns
    -------
    ndarray
        The leaf ordering, so the same permutation can be applied to other
        matrices for a like-for-like comparison.
    """
    R = cov_to_corr(Sigma)

    dist = 1 - np.abs(R)
    Z = linkage(squareform(dist, checks=False), method=method)
    order = leaves_list(Z)

    R_clustered = R[order][:, order]

    sns.heatmap(
        R_clustered,
        cmap="coolwarm",
        center=0,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
    )

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")

    return order


def corr_dendrogram(Sigma, labels=None, method="average", figsize=(12, 5)):
    """Dendrogram of the same linkage used by :func:`clustered_corr_heatmap`."""
    R = cov_to_corr(Sigma)
    # sqrt(2(1-r)) is the Euclidean distance between standardised variables.
    dist = np.sqrt(2 * (1 - R))
    np.fill_diagonal(dist, 0)

    Z = linkage(squareform(dist, checks=False), method=method)

    plt.figure(figsize=figsize)
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Variables")
    plt.ylabel("Distance")
    plt.tight_layout()
    return Z
