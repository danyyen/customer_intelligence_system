"""Shared chart styling used across every visualization in this project."""

from __future__ import annotations

import matplotlib.pyplot as plt

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"      # primary series
ORANGE = "#eb6834"    # highlight / flagged outliers

# Fixed categorical assignment for the five final customer segments --
# consistent color per segment across every chart, not re-cycled per plot.
SEGMENT_PALETTE = {
    "Recent developing": "#54A24B",
    "Champions": "#F58518",
    "Lapsed low-value": "#4C78A8",
    "At-risk established": "#B279A2",
    "Exceptional high-value": "#E45756",
}


def apply_style() -> None:
    """Set the shared matplotlib rcParams used across every chart in this
    project. Call once, near the top of a notebook, before plotting."""
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def style_ax(ax, axis: str = "y"):
    """Apply the shared recessive-gridline styling to a single matplotlib Axes."""
    ax.grid(axis=axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    return ax
