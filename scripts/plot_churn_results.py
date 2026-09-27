"""Plot the reported September holdout metrics for the README.

Run with: python scripts/plot_churn_results.py
These are the rounded summary values in README.md, not synthetic curves.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.30, top=0.72, wspace=0.65)
    fig.suptitle("Churn prediction | Holdout results", fontsize=21, fontweight="bold", y=0.96)
    fig.text(0.5, 0.86, "Logistic regression • Untouched September test • 90-day inactivity target",
             ha="center", color="#555555", fontsize=12)

    panels = [
        (["PR-AUC", "ROC-AUC"], [0.647, 0.759], ["0.647", "0.759"],
         "Ranking quality", "Score (0–1)", 1, "#6085ad"),
        (["Precision", "Recall"], [0.709, 0.361], ["70.9%", "36.1%"],
         "At 20% campaign capacity", "Share of respective group", 1, "#67a65c"),
        (["Lift"], [1.80], ["1.80×"],
         "Concentration of churn", "Multiple of population churn rate", 2.2, "#b486a7"),
    ]
    for ax, (labels, values, annotations, title, xlabel, xmax, color) in zip(axes, panels):
        positions = list(range(len(labels)))
        ax.barh(positions, values, height=0.42, color=color, zorder=3)
        ax.set_yticks(positions, labels)
        ax.set_ylim(1.55, -0.65)
        ax.set_xlim(0, xmax)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=18)
        ax.set_xlabel(xlabel, fontsize=10, labelpad=10)
        ax.grid(axis="x", color="#e4e4e4", zorder=0)
        ax.tick_params(axis="both", length=0, pad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for y, value, annotation in zip(positions, values, annotations):
            ax.text(value + xmax * 0.035, y, annotation, va="center", fontweight="bold")

    axes[0].set_xticks([0, 0.5, 1])
    axes[1].set_xticks([0, 0.5, 1])
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[2].set_xticks([0, 1, 2], ["0×", "1×", "2×"])
    axes[2].axvline(1, color="#777777", linestyle="--", linewidth=1.2, zorder=4)
    axes[2].text(1, 0.53, "Population baseline", ha="center", fontsize=9, color="#555555")
    fig.text(0.5, 0.14,
             "Contacting the highest-risk fifth: about 71 in 100 selected customers churned;",
             ha="center", fontsize=12)
    fig.text(0.5, 0.085,
             "the queue captured 36.1% of all churners, at 1.80× the population churn rate.",
             ha="center", fontsize=12)
    destination = ROOT / "images" / "churn_holdout_results.png"
    destination.parent.mkdir(exist_ok=True)
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Exported {destination}")


if __name__ == "__main__":
    main()
