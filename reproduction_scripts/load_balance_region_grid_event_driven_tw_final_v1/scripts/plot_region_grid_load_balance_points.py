"""Plot region-grid load-balance points in the context-noise figure style."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-yifu")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs/research/rmab_vm_outputs/load_balance_region_grid_event_driven_tw_final_v1"
INPUT = OUT_DIR / "all_strategy_load_balance_appendix_table.csv"

METHODS = [
    ("State Thompson", "State\nThompson", "#6b7280", "--", "s"),
    ("TW", "TW", "#0b1f66", "--", "^"),
    ("Local UCB+TW", "Local\nUCB+TW", "#1d4ed8", "--", "s"),
    ("Global UCB+TW", "Global\nUCB+TW", "#0284c7", "--", "s"),
    ("EXP4", "EXP4", "#14b8a6", "--", "s"),
    ("TM--TW", "Original\nTM--TW", "#7c3aed", "--", "^"),
    ("Adp. TM--TW", "Adaptive\ntrust", "#f97316", "-", "o"),
    ("Adp. TM--TW + beta gate prior", "Adaptive\n+beta gate", "#dc2626", "-.", "D"),
    (
        "Adp. TM--TW + beta gate + low-rank",
        "Adaptive\n+beta gate+LR",
        "#b45309",
        "-.",
        "D",
    ),
    ("Adp. TM--TW + support/offline prior", "Adaptive\n+support", "#16a34a", "-", "o"),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def values_for(df: pd.DataFrame, type_mix: str, strategy: str) -> tuple[list[int], list[float]]:
    sub = df[(df["type_mix"] == type_mix) & (df["strategy"] == strategy)].sort_values("G")
    return sub["G"].astype(int).tolist(), sub["pct_oracle"].astype(float).tolist()


def main() -> None:
    setup_style()
    df = pd.read_csv(INPUT)
    df = df[df["strategy"] != "Oracle"].copy()

    all_values = df["pct_oracle"].astype(float).to_numpy()
    ymin = max(0.0, float(np.nanmin(all_values)) - 4.0)
    ymax = min(104.0, float(np.nanmax(all_values)) + 4.0)

    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.9), sharex=True, sharey=True)
    for ax, type_mix, title in zip(axes, ["Homo.", "Heter."], ["Homogeneous", "Heterogeneous"]):
        for strategy, _short, color, linestyle, marker in METHODS:
            xs, ys = values_for(df, type_mix, strategy)
            if not xs:
                continue
            linewidth = 2.6 if strategy.startswith("Adp.") else 2.0
            alpha = 0.92 if strategy.startswith("Adp.") or strategy == "TW" else 0.78
            ax.plot(
                xs,
                ys,
                marker=marker,
                markersize=6.5,
                linewidth=linewidth,
                linestyle=linestyle,
                color=color,
                alpha=alpha,
            )
        ax.set_title(title, loc="left", fontweight="bold", fontsize=15, pad=8)
        ax.grid(True, axis="both", color="#d7d7d7", linewidth=0.7, alpha=0.75)
        ax.set_xticks([5, 10, 15])
        ax.set_xlabel("Grid-region states $G$", fontsize=13)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis="both", labelsize=12)

    axes[0].set_ylabel("Reward (% oracle)", fontsize=13)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=2.6 if strategy.startswith("Adp.") else 2.0,
            linestyle=linestyle,
            marker=marker,
            markersize=6,
        )
        for strategy, _, color, linestyle, marker in METHODS
    ]
    legend_labels = [short for _, short, *_ in METHODS]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.52, -0.02),
        handlelength=2.5,
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0.21, 1, 1.0], w_pad=2.0)
    fig.savefig(OUT_DIR / "fig_region_grid_load_balance_points.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_region_grid_load_balance_points.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
