"""Plot normalized regional carbon intensity and electricity price inputs."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-yifu")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "supporting_inputs/grid_cost_features_v1/grid_cost_region_averages_2023_normalized.csv"
OUT_DIR = ROOT / "supporting_inputs/grid_cost_features_v1"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def main() -> None:
    setup_style()
    df = pd.read_csv(INPUT).sort_values("grid_rank_low_to_high").reset_index(drop=True)
    labels = df["region_code"].tolist()
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.bar(
        x - width / 2,
        df["carbon_direct_norm"],
        width,
        label="Carbon intensity",
        color="#2563eb",
    )
    ax.bar(
        x + width / 2,
        df["electricity_price_norm"],
        width,
        label="Electricity price",
        color="#f97316",
    )

    ax.set_title("Normalized 2023 Grid/Carbon Conditions by Region", loc="left", fontweight="bold")
    ax.set_ylabel("Normalized value")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, axis="y", color="#d7d7d7", linewidth=0.7, alpha=0.75)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_normalized_carbon_electricity_prices.png", dpi=300)
    fig.savefig(OUT_DIR / "fig_normalized_carbon_electricity_prices.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
