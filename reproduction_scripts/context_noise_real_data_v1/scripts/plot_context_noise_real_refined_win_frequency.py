"""Rank refined variants by win frequency in the real-data context-noise sweep."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-yifu")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs/research/rmab_vm_outputs/context_noise_real_data_v1"
SUMMARY_PATH = OUT_DIR / "context_noise_summary_with_best_beta_gate.csv"

REFINED_LABELS = [
    "tm_tw_refined_dense",
    "tm_tw_refined_gated_offline",
    "tm_tw_refined_gated_offline_low_rank",
    "tm_tw_refined_support_offline",
    "tm_tw_refined_gated_offline_best_beta",
    "tm_tw_refined_gated_offline_low_rank_best_beta",
    "tm_tw_refined_support_gated_offline_low_rank_best_beta",
]

DISPLAY = {
    "tm_tw_refined_dense": "Adp. TM--TW",
    "tm_tw_refined_gated_offline": "Adp. + gated prior",
    "tm_tw_refined_gated_offline_low_rank": "Adp. + gated + LR",
    "tm_tw_refined_support_offline": "Adp. + support/\noffline",
    "tm_tw_refined_gated_offline_best_beta": "Adp. + beta\ngate prior",
    "tm_tw_refined_gated_offline_low_rank_best_beta": "Adp. + beta\ngate + LR",
    "tm_tw_refined_support_gated_offline_low_rank_best_beta": "Adp. + beta support-gate\n+ LR",
}


def winner_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for state in sorted(df["S"].unique()):
        for noise in sorted(df["context_noise_level"].unique()):
            cell = df[
                (df["S"] == state)
                & (df["context_noise_level"] == noise)
                & (df["policy_label"].isin(REFINED_LABELS))
            ]
            winner = cell.loc[cell["mean_reward_pct_oracle"].idxmax()]
            rows.append(
                {
                    "S": int(state),
                    "context_noise_level": float(noise),
                    "policy_label": str(winner["policy_label"]),
                    "display_name": DISPLAY[str(winner["policy_label"])],
                    "mean_reward_pct_oracle": float(winner["mean_reward_pct_oracle"]),
                }
            )
    return rows


def main() -> None:
    df = pd.read_csv(SUMMARY_PATH)
    winners = winner_rows(df)
    counts = Counter(row["policy_label"] for row in winners)
    ranked = sorted(
        (
            {
                "policy_label": label,
                "display_name": DISPLAY[label],
                "wins": count,
                "win_share": count / len(winners),
            }
            for label, count in counts.items()
        ),
        key=lambda row: (-row["wins"], row["display_name"]),
    )

    pd.DataFrame(ranked).to_csv(OUT_DIR / "context_noise_refined_win_frequency.csv", index=False)

    plot_rows = list(reversed(ranked))
    labels = [row["display_name"] for row in plot_rows]
    wins = [row["wins"] for row in plot_rows]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    bars = ax.barh(range(len(labels)), wins, color="#d9812e", edgecolor="#1f1f1f", linewidth=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=17)
    ax.set_xlabel("Number of wins across 16 $(|S|, \\rho)$ cells", fontsize=17)
    ax.tick_params(axis="x", labelsize=16)
    ax.set_xlim(0, max(wins) + 1)
    ax.grid(axis="x", color="#d7d7d7", linewidth=0.7, alpha=0.75)
    for bar, win in zip(bars, wins):
        ax.text(
            bar.get_width() + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{win}/16",
            va="center",
            ha="left",
            fontsize=16,
        )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_context_noise_real_refined_win_frequency.png", dpi=300)
    fig.savefig(OUT_DIR / "fig_context_noise_real_refined_win_frequency.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
