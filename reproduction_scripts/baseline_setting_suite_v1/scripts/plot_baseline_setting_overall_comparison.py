"""Plot the baseline-setting comparison from the one-dimensional zero-noise run."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-yifu")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs/research/rmab_vm_outputs/baseline_setting_suite_v1"
SUMMARY = OUT_DIR / "context_noise_summary.csv"
SUPPLEMENT_SUMMARY = (
    ROOT
    / "docs/research/rmab_vm_outputs/baseline_setting_beta_offline_supplement_v1/context_noise_summary.csv"
)

METHODS = [
    ("state_thompson", "ST", "Baselines"),
    ("tw_dense", "TW", "Baselines"),
    ("local_ucb_tw_dense", "Local\nUCB+TW", "Baselines"),
    ("global_ucb_tw_dense", "Global\nUCB+TW", "Baselines"),
    ("exp4_dense", "EXP4", "Baselines"),
    ("tm_tw_dense", "TM-TW", "Original main"),
    ("tm_tw_refined_dense", "Adaptive\nTM-TW", "Refined TM-TW"),
    ("tm_tw_refined_offline", "Offline\nprior", "Refined TM-TW"),
    ("tm_tw_refined_gated_offline", "Gated\nprior", "Refined TM-TW"),
    ("tm_tw_refined_gated_offline_low_rank", "Gated\n+ LR", "Refined TM-TW"),
    ("tm_tw_refined_gated_offline_beta", "Beta-gate\nprior", "Refined TM-TW"),
    ("tm_tw_refined_gated_offline_low_rank_beta", "Beta-gate\n+ LR", "Refined TM-TW"),
    ("tm_tw_refined_support_offline", "Support\n+ offline", "Refined TM-TW"),
]

COLORS = {
    "Baselines": "#94a3b8",
    "Original main": "#2f5f8f",
    "Refined TM-TW": "#d9812e",
}


def main() -> None:
    df = pd.read_csv(SUMMARY)
    values_by_label = dict(zip(df["policy_label"], df["mean_reward_pct_oracle"]))
    if SUPPLEMENT_SUMMARY.exists():
        supplement = pd.read_csv(SUPPLEMENT_SUMMARY)
        deterministic = supplement[supplement["gate_mode"] == "deterministic"]
        beta = supplement[supplement["gate_mode"] == "beta"]
        for _, row in deterministic.iterrows():
            values_by_label[row["policy_label"]] = row["mean_reward_pct_oracle"]
        beta_keys = {
            "tm_tw_refined_gated_offline": "tm_tw_refined_gated_offline_beta",
            "tm_tw_refined_gated_offline_low_rank": "tm_tw_refined_gated_offline_low_rank_beta",
        }
        for source, target in beta_keys.items():
            rows = beta[beta["policy_label"] == source]
            if not rows.empty:
                values_by_label[target] = rows["mean_reward_pct_oracle"].iloc[0]
    methods = [(key, label, role) for key, label, role in METHODS if key in values_by_label]
    values = [float(values_by_label[key]) for key, _, _ in methods]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    xs = range(len(methods))
    bars = ax.bar(
        xs,
        values,
        color=[COLORS[role] for _, _, role in methods],
        edgecolor="#1f1f1f",
        linewidth=0.55,
        width=0.72,
    )

    ax.set_ylim(max(0.0, min(values) - 4.0), min(103.5, max(values) + 3.5))
    ax.set_xticks(list(xs))
    ax.set_xticklabels([label for _, label, _ in methods], rotation=60, ha="right", fontsize=10)
    ax.set_ylabel("Cumulative reward (% of true-state Whittle)")
    ax.set_title("Baseline setting: one-dimensional state, no contextual noise", loc="left", fontweight="bold", pad=8)
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.7, alpha=0.75)
    ax.axhline(100.0, color="#222222", linestyle="--", linewidth=1.0, alpha=0.72)
    ax.text(0.98, 0.97, "Oracle = 100%", transform=ax.transAxes, ha="right", va="top", fontsize=8)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.18,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    base = values_by_label.get("tm_tw_dense")
    refined = values_by_label.get("tm_tw_refined_dense")
    gated = values_by_label.get("tm_tw_refined_gated_offline")
    if base is not None and refined is not None:
        ax.annotate(
            f"adaptive trust +{float(refined) - float(base):.1f}",
            xy=(6, float(refined)),
            xytext=(4.9, float(refined) + 1.0),
            arrowprops={"arrowstyle": "->", "color": "#1f9a8a", "lw": 1.1},
            color="#12685e",
            fontsize=8.5,
            fontweight="bold",
        )
    if base is not None and gated is not None:
        ax.annotate(
            f"gated +{float(gated) - float(base):.1f}",
            xy=(7, float(gated)),
            xytext=(7.3, float(gated) + 1.2),
            arrowprops={"arrowstyle": "->", "color": "#d9812e", "lw": 1.1},
            color="#a65f16",
            fontsize=8.5,
            fontweight="bold",
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[role], ec="#1f1f1f", lw=0.55)
        for role in ["Baselines", "Original main", "Refined TM-TW"]
    ]
    fig.legend(
        handles,
        ["Baselines", "Original TM-TW", "Refined TM-TW"],
        frameon=False,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.55, 1.02),
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_overall_baseline_comparison.png", dpi=300)
    fig.savefig(OUT_DIR / "fig_overall_baseline_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
