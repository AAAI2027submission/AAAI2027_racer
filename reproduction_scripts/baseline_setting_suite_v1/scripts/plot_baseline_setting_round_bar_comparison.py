"""Plot baseline-setting bar comparisons at selected learning-curve rounds."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-yifu")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs/research/rmab_vm_outputs/baseline_setting_suite_v1"
CURVES_BY_SEED = OUT_DIR / "baseline_setting_learning_curves_by_seed.csv"
OUT_CSV = OUT_DIR / "baseline_setting_round_bar_comparison.csv"
ROUNDS = [100, 1000]

METHODS = [
    ("State Thompson (ST)", "ST", "Baselines"),
    ("TW", "TW", "TW"),
    ("Local UCB + TW", "Local\nUCB+TW", "Baselines"),
    ("Global UCB + TW", "Global\nUCB+TW", "Baselines"),
    ("EXP4-based", "EXP4", "Baselines"),
    ("TM-TW", "TM-TW", "Baselines"),
    ("Adp. TM-TW", "Adaptive\nTM-TW", "Refined TM-TW"),
    ("Adp. + offline prior", "Offline\nprior", "Refined TM-TW"),
    ("Adp. + gated prior", "Gated\nprior", "Refined TM-TW"),
    ("Adp. + gated + LR", "Gated\n+ LR", "Refined TM-TW"),
    ("Adp. + beta gate prior", "Beta-gate\nprior", "Refined TM-TW"),
    ("Adp. + beta gate + LR", "Beta-gate\n+ LR", "Refined TM-TW"),
    ("Adp. + support/offline", "Support\n+ offline", "Refined TM-TW"),
]

COLORS = {
    "Baselines": "#94a3b8",
    "TW": "#08244f",
    "Refined TM-TW": "#d9812e",
}


def make_comparison(curves_by_seed: pd.DataFrame) -> pd.DataFrame:
    rows = curves_by_seed[curves_by_seed["round"].isin(ROUNDS)].copy()
    oracle_by_seed_round = rows[rows["strategy"] == "Oracle Whittle"][
        ["seed_index", "round", "cumulative_average_reward"]
    ].rename(columns={"cumulative_average_reward": "oracle_cumulative_average_reward"})
    methods = pd.DataFrame(
        [
            {"strategy": strategy, "display_name": display_name, "role": role, "order": order}
            for order, (strategy, display_name, role) in enumerate(METHODS)
        ]
    )
    rows = rows.merge(methods, on="strategy", how="inner")
    rows = rows.merge(oracle_by_seed_round, on=["seed_index", "round"], how="left")
    rows["reward_pct_oracle_seed"] = (
        100.0 * rows["cumulative_average_reward"] / rows["oracle_cumulative_average_reward"]
    )
    rows = (
        rows.groupby(["round", "strategy", "display_name", "role", "order"], as_index=False)
        .agg(
            mean_cumulative_average_reward=("cumulative_average_reward", "mean"),
            reward_pct_oracle=("reward_pct_oracle_seed", "mean"),
            std_reward_pct_oracle=("reward_pct_oracle_seed", "std"),
            n=("reward_pct_oracle_seed", "count"),
        )
        .sort_values(["round", "order"])
    )
    rows["std_reward_pct_oracle"] = rows["std_reward_pct_oracle"].fillna(0.0)
    rows[
        [
            "round",
            "strategy",
            "display_name",
            "role",
            "mean_cumulative_average_reward",
            "reward_pct_oracle",
            "std_reward_pct_oracle",
            "n",
        ]
    ].to_csv(OUT_CSV, index=False)
    return rows


def plot_comparison(summary: pd.DataFrame) -> None:
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

    fig, axes = plt.subplots(len(ROUNDS), 1, figsize=(7.4, 7.4), sharey=True)
    for ax, round_value in zip(axes, ROUNDS):
        data = summary[summary["round"] == round_value].sort_values("order")
        xs = range(len(data))
        values = data["reward_pct_oracle"].tolist()
        errors = data["std_reward_pct_oracle"].tolist()
        bars = ax.bar(
            xs,
            values,
            yerr=errors,
            color=[COLORS[role] for role in data["role"]],
            edgecolor="#1f1f1f",
            ecolor="#1f1f1f",
            capsize=2.8,
            error_kw={"elinewidth": 0.75, "capthick": 0.75},
            linewidth=0.55,
            width=0.72,
        )

        ax.set_title(f"Round {round_value}", loc="left", fontweight="bold", pad=8)
        ax.set_ylabel("Cumulative average reward\n(% of Oracle)")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(data["display_name"].tolist(), rotation=60, ha="right", fontsize=10)
        ax.grid(axis="y", color="#d7d7d7", linewidth=0.7, alpha=0.75)
        ax.axhline(100.0, color="#222222", linestyle="--", linewidth=1.0, alpha=0.72)
        ax.text(0.98, 0.96, "Oracle = 100%", transform=ax.transAxes, ha="right", va="top", fontsize=8)

        for bar, value, err in zip(bars, values, errors):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + err + 0.18,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=90,
            )

    y_min = max(0.0, (summary["reward_pct_oracle"] - summary["std_reward_pct_oracle"]).min() - 4.0)
    y_max = min(106.0, (summary["reward_pct_oracle"] + summary["std_reward_pct_oracle"]).max() + 4.0)
    axes[0].set_ylim(y_min, y_max)
    legend_roles = ["Baselines", "TW", "Refined TM-TW"]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[role], ec="#1f1f1f", lw=0.55)
        for role in legend_roles
    ]
    axes[0].legend(
        handles,
        ["Baselines", "TW", "Refined TM-TW"],
        frameon=False,
        loc="upper left",
        ncol=2,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_baseline_setting_round_bar_comparison.png", dpi=300)
    fig.savefig(OUT_DIR / "fig_baseline_setting_round_bar_comparison.pdf")
    plt.close(fig)


def main() -> None:
    curves_by_seed = pd.read_csv(CURVES_BY_SEED)
    summary = make_comparison(curves_by_seed)
    plot_comparison(summary)


if __name__ == "__main__":
    main()
