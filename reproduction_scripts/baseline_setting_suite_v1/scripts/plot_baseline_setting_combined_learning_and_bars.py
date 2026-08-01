"""Combine baseline-setting learning curves and round bar comparisons."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-yifu")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_OUT_DIR = ROOT / "docs/research/rmab_vm_outputs/baseline_setting_suite_v1"
BUNDLE_OUT_DIR = REPO_ROOT / "experiments/baseline_setting_suite_v1"
OUT_DIR = LEGACY_OUT_DIR if LEGACY_OUT_DIR.exists() else BUNDLE_OUT_DIR
CURVES = OUT_DIR / "baseline_setting_learning_curves.csv"
CURVES_BY_SEED = OUT_DIR / "baseline_setting_learning_curves_by_seed.csv"
ROUNDS = [100, 1000]

STRATEGY_ORDER = [
    "Oracle Whittle",
    "State Thompson (ST)",
    "TW",
    "Local UCB + TW",
    "Global UCB + TW",
    "EXP4-based",
    "Adp. TM-TW",
    "Adp. + offline prior",
    "Adp. + gated prior",
    "Adp. + gated + LR",
    "Adp. + beta gate prior",
    "Adp. + beta gate + LR",
]

BAR_METHODS = [
    ("State Thompson (ST)", "ST", "Baselines"),
    ("TW", "TW", "TW"),
    ("Local UCB + TW", "Local\nUCB+TW", "Baselines"),
    ("Global UCB + TW", "Global\nUCB+TW", "Baselines"),
    ("EXP4-based", "EXP4", "Baselines"),
    ("Adp. TM-TW", "Adaptive\nTW", "Refined TW"),
    ("Adp. + offline prior", "Offline\nprior", "Refined TW"),
    ("Adp. + gated prior", "Gated\nprior", "Refined TW"),
    ("Adp. + gated + LR", "Gated\n+ LR", "Refined TW"),
    ("Adp. + beta gate prior", "Beta-gate\nprior", "Refined TW"),
    ("Adp. + beta gate + LR", "Beta-gate\n+ LR", "Refined TW"),
]

GROUP_STYLE = {
    "oracle": {"color": "#000000", "linestyle": "-", "linewidth": 2.2, "alpha": 0.95},
    "baseline": {"color": "#2f5f8f", "linestyle": "-", "linewidth": 1.45, "alpha": 0.70},
    "tw": {"color": "#08244f", "linestyle": "-.", "linewidth": 2.2, "alpha": 0.98},
    "refined": {"color": "#d9812e", "linestyle": "-", "linewidth": 1.65, "alpha": 0.78},
}

BAR_COLORS = {
    "Baselines": "#94a3b8",
    "TW": "#08244f",
    "Refined TW": "#d9812e",
}


def strategy_style(row: pd.Series) -> dict:
    if row["strategy"] == "TW":
        return GROUP_STYLE["tw"]
    return GROUP_STYLE[row["group"]]


def round_bar_data(curves_by_seed: pd.DataFrame) -> pd.DataFrame:
    rows = curves_by_seed[curves_by_seed["round"].isin(ROUNDS)].copy()
    oracle_by_seed_round = rows[rows["strategy"] == "Oracle Whittle"][
        ["seed_index", "round", "cumulative_average_reward"]
    ].rename(columns={"cumulative_average_reward": "oracle_cumulative_average_reward"})
    methods = pd.DataFrame(
        [
            {"strategy": strategy, "display_name": display_name, "role": role, "order": order}
            for order, (strategy, display_name, role) in enumerate(BAR_METHODS)
        ]
    )
    rows = rows.merge(methods, on="strategy", how="inner")
    rows = rows.merge(oracle_by_seed_round, on=["seed_index", "round"], how="left")
    rows["reward_pct_oracle_seed"] = (
        100.0 * rows["cumulative_average_reward"] / rows["oracle_cumulative_average_reward"]
    )
    summary = (
        rows.groupby(["round", "strategy", "display_name", "role", "order"], as_index=False)
        .agg(
            mean_cumulative_average_reward=("cumulative_average_reward", "mean"),
            reward_pct_oracle=("reward_pct_oracle_seed", "mean"),
            std_reward_pct_oracle=("reward_pct_oracle_seed", "std"),
            n=("reward_pct_oracle_seed", "count"),
        )
        .sort_values(["round", "order"])
    )
    summary["std_reward_pct_oracle"] = summary["std_reward_pct_oracle"].fillna(0.0)
    return summary


def main() -> None:
    curves = pd.read_csv(CURVES)
    curves_by_seed = pd.read_csv(CURVES_BY_SEED)
    bars = round_bar_data(curves_by_seed)

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

    fig = plt.figure(figsize=(15.2, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.15], wspace=0.24)
    ax_curve = fig.add_subplot(gs[0, 0])
    bar_gs = gs[0, 1].subgridspec(2, 1, hspace=0.72)
    ax_bar_100 = fig.add_subplot(bar_gs[0, 0])
    ax_bar_1000 = fig.add_subplot(bar_gs[1, 0], sharey=ax_bar_100)
    bar_axes = [ax_bar_100, ax_bar_1000]

    for strategy in STRATEGY_ORDER:
        rows = curves[curves["strategy"] == strategy]
        if rows.empty:
            continue
        style = strategy_style(rows.iloc[0])
        ax_curve.plot(
            rows["round"],
            rows["mean_cumulative_average_reward"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
        )

    ax_curve.text(0.0, 1.075, "(a)", transform=ax_curve.transAxes, fontsize=13, fontweight="bold", va="bottom")
    ax_curve.set_xlabel("Round", fontsize=14)
    ax_curve.set_ylabel("Average reward across rounds", fontsize=12)
    ax_curve.set_xlim(0, 1000)
    ax_curve.tick_params(axis="both", labelsize=10)
    ax_curve.grid(True, color="#d7d7d7", linewidth=0.7, alpha=0.75)

    legend_handles = [
        Line2D([0], [0], color=GROUP_STYLE["oracle"]["color"], linestyle=GROUP_STYLE["oracle"]["linestyle"],
               linewidth=GROUP_STYLE["oracle"]["linewidth"]),
        Line2D([0], [0], color=GROUP_STYLE["baseline"]["color"], linestyle=GROUP_STYLE["baseline"]["linestyle"],
               linewidth=GROUP_STYLE["baseline"]["linewidth"]),
        Line2D([0], [0], color=GROUP_STYLE["tw"]["color"], linestyle=GROUP_STYLE["tw"]["linestyle"],
               linewidth=GROUP_STYLE["tw"]["linewidth"]),
        Line2D([0], [0], color=GROUP_STYLE["refined"]["color"], linestyle=GROUP_STYLE["refined"]["linestyle"],
               linewidth=GROUP_STYLE["refined"]["linewidth"]),
    ]
    ax_curve.legend(
        legend_handles,
        ["Oracle Whittle", "Baseline strategies", "TW", "Refined variants"],
        frameon=False,
        loc="lower right",
        fontsize=11.0,
    )

    for ax, round_value in zip(bar_axes, ROUNDS):
        data = bars[bars["round"] == round_value].sort_values("order")
        xs = range(len(data))
        values = data["reward_pct_oracle"].tolist()
        errors = data["std_reward_pct_oracle"].tolist()
        bar_artists = ax.bar(
            xs,
            values,
            yerr=errors,
            color=[BAR_COLORS[role] for role in data["role"]],
            edgecolor="#1f1f1f",
            ecolor="#1f1f1f",
            capsize=2.8,
            error_kw={"elinewidth": 0.75, "capthick": 0.75},
            linewidth=0.55,
            width=0.72,
        )
        ax.set_title(f"Round {round_value}", loc="left", fontweight="bold", pad=6)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(data["display_name"].tolist(), rotation=60, ha="right", fontsize=12.0)
        ax.set_ylabel("Cumulative average reward\n(% of Oracle)", fontsize=10.5)
        ax.grid(axis="y", color="#d7d7d7", linewidth=0.7, alpha=0.75)
        ax.axhline(100.0, color="#222222", linestyle="--", linewidth=1.0, alpha=0.72)
        best_value = max(values)
        for bar, value, err in zip(bar_artists, values, errors):
            if value != best_value:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + err + 0.55,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9.0,
                fontweight="bold" if value == best_value else "normal",
                rotation=90,
            )

    y_min = max(0.0, (bars["reward_pct_oracle"] - bars["std_reward_pct_oracle"]).min() - 4.0)
    y_max = min(106.0, (bars["reward_pct_oracle"] + bars["std_reward_pct_oracle"]).max() + 4.0)
    ax_bar_100.set_ylim(y_min, y_max)
    ax_bar_100.text(0.0, 1.20, "(b)", transform=ax_bar_100.transAxes, fontsize=13, fontweight="bold", va="bottom")

    bar_legend_roles = ["Baselines", "TW", "Refined TW"]
    bar_handles = [
        plt.Rectangle((0, 0), 1, 1, color=BAR_COLORS[role], ec="#1f1f1f", lw=0.55)
        for role in bar_legend_roles
    ]
    ax_bar_100.legend(
        bar_handles,
        ["Baselines", "TW", "Refined TW"],
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.08),
        ncol=3,
        fontsize=11.0,
        handlelength=1.8,
        handleheight=0.85,
        columnspacing=1.8,
    )

    fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.34)
    fig.savefig(OUT_DIR / "fig_baseline_setting_learning_and_round_bar.png", dpi=300)
    fig.savefig(OUT_DIR / "fig_baseline_setting_learning_and_round_bar.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
