"""Generate per-round baseline-setting learning curves and plot them.

This reruns the one-dimensional baseline setting (S=8, T=1000, 10 seeds,
zero contextual noise) so that per-round rewards are available for every
strategy.  The figure follows the older EXP4-vs-ablation reward-curve style:
cumulative average reward over rounds, with baseline comparisons in grey/blue
and refined variants in orange.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-yifu")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_experiments import make_instance, run_single_policy  # noqa: E402


OUT_DIR = ROOT / "docs/research/rmab_vm_outputs/baseline_setting_suite_v1"

N_ARMS = 5
N_STATES = 8
ROUNDS = 1000
SEEDS = 10
BASE_SEED = 20260425
SPARSITY = 2
TRANSITION_DOMINANCE = 0.45

STRATEGIES = [
    {
        "name": "Oracle Whittle",
        "policy": "oracle",
        "variant": "dense",
        "gate_mode": "deterministic",
        "group": "oracle",
        "color": "#000000",
        "linestyle": "--",
        "linewidth": 2.4,
    },
    {
        "name": "State Thompson (ST)",
        "policy": "state_thompson",
        "variant": "dense",
        "gate_mode": "deterministic",
        "group": "baseline",
        "color": "#9ca3af",
        "linestyle": "-",
        "linewidth": 1.7,
    },
    {
        "name": "TW",
        "policy": "tw",
        "variant": "dense",
        "gate_mode": "deterministic",
        "group": "baseline",
        "color": "#0f3d66",
        "linestyle": "-",
        "linewidth": 2.8,
    },
    {
        "name": "Local UCB + TW",
        "policy": "local_ucb_tw",
        "variant": "dense",
        "gate_mode": "deterministic",
        "group": "baseline",
        "color": "#60a5fa",
        "linestyle": "-",
        "linewidth": 1.8,
    },
    {
        "name": "Global UCB + TW",
        "policy": "global_ucb_tw",
        "variant": "dense",
        "gate_mode": "deterministic",
        "group": "baseline",
        "color": "#93c5fd",
        "linestyle": "-",
        "linewidth": 1.8,
    },
    {
        "name": "EXP4-based",
        "policy": "exp4",
        "variant": "dense",
        "gate_mode": "deterministic",
        "group": "baseline",
        "color": "#64748b",
        "linestyle": "-",
        "linewidth": 1.8,
    },
    {
        "name": "TM-TW",
        "policy": "tm_tw",
        "variant": "dense",
        "gate_mode": "deterministic",
        "group": "baseline",
        "color": "#2f5f8f",
        "linestyle": "-",
        "linewidth": 2.3,
    },
    {
        "name": "Adp. TM-TW",
        "policy": "tm_tw_refined",
        "variant": "dense",
        "gate_mode": "deterministic",
        "group": "refined",
        "color": "#f59e0b",
        "linestyle": "-",
        "linewidth": 2.2,
    },
    {
        "name": "Adp. + offline prior",
        "policy": "tm_tw_refined",
        "variant": "offline",
        "gate_mode": "deterministic",
        "group": "refined",
        "color": "#d97706",
        "linestyle": "--",
        "linewidth": 2.0,
    },
    {
        "name": "Adp. + gated prior",
        "policy": "tm_tw_refined",
        "variant": "gated_offline",
        "gate_mode": "deterministic",
        "group": "refined",
        "color": "#ea580c",
        "linestyle": "-",
        "linewidth": 2.0,
    },
    {
        "name": "Adp. + gated + LR",
        "policy": "tm_tw_refined",
        "variant": "gated_offline_low_rank",
        "gate_mode": "deterministic",
        "group": "refined",
        "color": "#c2410c",
        "linestyle": "-.",
        "linewidth": 2.0,
    },
    {
        "name": "Adp. + beta gate prior",
        "policy": "tm_tw_refined",
        "variant": "gated_offline",
        "gate_mode": "beta",
        "group": "refined",
        "color": "#fb923c",
        "linestyle": "-",
        "linewidth": 2.0,
    },
    {
        "name": "Adp. + beta gate + LR",
        "policy": "tm_tw_refined",
        "variant": "gated_offline_low_rank",
        "gate_mode": "beta",
        "group": "refined",
        "color": "#9a3412",
        "linestyle": ":",
        "linewidth": 2.2,
    },
    {
        "name": "Adp. + support/offline",
        "policy": "tm_tw_refined",
        "variant": "support_offline",
        "gate_mode": "deterministic",
        "group": "refined",
        "color": "#fdba74",
        "linestyle": "--",
        "linewidth": 2.0,
    },
]


def stable_strategy_seed(seed: int, policy: str, variant: str, gate_mode: str) -> int:
    key = f"{policy}:{variant}:{gate_mode}"
    return seed + sum(ord(ch) for ch in key)


def generate_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict] = []
    aggregate_rows: list[dict] = []

    curves_by_strategy: dict[str, list[np.ndarray]] = {strategy["name"]: [] for strategy in STRATEGIES}

    for seed_idx in range(SEEDS):
        instance = make_instance(
            seed=BASE_SEED + seed_idx + 31 * N_STATES,
            n_arms=N_ARMS,
            n_states=N_STATES,
            sparsity=SPARSITY,
            transition_dominance=TRANSITION_DOMINANCE,
        )
        for strategy in STRATEGIES:
            run_seed = stable_strategy_seed(
                BASE_SEED + seed_idx,
                strategy["policy"],
                strategy["variant"],
                strategy["gate_mode"],
            )
            _, curve = run_single_policy(
                instance,
                strategy["policy"],
                seed=run_seed,
                rounds=ROUNDS,
                noise_level=0.0,
                transition_variant=strategy["variant"],
                gate_mode=strategy["gate_mode"],
            )
            curves_by_strategy[strategy["name"]].append(curve)
            cumulative_average = np.cumsum(curve) / np.arange(1, ROUNDS + 1)
            for round_idx, (reward, cumavg) in enumerate(zip(curve, cumulative_average), start=1):
                detail_rows.append(
                    {
                        "strategy": strategy["name"],
                        "group": strategy["group"],
                        "policy": strategy["policy"],
                        "transition_variant": strategy["variant"],
                        "gate_mode": strategy["gate_mode"],
                        "seed_index": seed_idx,
                        "round": round_idx,
                        "reward": float(reward),
                        "cumulative_average_reward": float(cumavg),
                    }
                )

    for strategy in STRATEGIES:
        curves = np.vstack(curves_by_strategy[strategy["name"]])
        mean_reward = curves.mean(axis=0)
        mean_cumavg = np.cumsum(mean_reward) / np.arange(1, ROUNDS + 1)
        for round_idx, (reward, cumavg) in enumerate(zip(mean_reward, mean_cumavg), start=1):
            aggregate_rows.append(
                {
                    "strategy": strategy["name"],
                    "group": strategy["group"],
                    "policy": strategy["policy"],
                    "transition_variant": strategy["variant"],
                    "gate_mode": strategy["gate_mode"],
                    "round": round_idx,
                    "mean_reward": float(reward),
                    "mean_cumulative_average_reward": float(cumavg),
                }
            )

    detail = pd.DataFrame(detail_rows)
    aggregate = pd.DataFrame(aggregate_rows)
    detail.to_csv(OUT_DIR / "baseline_setting_learning_curves_by_seed.csv", index=False)
    aggregate.to_csv(OUT_DIR / "baseline_setting_learning_curves.csv", index=False)
    return detail, aggregate


def plot_curves(aggregate: pd.DataFrame) -> None:
    group_style = {
        "oracle": {"color": "#000000", "linestyle": "--", "linewidth": 2.4, "alpha": 0.95},
        "baseline": {"color": "#2f5f8f", "linestyle": "-", "linewidth": 1.6, "alpha": 0.72},
        "tw": {"color": "#08244f", "linestyle": "-.", "linewidth": 2.4, "alpha": 0.98},
        "refined": {"color": "#d9812e", "linestyle": "-", "linewidth": 1.8, "alpha": 0.78},
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    for strategy in STRATEGIES:
        rows = aggregate[aggregate["strategy"] == strategy["name"]]
        style_key = "tw" if strategy["name"] == "TW" else strategy["group"]
        style = group_style[style_key]
        ax.plot(
            rows["round"],
            rows["mean_cumulative_average_reward"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
        )

    ax.set_xlabel("Round", fontsize=14)
    ax.set_ylabel(r"$T^{-1}\sum_{t=1}^{T} r_t$", fontsize=14)
    ax.set_xlim(0, ROUNDS)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, color="#d7d7d7", linewidth=0.7, alpha=0.75)
    legend_handles = [
        Line2D([0], [0], color=group_style["oracle"]["color"], linestyle=group_style["oracle"]["linestyle"],
               linewidth=group_style["oracle"]["linewidth"]),
        Line2D([0], [0], color=group_style["baseline"]["color"], linestyle=group_style["baseline"]["linestyle"],
               linewidth=group_style["baseline"]["linewidth"]),
        Line2D([0], [0], color=group_style["tw"]["color"], linestyle=group_style["tw"]["linestyle"],
               linewidth=group_style["tw"]["linewidth"]),
        Line2D([0], [0], color=group_style["refined"]["color"], linestyle=group_style["refined"]["linestyle"],
               linewidth=group_style["refined"]["linewidth"]),
    ]
    ax.legend(
        legend_handles,
        ["Oracle Whittle", "Baseline strategies", "TW", "Refined variants"],
        loc="best",
        fontsize=11,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_baseline_setting_learning_curves.png", dpi=300)
    fig.savefig(OUT_DIR / "fig_baseline_setting_learning_curves.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Only redraw the figure from baseline_setting_learning_curves.csv.",
    )
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.reuse_existing:
        aggregate = pd.read_csv(OUT_DIR / "baseline_setting_learning_curves.csv")
    else:
        _, aggregate = generate_curves()
    plot_curves(aggregate)


if __name__ == "__main__":
    main()
