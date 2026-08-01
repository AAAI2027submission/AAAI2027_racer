"""Plot Sweep A real-data context noise with added best beta-gate variants."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-yifu")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_OUT_DIR = ROOT / "docs/research/rmab_vm_outputs/context_noise_real_data_v1"
BUNDLE_OUT_DIR = REPO_ROOT / "experiments/context_noise_real_data_v1"
OUT_DIR = LEGACY_OUT_DIR if LEGACY_OUT_DIR.exists() else BUNDLE_OUT_DIR
SUMMARY_PATH = OUT_DIR / "context_noise_summary_with_best_beta_gate.csv"
RESULTS_PATH = OUT_DIR / "context_noise_results_with_best_beta_gate.csv"
BASE_SEED = 20260425
SUMMARY_GROUP_COLUMNS = [
    "S",
    "context_noise_level",
    "trust_scale_mult",
    "gate_scale_mult",
    "gate_mode",
    "trust_floor",
    "policy_label",
]

BASELINE_LABELS = [
    "state_thompson",
    "tw_dense",
    "local_ucb_tw_dense",
    "global_ucb_tw_dense",
    "exp4_dense",
]

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
    "state_thompson": "State Thompson",
    "tw_dense": "TW",
    "local_ucb_tw_dense": "Local UCB+TW",
    "global_ucb_tw_dense": "Global UCB+TW",
    "exp4_dense": "EXP4",
    "tm_tw_dense": "TM--TW",
    "tm_tw_refined_dense": "Adp. TW",
    "tm_tw_refined_gated_offline": "Adp. TW + gated prior",
    "tm_tw_refined_gated_offline_low_rank": "Adp. TW + gated prior + low-rank",
    "tm_tw_refined_support_offline": "Adp. TW + support/offline prior",
    "tm_tw_refined_gated_offline_best_beta": "Adp. TW + beta gated prior",
    "tm_tw_refined_gated_offline_low_rank_best_beta": (
        "Adp. TW + beta gated prior + low-rank"
    ),
    "tm_tw_refined_support_gated_offline_low_rank_best_beta": (
        "Adp. TW + beta support-gated prior + low-rank"
    ),
}

METHODS = [
    ("best_baseline", "Best paper\nbaseline", "#7a8794", "--", "s"),
    ("tm_tw_dense", "TW", "#2f5f8f", "--", "^"),
    ("tm_tw_refined_dense", "Adaptive\nTW", "#2c7fb8", "-", "o"),
    ("tm_tw_refined_gated_offline", "Adaptive\n+gated", "#d9812e", "-", "o"),
    ("tm_tw_refined_gated_offline_low_rank", "Adaptive\n+gated+LR", "#c2410c", "-", "o"),
    ("tm_tw_refined_support_offline", "Adaptive\n+support", "#1f9a8a", "-", "o"),
    ("tm_tw_refined_gated_offline_best_beta", "Best beta\n+gated", "#9467bd", "-.", "D"),
    (
        "tm_tw_refined_gated_offline_low_rank_best_beta",
        "Best beta\n+gated+LR",
        "#6b21a8",
        "-.",
        "D",
    ),
    (
        "tm_tw_refined_support_gated_offline_low_rank_best_beta",
        "Best beta\n+support+gated+LR",
        "#0f766e",
        "-.",
        "D",
    ),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(SUMMARY_GROUP_COLUMNS, dropna=False)
        .agg(
            n=("reward_pct_oracle", "size"),
            mean_reward_pct_oracle=("reward_pct_oracle", "mean"),
            mean_transition_l1_error=("transition_l1_error", "mean"),
            mean_off_support_leakage=("off_support_leakage", "mean"),
            mean_top1_agreement=("top1_agreement", "mean"),
            mean_top2_agreement=("top2_agreement", "mean"),
            mean_cum_regret=("cum_regret", "mean"),
        )
        .reset_index()
    )


def cn_val(df: pd.DataFrame, state: int, noise: float, label: str) -> float:
    mask = (
        (df["S"] == state)
        & (abs(df["context_noise_level"] - noise) < 1e-9)
        & (df["policy_label"] == label)
    )
    rows = df[mask]
    return float(rows["mean_reward_pct_oracle"].iloc[0]) if not rows.empty else float("nan")


def best_baseline(df: pd.DataFrame, state: int, noise: float) -> float:
    return float(np.nanmax([cn_val(df, state, noise, label) for label in BASELINE_LABELS]))


def best_baseline_with_label(df: pd.DataFrame, state: int, noise: float) -> tuple[str, float]:
    values = [(label, cn_val(df, state, noise, label)) for label in BASELINE_LABELS]
    return max(values, key=lambda item: item[1] if not np.isnan(item[1]) else -999.0)


def best_refined_with_label(df: pd.DataFrame, state: int, noise: float) -> tuple[str, float]:
    values = [(label, cn_val(df, state, noise, label)) for label in REFINED_LABELS]
    return max(values, key=lambda item: item[1] if not np.isnan(item[1]) else -999.0)


def stable_policy_offset(policy: str, variant: str) -> int:
    return sum(ord(ch) for ch in f"{policy}:{variant}")


def seed_level_values(results: pd.DataFrame, state: int, noise: float, label: str) -> pd.DataFrame:
    rows = results[
        (results["S"] == state)
        & (abs(results["context_noise_level"] - noise) < 1e-9)
        & (results["policy_label"] == label)
    ].copy()
    if rows.empty:
        return pd.DataFrame(columns=["seed_index", "reward_pct_oracle"])

    offsets = [
        stable_policy_offset(str(row.policy), str(row.transition_variant))
        for row in rows.itertuples(index=False)
    ]
    rows["seed_index"] = rows["seed"].astype(int) - BASE_SEED - np.asarray(offsets, dtype=int)
    return rows[["seed_index", "reward_pct_oracle"]]


def paired_margin_std(
    results: pd.DataFrame,
    state: int,
    noise: float,
    refined_label: str,
    comparator_label: str,
) -> float:
    refined = seed_level_values(results, state, noise, refined_label).rename(
        columns={"reward_pct_oracle": "refined_reward_pct_oracle"}
    )
    comparator = seed_level_values(results, state, noise, comparator_label).rename(
        columns={"reward_pct_oracle": "comparator_reward_pct_oracle"}
    )
    paired = refined.merge(comparator, on="seed_index", how="inner")
    if len(paired) <= 1:
        return 0.0
    margins = paired["refined_reward_pct_oracle"] - paired["comparator_reward_pct_oracle"]
    return float(margins.std(ddof=1))


def paired_margin_p_value(
    results: pd.DataFrame,
    state: int,
    noise: float,
    refined_label: str,
    comparator_label: str,
) -> float:
    refined = seed_level_values(results, state, noise, refined_label).rename(
        columns={"reward_pct_oracle": "refined_reward_pct_oracle"}
    )
    comparator = seed_level_values(results, state, noise, comparator_label).rename(
        columns={"reward_pct_oracle": "comparator_reward_pct_oracle"}
    )
    paired = refined.merge(comparator, on="seed_index", how="inner")
    if len(paired) <= 1:
        return float("nan")
    margins = paired["refined_reward_pct_oracle"] - paired["comparator_reward_pct_oracle"]
    try:
        from scipy import stats

        return float(stats.ttest_1samp(margins.to_numpy(), 0.0).pvalue)
    except Exception:
        return float("nan")


def significance_stars(p_value: float) -> str:
    if np.isnan(p_value):
        return ""
    if p_value < 0.001:
        return r"$^{***}$"
    if p_value < 0.01:
        return r"$^{**}$"
    if p_value < 0.05:
        return r"$^{*}$"
    return ""


def margin_cell(mean_margin: float, std_margin: float, p_value: float) -> str:
    return rf"{mean_margin:+.2f}$\pm${std_margin:.2f}{significance_stars(p_value)}"


def plot_context_noise(df: pd.DataFrame) -> None:
    states = [8, 20, 50, 100]
    noises = [0.0, 0.1, 0.2, 0.3]

    all_values: list[float] = []
    for state in states:
        for label, *_ in METHODS:
            for noise in noises:
                value = best_baseline(df, state, noise) if label == "best_baseline" else cn_val(df, state, noise, label)
                if not np.isnan(value):
                    all_values.append(value)

    ymin = max(0.0, min(all_values) - 3.0)
    ymax = min(106.0, max(all_values) + 3.0)

    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.2), sharex=True, sharey=True)
    for ax, state in zip(axes.flat, states):
        for label, short, color, linestyle, marker in METHODS:
            values = [
                best_baseline(df, state, noise) if label == "best_baseline" else cn_val(df, state, noise, label)
                for noise in noises
            ]
            if all(np.isnan(value) for value in values):
                continue
            linewidth = 1.8 if label == "best_baseline" else 2.1
            ax.plot(
                noises,
                values,
                marker=marker,
                linewidth=linewidth,
                linestyle=linestyle,
                color=color,
                label=short,
            )
        ax.set_title(f"S={state}", loc="left", fontweight="bold", fontsize=13, pad=7)
        ax.grid(True, axis="both", color="#d7d7d7", linewidth=0.7, alpha=0.75)
        ax.set_xticks(noises)
        ax.set_ylim(ymin, ymax)

    axes[0, 0].set_ylabel("Reward (% oracle)", fontsize=12)
    axes[1, 0].set_ylabel("Reward (% oracle)", fontsize=12)
    axes[1, 0].set_xlabel("Contextual state-noise probability", fontsize=12)
    axes[1, 1].set_xlabel("Contextual state-noise probability", fontsize=12)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=1.8 if label == "best_baseline" else 2.1,
            linestyle=linestyle,
            marker=marker,
            markersize=6.5,
        )
        for label, _, color, linestyle, marker in METHODS
    ]
    legend_labels = [short for _, short, *_ in METHODS]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.52, 0.0),
        handlelength=2.8,
        handletextpad=0.7,
        columnspacing=1.5,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.18, 1, 1.0])
    fig.savefig(OUT_DIR / "fig_context_noise_real_with_best_beta_gate.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_context_noise_real_with_best_beta_gate.pdf", bbox_inches="tight")
    plt.close(fig)


def write_table(df: pd.DataFrame, results: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sweep A (real VM data): contextual-noise robustness with best beta-gate variants. "
        r"For each $(|S|, \text{noise})$ cell the table reports the best paper-style baseline "
        r"(with its name), original TM--TW, and the best refined variant, including the added "
        r"best beta-gate policies where available.}",
        r"\label{tab:sweep_a_real_best_beta_gate}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{cclccclcc}",
        r"\toprule",
        r"$|S|$ & Noise & Best baseline name & Best base & TM--TW "
        r"& Best refined & Best refined variant & Margin vs baseline & Margin vs TW \\",
        r"\midrule",
    ]
    for state in [8, 20, 50, 100]:
        for noise in [0.0, 0.1, 0.2, 0.3]:
            baseline_label, baseline_value = best_baseline_with_label(df, state, noise)
            tw_value = cn_val(df, state, noise, "tw_dense")
            tmtw_value = cn_val(df, state, noise, "tm_tw_dense")
            refined_label, refined_value = best_refined_with_label(df, state, noise)
            margin_vs_baseline = refined_value - baseline_value
            margin_vs_tw = refined_value - tw_value
            margin_vs_baseline_std = paired_margin_std(
                results,
                state,
                noise,
                refined_label,
                baseline_label,
            )
            margin_vs_tw_std = paired_margin_std(
                results,
                state,
                noise,
                refined_label,
                "tw_dense",
            )
            margin_vs_baseline_p = paired_margin_p_value(
                results,
                state,
                noise,
                refined_label,
                baseline_label,
            )
            margin_vs_tw_p = paired_margin_p_value(
                results,
                state,
                noise,
                refined_label,
                "tw_dense",
            )
            lines.append(
                f"{state} & {noise:.1f} & {DISPLAY.get(baseline_label, baseline_label)} & "
                f"{baseline_value:.2f} & {tmtw_value:.2f} & {refined_value:.2f} & "
                rf"\textbf{{{DISPLAY.get(refined_label, refined_label)}}} & "
                f"{margin_cell(margin_vs_baseline, margin_vs_baseline_std, margin_vs_baseline_p)} & "
                f"{margin_cell(margin_vs_tw, margin_vs_tw_std, margin_vs_tw_p)} \\\\"
            )
        if state != 100:
            lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{0.25em}",
        r"\footnotesize{Margins are mean$\pm$one standard deviation over 10 paired seeds; "
        r"$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ by paired $t$-test.}",
        r"\end{table}",
        "",
    ]
    (OUT_DIR / "tab_context_noise_real_with_best_beta_gate.tex").write_text("\n".join(lines))


def main() -> None:
    setup_style()
    results = pd.read_csv(RESULTS_PATH)
    df = summarize_results(results)
    df.to_csv(SUMMARY_PATH, index=False)
    plot_context_noise(df)
    write_table(df, results)
    print("Saved context_noise_summary_with_best_beta_gate.csv")
    print("Saved fig_context_noise_real_with_best_beta_gate.pdf/png")
    print("Saved tab_context_noise_real_with_best_beta_gate.tex")


if __name__ == "__main__":
    main()
