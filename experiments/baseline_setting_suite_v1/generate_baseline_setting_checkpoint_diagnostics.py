"""Generate averaged checkpoint diagnostics from per-seed diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
BY_SEED_PATH = HERE / "baseline_setting_checkpoint_diagnostics_by_seed.csv"
SUMMARY_PATH = HERE / "baseline_setting_checkpoint_diagnostics.csv"
LATEX_PATH = HERE / "tab_baseline_setting_checkpoint_diagnostics_ranked.tex"
REDUCTION_LATEX_PATH = HERE / "tab_baseline_setting_checkpoint_reductions.tex"

GROUP_COLUMNS = [
    "round",
    "strategy",
    "display_name",
    "role",
    "policy",
    "transition_variant",
    "gate_mode",
]


def latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def write_latex_table(summary: pd.DataFrame, latex_path: Path = LATEX_PATH) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Baseline-setting checkpoint diagnostics ranked by reward versus oracle. "
        r"Rank 1 is best within each checkpoint round.}",
        r"\label{tab:baseline_setting_checkpoint_diagnostics_ranked}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{rlrrrrr}",
        r"\toprule",
        r"Round & Method & Reward vs. oracle (\%) $\downarrow$ & L1 error & Leakage & Top-1 & Top-2 \\",
        r"\midrule",
    ]
    for round_value, rows in summary.groupby("round", sort=True):
        for _, row in rows.iterrows():
            lines.append(
                f"{int(round_value)} & "
                f"{latex_escape(row['display_name'])} & "
                f"{row['reward_vs_oracle_pct']:.2f} & "
                f"{row['mean_transition_l1_error']:.3f} & "
                f"{row['mean_off_support_leakage']:.3f} & "
                f"{row['mean_top1_agreement']:.3f} & "
                f"{row['mean_top2_agreement']:.3f} \\\\"
            )
        if int(round_value) != int(summary["round"].max()):
            lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ]
    latex_path.write_text("\n".join(lines))


def write_reduction_latex_table(
    summary: pd.DataFrame,
    latex_path: Path = REDUCTION_LATEX_PATH,
) -> None:
    rows = []
    for display_name, method_rows in summary.groupby("display_name", sort=False):
        by_round = method_rows.set_index("round")
        if 100 not in by_round.index or 1000 not in by_round.index:
            continue
        early = by_round.loc[100]
        late = by_round.loc[1000]
        delta_l1 = early["mean_transition_l1_error"] - late["mean_transition_l1_error"]
        delta_leakage = early["mean_off_support_leakage"] - late["mean_off_support_leakage"]
        rows.append(
            {
                "display_name": display_name,
                "rank": int(late["reward_vs_oracle_rank"]),
                "delta_l1": delta_l1,
                "l1_reduction_pct": 100.0 * delta_l1 / early["mean_transition_l1_error"],
                "delta_leakage": delta_leakage,
                "leakage_reduction_pct": (
                    100.0 * delta_leakage / early["mean_off_support_leakage"]
                    if early["mean_off_support_leakage"] != 0
                    else 0.0
                ),
            }
        )

    reductions = pd.DataFrame(rows).sort_values(["rank", "display_name"])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Baseline-setting diagnostic reductions from round 100 to round 1000. "
        r"Rows are ordered by the round-1000 reward-versus-oracle ranking.}",
        r"\label{tab:baseline_setting_checkpoint_reductions}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Strategy & Reward rank & $\Delta$ L1 & L1 red. (\%) & $\Delta$ leakage & Leakage red. (\%) \\",
        r"\midrule",
    ]
    for _, row in reductions.iterrows():
        lines.append(
            f"{latex_escape(row['display_name'])} & "
            f"{int(row['rank'])} & "
            f"{row['delta_l1']:.5f} & "
            f"{row['l1_reduction_pct']:.1f} & "
            f"{row['delta_leakage']:.5g} & "
            f"{row['leakage_reduction_pct']:.1f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    latex_path.write_text("\n".join(lines))


def generate_summary(
    by_seed_path: Path = BY_SEED_PATH,
    summary_path: Path = SUMMARY_PATH,
    latex_path: Path = LATEX_PATH,
    reduction_latex_path: Path = REDUCTION_LATEX_PATH,
) -> None:
    by_seed = pd.read_csv(by_seed_path)
    by_seed = by_seed[by_seed["display_name"] != "TM-TW"].copy()
    by_seed["strategy"] = by_seed["strategy"].replace({"Adp. TM-TW": "Adaptive TW"})
    by_seed["display_name"] = by_seed["display_name"].replace({"Adaptive TM-TW": "Adaptive TW"})
    by_seed["role"] = by_seed["role"].replace({"Refined TM-TW": "Refined TW"})
    summary = (
        by_seed.groupby(GROUP_COLUMNS, as_index=False)
        .agg(
            n=("seed_index", "count"),
            mean_transition_l1_error=("transition_l1_error", "mean"),
            std_transition_l1_error=("transition_l1_error", "std"),
            mean_off_support_leakage=("off_support_leakage", "mean"),
            std_off_support_leakage=("off_support_leakage", "std"),
            mean_reward_pct_oracle=("reward_pct_oracle", "mean"),
            mean_top1_agreement=("top1_agreement", "mean"),
            mean_top2_agreement=("top2_agreement", "mean"),
        )
    )
    summary["reward_vs_oracle_pct"] = summary["mean_reward_pct_oracle"]
    summary["reward_vs_oracle_rank"] = (
        summary.groupby("round")["reward_vs_oracle_pct"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    front_columns = GROUP_COLUMNS + ["n", "reward_vs_oracle_pct", "reward_vs_oracle_rank"]
    remaining_columns = [column for column in summary.columns if column not in front_columns]
    summary = summary[front_columns + remaining_columns].sort_values(
        ["round", "reward_vs_oracle_rank", "strategy"]
    )
    summary.to_csv(summary_path, index=False)
    write_latex_table(summary, latex_path)
    write_reduction_latex_table(summary, reduction_latex_path)


if __name__ == "__main__":
    generate_summary()
