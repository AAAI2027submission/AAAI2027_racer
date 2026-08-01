"""Compute L1/leakage diagnostics for baseline-setting checkpoints.

Round 100 diagnostics are recomputed because the learning-curve files store
rewards only. Round 1000 diagnostics are read from the existing experiment
result CSVs to avoid rerunning the full horizon.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-yifu")

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_experiments import make_instance, run_single_policy  # noqa: E402
from scripts.plot_baseline_setting_learning_curves import (  # noqa: E402
    BASE_SEED,
    N_ARMS,
    N_STATES,
    OUT_DIR,
    SEEDS,
    SPARSITY,
    STRATEGIES,
    TRANSITION_DOMINANCE,
    stable_strategy_seed,
)


SUPPLEMENT_DIR = ROOT / "docs/research/rmab_vm_outputs/baseline_setting_beta_offline_supplement_v1"
MAIN_RESULTS = OUT_DIR / "context_noise_results.csv"
SUPPLEMENT_RESULTS = SUPPLEMENT_DIR / "context_noise_results.csv"
OUT_SUMMARY = OUT_DIR / "baseline_setting_checkpoint_diagnostics.csv"
OUT_BY_SEED = OUT_DIR / "baseline_setting_checkpoint_diagnostics_by_seed.csv"

DISPLAY_NAMES = {
    "State Thompson (ST)": "ST",
    "TW": "TW",
    "Local UCB + TW": "Local UCB+TW",
    "Global UCB + TW": "Global UCB+TW",
    "EXP4-based": "EXP4",
    "TM-TW": "TM-TW",
    "Adp. TM-TW": "Adaptive TM-TW",
    "Adp. + offline prior": "Offline prior",
    "Adp. + gated prior": "Gated prior",
    "Adp. + gated + LR": "Gated + LR",
    "Adp. + beta gate prior": "Beta-gate prior",
    "Adp. + beta gate + LR": "Beta-gate + LR",
    "Adp. + support/offline": "Support + offline",
}

POLICY_LABELS = {
    "State Thompson (ST)": "state_thompson",
    "TW": "tw_dense",
    "Local UCB + TW": "local_ucb_tw_dense",
    "Global UCB + TW": "global_ucb_tw_dense",
    "EXP4-based": "exp4_dense",
    "TM-TW": "tm_tw_dense",
    "Adp. TM-TW": "tm_tw_refined_dense",
    "Adp. + offline prior": "tm_tw_refined_offline",
    "Adp. + gated prior": "tm_tw_refined_gated_offline",
    "Adp. + gated + LR": "tm_tw_refined_gated_offline_low_rank",
    "Adp. + beta gate prior": "tm_tw_refined_gated_offline",
    "Adp. + beta gate + LR": "tm_tw_refined_gated_offline_low_rank",
    "Adp. + support/offline": "tm_tw_refined_support_offline",
}


def role_for(strategy: dict) -> str:
    if strategy["name"] == "TW":
        return "TW"
    if strategy["group"] == "refined":
        return "Refined TM-TW"
    return "Baselines"


def strategy_rows() -> list[dict]:
    return [strategy for strategy in STRATEGIES if strategy["name"] != "Oracle Whittle"]


def round_100_rows() -> list[dict]:
    rows: list[dict] = []
    for seed_idx in range(SEEDS):
        instance = make_instance(
            seed=BASE_SEED + seed_idx + 31 * N_STATES,
            n_arms=N_ARMS,
            n_states=N_STATES,
            sparsity=SPARSITY,
            transition_dominance=TRANSITION_DOMINANCE,
        )
        for strategy in strategy_rows():
            run_seed = stable_strategy_seed(
                BASE_SEED + seed_idx,
                strategy["policy"],
                strategy["variant"],
                strategy["gate_mode"],
            )
            result, _ = run_single_policy(
                instance,
                strategy["policy"],
                seed=run_seed,
                rounds=100,
                noise_level=0.0,
                transition_variant=strategy["variant"],
                gate_mode=strategy["gate_mode"],
            )
            rows.append(format_row(100, strategy, seed_idx, result))
    return rows


def round_1000_rows() -> list[dict]:
    main = pd.read_csv(MAIN_RESULTS)
    supplement = pd.read_csv(SUPPLEMENT_RESULTS)
    all_results = pd.concat([main, supplement], ignore_index=True)
    rows: list[dict] = []
    for strategy in strategy_rows():
        gate_mode = strategy["gate_mode"]
        policy_label = POLICY_LABELS[strategy["name"]]
        source_rows = all_results[
            (all_results["policy_label"] == policy_label)
            & (all_results["gate_mode"] == gate_mode)
            & (all_results["rounds"] == 1000)
        ].copy()
        source_rows = source_rows.drop_duplicates(["seed", "policy_label", "gate_mode"])
        for seed_idx, (_, result) in enumerate(source_rows.sort_values("seed").iterrows()):
            rows.append(format_row(1000, strategy, seed_idx, result))
    return rows


def format_row(round_value: int, strategy: dict, seed_idx: int, result: pd.Series | dict) -> dict:
    name = strategy["name"]
    return {
        "round": round_value,
        "strategy": name,
        "display_name": DISPLAY_NAMES[name],
        "role": role_for(strategy),
        "policy": strategy["policy"],
        "transition_variant": strategy["variant"],
        "gate_mode": strategy["gate_mode"],
        "seed_index": seed_idx,
        "transition_l1_error": float(result["transition_l1_error"]),
        "off_support_leakage": float(result["off_support_leakage"]),
        "reward_pct_oracle": float(result["reward_pct_oracle"]),
        "top1_agreement": float(result["top1_agreement"]),
        "top2_agreement": float(result["top2_agreement"]),
    }


def main() -> None:
    by_seed = pd.DataFrame(round_100_rows() + round_1000_rows())
    by_seed.to_csv(OUT_BY_SEED, index=False)

    summary = (
        by_seed.groupby(
            ["round", "strategy", "display_name", "role", "policy", "transition_variant", "gate_mode"],
            as_index=False,
        )
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
        .sort_values(["round", "role", "strategy"])
    )
    summary.to_csv(OUT_SUMMARY, index=False)


if __name__ == "__main__":
    main()
