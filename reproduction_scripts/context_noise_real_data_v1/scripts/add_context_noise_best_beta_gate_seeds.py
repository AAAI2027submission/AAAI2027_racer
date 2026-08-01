"""Add more seeds to the preserved context-noise best-beta-gate results.

The original assembler for context_noise_results_with_best_beta_gate.csv was
not preserved. This script extends the saved selected configurations in that
CSV by running additional seed indices with the same policy/hyperparameter
settings, then rewrites the result and summary CSVs in the cleaned bundle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
EXPERIMENT_DIR = REPO_ROOT / "experiments/context_noise_real_data_v1"
RESULTS_PATH = EXPERIMENT_DIR / "context_noise_results_with_best_beta_gate.csv"
SUMMARY_PATH = EXPERIMENT_DIR / "context_noise_summary_with_best_beta_gate.csv"
DATA_DIR = REPO_ROOT / "datasets/datacenter_with_metrics"
RUNNER_DIR = REPO_ROOT / "reproduction_scripts/context_noise_real_data_v1/experiments"
BASE_SEED = 20260425
NEW_SEED_INDICES = range(3, 10)
SUMMARY_GROUP_FIELDS = [
    "S",
    "context_noise_level",
    "trust_scale_mult",
    "gate_scale_mult",
    "gate_mode",
    "trust_floor",
    "policy_label",
]


def load_runner():
    sys.path.insert(0, str(RUNNER_DIR))
    import run_experiments as runner  # noqa: PLC0415

    return runner


def write_summary(results: pd.DataFrame) -> None:
    numeric_fields = [
        "reward_pct_oracle",
        "transition_l1_error",
        "off_support_leakage",
        "top1_agreement",
        "top2_agreement",
        "cum_regret",
    ]
    grouped = (
        results.groupby(SUMMARY_GROUP_FIELDS, as_index=False)
        .agg(
            n=("reward_pct_oracle", "count"),
            **{f"mean_{field}": (field, "mean") for field in numeric_fields},
        )
        .sort_values(SUMMARY_GROUP_FIELDS)
    )
    grouped.to_csv(SUMMARY_PATH, index=False)


def main() -> None:
    runner = load_runner()
    existing = pd.read_csv(RESULTS_PATH)
    selected_configs = (
        existing[
            [
                "S",
                "context_noise_level",
                "policy",
                "transition_variant",
                "trust_scale_mult",
                "gate_scale_mult",
                "gate_mode",
                "beta_gate_concentration",
                "trust_floor",
                "trust_cap",
                "policy_label",
            ]
        ]
        .drop_duplicates()
        .sort_values(["S", "context_noise_level", "policy_label"])
    )
    datacenter_dfs = runner.load_datacenter_dfs(str(DATA_DIR))
    rows: list[dict] = []
    instance_cache = {}
    existing_keys = {
        (
            int(row.S),
            float(row.context_noise_level),
            row.policy_label,
            int(row.seed),
        )
        for row in existing.itertuples(index=False)
    }

    total = len(selected_configs) * len(list(NEW_SEED_INDICES))
    done = 0
    for config in selected_configs.itertuples(index=False):
        state_count = int(config.S)
        noise_level = float(config.context_noise_level)
        policy = str(config.policy)
        variant = str(config.transition_variant)
        policy_label = str(config.policy_label)
        policy_offset = runner.stable_policy_offset(policy, variant)

        for seed_index in NEW_SEED_INDICES:
            run_seed = BASE_SEED + seed_index + policy_offset
            key = (state_count, noise_level, policy_label, run_seed)
            done += 1
            if key in existing_keys:
                continue

            instance_key = (state_count, seed_index)
            if instance_key not in instance_cache:
                instance_cache[instance_key] = runner.make_instance(
                    seed=BASE_SEED + seed_index + 31 * state_count,
                    n_states=state_count,
                    sparsity=2,
                    transition_dominance=0.45,
                    datacenter_dfs=datacenter_dfs,
                )

            row, _ = runner.run_single_policy(
                instance_cache[instance_key],
                policy,
                seed=run_seed,
                rounds=100,
                noise_level=noise_level,
                transition_variant=variant,
                trust_scale_mult=float(config.trust_scale_mult),
                gate_scale_mult=float(config.gate_scale_mult),
                gate_mode=str(config.gate_mode),
                beta_gate_concentration=float(config.beta_gate_concentration),
                trust_floor=float(config.trust_floor),
                trust_cap=float(config.trust_cap),
            )
            row["policy_label"] = policy_label
            row["experiment_id"] = "context_noise_refinement"
            rows.append(row)

            if len(rows) % 50 == 0:
                print(f"generated {len(rows)} new rows ({done}/{total} configs checked)", flush=True)

    if not rows:
        print("No missing rows to add.")
        write_summary(existing)
        return

    updated = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    updated = updated.sort_values(["S", "context_noise_level", "policy_label", "seed"])
    updated.to_csv(RESULTS_PATH, index=False)
    write_summary(updated)
    print(f"Added {len(rows)} rows.")
    print(f"Updated {RESULTS_PATH}")
    print(f"Updated {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
