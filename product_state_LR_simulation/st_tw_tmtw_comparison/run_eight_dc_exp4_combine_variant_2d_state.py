"""Run eight-DC baseline variants with two-dimensional per-DC state.

The state is s=(q,o), where q is the queue state and o is the operation type:
  o=0: batch-flexible training jobs
  o=1: interactive-heavy inference jobs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_EXP_DIR = REPO_ROOT / "baseline_setting_suite_v1" / "experiments"
sys.path.insert(0, str(BASELINE_EXP_DIR))

import run_experiments as baseline_exp  # noqa: E402
from run_experiments import (  # noqa: E402
    LearningPolicy,
    RMABInstance,
    all_arm_indices,
    corrupt_states,
    row_normalize,
    step_environment,
    write_group_summary,
)


SOURCE_RESULTS = REPO_ROOT / "eight_dc_exp4_results" / "local_global_tw_oracle_tw_n40_summary.csv"
DATACENTER_DIR = REPO_ROOT / "datasets" / "datacenter_with_metrics"
DEFAULT_OUTPUT = REPO_ROOT / "eight_dc_exp4_combine_variant_2d_state_results"
NO_TMTW_RANK10_OUTPUT = REPO_ROOT / "eight_dc_exp4_combine_variant_2d_state_no_tmtw_rank10_results"
NO_TMTW_BLOCK_RANK10_OUTPUT = (
    REPO_ROOT / "eight_dc_exp4_combine_variant_2d_state_no_tmtw_block_rank10_results"
)

OP_BATCH_TRAINING = 0
OP_INTERACTIVE_INFERENCE = 1
N_OPS = 2

POLICIES = [
    "state_thompson",
    "local_ucb_tw",
    "global_ucb_tw",
    "exp4",
    "tw",
    "tm_tw",
    "tm_tw_refined",
]

VARIANTS = [
    "dense",
    "gated_offline",
    "gated_offline_low_rank",
    "support_offline",
]

POLICY_DISPLAY = {
    "oracle": "Oracle Whittle",
    "state_thompson": "State Thompson",
    "local_ucb_tw": "Local UCB + TW",
    "global_ucb_tw": "Global UCB + TW",
    "exp4": "EXP4",
    "tw": "TW",
    "tm_tw": "TM-TW",
    "tm_tw_refined": "Adaptive TM-TW",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, default=SOURCE_RESULTS)
    parser.add_argument("--datacenter-dir", type=Path, default=DATACENTER_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rounds", type=int, default=1500)
    parser.add_argument("--queue-states", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--lambda-lmp", type=float, default=1.0)
    parser.add_argument("--batch-epsilon", type=float, default=0.9)
    parser.add_argument("--inference-epsilon", type=float, default=0.5)
    parser.add_argument("--batch-delay-weight", type=float, default=0.04)
    parser.add_argument("--inference-delay-weight", type=float, default=0.35)
    parser.add_argument("--trust-floor", type=float, default=0.10)
    parser.add_argument("--trust-cap", type=float, default=0.95)
    parser.add_argument("--gate-scale-mult", type=float, default=1.0)
    parser.add_argument("--beta-gate-concentration", type=float, default=20.0)
    parser.add_argument("--offline-prior-weight", type=float, default=1.50)
    parser.add_argument("--support-offline-leak", type=float, default=0.0)
    parser.add_argument("--support-trust-cap", type=float, default=None)
    parser.add_argument("--support-local-bonus", type=float, default=0.0)
    parser.add_argument("--low-rank-blend-scale", type=float, default=0.7)
    parser.add_argument("--low-rank-projection-weight", type=float, default=1.0)
    parser.add_argument("--low-rank-local-bonus", type=float, default=0.0)
    parser.add_argument("--disable-model-informed-offline-support", action="store_true")
    parser.add_argument(
        "--offline-support-mode",
        choices=["model", "rules", "model_rules_union", "model_rules_intersection", "true", "dense"],
        default="model",
        help="Offline support source: model assignments, hand-coded rules, hybrids, true support, or dense no-domain support.",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated source seeds to run, e.g. 42.",
    )
    parser.add_argument("--exclude-tm-tw-family", action="store_true")
    parser.add_argument("--exclude-plain-tm-tw", action="store_true")
    parser.add_argument("--include-support-gated-low-rank", action="store_true")
    parser.add_argument(
        "--policy-labels",
        default="",
        help="Optional comma-separated policy labels to run, e.g. oracle,tm_tw_refined_dense.",
    )
    parser.add_argument("--low-rank-max-rank", type=int, default=3)
    parser.add_argument(
        "--block-low-rank",
        action="store_true",
        help="Apply low-rank smoothing within operation/action queue-transition blocks.",
    )
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def encode_state(q: int, o: int) -> int:
    return q * N_OPS + o


def decode_state(state: int) -> tuple[int, int]:
    return state // N_OPS, state % N_OPS


def stable_policy_seed(seed: int, policy: str, variant: str) -> int:
    key = f"{policy}:{variant}:2d_state"
    return seed + sum(ord(ch) for ch in key)


def policy_variant_grid(
    exclude_tm_tw_family: bool = False,
    exclude_plain_tm_tw: bool = False,
    include_support_gated_low_rank: bool = False,
) -> list[tuple[str, str]]:
    grid: list[tuple[str, str]] = [("oracle", "dense")]
    variants = VARIANTS + (["support_gated_offline_low_rank"] if include_support_gated_low_rank else [])
    for policy in POLICIES:
        if exclude_tm_tw_family and policy in {"tm_tw", "tm_tw_refined"}:
            continue
        if exclude_plain_tm_tw and policy == "tm_tw":
            continue
        if policy in {"state_thompson", "local_ucb_tw", "global_ucb_tw", "exp4"}:
            grid.append((policy, "dense"))
        else:
            grid.extend((policy, variant) for variant in variants)
    return grid


def configure_low_rank_cap(max_rank: int, block_low_rank: bool = False) -> None:
    def low_rank_transition_with_cap(p: np.ndarray, rank: int) -> np.ndarray:
        effective_rank = min(max(1, max_rank), p.shape[-1], p.shape[-2])
        smoothed = np.zeros_like(p)
        for arm in range(p.shape[0]):
            for action in range(p.shape[1]):
                u, s, vt = np.linalg.svd(p[arm, action], full_matrices=False)
                r = min(effective_rank, s.size)
                approx = (u[:, :r] * s[:r]) @ vt[:r, :]
                smoothed[arm, action] = np.maximum(approx, 0.0)
        return row_normalize(smoothed)

    def block_low_rank_transition(p: np.ndarray, rank: int) -> np.ndarray:
        effective_rank = min(max(1, max_rank), p.shape[-1] // N_OPS)
        smoothed = np.zeros_like(p)
        queue_states = p.shape[-1] // N_OPS
        for arm in range(p.shape[0]):
            for action in range(p.shape[1]):
                for source_op in range(N_OPS):
                    source_states = np.arange(source_op, p.shape[-2], N_OPS)
                    for next_op in range(N_OPS):
                        next_states = np.arange(next_op, p.shape[-1], N_OPS)
                        block = p[arm, action][np.ix_(source_states, next_states)]
                        u, s, vt = np.linalg.svd(block, full_matrices=False)
                        r = min(effective_rank, s.size, queue_states)
                        approx = (u[:, :r] * s[:r]) @ vt[:r, :]
                        smoothed[arm, action][np.ix_(source_states, next_states)] = np.maximum(approx, 0.0)
        return row_normalize(smoothed)

    baseline_exp.low_rank_transition = block_low_rank_transition if block_low_rank else low_rank_transition_with_cap


def policy_label(policy: str, variant: str) -> str:
    if policy in {"oracle", "state_thompson"}:
        return policy
    return f"{policy}_{variant}"


def parse_label_filter(labels: str) -> set[str] | None:
    parsed = {label.strip() for label in labels.split(",") if label.strip()}
    return parsed or None


def load_source_settings(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    required = {"seed", "n_dc", "n_jobs", "budget", "datacenter_ids"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return (
        source[["seed", "n_dc", "n_jobs", "budget", "datacenter_ids"]]
        .drop_duplicates()
        .sort_values(["seed", "n_dc"])
        .reset_index(drop=True)
    )


def load_datacenter_dfs(datacenter_dir: Path, datacenter_ids: str) -> list[pd.DataFrame]:
    dfs: list[pd.DataFrame] = []
    for dc_id in str(datacenter_ids).split("|"):
        path = datacenter_dir / f"datacenter_{int(dc_id)}_with_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing datacenter file: {path}")
        dfs.append(pd.read_csv(path))
    return dfs


def split_model_value(value: object) -> list[str]:
    models = [part.strip() for part in str(value).split("|") if part.strip()]
    return models or ["unlabeled"]


def datacenter_model_weights(df: pd.DataFrame) -> dict[str, float]:
    if "model" not in df.columns:
        return {"unlabeled": 1.0}
    counts: Counter[str] = Counter()
    for value in df["model"]:
        models = split_model_value(value)
        weight = 1.0 / len(models)
        for model in models:
            counts[model] += weight
    total = sum(counts.values())
    if total <= 0:
        return {"unlabeled": 1.0}
    return {model: count / total for model, count in counts.items()}


def build_model_informed_offline_support(
    support_mask: np.ndarray,
    datacenter_dfs: list[pd.DataFrame],
    min_model_weight: float = 0.05,
) -> tuple[np.ndarray, list[dict]]:
    model_weights = [datacenter_model_weights(df) for df in datacenter_dfs]
    model_support: dict[str, np.ndarray] = {}
    for arm, weights in enumerate(model_weights):
        for model, weight in weights.items():
            if weight < min_model_weight:
                continue
            if model not in model_support:
                model_support[model] = np.zeros_like(support_mask[arm], dtype=bool)
            model_support[model] |= support_mask[arm]

    offline_support = np.zeros_like(support_mask, dtype=bool)
    metadata_rows: list[dict] = []
    for arm, weights in enumerate(model_weights):
        active_models = [model for model, weight in weights.items() if weight >= min_model_weight]
        if not active_models:
            active_models = [max(weights, key=weights.get)]
        for model in active_models:
            offline_support[arm] |= model_support.get(model, support_mask[arm])
        if not offline_support[arm].any():
            offline_support[arm] = support_mask[arm]
        metadata_rows.append(
            {
                "arm": arm,
                "model_weights": "|".join(f"{model}:{weights[model]:.3f}" for model in sorted(weights)),
                "active_models": "|".join(active_models),
                "offline_support_edges": int(offline_support[arm].sum()),
            }
        )
    return offline_support, metadata_rows


def build_rule_based_offline_support(
    support_mask: np.ndarray,
    queue_states: int,
) -> tuple[np.ndarray, list[dict]]:
    """Build support from workload-specific offline rules.

    The rule graph encodes: delay-tolerant training remains queued under active
    actions and continues under passive actions; latency-sensitive inference is
    rarely deferred and moves toward completion or the next batch; operation
    type is persistent except at low-pressure/completion boundaries.
    """
    offline_support = np.zeros_like(support_mask, dtype=bool)
    n_arms = support_mask.shape[0]
    for arm in range(n_arms):
        for state in range(queue_states * N_OPS):
            q, operation = decode_state(state)
            pressure_boundary = q <= 1
            for action in [0, 1]:
                for next_state in np.flatnonzero(support_mask[arm, action, state]):
                    next_q, next_operation = decode_state(int(next_state))
                    same_operation = next_operation == operation
                    switches_at_boundary = (not same_operation) and pressure_boundary
                    persistent = same_operation or switches_at_boundary

                    if operation == OP_BATCH_TRAINING:
                        if action == 1:
                            # Active training is delay-tolerant: allow queued
                            # persistence instead of broad completion jumps.
                            allowed_queue = next_q in {q, min(queue_states - 1, q + 1)}
                        else:
                            # Passive training continues execution or holds.
                            allowed_queue = next_q in {max(0, q - 1), q}
                    else:
                        # Inference is latency-sensitive: it should not be
                        # deferred upward; completion/subsequent batches move
                        # downward or hold briefly.
                        allowed_queue = next_q <= q

                    if persistent and allowed_queue:
                        offline_support[arm, action, state, next_state] = True

                if not offline_support[arm, action, state].any():
                    offline_support[arm, action, state] = support_mask[arm, action, state]

    metadata_rows = [
        {
            "arm": arm,
            "model_weights": "rule_based",
            "active_models": "training_delay_tolerant|inference_latency_sensitive|workload_persistence",
            "offline_support_edges": int(offline_support[arm].sum()),
        }
        for arm in range(n_arms)
    ]
    return offline_support, metadata_rows


def normalize_contexts(contexts: np.ndarray) -> np.ndarray:
    flat = contexts.reshape(-1, contexts.shape[-1])
    lo = flat.min(axis=0)
    hi = flat.max(axis=0)
    scale = np.where(hi > lo, hi - lo, 1.0)
    return (contexts - lo) / scale


def filtered_jobs(df: pd.DataFrame) -> pd.DataFrame:
    utilization = pd.to_numeric(df["avgcpu"], errors="coerce") / 100.0
    core_hours = pd.to_numeric(df["corehour"], errors="coerce")
    keep = (core_hours >= 1.0) & (utilization >= 0.10)
    return df[keep].copy().reset_index(drop=True)


def sample_operation_jobs(
    df: pd.DataFrame,
    rng: np.random.Generator,
    operation: int,
    n_jobs: int,
) -> pd.DataFrame:
    jobs = filtered_jobs(df)
    if operation == OP_BATCH_TRAINING:
        pool = jobs[jobs["vmcategory"] == "Delay-insensitive"]
    else:
        pool = jobs[jobs["vmcategory"] == "Interactive"]
    if pool.empty:
        pool = jobs
    indices = rng.choice(len(pool), size=n_jobs, replace=(len(pool) < n_jobs))
    return pool.iloc[indices].reset_index(drop=True)


def build_operation_rewards_and_contexts(
    df: pd.DataFrame,
    rng: np.random.Generator,
    queue_states: int,
    batch_size: int,
    lambda_lmp: float,
    batch_epsilon: float,
    inference_epsilon: float,
    batch_delay_weight: float,
    inference_delay_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_states = queue_states * N_OPS
    rewards = np.zeros(n_states)
    contexts = np.zeros((n_states, 4))
    n_jobs = queue_states * batch_size

    for operation in [OP_BATCH_TRAINING, OP_INTERACTIVE_INFERENCE]:
        jobs = sample_operation_jobs(df, rng, operation, n_jobs)
        power = pd.to_numeric(jobs["power_saving_index"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        core = pd.to_numeric(jobs["corehour"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        avgcpu = pd.to_numeric(jobs["avgcpu"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / 100.0
        qos = pd.to_numeric(jobs["qos_cost"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        for q in range(queue_states):
            start = q * batch_size
            batch_idx = np.arange(start, start + batch_size) % n_jobs
            window_idx = np.arange(start, min(start + 2 * batch_size, n_jobs)) % n_jobs
            chosen = window_idx[np.argsort(power[window_idx])[:batch_size]]
            deferred = np.setdiff1d(batch_idx, chosen, assume_unique=False)

            default_power = power[batch_idx].sum()
            chosen_power = power[chosen].sum()
            queue_pressure = q / max(1, queue_states - 1)
            if operation == OP_BATCH_TRAINING:
                epsilon = batch_epsilon
                delay_weight = batch_delay_weight
                delay_cost = queue_pressure * core[deferred].sum()
            else:
                epsilon = inference_epsilon
                delay_weight = inference_delay_weight
                delay_cost = (1.0 + queue_pressure) * qos[deferred].sum()
            expected_chosen_power = epsilon * chosen_power + (1.0 - epsilon) * default_power

            state = encode_state(q, operation)
            rewards[state] = max(
                0.0,
                lambda_lmp * (default_power - expected_chosen_power)
                - delay_weight * delay_cost,
            )
            contexts[state, 0] = power[batch_idx].mean()
            contexts[state, 1] = core[batch_idx].mean()
            contexts[state, 2] = 0.15 if operation == OP_BATCH_TRAINING else 0.85
            contexts[state, 3] = queue_pressure
    return rewards, contexts


def add_transition_edge(
    p: np.ndarray,
    mask: np.ndarray,
    arm: int,
    action: int,
    state: int,
    next_q: int,
    next_o: int,
    weight: float,
) -> None:
    next_state = encode_state(next_q, next_o)
    p[arm, action, state, next_state] += weight
    mask[arm, action, state, next_state] = True


def make_two_dimensional_instance(
    seed: int,
    datacenter_dfs: list[pd.DataFrame],
    queue_states: int,
    batch_size: int,
    lambda_lmp: float,
    batch_epsilon: float,
    inference_epsilon: float,
    batch_delay_weight: float,
    inference_delay_weight: float,
    offline_support_mode: str = "model",
) -> RMABInstance:
    rng = np.random.default_rng(seed)
    n_arms = len(datacenter_dfs)
    n_states = queue_states * N_OPS
    rewards = np.zeros((n_arms, n_states))
    contexts = np.zeros((n_arms, n_states, 4))
    transitions = np.zeros((n_arms, 2, n_states, n_states))
    support_mask = np.zeros((n_arms, 2, n_states, n_states), dtype=bool)

    for arm, df in enumerate(datacenter_dfs):
        rewards[arm], contexts[arm] = build_operation_rewards_and_contexts(
            df,
            rng,
            queue_states,
            batch_size,
            lambda_lmp,
            batch_epsilon,
            inference_epsilon,
            batch_delay_weight,
            inference_delay_weight,
        )
        arm_shift = rng.uniform(-0.04, 0.04)
        for state in range(n_states):
            q, operation = decode_state(state)
            pressure = q / max(1, queue_states - 1)

            # Passive: queues tend to grow, and operation type is sticky.
            passive_up = np.clip(0.46 + 0.20 * pressure + arm_shift, 0.20, 0.78)
            passive_stay = 0.34
            passive_down = max(0.0, 1.0 - passive_up - passive_stay)
            passive_switch = 0.07 if operation == OP_BATCH_TRAINING else 0.05

            for next_q, weight in [
                (min(queue_states - 1, q + 1), passive_up),
                (q, passive_stay),
                (max(0, q - 1), passive_down),
            ]:
                add_transition_edge(
                    transitions,
                    support_mask,
                    arm,
                    0,
                    state,
                    next_q,
                    operation,
                    weight * (1.0 - passive_switch),
                )
                add_transition_edge(
                    transitions,
                    support_mask,
                    arm,
                    0,
                    state,
                    next_q,
                    1 - operation,
                    weight * passive_switch,
                )

            # Active: batch training can be deferred broadly; inference gets
            # immediate queue relief but is less tolerant to operation switching.
            if operation == OP_BATCH_TRAINING:
                active_candidates = [
                    (max(0, q - 2), 0.44),
                    (max(0, q - 1), 0.34),
                    (q, 0.14),
                    (min(queue_states - 1, q + 1), 0.08),
                ]
                active_switch = 0.12
            else:
                active_candidates = [
                    (max(0, q - 1), 0.70),
                    (q, 0.24),
                    (min(queue_states - 1, q + 1), 0.06),
                ]
                active_switch = 0.03
            for next_q, weight in active_candidates:
                add_transition_edge(
                    transitions,
                    support_mask,
                    arm,
                    1,
                    state,
                    next_q,
                    operation,
                    weight * (1.0 - active_switch),
                )
                add_transition_edge(
                    transitions,
                    support_mask,
                    arm,
                    1,
                    state,
                    next_q,
                    1 - operation,
                    weight * active_switch,
                )

    transitions = row_normalize(transitions)
    contexts = normalize_contexts(contexts)
    if offline_support_mode == "model":
        offline_support_mask, offline_support_metadata = build_model_informed_offline_support(
            support_mask,
            datacenter_dfs,
        )
        offline_support_source = "model_informed_from_datacenter_model_assignment"
    elif offline_support_mode == "rules":
        offline_support_mask, offline_support_metadata = build_rule_based_offline_support(
            support_mask,
            queue_states,
        )
        offline_support_source = "rule_based_workload_domain_knowledge"
    elif offline_support_mode in {"model_rules_union", "model_rules_intersection"}:
        model_mask, model_metadata = build_model_informed_offline_support(support_mask, datacenter_dfs)
        rule_mask, rule_metadata = build_rule_based_offline_support(support_mask, queue_states)
        if offline_support_mode == "model_rules_union":
            offline_support_mask = model_mask | rule_mask
            offline_support_source = "model_informed_union_rule_based_support"
        else:
            offline_support_mask = model_mask & rule_mask
            for arm in range(n_arms):
                empty_rows = ~offline_support_mask[arm].any(axis=-1)
                offline_support_mask[arm][empty_rows] = rule_mask[arm][empty_rows]
            offline_support_source = "model_informed_intersection_rule_based_support"
        offline_support_metadata = []
        rule_by_arm = {int(row["arm"]): row for row in rule_metadata}
        for row in model_metadata:
            arm = int(row["arm"])
            offline_support_metadata.append(
                {
                    **row,
                    "rule_active_models": rule_by_arm[arm]["active_models"],
                    "rule_support_edges": rule_by_arm[arm]["offline_support_edges"],
                    "offline_support_edges": int(offline_support_mask[arm].sum()),
                }
            )
    else:
        if offline_support_mode == "dense":
            offline_support_mask = np.ones_like(support_mask, dtype=bool)
            offline_support_source = "dense_no_domain_support"
        else:
            offline_support_mask = support_mask.copy()
            offline_support_source = "noncontextual_true_support"
        offline_support_metadata = [
            {
                "arm": arm,
                "model_weights": "none",
                "active_models": "none",
                "offline_support_edges": int(offline_support_mask[arm].sum()),
            }
            for arm in range(n_arms)
        ]
    initial_states = np.asarray(
        [encode_state(int(rng.integers(0, queue_states)), int(rng.integers(0, N_OPS))) for _ in range(n_arms)]
    )
    return RMABInstance(
        rewards=rewards,
        passive_p=transitions[:, 0],
        active_p=transitions[:, 1],
        contexts=contexts,
        support_mask=support_mask,
        offline_support_mask=offline_support_mask,
        initial_states=initial_states,
        description={
            "seed": seed,
            "n_arms": n_arms,
            "n_states": n_states,
            "queue_states": queue_states,
            "operation_states": N_OPS,
            "batch_size": batch_size,
            "sparsity": int(np.ceil(support_mask.sum(axis=-1).mean())),
            "offline_support_sparsity": int(np.ceil(offline_support_mask.sum(axis=-1).mean())),
            "offline_support_source": offline_support_source,
            "offline_support_metadata": offline_support_metadata,
            "top_gap_lambda": 0.0,
            "transition_dominance": 0.0,
            "state_space": "two-dimensional queue_operation",
            "operation_type_0": "batch-flexible training",
            "operation_type_1": "interactive-heavy inference",
            "reward_function": (
                "max(0, lambda_lmp * (P_default - P_hat_chosen) - lambda_delay_o * c_delay); "
                "P_hat_chosen = epsilon_o * P_chosen + (1 - epsilon_o) * P_default"
            ),
            "lambda_lmp": lambda_lmp,
            "batch_epsilon": batch_epsilon,
            "inference_epsilon": inference_epsilon,
            "batch_delay_weight": batch_delay_weight,
            "inference_delay_weight": inference_delay_weight,
        },
    )


def run_single_policy_same_path_oracle(
    instance: RMABInstance,
    policy_name: str,
    seed: int,
    rounds: int,
    budget: int,
    transition_variant: str,
    gate_mode: str,
    gate_scale_mult: float,
    beta_gate_concentration: float,
    trust_floor: float,
    trust_cap: float,
    offline_prior_weight: float,
    support_offline_leak: float,
    support_trust_cap: float | None,
    support_local_bonus: float,
    low_rank_blend_scale: float,
    low_rank_projection_weight: float,
    low_rank_local_bonus: float,
    oracle_cum_override: float | None = None,
) -> tuple[dict, np.ndarray]:
    """Evaluate a policy, optionally normalized by a fixed oracle rollout."""
    rng = np.random.default_rng(seed)
    n_arms, n_states = instance.rewards.shape
    true_indices = all_arm_indices(instance.rewards, instance.active_p, instance.passive_p)
    policy = LearningPolicy(
        instance,
        policy_name,
        rng,
        transition_variant=transition_variant,
        gate_mode=gate_mode,
        gate_scale_mult=gate_scale_mult,
        beta_gate_concentration=beta_gate_concentration,
        trust_floor=trust_floor,
        trust_cap=trust_cap,
        offline_prior_weight=offline_prior_weight,
        support_offline_leak=support_offline_leak,
        support_trust_cap=support_trust_cap,
        support_local_bonus=support_local_bonus,
        low_rank_blend_scale=low_rank_blend_scale,
        low_rank_projection_weight=low_rank_projection_weight,
        low_rank_local_bonus=low_rank_local_bonus,
    )

    states = instance.initial_states.copy()
    rewards_by_round = np.zeros(rounds)
    top1_agree = 0
    top2_agree = 0
    start = time.perf_counter()

    for t in range(rounds):
        observed = corrupt_states(rng, states, n_states, 0.0)
        actions = policy.select(observed, t, budget, true_indices=true_indices)
        next_states, rewards = step_environment(rng, instance, states, actions)
        next_observed = corrupt_states(rng, next_states, n_states, 0.0)
        policy.update(observed, actions, next_observed, rewards)

        rewards_by_round[t] = rewards[actions].sum()
        oracle_rank = np.argsort(true_indices[np.arange(n_arms), states])[::-1]
        if int(actions[0]) == int(oracle_rank[0]):
            top1_agree += 1
        if int(actions[0]) in set(int(x) for x in oracle_rank[:2]):
            top2_agree += 1
        states = next_states

    runtime = time.perf_counter() - start
    cum_reward = float(rewards_by_round.sum())
    oracle_cum = float(cum_reward if oracle_cum_override is None else oracle_cum_override)
    feasible_l1, off_support = policy.transition_error()
    summary = {
        "policy": policy_name,
        "seed": seed,
        "rounds": rounds,
        "S": n_states,
        "k": instance.description["sparsity"],
        "top_gap_lambda": instance.description["top_gap_lambda"],
        "transition_dominance": instance.description["transition_dominance"],
        "context_noise_level": 0.0,
        "transition_variant": transition_variant,
        "trust_scale_mult": 1.0,
        "gate_scale_mult": gate_scale_mult,
        "gate_mode": gate_mode,
        "beta_gate_concentration": beta_gate_concentration,
        "trust_floor": trust_floor,
        "trust_cap": trust_cap,
        "offline_prior_weight": offline_prior_weight,
        "support_offline_leak": support_offline_leak,
        "support_trust_cap": support_trust_cap,
        "support_local_bonus": support_local_bonus,
        "low_rank_blend_scale": low_rank_blend_scale,
        "low_rank_projection_weight": low_rank_projection_weight,
        "low_rank_local_bonus": low_rank_local_bonus,
        "support_mask_enabled": "support" in transition_variant,
        "avg_reward": float(np.mean(rewards_by_round)),
        "cum_reward": cum_reward,
        "oracle_cum_reward": oracle_cum,
        "reward_pct_oracle": 100.0 * cum_reward / max(oracle_cum, 1e-9),
        "cum_regret": oracle_cum - cum_reward,
        "top1_agreement": top1_agree / rounds,
        "top2_agreement": top2_agree / rounds,
        "transition_l1_error": feasible_l1,
        "off_support_leakage": off_support,
        "runtime_seconds": runtime,
        "oracle_comparison": "fixed_whittle_oracle_rollout",
    }
    return summary, rewards_by_round


def main() -> None:
    args = parse_args()
    if args.exclude_tm_tw_family and args.block_low_rank and args.output == DEFAULT_OUTPUT:
        args.output = NO_TMTW_BLOCK_RANK10_OUTPUT
    elif args.exclude_tm_tw_family and args.output == DEFAULT_OUTPUT:
        args.output = NO_TMTW_RANK10_OUTPUT
    configure_low_rank_cap(args.low_rank_max_rank, args.block_low_rank)
    args.output.mkdir(parents=True, exist_ok=True)
    settings = load_source_settings(args.source_results)
    if args.seeds:
        seed_filter = set(parse_int_list(args.seeds))
        settings = settings[settings["seed"].isin(seed_filter)].reset_index(drop=True)
        if settings.empty:
            raise ValueError(f"No source settings matched --seeds={args.seeds!r}")
    result_rows: list[dict] = []
    round_rows: list[dict] = []
    offline_support_rows: list[dict] = []
    config_rows = settings.to_dict(orient="records")
    label_filter = parse_label_filter(args.policy_labels)

    for setting in config_rows:
        seed = int(setting["seed"])
        n_dc = int(setting["n_dc"])
        budget = int(setting["budget"])
        datacenter_ids = str(setting["datacenter_ids"])
        datacenter_dfs = load_datacenter_dfs(args.datacenter_dir, datacenter_ids)
        instance = make_two_dimensional_instance(
            seed=seed + 31 * args.queue_states,
            datacenter_dfs=datacenter_dfs,
            queue_states=args.queue_states,
            batch_size=args.batch_size,
            lambda_lmp=args.lambda_lmp,
            batch_epsilon=args.batch_epsilon,
            inference_epsilon=args.inference_epsilon,
            batch_delay_weight=args.batch_delay_weight,
            inference_delay_weight=args.inference_delay_weight,
            offline_support_mode=(
                "true" if args.disable_model_informed_offline_support else args.offline_support_mode
            ),
        )
        for metadata in instance.description.get("offline_support_metadata", []):
            offline_support_rows.append(
                {
                    "seed": seed,
                    "n_dc": n_dc,
                    "budget": budget,
                    "datacenter_ids": datacenter_ids,
                    **metadata,
                }
            )

        oracle_cum_by_setting: float | None = None
        grid = policy_variant_grid(
            args.exclude_tm_tw_family,
            args.exclude_plain_tm_tw,
            args.include_support_gated_low_rank,
        )
        if label_filter is not None:
            grid = [(policy, variant) for policy, variant in grid if policy_label(policy, variant) in label_filter]
            if not grid or grid[0][0] != "oracle":
                raise ValueError("--policy-labels must include oracle so the fixed denominator is available")

        for policy, variant in grid:
            run_seed = stable_policy_seed(seed, policy, variant)
            summary, rewards = run_single_policy_same_path_oracle(
                instance=instance,
                policy_name=policy,
                seed=run_seed,
                rounds=args.rounds,
                budget=budget,
                transition_variant=variant,
                gate_mode="deterministic",
                gate_scale_mult=args.gate_scale_mult,
                beta_gate_concentration=args.beta_gate_concentration,
                trust_floor=args.trust_floor,
                trust_cap=args.trust_cap,
                offline_prior_weight=args.offline_prior_weight,
                support_offline_leak=args.support_offline_leak,
                support_trust_cap=args.support_trust_cap,
                support_local_bonus=args.support_local_bonus,
                low_rank_blend_scale=args.low_rank_blend_scale,
                low_rank_projection_weight=args.low_rank_projection_weight,
                low_rank_local_bonus=args.low_rank_local_bonus,
                oracle_cum_override=oracle_cum_by_setting,
            )
            if policy == "oracle":
                oracle_cum_by_setting = float(summary["cum_reward"])
                summary["oracle_cum_reward"] = oracle_cum_by_setting
                summary["reward_pct_oracle"] = 100.0
                summary["cum_regret"] = 0.0
            label = policy_label(policy, variant)
            run_seed_recorded = int(summary["seed"])
            summary.update(
                {
                    "source_results": str(args.source_results),
                    "source_seed": seed,
                    "run_seed": run_seed_recorded,
                    "seed": seed,
                    "n_dc": n_dc,
                    "n_jobs": int(setting["n_jobs"]),
                    "budget": budget,
                    "datacenter_ids": datacenter_ids,
                    "queue_states": args.queue_states,
                    "operation_states": N_OPS,
                    "offline_support_sparsity": instance.description["offline_support_sparsity"],
                    "offline_support_source": instance.description["offline_support_source"],
                    "state_definition": "s=(q,o)",
                    "operation_type_0": "batch-flexible training",
                    "operation_type_1": "interactive-heavy inference",
                    "reward_function": (
                        "max(0, lambda_lmp * (P_default - P_hat_chosen) - lambda_delay_o * c_delay); "
                        "P_hat_chosen = epsilon_o * P_chosen + (1 - epsilon_o) * P_default"
                    ),
                    "lambda_lmp": args.lambda_lmp,
                    "batch_epsilon": args.batch_epsilon,
                    "inference_epsilon": args.inference_epsilon,
                    "batch_delay_weight": args.batch_delay_weight,
                    "inference_delay_weight": args.inference_delay_weight,
                    "policy_label": label,
                    "policy_display": POLICY_DISPLAY[policy],
                    "experiment_id": "eight_dc_exp4_combine_variant_2d_state",
                }
            )
            result_rows.append(summary)

            running_avg = np.cumsum(rewards) / np.arange(1, args.rounds + 1)
            for round_idx, (reward, avg_reward) in enumerate(zip(rewards, running_avg), start=1):
                round_rows.append(
                    {
                        "seed": seed,
                        "n_dc": n_dc,
                        "budget": budget,
                        "datacenter_ids": datacenter_ids,
                        "queue_states": args.queue_states,
                        "operation_states": N_OPS,
                        "policy": policy,
                        "transition_variant": variant,
                        "policy_label": label,
                        "round": round_idx,
                        "reward": float(reward),
                        "running_avg_reward": float(avg_reward),
                    }
                )

            if args.progress:
                print(
                    f"finished seed={seed} n_dc={n_dc} budget={budget} "
                    f"policy={label} reward_pct={summary['reward_pct_oracle']:.2f}",
                    flush=True,
                )

    results_path = args.output / "combine_variant_2d_state_results.csv"
    rounds_path = args.output / "combine_variant_2d_state_round_rewards.csv"
    summary_path = args.output / "combine_variant_2d_state_summary.csv"
    offline_support_path = args.output / "model_informed_offline_support.csv"
    config_path = args.output / "run_config.json"

    pd.DataFrame(result_rows).to_csv(results_path, index=False)
    pd.DataFrame(round_rows).to_csv(rounds_path, index=False)
    pd.DataFrame(offline_support_rows).to_csv(offline_support_path, index=False)
    write_group_summary(
        result_rows,
        [
            "n_dc",
            "budget",
            "S",
            "queue_states",
            "operation_states",
            "policy_label",
            "transition_variant",
            "gate_mode",
            "trust_floor",
        ],
        summary_path,
    )
    config_path.write_text(
        json.dumps(
            {
                "source_results": str(args.source_results),
                "datacenter_dir": str(args.datacenter_dir),
                "output": str(args.output),
                "rounds": args.rounds,
                "queue_states": args.queue_states,
                "operation_states": N_OPS,
                "total_states_per_datacenter": args.queue_states * N_OPS,
                "offline_support_source": (
                    "noncontextual_true_support"
                    if args.disable_model_informed_offline_support
                    else (
                        "model_informed_from_datacenter_model_assignment"
                        if args.offline_support_mode == "model"
                        else "rule_based_workload_domain_knowledge"
                        if args.offline_support_mode == "rules"
                        else "model_informed_union_rule_based_support"
                        if args.offline_support_mode == "model_rules_union"
                        else "model_informed_intersection_rule_based_support"
                        if args.offline_support_mode == "model_rules_intersection"
                        else "dense_no_domain_support"
                        if args.offline_support_mode == "dense"
                        else "noncontextual_true_support"
                    )
                ),
                "offline_support_rules": [
                    "Training batches are delay-tolerant and more likely to remain queued under active actions and continue execution under passive actions.",
                    "Inference batches are latency-sensitive and unlikely to be deferred, while rapidly transitioning to completion or subsequent batches.",
                    "Consecutive batches exhibit workload persistence.",
                    "Only transitions consistent with the rules are admitted in M_i,a,s,s'.",
                ]
                if args.offline_support_mode == "rules" and not args.disable_model_informed_offline_support
                else [],
                "offline_support_min_model_weight": 0.05,
                "state_definition": "s=(q,o)",
                "operation_type_0": "batch-flexible training",
                "operation_type_1": "interactive-heavy inference",
                "reward_function": (
                    "max(0, lambda_lmp * (P_default - P_hat_chosen) - lambda_delay_o * c_delay); "
                    "P_hat_chosen = epsilon_o * P_chosen + (1 - epsilon_o) * P_default"
                ),
                "lambda_lmp": args.lambda_lmp,
                "batch_epsilon": args.batch_epsilon,
                "inference_epsilon": args.inference_epsilon,
                "batch_delay_weight": args.batch_delay_weight,
                "inference_delay_weight": args.inference_delay_weight,
                "policies": [
                    policy
                    for policy in POLICIES
                    if not (args.exclude_tm_tw_family and policy in {"tm_tw", "tm_tw_refined"})
                    and not (args.exclude_plain_tm_tw and policy == "tm_tw")
                ],
                "variants": VARIANTS
                + (["support_gated_offline_low_rank"] if args.include_support_gated_low_rank else []),
                "exclude_tm_tw_family": args.exclude_tm_tw_family,
                "exclude_plain_tm_tw": args.exclude_plain_tm_tw,
                "include_support_gated_low_rank": args.include_support_gated_low_rank,
                "policy_labels": sorted(label_filter) if label_filter is not None else None,
                "gate_scale_mult": args.gate_scale_mult,
                "offline_prior_weight": args.offline_prior_weight,
                "support_offline_leak": args.support_offline_leak,
                "support_trust_cap": args.support_trust_cap,
                "support_local_bonus": args.support_local_bonus,
                "low_rank_blend_scale": args.low_rank_blend_scale,
                "low_rank_projection_weight": args.low_rank_projection_weight,
                "low_rank_local_bonus": args.low_rank_local_bonus,
                "low_rank_max_rank": args.low_rank_max_rank,
                "block_low_rank": args.block_low_rank,
                "oracle_comparison": "fixed_whittle_oracle_rollout",
                "settings": config_rows,
            },
            indent=2,
        )
    )
    print(f"Saved: {results_path}")
    print(f"Saved: {rounds_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {offline_support_path}")
    print(f"Saved: {config_path}")


if __name__ == "__main__":
    main()
