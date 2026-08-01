"""Run standalone RMAB/VM scheduling experiments.

The original VM/RMAB experiment source is not present in this repository.
This module reconstructs the data-generation and scheduling mechanics from
Paper_3.pdf and provides a reproducible scaffold for the proposed new
experiments.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-yifu")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PASSIVE = 0
ACTIVE = 1


@dataclass
class VMJobs:
    power: np.ndarray
    core_hours: np.ndarray
    qos_cost: np.ndarray
    is_interactive: np.ndarray
    utilization: np.ndarray


@dataclass
class RMABInstance:
    rewards: np.ndarray  # arms x states
    passive_p: np.ndarray  # arms x states x states
    active_p: np.ndarray
    contexts: np.ndarray  # arms x states x features
    support_mask: np.ndarray  # arms x actions x states x states
    initial_states: np.ndarray
    description: dict


@dataclass
class RunResult:
    rows: list[dict]
    learning_curves: dict[str, np.ndarray]


DEFAULT_VARIANTS = [
    "dense",
    "threshold",
    "low_rank",
    "offline",
    "gated_offline",
    "gated_offline_low_rank",
    "support",
    "support_threshold",
    "support_low_rank",
    "support_offline",
    "support_gated_offline_low_rank",
]


def piecewise_power_per_core_hour(utilization: np.ndarray) -> np.ndarray:
    """Paper-style piecewise-linear CPU-utilization power model."""
    u = utilization
    return np.where(
        u < 0.4,
        0.35 + 0.85 * u,
        np.where(u < 0.8, 0.55 + 1.05 * u, 0.85 + 1.35 * u),
    )


def sample_vm_jobs(rng: np.random.Generator, n_jobs: int, quality_shift: float = 0.0) -> VMJobs:
    """Generate filtered Azure-VM-like traces from the paper description.

    Paper_3.pdf says the Azure VM traces are filtered at core-hour >= 1 and
    utilization >= 10%, power is a piecewise-linear function of CPU utilization,
    and QoS costs are assigned by workload class.
    """
    power_parts: list[np.ndarray] = []
    core_parts: list[np.ndarray] = []
    qos_parts: list[np.ndarray] = []
    interactive_parts: list[np.ndarray] = []
    util_parts: list[np.ndarray] = []

    while sum(part.size for part in power_parts) < n_jobs:
        batch = max(256, 2 * n_jobs)
        utilization = rng.beta(2.4 + quality_shift, 2.2, size=batch)
        core_hours = rng.lognormal(mean=0.45 + 0.15 * quality_shift, sigma=0.55, size=batch)
        keep = (core_hours >= 1.0) & (utilization >= 0.10)
        utilization = utilization[keep]
        core_hours = core_hours[keep]
        if utilization.size == 0:
            continue

        classes = rng.choice(3, size=utilization.size, p=[0.25, 0.45, 0.30])
        qos_rate = np.zeros(utilization.size)
        qos_rate[classes == 0] = 10.0
        qos_rate[classes == 1] = 0.0
        qos_rate[classes == 2] = rng.uniform(0.0, 10.0, size=np.sum(classes == 2))

        per_core_power = piecewise_power_per_core_hour(utilization)
        power = core_hours * per_core_power * (5.0 + quality_shift)
        qos_cost = core_hours * qos_rate

        power_parts.append(power)
        core_parts.append(core_hours)
        qos_parts.append(qos_cost)
        interactive_parts.append(classes == 0)
        util_parts.append(utilization)

    power = np.concatenate(power_parts)[:n_jobs]
    core_hours = np.concatenate(core_parts)[:n_jobs]
    qos_cost = np.concatenate(qos_parts)[:n_jobs]
    is_interactive = np.concatenate(interactive_parts)[:n_jobs]
    utilization = np.concatenate(util_parts)[:n_jobs]
    return VMJobs(power, core_hours, qos_cost, is_interactive, utilization)


def load_datacenter_dfs(data_dir: str) -> list[pd.DataFrame]:
    """Load pre-processed datacenter CSV files from *data_dir*, sorted by index.

    Expects files named datacenter_*_with_metrics.csv as produced by the
    VM-trace cleaning pipeline.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "datacenter_*_with_metrics.csv")))
    if not paths:
        raise FileNotFoundError(f"No datacenter CSV files found in {data_dir!r}")
    return [pd.read_csv(p) for p in paths]


def load_vm_jobs_from_csv(
    df: pd.DataFrame,
    rng: np.random.Generator,
    n_jobs: int,
) -> VMJobs:
    """Sample VM jobs from a pre-processed datacenter_with_metrics CSV.

    Applies the same quality filter as the synthetic generator
    (corehour >= 1.0 and avgcpu/100 >= 0.10).  Columns used:
      avgcpu      – CPU utilisation in percent [0, 100]
      corehour    – core-hours consumed by the job
      vmcategory  – workload class ("Interactive", "Delay-insensitive", "Unknown")
      qos_cost    – pre-computed QoS cost (equals corehour for Interactive jobs)
    Power is computed from avgcpu via the same piecewise-linear model used by
    the synthetic generator so that reward scales are comparable.
    """
    utilization_raw = df["avgcpu"].values / 100.0
    core_hours_raw = df["corehour"].values
    keep = (core_hours_raw >= 1.0) & (utilization_raw >= 0.10)
    df_filtered = df[keep].reset_index(drop=True)

    n_avail = len(df_filtered)
    if n_avail == 0:
        raise ValueError("No jobs pass the quality filter in the provided DataFrame.")
    idx = rng.choice(n_avail, size=n_jobs, replace=(n_avail < n_jobs))
    df_s = df_filtered.iloc[idx].reset_index(drop=True)

    utilization = df_s["avgcpu"].values / 100.0
    core_hours = df_s["corehour"].values
    is_interactive = (df_s["vmcategory"].values == "Interactive")
    qos_cost = df_s["qos_cost"].values
    power = core_hours * piecewise_power_per_core_hour(utilization) * 5.0

    return VMJobs(power, core_hours, qos_cost, is_interactive, utilization)


def _arm_stats_from_df(df: pd.DataFrame) -> dict:
    """Compute arm-level calibration stats from a real datacenter CSV.

    Uses vmcategory fractions for type_mix and power_saving_index (max=25) for
    capacity_quality and renewable_profile, so the load-balance reward function
    reflects actual workload composition and scheduling headroom per datacenter.
    """
    vc = df["vmcategory"].value_counts(normalize=True)
    interactive_frac = float(vc.get("Interactive", 0.0))
    batch_frac = float(vc.get("Delay-insensitive", 0.0))
    geo_frac = max(0.0, 1.0 - interactive_frac - batch_frac)
    type_mix = np.maximum(np.array([interactive_frac, batch_frac, geo_frac]), 1e-3)
    type_mix = type_mix / type_mix.sum()
    psi_norm = float(np.clip(df["power_saving_index"].mean() / 25.0, 0.0, 1.0))
    return {
        "type_mix": type_mix,
        "capacity_quality": 0.85 + 0.40 * psi_norm,
        "renewable_profile": psi_norm,
    }


def local_transition_matrix(
    rng: np.random.Generator,
    n_states: int,
    sparsity: int,
    active: bool,
    jitter: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a sparse circular-queue transition matrix and support mask."""
    k = max(1, min(sparsity, n_states))
    p = np.zeros((n_states, n_states))
    mask = np.zeros((n_states, n_states), dtype=bool)
    offsets = np.arange(1, k + 1)
    if active:
        weights = np.exp(-0.35 * np.abs(offsets - min(2, k)))
    else:
        weights = np.exp(-0.75 * (offsets - 1))
    weights = weights / weights.sum()

    for s in range(n_states):
        next_states = (s + offsets) % n_states
        mask[s, next_states] = True
        noisy_weights = weights + rng.uniform(0.0, jitter, size=k)
        noisy_weights = noisy_weights / noisy_weights.sum()
        p[s, next_states] = noisy_weights
    return p, mask


def build_arm_from_jobs(
    jobs: VMJobs,
    batch_size: int,
    lookahead_jobs: int,
    delay_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute active rewards and local context vectors for one data center."""
    n_jobs = jobs.power.size
    n_states = n_jobs // batch_size
    rewards = np.zeros(n_states)
    contexts = np.zeros((n_states, 4))
    for s in range(n_states):
        start = s * batch_size
        batch_idx = np.arange(start, start + batch_size) % n_jobs
        window_idx = np.arange(start, start + lookahead_jobs) % n_jobs
        chosen = window_idx[np.argsort(jobs.power[window_idx])[:batch_size]]

        default_power = jobs.power[batch_idx].sum()
        chosen_power = jobs.power[chosen].sum()
        skipped = np.setdiff1d(batch_idx, chosen, assume_unique=False)
        delay_cost = jobs.qos_cost[skipped][jobs.is_interactive[skipped]].sum()
        rewards[s] = max(0.0, default_power - chosen_power - delay_weight * delay_cost)

        contexts[s, 0] = jobs.power[batch_idx].mean()
        contexts[s, 1] = jobs.core_hours[batch_idx].mean()
        contexts[s, 2] = jobs.is_interactive[batch_idx].mean()
        contexts[s, 3] = s / max(1, n_states - 1)

    return rewards, contexts


def normalize_contexts(contexts: np.ndarray) -> np.ndarray:
    flat = contexts.reshape(-1, contexts.shape[-1])
    lo = flat.min(axis=0)
    hi = flat.max(axis=0)
    scale = np.where(hi > lo, hi - lo, 1.0)
    return (contexts - lo) / scale


def make_instance(
    seed: int,
    n_arms: int = 5,
    n_states: int = 8,
    batch_size: int = 5,
    sparsity: int = 2,
    top_gap_lambda: float = 0.0,
    high_value_first_arm: bool = True,
    transition_dominance: float = 0.45,
    datacenter_dfs: list[pd.DataFrame] | None = None,
) -> RMABInstance:
    rng = np.random.default_rng(seed)
    n_jobs = n_states * batch_size
    lookahead_jobs = min(n_jobs, 2 * batch_size)
    delay_weight = 0.08

    rewards = []
    contexts = []
    passive_ps = []
    active_ps = []
    masks = []

    for arm in range(n_arms):
        if datacenter_dfs is not None:
            jobs = load_vm_jobs_from_csv(datacenter_dfs[arm % len(datacenter_dfs)], rng, n_jobs)
        else:
            quality_shift = 0.65 if (arm == 0 and high_value_first_arm) else rng.normal(0.0, 0.12)
            jobs = sample_vm_jobs(rng, n_jobs, quality_shift=quality_shift)
        arm_rewards, arm_contexts = build_arm_from_jobs(jobs, batch_size, lookahead_jobs, delay_weight)
        passive_p, passive_mask = local_transition_matrix(rng, n_states, sparsity, active=False)
        active_p, active_mask = local_transition_matrix(rng, n_states, sparsity + 1, active=True)

        rewards.append(arm_rewards)
        contexts.append(arm_contexts)
        passive_ps.append(passive_p)
        active_ps.append(active_p)
        masks.append(np.stack([passive_mask, active_mask], axis=0))

    rewards_arr = np.asarray(rewards)
    passive_arr = np.asarray(passive_ps)
    active_arr = np.asarray(active_ps)
    support_mask = np.asarray(masks)

    if transition_dominance > 0:
        rewards_arr, passive_arr, active_arr, support_mask = add_gateway_structure(
            rewards_arr,
            passive_arr,
            active_arr,
            support_mask,
            strength=transition_dominance,
        )

    if n_arms >= 2 and top_gap_lambda > 0.0:
        best_arm = int(np.argmax(rewards_arr.mean(axis=1)))
        order = np.argsort(rewards_arr.mean(axis=1))[::-1]
        second_arm = int(order[1] if order[0] == best_arm else order[0])
        rewards_arr[second_arm] = (
            (1.0 - top_gap_lambda) * rewards_arr[second_arm]
            + top_gap_lambda * rewards_arr[best_arm]
        )

    context_arr = normalize_contexts(np.asarray(contexts))
    initial_states = rng.integers(0, n_states, size=n_arms)
    return RMABInstance(
        rewards=rewards_arr,
        passive_p=passive_arr,
        active_p=active_arr,
        contexts=context_arr,
        support_mask=support_mask,
        initial_states=initial_states,
        description={
            "seed": seed,
            "n_arms": n_arms,
            "n_states": n_states,
            "batch_size": batch_size,
            "sparsity": sparsity,
            "top_gap_lambda": top_gap_lambda,
            "transition_dominance": transition_dominance,
        },
    )


def encode_load_balance_state(queue_state: int, grid_state: int, op_state: int, grid_states: int, op_states: int) -> int:
    return (queue_state * grid_states + grid_state) * op_states + op_state


def decode_load_balance_state(state: int, grid_states: int, op_states: int) -> tuple[int, int, int]:
    op_state = state % op_states
    tmp = state // op_states
    grid_state = tmp % grid_states
    queue_state = tmp // grid_states
    return queue_state, grid_state, op_state


def make_load_balance_instance(
    seed: int,
    n_arms: int = 8,
    queue_states: int = 12,
    grid_states: int = 3,
    op_states: int = 3,
    transition_dominance: float = 0.65,
    heterogeneity: float = 0.35,
    datacenter_dfs: list[pd.DataFrame] | None = None,
) -> RMABInstance:
    """Build a larger state-space proxy for load-balanced data-center operations.

    State is a product of queue/backlog bucket, grid/carbon condition, and
    operation type. This is not meant to replace Zixi's model. It is a fast
    stress test for whether the refinement survives the kind of product state
    space induced by temporal rescheduling plus spatial/load-balancing features.
    """
    rng = np.random.default_rng(seed)
    n_states = queue_states * grid_states * op_states
    rewards = np.zeros((n_arms, n_states))
    contexts = np.zeros((n_arms, n_states, 6))
    passive_p = np.zeros((n_arms, n_states, n_states))
    active_p = np.zeros((n_arms, n_states, n_states))
    support_mask = np.zeros((n_arms, 2, n_states, n_states), dtype=bool)

    # Operation types: 0 interactive-heavy, 1 batch/flexible, 2 geo-migratable.
    type_mix = rng.dirichlet([2.0, 2.2, 1.5], size=n_arms)
    renewable_profile = rng.beta(2.0, 2.0, size=n_arms)
    migration_cost = rng.uniform(0.15, 0.55, size=n_arms)
    capacity_quality = rng.uniform(0.85, 1.25, size=n_arms)

    if datacenter_dfs is not None:
        for arm in range(n_arms):
            stats = _arm_stats_from_df(datacenter_dfs[arm % len(datacenter_dfs)])
            type_mix[arm] = stats["type_mix"]
            renewable_profile[arm] = stats["renewable_profile"]
            capacity_quality[arm] = stats["capacity_quality"]

    # Grid/carbon transitions are sticky but can drift into adjacent stress
    # levels. The common case has three states: low, normal, stressed.
    if grid_states == 3:
        grid_base = np.asarray(
            [
                [0.70, 0.25, 0.05],
                [0.16, 0.66, 0.18],
                [0.06, 0.29, 0.65],
            ]
        )
    else:
        grid_base = np.zeros((grid_states, grid_states))
        for g in range(grid_states):
            weights = np.exp(-1.15 * np.abs(np.arange(grid_states) - g))
            weights[g] += 1.0
            grid_base[g] = weights / weights.sum()

    for arm in range(n_arms):
        for state in range(n_states):
            q, g, op = decode_load_balance_state(state, grid_states, op_states)
            backlog = q / max(1, queue_states - 1)
            grid_stress = g / max(1, grid_states - 1)
            interactive_frac = 0.68 if op == 0 else (0.18 if op == 1 else 0.32)
            flexible_frac = 1.0 - interactive_frac
            spatial_frac = 0.10 if op == 0 else (0.28 if op == 1 else 0.72)

            carbon_value = 0.65 + 1.35 * grid_stress - 0.35 * renewable_profile[arm]
            relief_value = capacity_quality[arm] * (0.6 + 1.4 * backlog) * carbon_value
            delay_penalty = (0.75 + 0.90 * backlog) * interactive_frac
            migration_penalty = migration_cost[arm] * spatial_frac * (0.3 + grid_stress)
            type_bonus = 0.15 * flexible_frac + 0.12 * spatial_frac
            rewards[arm, state] = max(0.0, 8.0 * relief_value + 2.0 * type_bonus - 3.2 * delay_penalty - migration_penalty)

            contexts[arm, state, 0] = relief_value
            contexts[arm, state, 1] = capacity_quality[arm] * (0.8 + backlog)
            contexts[arm, state, 2] = interactive_frac
            contexts[arm, state, 3] = backlog
            contexts[arm, state, 4] = grid_stress
            contexts[arm, state, 5] = spatial_frac

            # Passive operation: backlog drifts up, grid follows exogenous carbon
            # state, and operation type is mostly sticky.
            passive_q_candidates = [min(queue_states - 1, q), min(queue_states - 1, q + 1)]
            if q > 0:
                passive_q_candidates.append(q - 1)
            passive_q_weights = np.asarray([0.46, 0.42] + ([0.12] if q > 0 else []), dtype=float)
            passive_q_weights = passive_q_weights / passive_q_weights.sum()

            # Active operation: interactive-heavy centers mainly reduce current
            # backlog locally; batch-heavy centers can defer to a better future
            # bucket; geo-migratable centers have wider support because load can
            # be redistributed across operation modes.
            if op == 0:
                active_q_candidates = [max(0, q - 1), q]
                active_q_weights = np.asarray([0.70, 0.30])
            elif op == 1:
                active_q_candidates = sorted({max(0, q - 2), max(0, q - 1), min(queue_states - 1, q + 1)})
                active_q_weights = np.asarray([0.40, 0.38, 0.22])[: len(active_q_candidates)]
                active_q_weights = active_q_weights / active_q_weights.sum()
            else:
                active_q_candidates = sorted({max(0, q - 2), max(0, q - 1), q, min(queue_states - 1, q + 1)})
                active_q_weights = np.asarray([0.32, 0.28, 0.20, 0.20])[: len(active_q_candidates)]
                active_q_weights = active_q_weights / active_q_weights.sum()

            active_grid = grid_base[g].copy()
            if g == grid_states - 1:
                relief_target = np.exp(-0.85 * np.arange(grid_states))
                relief_target = relief_target / relief_target.sum()
                active_grid = (1.0 - transition_dominance) * active_grid + transition_dominance * relief_target
            active_grid = active_grid / active_grid.sum()

            op_stay = 0.82 - 0.18 * spatial_frac
            op_probs = np.full(op_states, (1.0 - op_stay) / max(1, op_states - 1))
            op_probs[op] = op_stay

            for q2, qw in zip(passive_q_candidates, passive_q_weights):
                for g2, gw in enumerate(grid_base[g]):
                    for op2, ow in enumerate(op_probs):
                        ns = encode_load_balance_state(q2, g2, op2, grid_states, op_states)
                        passive_p[arm, state, ns] += qw * gw * ow
                        support_mask[arm, PASSIVE, state, ns] = True

            for q2, qw in zip(active_q_candidates, active_q_weights):
                for g2, gw in enumerate(active_grid):
                    for op2, ow in enumerate(op_probs):
                        ns = encode_load_balance_state(q2, g2, op2, grid_states, op_states)
                        active_p[arm, state, ns] += qw * gw * ow
                        support_mask[arm, ACTIVE, state, ns] = True

            # Add heterogeneous rare migration edges. These make the product-state
            # transition less globally low-rank, which is exactly the concern to test.
            if op == 2 and rng.random() < heterogeneity:
                target_op = int(rng.integers(0, op_states))
                target_q = max(0, q - int(rng.integers(1, 3)))
                target_g = int(rng.integers(0, grid_states))
                ns = encode_load_balance_state(target_q, target_g, target_op, grid_states, op_states)
                active_p[arm, state] *= 0.90
                active_p[arm, state, ns] += 0.10
                support_mask[arm, ACTIVE, state, ns] = True

    passive_p = row_normalize(passive_p)
    active_p = row_normalize(active_p)
    contexts = normalize_contexts(contexts)
    initial_states = rng.integers(0, n_states, size=n_arms)
    return RMABInstance(
        rewards=rewards,
        passive_p=passive_p,
        active_p=active_p,
        contexts=contexts,
        support_mask=support_mask,
        initial_states=initial_states,
        description={
            "seed": seed,
            "n_arms": n_arms,
            "n_states": n_states,
            "queue_states": queue_states,
            "grid_states": grid_states,
            "op_states": op_states,
            "sparsity": int(np.mean(support_mask.sum(axis=-1))),
            "top_gap_lambda": 0.0,
            "transition_dominance": transition_dominance,
            "heterogeneity": heterogeneity,
        },
    )


def add_gateway_structure(
    rewards: np.ndarray,
    passive_p: np.ndarray,
    active_p: np.ndarray,
    support_mask: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Add transition-driven gateway states.

    Meeting note motivation: reward is relatively easy to learn, while sparse
    transition noise corrupts Whittle indices. Gateway states make transition
    structure consequential: a gateway has only moderate immediate reward, but
    active scheduling moves the arm toward a high-opportunity state whereas
    passive evolution moves it away. Myopic state-reward learners therefore
    lose information that Whittle-style policies can use.
    """
    n_arms, n_states = rewards.shape
    if n_states < 6:
        return rewards, passive_p, active_p, support_mask

    rewards = rewards.copy()
    passive_p = passive_p.copy()
    active_p = active_p.copy()
    support_mask = support_mask.copy()
    alpha = float(np.clip(strength, 0.0, 1.0))

    for arm in range(n_arms):
        high_state = int(np.argmax(rewards[arm]))
        gateway = (high_state - 1) % n_states
        sink = (high_state + n_states // 2) % n_states

        high_reward = rewards[arm, high_state]
        median_reward = float(np.median(rewards[arm]))
        rewards[arm, gateway] = (1.0 - alpha) * rewards[arm, gateway] + alpha * (
            0.55 * high_reward + 0.45 * median_reward
        )
        rewards[arm, sink] = (1.0 - 0.5 * alpha) * rewards[arm, sink] + 0.5 * alpha * median_reward

        active_p[arm, gateway] = 0.0
        passive_p[arm, gateway] = 0.0
        active_p[arm, gateway, high_state] = 0.82
        active_p[arm, gateway, (high_state + 1) % n_states] = 0.18
        passive_p[arm, gateway, sink] = 0.80
        passive_p[arm, gateway, (sink + 1) % n_states] = 0.20

        support_mask[arm, ACTIVE, gateway] = False
        support_mask[arm, PASSIVE, gateway] = False
        support_mask[arm, ACTIVE, gateway, high_state] = True
        support_mask[arm, ACTIVE, gateway, (high_state + 1) % n_states] = True
        support_mask[arm, PASSIVE, gateway, sink] = True
        support_mask[arm, PASSIVE, gateway, (sink + 1) % n_states] = True

    return rewards, passive_p, active_p, support_mask


def dynamic_indices(
    rewards: np.ndarray,
    active_p: np.ndarray,
    passive_p: np.ndarray,
    beta: float = 0.95,
    max_iter: int = 80,
    tol: float = 1e-5,
) -> np.ndarray:
    """Fast Whittle-style dynamic priority proxy.

    It estimates each state's marginal active value after solving a relaxed
    single-arm dynamic program. This keeps the scaffold fast enough for large
    sweeps while preserving the dependence on rewards and transition dynamics.
    """
    n_states = rewards.size
    values = np.zeros(n_states)
    for _ in range(max_iter):
        q_active = rewards + beta * active_p.dot(values)
        q_passive = beta * passive_p.dot(values)
        updated = np.maximum(q_active, q_passive)
        if np.max(np.abs(updated - values)) < tol:
            values = updated
            break
        values = updated
    return rewards + beta * (active_p - passive_p).dot(values)


def all_arm_indices(rewards: np.ndarray, active_p: np.ndarray, passive_p: np.ndarray) -> np.ndarray:
    return np.asarray(
        [dynamic_indices(rewards[i], active_p[i], passive_p[i]) for i in range(rewards.shape[0])]
    )


def variant_has(variant: str, token: str) -> bool:
    return token in variant.split("_") or token in variant


def row_normalize(p: np.ndarray) -> np.ndarray:
    denom = p.sum(axis=-1, keepdims=True)
    return np.divide(p, denom, out=np.zeros_like(p), where=denom > 0)


def apply_support_mask(p: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    masked = np.where(support_mask, p, 0.0)
    row_sum = masked.sum(axis=-1, keepdims=True)
    fallback = np.where(support_mask, 1.0, 0.0)
    fallback = row_normalize(fallback)
    return np.where(row_sum > 0, masked / np.maximum(row_sum, 1e-12), fallback)


def threshold_transition(
    p: np.ndarray,
    threshold: float,
    support_mask: np.ndarray | None = None,
) -> np.ndarray:
    pruned = np.where(p >= threshold, p, 0.0)
    if support_mask is not None:
        pruned = np.where(support_mask, pruned, 0.0)
    row_sum = pruned.sum(axis=-1, keepdims=True)
    fallback = p if support_mask is None else apply_support_mask(p, support_mask)
    return np.where(row_sum > 0, pruned / np.maximum(row_sum, 1e-12), fallback)


def low_rank_transition(p: np.ndarray, rank: int) -> np.ndarray:
    smoothed = np.zeros_like(p)
    for arm in range(p.shape[0]):
        for action in range(p.shape[1]):
            u, s, vt = np.linalg.svd(p[arm, action], full_matrices=False)
            r = min(rank, s.size)
            approx = (u[:, :r] * s[:r]) @ vt[:r, :]
            smoothed[arm, action] = np.maximum(approx, 0.0)
    return row_normalize(smoothed)


def make_offline_transition_prior(
    true_p: np.ndarray,
    support_mask: np.ndarray,
    support_only: bool,
) -> np.ndarray:
    """Build a simple generative/offline prior from queue-feasible structure.

    This approximates the meeting suggestion of using offline/generative
    information to initialize transition rows before online visits arrive.
    """
    support_uniform = row_normalize(np.where(support_mask, 1.0, 0.0))
    dense_uniform = np.full_like(true_p, 1.0 / true_p.shape[-1])
    background = support_uniform if support_only else dense_uniform
    prior = 0.70 * true_p + 0.30 * background
    if support_only:
        prior = apply_support_mask(prior, support_mask)
    return row_normalize(prior)


def gated_row_blend(
    base_p: np.ndarray,
    prior_p: np.ndarray,
    observed_counts: np.ndarray,
    gate_scale: float,
    rng: np.random.Generator | None = None,
    gate_mode: str = "deterministic",
    beta_concentration: float = 20.0,
) -> np.ndarray:
    """Blend a sampled row with a structural prior using visit-dependent trust."""
    gate_mean = gate_scale / (gate_scale + observed_counts)
    if gate_mode == "deterministic":
        gate = gate_mean
    elif gate_mode == "beta":
        if rng is None:
            raise ValueError("beta gate requires an RNG")
        concentration = max(beta_concentration, 2.0)
        alpha = np.maximum(gate_mean * concentration, 1e-3)
        beta = np.maximum((1.0 - gate_mean) * concentration, 1e-3)
        gate = rng.beta(alpha, beta)
    else:
        raise ValueError(f"unknown gate_mode: {gate_mode}")
    return row_normalize((1.0 - gate) * base_p + gate * prior_p)


class LearningPolicy:
    def __init__(
        self,
        instance: RMABInstance,
        policy: str,
        rng: np.random.Generator,
        transition_variant: str = "dense",
        prior_count: float = 1.0,
        trust_scale_mult: float = 1.0,
        gate_scale_mult: float = 1.0,
        gate_mode: str = "deterministic",
        beta_gate_concentration: float = 20.0,
        trust_floor: float = 0.10,
        trust_cap: float = 0.95,
    ) -> None:
        self.instance = instance
        self.policy = policy
        self.rng = rng
        self.n_arms, self.n_states = instance.rewards.shape
        self.transition_variant = transition_variant
        self.support_prior = variant_has(transition_variant, "support")
        self.threshold_prior = variant_has(transition_variant, "threshold")
        self.low_rank_prior = variant_has(transition_variant, "low_rank")
        self.offline_prior = variant_has(transition_variant, "offline")
        self.gated_prior = variant_has(transition_variant, "gated")
        self.exp4_log_weights = np.zeros(3)
        self.exp4_expert_arms = np.zeros(3, dtype=int)
        self.reward_prior_count = prior_count
        self.trust_scale_mult = trust_scale_mult
        self.gate_scale_mult = gate_scale_mult
        self.gate_mode = gate_mode
        self.beta_gate_concentration = beta_gate_concentration
        self.trust_floor = trust_floor
        self.trust_cap = trust_cap

        base = np.full((self.n_arms, 2, self.n_states, self.n_states), 0.05)
        if self.support_prior:
            base = np.where(instance.support_mask, 0.20, 1e-6)
        if self.offline_prior:
            true_p = np.stack([instance.passive_p, instance.active_p], axis=1)
            noisy_prior = make_offline_transition_prior(true_p, instance.support_mask, self.support_prior)
            base += 1.50 * noisy_prior
        self.prior_trans_counts = base.astype(float)
        self.observed_trans_counts = np.zeros_like(self.prior_trans_counts)
        self.trans_counts = self.prior_trans_counts.copy()

        context_proxy = self._context_reward_proxy(instance.contexts)
        self.reward_sum = context_proxy * prior_count
        self.reward_count = np.full((self.n_arms, self.n_states), prior_count)
        self.global_sum = context_proxy.mean(axis=1) * prior_count
        self.global_count = np.full(self.n_arms, prior_count)

    @staticmethod
    def _context_reward_proxy(contexts: np.ndarray) -> np.ndarray:
        avg_power = contexts[:, :, 0]
        avg_core = contexts[:, :, 1]
        delay_frac = contexts[:, :, 2]
        return np.maximum(0.0, 8.0 * avg_power + 2.0 * avg_core - 4.0 * delay_frac)

    def _sample_transition(self) -> tuple[np.ndarray, np.ndarray]:
        p = np.zeros_like(self.trans_counts)
        for i in range(self.n_arms):
            for a in range(2):
                for s in range(self.n_states):
                    p[i, a, s] = self.rng.dirichlet(self.trans_counts[i, a, s])
        prior_p = row_normalize(self.prior_trans_counts)
        observed_rows = self.observed_trans_counts.sum(axis=-1, keepdims=True)
        gate_scale = self.gate_scale_mult * (2.0 + np.sqrt(float(self.n_states)))
        if self.gated_prior:
            p = gated_row_blend(
                p,
                prior_p,
                observed_rows,
                gate_scale,
                rng=self.rng,
                gate_mode=self.gate_mode,
                beta_concentration=self.beta_gate_concentration,
            )
        if self.threshold_prior:
            thresh = threshold_transition(
                p,
                threshold=0.025,
                support_mask=self.instance.support_mask if self.support_prior else None,
            )
            if self.gated_prior:
                p = gated_row_blend(
                    p,
                    thresh,
                    observed_rows,
                    0.5 * gate_scale,
                    rng=self.rng,
                    gate_mode=self.gate_mode,
                    beta_concentration=self.beta_gate_concentration,
                )
            else:
                p = thresh
        if self.low_rank_prior:
            low_rank = low_rank_transition(p, rank=min(3, max(1, self.n_states // 8)))
            if self.gated_prior:
                p = gated_row_blend(
                    p,
                    low_rank,
                    observed_rows,
                    0.7 * gate_scale,
                    rng=self.rng,
                    gate_mode=self.gate_mode,
                    beta_concentration=self.beta_gate_concentration,
                )
            else:
                p = low_rank
        if self.support_prior:
            if self.gated_prior:
                support_multiplier = np.where(self.instance.support_mask, 1.0, 0.05)
                p = row_normalize(p * support_multiplier)
            else:
                p = apply_support_mask(p, self.instance.support_mask)
        return p[:, ACTIVE], p[:, PASSIVE]

    def _sample_rewards(self) -> np.ndarray:
        mean = self.reward_sum / np.maximum(self.reward_count, 1e-9)
        scale = 3.0 / np.sqrt(np.maximum(self.reward_count, 1.0))
        return np.maximum(0.0, self.rng.normal(mean, scale))

    def select(
        self,
        observed_states: np.ndarray,
        t: int,
        budget: int,
        true_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.policy == "oracle":
            assert true_indices is not None
            scores = true_indices[np.arange(self.n_arms), observed_states]
        elif self.policy == "state_thompson":
            sampled_rewards = self._sample_rewards()
            scores = sampled_rewards[np.arange(self.n_arms), observed_states]
        elif self.policy in {"tw", "tm_tw", "tm_tw_refined", "local_ucb_tw", "global_ucb_tw", "exp4"}:
            active_p, passive_p = self._sample_transition()
            sampled_rewards = self._sample_rewards()
            indices = all_arm_indices(sampled_rewards, active_p, passive_p)
            whittle_scores = indices[np.arange(self.n_arms), observed_states]
            horizon = max(80.0, 3.0 * self.n_states)
            exploration_weight = max(0.0, 1.0 - t / horizon)
            if self.policy == "tw":
                scores = whittle_scores
            elif self.policy == "tm_tw_refined":
                local_scores = self._local_scores(observed_states, t)
                trust = self._transition_trust(observed_states)
                scores = trust * zscore(whittle_scores) + (1.0 - trust) * zscore(local_scores)
            elif self.policy == "local_ucb_tw":
                local_scores = self._local_scores(observed_states, t)
                scores = zscore(whittle_scores) + 0.55 * exploration_weight * zscore(local_scores)
            elif self.policy == "global_ucb_tw":
                global_scores = self._global_scores(t)
                scores = zscore(whittle_scores) + 0.55 * exploration_weight * zscore(global_scores)
            elif self.policy == "exp4":
                global_scores = self._global_scores(t)
                local_scores = self._local_scores(observed_states, t)
                experts = np.vstack(
                    [
                        zscore(global_scores),
                        zscore(local_scores),
                        zscore(whittle_scores),
                    ]
                )
                self.exp4_expert_arms = np.argmax(experts, axis=1)
                weights = np.exp(self.exp4_log_weights - np.max(self.exp4_log_weights))
                weights /= np.sum(weights)
                scores = weights @ experts
            else:
                greedy_scores = self._greedy_scores(observed_states, t)
                scores = zscore(whittle_scores) + 0.55 * exploration_weight * zscore(greedy_scores)
        else:
            raise ValueError(f"unknown policy: {self.policy}")
        return np.argsort(scores)[-budget:][::-1]

    def _local_scores(self, observed_states: np.ndarray, t: int) -> np.ndarray:
        local_mean = self.reward_sum / np.maximum(self.reward_count, 1e-9)
        local_bonus = np.sqrt(np.log(t + 2.0) / np.maximum(self.reward_count, 1.0))
        return local_mean[np.arange(self.n_arms), observed_states] + 2.0 * local_bonus[
            np.arange(self.n_arms), observed_states
        ]

    def _transition_trust(self, observed_states: np.ndarray) -> np.ndarray:
        state_visits = self.observed_trans_counts.sum(axis=(1, 3))
        visits = state_visits[np.arange(self.n_arms), observed_states]
        scale = self.trust_scale_mult * (2.0 + np.sqrt(float(self.n_states)))
        trust = visits / (visits + scale)
        return np.clip(trust, self.trust_floor, self.trust_cap)

    def _global_scores(self, t: int) -> np.ndarray:
        global_mean = self.global_sum / np.maximum(self.global_count, 1e-9)
        global_bonus = np.sqrt(np.log(t + 2.0) / np.maximum(self.global_count, 1.0))
        return global_mean + 2.0 * global_bonus

    def _greedy_scores(self, observed_states: np.ndarray, t: int) -> np.ndarray:
        local = self._local_scores(observed_states, t)
        global_score = self._global_scores(t)
        w = min(1.0, t / 150.0)
        return (1.0 - w) * global_score + w * local

    def update(
        self,
        prev_observed_states: np.ndarray,
        actions: np.ndarray,
        next_observed_states: np.ndarray,
        rewards: np.ndarray,
    ) -> None:
        active_set = set(int(a) for a in actions)
        for arm in range(self.n_arms):
            action = ACTIVE if arm in active_set else PASSIVE
            s = int(prev_observed_states[arm])
            ns = int(next_observed_states[arm])
            self.observed_trans_counts[arm, action, s, ns] += 1.0
            self.trans_counts[arm, action, s, ns] += 1.0
            if action == ACTIVE:
                r = rewards[arm]
                self.reward_sum[arm, s] += r
                self.reward_count[arm, s] += 1.0
                self.global_sum[arm] += r
                self.global_count[arm] += 1.0
        if self.policy == "exp4":
            expert_rewards = rewards[self.exp4_expert_arms]
            scale = max(10.0, float(np.max(np.abs(expert_rewards))))
            self.exp4_log_weights += 0.15 * expert_rewards / scale

    def transition_error(self) -> tuple[float, float]:
        counts = self.trans_counts / self.trans_counts.sum(axis=-1, keepdims=True)
        true_p = np.stack([self.instance.passive_p, self.instance.active_p], axis=1)
        mask = self.instance.support_mask
        feasible_l1 = np.abs(counts - true_p)[mask].mean()
        off_support = counts[~mask].mean() if np.any(~mask) else 0.0
        return float(feasible_l1), float(off_support)


def zscore(x: np.ndarray) -> np.ndarray:
    std = np.std(x)
    if std < 1e-9:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def corrupt_states(
    rng: np.random.Generator,
    states: np.ndarray,
    n_states: int,
    noise_level: float,
) -> np.ndarray:
    observed = states.copy()
    if noise_level <= 0:
        return observed
    flips = rng.random(states.size) < noise_level
    observed[flips] = rng.integers(0, n_states, size=np.sum(flips))
    return observed


def step_environment(
    rng: np.random.Generator,
    instance: RMABInstance,
    states: np.ndarray,
    active_arms: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    active_set = set(int(a) for a in active_arms)
    rewards = np.zeros(states.size)
    next_states = np.zeros_like(states)
    for arm, state in enumerate(states):
        action = ACTIVE if arm in active_set else PASSIVE
        if action == ACTIVE:
            rewards[arm] = instance.rewards[arm, state]
            probs = instance.active_p[arm, state]
        else:
            probs = instance.passive_p[arm, state]
        next_states[arm] = rng.choice(instance.rewards.shape[1], p=probs)
    return next_states, rewards


def run_single_policy(
    instance: RMABInstance,
    policy_name: str,
    seed: int,
    rounds: int,
    budget: int = 1,
    noise_level: float = 0.0,
    transition_variant: str = "dense",
    trust_scale_mult: float = 1.0,
    gate_scale_mult: float = 1.0,
    gate_mode: str = "deterministic",
    beta_gate_concentration: float = 20.0,
    trust_floor: float = 0.10,
    trust_cap: float = 0.95,
) -> tuple[dict, np.ndarray]:
    rng        = np.random.default_rng(seed)
    oracle_rng = np.random.default_rng(seed)   # independent stream, same seed → fair comparison
    n_arms, n_states = instance.rewards.shape
    true_indices = all_arm_indices(instance.rewards, instance.active_p, instance.passive_p)
    true_policy = LearningPolicy(instance, "oracle", oracle_rng)
    policy = LearningPolicy(
        instance,
        policy_name,
        rng,
        transition_variant=transition_variant,
        trust_scale_mult=trust_scale_mult,
        gate_scale_mult=gate_scale_mult,
        gate_mode=gate_mode,
        beta_gate_concentration=beta_gate_concentration,
        trust_floor=trust_floor,
        trust_cap=trust_cap,
    )

    states = instance.initial_states.copy()
    rewards_by_round = np.zeros(rounds)
    oracle_rewards_by_round = np.zeros(rounds)
    top1_agree = 0
    top2_agree = 0
    start = time.perf_counter()

    for t in range(rounds):
        observed = corrupt_states(rng, states, n_states, noise_level)
        oracle_actions = true_policy.select(states, t, budget, true_indices=true_indices)
        actions = policy.select(observed, t, budget, true_indices=true_indices)

        rng_state = rng.bit_generator.state          # snapshot before transitions
        next_states, rewards = step_environment(rng, instance, states, actions)
        oracle_rng.bit_generator.state = rng_state  # reset oracle to same draw position
        oracle_next_states, oracle_rewards = step_environment(oracle_rng, instance, states, oracle_actions)
        next_observed = corrupt_states(rng, next_states, n_states, noise_level)
        policy.update(observed, actions, next_observed, rewards)

        rewards_by_round[t] = rewards[actions].sum()
        oracle_rewards_by_round[t] = oracle_rewards[oracle_actions].sum()
        oracle_rank = np.argsort(true_indices[np.arange(n_arms), states])[::-1]
        if int(actions[0]) == int(oracle_rank[0]):
            top1_agree += 1
        if int(actions[0]) in set(int(x) for x in oracle_rank[:2]):
            top2_agree += 1
        states = next_states
        _ = oracle_next_states

    runtime = time.perf_counter() - start
    oracle_cum = float(oracle_rewards_by_round.sum())
    cum_reward = float(rewards_by_round.sum())
    feasible_l1, off_support = policy.transition_error()
    summary = {
        "policy": policy_name,
        "seed": seed,
        "rounds": rounds,
        "S": n_states,
        "k": instance.description["sparsity"],
        "top_gap_lambda": instance.description["top_gap_lambda"],
        "transition_dominance": instance.description["transition_dominance"],
        "context_noise_level": noise_level,
        "transition_variant": transition_variant,
        "trust_scale_mult": trust_scale_mult,
        "gate_scale_mult": gate_scale_mult,
        "gate_mode": gate_mode,
        "beta_gate_concentration": beta_gate_concentration,
        "trust_floor": trust_floor,
        "trust_cap": trust_cap,
        "support_mask_enabled": variant_has(transition_variant, "support"),
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
    }
    return summary, rewards_by_round


def run_policy_suite(
    instance: RMABInstance,
    seed: int,
    rounds: int,
    noise_level: float = 0.0,
    include_masked: bool = False,
    variants: list[str] | None = None,
    policies_filter: set[str] | None = None,
    trust_scale_mult: float = 1.0,
    gate_scale_mult: float = 1.0,
    gate_mode: str = "deterministic",
    beta_gate_concentration: float = 20.0,
    trust_floor: float = 0.10,
    trust_cap: float = 0.95,
) -> RunResult:
    variants = variants or (["dense", "support"] if include_masked else ["dense"])
    policies: list[tuple[str, str]] = [
        ("oracle", "dense"),
        ("state_thompson", "dense"),
        ("global_ucb_tw", "dense"),
        ("local_ucb_tw", "dense"),
        ("exp4", "dense"),
    ]
    for variant in variants:
        policies.extend([("tw", variant), ("tm_tw", variant), ("tm_tw_refined", variant)])
    if policies_filter is not None:
        policies = [(policy, variant) for policy, variant in policies if policy == "oracle" or policy in policies_filter]

    rows = []
    curves = {}
    seen = set()
    for policy, variant in policies:
        key = (policy, variant)
        if key in seen:
            continue
        seen.add(key)
        row, curve = run_single_policy(
            instance,
            policy,
            seed=seed + stable_policy_offset(policy, variant),
            rounds=rounds,
            noise_level=noise_level,
            transition_variant=variant,
            trust_scale_mult=trust_scale_mult,
            gate_scale_mult=gate_scale_mult,
            gate_mode=gate_mode,
            beta_gate_concentration=beta_gate_concentration,
            trust_floor=trust_floor,
            trust_cap=trust_cap,
        )
        label = policy if policy in {"oracle", "state_thompson"} else f"{policy}_{variant}"
        row["policy_label"] = label
        rows.append(row)
        curves[label] = curve
    return RunResult(rows, curves)


def stable_policy_offset(policy: str, variant: str) -> int:
    return sum(ord(ch) for ch in f"{policy}:{variant}")


def moving_average(x: np.ndarray, window: int = 25) -> np.ndarray:
    if x.size < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_gap(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted({row["policy_label"] for row in rows})
    lambdas = sorted({float(row["top_gap_lambda"]) for row in rows})
    plt.figure(figsize=(7, 4.5))
    for label in labels:
        means = []
        for lam in lambdas:
            vals = [
                float(row["reward_pct_oracle"])
                for row in rows
                if row["policy_label"] == label and float(row["top_gap_lambda"]) == lam
            ]
            means.append(np.mean(vals))
        plt.plot(lambdas, means, marker="o", label=label)
    plt.xlabel("top-gap interpolation lambda")
    plt.ylabel("cumulative reward / oracle (%)")
    plt.title("Arm-separation sensitivity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "gap_sensitivity_reward_pct.png", dpi=200)
    plt.close()


def plot_sparse_heatmap(rows: list[dict], out_dir: Path, policy_label: str = "tm_tw") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered = [row for row in rows if row["policy_label"] == policy_label]
    if not filtered:
        return
    states = sorted({int(row["S"]) for row in filtered})
    rounds = sorted({int(row["rounds"]) for row in filtered})
    matrix = np.zeros((len(states), len(rounds)))
    for i, s in enumerate(states):
        for j, t in enumerate(rounds):
            vals = [
                float(row["cum_regret"])
                for row in filtered
                if int(row["S"]) == s and int(row["rounds"]) == t
            ]
            matrix[i, j] = np.mean(vals) if vals else np.nan
    plt.figure(figsize=(6.5, 4.5))
    plt.imshow(matrix, aspect="auto", origin="lower")
    plt.colorbar(label="cumulative regret to oracle")
    plt.xticks(range(len(rounds)), rounds)
    plt.yticks(range(len(states)), states)
    plt.xlabel("round budget T")
    plt.ylabel("state count S")
    plt.title(f"Sparse-state heatmap: {policy_label}")
    plt.tight_layout()
    plt.savefig(out_dir / f"sparse_heatmap_{policy_label}.png", dpi=200)
    plt.close()


def plot_learning_curves(curves: dict[str, list[np.ndarray]], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    for label, series in sorted(curves.items()):
        stacked = np.vstack(series)
        mean_curve = stacked.mean(axis=0)
        plt.plot(moving_average(mean_curve), label=label)
    plt.xlabel("round")
    plt.ylabel("moving-average reward")
    plt.title(stem.replace("_", " "))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_learning_curves.png", dpi=200)
    plt.close()


def plot_variant_summary(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted({row["policy_label"] for row in rows if row["policy_label"] != "oracle"})
    means = []
    for label in labels:
        vals = [float(row["reward_pct_oracle"]) for row in rows if row["policy_label"] == label]
        means.append(float(np.mean(vals)))
    plt.figure(figsize=(9, 4.8))
    plt.bar(np.arange(len(labels)), means)
    plt.xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    plt.ylabel("mean cumulative reward / oracle (%)")
    plt.title("Meeting-improvement variants")
    plt.tight_layout()
    plt.savefig(out_dir / "improvement_variant_summary.png", dpi=200)
    plt.close()


def write_group_summary(rows: list[dict], group_fields: list[str], path: Path) -> None:
    numeric_fields = [
        "reward_pct_oracle",
        "transition_l1_error",
        "off_support_leakage",
        "top1_agreement",
        "top2_agreement",
        "cum_regret",
    ]
    grouped: dict[tuple[str, ...], list[dict]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in group_fields)
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for key, group in sorted(grouped.items()):
        out = {field: value for field, value in zip(group_fields, key)}
        out["n"] = len(group)
        for field in numeric_fields:
            vals = [float(row[field]) for row in group if field in row]
            if vals:
                out[f"mean_{field}"] = float(np.mean(vals))
        summary_rows.append(out)
    write_rows(path, summary_rows)


def plot_context_noise(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    default_rows = [
        row
        for row in rows
        if float(row.get("trust_scale_mult", 1.0)) == 1.0
        and float(row.get("gate_scale_mult", 1.0)) == 1.0
        and row.get("gate_mode", "deterministic") == "deterministic"
    ]
    plot_rows = default_rows or rows
    labels = [
        "state_thompson",
        "tw_dense",
        "local_ucb_tw_dense",
        "exp4_dense",
        "tm_tw_dense",
        "tm_tw_refined_dense",
        "tm_tw_refined_gated_offline",
    ]
    states = sorted({int(row["S"]) for row in plot_rows})
    fig, axes = plt.subplots(1, len(states), figsize=(5.2 * len(states), 4.0), sharey=True)
    if len(states) == 1:
        axes = [axes]
    for ax, state_count in zip(axes, states):
        state_rows = [row for row in plot_rows if int(row["S"]) == state_count]
        noise_levels = sorted({float(row["context_noise_level"]) for row in state_rows})
        for label in labels:
            means = []
            for noise in noise_levels:
                vals = [
                    float(row["reward_pct_oracle"])
                    for row in state_rows
                    if row["policy_label"] == label and float(row["context_noise_level"]) == noise
                ]
                means.append(float(np.mean(vals)) if vals else np.nan)
            if not np.all(np.isnan(means)):
                ax.plot(noise_levels, means, marker="o", label=label)
        ax.set_title(f"S={state_count}")
        ax.set_xlabel("contextual state noise")
        ax.grid(True, alpha=0.35)
    axes[0].set_ylabel("cumulative reward / oracle (%)")
    axes[-1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.suptitle("Contextual-noise robustness")
    fig.tight_layout()
    fig.savefig(out_dir / "context_noise_reward_pct.png", dpi=200)
    plt.close(fig)


def write_reconstructed_data(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    n_jobs = args.states * 5
    batch_size = 5
    lookahead_jobs = min(n_jobs, 2 * batch_size)
    delay_weight = 0.08

    job_rows: list[dict] = []
    rewards = []
    contexts = []
    passive_ps = []
    active_ps = []
    masks = []

    for arm in range(5):
        quality_shift = 0.65 if arm == 0 else rng.normal(0.0, 0.12)
        jobs = sample_vm_jobs(rng, n_jobs, quality_shift=quality_shift)
        for job_id in range(n_jobs):
            job_rows.append(
                {
                    "arm": arm,
                    "job_id": job_id,
                    "utilization": jobs.utilization[job_id],
                    "core_hours": jobs.core_hours[job_id],
                    "power": jobs.power[job_id],
                    "qos_cost": jobs.qos_cost[job_id],
                    "is_interactive": bool(jobs.is_interactive[job_id]),
                }
            )
        arm_rewards, arm_contexts = build_arm_from_jobs(jobs, batch_size, lookahead_jobs, delay_weight)
        passive_p, passive_mask = local_transition_matrix(rng, args.states, args.sparsity, active=False)
        active_p, active_mask = local_transition_matrix(rng, args.states, args.sparsity + 1, active=True)
        rewards.append(arm_rewards)
        contexts.append(arm_contexts)
        passive_ps.append(passive_p)
        active_ps.append(active_p)
        masks.append(np.stack([passive_mask, active_mask], axis=0))

    rewards_arr = np.asarray(rewards)
    contexts_arr = normalize_contexts(np.asarray(contexts))
    passive_arr = np.asarray(passive_ps)
    active_arr = np.asarray(active_ps)
    support_mask = np.asarray(masks)
    rewards_arr, passive_arr, active_arr, support_mask = add_gateway_structure(
        rewards_arr,
        passive_arr,
        active_arr,
        support_mask,
        strength=args.transition_dominance,
    )

    state_rows: list[dict] = []
    for arm in range(rewards_arr.shape[0]):
        for state in range(rewards_arr.shape[1]):
            state_rows.append(
                {
                    "arm": arm,
                    "state": state,
                    "reward": rewards_arr[arm, state],
                    "avg_power_context": contexts_arr[arm, state, 0],
                    "avg_core_context": contexts_arr[arm, state, 1],
                    "interactive_fraction_context": contexts_arr[arm, state, 2],
                    "normalized_queue_position": contexts_arr[arm, state, 3],
                }
            )

    transition_rows: list[dict] = []
    for arm in range(rewards_arr.shape[0]):
        for action_name, action, p_arr in [("passive", PASSIVE, passive_arr), ("active", ACTIVE, active_arr)]:
            for state in range(rewards_arr.shape[1]):
                for next_state in range(rewards_arr.shape[1]):
                    prob = p_arr[arm, state, next_state]
                    if prob > 0 or support_mask[arm, action, state, next_state]:
                        transition_rows.append(
                            {
                                "arm": arm,
                                "action": action_name,
                                "state": state,
                                "next_state": next_state,
                                "probability": prob,
                                "support_feasible": bool(support_mask[arm, action, state, next_state]),
                            }
                        )

    out_dir = Path(args.output)
    write_rows(out_dir / "reconstructed_vm_jobs.csv", job_rows)
    write_rows(out_dir / "reconstructed_states.csv", state_rows)
    write_rows(out_dir / "reconstructed_transitions.csv", transition_rows)


def _maybe_load_dfs(args: argparse.Namespace) -> list[pd.DataFrame] | None:
    data_dir = getattr(args, "data_dir", None)
    if not data_dir:
        return None
    dfs = load_datacenter_dfs(data_dir)
    print(f"Loaded {len(dfs)} datacenter CSV files from {data_dir!r}")
    return dfs


def run_gap_experiment(args: argparse.Namespace) -> None:
    rows: list[dict] = []
    all_curves: dict[str, list[np.ndarray]] = {}
    datacenter_dfs = _maybe_load_dfs(args)
    for lam in args.gap_lambdas:
        for seed in range(args.seeds):
            instance = make_instance(
                seed=args.seed + seed,
                n_states=args.states,
                sparsity=args.sparsity,
                top_gap_lambda=lam,
                transition_dominance=args.transition_dominance,
                datacenter_dfs=datacenter_dfs,
            )
            result = run_policy_suite(instance, seed=args.seed + seed, rounds=args.rounds)
            for row in result.rows:
                row["experiment_id"] = "gap_sensitivity"
                rows.append(row)
            if lam == args.gap_lambdas[-1]:
                for label, curve in result.learning_curves.items():
                    all_curves.setdefault(label, []).append(curve)
    out_dir = Path(args.output)
    write_rows(out_dir / "gap_sensitivity_results.csv", rows)
    plot_gap(rows, out_dir)
    plot_learning_curves(all_curves, out_dir, "narrow_gap")


def run_sparse_experiment(args: argparse.Namespace) -> None:
    rows: list[dict] = []
    selected_curves: dict[str, list[np.ndarray]] = {}
    datacenter_dfs = _maybe_load_dfs(args)
    for s in args.state_grid:
        for t_budget in args.round_grid:
            for k in args.sparsity_grid:
                for seed in range(args.seeds):
                    instance = make_instance(
                        seed=args.seed + seed + 31 * s + 7 * k,
                        n_states=s,
                        sparsity=k,
                        transition_dominance=args.transition_dominance,
                        datacenter_dfs=datacenter_dfs,
                    )
                    result = run_policy_suite(
                        instance,
                        seed=args.seed + seed,
                        rounds=t_budget,
                        include_masked=args.include_masked,
                        variants=args.variants if hasattr(args, "variants") and args.variants else None,
                    )
                    for row in result.rows:
                        row["experiment_id"] = "sparse_state_limited_rounds"
                        rows.append(row)
                    if s == args.state_grid[-1] and t_budget == args.round_grid[0] and k == args.sparsity_grid[0]:
                        for label, curve in result.learning_curves.items():
                            selected_curves.setdefault(label, []).append(curve)
    out_dir = Path(args.output)
    write_rows(out_dir / "sparse_state_results.csv", rows)
    plot_sparse_heatmap(rows, out_dir, "tm_tw_dense")
    variants = getattr(args, "variants", None) or []
    if args.include_masked or "support" in variants:
        plot_sparse_heatmap(rows, out_dir, "tm_tw_support")
    plot_learning_curves(selected_curves, out_dir, "sparse_hard_setting")
    if hasattr(args, "plot_variants") and args.plot_variants:
        plot_variant_summary(rows, out_dir)


def run_improvements_experiment(args: argparse.Namespace) -> None:
    improvement_args = argparse.Namespace(**vars(args))
    improvement_args.variants = args.variants or DEFAULT_VARIANTS
    improvement_args.include_masked = True
    improvement_args.plot_variants = True
    run_sparse_experiment(improvement_args)
    out_dir = Path(args.output)
    raw_rows = list(__import__("csv").DictReader((out_dir / "sparse_state_results.csv").open()))
    write_group_summary(raw_rows, ["policy_label"], out_dir / "summary.csv")


def run_context_noise_experiment(args: argparse.Namespace) -> None:
    rows: list[dict] = []
    variants = args.variants or [
        "dense",
        "offline",
        "low_rank",
        "gated_offline",
        "gated_offline_low_rank",
        "support_offline",
        "support_gated_offline_low_rank",
    ]
    datacenter_dfs = _maybe_load_dfs(args)
    for state_count in args.state_grid:
        for noise_level in args.noise_grid:
            for trust_scale_mult in args.trust_scale_mults:
                for gate_scale_mult in args.gate_scale_mults:
                    for gate_mode in args.gate_modes:
                        for trust_floor in args.trust_floors:
                            for seed in range(args.seeds):
                                instance = make_instance(
                                    seed=args.seed + seed + 31 * state_count,
                                    n_states=state_count,
                                    sparsity=args.sparsity,
                                    transition_dominance=args.transition_dominance,
                                    datacenter_dfs=datacenter_dfs,
                                )
                                result = run_policy_suite(
                                    instance,
                                    seed=args.seed + seed,
                                    rounds=args.rounds,
                                    noise_level=noise_level,
                                    include_masked=True,
                                    variants=variants,
                                    policies_filter=set(args.policies) if args.policies else None,
                                    trust_scale_mult=trust_scale_mult,
                                    gate_scale_mult=gate_scale_mult,
                                    gate_mode=gate_mode,
                                    beta_gate_concentration=args.beta_gate_concentration,
                                    trust_floor=trust_floor,
                                    trust_cap=args.trust_cap,
                                )
                                for row in result.rows:
                                    row["experiment_id"] = "context_noise_refinement"
                                    rows.append(row)
                                if args.flush_every > 0 and len(rows) % args.flush_every == 0:
                                    out_dir = Path(args.output)
                                    write_rows(out_dir / "context_noise_results.partial.csv", rows)
                                if args.progress:
                                    print(
                                        "finished",
                                        f"S={state_count}",
                                        f"noise={noise_level}",
                                        f"trust={trust_scale_mult}",
                                        f"gate={gate_scale_mult}",
                                        f"mode={gate_mode}",
                                        f"floor={trust_floor}",
                                        f"seed={seed}",
                                        flush=True,
                                    )

    out_dir = Path(args.output)
    write_rows(out_dir / "context_noise_results.csv", rows)
    write_group_summary(
        rows,
        ["S", "context_noise_level", "trust_scale_mult", "gate_scale_mult", "gate_mode", "trust_floor", "policy_label"],
        out_dir / "context_noise_summary.csv",
    )
    plot_context_noise(rows, out_dir)


def run_load_balance_experiment(args: argparse.Namespace) -> None:
    rows: list[dict] = []
    variants = args.variants or [
        "dense",
        "gated_offline",
        "low_rank",
        "gated_offline_low_rank",
        "support_offline",
    ]
    datacenter_dfs = _maybe_load_dfs(args)
    for queue_states in args.queue_state_grid:
        for grid_states in args.grid_state_grid:
            for op_states in args.op_state_grid:
                for noise_level in args.noise_grid:
                    for gate_mode in args.gate_modes:
                        for trust_floor in args.trust_floors:
                            for seed in range(args.seeds):
                                instance = make_load_balance_instance(
                                    seed=args.seed + seed + 37 * queue_states + 11 * grid_states + 5 * op_states,
                                    n_arms=args.arms,
                                    queue_states=queue_states,
                                    grid_states=grid_states,
                                    op_states=op_states,
                                    transition_dominance=args.transition_dominance,
                                    datacenter_dfs=datacenter_dfs,
                                )
                                result = run_policy_suite(
                                    instance,
                                    seed=args.seed + seed,
                                    rounds=args.rounds,
                                    noise_level=noise_level,
                                    include_masked=True,
                                    variants=variants,
                                    policies_filter=set(args.policies) if args.policies else None,
                                    trust_scale_mult=args.trust_scale_mults[0],
                                    gate_scale_mult=args.gate_scale_mults[0],
                                    gate_mode=gate_mode,
                                    beta_gate_concentration=args.beta_gate_concentration,
                                    trust_floor=trust_floor,
                                    trust_cap=args.trust_cap,
                                )
                                for row in result.rows:
                                    row["experiment_id"] = "load_balance_product_state"
                                    row["queue_states"] = queue_states
                                    row["grid_states"] = grid_states
                                    row["op_states"] = op_states
                                    row["arms"] = args.arms
                                    rows.append(row)
                                if args.flush_every > 0 and len(rows) % args.flush_every == 0:
                                    out_dir = Path(args.output)
                                    write_rows(out_dir / "load_balance_results.partial.csv", rows)
                                if args.progress:
                                    print(
                                        "finished",
                                        f"Q={queue_states}",
                                        f"G={grid_states}",
                                        f"O={op_states}",
                                        f"S={instance.rewards.shape[1]}",
                                        f"noise={noise_level}",
                                        f"mode={gate_mode}",
                                        f"floor={trust_floor}",
                                        f"seed={seed}",
                                        flush=True,
                                    )

    out_dir = Path(args.output)
    write_rows(out_dir / "load_balance_results.csv", rows)
    write_group_summary(
        rows,
        [
            "queue_states",
            "grid_states",
            "op_states",
            "S",
            "context_noise_level",
            "gate_mode",
            "trust_floor",
            "policy_label",
        ],
        out_dir / "load_balance_summary.csv",
    )


def run_quick(args: argparse.Namespace) -> None:
    quick_dir = Path(args.output)
    gap_args = argparse.Namespace(**vars(args))
    gap_args.gap_lambdas = [0.0, 0.8]
    gap_args.states = 8
    gap_args.sparsity = 2
    gap_args.rounds = min(args.rounds, 120)
    gap_args.seeds = min(args.seeds, 2)
    gap_args.output = str(quick_dir / "quick_gap")
    run_gap_experiment(gap_args)

    sparse_args = argparse.Namespace(**vars(args))
    sparse_args.state_grid = [8, 20]
    sparse_args.round_grid = [80, 120]
    sparse_args.sparsity_grid = [2]
    sparse_args.rounds = min(args.rounds, 120)
    sparse_args.seeds = min(args.seeds, 2)
    sparse_args.include_masked = True
    sparse_args.output = str(quick_dir / "quick_sparse")
    run_sparse_experiment(sparse_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=["quick", "reconstruct", "gap", "sparse", "improvements", "context_noise", "load_balance"],
        default="quick",
    )
    parser.add_argument("--output", default="docs/research/rmab_vm_outputs")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to directory containing datacenter_*_with_metrics.csv files. "
             "When provided, real VM trace data is used instead of synthetic sampling.",
    )
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--arms", type=int, default=8)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--sparsity", type=int, default=2)
    parser.add_argument("--transition-dominance", type=float, default=0.45)
    parser.add_argument("--include-masked", action="store_true")
    parser.add_argument("--variants", nargs="+", default=None, help="Transition-learning variants for TW/TM-TW.")
    parser.add_argument("--policies", nargs="+", default=None, help="Optional policy filter for expensive sweeps.")
    parser.add_argument("--gap-lambdas", type=float, nargs="+", default=[0.0, 0.5, 0.8, 0.95])
    parser.add_argument("--state-grid", type=int, nargs="+", default=[8, 20, 50, 100])
    parser.add_argument("--round-grid", type=int, nargs="+", default=[250, 500, 1000])
    parser.add_argument("--sparsity-grid", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--noise-grid", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--queue-state-grid", type=int, nargs="+", default=[8, 12])
    parser.add_argument("--grid-state-grid", type=int, nargs="+", default=[3])
    parser.add_argument("--op-state-grid", type=int, nargs="+", default=[3])
    parser.add_argument("--trust-scale-mults", type=float, nargs="+", default=[1.0])
    parser.add_argument("--gate-scale-mults", type=float, nargs="+", default=[1.0])
    parser.add_argument("--gate-modes", choices=["deterministic", "beta"], nargs="+", default=["deterministic"])
    parser.add_argument("--beta-gate-concentration", type=float, default=20.0)
    parser.add_argument("--trust-floor", type=float, default=0.10)
    parser.add_argument("--trust-floors", type=float, nargs="+", default=[0.10])
    parser.add_argument("--trust-cap", type=float, default=0.95)
    parser.add_argument("--flush-every", type=int, default=0)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiment == "quick":
        run_quick(args)
    elif args.experiment == "reconstruct":
        write_reconstructed_data(args)
    elif args.experiment == "gap":
        run_gap_experiment(args)
    elif args.experiment == "sparse":
        run_sparse_experiment(args)
    elif args.experiment == "improvements":
        run_improvements_experiment(args)
    elif args.experiment == "context_noise":
        run_context_noise_experiment(args)
    elif args.experiment == "load_balance":
        run_load_balance_experiment(args)
    else:
        raise ValueError(args.experiment)


if __name__ == "__main__":
    main()
