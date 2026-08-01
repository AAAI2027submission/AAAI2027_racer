"""Run load-balance experiments with regional carbon/electricity grid states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.run_experiments import (  # noqa: E402
    ACTIVE,
    PASSIVE,
    RMABInstance,
    load_datacenter_dfs,
    load_vm_jobs_from_csv,
    normalize_contexts,
    row_normalize,
    run_single_policy,
    write_group_summary,
    write_rows,
)


DEFAULT_OUTPUT = ROOT / "docs/research/rmab_vm_outputs/load_balance_region_grid_v1"
GRID_COST_CSV = ROOT / "docs/research/rmab_vm_outputs/grid_cost_features_v1/grid_cost_region_averages_2023.csv"
DATACENTER_DIR = ROOT.parent / "datacenter_with_metrics"

STRATEGIES = [
    ("oracle", "oracle", "dense", "deterministic"),
    ("State Thompson", "state_thompson", "dense", "deterministic"),
    ("Local UCB+TW", "local_ucb_tw", "dense", "deterministic"),
    ("Global UCB+TW", "global_ucb_tw", "dense", "deterministic"),
    ("EXP4", "exp4", "dense", "deterministic"),
    ("TW", "tw", "dense", "deterministic"),
    ("TM-TW", "tm_tw", "dense", "deterministic"),
    ("Adaptive TM-TW", "tm_tw_refined", "dense", "deterministic"),
    ("Adaptive + beta-gate prior", "tm_tw_refined", "gated_offline", "beta"),
    ("Adaptive + beta-gate + LR", "tm_tw_refined", "gated_offline_low_rank", "beta"),
    ("Adaptive + support/offline", "tm_tw_refined", "support_offline", "deterministic"),
]

THETA = {
    0: 2.0,  # interactive-heavy
    1: 1.0,  # batch-flexible
    2: 0.0,  # geo-migratable
}


def encode_state(queue_state: int, region_state: int, op_state: int, region_states: int, op_states: int) -> int:
    return (queue_state * region_states + region_state) * op_states + op_state


def decode_state(state: int, region_states: int, op_states: int) -> tuple[int, int, int]:
    op_state = state % op_states
    tmp = state // op_states
    region_state = tmp % region_states
    queue_state = tmp // region_states
    return queue_state, region_state, op_state


def selected_regions(grid_costs: pd.DataFrame, region_states: int) -> pd.DataFrame:
    df = grid_costs.copy()
    carbon = df.get("carbon_direct_norm", df["mean_carbon_intensity_direct_gco2eq_per_kwh"])
    price = df.get("electricity_price_norm", df["mean_electricity_price_usd_per_mwh"])
    df["carbon_norm"] = (carbon - carbon.min()) / max(float(carbon.max() - carbon.min()), 1e-9)
    df["price_norm"] = (price - price.min()) / max(float(price.max() - price.min()), 1e-9)
    df["combined_grid_cost"] = 0.5 * df["carbon_norm"] + 0.5 * df["price_norm"]
    df = df.sort_values("combined_grid_cost").reset_index(drop=True)
    if region_states < len(df):
        idx = np.linspace(0, len(df) - 1, region_states).round().astype(int)
        df = df.iloc[idx].drop_duplicates("region_code").reset_index(drop=True)

    # Re-normalize within the selected scenario so G=5,10,15 use comparable
    # low-to-high cost scales instead of inheriting different absolute ranges.
    carbon = df["mean_carbon_intensity_direct_gco2eq_per_kwh"]
    price = df["mean_electricity_price_usd_per_mwh"]
    df["carbon_norm"] = (carbon - carbon.min()) / max(float(carbon.max() - carbon.min()), 1e-9)
    df["price_norm"] = (price - price.min()) / max(float(price.max() - price.min()), 1e-9)
    df["combined_grid_cost"] = 0.5 * df["carbon_norm"] + 0.5 * df["price_norm"]
    return df.sort_values("combined_grid_cost").reset_index(drop=True)


def arm_power_profile(datacenter_df: pd.DataFrame, rng: np.random.Generator, queue_states: int) -> tuple[np.ndarray, np.ndarray]:
    jobs = load_vm_jobs_from_csv(datacenter_df, rng, n_jobs=queue_states * 5)
    savings = np.zeros(queue_states)
    delay_costs = np.zeros(queue_states)
    for q in range(queue_states):
        start = q * 5
        batch_idx = np.arange(start, start + 5) % jobs.power.size
        window_idx = np.arange(start, start + min(jobs.power.size, 10)) % jobs.power.size
        chosen = window_idx[np.argsort(jobs.power[window_idx])[:5]]
        skipped = np.setdiff1d(batch_idx, chosen, assume_unique=False)
        savings[q] = max(0.0, jobs.power[batch_idx].sum() - jobs.power[chosen].sum())
        delay_costs[q] = jobs.qos_cost[skipped][jobs.is_interactive[skipped]].sum()
    return savings, delay_costs


def make_region_grid_instance(
    seed: int,
    grid_costs: pd.DataFrame,
    datacenter_dfs: list[pd.DataFrame],
    n_arms: int,
    queue_states: int,
    region_states: int,
    op_states: int,
) -> tuple[RMABInstance, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    regions = selected_regions(grid_costs, region_states)
    n_states = queue_states * len(regions) * op_states
    rewards = np.zeros((n_arms, n_states))
    contexts = np.zeros((n_arms, n_states, 7))
    passive_p = np.zeros((n_arms, n_states, n_states))
    active_p = np.zeros((n_arms, n_states, n_states))
    support_mask = np.zeros((n_arms, 2, n_states, n_states), dtype=bool)

    grid_cost = regions["combined_grid_cost"].to_numpy()

    for arm in range(n_arms):
        savings, delay_costs = arm_power_profile(datacenter_dfs[arm % len(datacenter_dfs)], rng, queue_states)
        migration_base = rng.uniform(0.12, 0.45)
        arm_scale = rng.uniform(0.9, 1.15)
        for state in range(n_states):
            q, g, op = decode_state(state, len(regions), op_states)
            backlog = q / max(1, queue_states - 1)
            theta = THETA[op]
            delay_multiplier = 1.0 + theta
            migration_multiplier = 1.0 + (1.0 if op == 2 else 0.25 * op)
            migration_cost = migration_base * migration_multiplier * (1.0 + backlog)
            delay_cost = delay_multiplier * delay_costs[q]
            rewards[arm, state] = max(
                0.0,
                arm_scale * grid_cost[g] * savings[q] - migration_cost - 0.08 * delay_cost,
            )

            contexts[arm, state, 0] = backlog
            contexts[arm, state, 1] = regions.loc[g, "carbon_norm"]
            contexts[arm, state, 2] = regions.loc[g, "price_norm"]
            contexts[arm, state, 3] = grid_cost[g]
            contexts[arm, state, 4] = float(op == 0)
            contexts[arm, state, 5] = float(op == 1)
            contexts[arm, state, 6] = float(op == 2)

            passive_q = [q, min(queue_states - 1, q + 1)]
            passive_q_w = np.array([0.50, 0.50])
            if q > 0:
                passive_q.append(q - 1)
                passive_q_w = np.array([0.45, 0.42, 0.13])
            passive_q_w = passive_q_w / passive_q_w.sum()

            if op == 0:
                active_q = [max(0, q - 1), q]
                active_q_w = np.array([0.72, 0.28])
            elif op == 1:
                active_q = sorted({max(0, q - 2), max(0, q - 1), q})
                active_q_w = np.array([0.42, 0.38, 0.20])[: len(active_q)]
            else:
                active_q = sorted({max(0, q - 2), max(0, q - 1), q, min(queue_states - 1, q + 1)})
                active_q_w = np.array([0.30, 0.28, 0.22, 0.20])[: len(active_q)]
            active_q_w = active_q_w / active_q_w.sum()

            region_passive = np.exp(-1.2 * np.abs(np.arange(len(regions)) - g))
            region_passive[g] += 1.0
            region_passive = region_passive / region_passive.sum()
            region_active = np.exp(-0.75 * np.abs(np.arange(len(regions)) - np.argmin(grid_cost)))
            region_active = 0.55 * region_passive + 0.45 * region_active / region_active.sum()
            region_active = region_active / region_active.sum()

            op_stay = 0.82 if op != 2 else 0.62
            op_probs = np.full(op_states, (1.0 - op_stay) / max(1, op_states - 1))
            op_probs[op] = op_stay

            for q2, qw in zip(passive_q, passive_q_w):
                for g2, gw in enumerate(region_passive):
                    for op2, ow in enumerate(op_probs):
                        ns = encode_state(q2, g2, op2, len(regions), op_states)
                        passive_p[arm, state, ns] += qw * gw * ow
                        support_mask[arm, PASSIVE, state, ns] = True
            for q2, qw in zip(active_q, active_q_w):
                for g2, gw in enumerate(region_active):
                    for op2, ow in enumerate(op_probs):
                        ns = encode_state(q2, g2, op2, len(regions), op_states)
                        active_p[arm, state, ns] += qw * gw * ow
                        support_mask[arm, ACTIVE, state, ns] = True

    instance = RMABInstance(
        rewards=rewards,
        passive_p=row_normalize(passive_p),
        active_p=row_normalize(active_p),
        contexts=normalize_contexts(contexts),
        support_mask=support_mask,
        initial_states=rng.integers(0, n_states, size=n_arms),
        description={
            "seed": seed,
            "n_arms": n_arms,
            "n_states": n_states,
            "queue_states": queue_states,
            "grid_states": len(regions),
            "op_states": op_states,
            "sparsity": int(np.mean(support_mask.sum(axis=-1))),
            "top_gap_lambda": 0.0,
            "transition_dominance": 0.45,
        },
    )
    return instance, regions


def stable_strategy_offset(policy: str, variant: str, gate_mode: str) -> int:
    return sum(ord(ch) for ch in f"{policy}:{variant}:{gate_mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-cost-csv", type=Path, default=GRID_COST_CSV)
    parser.add_argument("--data-dir", type=Path, default=DATACENTER_DIR)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=250)
    parser.add_argument("--arms", type=int, default=8)
    parser.add_argument("--queue-states", type=int, default=12)
    parser.add_argument("--region-state-grid", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--op-states", type=int, default=3)
    parser.add_argument("--noise-level", type=float, default=0.2)
    parser.add_argument("--gate-mode", choices=["deterministic", "beta"], default="deterministic")
    parser.add_argument("--trust-floor", type=float, default=0.10)
    parser.add_argument("--trust-cap", type=float, default=0.95)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_costs = pd.read_csv(args.grid_cost_csv)
    datacenter_dfs = load_datacenter_dfs(str(args.data_dir))

    rows: list[dict] = []
    region_rows: list[dict] = []
    for region_states in args.region_state_grid:
        for seed_idx in range(args.seeds):
            instance, regions = make_region_grid_instance(
                seed=args.seed + seed_idx + 101 * region_states,
                grid_costs=grid_costs,
                datacenter_dfs=datacenter_dfs,
                n_arms=args.arms,
                queue_states=args.queue_states,
                region_states=region_states,
                op_states=args.op_states,
            )
            for display_name, policy, variant, gate_mode in STRATEGIES:
                row, _ = run_single_policy(
                    instance,
                    policy,
                    seed=args.seed + seed_idx + stable_strategy_offset(policy, variant, gate_mode),
                    rounds=args.rounds,
                    noise_level=args.noise_level,
                    transition_variant=variant,
                    gate_mode=gate_mode,
                    trust_floor=args.trust_floor,
                    trust_cap=args.trust_cap,
                )
                label = policy if policy in {"oracle", "state_thompson"} else f"{policy}_{variant}"
                if gate_mode == "beta":
                    label = f"{label}_beta"
                row["policy_label"] = label
                row["display_name"] = display_name
                row["experiment_id"] = "load_balance_region_grid"
                row["queue_states"] = args.queue_states
                row["grid_states"] = region_states
                row["op_states"] = args.op_states
                row["arms"] = args.arms
                rows.append(row)
            for order, region in regions.reset_index(drop=True).iterrows():
                region_rows.append(
                    {
                        "grid_states": region_states,
                        "seed_index": seed_idx,
                        "region_order": order,
                        "region_code": region["region_code"],
                        "region_name": region["region_name"],
                        "combined_grid_cost": region["combined_grid_cost"],
                        "carbon_norm": region["carbon_norm"],
                        "price_norm": region["price_norm"],
                        "mean_carbon_intensity_direct_gco2eq_per_kwh": region[
                            "mean_carbon_intensity_direct_gco2eq_per_kwh"
                        ],
                        "mean_electricity_price_usd_per_mwh": region["mean_electricity_price_usd_per_mwh"],
                    }
                )
            if args.progress:
                print(f"finished G={region_states} seed={seed_idx} S={instance.rewards.shape[1]}", flush=True)

    write_rows(out_dir / "load_balance_region_grid_results.csv", rows)
    write_group_summary(
        rows,
        ["queue_states", "grid_states", "op_states", "S", "context_noise_level", "gate_mode", "policy_label"],
        out_dir / "load_balance_region_grid_summary.csv",
    )
    pd.DataFrame(region_rows).drop_duplicates(["grid_states", "region_order", "region_code"]).to_csv(
        out_dir / "selected_regions.csv", index=False
    )
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "setting": "Region-grid load-balance",
                "arms": args.arms,
                "rounds": args.rounds,
                "seeds": args.seeds,
                "queue_states": args.queue_states,
                "region_state_grid": args.region_state_grid,
                "op_states": args.op_states,
                "noise_level": args.noise_level,
                "gate_mode": args.gate_mode,
                "strategies": [strategy[0] for strategy in STRATEGIES],
                "grid_cost_csv": str(args.grid_cost_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
