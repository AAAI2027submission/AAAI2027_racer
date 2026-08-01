"""Run event-driven TW on the region-grid load-balance state spaces."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))

from Event_driven_TW_varying_data_center_jobs import (  # noqa: E402
    BudgetEventDrivenTWPolicy,
    BudgetOracleWhittlePolicy,
    run_policy_budget,
)
from experiments.run_experiments import (  # noqa: E402
    ACTIVE,
    PASSIVE,
    apply_support_mask,
    gated_row_blend,
    load_datacenter_dfs,
    low_rank_transition,
    make_offline_transition_prior,
    row_normalize,
    threshold_transition,
    variant_has,
)
from experiments.run_region_grid_load_balance_sweep import (  # noqa: E402
    DATACENTER_DIR,
    GRID_COST_CSV,
    make_region_grid_instance,
    stable_strategy_offset,
)
from multi_armed_bandits_mdp_thompson_whittle_greedy import (  # noqa: E402
    Config,
    RMABEnvironment,
    compute_whittle_table,
)


DEFAULT_OUTPUT = ROOT / "docs/research/rmab_vm_outputs/load_balance_region_grid_event_driven_tw_v1"

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


def normalize_scores_01(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


class VariantTransitionPosterior:
    """Transition posterior with the same priors/gates as the context-noise suite."""

    def __init__(
        self,
        active_p: np.ndarray,
        passive_p: np.ndarray,
        support_mask: np.ndarray,
        transition_variant: str,
        gate_mode: str,
        rng: np.random.Generator,
        gate_scale_mult: float = 1.0,
        beta_gate_concentration: float = 20.0,
    ) -> None:
        self.transition_variant = transition_variant
        self.support_prior = variant_has(transition_variant, "support")
        self.threshold_prior = variant_has(transition_variant, "threshold")
        self.low_rank_prior = variant_has(transition_variant, "low_rank")
        self.offline_prior = variant_has(transition_variant, "offline")
        self.gated_prior = variant_has(transition_variant, "gated")
        self.support_mask = support_mask
        self.rng = rng
        self.gate_mode = gate_mode
        self.gate_scale_mult = gate_scale_mult
        self.beta_gate_concentration = beta_gate_concentration

        n_dc, n_states = active_p.shape[:2]
        base = np.full((n_dc, 2, n_states, n_states), 0.05)
        if self.support_prior:
            base = np.where(support_mask, 0.20, 1e-6)
        if self.offline_prior:
            true_p = np.stack([passive_p, active_p], axis=1)
            noisy_prior = make_offline_transition_prior(true_p, support_mask, self.support_prior)
            base += 1.50 * noisy_prior
        self.alpha = base.transpose(0, 2, 1, 3).astype(float)
        self.visit_counts = np.zeros((n_dc, n_states, 2), dtype=float)

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        n_dc, n_states, _, _ = self.alpha.shape
        p_old = np.zeros_like(self.alpha)
        for dc in range(n_dc):
            for state in range(n_states):
                for action in range(2):
                    p_old[dc, state, action] = rng.dirichlet(self.alpha[dc, state, action])

        p = p_old.transpose(0, 2, 1, 3)
        prior_p = row_normalize(self.alpha.transpose(0, 2, 1, 3))
        observed_rows = self.visit_counts.transpose(0, 2, 1)[..., None]
        gate_scale = self.gate_scale_mult * (2.0 + np.sqrt(float(n_states)))

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
                support_mask=self.support_mask if self.support_prior else None,
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
            low_rank = low_rank_transition(p, rank=min(3, max(1, n_states // 8)))
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
                p = row_normalize(p * np.where(self.support_mask, 1.0, 0.05))
            else:
                p = apply_support_mask(p, self.support_mask)
        return p[:, ACTIVE], p[:, PASSIVE]


class BudgetVariantEventDrivenTWPolicy(BudgetEventDrivenTWPolicy):
    """Budget event-driven TW with context-noise-suite policy scoring variants."""

    def __init__(
        self,
        policy_name: str,
        transition_variant: str,
        gate_mode: str,
        active_p: np.ndarray,
        passive_p: np.ndarray,
        support_mask: np.ndarray,
        n_dc: int,
        n_states: int,
        batch_size: int,
        cfg: Config,
        rng_seed: int,
        gate_scale_mult: float = 1.0,
        beta_gate_concentration: float = 20.0,
    ) -> None:
        super().__init__(n_dc, n_states, batch_size, cfg, rng_seed)
        self.policy_name = policy_name
        self.exp4_log_weights = np.zeros(3)
        self.exp4_expert_arms = np.zeros(3, dtype=int)
        self.trans_post = VariantTransitionPosterior(
            active_p=active_p,
            passive_p=passive_p,
            support_mask=support_mask,
            transition_variant=transition_variant,
            gate_mode=gate_mode,
            rng=self.rng,
            gate_scale_mult=gate_scale_mult,
            beta_gate_concentration=beta_gate_concentration,
        )

    def _local_scores(self, states: np.ndarray, t: int) -> np.ndarray:
        mu = self.reward_post.mu[self.arm_ids, states]
        std = np.sqrt(1.0 / np.maximum(self.reward_post.precision[self.arm_ids, states], 1e-9))
        return mu + 2.0 * np.sqrt(np.log(t + 2.0)) * std

    def select_arms(self, states: np.ndarray, t: int, budget: int) -> list[int]:
        if self.policy_name == "state_thompson":
            sampled_rewards = self.reward_post.sample(self.rng)
            scores = sampled_rewards[self.arm_ids, states]
            return list(np.argsort(scores)[-budget:][::-1])

        if t < self.cfg.mix_warmup_rounds:
            return [int((t + k) % self.n_dc) for k in range(budget)]

        interval_due = (t - self.last_replan_t) >= self.cfg.replan_interval
        new_state = any(int(states[dc]) not in self._visited[dc] for dc in range(self.n_dc))
        if interval_due or new_state:
            if new_state and not interval_due:
                self._event_replan_count += 1
            self._replan(t)

        whittle_raw = self._whittle_scores(states)
        horizon = max(80.0, 3.0 * self.n_states)
        exploration_weight = max(0.0, 1.0 - t / horizon)
        if self.policy_name == "tw":
            scores = whittle_raw
        elif self.policy_name == "local_ucb_tw":
            scores = normalize_scores_01(whittle_raw) + 0.55 * exploration_weight * normalize_scores_01(
                self._local_scores(states, t)
            )
        elif self.policy_name == "global_ucb_tw":
            scores = normalize_scores_01(whittle_raw) + 0.55 * exploration_weight * normalize_scores_01(
                self._global_ucb_score(t)
            )
        elif self.policy_name == "exp4":
            experts = np.vstack(
                [
                    normalize_scores_01(self._global_ucb_score(t)),
                    normalize_scores_01(self._local_scores(states, t)),
                    normalize_scores_01(whittle_raw),
                ]
            )
            self.exp4_expert_arms = np.argmax(experts, axis=1)
            weights = np.exp(self.exp4_log_weights - np.max(self.exp4_log_weights))
            weights /= np.sum(weights)
            scores = weights @ experts
        elif self.policy_name == "tm_tw_refined":
            trust_t = self._compute_trust_index(states, t)
            self.trust_history.append(trust_t)
            scores = trust_t * normalize_scores_01(whittle_raw) + (1.0 - trust_t) * normalize_scores_01(
                self._local_scores(states, t)
            )
        else:
            scores = normalize_scores_01(whittle_raw) + 0.35 * normalize_scores_01(self._local_scores(states, t))
        return list(np.argsort(scores)[-budget:][::-1])

    def update_multi(
        self,
        states: np.ndarray,
        chosen_arms: list[int],
        per_arm_rewards: np.ndarray,
        next_states: np.ndarray,
    ) -> None:
        super().update_multi(states, chosen_arms, per_arm_rewards, next_states)
        if self.policy_name == "exp4":
            expert_rewards = per_arm_rewards[self.exp4_expert_arms]
            scale = max(10.0, float(np.max(np.abs(expert_rewards))))
            self.exp4_log_weights += 0.15 * expert_rewards / scale


class BudgetVariantFullRecomputeTWPolicy(BudgetVariantEventDrivenTWPolicy):
    """Budget TW variant with interval-based full-state Whittle recomputation."""

    def _replan(self, t: int) -> None:
        sampled_rewards = self.reward_post.sample(self.rng)
        sampled_pa, sampled_pp = self.trans_post.sample(self.rng)
        new_w = compute_whittle_table(
            r_active_dc_state=sampled_rewards,
            p_active=sampled_pa,
            p_passive=sampled_pp,
            cfg=self.cfg,
        )
        if self.last_replan_t > -(10**8):
            denom = max(float(np.mean(np.abs(self.cached_W))), 1e-8)
            self.whittle_change = float(np.mean(np.abs(new_w - self.cached_W)) / denom)
        else:
            self.whittle_change = 1.0
        self.cached_W = new_w
        self.last_replan_t = t
        self._replan_counts += 1
        self._states_computed.append(self.n_dc * self.n_states)

    def select_arms(self, states: np.ndarray, t: int, budget: int) -> list[int]:
        if self.policy_name == "state_thompson":
            return super().select_arms(states, t, budget)
        if t < self.cfg.mix_warmup_rounds:
            return [int((t + k) % self.n_dc) for k in range(budget)]
        if (t - self.last_replan_t) >= self.cfg.replan_interval:
            self._replan(t)

        whittle_raw = self._whittle_scores(states)
        horizon = max(80.0, 3.0 * self.n_states)
        exploration_weight = max(0.0, 1.0 - t / horizon)
        if self.policy_name == "tw":
            scores = whittle_raw
        elif self.policy_name == "local_ucb_tw":
            scores = normalize_scores_01(whittle_raw) + 0.55 * exploration_weight * normalize_scores_01(
                self._local_scores(states, t)
            )
        elif self.policy_name == "global_ucb_tw":
            scores = normalize_scores_01(whittle_raw) + 0.55 * exploration_weight * normalize_scores_01(
                self._global_ucb_score(t)
            )
        elif self.policy_name == "exp4":
            experts = np.vstack(
                [
                    normalize_scores_01(self._global_ucb_score(t)),
                    normalize_scores_01(self._local_scores(states, t)),
                    normalize_scores_01(whittle_raw),
                ]
            )
            self.exp4_expert_arms = np.argmax(experts, axis=1)
            weights = np.exp(self.exp4_log_weights - np.max(self.exp4_log_weights))
            weights /= np.sum(weights)
            scores = weights @ experts
        elif self.policy_name == "tm_tw_refined":
            trust_t = self._compute_trust_index(states, t)
            self.trust_history.append(trust_t)
            scores = trust_t * normalize_scores_01(whittle_raw) + (1.0 - trust_t) * normalize_scores_01(
                self._local_scores(states, t)
            )
        else:
            scores = normalize_scores_01(whittle_raw) + 0.35 * normalize_scores_01(self._local_scores(states, t))
        return list(np.argsort(scores)[-budget:][::-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-cost-csv", type=Path, default=GRID_COST_CSV)
    parser.add_argument("--data-dir", type=Path, default=DATACENTER_DIR)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=250)
    parser.add_argument("--arms", type=int, default=8)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--queue-states", type=int, default=12)
    parser.add_argument("--region-state-grid", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--op-states", type=int, default=3)
    parser.add_argument("--computation-mode", choices=["event", "full"], default="event")
    parser.add_argument("--mix-trust-min", type=float, default=0.01)
    parser.add_argument("--mix-trust-max", type=float, default=0.98)
    parser.add_argument("--gate-scale-mult", type=float, default=1.0)
    parser.add_argument("--beta-gate-concentration", type=float, default=20.0)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Optional policy_label filter, e.g. tw_dense tm_tw_refined_gated_offline_beta.",
    )
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def policy_label(policy: str, variant: str, gate_mode: str) -> str:
    label = policy if policy in {"oracle", "state_thompson"} else f"{policy}_{variant}"
    if gate_mode == "beta":
        label = f"{label}_beta"
    return label


def make_policy(
    args: argparse.Namespace,
    policy: str,
    variant: str,
    gate_mode: str,
    instance,
    n_states: int,
    cfg: Config,
    seed: int,
):
    if policy == "oracle":
        return BudgetOracleWhittlePolicy(instance.rewards.T, instance.active_p, instance.passive_p, cfg)
    policy_cls = BudgetVariantFullRecomputeTWPolicy if args.computation_mode == "full" else BudgetVariantEventDrivenTWPolicy
    return policy_cls(
        policy_name=policy,
        transition_variant=variant,
        gate_mode=gate_mode,
        active_p=instance.active_p,
        passive_p=instance.passive_p,
        support_mask=instance.support_mask,
        n_dc=args.arms,
        n_states=n_states,
        batch_size=cfg.batch_size,
        cfg=cfg,
        rng_seed=seed,
        gate_scale_mult=args.gate_scale_mult,
        beta_gate_concentration=args.beta_gate_concentration,
    )


def run_one(
    args: argparse.Namespace,
    grid_costs: pd.DataFrame,
    datacenter_dfs: list[pd.DataFrame],
    region_states: int,
    seed_idx: int,
    display_name: str,
    policy_name: str,
    variant: str,
    gate_mode: str,
) -> dict:
    instance, regions = make_region_grid_instance(
        seed=args.seed + seed_idx + 101 * region_states,
        grid_costs=grid_costs,
        datacenter_dfs=datacenter_dfs,
        n_arms=args.arms,
        queue_states=args.queue_states,
        region_states=region_states,
        op_states=args.op_states,
    )
    n_states = instance.rewards.shape[1]
    cfg = Config(
        n_dc=args.arms,
        n_jobs_sample=n_states,
        n_rounds=args.rounds,
        n_sims=args.seeds,
        random_state=args.seed + seed_idx,
        replan_interval=25,
        mix_warmup_rounds=5,
        mix_eps_start=0.0,
        mix_eps_end=0.0,
        mix_trust_min=args.mix_trust_min,
        mix_trust_max=args.mix_trust_max,
        trans_prior_alpha=0.05,
        trans_structural_bias=0.2,
        vi_max_iters=80,
        binary_iters=8,
        whittle_tol=1e-2,
    )
    env = RMABEnvironment(
        reward_table=instance.rewards.T,
        p_active=instance.active_p,
        p_passive=instance.passive_p,
        reward_noise_std=0.0,
        rng_seed=args.seed + seed_idx + 500,
    )
    run_seed = args.seed + seed_idx + stable_strategy_offset(policy_name, variant, gate_mode)
    policy = make_policy(args, policy_name, variant, gate_mode, instance, n_states, cfg, run_seed)
    t0 = time.perf_counter()
    rewards, _ = run_policy_budget(env, policy, args.rounds, args.budget)
    walltime = time.perf_counter() - t0
    replan_calls = int(getattr(policy, "_replan_counts", 0))
    event_replans = int(getattr(policy, "_event_replan_count", 0))
    states_computed = int(np.sum(getattr(policy, "_states_computed", [])))
    baseline_states_total = max(float(max(replan_calls, 1) * args.arms * n_states), 1.0)
    return {
        "policy": policy_name,
        "transition_variant": variant,
        "gate_mode": gate_mode,
        "policy_label": policy_label(policy_name, variant, gate_mode),
        "display_name": display_name,
        "computation": "Full recompute TW" if args.computation_mode == "full" else "Event-Driven TW (M1+M3)",
        "queue_states": args.queue_states,
        "grid_states": region_states,
        "op_states": args.op_states,
        "S": n_states,
        "arms": args.arms,
        "budget": args.budget,
        "rounds": args.rounds,
        "seed_index": seed_idx,
        "avg_reward": float(np.mean(rewards)),
        "cum_reward": float(np.sum(rewards)),
        "walltime_seconds": walltime,
        "replan_calls": replan_calls,
        "event_replans": event_replans,
        "states_computed": states_computed,
        "computation_saving_pct": 100.0 * (1.0 - states_computed / baseline_states_total),
        "mix_trust_min": args.mix_trust_min,
        "mix_trust_max": args.mix_trust_max,
        "gate_scale_mult": args.gate_scale_mult,
        "beta_gate_concentration": args.beta_gate_concentration,
        "regions": ",".join(regions["region_code"].astype(str)),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    grid_costs = pd.read_csv(args.grid_cost_csv)
    datacenter_dfs = load_datacenter_dfs(str(args.data_dir))

    rows = []
    strategies = [
        strategy
        for strategy in STRATEGIES
        if args.strategies is None or policy_label(strategy[1], strategy[2], strategy[3]) in set(args.strategies)
    ]
    for region_states in args.region_state_grid:
        for seed_idx in range(args.seeds):
            for display_name, policy, variant, gate_mode in strategies:
                row = run_one(args, grid_costs, datacenter_dfs, region_states, seed_idx, display_name, policy, variant, gate_mode)
                rows.append(row)
                if args.progress:
                    print(
                        f"G={region_states} seed={seed_idx} {row['policy_label']} S={row['S']} "
                        f"reward={row['avg_reward']:.4f} save={row['computation_saving_pct']:.1f}% "
                        f"time={row['walltime_seconds']:.1f}s",
                        flush=True,
                    )

    detail = pd.DataFrame(rows)
    detail.to_csv(args.output / "event_driven_tw_region_grid_results.csv", index=False)
    summary = (
        detail.groupby(
            [
                "computation",
                "queue_states",
                "grid_states",
                "op_states",
                "S",
                "arms",
                "budget",
                "rounds",
                "policy_label",
                "display_name",
                "policy",
                "transition_variant",
                "gate_mode",
            ],
            as_index=False,
        )
        .agg(
            n=("seed_index", "count"),
            mean_avg_reward=("avg_reward", "mean"),
            mean_cum_reward=("cum_reward", "mean"),
            mean_walltime_seconds=("walltime_seconds", "mean"),
            mean_replan_calls=("replan_calls", "mean"),
            mean_event_replans=("event_replans", "mean"),
            mean_states_computed=("states_computed", "mean"),
            mean_computation_saving_pct=("computation_saving_pct", "mean"),
        )
    )
    summary.to_csv(args.output / "event_driven_tw_region_grid_summary.csv", index=False)
    (args.output / "run_config.json").write_text(
        json.dumps(
            {
                "setting": "Region-grid event-driven TW",
                "arms": args.arms,
                "budget": args.budget,
                "rounds": args.rounds,
                "seeds": args.seeds,
                "queue_states": args.queue_states,
                "region_state_grid": args.region_state_grid,
                "op_states": args.op_states,
                "state_sizes": [args.queue_states * g * args.op_states for g in args.region_state_grid],
                "computation": "Full recompute TW" if args.computation_mode == "full" else "Event-Driven TW (M1+M3)",
                "computation_mode": args.computation_mode,
                "mix_trust_min": args.mix_trust_min,
                "mix_trust_max": args.mix_trust_max,
                "gate_scale_mult": args.gate_scale_mult,
                "beta_gate_concentration": args.beta_gate_concentration,
                "strategies": [strategy[0] for strategy in strategies],
                "policy_labels": [policy_label(strategy[1], strategy[2], strategy[3]) for strategy in strategies],
                "source_style": "Event_driven_TW_varying_data_center_jobs.py",
                "grid_cost_csv": str(args.grid_cost_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
