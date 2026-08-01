"""
Event-Driven TW — sensitivity to n_dc and n_jobs_sample.

Budget = 2: two datacenters are selected (activated) every round.

For each (n_dc, n_jobs_sample) combination in a user-defined grid, this script
runs three policies with a per-round activation budget of 2 and records:

  Oracle Whittle          — true model, upper bound
  Trust-Mixed TW          — original baseline
  Event-Driven TW (M1+M3) — lazy + event-driven replan (from improved_tw)

Outputs
-------
  results/plots/ed_tw_varying_ndc_reward.png
  results/plots/ed_tw_varying_ndc_regret.png
  results/plots/ed_tw_varying_ndc_walltime.png
  results/ed_tw_varying_summary.csv
"""

from __future__ import annotations

import glob
import os
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multi_armed_bandits_mdp_thompson_whittle_greedy import (
    Config,
    build_reward_table,
    build_true_transitions,
    RMABEnvironment,
    OracleWhittlePolicy,
    TrustMixedThompsonWhittlePolicy,
    normalize_scores_01,
)
from multi_armed_bandits_improved_tw import EventDrivenTWPolicy


# ---------------------------------------------------------------------------
# Experiment grid — edit these to change the sweep
# ---------------------------------------------------------------------------
N_DC_VALUES       = [3, 5, 8, 10]       # number of datacenters (arms)
N_JOBS_VALUES     = [20, 40, 60, 80]    # n_jobs_sample (state-space size)
BUDGET            = 2                    # arms activated per round
N_SIMS            = 4
N_ROUNDS          = 600
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Multi-arm environment helpers
# ---------------------------------------------------------------------------

def step_multi(
    env: RMABEnvironment,
    states: np.ndarray,
    chosen_arms: List[int],
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Step the environment with `budget` active arms.

    Returns
    -------
    next_states      : np.ndarray, shape (n_dc,)
    total_reward     : float  — sum of rewards from all active arms
    per_arm_rewards  : np.ndarray, shape (n_dc,)  — reward per arm (0 for passive)
    """
    chosen_set = set(chosen_arms)
    next_states = np.zeros_like(states)
    per_arm_rewards = np.zeros(env.n_dc, dtype=float)
    total_reward = 0.0

    for dc in range(env.n_dc):
        s = int(states[dc])
        is_active = dc in chosen_set
        p = env.Pa[dc, s] if is_active else env.Pp[dc, s]
        next_states[dc] = int(env.rng.choice(env.n_states, p=p))
        if is_active:
            noise = env.rng.normal(0.0, env.reward_noise_std) if env.reward_noise_std > 0 else 0.0
            per_arm_rewards[dc] = float(env.R[s, dc] + noise)
            total_reward += per_arm_rewards[dc]

    return next_states, total_reward, per_arm_rewards


def run_policy_budget(
    env: RMABEnvironment,
    policy: "BudgetPolicy",
    n_rounds: int,
    budget: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a BudgetPolicy for n_rounds, selecting `budget` arms each round."""
    states = env.reset()
    rewards = np.zeros(n_rounds, dtype=float)
    chosen = np.zeros((n_rounds, budget), dtype=int)

    for t in range(n_rounds):
        arms = policy.select_arms(states, t, budget)
        next_states, reward, per_arm_rewards = step_multi(env, states, arms)
        policy.update_multi(states, arms, per_arm_rewards, next_states)

        rewards[t] = reward
        chosen[t] = arms
        states = next_states

    return rewards, chosen


# ---------------------------------------------------------------------------
# Budget-aware policy wrappers
# ---------------------------------------------------------------------------

class BudgetOracleWhittlePolicy(OracleWhittlePolicy):
    """Oracle Whittle with budget > 1: picks top-k arms by Whittle index."""

    def select_arms(self, states: np.ndarray, _t: int, budget: int) -> List[int]:
        scores = np.array([self.W[states[dc], dc] for dc in range(len(states))], dtype=float)
        return list(np.argsort(scores)[-budget:][::-1])

    def update_multi(self, *_args, **_kwargs) -> None:
        pass  # oracle knows the true model — no learning needed


class BudgetTrustMixedTWPolicy(TrustMixedThompsonWhittlePolicy):
    """
    Trust-Mixed TW with budget > 1.

    select_arms returns the top-`budget` arms by mixed score.
    update_multi updates the reward posterior for each active arm and the
    transition posterior with all active arms marked simultaneously.
    """

    def select_arms(self, states: np.ndarray, t: int, budget: int) -> List[int]:
        if t < self.cfg.mix_warmup_rounds:
            return [int((t + k) % self.n_dc) for k in range(budget)]

        eps_t = self._linear_decay(
            t=t,
            start=self.cfg.mix_eps_start,
            end=self.cfg.mix_eps_end,
            horizon=self.cfg.mix_eps_decay_rounds,
        )
        if self.rng.random() < eps_t:
            return list(self.rng.choice(self.n_dc, size=budget, replace=False))

        self._maybe_replan(t)

        whittle_raw = self._whittle_scores(states)
        trust_t     = self._compute_trust_index(states, t)
        greedy_raw  = self._greedy_score(states, t)
        mixed_score = (
            trust_t       * normalize_scores_01(whittle_raw) +
            (1 - trust_t) * normalize_scores_01(greedy_raw)
        )
        self.trust_history.append(trust_t)
        return list(np.argsort(mixed_score)[-budget:][::-1])

    def update_multi(
        self,
        states: np.ndarray,
        chosen_arms: List[int],
        per_arm_rewards: np.ndarray,
        next_states: np.ndarray,
    ) -> None:
        # Reward posterior: one update per active arm
        for arm in chosen_arms:
            s_sel = int(states[arm])
            self.reward_post.update((arm, s_sel), float(per_arm_rewards[arm]))

        # Transition posterior: mark all active arms at once, update in one pass
        self._actions_buf.fill(0)
        for arm in chosen_arms:
            self._actions_buf[arm] = 1
        self.trans_post.alpha[self.arm_ids, states, self._actions_buf, next_states] += 1.0
        self.trans_post.visit_counts[self.arm_ids, states, self._actions_buf] += 1.0

        # Per-arm pull / reward tracking (used by trust index)
        for arm in chosen_arms:
            self.arm_pull_counts[arm] += 1.0
            self.arm_reward_sums[arm] += float(per_arm_rewards[arm])


class BudgetEventDrivenTWPolicy(EventDrivenTWPolicy):
    """
    Event-Driven TW (M1+M3) with budget > 1.

    Inherits lazy replan and new-state event trigger from EventDrivenTWPolicy.
    select_arms / update_multi extend the base to handle multiple active arms.
    """

    def select_arms(self, states: np.ndarray, t: int, budget: int) -> List[int]:
        if t < self.cfg.mix_warmup_rounds:
            return [int((t + k) % self.n_dc) for k in range(budget)]

        eps_t = self._linear_decay(
            t=t,
            start=self.cfg.mix_eps_start,
            end=self.cfg.mix_eps_end,
            horizon=self.cfg.mix_eps_decay_rounds,
        )
        if self.rng.random() < eps_t:
            return list(self.rng.choice(self.n_dc, size=budget, replace=False))

        interval_due = (t - self.last_replan_t) >= self.cfg.replan_interval
        new_state    = any(int(states[dc]) not in self._visited[dc] for dc in range(self.n_dc))
        if interval_due or new_state:
            if new_state and not interval_due:
                self._event_replan_count += 1
            self._replan(t)

        whittle_raw = self._whittle_scores(states)
        trust_t     = self._compute_trust_index(states, t)
        greedy_raw  = self._greedy_score(states, t)
        mixed_score = (
            trust_t       * normalize_scores_01(whittle_raw) +
            (1 - trust_t) * normalize_scores_01(greedy_raw)
        )
        self.trust_history.append(trust_t)
        return list(np.argsort(mixed_score)[-budget:][::-1])

    def update_multi(
        self,
        states: np.ndarray,
        chosen_arms: List[int],
        per_arm_rewards: np.ndarray,
        next_states: np.ndarray,
    ) -> None:
        # Reward posterior: one update per active arm
        for arm in chosen_arms:
            s_sel = int(states[arm])
            self.reward_post.update((arm, s_sel), float(per_arm_rewards[arm]))

        # Transition posterior: all active arms marked simultaneously
        self._actions_buf.fill(0)
        for arm in chosen_arms:
            self._actions_buf[arm] = 1
        self.trans_post.alpha[self.arm_ids, states, self._actions_buf, next_states] += 1.0
        self.trans_post.visit_counts[self.arm_ids, states, self._actions_buf] += 1.0

        # Per-arm tracking
        for arm in chosen_arms:
            self.arm_pull_counts[arm] += 1.0
            self.arm_reward_sums[arm] += float(per_arm_rewards[arm])

        # Update visited sets (for lazy / event-driven logic)
        for dc in range(self.n_dc):
            self._visited[dc].add(int(states[dc]))
            self._visited[dc].add(int(next_states[dc]))


# Type alias for run_policy_budget
BudgetPolicy = BudgetOracleWhittlePolicy  # any of the three above


# ---------------------------------------------------------------------------
# Per-config runner
# ---------------------------------------------------------------------------

def run_config(
    n_dc: int,
    n_jobs: int,
    all_dfs: List[pd.DataFrame],
    n_sims: int,
    n_rounds: int,
    budget: int,
    base_seed: int = 42,
) -> Dict:
    cfg = Config(
        n_dc=n_dc,
        n_jobs_sample=n_jobs,
        n_rounds=n_rounds,
        n_sims=n_sims,
        random_state=base_seed,
    )

    raw_dfs = all_dfs[:n_dc]

    cum_rewards: np.ndarray = np.zeros(n_rounds)
    wall_times:  List[float] = []
    avg_rewards: List[float] = []

    replan_counts_list:   List[int] = []
    states_computed_list: List[int] = []
    event_counts_list:    List[int] = []

    for sim in range(n_sims):
        sim_seed = base_seed + sim * 13

        R = build_reward_table(
            raw_dfs=raw_dfs,
            random_state=sim_seed,
            n_jobs_sample=n_jobs,
            n_dc=n_dc,
            batch_size=cfg.batch_size,
            lookahead_size=cfg.lookahead_size,
        )
        p_active_true, p_passive_true = build_true_transitions(
            n_states=n_jobs,
            batch_size=cfg.batch_size,
            n_dc=n_dc,
        )

        env    = RMABEnvironment(R, p_active_true, p_passive_true, 0.0, sim_seed + 500)
        policy = BudgetEventDrivenTWPolicy(n_dc, n_jobs, cfg.batch_size, cfg, sim_seed + 401)

        t0 = time.perf_counter()
        rewards, _ = run_policy_budget(env, policy, n_rounds, budget)
        wall_times.append(time.perf_counter() - t0)

        cum_rewards += rewards
        avg_rewards.append(float(np.mean(rewards)))

        replan_counts_list.append(policy._replan_counts)
        states_computed_list.append(int(np.sum(policy._states_computed)))
        event_counts_list.append(policy._event_replan_count)

    baseline_states_total = float(np.mean(replan_counts_list)) * n_dc * n_jobs
    avg_computed          = float(np.mean(states_computed_list))
    saving_pct            = 100.0 * (1.0 - avg_computed / max(baseline_states_total, 1.0))

    return {
        "n_dc":                   n_dc,
        "n_jobs":                 n_jobs,
        "budget":                 budget,
        "avg_reward_ed":          float(np.mean(avg_rewards)),
        "avg_walltime_ed_s":      float(np.mean(wall_times)),
        "avg_replan_calls":       float(np.mean(replan_counts_list)),
        "avg_states_computed":    avg_computed,
        "computation_saving_pct": saving_pct,
        "avg_event_replans":      float(np.mean(event_counts_list)),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    row_labels: List,
    col_labels: List,
    title: str,
    fmt: str = ".3f",
    cmap: str = "YlOrRd",
) -> None:
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels([str(v) for v in col_labels], fontsize=11)
    ax.set_yticklabels([str(v) for v in row_labels], fontsize=11)
    ax.set_xlabel("n_jobs_sample", fontsize=12)
    ax.set_ylabel("n_dc", fontsize=12)
    ax.set_title(title, fontsize=12)
    for r in range(len(row_labels)):
        for c in range(len(col_labels)):
            ax.text(c, r, format(data[r, c], fmt),
                    ha="center", va="center", fontsize=9, color="black")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs("event_driven_TW/plots", exist_ok=True)

    all_files = sorted(glob.glob("datacenter_with_metrics/datacenter_*_with_metrics.csv"))
    max_dc_needed = max(N_DC_VALUES)
    if len(all_files) < max_dc_needed:
        raise RuntimeError(
            f"Need {max_dc_needed} datacenter files but only found {len(all_files)}"
        )
    all_dfs = [pd.read_csv(fp) for fp in all_files[:max_dc_needed]]
    print(f"Loaded {len(all_dfs)} datacenter files.")

    total_configs = len(N_DC_VALUES) * len(N_JOBS_VALUES)
    print(f"Budget={BUDGET} | {total_configs} configs × {N_SIMS} sims × {N_ROUNDS} rounds\n")
    print(f"{'n_dc':>5} {'n_jobs':>7} {'R_ed':>9} {'Save%':>7} {'t_ed(s)':>9}")
    print("-" * 50)

    rows = []
    for n_dc in N_DC_VALUES:
        for n_jobs in N_JOBS_VALUES:
            res = run_config(
                n_dc=n_dc,
                n_jobs=n_jobs,
                all_dfs=all_dfs,
                n_sims=N_SIMS,
                n_rounds=N_ROUNDS,
                budget=BUDGET,
            )
            rows.append(res)
            print(
                f"{n_dc:>5} {n_jobs:>7} "
                f"{res['avg_reward_ed']:>9.4f} "
                f"{res['computation_saving_pct']:>7.1f} "
                f"{res['avg_walltime_ed_s']:>9.1f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    csv_path = "event_driven_TW/ed_tw_varying_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Build heatmap arrays (rows = n_dc, cols = n_jobs)
    ndc_idx   = {v: i for i, v in enumerate(N_DC_VALUES)}
    njobs_idx = {v: i for i, v in enumerate(N_JOBS_VALUES)}
    shape = (len(N_DC_VALUES), len(N_JOBS_VALUES))

    hm_reward_ed   = np.zeros(shape)
    hm_walltime_ed = np.zeros(shape)
    hm_saving      = np.zeros(shape)

    for res in rows:
        r, c = ndc_idx[res["n_dc"]], njobs_idx[res["n_jobs"]]
        hm_reward_ed[r, c]   = res["avg_reward_ed"]
        hm_walltime_ed[r, c] = res["avg_walltime_ed_s"]
        hm_saving[r, c]      = res["computation_saving_pct"]

    # Plot 1: reward heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    heatmap(ax, hm_reward_ed, N_DC_VALUES, N_JOBS_VALUES,
            f"Event-Driven TW — avg reward (budget={BUDGET})", cmap="Oranges")
    plt.tight_layout()
    out1 = "event_driven_TW/plots/ed_tw_varying_ndc_reward.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved: {out1}")
    plt.close()

    # Plot 2: wall-time and computation saving
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    heatmap(axes[0], hm_walltime_ed, N_DC_VALUES, N_JOBS_VALUES,
            "Event-Driven TW — wall time (s)", fmt=".1f", cmap="PuBu")
    heatmap(axes[1], hm_saving, N_DC_VALUES, N_JOBS_VALUES,
            "Event-Driven TW — computation saving (%)", fmt=".1f", cmap="YlGn")
    fig.suptitle(f"Efficiency vs (n_dc, n_jobs_sample) — budget={BUDGET}", fontsize=14, y=1.02)
    plt.tight_layout()
    out2 = "event_driven_TW/plots/ed_tw_varying_ndc_efficiency.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
