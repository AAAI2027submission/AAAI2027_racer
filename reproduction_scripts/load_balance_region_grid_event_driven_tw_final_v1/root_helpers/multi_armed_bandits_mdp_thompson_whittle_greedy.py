"""
Slide-20 TODO solution: Thompson + Whittle under unknown reward/transition.

This script addresses the open question:
  "If reward and transition are unknown, can Whittle-style structure improve
   over black-box learning?"

Environment
-----------
Independent-state RMAB with one activation budget per round:
  - N_DC datacenters (arms), each with its own queue-position state s_dc.
  - If selected: reward = R[s_dc, dc], state follows active transition.
  - If not selected: reward = 0, state follows passive transition.

True rewards are generated from sampled VM traces (same reward logic as the
existing MDP scripts), while the learner does not know reward/transition.

Policies compared
-----------------
1) Thompson-Whittle (proposed):
   - Reward posterior: Gaussian per (dc, state) for active reward.
   - Transition posterior: Dirichlet per (dc, state, action).
   - At each replan step, sample a model and compute Whittle indices.
2) Trust-Mixed Thompson-Whittle (new):
   - Blends greedy and sampled Whittle scores via a trust index.
   - Trust increases with rounds and posterior confidence.
3) State Thompson (myopic):
   - Gaussian posterior per (dc, state), no transition planning/index.
4) Black-box Thompson (binary feedback):
   - Beta-Bernoulli posterior per arm only (ignores state/transition structure).
   - Uses y=1(reward>0), matching the binary-style Thompson setting often used
     in earlier experiments.
5) Oracle Whittle:
   - Uses true reward + true transition as an upper-bound benchmark.

Outputs
-------
- results/plots/thompson_whittle_greedy_vs_blackbox.png
- results/plots/thompson_whittle_greedy_arm_selection.png
- results/thompson_whittle_greedy_summary.csv
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class Config:
    n_dc: int = 5
    n_jobs_sample: int = 40
    batch_size: int = 5
    lookahead_size: int = 10
    gamma: float = 0.9
    vi_theta: float = 1e-4
    vi_max_iters: int = 120
    binary_iters: int = 10
    whittle_tol: float = 5e-3
    n_rounds: int = 600
    n_sims: int = 4
    replan_interval: int = 10
    transition_noise: float = 0.0
    reward_noise_std: float = 0.0
    reward_prior_mean: float = 0.0
    reward_prior_var: float = 25.0
    reward_obs_var: float = 4.0
    trans_prior_alpha: float = 0.1
    trans_structural_bias: float = 0.9
    tw_warmup_rounds: int = 20
    tw_eps_start: float = 0.20
    tw_eps_end: float = 0.02
    tw_eps_decay_rounds: int = 400
    mix_warmup_rounds: int = 5
    mix_trust_warmup_rounds: int = 10
    mix_trust_ramp_rounds: int = 450
    mix_trust_min: float = 0.01
    mix_trust_max: float = 0.98
    mix_trust_conf_weight: float = 0.45
    mix_trust_progress_weight: float = 0.25
    mix_trust_stability_weight: float = 0.30
    mix_whittle_change_scale: float = 0.20
    mix_trust_switch_round: int = 120
    mix_trust_switch_temp: float = 18.0
    mix_reward_conf_scale: float = 6.0
    mix_trans_conf_scale: float = 8.0
    mix_global_conf_scale: float = 10.0
    mix_global_ucb_coef: float = 4.0
    mix_global_prior_count: float = 1.0
    mix_greedy_global_weight: float = 0.95
    mix_greedy_global_decay_rounds: int = 180
    mix_eps_start: float = 0.00
    mix_eps_end: float = 0.00
    mix_eps_decay_rounds: int = 1
    mix_greedy_bonus_coef: float = 0.8
    random_state: int = 42

def build_reward_table(
    raw_dfs: List[pd.DataFrame],
    random_state: int,
    n_jobs_sample: int,
    n_dc: int,
    batch_size: int,
    lookahead_size: int,
) -> np.ndarray:
    """
    Build reward table R[s, dc] from sampled jobs.
    """
    powers, corehrs, ch_norms, is_ints = [], [], [], []

    for dc, df in enumerate(raw_dfs[:n_dc]):
        df_s = df.sample(
            n=min(n_jobs_sample, len(df)),
            random_state=random_state + dc + 1,
        ).reset_index(drop=True)

        corehour = df_s["corehour"].values
        ch_norm = (corehour - corehour.min()) / max(corehour.max() - corehour.min(), 1e-8)
        power = df_s["avgcpu"].values * ch_norm
        is_int = df_s["vmcategory"].values == "Interactive"

        powers.append(power)
        corehrs.append(corehour)
        ch_norms.append(ch_norm)
        is_ints.append(is_int)

    def idxs(s: int, size: int) -> List[int]:
        return [(s + i) % n_jobs_sample for i in range(size)]

    def reward(s: int, dc: int) -> float:
        default_batch = idxs(s, batch_size)
        pool = idxs(s, lookahead_size)
        sorted_pool = sorted(pool, key=lambda i: corehrs[dc][i])
        selected = sorted_pool[:batch_size]
        delayed = sorted_pool[batch_size:]

        saving = sum(powers[dc][i] for i in default_batch) - sum(powers[dc][i] for i in selected)
        penalty = sum(10.0 * ch_norms[dc][i] for i in delayed if is_ints[dc][i])
        return float(saving - penalty)

    return np.array([[reward(s, dc) for dc in range(n_dc)] for s in range(n_jobs_sample)], dtype=float)


def build_true_transitions(
    n_states: int,
    batch_size: int,
    n_dc: int,
    noise: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return true transition tensors:
      P_active[dc, s, s']
      P_passive[dc, s, s']
    """
    p_active = np.zeros((n_dc, n_states, n_states), dtype=float)
    p_passive = np.zeros((n_dc, n_states, n_states), dtype=float)

    for dc in range(n_dc):
        for s in range(n_states):
            s_next = (s + batch_size) % n_states
            p_active[dc, s, s_next] = 1.0
            p_passive[dc, s, s] = 1.0

    if noise > 0.0:
        uniform = np.ones((n_states,), dtype=float) / n_states
        p_active = (1.0 - noise) * p_active + noise * uniform[np.newaxis, np.newaxis, :]
        p_passive = (1.0 - noise) * p_passive + noise * uniform[np.newaxis, np.newaxis, :]

    return p_active, p_passive


class RMABEnvironment:
    def __init__(
        self,
        reward_table: np.ndarray,
        p_active: np.ndarray,
        p_passive: np.ndarray,
        reward_noise_std: float,
        rng_seed: int,
    ) -> None:
        self.R = reward_table
        self.Pa = p_active
        self.Pp = p_passive
        self.n_states, self.n_dc = self.R.shape
        self.reward_noise_std = reward_noise_std
        self.rng = np.random.default_rng(rng_seed)

    def reset(self) -> np.ndarray:
        return np.zeros((self.n_dc,), dtype=int)

    def step(self, states: np.ndarray, chosen_arm: int) -> Tuple[np.ndarray, float]:
        next_states = np.zeros_like(states)
        reward = 0.0
        for dc in range(self.n_dc):
            s = int(states[dc])
            is_active = int(dc == chosen_arm)
            p = self.Pa[dc, s] if is_active else self.Pp[dc, s]
            next_states[dc] = int(self.rng.choice(self.n_states, p=p))
            if is_active:
                mean_r = self.R[s, dc]
                noise = self.rng.normal(0.0, self.reward_noise_std) if self.reward_noise_std > 0 else 0.0
                reward = float(mean_r + noise)
        return next_states, reward


def solve_subsidy_mdp(
    r_active: np.ndarray,
    p_active: np.ndarray,
    p_passive: np.ndarray,
    lam: float,
    gamma: float,
    theta: float,
    max_iters: int,
    v_init: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solve single-arm subsidy MDP:
      V(s) = max( r_active(s) + gamma * E[V(s'|active)],
                  lam + gamma * E[V(s'|passive)] )
    """
    n_states = r_active.shape[0]
    V = np.zeros((n_states,), dtype=float) if v_init is None else v_init.copy()

    for _ in range(max_iters):
        q_active = r_active + gamma * (p_active @ V)
        q_passive = lam + gamma * (p_passive @ V)
        new_V = np.maximum(q_active, q_passive)
        if np.max(np.abs(new_V - V)) < theta:
            return new_V
        V = new_V

    return V


def compute_whittle_index_for_state(
    s: int,
    r_active: np.ndarray,
    p_active: np.ndarray,
    p_passive: np.ndarray,
    gamma: float,
    vi_theta: float,
    vi_max_iters: int,
    binary_iters: int,
    whittle_tol: float,
    lam_lo: float,
    lam_hi: float,
) -> float:
    """
    Binary search lambda where active/passive are indifferent at state s.
    """
    lo, hi = lam_lo, lam_hi
    V = np.zeros_like(r_active)
    for _ in range(binary_iters):
        if hi - lo < whittle_tol:
            break
        mid = 0.5 * (lo + hi)
        V = solve_subsidy_mdp(
            r_active=r_active,
            p_active=p_active,
            p_passive=p_passive,
            lam=mid,
            gamma=gamma,
            theta=vi_theta,
            max_iters=vi_max_iters,
            v_init=V,
        )
        q_active = r_active[s] + gamma * float(np.dot(p_active[s], V))
        q_passive = mid + gamma * float(np.dot(p_passive[s], V))
        if q_active >= q_passive:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def compute_whittle_table(
    r_active_dc_state: np.ndarray,
    p_active: np.ndarray,
    p_passive: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """
    Returns W[state, dc].
    """
    n_dc, n_states = r_active_dc_state.shape
    W = np.zeros((n_states, n_dc), dtype=float)

    for dc in range(n_dc):
        r_dc = r_active_dc_state[dc]
        lam_lo = float(r_dc.min() - abs(r_dc.min()) - 1.0)
        lam_hi = float(r_dc.max() / max(1.0 - cfg.gamma, 1e-6) + 1.0)
        for s in range(n_states):
            W[s, dc] = compute_whittle_index_for_state(
                s=s,
                r_active=r_dc,
                p_active=p_active[dc],
                p_passive=p_passive[dc],
                gamma=cfg.gamma,
                vi_theta=cfg.vi_theta,
                vi_max_iters=cfg.vi_max_iters,
                binary_iters=cfg.binary_iters,
                whittle_tol=cfg.whittle_tol,
                lam_lo=lam_lo,
                lam_hi=lam_hi,
            )
    return W


class GaussianPosterior:
    """
    Independent Gaussian posterior over means with known observation variance.
    """
    def __init__(self, shape: Tuple[int, ...], prior_mean: float, prior_var: float, obs_var: float) -> None:
        self.mu = np.full(shape, prior_mean, dtype=float)
        self.precision = np.full(shape, 1.0 / prior_var, dtype=float)
        self.obs_precision = 1.0 / obs_var

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        std = np.sqrt(1.0 / self.precision)
        return rng.normal(loc=self.mu, scale=std)

    def update(self, index: Tuple[int, ...], reward: float) -> None:
        old_prec = self.precision[index]
        old_mu = self.mu[index]
        new_prec = old_prec + self.obs_precision
        new_mu = (old_prec * old_mu + self.obs_precision * reward) / new_prec
        self.precision[index] = new_prec
        self.mu[index] = new_mu


class TransitionPosterior:
    """
    Dirichlet posterior for transitions per (dc, state, action).
    action=0 passive, action=1 active.
    """
    def __init__(
        self,
        n_dc: int,
        n_states: int,
        batch_size: int,
        prior_alpha: float,
        structural_bias: float,
    ) -> None:
        self.alpha = np.full((n_dc, n_states, 2, n_states), prior_alpha, dtype=float)
        self.visit_counts = np.zeros((n_dc, n_states, 2), dtype=float)

        # Weak structural prior toward expected queue dynamics.
        for dc in range(n_dc):
            for s in range(n_states):
                s_active_next = (s + batch_size) % n_states
                self.alpha[dc, s, 1, s_active_next] += structural_bias
                self.alpha[dc, s, 0, s] += structural_bias

    def sample(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        n_dc, n_states = self.alpha.shape[:2]
        p_active = np.zeros((n_dc, n_states, n_states), dtype=float)
        p_passive = np.zeros((n_dc, n_states, n_states), dtype=float)
        for dc in range(n_dc):
            for s in range(n_states):
                p_passive[dc, s] = rng.dirichlet(self.alpha[dc, s, 0])
                p_active[dc, s] = rng.dirichlet(self.alpha[dc, s, 1])
        return p_active, p_passive

    def update(self, dc: int, state: int, action: int, next_state: int) -> None:
        self.alpha[dc, state, action, next_state] += 1.0
        self.visit_counts[dc, state, action] += 1.0


class Policy:
    def select_arm(self, states: np.ndarray, t: int) -> int:
        raise NotImplementedError

    def update(self, states: np.ndarray, chosen_arm: int, reward: float, next_states: np.ndarray) -> None:
        return


def normalize_scores_01(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalize scores into [0, 1] for stable score fusion.
    """
    s_min = float(np.min(scores))
    s_max = float(np.max(scores))
    if s_max - s_min < 1e-12:
        return np.zeros_like(scores, dtype=float)
    return (scores - s_min) / (s_max - s_min)


class OracleWhittlePolicy(Policy):
    def __init__(self, reward_table: np.ndarray, p_active: np.ndarray, p_passive: np.ndarray, cfg: Config) -> None:
        self.W = compute_whittle_table(
            r_active_dc_state=reward_table.T,
            p_active=p_active,
            p_passive=p_passive,
            cfg=cfg,
        )

    def select_arm(self, states: np.ndarray, t: int) -> int:
        scores = np.array([self.W[states[dc], dc] for dc in range(len(states))], dtype=float)
        return int(np.argmax(scores))


class ThompsonWhittlePolicy(Policy):
    def __init__(self, n_dc: int, n_states: int, batch_size: int, cfg: Config, rng_seed: int) -> None:
        self.cfg = cfg
        self.n_dc = n_dc
        self.n_states = n_states
        self.arm_ids = np.arange(n_dc, dtype=int)
        self.rng = np.random.default_rng(rng_seed)
        self.reward_post = GaussianPosterior(
            shape=(n_dc, n_states),
            prior_mean=cfg.reward_prior_mean,
            prior_var=cfg.reward_prior_var,
            obs_var=cfg.reward_obs_var,
        )
        self.trans_post = TransitionPosterior(
            n_dc=n_dc,
            n_states=n_states,
            batch_size=batch_size,
            prior_alpha=cfg.trans_prior_alpha,
            structural_bias=cfg.trans_structural_bias,
        )
        self._actions_buf = np.zeros((n_dc,), dtype=int)
        self.cached_W = np.zeros((n_states, n_dc), dtype=float)
        self.whittle_change = 1.0
        self.last_replan_t = -10**9

    @staticmethod
    def _linear_decay(t: int, start: float, end: float, horizon: int) -> float:
        decay_frac = max(0.0, 1.0 - (t / max(horizon, 1)))
        return end + (start - end) * decay_frac

    def _maybe_replan(self, t: int) -> None:
        if (t - self.last_replan_t) >= self.cfg.replan_interval:
            self._replan(t)

    def _whittle_scores(self, states: np.ndarray) -> np.ndarray:
        return self.cached_W[states, self.arm_ids]

    def _replan(self, t: int) -> None:
        sampled_rewards = self.reward_post.sample(self.rng)
        sampled_pa, sampled_pp = self.trans_post.sample(self.rng)
        new_W = compute_whittle_table(
            r_active_dc_state=sampled_rewards,
            p_active=sampled_pa,
            p_passive=sampled_pp,
            cfg=self.cfg,
        )
        if self.last_replan_t > -10**8:
            denom = max(float(np.mean(np.abs(self.cached_W))), 1e-8)
            self.whittle_change = float(np.mean(np.abs(new_W - self.cached_W)) / denom)
        else:
            self.whittle_change = 1.0
        self.cached_W = new_W
        self.last_replan_t = t

    def select_arm(self, states: np.ndarray, t: int) -> int:
        # Early round-robin warmup ensures each arm is observed at least a few times.
        if t < self.cfg.tw_warmup_rounds:
            return int(t % self.n_dc)

        # Decaying epsilon exploration reduces the chance of early lock-in.
        eps_t = self._linear_decay(
            t=t,
            start=self.cfg.tw_eps_start,
            end=self.cfg.tw_eps_end,
            horizon=self.cfg.tw_eps_decay_rounds,
        )
        if self.rng.random() < eps_t:
            return int(self.rng.integers(0, self.n_dc))

        self._maybe_replan(t)
        return int(np.argmax(self._whittle_scores(states)))

    def update(self, states: np.ndarray, chosen_arm: int, reward: float, next_states: np.ndarray) -> None:
        s_sel = int(states[chosen_arm])
        self.reward_post.update((chosen_arm, s_sel), reward)

        self._actions_buf.fill(0)
        self._actions_buf[chosen_arm] = 1
        self.trans_post.alpha[self.arm_ids, states, self._actions_buf, next_states] += 1.0
        self.trans_post.visit_counts[self.arm_ids, states, self._actions_buf] += 1.0


class TrustMixedThompsonWhittlePolicy(ThompsonWhittlePolicy):
    """
    Mixed strategy:
      score = trust_t * WhittleScore + (1 - trust_t) * GreedyScore

    trust_t is dynamic and increases with:
      1) round progress
      2) posterior confidence of reward/transition at current states
    """

    def __init__(self, n_dc: int, n_states: int, batch_size: int, cfg: Config, rng_seed: int) -> None:
        super().__init__(n_dc=n_dc, n_states=n_states, batch_size=batch_size, cfg=cfg, rng_seed=rng_seed)
        self.trust_history: List[float] = []
        self._prior_precision = 1.0 / max(cfg.reward_prior_var, 1e-8)
        self._obs_precision = 1.0 / max(cfg.reward_obs_var, 1e-8)
        self._mix_reward_conf_scale = max(cfg.mix_reward_conf_scale, 1e-8)
        self._mix_trans_conf_scale = max(cfg.mix_trans_conf_scale, 1e-8)
        self._mix_global_conf_scale = max(cfg.mix_global_conf_scale, 1e-8)
        self._mix_whittle_change_scale = max(cfg.mix_whittle_change_scale, 1e-8)
        self._mix_trust_switch_temp = max(cfg.mix_trust_switch_temp, 1e-8)
        self._mix_global_prior_count = max(cfg.mix_global_prior_count, 1e-8)
        self.arm_pull_counts = np.zeros((n_dc,), dtype=float)
        self.arm_reward_sums = np.zeros((n_dc,), dtype=float)
        w_conf = max(cfg.mix_trust_conf_weight, 0.0)
        w_prog = max(cfg.mix_trust_progress_weight, 0.0)
        w_stab = max(cfg.mix_trust_stability_weight, 0.0)
        w_sum = max(w_conf + w_prog + w_stab, 1e-8)
        self._w_conf = w_conf / w_sum
        self._w_prog = w_prog / w_sum
        self._w_stab = w_stab / w_sum

    def _compute_model_confidence(self, states: np.ndarray) -> float:
        # Effective reward observations inferred from posterior precision.
        prec = self.reward_post.precision[self.arm_ids, states]
        n_eff = np.maximum((prec - self._prior_precision) / self._obs_precision, 0.0)
        reward_conf = 1.0 - np.exp(-n_eff / self._mix_reward_conf_scale)

        # Active transition confidence from observed (dc, state, action=1) counts.
        active_visits = self.trans_post.visit_counts[self.arm_ids, states, 1]
        trans_conf = 1.0 - np.exp(-active_visits / self._mix_trans_conf_scale)

        mean_reward_conf = float(np.mean(reward_conf))
        mean_trans_conf = float(np.mean(trans_conf))
        state_conf = 0.5 * mean_reward_conf + 0.5 * mean_trans_conf

        avg_pulls_per_arm = float(np.mean(self.arm_pull_counts))
        global_conf = 1.0 - np.exp(-avg_pulls_per_arm / self._mix_global_conf_scale)
        return 0.5 * state_conf + 0.5 * global_conf

    def _global_ucb_score(self, t: int) -> np.ndarray:
        pulls = self.arm_pull_counts + self._mix_global_prior_count
        prior_sum = self.cfg.reward_prior_mean * self._mix_global_prior_count
        means = (self.arm_reward_sums + prior_sum) / pulls
        bonus = self.cfg.mix_global_ucb_coef * np.sqrt(np.log(t + 2.0) / pulls)
        return means + bonus

    def _greedy_score(self, states: np.ndarray, t: int) -> np.ndarray:
        mu = self.reward_post.mu[self.arm_ids, states]
        std = np.sqrt(1.0 / self.reward_post.precision[self.arm_ids, states])
        local_ucb = mu + self.cfg.mix_greedy_bonus_coef * std
        global_ucb = self._global_ucb_score(t)

        # Early rounds rely more on global arm quality; then decay to local-state estimates.
        w_global = self.cfg.mix_greedy_global_weight * max(
            0.0,
            1.0 - (t / max(self.cfg.mix_greedy_global_decay_rounds, 1)),
        )
        w_global = float(np.clip(w_global, 0.0, 1.0))
        return w_global * global_ucb + (1.0 - w_global) * local_ucb

    def _compute_trust_index(self, states: np.ndarray, t: int) -> float:
        if t < self.cfg.mix_trust_warmup_rounds:
            progress = 0.0
        else:
            progress = min(
                1.0,
                (t - self.cfg.mix_trust_warmup_rounds) / max(self.cfg.mix_trust_ramp_rounds, 1),
            )

        model_conf = self._compute_model_confidence(states)
        whittle_stability = float(np.exp(-self.whittle_change / self._mix_whittle_change_scale))
        avg_pulls_per_arm = float(np.mean(self.arm_pull_counts))
        active_cov = 1.0 - np.exp(-avg_pulls_per_arm / self._mix_global_conf_scale)

        trust_raw = self._w_conf * model_conf + self._w_prog * progress + self._w_stab * whittle_stability
        trust_gate = 0.25 + 0.75 * active_cov
        trust_t = trust_gate * trust_raw
        # Smoothly force transition toward TW after the early-learning stage.
        switch_floor = 1.0 / (1.0 + np.exp(-(t - self.cfg.mix_trust_switch_round) / self._mix_trust_switch_temp))
        trust_t = max(trust_t, self.cfg.mix_trust_max * switch_floor)
        return float(np.clip(trust_t, self.cfg.mix_trust_min, self.cfg.mix_trust_max))

    def select_arm(self, states: np.ndarray, t: int) -> int:
        # Short warmup to get one-pass observations per arm, then switch to mixed policy.
        if t < self.cfg.mix_warmup_rounds:
            return int(t % self.n_dc)

        eps_t = self._linear_decay(
            t=t,
            start=self.cfg.mix_eps_start,
            end=self.cfg.mix_eps_end,
            horizon=self.cfg.mix_eps_decay_rounds,
        )
        if self.rng.random() < eps_t:
            return int(self.rng.integers(0, self.n_dc))

        self._maybe_replan(t)

        whittle_raw = self._whittle_scores(states)
        trust_t = self._compute_trust_index(states, t)
        greedy_raw = self._greedy_score(states, t)
        whittle_score = normalize_scores_01(whittle_raw)
        greedy_score = normalize_scores_01(greedy_raw)
        mixed_score = trust_t * whittle_score + (1.0 - trust_t) * greedy_score
        chosen_arm = int(np.argmax(mixed_score))
        self.trust_history.append(trust_t)
        return int(chosen_arm)

    def update(self, states: np.ndarray, chosen_arm: int, reward: float, next_states: np.ndarray) -> None:
        super().update(states, chosen_arm, reward, next_states)
        self.arm_pull_counts[chosen_arm] += 1.0
        self.arm_reward_sums[chosen_arm] += reward


class StateThompsonPolicy(Policy):
    def __init__(self, n_dc: int, n_states: int, cfg: Config, rng_seed: int) -> None:
        self.n_dc = n_dc
        self.rng = np.random.default_rng(rng_seed)
        self.reward_post = GaussianPosterior(
            shape=(n_dc, n_states),
            prior_mean=cfg.reward_prior_mean,
            prior_var=cfg.reward_prior_var,
            obs_var=cfg.reward_obs_var,
        )

    def select_arm(self, states: np.ndarray, t: int) -> int:
        sampled = self.reward_post.sample(self.rng)
        scores = np.array([sampled[dc, states[dc]] for dc in range(self.n_dc)], dtype=float)
        return int(np.argmax(scores))

    def update(self, states: np.ndarray, chosen_arm: int, reward: float, next_states: np.ndarray) -> None:
        s_sel = int(states[chosen_arm])
        self.reward_post.update((chosen_arm, s_sel), reward)


class BlackBoxThompsonPolicy(Policy):
    def __init__(self, n_dc: int, rng_seed: int) -> None:
        self.n_dc = n_dc
        self.rng = np.random.default_rng(rng_seed)
        self.alpha = np.ones((n_dc,), dtype=float)
        self.beta = np.ones((n_dc,), dtype=float)

    def select_arm(self, states: np.ndarray, t: int) -> int:
        sampled = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(sampled))

    def update(self, states: np.ndarray, chosen_arm: int, reward: float, next_states: np.ndarray) -> None:
        y = 1.0 if reward > 0.0 else 0.0
        self.alpha[chosen_arm] += y
        self.beta[chosen_arm] += (1.0 - y)


def run_policy(env: RMABEnvironment, policy: Policy, n_rounds: int) -> Tuple[np.ndarray, np.ndarray]:
    states = env.reset()
    rewards = np.zeros((n_rounds,), dtype=float)
    chosen = np.zeros((n_rounds,), dtype=int)

    for t in range(n_rounds):
        arm = policy.select_arm(states, t)
        next_states, reward = env.step(states, arm)
        policy.update(states, arm, reward, next_states)

        rewards[t] = reward
        chosen[t] = arm
        states = next_states

    return rewards, chosen


def ensure_output_dirs() -> None:
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)


def tagged_output_path(path: str, output_tag: str) -> str:
    if not output_tag:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_{output_tag}{ext}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thompson + Whittle for unknown-model RMAB")
    parser.add_argument("--n-jobs-sample", type=int, default=40, help="Sampled jobs per datacenter (state count)")
    parser.add_argument("--rounds", type=int, default=600, help="Rounds per simulation")
    parser.add_argument("--sims", type=int, default=4, help="Number of simulations")
    parser.add_argument("--replan-interval", type=int, default=10, help="Recompute sampled Whittle table every K rounds")
    parser.add_argument("--binary-iters", type=int, default=10, help="Binary search iterations for Whittle index")
    parser.add_argument("--transition-noise", type=float, default=0.0, help="Transition randomness level")
    parser.add_argument("--reward-noise-std", type=float, default=0.0, help="Observation noise std for active reward")
    parser.add_argument("--trans-prior-alpha", type=float, default=0.1, help="Dirichlet prior base concentration")
    parser.add_argument("--trans-structural-bias", type=float, default=0.9, help="Extra prior mass on expected active/passive next state")
    parser.add_argument("--tw-warmup-rounds", type=int, default=20, help="Round-robin warmup rounds for Thompson-Whittle")
    parser.add_argument("--tw-eps-start", type=float, default=0.20, help="Initial epsilon exploration rate for Thompson-Whittle")
    parser.add_argument("--tw-eps-end", type=float, default=0.02, help="Final epsilon exploration rate for Thompson-Whittle")
    parser.add_argument("--tw-eps-decay-rounds", type=int, default=400, help="Linear decay horizon for Thompson-Whittle epsilon")
    parser.add_argument("--mix-warmup-rounds", type=int, default=5, help="Initial warmup rounds for Trust-Mixed TW")
    parser.add_argument("--mix-trust-warmup-rounds", type=int, default=10, help="Rounds before trust ramp begins")
    parser.add_argument("--mix-trust-ramp-rounds", type=int, default=450, help="Trust ramp length in rounds")
    parser.add_argument("--mix-trust-min", type=float, default=0.01, help="Lower clip for trust index")
    parser.add_argument("--mix-trust-max", type=float, default=0.98, help="Upper clip for trust index")
    parser.add_argument("--mix-trust-conf-weight", type=float, default=0.45, help="Weight on posterior confidence in trust index")
    parser.add_argument("--mix-trust-progress-weight", type=float, default=0.25, help="Weight on round-progress term in trust index")
    parser.add_argument("--mix-trust-stability-weight", type=float, default=0.3, help="Weight on Whittle-stability term in trust index")
    parser.add_argument("--mix-whittle-change-scale", type=float, default=0.2, help="Scale for mapping Whittle-table change to stability")
    parser.add_argument("--mix-trust-switch-round", type=int, default=120, help="Round where Trust-Mixed starts quickly transitioning toward TW")
    parser.add_argument("--mix-trust-switch-temp", type=float, default=18.0, help="Temperature of smooth trust-switch (smaller = sharper)")
    parser.add_argument("--mix-reward-conf-scale", type=float, default=6.0, help="Reward confidence scale (larger = slower trust growth)")
    parser.add_argument("--mix-trans-conf-scale", type=float, default=8.0, help="Transition confidence scale (larger = slower trust growth)")
    parser.add_argument("--mix-global-conf-scale", type=float, default=10.0, help="State-active coverage scale for trust gating")
    parser.add_argument("--mix-global-ucb-coef", type=float, default=4.0, help="UCB bonus coefficient for global-arm greedy score")
    parser.add_argument("--mix-global-prior-count", type=float, default=1.0, help="Pseudo-count in global arm-mean estimate")
    parser.add_argument("--mix-greedy-global-weight", type=float, default=0.95, help="Early-round weight on global UCB in greedy score")
    parser.add_argument("--mix-greedy-global-decay-rounds", type=int, default=180, help="Rounds to decay global-UCB weight")
    parser.add_argument("--mix-eps-start", type=float, default=0.00, help="Initial epsilon exploration for Trust-Mixed TW")
    parser.add_argument("--mix-eps-end", type=float, default=0.00, help="Final epsilon exploration for Trust-Mixed TW")
    parser.add_argument("--mix-eps-decay-rounds", type=int, default=1, help="Linear epsilon decay rounds for Trust-Mixed TW")
    parser.add_argument("--mix-greedy-bonus-coef", type=float, default=0.8, help="UCB-like bonus coefficient for local-state greedy score")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--output-tag", type=str, default="", help="Optional suffix for output files to avoid collisions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_kwargs = vars(args).copy()
    output_tag = str(cfg_kwargs.pop("output_tag")).strip()
    cfg_kwargs["n_rounds"] = cfg_kwargs.pop("rounds")
    cfg_kwargs["n_sims"] = cfg_kwargs.pop("sims")
    cfg_kwargs["random_state"] = cfg_kwargs.pop("seed")
    cfg = Config(**cfg_kwargs)

    ensure_output_dirs()

    datacenter_files = sorted(glob.glob("datacenter_with_metrics/datacenter_*_with_metrics.csv"))[:cfg.n_dc]
    if len(datacenter_files) < cfg.n_dc:
        raise RuntimeError(f"Expected {cfg.n_dc} datacenter files, found {len(datacenter_files)}")

    raw_dfs = [pd.read_csv(fp) for fp in datacenter_files]
    print(f"Loaded {cfg.n_dc} datacenters.")
    for dc, fp in enumerate(datacenter_files):
        print(f"  DC {dc}: {fp}")

    policy_names = [
        "Oracle Whittle",
        "Trust-Mixed TW",
        "Thompson-Whittle",
        "State Thompson",
        "Black-box Thompson (binary)",
    ]

    cum_rewards = {name: np.zeros((cfg.n_rounds,), dtype=float) for name in policy_names}
    cum_regrets = {name: np.zeros((cfg.n_rounds,), dtype=float) for name in policy_names if name != "Oracle Whittle"}
    avg_rewards = {name: [] for name in policy_names}
    sel_counts = {name: np.zeros((cfg.n_dc,), dtype=int) for name in policy_names}

    print("=" * 72)
    print(
        f"Running {cfg.n_sims} simulations  |  rounds={cfg.n_rounds}  |  "
        f"replan_interval={cfg.replan_interval}"
    )
    print("=" * 72)

    for sim in range(cfg.n_sims):
        sim_seed = cfg.random_state + sim * 13
        R = build_reward_table(
            raw_dfs=raw_dfs,
            random_state=sim_seed,
            n_jobs_sample=cfg.n_jobs_sample,
            n_dc=cfg.n_dc,
            batch_size=cfg.batch_size,
            lookahead_size=cfg.lookahead_size,
        )
        p_active_true, p_passive_true = build_true_transitions(
            n_states=cfg.n_jobs_sample,
            batch_size=cfg.batch_size,
            n_dc=cfg.n_dc,
            noise=cfg.transition_noise,
        )

        run_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        oracle_policy = OracleWhittlePolicy(
            reward_table=R,
            p_active=p_active_true,
            p_passive=p_passive_true,
            cfg=cfg,
        )
        oracle_env = RMABEnvironment(
            reward_table=R,
            p_active=p_active_true,
            p_passive=p_passive_true,
            reward_noise_std=cfg.reward_noise_std,
            rng_seed=sim_seed + 101,
        )
        run_results["Oracle Whittle"] = run_policy(oracle_env, oracle_policy, cfg.n_rounds)

        tw_policy = ThompsonWhittlePolicy(
            n_dc=cfg.n_dc,
            n_states=cfg.n_jobs_sample,
            batch_size=cfg.batch_size,
            cfg=cfg,
            rng_seed=sim_seed + 201,
        )
        tw_env = RMABEnvironment(
            reward_table=R,
            p_active=p_active_true,
            p_passive=p_passive_true,
            reward_noise_std=cfg.reward_noise_std,
            rng_seed=sim_seed + 202,
        )
        run_results["Thompson-Whittle"] = run_policy(tw_env, tw_policy, cfg.n_rounds)

        mix_policy = TrustMixedThompsonWhittlePolicy(
            n_dc=cfg.n_dc,
            n_states=cfg.n_jobs_sample,
            batch_size=cfg.batch_size,
            cfg=cfg,
            rng_seed=sim_seed + 251,
        )
        mix_env = RMABEnvironment(
            reward_table=R,
            p_active=p_active_true,
            p_passive=p_passive_true,
            reward_noise_std=cfg.reward_noise_std,
            rng_seed=sim_seed + 252,
        )
        run_results["Trust-Mixed TW"] = run_policy(mix_env, mix_policy, cfg.n_rounds)

        st_policy = StateThompsonPolicy(
            n_dc=cfg.n_dc,
            n_states=cfg.n_jobs_sample,
            cfg=cfg,
            rng_seed=sim_seed + 301,
        )
        st_env = RMABEnvironment(
            reward_table=R,
            p_active=p_active_true,
            p_passive=p_passive_true,
            reward_noise_std=cfg.reward_noise_std,
            rng_seed=sim_seed + 302,
        )
        run_results["State Thompson"] = run_policy(st_env, st_policy, cfg.n_rounds)

        bb_policy = BlackBoxThompsonPolicy(
            n_dc=cfg.n_dc,
            rng_seed=sim_seed + 401,
        )
        bb_env = RMABEnvironment(
            reward_table=R,
            p_active=p_active_true,
            p_passive=p_passive_true,
            reward_noise_std=cfg.reward_noise_std,
            rng_seed=sim_seed + 402,
        )
        run_results["Black-box Thompson (binary)"] = run_policy(bb_env, bb_policy, cfg.n_rounds)

        oracle_rewards = run_results["Oracle Whittle"][0]
        oracle_cum = np.cumsum(oracle_rewards)

        msg_parts = [f"sim {sim + 1:>2}/{cfg.n_sims}"]
        for name in policy_names:
            rewards, chosen = run_results[name]
            cum_rewards[name] += rewards
            avg_rewards[name].append(float(np.mean(rewards)))
            for dc in range(cfg.n_dc):
                sel_counts[name][dc] += int(np.sum(chosen == dc))
            msg_parts.append(f"{name}={np.mean(rewards):.3f}")

            if name != "Oracle Whittle":
                cum_regrets[name] += (oracle_cum - np.cumsum(rewards))

        print("  " + "  |  ".join(msg_parts))

    mean_rewards = {name: cum_rewards[name] / cfg.n_sims for name in policy_names}
    mean_regrets = {name: cum_regrets[name] / cfg.n_sims for name in cum_regrets}

    print("\nAverage reward over simulations:")
    for name in policy_names:
        print(f"  {name:<18} {np.mean(avg_rewards[name]):.4f}")

    first_k = min(100, cfg.n_rounds)
    tw_first_avg = float(np.mean(mean_rewards["Thompson-Whittle"][:first_k]))
    mix_first_avg = float(np.mean(mean_rewards["Trust-Mixed TW"][:first_k]))
    early_gap = mix_first_avg - tw_first_avg
    print(
        f"\nEarly-stage diagnostic (first {first_k} rounds):\n"
        f"  Thompson-Whittle avg reward : {tw_first_avg:.4f}\n"
        f"  Trust-Mixed TW avg reward   : {mix_first_avg:.4f}\n"
        f"  Gap (Mix - TW)              : {early_gap:+.4f}"
    )

    # Convergence diagnostic: last-window slope of cumulative-average reward/regret.
    rounds_axis = np.arange(1, cfg.n_rounds + 1)
    tail_k = min(100, max(20, cfg.n_rounds // 4))
    tail_x = np.arange(cfg.n_rounds - tail_k, cfg.n_rounds)
    tw_curve = np.cumsum(mean_rewards["Thompson-Whittle"]) / rounds_axis
    tw_reg_curve = mean_regrets["Thompson-Whittle"] / rounds_axis
    tw_reward_slope = float(np.polyfit(tail_x, tw_curve[-tail_k:], 1)[0])
    tw_regret_slope = float(np.polyfit(tail_x, tw_reg_curve[-tail_k:], 1)[0])
    mix_curve = np.cumsum(mean_rewards["Trust-Mixed TW"]) / rounds_axis
    mix_reg_curve = mean_regrets["Trust-Mixed TW"] / rounds_axis
    mix_reward_slope = float(np.polyfit(tail_x, mix_curve[-tail_k:], 1)[0])
    mix_regret_slope = float(np.polyfit(tail_x, mix_reg_curve[-tail_k:], 1)[0])
    print(
        f"\nConvergence diagnostic (last {tail_k} rounds):\n"
        f"  Thompson-Whittle reward slope: {tw_reward_slope:+.6f} per round\n"
        f"  Thompson-Whittle regret slope: {tw_regret_slope:+.6f} per round\n"
        f"  Trust-Mixed TW reward slope  : {mix_reward_slope:+.6f} per round\n"
        f"  Trust-Mixed TW regret slope  : {mix_regret_slope:+.6f} per round"
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for i, name in enumerate(policy_names):
        ax.plot(
            rounds_axis,
            np.cumsum(mean_rewards[name]) / rounds_axis,
            label=f"{name} (avg={np.mean(avg_rewards[name]):.3f})",
            linewidth=1.8,
            color=colors[i % len(colors)],
        )
    ax.set_xlabel("Round")
    ax.set_ylabel(r"$T^{-1}\sum_{t=1}^T r_t$")
    ax.set_title("Per-round average reward")
    ax.legend(loc="best")

    ax = axes[1]
    regret_policy_names = [name for name in policy_names if name != "Oracle Whittle"]
    for i, name in enumerate(regret_policy_names):
        ax.plot(
            rounds_axis,
            mean_regrets[name] / rounds_axis,
            label=name,
            linewidth=1.8,
            color=colors[(i + 1) % len(colors)],
        )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Round")
    ax.set_ylabel(r"$T^{-1}\sum_{t=1}^T (r_t^* - r_t)$")
    ax.set_title("Per-round average regret vs Oracle Whittle")
    ax.legend(loc="best")

    plt.suptitle(
        "Unknown-model RMAB: Thompson-Whittle vs black-box learning\n"
        f"({cfg.n_dc} DCs, {cfg.n_jobs_sample} states, rounds={cfg.n_rounds}, sims={cfg.n_sims})",
        fontsize=11,
    )
    plt.tight_layout()
    out_plot_main = tagged_output_path("results/plots/thompson_whittle_greedy_vs_blackbox.png", output_tag)
    plt.savefig(out_plot_main, bbox_inches="tight")
    print(f"\nSaved: {out_plot_main}")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(cfg.n_dc)
    width = 0.2
    for i, name in enumerate(policy_names):
        ax.bar(
            x + (i - 1.5) * width,
            100.0 * sel_counts[name] / (cfg.n_sims * cfg.n_rounds),
            width=width,
            label=name,
            color=colors[i % len(colors)],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"DC{dc}" for dc in range(cfg.n_dc)])
    ax.set_ylabel("Selection frequency (%)")
    ax.set_title("Arm selection frequency")
    ax.legend(loc="best")
    plt.tight_layout()
    out_plot_sel = tagged_output_path("results/plots/thompson_whittle_greedy_arm_selection.png", output_tag)
    plt.savefig(out_plot_sel, bbox_inches="tight")
    print(f"Saved: {out_plot_sel}")

    summary_rows = []
    for name in policy_names:
        summary_rows.append(
            {
                "policy": name,
                "avg_reward": float(np.mean(avg_rewards[name])),
                "cum_reward": float(np.sum(mean_rewards[name])),
                "first100_avg_reward": float(np.mean(mean_rewards[name][:first_k])),
                "avg_regret_vs_oracle": 0.0 if name == "Oracle Whittle" else float(np.mean(mean_regrets[name])),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    out_csv = tagged_output_path("results/thompson_whittle_greedy_summary.csv", output_tag)
    summary_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()
