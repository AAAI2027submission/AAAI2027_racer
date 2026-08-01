"""Run the one-dimensional baseline setting with the refinement policy suite.

The setting matches the native/baseline row:

    arms = 5, states = 8, batch size = 5, rounds = 1000, seeds = 10,
    contextual noise = 0.

The policy/variant family matches the transition-stress experiments: paper
baselines plus TW/TM--TW/adaptive TM--TW over dense, gated-prior,
gated-prior+low-rank, and support/offline-prior variants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_experiments import run_context_noise_experiment


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "docs/research/rmab_vm_outputs/baseline_setting_suite_v1"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--arms", type=int, default=5)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--sparsity", type=int, default=2)
    parser.add_argument("--transition-dominance", type=float, default=0.45)
    parser.add_argument("--trust-floor", type=float, default=0.10)
    parser.add_argument("--trust-cap", type=float, default=0.95)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_args = argparse.Namespace(
        output=str(args.output),
        data_dir=None,
        seed=args.seed,
        seeds=args.seeds,
        rounds=args.rounds,
        arms=args.arms,
        state_grid=[args.states],
        sparsity=args.sparsity,
        transition_dominance=args.transition_dominance,
        noise_grid=[0.0],
        variants=VARIANTS,
        policies=POLICIES,
        trust_scale_mults=[1.0],
        gate_scale_mults=[1.0],
        gate_modes=["deterministic"],
        beta_gate_concentration=20.0,
        trust_floor=args.trust_floor,
        trust_floors=[args.trust_floor],
        trust_cap=args.trust_cap,
        flush_every=0,
        progress=args.progress,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_config.json").write_text(
        json.dumps(
            {
                "setting": "Baseline setting",
                "state_dimension": "one-dimensional",
                "contextual_noise": 0.0,
                "arms": args.arms,
                "states": args.states,
                "batch_size": 5,
                "rounds": args.rounds,
                "seeds": args.seeds,
                "policies": POLICIES,
                "variants": VARIANTS,
                "output_files": [
                    "context_noise_results.csv",
                    "context_noise_summary.csv",
                    "context_noise_reward_pct.png",
                ],
            },
            indent=2,
        )
    )
    run_context_noise_experiment(run_args)


if __name__ == "__main__":
    main()
