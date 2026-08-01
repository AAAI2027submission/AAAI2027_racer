"""Run supplemental beta-gate/offline-prior rows for the baseline setting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_experiments import run_context_noise_experiment


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "docs/research/rmab_vm_outputs/baseline_setting_beta_offline_supplement_v1"


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
    variants = [
        "offline",
        "gated_offline",
        "gated_offline_low_rank",
        "support_offline",
    ]
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
        variants=variants,
        policies=["tm_tw_refined"],
        trust_scale_mults=[1.0],
        gate_scale_mults=[1.0],
        gate_modes=["deterministic", "beta"],
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
                "setting": "Baseline setting supplemental rows",
                "state_dimension": "one-dimensional",
                "contextual_noise": 0.0,
                "arms": args.arms,
                "states": args.states,
                "batch_size": 5,
                "rounds": args.rounds,
                "seeds": args.seeds,
                "policies": ["tm_tw_refined"],
                "variants": variants,
                "gate_modes": ["deterministic", "beta"],
            },
            indent=2,
        )
    )
    run_context_noise_experiment(run_args)


if __name__ == "__main__":
    main()
