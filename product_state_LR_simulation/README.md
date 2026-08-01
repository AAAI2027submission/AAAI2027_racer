# Product-State LR Simulation

This folder contains the simulation code and run context for the product-state
block/local versus global low-rank 10-seed experiment.

## Main Script

Run the simulations with:

```bash
python st_tw_tmtw_comparison/run_eight_dc_exp4_combine_variant_2d_state.py
```

The script writes:

- `combine_variant_2d_state_results.csv`
- `combine_variant_2d_state_round_rewards.csv`
- `combine_variant_2d_state_summary.csv`
- `model_informed_offline_support.csv`
- `run_config.json`

It imports shared RMAB utilities from:

```text
baseline_setting_suite_v1/experiments/run_experiments.py
```

## Global LR Run

```bash
python st_tw_tmtw_comparison/run_eight_dc_exp4_combine_variant_2d_state.py \
  --source-results eight_dc_exp4_block_vs_global_lr_10seed_source.csv \
  --datacenter-dir datasets/datacenter_with_metrics \
  --output eight_dc_exp4_combine_variant_2d_state_block_vs_global_lr_10seed_global_results \
  --rounds 1500 \
  --queue-states 40 \
  --exclude-plain-tm-tw \
  --policy-labels oracle,tw_gated_offline_low_rank,tm_tw_refined_gated_offline_low_rank \
  --low-rank-max-rank 10
```

## Block/Local LR Run

```bash
python st_tw_tmtw_comparison/run_eight_dc_exp4_combine_variant_2d_state.py \
  --source-results eight_dc_exp4_block_vs_global_lr_10seed_source.csv \
  --datacenter-dir datasets/datacenter_with_metrics \
  --output eight_dc_exp4_combine_variant_2d_state_block_vs_global_lr_10seed_block_results \
  --rounds 1500 \
  --queue-states 40 \
  --exclude-plain-tm-tw \
  --policy-labels oracle,tw_gated_offline_low_rank,tm_tw_refined_gated_offline_low_rank \
  --low-rank-max-rank 10 \
  --block-low-rank
```

## Included Context

- `eight_dc_exp4_block_vs_global_lr_10seed_source.csv`: source seeds/settings.
- `configs/global_lr_run_config.json`: saved config from the global LR run.
- `configs/block_local_lr_run_config.json`: saved config from the block/local LR run.

The commands above expect `datasets/datacenter_with_metrics` to exist at the
repository root.
