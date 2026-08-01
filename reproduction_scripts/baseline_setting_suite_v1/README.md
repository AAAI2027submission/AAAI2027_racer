# Baseline Setting Context-Noise Suite Scripts

This folder preserves the original Python scripts used to generate the
`baseline_setting_suite_v1` experiment outputs.

The files are copied from:

- `ruicheng_codebase_2026-04-28/experiments/rmab_vm/`
- `ruicheng_codebase_2026-04-28/scripts/`

## Experiment

- `experiments/run_baseline_setting_suite.py`
  - Main baseline-setting experiment runner.
  - Calls `run_context_noise_experiment` from `experiments/run_experiments.py`.

- `experiments/run_experiments.py`
  - Shared RMAB VM experiment implementation and plotting helpers.

- `experiments/run_baseline_setting_beta_offline_supplement.py`
  - Supplementary beta/offline baseline run used by some comparison and
    diagnostic scripts.

## Plot and Diagnostic Scripts

- `scripts/plot_baseline_setting_learning_curves.py`
  - Generates per-seed and aggregate learning-curve CSVs plus
    `fig_baseline_setting_learning_curves.{png,pdf}`.

- `scripts/plot_baseline_setting_round_bar_comparison.py`
  - Generates `baseline_setting_round_bar_comparison.csv` and the standalone
    round-bar comparison figure.

- `scripts/plot_baseline_setting_combined_learning_and_bars.py`
  - Generates the combined learning-curve and round-bar paper figure.

- `scripts/plot_baseline_setting_overall_comparison.py`
  - Generates the overall baseline comparison figure from summary CSVs.

- `scripts/compute_baseline_setting_checkpoint_diagnostics.py`
  - Generates checkpoint L1/leakage diagnostics CSVs.

## Notes

These are source snapshots for traceability. The cleaned result artifacts are
stored separately under:

`clean_research_bundle/experiments/baseline_setting_suite_v1/`
