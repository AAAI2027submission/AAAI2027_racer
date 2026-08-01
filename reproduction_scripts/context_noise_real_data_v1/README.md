# Real-Data Context-Noise Scripts

This folder preserves the Python scripts used to generate and plot the
`context_noise_real_data_v1` outputs.

The files are copied from:

- `ruicheng_codebase_2026-04-28/experiments/rmab_vm/`
- `ruicheng_codebase_2026-04-28/scripts/`

## Experiment

- `experiments/run_experiments.py`
  - Generic RMAB VM experiment runner.
  - The real-data context-noise sweep is produced with
    `--experiment context_noise` and `--data-dir` pointing at the processed
    datacenter metric CSVs.

Likely raw-run command:

```bash
python experiments/run_experiments.py \
  --experiment context_noise \
  --output docs/research/rmab_vm_outputs/context_noise_real_data_v1 \
  --data-dir datacenter_with_metrics \
  --seeds 3 \
  --rounds 100 \
  --state-grid 8 20 50 100 \
  --noise-grid 0.0 0.1 0.2 0.3 \
  --variants dense gated_offline gated_offline_low_rank support_offline \
  --policies state_thompson local_ucb_tw global_ucb_tw exp4 tw tm_tw tm_tw_refined
```

This produces:

- `context_noise_results.csv`
- `context_noise_summary.csv`
- `context_noise_reward_pct.png`

## Plot and Table Scripts

- `scripts/plot_real_data_figures.py`
  - Reads `context_noise_real_data_v1/context_noise_summary.csv`.
  - Generates the original Sweep A real-data figure/table:
    - `fig_context_noise_real.{png,pdf}`
    - `tab_context_noise_real.tex`

- `scripts/plot_context_noise_real_with_best_beta_gate.py`
  - Reads `context_noise_summary_with_best_beta_gate.csv`.
  - Generates:
    - `fig_context_noise_real_with_best_beta_gate.{png,pdf}`
    - `tab_context_noise_real_with_best_beta_gate.tex`

- `scripts/plot_context_noise_real_refined_win_frequency.py`
  - Reads `context_noise_summary_with_best_beta_gate.csv`.
  - Generates:
    - `context_noise_refined_win_frequency.csv`
    - `fig_context_noise_real_refined_win_frequency.{png,pdf}`

## Notes

The cleaned result artifacts are stored under:

`clean_research_bundle/experiments/context_noise_real_data_v1/`

The preserved codebase did not contain a standalone script for assembling
`context_noise_results_with_best_beta_gate.csv` or
`context_noise_summary_with_best_beta_gate.csv`. Those files appear to have
been assembled from the base real-data sweep plus supplemental beta-gate runs.
