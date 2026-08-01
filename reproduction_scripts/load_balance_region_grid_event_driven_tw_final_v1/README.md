# Region-Grid Event-Driven TW Scripts

This folder preserves the Python scripts used to generate and plot the
`load_balance_region_grid_event_driven_tw_final_v1` outputs.

The files are copied from:

- `ruicheng_codebase_2026-04-28/experiments/rmab_vm/`
- `ruicheng_codebase_2026-04-28/scripts/`
- repository-root helper modules used by the event-driven TW runner

## Main Experiment Runner

- `experiments/run_region_grid_event_driven_tw.py`
  - Main event-driven TW region-grid load-balancing runner.
  - Produces:
    - `event_driven_tw_region_grid_results.csv`
    - `event_driven_tw_region_grid_summary.csv`
    - `run_config.json`

- `experiments/run_region_grid_load_balance_sweep.py`
  - Provides the region-grid instance construction used by the event-driven
    runner.

- `experiments/run_experiments.py`
  - Shared RMAB VM utilities, priors, policy helpers, and datacenter loading.

- `root_helpers/Event_driven_TW_varying_data_center_jobs.py`
  - Event-driven TW and oracle Whittle policy implementation used by the
    runner.

- `root_helpers/multi_armed_bandits_mdp_thompson_whittle_greedy.py`
  - Whittle-table and RMAB environment helpers used by the event-driven runner.

## Commands Matching Preserved Run Configs

Homogeneous run (`op_states=1`, state sizes 20/40/60):

```bash
python experiments/run_region_grid_event_driven_tw.py \
  --output docs/research/rmab_vm_outputs/load_balance_region_grid_event_driven_tw_q4_homo_norm_v1 \
  --grid-cost-csv docs/research/rmab_vm_outputs/grid_cost_features_v1/grid_cost_region_averages_2023.csv \
  --data-dir datacenter_with_metrics \
  --seeds 2 \
  --rounds 250 \
  --arms 8 \
  --budget 2 \
  --queue-states 4 \
  --region-state-grid 5 10 15 \
  --op-states 1 \
  --computation-mode event
```

Heterogeneous run (`op_states=3`, state sizes 60/120/180):

```bash
python experiments/run_region_grid_event_driven_tw.py \
  --output docs/research/rmab_vm_outputs/load_balance_region_grid_event_driven_tw_q4_heter_norm_v1 \
  --grid-cost-csv docs/research/rmab_vm_outputs/grid_cost_features_v1/grid_cost_region_averages_2023.csv \
  --data-dir datacenter_with_metrics \
  --seeds 2 \
  --rounds 250 \
  --arms 8 \
  --budget 2 \
  --queue-states 4 \
  --region-state-grid 5 10 15 \
  --op-states 3 \
  --computation-mode event
```

Timing comparison, event-driven TW only:

```bash
python experiments/run_region_grid_event_driven_tw.py \
  --output docs/research/rmab_vm_outputs/load_balance_region_grid_event_driven_tw_final_v1/timing_tw_event_s60_s180_one_time \
  --grid-cost-csv docs/research/rmab_vm_outputs/grid_cost_features_v1/grid_cost_region_averages_2023.csv \
  --data-dir datacenter_with_metrics \
  --seeds 1 \
  --rounds 250 \
  --arms 8 \
  --budget 2 \
  --queue-states 4 \
  --region-state-grid 5 15 \
  --op-states 3 \
  --computation-mode event \
  --strategies tw_dense
```

Timing comparison, full recompute TW only:

```bash
python experiments/run_region_grid_event_driven_tw.py \
  --output docs/research/rmab_vm_outputs/load_balance_region_grid_event_driven_tw_final_v1/timing_tw_full_s60_s180_one_time \
  --grid-cost-csv docs/research/rmab_vm_outputs/grid_cost_features_v1/grid_cost_region_averages_2023.csv \
  --data-dir datacenter_with_metrics \
  --seeds 1 \
  --rounds 250 \
  --arms 8 \
  --budget 2 \
  --queue-states 4 \
  --region-state-grid 5 15 \
  --op-states 3 \
  --computation-mode full \
  --strategies tw_dense
```

## Plot Script

- `scripts/plot_region_grid_load_balance_points.py`
  - Reads `all_strategy_load_balance_appendix_table.csv`.
  - Generates:
    - `fig_region_grid_load_balance_points.png`
    - `fig_region_grid_load_balance_points.pdf`

## Notes

The cleaned final artifacts are stored under:

`clean_research_bundle/experiments/load_balance_region_grid_event_driven_tw_final_v1/`

That final folder is a consolidation of several source runs. The copied
`manifest.json` records the main kept source directories:

- `load_balance_region_grid_event_driven_tw_q4_homo_norm_v1`
- `load_balance_region_grid_event_driven_tw_q4_heter_norm_v1`
- `load_balance_region_grid_beta_gate_param_homo_g15_v1`

The preserved codebase does not appear to contain standalone Python scripts for
assembling the final prefixed CSV names, oracle-comparison summary CSVs, beta
gate tuning summary/table, or timing LaTeX table. Those final derived artifacts
are preserved in the clean experiment folder, while this directory preserves the
available source runners and plotting script.
