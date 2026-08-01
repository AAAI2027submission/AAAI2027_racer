# RACER: Robust and Adaptive Computing and Energy Resource Coordination Framework

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+"></a>
</p>

This repository contains the datasets and experimental results associated with the submission: *Refined Thompson Learning for Adaptive Restless Bandits: Power-Efficient Flexibility Scheduling Across Data Centers*.


![RACER GitHub cover](Readme_graph/Github_cover.png)

## Contents

### Datasets

| Path | Contents | Use / Notes |
| --- | --- | --- |
| `datasets/datacenter_with_metrics/` | Processed data-center VM/job datasets named `datacenter_*_with_metrics.csv`. | Consumed by the real-data and region-grid load-balancing experiments. |
| `datasets/carbon_intensity/` | Regional hourly carbon-intensity data. | Used to construct grid/carbon-cost features for load-balancing experiments. |
| `datasets/electricity_prices_processed/` | Processed regional electricity-price CSV files. | Provides regional price inputs. |
| `datasets/electricity_prices_standardized/` | Standardized electricity-price CSV files aligned for cross-region use. | Kept because the region-grid experiments use standardized grid-cost inputs. |
| `datasets/power_qos_distribution/` | Power-saving index and QoS-cost distribution figure. | Includes the generated figure and plotting script copied from `current_paper_artifacts/power_qos_distribution/`. |

### Experiment Outputs and Reproduction Scripts

The experiments cover two main settings:

- **Baseline:** simple 8-state setting with 5 arms, batch size 5, 1000
  rounds, and 10 random seeds. This setting evaluates performance when the
  state space is small and online learning has many rounds.
- **Transition stress:** larger sparse-transition setting with 5 arms, 8--100
  states, batch size 5, 100--200 rounds, and 3 random seeds. This setting
  stresses transition learning when online observations are limited.

Across these settings, we use multiple random seeds and hyperparameter sweeps
for refinement parameters such as the trust floor and beta gate.

- **Baseline:** `baseline_setting_suite_v1`
  - Output: `experiments/baseline_setting_suite_v1/`
  - Scripts: `reproduction_scripts/baseline_setting_suite_v1/`
  - Summary: baseline synthetic suite with reward summaries, learning curves,
    round-bar comparisons, and seed-level checkpoint diagnostics.

![Baseline setting learning curves and round-bar comparison](experiments/baseline_setting_suite_v1/fig_baseline_setting_learning_and_round_bar.png)

**Highlight:** The refined Thompson learning variants generally outperform the
baseline, and the final reward exceeds the state-of-the-art EXP4 framework.

- **Transition stress:** `context_noise_real_data_v1`
  - Output: `experiments/context_noise_real_data_v1/`
  - Scripts: `reproduction_scripts/context_noise_real_data_v1/`
  - Summary: transition-stress context-noise outputs, including summary CSVs,
    figures, LaTeX tables, best-beta-gate analysis, and refined win-frequency
    plots.

### Ablation Experiments

The repository also includes two ablation experiment summaries:

- **Block-local vs. global low-rank refinement:** `block_vs_global_lr_10seed_summary`
  - Output: `experiments/block_vs_global_lr_10seed_summary/`
  - Scripts: `reproduction_scripts/product_state_LR_simulation/`
  - Summary: 10-seed comparison of block-local and global low-rank refinement
    modes, including reward summaries, selected-policy seed results, support
    edge counts, and plots.

![Block-local vs. global low-rank refinement comparison](experiments/block_vs_global_lr_10seed_summary/plots/block_vs_global_lr_10seed_error_bars.png)

- **Support-construction ablations:** `four_support_experiments_summary`
  - Output: `experiments/four_support_experiments_summary/`
  - Scripts: `reproduction_scripts/four_support_experiments_simulation/`
  - Summary: four support experiments comparing support-construction variants,
    including combined reward bars, per-experiment summaries, selected-policy
    seed results, support edge counts, and plots.

![Support-construction ablation comparison](experiments/four_support_experiments_summary/plots/all_support_experiments_bars.png)

Generated summary artifacts can be rebuilt from the retained source outputs:

```bash
python experiments/baseline_setting_suite_v1/generate_baseline_setting_checkpoint_diagnostics.py
```

## Notes

- Python 3.9 or later is recommended for running the reproduction scripts.
- Required Python packages: `numpy`, `pandas`, `matplotlib`, and `seaborn`.
- Install the package dependencies with:

```bash
pip install numpy pandas matplotlib seaborn
```

- The scripts use `matplotlib` for figure generation and can run in headless
  environments using the non-interactive `Agg` backend configured in the
  plotting scripts.

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for
details.
