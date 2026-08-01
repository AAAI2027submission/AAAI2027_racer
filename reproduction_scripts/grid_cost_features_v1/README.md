# Grid-Cost Feature Scripts

This folder preserves the available source context for generating
`grid_cost_features_v1` and adds a clean reproduction script for the preserved
bundle layout.

## What Was Found

In `ruicheng_codebase_2026-04-28`, the preserved files are:

- `docs/research/rmab_vm_outputs/grid_cost_features_v1/grid_cost_region_averages_2023.csv`
- `docs/research/rmab_vm_outputs/grid_cost_features_v1/grid_cost_region_averages_2023_normalized.csv`

I did not find a standalone Python generator for these two CSVs in Ruicheng's
folder. The codebase only references these files from downstream region-grid
load-balancing scripts.

## Included Files

- `source_analysis_scripts/analyze_carbon_intensity_data.py`
  - Copied from `ruicheng_codebase_2026-04-28/carbon_intensity/`.

- `source_analysis_scripts/analyze_electricity_prices.py`
  - Copied from `ruicheng_codebase_2026-04-28/electricity_prices/`.

- `scripts/generate_grid_cost_features.py`
  - Reconstructs the grid-cost feature CSVs from:
    - `clean_research_bundle/datasets/carbon_intensity/`
    - `clean_research_bundle/datasets/electricity_prices_standardized/`

## Reproduction Command

From this repository root:

```bash
python clean_research_bundle/reproduction_scripts/grid_cost_features_v1/scripts/generate_grid_cost_features.py
```

This writes:

- `clean_research_bundle/supporting_inputs/grid_cost_features_v1/grid_cost_region_averages_2023.csv`
- `clean_research_bundle/supporting_inputs/grid_cost_features_v1/grid_cost_region_averages_2023_normalized.csv`

The script includes regions that have both 2023 carbon-intensity data and 2023
standardized electricity-price data.

