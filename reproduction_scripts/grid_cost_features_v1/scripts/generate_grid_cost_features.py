"""Generate region-level grid-cost features from carbon and price datasets.

This reconstructs the supporting inputs preserved in:

    clean_research_bundle/supporting_inputs/grid_cost_features_v1/

The output combines yearly average direct/LCA carbon intensity with yearly
average standardized electricity price for regions present in both datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REGION_NAMES = {
    "AU-NSW": "New South Wales, Australia",
    "AU-VIC": "Victoria, Australia",
    "BR-SP": "São Paulo, Brazil",
    "CA-ON": "Ontario, Canada",
    "CL-SIC": "Central Chile (SIC)",
    "DE-LU": "Germany/Luxembourg",
    "IN-WE": "Western India",
    "JP-TK": "Tokyo, Japan",
    "KR": "South Korea",
    "SG": "Singapore",
    "US-CAL-CISO": "California, USA (CAISO)",
    "US-MIDA-PJM": "Mid-Atlantic, USA (PJM)",
    "US-NY-NYIS": "New York, USA (NYISO)",
    "US-TEX-ERCO": "Texas, USA (ERCOT)",
    "ZA": "South Africa",
}


def minmax(series: pd.Series) -> pd.Series:
    span = float(series.max() - series.min())
    if span < 1e-12:
        return series * 0.0
    return (series - series.min()) / span


def carbon_summary(carbon_root: Path, region_code: str, year: int) -> dict | None:
    path = carbon_root / region_code / str(year) / f"{region_code}_{year}_hourly.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(
        path,
        usecols=[
            "Carbon Intensity gCO₂eq/kWh (direct)",
            "Carbon Intensity gCO₂eq/kWh (LCA)",
        ],
    )
    return {
        "mean_carbon_intensity_direct_gco2eq_per_kwh": float(
            df["Carbon Intensity gCO₂eq/kWh (direct)"].mean()
        ),
        "mean_carbon_intensity_lca_gco2eq_per_kwh": float(
            df["Carbon Intensity gCO₂eq/kWh (LCA)"].mean()
        ),
        "carbon_hour_count": int(len(df)),
    }


def price_summary(price_root: Path, region_code: str, year: int) -> dict | None:
    path = price_root / region_code / str(year) / f"{region_code}_electricity_prices_{year}.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path, usecols=["Price (USD/MWh)"])
    return {
        "mean_electricity_price_usd_per_mwh": float(df["Price (USD/MWh)"].mean()),
        "electricity_price_hour_count": int(len(df)),
    }


def build_features(bundle_root: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    carbon_root = bundle_root / "datasets" / "carbon_intensity"
    price_root = bundle_root / "datasets" / "electricity_prices_standardized"

    rows = []
    for carbon_dir in sorted(path for path in carbon_root.iterdir() if path.is_dir()):
        region_code = carbon_dir.name
        carbon = carbon_summary(carbon_root, region_code, year)
        price = price_summary(price_root, region_code, year)
        if carbon is None or price is None:
            continue
        rows.append(
            {
                "region_code": region_code,
                "region_name": REGION_NAMES.get(region_code, region_code),
                "year": year,
                **carbon,
                **price,
            }
        )

    raw_columns = [
        "region_code",
        "region_name",
        "year",
        "mean_carbon_intensity_direct_gco2eq_per_kwh",
        "mean_carbon_intensity_lca_gco2eq_per_kwh",
        "mean_electricity_price_usd_per_mwh",
        "carbon_hour_count",
        "electricity_price_hour_count",
    ]
    raw = pd.DataFrame(rows).sort_values("region_code").reset_index(drop=True)
    raw = raw[raw_columns]

    normalized = raw.copy()
    normalized["carbon_direct_norm"] = minmax(
        normalized["mean_carbon_intensity_direct_gco2eq_per_kwh"]
    )
    normalized["carbon_lca_norm"] = minmax(
        normalized["mean_carbon_intensity_lca_gco2eq_per_kwh"]
    )
    normalized["electricity_price_norm"] = minmax(
        normalized["mean_electricity_price_usd_per_mwh"]
    )
    normalized["combined_direct_carbon_electricity_norm"] = 0.5 * (
        normalized["carbon_direct_norm"] + normalized["electricity_price_norm"]
    )
    normalized["combined_lca_carbon_electricity_norm"] = 0.5 * (
        normalized["carbon_lca_norm"] + normalized["electricity_price_norm"]
    )
    normalized = normalized.sort_values(
        ["combined_direct_carbon_electricity_norm", "region_code"]
    ).reset_index(drop=True)
    normalized.insert(0, "grid_rank_low_to_high", range(1, len(normalized) + 1))
    normalized = normalized[
        [
            "grid_rank_low_to_high",
            *raw_columns,
            "carbon_direct_norm",
            "carbon_lca_norm",
            "electricity_price_norm",
            "combined_direct_carbon_electricity_norm",
            "combined_lca_carbon_electricity_norm",
        ]
    ]
    return raw, normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Path to clean_research_bundle.",
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to supporting_inputs/grid_cost_features_v1 under bundle root.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (
        args.bundle_root / "supporting_inputs" / "grid_cost_features_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    raw, normalized = build_features(args.bundle_root, args.year)
    raw.to_csv(output_dir / f"grid_cost_region_averages_{args.year}.csv", index=False)
    normalized.to_csv(
        output_dir / f"grid_cost_region_averages_{args.year}_normalized.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
