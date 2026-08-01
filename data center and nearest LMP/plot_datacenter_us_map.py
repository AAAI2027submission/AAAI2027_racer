"""
Plot all database data center locations and highlight selected market locations.
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "racer_cdc_mplconfig"))

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATABASE_XLSX_PATH = SCRIPT_DIR / "Data_Centers_Database.xlsx"
DATABASE_CSV_PATH = SCRIPT_DIR / "Data_Centers_Database.csv"
MAP_DATA_PATH = SCRIPT_DIR / "data_center_us_map.csv"
OUTPUT_PATH = SCRIPT_DIR / "datacenter_us_map.png"
LABEL_PATH = SCRIPT_DIR / "datacenter_us_map_labels.csv"
DATABASE_SHEET = "DB_Output_V2"

MARKETS = [
    {
        "market": "CAISO",
        "file": "caiso_finalized.csv",
        "skiprows": 0,
        "data_center_col": "Data Center",
    },
    {
        "market": "ERCOT",
        "file": "ercot_load_zone_15min.csv",
        "skiprows": 1,
        "data_center_col": "Data Center",
    },
    {
        "market": "ISONE",
        "file": "combined_ISONE_5min_lmp.csv",
        "skiprows": 1,
        "data_center_col": "Data Center",
    },
    {
        "market": "PJM",
        "file": "combined_PJM_5min_lmp.csv",
        "skiprows": 1,
        "data_center_col": "Data center",
    },
    {
        "market": "NYISO",
        "file": "NYISO_zone_lmp_5min.csv",
        "skiprows": 1,
        "data_center_col": "Data center",
    },
]

COLORS = {
    "CAISO": "#1f77b4",
    "ERCOT": "#d62728",
    "ISONE": "#2ca02c",
    "PJM": "#9467bd",
    "NYISO": "#ff7f0e",
}


def load_locations() -> pd.DataFrame:
    frames = []
    for config in MARKETS:
        df = pd.read_csv(SCRIPT_DIR / config["file"], skiprows=config["skiprows"])
        subset = df[[config["data_center_col"], "Latitude", "Longitude"]].copy()
        subset = subset.rename(columns={config["data_center_col"]: "Data Center"})
        subset["Market"] = config["market"]
        subset["Source File"] = config["file"]
        frames.append(subset)

    locations = pd.concat(frames, ignore_index=True)
    locations["Latitude"] = pd.to_numeric(locations["Latitude"], errors="coerce")
    locations["Longitude"] = pd.to_numeric(locations["Longitude"], errors="coerce")
    locations = locations.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)

    # A few source rows use positive west longitudes. Keep CSVs untouched, but
    # normalize for map placement within the contiguous US.
    locations.loc[locations["Longitude"] > 0, "Longitude"] *= -1

    locations["Map Label"] = (
        locations["Market"] + "-" + (locations.groupby("Market").cumcount() + 1).astype(str)
    )
    return locations


def normalize_longitude(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.mask(values > 0, -values)


def load_database_locations() -> pd.DataFrame:
    database = pd.read_excel(DATABASE_XLSX_PATH, sheet_name=DATABASE_SHEET)
    database.to_csv(DATABASE_CSV_PATH, index=False)

    locations = database[["facility_name", "lat", "long", "status", "state"]].copy()
    locations = locations.rename(
        columns={
            "facility_name": "Data Center",
            "lat": "Latitude",
            "long": "Longitude",
            "status": "Status",
            "state": "State",
        }
    )
    locations["Latitude"] = pd.to_numeric(locations["Latitude"], errors="coerce")
    locations["Longitude"] = normalize_longitude(locations["Longitude"])
    locations = locations.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
    locations["Market"] = "Not selected"
    locations["Source File"] = DATABASE_XLSX_PATH.name
    locations["Selected"] = False
    locations["Map Label"] = ""
    return locations


def within_contiguous_us(locations: pd.DataFrame) -> pd.Series:
    return (
        locations["Longitude"].between(-126, -66)
        & locations["Latitude"].between(24, 50)
    )


def main() -> None:
    database_locations = load_database_locations()
    selected_locations = load_locations()
    selected_locations["Selected"] = True
    selected_locations["Status"] = ""
    selected_locations["State"] = ""

    all_locations = pd.concat(
        [
            database_locations[
                [
                    "Map Label",
                    "Market",
                    "Data Center",
                    "Latitude",
                    "Longitude",
                    "Selected",
                    "Status",
                    "State",
                    "Source File",
                ]
            ],
            selected_locations[
                [
                    "Map Label",
                    "Market",
                    "Data Center",
                    "Latitude",
                    "Longitude",
                    "Selected",
                    "Status",
                    "State",
                    "Source File",
                ]
            ],
        ],
        ignore_index=True,
    )
    all_locations.to_csv(MAP_DATA_PATH, index=False)

    database_to_plot = database_locations[within_contiguous_us(database_locations)]
    selected_to_plot = selected_locations[within_contiguous_us(selected_locations)]

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    usa = world[world["name"] == "United States of America"]

    fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True)
    usa.plot(ax=ax, color="#f3f0e8", edgecolor="#8f8f8f", linewidth=0.8)

    ax.set_xlim(-126, -66)
    ax.set_ylim(24, 50)
    ax.set_aspect("equal", adjustable="box")

    ax.scatter(
        database_to_plot["Longitude"],
        database_to_plot["Latitude"],
        s=16,
        facecolors="none",
        edgecolors="#8a8a8a",
        linewidth=0.45,
        alpha=0.55,
        label=f"Not selected ({len(database_to_plot)})",
        zorder=2,
    )

    label_offsets = {
        "CAISO": [(8, 8), (8, -3), (8, 18), (8, -15), (8, -26)],
        "ERCOT": [(8, 0), (8, 10), (8, -6), (8, 0), (8, 6), (8, -10)],
        "ISONE": [(8, 12), (8, -2), (8, -16)],
        "PJM": [(8, 2), (8, 12), (8, -8), (8, 6), (8, -8)],
        "NYISO": [(8, 10), (8, 6), (8, -8), (-32, 8), (-32, -8)],
    }

    for market, sub in selected_to_plot.groupby("Market", sort=False):
        ax.scatter(
            sub["Longitude"],
            sub["Latitude"],
            s=72,
            color=COLORS[market],
            edgecolor="white",
            linewidth=0.9,
            label=f"{market} ({len(sub)})",
            zorder=3,
        )
        for offset, (_, row) in zip(label_offsets[market], sub.iterrows()):
            ax.annotate(
                row["Map Label"],
                (row["Longitude"], row["Latitude"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                color="#202020",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#808080",
                    "linewidth": 0.4,
                    "shrinkA": 0,
                    "shrinkB": 3,
                },
                zorder=4,
            )

    ax.set_title("US Data Center Locations and Selected Electricity Market Sites", fontsize=16)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.grid(True, color="#d7d7d7", linewidth=0.6, alpha=0.7)
    ax.legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=10)

    selected_locations[
        ["Map Label", "Market", "Data Center", "Latitude", "Longitude", "Source File"]
    ].to_csv(LABEL_PATH, index=False)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {DATABASE_CSV_PATH}")
    print(f"Saved: {MAP_DATA_PATH}")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved: {LABEL_PATH}")


if __name__ == "__main__":
    main()
