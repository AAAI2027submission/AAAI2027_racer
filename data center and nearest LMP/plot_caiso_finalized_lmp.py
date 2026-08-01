"""
Plot finalized CAISO 5-minute LMP traces for the data centers in one figure.
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "racer_cdc_mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "caiso_finalized.csv"
OUTPUT_PATH = SCRIPT_DIR / "caiso_finalized_lmp_at5dc.png"

META_COLUMNS = ["Date", "Data Center", "Latitude", "Longitude", "NODE", "Unnamed: 5"]


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    time_columns = [col for col in df.columns if col not in META_COLUMNS]

    fig, ax = plt.subplots(figsize=(13, 5.8), constrained_layout=True)
    x = range(len(time_columns))
    tick_positions = list(range(0, len(time_columns), 24))
    tick_labels = [time_columns[i] for i in tick_positions]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    for color, (_, row) in zip(colors, df.sort_values("Data Center").iterrows()):
        values = pd.to_numeric(row[time_columns], errors="coerce")
        ax.plot(x, values, linewidth=1.8, color=color, label=row["Data Center"])

    date = df["Date"].iloc[0]
    ax.set_title(f"Finalized CAISO 5-Minute LMP at Five Data Centers ({date})", fontsize=15)
    ax.set_xlabel("Time of day", fontsize=12)
    ax.set_ylabel("LMP ($/MWh)", fontsize=12)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=False, borderaxespad=0)

    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
