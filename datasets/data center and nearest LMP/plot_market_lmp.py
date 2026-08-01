"""
Plot LMP traces for ERCOT, NYISO, ISONE, and PJM in separate figures.
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "racer_cdc_mplconfig"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

MARKETS = [
    {
        "name": "ERCOT",
        "input": SCRIPT_DIR / "ERCOT_finalized.csv",
        "output": SCRIPT_DIR / "ercot_lmp_at_datacenters.png",
        "date_col": "Date",
        "data_center_col": "Data Center",
        "meta_cols": [
            "Date",
            "Data Center",
            "Settlement.Point.Name",
            "Latitude",
            "Longitude",
        ],
        "legend_groups": [
            {
                "heading": "Northwestern ERCOT",
                "series": [
                    {"label": "QTS_Irving", "color": "#7DB7E8"},
                    {"label": "Denton Data center", "color": "#7DB7E8"},
                    {
                        "label": "Open AI/ SoftBank Stargate Data Center (Lancium Clean Campus)",
                        "color": "#08306B",
                    },
                ],
            },
            {
                "heading": "Urban South ERCOT",
                "series": [
                    {"label": "Houston One Data Center", "color": "#8B0000"},
                    {"label": "Oracle-Austin Data Center", "color": "#F28E8E"},
                    {"label": "Microsoft San Antonio Data Center", "color": "#F28E8E"},
                ],
            },
        ],
        "legend_zone_col": "Settlement.Point.Name",
    },
    {
        "name": "ISONE",
        "input": SCRIPT_DIR / "ISONE_finalized.csv",
        "output": SCRIPT_DIR / "isone_lmp_at_datacenters.png",
        "date_col": "Date",
        "data_center_col": "Data Center",
        "meta_cols": [
            "Date",
            "Data Center",
            "Latitude",
            "Longitude",
            "location_id",
            "Load Zone",
        ],
        "series_colors": {
            "Medford Data Center": "#d62728",
            "Markely Data Center": "#d62728",
        },
        "legend_zone_col": "Load Zone",
    },
    {
        "name": "NYISO",
        "input": SCRIPT_DIR / "NYISO_finalized.csv",
        "output": SCRIPT_DIR / "nyiso_lmp_at_datacenters.png",
        "date_col": "Date",
        "data_center_col": "Data center",
        "meta_cols": [
            "Date",
            "Data center",
            "Latitude",
            "Longitude",
            "Load Zone",
            "PTID",
        ],
        "series_colors": {
            "BlockFusion aka North East Data Center LLC": "#1f77b4",
            "North Tonawanda Data Center": "#1f77b4",
        },
        "series_order": [
            "BlockFusion aka North East Data Center LLC",
            "North Tonawanda Data Center",
            "Global AI",
            "LGA1",
            "Orangeburg Data Center",
        ],
        "legend_zone_col": "Load Zone",
        "legend_loc": "upper left",
        "legend_bbox_to_anchor": (0.01, 0.99),
    },
    {
        "name": "PJM",
        "input": SCRIPT_DIR / "PJM_finalized.csv",
        "output": SCRIPT_DIR / "pjm_lmp_at_datacenters.png",
        "date_col": "date",
        "data_center_col": "Data center",
        "meta_cols": [
            "date",
            "Data center",
            "Latitude",
            "Longitude",
            "pnode_id",
            "pnode_name",
            "type",
            "zone",
        ],
        "legend_zone_col": "pnode_name",
        "legend_loc": "upper left",
        "legend_bbox_to_anchor": (0.01, 0.99),
    },
]


def short_time_label(label: str) -> str:
    return label[:5]


def format_series_label(label: str, row: pd.Series, config: dict[str, object]) -> str:
    zone_col = config.get("legend_zone_col")
    if not zone_col:
        return label

    zone = row.get(zone_col)
    if pd.isna(zone):
        return label

    return f"{label} ({zone})"


def series_for_plot(df: pd.DataFrame, config: dict[str, object]) -> list[tuple[str, pd.Series, str]]:
    data_center_col = config["data_center_col"]
    legend_groups = config.get("legend_groups")
    series_colors = config.get("series_colors", {})
    if not legend_groups:
        sorted_rows = list(df.sort_values(data_center_col).iterrows())
        rows_by_label = {row[data_center_col]: row for _, row in sorted_rows}
        color_by_label = {
            row[data_center_col]: DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            for i, (_, row) in enumerate(sorted_rows)
        }
        ordered_labels = [label for label in config.get("series_order", []) if label in rows_by_label]
        used_labels = set(ordered_labels)
        ordered_rows = [rows_by_label[label] for label in ordered_labels]
        ordered_rows.extend(row for _, row in sorted_rows if row[data_center_col] not in used_labels)

        return [
            (
                row[data_center_col],
                row,
                series_colors.get(row[data_center_col], color_by_label[row[data_center_col]]),
            )
            for row in ordered_rows
        ]

    rows_by_label = {row[data_center_col]: row for _, row in df.iterrows()}
    used_labels: set[str] = set()
    series = []

    for group in legend_groups:
        for item in group["series"]:
            label = item["label"]
            row = rows_by_label.get(label)
            if row is None:
                continue

            series.append((label, row, item["color"]))
            used_labels.add(label)

    fallback_colors = iter(DEFAULT_COLORS)
    for _, row in df.sort_values(data_center_col).iterrows():
        label = row[data_center_col]
        if label not in used_labels:
            series.append((label, row, next(fallback_colors, "#333333")))

    return series


def add_legend(ax: plt.Axes, config: dict[str, object], plotted_lines: dict[str, object]) -> None:
    legend_groups = config.get("legend_groups")
    if not legend_groups:
        legend_kwargs = {
            "fontsize": 8.5,
            "loc": config.get("legend_loc", "upper right"),
            "frameon": False,
        }
        if "legend_bbox_to_anchor" in config:
            legend_kwargs["bbox_to_anchor"] = config["legend_bbox_to_anchor"]
            legend_kwargs["borderaxespad"] = 0
        ax.legend(**legend_kwargs)
        return

    handles = []
    labels = []
    headings = set()

    for group in legend_groups:
        heading = group["heading"]
        handles.append(Line2D([], [], linestyle="none", color="none"))
        labels.append(heading)
        headings.add(heading)

        for item in group["series"]:
            label = item["label"]
            line = plotted_lines.get(label)
            if line is not None:
                handles.append(line)
                labels.append(line.get_label())

    legend = ax.legend(
        handles,
        labels,
        fontsize=8.5,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=False,
        borderaxespad=0,
        handlelength=2.6,
        handletextpad=0.7,
    )
    for text in legend.get_texts():
        if text.get_text() in headings:
            text.set_fontweight("bold")


def plot_market(config: dict[str, object]) -> None:
    df = pd.read_csv(config["input"], skiprows=1)
    meta_cols = set(config["meta_cols"])
    time_columns = [col for col in df.columns if col not in meta_cols]

    fig, ax = plt.subplots(figsize=(13, 5.8), constrained_layout=True)
    x = range(len(time_columns))
    tick_step = 4 if len(time_columns) <= 96 else 24
    tick_positions = list(range(0, len(time_columns), tick_step))
    tick_labels = [short_time_label(time_columns[i]) for i in tick_positions]

    plotted_lines = {}
    for label, row, color in series_for_plot(df, config):
        values = pd.to_numeric(row[time_columns], errors="coerce")
        display_label = format_series_label(label, row, config)
        (line,) = ax.plot(x, values, linewidth=1.7, color=color, label=display_label)
        plotted_lines[label] = line

    date = df[config["date_col"]].iloc[0]
    ax.set_title(f"{config['name']} LMP at Data Centers ({date})", fontsize=15)
    ax.set_xlabel("Time of day", fontsize=12)
    ax.set_ylabel("LMP ($/MWh)", fontsize=12)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    add_legend(ax, config, plotted_lines)

    fig.savefig(config["output"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {config['output']}")


def main() -> None:
    for config in MARKETS:
        plot_market(config)


if __name__ == "__main__":
    main()
