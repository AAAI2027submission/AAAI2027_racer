"""
Plot power consumption (avgcpu * corehour_norm) and QoS cost (10 * corehour_norm for
Interactive VMs) distributions across all datacenter VM traces.
corehour_norm is min-max normalised per datacenter to [0, 1].
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Load data (normalise corehour per datacenter, matching model convention) ───
SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = SCRIPT_DIR.parents[1]
files = sorted((BUNDLE_DIR / "datasets" / "datacenter_with_metrics").glob("datacenter_*.csv"))
dfs = []
for f in files:
    d = pd.read_csv(f)
    ch_min, ch_max = d["corehour"].min(), d["corehour"].max()
    d["corehour_norm"] = (d["corehour"] - ch_min) / max(ch_max - ch_min, 1e-8)
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)

df = df[(df["avgcpu"] > 10) & (df["corehour"] >= 1)].reset_index(drop=True)

df["power"] = df["avgcpu"] * df["corehour_norm"]
df["qos_cost"] = df.apply(
    lambda r: 10.0 * r["corehour_norm"] if r["vmcategory"] == "Interactive" else 0.0,
    axis=1,
)

categories = ["Interactive", "Delay-insensitive", "Unknown"]
colors     = {"Interactive": "#d62728", "Delay-insensitive": "#1f77b4", "Unknown": "#7f7f7f"}

BINS = 80
out_dir = SCRIPT_DIR
out_dir.mkdir(parents=True, exist_ok=True)

FS_LABEL  = 20   # axis labels
FS_TITLE  = 21   # panel titles
FS_TICK   = 18   # tick labels
FS_LEGEND = 18   # legend

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── Panel A: Power consumption distribution ────────────────────────────────────
ax = axes[0]
log_power = np.log10(df["power"].clip(lower=1e-3))
bins = np.linspace(log_power.min(), log_power.max(), BINS + 1)

for cat in categories:
    sub = np.log10(df.loc[df["vmcategory"] == cat, "power"].clip(lower=1e-3))
    ax.hist(sub, bins=bins, alpha=0.6, label=cat, color=colors[cat], density=True)

ax.set_xlabel(r"Power Consumption (log scale)", fontsize=FS_LABEL)
ax.set_ylabel("Density", fontsize=FS_LABEL)
ax.set_title("(a) Power Consumption Distribution", fontsize=FS_TITLE)
handles, labels = ax.get_legend_handles_labels()
labels = ["Delay-\ninsensitive" if l == "Delay-insensitive" else l for l in labels]
ax.legend(handles, labels, fontsize=FS_LEGEND, loc="upper left")
ax.tick_params(labelsize=FS_TICK)


# ── Panel B: QoS cost distribution (Interactive VMs only) ──────────────────────
ax = axes[1]
interactive_qos = df.loc[df["vmcategory"] == "Interactive", "qos_cost"]
log_qos = np.log10(interactive_qos.clip(lower=1e-3))
bins_qos = np.linspace(log_qos.min(), log_qos.max(), BINS + 1)

ax.hist(log_qos, bins=bins_qos, color=colors["Interactive"], alpha=0.75,
        density=True, label="Interactive")

# annotate median and 95th percentile
for pct, ls in [(50, "--"), (95, ":")]:
    val = np.percentile(interactive_qos, pct)
    ax.axvline(np.log10(val), color="k", linestyle=ls, linewidth=1.4,
               label=f"p{pct} = {val:.0f}")

ax.set_xlabel(r"QoS Cost (log scale)", fontsize=FS_LABEL)
ax.set_ylabel("Density", fontsize=FS_LABEL)
ax.set_title("(b) QoS Cost Distribution\n(Interactive VMs only)", fontsize=FS_TITLE)
ax.legend(fontsize=FS_LEGEND)
ax.tick_params(labelsize=FS_TICK)


plt.tight_layout()
out_path = out_dir / "power_qos_distribution.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
