"""Paper-ready figures and tables for all three real-data sweeps.

Sweep A  context_noise_real_data_v1   S in {8,20,50,100}, noise in {0,.1,.2,.3}
Sweep B  context_noise_schedule_real_v1  S=100, noise in {0,.2}, schedule sweep
Sweep C  load_balance_real_v1          S in {72,108}, noise in {0,.2}
"""

from __future__ import annotations

import os
from pathlib import Path
from matplotlib.lines import Line2D

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-yifu")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[1]
OUTROOT = ROOT / "docs/research/rmab_vm_outputs"

CN_PATH  = OUTROOT / "context_noise_real_data_v1/context_noise_summary.csv"
SCH_PATH = OUTROOT / "context_noise_schedule_real_v1/context_noise_summary.csv"
LB_PATH  = OUTROOT / "load_balance_real_v1/load_balance_summary.csv"

CN_OUT  = OUTROOT / "context_noise_real_data_v1"
SCH_OUT = OUTROOT / "context_noise_schedule_real_v1"
LB_OUT  = OUTROOT / "load_balance_real_v1"

BASELINE_LABELS = [
    "state_thompson", "tw_dense", "local_ucb_tw_dense",
    "global_ucb_tw_dense", "exp4_dense",
]
REFINED_CN_LABELS = [
    "tm_tw_refined_dense",
    "tm_tw_refined_gated_offline",
    "tm_tw_refined_gated_offline_low_rank",
    "tm_tw_refined_support_offline",
]
DISPLAY = {
    "state_thompson":                       "State Thompson",
    "tw_dense":                             "TW",
    "local_ucb_tw_dense":                   "Local UCB+TW",
    "global_ucb_tw_dense":                  "Global UCB+TW",
    "exp4_dense":                           "EXP4",
    "tm_tw_dense":                          "TM--TW",
    "tm_tw_refined_dense":                  "Adaptive TM--TW",
    "tm_tw_refined_gated_offline":          "Adaptive TM--TW + gated prior",
    "tm_tw_refined_gated_offline_low_rank": "Adaptive TM--TW + gated prior + low-rank",
    "tm_tw_refined_support_offline":        "Adaptive TM--TW + support/offline prior",
    "tm_tw_refined_low_rank":               "Adaptive TM--TW + low-rank",
}

CN_METHODS = [
    ("best_baseline",                         "Best paper\nbaseline",          "#7a8794"),
    ("tm_tw_dense",                           "Original\nTM-TW",               "#2f5f8f"),
    ("tm_tw_refined_dense",                   "Adaptive\ntrust",               "#2c7fb8"),
    ("tm_tw_refined_gated_offline",           "Adaptive\n+gated",              "#d9812e"),
    ("tm_tw_refined_gated_offline_low_rank",  "Adaptive\n+gated+LR",           "#c2410c"),
    ("tm_tw_refined_support_offline",         "Adaptive\n+support",            "#1f9a8a"),
]

LB_PLOT_METHODS = [
    ("state_thompson",                        "ST",                "#94a3b8", "Baselines"),
    ("tw_dense",                              "TW",                "#94a3b8", "Baselines"),
    ("local_ucb_tw_dense",                    "Local\nUCB+TW",     "#94a3b8", "Baselines"),
    ("exp4_dense",                            "EXP4",              "#94a3b8", "Baselines"),
    ("tm_tw_dense",                           "TM-TW",             "#2f5f8f", "Original"),
    ("tm_tw_refined_dense",                   "Adaptive\ntrust",   "#1f9a8a", "Refined"),
    ("tm_tw_refined_gated_offline",           "Adaptive\n+gated",  "#d9812e", "Refined"),
    ("tm_tw_refined_low_rank",                "Adaptive\n+LR",     "#c2410c", "Refined"),
    ("tm_tw_refined_gated_offline_low_rank",  "Adaptive\n+gated+LR","#7e22ce","Refined"),
    ("tm_tw_refined_support_offline",         "Adaptive\n+support","#0e7490", "Refined"),
]
LB_REFINED_LABELS = [lbl for lbl, *_ in LB_PLOT_METHODS if _ and _[-1] == "Refined"]
LB_GROUP_COLORS   = {"Baselines": "#94a3b8", "Original": "#2f5f8f", "Refined": "#d9812e"}


# ── helpers ───────────────────────────────────────────────────────────────────

def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.bbox": "tight",
    })


def cn_val(df: pd.DataFrame, S: int, noise: float, label: str) -> float:
    mask = (
        (df["S"] == S) &
        (abs(df["context_noise_level"] - noise) < 1e-9) &
        (df["policy_label"] == label)
    )
    r = df[mask]
    return float(r["mean_reward_pct_oracle"].iloc[0]) if not r.empty else float("nan")


def best_cn_baseline(df: pd.DataFrame, S: int, noise: float) -> float:
    return float(np.nanmax([cn_val(df, S, noise, l) for l in BASELINE_LABELS]))


def best_cn_baseline_with_label(df: pd.DataFrame, S: int, noise: float) -> tuple[str, float]:
    vals = [(l, cn_val(df, S, noise, l)) for l in BASELINE_LABELS]
    label, v = max(vals, key=lambda x: x[1] if not np.isnan(x[1]) else -999)
    return label, v


def best_cn_refined(df: pd.DataFrame, S: int, noise: float) -> tuple[str, float]:
    vals = [(l, cn_val(df, S, noise, l)) for l in REFINED_CN_LABELS]
    label, v = max(vals, key=lambda x: x[1] if not np.isnan(x[1]) else -999)
    return label, v


def sch_val(df: pd.DataFrame, noise: float, label: str,
            trust: float = 1.0, gate: float = 1.0, mode: str = "deterministic") -> float:
    mask = (
        (abs(df["context_noise_level"] - noise) < 1e-9) &
        (df["policy_label"] == label) &
        (abs(df["trust_scale_mult"] - trust) < 1e-9) &
        (abs(df["gate_scale_mult"] - gate) < 1e-9) &
        (df["gate_mode"] == mode)
    )
    r = df[mask]
    return float(r["mean_reward_pct_oracle"].iloc[0]) if not r.empty else float("nan")


def best_beta(df: pd.DataFrame, noise: float, label: str) -> tuple[float, float, float]:
    sub = df[(abs(df["context_noise_level"] - noise) < 1e-9) &
             (df["policy_label"] == label) &
             (df["gate_mode"] == "beta")]
    if sub.empty:
        return float("nan"), float("nan"), float("nan")
    best = sub.loc[sub["mean_reward_pct_oracle"].idxmax()]
    return float(best["mean_reward_pct_oracle"]), float(best["trust_scale_mult"]), float(best["gate_scale_mult"])


def lb_val(df: pd.DataFrame, S: int, noise: float, label: str,
           gate_mode: str = "deterministic", trust_floor: float = 0.10) -> float:
    mask = (
        (df["S"] == S) &
        (abs(df["context_noise_level"] - noise) < 1e-9) &
        (df["gate_mode"] == gate_mode) &
        (abs(df["trust_floor"] - trust_floor) < 1e-9) &
        (df["policy_label"] == label)
    )
    r = df[mask]
    return float(r["mean_reward_pct_oracle"].iloc[0]) if not r.empty else float("nan")


def best_lb_baseline(df, S, noise, gate_mode="deterministic", trust_floor=0.10) -> float:
    return float(np.nanmax([lb_val(df, S, noise, l, gate_mode, trust_floor) for l in BASELINE_LABELS]))


# ── Sweep A: context-noise robustness ─────────────────────────────────────────

def plot_sweep_a(df: pd.DataFrame) -> None:
    states  = [8, 20, 50, 100]
    noises  = [0.0, 0.1, 0.2, 0.3]

    # compute global y range across all plotted series so nothing is clipped
    all_vals = []
    for S in states:
        for label, _, _ in CN_METHODS:
            for noise in noises:
                v = best_cn_baseline(df, S, noise) if label == "best_baseline" \
                    else cn_val(df, S, noise, label)
                if not np.isnan(v):
                    all_vals.append(v)
    ymin = max(0.0, min(all_vals) - 3.0)
    ymax = min(106.0, max(all_vals) + 3.0)

    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), sharex=True, sharey=True)
    for ax, S in zip(axes.flat, states):
        for label, short, color in CN_METHODS:
            vals = []
            for noise in noises:
                if label == "best_baseline":
                    vals.append(best_cn_baseline(df, S, noise))
                else:
                    vals.append(cn_val(df, S, noise, label))
            ls  = "--" if label in {"best_baseline", "tm_tw_dense"} else "-"
            lw  = 1.8  if label == "best_baseline" else 2.1
            mk  = "s"  if label == "best_baseline" else ("^" if label == "tm_tw_dense" else "o")
            ax.plot(noises, vals, marker=mk, linewidth=lw, linestyle=ls, color=color, label=short)
        ax.set_title(f"S={S}", loc="left", fontweight="bold", fontsize=13, pad=7)
        ax.grid(True, axis="both", color="#d7d7d7", linewidth=0.7, alpha=0.75)
        ax.set_xticks(noises)
        ax.set_ylim(ymin, ymax)
    axes[0, 0].set_ylabel("Reward (% oracle)", fontsize=12)
    axes[1, 0].set_ylabel("Reward (% oracle)", fontsize=12)
    axes[1, 0].set_xlabel("Contextual state-noise probability", fontsize=12)
    axes[1, 1].set_xlabel("Contextual state-noise probability", fontsize=12)
    legend_handles = [
        Line2D([0], [0], color=color,
               linewidth=1.8 if label == "best_baseline" else 2.1,
               linestyle="--" if label in {"best_baseline", "tm_tw_dense"} else "-",
               marker="s" if label == "best_baseline" else ("^" if label == "tm_tw_dense" else "o"),
               markersize=5)
        for label, _, color in CN_METHODS
    ]
    legend_labels = [short for _, short, _ in CN_METHODS]
    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=6,
               frameon=False, bbox_to_anchor=(0.52, 0.0), handlelength=2.4, fontsize=11)
    fig.tight_layout(rect=[0, 0.10, 1, 1.0])
    fig.savefig(CN_OUT / "fig_context_noise_real.png", dpi=300, bbox_inches="tight")
    fig.savefig(CN_OUT / "fig_context_noise_real.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_context_noise_real.pdf/png")


def table_sweep_a(df: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sweep A (real VM data): contextual-noise robustness. "
        r"For each $(|S|, \text{noise})$ cell the table reports the best paper-style baseline "
        r"(with its name), original TM--TW, and the best refined variant with its margin over the best baseline.}",
        r"\label{tab:sweep_a_real}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{cclccclc}",
        r"\toprule",
        r"$|S|$ & Noise & Best baseline name & Best base & TM--TW "
        r"& Best refined & Best refined variant & Margin vs baseline \\",
        r"\midrule",
    ]
    for S in [8, 20, 50, 100]:
        for noise in [0.0, 0.1, 0.2, 0.3]:
            bb_lbl, bb     = best_cn_baseline_with_label(df, S, noise)
            tmtw           = cn_val(df, S, noise, "tm_tw_dense")
            best_lbl, best_v = best_cn_refined(df, S, noise)
            lines.append(
                f"{S} & {noise:.1f} & {DISPLAY.get(bb_lbl, bb_lbl)} & {bb:.2f} & "
                f"{tmtw:.2f} & {best_v:.2f} & "
                rf"\textbf{{{DISPLAY.get(best_lbl, best_lbl)}}} & "
                f"{best_v - bb:+.2f} \\\\"
            )
        if S != 100:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    (CN_OUT / "tab_context_noise_real.tex").write_text("\n".join(lines))
    print("Saved tab_context_noise_real.tex")


# ── Sweep B: schedule-conflict ablation ───────────────────────────────────────

def plot_sweep_b(df: pd.DataFrame, s_label: str = "S=100", out_path: Path = None) -> None:
    if out_path is None:
        out_path = SCH_OUT
    noises = [0.0, 0.2]
    bar_specs = [
        ("Adaptive\ntrust",          "#2c7fb8", "tm_tw_refined_dense",
         dict(trust=1.0, gate=1.0, mode="deterministic")),
        ("Adaptive\n+beta gate",     "#5b9bd5", "tm_tw_refined_dense",
         None),  # best beta, no offline prior
        ("Default gated\nprior",     "#d9812e", "tm_tw_refined_gated_offline",
         dict(trust=1.0, gate=1.0, mode="deterministic")),
        ("Best beta\ngated prior",   "#e8a44a", "tm_tw_refined_gated_offline",
         None),  # best beta with offline prior
        ("Default gated\n+low-rank", "#b7791f", "tm_tw_refined_gated_offline_low_rank",
         dict(trust=1.0, gate=1.0, mode="deterministic")),
        ("Best beta\ngated+LR",      "#c2410c", "tm_tw_refined_gated_offline_low_rank",
         None),  # resolved below
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), sharey=True)
    for ax, noise in zip(axes, noises):
        values = []
        annotations = []
        for short, color, label, kw in bar_specs:
            if kw is None:
                v, ts, gs = best_beta(df, noise, label)
                annotations.append(f"best beta: trust={ts:.2g} gate={gs:.2g}")
            else:
                v = sch_val(df, noise, label, **kw)
                annotations.append("")
            values.append(v)
        xs   = np.arange(len(bar_specs))
        bars = ax.bar(xs, values,
                      color=[c for _, c, _, _ in bar_specs],
                      edgecolor="#1f1f1f", linewidth=0.55, width=0.68)
        baseline = values[0]
        ax.axhline(baseline, linestyle="--", color="#2c7fb8", linewidth=1.0, alpha=0.8)
        ax.set_title(f"{s_label},  noise={noise:.1f}", loc="left", fontweight="bold", pad=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([s.replace("\n", " ") for s, *_ in bar_specs],
                           rotation=45, ha="right")
        ax.grid(axis="y", color="#d7d7d7", linewidth=0.7, alpha=0.75)
        yvals = [v for v in values if not np.isnan(v)]
        ax.set_ylim(max(0, min(yvals) - 4), 102)
        for bar, v, note in zip(bars, values, annotations):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            if note:
                ax.text(0.98, 0.05, note, transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=7.5, color="#333333")
    axes[0].set_ylabel("Reward (% oracle)")
    fig.tight_layout()
    stem = "fig_schedule_conflict_real"
    fig.savefig(out_path / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_path / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path / stem}.pdf/png")


def table_sweep_b(df: pd.DataFrame) -> None:
    specs = [
        ("Adaptive trust",             "tm_tw_refined_dense",
         dict(trust=1.0, gate=1.0, mode="deterministic")),
        ("Adaptive trust + beta gate", "tm_tw_refined_dense", None),
        ("Default gated prior",        "tm_tw_refined_gated_offline",
         dict(trust=1.0, gate=1.0, mode="deterministic")),
        ("Best beta gated prior",      "tm_tw_refined_gated_offline", None),
        ("Default gated+low-rank",     "tm_tw_refined_gated_offline_low_rank",
         dict(trust=1.0, gate=1.0, mode="deterministic")),
        ("Best beta gated+low-rank",   "tm_tw_refined_gated_offline_low_rank", None),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sweep B (real VM data): schedule ablation at $|S|=100$. "
        r"On real data the default gated prior consistently \emph{helps} adaptive trust, "
        r"unlike the synthetic setting where it was counterproductive at high noise.}",
        r"\label{tab:sweep_b_real}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{clcccc}",
        r"\toprule",
        r"Noise & Variant & Gate mode & Trust scale & Gate scale & Reward (\% oracle) \\",
        r"\midrule",
    ]
    for noise in [0.0, 0.2]:
        first = True
        for label_str, policy, kw in specs:
            prefix = f"{noise:.1f}" if first else ""
            first  = False
            if kw is None:
                v, ts, gs = best_beta(df, noise, policy)
                mode, trust, gate = "beta", ts, gs
                method = rf"\textbf{{{label_str}}}"
            else:
                v     = sch_val(df, noise, policy, **kw)
                mode  = kw["mode"]
                trust = kw["trust"]
                gate  = kw["gate"]
                method = label_str
            lines.append(
                f"{prefix} & {method} & {mode} & "
                f"{trust:.2f} & {gate:.2f} & {v:.2f} \\\\"
            )
        if noise != 0.2:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    (SCH_OUT / "tab_schedule_conflict_real.tex").write_text("\n".join(lines))
    print("Saved tab_schedule_conflict_real.tex")


# ── Sweep C: load-balance (figure + tables already generated separately,
#    but regenerate here for consistency) ─────────────────────────────────────

def plot_sweep_c(df: pd.DataFrame) -> None:
    # refined variants only as bars; best-baseline and TM-TW as reference lines
    REFINED_METHODS = [
        ("tm_tw_refined_dense",                  "Adaptive\ntrust",    "#1f9a8a"),
        ("tm_tw_refined_gated_offline",          "Adaptive\n+gated",   "#d9812e"),
        ("tm_tw_refined_low_rank",               "Adaptive\n+LR",      "#c2410c"),
        ("tm_tw_refined_gated_offline_low_rank", "Adaptive\n+gated+LR","#7e22ce"),
        ("tm_tw_refined_support_offline",        "Adaptive\n+support", "#0e7490"),
    ]

    configs = [(72, 0.0), (72, 0.2), (108, 0.0), (108, 0.2)]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharey=False)
    for ax, (S, noise) in zip(axes.flat, configs):
        available = [(lbl, short, color) for lbl, short, color in REFINED_METHODS
                     if not np.isnan(lb_val(df, S, noise, lbl))]
        values = [lb_val(df, S, noise, lbl) for lbl, *_ in available]
        xs = np.arange(len(available))
        bars = ax.bar(xs, values,
                      color="white",
                      edgecolor=[color for _, _, color in available],
                      linewidth=1.2, width=0.72,
                      hatch="//")

        bb   = best_lb_baseline(df, S, noise)
        tmtw = lb_val(df, S, noise, "tm_tw_dense")
        ax.axhline(bb,    color="#94a3b8", linestyle="--", linewidth=2.8, alpha=1.0)
        ax.axhline(tmtw,  color="#2f5f8f", linestyle="--", linewidth=2.8, alpha=1.0)
        ax.axhline(100.0, color="#222",    linestyle=":",  linewidth=0.9, alpha=0.5)

        all_vals = values + [bb, tmtw]
        ymin = max(0.0, min(v for v in all_vals if not np.isnan(v)) - 5.0)
        ymax = min(104.0, max(v for v in all_vals if not np.isnan(v)) + 4.0)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(xs)
        ax.set_xticklabels([short for _, short, _ in available], fontsize=7.5)
        ax.set_ylabel("Reward (% oracle)")
        ax.set_title(f"S={S},  noise={noise:.1f}", loc="left", fontweight="bold", pad=6)
        ax.grid(axis="y", color="#d7d7d7", linewidth=0.6, alpha=0.75)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
        best_r = max(values)
        ax.annotate(f"best refined: +{best_r - tmtw:.1f} vs TM-TW",
                    xy=(0.98, 0.04), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=7.5, color="#c2410c", fontweight="bold")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([0], [0], color="#94a3b8", linestyle="--", linewidth=2.8),
        Line2D([0], [0], color="#2f5f8f", linestyle="--", linewidth=2.8),
        Patch(facecolor="white", edgecolor="#d9812e", linewidth=1.2, hatch="//"),
    ]
    fig.legend(handles, ["Best paper baseline", "Original TM-TW", "Refined TM-TW"],
               frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.52, 1.01))
    fig.suptitle("Sweep C — Load-balance stress test (real VM data)",
                 x=0.02, y=1.04, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(LB_OUT / "fig_load_balance_real.png", dpi=300)
    fig.savefig(LB_OUT / "fig_load_balance_real.pdf")
    plt.close(fig)
    print("Saved fig_load_balance_real.pdf/png")


def table_sweep_c_key(df: pd.DataFrame) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"States & Noise & Best base & TM--TW & Adaptive & +gated & +low-rank & +gated+LR \\",
        r"\midrule",
    ]
    for S in [72, 108]:
        for noise in [0.0, 0.2]:
            lines.append(
                f"{S} & {noise:.1f} & "
                f"{best_lb_baseline(df,S,noise):.2f} & "
                f"{lb_val(df,S,noise,'tm_tw_dense'):.2f} & "
                f"{lb_val(df,S,noise,'tm_tw_refined_dense'):.2f} & "
                f"{lb_val(df,S,noise,'tm_tw_refined_gated_offline'):.2f} & "
                f"{lb_val(df,S,noise,'tm_tw_refined_low_rank'):.2f} & "
                f"{lb_val(df,S,noise,'tm_tw_refined_gated_offline_low_rank'):.2f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (LB_OUT / "tab_load_balance_real_key_results.tex").write_text("\n".join(lines))
    print("Saved tab_load_balance_real_key_results.tex")


def table_sweep_c_aggregate(df: pd.DataFrame) -> None:
    configs = [
        (S, noise, gm, tf)
        for S     in [72, 108]
        for noise in [0.0, 0.2]
        for gm    in ["deterministic", "beta"]
        for tf    in [0.10, 0.20]
    ]
    refined_labels = [
        ("tm_tw_refined_dense",                 "Adaptive trust"),
        ("tm_tw_refined_gated_offline",         "Adaptive + gated prior"),
        ("tm_tw_refined_low_rank",              "Adaptive + low-rank"),
        ("tm_tw_refined_gated_offline_low_rank","Adaptive + gated + low-rank"),
        ("tm_tw_refined_support_offline",       "Adaptive + support/offline prior"),
    ]
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Refinement & Avg.\ gain over TM--TW & Wins & Worst gain \\",
        r"\midrule",
    ]
    for label, disp in refined_labels:
        gains = []
        for S, noise, gm, tf in configs:
            base = lb_val(df, S, noise, "tm_tw_dense", gm, tf)
            ref  = lb_val(df, S, noise, label, gm, tf)
            if not (np.isnan(base) or np.isnan(ref)):
                gains.append(ref - base)
        if not gains:
            continue
        lines.append(
            f"{disp} & {np.mean(gains):+.2f} & "
            f"{sum(g>0 for g in gains)}/{len(gains)} & "
            f"{np.min(gains):+.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (LB_OUT / "tab_load_balance_real_aggregate.tex").write_text("\n".join(lines))
    print("Saved tab_load_balance_real_aggregate.tex")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_style()
    cn_df  = pd.read_csv(CN_PATH)
    sch_df = pd.read_csv(SCH_PATH)
    lb_df  = pd.read_csv(LB_PATH)

    print("=== Sweep A ===")
    plot_sweep_a(cn_df)
    table_sweep_a(cn_df)

    print("=== Sweep B ===")
    plot_sweep_b(sch_df)
    table_sweep_b(sch_df)

    print("=== Sweep C ===")
    plot_sweep_c(lb_df)
    table_sweep_c_key(lb_df)
    table_sweep_c_aggregate(lb_df)


if __name__ == "__main__":
    main()
