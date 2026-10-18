"""Generate independent explanation and computational trade-off panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
COLORS = {17: BLUE, 42: ORANGE, 73: GREEN}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "axes.linewidth": 1.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_panel(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_latency(evidence: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    x = np.arange(len(evidence))
    width = 0.34
    ax.bar(x - width / 2, evidence["decision_p95_ms"], width,
           color=BLUE, label="Decision availability")
    ax.bar(x + width / 2, evidence["shap_p95_ms"], width,
           color=ORANGE, label="TreeSHAP completion")
    for index, row in evidence.reset_index(drop=True).iterrows():
        ax.text(index + width / 2, row["shap_p95_ms"] + 1.5,
                f"n={int(row['explanations']):,}", ha="center", va="bottom", fontsize=16)
    ax.set_xticks(x, [str(value) for value in evidence["configuration"]])
    ax.set_xlabel("Grouped configuration")
    ax.set_ylabel("p95 latency (ms)")
    ax.set_ylim(0, max(evidence["shap_p95_ms"]) * 1.25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.80, bottom=0.18)
    save_panel(fig, output_dir, "fig_qut_cost_a_latency")


def plot_compact_tradeoff(evidence: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for _, row in evidence.iterrows():
        configuration = int(row["configuration"])
        color = COLORS[configuration]
        primary_size_mb = row["primary_size_bytes"] / (1024 ** 2)
        compact_size_mb = row["compact_size_bytes"] / (1024 ** 2)
        ax.annotate(
            "", xy=(compact_size_mb, row["compact_f1"]),
            xytext=(primary_size_mb, row["primary_f1"]),
            arrowprops={"arrowstyle": "->", "color": color, "linewidth": 2.5},
        )
        ax.scatter(primary_size_mb, row["primary_f1"], s=170, marker="o", color=color,
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.scatter(compact_size_mb, row["compact_f1"], s=170, marker="s", color=color,
                   edgecolor="white", linewidth=1.5, zorder=3)
        speedup = row["primary_p95_ms"] / row["compact_p95_ms"]
        vertical_offset = {17: -22, 42: 0, 73: 22}[configuration]
        ax.annotate(
            f"{configuration}: {speedup:.1f}$\\times$",
            xy=(compact_size_mb, row["compact_f1"]), xytext=(12, vertical_offset),
            textcoords="offset points", va="center", fontsize=16, color=color,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Serialized model size (MiB, log scale)")
    ax.set_ylabel("Test F1")
    ax.set_ylim(min(evidence["compact_f1"]) - 0.008, max(evidence["primary_f1"]) + 0.006)
    ax.grid(alpha=0.25, which="both")
    primary = plt.Line2D([], [], marker="o", linestyle="none", markersize=10,
                         color="#374151", label="Primary Random Forest")
    compact = plt.Line2D([], [], marker="s", linestyle="none", markersize=10,
                         color="#374151", label="Compact tree")
    ax.legend(handles=[primary, compact], frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.96, bottom=0.18)
    save_panel(fig, output_dir, "fig_qut_cost_b_compact")


def plot_shap_magnitude(shap: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ordered = shap.iloc[::-1].reset_index(drop=True)
    labels = [value.replace("_", " ") for value in ordered["feature"]]
    bars = ax.barh(labels, ordered["mean_abs_shap"], color=BLUE, alpha=0.90)
    for bar, value in zip(bars, ordered["mean_abs_shap"]):
        ax.text(value + shap["mean_abs_shap"].max() * 0.012,
                bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                va="center", ha="left", fontsize=16)
    ax.set_xlabel("Mean absolute TreeSHAP contribution")
    ax.set_xlim(0, shap["mean_abs_shap"].max() * 1.17)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=16)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.98, bottom=0.16)
    save_panel(fig, output_dir, "fig_qut_cost_c_shap")


def main() -> None:
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--data-dir", type=Path, default=default_root / "results" / "qut_dv25",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default_root / "figures" / "output",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    evidence = pd.read_csv(args.data_dir / "evidence_results.csv")
    shap = pd.read_csv(args.data_dir / "shap_global.csv").head(8)
    configure_style()
    plot_latency(evidence, args.output_dir)
    plot_compact_tradeoff(evidence, args.output_dir)
    plot_shap_magnitude(shap, args.output_dir)


if __name__ == "__main__":
    main()
