"""Generate the three independent QUT-DV25 component-effect panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
GRAY = "#8A94A3"
INK = "#1F2937"


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


def plot_profile_effect(profiles: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    order = ["baseline", "optimized", "collaborative"]
    x = np.arange(len(order))
    colors = {17: BLUE, 42: ORANGE, 73: GREEN}
    for configuration, group in profiles.groupby("configuration"):
        values = group.set_index("profile").loc[order, "mcc"].to_numpy()
        ax.plot(
            x, values, marker="o", markersize=8, linewidth=2.4,
            color=colors[int(configuration)], label=f"Configuration {configuration}",
        )
    means = profiles.groupby("profile")["mcc"].mean().reindex(order).to_numpy()
    ax.plot(x, means, color=INK, linewidth=3.2, marker="D", markersize=8, label="Mean")
    ax.set_xticks(x, ["Baseline", "Optimized", "Collaborative"], rotation=18, ha="right")
    ax.set_ylabel("MCC (expanded axis)")
    ax.set_ylim(0.95, 0.986)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.78, bottom=0.20)
    save_panel(fig, output_dir, "fig_qut_effects_a_profile")


def plot_feature_deployment(optimization: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    x = np.arange(len(optimization))
    ax.axhline(36, color=GRAY, linestyle="--", linewidth=2)
    for index, row in optimization.reset_index(drop=True).iterrows():
        accepted = bool(row["accepted"])
        color = GREEN if accepted else RED
        ax.plot(
            [index, index], [row["candidate_features"], row["deployed_features"]],
            color=color, linewidth=3, zorder=1,
        )
        ax.scatter(index, row["candidate_features"], marker="o", s=150,
                   facecolor="white", edgecolor=color, linewidth=2.5, zorder=3)
        ax.scatter(index, row["deployed_features"], marker="D", s=130,
                   color=color, zorder=4)
        delta = row["candidate_calibration_mcc"] - row["full_calibration_mcc"]
        ax.text(index, 27.6, f"{delta:+.3f}", ha="center", va="top", fontsize=16)
    ax.set_xticks(x, [str(value) for value in optimization["configuration"]])
    ax.set_xlabel("Grouped configuration")
    ax.set_ylabel("Number of features")
    ax.set_ylim(26, 38)
    ax.grid(axis="y", alpha=0.25)
    candidate = plt.Line2D([], [], marker="o", linestyle="none", markersize=9,
                           markerfacecolor="white", markeredgecolor=INK, label="Candidate")
    deployed = plt.Line2D([], [], marker="D", linestyle="none", markersize=8,
                          color=INK, label="Deployed")
    ax.legend(handles=[candidate, deployed], frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.80, bottom=0.18)
    save_panel(fig, output_dir, "fig_qut_effects_b_features")


def plot_conflict_outcomes(collaboration: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 3.7))
    y = np.arange(len(collaboration))
    preserved = collaboration["requests_conflicts"] - collaboration["changed_decisions"]
    corrections = collaboration["corrections"]
    harms = collaboration["harms"]
    totals = collaboration["requests_conflicts"]
    preserved_pct = preserved / totals * 100
    corrections_pct = corrections / totals * 100
    harms_pct = harms / totals * 100
    ax.barh(y, preserved_pct, color="#C9D1D9", label="Preserved")
    ax.barh(y, corrections_pct, left=preserved_pct, color=GREEN, label="Correction")
    ax.barh(y, harms_pct, left=preserved_pct + corrections_pct, color=RED, label="Harm")
    for index in range(len(collaboration)):
        ax.text(2, index, f"{int(preserved.iloc[index])} preserved", va="center", fontsize=16)
        changed_x = preserved_pct.iloc[index] + (corrections_pct.iloc[index] + harms_pct.iloc[index]) / 2
        ax.text(changed_x, index, f"{int(corrections.iloc[index])}/{int(harms.iloc[index])}",
                ha="center", va="center", fontsize=16, color="white", fontweight="bold")
        ax.text(104, index, f"n={int(totals.iloc[index])}", va="center", fontsize=16)
    ax.set_yticks(y, [str(value) for value in collaboration["configuration"]])
    ax.set_ylabel("Grouped configuration")
    ax.set_xlim(0, 114)
    ax.set_xlabel("Share of classifier conflicts (%)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.78, bottom=0.24)
    save_panel(fig, output_dir, "fig_qut_effects_c_conflicts")


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

    profiles = pd.read_csv(args.data_dir / "profile_results.csv")
    profiles = profiles.loc[profiles["profile"].isin(["baseline", "optimized", "collaborative"])]
    optimization = pd.read_csv(args.data_dir / "optimization_results.csv")
    collaboration = pd.read_csv(args.data_dir / "collaboration_results.csv")

    configure_style()
    plot_profile_effect(profiles, args.output_dir)
    plot_feature_deployment(optimization, args.output_dir)
    plot_conflict_outcomes(collaboration, args.output_dir)


if __name__ == "__main__":
    main()
