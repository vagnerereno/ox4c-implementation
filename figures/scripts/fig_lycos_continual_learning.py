"""Generate the four independent LYCOS-IDS2017 adaptation panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"frozen": "#8A94A3", "latest_batch": "#E69F00", "rolling_replay": "#0072B2"}
LABELS = {"frozen": "Frozen", "latest_batch": "Latest-batch", "rolling_replay": "Rolling replay"}
FAMILY_LABELS = {
    "bot": "Bot",
    "ddos": "Distributed Denial of Service",
    "dos_goldeneye": "Denial of Service GoldenEye",
    "dos_hulk": "Denial of Service Hulk",
    "dos_slowhttptest": "Denial of Service Slowhttptest",
    "dos_slowloris": "Denial of Service Slowloris",
    "ftp_patator": "FTP-Patator",
    "heartbleed": "Heartbleed",
    "portscan": "PortScan",
    "ssh_patator": "SSH-Patator",
    "webattack_bruteforce": "Web Attack Brute Force",
    "webattack_sql_injection": "Web Attack SQL Injection",
    "webattack_xss": "Web Attack Cross-site Scripting",
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 20,
        "axes.labelsize": 22,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "legend.fontsize": 18,
        "axes.linewidth": 1.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_panel(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_policy_curves(ax, frame: pd.DataFrame, metric: str) -> None:
    grouped = frame.groupby(["label_delay_batches", "policy"])[metric].agg(["mean", "std"]).reset_index()
    for policy in ("frozen", "latest_batch", "rolling_replay"):
        data = grouped.loc[grouped["policy"].eq(policy)].sort_values("label_delay_batches")
        ax.errorbar(
            data["label_delay_batches"], data["mean"], yerr=data["std"],
            color=COLORS[policy], label=LABELS[policy], marker="o", markersize=9,
            linewidth=3.1, capsize=5.5,
        )
    ax.set_xticks([0, 1, 2, 5])
    ax.set_xlabel("Label delay (batches)")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def plot_online_metric(aggregate: pd.DataFrame, metric: str, output_dir: Path, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    add_policy_curves(ax, aggregate, metric)
    ax.set_ylabel(metric.upper())
    ax.set_ylim(-0.03, 1.02)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.96, bottom=0.20)
    save_panel(fig, output_dir, stem)


def plot_retention(retention: pd.DataFrame, output_dir: Path) -> None:
    final_retention = retention.loc[retention["stage_day"].eq("friday")]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    add_policy_curves(ax, final_retention, "macro_attack_recall")
    ax.set_ylabel("Friday macro recall")
    ax.set_ylim(0, 0.72)
    fig.subplots_adjust(left=0.21, right=0.98, top=0.96, bottom=0.20)
    save_panel(fig, output_dir, "fig_lycos_c_retention")


def plot_family_heatmap(family: pd.DataFrame, output_dir: Path) -> None:
    replay = family.loc[
        family["stage_day"].eq("friday") & family["policy"].eq("rolling_replay")
    ]
    matrix = replay.pivot_table(
        index="attack_family", columns="label_delay_batches", values="recall", aggfunc="mean"
    ).sort_index()
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    delay_labels = [
        "Immediate (0)" if value == 0 else
        f"{int(value)} batch" if value == 1 else
        f"{int(value)} batches"
        for value in matrix.columns
    ]
    ax.set_xticks(np.arange(len(matrix.columns)), delay_labels)
    ax.set_yticks(
        np.arange(len(matrix.index)),
        [FAMILY_LABELS.get(value, value.replace("_", " ").title()) for value in matrix.index],
    )
    ax.set_xlabel("Trusted-label delay under rolling replay")
    ax.set_ylabel("Attack family")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean Friday recall")
    fig.subplots_adjust(left=0.25, right=0.92, top=0.98, bottom=0.16)
    save_panel(fig, output_dir, "fig_lycos_d_family_heatmap")


def main() -> None:
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--data-dir", type=Path, default=default_root / "results" / "lycos_ids2017",
    )
    parser.add_argument(
        "--additional-data-dir", type=Path, action="append", default=[],
        help="additional result directory to merge, such as a zero-delay run",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default_root / "figures" / "output",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_dirs = [args.data_dir, *args.additional_data_dir]
    aggregate = pd.concat(
        [pd.read_csv(path / "aggregate_results.csv") for path in data_dirs],
        ignore_index=True,
    )
    retention = pd.concat(
        [pd.read_csv(path / "retention_results.csv") for path in data_dirs],
        ignore_index=True,
    )
    family = pd.concat(
        [pd.read_csv(path / "retention_by_family.csv") for path in data_dirs],
        ignore_index=True,
    )

    configure_style()
    plot_online_metric(aggregate, "f1", args.output_dir, "fig_lycos_a_f1")
    plot_online_metric(aggregate, "mcc", args.output_dir, "fig_lycos_b_mcc")
    plot_retention(retention, args.output_dir)
    plot_family_heatmap(family, args.output_dir)


if __name__ == "__main__":
    main()
