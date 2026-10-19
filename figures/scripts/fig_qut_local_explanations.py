"""Generate reproducible local TreeSHAP explanation panels for QUT-DV25."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox


POSITIVE = "#009E73"
NEGATIVE = "#D55E00"
NEUTRAL = "#6B7280"


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "axes.linewidth": 1.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_events(event_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(event_root.glob("configuration_*/event_log.csv")):
        frame = pd.read_csv(path)
        frame["configuration"] = int(path.parent.name.replace("configuration_", ""))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No event logs found under {event_root}")
    events = pd.concat(frames, ignore_index=True)
    events["local_prediction"] = (events["local_probability"] >= 0.5).astype(int)
    return events


def median_confidence_case(frame: pd.DataFrame) -> pd.Series:
    ordered = frame.sort_values(
        ["confidence", "configuration", "sample_id"], kind="mergesort",
    ).reset_index(drop=True)
    return ordered.iloc[(len(ordered) - 1) // 2]


def select_cases(events: pd.DataFrame) -> list[dict[str, object]]:
    explained = events[events["explanation"].notna()].copy()

    primary = explained[
        (explained["true_label"] == 1)
        & (explained["prediction"] == 1)
        & (~explained["collaborative"])
        & (explained["decision_source"] == "random_forest")
    ]
    corrections = explained[
        explained["conflict"]
        & (explained["prediction"] == explained["true_label"])
        & (explained["local_prediction"] != explained["true_label"])
    ]
    if primary.empty or corrections.empty:
        raise ValueError("The required primary and collaborative cases were not found.")

    selected = []
    for case_type, eligible in (
        ("primary_true_positive", primary),
        ("collaborative_correction", corrections),
    ):
        row = median_confidence_case(eligible)
        explanation = ast.literal_eval(row["explanation"])
        contributions = explanation["full_contributions"]
        selected.append({
            "case_type": case_type,
            "eligible_cases": int(len(eligible)),
            "configuration": int(row["configuration"]),
            "sample_id": row["sample_id"],
            "true_label": int(row["true_label"]),
            "local_probability": float(row["local_probability"]),
            "final_probability": float(row["final_probability"]),
            "decision_source": row["decision_source"],
            "model_algorithm": explanation["model_algorithm"],
            "model_version": int(explanation["model_version"]),
            "output_space": explanation["output_space"],
            "base_value": float(explanation["base_value"]),
            "model_output": float(explanation["additive_reconstruction"]),
            "contributions": contributions,
        })
    return selected


def summarized_contributions(
    contributions: dict[str, float], top_k: int = 7,
) -> tuple[list[str], np.ndarray]:
    ranked = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    leading = ranked[:top_k]
    remaining = ranked[top_k:]
    labels = [name.replace("_", " ") for name, _ in leading]
    values = [float(value) for _, value in leading]
    if remaining:
        labels.append(f"Remaining (n={len(remaining)})")
        values.append(float(sum(value for _, value in remaining)))
    return labels, np.asarray(values, dtype=float)


def save_case(case: dict[str, object], output_dir: Path, stem: str) -> None:
    labels, values = summarized_contributions(case["contributions"])
    order = np.arange(len(labels))[::-1]
    colors = [POSITIVE if value >= 0 else NEGATIVE for value in values]

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    bars = ax.barh(order, values, color=colors, alpha=0.94, height=0.68)
    ax.axvline(0, color="#1F2937", linewidth=1.0)
    ax.set_yticks(order, labels)

    span = max(values.max(), abs(values.min()))
    padding = span * 0.10
    for bar, value in zip(bars, values):
        horizontal = value + padding * (0.20 if value >= 0 else -0.20)
        ax.text(
            horizontal,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=14,
            color="#111827",
        )

    if case["output_space"] == "probability":
        x_label = "TreeSHAP contribution\n(attack probability)"
    else:
        x_label = "TreeSHAP contribution\n(attack log-odds)"

    ax.set_xlabel(x_label)
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    limit = span * 1.35
    ax.set_xlim(-limit, limit)
    fig.subplots_adjust(left=0.34, right=0.97, top=0.97, bottom=0.20)
    crop = Bbox.from_extents(0.025, 0.06, fig.get_figwidth(), fig.get_figheight())
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches=crop)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches=crop)
    plt.close(fig)


def write_case_index(cases: list[dict[str, object]], output_path: Path) -> None:
    rows = []
    for case in cases:
        rows.append({key: value for key, value in case.items() if key != "contributions"})
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--event-dir", type=Path,
        default=root / "results" / "qut_dv25" / "complete",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "figures" / "output",
    )
    parser.add_argument(
        "--case-index", type=Path,
        default=root / "results" / "qut_dv25" / "local_explanation_cases.csv",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.case_index.parent.mkdir(parents=True, exist_ok=True)

    configure_style()
    cases = select_cases(load_events(args.event_dir))
    save_case(cases[0], args.output_dir, "fig_qut_local_a_primary")
    save_case(cases[1], args.output_dir, "fig_qut_local_b_correction")
    write_case_index(cases, args.case_index)


if __name__ == "__main__":
    main()
