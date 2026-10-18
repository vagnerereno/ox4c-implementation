"""Integrity audit for the LYCOS continual-learning experiment outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


POLICIES = ("frozen", "latest_batch", "rolling_replay")
ROOT = Path(__file__).resolve().parents[1]


def confusion_metrics(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn - fp * fn) / denominator) if denominator else 0.0
    return {"f1": float(f1), "mcc": float(mcc), "recall": float(recall)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=ROOT / "results" / "lycos_ids2017",
    )
    args = parser.parse_args()
    summary = json.loads((args.input / "summary.json").read_text(encoding="utf-8"))
    batches = pd.read_csv(args.input / "batch_results.csv")
    updates = pd.read_csv(args.input / "update_results.csv")
    aggregate = pd.read_csv(args.input / "aggregate_results.csv")
    retention = pd.read_csv(args.input / "retention_results.csv")
    retention_family = pd.read_csv(args.input / "retention_by_family.csv")

    configurations = summary["configurations"]
    delays = summary["label_delays_batches"]
    combinations = len(configurations) * len(delays)
    batches_per_combination = int(
        batches.groupby(["configuration", "label_delay_batches"])["global_batch"].nunique().iloc[0]
    )
    checks: dict[str, dict] = {}

    def record(name: str, passed: bool, evidence) -> None:
        checks[name] = {"passed": bool(passed), "evidence": evidence}

    observed_combinations = batches.groupby(
        ["configuration", "label_delay_batches"], observed=True,
    ).ngroups
    record(
        "all_configuration_delay_combinations",
        observed_combinations == combinations,
        {"expected": combinations, "observed": observed_combinations},
    )
    expected_batch_rows = combinations * batches_per_combination * len(POLICIES)
    record(
        "three_policy_rows_for_every_batch",
        len(batches) == expected_batch_rows
        and batches.groupby(
            ["configuration", "label_delay_batches", "global_batch"], observed=True,
        )["policy"].nunique().eq(len(POLICIES)).all(),
        {"expected_rows": expected_batch_rows, "observed_rows": len(batches)},
    )
    record(
        "policy_inputs_identical_within_batch",
        batches.groupby(
            ["configuration", "label_delay_batches", "global_batch"], observed=True,
        )[["events", "attacks"]].nunique().le(1).all().all(),
        "events and attacks agree across policies for every batch",
    )
    record(
        "feedback_released_after_exact_configured_delay",
        (
            updates["after_global_batch"] - updates["source_global_batch"]
            == updates["label_delay_batches"]
        ).all(),
        {
            "minimum_observed_delay": int(
                (updates["after_global_batch"] - updates["source_global_batch"]).min()
            ),
            "maximum_observed_delay": int(
                (updates["after_global_batch"] - updates["source_global_batch"]).max()
            ),
        },
    )

    replay_batches = batches.loc[batches["policy"].eq("rolling_replay")]
    expected_version = np.maximum(
        1,
        replay_batches["global_batch"].to_numpy()
        - replay_batches["label_delay_batches"].to_numpy(),
    )
    record(
        "replay_version_has_no_early_feedback",
        np.array_equal(replay_batches["model_version"].to_numpy(), expected_version),
        "prediction at batch t uses only labels released before batch t",
    )
    record(
        "frozen_model_never_changes",
        batches.loc[batches["policy"].eq("frozen"), "model_version"].eq(1).all(),
        "all frozen predictions use version 1",
    )

    maximum_metric_difference = 0.0
    for keys, group in batches.groupby(
        ["configuration", "label_delay_batches", "policy"], observed=True,
    ):
        tn, fp, fn, tp = (int(group[column].sum()) for column in ("tn", "fp", "fn", "tp"))
        recomputed = confusion_metrics(tn, fp, fn, tp)
        row = aggregate.loc[
            aggregate["configuration"].eq(keys[0])
            & aggregate["label_delay_batches"].eq(keys[1])
            & aggregate["policy"].eq(keys[2])
        ].iloc[0]
        maximum_metric_difference = max(
            maximum_metric_difference,
            *(abs(recomputed[metric] - float(row[metric])) for metric in recomputed),
        )
    record(
        "aggregate_metrics_reproduce_from_batch_confusions",
        maximum_metric_difference < 1e-12,
        {"maximum_absolute_difference": maximum_metric_difference},
    )
    expected_retention_rows = combinations * len(summary["prequential_stream"]) * len(POLICIES)
    record(
        "retention_evaluated_after_every_stream_day",
        len(retention) == expected_retention_rows,
        {"expected_rows": expected_retention_rows, "observed_rows": len(retention)},
    )
    record(
        "family_retention_contains_no_duplicate_keys",
        not retention_family.duplicated([
            "configuration", "label_delay_batches", "stage_day", "policy", "attack_family",
        ]).any(),
        {"rows": len(retention_family)},
    )
    key_columns = [
        "configuration", "label_delay_batches", "global_batch", "policy",
        "model_version", "events", "tn", "fp", "fn", "tp", "f1", "mcc",
    ]
    record(
        "batch_key_fields_complete",
        not batches[key_columns].isna().any().any(),
        {"rows": len(batches), "key_columns": key_columns},
    )
    passed = all(check["passed"] for check in checks.values())
    report = {
        "status": "passed" if passed else "failed",
        "protocol": summary["protocol"],
        "checks": checks,
        "output_row_counts": {
            "batch_results": len(batches),
            "update_results": len(updates),
            "aggregate_results": len(aggregate),
            "retention_results": len(retention),
            "retention_by_family": len(retention_family),
        },
    }
    (args.input / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
