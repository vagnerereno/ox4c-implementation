"""Confirmatory continual-learning experiment on LYCOS-IDS2017.

The experiment isolates the OX4C Continuous Learning dimension.  Every batch
is predicted before its labels become available.  It compares:

* ``frozen``: no model update;
* ``latest_batch``: naive retraining from only the latest released batch;
* ``rolling_replay``: retraining from a bounded history of previous and new
  knowledge.

Label delay is measured in complete future batches.  A label released after
delay ``d`` can only affect predictions from batch ``t + d + 1`` onward.  A
separate retention panel never enters training and is evaluated after each
capture day to quantify attack-family retention and forgetting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ox4c.adapters import LYCOSIDS2017TemporalAdapter, TemporalDayData  # noqa: E402
from ox4c.config import ModelConfig  # noqa: E402
from ox4c.services.model import DetectorModel  # noqa: E402

DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday")
INITIAL_DAYS = ("monday", "tuesday")
STREAM_DAYS = ("wednesday", "thursday", "friday")
POLICIES = ("frozen", "latest_batch", "rolling_replay")
PRIMARY_METRICS = ("f1", "mcc", "recall", "specificity", "false_positive_rate")


def score(y: pd.Series, probability: np.ndarray) -> dict[str, float | int]:
    """Compute binary detection metrics for a prequential batch."""
    truth = np.asarray(y, dtype=int)
    prediction = (np.asarray(probability, dtype=float) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "specificity": float(specificity),
        "false_positive_rate": float(1.0 - specificity),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(truth, prediction)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


@dataclass
class DayPartition:
    stream_X: pd.DataFrame
    stream_y: pd.Series
    stream_labels: pd.Series
    stream_ids: pd.Series
    retention_X: pd.DataFrame
    retention_y: pd.Series
    retention_labels: pd.Series
    retention_ids: pd.Series


def _systematic_indices(indices: np.ndarray, count: int) -> np.ndarray:
    if count >= len(indices):
        return indices.copy()
    positions = ((np.arange(count, dtype=float) + 0.5) * len(indices) / count)
    return indices[np.minimum(positions.astype(int), len(indices) - 1)]


def partition_retention(
    data: TemporalDayData,
    fraction: float,
    maximum_per_label: int,
) -> DayPartition:
    """Create a fixed, temporally distributed panel excluded from learning."""
    retained: list[np.ndarray] = []
    labels = data.attack_labels.reset_index(drop=True)
    for label in sorted(labels.unique()):
        indices = np.flatnonzero(labels.to_numpy() == label)
        if len(indices) <= 1:
            continue
        count = max(1, int(np.floor(len(indices) * fraction)))
        count = min(maximum_per_label, count, len(indices) - 1)
        retained.append(_systematic_indices(indices, count))
    retention_indices = (
        np.sort(np.concatenate(retained)) if retained else np.empty(0, dtype=int)
    )
    stream_mask = np.ones(len(labels), dtype=bool)
    stream_mask[retention_indices] = False
    stream_indices = np.flatnonzero(stream_mask)

    def take(frame, indices):
        return frame.iloc[indices].reset_index(drop=True)

    return DayPartition(
        stream_X=take(data.X, stream_indices),
        stream_y=take(data.y, stream_indices),
        stream_labels=take(labels, stream_indices),
        stream_ids=take(data.sample_ids, stream_indices),
        retention_X=take(data.X, retention_indices),
        retention_y=take(data.y, retention_indices),
        retention_labels=take(labels, retention_indices),
        retention_ids=take(data.sample_ids, retention_indices),
    )


def bounded_append(
    history_X: pd.DataFrame,
    history_y: pd.Series,
    feedback_X: pd.DataFrame,
    feedback_y: pd.Series,
    maximum_rows: int,
) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.concat([history_X, feedback_X], ignore_index=True)
    y = pd.concat([history_y, feedback_y], ignore_index=True)
    if len(y) > maximum_rows:
        X = X.iloc[-maximum_rows:].reset_index(drop=True)
        y = y.iloc[-maximum_rows:].reset_index(drop=True)
    return X, y


def systematic_limit(
    X: pd.DataFrame, y: pd.Series, maximum_rows: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(y) <= maximum_rows:
        return X.reset_index(drop=True), y.reset_index(drop=True)
    indices = _systematic_indices(np.arange(len(y)), maximum_rows)
    return X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)


def aggregate(parts_y: list[np.ndarray], parts_probability: list[np.ndarray]) -> dict:
    return score(
        pd.Series(np.concatenate(parts_y), dtype=int),
        np.concatenate(parts_probability),
    )


def train_model(
    policy: str,
    configuration: int,
    version: int,
    features: list[str],
    X: pd.DataFrame,
    y: pd.Series,
    model_config: ModelConfig,
) -> DetectorModel:
    return DetectorModel.train(
        f"lycos-{policy}-c{configuration}", version, features, X, y, model_config,
    )


def retention_evaluation(
    models: dict[str, DetectorModel],
    partitions: dict[str, DayPartition],
    available_days: tuple[str, ...],
    features: list[str],
    best_recall: dict[str, dict[str, float]],
    configuration: int,
    delay: int,
    stage_day: str,
) -> tuple[list[dict], list[dict]]:
    X = pd.concat(
        [partitions[day].retention_X for day in available_days], ignore_index=True,
    ).loc[:, features]
    y = pd.concat(
        [partitions[day].retention_y for day in available_days], ignore_index=True,
    )
    labels = pd.concat(
        [partitions[day].retention_labels for day in available_days], ignore_index=True,
    )
    aggregate_rows: list[dict] = []
    family_rows: list[dict] = []
    for policy, model in models.items():
        probability = model.predict_proba(X)
        prediction = probability >= 0.5
        metrics = score(y, probability)
        attack_recalls: list[float] = []
        forgetting_values: list[float] = []
        for label in sorted(value for value in labels.unique() if value != "benign"):
            mask = labels.eq(label).to_numpy()
            current_recall = float(prediction[mask].mean())
            previous_best = best_recall[policy].get(label, current_recall)
            forgetting = max(0.0, previous_best - current_recall)
            best_recall[policy][label] = max(previous_best, current_recall)
            attack_recalls.append(current_recall)
            forgetting_values.append(forgetting)
            family_rows.append({
                "configuration": configuration,
                "label_delay_batches": delay,
                "stage_day": stage_day,
                "policy": policy,
                "model_version": model.version,
                "attack_family": label,
                "events": int(mask.sum()),
                "recall": current_recall,
                "best_previous_or_current_recall": previous_best,
                "forgetting": forgetting,
            })
        aggregate_rows.append({
            "configuration": configuration,
            "label_delay_batches": delay,
            "stage_day": stage_day,
            "policy": policy,
            "model_version": model.version,
            "retention_events": int(len(y)),
            "attack_families": int(len(attack_recalls)),
            "macro_attack_recall": (
                float(np.mean(attack_recalls)) if attack_recalls else np.nan
            ),
            "average_forgetting": (
                float(np.mean(forgetting_values)) if forgetting_values else 0.0
            ),
            **metrics,
        })
    return aggregate_rows, family_rows


def run_protocol(
    args,
    partitions: dict[str, DayPartition],
    features: list[str],
    configuration: int,
    delay: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    started = time.perf_counter()
    initial_X = pd.concat(
        [partitions[day].stream_X for day in INITIAL_DAYS], ignore_index=True,
    ).loc[:, features]
    initial_y = pd.concat(
        [partitions[day].stream_y for day in INITIAL_DAYS], ignore_index=True,
    )
    initial_X, initial_y = systematic_limit(
        initial_X, initial_y, args.max_training_rows,
    )
    model_config = ModelConfig(
        n_estimators=args.estimators,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=configuration,
    )
    training_started = time.perf_counter()
    initial = train_model(
        "initial", configuration, 1, features, initial_X, initial_y, model_config,
    )
    initial_training_seconds = time.perf_counter() - training_started
    models = {
        "frozen": initial,
        "latest_batch": initial,
        "rolling_replay": initial,
    }
    replay_X, replay_y = initial_X, initial_y
    pending: deque[tuple[int, int, str, pd.DataFrame, pd.Series]] = deque()
    batches: list[dict] = []
    updates: list[dict] = []
    retention_rows: list[dict] = []
    retention_family_rows: list[dict] = []
    overall_parts = {
        policy: {"y": [], "probability": []} for policy in POLICIES
    }
    day_parts = {
        day: {
            policy: {"y": [], "probability": []} for policy in POLICIES
        }
        for day in STREAM_DAYS
    }
    best_recall = {policy: {} for policy in POLICIES}
    global_batch = 0
    released_events = 0

    for day_index, day in enumerate(STREAM_DAYS, start=len(INITIAL_DAYS)):
        partition = partitions[day]
        for day_batch, start in enumerate(
            range(0, len(partition.stream_y), args.batch_size), start=1,
        ):
            stop = min(start + args.batch_size, len(partition.stream_y))
            X = partition.stream_X.iloc[start:stop].loc[:, features].reset_index(drop=True)
            y = partition.stream_y.iloc[start:stop].reset_index(drop=True)
            labels = partition.stream_labels.iloc[start:stop].reset_index(drop=True)
            global_batch += 1

            for policy in POLICIES:
                inference_started = time.perf_counter()
                probability = models[policy].predict_proba(X)
                inference_seconds = time.perf_counter() - inference_started
                overall_parts[policy]["y"].append(y.to_numpy())
                overall_parts[policy]["probability"].append(probability)
                day_parts[day][policy]["y"].append(y.to_numpy())
                day_parts[day][policy]["probability"].append(probability)
                values = score(y, probability)
                batches.append({
                    "configuration": configuration,
                    "label_delay_batches": delay,
                    "global_batch": global_batch,
                    "day": day,
                    "day_batch": day_batch,
                    "policy": policy,
                    "model_version": models[policy].version,
                    "events": int(len(y)),
                    "attacks": int(y.sum()),
                    "attack_prevalence": float(y.mean()),
                    "attack_families": "|".join(sorted(
                        value for value in labels.unique() if value != "benign"
                    )),
                    "inference_seconds": inference_seconds,
                    **values,
                })

            # The current labels are queued only after all policies predicted.
            pending.append((global_batch + delay, global_batch, day, X, y))
            while pending and pending[0][0] <= global_batch:
                _, source_batch, source_day, feedback_X, feedback_y = pending.popleft()
                released_events += len(feedback_y)
                if feedback_y.nunique() == 2:
                    update_started = time.perf_counter()
                    models["latest_batch"] = train_model(
                        "latest-batch", configuration,
                        models["latest_batch"].version + 1,
                        features, feedback_X, feedback_y, model_config,
                    )
                    updates.append({
                        "configuration": configuration,
                        "label_delay_batches": delay,
                        "after_global_batch": global_batch,
                        "source_global_batch": source_batch,
                        "source_day": source_day,
                        "policy": "latest_batch",
                        "status": "updated",
                        "reason": "latest released batch contains both classes",
                        "new_model_version": models["latest_batch"].version,
                        "feedback_events": int(len(feedback_y)),
                        "feedback_attacks": int(feedback_y.sum()),
                        "training_events": int(len(feedback_y)),
                        "retraining_seconds": time.perf_counter() - update_started,
                    })
                else:
                    updates.append({
                        "configuration": configuration,
                        "label_delay_batches": delay,
                        "after_global_batch": global_batch,
                        "source_global_batch": source_batch,
                        "source_day": source_day,
                        "policy": "latest_batch",
                        "status": "skipped",
                        "reason": "latest released batch has only one class",
                        "new_model_version": models["latest_batch"].version,
                        "feedback_events": int(len(feedback_y)),
                        "feedback_attacks": int(feedback_y.sum()),
                        "training_events": 0,
                        "retraining_seconds": 0.0,
                    })

                replay_X, replay_y = bounded_append(
                    replay_X, replay_y, feedback_X, feedback_y,
                    args.max_training_rows,
                )
                update_started = time.perf_counter()
                models["rolling_replay"] = train_model(
                    "rolling-replay", configuration,
                    models["rolling_replay"].version + 1,
                    features, replay_X, replay_y, model_config,
                )
                updates.append({
                    "configuration": configuration,
                    "label_delay_batches": delay,
                    "after_global_batch": global_batch,
                    "source_global_batch": source_batch,
                    "source_day": source_day,
                    "policy": "rolling_replay",
                    "status": "updated",
                    "reason": "released feedback appended to bounded replay history",
                    "new_model_version": models["rolling_replay"].version,
                    "feedback_events": int(len(feedback_y)),
                    "feedback_attacks": int(feedback_y.sum()),
                    "training_events": int(len(replay_y)),
                    "retraining_seconds": time.perf_counter() - update_started,
                })

            if global_batch % args.progress_every_batches == 0:
                print(
                    f"  c={configuration}, delay={delay}: batch {global_batch}, "
                    f"day={day}, replay_version={models['rolling_replay'].version}",
                    flush=True,
                )

        available_days = DAYS[: day_index + 1]
        aggregate_retention, family_retention = retention_evaluation(
            models, partitions, available_days, features, best_recall,
            configuration, delay, day,
        )
        retention_rows.extend(aggregate_retention)
        retention_family_rows.extend(family_retention)

    overall = {
        policy: aggregate(
            overall_parts[policy]["y"], overall_parts[policy]["probability"],
        )
        for policy in POLICIES
    }
    per_day = {
        day: {
            policy: aggregate(
                day_parts[day][policy]["y"],
                day_parts[day][policy]["probability"],
            )
            for policy in POLICIES
        }
        for day in STREAM_DAYS
    }
    final_retention = {
        row["policy"]: row
        for row in retention_rows if row["stage_day"] == STREAM_DAYS[-1]
    }
    result = {
        "configuration": configuration,
        "label_delay_batches": delay,
        "initial_training_events": int(len(initial_y)),
        "initial_training_attacks": int(initial_y.sum()),
        "overall": overall,
        "per_day": per_day,
        "final_retention": final_retention,
        "released_feedback_events": int(released_events),
        "unreleased_tail_events": int(sum(len(item[4]) for item in pending)),
        "model_versions": {policy: model.version for policy, model in models.items()},
        "runtime": {
            "initial_training_seconds": initial_training_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }
    return batches, updates, retention_rows, retention_family_rows, result


def source_manifest(adapter: LYCOSIDS2017TemporalAdapter) -> list[dict]:
    manifest = []
    for day, filename in adapter.DAY_ARCHIVES.items():
        path = adapter.root / filename
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        manifest.append({
            "day": day,
            "archive": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        })
    return manifest


def summarize(results: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for delay in sorted({result["label_delay_batches"] for result in results}):
        delay_results = [
            result for result in results if result["label_delay_batches"] == delay
        ]
        delay_summary: dict[str, dict] = {}
        for policy in POLICIES:
            delay_summary[policy] = {}
            for metric in PRIMARY_METRICS:
                values = np.asarray([
                    result["overall"][policy][metric] for result in delay_results
                ], dtype=float)
                delay_summary[policy][metric] = {
                    "mean": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "values": values.tolist(),
                }
            for metric in ("macro_attack_recall", "average_forgetting"):
                values = np.asarray([
                    result["final_retention"][policy][metric]
                    for result in delay_results
                ], dtype=float)
                delay_summary[policy][f"final_retention_{metric}"] = {
                    "mean": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "values": values.tolist(),
                }
        summary[str(delay)] = delay_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "lycos_ids2017",
    )
    parser.add_argument("--configurations", type=int, nargs="+", default=[17, 42, 73])
    parser.add_argument("--label-delays", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument(
        "--max-events-per-day", type=int, default=100_000,
        help="0 uses every eligible event; positive values use systematic temporal sampling",
    )
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--max-training-rows", type=int, default=200_000)
    parser.add_argument("--retention-fraction", type=float, default=0.20)
    parser.add_argument("--retention-max-per-label", type=int, default=1000)
    parser.add_argument("--estimators", type=int, default=100)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--progress-every-batches", type=int, default=10)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_training_rows < args.batch_size:
        parser.error("batch-size must be positive and fit inside max-training-rows")
    if any(delay < 1 for delay in args.label_delays):
        parser.error("label delays must be at least one complete batch")
    if not 0 < args.retention_fraction < 1:
        parser.error("retention-fraction must be between zero and one")
    args.output.mkdir(parents=True, exist_ok=True)

    overall_started = time.perf_counter()
    adapter = LYCOSIDS2017TemporalAdapter(
        args.data_root,
        max_events_per_day=(
            None if args.max_events_per_day == 0 else args.max_events_per_day
        ),
    )
    print("Loading and chronologically ordering LYCOS-IDS2017...", flush=True)
    load_started = time.perf_counter()
    days = adapter.load()
    load_seconds = time.perf_counter() - load_started
    partitions = {
        day: partition_retention(
            data, args.retention_fraction, args.retention_max_per_label,
        )
        for day, data in days.items()
    }
    initial_X = pd.concat(
        [partitions[day].stream_X for day in INITIAL_DAYS], ignore_index=True,
    )
    features = [
        column for column in initial_X.columns
        if initial_X[column].notna().any()
    ]
    if not features:
        raise ValueError("No usable LYCOS-IDS2017 predictive features")

    all_batches: list[dict] = []
    all_updates: list[dict] = []
    all_retention: list[dict] = []
    all_retention_family: list[dict] = []
    results: list[dict] = []
    for configuration in args.configurations:
        for delay in args.label_delays:
            print(f"Running configuration {configuration}, delay {delay}...", flush=True)
            batches, updates, retention, retention_family, result = run_protocol(
                args, partitions, features, configuration, delay,
            )
            all_batches.extend(batches)
            all_updates.extend(updates)
            all_retention.extend(retention)
            all_retention_family.extend(retention_family)
            results.append(result)
            replay = result["overall"]["rolling_replay"]
            print(
                f"  replay: F1={replay['f1']:.6f}, MCC={replay['mcc']:.6f}, "
                f"recall={replay['recall']:.6f}",
                flush=True,
            )

    across = summarize(results)
    summary = {
        "protocol": "lycos-prequential-delayed-supervision-and-retention-v1",
        "dataset": "LYCOS-IDS2017 corrected CIC-IDS2017 flows",
        "study_scope": "OX4C Continuous Learning dimension only",
        "initial_period": list(INITIAL_DAYS),
        "prequential_stream": list(STREAM_DAYS),
        "test_then_train": True,
        "strategies": {
            "frozen": "initial model; no updates",
            "latest_batch": (
                "naive retraining from the latest released batch; one-class batches "
                "are skipped because the binary learner cannot be fitted"
            ),
            "rolling_replay": (
                "retraining from released feedback plus a bounded rolling history"
            ),
        },
        "label_delay_semantics": (
            "labels from batch t are released after predictions for batch t+d and "
            "can first affect batch t+d+1"
        ),
        "configurations": args.configurations,
        "label_delays_batches": args.label_delays,
        "parameters": {
            "max_events_per_day": (
                None if args.max_events_per_day == 0 else args.max_events_per_day
            ),
            "sampling": (
                "complete chronological stream" if args.max_events_per_day == 0 else
                "deterministic systematic temporal sample spanning each complete day"
            ),
            "batch_size": args.batch_size,
            "max_training_rows": args.max_training_rows,
            "retention_fraction": args.retention_fraction,
            "retention_max_per_label_per_day": args.retention_max_per_label,
            "estimators": args.estimators,
            "decision_threshold": 0.5,
        },
        "features_used": len(features),
        "feature_names": features,
        "day_counts": {
            day: {
                "selected_events": int(len(days[day].y)),
                "stream_events": int(len(partitions[day].stream_y)),
                "retention_events": int(len(partitions[day].retention_y)),
                "stream_labels": {
                    str(label): int(count) for label, count in
                    partitions[day].stream_labels.value_counts().items()
                },
                "ordering": days[day].ordering,
            }
            for day in DAYS
        },
        "configuration_results": results,
        "across_configuration_summary": across,
        "source_manifest": source_manifest(adapter),
        "runtime": {
            "data_loading_seconds": load_seconds,
            "total_seconds": time.perf_counter() - overall_started,
        },
        "limitations": [
            "The experiment uses one natural capture-day order and does not claim order robustness.",
            "Label availability is simulated as trusted supervision after a configurable delay.",
            "Algorithmic configurations quantify learner stability, not population replications.",
            "The systematic pilot sample thins the stream when max_events_per_day is set.",
            "This evaluates binary intrusion detection; family labels are used only for retention analysis.",
        ],
    }
    pd.DataFrame(all_batches).to_csv(args.output / "batch_results.csv", index=False)
    pd.DataFrame(all_updates).to_csv(args.output / "update_results.csv", index=False)
    pd.DataFrame(all_retention).to_csv(args.output / "retention_results.csv", index=False)
    pd.DataFrame(all_retention_family).to_csv(
        args.output / "retention_by_family.csv", index=False,
    )
    pd.DataFrame([
        {
            "configuration": result["configuration"],
            "label_delay_batches": result["label_delay_batches"],
            "policy": policy,
            **result["overall"][policy],
        }
        for result in results for policy in POLICIES
    ]).to_csv(args.output / "aggregate_results.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    (args.output / "protocol.json").write_text(
        json.dumps({
            "arguments": vars(args) | {
                "data_root": str(args.data_root), "output": str(args.output),
            },
            "source_archives": adapter.DAY_ARCHIVES,
            "official_label_rules": "LYCOS-IDS2017 labelling.py, reproduced by adapter",
        }, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(across, indent=2))
    print(f"Wrote LYCOS continual-learning results to {args.output}")


if __name__ == "__main__":
    main()
