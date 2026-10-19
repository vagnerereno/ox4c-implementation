"""Integrity and protocol audit for the confirmatory QUT-DV25 outputs."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import pandas as pd


PROFILES = (
    "baseline",
    "optimized",
    "collaboration_only",
    "collaborative",
    "complete",
)
ROOT = Path(__file__).resolve().parents[1]


def binary_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    y = frame["true_label"].astype(int)
    p = frame["prediction"].astype(int)
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tp = int(((y == 1) & (p == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "f1": f1, "mcc": mcc}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "qut_dv25")
    args = parser.parse_args()
    manifest = json.loads((args.input / "manifest.json").read_text(encoding="utf-8"))
    configurations = [int(value) for value in manifest["configurations"]]
    checks: dict[str, dict] = {}

    def record(name: str, passed: bool, evidence) -> None:
        checks[name] = {"passed": bool(passed), "evidence": evidence}

    summaries: dict[tuple[str, int], dict] = {}
    logs: dict[tuple[str, int], pd.DataFrame] = {}
    missing: list[str] = []
    for profile in PROFILES:
        for configuration in configurations:
            root = args.input / profile / f"configuration_{configuration}"
            if not (root / "summary.json").exists() or not (root / "event_log.csv").exists():
                missing.append(str(root))
                continue
            summaries[(profile, configuration)] = json.loads(
                (root / "summary.json").read_text(encoding="utf-8")
            )
            logs[(profile, configuration)] = pd.read_csv(root / "event_log.csv")
    expected_runs = len(PROFILES) * len(configurations)
    record(
        "all_expected_runs_present",
        len(summaries) == expected_runs and not missing,
        {"expected": expected_runs, "observed": len(summaries), "missing": missing},
    )

    metric_error = 0.0
    row_counts: dict[str, int] = {}
    for key, summary in summaries.items():
        frame = logs[key]
        row_counts[f"{key[0]}:{key[1]}"] = len(frame)
        recomputed = binary_metrics(frame)
        reported = summary["final_metrics"]
        metric_error = max(
            metric_error,
            *(abs(float(recomputed[name]) - float(reported[name])) for name in recomputed),
        )
    record(
        "final_metrics_reproduce_from_event_logs",
        metric_error < 1e-12,
        {"maximum_absolute_difference": metric_error},
    )
    record(
        "equal_test_event_counts",
        len(set(row_counts.values())) == 1 and next(iter(row_counts.values()), 0) == 2140,
        row_counts,
    )
    record(
        "continuous_learning_disabled_on_qut",
        all(
            not summary["learning"]["enabled"]
            and summary["learning"]["attempts"] == 0
            and summary["learning"]["promotions"] == 0
            for summary in summaries.values()
        ),
        "QUT isolates non-temporal dimensions; LYCOS evaluates Continuous Learning",
    )

    feature_evidence = []
    for configuration in configurations:
        optimized = summaries[("optimized", configuration)]
        deployment = optimized["features"]["metadata"]["deployment"]
        feature_evidence.append({
            "configuration": configuration,
            "candidate": deployment["candidate_count"],
            "deployed": deployment["deployed_count"],
            "accepted": deployment["accepted"],
        })
    record(
        "feature_candidate_and_deployed_subsets_distinguished",
        all(
            row["deployed"] == (
                row["candidate"] if row["accepted"] else 36
            )
            for row in feature_evidence
        ),
        feature_evidence,
    )

    accounting = []
    collaboration_valid = True
    resolver_valid = True
    for configuration in configurations:
        summary = summaries[("collaborative", configuration)]
        c = summary["collaboration"]
        row = {
            "configuration": configuration,
            "requests": c["conflicts"],
            "accepted": c["authorized_interventions"],
            "changed": c["changed_decisions"],
            "corrections": c["corrections"],
            "harms": c["harms"],
            "net": c["net_corrections"],
        }
        accounting.append(row)
        collaboration_valid &= (
            row["accepted"] <= row["requests"]
            and row["changed"] <= row["accepted"]
            and row["changed"] == row["corrections"] + row["harms"]
            and row["net"] == row["corrections"] - row["harms"]
        )
        activation = c["competence"]["activation"]
        resolver_valid &= bool(activation["authorized"] and activation["mcc_gain"] > 0)
    record("collaboration_accounting_is_consistent", collaboration_valid, accounting)
    record(
        "collaboration_authorized_only_after_calibration_gain",
        resolver_valid,
        "all three resolvers have positive calibration MCC gain",
    )

    predictive_equivalence = []
    for configuration in configurations:
        collaborative = summaries[("collaborative", configuration)]["final_metrics"]
        complete = summaries[("complete", configuration)]["final_metrics"]
        equal = all(
            abs(float(collaborative[name]) - float(complete[name])) < 1e-12
            for name in ("tn", "fp", "fn", "tp", "f1", "mcc")
        )
        predictive_equivalence.append({"configuration": configuration, "equal": equal})
    record(
        "post_decision_evidence_does_not_change_predictions",
        all(row["equal"] for row in predictive_equivalence),
        predictive_equivalence,
    )

    explanation_evidence = []
    explanations_valid = True
    conformance_valid = True
    for configuration in configurations:
        summary = summaries[("complete", configuration)]
        frame = logs[("complete", configuration)]
        explained = frame.loc[frame["explanation"].notna() & frame["explanation"].ne("{}")]
        sampled = explained.head(25)
        max_additive_error = 0.0
        lengths: list[int] = []
        for raw in sampled["explanation"]:
            payload = ast.literal_eval(raw)
            lengths.append(len(payload.get("full_contributions", {})))
            max_additive_error = max(
                max_additive_error,
                abs(
                    float(payload["probability_reconstruction"])
                    - float(payload["model_probability"])
                ),
            )
        selected = summary["features"]["selected_count"]
        valid = (
            len(explained) == summary["performance"]["counters"]["explanations_generated"]
            and bool(lengths)
            and all(length == selected for length in lengths)
            and max_additive_error < 1e-9
        )
        explanations_valid &= valid
        conformance = summary["technical_conformance"]
        conformance_valid &= bool(
            conformance["allowed"]
            and conformance["controls"]
            and all(control["passed"] for control in conformance["controls"])
        )
        explanation_evidence.append({
            "configuration": configuration,
            "explanations": len(explained),
            "selected_features": selected,
            "sampled_full_vector_lengths": sorted(set(lengths)),
            "maximum_sampled_additive_error": max_additive_error,
        })
    record("full_vector_treeshap_evidence_is_additive", explanations_valid, explanation_evidence)
    record(
        "technical_conformance_controls_pass",
        conformance_valid,
        "machine-readable technical-policy checks",
    )

    passed = all(item["passed"] for item in checks.values())
    report = {
        "status": "passed" if passed else "failed",
        "protocol": manifest["protocol"],
        "checks": checks,
    }
    (args.input / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
