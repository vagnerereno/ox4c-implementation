"""Consolidate the confirmatory QUT-DV25 study into auditable tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROFILE_ORDER = ("baseline", "optimized", "collaborative", "complete")
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "qut_dv25")
    args = parser.parse_args()

    profile_rows: list[dict] = []
    optimization_rows: list[dict] = []
    collaboration_rows: list[dict] = []
    evidence_rows: list[dict] = []

    for profile in PROFILE_ORDER:
        for path in sorted((args.input / profile).glob("configuration_*/summary.json")):
            summary = json.loads(path.read_text(encoding="utf-8"))
            configuration = int(summary["experiment"]["configuration"])
            metrics = summary["final_metrics"]
            profile_rows.append({
                "configuration": configuration,
                "profile": profile,
                "features": summary["features"]["selected_count"],
                "f1": metrics["f1"],
                "mcc": metrics["mcc"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "false_positive_rate": metrics["false_positive_rate"],
                "false_negatives": metrics["fn"],
                "false_positives": metrics["fp"],
            })

            if profile == "optimized":
                deployment = summary["features"]["metadata"]["deployment"]
                optimization_rows.append({
                    "configuration": configuration,
                    "available_features": summary["features"]["available_count"],
                    "candidate_features": deployment["candidate_count"],
                    "deployed_features": deployment["deployed_count"],
                    "accepted": deployment["accepted"],
                    "search_evaluations": summary["features"]["metadata"]["evaluations"],
                    "search_runtime_seconds": summary["features"]["runtime_seconds"],
                    "search_stop_reason": summary["features"]["metadata"]["stop_reason"],
                    "candidate_calibration_mcc": deployment["candidate_mcc"],
                    "full_calibration_mcc": deployment["full_feature_mcc"],
                    "candidate_test_f1": deployment["candidate_test_metrics"]["f1"],
                    "full_test_f1": deployment["full_feature_test_metrics"]["f1"],
                    "candidate_test_mcc": deployment["candidate_test_metrics"]["mcc"],
                    "full_test_mcc": deployment["full_feature_test_metrics"]["mcc"],
                    "candidate_p95_ms": deployment["candidate_latency"]["p95_ms"],
                    "full_p95_ms": deployment["full_feature_latency"]["p95_ms"],
                    "candidate_size_bytes": deployment["candidate_size_bytes"],
                    "full_size_bytes": deployment["full_feature_size_bytes"],
                })

            if profile == "collaborative":
                collab = summary["collaboration"]
                activation = collab["competence"]["activation"]
                collaboration_rows.append({
                    "configuration": configuration,
                    "requests_conflicts": collab["conflicts"],
                    "accepted_consensuses": collab["authorized_interventions"],
                    "changed_decisions": collab["changed_decisions"],
                    "corrections": collab["corrections"],
                    "harms": collab["harms"],
                    "net_corrections": collab["net_corrections"],
                    "primary_false_negatives": collab["primary_false_negatives"],
                    "false_negatives_with_conflict": collab["primary_false_negatives_with_conflict"],
                    "final_false_negatives": metrics["fn"],
                    "calibration_conflicts": activation["observed_conflicts"],
                    "calibration_mcc_gain": activation["mcc_gain"],
                    "selected_minimum_attack_votes": activation["selected_minimum_attack_votes"],
                    "resolver_authorized": activation["authorized"],
                })

            if profile == "complete":
                performance = summary["performance"]
                compact = summary["compact_model"]
                evidence_rows.append({
                    "configuration": configuration,
                    "explanations": performance["counters"]["explanations_generated"],
                    "decision_p95_ms": performance["timings"]["decision_availability"]["p95_ms"],
                    "shap_p95_ms": performance["timings"]["shap"]["p95_ms"],
                    "pipeline_p95_ms": performance["timings"]["pipeline_completion"]["p95_ms"],
                    "throughput_events_per_second": performance["throughput_events_per_second"],
                    "technical_conformance_allowed": summary["technical_conformance"]["allowed"],
                    "technical_controls": len(summary["technical_conformance"]["controls"]),
                    "compact_f1": compact["test_metrics"]["f1"],
                    "primary_f1": compact["primary_test_metrics"]["f1"],
                    "compact_size_bytes": compact["compact_size_bytes"],
                    "primary_size_bytes": compact["primary_size_bytes"],
                    "compact_p95_ms": compact["compact_latency"]["p95_ms"],
                    "primary_p95_ms": compact["primary_latency"]["p95_ms"],
                })

    profiles = pd.DataFrame(profile_rows)
    optimization = pd.DataFrame(optimization_rows)
    collaboration = pd.DataFrame(collaboration_rows)
    evidence = pd.DataFrame(evidence_rows)
    if profiles.empty:
        raise SystemExit(f"No confirmatory summaries found under {args.input}")

    profiles.to_csv(args.input / "profile_results.csv", index=False)
    optimization.to_csv(args.input / "optimization_results.csv", index=False)
    collaboration.to_csv(args.input / "collaboration_results.csv", index=False)
    evidence.to_csv(args.input / "evidence_results.csv", index=False)

    profile_aggregate = (
        profiles.groupby("profile", sort=False)
        .agg(
            configurations=("configuration", "count"),
            f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
            mcc_mean=("mcc", "mean"), mcc_sd=("mcc", "std"),
            features_mean=("features", "mean"),
            fn_total=("false_negatives", "sum"), fp_total=("false_positives", "sum"),
        )
        .reindex(PROFILE_ORDER)
        .reset_index()
    )
    profile_aggregate.to_csv(args.input / "profile_aggregate.csv", index=False)

    summary = {
        "protocol": "qut-ox4c-nontemporal-confirmatory-v1",
        "configurations": sorted(profiles["configuration"].unique().tolist()),
        "profile_aggregate": profile_aggregate.to_dict(orient="records"),
        "optimization": {
            "accepted_configurations": int(optimization["accepted"].sum()),
            "total_configurations": len(optimization),
            "deployed_features": optimization.set_index("configuration")["deployed_features"].to_dict(),
            "offline_search_runtime_seconds_total": float(optimization["search_runtime_seconds"].sum()),
        },
        "collaboration": {
            "requests_conflicts": int(collaboration["requests_conflicts"].sum()),
            "accepted_consensuses": int(collaboration["accepted_consensuses"].sum()),
            "changed_decisions": int(collaboration["changed_decisions"].sum()),
            "corrections": int(collaboration["corrections"].sum()),
            "harms": int(collaboration["harms"].sum()),
            "net_corrections": int(collaboration["net_corrections"].sum()),
        },
        "evidence": {
            "explanations_total": int(evidence["explanations"].sum()),
            "all_technical_conformance_runs_allowed": bool(
                evidence["technical_conformance_allowed"].all()
            ),
            "decision_p95_ms_mean": float(evidence["decision_p95_ms"].mean()),
            "shap_p95_ms_mean": float(evidence["shap_p95_ms"].mean()),
            "compact_size_reduction_mean": float(
                (1 - evidence["compact_size_bytes"] / evidence["primary_size_bytes"]).mean()
            ),
            "compact_p95_speedup_mean": float(
                (evidence["primary_p95_ms"] / evidence["compact_p95_ms"]).mean()
            ),
            "compact_f1_delta_mean": float(
                (evidence["compact_f1"] - evidence["primary_f1"]).mean()
            ),
        },
    }
    (args.input / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
