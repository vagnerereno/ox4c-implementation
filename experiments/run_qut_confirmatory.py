"""QUT-DV25 experiment for the non-temporal OX4C dimensions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from ox4c import OX4CConfig, OX4COrchestrator  # noqa: E402
from ox4c.adapters import QUTDV25Adapter  # noqa: E402
from ox4c.services.optimization import SelectionResult  # noqa: E402
from common import summarize_run  # noqa: E402


PROFILES = {
    "baseline": dict(optimization=False, collaboration=False, evidence=False),
    "optimized": dict(optimization=True, collaboration=False, evidence=False),
    "collaboration_only": dict(optimization=False, collaboration=True, evidence=False),
    "collaborative": dict(optimization=True, collaboration=True, evidence=False),
    "complete": dict(optimization=True, collaboration=True, evidence=True),
}
COUNSELORS = ("hist_gradient_boosting", "compact_tree")


def default_qut() -> Path:
    candidates = list((ROOT / "data" / "qut_dv25").glob("*.csv"))
    if candidates:
        return max(candidates, key=lambda item: item.stat().st_size)
    return ROOT / "data" / "qut_dv25" / "QUT-DV25.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selection_path(root: Path, configuration: int) -> Path:
    return root / f"configuration_{configuration}.json"


def configure(output: Path, configuration: int, profile: str) -> OX4CConfig:
    switches = PROFILES[profile]
    config = OX4CConfig()
    config.output_dir = str(output)
    config.model.random_state = configuration
    config.optimization.algorithm = "agraspqfs" if switches["optimization"] else "none"
    config.optimization.max_validation_mcc_loss = 0.02
    config.optimization.min_feature_reduction_fraction = 0.25
    config.collaboration.enabled = switches["collaboration"]
    config.collaboration.counselor_algorithms = list(COUNSELORS)
    config.collaboration.consensus = "calibration_selected_vote"
    config.learning.enabled = False
    config.explainability.enabled = switches["evidence"]
    config.explainability.policy = "alerts_and_collaborative"
    config.compliance.enabled = switches["evidence"]
    config.compliance.enforce = True
    config.compact_model.enabled = switches["evidence"]
    config.compact_model.algorithm = "compact_tree"
    config.compact_model.benchmark_samples = 512
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qut-data", type=Path, default=default_qut())
    parser.add_argument(
        "--selection-cache-root", type=Path,
        default=ROOT / "config" / "feature_selection",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "qut_dv25",
    )
    parser.add_argument("--configurations", nargs="+", type=int, default=[17, 42, 73])
    parser.add_argument(
        "--profiles", nargs="+", choices=tuple(PROFILES), default=list(PROFILES),
    )
    parser.add_argument("--event-limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cache_manifest = []
    cached_selections: dict[int, SelectionResult] = {}
    for configuration in args.configurations:
        path = selection_path(args.selection_cache_root, configuration)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing frozen robust A-GRASPQ-FS result for configuration {configuration}: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = SelectionResult(**payload)
        resolved = result.metadata.get("resolved_search", {})
        evaluations = int(result.metadata.get("evaluations", 0))
        elapsed_seconds = float(result.elapsed_seconds)
        reached_evaluation_budget = evaluations >= int(resolved.get("max_evaluations", 250))
        reached_time_budget = (
            result.metadata.get("stop_reason") == "budget_exhausted"
            and elapsed_seconds >= 0.99 * float(resolved.get("time_budget", 900.0))
        )
        if result.algorithm != "agraspqfs" or not (
            reached_evaluation_budget or reached_time_budget
        ):
            raise ValueError(f"Selection cache is not the expected robust search artifact: {path}")
        cached_selections[configuration] = result
        cache_manifest.append({
            "configuration": configuration,
            "path": str(path),
            "sha256": sha256(path),
            "evaluations": result.metadata.get("evaluations"),
            "elapsed_seconds": result.elapsed_seconds,
            "stop_reason": result.metadata.get("stop_reason"),
            "constructive_iterations": resolved.get("constructive_iterations"),
            "local_iterations": resolved.get("local_iterations"),
            "priority_queue_size": resolved.get("priority_queue_size"),
            "candidate_count": result.metadata.get("deployment", {}).get("candidate_count"),
            "deployed_count": len(result.selected_features),
            "deployment_accepted": result.metadata.get("deployment", {}).get("accepted"),
        })

    manifest = {
        "protocol": "qut-ox4c-nontemporal-confirmatory-v2",
        "created_at": time.time(),
        "dataset": "QUT-DV25 software supply-chain attack detection",
        "data_path": str(args.qut_data),
        "data_sha256": sha256(args.qut_data),
        "configurations": args.configurations,
        "configuration_role": "grouped configuration (group split plus algorithmic randomness)",
        "profiles": args.profiles,
        "profile_definition": PROFILES,
        "continuous_learning": (
            "disabled on QUT-DV25; evaluated separately on the chronological LYCOS-IDS2017 stream"
        ),
        "optimization": {
            "method": "A-GRASPQ-FS",
            "frozen_search_artifacts": cache_manifest,
            "selection_partition": "calibration",
            "test_partition_used_for_selection": False,
        },
        "collaboration": {
            "primary": "random_forest",
            "counselors": list(COUNSELORS),
            "trigger": "predicted-label disagreement",
            "resolver": "minimum attack votes selected by calibration MCC",
            "authorization": "resolver enabled only for positive calibration MCC gain",
        },
        "evidence": {
            "explanation": "post-decision TreeSHAP for alerts and collaborative decisions",
            "compliance": "machine-readable technical artifact conformance",
            "performance": "measured compact candidate, latency, serialized size and runtime resources",
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )

    for configuration in args.configurations:
        bundle = QUTDV25Adapter(args.qut_data, random_state=configuration).load()
        for profile in args.profiles:
            output = args.output / profile / f"configuration_{configuration}"
            if (output / "summary.json").exists() and not args.force:
                print(f"[QUT] skip {profile}/configuration_{configuration}", flush=True)
                continue
            config = configure(output, configuration, profile)
            cached = (
                deepcopy(cached_selections[configuration])
                if PROFILES[profile]["optimization"] else None
            )
            print(f"[QUT] start {profile}/configuration_{configuration}", flush=True)
            started = time.perf_counter()
            system = OX4COrchestrator(config).fit(bundle, cached)
            events = system.run(args.event_limit, args.progress_every)
            result = summarize_run(events, system, bundle.domain, output, {
                "dataset": "qut",
                "profile": profile,
                "configuration": configuration,
                "configuration_role": "grouped_configuration",
                "data_path": str(args.qut_data),
                "event_limit": args.event_limit,
                "selection_cache": (
                    str(selection_path(args.selection_cache_root, configuration))
                    if cached is not None else None
                ),
                "continuous_learning_enabled": False,
                "runtime_seconds": time.perf_counter() - started,
            })
            print(
                f"[QUT] complete {profile}/configuration_{configuration}: "
                f"F1={result['final_metrics']['f1']:.6f} "
                f"MCC={result['final_metrics']['mcc']:.6f} "
                f"features={result['features']['selected_count']} "
                f"conflicts={result['collaboration']['conflicts']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
