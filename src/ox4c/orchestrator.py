from __future__ import annotations

import json
import time
from collections import deque
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ox4c.config import OX4CConfig
from ox4c.services import (
    AuditService, CompliancePolicyGate, ConflictDrivenCounselors, DetectorModel,
    FeatureOptimizer, LabeledObservation, ModelRegistry, PerformanceMonitor,
    ShapExplainer, ContinuousLearningService, metrics,
)
from ox4c.types import DatasetBundle, Decision


class OX4COrchestrator:
    """OX4C pipeline centered on explicit classifier conflict."""

    def __init__(self, config: OX4CConfig) -> None:
        self.config = config
        self.output = Path(config.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.audit = AuditService(self.output / "audit")
        self.performance = PerformanceMonitor()
        self.registry = ModelRegistry(self.output / "models")
        self.explainer = ShapExplainer(config.explainability.top_k)
        self.bundle: DatasetBundle | None = None
        self.selection = None
        self.primary: DetectorModel | None = None
        self.frozen_primary: DetectorModel | None = None
        self.counselors: ConflictDrivenCounselors | None = None
        self.learning: ContinuousLearningService | None = None
        self.compact_model: DetectorModel | None = None
        self.compliance_decision = None

    def fit(self, bundle: DatasetBundle, selection_result=None):
        self.bundle = bundle
        if selection_result is None:
            with self.performance.measure("feature_selection"):
                self.selection = FeatureOptimizer(
                    self.config.optimization, self.config.model,
                ).select(bundle.train.X, bundle.train.y)
        else:
            self.selection = deepcopy(selection_result)
            self.performance.observe("feature_selection", self.selection.elapsed_seconds * 1000)
            self.performance.increment("feature_selection_cache_reused")
        primary_name = self.config.collaboration.primary_algorithm
        candidate_features = list(self.selection.selected_features)
        selection_already_evaluated = "deployment" in self.selection.metadata
        if self.selection.algorithm != "none" and not selection_already_evaluated:
            with self.performance.measure("candidate_feature_model_training"):
                candidate_model = DetectorModel.train(
                    primary_name, 1, candidate_features, bundle.train.X, bundle.train.y,
                    self.config.model, algorithm=primary_name,
                )
            with self.performance.measure("full_feature_model_training"):
                full_model = DetectorModel.train(
                    primary_name, 1, list(bundle.feature_names), bundle.train.X,
                    bundle.train.y, self.config.model, algorithm=primary_name,
                )
            candidate_mcc = metrics(
                candidate_model, bundle.calibration.X, bundle.calibration.y,
            )["mcc"]
            full_mcc = metrics(full_model, bundle.calibration.X, bundle.calibration.y)["mcc"]
            mcc_loss = float(full_mcc - candidate_mcc)
            reduction_fraction = float(
                1.0 - len(candidate_features) / max(1, len(bundle.feature_names))
            )
            predictive_improvement = candidate_mcc >= full_mcc
            efficient_tradeoff = (
                len(candidate_features) < len(bundle.feature_names)
                and reduction_fraction >= self.config.optimization.min_feature_reduction_fraction
                and mcc_loss <= self.config.optimization.max_validation_mcc_loss
            )
            accepted = bool(predictive_improvement or efficient_tradeoff)
            self.primary = candidate_model if accepted else full_model
            features = candidate_features if accepted else list(bundle.feature_names)
            candidate_test = metrics(candidate_model, bundle.test.X, bundle.test.y)
            full_test = metrics(full_model, bundle.test.X, bundle.test.y)
            self.selection.metadata["deployment"] = {
                "rule": "deploy an improving subset, or a materially smaller subset within the predeclared calibration MCC-loss tolerance",
                "selection_partition": "calibration",
                "test_metrics_role": "reporting only; never used for deployment",
                "candidate_features": candidate_features,
                "candidate_count": len(candidate_features),
                "candidate_mcc": candidate_mcc,
                "full_feature_mcc": full_mcc,
                "calibration_mcc_loss": mcc_loss,
                "max_allowed_mcc_loss": self.config.optimization.max_validation_mcc_loss,
                "feature_reduction_fraction": reduction_fraction,
                "minimum_feature_reduction_fraction": self.config.optimization.min_feature_reduction_fraction,
                "predictive_improvement": bool(predictive_improvement),
                "efficient_tradeoff": bool(efficient_tradeoff),
                "accepted": accepted,
                "deployed_features": features,
                "deployed_count": len(features),
                "candidate_test_metrics": candidate_test,
                "full_feature_test_metrics": full_test,
                "candidate_latency": self._latency(candidate_model, bundle.test.X, 256),
                "full_feature_latency": self._latency(full_model, bundle.test.X, 256),
                "candidate_size_bytes": self._serialized_size(candidate_model),
                "full_feature_size_bytes": self._serialized_size(full_model),
            }
            self.selection.selected_features = features
        else:
            features = list(self.selection.selected_features)
            with self.performance.measure("primary_training"):
                self.primary = DetectorModel.train(
                    primary_name, 1, features, bundle.train.X, bundle.train.y,
                    self.config.model, algorithm=primary_name,
                )
        models = {primary_name: self.primary}
        if self.config.collaboration.enabled:
            for algorithm in self.config.collaboration.counselor_algorithms:
                with self.performance.measure(f"counselor_training_{algorithm}"):
                    models[algorithm] = DetectorModel.train(
                        algorithm, 1, features, bundle.train.X, bundle.train.y,
                        self.config.model, algorithm=algorithm,
                    )
        self.counselors = ConflictDrivenCounselors(
            models, primary_name, bundle.calibration.X, bundle.calibration.y,
            self.config.collaboration,
        )
        validation_metrics = metrics(self.primary, bundle.validation.X, bundle.validation.y)
        self.registry.promote(self.primary, validation_metrics, {
            "domain": bundle.domain,
            "feature_selection": self.selection.to_dict(),
            "architecture": "ox4c-conflict-driven-v1",
        })
        self.frozen_primary = deepcopy(self.primary)
        self.learning = ContinuousLearningService(
            bundle.train, bundle.validation, self.config.learning,
            self.config.model, self.registry,
        )
        if self.config.compact_model.enabled:
            compact_algorithm = self.config.compact_model.algorithm
            self.compact_model = models.get(compact_algorithm)
            if self.compact_model is None:
                with self.performance.measure("compact_model_training"):
                    self.compact_model = DetectorModel.train(
                        "compact", 1, features, bundle.train.X, bundle.train.y,
                        self.config.model, algorithm=compact_algorithm,
                    )
            joblib.dump(self.compact_model, self.output / "models" / "compact-candidate.joblib")
        if self.config.compliance.enabled:
            self.compliance_decision = CompliancePolicyGate(self.config.compliance).evaluate(
                bundle, self.primary, audit_available=self.audit.path.exists(),
                explanation_available=self.config.explainability.enabled,
            )
            payload = self.compliance_decision.to_dict()
            (self.output / "technical_conformance.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8",
            )
            self.audit.record("technical_conformance_evaluated", payload)
            if self.config.compliance.enforce and not self.compliance_decision.allowed:
                raise RuntimeError(
                    "Technical conformance rejected the run: "
                    + ", ".join(self.compliance_decision.violations)
                )
        (self.output / "run_config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8",
        )
        (self.output / "dataset_provenance.json").write_text(
            json.dumps(bundle.provenance, indent=2, default=str), encoding="utf-8",
        )
        (self.output / "feature_selection.json").write_text(
            json.dumps(self.selection.to_dict(), indent=2), encoding="utf-8",
        )
        if self.selection.search_history:
            pd.DataFrame(self.selection.search_history).to_json(
                self.output / "feature_selection_history.jsonl",
                orient="records", lines=True,
            )
        (self.output / "classifier_competence.json").write_text(
            json.dumps(self.counselors.metadata(), indent=2), encoding="utf-8",
        )
        self.audit.record("training_completed", {
            "domain": bundle.domain,
            "features": features,
            "validation": validation_metrics,
            "collaboration": self.counselors.metadata(),
        })
        self.audit.flush()
        return self

    def predict_one(self, row: pd.DataFrame, sample_id: str) -> Decision:
        if self.primary is None or self.counselors is None or self.learning is None:
            raise RuntimeError("fit must be called first")
        started = time.perf_counter()
        if self.config.collaboration.enabled:
            with self.performance.measure("classifier_pool_inference"):
                resolution = self.counselors.resolve(row)
        else:
            with self.performance.measure("primary_inference"):
                resolution = self.counselors.resolve(row)
        local_probability = resolution.primary_probability
        final_probability = resolution.final_probability
        conflict = bool(resolution.conflict and self.config.collaboration.enabled)
        collaborative = bool(resolution.authorized and self.config.collaboration.enabled)
        if conflict:
            self.performance.increment("classifier_conflicts")
            if collaborative and resolution.changed_primary_decision:
                self.performance.increment("conflict_decision_changes")
        selected_model = self.counselors.models[resolution.selected_classifier]
        if self.primary.version == 1:
            frozen_probability = local_probability
        else:
            with self.performance.measure("frozen_inference"):
                frozen_probability = float(self.frozen_primary.predict_proba(row)[0])
        prediction = int(final_probability >= 0.5)
        decision_available_ms = (time.perf_counter() - started) * 1000
        self.performance.observe("decision_availability", decision_available_ms)
        should_explain = self.config.explainability.enabled and (
            self.config.explainability.policy == "all"
            or (
                self.config.explainability.policy == "alerts_and_collaborative"
                and (prediction == 1 or collaborative)
            )
        )
        explanation = {}
        if should_explain:
            with self.performance.measure("shap"):
                explanation = self.explainer.explain(selected_model, row)
            self.performance.increment("explanations_generated" if explanation else "xai_failures")
        rationale = resolution.to_dict() if conflict else {
            "conflict": False,
            "selected_classifier": resolution.selected_classifier,
            "predictions": resolution.predictions,
        }
        decision = Decision(
            sample_id=str(sample_id),
            frozen_probability=frozen_probability,
            local_probability=local_probability,
            final_probability=final_probability,
            prediction=prediction,
            confidence=max(final_probability, 1 - final_probability),
            model_version=selected_model.version,
            counselors_requested=conflict,
            collaborative=collaborative,
            conflict=conflict,
            decision_source=resolution.selected_classifier,
            advice=resolution.advice if conflict else [],
            explanation=explanation,
            collaboration_explanation=rationale,
            latency_ms=decision_available_ms,
        )
        with self.performance.measure("audit_completion"):
            decision.audit_id = self.audit.record("decision", {
                **decision.to_dict(), "true_label_available_after_decision": False,
            })
        self.performance.observe("pipeline_completion", (time.perf_counter() - started) * 1000)
        return decision

    def observe_feedback(self, row: pd.DataFrame, decision: Decision, true_label: int) -> None:
        if self.primary is None or self.learning is None or self.counselors is None:
            raise RuntimeError("fit must be called first")
        self.performance.increment("feedback_released")
        if not self.config.learning.enabled:
            return
        self.learning.admit(LabeledObservation(row.copy(), int(true_label), decision.sample_id))
        self.performance.increment("knowledge_admitted")
        if not self.learning.ready():
            return
        with self.performance.measure("candidate_retraining"):
            updated, result = self.learning.update_if_ready(self.primary)
        if result is None:
            return
        promoted = updated.version > self.primary.version
        self.performance.increment("updates_promoted" if promoted else "updates_rejected")
        self.audit.record("model_promoted" if promoted else "model_rejected", result)
        if promoted:
            self.primary = updated
            self.counselors.replace_primary(
                updated, self.bundle.calibration.X, self.bundle.calibration.y,
            )

    @staticmethod
    def _latency(model: DetectorModel, X: pd.DataFrame, count: int) -> dict[str, float]:
        count = min(max(1, count), len(X))
        values = []
        for position in range(count):
            started = time.perf_counter()
            model.predict_proba(X.iloc[[position]])
            values.append((time.perf_counter() - started) * 1000)
        return {
            "samples": count,
            "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
        }

    @staticmethod
    def _serialized_size(model: DetectorModel) -> int:
        buffer = BytesIO()
        joblib.dump(model, buffer)
        return int(buffer.tell())

    def _write_compact_report(self) -> None:
        if self.compact_model is None or self.bundle is None or self.frozen_primary is None:
            return
        compact_path = self.output / "models" / "compact-candidate.joblib"
        primary_path = self.output / "models" / "primary-v1.joblib"
        sample_count = self.config.compact_model.benchmark_samples
        payload = {
            "role": "lightweight candidate; not a TinyML hardware claim",
            "algorithm": self.compact_model.algorithm,
            "feature_count": len(self.compact_model.features),
            "validation_metrics": metrics(
                self.compact_model, self.bundle.validation.X, self.bundle.validation.y,
            ),
            "test_metrics": metrics(self.compact_model, self.bundle.test.X, self.bundle.test.y),
            "reference": "frozen pre-stream primary model (version 1)",
            "primary_test_metrics": metrics(
                self.frozen_primary, self.bundle.test.X, self.bundle.test.y,
            ),
            "compact_size_bytes": compact_path.stat().st_size,
            "primary_size_bytes": primary_path.stat().st_size if primary_path.exists() else None,
            "compact_latency": self._latency(
                self.compact_model, self.bundle.calibration.X, sample_count,
            ),
            "primary_latency": self._latency(
                self.frozen_primary, self.bundle.calibration.X, sample_count,
            ),
        }
        (self.output / "compact_model.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )

    def run(self, limit: int | None = None, progress_every: int = 1000) -> pd.DataFrame:
        if self.bundle is None:
            raise RuntimeError("fit must be called first")
        count = len(self.bundle.test.y) if limit is None else min(limit, len(self.bundle.test.y))
        rows = []
        pending = deque()
        delay = max(0, int(self.config.learning.feedback_delay_events))
        started = time.perf_counter()
        for index in range(count):
            while pending and pending[0][0] <= index:
                _, feedback_row, feedback_decision, feedback_label = pending.popleft()
                self.observe_feedback(feedback_row, feedback_decision, feedback_label)
            row = self.bundle.test.X.iloc[[index]]
            decision = self.predict_one(row, str(self.bundle.test.ids.iloc[index]))
            label = int(self.bundle.test.y.iloc[index])
            if self.config.learning.enabled:
                if delay == 0:
                    self.observe_feedback(row.copy(), decision, label)
                else:
                    pending.append((index + delay, row.copy(), decision, label))
            record = decision.to_dict()
            record["true_label"] = label
            rows.append(record)
            if progress_every and ((index + 1) % progress_every == 0 or index + 1 == count):
                print(
                    f"[OX4C] {index + 1}/{count} events in "
                    f"{time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        frame = pd.DataFrame(rows)
        self.performance.increment("feedback_pending_at_stream_end", len(pending))
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.performance.observe("stream_wall", elapsed_ms)
        self.performance.increment("stream_events", count)
        self.audit.flush()
        frame.to_csv(self.output / "event_log.csv", index=False)
        (self.output / "performance.json").write_text(
            json.dumps(self.performance.snapshot(), indent=2), encoding="utf-8",
        )
        self._write_compact_report()
        return frame
