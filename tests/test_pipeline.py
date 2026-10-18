from __future__ import annotations

from pathlib import Path

from ox4c import OX4CConfig, OX4COrchestrator
from ox4c.adapters import QUTDV25Adapter
from ox4c.services import CompliancePolicyGate, DetectorModel


def _config(tmp_path: Path) -> OX4CConfig:
    config = OX4CConfig()
    config.output_dir = str(tmp_path / "artifacts")
    config.optimization.algorithm = "none"
    config.model.n_estimators = 20
    config.model.hgb_max_iter = 20
    config.model.compact_max_depth = 4
    config.collaboration.counselor_algorithms = [
        "hist_gradient_boosting", "compact_tree",
    ]
    config.learning.enabled = False
    config.explainability.enabled = True
    config.explainability.policy = "all"
    config.compliance.enabled = True
    config.compact_model.benchmark_samples = 5
    return config


def test_qut_pipeline_produces_decisions_and_evidence(qut_csv: Path, tmp_path: Path) -> None:
    bundle = QUTDV25Adapter(qut_csv, random_state=17).load()
    system = OX4COrchestrator(_config(tmp_path)).fit(bundle)
    events = system.run(limit=25, progress_every=0)

    assert len(events) == 25
    assert {
        "conflict", "decision_source", "local_probability", "final_probability",
        "explanation", "true_label",
    } <= set(events)
    assert events["explanation"].map(bool).all()
    assert system.compliance_decision and system.compliance_decision.allowed
    assert (tmp_path / "artifacts" / "classifier_competence.json").exists()
    assert (tmp_path / "artifacts" / "compact_model.json").exists()
    assert (tmp_path / "artifacts" / "technical_conformance.json").exists()


def test_technical_policy_accepts_and_rejects_evidence(qut_csv: Path, tmp_path: Path) -> None:
    bundle = QUTDV25Adapter(qut_csv, random_state=17).load()
    config = _config(tmp_path)
    model = DetectorModel.train(
        "primary", 1, list(bundle.feature_names), bundle.train.X, bundle.train.y,
        config.model,
    )

    allowed = CompliancePolicyGate(config.compliance).evaluate(
        bundle, model, audit_available=True, explanation_available=True,
    )
    assert allowed.allowed

    config.compliance.feature_denylist = [model.features[0]]
    rejected = CompliancePolicyGate(config.compliance).evaluate(
        bundle, model, audit_available=True, explanation_available=True,
    )
    assert not rejected.allowed
    assert "feature_deny_rules" in rejected.violations
