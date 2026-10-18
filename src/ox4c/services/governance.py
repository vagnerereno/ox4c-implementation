from __future__ import annotations

from dataclasses import asdict, dataclass

from ox4c.config import ComplianceConfig
from ox4c.services.model import DetectorModel
from ox4c.types import DatasetBundle


@dataclass
class ComplianceDecision:
    policy_id: str
    allowed: bool
    controls: list[dict]
    violations: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class CompliancePolicyGate:
    """Evaluate technical policy conformance from machine-readable evidence."""

    def __init__(self, config: ComplianceConfig) -> None:
        self.config = config

    def evaluate(
        self, bundle: DatasetBundle, model: DetectorModel, *,
        audit_available: bool, explanation_available: bool,
    ) -> ComplianceDecision:
        controls: list[dict] = []

        def add(control: str, passed: bool, evidence) -> None:
            controls.append({"control": control, "passed": bool(passed), "evidence": evidence})

        missing = [
            field for field in self.config.required_provenance_fields
            if field not in bundle.provenance or bundle.provenance[field] in (None, "", {}, [])
        ]
        add("required_provenance", not missing, {"required": self.config.required_provenance_fields, "missing": missing})
        model_evidence = bool(model.name and model.version >= 1 and model.features)
        add(
            "model_version_evidence",
            not self.config.require_model_version_evidence or model_evidence,
            {"name": model.name, "version": model.version, "feature_count": len(model.features)},
        )
        add("audit_record_capability", not self.config.require_audit_record or audit_available, {"available": audit_available})
        add(
            "explanation_evidence_capability",
            not self.config.require_explanation_capability or explanation_available,
            {"available": explanation_available},
        )
        denied = sorted(set(model.features) & set(self.config.feature_denylist))
        outside_allowlist = sorted(
            set(model.features) - set(self.config.feature_allowlist)
        ) if self.config.feature_allowlist else []
        add("feature_deny_rules", not denied, {"denied_features_present": denied})
        add(
            "feature_allow_rules", not outside_allowlist,
            {"allowlist_active": bool(self.config.feature_allowlist), "features_outside_allowlist": outside_allowlist},
        )
        violations = [item["control"] for item in controls if not item["passed"]]
        return ComplianceDecision(self.config.policy_id, not violations, controls, violations)
