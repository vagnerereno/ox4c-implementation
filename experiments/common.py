from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score,
)


def _score(y: pd.Series, prediction: pd.Series) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "false_positive_rate": float(fp / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, prediction)),
    }


def summarize_run(events, system, domain: str, output: Path, experiment: dict) -> dict:
    y = events.true_label.astype(int)
    frozen = (events.frozen_probability >= 0.5).astype(int)
    local = (events.local_probability >= 0.5).astype(int)
    final = events.prediction.astype(int)
    conflict = events.conflict.astype(bool)
    collaborative = events.collaborative.astype(bool)
    changed = collaborative & (local != final)
    compact_path = output / "compact_model.json"
    compact = json.loads(compact_path.read_text(encoding="utf-8")) if compact_path.exists() else None
    payload = {
        "protocol": "ox4c-conflict-driven-v1",
        "domain": domain,
        "events": len(events),
        "experiment": experiment,
        "frozen_metrics": _score(y, frozen),
        "local_metrics": _score(y, local),
        "final_metrics": _score(y, final),
        "features": {
            "algorithm": system.selection.algorithm,
            "selected": system.selection.selected_features,
            "selected_count": len(system.selection.selected_features),
            "available_count": len(system.bundle.feature_names),
            "runtime_seconds": system.selection.elapsed_seconds,
            "metadata": system.selection.metadata,
            "candidate_count": system.selection.metadata.get("deployment", {}).get(
                "candidate_count", len(system.selection.selected_features)
            ),
            "deployment_accepted": system.selection.metadata.get("deployment", {}).get(
                "accepted", True
            ),
        },
        "collaboration": {
            "conflicts": int(conflict.sum()),
            "primary_negative_conflicts": int(((local == 0) & conflict).sum()),
            "primary_positive_conflicts": int(((local == 1) & conflict).sum()),
            "primary_false_negatives": int(((local == 0) & (y == 1)).sum()),
            "primary_false_negatives_with_conflict": int(
                ((local == 0) & (y == 1) & conflict).sum()
            ),
            "primary_false_positives": int(((local == 1) & (y == 0)).sum()),
            "primary_false_positives_with_conflict": int(
                ((local == 1) & (y == 0) & conflict).sum()
            ),
            "authorized_interventions": int(collaborative.sum()),
            "changed_decisions": int(changed.sum()),
            "corrections": int(((local != y) & (final == y) & changed).sum()),
            "harms": int(((local == y) & (final != y) & changed).sum()),
            "net_corrections": int(((local != y) & (final == y) & changed).sum())
            - int(((local == y) & (final != y) & changed).sum()),
            "decision_sources": dict(Counter(events.decision_source.astype(str))),
            "competence": system.counselors.metadata(),
        },
        "learning": {
            "enabled": system.config.learning.enabled,
            "attempts": system.learning.attempts,
            "promotions": system.learning.promotions,
            "final_primary_version": system.primary.version,
            "pending_labels": len(system.learning.buffer),
            "promotion_rule": "candidate validation MCC > current validation MCC",
        },
        "compact_model": compact,
        "technical_conformance": (
            system.compliance_decision.to_dict() if system.compliance_decision else None
        ),
        "performance": system.performance.snapshot(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
