from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, recall_score

from ox4c.config import CollaborationConfig
from ox4c.services.model import DetectorModel
from ox4c.types import Advice


@dataclass(frozen=True)
class ClassifierCompetence:
    classifier: str
    global_mcc: float
    predictive_value: dict[int, float]
    class_recall: dict[int, float]


@dataclass
class ConflictResolution:
    conflict: bool
    authorized: bool
    selected_classifier: str
    proposed_classifier: str
    primary_probability: float
    final_probability: float
    predictions: dict[str, int]
    probabilities: dict[str, float]
    vote_scores: dict[int, float]
    advice: list[Advice]

    @property
    def changed_primary_decision(self) -> bool:
        return (self.primary_probability >= 0.5) != (self.final_probability >= 0.5)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["changed_primary_decision"] = self.changed_primary_decision
        return payload


class ConflictDrivenCounselors:
    """Resolve explicit disagreement with a validation-selected voting rule.

    Reliability is the predictive value of the label emitted by each model,
    P(y=c | model predicts c). This differs from class recall and avoids treating
    a model with high specificity as automatically trustworthy whenever it emits
    a negative prediction. Calibration selects the minimum number of attack
    votes (or, optionally, a reliability-weighted consensus), and the complete
    resolver is enabled only when it improves calibration MCC.
    """

    def __init__(
        self, models: dict[str, DetectorModel], primary_name: str,
        X_validation: pd.DataFrame, y_validation: pd.Series,
        config: CollaborationConfig | None = None,
    ) -> None:
        if primary_name not in models:
            raise KeyError(f"Primary classifier {primary_name!r} is not in the pool")
        self.models = dict(models)
        self.primary_name = primary_name
        self.config = config or CollaborationConfig()
        if self.config.consensus not in {"reliability_weighted", "calibration_selected_vote"}:
            raise ValueError(f"Unsupported consensus: {self.config.consensus}")
        self.competence = self._estimate_competence(X_validation, y_validation)
        self.calibration = self._calibrate_activation(X_validation, y_validation)
        self.enabled = bool(self.calibration["authorized"])

    def _estimate_competence(
        self, X: pd.DataFrame, y: pd.Series,
    ) -> dict[str, ClassifierCompetence]:
        result: dict[str, ClassifierCompetence] = {}
        truth = y.to_numpy(dtype=int)
        for name, model in self.models.items():
            prediction = (model.predict_proba(X) >= 0.5).astype(int)
            predictive_value: dict[int, float] = {}
            class_recall: dict[int, float] = {}
            for label in (0, 1):
                predicted_mask = prediction == label
                truth_mask = truth == label
                predictive_value[label] = (
                    float((truth[predicted_mask] == label).mean())
                    if predicted_mask.any() else 0.0
                )
                class_recall[label] = (
                    float((prediction[truth_mask] == label).mean())
                    if truth_mask.any() else 0.0
                )
            result[name] = ClassifierCompetence(
                classifier=name,
                global_mcc=float(matthews_corrcoef(truth, prediction)),
                predictive_value=predictive_value,
                class_recall=class_recall,
            )
        return result

    def replace_primary(
        self, model: DetectorModel, X_validation: pd.DataFrame, y_validation: pd.Series,
    ) -> None:
        self.models[self.primary_name] = model
        self.competence = self._estimate_competence(X_validation, y_validation)
        self.calibration = self._calibrate_activation(X_validation, y_validation)
        self.enabled = bool(self.calibration["authorized"])

    def _weighted_choice(self, labels: dict[str, int]) -> tuple[str, dict[int, float]]:
        scores = {0: 0.0, 1: 0.0}
        for name, label in labels.items():
            scores[label] += self.competence[name].predictive_value[label]
        primary_label = labels[self.primary_name]
        if np.isclose(scores[0], scores[1]):
            consensus_label = primary_label
        else:
            consensus_label = int(scores[1] > scores[0])
        voters = [name for name, label in labels.items() if label == consensus_label]
        selected = max(
            voters,
            key=lambda name: (
                self.competence[name].predictive_value[consensus_label],
                self.competence[name].global_mcc,
                int(name == self.primary_name),
            ),
        )
        return selected, scores

    def _representative(self, labels: dict[str, int], consensus_label: int) -> str:
        voters = [name for name, label in labels.items() if label == consensus_label]
        return max(
            voters,
            key=lambda name: (
                self.competence[name].predictive_value[consensus_label],
                self.competence[name].global_mcc,
                int(name == self.primary_name),
            ),
        )

    def _propose(self, labels: dict[str, int]) -> tuple[str, dict[int, float]]:
        if self.config.consensus == "reliability_weighted":
            return self._weighted_choice(labels)
        attack_votes = sum(labels.values())
        minimum_votes = int(self.calibration["selected_minimum_attack_votes"])
        consensus_label = int(attack_votes >= minimum_votes)
        return self._representative(labels, consensus_label), {
            0: float(len(labels) - attack_votes), 1: float(attack_votes),
        }

    @staticmethod
    def _classification_snapshot(truth: np.ndarray, prediction: np.ndarray) -> dict:
        tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
        return {
            "mcc": float(matthews_corrcoef(truth, prediction)),
            "f1": float(f1_score(truth, prediction, zero_division=0)),
            "recall": float(recall_score(truth, prediction, zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }

    def _calibrate_activation(self, X: pd.DataFrame, y: pd.Series) -> dict:
        probabilities = {
            name: model.predict_proba(X) for name, model in self.models.items()
        }
        predictions = {
            name: (values >= 0.5).astype(int) for name, values in probabilities.items()
        }
        primary = predictions[self.primary_name].copy()
        truth = y.to_numpy(dtype=int)
        conflicts = changes = primary_negative_conflicts = primary_positive_conflicts = 0
        fn_with_conflict = fp_with_conflict = 0
        for position in range(len(X)):
            labels = {name: int(values[position]) for name, values in predictions.items()}
            if len(set(labels.values())) <= 1:
                continue
            conflicts += 1
            primary_label = labels[self.primary_name]
            primary_negative_conflicts += int(primary_label == 0)
            primary_positive_conflicts += int(primary_label == 1)
            fn_with_conflict += int(primary_label == 0 and truth[position] == 1)
            fp_with_conflict += int(primary_label == 1 and truth[position] == 0)
        primary_metrics = self._classification_snapshot(truth, primary)
        vote_candidates = []
        selected_minimum_attack_votes = None
        if self.config.consensus == "calibration_selected_vote":
            vote_count = np.column_stack(list(predictions.values())).sum(axis=1)
            centre = (len(self.models) + 1) / 2
            for minimum_votes in range(1, len(self.models) + 1):
                candidate = (vote_count >= minimum_votes).astype(int)
                candidate_metrics = self._classification_snapshot(truth, candidate)
                vote_candidates.append({
                    "minimum_attack_votes": minimum_votes,
                    **candidate_metrics,
                })
            selected = max(
                vote_candidates,
                key=lambda item: (item["mcc"], -abs(item["minimum_attack_votes"] - centre)),
            )
            selected_minimum_attack_votes = int(selected["minimum_attack_votes"])
            resolved = (vote_count >= selected_minimum_attack_votes).astype(int)
            changes = int((resolved != primary).sum())
        else:
            resolved = primary.copy()
            for position in range(len(X)):
                labels = {name: int(values[position]) for name, values in predictions.items()}
                if len(set(labels.values())) <= 1:
                    continue
                selected, _ = self._weighted_choice(labels)
                resolved[position] = labels[selected]
                changes += int(resolved[position] != primary[position])
        resolved_metrics = self._classification_snapshot(truth, resolved)
        gain = resolved_metrics["mcc"] - primary_metrics["mcc"]
        return {
            "rule": "select the voting rule on calibration MCC and authorize it only when calibration MCC improves",
            "reliability_definition": "P(true label = c | classifier predicts c)",
            "consensus": self.config.consensus,
            "vote_candidates": vote_candidates,
            "selected_minimum_attack_votes": selected_minimum_attack_votes,
            "primary_metrics": primary_metrics,
            "resolved_metrics": resolved_metrics,
            "primary_mcc": primary_metrics["mcc"],
            "resolved_mcc": resolved_metrics["mcc"],
            "mcc_gain": gain,
            "minimum_mcc_gain": self.config.minimum_calibration_mcc_gain,
            "observed_conflicts": conflicts,
            "primary_negative_conflicts": primary_negative_conflicts,
            "primary_positive_conflicts": primary_positive_conflicts,
            "primary_false_negatives_with_conflict": fn_with_conflict,
            "primary_false_positives_with_conflict": fp_with_conflict,
            "changed_decisions": changes,
            "authorized": bool(gain > self.config.minimum_calibration_mcc_gain),
        }

    def resolve(self, row: pd.DataFrame) -> ConflictResolution:
        probabilities = {
            name: float(model.predict_proba(row)[0]) for name, model in self.models.items()
        }
        predictions = {name: int(probability >= 0.5) for name, probability in probabilities.items()}
        primary_probability = probabilities[self.primary_name]
        conflict = len(set(predictions.values())) > 1
        proposed = self.primary_name
        scores = {0: 0.0, 1: 0.0}
        if conflict:
            proposed, scores = self._propose(predictions)
        selected = proposed if conflict and self.enabled else self.primary_name
        advice = [
            Advice(
                counselor=name,
                probability=probabilities[name],
                confidence=max(probabilities[name], 1 - probabilities[name]),
                reliability=self.competence[name].predictive_value[predictions[name]],
            )
            for name in self.models
            if name != self.primary_name
        ]
        return ConflictResolution(
            conflict=conflict,
            authorized=bool(conflict and self.enabled),
            selected_classifier=selected,
            proposed_classifier=proposed,
            primary_probability=primary_probability,
            final_probability=probabilities[selected],
            predictions=predictions,
            probabilities=probabilities,
            vote_scores=scores,
            advice=advice,
        )

    def metadata(self) -> dict:
        return {
            "primary_classifier": self.primary_name,
            "classifiers": list(self.models),
            "consensus": self.config.consensus,
            "competence": {
                name: asdict(value) for name, value in self.competence.items()
            },
            "conflict_rule": "request counsel when predicted labels disagree",
            "resolution_rule": (
                "calibration-selected minimum attack votes; select the most reliable voter for the consensus label"
                if self.config.consensus == "calibration_selected_vote"
                else "reliability-weighted vote; select the most reliable voter for the consensus label"
            ),
            "activation": self.calibration,
        }
