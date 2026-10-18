from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ox4c.config import LearningConfig, ModelConfig
from ox4c.services.model import DetectorModel, metrics
from ox4c.services.registry import ModelRegistry
from ox4c.types import SplitData


@dataclass
class LabeledObservation:
    X: pd.DataFrame
    label: int
    sample_id: str


class ContinuousLearningService:
    """Periodic supervised retraining with one transparent promotion rule."""

    def __init__(
        self, train: SplitData, validation: SplitData, config: LearningConfig,
        model_config: ModelConfig, registry: ModelRegistry,
    ) -> None:
        self.X = train.X.copy()
        self.y = train.y.copy()
        self.validation = validation
        self.config = config
        self.model_config = model_config
        self.registry = registry
        self.buffer: list[LabeledObservation] = []
        self.next_attempt_size = config.batch_size
        self.promotions = 0
        self.attempts = 0

    def admit(self, item: LabeledObservation) -> None:
        self.buffer.append(item)

    def ready(self) -> bool:
        return (
            self.config.enabled
            and self.promotions < self.config.max_promotions
            and len(self.buffer) >= self.next_attempt_size
        )

    def update_if_ready(self, current: DetectorModel) -> tuple[DetectorModel, dict | None]:
        if not self.ready():
            return current, None
        self.attempts += 1
        new_X = pd.concat([item.X for item in self.buffer], ignore_index=True)
        new_y = pd.Series([item.label for item in self.buffer], dtype=int)
        candidate_X = pd.concat([self.X, new_X], ignore_index=True)
        candidate_y = pd.concat([self.y, new_y], ignore_index=True)
        candidate = DetectorModel.train(
            current.name, current.version + 1, current.features,
            candidate_X, candidate_y, self.model_config, algorithm=current.algorithm,
        )
        current_metrics = metrics(current, self.validation.X, self.validation.y)
        candidate_metrics = metrics(candidate, self.validation.X, self.validation.y)
        accepted = candidate_metrics["mcc"] > current_metrics["mcc"]
        result = {
            "accepted": accepted,
            "promotion_rule": "candidate validation MCC > current validation MCC",
            "current": current_metrics,
            "candidate": candidate_metrics,
            "knowledge_items": len(self.buffer),
        }
        if accepted:
            self.X, self.y = candidate_X, candidate_y
            self.buffer.clear()
            self.next_attempt_size = self.config.batch_size
            self.promotions += 1
            self.registry.promote(candidate, candidate_metrics, {
                "parent_version": current.version,
                "knowledge_items": result["knowledge_items"],
                "promotion_rule": result["promotion_rule"],
            })
            return candidate, result
        self.next_attempt_size += self.config.batch_size
        return current, result
