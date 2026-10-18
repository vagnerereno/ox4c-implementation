from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SplitData:
    X: pd.DataFrame
    y: pd.Series
    ids: pd.Series
    timestamps: pd.Series


@dataclass(frozen=True)
class DatasetBundle:
    domain: str
    train: SplitData
    calibration: SplitData
    validation: SplitData
    test: SplitData
    feature_names: tuple[str, ...]
    feature_views: dict[str, list[str]]
    provenance: dict[str, Any]


@dataclass
class Advice:
    counselor: str
    probability: float
    confidence: float
    reliability: float


@dataclass
class Decision:
    sample_id: str
    frozen_probability: float
    local_probability: float
    final_probability: float
    prediction: int
    confidence: float
    model_version: int
    counselors_requested: bool
    collaborative: bool
    conflict: bool = False
    decision_source: str = "primary"
    advice: list[Advice] = field(default_factory=list)
    explanation: dict[str, Any] = field(default_factory=dict)
    collaboration_explanation: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    audit_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
