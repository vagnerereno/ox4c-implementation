from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, matthews_corrcoef

from ox4c.config import ModelConfig, OptimizationConfig


@dataclass
class SelectionResult:
    algorithm: str
    selected_features: list[str]
    elapsed_seconds: float
    metadata: dict
    search_history: list[dict] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.pop("search_history", None)
        return payload


class FeatureOptimizer:
    def __init__(self, config: OptimizationConfig, model_config: ModelConfig) -> None:
        self.config = config
        self.model_config = model_config
        self.random_state = model_config.random_state

    def select(self, X: pd.DataFrame, y: pd.Series) -> SelectionResult:
        started = time.perf_counter()
        if self.config.algorithm == "none":
            return SelectionResult("none", list(X.columns), time.perf_counter() - started, {})
        if self.config.algorithm != "agraspqfs":
            raise ValueError(f"Unknown optimization algorithm: {self.config.algorithm}")
        from agraspqfs import AGraspQFeatureSelector

        # A-GRASPQ-FS ranks features before the detector pipeline is fitted and
        # therefore cannot rely on the detector's imputer. Fit a separate
        # training-only imputer so feature selection receives finite values
        # without leaking validation or test statistics.
        missing_values = int(X.isna().sum().sum())
        prepared = X.replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        prepared = pd.DataFrame(
            imputer.fit_transform(prepared), columns=X.columns, index=X.index,
        )
        if not np.isfinite(prepared.to_numpy()).all():
            raise ValueError("Feature optimization requires finite numeric training features")

        maximum = X.shape[1] if self.config.max_features is None else min(self.config.max_features, X.shape[1])
        minimum = min(self.config.min_features, maximum)
        sample = float(self.config.evaluation_sample_size)
        sample_fraction = sample if sample <= 1 else min(1.0, sample / len(X))
        selector = AGraspQFeatureSelector(
            estimator=RandomForestClassifier(
                n_estimators=self.model_config.n_estimators,
                min_samples_leaf=self.model_config.min_samples_leaf,
                class_weight=self.model_config.class_weight,
                random_state=self.random_state,
                n_jobs=1,
            ),
            scoring=make_scorer(matthews_corrcoef), preset=self.config.preset,
            min_features=minimum, max_features=maximum,
            constructive_iterations=self.config.constructive_iterations,
            local_iterations=self.config.local_iterations,
            priority_queue_size=self.config.priority_queue_size,
            rcl_size=self.config.rcl_size,
            alpha=self.config.alpha,
            update_interval=self.config.update_interval,
            ranking_method=self.config.ranking_method,
            validation_fraction=self.config.validation_fraction,
            cv=self.config.cv,
            max_evaluations=self.config.max_evaluations,
            early_stopping_rounds=self.config.early_stopping_rounds,
            time_budget=self.config.time_budget,
            evaluation_sample_size=sample_fraction,
            random_state=self.random_state,
        )
        selector.fit(prepared, y)
        selected = list(selector.selected_features_)
        history = selector.history_.to_dict(orient="records")
        phase_counts = selector.history_["phase"].value_counts().to_dict()
        resolved = selector._params_
        metadata = {
            "best_score": float(selector.best_score_),
            "baseline_score": float(selector.baseline_score_),
            "evaluations": int(selector.n_evaluations_),
            "cache_hits": int(selector.n_cache_hits_),
            "evaluation_sample_fraction": sample_fraction,
            "imputation": "training_median",
            "missing_values_imputed": missing_values,
            "fallback": False,
            "subset_evaluator": "RandomForestClassifier (same downstream configuration)",
            "scoring": "matthews_corrcoef",
            "stop_reason": str(selector.stop_reason_),
            "search_history_rows": len(history),
            "search_phase_counts": {str(name): int(count) for name, count in phase_counts.items()},
            "final_full_evaluations": int(phase_counts.get("final_full_evaluation", 0)),
            "selected_candidate_revalidated_on_full_search_data": bool(
                phase_counts.get("final_full_evaluation", 0)
            ),
            "resolved_search": {
                "constructive_iterations": int(resolved.constructive_iterations),
                "local_iterations": int(resolved.local_iterations),
                "priority_queue_size": int(resolved.priority_queue_size),
                "evaluation_sample_size": float(resolved.evaluation_sample_size),
                "validation_fraction": float(resolved.validation_fraction),
                "rcl_size": int(resolved.rcl_size),
                "min_features": int(resolved.min_features),
                "max_features": int(resolved.max_features),
                "alpha": float(resolved.alpha),
                "update_interval": int(resolved.update_interval),
                "max_evaluations": resolved.max_evaluations,
                "early_stopping_rounds": resolved.early_stopping_rounds,
                "time_budget": resolved.time_budget,
                "cv": resolved.cv,
            },
        }
        return SelectionResult(
            "agraspqfs", selected, time.perf_counter() - started, metadata, history,
        )
