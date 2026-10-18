from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ox4c.config import ModelConfig


@dataclass
class DetectorModel:
    name: str
    version: int
    features: list[str]
    pipeline: Pipeline
    algorithm: str = "random_forest"

    @classmethod
    def train(
        cls, name: str, version: int, features: list[str], X: pd.DataFrame,
        y: pd.Series, config: ModelConfig, algorithm: str | None = None,
    ):
        if len(features) == 0 or y.nunique() != 2:
            raise ValueError(f"Cannot train {name}: features and both classes are required")
        algorithm = algorithm or config.algorithm
        if algorithm == "random_forest":
            classifier = RandomForestClassifier(
                n_estimators=config.n_estimators,
                min_samples_leaf=config.min_samples_leaf,
                class_weight=config.class_weight,
                random_state=config.random_state,
                n_jobs=-1,
            )
            steps = [("imputer", SimpleImputer(strategy="median")), ("classifier", classifier)]
        elif algorithm == "unweighted_random_forest":
            classifier = RandomForestClassifier(
                n_estimators=config.n_estimators,
                min_samples_leaf=config.min_samples_leaf,
                class_weight=None,
                random_state=config.random_state,
                n_jobs=-1,
            )
            steps = [("imputer", SimpleImputer(strategy="median")), ("classifier", classifier)]
        elif algorithm == "attack_sensitive_forest":
            classifier = RandomForestClassifier(
                n_estimators=config.n_estimators,
                min_samples_leaf=config.min_samples_leaf,
                class_weight={0: 1.0, 1: config.attack_sensitive_weight},
                random_state=config.random_state,
                n_jobs=-1,
            )
            steps = [("imputer", SimpleImputer(strategy="median")), ("classifier", classifier)]
        elif algorithm == "extra_trees":
            classifier = ExtraTreesClassifier(
                n_estimators=config.extra_trees_estimators,
                min_samples_leaf=config.min_samples_leaf,
                class_weight=config.class_weight,
                random_state=config.random_state,
                n_jobs=-1,
            )
            steps = [("imputer", SimpleImputer(strategy="median")), ("classifier", classifier)]
        elif algorithm == "hist_gradient_boosting":
            classifier = HistGradientBoostingClassifier(
                max_iter=config.hgb_max_iter,
                learning_rate=config.hgb_learning_rate,
                max_leaf_nodes=config.hgb_max_leaf_nodes,
                l2_regularization=config.hgb_l2_regularization,
                class_weight=config.class_weight,
                random_state=config.random_state,
            )
            steps = [("imputer", SimpleImputer(strategy="median")), ("classifier", classifier)]
        elif algorithm == "logistic_regression":
            classifier = LogisticRegression(
                max_iter=config.logistic_max_iter,
                class_weight=config.class_weight,
                random_state=config.random_state,
            )
            steps = [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        elif algorithm == "compact_tree":
            classifier = DecisionTreeClassifier(
                max_depth=config.compact_max_depth,
                min_samples_leaf=config.min_samples_leaf,
                class_weight=config.class_weight,
                random_state=config.random_state,
            )
            steps = [("imputer", SimpleImputer(strategy="median")), ("classifier", classifier)]
        else:
            raise ValueError(f"Unsupported detector algorithm: {algorithm}")
        pipeline = Pipeline(steps)
        pipeline.fit(X.loc[:, features], y)
        # Training benefits from all cores, while joblib parallel dispatch is
        # counterproductive for the framework's one-event-at-a-time inference.
        if hasattr(pipeline.named_steps["classifier"], "n_jobs"):
            pipeline.named_steps["classifier"].n_jobs = config.inference_jobs
        return cls(name, version, list(features), pipeline, algorithm)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X.loc[:, self.features])[:, 1]


def metrics(model: DetectorModel, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    probability = model.predict_proba(X)
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "roc_auc": float(roc_auc_score(y, probability)) if y.nunique() > 1 else 0.5,
    }
