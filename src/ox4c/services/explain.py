from __future__ import annotations

import numpy as np
import pandas as pd

from ox4c.services.model import DetectorModel


class ShapExplainer:
    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k
        self._cache = {}

    def explain(self, model: DetectorModel, row: pd.DataFrame) -> dict:
        try:
            import shap

            transformed = model.pipeline[:-1].transform(row.loc[:, model.features])
            cache_key = (model.name, model.version, model.algorithm)
            explainer = self._cache.setdefault(
                cache_key, shap.TreeExplainer(model.pipeline.named_steps["classifier"])
            )
            values = np.asarray(explainer.shap_values(transformed))
            if values.ndim == 3:
                vector = values[0, :, -1]
            elif values.ndim == 2:
                vector = values[0]
            else:
                vector = np.asarray(values[-1])[0]
            ranked = sorted(zip(model.features, vector), key=lambda item: abs(float(item[1])), reverse=True)
            expected = np.asarray(explainer.expected_value)
            base_value = float(expected.reshape(-1)[-1])
            full_contributions = {name: float(value) for name, value in zip(model.features, vector)}
            top_contributions = {name: float(value) for name, value in ranked[: self.top_k]}
            omitted_contribution = float(
                sum(value for name, value in full_contributions.items() if name not in top_contributions)
            )
            raw_reconstruction = base_value + float(sum(full_contributions.values()))
            if model.algorithm == "hist_gradient_boosting":
                output_space = "raw_log_odds"
                probability_reconstruction = float(1.0 / (1.0 + np.exp(-raw_reconstruction)))
            else:
                output_space = "probability"
                probability_reconstruction = raw_reconstruction
            return {
                "scope": "decision_model",
                "model_name": model.name,
                "model_algorithm": model.algorithm,
                "model_version": model.version,
                "model_probability": float(model.predict_proba(row)[0]),
                "base_value": base_value,
                "contributions": top_contributions,
                "full_contributions": full_contributions,
                "omitted_contribution": omitted_contribution,
                "output_space": output_space,
                "additive_reconstruction": raw_reconstruction,
                "probability_reconstruction": probability_reconstruction,
            }
        except Exception:
            return {}
