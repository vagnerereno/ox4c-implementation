from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from ox4c.types import DatasetBundle, SplitData


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and set(numeric.unique()) <= {0, 1}:
        return numeric.astype(int)
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map({
        "0": 0, "false": 0, "benign": 0, "normal": 0, "clean": 0,
        "1": 1, "true": 1, "malicious": 1, "attack": 1, "anomaly": 1,
    })
    if mapped.isna().any():
        raise ValueError(f"Unsupported binary labels: {sorted(text[mapped.isna()].unique())[:10]}")
    return mapped.astype(int)


class SecurityDataAdapter(ABC):
    """Leakage-aware preparation shared by both cybersecurity datasets."""

    target_candidates: tuple[str, ...] = ()
    id_candidates: tuple[str, ...] = ()
    timestamp_candidates: tuple[str, ...] = ()
    excluded_columns: tuple[str, ...] = ()
    csv_encoding: str | None = None

    def __init__(self, csv_path: str | Path, max_rows: int | None = None, random_state: int = 42) -> None:
        self.csv_path = Path(csv_path)
        self.max_rows = max_rows
        self.random_state = random_state

    @abstractmethod
    def label(self, frame: pd.DataFrame) -> tuple[pd.Series, str]:
        raise NotImplementedError

    @abstractmethod
    def views(self, feature_names: list[str]) -> dict[str, list[str]]:
        raise NotImplementedError

    def _find(self, columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
        lookup = {str(column).strip().lower(): str(column) for column in columns}
        for candidate in candidates:
            if candidate.lower() in lookup:
                return lookup[candidate.lower()]
        return None

    def _split_indices(
        self, frame: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series, str]:
        timestamp_name = self._find(frame.columns, self.timestamp_candidates)
        if timestamp_name:
            numeric = pd.to_numeric(frame[timestamp_name], errors="coerce")
            if "epoch" in timestamp_name.lower() and numeric.notna().mean() >= 0.95:
                unit = "ms" if numeric.dropna().median() > 100_000_000_000 else "s"
                timestamps = pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
            else:
                timestamps = pd.to_datetime(
                    frame[timestamp_name], format="mixed", errors="coerce", utc=True,
                )
            if timestamps.notna().mean() >= 0.95:
                order = np.argsort(timestamps.fillna(timestamps.min()).to_numpy(), kind="stable")
                a, b, c = int(len(frame) * 0.60), int(len(frame) * 0.75), int(len(frame) * 0.85)
                candidates = (order[:a], order[a:b], order[b:c], order[c:])
                if all(y.iloc[index].nunique() == 2 for index in candidates):
                    if groups is None or all(
                        not set(groups.iloc[left]).intersection(set(groups.iloc[right]))
                        for position, left in enumerate(candidates)
                        for right in candidates[position + 1:]
                    ):
                        return *candidates, timestamps, f"chronological 60/15/10/15 using {timestamp_name}"

        if groups is not None and groups.nunique() >= 8:
            folds = min(20, int(groups.nunique()))
            assignment = np.full(len(frame), -1, dtype=int)
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
            for fold, (_, held_out) in enumerate(splitter.split(frame, y, groups)):
                assignment[held_out] = fold
            train_end = max(1, round(folds * 0.60))
            calibration_end = max(train_end + 1, round(folds * 0.75))
            validation_end = max(calibration_end + 1, round(folds * 0.85))
            validation_end = min(validation_end, folds - 1)
            candidates = (
                np.flatnonzero(assignment < train_end),
                np.flatnonzero((assignment >= train_end) & (assignment < calibration_end)),
                np.flatnonzero((assignment >= calibration_end) & (assignment < validation_end)),
                np.flatnonzero(assignment >= validation_end),
            )
            if all(len(index) and y.iloc[index].nunique() == 2 for index in candidates):
                rng = np.random.default_rng(self.random_state)
                candidates = tuple(rng.permutation(index) for index in candidates)
                timestamps = pd.Series(
                    pd.date_range("2000-01-01", periods=len(frame), freq="s", tz="UTC"), index=frame.index,
                )
                return *candidates, timestamps, (
                    "stratified-group 60/15/10/15; repeated entities kept in one partition; "
                    "deterministic within-partition shuffle"
                )

        indices = np.arange(len(frame))
        train, remainder = train_test_split(indices, train_size=0.60, stratify=y, random_state=self.random_state)
        calibration, remainder = train_test_split(
            remainder, train_size=0.375, stratify=y.iloc[remainder], random_state=self.random_state,
        )
        validation, test = train_test_split(
            remainder, train_size=0.40, stratify=y.iloc[remainder], random_state=self.random_state,
        )
        timestamps = pd.Series(pd.date_range("2000-01-01", periods=len(frame), freq="s", tz="UTC"), index=frame.index)
        return train, calibration, validation, test, timestamps, "stratified 60/15/10/15; source has no usable class-safe timestamp"

    def _numeric_features(
        self, frame: pd.DataFrame, excluded: set[str], train_indices: np.ndarray
    ) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
        result: dict[str, pd.Series] = {}
        encoders: dict[str, dict[str, float]] = {}
        for column in frame.columns:
            if column in excluded:
                continue
            series = frame[column]
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().mean() >= 0.95:
                result[str(column)] = numeric.replace([np.inf, -np.inf], np.nan)
                continue
            text = series.fillna("<missing>").astype(str)
            frequencies = text.iloc[train_indices].value_counts(normalize=True).to_dict()
            encoders[str(column)] = {str(key): float(value) for key, value in frequencies.items()}
            result[str(column)] = text.map(frequencies).fillna(0.0).astype(float)
        if not result:
            raise ValueError("No usable predictive columns were found")
        return pd.DataFrame(result, index=frame.index), encoders

    def load(self) -> DatasetBundle:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")
        frame = pd.read_csv(self.csv_path, low_memory=False, encoding=self.csv_encoding)
        if self.max_rows and len(frame) > self.max_rows:
            frame, _ = train_test_split(
                frame, train_size=self.max_rows, stratify=self.label(frame)[0], random_state=self.random_state
            )
            frame = frame.reset_index(drop=True)
        y, target_name = self.label(frame)
        if y.nunique() != 2:
            raise ValueError("Binary detection requires both benign and malicious observations")
        id_name = self._find(frame.columns, self.id_candidates)
        ids = frame[id_name].astype(str) if id_name else pd.Series([f"sample-{index:08d}" for index in range(len(frame))])
        timestamp_name = self._find(frame.columns, self.timestamp_candidates)
        excluded = {target_name, id_name, timestamp_name, *self.excluded_columns, None}
        predictor_columns = [column for column in frame.columns if column not in excluded]
        fingerprints = pd.util.hash_pandas_object(frame.loc[:, predictor_columns], index=False).astype(str)
        repeated_ids = bool(id_name and ids.duplicated(keep=False).any())
        repeated_features = fingerprints.duplicated(keep=False)
        groups = ids if repeated_ids else (fingerprints if repeated_features.any() else None)
        group_source = id_name if repeated_ids else ("raw_feature_fingerprint" if groups is not None else None)
        train_idx, calibration_idx, validation_idx, test_idx, timestamps, split_policy = self._split_indices(frame, y, groups)
        X, encoders = self._numeric_features(frame, excluded, train_idx)

        def split(indices: np.ndarray) -> SplitData:
            return SplitData(
                X.iloc[indices].reset_index(drop=True),
                y.iloc[indices].reset_index(drop=True),
                ids.iloc[indices].reset_index(drop=True),
                timestamps.iloc[indices].reset_index(drop=True),
            )

        feature_names = list(X.columns)
        feature_views = {name: values for name, values in self.views(feature_names).items() if values}
        if len(feature_views) < 2:
            midpoint = max(1, len(feature_names) // 2)
            feature_views = {"behavior_a": feature_names[:midpoint], "behavior_b": feature_names[midpoint:]}
        partitions = {
            "train": train_idx, "calibration": calibration_idx,
            "validation": validation_idx, "test": test_idx,
        }
        group_overlap = {}
        if groups is not None:
            for left_position, (left_name, left_indices) in enumerate(partitions.items()):
                for right_name, right_indices in list(partitions.items())[left_position + 1:]:
                    group_overlap[f"{left_name}__{right_name}"] = len(
                        set(groups.iloc[left_indices]).intersection(set(groups.iloc[right_indices]))
                    )
        return DatasetBundle(
            self.domain,
            split(train_idx), split(calibration_idx), split(validation_idx), split(test_idx),
            tuple(feature_names), feature_views,
            {
                "source_file": str(self.csv_path),
                "sha256": sha256(self.csv_path),
                "rows": len(frame),
                "class_counts": y.value_counts().sort_index().to_dict(),
                "target": target_name,
                "sample_id": id_name,
                "split_policy": split_policy,
                "split_sizes": {name: len(indices) for name, indices in partitions.items()},
                "split_class_counts": {
                    name: y.iloc[indices].value_counts().sort_index().to_dict()
                    for name, indices in partitions.items()
                },
                "leakage_audit": {
                    "rows_with_repeated_raw_features": int(repeated_features.sum()),
                    "unique_sample_ids": int(ids.nunique()),
                    "repeated_sample_ids": int(ids.duplicated(keep=False).sum()),
                    "group_source": group_source,
                    "cross_partition_group_overlap": group_overlap,
                },
                "categorical_encoding": "training-frequency encoding",
                "categorical_columns": sorted(encoders),
                "feature_views": feature_views,
            },
        )
