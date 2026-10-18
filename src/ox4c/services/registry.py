from __future__ import annotations

import json
import os
from pathlib import Path

import joblib

from ox4c.services.model import DetectorModel


class ModelRegistry:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "registry.json"
        if not self.index_path.exists():
            self._write({"current": None, "versions": {}})

    def _read(self) -> dict:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _write(self, payload: dict) -> None:
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, self.index_path)

    def promote(self, model: DetectorModel, metrics: dict, provenance: dict) -> None:
        model_path = self.directory / f"primary-v{model.version}.joblib"
        temporary = model_path.with_suffix(".tmp")
        joblib.dump(model, temporary)
        os.replace(temporary, model_path)
        index = self._read()
        index["versions"][str(model.version)] = {
            "version": model.version, "path": str(model_path), "features": model.features,
            "metrics": metrics, "provenance": provenance,
        }
        index["current"] = model.version
        self._write(index)

    def rollback(self, version: int) -> DetectorModel:
        index = self._read()
        if str(version) not in index["versions"]:
            raise KeyError(version)
        index["current"] = version
        self._write(index)
        return joblib.load(index["versions"][str(version)]["path"])
