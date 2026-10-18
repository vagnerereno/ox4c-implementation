from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np


class AuditService:
    def __init__(self, directory: str | Path, commit_every: int = 100) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "audit.sqlite3"
        self.commit_every = max(1, int(commit_every))
        self.pending = 0
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, occurred_at TEXT, event_type TEXT, payload TEXT)"
        )
        self.connection.commit()

    def record(self, event_type: str, payload: dict) -> str:
        event_id = str(uuid.uuid4())
        self.connection.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?)",
            (event_id, datetime.now(timezone.utc).isoformat(), event_type, json.dumps(payload, default=str)),
        )
        self.pending += 1
        if self.pending >= self.commit_every:
            self.flush()
        return event_id

    def flush(self) -> None:
        self.connection.commit()
        self.pending = 0

    def close(self) -> None:
        self.flush()
        self.connection.close()

    def __del__(self) -> None:
        try:
            self.connection.close()
        except (AttributeError, sqlite3.Error):
            pass


class PerformanceMonitor:
    def __init__(self) -> None:
        self.times = defaultdict(list)
        self.counters = Counter()

    @contextmanager
    def measure(self, component: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.times[component].append((time.perf_counter() - started) * 1000)

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] += amount

    def observe(self, component: str, milliseconds: float) -> None:
        self.times[component].append(float(milliseconds))

    def snapshot(self) -> dict:
        result = {
            "timings": {
                name: {
                    "count": len(values),
                    "mean_ms": mean(values),
                    "p50_ms": float(np.percentile(values, 50)),
                    "p95_ms": float(np.percentile(values, 95)),
                    "p99_ms": float(np.percentile(values, 99)),
                    "max_ms": max(values),
                    "total_ms": sum(values),
                }
                for name, values in self.times.items() if values
            },
            "counters": dict(self.counters),
        }
        if self.times.get("stream_wall") and self.counters.get("stream_events"):
            seconds = self.times["stream_wall"][-1] / 1000
            result["throughput_events_per_second"] = self.counters["stream_events"] / seconds if seconds else None
        return result
