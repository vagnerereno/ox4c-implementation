from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalDayData:
    """Chronologically ordered events and provenance for one capture day."""

    day: str
    X: pd.DataFrame
    y: pd.Series
    attack_labels: pd.Series
    sample_ids: pd.Series
    source_files: tuple[str, ...]
    ordering: str


@dataclass(frozen=True)
class LycosSource:
    day: str
    archive: Path
    member: str


class LYCOSIDS2017TemporalAdapter:
    """Read the corrected LYCOS-IDS2017 flows in chronological order.

    The public repository distributes one ZIP archive per capture day.  Those
    CSVs contain microsecond Unix timestamps and the raw ``NeedLabel`` marker.
    This adapter applies the deterministic labelling rules published with the
    dataset, removes identifiers/timestamps from the predictive feature set,
    and retains timestamps solely for ordering and provenance.

    ``max_events_per_day`` enables a deterministic pilot without randomising
    the stream: evenly spaced events are selected across the complete ordered
    day.  ``None`` retains every eligible event.
    """

    DAY_ARCHIVES = {
        "monday": "Monday-WorkingHours.pcap_lycos.zip",
        "tuesday": "Tuesday-WorkingHours.pcap_lycos.zip",
        "wednesday": "Wednesday-WorkingHours.pcap_lycos.zip",
        "thursday": "Thursday-WorkingHours.pcap_lycos.zip",
        "friday": "Friday-WorkingHours.pcap_lycos.zip",
    }
    DAY_MEMBERS = {
        day: filename.removesuffix(".zip") + ".csv"
        for day, filename in DAY_ARCHIVES.items()
    }
    NON_PREDICTIVE_COLUMNS = {
        "flow_id", "src_addr", "dst_addr", "timestamp", "label",
    }

    def __init__(
        self,
        root: str | Path,
        max_events_per_day: int | None = None,
        chunksize: int = 100_000,
    ) -> None:
        root = Path(root)
        self.root = root / "pcap_lycos" if (root / "pcap_lycos").is_dir() else root
        self.max_events_per_day = (
            None if max_events_per_day is None else int(max_events_per_day)
        )
        self.chunksize = int(chunksize)
        if self.max_events_per_day is not None and self.max_events_per_day <= 0:
            raise ValueError("max_events_per_day must be positive or None")
        if self.chunksize <= 0:
            raise ValueError("chunksize must be positive")

    def _source(self, day: str) -> LycosSource:
        if day not in self.DAY_ARCHIVES:
            raise ValueError(f"Unsupported LYCOS-IDS2017 day: {day}")
        archive = self.root / self.DAY_ARCHIVES[day]
        if not archive.exists():
            raise FileNotFoundError(f"Missing LYCOS-IDS2017 archive: {archive}")
        member = self.DAY_MEMBERS[day]
        with ZipFile(archive) as zipped:
            if member not in zipped.namelist():
                raise FileNotFoundError(f"Archive {archive} does not contain {member}")
        return LycosSource(day=day, archive=archive, member=member)

    @staticmethod
    def _label_day(frame: pd.DataFrame, day: str) -> pd.DataFrame:
        """Apply the rules from the official LYCOS ``labelling.py`` artifact."""
        frame = frame.copy()
        frame["label"] = frame["label"].astype(str).str.strip()
        timestamp = pd.to_numeric(frame["timestamp"], errors="coerce")

        if day == "monday":
            frame["label"] = "benign"
            return frame

        if day == "tuesday":
            endpoint = (
                frame["src_addr"].eq("172.16.0.1")
                & frame["dst_addr"].eq("192.168.10.50")
            )
            frame.loc[
                endpoint & timestamp.between(1499170620000000, 1499175000000000),
                "label",
            ] = "ftp_patator"
            frame.loc[
                endpoint & timestamp.between(1499188140000000, 1499191860000000),
                "label",
            ] = "ssh_patator"

        elif day == "wednesday":
            frame.loc[
                frame["src_addr"].eq("172.16.0.1")
                & frame["src_port"].eq(45022)
                & frame["dst_addr"].eq("192.168.10.51")
                & frame["dst_port"].eq(444),
                "label",
            ] = "heartbleed"
            endpoint = (
                frame["src_addr"].eq("172.16.0.1")
                & frame["dst_addr"].eq("192.168.10.50")
                & frame["dst_port"].eq(80)
            )
            slowloris = timestamp.between(1499256060000000, 1499260260000000) | (
                timestamp.between(1499275440000000, 1499275500000000)
            )
            frame.loc[endpoint & slowloris, "label"] = "dos_slowloris"
            frame.loc[
                endpoint & timestamp.between(1499260500000000, 1499261820000000),
                "label",
            ] = "dos_slowhttptest"
            frame.loc[
                endpoint & timestamp.between(1499262180000000, 1499263620000000),
                "label",
            ] = "dos_hulk"
            frame.loc[
                endpoint & timestamp.between(1499263800000000, 1499264340000000),
                "label",
            ] = "dos_goldeneye"

        elif day == "thursday":
            endpoint = (
                frame["src_addr"].eq("172.16.0.1")
                & frame["dst_addr"].eq("192.168.10.50")
                & frame["ip_prot"].eq(6)
            )
            frame.loc[
                endpoint & timestamp.between(1499343300000000, 1499346000000000),
                "label",
            ] = "webattack_bruteforce"
            frame.loc[
                endpoint & timestamp.between(1499346935000000, 1499348100000000),
                "label",
            ] = "webattack_xss"
            frame.loc[
                endpoint & timestamp.between(1499348400000000, 1499348576000000),
                "label",
            ] = "webattack_sql_injection"
            frame = frame.loc[timestamp < 1499353200000000].copy()

        elif day == "friday":
            keep = ~(
                frame["dst_addr"].eq("205.174.165.73")
                & timestamp.gt(1499436193000000)
            )
            keep &= ~frame["dst_addr"].isin(["52.6.13.28", "52.7.235.158"])
            frame = frame.loc[keep].copy()
            timestamp = pd.to_numeric(frame["timestamp"], errors="coerce")
            frame.loc[
                frame["dst_addr"].eq("205.174.165.73")
                & timestamp.between(1499430840000000, 1499436122000000),
                "label",
            ] = "bot"
            endpoint = (
                frame["src_addr"].eq("172.16.0.1")
                & frame["dst_addr"].eq("192.168.10.50")
                & frame["ip_prot"].eq(6)
            )
            frame.loc[
                endpoint & timestamp.between(1499453791000000, 1499454973000000),
                "label",
            ] = "ddos"
            frame.loc[
                endpoint & timestamp.between(1499443530000000, 1499451842000000),
                "label",
            ] = "portscan"

        else:  # pragma: no cover - validated by _source
            raise ValueError(f"Unsupported LYCOS-IDS2017 day: {day}")

        frame.loc[frame["label"].eq("NeedLabel"), "label"] = "benign"
        return frame

    @staticmethod
    def _systematic_positions(length: int, target: int | None) -> np.ndarray:
        if target is None or target >= length:
            return np.arange(length, dtype=np.int64)
        # Bin midpoints cover the complete day without randomising event order.
        positions = ((np.arange(target, dtype=np.float64) + 0.5) * length / target)
        return np.minimum(positions.astype(np.int64), length - 1)

    def load_day(self, day: str) -> TemporalDayData:
        source = self._source(day)
        parts: list[pd.DataFrame] = []
        source_offset = 0
        with ZipFile(source.archive) as zipped, zipped.open(source.member) as handle:
            for chunk in pd.read_csv(handle, chunksize=self.chunksize, low_memory=False):
                chunk.columns = [str(column).strip() for column in chunk.columns]
                chunk["__source_row"] = np.arange(
                    source_offset, source_offset + len(chunk), dtype=np.int64,
                )
                source_offset += len(chunk)
                labelled = self._label_day(chunk, day)
                if not labelled.empty:
                    parts.append(labelled)
        if not parts:
            raise ValueError(f"No eligible LYCOS-IDS2017 events found for {day}")

        frame = pd.concat(parts, ignore_index=True)
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
        if frame["timestamp"].isna().any():
            raise ValueError(f"Invalid LYCOS timestamp found on {day}")
        frame = frame.sort_values(
            ["timestamp", "src_port", "dst_port", "src_addr", "dst_addr", "__source_row"],
            kind="stable",
        ).reset_index(drop=True)
        complete_day = self.max_events_per_day is None or self.max_events_per_day >= len(frame)
        positions = self._systematic_positions(len(frame), self.max_events_per_day)
        frame = frame.iloc[positions].reset_index(drop=True)

        labels = frame["label"].astype(str).str.strip().str.lower()
        y = labels.ne("benign").astype(int)
        source_rows = frame["__source_row"].astype(int)
        timestamps = frame["timestamp"].astype(np.int64)
        sample_ids = pd.Series(
            [f"{day}:{timestamp}:{row}" for timestamp, row in zip(timestamps, source_rows)],
            dtype=str,
        )
        feature_columns = [
            column for column in frame.columns
            if column not in self.NON_PREDICTIVE_COLUMNS and column != "__source_row"
        ]
        X = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        return TemporalDayData(
            day=day,
            X=X.reset_index(drop=True),
            y=y.reset_index(drop=True),
            attack_labels=labels.reset_index(drop=True),
            sample_ids=sample_ids,
            source_files=(source.archive.name,),
            ordering=(
                "microsecond Unix timestamp with documented LYCOS tie-breakers; "
                + (
                    "complete eligible day"
                    if complete_day
                    else "deterministic systematic temporal sample across the complete day"
                )
            ),
        )

    def load(self) -> dict[str, TemporalDayData]:
        days = {day: self.load_day(day) for day in self.DAY_ARCHIVES}
        columns = list(days["monday"].X.columns)
        for day, data in days.items():
            if list(data.X.columns) != columns:
                raise ValueError(f"LYCOS feature schema differs on {day}")
        return days
