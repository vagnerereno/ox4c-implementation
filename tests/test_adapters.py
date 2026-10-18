from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from ox4c.adapters import LYCOSIDS2017TemporalAdapter, QUTDV25Adapter


def _write_day(root: Path, day: str, rows: list[list]) -> None:
    filename = LYCOSIDS2017TemporalAdapter.DAY_ARCHIVES[day]
    member = LYCOSIDS2017TemporalAdapter.DAY_MEMBERS[day]
    frame = pd.DataFrame(rows, columns=[
        "flow_id", "src_addr", "src_port", "dst_addr", "dst_port",
        "ip_prot", "timestamp", "flow_duration", "label",
    ])
    with ZipFile(root / filename, "w", ZIP_DEFLATED) as zipped:
        zipped.writestr(member, frame.to_csv(index=False).encode("utf-8"))


def _lycos_archives(root: Path) -> None:
    benign = ["b", "10.0.0.1", 10, "10.0.0.2", 20, 6, 1, 5, "NeedLabel"]
    _write_day(root, "monday", [benign, benign[:6] + [2, 6, "NeedLabel"]])
    _write_day(root, "tuesday", [
        benign,
        ["ftp", "172.16.0.1", 123, "192.168.10.50", 21, 6,
         1499171000000000, 9, "NeedLabel"],
    ])
    _write_day(root, "wednesday", [
        benign,
        ["hulk", "172.16.0.1", 123, "192.168.10.50", 80, 6,
         1499262500000000, 10, "NeedLabel"],
    ])
    _write_day(root, "thursday", [
        benign,
        ["web", "172.16.0.1", 123, "192.168.10.50", 80, 6,
         1499344000000000, 11, "NeedLabel"],
        ["dropped", "10.0.0.1", 10, "10.0.0.2", 20, 6,
         1499353200000000, 12, "NeedLabel"],
    ])
    _write_day(root, "friday", [
        benign,
        ["ddos", "172.16.0.1", 123, "192.168.10.50", 80, 6,
         1499454000000000, 13, "NeedLabel"],
        ["dropped", "10.0.0.1", 10, "52.6.13.28", 20, 6,
         1499431000000000, 14, "NeedLabel"],
    ])


def test_qut_adapter_builds_grouped_partitions(qut_csv: Path) -> None:
    bundle = QUTDV25Adapter(qut_csv, random_state=17).load()

    assert bundle.domain == "software_supply_chain_security"
    assert len(bundle.feature_names) == 13
    assert sum(len(split.y) for split in (
        bundle.train, bundle.calibration, bundle.validation, bundle.test,
    )) == 500
    assert bundle.provenance["split_policy"]
    assert bundle.provenance["sha256"]


def test_lycos_adapter_reproduces_labels_and_temporal_order(tmp_path: Path) -> None:
    _lycos_archives(tmp_path)

    days = LYCOSIDS2017TemporalAdapter(tmp_path, chunksize=1).load()

    assert list(days) == ["monday", "tuesday", "wednesday", "thursday", "friday"]
    assert days["tuesday"].attack_labels.tolist() == ["benign", "ftp_patator"]
    assert "dos_hulk" in set(days["wednesday"].attack_labels)
    assert "webattack_bruteforce" in set(days["thursday"].attack_labels)
    assert "ddos" in set(days["friday"].attack_labels)
    assert len(days["thursday"].y) == 2
    assert len(days["friday"].y) == 2
    assert "timestamp" not in days["monday"].X.columns
    assert days["friday"].ordering.startswith("microsecond Unix timestamp")


def test_lycos_sampling_is_deterministic_and_ordered(tmp_path: Path) -> None:
    rows = [
        [f"f{i}", "10.0.0.1", i, "10.0.0.2", 80, 6, 100 + i, i, "NeedLabel"]
        for i in range(10)
    ]
    for day in LYCOSIDS2017TemporalAdapter.DAY_ARCHIVES:
        _write_day(tmp_path, day, rows)

    adapter = LYCOSIDS2017TemporalAdapter(tmp_path, max_events_per_day=4, chunksize=3)
    first = adapter.load()
    second = adapter.load()

    assert all(len(data.y) == 4 for data in first.values())
    assert first["monday"].sample_ids.tolist() == second["monday"].sample_ids.tolist()
    timestamps = [int(value.split(":")[1]) for value in first["monday"].sample_ids]
    assert timestamps == sorted(timestamps)
