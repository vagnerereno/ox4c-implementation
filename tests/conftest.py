from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))


@pytest.fixture
def qut_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    rows = 500
    label = rng.binomial(1, 0.5, rows)
    frame = pd.DataFrame({
        "Package_Name": [f"package-{index}" for index in range(rows)],
        "Read_Processes": rng.poisson(10 - label * 3),
        "Write_Processes": rng.poisson(10 + label * 3),
        "Total_Dependencies": rng.poisson(5 - label * 2),
        "Direct_Dependencies": rng.poisson(2 - label),
        "Root_DIR_Access": rng.poisson(3 + label * 5),
        "Temp_DIR_Access": rng.poisson(5 + label),
        "State_Transition": np.where(label, "SYN_SENT_CLOSE", "ESTABLISHED_CLOSE"),
        "Remote_IPs_Access": rng.poisson(2 + label * 4),
        "IO_Operations": rng.poisson(20 - label * 4),
        "Security_Operations": rng.poisson(1 + label * 3),
        "Process_Operations": rng.poisson(3 + label * 2),
        "Pattern_1": np.where(label, "suspicious", "normal"),
        "Pattern_4": np.where(label, "socket_bind", "none"),
        "Labels": label,
    })
    path = tmp_path / "QUT-DV25.csv"
    frame.to_csv(path, index=False)
    return path
