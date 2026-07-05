from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pickleball_moe.demo import generate_demo_csv


def test_generate_demo_csv(tmp_path: Path) -> None:
    out = generate_demo_csv(tmp_path / "demo.csv", n_samples=32, seed=7)
    with out.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 32
    assert {"stance_x", "ball_speed", "angle", "height", "label"}.issubset(rows[0])
    assert len({row["label"] for row in rows}) == 4
