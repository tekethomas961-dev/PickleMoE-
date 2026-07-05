from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


DEMO_FEATURES = [
    "stance_x",
    "stance_y",
    "swing_speed",
    "ball_speed",
    "landing_x",
    "landing_y",
    "angle",
    "height",
]

DEMO_LABEL_COL = "label"
DEMO_CLASS_NAMES = ["正手抽球", "反手削球", "网前吊球", "高压球"]


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def generate_demo_rows(n_samples: int = 600, seed: int = 42) -> list[dict[str, object]]:
    """Generate a small physically plausible synthetic pickleball table dataset."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    prototypes = {
        0: {
            "name": "正手抽球",
            "stance_x": 1.15,
            "stance_y": 5.6,
            "swing_speed": 16.0,
            "ball_speed": 34.0,
            "landing_x": 1.0,
            "landing_y": 6.4,
            "angle": 8.0,
            "height": 0.95,
        },
        1: {
            "name": "反手削球",
            "stance_x": -1.15,
            "stance_y": 5.1,
            "swing_speed": 10.5,
            "ball_speed": 22.0,
            "landing_x": -0.85,
            "landing_y": 5.7,
            "angle": -16.0,
            "height": 0.82,
        },
        2: {
            "name": "网前吊球",
            "stance_x": 0.15,
            "stance_y": 2.1,
            "swing_speed": 7.2,
            "ball_speed": 15.5,
            "landing_x": 0.15,
            "landing_y": 2.3,
            "angle": 20.0,
            "height": 0.62,
        },
        3: {
            "name": "高压球",
            "stance_x": 0.35,
            "stance_y": 4.2,
            "swing_speed": 18.5,
            "ball_speed": 39.0,
            "landing_x": 0.3,
            "landing_y": 7.6,
            "angle": -4.0,
            "height": 1.45,
        },
    }

    for i in range(n_samples):
        label = i % len(DEMO_CLASS_NAMES)
        proto = prototypes[label]

        stance_x = _clip(rng.normal(proto["stance_x"], 0.55), -3.1, 3.1)
        stance_y = _clip(rng.normal(proto["stance_y"], 0.75), 0.2, 8.9)
        swing_speed = _clip(rng.normal(proto["swing_speed"], 1.8), 2.0, 25.0)
        ball_speed = _clip(rng.normal(proto["ball_speed"], 3.5), 5.0, 50.0)
        landing_x = _clip(rng.normal(proto["landing_x"] + 0.15 * stance_x, 0.75), -3.1, 3.1)
        landing_y = _clip(rng.normal(proto["landing_y"], 0.9), 0.2, 8.9)
        angle = _clip(rng.normal(proto["angle"], 5.5), -35.0, 45.0)
        height = _clip(rng.normal(proto["height"], 0.12), 0.25, 1.8)

        rows.append(
            {
                "rally_id": i // 6,
                "stance_x": round(stance_x, 4),
                "stance_y": round(stance_y, 4),
                "swing_speed": round(swing_speed, 4),
                "ball_speed": round(ball_speed, 4),
                "landing_x": round(landing_x, 4),
                "landing_y": round(landing_y, 4),
                "angle": round(angle, 4),
                "height": round(height, 4),
                "label": label,
            }
        )

    rng.shuffle(rows)
    return rows


def generate_demo_csv(path: str | Path, n_samples: int = 600, seed: int = 42) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_demo_rows(n_samples=n_samples, seed=seed)
    fieldnames = ["rally_id", *DEMO_FEATURES, DEMO_LABEL_COL]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
