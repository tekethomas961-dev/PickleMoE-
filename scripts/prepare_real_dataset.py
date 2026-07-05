from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REQUIRED_FEATURES = [
    "stance_x",
    "stance_y",
    "swing_speed",
    "ball_speed",
    "landing_x",
    "landing_y",
    "angle",
    "height",
]
OUTPUT_COLUMNS = [*REQUIRED_FEATURES, "label"]


def parse_mapping(values: list[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Mapping must use target=source format, got: {value}")
        target, source = value.split("=", 1)
        target = target.strip()
        source = source.strip()
        if not target or not source:
            raise ValueError(f"Invalid empty mapping entry: {value}")
        mapping[target] = source
    return mapping


def parse_label_map(value: str | None) -> dict[str, int] | None:
    if not value:
        return None
    result: dict[str, int] = {}
    for item in value.split(","):
        if "=" not in item:
            raise ValueError(f"Label map must use class_name=index format, got: {item}")
        name, idx = item.split("=", 1)
        result[name.strip()] = int(idx.strip())
    return result


def normalize_labels(series: pd.Series, label_map: dict[str, int] | None) -> tuple[pd.Series, dict[str, int]]:
    numeric = pd.to_numeric(series, errors="coerce")
    if label_map is None and not numeric.isna().any():
        labels = numeric.astype(int)
        discovered = {str(i): int(i) for i in sorted(labels.unique())}
        return labels, discovered

    values = series.astype(str).str.strip()
    if label_map is None:
        classes = sorted(values.dropna().unique().tolist())
        if len(classes) != 4:
            raise ValueError(
                f"Expected 4 classes when --label-map is omitted, found {len(classes)}: {classes}. "
                "Pass --label-map to choose the four target classes explicitly."
            )
        label_map = {name: i for i, name in enumerate(classes)}

    labels = values.map(label_map)
    if labels.isna().any():
        unknown = sorted(values[labels.isna()].unique().tolist())
        raise ValueError(f"Found labels not covered by --label-map: {unknown}")
    return labels.astype(int), label_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a real CSV dataset into the 8-feature pickleball MoE training format."
    )
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--out", default="data/real_pickleball.csv", help="Output normalized CSV path.")
    parser.add_argument("--label-col", help="Source label column. Defaults to mapped label source or 'label'.")
    parser.add_argument(
        "--map",
        nargs="*",
        help=(
            "Column mappings in target=source format. Targets are stance_x, stance_y, swing_speed, "
            "ball_speed, landing_x, landing_y, angle, height, and optionally label."
        ),
    )
    parser.add_argument(
        "--label-map",
        help="Optional class mapping, e.g. forehand=0,backhand=1,dink=2,overhead=3.",
    )
    parser.add_argument("--limit", type=int, help="Optional row limit for quick smoke tests.")
    parser.add_argument("--show-columns", action="store_true", help="Print input columns and exit.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if args.show_columns:
        print("\n".join(df.columns))
        return

    mapping = parse_mapping(args.map)
    label_col = args.label_col or mapping.get("label") or "label"

    output = pd.DataFrame()
    missing: list[str] = []
    for target in REQUIRED_FEATURES:
        source = mapping.get(target, target)
        if source not in df.columns:
            missing.append(f"{target}<-{source}")
            continue
        output[target] = pd.to_numeric(df[source], errors="coerce")

    if label_col not in df.columns:
        missing.append(f"label<-{label_col}")
    if missing:
        raise ValueError(
            "Missing required source columns: "
            + ", ".join(missing)
            + ". Use --show-columns and --map target=source to align your dataset."
        )

    labels, label_map = normalize_labels(df[label_col], parse_label_map(args.label_map))
    output["label"] = labels
    output = output.dropna(subset=OUTPUT_COLUMNS)

    if args.limit:
        output = output.head(args.limit)

    classes = sorted(output["label"].unique().tolist())
    if classes != [0, 1, 2, 3]:
        raise ValueError(f"Normalized labels must cover exactly [0, 1, 2, 3], got: {classes}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False, encoding="utf-8")

    label_map_path = out_path.with_suffix(".label_map.json")
    label_map_path.write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {out_path} with shape {output.shape}")
    print(f"Wrote {label_map_path}")
    print(output["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
