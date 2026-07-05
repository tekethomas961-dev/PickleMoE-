from __future__ import annotations

import argparse
from pathlib import Path

from .demo import DEMO_LABEL_COL, generate_demo_csv
from .train import run_experiment
from .utils import load_yaml


def _parse_args(default_config: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pickleball shot classification experiments.")
    parser.add_argument("--config", default=default_config, help="Path to YAML config.")
    parser.add_argument("--data", help="CSV path. Overrides data.csv_path.")
    parser.add_argument("--label-col", help="Label column. Overrides data.label_col.")
    parser.add_argument("--label-names", nargs="+", help="Class names ordered by numeric label id.")
    parser.add_argument("--feature-cols", nargs="+", help="Feature columns. Defaults to numeric columns.")
    parser.add_argument("--group-col", help="Group column for group-aware split.")
    parser.add_argument("--output-dir", help="Run output directory.")
    parser.add_argument("--demo", action="store_true", help="Generate and use the built-in synthetic demo CSV.")
    parser.add_argument("--demo-samples", type=int, default=800, help="Number of synthetic demo rows.")
    parser.add_argument("--device", help="Device, e.g. cpu, cuda, or auto.")
    parser.add_argument("--k", type=int, help="Override MoE top-k.")
    parser.add_argument("--max-epochs", type=int, help="Override max training epochs.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    return parser.parse_args()


def main(default_config: str) -> None:
    args = _parse_args(default_config)
    config = load_yaml(args.config)

    if args.demo:
        demo_path = generate_demo_csv(
            config.get("data", {}).get("csv_path", "data/demo_pickleball.csv"),
            n_samples=args.demo_samples,
        )
        config.setdefault("data", {})["csv_path"] = str(demo_path)
        config["data"]["label_col"] = DEMO_LABEL_COL

    if args.data:
        config.setdefault("data", {})["csv_path"] = args.data
    if args.label_col:
        config.setdefault("data", {})["label_col"] = args.label_col
    if args.label_names:
        config.setdefault("data", {})["label_names"] = args.label_names
    if args.feature_cols:
        config.setdefault("data", {})["feature_cols"] = args.feature_cols
    if args.group_col:
        config.setdefault("data", {})["group_col"] = args.group_col
    if args.device:
        config.setdefault("training", {})["device"] = args.device
    if args.k is not None:
        config.setdefault("model", {})["k"] = args.k
        config.setdefault("model", {})["type"] = "moe"
    if args.max_epochs is not None:
        config.setdefault("training", {})["max_epochs"] = args.max_epochs
    if args.seed is not None:
        config.setdefault("training", {})["seed"] = args.seed

    run_dir = run_experiment(config, output_dir=Path(args.output_dir) if args.output_dir else None)
    print(f"Experiment finished: {run_dir}")
