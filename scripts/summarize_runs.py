from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "run",
    "model_type",
    "top_k",
    "lambda_lb",
    "optimizer",
    "best_epoch",
    "total_parameters",
    "active_parameters_per_sample",
    "test_loss",
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "load_cv",
    "max_load_over_mean",
    "gate_entropy_mean",
]


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def read_run(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expert = metrics.get("expert_statistics", {})
    row = {
        "run": run_dir.name,
        "model_type": metrics.get("model_type"),
        "top_k": metrics.get("top_k"),
        "lambda_lb": metrics.get("lambda_lb"),
        "optimizer": metrics.get("optimizer"),
        "best_epoch": metrics.get("best_epoch"),
        "total_parameters": metrics.get("total_parameters"),
        "active_parameters_per_sample": metrics.get("active_parameters_per_sample"),
        "test_loss": metrics.get("test_loss"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "load_cv": expert.get("load_cv"),
        "max_load_over_mean": expert.get("max_load_over_mean"),
        "gate_entropy_mean": expert.get("gate_entropy_mean"),
    }
    return {k: _round(v) for k, v in row.items()}


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment Summary",
        "",
        "| Run | Model | Best Epoch | Params | Active Params | Test Loss | Acc | Macro F1 | Balanced Acc | CV(load) | Max/Mean | Gate Entropy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        model_label = row["model_type"]
        if row.get("top_k") not in (None, ""):
            model_label = f"{model_label}-K={row['top_k']}, lambda_lb={row['lambda_lb']}"
        lines.append(
            "| {run} | {model_label} | {best_epoch} | {total_parameters} | {active_parameters_per_sample} | "
            "{test_loss} | {accuracy} | {macro_f1} | {balanced_accuracy} | {load_cv} | "
            "{max_load_over_mean} | {gate_entropy_mean} |".format(
                model_label=model_label,
                **{key: "" if value is None else value for key, value in row.items()},
            )
        )
    lines.append("")
    lines.append("Figures are stored under each run directory's `figures/` folder.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment metrics into CSV and Markdown.")
    parser.add_argument("runs", nargs="+", help="Run directories containing metrics.json.")
    parser.add_argument("--out-dir", default="runs", help="Output directory for summary files.")
    args = parser.parse_args()

    rows = [read_run(Path(run)) for run in args.runs]
    out_dir = Path(args.out_dir)
    write_csv(rows, out_dir / "summary.csv")
    write_markdown(rows, out_dir / "summary.md")
    print(f"Wrote {out_dir / 'summary.csv'}")
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
