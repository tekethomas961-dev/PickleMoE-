from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_URL = "https://data.scorenetwork.org/tennis/tennis-shot-level-data.html"
DEFAULT_DATA_URL = "../data/tennis-w-shots-wim.csv.gz"

LABEL_MAP = {
    "groundstroke": "groundstroke",
    "slice": "slice",
    "volley": "volley",
    "overhead": "overhead",
}

FEATURE_COLUMNS = [
    "ShotHand",
    "ShotDirection",
    "ServeDirection",
    "ShotDepth",
    "OutcomeType",
    "ErrorType",
    "Serve",
    "Round",
    "Tournament",
    "Shot",
]


def ensure_chinese_font() -> None:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            from matplotlib import font_manager

            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=str(path)).get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def download_dataset(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = url.rstrip("/").split("/")[-1]
    path = out_dir / filename
    if not path.exists():
        print(f"Downloading {url}")
        urlretrieve(url, path)
    return path


def load_public_data(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        df = pd.read_csv(f)
    if max_rows:
        df = df.head(max_rows)
    return df


def make_dataset(df: pd.DataFrame, samples_per_class: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df = df[df["ShotType"].isin(LABEL_MAP)].copy()
    df["label"] = df["ShotType"].map(LABEL_MAP)
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Public dataset is missing expected columns: {missing}")

    groups = []
    for label, group in df.groupby("label", sort=True):
        groups.append(group.sample(n=min(samples_per_class, len(group)), random_state=seed))
    balanced = pd.concat(groups, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    x = balanced[FEATURE_COLUMNS]
    y = balanced["label"]
    return x, y


def build_model(model_name: str, categorical_cols: list[str], numeric_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
        ]
    )
    if model_name == "logreg":
        clf = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=None, random_state=42)
    elif model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=240,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, labels: list[str]) -> dict[str, object]:
    pred = model.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "classification_report": classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
        "confusion_matrix_normalized": confusion_matrix(y_test, pred, labels=labels, normalize="true").tolist(),
    }


def plot_confusion(cm: list[list[float]], labels: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    disp = ConfusionMatrixDisplay(np.array(cm), display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=True)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_summary(rows: list[dict[str, object]], out_dir: Path) -> None:
    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    lines = [
        "# Public SCORE Tennis Test Summary",
        "",
        "| Model | Accuracy | Macro F1 | Balanced Acc | Train | Val | Test |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['balanced_accuracy']:.4f} | {row['train_size']} | {row['val_size']} | {row['test_size']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a non-PyTorch public-data test on SCORE Grand Slam tennis shot-level data."
    )
    parser.add_argument("--url", default=DEFAULT_DATA_URL, help="Direct .csv.gz URL from SCORE.")
    parser.add_argument("--out-dir", default="public_data_test/score_tennis", help="Output directory.")
    parser.add_argument("--samples-per-class", type=int, default=2500, help="Balanced samples per target class.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, help="Optional row cap after loading the public CSV.")
    args = parser.parse_args()

    ensure_chinese_font()
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data_url = urljoin(BASE_URL, args.url)
    raw_path = download_dataset(data_url, raw_dir)
    raw_df = load_public_data(raw_path, max_rows=args.max_rows)
    x, y = make_dataset(raw_df, samples_per_class=args.samples_per_class, seed=args.seed)

    labels = sorted(y.unique().tolist())
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=0.20, random_state=args.seed, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.125, random_state=args.seed, stratify=y_train_val
    )

    categorical_cols = [col for col in FEATURE_COLUMNS if col != "Shot"]
    numeric_cols = ["Shot"]
    rows: list[dict[str, object]] = []
    full_metrics: dict[str, object] = {
        "source_url": data_url,
        "source_page": BASE_URL,
        "target_labels": labels,
        "feature_columns": FEATURE_COLUMNS,
        "dataset_shape_after_balancing": [int(x.shape[0]), int(x.shape[1])],
        "split_sizes": {"train": len(x_train), "val": len(x_val), "test": len(x_test)},
        "models": {},
    }

    for model_name in ["logreg", "rf"]:
        model = build_model(model_name, categorical_cols, numeric_cols)
        model.fit(x_train, y_train)
        val_metrics = evaluate_model(model, x_val, y_val, labels)
        test_metrics = evaluate_model(model, x_test, y_test, labels)
        full_metrics["models"][model_name] = {"val": val_metrics, "test": test_metrics}
        plot_confusion(
            test_metrics["confusion_matrix_normalized"],
            labels,
            f"SCORE tennis public test - {model_name}",
            fig_dir / f"{model_name}_confusion_matrix.png",
        )
        rows.append(
            {
                "model": model_name,
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "balanced_accuracy": test_metrics["balanced_accuracy"],
                "train_size": len(x_train),
                "val_size": len(x_val),
                "test_size": len(x_test),
            }
        )

    (out_dir / "metrics.json").write_text(json.dumps(full_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(rows, out_dir)
    print(f"Wrote {out_dir / 'summary.md'}")
    print(f"Wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
