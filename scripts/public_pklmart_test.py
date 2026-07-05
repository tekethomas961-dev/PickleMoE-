from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from zipfile import ZipFile

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
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_TYPES = {
    "D": "Dink",
    "HB": "Hand Battle",
    "tsDrp": "3rd Shot Drop",
    "tsDrv": "3rd Shot Drive",
}

NUMERIC_FEATURES = [
    "shot_nbr",
    "loc_x",
    "loc_y",
    "next_loc_x",
    "next_loc_y",
    "delta_x",
    "delta_y",
    "shot_distance",
    "shot_angle",
    "start_dist_to_center",
    "end_dist_to_center",
    "start_dist_to_nvz",
    "end_dist_to_nvz",
]

CATEGORICAL_FEATURES = [
    "skill_lvl",
    "scoring_type",
    "ball_type",
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


def read_zip_csv(zip_path: Path, name: str, **kwargs) -> pd.DataFrame:
    with ZipFile(zip_path) as z:
        with z.open(name) as f:
            return pd.read_csv(f, **kwargs)


def build_dataset(zip_path: Path, samples_per_class: int, seed: int) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    shot = read_zip_csv(zip_path, "shot.csv")
    rally = read_zip_csv(
        zip_path,
        "rally.csv",
        usecols=["rally_id", "game_id", "match_id"],
    )
    game = read_zip_csv(
        zip_path,
        "game.csv",
        usecols=["game_id", "skill_lvl", "scoring_type", "ball_type"],
    )

    df = shot[shot["shot_type"].isin(TARGET_TYPES)].copy()
    df = df.dropna(subset=["loc_x", "loc_y", "next_loc_x", "next_loc_y", "shot_nbr"])
    df = df.merge(rally, on="rally_id", how="left").merge(game, on="game_id", how="left")
    df["label"] = df["shot_type"].map(TARGET_TYPES)

    df["delta_x"] = df["next_loc_x"] - df["loc_x"]
    df["delta_y"] = df["next_loc_y"] - df["loc_y"]
    df["shot_distance"] = np.sqrt(df["delta_x"] ** 2 + df["delta_y"] ** 2)
    df["shot_angle"] = np.degrees(np.arctan2(df["delta_y"], df["delta_x"]))
    df["start_dist_to_center"] = np.abs(df["loc_x"] - 10.0)
    df["end_dist_to_center"] = np.abs(df["next_loc_x"] - 10.0)
    df["start_dist_to_nvz"] = np.abs(df["loc_y"] - 7.0)
    df["end_dist_to_nvz"] = np.abs(df["next_loc_y"] - 7.0)

    balanced = []
    for label, group in df.groupby("label", sort=True):
        balanced.append(group.sample(n=min(samples_per_class, len(group)), random_state=seed))
    data = pd.concat(balanced, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    metadata = {
        "zip_path": str(zip_path),
        "target_types": TARGET_TYPES,
        "raw_target_counts": df["label"].value_counts().to_dict(),
        "balanced_counts": data["label"].value_counts().to_dict(),
        "feature_columns": [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES],
    }
    x = data[[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]
    y = data["label"]
    groups = data["match_id"].fillna(data["rally_id"]).astype(str)
    return x, y, groups, metadata


def split_data(x: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int):
    if groups.nunique() >= 10:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_val_idx, test_idx = next(splitter.split(x, y, groups=groups))
        groups_train_val = groups.iloc[train_val_idx]
        val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
        train_local, val_local = next(
            val_splitter.split(x.iloc[train_val_idx], y.iloc[train_val_idx], groups=groups_train_val)
        )
        train_idx = train_val_idx[train_local]
        val_idx = train_val_idx[val_local]
        return train_idx, val_idx, test_idx, "GroupShuffleSplit(match_id)"

    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=seed, stratify=y)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.125, random_state=seed, stratify=y.iloc[train_val_idx]
    )
    return train_idx, val_idx, test_idx, "stratified train_test_split"


def build_model(name: str) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    if name == "logreg":
        clf = LogisticRegression(max_iter=700, class_weight="balanced", random_state=42)
    elif name == "rf":
        clf = RandomForestClassifier(
            n_estimators=260,
            max_depth=16,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("preprocess", preprocess), ("model", clf)])


def evaluate(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, labels: list[str]) -> dict:
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
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    disp = ConfusionMatrixDisplay(np.array(cm), display_labels=labels)
    disp.plot(ax=ax, cmap="Greens", values_format=".2f", colorbar=True)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_summary(rows: list[dict], out_dir: Path) -> None:
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# PKLMart Pickleball Public Test Summary",
        "",
        "| Model | Accuracy | Macro F1 | Balanced Acc | Train | Val | Test |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['balanced_accuracy']:.4f} | {row['train_size']} | {row['val_size']} | {row['test_size']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a non-PyTorch test on the downloaded Kaggle PKLMart pickleball dataset."
    )
    parser.add_argument("--zip", default=r"C:\Users\13342\Downloads\archive (1).zip", help="Downloaded Kaggle zip path.")
    parser.add_argument("--out-dir", default="public_data_test/pklmart_pickleball")
    parser.add_argument("--samples-per-class", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_chinese_font()
    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise FileNotFoundError(f"Kaggle zip not found: {zip_path}")

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    x, y, groups, metadata = build_dataset(zip_path, args.samples_per_class, args.seed)
    train_idx, val_idx, test_idx, split_method = split_data(x, y, groups, args.seed)
    labels = sorted(y.unique().tolist())

    x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
    x_val, y_val = x.iloc[val_idx], y.iloc[val_idx]
    x_test, y_test = x.iloc[test_idx], y.iloc[test_idx]

    full_metrics = {
        **metadata,
        "method": "scikit-learn Pipeline, non-PyTorch",
        "split_method": split_method,
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "labels": labels,
        "models": {},
    }
    rows = []

    for model_name in ["logreg", "rf"]:
        model = build_model(model_name)
        model.fit(x_train, y_train)
        val_metrics = evaluate(model, x_val, y_val, labels)
        test_metrics = evaluate(model, x_test, y_test, labels)
        full_metrics["models"][model_name] = {"val": val_metrics, "test": test_metrics}
        plot_confusion(
            test_metrics["confusion_matrix_normalized"],
            labels,
            f"PKLMart pickleball public test - {model_name}",
            fig_dir / f"{model_name}_confusion_matrix.png",
        )
        rows.append(
            {
                "model": model_name,
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "balanced_accuracy": test_metrics["balanced_accuracy"],
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "test_size": len(test_idx),
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
