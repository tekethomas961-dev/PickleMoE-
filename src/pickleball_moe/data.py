from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class ShotDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    d_in: int
    n_classes: int
    feature_cols: list[str]
    label_map: dict[str, int]
    class_names: list[str]
    class_weights: np.ndarray
    split_sizes: dict[str, int]
    x_test_raw_index: np.ndarray


def _encode_labels(series: pd.Series, label_names: list[str] | None = None) -> tuple[np.ndarray, dict[str, int], list[str]]:
    if label_names:
        class_names = [str(name) for name in label_names]
        numeric = pd.to_numeric(series, errors="coerce")
        if not numeric.isna().any():
            y = numeric.astype(int).to_numpy(dtype=np.int64)
            if y.min(initial=0) < 0 or y.max(initial=0) >= len(class_names):
                raise ValueError("Numeric labels must be in the range covered by data.label_names")
            return y, {name: i for i, name in enumerate(class_names)}, class_names

    values = series.astype(str)
    class_names = [str(name) for name in label_names] if label_names else sorted(values.dropna().unique().tolist())
    label_map = {name: i for i, name in enumerate(class_names)}
    y = values.map(label_map)
    if y.isna().any():
        unknown = sorted(values[y.isna()].unique().tolist())
        raise ValueError(f"Found labels not present in data.label_names: {unknown}")
    return y.to_numpy(dtype=np.int64), label_map, class_names


def _infer_feature_cols(df: pd.DataFrame, label_col: str, group_col: str | None) -> list[str]:
    excluded = {label_col}
    if group_col:
        excluded.add(group_col)
    candidates = [c for c in df.columns if c not in excluded]
    numeric_cols = df[candidates].select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric feature columns found. Pass --feature-cols explicitly.")
    return numeric_cols


def _safe_train_test_split(
    indices: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    stratify_y = y if stratify else None
    try:
        return train_test_split(indices, test_size=test_size, random_state=seed, stratify=stratify_y)
    except ValueError:
        return train_test_split(indices, test_size=test_size, random_state=seed, stratify=None)


def _split_indices(
    y: np.ndarray,
    val_size: float,
    test_size: float,
    seed: int,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    if groups is not None:
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_idx, test_idx = next(gss_test.split(indices, y, groups=groups))
        val_ratio = val_size / max(1.0 - test_size, 1e-9)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_local, val_local = next(gss_val.split(train_val_idx, y[train_val_idx], groups=groups[train_val_idx]))
        return train_val_idx[train_local], train_val_idx[val_local], test_idx

    train_val_idx, test_idx = _safe_train_test_split(indices, y, test_size=test_size, seed=seed)
    val_ratio = val_size / max(1.0 - test_size, 1e-9)
    train_idx, val_idx = _safe_train_test_split(
        train_val_idx,
        y[train_val_idx],
        test_size=val_ratio,
        seed=seed,
    )
    return train_idx, val_idx, test_idx


def _balanced_class_weights(y_train: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = y_train.shape[0] / (n_classes * counts)
    return weights.astype(np.float32)


def build_data_bundle(config: dict[str, Any], output_dir: str | Path | None = None) -> DataBundle:
    data_cfg = config["data"]
    train_cfg = config["training"]

    csv_path = Path(data_cfg["csv_path"])
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    label_col = data_cfg["label_col"]
    group_col = data_cfg.get("group_col")
    feature_cols = data_cfg.get("feature_cols") or _infer_feature_cols(df, label_col, group_col)

    missing = [c for c in [label_col, *feature_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    y, label_map, class_names = _encode_labels(df[label_col], data_cfg.get("label_names"))
    n_classes = len(class_names)
    groups = df[group_col].to_numpy() if group_col else None
    train_idx, val_idx, test_idx = _split_indices(
        y,
        val_size=float(data_cfg.get("val_size", 0.15)),
        test_size=float(data_cfg.get("test_size", 0.15)),
        seed=int(train_cfg.get("seed", 42)),
        groups=groups,
    )

    x_df = df[feature_cols].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    x_train = scaler.fit_transform(imputer.fit_transform(x_df.iloc[train_idx]))
    x_val = scaler.transform(imputer.transform(x_df.iloc[val_idx]))
    x_test = scaler.transform(imputer.transform(x_df.iloc[test_idx]))

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    batch_size = int(data_cfg.get("batch_size", 128))
    num_workers = int(data_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        ShotDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        ShotDataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        ShotDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"imputer": imputer, "scaler": scaler, "feature_cols": feature_cols}, output_dir / "preprocessor.joblib")

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        d_in=len(feature_cols),
        n_classes=n_classes,
        feature_cols=feature_cols,
        label_map=label_map,
        class_names=class_names,
        class_weights=_balanced_class_weights(y_train, n_classes),
        split_sizes={"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        x_test_raw_index=test_idx,
    )
