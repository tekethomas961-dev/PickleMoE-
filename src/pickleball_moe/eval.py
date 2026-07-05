from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_normalized": confusion_matrix(y_true, y_pred, labels=labels, normalize="true").tolist(),
    }


def expert_statistics(
    y_true: np.ndarray,
    topk_idx: np.ndarray,
    router_probs: np.ndarray,
    class_names: list[str],
    sparse_probs: np.ndarray | None = None,
) -> dict[str, Any]:
    num_classes = len(class_names)
    num_experts = router_probs.shape[1]
    flat_idx = topk_idx.reshape(-1)
    load_counts = np.bincount(flat_idx, minlength=num_experts).astype(float)
    load_freq = load_counts / max(load_counts.sum(), 1.0)
    mean_prob = router_probs.mean(axis=0)
    load_cv = float(load_freq.std() / (load_freq.mean() + 1e-12))
    max_over_mean = float(load_freq.max() / (load_freq.mean() + 1e-12))

    class_to_expert = np.zeros((num_classes, num_experts), dtype=float)
    for c in range(num_classes):
        mask = y_true == c
        denom = max(int(mask.sum()), 1)
        for e in range(num_experts):
            class_to_expert[c, e] = np.any(topk_idx[mask] == e, axis=1).sum() / denom

    expert_to_class = np.zeros((num_experts, num_classes), dtype=float)
    for e in range(num_experts):
        selected_rows = np.any(topk_idx == e, axis=1)
        denom = max(int(selected_rows.sum()), 1)
        for c in range(num_classes):
            expert_to_class[e, c] = ((y_true == c) & selected_rows).sum() / denom

    mean_gate_by_class = np.zeros((num_classes, num_experts), dtype=float)
    mean_selected_gate_by_class = np.zeros((num_classes, num_experts), dtype=float)
    for c in range(num_classes):
        mask = y_true == c
        if int(mask.sum()) > 0:
            mean_gate_by_class[c] = router_probs[mask].mean(axis=0)
            if sparse_probs is not None:
                mean_selected_gate_by_class[c] = sparse_probs[mask].mean(axis=0)

    entropy = -np.sum(router_probs * np.log(router_probs + 1e-12), axis=1)
    return {
        "load_counts": load_counts.tolist(),
        "load_frequency": load_freq.tolist(),
        "mean_router_probability": mean_prob.tolist(),
        "load_cv": load_cv,
        "max_load_over_mean": max_over_mean,
        "class_to_expert": class_to_expert.tolist(),
        "expert_to_class": expert_to_class.tolist(),
        "mean_gate_by_class": mean_gate_by_class.tolist(),
        "mean_selected_gate_by_class": mean_selected_gate_by_class.tolist(),
        "gate_entropy_mean": float(entropy.mean()),
        "gate_entropy_std": float(entropy.std()),
    }


def write_predictions(
    path: str | Path,
    raw_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row_index", "y_true", "y_pred", "true_label", "pred_label"])
        writer.writeheader()
        for idx, yt, yp in zip(raw_indices, y_true, y_pred, strict=False):
            writer.writerow(
                {
                    "row_index": int(idx),
                    "y_true": int(yt),
                    "y_pred": int(yp),
                    "true_label": class_names[int(yt)],
                    "pred_label": class_names[int(yp)],
                }
            )


def write_matrix_csv(path: str | Path, matrix: np.ndarray, row_labels: list[str], col_labels: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([""] + col_labels)
        for label, row in zip(row_labels, matrix, strict=False):
            writer.writerow([label] + [f"{float(value):.6f}" for value in row])
