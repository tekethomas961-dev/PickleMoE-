from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def _configure_chinese_fonts() -> None:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


_configure_chinese_fonts()


def plot_training_curves(history: list[dict[str, Any]], save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, [row["train_loss"] for row in history], label="train_loss")
    ax1.plot(epochs, [row["val_loss"] for row in history], label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, [row["val_acc"] for row in history], label="val_acc", color="tab:green")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: str | Path, title: str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    save_path: str | Path,
    title: str,
    cmap: str = "magma",
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_gate_entropy(router_probs: np.ndarray, save_path: str | Path) -> None:
    entropy = -np.sum(router_probs * np.log(router_probs + 1e-12), axis=1)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(entropy, bins=24, color="tab:purple", alpha=0.85)
    ax.set_xlabel("Gate entropy")
    ax.set_ylabel("Samples")
    ax.set_title("Gate Entropy Distribution")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
