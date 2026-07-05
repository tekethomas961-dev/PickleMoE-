from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import DataBundle, build_data_bundle
from .eval import classification_metrics, expert_statistics, write_matrix_csv, write_predictions
from .models import DenseMLP, SparseMoE
from .utils import (
    count_parameters,
    ensure_dir,
    resolve_device,
    save_json,
    save_yaml,
    set_seed,
    timestamp,
    write_history_csv,
)
from .viz import plot_confusion_matrix, plot_gate_entropy, plot_heatmap, plot_training_curves


def build_model(config: dict[str, Any], d_in: int, n_classes: int) -> nn.Module:
    model_cfg = config["model"]
    model_type = model_cfg.get("type", "mlp")
    if model_type == "mlp":
        return DenseMLP(
            d_in=d_in,
            n_classes=n_classes,
            hidden_dims=model_cfg.get("hidden_dims", [128, 64, 32]),
            dropout=model_cfg.get("dropout", [0.2, 0.2, 0.1]),
            batch_norm=bool(model_cfg.get("batch_norm", True)),
        )
    if model_type == "moe":
        return SparseMoE(
            d_in=d_in,
            n_classes=n_classes,
            num_experts=int(model_cfg.get("num_experts", 4)),
            k=int(model_cfg.get("k", 1)),
            stem_dim=int(model_cfg.get("stem_dim", 64)),
            expert_hidden_dim=int(model_cfg.get("expert_hidden_dim", 64)),
            gate_hidden_dim=int(model_cfg.get("gate_hidden_dim", 32)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            noisy_gating=bool(model_cfg.get("noisy_gating", False)),
        )
    raise ValueError(f"Unknown model type: {model_type}")


def _criterion(config: dict[str, Any], data: DataBundle, device: str) -> nn.Module:
    class_weight = config["training"].get("class_weight")
    if class_weight == "balanced":
        weights = torch.as_tensor(data.class_weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weights)
    return nn.CrossEntropyLoss()


def _forward_loss(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    criterion: nn.Module,
    lambda_lb: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    out = model(xb)
    if isinstance(out, tuple):
        logits, lb_loss, aux = out
        ce_loss = criterion(logits, yb)
        loss = ce_loss + lambda_lb * lb_loss
        return loss, ce_loss.detach(), lb_loss.detach(), (logits, aux)

    logits = out
    ce_loss = criterion(logits, yb)
    zero = torch.zeros((), device=xb.device)
    return ce_loss, ce_loss.detach(), zero, (logits, None)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    lambda_lb: float,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "ce": 0.0, "lb": 0.0, "correct": 0.0, "n": 0.0}
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, ce_loss, lb_loss, payload = _forward_loss(model, xb, yb, criterion, lambda_lb)
        logits, _ = payload
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = xb.shape[0]
        totals["loss"] += float(loss.detach().item()) * batch_size
        totals["ce"] += float(ce_loss.item()) * batch_size
        totals["lb"] += float(lb_loss.item()) * batch_size
        totals["correct"] += float((logits.argmax(dim=1) == yb).sum().item())
        totals["n"] += batch_size

    n = max(totals["n"], 1.0)
    return {
        "loss": totals["loss"] / n,
        "ce": totals["ce"] / n,
        "lb": totals["lb"] / n,
        "acc": totals["correct"] / n,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
    lambda_lb: float,
    collect_router: bool = False,
) -> dict[str, Any]:
    model.eval()
    totals = {"loss": 0.0, "ce": 0.0, "lb": 0.0, "n": 0.0}
    all_y: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_router: list[np.ndarray] = []
    all_sparse: list[np.ndarray] = []
    all_topk: list[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        loss, ce_loss, lb_loss, payload = _forward_loss(model, xb, yb, criterion, lambda_lb)
        logits, aux = payload
        pred = logits.argmax(dim=1)
        batch_size = xb.shape[0]
        totals["loss"] += float(loss.item()) * batch_size
        totals["ce"] += float(ce_loss.item()) * batch_size
        totals["lb"] += float(lb_loss.item()) * batch_size
        totals["n"] += batch_size
        all_y.append(yb.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        if collect_router and aux is not None:
            all_router.append(aux.router_probs.cpu().numpy())
            all_sparse.append(aux.sparse_probs.cpu().numpy())
            all_topk.append(aux.topk_idx.cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    n = max(totals["n"], 1.0)
    result: dict[str, Any] = {
        "loss": totals["loss"] / n,
        "ce": totals["ce"] / n,
        "lb": totals["lb"] / n,
        "acc": float((y_true == y_pred).mean()),
        "y_true": y_true,
        "y_pred": y_pred,
    }
    if all_router:
        result["router_probs"] = np.concatenate(all_router, axis=0)
        result["sparse_probs"] = np.concatenate(all_sparse, axis=0)
        result["topk_idx"] = np.concatenate(all_topk, axis=0)
    return result


def run_experiment(config: dict[str, Any], output_dir: str | Path | None = None) -> Path:
    train_cfg = config["training"]
    set_seed(int(train_cfg.get("seed", 42)))

    experiment_name = config.get("experiment_name", config["model"].get("type", "experiment"))
    run_dir = ensure_dir(output_dir or Path("runs") / f"{experiment_name}_{timestamp()}")
    figures_dir = ensure_dir(run_dir / "figures")
    save_yaml(config, run_dir / "config.yaml")

    data = build_data_bundle(config, output_dir=run_dir)
    model = build_model(config, d_in=data.d_in, n_classes=data.n_classes)
    device = resolve_device(train_cfg.get("device", "auto"))
    model.to(device)

    criterion = _criterion(config, data, device)
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    optimizer_cls = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    optimizer = optimizer_cls(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_cfg.get("scheduler_factor", 0.5)),
        patience=int(train_cfg.get("scheduler_patience", 5)),
    )

    lambda_lb = float(train_cfg.get("lambda_lb", 0.0))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    patience = int(train_cfg.get("early_stopping_patience", 15))
    max_epochs = int(train_cfg.get("max_epochs", 80))

    best_val = float("inf")
    best_epoch = 0
    wait = 0
    history: list[dict[str, Any]] = []
    best_path = run_dir / "best_model.pt"

    for epoch in range(1, max_epochs + 1):
        train_stats = train_one_epoch(
            model,
            data.train_loader,
            optimizer,
            criterion,
            device,
            lambda_lb=lambda_lb,
            grad_clip=grad_clip,
        )
        val_stats = evaluate(model, data.val_loader, criterion, device, lambda_lb=lambda_lb)
        scheduler.step(val_stats["loss"])

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_ce": train_stats["ce"],
            "train_lb": train_stats["lb"],
            "train_acc": train_stats["acc"],
            "val_loss": val_stats["loss"],
            "val_ce": val_stats["ce"],
            "val_lb": val_stats["lb"],
            "val_acc": val_stats["acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        if val_stats["loss"] < best_val - 1e-6:
            best_val = val_stats["loss"]
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "feature_cols": data.feature_cols,
                    "label_map": data.label_map,
                    "class_names": data.class_names,
                    "best_epoch": best_epoch,
                },
                best_path,
            )
        else:
            wait += 1
            if wait >= patience:
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_stats = evaluate(model, data.test_loader, criterion, device, lambda_lb=lambda_lb, collect_router=True)
    metrics = classification_metrics(test_stats["y_true"], test_stats["y_pred"], data.class_names)

    active_params = model.active_parameters_count() if hasattr(model, "active_parameters_count") else count_parameters(model)
    summary: dict[str, Any] = {
        "experiment_name": experiment_name,
        "model_type": config["model"].get("type"),
        "best_epoch": best_epoch,
        "device": device,
        "split_sizes": data.split_sizes,
        "feature_cols": data.feature_cols,
        "label_map": data.label_map,
        "total_parameters": count_parameters(model),
        "active_parameters_per_sample": active_params,
        "optimizer": optimizer_name,
        "lambda_lb": lambda_lb,
        "top_k": config["model"].get("k") if config["model"].get("type") == "moe" else None,
        "test_loss": test_stats["loss"],
        **metrics,
    }

    if "router_probs" in test_stats:
        expert_stats = expert_statistics(
            test_stats["y_true"],
            test_stats["topk_idx"],
            test_stats["router_probs"],
            data.class_names,
            sparse_probs=test_stats["sparse_probs"],
        )
        summary["expert_statistics"] = expert_stats
        expert_cols = [f"Expert {i}" for i in range(test_stats["router_probs"].shape[1])]
        plot_heatmap(
            np.asarray(expert_stats["class_to_expert"]),
            data.class_names,
            expert_cols,
            figures_dir / "class_to_expert_heatmap.png",
            "Class to Expert Routing",
        )
        plot_heatmap(
            np.asarray(expert_stats["mean_gate_by_class"]),
            data.class_names,
            expert_cols,
            figures_dir / "mean_gate_by_class_heatmap.png",
            "Mean Gate Probability by Class",
            cmap="plasma",
        )
        plot_heatmap(
            np.asarray(expert_stats["mean_selected_gate_by_class"]),
            data.class_names,
            expert_cols,
            figures_dir / "mean_selected_gate_by_class_heatmap.png",
            "Mean Selected Gate Weight by Class",
            cmap="plasma",
        )
        write_matrix_csv(
            run_dir / "class_to_expert.csv",
            np.asarray(expert_stats["class_to_expert"]),
            data.class_names,
            expert_cols,
        )
        write_matrix_csv(
            run_dir / "mean_gate_by_class.csv",
            np.asarray(expert_stats["mean_gate_by_class"]),
            data.class_names,
            expert_cols,
        )
        write_matrix_csv(
            run_dir / "mean_selected_gate_by_class.csv",
            np.asarray(expert_stats["mean_selected_gate_by_class"]),
            data.class_names,
            expert_cols,
        )
        plot_heatmap(
            np.asarray(expert_stats["expert_to_class"]),
            expert_cols,
            data.class_names,
            figures_dir / "expert_to_class_heatmap.png",
            "Expert to Class Purity",
            cmap="viridis",
        )
        plot_gate_entropy(test_stats["router_probs"], figures_dir / "gate_entropy.png")

    write_history_csv(history, run_dir / "history.csv")
    save_json(summary, run_dir / "metrics.json")
    save_json(data.label_map, run_dir / "label_map.json")
    write_predictions(run_dir / "predictions.csv", data.x_test_raw_index, test_stats["y_true"], test_stats["y_pred"], data.class_names)
    plot_training_curves(history, figures_dir / "training_curves.png")
    plot_confusion_matrix(
        np.asarray(metrics["confusion_matrix_normalized"]),
        data.class_names,
        figures_dir / "confusion_matrix_normalized.png",
        "Normalized Confusion Matrix",
    )

    return run_dir
