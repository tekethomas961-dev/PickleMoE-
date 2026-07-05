from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class RouterAux:
    router_probs: torch.Tensor
    sparse_probs: torch.Tensor
    topk_idx: torch.Tensor
    topk_w: torch.Tensor
    load_freq: torch.Tensor
    mean_prob: torch.Tensor
    lb_loss: torch.Tensor


class Expert(nn.Module):
    """A small expert MLP that directly outputs class logits."""

    def __init__(self, d_in: int = 64, d_hidden: int = 64, n_classes: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(d_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature(x))


def topk_router(
    router_logits: torch.Tensor,
    k: int,
    noise_logits: torch.Tensor | None = None,
    training: bool = False,
    eps: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Softmax routing followed by sparse top-k renormalization."""
    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape [batch, num_experts]")
    num_experts = router_logits.shape[1]
    if not 1 <= k <= num_experts:
        raise ValueError(f"k must be in [1, {num_experts}], got {k}")

    if training and noise_logits is not None:
        noise_std = F.softplus(noise_logits)
        router_logits = router_logits + torch.randn_like(router_logits) * noise_std

    probs = F.softmax(router_logits, dim=-1)
    topk_vals, topk_idx = torch.topk(probs, k=k, dim=-1)
    topk_w = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + eps)

    sparse_probs = torch.zeros_like(probs)
    sparse_probs.scatter_(1, topk_idx, topk_w)
    return probs, sparse_probs, topk_idx, topk_w


def switch_style_load_balance_loss(
    router_probs: torch.Tensor,
    topk_idx: torch.Tensor,
    num_experts: int,
    k: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Switch-style load-balancing loss with a natural top-k extension."""
    batch_size = router_probs.shape[0]
    flat_idx = topk_idx.reshape(-1)
    counts = torch.zeros(num_experts, device=router_probs.device)
    counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float))
    f = counts / max(batch_size * k, 1)
    p = router_probs.mean(dim=0)
    loss = num_experts * torch.sum(f * p)
    return loss, {"f": f.detach(), "P": p.detach()}


class SparseMoE(nn.Module):
    """Small sparse MoE where only top-k experts produce weighted class logits."""

    def __init__(
        self,
        d_in: int,
        n_classes: int = 4,
        num_experts: int = 4,
        k: int = 1,
        stem_dim: int = 64,
        expert_hidden_dim: int = 64,
        gate_hidden_dim: int = 32,
        dropout: float = 0.1,
        noisy_gating: bool = False,
    ) -> None:
        super().__init__()
        if not 1 <= k <= num_experts:
            raise ValueError("k must be between 1 and num_experts")
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating

        self.stem = nn.Sequential(
            nn.Linear(d_in, stem_dim),
            nn.BatchNorm1d(stem_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(stem_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, num_experts),
        )
        self.noise_gate = nn.Linear(stem_dim, num_experts) if noisy_gating else None
        self.experts = nn.ModuleList(
            [Expert(stem_dim, expert_hidden_dim, n_classes=n_classes, dropout=dropout) for _ in range(num_experts)]
        )
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, RouterAux]:
        h = self.stem(x)
        router_logits = self.gate(h)
        noise_logits = self.noise_gate(h) if self.noise_gate is not None else None
        router_probs, sparse_probs, topk_idx, topk_w = topk_router(
            router_logits,
            k=self.k,
            noise_logits=noise_logits,
            training=self.training,
        )

        combined_logits = torch.zeros(x.shape[0], self.n_classes, device=x.device, dtype=h.dtype)
        for expert_id, expert in enumerate(self.experts):
            selected = (topk_idx == expert_id).any(dim=1)
            if not torch.any(selected):
                continue
            expert_logits = expert(h[selected])
            weights = sparse_probs[selected, expert_id].unsqueeze(-1)
            combined_logits[selected] += weights * expert_logits

        lb_loss, lb_stats = switch_style_load_balance_loss(router_probs, topk_idx, self.num_experts, self.k)
        aux = RouterAux(
            router_probs=router_probs.detach(),
            sparse_probs=sparse_probs.detach(),
            topk_idx=topk_idx.detach(),
            topk_w=topk_w.detach(),
            load_freq=lb_stats["f"],
            mean_prob=lb_stats["P"],
            lb_loss=lb_loss.detach(),
        )
        return combined_logits, lb_loss, aux

    def active_parameters_count(self) -> int:
        shared = list(self.stem.parameters()) + list(self.gate.parameters())
        if self.noise_gate is not None:
            shared += list(self.noise_gate.parameters())
        shared_count = sum(p.numel() for p in shared if p.requires_grad)
        expert_counts = [sum(p.numel() for p in expert.parameters() if p.requires_grad) for expert in self.experts]
        return int(shared_count + sum(expert_counts[: self.k]))
