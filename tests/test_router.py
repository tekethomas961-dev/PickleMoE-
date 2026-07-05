from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pickleball_moe.models.moe import switch_style_load_balance_loss, topk_router


def test_topk_router_has_sparse_normalized_rows() -> None:
    logits = torch.tensor([[1.0, 2.0, 0.0, -1.0], [0.5, 0.2, 3.0, 1.0]])
    probs, sparse_probs, topk_idx, topk_w = topk_router(logits, k=2)
    assert probs.shape == (2, 4)
    assert topk_idx.shape == (2, 2)
    assert topk_w.shape == (2, 2)
    assert torch.allclose(sparse_probs.sum(dim=1), torch.ones(2))
    assert torch.equal((sparse_probs > 0).sum(dim=1), torch.full((2,), 2))


def test_load_balance_loss_is_positive() -> None:
    logits = torch.randn(8, 4)
    probs, _, topk_idx, _ = topk_router(logits, k=1)
    loss, stats = switch_style_load_balance_loss(probs, topk_idx, num_experts=4, k=1)
    assert loss.item() > 0
    assert stats["f"].shape == (4,)
