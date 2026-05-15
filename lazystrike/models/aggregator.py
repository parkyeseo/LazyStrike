from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class TopKChannelAggregator(nn.Module):
    """LaSt-ViT Eq. (6),(7): per-channel top-K patch averaging."""

    def __init__(self, K: int = 98):
        super().__init__()
        self.K = int(K)

    def forward(self, x_patch: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        batch, num_patches, _dim = x_patch.shape
        k = min(self.K, num_patches)
        _, idx = torch.topk(scores, k=k, dim=1, largest=True)
        gathered = torch.gather(x_patch, dim=1, index=idx)
        return gathered.mean(dim=1)

    def vote_count(self, scores: torch.Tensor) -> torch.Tensor:
        batch, num_patches, _dim = scores.shape
        k = min(self.K, num_patches)
        _, idx = torch.topk(scores, k=k, dim=1, largest=True)
        votes = torch.zeros(batch, num_patches, dtype=torch.long, device=scores.device)
        flat_idx = idx.reshape(batch, -1)
        ones = torch.ones_like(flat_idx, dtype=torch.long)
        votes.scatter_add_(dim=1, index=flat_idx, src=ones)
        return votes


class TopKPatchAggregator(nn.Module):
    """Patch-level top-K averaging for E3 ablation."""

    def __init__(self, K: int = 98, channel_aggregator: str = "mean"):
        super().__init__()
        if channel_aggregator not in {"mean", "max"}:
            raise ValueError("channel_aggregator must be 'mean' or 'max'")
        self.K = int(K)
        self.reduce = channel_aggregator

    def forward(self, x_patch: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        _batch, num_patches, dim = x_patch.shape
        k = min(self.K, num_patches)
        if self.reduce == "mean":
            patch_scores = scores.mean(dim=-1)
        else:
            patch_scores = scores.amax(dim=-1)
        _, idx = torch.topk(patch_scores, k=k, dim=1, largest=True)
        idx = idx.unsqueeze(-1).expand(-1, -1, dim)
        gathered = torch.gather(x_patch, dim=1, index=idx)
        return gathered.mean(dim=1)


AGGREGATOR_REGISTRY: dict[str, type[nn.Module]] = {
    "topk_channel": TopKChannelAggregator,
    "topk_patch": TopKPatchAggregator,
}


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return cfg.get(key, default)


def build_aggregator(cfg: Any) -> nn.Module:
    name = _cfg_get(cfg, "name")
    if name not in AGGREGATOR_REGISTRY:
        raise KeyError(f"Unknown aggregator {name!r}. Choices: {sorted(AGGREGATOR_REGISTRY)}")
    kwargs = dict(_cfg_get(cfg, "kwargs", {}) or {})
    return AGGREGATOR_REGISTRY[name](**kwargs)

