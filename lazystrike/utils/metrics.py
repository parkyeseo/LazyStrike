from __future__ import annotations

import torch


def topk_correct(logits: torch.Tensor, targets: torch.Tensor, ks: tuple[int, ...] = (1, 5)) -> list[int]:
    max_k = min(max(ks), logits.shape[1])
    _, pred = logits.topk(max_k, dim=1)
    correct = pred.eq(targets.unsqueeze(1))
    return [int(correct[:, : min(k, max_k)].any(dim=1).sum().item()) for k in ks]

