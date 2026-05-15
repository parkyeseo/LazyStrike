from __future__ import annotations

import torch

from lazystrike.utils.metrics import topk_correct


@torch.no_grad()
def evaluate_classification(model, loader, device) -> tuple[float, float, int]:
    model.eval()
    correct1 = correct5 = total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        c1, c5 = topk_correct(logits, targets, (1, 5))
        correct1 += c1
        correct5 += c5
        total += int(targets.numel())
    return correct1 / max(total, 1), correct5 / max(total, 1), total

