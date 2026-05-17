from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

DEFAULT_IMAGE_SIZE = 224
DEFAULT_PATCH_SIZE = 16
DEFAULT_RESIZE_SHORT = 256


@dataclass(frozen=True)
class PiBResult:
    score_method: str
    pib: float
    hits: int
    total: int
    empty_patch_sets: int
    image_size: int
    patch_size: int
    resize_short: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bbox_to_patch_set(
    bbox_xyxy: torch.Tensor | list[float] | tuple[float, float, float, float],
    original_width: int,
    original_height: int,
    image_size: int = DEFAULT_IMAGE_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    resize_short: int = DEFAULT_RESIZE_SHORT,
) -> set[int]:
    """Map an original-image bbox to post-transform patch indices.

    This mirrors the official LAST-ViT patch-bbox evaluation: resize the shorter
    image side to 256, center-crop 224x224, then count every patch touched by
    the transformed bbox.
    """
    if isinstance(bbox_xyxy, torch.Tensor):
        xmin, ymin, xmax, ymax = bbox_xyxy.tolist()
    else:
        xmin, ymin, xmax, ymax = bbox_xyxy

    scale = float(resize_short) / float(min(original_width, original_height))
    resized_width = int(original_width * scale)
    resized_height = int(original_height * scale)
    crop_left = (resized_width - image_size) // 2
    crop_top = (resized_height - image_size) // 2

    xf_min = max(0.0, min(float(image_size), xmin * scale - crop_left))
    yf_min = max(0.0, min(float(image_size), ymin * scale - crop_top))
    xf_max = max(0.0, min(float(image_size), xmax * scale - crop_left))
    yf_max = max(0.0, min(float(image_size), ymax * scale - crop_top))

    grid = image_size // patch_size
    indices: set[int] = set()
    y_start = int(yf_min // patch_size)
    y_end = min(int(yf_max // patch_size) + 1, grid)
    x_start = int(xf_min // patch_size)
    x_end = min(int(xf_max // patch_size) + 1, grid)
    for py in range(y_start, y_end):
        for px in range(x_start, x_end):
            indices.add(py * grid + px)
    return indices


def shi_patch_score(patches: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
    p_norm = torch.nn.functional.normalize(patches, dim=-1)
    c_norm = torch.nn.functional.normalize(cls.unsqueeze(1), dim=-1)
    return (p_norm * c_norm).sum(dim=-1)


def patch_scores_from_model_output(
    model,
    patches: torch.Tensor,
    scores: torch.Tensor | None,
    cls: torch.Tensor,
    score_method: str,
) -> torch.Tensor:
    if score_method == "patch_score_shi":
        return shi_patch_score(patches, cls)
    if score_method == "raw_score":
        if scores is None:
            raise RuntimeError("raw_score requires a stability score model")
        return scores.mean(dim=-1)
    if score_method == "vote_count":
        if scores is None or not hasattr(model.aggregator, "vote_count"):
            raise RuntimeError("vote_count requires TopKChannelAggregator")
        return model.aggregator.vote_count(scores).float()
    raise ValueError(f"Unknown PiB score method: {score_method}")


@torch.no_grad()
def evaluate_pib(
    model,
    dataloader,
    device: torch.device,
    score_method: str = "patch_score_shi",
    image_size: int = DEFAULT_IMAGE_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    resize_short: int = DEFAULT_RESIZE_SHORT,
) -> PiBResult:
    model.eval()
    hits = total = empty_patch_sets = 0
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        bboxes = batch["bbox"]
        widths = batch["image_width"]
        heights = batch["image_height"]

        patches, scores, cls, _logits = model.forward_with_scores(images)
        patch_scores = patch_scores_from_model_output(model, patches, scores, cls, score_method)
        top1 = patch_scores.argmax(dim=1).cpu()

        for i in range(int(top1.numel())):
            patch_set = bbox_to_patch_set(
                bboxes[i],
                int(widths[i].item() if isinstance(widths[i], torch.Tensor) else widths[i]),
                int(heights[i].item() if isinstance(heights[i], torch.Tensor) else heights[i]),
                image_size=image_size,
                patch_size=patch_size,
                resize_short=resize_short,
            )
            if not patch_set:
                empty_patch_sets += 1
            hits += int(int(top1[i].item()) in patch_set)
            total += 1

    return PiBResult(
        score_method=score_method,
        pib=hits / max(total, 1),
        hits=hits,
        total=total,
        empty_patch_sets=empty_patch_sets,
        image_size=image_size,
        patch_size=patch_size,
        resize_short=resize_short,
    )
