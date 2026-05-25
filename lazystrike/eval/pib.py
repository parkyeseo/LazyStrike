from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

DEFAULT_IMAGE_SIZE = 224
DEFAULT_PATCH_SIZE = 16
DEFAULT_RESIZE_SHORT = 256


@dataclass(frozen=True)
class PiBResult:
    score_methods: list[str]
    pibs: dict[str, float | None]
    hits_by_method: dict[str, int | None]
    total: int
    empty_patch_sets: int
    image_size: int
    patch_size: int
    resize_short: int

    @property
    def score_method(self) -> str:
        if len(self.score_methods) != 1:
            raise AttributeError("score_method is only defined for single-method PiB results")
        return self.score_methods[0]

    @property
    def pib(self) -> float | None:
        if len(self.score_methods) != 1:
            raise AttributeError("pib is only defined for single-method PiB results")
        return self.pibs[self.score_methods[0]]

    @property
    def hits(self) -> int | None:
        if len(self.score_methods) != 1:
            raise AttributeError("hits is only defined for single-method PiB results")
        return self.hits_by_method[self.score_methods[0]]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "score_methods": self.score_methods,
            "pibs": self.pibs,
            "hits_by_method": self.hits_by_method,
            "total": self.total,
            "empty_patch_sets": self.empty_patch_sets,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "resize_short": self.resize_short,
        }
        if len(self.score_methods) == 1:
            method = self.score_methods[0]
            payload.update(
                {
                    "score_method": method,
                    "pib": self.pibs[method],
                    "hits": self.hits_by_method[method],
                }
            )
        return payload


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
    pooled_cls: torch.Tensor,
    encoder_cls: torch.Tensor | None,
    score_method: str,
) -> torch.Tensor:
    if score_method in {"patch_score_shi", "patch_score_encoder_cls", "encoder_cls"}:
        if encoder_cls is None:
            raise RuntimeError("patch_score_shi requires an encoder CLS token")
        return shi_patch_score(patches, encoder_cls)
    if score_method in {"patch_score_qcls", "patch_score_pooled", "qcls"}:
        return shi_patch_score(patches, pooled_cls)
    if score_method in {"patch_score_mean", "mean_pool", "patch_mean"}:
        return shi_patch_score(patches, patches.mean(dim=1))
    if score_method == "raw_score":
        if scores is None:
            raise RuntimeError("raw_score requires a stability score model")
        return scores.mean(dim=-1)
    if score_method == "vote_count":
        if scores is None or not hasattr(model.aggregator, "vote_count"):
            raise RuntimeError("vote_count requires TopKChannelAggregator")
        return model.aggregator.vote_count(scores).float()
    if score_method == "patch_score_prototype":
        logits = model.head(pooled_cls)
        pred_class = logits.argmax(dim=1)
        prototypes = model.head.weight[pred_class]
        return shi_patch_score(patches, prototypes)
    raise ValueError(f"Unknown PiB score method: {score_method}")


@torch.no_grad()
def evaluate_pib(
    model,
    dataloader,
    device: torch.device,
    score_method: str | None = None,
    score_methods: list[str] | tuple[str, ...] | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    resize_short: int = DEFAULT_RESIZE_SHORT,
) -> PiBResult:
    if score_method is not None and score_methods is not None:
        raise ValueError("Use either score_method or score_methods, not both")
    if score_methods is None:
        score_methods = [score_method or "patch_score_qcls"]
    score_methods = list(score_methods)
    if not score_methods:
        raise ValueError("At least one score method is required")

    model.eval()
    hits_by_method: dict[str, int | None] = {method: 0 for method in score_methods}
    valid_methods = set(score_methods)
    total = empty_patch_sets = 0
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        bboxes = batch["bbox"]
        widths = batch["image_width"]
        heights = batch["image_height"]

        encoder_cls, patches = model.forward_tokens(images)
        pooled_cls, scores = model.aggregate(patches, encoder_cls)

        batch_patch_sets = []
        for i in range(int(images.shape[0])):
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
            batch_patch_sets.append(patch_set)

        for method in list(valid_methods):
            try:
                patch_scores = patch_scores_from_model_output(
                    model,
                    patches,
                    scores,
                    pooled_cls,
                    encoder_cls,
                    method,
                )
            except RuntimeError:
                valid_methods.remove(method)
                hits_by_method[method] = None
                continue

            top1 = patch_scores.argmax(dim=1).cpu()
            for i in range(int(top1.numel())):
                assert hits_by_method[method] is not None
                hits_by_method[method] += int(int(top1[i].item()) in batch_patch_sets[i])

        total += int(images.shape[0])

    pibs: dict[str, float | None] = {}
    for method in score_methods:
        hits = hits_by_method[method]
        pibs[method] = None if hits is None else hits / max(total, 1)

    return PiBResult(
        score_methods=score_methods,
        pibs=pibs,
        hits_by_method=hits_by_method,
        total=total,
        empty_patch_sets=empty_patch_sets,
        image_size=image_size,
        patch_size=patch_size,
        resize_short=resize_short,
    )
