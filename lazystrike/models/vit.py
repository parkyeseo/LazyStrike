from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn

from .aggregator import build_aggregator
from .stability_scores import build_score


class ViTWithStabilityScore(nn.Module):
    """ViT backbone with stability-aware CLS replacement."""

    def __init__(
        self,
        backbone_name: str = "vit_small_patch16_224",
        num_classes: int = 100,
        score_cfg: Any = None,
        aggregator_cfg: Any = None,
        vanilla_pool: str = "cls",
        drop_path_rate: float = 0.1,
        pretrained: bool = False,
    ):
        super().__init__()
        if vanilla_pool not in {"cls", "mean"}:
            raise ValueError(f"Unsupported vanilla_pool={vanilla_pool!r}; expected 'cls' or 'mean'")
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=drop_path_rate,
        )
        self.embed_dim = int(self.backbone.embed_dim)
        self.num_patches = int(self.backbone.patch_embed.num_patches)
        self.vanilla_pool = vanilla_pool

        self.score = build_score(score_cfg, dim=self.embed_dim)
        self.aggregator = build_aggregator(aggregator_cfg) if self.score is not None else None

        self.head = nn.Linear(self.embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        feats = self.backbone.forward_features(x)
        if feats.ndim != 3:
            raise RuntimeError(f"Expected sequence features [B, T, D], got {tuple(feats.shape)}")
        if feats.shape[1] == self.num_patches + 1:
            return feats[:, 0, :], feats[:, 1:, :]
        if feats.shape[1] == self.num_patches:
            return None, feats
        raise RuntimeError(
            f"Unexpected backbone output shape {tuple(feats.shape)}; "
            f"expected {self.num_patches} or {self.num_patches + 1} tokens"
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        _cls_token, patches = self.forward_tokens(x)
        return patches

    def aggregate(
        self,
        patches: torch.Tensor,
        cls_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.score is None:
            if self.vanilla_pool == "mean":
                return patches.mean(dim=1), None
            if cls_token is None:
                raise RuntimeError("vanilla_pool='cls' requires a backbone CLS token")
            return cls_token, None
        scores = self.score(patches)
        cls = self.aggregator(patches, scores)
        return cls, scores

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token, patches = self.forward_tokens(x)
        cls, _scores = self.aggregate(patches, cls_token)
        return self.head(cls)

    @torch.no_grad()
    def forward_with_scores(self, x: torch.Tensor):
        cls_token, patches = self.forward_tokens(x)
        cls, scores = self.aggregate(patches, cls_token)
        logits = self.head(cls)
        return patches, scores, cls, logits


def build_model(cfg: Any, num_classes: int | None = None) -> ViTWithStabilityScore:
    n_classes = int(num_classes if num_classes is not None else cfg.data.num_classes)
    return ViTWithStabilityScore(
        backbone_name=cfg.model.backbone,
        num_classes=n_classes,
        score_cfg=cfg.model.score,
        aggregator_cfg=cfg.model.aggregator,
        vanilla_pool=str(cfg.model.get("vanilla_pool", "cls")),
        drop_path_rate=float(cfg.model.drop_path_rate),
        pretrained=bool(cfg.model.pretrained),
    )
