from __future__ import annotations

from pathlib import Path
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
        dynamic_img_size: bool = False,
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
            dynamic_img_size=dynamic_img_size,
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
        patch_size = getattr(self.backbone.patch_embed, "patch_size", None)
        if isinstance(patch_size, tuple):
            expected_patches = (x.shape[-2] // int(patch_size[0])) * (x.shape[-1] // int(patch_size[1]))
        elif patch_size is not None:
            expected_patches = (x.shape[-2] // int(patch_size)) * (x.shape[-1] // int(patch_size))
        else:
            expected_patches = self.num_patches
        if feats.shape[1] == expected_patches + 1:
            return feats[:, 0, :], feats[:, 1:, :]
        if feats.shape[1] == expected_patches:
            return None, feats
        raise RuntimeError(
            f"Unexpected backbone output shape {tuple(feats.shape)}; "
            f"expected {expected_patches} or {expected_patches + 1} tokens"
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


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return cfg.get(key, default)


def _read_wnids(path: str | Path) -> list[str]:
    wnids = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(set(wnids)) != len(wnids):
        raise RuntimeError(f"Duplicate wnids found in {path}")
    return wnids


def _imagenet_wnids_from_root(root: str | Path) -> list[str] | None:
    root = Path(root)
    for split in ("val", "train"):
        split_root = root / split
        if split_root.is_dir():
            wnids = sorted(p.name for p in split_root.iterdir() if p.is_dir())
            if wnids:
                return wnids
    return None


def _source_imagenet1k_wnids(cfg: Any) -> list[str]:
    model_cfg = cfg.model
    explicit_classes = _cfg_get(model_cfg, "pretrained_head_source_classes", None)
    if explicit_classes:
        wnids = _read_wnids(explicit_classes)
    else:
        imagenet_full = _cfg_get(getattr(cfg, "paths", None), "imagenet_full", None)
        wnids = _imagenet_wnids_from_root(imagenet_full) if imagenet_full else None
        if wnids is None:
            raise RuntimeError(
                "pretrained_head slicing requires model.pretrained_head_source_classes "
                "or paths.imagenet_full with train/val class directories"
            )
    if len(wnids) != 1000:
        raise RuntimeError(f"Expected 1000 ImageNet-1K source wnids, got {len(wnids)}")
    return wnids


def _target_wnids_for_head(cfg: Any) -> list[str]:
    model_cfg = cfg.model
    target_classes = _cfg_get(model_cfg, "pretrained_head_target_classes", None)
    if target_classes is None:
        target_classes = _cfg_get(getattr(cfg, "paths", None), "imagenet100_classes", None)
    if target_classes is None:
        raise RuntimeError(
            "Sliced pretrained head requires model.pretrained_head_target_classes "
            "or paths.imagenet100_classes"
        )
    wnids = _read_wnids(target_classes)
    if bool(_cfg_get(model_cfg, "pretrained_head_sort_target", True)):
        wnids = sorted(wnids)
    return wnids


def _load_init_checkpoint(model: ViTWithStabilityScore, init_ckpt: str | Path) -> None:
    state = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload from {init_ckpt}: {type(state)!r}")

    normalized = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key.startswith(("backbone.", "head.", "score.", "aggregator.")):
            normalized[key] = value
        else:
            normalized[f"backbone.{key}"] = value

    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in normalized.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            filtered[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(
        f"Loaded init_ckpt={init_ckpt} "
        f"(loaded={len(filtered)}, skipped_shape_or_missing={len(skipped)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)})"
    )


def _copy_pretrained_head(model: ViTWithStabilityScore, cfg: Any, num_classes: int) -> None:
    source = timm.create_model(
        cfg.model.backbone,
        pretrained=True,
        num_classes=1000,
        global_pool="",
    )
    source_head = getattr(source, "head", None)
    if source_head is None or not hasattr(source_head, "weight"):
        raise RuntimeError(f"Could not find a linear head on pretrained {cfg.model.backbone}")
    weight = source_head.weight.detach()
    bias = source_head.bias.detach() if source_head.bias is not None else None
    if weight.shape[1] != model.embed_dim:
        raise RuntimeError(f"Pretrained head dim {weight.shape[1]} does not match model dim {model.embed_dim}")

    if num_classes == weight.shape[0]:
        selected_weight = weight
        selected_bias = bias
    else:
        source_wnids = _source_imagenet1k_wnids(cfg)
        source_index = {wnid: idx for idx, wnid in enumerate(source_wnids)}
        target_wnids = _target_wnids_for_head(cfg)
        if len(target_wnids) != num_classes:
            raise RuntimeError(f"Target class count {len(target_wnids)} does not match num_classes={num_classes}")
        indices = []
        for wnid in target_wnids:
            if wnid not in source_index:
                raise RuntimeError(f"Target wnid {wnid} not found in ImageNet-1K source classes")
            indices.append(source_index[wnid])
        idx = torch.tensor(indices, dtype=torch.long)
        selected_weight = weight.index_select(0, idx)
        selected_bias = bias.index_select(0, idx) if bias is not None else None

    if tuple(model.head.weight.shape) != tuple(selected_weight.shape):
        raise RuntimeError(f"Head weight shape mismatch: {model.head.weight.shape} vs {selected_weight.shape}")
    with torch.no_grad():
        model.head.weight.copy_(selected_weight)
        if selected_bias is not None:
            model.head.bias.copy_(selected_bias)
    print(f"Initialized classifier head from ImageNet-1K pretrained head ({num_classes} classes)")


def build_model(cfg: Any, num_classes: int | None = None) -> ViTWithStabilityScore:
    n_classes = int(num_classes if num_classes is not None else cfg.data.num_classes)
    model = ViTWithStabilityScore(
        backbone_name=cfg.model.backbone,
        num_classes=n_classes,
        score_cfg=cfg.model.score,
        aggregator_cfg=cfg.model.aggregator,
        vanilla_pool=str(cfg.model.get("vanilla_pool", "cls")),
        drop_path_rate=float(cfg.model.drop_path_rate),
        pretrained=bool(cfg.model.pretrained),
        dynamic_img_size=bool(cfg.model.get("dynamic_img_size", False)),
    )
    init_ckpt = cfg.model.get("init_ckpt", None)
    if init_ckpt:
        _load_init_checkpoint(model, init_ckpt)
    if bool(cfg.model.get("pretrained_head", False)):
        _copy_pretrained_head(model, cfg, n_classes)
    return model
