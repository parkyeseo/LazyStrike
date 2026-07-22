#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from omegaconf import OmegaConf

from lazystrike.models.aggregator import TopKChannelAggregator, TopKPatchAggregator
from lazystrike.models.stability_scores import (
    FFTScore,
    GlobalVarianceScore,
    LocalWindowVarianceScore,
    TCIGScore,
)
from lazystrike.models.vit import build_model
from lazystrike.train.optimizer import build_optimizer


def main() -> None:
    batch, patches, dim = 2, 196, 384
    torch.manual_seed(0)
    x = torch.randn(batch, patches, dim)

    for cls, kwargs in [
        (FFTScore, {"dim": dim, "sigma": 24.0}),
        (GlobalVarianceScore, {"dim": dim}),
        (TCIGScore, {"dim": dim, "kernel_size": 3, "init_W": 2.0}),
        (LocalWindowVarianceScore, {"dim": dim, "window_size": 8}),
    ]:
        scores = cls(**kwargs)(x)
        assert scores.shape == (batch, patches, dim), (cls.__name__, scores.shape)
        assert torch.isfinite(scores).all(), cls.__name__
    print("score shape/finiteness ok")

    global_scores = GlobalVarianceScore(dim=dim)(x)
    local_scores = LocalWindowVarianceScore(dim=dim, window_size=dim)(x)
    assert torch.allclose(global_scores, local_scores, atol=1e-5)
    print("local_var w=D equals global_var ok")

    perm = torch.randperm(dim)
    inv = torch.argsort(perm)
    perm_scores = GlobalVarianceScore(dim=dim)(x[:, :, perm])[:, :, inv]
    assert torch.allclose(global_scores, perm_scores, atol=1e-5)
    print("global_var permutation invariance ok")

    channel_agg = TopKChannelAggregator(K=98)
    cls_token = channel_agg(x, torch.randn_like(x))
    assert cls_token.shape == (batch, dim)
    vote_scores = torch.randn(batch, 10, 7)
    votes = TopKChannelAggregator(K=3).vote_count(vote_scores)
    assert votes.shape == (batch, 10)
    assert torch.equal(votes.sum(dim=1), torch.full((batch,), 3 * 7, dtype=torch.long))
    patch_cls = TopKPatchAggregator(K=98)(x, torch.randn_like(x))
    assert patch_cls.shape == (batch, dim)
    print("aggregators ok")

    cfg = OmegaConf.create(
        {
            "data": {"num_classes": 100, "input_size": 224},
            "model": {
                "backbone": "vit_small_patch16_224",
                "drop_path_rate": 0.0,
                "pretrained": False,
                "score": {"name": "local_var", "kwargs": {"window_size": 8}},
                "aggregator": {"name": "topk_channel", "kwargs": {"K": 98}},
            },
            "train": {"lr": 5e-4, "weight_decay": 0.05, "gamma_lr_scale": 0.1},
        }
    )
    model = build_model(cfg, num_classes=100)
    logits = model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 100)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0]))
    loss.backward()
    assert sum(p.grad is not None for p in model.parameters()) > 0
    print("vit local_var forward/backward ok")

    cfg.model.score = {"name": "tcig", "kwargs": {"kernel_size": 3, "init_W": 2.0}}
    model = build_model(cfg, num_classes=100)
    optimizer = build_optimizer(cfg, model)
    assert not any(group.get("group_name") == "tcig_gamma" for group in optimizer.param_groups)
    assert "W_gamma" not in dict(model.named_parameters())
    print("tcig gamma fixed by default for hard top-k ok")

    cfg.model.score = {
        "name": "tcig",
        "kwargs": {"kernel_size": 3, "init_W": 2.0, "learnable_gamma": True},
    }
    model = build_model(cfg, num_classes=100)
    optimizer = build_optimizer(cfg, model)
    assert not any(group.get("group_name") == "tcig_gamma" for group in optimizer.param_groups)
    print("legacy learnable_gamma config remains parameter-free ok")
    print("SMOKE_VERIFY_OK")


if __name__ == "__main__":
    main()
