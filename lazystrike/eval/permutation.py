from __future__ import annotations

import torch


def random_channel_permutation(dim: int, seed: int, device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(dim, generator=generator).to(device)


@torch.no_grad()
def aggregate_with_score_permutation(model, images: torch.Tensor, perm: torch.Tensor):
    cls_token, patches = model.forward_tokens(images)
    if model.score is None:
        cls, _scores = model.aggregate(patches, cls_token)
        return cls
    inv = torch.argsort(perm)
    scores = model.score(patches[:, :, perm])[:, :, inv]
    return model.aggregator(patches, scores)
