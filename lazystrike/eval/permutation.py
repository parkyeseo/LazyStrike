from __future__ import annotations

import torch


def random_channel_permutation(dim: int, seed: int, device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(dim, generator=generator).to(device)


@torch.no_grad()
def aggregate_with_score_permutation(model, images: torch.Tensor, perm: torch.Tensor):
    patches = model.forward_features(images)
    if model.score is None:
        return patches.mean(dim=1)
    inv = torch.argsort(perm)
    scores = model.score(patches[:, :, perm])[:, :, inv]
    return model.aggregator(patches, scores)

