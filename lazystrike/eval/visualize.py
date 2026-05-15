from __future__ import annotations

import numpy as np
import torch


def normalize_heatmap(values: torch.Tensor, grid: int) -> np.ndarray:
    arr = values.reshape(grid, grid).detach().cpu().float().numpy()
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def patch_pca_rgb(patches: torch.Tensor, grid: int) -> np.ndarray:
    x = patches.detach().float().cpu()
    x = x - x.mean(dim=0, keepdim=True)
    u, s, _v = torch.pca_lowrank(x, q=3)
    rgb = u[:, :3] * s[:3]
    rgb = (rgb - rgb.min(dim=0).values) / (rgb.max(dim=0).values - rgb.min(dim=0).values + 1e-8)
    return rgb.reshape(grid, grid, 3).numpy()

