from __future__ import annotations

import torch

CROP_PCT = 0.875


def patch_center_in_bbox(idx_flat: int, bbox_in_orig: torch.Tensor, grid: int = 14) -> bool | None:
    margin = (1.0 - CROP_PCT) / 2.0
    x1, y1, x2, y2 = bbox_in_orig.tolist()
    bx1 = max(0.0, (x1 - margin) / CROP_PCT)
    by1 = max(0.0, (y1 - margin) / CROP_PCT)
    bx2 = min(1.0, (x2 - margin) / CROP_PCT)
    by2 = min(1.0, (y2 - margin) / CROP_PCT)
    if bx2 <= bx1 or by2 <= by1:
        return None
    row = idx_flat // grid
    col = idx_flat % grid
    px = (col + 0.5) / grid
    py = (row + 0.5) / grid
    return bx1 <= px <= bx2 and by1 <= py <= by2


def shi_patch_score(patches: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
    p_norm = torch.nn.functional.normalize(patches, dim=-1)
    c_norm = torch.nn.functional.normalize(cls.unsqueeze(1), dim=-1)
    return (p_norm * c_norm).sum(dim=-1)

