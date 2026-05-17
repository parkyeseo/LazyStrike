from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEFAULT_IMAGE_SIZE = 224
DEFAULT_RESIZE_SHORT = 256


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


def resize_shorter_side(image: Image.Image, resize_short: int = DEFAULT_RESIZE_SHORT) -> Image.Image:
    width, height = image.size
    scale = float(resize_short) / float(min(width, height))
    resized_width = int(width * scale)
    resized_height = int(height * scale)
    return image.resize((resized_width, resized_height), Image.BICUBIC)


def center_crop(image: Image.Image, image_size: int = DEFAULT_IMAGE_SIZE) -> Image.Image:
    width, height = image.size
    left = (width - image_size) // 2
    top = (height - image_size) // 2
    return image.crop((left, top, left + image_size, top + image_size))


def official_display_image(
    image: Image.Image,
    image_size: int = DEFAULT_IMAGE_SIZE,
    resize_short: int = DEFAULT_RESIZE_SHORT,
) -> Image.Image:
    """Display image aligned with LAST-ViT visualization preprocessing."""
    return center_crop(resize_shorter_side(image, resize_short), image_size)


def heatmap_to_image(heat: np.ndarray, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    heat_uint8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(heat_uint8).resize((image_size, image_size), Image.BILINEAR))


def upsample_patch_values(values: torch.Tensor, grid: int, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    x = values.reshape(1, 1, grid, grid).detach().cpu().float()
    return F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)[0, 0].numpy()


def token_selection_count(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Count how often each patch is selected across channels, as in LAST-ViT."""
    batch, num_patches, dim = scores.shape
    k = min(int(k), num_patches)
    _, indices = torch.topk(scores, k=k, dim=1, largest=True)
    counts = torch.zeros(batch, num_patches, dtype=torch.float32, device=scores.device)
    counts.scatter_add_(1, indices.reshape(batch, k * dim), torch.ones(batch, k * dim, device=scores.device))
    return counts
