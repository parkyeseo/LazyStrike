#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from lazystrike.data.transforms import build_val_transform
from lazystrike.models.vit import build_model


def heatmap(values: torch.Tensor, grid: int) -> np.ndarray:
    arr = values.reshape(grid, grid).detach().cpu().float().numpy()
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def overlay(image: Image.Image, heat: np.ndarray, alpha: float = 0.5):
    width, height = image.size
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((width, height), Image.BILINEAR)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.imshow(np.asarray(heat_img), cmap="jet", alpha=alpha)
    ax.axis("off")
    return fig


def pca_rgb(patches: torch.Tensor, grid: int) -> np.ndarray:
    x = patches.detach().float().cpu()
    x = x - x.mean(dim=0, keepdim=True)
    u, s, _v = torch.pca_lowrank(x, q=3)
    proj = u[:, :3] * s[:3]
    proj = (proj - proj.min(dim=0).values) / (proj.max(dim=0).values - proj.min(dim=0).values + 1e-8)
    return proj.reshape(grid, grid, 3).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--out-dir", default="/mnt/newdisk/yeseo_item/UADL_viz")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = build_model(cfg, int(cfg.data.num_classes)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    grid = int(model.num_patches ** 0.5)
    transform = build_val_transform(int(cfg.data.input_size))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_path in args.images:
        image = Image.open(image_path).convert("RGB")
        x = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            patches, scores, _cls, _logits = model.forward_with_scores(x)
        stem = Path(image_path).stem
        if scores is not None:
            fig = overlay(image, heatmap(scores[0].mean(dim=-1), grid))
            fig.savefig(out_dir / f"{stem}_score.png", bbox_inches="tight", dpi=120)
            plt.close(fig)
        fig = overlay(image, heatmap(patches[0].norm(dim=-1), grid))
        fig.savefig(out_dir / f"{stem}_norm.png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        pca = pca_rgb(patches[0], grid)
        Image.fromarray((pca * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR).save(
            out_dir / f"{stem}_pca.png"
        )
    print(f"[done] wrote visualizations to {os.fspath(out_dir)}")


if __name__ == "__main__":
    main()
