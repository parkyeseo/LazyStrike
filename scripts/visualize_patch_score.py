#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from lazystrike.data.transforms import build_val_transform
from lazystrike.eval.pib import shi_patch_score
from lazystrike.eval.visualize import (
    official_display_image,
    patch_pca_rgb,
    token_selection_count,
    upsample_patch_values,
)
from lazystrike.models.vit import build_model


def parse_ckpt(value: str) -> tuple[str, str]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name, path
    path = value
    name = Path(path).parent.name or Path(path).stem
    return name, path


def infer_patch_size(model) -> int:
    patch_size = getattr(model.backbone.patch_embed, "patch_size", None)
    if isinstance(patch_size, tuple):
        if patch_size[0] != patch_size[1]:
            raise ValueError(f"Visualization expects square patches, got patch_size={patch_size}")
        return int(patch_size[0])
    if patch_size is None:
        raise ValueError("Could not infer patch size from model.backbone.patch_embed")
    return int(patch_size)


def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    model = build_model(cfg, int(cfg.data.num_classes)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, cfg, ckpt


def save_official_patch_score_figure(
    image_display: Image.Image,
    score_values: torch.Tensor,
    logits: torch.Tensor,
    grid: int,
    patch_size: int,
    save_path: Path,
    top_k: int,
    title_prefix: str,
) -> None:
    patch_scores = score_values.detach().cpu().float().numpy()
    score_map = patch_scores.reshape(grid, grid)
    pred_class = int(logits.argmax().item())
    pred_prob = float(torch.softmax(logits, dim=0)[pred_class].item())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image_display)
    axes[0].set_title(f"{title_prefix}\nPred: Class {pred_class} ({pred_prob:.3f})", fontsize=12)
    axes[0].axis("off")

    im = axes[1].imshow(score_map, cmap="hot", interpolation="nearest")
    axes[1].set_title(
        f"Patch Scores ({grid}x{grid})\nMin: {score_map.min():.3f}, Max: {score_map.max():.3f}",
        fontsize=12,
    )
    axes[1].set_xlabel("Patch X")
    axes[1].set_ylabel("Patch Y")
    plt.colorbar(im, ax=axes[1])

    axes[2].imshow(image_display, alpha=0.6)
    upsampled = upsample_patch_values(score_values, grid, image_display.size[0])
    im2 = axes[2].imshow(upsampled, cmap="hot", alpha=0.4, interpolation="bilinear")
    if top_k > 0:
        flat_indices = np.argsort(patch_scores)[-top_k:]
        for idx in flat_indices:
            row = int(idx) // grid
            col = int(idx) % grid
            score = patch_scores[idx]
            rect = mpatches.Rectangle(
                (col * patch_size, row * patch_size),
                patch_size,
                patch_size,
                linewidth=2,
                edgecolor="cyan",
                facecolor="none",
            )
            axes[2].add_patch(rect)
            axes[2].text(
                col * patch_size + patch_size / 2,
                row * patch_size + patch_size / 2,
                f"{score:.2f}",
                color="white",
                fontsize=8,
                ha="center",
                va="center",
                bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.5},
            )
    axes[2].set_title(f"Overlay (Top-{top_k} patches)" if top_k > 0 else "Overlay", fontsize=12)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2])
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_overlay(
    image_display: Image.Image,
    values: torch.Tensor,
    grid: int,
    save_path: Path,
    title: str,
    cmap: str = "hot",
    alpha: float = 0.4,
) -> None:
    upsampled = upsample_patch_values(values, grid, image_display.size[0])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_display, alpha=0.65)
    im = ax.imshow(upsampled, cmap=cmap, alpha=alpha, interpolation="bilinear")
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_pca(image_display: Image.Image, patches: torch.Tensor, grid: int, save_path: Path) -> None:
    pca = patch_pca_rgb(patches, grid)
    Image.fromarray((pca * 255).astype(np.uint8)).resize(image_display.size, Image.BILINEAR).save(save_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", nargs="+", required=True, help="Checkpoint paths or name=/path/to/best.pth entries.")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--out-dir", default="/mnt/newdisk/yeseo_item/UADL_viz")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--selection-k", type=int, default=1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = []
    for ckpt_arg in args.ckpt:
        name, path = parse_ckpt(ckpt_arg)
        model, cfg, _ckpt = load_model_from_ckpt(path, device)
        input_size = int(cfg.data.input_size)
        patch_size = infer_patch_size(model)
        grid = int(model.num_patches**0.5)
        if grid * grid != int(model.num_patches):
            raise ValueError(f"{name}: expected square patch grid, got num_patches={model.num_patches}")
        if input_size != grid * patch_size:
            raise ValueError(f"{name}: input_size={input_size} does not match grid={grid}, patch_size={patch_size}")
        models.append((name, model, cfg, grid, patch_size))

    for image_path in args.images:
        image = Image.open(image_path).convert("RGB")
        stem = Path(image_path).stem
        for name, model, cfg, grid, patch_size in models:
            run_dir = out_dir / name
            run_dir.mkdir(parents=True, exist_ok=True)
            input_size = int(cfg.data.input_size)
            transform = build_val_transform(input_size)
            image_display = official_display_image(image, image_size=input_size)
            x = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                patches, scores, cls, logits = model.forward_with_scores(x)

            patch_score = shi_patch_score(patches, cls)[0]
            save_official_patch_score_figure(
                image_display=image_display,
                score_values=patch_score,
                logits=logits[0],
                grid=grid,
                patch_size=patch_size,
                save_path=run_dir / f"{stem}_patch_score_official.png",
                top_k=args.top_k,
                title_prefix=f"{name} / {stem}",
            )
            save_overlay(
                image_display,
                patches[0].norm(dim=-1),
                grid,
                run_dir / f"{stem}_feature_norm.png",
                "Feature Norm",
                cmap="hot",
            )
            save_pca(image_display, patches[0], grid, run_dir / f"{stem}_patch_pca.png")

            if scores is not None:
                save_overlay(
                    image_display,
                    scores[0].mean(dim=-1),
                    grid,
                    run_dir / f"{stem}_raw_stability_score.png",
                    "Mean Stability Score",
                    cmap="hot",
                )
                selection = token_selection_count(scores, k=args.selection_k)[0]
                save_overlay(
                    image_display,
                    selection,
                    grid,
                    run_dir / f"{stem}_token_selection_k{args.selection_k}.png",
                    f"Token Selection Count (k={args.selection_k})",
                    cmap="hot",
                )
    print(f"[done] wrote visualizations to {os.fspath(out_dir)}")


if __name__ == "__main__":
    main()
