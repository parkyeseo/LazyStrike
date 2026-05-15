#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from omegaconf import OmegaConf

from lazystrike.data.imagenet100 import build_imagenet100, build_loader
from lazystrike.data.transforms import build_val_transform
from lazystrike.models.vit import build_model
from lazystrike.utils.distributed import all_reduce_sum, cleanup_distributed, init_distributed, is_main
from lazystrike.utils.metrics import topk_correct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    rank, world_size, local_rank = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    model = build_model(cfg, int(cfg.data.num_classes)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    dataset = build_imagenet100(cfg, "val", build_val_transform(int(cfg.data.input_size)))
    loader = build_loader(cfg, dataset, is_train=False, world_size=world_size, rank=rank)

    correct1 = correct5 = total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            c1, c5 = topk_correct(logits, targets, (1, 5))
            correct1 += c1
            correct5 += c5
            total += int(targets.numel())
    stats = all_reduce_sum(torch.tensor([correct1, correct5, total], dtype=torch.float64, device=device))
    if is_main():
        print(f"Top-1: {stats[0].item() / max(stats[2].item(), 1.0):.4f}")
        print(f"Top-5: {stats[1].item() / max(stats[2].item(), 1.0):.4f}")
        print(f"N: {int(stats[2].item())}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
