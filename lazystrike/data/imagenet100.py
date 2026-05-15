from __future__ import annotations

import os

from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import ImageFolder


def build_imagenet100(cfg, split: str, transform):
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    root = os.path.join(str(cfg.data.root), split)
    dataset = ImageFolder(root, transform=transform)
    if len(dataset.classes) != int(cfg.data.num_classes):
        raise RuntimeError(
            f"Expected {cfg.data.num_classes} classes under {root}, got {len(dataset.classes)}"
        )
    return dataset


def build_loader(cfg, dataset, is_train: bool, world_size: int, rank: int):
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=is_train,
            drop_last=is_train,
        )
    return DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size_per_gpu),
        sampler=sampler,
        shuffle=(sampler is None and is_train),
        num_workers=int(cfg.data.num_workers),
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=int(cfg.data.num_workers) > 0,
    )

