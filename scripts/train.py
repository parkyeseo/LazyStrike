#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from lazystrike.data.imagenet100 import build_imagenet100, build_loader
from lazystrike.data.transforms import build_train_transform, build_val_transform
from lazystrike.models.vit import build_model
from lazystrike.train.optimizer import build_optimizer
from lazystrike.train.scheduler import build_scheduler
from lazystrike.train.trainer import run_training
from lazystrike.utils.distributed import cleanup_distributed, init_distributed, is_main
from lazystrike.utils.logging import NullLogger, WandbLogger
from lazystrike.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def load_cfg(args):
    cfg = OmegaConf.load("configs/base.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config))
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    OmegaConf.resolve(cfg)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)

    rank, world_size, local_rank = init_distributed()
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        if world_size > 1:
            raise RuntimeError("DDP training requires CUDA in this implementation")
        device = torch.device("cpu")
    set_seed(int(cfg.train.seed) + rank, deterministic=bool(cfg.train.deterministic))

    model = build_model(cfg, num_classes=int(cfg.data.num_classes)).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    train_ds = build_imagenet100(cfg, "train", build_train_transform(int(cfg.data.input_size)))
    val_ds = build_imagenet100(cfg, "val", build_val_transform(int(cfg.data.input_size)))
    train_loader = build_loader(cfg, train_ds, is_train=True, world_size=world_size, rank=rank)
    val_loader = build_loader(cfg, val_ds, is_train=False, world_size=world_size, rank=rank)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, len(train_loader))

    run_name = cfg.logging.get("run_name", None) or Path(args.config).stem
    epochs_dir = f"{int(cfg.train.epochs)}ep"
    ckpt_dir = args.ckpt_dir or os.path.join(str(cfg.paths.checkpoint_root), epochs_dir, run_name)
    if is_main():
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        print(f"[config] {OmegaConf.to_yaml(cfg)}")
        print(f"[run] name={run_name} world_size={world_size} ckpt_dir={ckpt_dir}")

    logger = WandbLogger(cfg, run_name) if is_main() else NullLogger()
    try:
        best = run_training(cfg, model, optimizer, scheduler, train_loader, val_loader, logger, device, ckpt_dir)
        if is_main():
            print(f"[done] best val top-1 = {best:.4f}")
    finally:
        logger.finish()
        cleanup_distributed()


if __name__ == "__main__":
    main()
