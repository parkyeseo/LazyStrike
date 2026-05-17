from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

from lazystrike.utils.distributed import all_reduce_sum, is_main
from lazystrike.utils.metrics import topk_correct


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _autocast_enabled(cfg) -> bool:
    return bool(cfg.train.amp) and torch.cuda.is_available()


class Trainer:
    def __init__(self, cfg, model, optimizer, scheduler, train_loader, val_loader, logger, device):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(enabled=_autocast_enabled(cfg))

        self.mixup = None
        if bool(cfg.train.mixup.enabled):
            self.mixup = Mixup(
                mixup_alpha=float(cfg.train.mixup.mixup_alpha),
                cutmix_alpha=float(cfg.train.mixup.cutmix_alpha),
                prob=float(cfg.train.mixup.prob),
                switch_prob=float(cfg.train.mixup.switch_prob),
                mode=str(cfg.train.mixup.mode),
                label_smoothing=float(cfg.train.label_smoothing),
                num_classes=int(cfg.data.num_classes),
            )
        self.criterion = (
            SoftTargetCrossEntropy()
            if self.mixup is not None
            else nn.CrossEntropyLoss(label_smoothing=float(cfg.train.label_smoothing))
        )

    def train_one_epoch(self, epoch: int) -> None:
        self.model.train()
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        n_batches = len(self.train_loader)
        loss_sum = 0.0
        correct = 0
        total = 0
        start = time.time()

        for it, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            loss_targets = targets
            if self.mixup is not None:
                images, loss_targets = self.mixup(images, targets)

            with torch.cuda.amp.autocast(enabled=_autocast_enabled(self.cfg)):
                logits = self.model(images)
                loss = self.criterion(logits, loss_targets)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if float(self.cfg.train.grad_clip) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.train.grad_clip))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step_update(epoch * n_batches + it + 1)

            loss_sum += float(loss.item())
            with torch.no_grad():
                correct += int((logits.argmax(dim=1) == targets).sum().item())
                total += int(targets.numel())

            if is_main() and it % int(self.cfg.logging.log_every) == 0:
                log_data = {
                    "train/loss_step": float(loss.item()),
                    "train/lr": float(self.optimizer.param_groups[0]["lr"]),
                    "train/lr_gamma": self._gamma_lr(),
                    "epoch": epoch,
                }
                gamma_value = self._gamma_value()
                if gamma_value is not None:
                    log_data["train/tcig_gamma"] = gamma_value
                self.logger.log(log_data, step=epoch * n_batches + it)

        stats = torch.tensor([loss_sum, correct, total], dtype=torch.float64, device=self.device)
        all_reduce_sum(stats)
        if is_main():
            global_batches = n_batches
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                global_batches *= torch.distributed.get_world_size()
            self.logger.log(
                {
                    "train/loss": float(stats[0].item() / max(global_batches, 1)),
                    "train/top1": float(stats[1].item() / max(stats[2].item(), 1.0)),
                    "train/time_sec": time.time() - start,
                    "epoch": epoch,
                }
            )

    @torch.no_grad()
    def evaluate(self, epoch: int) -> tuple[float, float]:
        self.model.eval()
        correct1 = 0
        correct5 = 0
        total = 0
        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=_autocast_enabled(self.cfg)):
                logits = self.model(images)
            c1, c5 = topk_correct(logits, targets, (1, 5))
            correct1 += c1
            correct5 += c5
            total += int(targets.numel())

        stats = torch.tensor([correct1, correct5, total], dtype=torch.float64, device=self.device)
        all_reduce_sum(stats)
        top1 = float(stats[0].item() / max(stats[2].item(), 1.0))
        top5 = float(stats[1].item() / max(stats[2].item(), 1.0))
        if is_main():
            self.logger.log({"val/top1": top1, "val/top5": top5, "epoch": epoch})
        return top1, top5

    def _gamma_lr(self) -> float:
        for group in self.optimizer.param_groups:
            if group.get("group_name") == "tcig_gamma":
                return float(group["lr"])
        return 0.0

    def _gamma_value(self) -> float | None:
        model = unwrap_model(self.model)
        score = getattr(model, "score", None)
        if score is not None and hasattr(score, "gamma"):
            return float(score.gamma.detach().cpu().item())
        return None


def save_checkpoint(cfg, model, optimizer, scheduler, epoch: int, top1, top5, ckpt_dir: str, name: str) -> None:
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    state = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "metrics": {"top1": top1, "top5": top5},
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(state, os.path.join(ckpt_dir, name))


def load_checkpoint(model, optimizer, scheduler, ckpt_path: str, device) -> int:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("epoch", -1)) + 1


def load_existing_best_top1(ckpt_dir: str, device) -> float:
    best_path = Path(ckpt_dir) / "best.pth"
    if not best_path.is_file():
        return 0.0
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    metrics = ckpt.get("metrics") or {}
    top1 = metrics.get("top1", 0.0)
    return float(top1) if top1 is not None else 0.0


def run_training(cfg, model, optimizer, scheduler, train_loader, val_loader, logger, device, ckpt_dir: str) -> float:
    trainer = Trainer(cfg, model, optimizer, scheduler, train_loader, val_loader, logger, device)
    start_epoch = 0
    if cfg.train.get("resume", None):
        start_epoch = load_checkpoint(model, optimizer, scheduler, str(cfg.train.resume), device)

    best_top1 = load_existing_best_top1(ckpt_dir, device)
    for epoch in range(start_epoch, int(cfg.train.epochs)):
        trainer.train_one_epoch(epoch)
        should_eval = (epoch + 1) % int(cfg.train.eval_every) == 0 or epoch == int(cfg.train.epochs) - 1
        if should_eval:
            top1, top5 = trainer.evaluate(epoch)
            if is_main() and top1 > best_top1:
                best_top1 = top1
                save_checkpoint(cfg, model, optimizer, scheduler, epoch, top1, top5, ckpt_dir, "best.pth")
        if is_main() and (epoch + 1) % int(cfg.train.save_every) == 0:
            save_checkpoint(cfg, model, optimizer, scheduler, epoch, None, None, ckpt_dir, f"epoch_{epoch:03d}.pth")
    if is_main():
        save_checkpoint(cfg, model, optimizer, scheduler, int(cfg.train.epochs) - 1, None, None, ckpt_dir, "last.pth")
    return best_top1
