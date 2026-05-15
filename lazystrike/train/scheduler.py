from timm.scheduler import CosineLRScheduler


def build_scheduler(cfg, optimizer, n_iter_per_epoch: int):
    total_steps = int(cfg.train.epochs) * int(n_iter_per_epoch)
    warmup_steps = int(cfg.train.warmup_epochs) * int(n_iter_per_epoch)
    return CosineLRScheduler(
        optimizer,
        t_initial=total_steps,
        lr_min=float(cfg.train.lr_min),
        warmup_t=warmup_steps,
        warmup_lr_init=float(cfg.train.warmup_lr),
        warmup_prefix=True,
        t_in_epochs=False,
    )

