from __future__ import annotations

import torch


def build_optimizer(cfg, model):
    gamma_params = []
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("W_gamma") or ".W_gamma" in name:
            gamma_params.append(param)
        elif param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    base_lr = float(cfg.train.lr)
    weight_decay = float(cfg.train.weight_decay)
    groups = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
            "lr": base_lr,
            "group_name": "decay",
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
            "lr": base_lr,
            "group_name": "no_decay",
        },
    ]
    if gamma_params:
        groups.append(
            {
                "params": gamma_params,
                "weight_decay": 0.0,
                "lr": base_lr * float(cfg.train.gamma_lr_scale),
                "group_name": "tcig_gamma",
            }
        )
    return torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.999), eps=1e-8)

