#!/usr/bin/env python
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from omegaconf import OmegaConf

from lazystrike.data.imagenet100 import build_imagenet100
from lazystrike.data.transforms import build_val_transform
from lazystrike.models.vit import build_model


@torch.no_grad()
def evaluate_with_perm(model, loader, device, perm: torch.Tensor, protocol: str) -> float:
    correct = 0
    total = 0
    inv = torch.argsort(perm)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        cls_token, patches = model.forward_tokens(images)
        if protocol == "score_only" and model.score is not None:
            scores = model.score(patches[:, :, perm])
            scores = scores[:, :, inv]
            cls = model.aggregator(patches, scores)
            logits = model.head(cls)
        elif protocol == "everything":
            if model.score is None:
                if model.vanilla_pool == "mean":
                    cls = patches[:, :, perm].mean(dim=1)
                else:
                    if cls_token is None:
                        raise RuntimeError("vanilla_pool='cls' requires a backbone CLS token")
                    cls = cls_token[:, perm]
            else:
                patches = patches[:, :, perm]
                scores = model.score(patches)
                cls = model.aggregator(patches, scores)
            logits = torch.nn.functional.linear(cls, model.head.weight[:, inv], model.head.bias)
        else:
            cls, _scores = model.aggregate(patches, cls_token)
            logits = model.head(cls)
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += int(targets.numel())
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num-perms", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--protocol", choices=["score_only", "everything"], default="score_only")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    cfg.model.init_ckpt = None
    cfg.model.pretrained = False
    cfg.model.pretrained_head = False
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = build_model(cfg, int(cfg.data.num_classes)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    dataset = build_imagenet100(cfg, "val", build_val_transform(int(cfg.data.input_size)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    dim = model.embed_dim
    identity = torch.arange(dim, device=device)
    base_acc = evaluate_with_perm(model, loader, device, identity, args.protocol)
    print(f"Identity Top-1: {base_acc:.4f}")

    accs = []
    for seed in args.seeds[: args.num_perms]:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        perm = torch.randperm(dim, generator=generator).to(device)
        acc = evaluate_with_perm(model, loader, device, perm, args.protocol)
        accs.append(acc)
        print(f"perm seed={seed}: Top-1={acc:.4f}, delta={acc - base_acc:+.4f}")
    print(f"Mean +/- Std: {st.mean(accs):.4f} +/- {st.pstdev(accs):.4f}")
    print(f"Drop from identity: {base_acc - st.mean(accs):+.4f}")


if __name__ == "__main__":
    main()
