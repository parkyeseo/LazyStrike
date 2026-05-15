#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from omegaconf import OmegaConf

from lazystrike.data.pib_dataset import ImageNetBBoxSubset
from lazystrike.data.transforms import build_val_transform
from lazystrike.models.vit import build_model

INPUT_SIZE = 224
PATCH_SIZE = 16
GRID = INPUT_SIZE // PATCH_SIZE
CROP_PCT = 0.875


def patch_center_in_bbox(idx_flat: int, bbox_in_orig: torch.Tensor) -> bool | None:
    margin = (1.0 - CROP_PCT) / 2.0
    x1, y1, x2, y2 = bbox_in_orig.tolist()
    bx1 = max(0.0, (x1 - margin) / CROP_PCT)
    by1 = max(0.0, (y1 - margin) / CROP_PCT)
    bx2 = min(1.0, (x2 - margin) / CROP_PCT)
    by2 = min(1.0, (y2 - margin) / CROP_PCT)
    if bx2 <= bx1 or by2 <= by1:
        return None
    row = idx_flat // GRID
    col = idx_flat % GRID
    px = (col + 0.5) / GRID
    py = (row + 0.5) / GRID
    return bx1 <= px <= bx2 and by1 <= py <= by2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--imagenet-root", default="/mnt/newdisk/yeseo_item/imagenet/full")
    parser.add_argument("--classes-txt", default="/mnt/newdisk/yeseo_item/UADL_data/imagenet100_classes.txt")
    parser.add_argument("--score-method", default="patch_score_shi", choices=["patch_score_shi", "raw_score", "vote_count"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = build_model(cfg, int(cfg.data.num_classes)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    wnids = [line.strip() for line in Path(args.classes_txt).read_text().splitlines() if line.strip()]
    dataset = ImageNetBBoxSubset(
        args.imagenet_root,
        wnids,
        transform=build_val_transform(INPUT_SIZE),
        input_size=INPUT_SIZE,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    hits = total = skipped = 0
    with torch.no_grad():
        for images, _targets, bboxes in loader:
            images = images.to(device, non_blocking=True)
            patches, scores, cls, _logits = model.forward_with_scores(images)
            if args.score_method == "patch_score_shi":
                p_norm = torch.nn.functional.normalize(patches, dim=-1)
                c_norm = torch.nn.functional.normalize(cls.unsqueeze(1), dim=-1)
                patch_scores = (p_norm * c_norm).sum(dim=-1)
            elif args.score_method == "raw_score":
                if scores is None:
                    raise RuntimeError("raw_score requires a stability score model")
                patch_scores = scores.mean(dim=-1)
            else:
                if scores is None or not hasattr(model.aggregator, "vote_count"):
                    raise RuntimeError("vote_count requires TopKChannelAggregator")
                patch_scores = model.aggregator.vote_count(scores).float()

            best_idx = patch_scores.argmax(dim=1).cpu()
            for i in range(images.shape[0]):
                hit = patch_center_in_bbox(int(best_idx[i].item()), bboxes[i])
                if hit is None:
                    skipped += 1
                else:
                    total += 1
                    hits += int(hit)
    print(f"PiB ({args.score_method}): {hits / max(total, 1):.4f} ({hits}/{total}, skipped={skipped})")


if __name__ == "__main__":
    main()
