#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from omegaconf import OmegaConf

from lazystrike.data.pib_dataset import ImageNetBBoxSubset
from lazystrike.data.transforms import build_val_transform
from lazystrike.eval.pib import DEFAULT_RESIZE_SHORT, evaluate_pib
from lazystrike.models.vit import build_model


def infer_patch_size(model) -> int:
    patch_size = getattr(model.backbone.patch_embed, "patch_size", None)
    if isinstance(patch_size, tuple):
        if patch_size[0] != patch_size[1]:
            raise ValueError(f"PiB expects square patches, got patch_size={patch_size}")
        return int(patch_size[0])
    if patch_size is None:
        raise ValueError("Could not infer patch size from model.backbone.patch_embed")
    return int(patch_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--imagenet-root", default="/mnt/newdisk/yeseo_item/imagenet/full")
    parser.add_argument("--classes-txt", default="/mnt/newdisk/yeseo_item/UADL_data/imagenet100_classes.txt")
    parser.add_argument(
        "--score-method",
        default="patch_score_shi",
        choices=["patch_score_shi", "patch_score_pooled", "raw_score", "vote_count"],
        help=(
            "patch_score_shi matches official evaluate_patch_hit.py: "
            "cosine(patch, encoder learned CLS). patch_score_pooled uses the "
            "model's final pooled token instead."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resize-short", type=int, default=DEFAULT_RESIZE_SHORT)
    parser.add_argument(
        "--max-bbox-area-ratio",
        type=float,
        default=0.25,
        help="Official LAST-ViT filter; set <=0 to disable.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional smoke-test limit; omit for paper runs.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model = build_model(cfg, int(cfg.data.num_classes)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    input_size = int(cfg.data.input_size)
    patch_size = infer_patch_size(model)
    grid = int(model.num_patches**0.5)
    if grid * grid != int(model.num_patches):
        raise ValueError(f"PiB expects a square patch grid, got num_patches={model.num_patches}")
    if grid != input_size // patch_size:
        raise ValueError(
            f"Patch grid mismatch: model grid={grid}, input_size={input_size}, patch_size={patch_size}"
        )

    wnids = [line.strip() for line in Path(args.classes_txt).read_text().splitlines() if line.strip()]
    dataset = ImageNetBBoxSubset(
        args.imagenet_root,
        wnids,
        transform=build_val_transform(input_size),
        input_size=input_size,
        max_bbox_area_ratio=args.max_bbox_area_ratio if args.max_bbox_area_ratio > 0 else None,
    )
    if args.max_samples is not None:
        dataset.samples = dataset.samples[: args.max_samples]
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    result = evaluate_pib(
        model,
        loader,
        device,
        score_method=args.score_method,
        image_size=input_size,
        patch_size=patch_size,
        resize_short=args.resize_short,
    )

    payload = result.to_dict()
    payload.update(
        {
            "checkpoint": args.ckpt,
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_metrics": ckpt.get("metrics"),
            "imagenet_root": args.imagenet_root,
            "classes_txt": args.classes_txt,
            "num_classes": int(cfg.data.num_classes),
            "num_pib_samples": len(dataset),
            "model_num_patches": int(model.num_patches),
            "max_bbox_area_ratio": args.max_bbox_area_ratio,
        }
    )
    print(
        f"PiB ({result.score_method}): {result.pib:.4f} "
        f"({result.hits}/{result.total}, empty_patch_sets={result.empty_patch_sets})"
    )
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
