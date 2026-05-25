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


def read_wnids(args) -> list[str]:
    if args.all_classes:
        val_root = Path(args.imagenet_root) / "val"
        wnids = sorted(p.name for p in val_root.iterdir() if p.is_dir())
        if not wnids:
            raise RuntimeError(f"No class directories found under {val_root}")
        return wnids
    return [line.strip() for line in Path(args.classes_txt).read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--imagenet-root", default="/home/yeseo_item/data_14T/UADL/data/imagenet/full")
    parser.add_argument("--classes-txt", default="/home/yeseo_item/data_14T/UADL/data/metadata/imagenet100_classes.txt")
    parser.add_argument("--all-classes", action="store_true", help="Use all wnid directories under imagenet-root/val.")
    parser.add_argument(
        "--score-method",
        default=None,
        help="Evaluate one score method. Kept for compatibility; prefer --score-methods.",
    )
    parser.add_argument(
        "--score-methods",
        nargs="+",
        default=None,
        help=(
            "Score methods to evaluate. Defaults to paper QCLS plus patch-mean auxiliary. "
            "Use 'all' for qcls, mean, encoder_cls, raw_score, vote_count, and prototype."
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
    if args.score_method and args.score_methods:
        raise ValueError("Use either --score-method or --score-methods, not both")
    if args.score_method:
        score_methods = [args.score_method]
    elif args.score_methods:
        score_methods = args.score_methods
    else:
        score_methods = ["patch_score_qcls", "patch_score_mean"]
    if score_methods == ["all"]:
        score_methods = [
            "patch_score_qcls",
            "patch_score_mean",
            "patch_score_shi",
            "raw_score",
            "vote_count",
            "patch_score_prototype",
        ]

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

    wnids = read_wnids(args)
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
        score_methods=score_methods,
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
            "classes_txt": "all_classes" if args.all_classes else args.classes_txt,
            "num_classes": int(cfg.data.num_classes),
            "num_pib_samples": len(dataset),
            "model_num_patches": int(model.num_patches),
            "max_bbox_area_ratio": args.max_bbox_area_ratio,
        }
    )
    for method in result.score_methods:
        pib = result.pibs[method]
        hits = result.hits_by_method[method]
        if pib is None:
            print(f"PiB ({method}): N/A")
        else:
            print(f"PiB ({method}): {pib:.4f} ({hits}/{result.total})")
    print(f"Total samples: {result.total}, empty_patch_sets={result.empty_patch_sets}")
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
