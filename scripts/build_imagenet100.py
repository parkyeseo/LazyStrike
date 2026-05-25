#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path


def read_wnids(path: Path) -> list[str]:
    wnids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(wnids) != 100:
        raise RuntimeError(f"Expected exactly 100 wnids in {path}, got {len(wnids)}")
    if len(set(wnids)) != 100:
        raise RuntimeError(f"Duplicate wnids found in {path}")
    return wnids


def symlink_force(src: Path, dst: Path) -> None:
    if dst.is_symlink():
        current = Path(os.readlink(dst))
        if current == src.resolve():
            return
        dst.unlink()
    elif dst.exists():
        raise FileExistsError(f"Refusing to overwrite non-symlink path: {dst}")
    os.symlink(src.resolve(), dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-root", default="/home/yeseo_item/data_14T/UADL/data/imagenet/full")
    parser.add_argument("--out-root", default="/home/yeseo_item/data_14T/UADL/data/imagenet-100")
    parser.add_argument("--classes-txt", default="/home/yeseo_item/data_14T/UADL/data/metadata/imagenet100_classes.txt")
    args = parser.parse_args()

    imagenet_root = Path(args.imagenet_root)
    out_root = Path(args.out_root)
    classes_txt = Path(args.classes_txt)
    wnids = read_wnids(classes_txt)

    for split in ("train", "val"):
        for wnid in wnids:
            src = imagenet_root / split / wnid
            if not src.is_dir():
                raise FileNotFoundError(f"Missing source class directory: {src}")
            dst = out_root / split / wnid
            dst.parent.mkdir(parents=True, exist_ok=True)
            symlink_force(src, dst)
    print(f"[done] wrote symlink ImageNet-100 to {out_root}")


if __name__ == "__main__":
    main()

