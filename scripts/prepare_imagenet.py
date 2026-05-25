#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import scipy.io as sio
from tqdm import tqdm

ARCHIVES = {
    "train": ("ILSVRC2012_img_train.tar", "1d675b47d978889d74fa0da5fadfb00e"),
    "val": ("ILSVRC2012_img_val.tar", "29b22e2961454d5413ddabcf34fc5622"),
    "devkit": ("ILSVRC2012_devkit_t12.tar.gz", "fa75699e90414af021442c21a62c3abf"),
    "bbox_val": ("ILSVRC2012_bbox_val_v3.tgz", "f4cd18b5ea29fe6bbea62ec9c20d80f0"),
}


def md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_archive(raw_root: Path, key: str, skip_md5: bool) -> Path:
    filename, expected = ARCHIVES[key]
    path = raw_root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    if not skip_md5:
        actual = md5(path)
        if actual != expected:
            raise RuntimeError(f"MD5 mismatch for {path}: expected {expected}, got {actual}")
    return path


def safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    root = path.resolve()
    for member in tar.getmembers():
        target = (path / member.name).resolve()
        if not str(target).startswith(str(root)):
            raise RuntimeError(f"Unsafe tar path: {member.name}")
    tar.extractall(path)


def parse_val_wnids(devkit_archive: Path) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(devkit_archive) as tar:
            safe_extract(tar, tmp_path)
        devkit_root = tmp_path / "ILSVRC2012_devkit_t12"
        meta = sio.loadmat(devkit_root / "data" / "meta.mat", squeeze_me=True)["synsets"]
        leaf_entries = [entry for entry in meta if int(entry[4]) == 0]
        idx_to_wnid = {int(entry[0]): str(entry[1]) for entry in leaf_entries}
        gt_path = devkit_root / "data" / "ILSVRC2012_validation_ground_truth.txt"
        val_indices = [int(line.strip()) for line in gt_path.read_text().splitlines() if line.strip()]
        return [idx_to_wnid[idx] for idx in val_indices]


def prepare_train(raw_root: Path, out_root: Path, skip_md5: bool) -> None:
    train_archive = verify_archive(raw_root, "train", skip_md5)
    train_root = out_root / "train"
    done = train_root / ".complete"
    if done.is_file():
        print(f"[skip] train already prepared: {train_root}")
        return
    train_root.mkdir(parents=True, exist_ok=True)

    print(f"[extract] {train_archive} -> {train_root}")
    with tarfile.open(train_archive) as tar:
        safe_extract(tar, train_root)

    class_archives = sorted(train_root.glob("n*.tar"))
    if len(class_archives) != 1000:
        raise RuntimeError(f"Expected 1000 class tar files, got {len(class_archives)}")
    for class_tar in tqdm(class_archives, desc="extract class tar"):
        class_dir = train_root / class_tar.stem
        class_dir.mkdir(exist_ok=True)
        if not any(class_dir.iterdir()):
            with tarfile.open(class_tar) as tar:
                safe_extract(tar, class_dir)
        class_tar.unlink()
    done.write_text("ok\n")


def prepare_val(raw_root: Path, out_root: Path, skip_md5: bool) -> None:
    val_archive = verify_archive(raw_root, "val", skip_md5)
    devkit_archive = verify_archive(raw_root, "devkit", skip_md5)
    val_root = out_root / "val"
    done = val_root / ".complete"
    if done.is_file():
        print(f"[skip] val already prepared: {val_root}")
        return
    val_root.mkdir(parents=True, exist_ok=True)
    val_wnids = parse_val_wnids(devkit_archive)
    if len(val_wnids) != 50000:
        raise RuntimeError(f"Expected 50000 val wnids, got {len(val_wnids)}")

    print(f"[extract] {val_archive} -> {val_root}")
    with tarfile.open(val_archive) as tar:
        safe_extract(tar, val_root)
    images = sorted(val_root.glob("ILSVRC2012_val_*.JPEG"))
    if len(images) != 50000:
        raise RuntimeError(f"Expected 50000 val images, got {len(images)}")
    for wnid in sorted(set(val_wnids)):
        (val_root / wnid).mkdir(exist_ok=True)
    for image_path, wnid in tqdm(list(zip(images, val_wnids)), desc="move val images"):
        shutil.move(str(image_path), str(val_root / wnid / image_path.name))
    done.write_text("ok\n")


def prepare_bbox(raw_root: Path, out_root: Path, skip_md5: bool) -> None:
    bbox_archive = verify_archive(raw_root, "bbox_val", skip_md5)
    bbox_root = out_root / "val_bbox"
    done = bbox_root / ".complete"
    if done.is_file():
        print(f"[skip] bbox already prepared: {bbox_root}")
        return
    bbox_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print(f"[extract] {bbox_archive} -> {bbox_root}")
        with tarfile.open(bbox_archive) as tar:
            safe_extract(tar, tmp_path)
        xml_files = sorted(tmp_path.rglob("*.xml"))
        if not xml_files:
            raise RuntimeError(f"No XML files found in {bbox_archive}")
        for xml_path in tqdm(xml_files, desc="copy bbox xml"):
            shutil.copy2(xml_path, bbox_root / xml_path.name)
    done.write_text("ok\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="/home/yeseo_item/data_14T/UADL/data/imagenet/raw")
    parser.add_argument("--out-root", default="/home/yeseo_item/data_14T/UADL/data/imagenet/full")
    parser.add_argument("--skip-md5", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--skip-bbox", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_train:
        prepare_train(raw_root, out_root, args.skip_md5)
    if not args.skip_val:
        prepare_val(raw_root, out_root, args.skip_md5)
    if not args.skip_bbox:
        prepare_bbox(raw_root, out_root, args.skip_md5)
    print(f"[done] prepared ImageNet at {out_root}")


if __name__ == "__main__":
    main()

