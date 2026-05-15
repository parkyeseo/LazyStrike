from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageNetBBoxSubset(Dataset):
    """ImageNet val subset with one-object bbox annotations filtered by wnid."""

    def __init__(
        self,
        imagenet_root: str,
        wnids: list[str],
        transform=None,
        input_size: int = 224,
        bbox_dir_name: str = "val_bbox",
    ):
        super().__init__()
        self.root = Path(imagenet_root)
        self.transform = transform
        self.input_size = int(input_size)
        self.samples: list[tuple[str, int, tuple[int, int, int, int], tuple[int, int]]] = []

        wnids = [w.strip() for w in wnids if w.strip()]
        wnid_to_idx = {wnid: i for i, wnid in enumerate(sorted(wnids))}
        bbox_dir = self.root / bbox_dir_name
        if not bbox_dir.is_dir():
            raise FileNotFoundError(f"Missing bbox directory: {bbox_dir}")

        for xml_path in sorted(bbox_dir.glob("*.xml")):
            tree = ET.parse(xml_path).getroot()
            objects = tree.findall("object")
            if len(objects) != 1:
                continue
            obj = objects[0]
            wnid = obj.findtext("name")
            if wnid not in wnid_to_idx:
                continue
            box = obj.find("bndbox")
            size = tree.find("size")
            if box is None or size is None:
                continue
            x1 = int(float(box.findtext("xmin")))
            y1 = int(float(box.findtext("ymin")))
            x2 = int(float(box.findtext("xmax")))
            y2 = int(float(box.findtext("ymax")))
            width = int(float(size.findtext("width")))
            height = int(float(size.findtext("height")))
            image_path = self.root / "val" / wnid / f"{xml_path.stem}.JPEG"
            if not image_path.is_file():
                flat_image_path = self.root / "val" / f"{xml_path.stem}.JPEG"
                image_path = flat_image_path
            if image_path.is_file():
                self.samples.append((str(image_path), wnid_to_idx[wnid], (x1, y1, x2, y2), (width, height)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label, bbox, image_size = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        x1, y1, x2, y2 = bbox
        width, height = image_size
        bbox_norm = torch.tensor([x1 / width, y1 / height, x2 / width, y2 / height], dtype=torch.float32)
        return image, label, bbox_norm

