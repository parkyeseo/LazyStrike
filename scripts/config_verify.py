#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf

from lazystrike.models.vit import build_model


def main() -> None:
    base = OmegaConf.load("configs/base.yaml")
    for path in sorted(Path("configs").glob("*/*.yaml")):
        cfg = OmegaConf.merge(base, OmegaConf.load(path))
        OmegaConf.resolve(cfg)
        model = build_model(cfg, num_classes=int(cfg.data.num_classes))
        score_name = type(model.score).__name__ if model.score is not None else "None"
        aggregator_name = type(model.aggregator).__name__ if model.aggregator is not None else "None"
        print(f"{path}: D={model.embed_dim} N={model.num_patches} score={score_name} agg={aggregator_name}")
    print("CONFIG_BUILD_OK")


if __name__ == "__main__":
    main()

