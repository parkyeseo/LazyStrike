from __future__ import annotations

from omegaconf import OmegaConf

from .distributed import is_main


class WandbLogger:
    def __init__(self, cfg, run_name: str):
        self.enabled = bool(cfg.logging.use_wandb) and is_main()
        self._wandb = None
        if self.enabled:
            import wandb

            self._wandb = wandb
            wandb.init(
                project=cfg.logging.project,
                entity=cfg.logging.get("entity", None),
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                tags=list(cfg.logging.get("tags", [])),
                resume="allow",
            )

    def log(self, data: dict, step: int | None = None) -> None:
        if self.enabled:
            self._wandb.log(data, step=step)

    def log_image(self, key: str, image, caption: str | None = None, step: int | None = None) -> None:
        if self.enabled:
            self._wandb.log({key: self._wandb.Image(image, caption=caption)}, step=step)

    def finish(self) -> None:
        if self.enabled:
            self._wandb.finish()


class NullLogger:
    def log(self, *args, **kwargs) -> None:
        pass

    def log_image(self, *args, **kwargs) -> None:
        pass

    def finish(self) -> None:
        pass

