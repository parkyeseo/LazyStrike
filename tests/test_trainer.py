import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("omegaconf")

from omegaconf import OmegaConf  # noqa: E402

from lazystrike.train.trainer import Trainer  # noqa: E402


class _Logger:
    def __init__(self):
        self.records = []

    def log(self, data, step=None):
        self.records.append((data, step))


class _ConstantLogitModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, images):
        return self.logits[: images.shape[0]]


def _cfg():
    return OmegaConf.create(
        {
            "train": {
                "amp": False,
                "label_smoothing": 0.1,
                "mixup": {"enabled": False},
            }
        }
    )


def test_evaluate_logs_hard_label_val_loss():
    logits = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
        ]
    )
    targets = torch.tensor([0, 1, 2])
    logger = _Logger()
    trainer = Trainer(
        _cfg(),
        _ConstantLogitModel(logits),
        optimizer=None,
        scheduler=None,
        train_loader=[],
        val_loader=[(torch.zeros(3, 1), targets)],
        logger=logger,
        device=torch.device("cpu"),
    )

    top1, top5, val_loss = trainer.evaluate(epoch=0)

    expected_loss = torch.nn.functional.cross_entropy(logits, targets).item()
    assert top1 == 1.0
    assert top5 == 1.0
    assert val_loss == pytest.approx(expected_loss)
    assert logger.records[-1][0]["val/loss"] == pytest.approx(expected_loss)
