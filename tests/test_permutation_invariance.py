import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("omegaconf")

from omegaconf import OmegaConf  # noqa: E402

from lazystrike.models.vit import build_model  # noqa: E402


def test_global_var_topk_is_perm_invariant():
    cfg = OmegaConf.create(
        {
            "data": {"num_classes": 100, "input_size": 224},
            "model": {
                "backbone": "vit_small_patch16_224",
                "drop_path_rate": 0.0,
                "pretrained": False,
                "score": {"name": "global_var", "kwargs": {}},
                "aggregator": {"name": "topk_channel", "kwargs": {"K": 98}},
            },
        }
    )
    model = build_model(cfg, num_classes=100).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        patches = model.forward_features(x)
        scores = model.score(patches)
        perm = torch.randperm(patches.shape[-1])
        inv = torch.argsort(perm)
        scores_back = model.score(patches[:, :, perm])[:, :, inv]
    idx1 = scores.topk(98, dim=1).indices.sort(dim=1).values
    idx2 = scores_back.topk(98, dim=1).indices.sort(dim=1).values
    assert torch.equal(idx1, idx2)

