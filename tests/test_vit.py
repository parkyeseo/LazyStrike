import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("omegaconf")

from omegaconf import OmegaConf  # noqa: E402

from lazystrike.models.vit import build_model  # noqa: E402
from lazystrike.train.optimizer import build_optimizer  # noqa: E402


def make_cfg(score_name, score_kwargs=None, K=98, vanilla_pool="cls"):
    return OmegaConf.create(
        {
            "data": {"num_classes": 100, "input_size": 224},
            "model": {
                "backbone": "vit_small_patch16_224",
                "drop_path_rate": 0.0,
                "pretrained": False,
                "vanilla_pool": vanilla_pool,
                "score": {"name": score_name, "kwargs": score_kwargs or {}},
                "aggregator": {"name": "topk_channel", "kwargs": {"K": K}},
            },
            "train": {"lr": 5e-4, "weight_decay": 0.05, "gamma_lr_scale": 0.1},
        }
    )


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("fft", {}),
        ("global_var", {}),
        ("tcig", {"kernel_size": 7, "init_W": 2.0}),
        ("tasc", {"init_W": 2.0}),
        ("dual_guard", {"k_small": 7, "k_large": 21}),
        ("local_var", {"window_size": 8}),
    ],
)
def test_vit_forward_shape_all_cells(name, kwargs):
    model = build_model(make_cfg(name, kwargs), num_classes=100).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 100)


def test_vit_backward_runs():
    model = build_model(make_cfg("local_var", {"window_size": 8}), num_classes=100).train()
    logits = model(torch.randn(1, 3, 224, 224))
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0]))
    loss.backward()
    assert sum(p.grad is not None for p in model.parameters()) > 0


def test_vanilla_uses_learned_cls_by_default():
    model = build_model(make_cfg("none"), num_classes=100).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        tokens = model.backbone.forward_features(x)
        expected_logits = model.head(tokens[:, 0, :])
        logits = model(x)
        patches, scores, cls, logits_with_scores = model.forward_with_scores(x)
    assert patches.shape == (1, model.num_patches, model.embed_dim)
    assert scores is None
    assert torch.allclose(cls, tokens[:, 0, :], atol=1e-6)
    assert torch.allclose(logits, expected_logits, atol=1e-6)
    assert torch.allclose(logits_with_scores, expected_logits, atol=1e-6)


def test_vanilla_mean_pool_legacy_mode():
    model = build_model(make_cfg("none", vanilla_pool="mean"), num_classes=100).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        _cls_token, patches = model.forward_tokens(x)
        expected_logits = model.head(patches.mean(dim=1))
        logits = model(x)
    assert torch.allclose(logits, expected_logits, atol=1e-6)


def test_tcig_gamma_is_fixed_by_default_for_hard_topk():
    model = build_model(make_cfg("tcig", {"kernel_size": 3, "init_W": 2.0}), num_classes=100)
    optimizer = build_optimizer(make_cfg("tcig", {"kernel_size": 3, "init_W": 2.0}), model)
    names = [group.get("group_name") for group in optimizer.param_groups]
    assert "tcig_gamma" not in names
    assert "W_gamma" not in dict(model.named_parameters())


def test_tcig_gamma_param_group_when_explicitly_learnable():
    cfg = make_cfg("tcig", {"kernel_size": 3, "init_W": 2.0, "learnable_gamma": True})
    model = build_model(cfg, num_classes=100)
    optimizer = build_optimizer(cfg, model)
    names = [group.get("group_name") for group in optimizer.param_groups]
    assert "tcig_gamma" in names
    gamma_group = next(group for group in optimizer.param_groups if group.get("group_name") == "tcig_gamma")
    assert abs(gamma_group["lr"] - 5e-5) < 1e-12
