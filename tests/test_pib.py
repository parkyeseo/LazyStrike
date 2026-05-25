import pytest

torch = pytest.importorskip("torch")

from lazystrike.eval.pib import (  # noqa: E402
    bbox_to_patch_set,
    evaluate_pib,
    patch_scores_from_model_output,
    shi_patch_score,
)


def test_bbox_to_patch_set_landscape_matches_last_vit_geometry():
    patch_set = bbox_to_patch_set(
        bbox_xyxy=(105, 204, 358, 279),
        original_width=500,
        original_height=368,
        image_size=224,
        patch_size=16,
        resize_short=256,
    )
    expected = {py * 14 + px for py in range(7, 12) for px in range(0, 12)}
    assert patch_set == expected


def test_bbox_to_patch_set_portrait_matches_last_vit_geometry():
    patch_set = bbox_to_patch_set(
        bbox_xyxy=(50, 250, 250, 400),
        original_width=300,
        original_height=600,
        image_size=224,
        patch_size=16,
        resize_short=256,
    )
    expected = {py * 14 + px for py in range(4, 13) for px in range(1, 13)}
    assert patch_set == expected


def test_shi_patch_score_shape():
    patches = torch.randn(2, 196, 384)
    cls = torch.randn(2, 384)
    scores = shi_patch_score(patches, cls)
    assert scores.shape == (2, 196)
    assert torch.isfinite(scores).all()


class DummyModel:
    aggregator = None

    def eval(self):
        return self

    def forward_tokens(self, images):
        batch = images.shape[0]
        patches = torch.zeros(batch, 196, 4)
        cls = torch.ones(batch, 4)
        return cls, patches

    def aggregate(self, patches, cls):
        batch = patches.shape[0]
        scores = torch.zeros(batch, 196, 4)
        scores[:, 15, :] = 10.0
        return cls, scores


def test_evaluate_pib_uses_patch_overlap_set():
    batch = {
        "image": torch.zeros(1, 3, 224, 224),
        "bbox": torch.tensor([[16.0, 16.0, 16.0, 16.0]]),
        "image_width": torch.tensor([224]),
        "image_height": torch.tensor([224]),
    }
    result = evaluate_pib(
        DummyModel(),
        [batch],
        torch.device("cpu"),
        score_method="raw_score",
        image_size=224,
        patch_size=16,
        resize_short=224,
    )
    assert result.total == 1
    assert result.hits == 1
    assert result.pib == 1.0


def test_patch_score_shi_uses_encoder_cls_not_pooled_cls():
    patches = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    encoder_cls = torch.tensor([[1.0, 0.0]])
    pooled_cls = torch.tensor([[0.0, 1.0]])
    scores = patch_scores_from_model_output(
        model=DummyModel(),
        patches=patches,
        scores=None,
        pooled_cls=pooled_cls,
        encoder_cls=encoder_cls,
        score_method="patch_score_shi",
    )
    assert int(scores.argmax(dim=1).item()) == 0


def test_patch_score_pooled_uses_final_representation():
    patches = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    encoder_cls = torch.tensor([[1.0, 0.0]])
    pooled_cls = torch.tensor([[0.0, 1.0]])
    scores = patch_scores_from_model_output(
        model=DummyModel(),
        patches=patches,
        scores=None,
        pooled_cls=pooled_cls,
        encoder_cls=encoder_cls,
        score_method="patch_score_pooled",
    )
    assert int(scores.argmax(dim=1).item()) == 1


def test_patch_score_qcls_alias_uses_final_representation():
    patches = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    encoder_cls = torch.tensor([[1.0, 0.0]])
    qcls = torch.tensor([[0.0, 1.0]])
    scores = patch_scores_from_model_output(
        model=DummyModel(),
        patches=patches,
        scores=None,
        pooled_cls=qcls,
        encoder_cls=encoder_cls,
        score_method="patch_score_qcls",
    )
    assert int(scores.argmax(dim=1).item()) == 1


def test_patch_score_mean_uses_patch_average():
    patches = torch.tensor([[[2.0, 0.0], [0.0, 1.0]]])
    scores = patch_scores_from_model_output(
        model=DummyModel(),
        patches=patches,
        scores=None,
        pooled_cls=torch.tensor([[0.0, 1.0]]),
        encoder_cls=torch.tensor([[0.0, 1.0]]),
        score_method="patch_score_mean",
    )
    assert int(scores.argmax(dim=1).item()) == 0


def test_evaluate_pib_multiple_methods():
    batch = {
        "image": torch.zeros(1, 3, 224, 224),
        "bbox": torch.tensor([[16.0, 16.0, 16.0, 16.0]]),
        "image_width": torch.tensor([224]),
        "image_height": torch.tensor([224]),
    }
    result = evaluate_pib(
        DummyModel(),
        [batch],
        torch.device("cpu"),
        score_methods=["raw_score", "patch_score_qcls"],
        image_size=224,
        patch_size=16,
        resize_short=224,
    )
    assert result.total == 1
    assert result.hits_by_method["raw_score"] == 1
    assert result.pibs["raw_score"] == 1.0
