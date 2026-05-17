import pytest

torch = pytest.importorskip("torch")

from lazystrike.eval.pib import bbox_to_patch_set, evaluate_pib, shi_patch_score  # noqa: E402


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

    def forward_with_scores(self, images):
        batch = images.shape[0]
        patches = torch.zeros(batch, 196, 4)
        scores = torch.zeros(batch, 196, 4)
        scores[:, 15, :] = 10.0
        cls = torch.zeros(batch, 4)
        logits = torch.zeros(batch, 1)
        return patches, scores, cls, logits


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
