import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from lazystrike.eval.visualize import official_display_image, token_selection_count, upsample_patch_values  # noqa: E402


def test_official_display_image_resizes_and_center_crops_landscape():
    image = Image.new("RGB", (500, 368))
    display = official_display_image(image, image_size=224, resize_short=256)
    assert display.size == (224, 224)


def test_token_selection_count_matches_topk_channel_votes():
    scores = torch.zeros(1, 4, 3)
    scores[:, 0, 0] = 5
    scores[:, 1, 1] = 5
    scores[:, 2, 2] = 5
    counts = token_selection_count(scores, k=1)
    assert counts.shape == (1, 4)
    assert torch.equal(counts, torch.tensor([[1.0, 1.0, 1.0, 0.0]]))


def test_token_selection_count_sums_to_k_times_channels():
    scores = torch.randn(2, 10, 7)
    counts = token_selection_count(scores, k=3)
    assert counts.shape == (2, 10)
    assert torch.equal(counts.sum(dim=1), torch.full((2,), 21.0))


def test_upsample_patch_values_preserves_raw_range():
    values = torch.tensor([0.25, 0.5, 0.75, 1.0])
    upsampled = upsample_patch_values(values, grid=2, image_size=4)
    assert upsampled.shape == (4, 4)
    assert upsampled.min() >= 0.25
    assert upsampled.max() <= 1.0
