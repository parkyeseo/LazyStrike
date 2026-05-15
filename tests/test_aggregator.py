import pytest

torch = pytest.importorskip("torch")

from lazystrike.models.aggregator import TopKChannelAggregator, TopKPatchAggregator  # noqa: E402


def test_topk_channel_shape():
    x = torch.randn(2, 196, 384)
    scores = torch.randn(2, 196, 384)
    cls = TopKChannelAggregator(K=98)(x, scores)
    assert cls.shape == (2, 384)


def test_topk_channel_correctness():
    x = torch.arange(1 * 5 * 4, dtype=torch.float32).reshape(1, 5, 4)
    scores = torch.zeros(1, 5, 4)
    scores[:, :3, :] = 1.0
    cls = TopKChannelAggregator(K=3)(x, scores)
    expected = x[:, :3, :].mean(dim=1)
    assert torch.allclose(cls, expected)


def test_topk_patch_shape():
    x = torch.randn(2, 196, 384)
    scores = torch.randn(2, 196, 384)
    cls = TopKPatchAggregator(K=98, channel_aggregator="mean")(x, scores)
    assert cls.shape == (2, 384)


def test_vote_count_sum():
    scores = torch.randn(2, 10, 7)
    votes = TopKChannelAggregator(K=3).vote_count(scores)
    assert votes.shape == (2, 10)
    assert torch.equal(votes.sum(dim=1), torch.full((2,), 3 * 7, dtype=torch.long))

