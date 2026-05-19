import pytest

torch = pytest.importorskip("torch")

from lazystrike.models.stability_scores import (  # noqa: E402
    FFTScore,
    GlobalVarianceScore,
    LocalWindowVarianceScore,
    TCIGScore,
)

B, N, D = 2, 196, 384


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(B, N, D)


@pytest.mark.parametrize(
    "score_cls,kwargs",
    [
        (FFTScore, {"dim": D}),
        (GlobalVarianceScore, {"dim": D}),
        (TCIGScore, {"dim": D, "kernel_size": 3}),
        (LocalWindowVarianceScore, {"dim": D, "window_size": 8}),
    ],
)
def test_score_shape_and_finiteness(x, score_cls, kwargs):
    module = score_cls(**kwargs).eval()
    with torch.no_grad():
        scores = module(x)
    assert scores.shape == (B, N, D)
    assert torch.isfinite(scores).all()


def test_local_var_wD_equals_global(x):
    global_var = GlobalVarianceScore(dim=D)
    local_var = LocalWindowVarianceScore(dim=D, window_size=D)
    with torch.no_grad():
        expected = global_var(x)
        actual = local_var(x)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_global_var_permutation_invariance(x):
    module = GlobalVarianceScore(dim=D)
    perm = torch.randperm(D)
    inv = torch.argsort(perm)
    with torch.no_grad():
        original = module(x)
        permuted = module(x[:, :, perm])[:, :, inv]
    assert torch.allclose(original, permuted, atol=1e-5)


def test_tcig_gamma_bound():
    module = TCIGScore(dim=D, init_W=-5.0)
    assert float(module.gamma.item()) >= 1.0


def test_tcig_output_range(x):
    module = TCIGScore(dim=D, kernel_size=3, init_W=2.0)
    with torch.no_grad():
        scores = module(x)
    assert bool((scores > 0).all())
    assert bool((scores <= 1.0 + 1e-6).all())


def test_fft_matches_manual_formula(x):
    module = FFTScore(dim=D, sigma=24.0)
    with torch.no_grad():
        scores = module(x)
        x_fft = torch.fft.fft(x.float(), dim=-1)
        x_fft = torch.fft.fftshift(x_fft, dim=-1) * module.gs_k
        x_fft = torch.fft.ifftshift(x_fft, dim=-1)
        x_hat = torch.fft.ifft(x_fft, dim=-1).real
        expected = x.float() / (torch.abs(x_hat - x.float()) + 1e-6)
    assert torch.allclose(scores, expected.to(scores.dtype), atol=1e-5)
