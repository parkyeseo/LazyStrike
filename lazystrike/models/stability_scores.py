from __future__ import annotations

from typing import Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


class StabilityScore(nn.Module):
    """Base interface for patch-channel stability scores.

    Input:
        x: Tensor[B, N, D]

    Output:
        scores: Tensor[B, N, D], where larger means more stable.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FFTScore(StabilityScore):
    """Cell 1: global + indirect LaSt-ViT style FFT score."""

    def __init__(self, dim: int, sigma: float | None = None, kernel_mode: str = "gaussian"):
        super().__init__()
        if kernel_mode != "gaussian":
            raise ValueError(f"Unsupported kernel_mode={kernel_mode!r}")
        self.dim = int(dim)
        self.sigma = float(sigma if sigma is not None else math.sqrt(dim))
        self.register_buffer(
            "gs_k",
            self._gaussian_kernel_1d(self.dim, self.sigma),
            persistent=False,
        )

    @staticmethod
    def _gaussian_kernel_1d(dim: int, sigma: float) -> torch.Tensor:
        idx = torch.arange(-dim // 2 + 1, dim // 2 + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (idx / float(sigma)) ** 2)
        return kernel / kernel.max()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        kernel = self.gs_k.to(device=x_fp32.device, dtype=x_fp32.dtype)
        x_fft = torch.fft.fft(x_fp32, dim=-1)
        x_fft = torch.fft.fftshift(x_fft, dim=-1)
        x_fft = x_fft * kernel
        x_fft = torch.fft.ifftshift(x_fft, dim=-1)
        x_hat = torch.fft.ifft(x_fft, dim=-1).real
        # Official LaSt-ViT uses the raw patch value as numerator:
        # diff = x_detach / abs(lowpass_fft(x_detach) - x_detach).
        scores = x_fp32 / (torch.abs(x_hat - x_fp32) + EPS)
        return scores.to(orig_dtype)


class GlobalVarianceScore(StabilityScore):
    """Cell 2: patch-wise negative z-score over all channels."""

    def __init__(self, dim: int | None = None):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False) + EPS
        return -(x - mu) / std


class TCIGScore(StabilityScore):
    """Cell 3: magnitude-aware local smoothing score.

    The score keeps channels whose local low-pass estimate remains large, while
    squaring the estimate suppresses high-frequency spikes more aggressively.
    ``init_W`` and ``learnable_gamma`` are accepted for config compatibility
    with earlier TCIG experiments, but the current hard top-K score does not use
    gamma.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        init_W: float = 2.0,
        learnable_gamma: bool = False,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd for symmetric padding")
        self.dim = int(dim)
        self.k = int(kernel_size)
        self.learnable_gamma = bool(learnable_gamma)
        raw_gamma = torch.tensor(float(init_W))
        if self.learnable_gamma:
            self.W_gamma = nn.Parameter(raw_gamma)
        else:
            self.register_buffer("W_gamma", raw_gamma)

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.W_gamma) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, patches, dim = x.shape
        x_flat = x.reshape(batch * patches, 1, dim)
        pad = self.k // 2
        x_pad = F.pad(x_flat, (pad, pad), mode="replicate")
        x_hat = F.avg_pool1d(x_pad, kernel_size=self.k, stride=1)
        x_hat = x_hat.reshape(batch, patches, dim)

        x_abs = x.abs()
        x_hat_abs = x_hat.abs()
        return x_hat_abs.pow(2) / (x_abs + EPS)


class LocalWindowVarianceScore(StabilityScore):
    """Cell 4: non-overlapping window-local negative z-score."""

    def __init__(self, dim: int, window_size: int = 8):
        super().__init__()
        dim = int(dim)
        window_size = int(window_size)
        if dim % window_size != 0:
            raise ValueError(f"dim ({dim}) must be divisible by window_size ({window_size})")
        self.D = dim
        self.w = window_size
        self.G = dim // window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, patches, dim = x.shape
        if dim != self.D:
            raise RuntimeError(f"Expected channel dim {self.D}, got {dim}")
        x_grouped = x.reshape(batch, patches, self.G, self.w)
        mu = x_grouped.mean(dim=-1, keepdim=True)
        std = x_grouped.std(dim=-1, keepdim=True, unbiased=False) + EPS
        scores = -((x_grouped - mu) / std)
        return scores.reshape(batch, patches, dim)


class TASCScore(StabilityScore):
    """Cell 5: channel-spike clipping score with magnitude awareness."""

    def __init__(
        self,
        dim: int,
        init_W: float = 2.0,
        learnable_gamma: bool = False,
    ):
        super().__init__()
        self.dim = int(dim)
        self.learnable_gamma = bool(learnable_gamma)
        raw_gamma = torch.tensor(float(init_W))
        if self.learnable_gamma:
            self.W_gamma = nn.Parameter(raw_gamma)
        else:
            self.register_buffer("W_gamma", raw_gamma)

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.W_gamma) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.abs().mean(dim=-1, keepdim=True)
        alpha = x.norm(p=1, dim=-1, keepdim=True) / (x.norm(p=2, dim=-1, keepdim=True) + EPS)
        x_hat = x.sign() * torch.min(x.abs(), mu * alpha)
        return x_hat.abs().pow(2) / (x.abs() + EPS)


class DualGuardScore(StabilityScore):
    """Cell 6: multi-scale local smoothing fused with TASC spike suppression."""

    def __init__(self, dim: int, k_small: int = 7, k_large: int = 21):
        super().__init__()
        self.dim = int(dim)
        self.k_small = int(k_small)
        self.k_large = int(k_large)
        if self.k_small % 2 != 1 or self.k_large % 2 != 1:
            raise ValueError("Kernel sizes must be odd")

    def _local_ratio(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        batch, patches, dim = x.shape
        x_flat = x.reshape(batch * patches, 1, dim)
        pad = kernel_size // 2
        x_pad = F.pad(x_flat, (pad, pad), mode="replicate")
        x_hat = F.avg_pool1d(x_pad, kernel_size=kernel_size, stride=1).reshape(batch, patches, dim)
        return (x_hat.abs() / (x.abs() + EPS)).clamp(max=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ratio_small = self._local_ratio(x, self.k_small)
        ratio_large = self._local_ratio(x, self.k_large)
        ratio_local = torch.min(ratio_small, ratio_large)

        mu = x.abs().mean(dim=-1, keepdim=True)
        alpha = x.norm(p=1, dim=-1, keepdim=True) / (x.norm(p=2, dim=-1, keepdim=True) + EPS)
        x_hat_spike = torch.min(x.abs(), mu * alpha)
        ratio_spike = x_hat_spike / (x.abs() + EPS)

        return ratio_local.pow(2) * ratio_spike * x.abs()


SCORE_REGISTRY: dict[str, type[StabilityScore]] = {
    "fft": FFTScore,
    "global_var": GlobalVarianceScore,
    "tcig": TCIGScore,
    "tasc": TASCScore,
    "local_var": LocalWindowVarianceScore,
    "dual_guard": DualGuardScore,
}


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return cfg.get(key, default)


def build_score(cfg: Any, dim: int) -> StabilityScore | None:
    if cfg is None:
        return None
    name = _cfg_get(cfg, "name")
    if name in (None, "none"):
        return None
    if name not in SCORE_REGISTRY:
        raise KeyError(f"Unknown score module {name!r}. Choices: {sorted(SCORE_REGISTRY)}")
    kwargs = dict(_cfg_get(cfg, "kwargs", {}) or {})
    return SCORE_REGISTRY[name](dim=dim, **kwargs)
