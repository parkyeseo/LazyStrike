# LazyStrike code specification

This document describes the public `v1.0.0` implementation. The final paper in `paper/main.tex` is the scientific source of truth; this specification maps its definitions to concrete modules, tensor shapes, configuration keys, and tests.

## 1. Runtime contract

- Python: `>=3.10,<3.12`
- Reference stack: PyTorch 2.1, torchvision 0.16, timm 0.9.16
- Primary input: RGB image tensor `[B, 3, H, W]`
- ViT patch tensor: `x_patch` with shape `[B, N, D]`
- Stability score tensor: `scores` with shape `[B, N, D]`
- Convention: a larger score means a more stable patch-channel entry and is preferred by top-K.

For ViT-S/16 at 224x224, `N=196` and `D=384`.

## 2. Forward data flow

1. `ViTWithStabilityScore.forward_tokens()` calls the timm backbone and separates the optional CLS token from patch tokens.
2. `build_score()` constructs the configured score module from `SCORE_REGISTRY`.
3. The score module maps `[B, N, D] -> [B, N, D]`.
4. `TopKChannelAggregator` performs top-K over the patch axis independently for each channel.
5. The selected values are averaged over K, producing a replacement CLS representation `[B, D]`.
6. The linear head maps `[B, D] -> [B, C]`.

If `model.score` is null, the model uses the original CLS token or patch mean according to `model.vanilla_pool`.

## 3. Stability scores

All implementations are in `lazystrike/models/stability_scores.py`. `EPS = 1e-6` is used for numerical stability.

### 3.1 FFT

Registry name: `fft`

```text
x_hat = IFFT(FFT(x) * gaussian_kernel)
score = x_hat / (abs(x_hat - x) + eps)
```

- FFT computation is promoted to FP32 under AMP.
- The Gaussian width defaults to `sqrt(D)`.
- The score depends on the ordering of the channel axis.

Regression test: `test_fft_matches_manual_formula`.

### 3.2 TCIG

Registry name: `tcig`

```text
x_hat = AvgPool1D(replicate_pad(x), kernel_size=k)
ratio = abs(abs(x) - abs(x_hat)) / (abs(x) + abs(x_hat) + eps)
score = exp(-ratio)
```

- The paper configuration uses `kernel_size=7`.
- Output is in `(0, 1]`.
- The score depends on local channel ordering.
- Historical configs may contain `init_W` and `learnable_gamma`. They are accepted so existing checkpoint configs can be loaded, but do not change the paper score.

Regression tests: `test_tcig_output_range`, `test_tcig_matches_paper_formula`.

### 3.3 Global Variance

Registry name: `global_var`

```text
mu = mean(x, channel)
sigma = std(x, channel, unbiased=False)
score = -(x - mu) / (sigma + eps)
```

- The definition is permutation equivariant over channels; top-K patch selection remains invariant under a consistent channel permutation.
- The signed score sums to approximately zero across channels. This is the structural degeneracy discussed for raw-score PiB in the paper.

Regression tests: `test_local_var_wD_equals_global`, `test_global_var_permutation_invariance`.

### 3.4 Local Window Variance

Registry name: `local_var`

- Split D into non-overlapping windows of width `window_size`.
- Apply the Global Variance definition within each window.
- `D % window_size` must equal zero.
- `window_size=D` exactly recovers Global Variance.

### 3.5 TASC

Registry name: `tasc`

```text
L1 = mean(abs(x), channel)
alpha = norm(x, p=1) / (norm(x, p=2) + eps)
threshold = L1 * alpha
x_hat = sign(x) * min(abs(x), threshold)
score = abs(x_hat)^2 / (abs(x) + eps)
```

- TASC is permutation equivariant over channels.
- There are no learnable parameters in the paper score.
- Historical gamma fields are retained only for checkpoint-config compatibility.

Regression test: `test_tasc_permutation_invariance`.

### 3.6 Dual Guard

Registry name: `dual_guard`

Dual Guard is an exploratory module, not one of the four principal paper comparisons. It combines small- and large-window local smoothing with TASC-style spike suppression. Its default windows are 7 and 21.

## 4. Aggregators

Implementations are in `lazystrike/models/aggregator.py`.

### TopKChannelAggregator

Registry name: `topk_channel`

- Input patches and scores: `[B, N, D]`
- `torch.topk(..., dim=1)` selects K patches independently for each channel.
- Gathered patch entries have shape `[B, K, D]`.
- Mean over K returns `[B, D]`.
- `vote_count()` scatters the per-channel selections into `[B, N]` patch vote totals.

### TopKPatchAggregator

Registry name: `topk_patch`

This ablation first reduces scores over D by mean or max, then selects K complete patch vectors. It is not the main-paper aggregation path.

## 5. Model construction and checkpoints

`lazystrike/models/vit.py` owns model construction.

- `model.backbone` selects a timm model.
- `model.pretrained` initializes the encoder from timm.
- `model.pretrained_head` optionally copies all or selected rows of the ImageNet-1K classifier.
- `model.init_ckpt` loads an initialization checkpoint before training.
- Evaluation scripts explicitly clear all three initialization fields before constructing a model, then restore the saved state with `strict=True`. This prevents downloads or accidental reinitialization during evaluation.

Training checkpoints contain:

```text
model       state_dict
optimizer   AdamW state
scheduler   scheduler state
epoch       zero-based epoch index
metrics     top1, top5, validation loss
cfg         resolved OmegaConf container
```

## 6. Configuration composition

`scripts/train.py` merges in order:

1. `configs/base.yaml`
2. the file passed to `--config`
3. command-line dot-list overrides

Example:

```bash
python scripts/train.py \
  --config configs/cell/tasc.yaml \
  data.root=/datasets/imagenet-100 \
  model.aggregator.kwargs.K=7 \
  logging.use_wandb=false
```

Main cells are under `configs/cell/`; controlled ablations are under `configs/ablation/`.

## 7. Data and evaluation

- `lazystrike/data/imagenet100.py`: ImageFolder datasets and distributed loaders.
- `lazystrike/data/pib_dataset.py`: ImageNet bounding-box subset used for Point-in-Box.
- `lazystrike/data/transforms.py`: train and validation preprocessing.
- `lazystrike/eval/pib.py`: raw score, vote count, pooled / QCLS, encoder CLS, and prototype score paths.
- `lazystrike/eval/permutation.py`: permutation helpers used by the CLI and tests.
- `lazystrike/eval/visualize.py`: patch-grid visualization.

The paper reports three PiB variants:

- `raw_score`: sum the score tensor over D, then rank patches.
- `vote_count`: count per-channel top-K selections.
- `patch_score_mean` / pooled: compare patches with the aggregated representation; treated as an auxiliary metric in the paper.

## 8. Validation matrix

Run:

```bash
pytest tests/ -x -v
python scripts/smoke_verify.py
```

The test suite covers:

- score shape and finiteness;
- FFT and TCIG paper-formula regressions;
- Global Variance / Local Variance equivalence at `w=D`;
- Global Variance and TASC permutation behavior;
- channel-wise top-K and vote-count semantics;
- PiB scoring and empty-set behavior;
- ViT forward/backward integration;
- trainer checkpoint and best-metric behavior;
- visualization output geometry.

## 9. Public-artifact boundaries

- ImageNet data and pretrained checkpoints are not bundled.
- Host-specific paths are not required; defaults are relative and can be overridden.
- `logs/`, `wandb/`, checkpoints, and generated artifacts are ignored by Git.
- The paper PDF is intentionally tracked as a release artifact; LaTeX auxiliary files are ignored.
