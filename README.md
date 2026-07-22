# LazyStrike

LazyStrike is a compact reproduction framework for studying high-norm artifacts and lazy aggregation in Vision Transformers. It keeps the ViT-S/16 backbone and channel-wise top-K aggregation fixed, then compares only the patch-stability score: FFT, TCIG, Global/Local Variance, and TASC.

TASC measures sparse channel activation spikes with an L1/L2-based score. On ImageNet-100 with K=7, it raises raw-score PiB from 28.42% (FFT) to 65.31% and vote-count PiB from 22.74% to 57.19%, while maintaining comparable classification accuracy.

[Download the paper (PDF)](https://github.com/parkyeseo/LazyStrike/raw/refs/heads/main/Beyond%20FFT%20-%20Frequency%20vs.%20Variance%20vs.%20Sparsity.pdf)

## Setup

```bash
conda env create -f environment.yml
conda activate lazy-strike
pip install -e .
```

## Data

Place the legally obtained ImageNet archives under `data/imagenet/raw`, then run:

```bash
python scripts/prepare_imagenet.py \
  --raw-root data/imagenet/raw \
  --out-root data/imagenet/full

python scripts/build_imagenet100.py \
  --imagenet-root data/imagenet/full \
  --out-root data/imagenet-100 \
  --classes-txt data/imagenet100_classes.txt
```

## Train

```bash
python scripts/train.py \
  --config configs/cell/tasc.yaml \
  data.root=data/imagenet-100 \
  paths.checkpoint_root=artifacts/checkpoints \
  logging.use_wandb=false
```

Available paper configurations are `fft.yaml`, `tcig.yaml`, `global_var.yaml`, `local_var.yaml`, `tasc.yaml`, and `_vanilla.yaml` under `configs/cell/`.

## Evaluate

```bash
python scripts/eval_classification.py \
  --ckpt artifacts/checkpoints/100ep/E1_tasc/best.pth

python scripts/eval_pib.py \
  --ckpt artifacts/checkpoints/100ep/E1_tasc/best.pth \
  --imagenet-root data/imagenet/full \
  --classes-txt data/imagenet100_classes.txt \
  --score-methods raw_score vote_count patch_score_mean \
  --output-json artifacts/eval/pib/E1_tasc.json

python scripts/eval_permutation.py \
  --ckpt artifacts/checkpoints/100ep/E1_tasc/best.pth \
  --num-perms 5 \
  --protocol score_only
```