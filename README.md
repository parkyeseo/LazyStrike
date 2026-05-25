# Lazy Strike

Implementation for the ViT stability score comparison study described in
`../Materials/UADL_code_spec_v1.md`.

The code keeps the LaSt-ViT aggregation interface fixed and swaps only the
stability score module across the cells:

- FFT: global + indirect
- Global Variance: global/order-invariant + direct
- TCIG: local sliding + magnitude-aware
- TASC: channel spike clipping + magnitude-aware
- Dual Guard: multi-scale local smoothing + TASC spike suppression
- Local Window Variance: local fixed + direct

Default paths are set for this server:

```text
raw ImageNet archives: /home/yeseo_item/data_14T/UADL/data/imagenet/raw
prepared ImageNet-1k:  /home/yeseo_item/data_14T/UADL/data/imagenet/full
ImageNet-100 subset:   /home/yeseo_item/data_14T/UADL/data/imagenet-100
class list:            /home/yeseo_item/data_14T/UADL/data/metadata/imagenet100_classes.txt
checkpoints:           /home/yeseo_item/data_14T/UADL/checkpoints
wandb local files:     /home/yeseo_item/data_14T/UADL/wandb
```

Quick checks:

```bash
pytest tests/ -x -v
```

Prepare data after downloads finish:

```bash
python scripts/prepare_imagenet.py \
  --raw-root /home/yeseo_item/data_14T/UADL/data/imagenet/raw \
  --out-root /home/yeseo_item/data_14T/UADL/data/imagenet/full

python scripts/build_imagenet100.py \
  --imagenet-root /home/yeseo_item/data_14T/UADL/data/imagenet/full \
  --out-root /home/yeseo_item/data_14T/UADL/data/imagenet-100 \
  --classes-txt /home/yeseo_item/data_14T/UADL/data/metadata/imagenet100_classes.txt
```

4-GPU DDP example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/train.py \
  --config configs/cell/local_var.yaml \
  data.batch_size_per_gpu=256
```

PiB evaluation defaults to the paper-facing QCLS patch score plus the patch-mean
auxiliary score:

```bash
python scripts/eval_pib.py \
  --ckpt /home/yeseo_item/data_14T/UADL/checkpoints/100ep/E1_tcig/best.pth \
  --score-methods patch_score_qcls patch_score_mean
```

ImageNet-1K fine-tuning keeps the pretrained classifier head when
`model.pretrained_head=true`:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --config configs/cell/tcig_in1k.yaml \
  data.batch_size_per_gpu=256

python scripts/eval_pib.py \
  --ckpt /home/yeseo_item/data_14T/UADL/checkpoints/100ep/E1k_tcig/best.pth \
  --all-classes \
  --score-methods patch_score_qcls patch_score_mean
```
