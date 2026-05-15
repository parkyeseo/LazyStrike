# Lazy Strike

Implementation for the ViT stability score comparison study described in
`../Materials/UADL_code_spec_v1.md`.

The code keeps the LaSt-ViT aggregation interface fixed and swaps only the
stability score module across the four cells:

- FFT: global + indirect
- Global Variance: global/order-invariant + direct
- TCIG: local sliding + indirect
- Local Window Variance: local fixed + direct

Default paths are set for this server:

```text
raw ImageNet archives: /mnt/newdisk/yeseo_item/imagenet/raw
prepared ImageNet-1k:  /mnt/newdisk/yeseo_item/imagenet/full
ImageNet-100 subset:   /mnt/newdisk/yeseo_item/imagenet-100
class list:            /mnt/newdisk/yeseo_item/UADL_data/imagenet100_classes.txt
checkpoints:           /mnt/newdisk/yeseo_item/UADL_checkpoints
```

Quick checks:

```bash
pytest tests/ -x -v
```

Prepare data after downloads finish:

```bash
python scripts/prepare_imagenet.py \
  --raw-root /mnt/newdisk/yeseo_item/imagenet/raw \
  --out-root /mnt/newdisk/yeseo_item/imagenet/full

python scripts/build_imagenet100.py \
  --imagenet-root /mnt/newdisk/yeseo_item/imagenet/full \
  --out-root /mnt/newdisk/yeseo_item/imagenet-100 \
  --classes-txt /mnt/newdisk/yeseo_item/UADL_data/imagenet100_classes.txt
```

4-GPU DDP example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/train.py \
  --config configs/cell/local_var.yaml \
  data.batch_size_per_gpu=256
```

