#!/usr/bin/env bash
set -euo pipefail

TIER=${1:-tier1}
NGPU=${NGPU:-4}
DATA_ROOT=${DATA_ROOT:-/mnt/newdisk/yeseo_item/imagenet-100}
IMAGENET_FULL=${IMAGENET_FULL:-/mnt/newdisk/yeseo_item/imagenet/full}
PIB_OUT=${PIB_OUT:-/mnt/newdisk/yeseo_item/UADL_eval/pib}

run() {
  local cfg="$1"; shift || true
  echo "[launch] $cfg $*"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} torchrun --standalone --nproc_per_node="${NGPU}" \
    scripts/train.py \
    --config "${cfg}" \
    data.root="${DATA_ROOT}" \
    data.batch_size_per_gpu=$((1024 / NGPU)) \
    "$@"
}

if [[ "${TIER}" == "tier1" || "${TIER}" == "all" ]]; then
  run configs/cell/_vanilla.yaml
  run configs/cell/fft.yaml
  run configs/cell/global_var.yaml
  run configs/cell/tcig.yaml
  run configs/cell/local_var.yaml

  for ck in /mnt/newdisk/yeseo_item/UADL_checkpoints/E1_*/best.pth; do
    [[ -f "${ck}" ]] || continue
    run_name="$(basename "$(dirname "${ck}")")"
    mkdir -p "${PIB_OUT}"
    python scripts/eval_permutation.py --ckpt "${ck}" --num-perms 5
    python scripts/eval_pib.py \
      --ckpt "${ck}" \
      --imagenet-root "${IMAGENET_FULL}" \
      --score-method patch_score_shi \
      --output-json "${PIB_OUT}/${run_name}_patch_score_shi.json"
  done
fi

if [[ "${TIER}" == "tier2" || "${TIER}" == "all" ]]; then
  run configs/ablation/E5_w4.yaml
  run configs/ablation/E5_w16.yaml
  run configs/ablation/E5_w32.yaml
  run configs/ablation/E3_patch_topk.yaml
  run configs/ablation/E6_K49.yaml
  run configs/ablation/E6_K147.yaml
fi

if [[ "${TIER}" == "tier3" || "${TIER}" == "all" ]]; then
  run configs/cell/tcig.yaml model.score.kwargs.kernel_size=5 logging.run_name=tcig_k5
  run configs/cell/tcig.yaml model.score.kwargs.kernel_size=7 logging.run_name=tcig_k7
  run configs/cell/tcig.yaml train.gamma_lr_scale=1.0 logging.run_name=tcig_gamma_lr1x
fi
