#!/usr/bin/env bash
set -euo pipefail

TIER=${1:-tier1}
NGPU=${NGPU:-4}
DATA_ROOT=${DATA_ROOT:-data/imagenet-100}
IMAGENET_FULL=${IMAGENET_FULL:-data/imagenet/full}
PIB_OUT=${PIB_OUT:-artifacts/eval/pib}
CKPT_ROOT=${CKPT_ROOT:-artifacts/checkpoints}
PIB_SCORE_METHODS=${PIB_SCORE_METHODS:-"patch_score_qcls patch_score_mean"}

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

run_pib() {
  mkdir -p "${PIB_OUT}"
  read -r -a pib_methods <<< "${PIB_SCORE_METHODS}"
  local method_slug
  method_slug="$(printf '%s_' "${pib_methods[@]}")"
  method_slug="${method_slug%_}"
  while IFS= read -r ck; do
    run_name="$(basename "$(dirname "${ck}")")"
    echo "[pib] ${run_name} (${PIB_SCORE_METHODS})"
    python scripts/eval_pib.py \
      --ckpt "${ck}" \
      --imagenet-root "${IMAGENET_FULL}" \
      --score-methods "${pib_methods[@]}" \
      --output-json "${PIB_OUT}/${run_name}_${method_slug}.json"
  done < <(find "${CKPT_ROOT}" -path '*/E1_*/best.pth' -print | sort)
}

if [[ "${TIER}" == "pib" ]]; then
  run_pib
  exit 0
fi

if [[ "${TIER}" == "tier1" || "${TIER}" == "all" ]]; then
  run configs/cell/_vanilla.yaml
  run configs/cell/fft.yaml
  run configs/cell/global_var.yaml
  run configs/cell/tcig.yaml
  run configs/cell/tasc.yaml
  run configs/cell/dual_guard.yaml
  run configs/cell/local_var.yaml

  while IFS= read -r ck; do
    python scripts/eval_permutation.py --ckpt "${ck}" --num-perms 5
  done < <(find "${CKPT_ROOT}" -path '*/E1_*/best.pth' -print | sort)
  run_pib
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
fi
