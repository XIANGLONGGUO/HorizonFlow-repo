#!/usr/bin/env bash
# Train HorizonFlow on Wan2.1-T2V-1.3B: 600 iterations, 4-step student, global batch = number of GPUs x 2
# (the paper uses 7 GPUs -> 14 videos per iteration, 8.4k samples in total).
#
#   GPUS=0,1,2,3,4,5,6 bash scripts/train_horizonflow.sh [extra config overrides]
#
# Environment variables (all optional):
#   GPUS         comma-separated GPU ids                        (default 0,1,2,3,4,5,6)
#   CKPT         directory with the Wan2.1 files (see README)   (default assets/checkpoints)
#   DATA         webdataset shard pattern of the training data  (default: the rCM synthetic set in assets/datasets)
#   OUTPUT_ROOT  where checkpoints and logs are written         (default ./outputs)
#   NAME         run name                                       (default horizonflow)
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3,4,5,6}
NGPU=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
CKPT=${CKPT:-assets/checkpoints}
DATA=${DATA:-assets/datasets/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard*.tar}

export PYTHONPATH=${PWD}
export IMAGINAIRE_OUTPUT_ROOT=${OUTPUT_ROOT:-${PWD}/outputs}
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200  # the first iteration compiles kernels and can take long

torchrun --standalone --nproc_per_node=${NGPU} -m scripts.train --config=rcm/configs/registry_distill.py -- \
  experiment=horizonflow_wan2pt1_1pt3B_t2v \
  job.name=${NAME:-horizonflow} \
  model.config.fsdp_shard_size=${NGPU} \
  model.config.teacher_ckpt=${CKPT}/Wan2.1-T2V-1.3B.dcp \
  model.config.tokenizer.vae_pth=${CKPT}/Wan2.1_VAE.pth \
  model.config.text_encoder_path=${CKPT}/models_t5_umt5-xxl-enc-bf16.pth \
  model.config.neg_embed_path=${CKPT}/umT5_wan_negative_emb.pt \
  "dataloader_train.tar_path_pattern='${DATA}'" \
  "$@"
