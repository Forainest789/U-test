#!/bin/bash
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -euo pipefail


export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-29628}"

cd "${REPO_DIR}"
if [[ -n "${CONDA_ENV:-}" ]]; then source activate "${CONDA_ENV}"; fi
CKPT_DIR="${CKPT_DIR:-/models/Wan2.2-I2V-A14B}"
: "${DATA_ROOT:?Set DATA_ROOT to the dataset directory}"
if [[ -f "${DATA_ROOT}/candidate_groups.csv" ]]; then
  DEFAULT_CANDIDATE_CSV="${DATA_ROOT}/candidate_groups.csv"
else
  DEFAULT_CANDIDATE_CSV="${DATA_ROOT}/caption/candidate_groups.csv"
fi
if [[ -d "${DATA_ROOT}/character_lists" ]]; then
  DEFAULT_CHARACTER_LISTS_DIR="${DATA_ROOT}/character_lists"
else
  DEFAULT_CHARACTER_LISTS_DIR="${DATA_ROOT}/caption/character_lists"
fi
CANDIDATE_CSV="${CANDIDATE_CSV:-${DEFAULT_CANDIDATE_CSV}}"
CHARACTER_LISTS_DIR="${CHARACTER_LISTS_DIR:-${DEFAULT_CHARACTER_LISTS_DIR}}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/video}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/experiments/slotmem_stage1}"
EXP_PREFIX_BASE="${EXP_PREFIX_BASE:-slotmem_stage1}"
CHECKPOINT_SAVE_EVERY_N_STEPS="${CHECKPOINT_SAVE_EVERY_N_STEPS:-200}"
LOW_EXPERT_CKPT_PATH="${LOW_EXPERT_CKPT_PATH:-}"
HIGH_EXPERT_CKPT_PATH="${HIGH_EXPERT_CKPT_PATH:-}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-$(python3 - <<'PY'
import os
value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
parts = [x for x in value.split(",") if x.strip()]
print(len(parts) if parts else 1)
PY
)}"

USER_TRAIN_NOISE_DOMAIN="${TRAIN_NOISE_DOMAIN:-}"
RUN_DOMAINS="${RUN_DOMAINS:-${USER_TRAIN_NOISE_DOMAIN:-low_noise,high_noise}}"

GLOBAL_MAX_STEPS="${MAX_STEPS:-}"
LOW_NOISE_MAX_STEPS="${LOW_NOISE_MAX_STEPS:-${GLOBAL_MAX_STEPS:-1700}}"
HIGH_NOISE_MAX_STEPS="${HIGH_NOISE_MAX_STEPS:-${GLOBAL_MAX_STEPS:-1100}}"

mkdir -p "${OUTPUT_ROOT}"

common_args=(
  --ckpt_dir "${CKPT_DIR}"
  --story_root ""
  --candidate_groups_csv "${CANDIDATE_CSV}"
  --character_lists_dir "${CHARACTER_LISTS_DIR}"
  --video_root "${VIDEO_ROOT}"
  --max_epochs 100
  --num_train_gpus "${NUM_TRAIN_GPUS}"
  --num_nodes 1
  --train_architecture lora
  --lora_rank "${LORA_RANK:-128}"
  --lora_alpha "${LORA_ALPHA:-128}"
  --lora_target_modules "${LORA_TARGET_MODULES:-q,k,v,o,ffn.0,ffn.2}"
  --noise_domain_boundary_ratio 0.9
  --learning_rate "${LEARNING_RATE:-1e-4}"
  --latent_dim 16
  --patch_dim 5120
  --num_frames 81
  --ref_pad_cfg
  --num_overlap_frame 5
  --use_gradient_checkpointing
  --use_gradient_checkpointing_offload
  --aggressive_vram_optimization
  --training_strategy "${TRAINING_STRATEGY:-ddp}"
  --model_slice_mode "none"
  --tp_size 1
  --extract_single_timestep_align_train
  --use_train_weights_for_extract_and_probe
  --precompute_image_emb
  --precompute_image_emb_strict
  --offload_image_encoder_after_extraction
  --no-keep_image_encoder_on_gpu
  --memory_injection_mode "context_only"
  --sparse_role_memory_layer_idx 3
  --sparse_role_memory_injection_layers "${SPARSE_ROLE_MEMORY_INJECTION_LAYERS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
  --sparse_role_memory_num_heads 8
  --sparse_role_memory_head_dim 128
  --sparse_role_memory_rope_dim 256
  --sparse_role_memory_use_half_role_heads
  --sparse_role_memory_feature_source "attn_out"
  --sparse_role_memory_init_scale 0.1
  --sparse_role_memory_time_gate
  --role_token_selection_mode "${ROLE_TOKEN_SELECTION_MODE:-two_role_diff}"
  --extract_layers "${EXTRACT_LAYERS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
  --use_attn_score_selection
  --top_visual_tokens 0.1
  --token_weight 1
  --cfg_scale_extraction 5.0
  --max_memory_characters 2
  --neighbor_filter_kernel 5
  --max_memory_tokens_per_character 512
  --slotmem_memory_bank_mode single
  --prompt_drop_prob 0.0
  --memory_drop_prob 0.0
  --slotmem_memory_encoder_layers "${SLOTMEM_MEMORY_ENCODER_LAYERS:-0-15}"
  --slotmem_memory_encoder_layer_groups "${SLOTMEM_MEMORY_ENCODER_LAYER_GROUPS:-0-4,5-10,11-15}"
  --slotmem_memory_encoder_slots "${SLOTMEM_MEMORY_ENCODER_SLOTS:-64}"
  --slotmem_memory_encoder_dim "${SLOTMEM_MEMORY_ENCODER_DIM:-512}"
  --slotmem_memory_encoder_hidden_dim "${SLOTMEM_MEMORY_ENCODER_HIDDEN_DIM:-1024}"
  --slotmem_memory_encoder_use_t_embed
  --slotmem_memory_encoder_use_slot_index_embed
  --train_stage stage1
  --checkpoint_save_every_n_epochs 0
  --checkpoint_save_every_n_steps "${CHECKPOINT_SAVE_EVERY_N_STEPS}"
  --seed -1
  --perf_log_interval 0
)

max_steps_for_domain() {
  case "$1" in
    low_noise) printf '%s\n' "${LOW_NOISE_MAX_STEPS}" ;;
    high_noise) printf '%s\n' "${HIGH_NOISE_MAX_STEPS}" ;;
    *) echo "Unsupported train_noise_domain: $1" >&2; return 1 ;;
  esac
}

pretrained_for_domain() {
  case "$1" in
    low_noise) printf '%s\n' "${LOW_EXPERT_CKPT_PATH}" ;;
    high_noise) printf '%s\n' "${HIGH_EXPERT_CKPT_PATH}" ;;
    *) echo "Unsupported train_noise_domain: $1" >&2; return 1 ;;
  esac
}

run_phase() {
  local train_noise_domain="$1"
  local max_steps="$2"
  local pretrained_lora_path="${3:-}"
  local output_dir
  local exp_prefix

  if [[ -n "${EXP_PREFIX:-}" && "${RUN_DOMAINS}" != *,* ]]; then
    exp_prefix="${EXP_PREFIX}"
    output_dir="${OUTPUT_ROOT}/${train_noise_domain}"
  else
    exp_prefix="${EXP_PREFIX_BASE}_memory_${train_noise_domain}"
    output_dir="${OUTPUT_ROOT}/memory/${train_noise_domain}"
  fi

  mkdir -p "${output_dir}"

  echo "=========================================="
  echo "Wan2.2 I2V SlotMem Stage1 Training"
  echo "=========================================="
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "NUM_TRAIN_GPUS=${NUM_TRAIN_GPUS}"
  echo "CKPT_DIR=${CKPT_DIR}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "OUTPUT_DIR=${output_dir}"
  echo "EXP_PREFIX=${exp_prefix}"
  echo "TRAIN_NOISE_DOMAIN=${train_noise_domain}"
  echo "SLOTMEM_MEMORY=enabled"
  echo "MAX_STEPS=${max_steps}"
  echo "PRETRAINED_LORA_PATH=${pretrained_lora_path:-<none>}"
  echo "=========================================="

  phase_args=(
    --output_path "${output_dir}"
    --exp_prefix "${exp_prefix}"
    --max_steps "${max_steps}"
    --train_noise_domain "${train_noise_domain}"
    --char_attn_noise_scope "${train_noise_domain}"
    --enable_sparse_role_memory_attn
    --slotmem_memory_encoder_mode "on"
    --slotmem_memory_writer_mode off
  )

  if [[ -n "${pretrained_lora_path}" ]]; then
    phase_args+=(--pretrained_lora_path "${pretrained_lora_path}")
  fi

  torchrun --nproc_per_node="${NUM_TRAIN_GPUS}" --master-port="${MASTER_PORT}" train_mem_Encoder.py \
    "${common_args[@]}" \
    "${phase_args[@]}"
}

IFS=',' read -r -a domain_list <<< "${RUN_DOMAINS}"

for domain in "${domain_list[@]}"; do
  domain="$(printf '%s' "${domain}" | xargs)"
  [[ -z "${domain}" ]] && continue
  run_phase "${domain}" "$(max_steps_for_domain "${domain}")" "$(pretrained_for_domain "${domain}")"
done
