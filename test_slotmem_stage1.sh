#!/bin/bash
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -euo pipefail

if [[ -n "${CONDA_ENV:-}" ]] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${REPO_DIR}"

CKPT_DIR="${CKPT_DIR:-/models/Wan2.2-I2V-A14B}"
HIGH_EXPERT_CKPT_PATH="${HIGH_EXPERT_CKPT_PATH:-${REPO_DIR}/ckpt/stage1/stage1_high.pt}"
NATIVE_WAN_INFERENCE="${NATIVE_WAN_INFERENCE:-0}"
LOW_EXPERT_CKPT_PATH="${LOW_EXPERT_CKPT_PATH:-${REPO_DIR}/ckpt/stage1/stage1_low.pt}"
JSON_PATH="${JSON_PATH:-}"
REF_IMAGE_PATH="${REF_IMAGE_PATH:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/inference_outputs/slotmem_stage1}"
EXP_PREFIX="${EXP_PREFIX:-}"
TRAIN_NOISE_DOMAIN="${TRAIN_NOISE_DOMAIN:-low_noise}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-flow_euler}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-5.0}"
SEED_BASE="${SEED_BASE:-42}"
MAX_CHUNKS="${MAX_CHUNKS:--1}"
DUAL_EXPERT_LOAD_MODE="${DUAL_EXPERT_LOAD_MODE:-active}"
DUAL_EXPERT_OFFLOAD_DTYPE="${DUAL_EXPERT_OFFLOAD_DTYPE:-bfloat16}"
DUAL_EXPERT_VRAM_LIMIT="${DUAL_EXPERT_VRAM_LIMIT:--1}"
DUAL_EXPERT_MANAGE_AUX_MODELS="${DUAL_EXPERT_MANAGE_AUX_MODELS:-0}"
OUTPUT_DIR="${OUTPUT_ROOT}"
if [[ -n "${EXP_PREFIX}" ]]; then
  OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_PREFIX}"
fi

if [[ -z "${JSON_PATH}" ]]; then
  echo "JSON_PATH is required"
  exit 1
fi
if [[ ! -f "${JSON_PATH}" ]]; then
  echo "JSON_PATH not found: ${JSON_PATH}"
  exit 1
fi
case "${NATIVE_WAN_INFERENCE,,}" in
  1|true|on|yes) native_wan_enabled=1 ;;
  *) native_wan_enabled=0 ;;
esac
if [[ "${native_wan_enabled}" == "0" ]]; then
  if [[ -z "${HIGH_EXPERT_CKPT_PATH}" && -z "${LOW_EXPERT_CKPT_PATH}" ]]; then
    echo "HIGH_EXPERT_CKPT_PATH/LOW_EXPERT_CKPT_PATH is required unless NATIVE_WAN_INFERENCE=1"
    exit 1
  fi
  if [[ -n "${HIGH_EXPERT_CKPT_PATH}" && ! -f "${HIGH_EXPERT_CKPT_PATH}" ]]; then
    echo "HIGH_EXPERT_CKPT_PATH not found: ${HIGH_EXPERT_CKPT_PATH}"
    exit 1
  fi
  if [[ -n "${LOW_EXPERT_CKPT_PATH}" && ! -f "${LOW_EXPERT_CKPT_PATH}" ]]; then
    echo "LOW_EXPERT_CKPT_PATH not found: ${LOW_EXPERT_CKPT_PATH}"
    exit 1
  fi
fi
if [[ -n "${REF_IMAGE_PATH}" && ! -f "${REF_IMAGE_PATH}" ]]; then
  echo "REF_IMAGE_PATH not found: ${REF_IMAGE_PATH}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

cmd=(
  python3 -u test_mem_Encoder.py
  --ckpt_dir "${CKPT_DIR}"
  --json_path "${JSON_PATH}"
  --ref_image_path "${REF_IMAGE_PATH}"
  --output_path "${OUTPUT_DIR}"
  --train_noise_domain "${TRAIN_NOISE_DOMAIN}"
  --noise_domain_boundary_ratio "${NOISE_DOMAIN_BOUNDARY_RATIO:-0.9}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --sample_solver "${SAMPLE_SOLVER}"
  --sample_shift "${SAMPLE_SHIFT}"
  --seed_base "${SEED_BASE}"
  --cfg_scale "${CFG_SCALE:-5.0}"
  --cfg_scale_extraction "${CFG_SCALE_EXTRACTION:-5.0}"
  --height "${HEIGHT:-480}"
  --width "${WIDTH:-832}"
  --context_frames "${CONTEXT_FRAMES:-81}"
  --num_overlap_frame "${NUM_OVERLAP_FRAME:-5}"
  --ref_pad_cfg
  --dual_expert_load_mode "${DUAL_EXPERT_LOAD_MODE}"
  --dual_expert_offload_dtype "${DUAL_EXPERT_OFFLOAD_DTYPE}"
  --dual_expert_vram_limit "${DUAL_EXPERT_VRAM_LIMIT}"
  --extract_layers "${EXTRACT_LAYERS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
  --role_token_selection_mode "${ROLE_TOKEN_SELECTION_MODE:-two_role_diff}"
  --top_visual_tokens "${TOP_VISUAL_TOKENS:-0.1}"
  --token_weight "${TOKEN_WEIGHT:-1.0}"
  --suffix_attention_scale "${SUFFIX_ATTENTION_SCALE:-1.0}"
  --max_memory_tokens_per_character "${MAX_MEMORY_TOKENS_PER_CHARACTER:-512}"
  --use_attn_score_selection
  --max_memory_characters "${MAX_MEMORY_CHARACTERS:-2}"
  --slotmem_memory_bank_mode single
  --neighbor_filter_kernel "${NEIGHBOR_FILTER_KERNEL:-5}"
  --neighbor_filter_any_window
  --lora_rank "${LORA_RANK:-128}"
  --lora_alpha "${LORA_ALPHA:-128}"
  --lora_target_modules "${LORA_TARGET_MODULES:-q,k,v,o,ffn.0,ffn.2}"
  --sparse_role_memory_layer_idx "${SPARSE_ROLE_MEMORY_LAYER_IDX:-3}"
  --sparse_role_memory_injection_layers "${SPARSE_ROLE_MEMORY_INJECTION_LAYERS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
  --memory_layer_binding_mode "${MEMORY_LAYER_BINDING_MODE:-layerwise}"
  --char_attn_noise_scope "${CHAR_ATTN_NOISE_SCOPE:-${TRAIN_NOISE_DOMAIN}}"
  --sparse_role_memory_num_heads "${SPARSE_ROLE_MEMORY_NUM_HEADS:-8}"
  --sparse_role_memory_head_dim "${SPARSE_ROLE_MEMORY_HEAD_DIM:-128}"
  --sparse_role_memory_rope_dim "${SPARSE_ROLE_MEMORY_ROPE_DIM:-256}"
  --sparse_role_memory_use_half_role_heads
  --sparse_role_memory_feature_source "${SPARSE_ROLE_MEMORY_FEATURE_SOURCE:-attn_out}"
  --sparse_role_memory_init_scale "${SPARSE_ROLE_MEMORY_INIT_SCALE:-0.1}"
  --sparse_role_memory_time_gate
  --slotmem_memory_encoder_mode "on"
  --slotmem_memory_encoder_layers "${SLOTMEM_MEMORY_ENCODER_LAYERS:-0-15}"
  --slotmem_memory_encoder_layer_groups "${SLOTMEM_MEMORY_ENCODER_LAYER_GROUPS:-0-4,5-10,11-15}"
  --slotmem_memory_encoder_slots "${SLOTMEM_MEMORY_ENCODER_SLOTS:-64}"
  --slotmem_memory_encoder_dim "${SLOTMEM_MEMORY_ENCODER_DIM:-512}"
  --slotmem_memory_encoder_hidden_dim "${SLOTMEM_MEMORY_ENCODER_HIDDEN_DIM:-1024}"
  --slotmem_memory_encoder_use_t_embed
  --slotmem_memory_encoder_use_slot_index_embed
  --train_stage stage1
  --slotmem_memory_writer_mode off
  --memory_runtime_log_every "${MEMORY_RUNTIME_LOG_EVERY:-1}"
  --no-save_denoise_step_edge_viz
  --enable_sparse_role_memory_attn
  --max_chunks "${MAX_CHUNKS}"
)

if [[ -n "${NUM_MOTION_LATENT:-}" ]]; then
  cmd+=(--num_motion_latent "${NUM_MOTION_LATENT}")
fi

case "${DUAL_EXPERT_MANAGE_AUX_MODELS,,}" in
  0|false|off|no)
    cmd+=(--no-dual_expert_manage_aux_models)
    ;;
  *)
    cmd+=(--dual_expert_manage_aux_models)
    ;;
esac

if [[ "${native_wan_enabled}" == "1" ]]; then
  cmd+=(--native_wan_inference)
else
  if [[ -n "${HIGH_EXPERT_CKPT_PATH}" ]]; then
    cmd+=(--high_expert_checkpoint_path "${HIGH_EXPERT_CKPT_PATH}")
  fi
  if [[ -n "${LOW_EXPERT_CKPT_PATH}" ]]; then
    cmd+=(--low_expert_checkpoint_path "${LOW_EXPERT_CKPT_PATH}")
  fi
fi

exec "${cmd[@]}"
