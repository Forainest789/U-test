#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/slotmem/output_root [output_root] [run_name]"
  exit 1
fi

INPUT_ROOT="$1"
OUTPUT_ROOT="${2:-${REPO_DIR}/benchmark_outputs/slotmem}"
RUN_NAME="${3:-slotmem_bench_$(date +%Y%m%d_%H%M%S)}"

# Release users should provide external paths and API credentials through the environment.
NARRASTREAM_REPO="${NARRASTREAM_REPO:?Set NARRASTREAM_REPO to your NarraStream-Bench checkout}"
NARRASTREAM_PATH_CONFIG="${NARRASTREAM_PATH_CONFIG:-${NARRASTREAM_REPO}/configs/paths_video_model.yaml}"
VBENCH_PYTHON="${VBENCH_PYTHON:?Set VBENCH_PYTHON to the Python interpreter with VBench installed}"
NARRASTREAM_API_PYTHON="${NARRASTREAM_API_PYTHON:?Set NARRASTREAM_API_PYTHON to the Python interpreter with NarraStream API dependencies installed}"
QWEN35_PYTHON="${QWEN35_PYTHON:?Set QWEN35_PYTHON to the Python interpreter with local Qwen3.5 dependencies installed}"
LOCAL_QWEN35_MODEL="${LOCAL_QWEN35_MODEL:-/models/Qwen3.5-4B}"
QWEN35_NARRASTREAM_METRICS=(${QWEN35_NARRASTREAM_METRICS:-entity_grounding vlm_score})

: "${OPENAI_COMPAT_API_KEY:?Set OPENAI_COMPAT_API_KEY for GPT-4.1-compatible API evaluation}"
: "${OPENAI_COMPAT_BASE_URL:?Set OPENAI_COMPAT_BASE_URL for GPT-4.1-compatible API evaluation}"
export OPENAI_COMPAT_API_KEY OPENAI_COMPAT_BASE_URL
export NARRASTREAM_API_PROVIDER=openai
export NARRASTREAM_API_BASE_URL="$OPENAI_COMPAT_BASE_URL"
export NARRASTREAM_API_KEY_ENV=OPENAI_COMPAT_API_KEY
export NARRASTREAM_API_MODEL="${NARRASTREAM_API_MODEL:-gpt-4.1}"
export NARRASTREAM_REPO NARRASTREAM_PATH_CONFIG LOCAL_QWEN35_MODEL
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-$NO_PROXY}"

mkdir -p "$OUTPUT_ROOT/$RUN_NAME/logs"

echo "[1/5] VBench"
PYTHON_BIN="$VBENCH_PYTHON" \
"$VBENCH_PYTHON" tools/run_slotmem_benchmarks.py \
  --infer-output "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --run-name "$RUN_NAME" \
  --benchmarks vbench \
  --python-bin "$VBENCH_PYTHON" \
  >"$OUTPUT_ROOT/$RUN_NAME/logs/vbench.log" 2>&1

echo "[2/5] NarraStream GPT-4.1"
PYTHONPATH="$NARRASTREAM_REPO" \
PYTHON_BIN="$NARRASTREAM_API_PYTHON" \
"$NARRASTREAM_API_PYTHON" tools/run_slotmem_benchmarks.py \
  --infer-output "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --run-name "${RUN_NAME}_narrastream_gpt41" \
  --benchmarks narrastream \
  --provider-mode api \
  --python-bin "$NARRASTREAM_API_PYTHON" \
  --narrastream-repo "$NARRASTREAM_REPO" \
  --narrastream-path-config "$NARRASTREAM_PATH_CONFIG" \
  --api-workers "${API_WORKERS_GPT41:-1}" \
  --device "${NARRASTREAM_DEVICE_GPT41:-cpu}" \
  >"$OUTPUT_ROOT/$RUN_NAME/logs/narrastream_gpt41.log" 2>&1

echo "[3/5] NarraStream Qwen3.5"
PYTHONPATH="$NARRASTREAM_REPO" \
PYTHON_BIN="$QWEN35_PYTHON" \
"$QWEN35_PYTHON" tools/run_slotmem_benchmarks.py \
  --infer-output "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --run-name "${RUN_NAME}_narrastream_qwen35" \
  --benchmarks narrastream \
  --provider-mode local-qwen35 \
  --python-bin "$QWEN35_PYTHON" \
  --narrastream-repo "$NARRASTREAM_REPO" \
  --narrastream-path-config "$NARRASTREAM_PATH_CONFIG" \
  --api-workers "${API_WORKERS_QWEN35:-1}" \
  --gpu-id "${QWEN35_GPU_ID:-0}" \
  --device "${NARRASTREAM_DEVICE_QWEN35:-auto}" \
  --narrastream-metrics "${QWEN35_NARRASTREAM_METRICS[@]}" \
  >"$OUTPUT_ROOT/$RUN_NAME/logs/narrastream_qwen35.log" 2>&1

echo "[4/5] ViStory PromptAlign GPT-4.1"
"$NARRASTREAM_API_PYTHON" tools/evaluate_slotmem_vistory_prompt_align.py \
  --infer-output "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --run-name "$RUN_NAME" \
  --provider api \
  --provider-id gpt41 \
  --api-key "$OPENAI_COMPAT_API_KEY" \
  --base-url "$OPENAI_COMPAT_BASE_URL" \
  --api-model "$NARRASTREAM_API_MODEL" \
  >"$OUTPUT_ROOT/$RUN_NAME/logs/vistory_gpt41.log" 2>&1

echo "[5/5] ViStory PromptAlign Qwen3.5"
PYTHONPATH="$NARRASTREAM_REPO" \
CUDA_VISIBLE_DEVICES="${QWEN35_GPU_ID:-0}" \
"$QWEN35_PYTHON" tools/evaluate_slotmem_vistory_prompt_align.py \
  --infer-output "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --run-name "$RUN_NAME" \
  --provider local-qwen35 \
  --provider-id qwen35 \
  --narrastream-repo "$NARRASTREAM_REPO" \
  --local-qwen35-model "$LOCAL_QWEN35_MODEL" \
  >"$OUTPUT_ROOT/$RUN_NAME/logs/vistory_qwen35.log" 2>&1

echo "Done: $OUTPUT_ROOT/$RUN_NAME"
