#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/slotmem/output_root [extra tools/run_slotmem_benchmarks.py args...]"
  exit 1
fi

INPUT_ROOT="$1"
shift

export PROVIDER_MODE=api
export NARRASTREAM_API_PROVIDER="${NARRASTREAM_API_PROVIDER:-openai}"
export NARRASTREAM_API_BASE_URL="${NARRASTREAM_API_BASE_URL:-https://api.openai.com/v1}"
export NARRASTREAM_API_KEY_ENV="${NARRASTREAM_API_KEY_ENV:-OPENAI_API_KEY}"
export NARRASTREAM_API_MODEL="${NARRASTREAM_API_MODEL:-gpt-4.1}"
export SEGMENT_DURATION="${SEGMENT_DURATION:-10}"

"${PYTHON_BIN:-python3}" tools/run_slotmem_benchmarks.py \
  --infer-output "${INPUT_ROOT}" \
  --provider-mode api \
  "$@"

# ViStoryBench is intentionally not run by default because it needs subject reference.
# Manual refs should be placed in the ViStory dataset-style character folders,
# e.g. data/dataset/ViStory/<story_id>/image/<character_name>/*.jpg, then run:
# "${PYTHON_BIN:-python3}" bench/vistorybench/bench_run.py \
#   --dataset_path /path/to/vistorybench/data/dataset \
#   --outputs_path /path/to/vistorybench/outputs \
#   --result_path /path/to/vistorybench/results \
#   --api_key "${OPENAI_API_KEY}" --base_url "${NARRASTREAM_API_BASE_URL}" --model_id "${NARRASTREAM_API_MODEL}" \
#   --method SlotMem --language en --mode manual_ref_images --metrics cids csd diversity aesthetic prompt_align
