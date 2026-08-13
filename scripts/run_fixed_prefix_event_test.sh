#!/usr/bin/env bash
# Run no_memory/zero/correct/wrong/random from one immutable real-story prefix.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${EVENT_JSON:?set EVENT_JSON to one event copied from the real e0.json report}"
: "${BASE_INFERENCE_ARGS:?set BASE_INFERENCE_ARGS to a validated M0a inference_args.yaml}"
: "${PLATFORM_MANIFEST:?set PLATFORM_MANIFEST to the stage-run platform.manifest.json}"
: "${DONOR_PAYLOAD:?set DONOR_PAYLOAD to a v2 donor payload dump from a different story}"
: "${DONOR_MANIFEST:?set DONOR_MANIFEST to the frozen matched-pair JSON}"
: "${EVENT_RUN_ROOT:?set EVENT_RUN_ROOT to a fresh remote output directory}"

if command -v conda >/dev/null 2>&1 && [[ -n "${UTEST_ENV:-}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${UTEST_ENV}"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
for required in "${EVENT_JSON}" "${BASE_INFERENCE_ARGS}" "${PLATFORM_MANIFEST}" "${DONOR_PAYLOAD}" "${DONOR_MANIFEST}"; do
  [[ -f "${required}" ]] || { echo "[event] missing file: ${required}" >&2; exit 2; }
done
[[ ! -e "${EVENT_RUN_ROOT}" ]] || { echo "[event] output already exists: ${EVENT_RUN_ROOT}" >&2; exit 2; }
mkdir -p "${EVENT_RUN_ROOT}"
cd "${REPO_DIR}"

"${PYTHON_BIN}" -m utest.event_harness prepare-prefix \
  --event "${EVENT_JSON}" \
  --output "${EVENT_RUN_ROOT}/prefix" \
  --platform-manifest "${PLATFORM_MANIFEST}" \
  --inference-args-file "${BASE_INFERENCE_ARGS}" \
  --arm-seed "${ARM_SEED:-0}"

"${PYTHON_BIN}" -m utest.event_harness run-arms \
  --prefix "${EVENT_RUN_ROOT}/prefix" \
  --output "${EVENT_RUN_ROOT}/prefix/arms" \
  --arms no_memory,zero,correct,wrong,random \
  --donor "${DONOR_PAYLOAD}" \
  --donor-manifest "${DONOR_MANIFEST}"

"${PYTHON_BIN}" -m utest.event_harness validate \
  --event-run "${EVENT_RUN_ROOT}/prefix/arms"

echo "[event] contract: ${EVENT_RUN_ROOT}/prefix/arms/intervention_contract.json"
