#!/usr/bin/env bash
# Compatibility entry point. The Q* runner is the only scientific SlotMem U-test.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STRICT="${REPO_DIR}/scripts/run_slotmem_qstar_event.sh"
export EVENT_JSON="${EVENT_JSON:-${REPO_DIR}/utest/events/person_reappearance_delta8.json}"

required=(
  FUTURE_TARGET_VIDEO FUTURE_TARGET_MANIFEST BASE_INFERENCE_ARGS PLATFORM_MANIFEST
  DONOR_PAYLOAD DONOR_MANIFEST EVENT_RUN_ROOT
)
missing=()
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || missing+=("${name}")
done
if (( ${#missing[@]} > 0 )); then
  echo "[slotmem-utest] legacy six-arm runner retired; missing strict inputs: ${missing[*]}" >&2
  cat >&2 <<'USAGE'
EVENT_JSON="$PWD/utest/events/person_reappearance_delta8.json" \
FUTURE_TARGET_VIDEO=/data/targets/person_reappearance_delta8_chunk_008_teacher.mp4 \
FUTURE_TARGET_MANIFEST=/data/targets/person_reappearance_delta8_chunk_008_teacher.manifest.json \
BASE_INFERENCE_ARGS=/data/runs/stage_gates/slotmem_m0_001/m0a/inference_args.yaml \
PLATFORM_MANIFEST=/data/runs/stage_gates/slotmem_m0_001/platform.manifest.json \
DONOR_PAYLOAD=/data/events/donor_payload.pt \
DONOR_MANIFEST=/data/events/donor_manifest.json \
EVENT_RUN_ROOT=/data/runs/qstar_person_reappearance_delta8 \
QSTAR_TIMESTEP_INDICES=0,12,25,37,49 \
RUN_ROLLOUT=1 UTEST_ENV=utest \
bash scripts/run_slotmem_qstar_event.sh
USAGE
  exit 2
fi
exec bash "${STRICT}"
