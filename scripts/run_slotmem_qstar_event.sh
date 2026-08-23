#!/usr/bin/env bash
# Strict SlotMem Q* probe plus optional seven-run rollout validation.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${EVENT_JSON:?set EVENT_JSON to the frozen recurrence event}"
: "${FUTURE_TARGET_VIDEO:?set FUTURE_TARGET_VIDEO to an independent held-out target clip}"
: "${FUTURE_TARGET_MANIFEST:?set FUTURE_TARGET_MANIFEST to its provenance manifest}"
: "${BASE_INFERENCE_ARGS:?set BASE_INFERENCE_ARGS to validated inference_args.yaml}"
: "${PLATFORM_MANIFEST:?set PLATFORM_MANIFEST to platform.manifest.json}"
: "${DONOR_PAYLOAD:?set DONOR_PAYLOAD to a frozen v2 donor payload}"
: "${DONOR_MANIFEST:?set DONOR_MANIFEST to its matched-pair manifest}"
: "${EVENT_RUN_ROOT:?set EVENT_RUN_ROOT to a fresh output directory}"

normalize_path() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -u "$1"; else printf '%s\n' "$1"; fi
}
EVENT_JSON="$(normalize_path "${EVENT_JSON}")"
FUTURE_TARGET_VIDEO="$(normalize_path "${FUTURE_TARGET_VIDEO}")"
FUTURE_TARGET_MANIFEST="$(normalize_path "${FUTURE_TARGET_MANIFEST}")"
BASE_INFERENCE_ARGS="$(normalize_path "${BASE_INFERENCE_ARGS}")"
PLATFORM_MANIFEST="$(normalize_path "${PLATFORM_MANIFEST}")"
DONOR_PAYLOAD="$(normalize_path "${DONOR_PAYLOAD}")"
DONOR_MANIFEST="$(normalize_path "${DONOR_MANIFEST}")"
EVENT_RUN_ROOT="$(normalize_path "${EVENT_RUN_ROOT}")"

for required in \
  "${EVENT_JSON}" "${FUTURE_TARGET_VIDEO}" "${FUTURE_TARGET_MANIFEST}" "${BASE_INFERENCE_ARGS}" \
  "${PLATFORM_MANIFEST}" "${DONOR_PAYLOAD}" "${DONOR_MANIFEST}"; do
  [[ -f "${required}" ]] || { echo "[qstar] missing file: ${required}" >&2; exit 2; }
done
[[ ! -e "${EVENT_RUN_ROOT}" ]] || { echo "[qstar] output already exists: ${EVENT_RUN_ROOT}" >&2; exit 2; }
if [[ "${ALLOW_DIRTY_SOURCE:-0}" != "1" ]] && [[ -n "$(git -C "${REPO_DIR}" status --porcelain)" ]]; then
  echo "[qstar] source tree is dirty; commit it or set ALLOW_DIRTY_SOURCE=1 for development only" >&2
  exit 2
fi

mkdir -p "${EVENT_RUN_ROOT}"
if command -v conda >/dev/null 2>&1 && [[ -n "${UTEST_ENV:-}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${UTEST_ENV}"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
QSTAR_TIMESTEP_INDICES="${QSTAR_TIMESTEP_INDICES:-0,12,25,37,49}"
RUN_ROLLOUT="${RUN_ROLLOUT:-1}"
export SLOTMEM_OFFLOAD_MODELS="${SLOTMEM_OFFLOAD_MODELS:-0}"
COMMAND_LOG="${EVENT_RUN_ROOT}/.commands.jsonl"
RUN_STATUS="failed"

record_command() {
  local name="$1"
  shift
  "${PYTHON_BIN}" -c \
    'import json,sys; open(sys.argv[1],"a",encoding="utf-8").write(json.dumps({"name":sys.argv[2],"argv":sys.argv[3:]},ensure_ascii=False)+"\n")' \
    "${COMMAND_LOG}" "${name}" "$@"
}

run_step() {
  local name="$1"
  shift
  record_command "${name}" "$@"
  printf '[qstar] %s: ' "${name}"
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != "1" ]]; then "$@"; fi
}

finalize_manifest() {
  local exit_code=$?
  trap - EXIT
  "${PYTHON_BIN}" -c '
import json, pathlib, sys
log, out, status, dry, timesteps, exit_code = sys.argv[1:]
log_path = pathlib.Path(log)
commands = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()] if log_path.exists() else []
arms_path = pathlib.Path(out).parent / "arms" / "arm_commands.json"
input_contract_path = pathlib.Path(out).parent / "input_contract.json"
payload = {
    "schema_version": 1,
    "status": status if int(exit_code) == 0 else "failed",
    "dry_run": dry == "1",
    "timestep_indices": [int(value) for value in timesteps.split(",") if value],
    "seven_runs": ["correct", "correct_repeat", "no_memory", "zero", "random", "wrong", "native"],
    "commands": commands,
}
if arms_path.is_file():
    payload["resolved_arm_commands"] = json.loads(arms_path.read_text(encoding="utf-8"))
if input_contract_path.is_file():
    payload["input_contract"] = json.loads(input_contract_path.read_text(encoding="utf-8"))
pathlib.Path(out).write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
' "${COMMAND_LOG}" "${EVENT_RUN_ROOT}/run_manifest.json" "${RUN_STATUS}" "${DRY_RUN:-0}" "${QSTAR_TIMESTEP_INDICES}" "${exit_code}"
  rm -f "${COMMAND_LOG}"
  exit "${exit_code}"
}
trap finalize_manifest EXIT

cd "${REPO_DIR}"
run_step content-self-check "${PYTHON_BIN}" -m utest.content_audit --self-check
run_step qstar-self-check "${PYTHON_BIN}" -m utest.qstar_probe --self-check
run_step input-contract-preflight \
  "${PYTHON_BIN}" -m utest.input_contract \
  --event "${EVENT_JSON}" \
  --donor "${DONOR_PAYLOAD}" \
  --donor-manifest "${DONOR_MANIFEST}" \
  --future-target-video "${FUTURE_TARGET_VIDEO}" \
  --future-target-manifest "${FUTURE_TARGET_MANIFEST}" \
  --arms-root "${EVENT_RUN_ROOT}/arms" \
  --report "${EVENT_RUN_ROOT}/input_contract.json"

PREPARE=(
  "${PYTHON_BIN}" -m utest.event_harness prepare-prefix
  --event "${EVENT_JSON}"
  --output "${EVENT_RUN_ROOT}/prefix"
  --platform-manifest "${PLATFORM_MANIFEST}"
  --inference-args-file "${BASE_INFERENCE_ARGS}"
  --future-target-video "${FUTURE_TARGET_VIDEO}"
  --future-target-manifest "${FUTURE_TARGET_MANIFEST}"
  --arms-root "${EVENT_RUN_ROOT}/arms"
  --timestep-indices "${QSTAR_TIMESTEP_INDICES}"
  --arm-seed "${ARM_SEED:-0}"
)
if [[ "${ALLOW_DIRTY_SOURCE:-0}" == "1" ]]; then PREPARE+=(--allow-dirty-source); fi
run_step prepare-prefix "${PREPARE[@]}"

QSTAR=(
  "${PYTHON_BIN}" -m utest.qstar_probe
  --prefix "${EVENT_RUN_ROOT}/prefix"
  --future-target-video "${FUTURE_TARGET_VIDEO}"
  --output "${EVENT_RUN_ROOT}/qstar"
  --arms-root "${EVENT_RUN_ROOT}/arms"
  --donor "${DONOR_PAYLOAD}"
  --donor-manifest "${DONOR_MANIFEST}"
  --timestep-indices "${QSTAR_TIMESTEP_INDICES}"
  --noise-seed "${QSTAR_NOISE_SEED:-0}"
  --repeat-loss-tolerance "${QSTAR_REPEAT_LOSS_TOLERANCE:-0}"
  --repeat-influence-tolerance "${QSTAR_REPEAT_INFLUENCE_TOLERANCE:-0}"
  --benefit-margin "${QSTAR_BENEFIT_MARGIN:-0}"
  --influence-floor "${QSTAR_INFLUENCE_FLOOR:-0}"
)
if [[ "${REQUIRE_DYNAMIC_WRITER:-0}" == "1" ]]; then QSTAR+=(--require-dynamic-writer); fi
run_step qstar-probe "${QSTAR[@]}"

if [[ "${RUN_ROLLOUT}" == "1" ]]; then
  ROLLOUT=(
    "${PYTHON_BIN}" -m utest.event_harness run-arms
    --prefix "${EVENT_RUN_ROOT}/prefix"
    --output "${EVENT_RUN_ROOT}/arms"
    --arms correct,no_memory,zero,random,wrong
    --donor "${DONOR_PAYLOAD}"
    --donor-manifest "${DONOR_MANIFEST}"
    --include-native
  )
  if [[ "${REQUIRE_DYNAMIC_WRITER:-0}" == "1" ]]; then ROLLOUT+=(--require-dynamic-writer); fi
  run_step seven-rollouts "${ROLLOUT[@]}"

  VALIDATE=(
    "${PYTHON_BIN}" -m utest.event_harness validate
    --event-run "${EVENT_RUN_ROOT}/arms"
    --require-native
  )
  if [[ "${REQUIRE_DYNAMIC_WRITER:-0}" == "1" ]]; then VALIDATE+=(--require-dynamic-writer); fi
  run_step validate-rollouts "${VALIDATE[@]}"

  if [[ -n "${CID_SCORER:-}" ]]; then
    CID_SCORER="$(normalize_path "${CID_SCORER}")"
    [[ -f "${CID_SCORER}" ]] || { echo "[qstar] missing C_id scorer: ${CID_SCORER}" >&2; exit 2; }
    run_step cid-score "${PYTHON_BIN}" "${CID_SCORER}" \
      --event-run "${EVENT_RUN_ROOT}/arms" \
      --output "${EVENT_RUN_ROOT}/arms/cid_report.json"
  fi
fi

RUN_STATUS=$([[ "${DRY_RUN:-0}" == "1" ]] && echo dry_run || echo passed)
echo "[qstar] report: ${EVENT_RUN_ROOT}/qstar/qstar_report.json"
echo "[qstar] manifest: ${EVENT_RUN_ROOT}/run_manifest.json"
