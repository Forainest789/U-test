#!/usr/bin/env bash
# Strict A100 entry point for the fast SlotMem identity-token causal probe.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${EVENT_JSON:?set EVENT_JSON to the frozen recurrence event}"
: "${FUTURE_TARGET_VIDEO:?set FUTURE_TARGET_VIDEO to the held-out target clip}"
: "${FUTURE_TARGET_MANIFEST:?set FUTURE_TARGET_MANIFEST to its provenance manifest}"
: "${BASE_INFERENCE_ARGS:?set BASE_INFERENCE_ARGS to validated inference_args.yaml}"
: "${PLATFORM_MANIFEST:?set PLATFORM_MANIFEST to platform.manifest.json}"
: "${DONOR_PAYLOAD:?set DONOR_PAYLOAD to a frozen v2 matched-wrong payload}"
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
  "${EVENT_JSON}" "${FUTURE_TARGET_VIDEO}" "${FUTURE_TARGET_MANIFEST}" \
  "${BASE_INFERENCE_ARGS}" "${PLATFORM_MANIFEST}" "${DONOR_PAYLOAD}" "${DONOR_MANIFEST}"; do
  [[ -f "${required}" ]] || { echo "[identity] missing file: ${required}" >&2; exit 2; }
done
[[ ! -e "${EVENT_RUN_ROOT}" ]] || { echo "[identity] output already exists: ${EVENT_RUN_ROOT}" >&2; exit 2; }
if [[ "${ALLOW_DIRTY_SOURCE:-0}" != "1" ]] && [[ -n "$(git -C "${REPO_DIR}" status --porcelain)" ]]; then
  echo "[identity] source tree is dirty; commit it or set ALLOW_DIRTY_SOURCE=1 for development only" >&2
  exit 2
fi

mkdir -p "${EVENT_RUN_ROOT}"
if command -v conda >/dev/null 2>&1 && [[ -n "${UTEST_ENV:-}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${UTEST_ENV}"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
IDENTITY_TIMESTEPS="${IDENTITY_TIMESTEPS:-0,25,49}"
IDENTITY_LAYER_GROUPS="${IDENTITY_LAYER_GROUPS:-0-4,5-10,11-15}"
export SLOTMEM_OFFLOAD_MODELS="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
if [[ "${ALLOW_ATTENTION_FALLBACK:-0}" == "1" ]]; then
  export DIFFSYNTH_ATTENTION_IMPLEMENTATION="${DIFFSYNTH_ATTENTION_IMPLEMENTATION:-torch}"
else
  export DIFFSYNTH_ATTENTION_IMPLEMENTATION="flash_attention_2"
fi

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
  printf '[identity] %s: ' "${name}"
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != "1" ]]; then "$@"; fi
}

finalize_manifest() {
  local exit_code=$?
  trap - EXIT
  "${PYTHON_BIN}" -c '
import json, os, pathlib, subprocess, sys
log, out, status, dry, exit_code = sys.argv[1:]
log_path = pathlib.Path(log)
commands = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()] if log_path.exists() else []
commit = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False).stdout.strip()
payload = {
    "schema_version": 1,
    "status": status if int(exit_code) == 0 else "failed",
    "dry_run": dry == "1",
    "source_commit": commit,
    "environment": {
        "DIFFSYNTH_ATTENTION_IMPLEMENTATION": os.environ.get("DIFFSYNTH_ATTENTION_IMPLEMENTATION"),
        "SLOTMEM_OFFLOAD_MODELS": os.environ.get("SLOTMEM_OFFLOAD_MODELS"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    },
    "commands": commands,
}
pathlib.Path(out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
' "${COMMAND_LOG}" "${EVENT_RUN_ROOT}/run_manifest.json" "${RUN_STATUS}" "${DRY_RUN:-0}" "${exit_code}"
  exit "${exit_code}"
}
trap finalize_manifest EXIT

cd "${REPO_DIR}"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import os, torch
assert torch.cuda.is_available(), "CUDA is unavailable"
name = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory
assert "A100" in name, f"expected A100, got {name}"
assert total >= 75 * 1024**3, f"expected >=75 GiB, got {total / 1024**3:.2f} GiB"
assert torch.cuda.is_bf16_supported(), "BF16 is unsupported"
if os.environ.get("DIFFSYNTH_ATTENTION_IMPLEMENTATION") == "flash_attention_2":
    import flash_attn
    from diffsynth.core.attention.attention import ATTENTION_IMPLEMENTATION
    assert ATTENTION_IMPLEMENTATION == "flash_attention_2", ATTENTION_IMPLEMENTATION
print(f"[identity] preflight device={name} memory_gib={total / 1024**3:.2f}")
PY
fi

run_step identity-self-check "${PYTHON_BIN}" -m utest.identity_token_probe --self-check
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
  --timestep-indices "${IDENTITY_TIMESTEPS}"
  --arm-seed "${ARM_SEED:-0}"
)
if [[ "${ALLOW_DIRTY_SOURCE:-0}" == "1" ]]; then PREPARE+=(--allow-dirty-source); fi
run_step prepare-prefix "${PREPARE[@]}"

PROBE=(
  "${PYTHON_BIN}" -m utest.identity_token_probe
  --prefix "${EVENT_RUN_ROOT}/prefix"
  --future-target-video "${FUTURE_TARGET_VIDEO}"
  --arms-root "${EVENT_RUN_ROOT}/arms"
  --donor "${DONOR_PAYLOAD}"
  --donor-manifest "${DONOR_MANIFEST}"
  --output "${EVENT_RUN_ROOT}/identity_probe"
  --timestep-indices "${IDENTITY_TIMESTEPS}"
  --layer-groups "${IDENTITY_LAYER_GROUPS}"
  --max-groups "${IDENTITY_MAX_GROUPS:-8}"
  --identity-budget "${IDENTITY_BUDGET:-0.25}"
  --noise-seed "${IDENTITY_NOISE_SEED:-0}"
  --repeat-loss-tolerance "${IDENTITY_REPEAT_LOSS_TOLERANCE:-0}"
  --repeat-influence-tolerance "${IDENTITY_REPEAT_INFLUENCE_TOLERANCE:-0}"
  --benefit-margin "${IDENTITY_BENEFIT_MARGIN:-0}"
  --influence-floor "${IDENTITY_INFLUENCE_FLOOR:-0}"
)
if [[ "${IDENTITY_SMOKE:-0}" == "1" ]]; then PROBE+=(--smoke); fi
if [[ "${RUN_DECODED_VALIDATION:-0}" == "1" ]]; then PROBE+=(--run-decoded-validation); fi
if [[ "${ALLOW_ATTENTION_FALLBACK:-0}" == "1" ]]; then PROBE+=(--allow-attention-fallback); fi
if [[ "${REQUIRE_DYNAMIC_WRITER:-0}" == "1" ]]; then PROBE+=(--require-dynamic-writer); fi
run_step identity-probe "${PROBE[@]}"

RUN_STATUS=$([[ "${DRY_RUN:-0}" == "1" ]] && echo dry_run || echo passed)
echo "[identity] report: ${EVENT_RUN_ROOT}/identity_probe/identity_probe_report.json"
echo "[identity] manifest: ${EVENT_RUN_ROOT}/run_manifest.json"
