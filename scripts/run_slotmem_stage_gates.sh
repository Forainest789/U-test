#!/usr/bin/env bash
# Real-server E0 + complete seven-chunk M0a + explicit M0b comparability report.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${NARRASTREAM_INPUT_ROOT:?set NARRASTREAM_INPUT_ROOT to converted real NarraStream stories}"
: "${WAN22_DIR:?set WAN22_DIR to Wan2.2-I2V-A14B}"
: "${CKPT_ROOT:?set CKPT_ROOT to the directory containing ckpt/stage1 and ckpt/stage2}"
: "${RUN_ROOT:?set RUN_ROOT to a server output directory}"
: "${UTEST_ENV:?set UTEST_ENV to the isolated SlotMem conda environment}"

for required in \
  "${NARRASTREAM_INPUT_ROOT}" \
  "${WAN22_DIR}/low_noise_model" \
  "${WAN22_DIR}/high_noise_model" \
  "${CKPT_ROOT}/ckpt/stage1/stage1_low.pt" \
  "${CKPT_ROOT}/ckpt/stage1/stage1_high.pt" \
  "${CKPT_ROOT}/ckpt/stage2/stage2_low.pt" \
  "${CKPT_ROOT}/ckpt/stage2/stage2_high.pt"; do
  [[ -e "${required}" ]] || { echo "[stage] missing required path: ${required}" >&2; exit 2; }
done

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${UTEST_ENV}"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_ROOT%/}/${RUN_ID}"
if [[ -e "${RUN_DIR}" && "${RESUME_RUN:-0}" != "1" ]]; then
  echo "[stage] run directory already exists: ${RUN_DIR}; set RESUME_RUN=1 with the same RUN_ID" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"
cd "${REPO_DIR}"

if [[ ! -s "${RUN_DIR}/e0.json" ]]; then
  echo "[stage] E0 -> ${RUN_DIR}/e0.json"
  "${PYTHON_BIN}" -m utest.eligibility \
    --data-root "${NARRASTREAM_INPUT_ROOT}" \
    --out "${RUN_DIR}/e0.json" | tee "${RUN_DIR}/e0.log"
else
  echo "[stage] reuse E0: ${RUN_DIR}/e0.json"
fi

E0_PASSES="$("${PYTHON_BIN}" - "${RUN_DIR}/e0.json" <<'PY'
import json, sys
print("1" if json.load(open(sys.argv[1], encoding="utf-8")).get("passes_min_viable") else "0")
PY
)"
if [[ "${E0_PASSES}" != "1" ]]; then
  echo "[stage] WARNING: E0 blocked; continuing M0 as a platform-only diagnostic" >&2
fi

CUDA_PREFLIGHT="${RUN_DIR}/cuda_preflight.json"
MIN_FREE_GPU_GB="${MIN_FREE_GPU_GB:-36}"
echo "[stage] CUDA preflight -> ${CUDA_PREFLIGHT}"
set +e
"${PYTHON_BIN}" -m utest.cuda_preflight \
  --out "${CUDA_PREFLIGHT}" \
  --min-free-gb "${MIN_FREE_GPU_GB}" \
  | tee "${RUN_DIR}/cuda_preflight.log"
cuda_preflight_status=${PIPESTATUS[0]}
set -e
if (( cuda_preflight_status != 0 )); then
  "${PYTHON_BIN}" - "${RUN_DIR}" "${cuda_preflight_status}" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
e0 = json.loads((root / "e0.json").read_text(encoding="utf-8"))
preflight = json.loads((root / "cuda_preflight.json").read_text(encoding="utf-8"))
summary = {
    "E0": "passed" if e0.get("passes_min_viable") else "blocked",
    "eligible_stories": e0.get("n_eligible_stories"),
    "CUDA_preflight": preflight.get("status", "failed"),
    "M0a": "blocked",
    "M0b": "not_run",
    "process_exit_codes": {"cuda_preflight": int(sys.argv[2])},
    "blocking_error": preflight.get("error"),
}
(root / "stage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
  echo "[stage] CUDA preflight failed; M0a was not attempted" >&2
  exit "${cuda_preflight_status}"
fi

PLATFORM_MANIFEST="${RUN_DIR}/platform.manifest.json"
if [[ ! -s "${PLATFORM_MANIFEST}" ]]; then
  echo "[stage] platform manifest -> ${PLATFORM_MANIFEST}"
  "${PYTHON_BIN}" - "${REPO_DIR}" "${CKPT_ROOT}" "${WAN22_DIR}" "${PLATFORM_MANIFEST}" <<'PY'
import hashlib, json, os, platform, subprocess, sys
from pathlib import Path

repo, ckpt_root, wan_root, output = map(Path, sys.argv[1:])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()

def command(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()

try:
    import torch
    torch_info = {
        "version": getattr(torch, "__version__", None),
        "cuda": getattr(getattr(torch, "version", None), "cuda", None),
    }
except Exception as exc:
    torch_info = {"error": str(exc)}

checkpoints = sorted((ckpt_root / "ckpt").rglob("*.pt"))
payload = {
    "schema_version": 1,
    "repo_commit": command("git", "-C", str(repo), "rev-parse", "HEAD"),
    "repo_dirty": bool(command("git", "-C", str(repo), "status", "--porcelain")),
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch_info,
    "gpu": command("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"),
    "checkpoints": {
        str(path.relative_to(ckpt_root / "ckpt")): {
            "sha256": sha256(path), "bytes": path.stat().st_size
        }
        for path in checkpoints
    },
    "wan22_files": {
        str(path.relative_to(wan_root)): path.stat().st_size
        for path in sorted(wan_root.rglob("*")) if path.is_file()
    },
}
output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
else
  echo "[stage] reuse platform manifest: ${PLATFORM_MANIFEST}"
fi

GPU_INDEX="${CUDA_VISIBLE_DEVICES:-0}"
GPU_INDEX="${GPU_INDEX%%,*}"
GPU_MIB="$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
MIN_GPU_MIB="${MIN_GPU_MIB:-36000}"
if (( GPU_MIB < MIN_GPU_MIB )) && [[ "${ALLOW_UNDERSIZED_GPU:-0}" != "1" ]]; then
  echo "[stage] GPU ${GPU_MIB} MiB is below the ${MIN_GPU_MIB} MiB M0a floor" >&2
  echo "[stage] E0 and platform manifest were saved; M0a was not attempted" >&2
  exit 3
fi

M0A_DIR="${RUN_DIR}/m0a"
mkdir -p "${M0A_DIR}"
echo "[stage] M0a seven chunks -> ${M0A_DIR}"
echo "[stage] active expert switching, bf16, 50 steps, resumable per chunk"
set +e
CONDA_ENV="${UTEST_ENV}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
DUAL_EXPERT_LOAD_MODE="${DUAL_EXPERT_LOAD_MODE:-active}" \
DUAL_EXPERT_MANAGE_AUX_MODELS="${DUAL_EXPERT_MANAGE_AUX_MODELS:-0}" \
CKPT_DIR="${WAN22_DIR}" \
STAGE1_LOW_CKPT_PATH="${CKPT_ROOT}/ckpt/stage1/stage1_low.pt" \
STAGE1_HIGH_CKPT_PATH="${CKPT_ROOT}/ckpt/stage1/stage1_high.pt" \
STAGE2_LOW_CKPT_PATH="${CKPT_ROOT}/ckpt/stage2/stage2_low.pt" \
STAGE2_HIGH_CKPT_PATH="${CKPT_ROOT}/ckpt/stage2/stage2_high.pt" \
JSON_PATH="${REPO_DIR}/sample/test/3_271/rewrite_caption.json" \
REF_IMAGE_PATH="${REPO_DIR}/sample/test/3_271/frame.jpg" \
OUTPUT_ROOT="${M0A_DIR}" \
MAX_CHUNKS=7 \
NUM_INFERENCE_STEPS=50 \
EFFICIENCY_METRICS_PATH="${M0A_DIR}/efficiency.json" \
EFFICIENCY_RUNTIME_LOG="${M0A_DIR}/efficiency.jsonl" \
RESUME_STATE_PATH="${M0A_DIR}/resume_state.pt" \
bash "${REPO_DIR}/test_slotmem_stage2.sh" >"${M0A_DIR}/run.log" 2>&1
inference_status=$?
"${PYTHON_BIN}" -m utest.stage_reports m0a \
  --output-dir "${M0A_DIR}" \
  --efficiency "${M0A_DIR}/efficiency.json" \
  --platform-manifest "${PLATFORM_MANIFEST}" \
  --out "${RUN_DIR}/m0a_report.json" | tee "${RUN_DIR}/m0a_validation.log"
m0a_status=$?
set -e

PREREQUISITES="${RUN_DIR}/m0b_prerequisites.json"
"${PYTHON_BIN}" - "${PREREQUISITES}" <<'PY'
import json, os, sys
truth = {"1", "true", "yes", "on"}
payload = {
    "official_inputs": os.environ.get("M0B_OFFICIAL_INPUTS", "0").lower() in truth,
    "official_preprocessing": os.environ.get("M0B_OFFICIAL_PREPROCESSING", "0").lower() in truth,
    "official_checkpoint": os.environ.get("M0B_OFFICIAL_CHECKPOINT", "0").lower() in truth,
    "official_evaluator": os.environ.get("M0B_OFFICIAL_EVALUATOR", "0").lower() in truth,
}
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(payload, indent=2))
PY

m0b_args=(
  "${PYTHON_BIN}" -m utest.stage_reports m0b
  --prerequisites "${PREREQUISITES}"
  --out "${RUN_DIR}/m0b_report.json"
)
if [[ -n "${M0B_METRIC_JSON:-}" ]]; then
  [[ -f "${M0B_METRIC_JSON}" ]] || { echo "[stage] M0B_METRIC_JSON not found" >&2; exit 2; }
  m0b_args+=(--metric-json "${M0B_METRIC_JSON}")
fi
set +e
"${m0b_args[@]}" | tee "${RUN_DIR}/m0b_validation.log"
m0b_status=$?
set -e

"${PYTHON_BIN}" - "${RUN_DIR}" "${inference_status}" "${m0a_status}" "${m0b_status}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
e0 = json.loads((root / "e0.json").read_text())
m0a = json.loads((root / "m0a_report.json").read_text()) if (root / "m0a_report.json").exists() else {"status": "failed"}
m0b = json.loads((root / "m0b_report.json").read_text()) if (root / "m0b_report.json").exists() else {"status": "failed"}
cuda_preflight = json.loads((root / "cuda_preflight.json").read_text())
summary = {
    "E0": "passed" if e0.get("passes_min_viable") else "blocked",
    "eligible_stories": e0.get("n_eligible_stories"),
    "CUDA_preflight": cuda_preflight.get("status"),
    "M0a": m0a.get("status"),
    "M0b": m0b.get("status"),
    "process_exit_codes": {"inference": int(sys.argv[2]), "m0a": int(sys.argv[3]), "m0b": int(sys.argv[4])},
}
(root / "stage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "[stage] artifacts: ${RUN_DIR}"
(( inference_status == 0 && m0a_status == 0 && m0b_status == 0 ))
