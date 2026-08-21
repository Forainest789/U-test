#!/usr/bin/env bash
# Prepare and run the Mara delta-8 Q* experiment on the configured cloud host.
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ROOT="${ASSET_ROOT:-/data/long_term_data/shixiao/videomem/U-test}"
INPUTS="${INPUTS:-${ASSET_ROOT}/runs/delta8_inputs}"

REFERENCE="${REFERENCE:-${ASSET_ROOT}/runs/mara.jpeg}"
TEACHER="${TEACHER:-${ASSET_ROOT}/runs/person_reappearance_delta8_chunk_008_teacher.mp4}"
EVENT_TEMPLATE="${SOURCE_ROOT}/utest/events/person_reappearance_delta8.json"
STORY="${SOURCE_ROOT}/utest/events/person_reappearance_delta8_story.json"
EVENT_JSON="${EVENT_JSON:-${INPUTS}/person_reappearance_delta8_event.json}"
TEACHER_MANIFEST="${FUTURE_TARGET_MANIFEST:-${INPUTS}/person_reappearance_delta8_chunk_008_teacher.manifest.json}"

BASE_INFERENCE_ARGS="${BASE_INFERENCE_ARGS:-${ASSET_ROOT}/runs/m0a_and_stage_gates/stage_gates/slotmem_m0_001/m0a/inference_args.yaml}"
PLATFORM_MANIFEST="${PLATFORM_MANIFEST:-${ASSET_ROOT}/runs/m0a_and_stage_gates/stage_gates/slotmem_m0_001/platform.manifest.json}"
DONOR_PAYLOAD="${DONOR_PAYLOAD:-${ASSET_ROOT}/runs/fixed_prefix_sample5_quick_20260819/donor_payload.pt}"
OLD_DONOR_MANIFEST="${OLD_DONOR_MANIFEST:-${ASSET_ROOT}/runs/fixed_prefix_sample5_quick_20260819/donor_manifest.json}"
DONOR_MANIFEST="${DONOR_MANIFEST:-${INPUTS}/person_reappearance_delta8_donor_manifest.json}"

UTEST_ENV="${UTEST_ENV:-slotmem}"
PYTHON_BIN="${PYTHON_BIN:-python}"
QSTAR_TIMESTEP_INDICES="${QSTAR_TIMESTEP_INDICES:-0,12,25,37,49}"
QSTAR_NOISE_SEED="${QSTAR_NOISE_SEED:-0}"
RUN_ROLLOUT="${RUN_ROLLOUT:-1}"
SLOTMEM_OFFLOAD_MODELS="${SLOTMEM_OFFLOAD_MODELS:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if command -v conda >/dev/null 2>&1 && [[ "${CONDA_DEFAULT_ENV:-}" != "${UTEST_ENV}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${UTEST_ENV}"
fi

if ! git -C "${SOURCE_ROOT}" cat-file -e c6fa98c^{commit} 2>/dev/null || \
   ! git -C "${SOURCE_ROOT}" merge-base --is-ancestor c6fa98c HEAD; then
  echo "[delta8] source must include c6fa98c (Q* current-step injection fix)" >&2
  exit 2
fi
if [[ -n "$(git -C "${SOURCE_ROOT}" status --porcelain)" ]]; then
  echo "[delta8] source worktree is dirty: ${SOURCE_ROOT}" >&2
  git -C "${SOURCE_ROOT}" status --short >&2
  exit 2
fi

mkdir -p "${INPUTS}"
required=(
  "${REFERENCE}" "${TEACHER}" "${EVENT_TEMPLATE}" "${STORY}"
  "${BASE_INFERENCE_ARGS}" "${PLATFORM_MANIFEST}"
  "${DONOR_PAYLOAD}" "${OLD_DONOR_MANIFEST}"
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || { echo "[delta8] missing file: ${path}" >&2; exit 2; }
done
command -v ffprobe >/dev/null 2>&1 || { echo "[delta8] ffprobe is required" >&2; exit 2; }

probe_json="$(ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,nb_read_frames -of json "${TEACHER}")"
"${PYTHON_BIN}" -c '
import json, sys
stream = json.load(sys.stdin)["streams"][0]
actual = {
    "width": int(stream["width"]),
    "height": int(stream["height"]),
    "r_frame_rate": stream["r_frame_rate"],
    "nb_read_frames": int(stream["nb_read_frames"]),
}
expected = {"width": 832, "height": 480, "r_frame_rate": "16/1", "nb_read_frames": 81}
if actual != expected:
    raise SystemExit(f"teacher format mismatch: expected={expected}, actual={actual}")
print("[delta8] teacher format:", actual)
' <<<"${probe_json}"

"${PYTHON_BIN}" - \
  "${EVENT_TEMPLATE}" "${STORY}" "${REFERENCE}" "${TEACHER}" \
  "${EVENT_JSON}" "${TEACHER_MANIFEST}" "${OLD_DONOR_MANIFEST}" \
  "${DONOR_PAYLOAD}" "${DONOR_MANIFEST}" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    event_template,
    story,
    reference,
    teacher,
    event_output,
    teacher_output,
    old_donor_manifest,
    donor_payload,
    donor_output,
) = map(pathlib.Path, sys.argv[1:])

def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def write_json(path: pathlib.Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)

event = json.loads(event_template.read_text(encoding="utf-8"))
event["reference_path"] = str(reference.resolve())
event["source_json_path"] = str(story.resolve())
write_json(event_output, event)

teacher_manifest = {
    "story_id": "person_reappearance_delta8",
    "target_chunk_idx": 8,
    "video_path": str(teacher.resolve()),
    "video_sha256": sha256(teacher),
    "source_type": "independent_teacher",
    "generated_by_arm": False,
    "generated_by_evaluated_model": False,
    "frame_count": 81,
    "fps": 16,
    "width": 832,
    "height": 480,
    "reference_path": str(reference.resolve()),
    "reference_sha256": sha256(reference),
}
write_json(teacher_output, teacher_manifest)

old_manifest = json.loads(old_donor_manifest.read_text(encoding="utf-8"))
row = dict(old_manifest["pairs"][0])
row["target_story_id"] = "person_reappearance_delta8"
row["target_entity_uid"] = "person_reappearance_delta8::mara"
row["payload_path"] = str(donor_payload.resolve())
row["payload_sha256"] = sha256(donor_payload)
write_json(donor_output, {"pairs": [row]})

print(f"[delta8] event: {event_output}")
print(f"[delta8] teacher manifest: {teacher_output}")
print(f"[delta8] donor manifest: {donor_output}")
PY

cd "${SOURCE_ROOT}"
"${PYTHON_BIN}" -m utest.content_audit --self-check
"${PYTHON_BIN}" -m utest.qstar_probe --self-check

PRECHECK="${INPUTS}/input_contract_preflight.json"
"${PYTHON_BIN}" -m utest.input_contract \
  --event "${EVENT_JSON}" \
  --donor "${DONOR_PAYLOAD}" \
  --donor-manifest "${DONOR_MANIFEST}" \
  --future-target-video "${TEACHER}" \
  --future-target-manifest "${TEACHER_MANIFEST}" \
  --arms-root "${ASSET_ROOT}/runs/qstar_person_reappearance_delta8/arms" \
  --report "${PRECHECK}"

STAMP="$(date +%Y%m%d_%H%M%S)"
DRY_ROOT="${ASSET_ROOT}/runs/qstar_delta8_dryrun_${STAMP}"
RUN_ROOT="${EVENT_RUN_ROOT:-${ASSET_ROOT}/runs/qstar_person_reappearance_delta8_${STAMP}}"
[[ ! -e "${DRY_ROOT}" ]] || { echo "[delta8] dry-run output exists: ${DRY_ROOT}" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "[delta8] full output exists: ${RUN_ROOT}" >&2; exit 2; }

common_env=(
  "EVENT_JSON=${EVENT_JSON}"
  "FUTURE_TARGET_VIDEO=${TEACHER}"
  "FUTURE_TARGET_MANIFEST=${TEACHER_MANIFEST}"
  "BASE_INFERENCE_ARGS=${BASE_INFERENCE_ARGS}"
  "PLATFORM_MANIFEST=${PLATFORM_MANIFEST}"
  "DONOR_PAYLOAD=${DONOR_PAYLOAD}"
  "DONOR_MANIFEST=${DONOR_MANIFEST}"
  "QSTAR_TIMESTEP_INDICES=${QSTAR_TIMESTEP_INDICES}"
  "QSTAR_NOISE_SEED=${QSTAR_NOISE_SEED}"
  "RUN_ROLLOUT=${RUN_ROLLOUT}"
  "SLOTMEM_OFFLOAD_MODELS=${SLOTMEM_OFFLOAD_MODELS}"
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "UTEST_ENV=${UTEST_ENV}"
  "PYTHON_BIN=${PYTHON_BIN}"
)
if [[ -n "${CID_SCORER:-}" ]]; then common_env+=("CID_SCORER=${CID_SCORER}"); fi

echo "[delta8] strict dry-run: ${DRY_ROOT}"
env "${common_env[@]}" \
  "EVENT_RUN_ROOT=${DRY_ROOT}" \
  "DRY_RUN=1" \
  bash "${SOURCE_ROOT}/scripts/run_slotmem_qstar_event.sh"
"${PYTHON_BIN}" -m json.tool "${DRY_ROOT}/run_manifest.json" >/dev/null

if [[ "${DRY_RUN_ONLY:-0}" == "1" ]]; then
  echo "[delta8] DRY_RUN_ONLY=1; preflight complete"
  echo "[delta8] manifest: ${DRY_ROOT}/run_manifest.json"
  exit 0
fi

LOG_FILE="${INPUTS}/$(basename "${RUN_ROOT}").log"
echo "[delta8] full run: ${RUN_ROOT}"
env "${common_env[@]}" \
  "EVENT_RUN_ROOT=${RUN_ROOT}" \
  "DRY_RUN=0" \
  bash "${SOURCE_ROOT}/scripts/run_slotmem_qstar_event.sh" 2>&1 | tee "${LOG_FILE}"

echo "[delta8] complete"
echo "[delta8] run root: ${RUN_ROOT}"
echo "[delta8] log: ${LOG_FILE}"
echo "[delta8] Q*: ${RUN_ROOT}/qstar/qstar_report.json"
echo "[delta8] arms: ${RUN_ROOT}/arms/intervention_contract.json"
