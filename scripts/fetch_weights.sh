#!/usr/bin/env bash
# Fetch the frozen platform: Wan2.2-I2V-A14B base + SlotMem Stage-2 checkpoints, then
# record exactly what was fetched. No code is cloned -- this repo IS the SlotMem fork.
#
# Run this from an already-active Conda environment:
#   SKIP_PIP=1 bash scripts/fetch_weights.sh
#
# Disk: ~126 GB base + ~21 GB checkpoints. VRAM at inference is a separate budget; see
# docs/research-plan.md.
main() (
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_ROOT="${CKPT_ROOT:-${REPO_DIR}}"
WAN22_DIR="${WAN22_DIR:-${REPO_DIR}/../wan_models/Wan2.2-I2V-A14B}"
WAN22_REPO="${WAN22_REPO:-Wan-AI/Wan2.2-I2V-A14B}"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${CONDA_PREFIX:?run this script from the already-active Conda environment}"
ACTIVE_ENV="${CONDA_DEFAULT_ENV:-$(basename "${CONDA_PREFIX}")}"

echo "[utest] using active Conda environment: ${ACTIVE_ENV}"

ensure_hf_cli() {
  command -v hf >/dev/null 2>&1 && return
  "${PYTHON_BIN}" -m pip install -U "huggingface_hub[cli]"
}

# 1. Dependencies (SlotMem's own pins).
if [[ "${SKIP_PIP:-0}" != "1" ]]; then
  "${PYTHON_BIN}" -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
  ( cd "${REPO_DIR}" && "${PYTHON_BIN}" -m pip install -e . \
      && "${PYTHON_BIN}" -m pip install -r requirements_slotmem.txt )
fi

# 2. SlotMem checkpoints. Stage-2 is the platform, so Stage-1 alone is not "present":
# checking only stage1 skips the download and the run then dies inside the launcher.
if compgen -G "${CKPT_ROOT}/ckpt/stage1/*.pt" >/dev/null \
   && compgen -G "${CKPT_ROOT}/ckpt/stage2/*.pt" >/dev/null; then
  echo "[utest] ckpt present (stage1 + stage2)"
else
  echo "[utest] fetching ckpt/* (~21 GB) -> ${CKPT_ROOT}"
  ensure_hf_cli
  hf download YilaiLiu-HKU/SlotMem --local-dir "${CKPT_ROOT}" --include "ckpt/*"
fi
for required in stage2/stage2_low.pt stage2/stage2_high.pt; do
  [[ -f "${CKPT_ROOT}/ckpt/${required}" ]] || {
    echo "[utest] FATAL: missing ckpt/${required}; Stage-2 is the frozen platform" >&2
    return 1
  }
done

# 3. Wan2.2 base.
if [[ -d "${WAN22_DIR}/low_noise_model" && -d "${WAN22_DIR}/high_noise_model" ]]; then
  echo "[utest] base model present: ${WAN22_DIR}"
else
  echo "[utest] downloading ${WAN22_REPO} (~126 GB) -> ${WAN22_DIR}"
  mkdir -p "${WAN22_DIR}"
  ensure_hf_cli
  hf download "${WAN22_REPO}" --local-dir "${WAN22_DIR}"
fi
# Assert AFTER the fetch, not only before it. hf download returns 0 on a resumed or
# partial pull, and a base model that is present-but-short fails deep inside the first
# denoising step instead of here. The floor is deliberately loose: it catches "a few
# files landed", not "one shard is truncated" -- the manifest sizes are for that.
wan_bytes=$(du -sb "${WAN22_DIR}" 2>/dev/null | cut -f1)
wan_shards=$(find "${WAN22_DIR}" \( -name '*.safetensors' -o -name '*.pth' \) | wc -l)
echo "[utest] base model: $((wan_bytes / 1024 / 1024 / 1024)) GiB in ${wan_shards} weight files"
for expert in low_noise_model high_noise_model; do
  [[ -d "${WAN22_DIR}/${expert}" ]] && [[ -n "$(ls -A "${WAN22_DIR}/${expert}" 2>/dev/null)" ]] || {
    echo "[utest] FATAL: ${WAN22_DIR}/${expert} missing or empty" >&2
    return 1
  }
done
if (( wan_bytes < 100 * 1024 * 1024 * 1024 )); then
  echo "[utest] FATAL: base model is $((wan_bytes / 1024 / 1024 / 1024)) GiB, expected ~126." >&2
  echo "[utest] Re-run this script; hf download resumes. Set HF_TOKEN for higher rate limits." >&2
  return 1
fi

# 4. Provenance. A commit hash describes the tree only when the tree is clean, which is
# why a bare git_commit has misled this project before; the dirty flag and the checkpoint
# hashes are what make a result attributable.
MANIFEST="${REPO_DIR}/platform.manifest.json"
echo "[utest] hashing checkpoints -> ${MANIFEST} (a few minutes for ~21 GB)"
CKPT_ROOT="${CKPT_ROOT}" WAN22_DIR="${WAN22_DIR}" REPO_DIR="${REPO_DIR}" \
MANIFEST="${MANIFEST}" "${PYTHON_BIN}" - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path

ckpt = Path(os.environ["CKPT_ROOT"]) / "ckpt"
wan = Path(os.environ["WAN22_DIR"])
repo = os.environ["REPO_DIR"]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()

def git(*args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args],
                          capture_output=True, text=True).stdout.strip()

Path(os.environ["MANIFEST"]).write_text(json.dumps({
    "repo_commit": git("rev-parse", "HEAD"),
    "repo_dirty": bool(git("status", "--porcelain")),
    "checkpoints": {
        str(p.relative_to(ckpt)): {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(ckpt.rglob("*.pt"))
    },
    # The 126 GB base is listed by size only: hashing it on every setup costs more than
    # it tells us, and the upstream release is versioned.
    "wan22_files": {
        str(p.relative_to(wan)): p.stat().st_size
        for p in sorted(wan.rglob("*")) if p.is_file()
    },
}, indent=2), encoding="utf-8")
print("[utest] manifest written")
PY

"${PYTHON_BIN}" -m utest.content_audit --self-check
"${PYTHON_BIN}" -m utest.eligibility --self-check

cat <<EOF

[utest] ready. manifest -> ${MANIFEST}

  M0a (official sample, Stage-2) -- record wall time and peak VRAM:
    CONDA_ENV=${ACTIVE_ENV} CUDA_VISIBLE_DEVICES=0 \\
    DUAL_EXPERT_LOAD_MODE=active DUAL_EXPERT_MANAGE_AUX_MODELS=1 \\
    CKPT_DIR=${WAN22_DIR} \\
    JSON_PATH=${REPO_DIR}/sample/test/3_271/rewrite_caption.json \\
    REF_IMAGE_PATH=${REPO_DIR}/sample/test/3_271/frame.jpg \\
    time bash ${REPO_DIR}/test_slotmem_stage2.sh

  E0 (zero GPU, run this FIRST -- it gates the whole method line):
    python -m utest.eligibility --data-root <narrastream-scripts> --out runs/e0.json
EOF
)

main "$@"
