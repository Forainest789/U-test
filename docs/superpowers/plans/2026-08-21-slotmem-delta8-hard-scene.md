# SlotMem Delta-8 Hard-Scene Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing Mara delta-8 story the default cloud Q* experiment without changing model logic or the completed sample-5 run.

**Architecture:** Reuse the existing generic strict runner and the already-tested `person_reappearance_delta8` story/event pair. Switch only the compatibility launcher default and documented cloud bindings, then verify the resolved dry-run command chain points at the delta-8 event.

**Tech Stack:** Bash, JSON, Python 3, pytest, existing SlotMem U-test runner.

## Global Constraints

- Do not modify artifacts under `runs/qstar_sample5_debug_20260820_182243`.
- Do not change checkpoints, model equations, scheduler behavior, prompts outside the frozen delta-8 story, or seven-arm intervention semantics.
- Reuse `utest/events/person_reappearance_delta8_story.json` and `utest/events/person_reappearance_delta8.json`; do not duplicate them.
- The target is Mara's first reappearance at chunk 8 after absence from chunks 1 through 7.
- The held-out teacher video and provenance manifest remain mandatory independent inputs.
- GPU generation is not performed in the local workspace.

---

### Task 1: Make delta-8 the compatibility launcher's default event

**Files:**
- Modify: `scripts/run_slotmem_utest.sh`
- Test: `utest/tests/test_qstar_runner.py`

**Interfaces:**
- Consumes: repository-relative `utest/events/person_reappearance_delta8.json` and the existing environment variables accepted by `scripts/run_slotmem_qstar_event.sh`.
- Produces: `EVENT_JSON` defaulting to the absolute repository event path when the caller does not provide it; an explicit caller value still wins.

- [ ] **Step 1: Write the failing launcher-default test**

Add a test that invokes `scripts/run_slotmem_utest.sh` with every strict input except `EVENT_JSON`, sets `DRY_RUN=1`, and reads `run_manifest.json`. Assert that the `input-contract-preflight` command's `--event` value resolves to `utest/events/person_reappearance_delta8.json`.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```powershell
python -m pytest utest/tests/test_qstar_runner.py -q
```

Expected: the new test fails because the compatibility launcher currently requires `EVENT_JSON`.

- [ ] **Step 3: Add the minimum default**

After `REPO_DIR` is resolved, export:

```bash
export EVENT_JSON="${EVENT_JSON:-${REPO_DIR}/utest/events/person_reappearance_delta8.json}"
```

Remove `EVENT_JSON` from the compatibility launcher's missing-variable list. Update its usage example to reference:

```text
utest/events/person_reappearance_delta8.json
/data/targets/person_reappearance_delta8_chunk_008_teacher.mp4
/data/targets/person_reappearance_delta8_chunk_008_teacher.manifest.json
/data/runs/qstar_person_reappearance_delta8
```

- [ ] **Step 4: Run the focused test and verify success**

Run:

```powershell
python -m pytest utest/tests/test_qstar_runner.py -q
```

Expected: both strict-runner and compatibility-default dry-run tests pass.

- [ ] **Step 5: Commit the launcher change**

```powershell
git add scripts/run_slotmem_utest.sh utest/tests/test_qstar_runner.py
git commit -m "Default SlotMem U-test to delta-8 event"
```

### Task 2: Update the cloud configuration documentation

**Files:**
- Modify: `utest/README.md`

**Interfaces:**
- Consumes: the launcher behavior from Task 1 and the fixed event/reference path in `utest/events/person_reappearance_delta8.json`.
- Produces: one copyable cloud command for the delta-8 teacher-forced and seven-arm experiment.

- [ ] **Step 1: Replace the sample-5 primary command**

Bind the documented command to these exact paths:

```bash
EVENT_JSON="$PWD/utest/events/person_reappearance_delta8.json" \
FUTURE_TARGET_VIDEO=/data/targets/person_reappearance_delta8_chunk_008_teacher.mp4 \
FUTURE_TARGET_MANIFEST=/data/targets/person_reappearance_delta8_chunk_008_teacher.manifest.json \
BASE_INFERENCE_ARGS=/data/runs/stage_gates/slotmem_m0_001/m0a/inference_args.yaml \
PLATFORM_MANIFEST=/data/runs/stage_gates/slotmem_m0_001/platform.manifest.json \
DONOR_PAYLOAD=/data/events/donor_payload.pt \
DONOR_MANIFEST=/data/events/donor_manifest.json \
EVENT_RUN_ROOT=/data/runs/qstar_person_reappearance_delta8 \
QSTAR_TIMESTEP_INDICES=0,12,25,37,49 \
QSTAR_NOISE_SEED=0 \
RUN_ROLLOUT=1 \
CID_SCORER=/data/videomem/scripts/score_identity.py \
SLOTMEM_OFFLOAD_MODELS=0 \
UTEST_ENV=utest \
bash scripts/run_slotmem_qstar_event.sh
```

- [ ] **Step 2: Replace the teacher manifest example**

Use `story_id: person_reappearance_delta8`, `target_chunk_idx: 8`, the chunk-8 teacher video path above, and retain the independent-teacher provenance fields required by `utest.input_contract`.

- [ ] **Step 3: Check documentation and shell syntax**

Run:

```powershell
rg -n "sample_5_qstar|sample_5_chunk_005|qstar_sample_5" utest/README.md scripts/run_slotmem_utest.sh
& 'C:\Program Files\Git\bin\bash.exe' -n scripts/run_slotmem_utest.sh scripts/run_slotmem_qstar_event.sh
```

Expected: no stale sample-5 primary configuration remains in those two files, and Bash syntax validation exits 0.

- [ ] **Step 4: Commit the documentation change**

```powershell
git add utest/README.md
git commit -m "Document delta-8 cloud Q-star configuration"
```

### Task 3: Verify the frozen hard-scene contract

**Files:**
- Verify only: `utest/events/person_reappearance_delta8_story.json`
- Verify only: `utest/events/person_reappearance_delta8.json`
- Verify only: `utest/tests/test_prefix_contract.py`

**Interfaces:**
- Consumes: the frozen story/event pair and launcher configuration from Tasks 1–2.
- Produces: a locally verified, no-GPU configuration ready to copy to the cloud host.

- [ ] **Step 1: Run the event and runner checks**

```powershell
python -m pytest utest/tests/test_prefix_contract.py::test_long_reappearance_fixture_has_one_establishment_and_one_return utest/tests/test_qstar_runner.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run dependency-light self-checks**

```powershell
python -m utest.content_audit --self-check
python -m utest.qstar_probe --self-check
```

Expected: both commands exit 0.

- [ ] **Step 3: Inspect repository integrity**

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; the prior untracked research note may remain, but no sample-5 run artifacts are modified.

