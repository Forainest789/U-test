# Q* Teacher-Forced Current-Step Injection Implementation Plan

> **For agentic workers:** Implement inline in this session; no subagent is requested. Use one red-green cycle per task.

**Goal:** Make a one-step teacher-forced Q* prediction consume role-query features from the same frozen timestep and fail closed when a present target payload is not actually injected.

**Architecture:** Add one focused `SlotMemInferenceEngine` helper that reuses the existing semantic-probe and payload-building methods. `ReferenceInferenceRuntime.generate_chunk` calls it only for teacher-forced non-`layer7_single` inference before the measured forward. Add a pure Q* diagnostic validator at the production predictor boundary.

**Tech Stack:** Python, PyTorch, pytest.

## Global Constraints

- Reuse the existing prefix snapshot and Q* inputs.
- Do not advance the scheduler, mutate weights, write the memory bank, or change normal rollout behavior.
- Add no dependency and no new inference mode.
- A present target payload with disabled, empty, non-finite, or zero measured injection fails closed.

---

### Task 1: Current-step query prepass

**Files:**
- Modify: `infer_slotmem.py`
- Modify: `reference_inference_runtime.py:1138-1217`
- Test: `utest/tests/test_inference_hotpath.py`

**Interfaces:**
- Produces: `SlotMemInferenceEngine._prepare_teacher_forced_query_payload(...) -> tuple[dict | None, dict | None]`
- Consumes: existing `_run_character_semantic_probe` and `_build_character_mask_payload_from_probe` behavior.

- [ ] **Step 1: Write the failing test**

Instantiate `SlotMemInferenceEngine` without its GPU constructor, replace the two existing probe helpers with deterministic fakes, and assert the new helper returns a non-empty current role box and query payload using the latent patch dimensions.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest utest/tests/test_inference_hotpath.py::test_teacher_forced_query_prepass_builds_current_step_payload -q`

Expected: FAIL because `_prepare_teacher_forced_query_payload` does not exist.

- [ ] **Step 3: Write minimal implementation**

Add the helper to `SlotMemInferenceEngine`, then call it from `generate_chunk` when `teacher_forced_probe is not None`, memory is active, and selection mode is not `layer7_single`. Assign the returned values to `active_query_role_boxes` and `active_query_feature_payload`; skip the old hook-and-next-step capture for that measured forward.

- [ ] **Step 4: Run test to verify it passes**

Run the focused test above, followed by `python -m pytest utest/tests/test_inference_hotpath.py -q`.

---

### Task 2: Production probe fail-closed diagnostic

**Files:**
- Modify: `utest/qstar_probe.py`
- Test: `utest/tests/test_qstar.py`

**Interfaces:**
- Produces: `validate_measured_injection(run_name: str, target_payload_present: bool, result: Mapping) -> None`
- Consumes: `sparse_role_memory_stats_by_layer` returned by `generate_chunk`.

- [ ] **Step 1: Write the failing test**

Assert that a correct-arm result with a present target payload and only disabled/zero layer diagnostics raises `ValueError`, while a positive finite effective layer delta passes and `no_memory` is exempt.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest utest/tests/test_qstar.py::test_present_payload_requires_measured_sparse_injection -q`

Expected: FAIL because `validate_measured_injection` does not exist.

- [ ] **Step 3: Write minimal implementation**

Validate the per-layer diagnostics immediately after `generate_chunk`; require at least one row with `enabled > 0`, `selected_memory_tokens > 0`, and finite `effective_delta_norm > 0`. Record the maximum effective delta as `injection_delta_norm` instead of the role-head norm.

- [ ] **Step 4: Run tests**

Run the focused test, `python -m pytest utest/tests/test_qstar.py -q`, then `python -m pytest utest/tests -q`.

---

### Task 3: Existing-prefix GPU rerun

**Files:**
- Reuse: `runs/qstar_sample5_debug_20260820_182243/prefix/prefix_state.pt`
- Replace after archiving: `runs/qstar_sample5_debug_20260820_182243/qstar/`

- [ ] **Step 1: Run the existing-prefix Q* command**

Use the recorded `qstar-probe` argv from `.commands.jsonl`, pointing `--prefix` to the existing prefix and a fresh Q* output directory so the invalid report remains recoverable.

- [ ] **Step 2: Verify the new evidence**

Require five cells, positive measured injection for payload-bearing confirmatory arms, exact correct-repeat reproducibility, and at least one prediction hash difference between `correct` and an intervention arm. Do not resume remaining rollouts until these checks pass.
