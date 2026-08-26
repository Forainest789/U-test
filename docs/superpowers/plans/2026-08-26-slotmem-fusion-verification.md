# SlotMem Fusion Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded `--verify-fusion` identity-probe mode that decomposes existing denoising effects and conditionally runs matched correct/wrong alpha sweeps.

**Architecture:** Keep the frozen SlotMem runtime and payload path unchanged. Add pure analysis helpers around existing teacher-forced predictions, reuse `sparse_role_memory_layer_scales` and token diagnostics for at most two V1 cells, and report the result without running S2/S3.

**Tech Stack:** Python 3.10+, PyTorch 2.7 on A100, pytest, existing SlotMem/Wan2.2 runtime, FlashAttention 2.

## Global Constraints

- No new model module, dependency, router, checkpoint field, payload format, or weight mutation.
- Q*, smoke, normal identity mode, S2, S3, and decoded generation retain existing behavior.
- Verification mode uses conditional-only arms and reports zero unconditional DiT calls.
- V0 adds zero forwards; V1 adds at most 16 measured arms; total ceiling is 41.
- Verification mode never runs S2/S3 and never emits identity-token labels.
- Prediction and token-feature tensors are transient and are not serialized.
- Alpha-one captured predictions must match S0/S1 prediction hashes exactly.
- Layer groups and scale mappings are restored in `finally` blocks.

## File Map

- Modify `utest/identity_token_probe.py`: V0 math, V1 selection/sweep, feature summaries, classification, CLI and report integration.
- Modify `utest/tests/test_identity_token_probe.py`: behavior-first tests.
- Modify `utest/README.md`: A100 command and interpretation.
- Do not modify `infer_slotmem.py`, `reference_inference_runtime.py`, `utest/qstar_probe.py`, weights, or assets.

---

### Task 1: V0 error decomposition

**Files:**
- Modify: `utest/identity_token_probe.py`
- Test: `utest/tests/test_identity_token_probe.py`

**Interfaces:**
- Produces `prediction_error_decomposition(prediction, baseline, target) -> dict`.
- Produces `_v0_cells(screening_cells, screening_records, cells) -> list[dict]`.

- [ ] **Step 1: Write one failing scalar test**

```python
def test_prediction_error_decomposition_reconstructs_loss_and_predicts_rescue():
    result = prediction_error_decomposition([1.5, 0.5], [1.0, 1.0], [0.0, 0.0])
    assert result["loss_delta_from_no_memory"] == 0.25
    assert result["directional_alignment"] == 0.0
    assert result["delta_energy"] == 0.25
    assert abs(result["decomposition_residual"]) < 1e-12
```

- [ ] **Step 2: Run RED**

```bash
python -m pytest utest/tests/test_identity_token_probe.py::test_prediction_error_decomposition_reconstructs_loss_and_predicts_rescue -q
```

Expected: import failure because the helper does not exist.

- [ ] **Step 3: Implement the minimal reducer**

Use device-local float tensor arithmetic in production and flat numeric sequences in tests. Compute `loss_delta`, `A=2*mean((baseline-target)*delta)`, `B=mean(delta^2)`, reconstruction residual, `alpha*=clip(-A/(2B),0,1)`, and predicted gain. Reject non-finite values and residuals above the specification tolerance.

- [ ] **Step 4: Run GREEN and compilation**

```bash
python -m pytest utest/tests/test_identity_token_probe.py::test_prediction_error_decomposition_reconstructs_loss_and_predicts_rescue -q
python -m py_compile utest/identity_token_probe.py
```

- [ ] **Step 5: Add V0 integration RED then GREEN**

Use fake screening records sharing a timestep baseline. Assert each correct/wrong cell gains `error_decomposition`, zero appears only where measured, and the helper performs no engine call. Match the baseline by timestep and target by frozen `ProbeCell`.

- [ ] **Step 6: Run identity tests and commit**

```bash
python -m pytest utest/tests/test_identity_token_probe.py -q
git add utest/identity_token_probe.py utest/tests/test_identity_token_probe.py
git commit -m "Add identity fusion error decomposition"
```

---

### Task 2: V1 trigger and deterministic selection

**Files:** same two Task 1 files.

**Interfaces:** Produces `select_fusion_verification_cells(cells, trigger_floor, max_cells=2) -> dict`.

- [ ] **Step 1: Write a failing selection test**

Build three qualifying cells and one positive-alignment rejection. Assert the primary has highest predicted gain and the second prefers a different timestep over a same-timestep runner-up.

- [ ] **Step 2: Run RED**

Expected: import failure because the selector does not exist.

- [ ] **Step 3: Implement frozen rules**

Require finite correct decomposition, `alignment < 0`, strict `0 < alpha < 1`, and `gain > floor`. Sort by `(-gain,-q_content,timestep,layer_group)`. Select first, then prefer different timestep, then different group, then next row. Return copied JSON-safe rows.

- [ ] **Step 4: Run GREEN and commit**

```bash
python -m pytest utest/tests/test_identity_token_probe.py -q
git add utest/identity_token_probe.py utest/tests/test_identity_token_probe.py
git commit -m "Select bounded fusion alpha probes"
```

---

### Task 3: Matched alpha sweep and host diagnostics

**Files:** same two Task 1 files.

**Interfaces:**
- Extends `_run_screening_forward(..., capture_token_diagnostics=False)`.
- Produces `run_fusion_alpha_sweep(context, selected_cells, screening_records, args) -> dict`.

- [ ] **Step 1: Write a failing fake-engine sweep test**

Assert eight calls per selected cell in alpha-major/arm-minor order, only selected layer scales change, original scales/layers restore after success and exception, and alpha-one SHAs match screening records.

- [ ] **Step 2: Run RED**

Expected: the sweep helper does not exist.

- [ ] **Step 3: Extend the existing forward wrapper**

Pass the optional diagnostic flag through the existing teacher-forced request. Keep raw result/features only as private transient fields for aggregation; normal callers are unchanged.

- [ ] **Step 4: Implement scoped alpha execution**

For alpha `(0.0,0.25,0.5,1.0)` and arms `(correct,wrong)`, set existing scale-map entries for the chosen group, execute conditional-only teacher forcing, then restore the original group and mapping in `finally`. Raise above 16 calls or on alpha-one SHA mismatch.

- [ ] **Step 5: Add host aggregation RED then GREEN**

With compact two-token fixtures, verify host/delta cosine, alpha-zero host similarity, host norm drift, and correct/wrong delta cosine on shared indices. Reject missing, misaligned, or non-finite tensors. Persist scalar aggregates only.

- [ ] **Step 6: Run identity tests and commit**

```bash
python -m pytest utest/tests/test_identity_token_probe.py -q
python -m py_compile utest/identity_token_probe.py
git add utest/identity_token_probe.py utest/tests/test_identity_token_probe.py
git commit -m "Add bounded SlotMem fusion alpha sweep"
```

---

### Task 4: Classification, orchestration, output, and docs

**Files:**
- Modify: `utest/identity_token_probe.py`
- Modify: `utest/tests/test_identity_token_probe.py`
- Modify: `utest/README.md`

**Interfaces:**
- Produces `classify_fusion_mechanism(v0, alpha_records, trigger_floor) -> dict`.
- Adds CLI `--verify-fusion`.
- Adds report section/file `fusion_verification` / `fusion_verification.json`.

- [ ] **Step 1: Classification RED then GREEN**

Test `supplement_candidate`, `representation_competition_candidate`, `direction_mismatch`, `path_or_routing_confound`, and `no_authority` with scalar curves. Return a primary label plus component flags; never call competition proven.

- [ ] **Step 2: End-to-end mode RED**

Using fake CPU contexts, assert V0 uses the original 25 calls, no candidate skips V1, a qualifying case adds no more than 16 calls, S2 stays absent, identity gate is pending for verification, and budget is 41.

- [ ] **Step 3: Integrate mode and count contract**

Parse the flag, run V1 only in verification mode, bypass S2/S3, merge V1 records into measured-arm and DiT reconciliation, set budget 41, reject unconditional counts, and write `fusion_verification.json` when present.

- [ ] **Step 4: Update README**

Document prefix reuse, `--verify-fusion`, 25+16 budget, no-S2 boundary, fields, and conservative interpretation.

- [ ] **Step 5: Run regression and commit**

```bash
python -m pytest utest/tests/test_identity_token_probe.py utest/tests/test_identity_token_runner.py utest/tests/test_qstar.py utest/tests/test_inference_hotpath.py -q
python -m utest.identity_token_probe --self-check
python -m py_compile utest/identity_token_probe.py infer_slotmem.py reference_inference_runtime.py utest/qstar_probe.py
git diff --check
git add utest/identity_token_probe.py utest/tests/test_identity_token_probe.py utest/README.md
git commit -m "Integrate SlotMem fusion verification mode"
```

---

### Task 5: Final verification, push, and A100 handoff

- [ ] **Step 1: Run fresh local evidence gate**

```bash
python -m pytest utest/tests --ignore=utest/tests/test_content_audit.py -q -rs
python -m utest.identity_token_probe --self-check
python -m py_compile infer_slotmem.py reference_inference_runtime.py utest/identity_token_probe.py utest/qstar_probe.py
git diff --check
git status --short
```

- [ ] **Step 2: Push and verify remote SHA**

```bash
git push origin fix/qstar-estimand-and-provenance
git fetch origin fix/qstar-estimand-and-provenance
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/fix/qstar-estimand-and-provenance)"
```

- [ ] **Step 3: Hand off A100 command**

Reuse the frozen prefix/arms with a fresh output, FlashAttention 2, conditional-only identity calls, `--verify-fusion`, and no decoded validation. Require decomposition reconstruction, alpha-one SHA parity, count reconciliation, and the 41-arm ceiling before interpreting mechanism labels.

