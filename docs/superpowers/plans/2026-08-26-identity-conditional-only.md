# Identity Probe Conditional-Only Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the unused unconditional DiT branch from identity teacher-forced probes and report the real semantic, conditional, and unconditional DiT invocation counts without changing the conditional estimand.

**Architecture:** Add an opt-in `conditional_only` request at the existing `teacher_forced_probe` boundary. The shared runtime preserves its default CFG path for Q* and normal generation, while identity S0/S1 and S2 opt in and aggregate per-call count metadata into an auditable report.

**Tech Stack:** Python 3.10+, PyTorch 2.7, DiffSynth/Wan2.2, pytest, NVIDIA A100 80GB, BF16, FlashAttention 2.

## Global Constraints

- Only identity teacher-forced probes become conditional-only; Q*, normal generation, and S3 retain CFG behavior.
- No model weights, prefix snapshots, payloads, loss equations, thresholds, gates, or semantic/query localization logic change.
- No new dependency or CUDA kernel is added.
- `forward_count` and `forward_budget` continue to mean measured arms; raw DiT invocations are reported separately.
- The negative smoke result for timestep 25 and layers 5–10 remains unchanged evidence.
- Every behavior change follows one red-green cycle and receives an independently reviewable commit.

## File Map

- `infer_slotmem.py`: records whether the truncated semantic/query prepass actually invoked a DiT.
- `reference_inference_runtime.py`: implements `conditional_only`, preserves the default CFG path, and returns per-call DiT counts and prediction semantics.
- `utest/identity_token_probe.py`: opts identity calls into conditional-only and aggregates per-call counts.
- `utest/tests/test_inference_hotpath.py`: guards runtime ordering, default CFG preservation, and count metadata.
- `utest/tests/test_identity_token_probe.py`: guards identity/Q* separation and report reconciliation.
- `utest/README.md`: documents measured-arm versus raw-DiT accounting and the optimized server rerun.

---

### Task 1: Conditional-only runtime boundary

**Files:**
- Modify: `infer_slotmem.py:1868-2015`
- Modify: `reference_inference_runtime.py:940-1412`
- Test: `utest/tests/test_inference_hotpath.py`

**Interfaces:**
- Consumes: `teacher_forced_probe: dict` and the existing `prediction_cond` tensor.
- Produces: opt-in `teacher_forced_probe["conditional_only"]: bool`, `prediction_semantics: str`, `cfg_composite_available: bool`, and `dit_forward_counts: dict[str, int]`.

- [ ] **Step 1: Write the failing runtime-boundary test**

Add a source-boundary test beside the existing teacher-forced hot-path tests:

```python
def test_conditional_only_skips_unconditional_but_default_keeps_cfg() -> None:
    root = Path(__file__).resolve().parents[2]
    runtime = (root / "reference_inference_runtime.py").read_text(encoding="utf-8")
    infer = (root / "infer_slotmem.py").read_text(encoding="utf-8")

    assert 'conditional_only = bool(teacher_forced_probe.get("conditional_only", False))' in runtime
    assert "if conditional_only:" in runtime
    assert '"prediction_semantics": prediction_semantics' in runtime
    assert '"cfg_composite_available": not conditional_only' in runtime
    assert '"dit_forward_counts": {' in runtime
    assert "self._last_teacher_forced_semantic_prepass_count = 1" in infer

    conditional_branch = runtime[runtime.index("if conditional_only:"):]
    default_unconditional = conditional_branch.index("noise_pred_uncond =")
    return_block = conditional_branch.index('"prediction_cond": noise_pred_cond')
    assert default_unconditional < return_block
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python -m pytest utest/tests/test_inference_hotpath.py::test_conditional_only_skips_unconditional_but_default_keeps_cfg -q
```

Expected: FAIL because `conditional_only` and count metadata do not exist.

- [ ] **Step 3: Record actual semantic prepass invocation**

At the start of `_run_character_semantic_probe`, reset the per-call counter. Set it only immediately before the native probe DiT invocation:

```python
self._last_teacher_forced_semantic_prepass_count = 0
# existing validation and hook setup
self._last_teacher_forced_semantic_prepass_count = 1
run_native_dit_forward(...)
```

This remains zero when role configuration produces no prepass and one when the truncated DiT is invoked.

- [ ] **Step 4: Implement the conditional-only branch and result metadata**

In `generate_chunk`, default the flag to false and reset the semantic counter for every teacher-forced request:

```python
conditional_only = False
if teacher_forced_probe is not None:
    conditional_only = bool(teacher_forced_probe.get("conditional_only", False))
    self._last_teacher_forced_semantic_prepass_count = 0
```

After the unchanged conditional forward and diagnostic snapshots, branch before the unconditional forward:

```python
if teacher_forced_probe is not None and conditional_only:
    noise_pred = noise_pred_cond
    unconditional_count = 0
    prediction_semantics = "conditional"
else:
    # existing memory-aware or native unconditional forward
    noise_pred = noise_pred_uncond + self.args.cfg_scale * (
        noise_pred_cond - noise_pred_uncond
    )
    unconditional_count = 1
    prediction_semantics = "cfg_composite"
```

Extend the teacher-forced result without changing `prediction_cond`:

```python
"prediction_semantics": prediction_semantics,
"cfg_composite_available": not conditional_only,
"dit_forward_counts": {
    "semantic_prepass": int(
        getattr(self, "_last_teacher_forced_semantic_prepass_count", 0)
    ),
    "conditional": 1,
    "unconditional": unconditional_count,
},
```

For `semantic_capture_only`, return counts with one semantic prepass and zero conditional/unconditional forwards.

- [ ] **Step 5: Run hot-path tests and verify GREEN**

Run:

```bash
python -m pytest utest/tests/test_inference_hotpath.py -q
python -m py_compile infer_slotmem.py reference_inference_runtime.py
```

Expected: new source-boundary test passes; Torch-dependent tests pass on the server or skip locally for the existing documented dependency reason.

- [ ] **Step 6: Commit the runtime slice**

```bash
git add infer_slotmem.py reference_inference_runtime.py utest/tests/test_inference_hotpath.py
git commit -m "Add conditional-only teacher probe runtime"
```

---

### Task 2: Identity-only opt-in with Q* preservation

**Files:**
- Modify: `utest/identity_token_probe.py:429-479`
- Modify: `utest/identity_token_probe.py:680-725`
- Test: `utest/tests/test_identity_token_probe.py`
- Test: `utest/tests/test_inference_hotpath.py`

**Interfaces:**
- Consumes: Task 1 `teacher_forced_probe["conditional_only"]`.
- Produces: conditional-only identity S0/S1 and S2 calls; unchanged Q* call dictionaries.

- [ ] **Step 1: Write the failing identity/Q* separation test**

```python
def test_identity_opts_into_conditional_only_without_changing_qstar() -> None:
    root = Path(__file__).resolve().parents[2]
    identity = (root / "utest" / "identity_token_probe.py").read_text(encoding="utf-8")
    qstar = (root / "utest" / "qstar_probe.py").read_text(encoding="utf-8")

    screening = identity[identity.index("def _run_screening_forward("):identity.index("def _screening_cells(")]
    s2 = identity[identity.index("def _s2_model_forward("):identity.index("def _text_positions(")]
    assert '"conditional_only": True' in screening
    assert '"conditional_only": True' in s2
    assert '"conditional_only"' not in qstar
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python -m pytest utest/tests/test_identity_token_probe.py::test_identity_opts_into_conditional_only_without_changing_qstar -q
```

Expected: FAIL because identity calls have not opted in.

- [ ] **Step 3: Add the flag to measured identity calls only**

Add this entry to the `teacher_forced_probe` dictionaries in `_run_screening_forward` and `_s2_model_forward`:

```python
"conditional_only": True,
```

Do not add it to `_semantic_capture`, Q*, S3, or normal generation.

- [ ] **Step 4: Run the focused identity and Q* tests**

```bash
python -m pytest utest/tests/test_identity_token_probe.py utest/tests/test_qstar.py utest/tests/test_inference_hotpath.py -q
```

Expected: identity separation test and existing conditional-Q*/CFG invariants pass.

- [ ] **Step 5: Commit the identity wiring slice**

```bash
git add utest/identity_token_probe.py utest/tests/test_identity_token_probe.py utest/tests/test_inference_hotpath.py
git commit -m "Use conditional-only identity interventions"
```

---

### Task 3: Reconciled DiT accounting and report schema

**Files:**
- Modify: `utest/identity_token_probe.py:429-1120`
- Modify: `utest/identity_token_probe.py:1184-1360`
- Modify: `utest/tests/test_identity_token_probe.py`
- Modify: `utest/README.md:231-330`

**Interfaces:**
- Consumes: Task 1 `dit_forward_counts` on each teacher-forced result.
- Produces: report-level measured-arm, warm-up, semantic-prepass, conditional, unconditional, and raw-invocation counts.

- [ ] **Step 1: Write a failing pure aggregation test**

Import a new pure helper `_sum_dit_forward_counts` and add:

```python
def test_dit_forward_counts_reconcile_measured_warmup_and_semantic_calls() -> None:
    records = [
        {"dit_forward_counts": {"semantic_prepass": 1, "conditional": 1, "unconditional": 0}},
        {"dit_forward_counts": {"semantic_prepass": 0, "conditional": 1, "unconditional": 0}},
        {"dit_forward_counts": {"semantic_prepass": 1, "conditional": 0, "unconditional": 0}},
    ]
    assert _sum_dit_forward_counts(records) == {
        "semantic_prepass": 2,
        "conditional": 2,
        "unconditional": 0,
        "raw": 4,
    }
```

- [ ] **Step 2: Run the aggregation test and verify RED**

```bash
python -m pytest utest/tests/test_identity_token_probe.py::test_dit_forward_counts_reconcile_measured_warmup_and_semantic_calls -q
```

Expected: collection error because `_sum_dit_forward_counts` does not exist.

- [ ] **Step 3: Implement the pure count reducer**

```python
def _sum_dit_forward_counts(records: Sequence[Mapping]) -> dict[str, int]:
    counts = {"semantic_prepass": 0, "conditional": 0, "unconditional": 0}
    for record in records:
        source = record.get("dit_forward_counts", {})
        for name in counts:
            counts[name] += int(source.get(name, 0) or 0)
    return {**counts, "raw": sum(counts.values())}
```

- [ ] **Step 4: Preserve per-call counts through all identity records**

Add `dit_forward_counts` to `_run_screening_forward` output, `_s2_model_forward` output,
`_public_intervention`, and semantic capture bookkeeping. Store the warm-up return record
instead of discarding it:

```python
warmup_record = _run_screening_forward(context, schedule[0])
```

Use the reducer over measured screening records, S2 public interventions, the two
semantic captures, and the warm-up record.

- [ ] **Step 5: Emit the corrected report fields**

Set:

```python
"measured_arm_count": measured_arm_count,
"warmup_arm_count": warmup_forward_count,
"semantic_prepass_count": dit_counts["semantic_prepass"],
"conditional_dit_count": dit_counts["conditional"],
"unconditional_dit_count": dit_counts["unconditional"],
"raw_dit_invocation_count": dit_counts["raw"],
"actual_model_forward_count": dit_counts["raw"],
```

Retain `forward_count == measured_arm_count` and the existing five/50 measured-arm
budgets. Add an invariant that raises if the raw count is not the sum of the three
component counts.

- [ ] **Step 6: Update smoke fake-engine behavior and assertions**

Have the fake `generate_chunk` return conditional-only counts and assert:

```python
assert result["measured_arm_count"] == 5
assert result["warmup_arm_count"] == 0  # fake CPU context performs no CUDA warm-up
assert result["conditional_dit_count"] == 5
assert result["unconditional_dit_count"] == 0
assert result["actual_model_forward_count"] == result["raw_dit_invocation_count"]
```

- [ ] **Step 7: Update A100 documentation**

In the fast identity section, state that the five/50 limits are measured-arm budgets,
that memory arms additionally perform a truncated semantic prepass, and that
conditional-only removes the unused unconditional DiT. Include the six report field
names from the design and retain the instruction not to advance to S2 after a blocked
content gate.

- [ ] **Step 8: Run report, runner, and self-check tests**

```bash
python -m pytest utest/tests/test_identity_token_probe.py utest/tests/test_identity_token_runner.py utest/tests/test_qstar.py utest/tests/test_inference_hotpath.py -q
python -m utest.identity_token_probe --self-check
python -m py_compile utest/identity_token_probe.py reference_inference_runtime.py infer_slotmem.py
```

Expected: all locally runnable tests pass; Torch-only tests may skip only on a machine
without Torch. Q* source tests continue to show no conditional-only flag.

- [ ] **Step 9: Commit the accounting slice**

```bash
git add utest/identity_token_probe.py utest/tests/test_identity_token_probe.py utest/README.md
git commit -m "Report identity DiT invocation counts"
```

---

### Task 4: Final verification, push, and A100 full-screen handoff

**Files:**
- Verify: all files modified by Tasks 1-3
- Do not modify: frozen prefix, teacher, donor, or prior smoke outputs

**Interfaces:**
- Consumes: Tasks 1-3 complete commits.
- Produces: a pushed branch and exact optimized full S0/S1 server command.

- [ ] **Step 1: Run the full locally available regression**

```bash
python -m pytest utest/tests --ignore=utest/tests/test_content_audit.py -q -rs
python -m utest.identity_token_probe --self-check
python -m py_compile infer_slotmem.py reference_inference_runtime.py utest/identity_token_probe.py utest/qstar_probe.py
git diff --check
```

Expected: no failures; any skips must state the existing missing-Torch reason. Do not
claim the CUDA branch is verified locally.

- [ ] **Step 2: Push and verify the remote commit**

```bash
git push origin fix/qstar-estimand-and-provenance
git fetch origin fix/qstar-estimand-and-provenance
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/fix/qstar-estimand-and-provenance)"
```

- [ ] **Step 3: Verify optimized smoke on A100 before the full grid**

Reuse the valid clean-`b06c7b9` prefix, use a fresh output directory, and run identity
smoke from the new source commit. Expected report invariants:

```text
measured_arm_count = 5
warmup_arm_count = 1
unconditional_dit_count = 0
correct prediction SHA = correct_repeat prediction SHA
content_causality = BLOCK for the already observed middle cell
```

The optimized smoke must reproduce the prior five conditional losses and SHAs exactly;
otherwise stop before the full grid.

- [ ] **Step 4: Run full S0/S1 only after parity**

Use `IDENTITY_SMOKE=0`, `RUN_DECODED_VALIDATION=0`, and a fresh strict-runner
`EVENT_RUN_ROOT`. S2 may execute only if a cell has positive content-specific benefit
above the configured margin and influence floor. A fully blocked grid is a valid null
result and ends the experiment without token labels.
