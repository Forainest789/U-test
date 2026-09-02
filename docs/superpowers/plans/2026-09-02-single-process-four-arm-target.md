# Single-Process Four-Arm Target Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate `full_correct`, `no_memory`, `zero_path`, and `wrong_subject` target chunks in one process with one loaded SlotMem engine.

**Architecture:** Add a focused U-test runner that reconstructs frozen arm payloads through the existing audited reader seam, then reuses one `SlotMemInferenceEngine` in a sequential `generate_chunk()` loop. Add one opt-in `target-preflight` harness command; leave the existing subprocess preflight and all SlotMem base/runtime files unchanged.

**Tech Stack:** Python 3.10+, PyTorch, existing SlotMem inference APIs, stdlib JSON/hashlib/time/pathlib, pytest.

## Global Constraints

- Do not modify `infer_slotmem.py`, `reference_inference_runtime.py`, or model code.
- Generate only the target chunk; never generate target+1 in this mode.
- Use exactly `full_correct`, `no_memory`, `zero_path`, and `wrong_subject`, in that order.
- Construct one `SlotMemInferenceEngine` per command invocation and reuse it sequentially.
- Require `offload_models=false`; keep the active dual-expert policy unchanged.
- Reload arm payload state from the immutable prefix and disable online memory collection.
- Preserve existing multi-process preflight behavior.

---

### Task 1: Build audited target-arm payload bundles

**Files:**
- Create: `utest/subject_reappearance_target_runner.py`
- Test: `utest/tests/test_subject_reappearance_target_runner.py`

**Interfaces:**
- Consumes: frozen prefix state, target chunk, event, subject mask, donor entry/artifact, and parsed inference arguments.
- Produces: `build_target_arm_bundles(...) -> dict[str, dict]` and an audit finalizer per arm.

- [ ] **Step 1: Write failing payload tests**

Create CPU fixtures with one target and one non-target layerwise memory payload. Assert that the helper:

```python
bundles = build_target_arm_bundles(
    infer_slotmem=fake_runtime,
    state=state,
    target_chunk=target_chunk,
    event=event,
    mask_manifest=mask,
    runtime_contract=runtime_contract,
    event_file_sha256="a" * 64,
    manifest_file_sha256="b" * 64,
    report_root=tmp_path,
    donor=donor,
)
assert tuple(bundles) == PREFLIGHT_ARMS
assert bundles["no_memory"].memory_bank_tokens is None
assert torch.count_nonzero(target_layer(bundles["zero_path"])) == 0
assert torch.equal(non_target_layer(bundles["full_correct"]), non_target_layer(bundles["wrong_subject"]))
```

Also assert every fresh manager reads at `event["target_chunk_idx"]`, and every audit finalizer restores the reader patch.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest -q utest/tests/test_subject_reappearance_target_runner.py
```

Expected: collection fails because `subject_reappearance_target_runner` does not exist.

- [ ] **Step 3: Implement the smallest payload builder**

In the new module:

- validate the mask and donor with existing `subject_subspace_audit` helpers;
- create a fresh `RoleWiseSlotMemoryBank` from the immutable CPU state for every arm;
- install `install_subject_subspace(...)` for that arm;
- read all characters and banks required by the target chunk;
- aggregate existing layerwise tokens and metadata into the arguments expected by `generate_chunk()`;
- return one dataclass containing the generation kwargs and `flush_audit()` callback;
- on any exception, flush/restore the installed patch before re-raising.

Do not reimplement `transform_slot_rows`; the patched reader remains authoritative.

- [ ] **Step 4: Run payload tests and existing audit tests**

Run:

```bash
python -m pytest -q utest/tests/test_subject_reappearance_target_runner.py utest/tests/test_subject_subspace.py utest/tests/test_content_audit.py
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add utest/subject_reappearance_target_runner.py utest/tests/test_subject_reappearance_target_runner.py
git commit -m "feat: build audited target-only arm payloads"
```

---

### Task 2: Reuse one engine for four target generations

**Files:**
- Modify: `utest/subject_reappearance_target_runner.py`
- Test: `utest/tests/test_subject_reappearance_target_runner.py`

**Interfaces:**
- Consumes: Task 1 arm bundles and a validated target-preflight context.
- Produces: `run_target_preflight(context, *, engine_factory=None, save_video_fn=None) -> dict`.

- [ ] **Step 1: Write failing single-engine loop tests**

Use a fake engine and saver. Assert:

```python
report = run_target_preflight(context, engine_factory=factory, save_video_fn=save)
assert factory.calls == 1
assert [call.arm for call in engine.calls] == list(PREFLIGHT_ARMS)
assert all(call.seed == 0 for call in engine.calls)
assert all(call.online_memory_chars == [] for call in engine.calls)
assert report["execution_mode"] == "single_process_target_only"
assert report["engine_initialization_count"] == 1
```

Assert only `chunk_<target>.mp4` is saved, per-arm diagnostics are reset, prefix SHA is checked after every arm, and an exception from arm two prevents arms three and four.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m pytest -q utest/tests/test_subject_reappearance_target_runner.py
```

Expected: failures for the missing execution loop.

- [ ] **Step 3: Implement the sequential runner**

Parse the frozen base argv after forcing only harness-owned values:

```python
argv = _set_option(argv, "--offload_models", None)
argv = _set_option(argv, "--no-offload_models", None)
argv.append("--no-offload_models")
runtime_args = infer_slotmem.parse_args(argv)
engine = engine_factory(runtime_args)
```

Load prompt and reference frames once. For each frozen arm:

- reset documented diagnostic accumulators;
- generate with identical prompt, seed, references, and arm-specific payload;
- pass empty online-memory collections;
- save the video to the existing arm directory;
- write compatible target metadata and efficiency JSON from the measured result;
- flush the arm audit in `finally`;
- verify the prefix SHA before advancing.

Write the phase report only after all four arms complete. Use exclusive writes and never overwrite an existing artifact.

- [ ] **Step 4: Run the focused tests**

```bash
python -m pytest -q utest/tests/test_subject_reappearance_target_runner.py
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add utest/subject_reappearance_target_runner.py utest/tests/test_subject_reappearance_target_runner.py
git commit -m "feat: reuse one engine across target preflight arms"
```

---

### Task 3: Expose the opt-in harness command and preserve validation

**Files:**
- Modify: `utest/subject_reappearance_harness.py`
- Modify: `utest/tests/test_subject_reappearance_harness.py`
- Modify: `utest/README.md`

**Interfaces:**
- Consumes: `run_target_preflight(...)` from Task 2.
- Produces: CLI command `python -m utest.subject_reappearance_harness target-preflight ...`.

- [ ] **Step 1: Write failing command-routing tests**

Assert that `target-preflight`:

- accepts `--manifest`, `--event-id`, `--seed`, and `--resume`;
- validates prefix, qualification, mask, donor, and snapshot before loading the engine;
- delegates once to the target runner;
- rejects multiple selected blocks because one engine may only serve one frozen event/seed context;
- leaves the existing `preflight` command artifact and subprocess path unchanged.

- [ ] **Step 2: Run routing tests and verify RED**

```bash
python -m pytest -q utest/tests/test_subject_reappearance_harness.py -k "target_preflight or arm_orders"
```

Expected: failures because the CLI and dispatch path do not exist.

- [ ] **Step 3: Implement routing and validation integration**

Add a dedicated target-preflight branch in `_execute_stage`. Build the context from the
already validated row and prefix contract, call the new runner, then call `validate_block`
against the four arm artifacts. Extend the returned validation object with:

```python
{
    "execution_mode": "single_process_target_only",
    "engine_initialization_count": 1,
    "target_chunk_idx": target_idx,
    "target_plus_one_generated": False,
}
```

When `--resume` is present, skip only individually valid completed arm directories; the
new invocation still initializes exactly one engine for the remaining arms.

- [ ] **Step 4: Document the exact Song command**

Replace only the Song pilot preflight invocation with:

```bash
CUDA_VISIBLE_DEVICES=0 \
DUAL_EXPERT_LOAD_MODE=active \
DUAL_EXPERT_MANAGE_AUX_MODELS=1 \
python -m utest.subject_reappearance_harness target-preflight \
  --manifest "$SONG_ROOT/target_run/run_manifest.json" \
  --event-id "$SONG_EVENT" \
  --seed 0
```

Document that it produces target only, loads one engine, and retains the old `preflight`
command for the two-chunk protocol.

- [ ] **Step 5: Run focused and full U-test suites**

```bash
python -m pytest -q utest/tests/test_subject_reappearance_target_runner.py utest/tests/test_subject_reappearance_harness.py
python -m pytest -q utest/tests
```

Expected: PASS, with CUDA-dependent tests skipped when CUDA is unavailable.

- [ ] **Step 6: Verify the protected base files are unchanged**

```bash
git diff 42ae148 -- infer_slotmem.py reference_inference_runtime.py diffsynth/
```

Expected: no output.

- [ ] **Step 7: Commit Task 3**

```bash
git add utest/subject_reappearance_harness.py utest/tests/test_subject_reappearance_harness.py utest/README.md
git commit -m "feat: add single-process target preflight command"
```
