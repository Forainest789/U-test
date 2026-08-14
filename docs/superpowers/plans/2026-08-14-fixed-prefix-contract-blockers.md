# Fixed-Prefix Contract Blockers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan.

**Goal:** Make the one-event fixed-prefix experiment causally valid by fixing state paths, validating actual arm runtimes, restricting intervention scope, and making the `wrong` and `random` controls strict and reproducible.

**Architecture:** Keep the existing inference path and the single `RoleWiseSlotMemoryBank.get_memory_payload_for_read` boundary. Prefix generation writes one immutable native state. Each arm loads that state, writes continuation state only in its own directory, emits an actual runtime contract, and changes only the target entity's read at the target chunk.

**Tech Stack:** Python 3.10, PyTorch, pytest, argparse, existing Bash runner.

## Global Constraints

- Do not modify the NVIDIA driver or create/change a CUDA environment.
- Preserve existing uncommitted user changes.
- Do not start a GPU run or the 12-story experiment in this implementation.
- Add no dependency and no second snapshot format.
- Keep the one-event pilot outside the later confirmatory 12 stories.

---

### Task 1: Fix immutable prefix state paths

**Files:**
- Modify: `utest/event_harness.py`
- Modify: `utest/prefix_contract.py`
- Test: `utest/tests/test_event_harness.py`
- Test: `utest/tests/test_prefix_contract.py`

**Step 1: Write failing path-contract tests**

Extend the arm command test:

```python
for run_name, command in commands.items():
    assert command[command.index("--resume_state_path") + 1] == str(snapshot)
    assert command[command.index("--save_state_path") + 1] == str(
        (tmp_path / "arms" / run_name / "resume_state.pt").resolve()
    )
```

Add a pure prefix-argument test:

```python
args = build_prefix_inference_args(event, tmp_path, [
    "--resume_state_path", "stale.pt", "--output_path", "old",
])
assert "--resume_state_path" not in args
assert args[args.index("--save_state_path") + 1] == str(tmp_path / "prefix_state.pt")
```

Also assert `save_state_path` and `resume_state_path` are excluded from frozen args.

**Step 2: Confirm RED**

```powershell
python -m pytest utest/tests/test_event_harness.py utest/tests/test_prefix_contract.py -q
```

**Step 3: Implement the shared path fix**

Let `_set_option` remove an option when `value is None`, then centralize prefix args:

```python
def build_prefix_inference_args(event: Mapping, output: Path, argv: Sequence[str]) -> list[str]:
    result = _set_option(argv, "--resume_state_path", None)
    result = _set_option(result, "--save_state_path", str(output / "prefix_state.pt"))
    result = _set_option(result, "--max_chunks", str(int(event["target_chunk_idx"])))
    result = _set_option(result, "--output_path", str(output / "prefix_generation"))
    return result
```

Keep the existing JSON/reference/efficiency options in that helper. In every arm command add:

```python
inference_args = _set_option(
    inference_args, "--save_state_path", str(arm_dir / "resume_state.pt")
)
```

Add `save_state_path` to `RUNTIME_ONLY_ARGS`.

**Step 4: Confirm GREEN and commit**

```powershell
python -m pytest utest/tests/test_event_harness.py utest/tests/test_prefix_contract.py -q
git add utest/event_harness.py utest/prefix_contract.py utest/tests/test_event_harness.py utest/tests/test_prefix_contract.py
git commit -m "fix: separate prefix save and arm resume paths"
```

---

### Task 2: Validate actual per-arm runtime contracts

**Files:**
- Modify: `utest/prefix_contract.py`
- Modify: `utest/content_audit.py`
- Modify: `utest/event_harness.py`
- Test: `utest/tests/test_prefix_contract.py`
- Test: `utest/tests/test_event_harness.py`

**Step 1: Write failing tests**

Test a reusable runtime builder:

```python
runtime = build_runtime_contract(event, args)
assert runtime["source_json_sha256"] == sha256_file(Path(event["source_json_path"]))
assert runtime["target_seed"] == 43
```

Build a synthetic event run where `random/audit.json` has a changed actual runtime and assert:

```python
report = validate_event_run(event_run)
assert "random:frozen_args_mismatch" in report["errors"]
```

**Step 2: Confirm RED**

```powershell
python -m pytest utest/tests/test_prefix_contract.py utest/tests/test_event_harness.py -q
```

**Step 3: Extract the runtime builder**

In `utest/prefix_contract.py`, reuse the current parsing and resolved-input logic:

```python
def build_runtime_contract(event: Mapping, inference_args: Sequence[str]) -> dict:
    parsed = _arguments(inference_args)
    source = Path(event.get("source_json_path") or parsed.get("json_path", "")).resolve()
    story = json.loads(source.read_text(encoding="utf-8"))
    chunks = story.get("chunks", story) if isinstance(story, dict) else story
    target_idx = int(event["target_chunk_idx"])
    prompt = str(chunks[target_idx].get("content") or chunks[target_idx].get("caption") or "")
    reference_text = event.get("reference_path") or parsed.get("ref_image_path", "")
    reference = Path(reference_text).resolve() if reference_text else None
    return {
        "frozen_args": normalized_frozen_args(inference_args),
        "source_json_sha256": sha256_file(source),
        "target_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "reference_sha256": sha256_file(reference) if reference and reference.is_file() else None,
        "target_seed": int(parsed.get("target_seed_override", int(parsed.get("seed_base", 42)) + target_idx)),
    }
```

Make `build_contract` call this helper.

**Step 4: Emit and validate actual artifacts**

Normalize `content_audit.main()` passthrough args before `install`, compute the runtime contract, and include it in `audit.json`:

```python
rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
actual_runtime = build_runtime_contract(event, rest)
flush = install(
    args.arm,
    args.seed,
    args.donor,
    args.dump_donor,
    args.report,
    event=event,
    donor_entry=donor_entry,
    runtime_contract=actual_runtime,
)
```

In `validate_event_run`, validate the six reports instead of comparing the expected object to itself:

```python
errors = validate_contract(contract, snapshot)
for run_name in (*ARMS, "correct_repeat"):
    report = reports.get(run_name)
    if report is None:
        errors.append(f"{run_name}:missing_report")
        continue
    errors.extend(
        f"{run_name}:{error}"
        for error in validate_contract(contract, snapshot, report.get("runtime_contract"))
    )
```

Keep scientific arm checks scoped to `ARMS`.

**Step 5: Confirm GREEN and commit**

```powershell
python -m pytest utest/tests/test_prefix_contract.py utest/tests/test_event_harness.py -q
git add utest/prefix_contract.py utest/content_audit.py utest/event_harness.py utest/tests/test_prefix_contract.py utest/tests/test_event_harness.py
git commit -m "fix: validate actual arm runtime contracts"
```

---

### Task 3: Restrict interventions to the frozen target read

**Files:**
- Modify: `infer_slotmem.py`
- Modify: `utest/content_audit.py`
- Modify: `utest/event_harness.py`
- Test: `utest/tests/test_content_audit.py`
- Test: `utest/tests/test_event_harness.py`

**Step 1: Write failing scope tests**

```python
event = {"character_name": "ana", "target_chunk_idx": 4}
assert intervention_applies(event, "ana", 4)
assert not intervention_applies(event, "bob", 4)
assert not intervention_applies(event, "ana", 5)
assert not intervention_applies(event, "ana", None)
```

Update the `no_memory` audit test so global native reads are allowed while the target return is absent:

```python
reports["no_memory"].update(
    returned_non_null_reads=3,
    target_source_non_null_reads=1,
    target_returned_non_null_reads=0,
)
assert validate_audit_group(reports) == []
```

**Step 2: Confirm RED**

```powershell
python -m pytest utest/tests/test_content_audit.py utest/tests/test_event_harness.py -q
```

**Step 3: Expose current chunk and scope the wrapper**

At the top of the existing inference loop:

```python
for chunk_idx, chunk in enumerate(chunks_to_iterate, start=start_chunk_idx):
    mem_manager.current_chunk_idx = int(chunk_idx)
```

Add the predicate:

```python
def intervention_applies(event: Mapping, character: object, chunk_idx: object) -> bool:
    try:
        return (
            str(character) == str(event["character_name"])
            and int(chunk_idx) == int(event["target_chunk_idx"])
        )
    except (KeyError, TypeError, ValueError):
        return False
```

Inside the patched reader, transform only when the predicate is true. Record `chunk_idx`, target-scoped source/return counts, exact source/returned payload hashes, and transformed count. Require `--event-json` for every non-self-check arm invocation.

`no_memory` effectiveness must use `target_source_non_null_reads > 0` and `target_returned_non_null_reads == 0`; global read counts remain diagnostics.

**Step 4: Confirm GREEN and commit**

```powershell
python -m pytest utest/tests/test_content_audit.py utest/tests/test_event_harness.py -q
git add infer_slotmem.py utest/content_audit.py utest/event_harness.py utest/tests/test_content_audit.py utest/tests/test_event_harness.py
git commit -m "fix: scope memory arms to the target read"
```

---

### Task 4: Make `wrong` exact-shape and `random` order-independent

**Files:**
- Modify: `utest/content_audit.py`
- Test: `utest/tests/test_content_audit.py`

**Step 1: Write failing strict-control tests**

```python
for donor in (torch.ones(5, 4), torch.ones(7, 4)):
    with pytest.raises(ValueError, match="exact shape"):
        transform_tokens(tokens, "wrong", torch.Generator().manual_seed(0), donor)
```

Require `payload_key` in the donor manifest. Add an order-independence test that transforms the same layers in forward and reverse dictionary order and compares output by layer key.

**Step 2: Confirm RED**

```powershell
python -m pytest utest/tests/test_content_audit.py -q
```

**Step 3: Enforce exact donor shape**

```python
def _match_rows(donor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if tuple(donor.shape) != tuple(target.shape):
        raise ValueError(
            f"wrong donor must have exact shape {tuple(target.shape)}, got {tuple(donor.shape)}"
        )
    return donor
```

Add `payload_key` to required manifest fields and remove fallback to the only payload in a donor file.

**Step 4: Seed each layer from canonical identifiers**

```python
def stable_transform_seed(event, target_idx, character, bank_idx, arm_seed, layer) -> int:
    canonical = json.dumps(
        [event.get("story_id"), event.get("event_id"), int(target_idx),
         str(character), int(bank_idx), int(arm_seed), str(layer)],
        ensure_ascii=False, separators=(",", ":"),
    )
    return int.from_bytes(hashlib.sha256(canonical.encode("utf-8")).digest()[-8:], "big")
```

Let `transform_payload` accept an optional `generator_for_layer` callable and use a fresh generator for `shared` or the stable layer key. Retain the current generator argument as a fallback for direct callers.

Keep moment computation in CPU float32 and test mean/std with `atol=rtol=1e-5` before restoring source dtype.

**Step 5: Update self-check, confirm GREEN, and commit**

```powershell
python -m pytest utest/tests/test_content_audit.py -q
python -m utest.content_audit --self-check
git add utest/content_audit.py utest/tests/test_content_audit.py
git commit -m "fix: make memory controls strict and reproducible"
```

---

### Task 5: Support target-only seed overrides from one prefix

**Files:**
- Modify: `infer_slotmem.py`
- Modify: `utest/event_harness.py`
- Modify: `utest/prefix_contract.py`
- Test: `utest/tests/test_event_harness.py`
- Test: `utest/tests/test_prefix_contract.py`

**Step 1: Write failing tests**

Build commands explicitly and assert each has the override:

```python
commands = build_arm_commands(
    contract,
    output_root=tmp_path / "arms",
    event_json=tmp_path / "event.json",
    arms=("correct", "random"),
    target_seed_override=271,
)
for command in commands.values():
    assert command[command.index("--target_seed_override") + 1] == "271"
```

Assert the option is runtime-only and changes the actual contract's `target_seed`.

**Step 2: Confirm RED**

```powershell
python -m pytest utest/tests/test_event_harness.py utest/tests/test_prefix_contract.py -q
```

**Step 3: Add the target-only override**

Add the parser option to `infer_slotmem.py`, then replace the current seed assignment:

```python
override = getattr(args, "target_seed_override", None)
chunk_seed = (
    int(override)
    if override is not None and chunk_idx == start_chunk_idx
    else int(getattr(args, "seed_base", 42)) + chunk_idx
)
```

This changes only the first resumed target chunk; target+1 returns to the native seed rule.

Add `--target-seed-override` to `run-arms`. Write a run-local contract copy with the selected expected target seed; never mutate the prefix directory's original contract. Add `target_seed_override` to `RUNTIME_ONLY_ARGS`.

**Step 4: Confirm GREEN and commit**

```powershell
python -m pytest utest/tests/test_event_harness.py utest/tests/test_prefix_contract.py -q
git add infer_slotmem.py utest/event_harness.py utest/prefix_contract.py utest/tests/test_event_harness.py utest/tests/test_prefix_contract.py
git commit -m "feat: vary target seeds from one frozen prefix"
```

---

### Task 6: Align E0 gating with the approved order

**Files:**
- Modify: `scripts/run_slotmem_stage_gates.sh`

**Step 1: Remove the contradictory early exit**

Replace the current E0 exit branch with:

```bash
if [[ "${E0_PASSES}" != "1" ]]; then
  echo "[stage] WARNING: E0 blocked; continuing M0 as a platform-only diagnostic" >&2
fi
```

This permits M0 diagnostics only; it does not authorize labels or the 12-story run.

**Step 2: Validate and commit**

```powershell
bash -n scripts/run_slotmem_stage_gates.sh
git add scripts/run_slotmem_stage_gates.sh
git commit -m "fix: keep M0 diagnostics available when E0 blocks labels"
```

---

### Task 7: Verify code and prepare the one-event handoff

**Files:** Modify only when verification exposes a defect in files already listed.

**Step 1: Run focused and full zero-GPU checks**

```powershell
python -m pytest utest/tests/test_content_audit.py utest/tests/test_event_harness.py utest/tests/test_prefix_contract.py -q
python -m pytest utest/tests -q
```

If local Python still exposes the known incomplete `torch` namespace, record it as an environment failure; do not weaken or skip logic tests in code.

**Step 2: Run syntax and self-checks**

```powershell
python -m py_compile infer_slotmem.py utest/content_audit.py utest/event_harness.py utest/prefix_contract.py
python -m utest.content_audit --self-check
bash -n scripts/run_slotmem_stage_gates.sh
```

**Step 3: Review the diff**

```powershell
git diff --check
git status --short
git diff -- infer_slotmem.py utest/content_audit.py utest/event_harness.py utest/prefix_contract.py scripts/run_slotmem_stage_gates.sh utest/tests/test_content_audit.py utest/tests/test_event_harness.py utest/tests/test_prefix_contract.py
```

Confirm no driver, CUDA environment, dataset, checkpoint, generated run, or unrelated file changed.

**Step 4: Prepare but do not launch the remote pilot**

Identify one real eligible recurrence event outside the future 12-story set and produce the exact fresh-output commands. The pilot must prove actual runtime matches, target-only read hashes, exact donor shapes, decoded divergence above `correct_repeat`, writer evidence, and immutable prefix SHA256.

---

## Deferred Until the One-Event Pilot Passes

- Isolated cu118 environment qualification (owned by the user for now).
- Real prefix/arm GPU execution.
- Frozen 12-story and donor-pair manifests.
- Seeded cross-GPU story scheduler and nuisance/block recording.
- Metric-card scoring and any utility/controller claim.
