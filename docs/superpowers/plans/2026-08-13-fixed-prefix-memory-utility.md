# Fixed-Prefix Memory Utility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an immutable-prefix five-arm SlotMem intervention harness plus machine-readable E0, M0a, and M0b remote-stage reports.

**Architecture:** Reuse SlotMem's native resume state to generate each prefix once, hash a JSON contract around it, and run all target arms in fresh processes from that state. Keep tensor intervention semantics, prefix contracts, event orchestration, decoded aggregation, and stage validation in separate modules; the remote Bash runner only wires those modules to the real server paths.

**Tech Stack:** Python 3.10+, PyTorch 2.7.1 on the remote CUDA server, stdlib JSON/hashlib/subprocess/pathlib, pytest, Bash, existing SlotMem inference and evaluator wrappers.

## Global Constraints

- Keep Wan2.2, SlotMem LoRA, Memory Encoder, Memory Writer, and Character-Wise Cross-Attention frozen.
- Use only `no_memory`, `zero`, `correct`, `wrong`, and `random` as confirmatory arms.
- Generate each prefix once and reject any cross-arm snapshot or contract mismatch.
- Define `random` as deterministic per-layer, per-feature-channel moment-matched Gaussian tokens.
- Require a frozen matched-donor manifest; never fall back to an arbitrary donor payload.
- Emit no utility label until the frozen metric vector is complete and M2 establishes content causality.
- Treat the local 6 GB GPU as incapable of M0a; real M0a evidence must come from the remote 14B-capable server.

---

### Task 1: Strict five-arm tensor interventions

**Files:**
- Modify: `utest/content_audit.py`
- Create: `utest/tests/test_content_audit.py`

**Interfaces:**
- Produces: `ARMS = ("no_memory", "zero", "correct", "wrong", "random")`.
- Produces: `transform_tokens(tokens, arm, generator, donor=None) -> torch.Tensor`.
- Produces: `transform_payload(payload, arm, generator, donor_tokens=None) -> tuple[object, int]`.
- Produces: `validate_donor_manifest(entry, event, donor_path) -> dict` and strict per-read JSON reports.

- [ ] **Step 1: Write failing tensor and donor tests**

```python
def test_random_is_deterministic_and_channel_moment_matched():
    tokens = torch.arange(48, dtype=torch.float32).reshape(8, 6)
    a = transform_tokens(tokens, "random", torch.Generator().manual_seed(7))
    b = transform_tokens(tokens, "random", torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    assert torch.allclose(a.mean(0), tokens.mean(0), atol=1e-5)
    assert torch.allclose(a.std(0, correction=0), tokens.std(0, correction=0), atol=1e-5)

def test_wrong_rejects_identity_or_hash_mismatch(tmp_path):
    with pytest.raises(ValueError, match="different entity_uid"):
        validate_donor_manifest(same_entity_entry, event, donor_path)
```

- [ ] **Step 2: Run the tests and confirm the new interfaces fail**

Run: `python -m pytest utest/tests/test_content_audit.py -q`  
Expected: import or assertion failures for the missing random/no-memory and donor validation behavior.

- [ ] **Step 3: Implement the minimum strict arm semantics**

Implement exact per-channel Gaussian moment matching by standardizing sampled noise and rescaling it to the source population mean and standard deviation. Preserve zero-variance channels at their source mean. Return `None` for `no_memory`, pass through for `correct`, zero for `zero`, validate and shape-match for `wrong`, and remove `scramble` from the CLI choices.

Record `attempted_reads`, `non_null_reads`, `payload_layers_seen`, `layers_transformed`, character/bank, slot shape, and before/after tensor hashes. Set `intervention_effective` only from observed arm semantics; never special-case `correct` as automatically effective.

- [ ] **Step 4: Run focused and existing tests**

Run: `python -m pytest utest/tests/test_content_audit.py utest/tests/test_memory_utility.py -q`  
Expected: all tests pass.

- [ ] **Step 5: Commit the arm implementation**

```bash
git add utest/content_audit.py utest/tests/test_content_audit.py
git commit -m "Add strict five-arm memory interventions"
```

### Task 2: Immutable prefix contracts and event orchestration

**Files:**
- Create: `utest/prefix_contract.py`
- Create: `utest/event_harness.py`
- Create: `utest/tests/test_prefix_contract.py`
- Create: `utest/tests/test_event_harness.py`
- Modify: `test_slotmem_stage2.sh`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str`.
- Produces: `build_contract(event: Mapping, snapshot: Path, inference_args: Sequence[str], platform_manifest: Path) -> dict`.
- Produces: `validate_contract(contract: Mapping, snapshot: Path, runtime: Mapping) -> list[str]`.
- Produces CLI subcommands `prepare-prefix`, `run-arms`, and `validate`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_snapshot_mutation_is_rejected(tmp_path):
    snapshot = tmp_path / "prefix.pt"
    snapshot.write_bytes(b"prefix-v1")
    contract = build_contract(event, snapshot, frozen_args, platform_manifest)
    snapshot.write_bytes(b"prefix-v2")
    assert validate_contract(contract, snapshot, runtime) == ["snapshot_sha256_mismatch"]

def test_arm_commands_share_resume_state_and_target_seed():
    commands = build_arm_commands(contract, arms=("correct", "wrong", "zero"))
    assert {value_after(cmd, "--resume_state_path") for cmd in commands} == {str(snapshot)}
    assert {value_after(cmd, "--max_chunks") for cmd in commands} == {str(target_idx + 2)}
```

- [ ] **Step 2: Run contract tests and confirm failure**

Run: `python -m pytest utest/tests/test_prefix_contract.py utest/tests/test_event_harness.py -q`  
Expected: missing-module failures.

- [ ] **Step 3: Implement prefix preparation by reusing native resume state**

`prepare-prefix` invokes native inference with `--max_chunks target_chunk_idx` and one explicit `--resume_state_path`. It validates `next_chunk_idx`, hashes the state, stores absolute input/reference/platform paths, target prompt bytes, seed rule, frozen generation arguments, commit, dirty flag, and platform-manifest hash. It refuses an existing non-matching output directory.

- [ ] **Step 4: Implement arm command construction and group validation**

Each arm starts a new Python process with the identical snapshot and original full story JSON, stops after target+1, and writes to a separate directory. Hash the snapshot before and after every process. Require target-character payload hits for all memory-bearing arms and an attempted absent return for `no_memory`. Write `intervention_contract.json` and `failure_ledger.json`.

Expose the existing inference efficiency paths from `test_slotmem_stage2.sh` through `EFFICIENCY_METRICS_PATH`, `EFFICIENCY_RUNTIME_LOG`, and `RESUME_STATE_PATH` environment variables so the runner does not reconstruct the long SlotMem command.

- [ ] **Step 5: Run orchestration tests**

Run: `python -m pytest utest/tests/test_prefix_contract.py utest/tests/test_event_harness.py -q`  
Expected: all tests pass with fake subprocess runners and synthetic snapshot files.

- [ ] **Step 6: Commit prefix orchestration**

```bash
git add utest/prefix_contract.py utest/event_harness.py utest/tests/test_prefix_contract.py utest/tests/test_event_harness.py test_slotmem_stage2.sh
git commit -m "Run memory arms from one immutable prefix"
```

### Task 3: Decoded measurement completeness and multi-arm reports

**Files:**
- Modify: `utest/memory_utility.py`
- Modify: `utest/tests/test_memory_utility.py`

**Interfaces:**
- Produces: `REQUIRED_OUTCOMES` containing every frozen outcome key.
- Produces: `measurement_completeness(outcomes: Mapping) -> tuple[bool, list[str]]`.
- Extends: `utility_census` with per-arm deltas and `measurement_incomplete` rows.

- [ ] **Step 1: Write failing completeness and five-arm tests**

```python
def test_incomplete_metric_vector_has_no_utility_label():
    report = utility_census(incomplete_records, content_causal=True, **frozen_rules)
    assert report["events"][0]["status"] == "measurement_incomplete"
    assert "label" not in report["events"][0]

def test_all_control_arms_are_reported_against_no_memory():
    report = utility_census(five_arm_records, content_causal=True, **frozen_rules)
    assert set(report["arm_populations"]) == {"correct", "wrong", "zero", "random"}
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `python -m pytest utest/tests/test_memory_utility.py -q`  
Expected: failures for absent completeness status and arm populations.

- [ ] **Step 3: Implement strict measurement gating and arm summaries**

Keep `correct` versus `no_memory` as the primary utility estimand. Report `wrong`, `zero`, and `random` as mechanism controls with the same story-cluster aggregation. Missing required metrics or margins creates a machine-readable incomplete row and never defaults a missing delta to zero.

- [ ] **Step 4: Run all package tests**

Run: `python -m pytest utest/tests -q`  
Expected: all tests pass.

- [ ] **Step 5: Commit decoded aggregation**

```bash
git add utest/memory_utility.py utest/tests/test_memory_utility.py
git commit -m "Gate utility labels on complete decoded outcomes"
```

### Task 4: E0, M0a, and M0b machine-readable stage reports

**Files:**
- Create: `utest/stage_reports.py`
- Create: `utest/tests/test_stage_reports.py`
- Modify: `infer_slotmem.py`

**Interfaces:**
- Produces: `validate_m0a(output_dir: Path, expected_chunks: int = 7) -> dict`.
- Produces: `evaluate_m0b(metric_result: Mapping | None, prerequisites: Mapping[str, bool]) -> dict`.
- Extends native inference metadata with aggregate read/write/runtime/VRAM evidence already collected during generation.

- [ ] **Step 1: Write failing synthetic artifact tests**

```python
def test_m0a_requires_all_seven_chunks_and_nonempty_read(tmp_path):
    make_m0_tree(tmp_path, chunks=2, memory_reads=0)
    report = validate_m0a(tmp_path)
    assert report["status"] == "failed"
    assert set(report["reasons"]) == {"completed_chunks:2/7", "no_nonempty_memory_read"}

def test_m0b_missing_official_inputs_is_non_comparable():
    report = evaluate_m0b(None, {"official_inputs": False, "official_evaluator": True})
    assert report["status"] == "non-comparable"
    assert report["missing"] == ["official_inputs"]
```

- [ ] **Step 2: Run stage-report tests and confirm failure**

Run: `python -m pytest utest/tests/test_stage_reports.py -q`  
Expected: missing-module failure.

- [ ] **Step 3: Persist native inference evidence**

Reuse `efficiency_chunk_records`, role states, sparse-memory statistics, and writer
statistics already present in `infer_slotmem.py`. Add aggregate non-empty-read evidence,
total/per-chunk timing, maximum allocated/reserved VRAM, loaded checkpoint domains, and
writer update evidence to the final metadata/efficiency JSON; do not add a second probe
path.

- [ ] **Step 4: Implement M0 validation and M0b comparability**

M0a passes only with seven videos, seven chunk metadata files, a completed manifest,
non-empty post-first-appearance reads, loaded Stage-2 checkpoints, wall time, and VRAM
fields. M0b compares Subject Consistency to `0.8771`, accepts absolute error at most
`0.02` or a bootstrap interval covering the anchor, and otherwise reports failure.
Missing official inputs, preprocessing, checkpoint equivalence, or evaluator produces
`non-comparable` with exact missing keys.

- [ ] **Step 5: Run stage and full unit tests**

Run: `python -m pytest utest/tests/test_stage_reports.py utest/tests -q`  
Expected: all tests pass.

- [ ] **Step 6: Commit stage reporting**

```bash
git add infer_slotmem.py utest/stage_reports.py utest/tests/test_stage_reports.py
git commit -m "Validate SlotMem E0 and M0 stage evidence"
```

### Task 5: Remote server runner and operator documentation

**Files:**
- Create: `scripts/run_slotmem_stage_gates.sh`
- Modify: `utest/README.md`
- Modify: `docs/experiment-protocol.md`

**Interfaces:**
- Consumes explicit `NARRASTREAM_INPUT_ROOT`, `WAN22_DIR`, `CKPT_ROOT`, `RUN_ROOT`, and `UTEST_ENV`.
- Produces `e0.json`, `platform.manifest.json`, `m0a_report.json`, `m0b_report.json`, logs, inference metadata, efficiency JSON, and remote test commands.

- [ ] **Step 1: Implement strict environment and path checks**

The Bash script uses `set -euo pipefail`, resolves every required path, refuses the local
6 GB GPU unless `ALLOW_UNDERSIZED_GPU=1` is explicitly set for a non-M0 smoke test, and
creates a fresh timestamped run directory. It runs `utest.eligibility` before GPU work
and stops controller-label work when E0 has fewer than 128 eligible stories.

- [ ] **Step 2: Run full seven-chunk M0a and validate it**

Invoke `test_slotmem_stage2.sh` with the official sample, `MAX_CHUNKS=7`, efficiency
paths, and the frozen platform manifest. Capture stdout/stderr and wall time, then call
`python -m utest.stage_reports m0a` to write the final report.

- [ ] **Step 3: Run or classify M0b**

When `M0B_INPUT_ROOT` and `NARRASTREAM_BENCH_REPO` are supplied, invoke the existing
benchmark wrapper and feed its Subject Consistency JSON to `utest.stage_reports m0b`.
Otherwise call the same command with explicit false prerequisite flags so the report is
`non-comparable`, not absent.

- [ ] **Step 4: Document exact quick and full remote commands**

Document one-event five-arm commands, required donor manifest schema, the full stage
runner invocation, output locations, expected statuses, resume behavior, and the rule
that logs alone are not completion evidence.

- [ ] **Step 5: Run shell and Python verification**

Run: `bash -n scripts/run_slotmem_stage_gates.sh`  
Run: `python -m compileall -q utest`  
Run: `python -m pytest utest/tests -q`  
Expected: shell syntax succeeds, compilation succeeds, and all tests pass.

- [ ] **Step 6: Commit the remote runner**

```bash
git add scripts/run_slotmem_stage_gates.sh utest/README.md docs/experiment-protocol.md
git commit -m "Add remote SlotMem stage gate runner"
```

### Task 6: Fresh completion audit

**Files:**
- Verify only; modify the smallest owning file if a check fails.

**Interfaces:**
- Produces the final evidence summary and exact remote command the user can run.

- [ ] **Step 1: Run the complete local verification suite**

Run: `python -m pytest utest/tests -q`  
Run: `python -m compileall -q utest`  
Run: `bash -n scripts/run_slotmem_stage_gates.sh`  
Run: `git diff --check`  
Expected: zero test failures, no compilation or shell errors, and no whitespace errors.

- [ ] **Step 2: Audit requirements against the approved design**

Verify five arms, immutable snapshot hashes, strict donor matching, random moment
matching, read/address/writer evidence, metric completeness, real E0 command, seven-chunk
M0a validation, and explicit M0b non-comparability behavior in code and tests.

- [ ] **Step 3: Report local versus remote evidence separately**

State the exact local test count. Do not claim E0 or M0 passed until the remote JSON
reports exist and validate. Provide the one-event quick command and full-stage command,
then list every expected artifact path.
