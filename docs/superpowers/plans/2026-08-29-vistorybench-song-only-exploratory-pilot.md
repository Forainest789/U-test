# ViStoryBench Song-Only Exploratory Pilot Implementation Plan

> **For implementation agents:** REQUIRED SUB-SKILL: use subagent-driven development task by task. Each implementation task is assigned to a fresh subagent, then reviewed by a different fresh subagent before root acceptance.

**Goal:** Reuse the current frozen ViStoryBench targets, complete 26-candidate donor survey, reviewed Song Yuchen donor decision, SlotMem top-8/64 runtime, donor pipeline, and subject-reappearance harness to generate exactly four Song Yuchen seed-0 exploratory videos.

**Architecture:** Add one explicit `exploratory_single_event` scope to the existing donor selection contract and make the donor harness and donor bundle derive their expected event IDs from that validated contract. Keep the formal exact-three default byte-compatible. Keep the target harness unchanged except for regressions: its existing partial donor-map behavior and `--event-id`/`--seed` selectors already provide the required one-block execution path.

**Tech Stack:** Python 3.10+, standard library, PyTorch already installed in the `slotmem` environment, pytest, existing SlotMem/Wan2.2 checkpoints, existing ViStoryBench snapshot.

## Frozen inputs and stop conditions

- Formal target authority remains `utest/events/vistorybench_reappearance_v1.json` with Song Yuchen, Bella, and Chen Sihan's Father.
- Exploratory target is exactly `vistory79_song_yuchen_s2_s8`; target seed and donor seed are both `0`.
- The complete donor survey remains the authority and must hash to `8d70b19a1b8c4c293495e5e2b853d54e4431ccd8bc9aa38c9f50d5c914baad57`.
- The only approved donor is Colonel Cromarty, candidate `ad04290d1a7ddd4691b8337c3a71afca0e8daee38a706be8c5a40f83bb725938`.
- The four decoded arms remain exactly `full_correct`, `no_memory`, `zero_path`, and `wrong_subject`.
- Geometry remains layers `0-15`, 64 slots, with the existing semantic top-8 mask protocol.
- Q* remains `not_available` and is not run, scored, or used to make a decision.
- Stop after the four preflight videos and their validation record. Do not run the eight-arm `full` stage and do not aggregate this pilot as a primary result.
- Reuse the active `slotmem` Conda environment. Do not add an environment, dependency, `exit`, or `SystemExit` path.

## Reused components

- `utest/vistory_donors.py`: survey validation, strict human-review schema, deterministic donor materialization, hashes, and no-clobber publication.
- `utest/vistory_donor_harness.py`: frozen prefix/dump commands, top-8/64 validation, completion records, and resume semantics.
- `utest/vistory_donor_bundle.py`: donor payload validation and event-level donor-map publication.
- `utest/subject_reappearance_harness.py`: full nine-block manifest, partial donor-map loading, source probe, four-arm preflight, and event/seed selection.
- `runs/_gate_b_review_ce2aed53/survey.json`: locally audited copy of the complete Gate-B survey; the server may deterministically regenerate the same file from its frozen inputs.
- Existing server paths under `runs/vistorybench_reappearance_v1`, `platform.manifest.json`, and `runs/m0a_slotmem_stage2/inference_args.yaml`.

---

### Task 1: Freeze the reviewed Song-only selection without weakening formal mode

**Files:**

- Modify: `utest/vistory_donors.py`
- Modify: `tools/prepare_vistory_donors.py`
- Create: `utest/events/vistorybench_song_yuchen_donor_review_v1.json`
- Test: `utest/tests/test_vistory_donors.py`

**Interfaces:**

- Add `EXPLORATORY_SINGLE_EVENT_SCOPE = "exploratory_single_event"`.
- Add `donor_selection_event_ids(selection: Mapping) -> frozenset[str]` as the single shared scope validator.
- Extend `freeze_donor_selection(*, data_root: Path, target_inputs_path: Path, survey_path: Path, review_path: Path, output_root: Path, exploratory_target_event_id: str | None = None)`.
- Add `prepare_vistory_donors.py freeze --exploratory-target-event-id EVENT_ID`.
- Formal selection JSON remains unchanged when the option is omitted.
- Exploratory selection adds only:

```json
{
  "protocol_scope": "exploratory_single_event",
  "target_event_ids": ["vistory79_song_yuchen_s2_s8"]
}
```

- [ ] **Step 1: Add the frozen three-row human-review artifact**

Create `utest/events/vistorybench_song_yuchen_donor_review_v1.json` with this exact payload:

```json
{
  "schema_version": 1,
  "dataset_commit": "92f845531b67e97a67ae04b256ec5d8c020e8341",
  "survey_sha256": "8d70b19a1b8c4c293495e5e2b853d54e4431ccd8bc9aa38c9f50d5c914baad57",
  "reviews": [
    {
      "target_event_id": "vistory79_song_yuchen_s2_s8",
      "candidate_id": "2e08901266442994503aa6c94b30cb8c75617d70266647a0a70425cc6dcfbc55",
      "target_presentation_class": "male",
      "donor_presentation_class": "male",
      "target_dominant_colour": "black",
      "donor_dominant_colour": "brown",
      "donor_source_visible": true,
      "donor_read_check_visible": true,
      "approved": false,
      "tie_group": null,
      "reviewer": "wangshixiao"
    },
    {
      "target_event_id": "vistory79_song_yuchen_s2_s8",
      "candidate_id": "6cde40c6ee47af0ec9fb2b0596a0088316c4510eea0f8a203dd40544e220fa13",
      "target_presentation_class": "male",
      "donor_presentation_class": "male",
      "target_dominant_colour": "black",
      "donor_dominant_colour": "blue-grey",
      "donor_source_visible": true,
      "donor_read_check_visible": true,
      "approved": false,
      "tie_group": null,
      "reviewer": "wangshixiao"
    },
    {
      "target_event_id": "vistory79_song_yuchen_s2_s8",
      "candidate_id": "ad04290d1a7ddd4691b8337c3a71afca0e8daee38a706be8c5a40f83bb725938",
      "target_presentation_class": "male",
      "donor_presentation_class": "male",
      "target_dominant_colour": "black",
      "donor_dominant_colour": "black",
      "donor_source_visible": true,
      "donor_read_check_visible": true,
      "approved": true,
      "tie_group": null,
      "reviewer": "wangshixiao"
    }
  ]
}
```

- [ ] **Step 2: Write failing scope and review-coverage tests**

Add focused tests using the existing `_three_target_fixture`, `_write_review_for_survey`, and JSON helpers:

```python
SONG_EVENT_ID = "vistory79_song_yuchen_s2_s8"


def test_exploratory_freeze_materializes_only_the_explicit_frozen_event(
    tmp_path: Path,
) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    review = _write_review_for_survey(review_path, survey_path, survey)
    review["reviews"] = [
        row for row in review["reviews"] if row["target_event_id"] == SONG_EVENT_ID
    ]
    review_path.write_text(json.dumps(review), encoding="utf-8")

    selection = freeze_donor_selection(
        data_root=data_root,
        target_inputs_path=targets,
        survey_path=survey_path,
        review_path=review_path,
        output_root=tmp_path / "selection",
        exploratory_target_event_id=SONG_EVENT_ID,
    )

    assert selection["protocol_scope"] == EXPLORATORY_SINGLE_EVENT_SCOPE
    assert selection["target_event_ids"] == [SONG_EVENT_ID]
    assert [row["target_event_id"] for row in selection["events"]] == [SONG_EVENT_ID]
    assert {row["target_event_id"] for row in selection["candidate_audit"]} == {SONG_EVENT_ID}


def test_exploratory_freeze_rejects_review_rows_outside_declared_event(
    tmp_path: Path,
) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    _write_review_for_survey(review_path, survey_path, survey)
    with pytest.raises(ValueError, match="outside exploratory target"):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=tmp_path / "selection",
            exploratory_target_event_id=SONG_EVENT_ID,
        )
```

Also test an unknown event ID, missing scoped review row, duplicate review row, and two approved scoped rows. Preserve `test_freeze_materializes_exactly_three_seed_zero_donor_events` as the formal-mode compatibility check. Add one static assertion that the checked-in Song review contains exactly the three frozen candidate IDs and approves only Colonel Cromarty.

- [ ] **Step 3: Run the tests and confirm the new mode is absent**

Run:

```bash
python -m pytest -q utest/tests/test_vistory_donors.py -k "exploratory or materializes_exactly_three or song_yuchen_donor_review"
```

Expected: the exploratory tests fail because the argument, constant, and scope validator do not exist; existing formal test passes.

- [ ] **Step 4: Implement the minimum shared scope contract**

Use one validator in `utest/vistory_donors.py`:

```python
EXPLORATORY_SINGLE_EVENT_SCOPE = "exploratory_single_event"


def donor_selection_event_ids(selection: Mapping) -> frozenset[str]:
    scope = selection.get("protocol_scope")
    declared = selection.get("target_event_ids")
    if scope is None and declared is None:
        return TARGET_EVENT_IDS
    if scope != EXPLORATORY_SINGLE_EVENT_SCOPE:
        raise ValueError("unsupported donor selection protocol_scope")
    if (
        not isinstance(declared, list)
        or len(declared) != 1
        or not isinstance(declared[0], str)
        or declared[0] not in TARGET_EVENT_IDS
    ):
        raise ValueError("exploratory donor selection must declare exactly one frozen target event ID")
    return frozenset(declared)
```

At freeze time, derive the scoped set from the optional CLI value before `_selected_reviews`. Filter eligible candidates and structural rejections by `target_event_id`, require review rows to cover exactly the eligible candidates in that scope, and reject rows from other targets. Keep formal tie handling unchanged; exploratory mode must have exactly one approved row. Materialize and audit only the scoped rows. Append `protocol_scope` and `target_event_ids` only in exploratory mode so the formal JSON shape and ordering stay unchanged.

- [ ] **Step 5: Wire the existing CLI**

Add the one optional `freeze` argument and pass it through:

```python
freeze.add_argument("--exploratory-target-event-id")

# freeze branch
exploratory_target_event_id=args.exploratory_target_event_id,
```

Do not create a second preparation script.

- [ ] **Step 6: Run the complete donor-selection suite**

Run:

```bash
python -m pytest -q utest/tests/test_vistory_donors.py
python tools/prepare_vistory_donors.py freeze --help
```

Expected: all tests pass; help shows the optional exploratory event selector; no GPU is used.

- [ ] **Step 7: Commit Task 1**

```bash
git add utest/vistory_donors.py tools/prepare_vistory_donors.py utest/events/vistorybench_song_yuchen_donor_review_v1.json utest/tests/test_vistory_donors.py
git commit -m "feat: freeze scoped exploratory donor selection"
```

---

### Task 2: Make the existing donor harness derive job count from selection scope

**Files:**

- Modify: `utest/vistory_donor_harness.py`
- Test: `utest/tests/test_vistory_donor_harness.py`

**Interfaces:**

- `validate_frozen_selection` calls `donor_selection_event_ids` and compares the actual event rows with that set.
- Exploratory run manifests repeat `protocol_scope` and `target_event_ids`; formal manifests remain unchanged.
- `_canonical_jobs` remains the job builder and naturally emits one job for a one-event selection.
- CLI success requires exactly the validated manifest job count, not the literal `3`.

- [ ] **Step 1: Generalize the existing selection test fixture**

Change the test helper signature to:

```python
def _selection(
    tmp_path: Path,
    target_ids: tuple[str, ...] = TARGET_IDS,
) -> Path:
```

Replace the current `reversed(TARGET_IDS)` loop source with `reversed(target_ids)`; retain
the loop body unchanged. At the final `_write_json`, add this concrete conditional mapping
so only the one-event fixture gains scope fields:

```python
scope_fields = (
    {
        "protocol_scope": EXPLORATORY_SINGLE_EVENT_SCOPE,
        "target_event_ids": list(target_ids),
    }
    if len(target_ids) == 1
    else {}
)
_write_json(selection, {
    "schema_version": 1,
    "dataset_commit": "dataset-commit",
    "selection_seed": 0,
    "donor_seed": 0,
    "path_contract": {
        "selection_paths_relative_to": "selection_parent",
        "event_paths_relative_to": "event_parent",
    },
    "target_inputs_sha256": "a" * 64,
    "survey_sha256": "b" * 64,
    "review_sha256": "c" * 64,
    "candidate_audit": [],
    **scope_fields,
    "events": events,
})
```

- [ ] **Step 2: Write failing one-job and mismatch tests**

```python
def test_exploratory_selection_builds_and_validates_one_seed_zero_job(
    tmp_path: Path,
) -> None:
    selection = _selection(tmp_path, ("vistory79_song_yuchen_s2_s8",))
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    assert run["protocol_scope"] == "exploratory_single_event"
    assert run["target_event_ids"] == ["vistory79_song_yuchen_s2_s8"]
    assert [job["target_event_id"] for job in run["jobs"]] == [
        "vistory79_song_yuchen_s2_s8"
    ]


def test_exploratory_run_rejects_scope_event_mismatch(tmp_path: Path) -> None:
    selection = _selection(tmp_path, ("vistory79_song_yuchen_s2_s8",))
    raw = json.loads(selection.read_text(encoding="utf-8"))
    raw["target_event_ids"] = ["vistory42_bella_s15_s21"]
    _write_json(selection, raw)
    with pytest.raises(ValueError, match="selection target event IDs"):
        validate_frozen_selection(selection)
```

Add tests for a missing scope field, unsupported scope, extra job, missing job, and completed-run selection mismatch. Keep the existing exact-three tests unchanged.

- [ ] **Step 3: Run and observe the exact-three failure**

Run:

```bash
python -m pytest -q utest/tests/test_vistory_donor_harness.py -k "exploratory or exactly_three or completed_run"
```

Expected: new one-event tests fail on the current exact-three guards.

- [ ] **Step 4: Replace numeric literals with the validated scoped set**

Import `donor_selection_event_ids`. In `validate_frozen_selection`, compute `expected_ids`, require `len(events) == len(expected_ids)`, and compare actual IDs to `expected_ids`. In `build_donor_run_manifest` and its canonical validation, copy the two exploratory fields only when present. In `validate_completed_donor_run`, compare both selection events and run jobs with the same expected set.

For CLI status, derive the expected count from the validated run manifest:

```python
result = run_stage(args.command, args.manifest)
run = validate_donor_run_manifest(args.manifest)
statuses = {row.get("status") for row in result["results"]}
return 0 if (
    len(result["results"]) == len(run["jobs"])
    and statuses <= {"completed", "skipped_valid"}
) else 2
```

Do not change job commands, payload formats, seed handling, top-8/64 checks, or the existing RuntimeError module-entrypoint behavior.

- [ ] **Step 5: Run the donor harness suite**

```bash
python -m pytest -q utest/tests/test_vistory_donor_harness.py
python -m utest.vistory_donor_harness dry-run --help
```

- [ ] **Step 6: Commit Task 2**

```bash
git add utest/vistory_donor_harness.py utest/tests/test_vistory_donor_harness.py
git commit -m "feat: run scoped exploratory donor jobs"
```

---

### Task 3: Publish a one-event donor map through the existing bundle validator

**Files:**

- Modify: `utest/vistory_donor_bundle.py`
- Test: `utest/tests/test_vistory_donor_bundle.py`

**Interfaces:**

- `validate_target_inputs` remains exact-three: target authority is not reduced.
- `build_validated_event_donor_map` derives selection/job IDs from `donor_selection_event_ids(selection)`.
- Exploratory `donor_map.json` records the same two scope fields and contains only Song.
- `tools/freeze_vistory_donor_map.py` is reused unchanged; the selection artifact is its scope authority.

- [ ] **Step 1: Extend the completed fixture without duplicating it**

Add an optional `selected_target_ids` parameter to `_completed_fixture`. Continue building all three prepared targets, but filter the selection events and donor jobs when an exploratory set is supplied, then add the two scope fields to selection and run.

- [ ] **Step 2: Write failing partial-map and cross-boundary tests**

```python
def test_exploratory_bundle_keeps_complete_targets_but_publishes_only_song(
    tmp_path: Path,
) -> None:
    targets, selection, run = _completed_fixture(
        tmp_path,
        selected_target_ids={"vistory79_song_yuchen_s2_s8"},
    )
    result = freeze_vistory_donor_map(
        target_inputs_path=targets,
        selection_path=selection,
        donor_run_manifest_path=run,
        output_root=tmp_path / "bundle",
    )
    assert result["protocol_scope"] == "exploratory_single_event"
    assert result["target_event_ids"] == ["vistory79_song_yuchen_s2_s8"]
    assert set(result["events"]) == {"vistory79_song_yuchen_s2_s8"}
```

Add parameterized mutations for:

- selection scope names Song but selection event is Bella;
- completed run has an extra Bella job;
- completed run omits Song;
- donor-map scope fields differ from the validated selection;
- formal mode still rejects a one-event selection and still publishes all three.

- [ ] **Step 3: Run and observe the exact-three bundle failure**

```bash
python -m pytest -q utest/tests/test_vistory_donor_bundle.py -k "exploratory or exactly_three or missing_or_extra"
```

- [ ] **Step 4: Generalize only the selection/job side of the bundle**

Replace the literal selection/job comparison and loop:

```python
expected_ids = donor_selection_event_ids(selection)
if set(target_by_id) != TARGET_EVENT_IDS:
    raise ValueError("target inputs must contain the exact frozen three")
if set(selected_by_id) != expected_ids or set(jobs_by_id) != expected_ids:
    raise ValueError("selection and completed donor jobs do not match their scope")

for event_id in sorted(expected_ids):
```

Keep the current loop body unchanged beneath that header. Copy `protocol_scope` and
`target_event_ids` into the returned donor map only for exploratory selections. Preserve
all existing payload, shape, hash, identity, repository, symlink, and no-clobber validation.

- [ ] **Step 5: Run the full bundle suite**

```bash
python -m pytest -q utest/tests/test_vistory_donor_bundle.py
python tools/freeze_vistory_donor_map.py --help
```

- [ ] **Step 6: Commit Task 3**

```bash
git add utest/vistory_donor_bundle.py utest/tests/test_vistory_donor_bundle.py
git commit -m "feat: publish scoped exploratory donor map"
```

---

### Task 4: Prove the existing target harness selects only Song seed 0

**Files:**

- Test: `utest/tests/test_subject_reappearance_harness.py`
- Modify only if a regression proves necessary: `utest/subject_reappearance_harness.py`

**Interfaces:**

- No new target-harness mode or CLI argument.
- Continue building the canonical nine blocks from complete target inputs.
- A one-event donor map makes all three Song seeds donor-ready and leaves Bella/Chen blocked.
- Existing `--event-id vistory79_song_yuchen_s2_s8 --seed 0` narrows execution to one block.

- [ ] **Step 1: Add a minimal one-event donor-map fixture**

Use synthetic paths because this test isolates map lookup and readiness; monkeypatch
`_validated_donor` so payload semantics remain covered by `test_vistory_donor_bundle.py`:

```python
{
    "schema_version": 1,
    "protocol_scope": "exploratory_single_event",
    "target_event_ids": ["vistory79_song_yuchen_s2_s8"],
    "events": {
        "vistory79_song_yuchen_s2_s8": {
            "payload": str(payload_path),
            "manifest": str(manifest_path),
        }
    },
}
```

- [ ] **Step 2: Write the nine-block readiness regression**

```python
def test_partial_song_donor_map_keeps_nine_blocks_and_readies_only_song(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _prepared_inputs(tmp_path)
    base = tmp_path / "args.json"
    base.write_text(json.dumps({"argv": VALID_BASE_ARGV}), encoding="utf-8")
    platform = tmp_path / "platform.json"
    platform.write_text("{}", encoding="utf-8")
    donor_map = tmp_path / "donor_map.json"
    donor_map.write_text(json.dumps({
        "schema_version": 1,
        "protocol_scope": "exploratory_single_event",
        "target_event_ids": ["vistory79_song_yuchen_s2_s8"],
        "events": {
            "vistory79_song_yuchen_s2_s8": {
                "payload": "song-donor.pt",
                "manifest": "song-pair.json",
            }
        },
    }), encoding="utf-8")
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._validated_donor",
        lambda entry, _event: (
            None if entry is None else {
                "payload_path": entry["payload"],
                "manifest_path": entry["manifest"],
                "payload_key": "Colonel Cromarty|0",
            }
        ),
    )
    manifest = build_run_manifest(
        inputs=inputs,
        output=tmp_path / "run",
        base_inference_args=base,
        platform_manifest=platform,
        donor_map=donor_map,
    )
    assert len(manifest["blocks"]) == 9
    ready = [
        row for row in manifest["blocks"]
        if row["commands"]["preflight"]["status"] == "deferred_until_prefix"
    ]
    assert {(row["event_id"], row["seed"]) for row in ready} == {
        ("vistory79_song_yuchen_s2_s8", 0),
        ("vistory79_song_yuchen_s2_s8", 1),
        ("vistory79_song_yuchen_s2_s8", 2),
    }
```

- [ ] **Step 3: Write the selector and arm invariants**

```python
def test_song_pilot_selector_returns_only_seed_zero_and_existing_four_arms() -> None:
    manifest = {"blocks": [
        {"event_id": event_id, "seed": seed}
        for event_id in sorted(FROZEN_EVENTS)
        for seed in (0, 1, 2)
    ]}
    selected = _selected_blocks(manifest, "vistory79_song_yuchen_s2_s8", 0)
    assert [(row["event_id"], row["seed"]) for row in selected] == [
        ("vistory79_song_yuchen_s2_s8", 0)
    ]
    assert PREFLIGHT_ARMS == (
        "full_correct", "no_memory", "zero_path", "wrong_subject"
    )
```

Retain the existing shared-snapshot/shared-target-seed assertion. Assert the Song block has Q* status `not_available` without a teacher map.

- [ ] **Step 4: Run tests before changing production**

```bash
python -m pytest -q utest/tests/test_subject_reappearance_harness.py -k "partial_song or song_pilot or arm_orders"
```

Expected: pass using current target code. If it passes, do not edit `utest/subject_reappearance_harness.py`. If it fails, make only the smallest root-cause fix and rerun the entire file.

- [ ] **Step 5: Commit Task 4**

```bash
git add utest/tests/test_subject_reappearance_harness.py
git commit -m "test: freeze Song-only pilot execution scope"
```

---

### Task 5: Integrate, audit, and independently review before server GPU work

**Files:**

- Modify: `utest/README.md`
- Review: all Task 1-4 changes

- [ ] **Step 1: Add one concise Song-only pilot section to the existing README**

Document the exact commands from Task 6 below. Mark the output `exploratory_single_event`, state that it is not part of the primary three-event aggregate, and state the stop after preflight.

- [ ] **Step 2: Run focused CPU verification**

```bash
python -m pytest -q \
  utest/tests/test_vistory_donors.py \
  utest/tests/test_vistory_donor_harness.py \
  utest/tests/test_vistory_donor_bundle.py \
  utest/tests/test_subject_reappearance_harness.py
python -m utest.subject_subspace_audit --self-check
python -m utest.eligibility --self-check
```

- [ ] **Step 3: Run the complete CPU test suite**

```bash
python -m pytest -q
```

- [ ] **Step 4: Perform two-level review**

Assign the combined diff to a fresh independent review subagent. The root then verifies:

- formal exact-three serialization is unchanged when no exploratory flag is supplied;
- every exploratory artifact self-identifies its scope and exact event ID;
- the full target input manifest remains exact-three;
- only selection/jobs/map become one-event;
- all strict hashes, paths, visual-match decisions, top-8/64 checks, and no-clobber rules remain active;
- no new dependency, environment logic, `exit`, `SystemExit`, Q* decision path, or unrelated cleanup was introduced.

- [ ] **Step 5: Commit documentation and any accepted review fixes**

```bash
git add utest/README.md
git commit -m "docs: add Song-only exploratory runbook"
```

Do not push until all reviews and verification pass and the user explicitly requests it.

---

### Task 6: Reuse the frozen server data and generate the four videos

This is an operational task after the reviewed code is pushed and pulled onto the A100 server. Run inside the already-active `slotmem` environment.

- [ ] **Step 1: Set existing paths and create a fresh output namespace**

```bash
cd /data/long_term_data/shixiao/videomem/U-test-vistory-8f0b728

export VM_ROOT=/data/long_term_data/shixiao/videomem
export VISTORY_REV=92f845531b67e97a67ae04b256ec5d8c020e8341
export VISTORY_DATA="$VM_ROOT/datasets/ViStoryBench-full-$VISTORY_REV/ViStoryBench"
export TARGET_INPUTS="$PWD/runs/vistorybench_reappearance_v1/inputs/manifest.json"
export BASE_ARGS="$PWD/runs/vistorybench_reappearance_v1/config/base_inference_args.json"
export PLATFORM_MANIFEST="$PWD/platform.manifest.json"
export SONG_EVENT=vistory79_song_yuchen_s2_s8
export SONG_ROOT="$PWD/runs/vistorybench_song_yuchen_exploratory_v1"

python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["SONG_ROOT"])
assert not root.exists(), f"fresh output root required: {root}"
for name in ("VISTORY_DATA", "TARGET_INPUTS", "BASE_ARGS", "PLATFORM_MANIFEST"):
    assert Path(os.environ[name]).exists(), f"missing {name}: {os.environ[name]}"
print("Song pilot root:", root)
PY
```

- [ ] **Step 2: Regenerate the complete survey and verify its frozen hash**

```bash
mkdir -p "$SONG_ROOT"

python tools/prepare_vistory_donors.py survey \
  --data-root "$VISTORY_DATA" \
  --targets "$TARGET_INPUTS" \
  --output "$SONG_ROOT/survey.json"

python - <<'PY'
import hashlib
import os
from pathlib import Path

path = Path(os.environ["SONG_ROOT"]) / "survey.json"
actual = hashlib.sha256(path.read_bytes()).hexdigest()
expected = "8d70b19a1b8c4c293495e5e2b853d54e4431ccd8bc9aa38c9f50d5c914baad57"
print("survey sha256:", actual)
assert actual == expected
PY
```

- [ ] **Step 3: Freeze the one-event selection from the checked-in review**

```bash
python tools/prepare_vistory_donors.py freeze \
  --data-root "$VISTORY_DATA" \
  --targets "$TARGET_INPUTS" \
  --survey "$SONG_ROOT/survey.json" \
  --review "$PWD/utest/events/vistorybench_song_yuchen_donor_review_v1.json" \
  --output-root "$SONG_ROOT/selection" \
  --exploratory-target-event-id "$SONG_EVENT"
```

Verify one event and the exact donor before GPU use:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

selection = json.loads(
    (Path(os.environ["SONG_ROOT"]) / "selection" / "selection.json").read_text()
)
assert selection["protocol_scope"] == "exploratory_single_event"
assert selection["target_event_ids"] == [os.environ["SONG_EVENT"]]
assert len(selection["events"]) == 1
assert selection["events"][0]["candidate_id"] == (
    "ad04290d1a7ddd4691b8337c3a71afca0e8daee38a706be8c5a40f83bb725938"
)
print("selection verified:", selection["events"][0]["donor_entity_uid"])
PY
```

- [ ] **Step 4: Build and execute the existing one-job donor harness**

```bash
python -m utest.vistory_donor_harness dry-run \
  --selection "$SONG_ROOT/selection/selection.json" \
  --output "$SONG_ROOT/donor_run" \
  --base-inference-args "$BASE_ARGS" \
  --platform-manifest "$PLATFORM_MANIFEST"

CUDA_VISIBLE_DEVICES=0 \
DUAL_EXPERT_LOAD_MODE=active \
DUAL_EXPERT_MANAGE_AUX_MODELS=1 \
python -m utest.vistory_donor_harness resume \
  --manifest "$SONG_ROOT/donor_run/run_manifest.json"
```

- [ ] **Step 5: Freeze the partial donor map**

```bash
python tools/freeze_vistory_donor_map.py \
  --targets "$TARGET_INPUTS" \
  --selection "$SONG_ROOT/selection/selection.json" \
  --donor-run-manifest "$SONG_ROOT/donor_run/run_manifest.json" \
  --output-root "$SONG_ROOT/donor_bundle"
```

- [ ] **Step 6: Build the canonical nine-block target manifest and audit readiness**

```bash
python -m utest.subject_reappearance_harness dry-run \
  --inputs "$TARGET_INPUTS" \
  --output "$SONG_ROOT/target_run" \
  --base-inference-args "$BASE_ARGS" \
  --platform-manifest "$PLATFORM_MANIFEST" \
  --donor-map "$SONG_ROOT/donor_bundle/donor_map.json"

python - <<'PY'
import json
import os
from pathlib import Path

run = json.loads(
    (Path(os.environ["SONG_ROOT"]) / "target_run" / "run_manifest.json").read_text()
)
assert len(run["blocks"]) == 9
ready = [
    (row["event_id"], row["seed"])
    for row in run["blocks"]
    if row["commands"]["preflight"]["status"] == "deferred_until_prefix"
]
assert ready == [(os.environ["SONG_EVENT"], seed) for seed in (0, 1, 2)]
selected = [
    row for row in run["blocks"]
    if row["event_id"] == os.environ["SONG_EVENT"] and row["seed"] == 0
]
assert len(selected) == 1
assert selected[0]["preflight_arms"] == [
    "full_correct", "no_memory", "zero_path", "wrong_subject"
]
assert selected[0]["qstar"] == {
    "status": "not_available", "reason": "independent_teacher_missing"
}
print("ready blocks:", ready)
PY
```

- [ ] **Step 7: Execute only Song seed 0 through preflight**

```bash
CUDA_VISIBLE_DEVICES=0 \
DUAL_EXPERT_LOAD_MODE=active \
DUAL_EXPERT_MANAGE_AUX_MODELS=1 \
python -m utest.subject_reappearance_harness prefix \
  --manifest "$SONG_ROOT/target_run/run_manifest.json" \
  --event-id "$SONG_EVENT" \
  --seed 0

CUDA_VISIBLE_DEVICES=0 \
DUAL_EXPERT_LOAD_MODE=active \
DUAL_EXPERT_MANAGE_AUX_MODELS=1 \
python -m utest.subject_reappearance_harness probe \
  --manifest "$SONG_ROOT/target_run/run_manifest.json" \
  --event-id "$SONG_EVENT" \
  --seed 0

CUDA_VISIBLE_DEVICES=0 \
DUAL_EXPERT_LOAD_MODE=active \
DUAL_EXPERT_MANAGE_AUX_MODELS=1 \
python -m utest.subject_reappearance_harness preflight \
  --manifest "$SONG_ROOT/target_run/run_manifest.json" \
  --event-id "$SONG_EVENT" \
  --seed 0
```

- [ ] **Step 8: Stop and package the four-arm evidence**

Confirm the validation record exists and collect only the Song seed-0 block:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

block = (
    Path(os.environ["SONG_ROOT"])
    / "target_run"
    / os.environ["SONG_EVENT"]
    / "seed_0"
)
validation = block / "preflight" / "validation.json"
report = json.loads(validation.read_text())
assert report["arms"] == [
    "full_correct", "no_memory", "zero_path", "wrong_subject"
]
print("validated:", validation)
print("videos:")
for path in sorted((block / "preflight" / "arms").rglob("*.mp4")):
    print(path)
PY

tar -czf "$SONG_ROOT.tar.gz" -C "$(dirname "$SONG_ROOT")" "$(basename "$SONG_ROOT")"
```

Do not invoke `full`, `qstar`, or an aggregate-analysis command until the four videos receive human visual review.

---

## Final acceptance checklist

- [ ] Formal donor selection, harness, and bundle tests still prove exact-three behavior.
- [ ] Exploratory scope is explicit and identical in selection, run manifest, completion validation, and donor map.
- [ ] The checked-in review is bound to the complete survey hash and covers exactly the three Song candidates.
- [ ] Colonel Cromarty is the only approved donor; Bella and Chen remain unapproved and blocked.
- [ ] Donor job count is one; target manifest block count is nine; executed block count is one.
- [ ] The four arms share the same Song seed-0 prefix snapshot and target noise.
- [ ] Payload geometry is layers `0-15` and 64 slots; selected semantic mask budget is top 8.
- [ ] Q* is unavailable and absent from execution/verdict logic.
- [ ] Four preflight videos and `validation.json` exist; no full-stage or formal aggregate output exists.
