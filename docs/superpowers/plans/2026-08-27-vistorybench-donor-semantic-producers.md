# ViStoryBench Donor And Source Semantic Producers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce three frozen matched-wrong SlotMem donor bundles and source-only `semantic_top8` score artifacts so the existing three-event, three-seed subject-reappearance harness can run all nine formal blocks without `blocked_missing_donor` or externally supplied semantic scores.

**Architecture:** Keep the existing event, prefix, donor, probe, mask, and evaluation contracts authoritative. Add one pure ViStoryBench donor-survey/freeze module, one thin donor orchestration module, and one source-semantic producer. The subject harness invokes the semantic producer between source capture and mask selection. Donor payloads are generated once at seed 0 per target event, frozen with complete provenance, and reused for target seeds 0, 1, and 2.

**Tech Stack:** Python 3.10 standard library, PyTorch already installed by SlotMem, existing `utest` contracts and CLIs, `pytest`, official ViStoryBench files, existing Wan2.2/SlotMem checkpoints.

## Global Constraints

- Work on the current branch; do not create a Conda environment or install another dependency.
- Do not add `exit`, shell `exit`, or Python `raise SystemExit` to repository scripts.
- Preserve the three frozen target events and target seeds `[0, 1, 2]`.
- The target event IDs are exactly `vistory79_song_yuchen_s2_s8`, `vistory15_gu_zhenzhen_s8_s20`, and `vistory16_chen_father_s1_s10`.
- Generate exactly three donor payloads: one donor seed-0 payload for each target event, shared by all target seeds.
- Donor identity must differ from the target identity and story. Donor matching must enforce official style/tag, source character count, horizon bucket, reviewed presentation class, and reviewed dominant clothing-colour class.
- `semantic_top8` must use only source token metadata and source MemoryEncoder attention. It must not read target latents, target frames, Q*, CIDS, or decoded outcomes.
- Reuse the existing `FROZEN_LAYER_GROUPS` keys and members exactly: `0-4=[0..4]`, `5-10=[5..10]`, and `11-15=[11..15]`; the selected mask is top 8 of 32 slots.
- Every generated artifact is atomic, no-clobber, content-hashed, and bound to repository/platform/source provenance.
- Q* remains descriptive only and continues to report unavailable unless an independent teacher artifact exists.
- Every task is implemented by a fresh subagent and reviewed before the next dependent task begins.

## File Map

- Modify `utest/subject_subspace.py`: add the pure source-token group construction used by the semantic producer.
- Modify `utest/prefix_contract.py`: add one standard-library atomic JSON no-clobber publisher reused by all new producers.
- Create `utest/source_semantic_scores.py`: validate source capture, construct source-only score vectors, project them through existing attention, and atomically write the semantic score artifact.
- Modify `utest/subject_reappearance_harness.py`: declare and execute semantic-score production before probe/mask stages.
- Create `utest/vistory_donors.py`: enumerate official recurrence candidates, enforce structured constraints, consume reviewed attributes, and freeze three donor event bundles.
- Create `tools/prepare_vistory_donors.py`: thin CLI for donor survey and selection freeze.
- Create `utest/vistory_donor_harness.py`: build and execute the three seed-0 donor prefix/dump jobs by calling the existing event harness.
- Create `utest/vistory_donor_bundle.py`: validate dumped donor payloads, write matched-pair manifests, and freeze the event-to-donor map consumed by the subject harness.
- Create `tools/freeze_vistory_donor_map.py`: thin CLI for the final donor map.
- Create focused test files for each new producer and extend the existing subject harness tests.
- Modify `utest/tests/test_prefix_contract.py`: verify atomic JSON publication and collision rejection.
- Modify `utest/README.md`: document the exact CPU/GPU sequence and success gates.

---

### Task 1: Implement the source-only semantic score producer

**Files:**

- Modify: `utest/subject_subspace.py`
- Modify: `utest/subject_subspace_probe.py`
- Modify: `utest/prefix_contract.py`
- Create: `utest/source_semantic_scores.py`
- Modify: `utest/tests/test_prefix_contract.py`
- Create: `utest/tests/test_source_semantic_scores.py`

- [ ] **Step 1: Write failing tests for the frozen token-group formula**

Add tests that construct raw-token metadata with target-inside, target-outside, other-character, and background tokens. Assert the exact four groups:

```python
def test_source_metadata_groups_follow_frozen_formula():
    metadata = [
        {"char_id": "alice", "inside_box": True, "tau_local": 0.0},
        {"char_id": "alice", "inside_box": True, "tau_local": 1.0},
        {"char_id": "alice", "inside_box": False, "tau_local": 0.0},
        {"char_id": "bob", "inside_box": True, "tau_local": 0.0},
        {"char_id": "", "inside_box": False, "tau_local": 0.0},
    ]

    groups = source_metadata_semantic_groups(metadata, "alice")

    assert groups["identity_name"] == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert groups["stable_attributes"] == pytest.approx(
        [1.0, math.exp(-0.5), 0.0, 0.0, 0.0]
    )
    assert groups["other_characters"] == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert groups["action_scene"] == [0.0, 0.0, 1.0, 0.0, 0.0]
```

Also assert fail-closed behavior for no target tokens, no inside-box target token, non-finite `tau_local`, and vector length mismatch.

Add a prefix-contract test that publishes canonical JSON twice to different paths and obtains identical bytes, then asserts publication to an existing path raises `FileExistsError` without changing the original bytes.

- [ ] **Step 2: Run the focused tests and confirm the new API is absent**

Run:

```bash
pytest -q utest/tests/test_source_semantic_scores.py
```

Expected: collection/import failure because `source_metadata_semantic_groups` and `source_semantic_scores` do not exist.

- [ ] **Step 3: Add the minimum pure grouping function**

Add this public signature to `utest/subject_subspace.py`:

```python
def source_metadata_semantic_groups(
    raw_token_metadata: Sequence[Mapping[str, object]],
    subject_char_id: str,
) -> dict[str, list[float]]:
    """Return frozen source-only semantic vectors in raw-token order."""
```

Implement exactly:

```python
is_target = float(char_id == subject_char_id)
inside = is_target * float(inside_box)
centre = inside * math.exp(-(tau_local ** 2) / 2.0)
is_other = float(bool(char_id) and char_id != subject_char_id)
outside = is_target * float(not inside_box)
```

Return `identity_name=is_target`, `stable_attributes=centre`, `other_characters=is_other`, and `action_scene=outside`. Reject empty subject IDs, malformed metadata, non-finite coordinates, missing target tokens, and missing inside-box target evidence.

- [ ] **Step 4: Write failing producer tests around existing artifact contracts**

Use a tiny source-capture fixture containing:

- source JSON hash and subject ID;
- raw-token metadata;
- subject-capture rows whose attention tensors cover every required layer ID;
- source seed plus repository, runtime, and model provenance.

Assert that `produce_source_semantic_scores`:

- calls the existing `build_semantic_score_artifact` path rather than reproducing its provenance schema;
- produces the four frozen group names;
- preserves per-capture token score vectors, bank IDs, and layer IDs;
- records the existing producer kind `slotmem_source_semantic_token_scores`, source-capture hash, source JSON hash, code/model identity, and semantic-vocabulary hash; runtime/platform-manifest binding remains enforced by the source capture and enclosing formal run manifest;
- records the exact formula contract `{"name": "source_role_box_centre", "version": 1}`, subject character, source seed, and `target_evidence_read=false` in the existing artifact/producer schema;
- cannot overwrite an existing output;
- rejects missing layers, wrong token counts, a changed source JSON, and provenance mismatch.

- [ ] **Step 5: Implement the producer and CLI**

Create `utest/source_semantic_scores.py` with these public entry points:

```python
SOURCE_SEMANTIC_FORMULA = {"name": "source_role_box_centre", "version": 1}

def produce_source_semantic_scores(
    *,
    event_path: Path,
    source_capture_path: Path,
    output_path: Path,
    repo_root: Path,
) -> dict[str, object]:
    event, story = load_event_and_story(event_path)
    capture, subject_rows = load_and_validate_source_capture(
        source_capture_path, event, repo_root
    )
    score_rows = [
        {
            "character": row["character"],
            "bank": row["bank"],
            "layer": row["layer"],
            "groups": source_metadata_semantic_groups(
                row["raw_token_meta"], event["character_name"]
            ),
        }
        for row in subject_rows
    ]
    artifact = build_semantic_score_artifact(
        event_id=event["event_id"],
        source_capture_sha256=sha256_file(source_capture_path),
        source_capture_canonical_artifact_sha256=capture["canonical_artifact_sha256"],
        semantic_manifest=source_only_semantic_group_manifest(story, event),
        source_provenance=capture["provenance"],
        captures=score_rows,
        formula=SOURCE_SEMANTIC_FORMULA,
    )
    write_json_no_clobber(output_path, artifact)
    return artifact

def validate_source_semantic_scores_file(
    *, event_path: Path, source_capture_path: Path, scores_path: Path, repo_root: Path
) -> dict[tuple[int, int], Mapping[str, object]]:
    return validate_scores_against_source_files(
        event_path, source_capture_path, scores_path, repo_root
    )

def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    produce_source_semantic_scores(
        event_path=args.event,
        source_capture_path=args.source_capture,
        output_path=args.output,
        repo_root=args.repo_root,
    )
    return 0
```

The CLI is:

```bash
python -m utest.source_semantic_scores \
  --event EVENT_JSON \
  --source-capture SOURCE_CAPTURE_PT \
  --output SEMANTIC_SCORES_JSON \
  --repo-root REPOSITORY_ROOT
```

Add `write_json_no_clobber(path, value)` to `utest/prefix_contract.py` using a same-directory temporary file plus an atomic hard-link publication that fails if the destination already exists. Validate the source capture with existing helpers, compute raw vectors with `source_metadata_semantic_groups` for every subject capture, and delegate provenance packaging to `build_semantic_score_artifact`. Extend that existing builder and validator with the exact `SOURCE_SEMANTIC_FORMULA`, subject, and source-seed fields; update the existing `subject_subspace_probe` self-check to use the same contract. Do not project attention in the producer: the existing probe must continue to call `slot_attention_matrix`, then `semantic_slot_scores`, through `aggregate_semantic_slot_scores`, and average each existing frozen layer group. Write the JSON artifact through `write_json_no_clobber`.

- [ ] **Step 6: Run focused and neighboring tests**

Run:

```bash
pytest -q utest/tests/test_prefix_contract.py utest/tests/test_source_semantic_scores.py utest/tests/test_subject_subspace.py
python -m utest.subject_subspace_probe --self-check
python -m utest.source_semantic_scores --help
```

Expected: all tests pass and the CLI exposes only event, source capture, output, and repository-root inputs; no target path is accepted.

- [ ] **Step 7: Commit Task 1**

```bash
git add utest/prefix_contract.py utest/subject_subspace.py utest/subject_subspace_probe.py utest/source_semantic_scores.py utest/tests/test_prefix_contract.py utest/tests/test_source_semantic_scores.py utest/tests/test_subject_subspace.py
git commit -m "feat: produce source-only semantic slot scores"
```

---

### Task 2: Insert semantic production into the subject-reappearance harness

**Files:**

- Modify: `utest/subject_reappearance_harness.py`
- Modify: `utest/tests/test_subject_reappearance_harness.py`

- [ ] **Step 1: Write failing manifest-construction tests**

Extend dry-run tests to require a generated command entry:

```python
semantic = block["commands"]["semantic_scores"]
assert semantic[:3] == [sys.executable, "-m", "utest.source_semantic_scores"]
assert block["source_capture"] in semantic
assert block["semantic_scores"] in semantic
assert not any("target_latent" in arg or "target_frame" in arg for arg in semantic)
```

Assert that a fresh dry-run no longer classifies semantic scores as a user-supplied external input.

- [ ] **Step 2: Run focused tests and observe the missing command**

Run:

```bash
pytest -q utest/tests/test_subject_reappearance_harness.py -k "semantic or dry_run"
```

Expected: assertions fail because the current manifest contains only the downstream probe command and treats the score file as external.

- [ ] **Step 3: Add the semantic producer command without changing artifact paths**

In `build_run_manifest`, keep the existing per-block paths:

```python
source_capture = block_dir / "subspace" / "source_capture.pt"
semantic_scores = block_dir / "subspace" / "semantic_scores.json"
```

Add a `commands["semantic_scores"]` argv calling Task 1's module with:

- the frozen event JSON;
- that block's source-capture artifact;
- the existing semantic score path;
- the harness module's resolved repository root.

Add `semantic_scores` as a top-level block artifact path and remove it from `required_external_inputs`; the generated artifact remains required by probe and mask validation. Add separate semantic-producer stdout/stderr log paths beside the existing probe logs.

- [ ] **Step 4: Write failing execution-order and resume tests**

Add tests with subprocess execution stubbed at the harness boundary. Assert:

1. `probe` stage validates source capture;
2. it executes `semantic_scores` when the file is absent;
3. it validates the produced semantic artifact;
4. it executes the existing probe command;
5. resume skips a valid score artifact;
6. resume rejects a present but invalid or provenance-mismatched artifact instead of overwriting it.

- [ ] **Step 5: Implement the execution ordering**

Factor the smallest shared helper needed by both normal and resume paths:

```python
def _ensure_semantic_scores(block: Mapping[str, object]) -> None:
    """Produce once, validate always, and never overwrite an invalid artifact."""
```

Call it at the start of the existing probe path, including before the completed-mask resume fast path. If the score file exists, call Task 1's `validate_source_semantic_scores_file`; if it does not, execute the producer command and then validate it. Do not add a new top-level experiment stage: semantic scoring is a deterministic producer owned by `probe`.

- [ ] **Step 6: Run the harness regression tests**

Run:

```bash
pytest -q utest/tests/test_subject_reappearance_harness.py utest/tests/test_subject_subspace.py utest/tests/test_source_semantic_scores.py
python -m utest.subject_reappearance_harness dry-run --help
```

Expected: all tests pass; existing stage names and artifact locations remain stable.

- [ ] **Step 7: Commit Task 2**

```bash
git add utest/subject_reappearance_harness.py utest/tests/test_subject_reappearance_harness.py
git commit -m "feat: materialize semantic scores before probing"
```

---

### Task 3: Survey and freeze matched-wrong ViStoryBench donor events

**Files:**

- Create: `utest/vistory_donors.py`
- Create: `tools/prepare_vistory_donors.py`
- Create: `utest/tests/test_vistory_donors.py`

- [ ] **Step 1: Write failing tests for official recurrence enumeration**

Build small official-story fixtures with explicit character lists, shot prompts, tags, reference images, and recurrence gaps. Assert:

- candidate identity/story differs from target;
- candidate is visible at source and read-check shots;
- every intermediate shot excludes that identity;
- realistic/non-realistic style and official character tag equal the target;
- source-shot character count equals the target source-shot count;
- horizon buckets are exactly `5-7`, `8-10`, or `11-13`;
- invalid reference paths and ambiguous duplicate character names are rejected;
- candidates are sorted by stable candidate ID, independent of filesystem order.

Define `style_class` mechanically from the official tag: `realistic_human -> realistic`; `unrealistic_human` and `non_human -> non_realistic`. Exact tag equality remains a separate hard constraint. Treat `Characters Appearing.en` as structural presence only; reviewed visibility is enforced during freeze.

Use these signatures:

```python
def horizon_bucket(horizon: int) -> str:
    for lower, upper in ((5, 7), (8, 10), (11, 13)):
        if lower <= horizon <= upper:
            return f"{lower}-{upper}"
    raise ValueError(f"unsupported donor horizon: {horizon}")

def build_donor_candidate_survey(
    *, data_root: Path, target_inputs_path: Path, output_path: Path
) -> dict[str, object]:
    targets = load_target_inputs(target_inputs_path)
    candidates, rejections = enumerate_official_recurrences(data_root, targets)
    survey = freeze_survey_record(targets, candidates, rejections, selection_seed=0)
    write_json_no_clobber(output_path, survey)
    return survey
```

- [ ] **Step 2: Run the survey tests and confirm failure**

Run:

```bash
pytest -q utest/tests/test_vistory_donors.py -k survey
```

Expected: import failure because the donor module does not exist.

- [ ] **Step 3: Implement deterministic survey generation**

Reuse the existing ViStoryBench story discovery/parsing and `convert_event` validation. Enumerate all valid consecutive appearances of every non-target identity whose gap falls in the target horizon bucket. Derive a candidate ID from canonical JSON containing:

```python
{
    "target_event_id": target_event_id,
    "donor_story_id": donor_story_id,
    "donor_char_id": donor_char_id,
    "source_shot": source_shot,
    "read_shot": read_shot,
}
```

The survey records every accepted candidate plus every rejection reason, official story/reference hashes, prompts needed for review, dataset commit, target-input manifest hash, and `selection_seed=0`.

- [ ] **Step 4: Write failing tests for reviewed-attribute freeze**

The review input has this strict schema:

```json
{
  "schema_version": 1,
  "dataset_commit": "official commit",
  "reviews": [
    {
      "target_event_id": "target event",
      "candidate_id": "survey candidate hash",
      "target_presentation_class": "male",
      "donor_presentation_class": "male",
      "target_dominant_colour": "black",
      "donor_dominant_colour": "black",
      "donor_source_visible": true,
      "donor_read_check_visible": true,
      "approved": true,
      "tie_group": null,
      "reviewer": "human"
    }
  ]
}
```

Assert freeze rejects unreviewed candidates, unequal presentation/colour classes, a false source/read visibility decision, stale candidate IDs, dataset/hash mismatch, zero approved candidates, undeclared multiple matches, and a non-zero selection seed. If multiple approved candidates for one target all declare the same non-empty `tie_group`, sort them by candidate ID and use `random.Random(0)` only to break that declared tie; test that the result is deterministic. Assert the final selection produces exactly three donor event bundles.

- [ ] **Step 5: Implement the freeze path and thin CLI**

Add:

```python
def freeze_donor_selection(
    *,
    data_root: Path,
    target_inputs_path: Path,
    survey_path: Path,
    review_path: Path,
    output_root: Path,
) -> dict[str, object]:
    survey = validate_survey(survey_path, data_root, target_inputs_path)
    reviews = validate_reviews(review_path, survey)
    selection = materialize_reviewed_donor_events(data_root, survey, reviews, output_root)
    write_json_no_clobber(output_root / "selection.json", selection)
    return selection
```

For each approved candidate, call the existing ViStoryBench event converter and write a donor event directory containing derived story JSON, event JSON, reference image, and hashes. The top-level `selection.json` records all accepted/rejected candidates, review fields, target-to-donor mapping, and the fixed `donor_seed=0`. Use atomic/no-clobber writes.

Expose:

```bash
python tools/prepare_vistory_donors.py survey \
  --data-root VISTORY_ROOT \
  --targets TARGET_INPUT_MANIFEST \
  --output DONOR_SURVEY_JSON

python tools/prepare_vistory_donors.py freeze \
  --data-root VISTORY_ROOT \
  --targets TARGET_INPUT_MANIFEST \
  --survey DONOR_SURVEY_JSON \
  --review DONOR_REVIEW_JSON \
  --output-root DONOR_SELECTION_ROOT
```

- [ ] **Step 6: Run focused tests and CLI self-checks**

Run:

```bash
pytest -q utest/tests/test_vistory_donors.py
python tools/prepare_vistory_donors.py --help
python tools/prepare_vistory_donors.py survey --help
python tools/prepare_vistory_donors.py freeze --help
```

Expected: deterministic hashes are identical across repeated fixture runs and all fail-closed cases pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add utest/vistory_donors.py tools/prepare_vistory_donors.py utest/tests/test_vistory_donors.py
git commit -m "feat: freeze matched ViStoryBench donor events"
```

---

### Task 4: Orchestrate exactly three seed-0 donor prefix and dump jobs

**Files:**

- Create: `utest/vistory_donor_harness.py`
- Create: `utest/tests/test_vistory_donor_harness.py`

- [ ] **Step 1: Write failing dry-run contract tests**

Construct a three-entry frozen donor selection and assert:

```python
run = build_donor_run_manifest(
    selection_path=selection_path,
    output_root=output_root,
    base_inference_args_path=base_args_path,
    platform_manifest_path=platform_manifest_path,
    python_executable=sys.executable,
)
assert len(run["jobs"]) == 3
assert {job["target_event_id"] for job in run["jobs"]} == set(TARGET_IDS)
assert {job["donor_seed"] for job in run["jobs"]} == {0}
assert all(job["commands"]["prefix"]["argv"][1:4] == ["-m", "utest.event_harness", "prepare-prefix"] for job in run["jobs"])
assert all(job["commands"]["dump"]["argv"][1:4] == ["-m", "utest.event_harness", "dump-donor"] for job in run["jobs"])
```

Also assert the base inference argv is sanitized by existing helpers and the manifest binds selection hash, base-argv hash, platform-manifest hash, repository commit, and dirty state.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
pytest -q utest/tests/test_vistory_donor_harness.py
```

Expected: import failure because the orchestration module does not exist.

- [ ] **Step 3: Implement a thin wrapper over the existing event harness**

Provide:

```python
def build_donor_run_manifest(
    *,
    selection_path: Path,
    output_root: Path,
    base_inference_args_path: Path,
    platform_manifest_path: Path,
    python_executable: str,
) -> dict[str, object]:
    selection = validate_frozen_selection(selection_path)
    jobs = build_seed_zero_event_harness_jobs(selection, output_root, python_executable)
    return freeze_donor_run_record(
        selection, jobs, base_inference_args_path, platform_manifest_path
    )

def run_stage(stage: str, manifest_path: Path) -> dict[str, object]:
    run = validate_donor_run_manifest(manifest_path)
    results = [run_one_donor_job(stage, job) for job in run["jobs"]]
    return {"stage": stage, "results": results}
```

Supported stages are `dry-run`, `prefix`, `dump`, and `resume`. Each job delegates to the existing event harness; do not copy its inference or donor serialization logic. `dump` must validate the prefix before invocation. `resume` skips a valid prefix/payload-info pair, rejects invalid present artifacts, and never overwrites.

Do not expose a seed option: the module writes seed 0 unconditionally and validates the selection's frozen value.

- [ ] **Step 4: Add execution and failure-isolation tests**

Stub subprocess execution and assert:

- jobs run in stable target-event order;
- a failing job stops before later GPU work and records its log path;
- dump is not attempted before a valid prefix exists;
- successful dump produces the existing v2 donor payload and payload-info paths;
- resume performs validation, not mere existence checks;
- no code path raises `SystemExit`.

- [ ] **Step 5: Implement CLI status output without terminal exit commands**

Expose:

```bash
python -m utest.vistory_donor_harness dry-run \
  --selection DONOR_SELECTION_JSON \
  --output DONOR_RUN_ROOT \
  --base-inference-args BASE_ARGS_JSON \
  --platform-manifest PLATFORM_MANIFEST_JSON

python -m utest.vistory_donor_harness prefix --manifest DONOR_RUN_MANIFEST
python -m utest.vistory_donor_harness dump --manifest DONOR_RUN_MANIFEST
python -m utest.vistory_donor_harness resume --manifest DONOR_RUN_MANIFEST
```

Return integer status from `main`; invoke it directly under `if __name__ == "__main__"` without `raise SystemExit`.

- [ ] **Step 6: Run focused and event-harness regression tests**

Run:

```bash
pytest -q utest/tests/test_vistory_donor_harness.py utest/tests/test_event_harness.py
python -m utest.vistory_donor_harness --help
```

- [ ] **Step 7: Commit Task 4**

```bash
git add utest/vistory_donor_harness.py utest/tests/test_vistory_donor_harness.py
git commit -m "feat: orchestrate frozen donor generation"
```

---

### Task 5: Freeze validated donor payloads into the subject-harness donor map

**Files:**

- Create: `utest/vistory_donor_bundle.py`
- Create: `tools/freeze_vistory_donor_map.py`
- Create: `utest/tests/test_vistory_donor_bundle.py`

- [ ] **Step 1: Write failing bundle-validation tests**

Using minimal torch payload fixtures compatible with the existing v2 donor contract, assert:

- exactly one donor payload and one matched-pair manifest are emitted per target event;
- the top-level map has exactly three event keys and no seed keys;
- each event entry is therefore reusable by all target seeds;
- donor seed must be 0;
- payload key, slot count, tensor shape/dtype, payload hash, source-event hash, repository commit, and platform hash agree across selection, donor-run manifest, payload-info, and payload;
- target and donor identities/stories differ;
- any tamper, missing job, extra job, or cross-wired payload is rejected;
- existing outputs are never overwritten.

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
pytest -q utest/tests/test_vistory_donor_bundle.py
```

Expected: import failure because the bundle module does not exist.

- [ ] **Step 3: Implement the bundle freezer using existing donor validators**

Provide:

```python
def freeze_vistory_donor_map(
    *,
    target_inputs_path: Path,
    selection_path: Path,
    donor_run_manifest_path: Path,
    output_root: Path,
) -> dict[str, object]:
    targets = validate_target_inputs(target_inputs_path)
    selection = validate_frozen_selection(selection_path)
    donor_run = validate_completed_donor_run(donor_run_manifest_path, selection)
    result = build_validated_event_donor_map(targets, selection, donor_run, output_root)
    write_json_no_clobber(output_root / "donor_map.json", result)
    return result
```

For each target event:

1. resolve the matching donor run job;
2. validate its payload-info and v2 payload;
3. write a matched-pair manifest accepted by the existing `validate_donor_bundle` function;
4. validate the complete pair through that function;
5. add one event-level entry to the top-level map.

Populate every field already required by `validate_donor_bundle`: `target_story_id`, `target_entity_uid`, `donor_story_id`, `donor_entity_uid`, `payload_path`, `payload_sha256`, `coarse_class`, `colour`, `character_count`, `source_visible`, `gap_bucket`, `slot_shape`, `selection_seed`, and `payload_key`. Derive matching fields only from the frozen reviewed selection and derive payload fields only from `donor_payload_info.json` plus the v2 payload. Additional donor-run repository/platform provenance may be recorded, but cannot replace any required field.

The top-level artifact is:

```python
{
    "schema_version": 1,
    "selection_sha256": selection_sha256,
    "donor_run_manifest_sha256": donor_run_manifest_sha256,
    "events": {
        target_event_id: {
            "payload": str(donor_payload_path.resolve()),
            "manifest": str(pair_manifest_path.resolve()),
        }
    },
}
```

Resolve and hash paths before writing. Use atomic/no-clobber helpers.

- [ ] **Step 4: Add the thin CLI and an existing-harness acceptance test**

Expose:

```bash
python tools/freeze_vistory_donor_map.py \
  --targets TARGET_INPUT_MANIFEST \
  --selection DONOR_SELECTION_JSON \
  --donor-run-manifest DONOR_RUN_MANIFEST_JSON \
  --output-root DONOR_BUNDLE_ROOT
```

In the test, feed the generated donor map to `build_run_manifest` and assert all nine blocks report donor preflight readiness while every seed of a target resolves the same payload hash.

- [ ] **Step 5: Run focused and cross-contract tests**

Run:

```bash
pytest -q \
  utest/tests/test_vistory_donor_bundle.py \
  utest/tests/test_event_harness.py \
  utest/tests/test_subject_reappearance_harness.py
python tools/freeze_vistory_donor_map.py --help
```

- [ ] **Step 6: Commit Task 5**

```bash
git add utest/vistory_donor_bundle.py tools/freeze_vistory_donor_map.py utest/tests/test_vistory_donor_bundle.py
git commit -m "feat: freeze validated donor maps"
```

---

### Task 6: Document and verify the complete CPU/GPU workflow

**Files:**

- Modify: `utest/README.md`
- Modify only if a discovered regression requires it: tests owned by Tasks 1-5

- [ ] **Step 1: Add exact workflow commands to the README**

Document this order with concrete repository-relative output locations:

1. synchronize the complete official ViStoryBench tree;
2. regenerate the three frozen target inputs;
3. run donor survey;
4. inspect reference images/prompts and write the strict review JSON;
5. freeze exactly three donor events;
6. build the donor dry-run manifest;
7. run three seed-0 donor prefixes and dumps on GPU;
8. freeze the donor map;
9. rebuild the nine-block subject dry-run with `--donor-map`;
10. run target prefix/capture, source semantic production through probe, masks, preflight, full injection, CIDS, Q* status, and report.

State explicitly that the active `slotmem` Conda environment, existing Wan2.2 directory, and existing four SlotMem checkpoints are reused.

- [ ] **Step 2: Add observable gates, not prose-only assurances**

The README must list these checks:

```text
donor jobs:                 3
donor seeds:                [0]
formal target blocks:       9
target seeds:               [0, 1, 2]
donor preflight statuses:   ready
semantic producer kind:     slotmem_source_semantic_token_scores
semantic mask cardinality:  8 / 32
semantic layer groups:      0-4, 5-10, 11-15
Q*:                         descriptive or unavailable
```

Include a short Python audit command that reads the frozen manifests and asserts these values without loading GPU weights.

- [ ] **Step 3: Run source-level safety scans**

Run:

```bash
rg -n "target_latent|target_frame|decoded_video|cids|qstar" \
  utest/source_semantic_scores.py utest/subject_subspace.py
rg -n "raise SystemExit|(^|[;&[:space:]])exit([[:space:]]|$)" \
  utest/vistory_donor_harness.py tools/prepare_vistory_donors.py tools/freeze_vistory_donor_map.py
```

Expected: the semantic producer scan has no target/Q*/CIDS access; the exit scan is empty.

- [ ] **Step 4: Run complete zero-GPU verification**

Run:

```bash
python -m compileall -q utest tools
pytest -q
```

Expected: all tests pass with the existing CUDA-dependent skips only.

- [ ] **Step 5: Perform a clean fixture rehearsal**

In a temporary directory, run the fixture-sized equivalents of:

```bash
python tools/prepare_vistory_donors.py survey --data-root "$FIXTURE_ROOT/data" --targets "$FIXTURE_ROOT/targets/manifest.json" --output "$FIXTURE_ROOT/donors/survey.json"
python tools/prepare_vistory_donors.py freeze --data-root "$FIXTURE_ROOT/data" --targets "$FIXTURE_ROOT/targets/manifest.json" --survey "$FIXTURE_ROOT/donors/survey.json" --review "$FIXTURE_ROOT/donors/review.json" --output-root "$FIXTURE_ROOT/donors/selection"
python -m utest.vistory_donor_harness dry-run --selection "$FIXTURE_ROOT/donors/selection/selection.json" --output "$FIXTURE_ROOT/donor_run" --base-inference-args "$FIXTURE_ROOT/base_args.json" --platform-manifest "$FIXTURE_ROOT/platform.manifest.json"
python tools/freeze_vistory_donor_map.py --targets "$FIXTURE_ROOT/targets/manifest.json" --selection "$FIXTURE_ROOT/donors/selection/selection.json" --donor-run-manifest "$FIXTURE_ROOT/donor_run/run_manifest.json" --output-root "$FIXTURE_ROOT/donor_bundle"
python -m utest.subject_reappearance_harness dry-run --inputs "$FIXTURE_ROOT/targets/manifest.json" --output "$FIXTURE_ROOT/formal" --base-inference-args "$FIXTURE_ROOT/base_args.json" --platform-manifest "$FIXTURE_ROOT/platform.manifest.json" --donor-map "$FIXTURE_ROOT/donor_bundle/donor_map.json"
```

Assert three donor jobs, nine formal blocks, shared donor hash across each target's three seeds, and a semantic producer command for every block.

- [ ] **Step 6: Review the diff against the frozen design**

Verify each statement directly from code/tests:

- no new dependency;
- no target-conditioned semantic input;
- no alpha sweep or learned probe;
- donor seed fixed to 0;
- one donor bundle per target event;
- all artifacts no-clobber and provenance-bound;
- Q* behavior unchanged;
- no unrelated workspace files staged.

- [ ] **Step 7: Commit Task 6**

```bash
git add utest/README.md
git commit -m "docs: add donor and semantic production workflow"
```

---

## Server Execution Gate After Implementation

Do not regenerate `platform.manifest.json` after every implementation commit. Once all six tasks are reviewed, merged on the working branch, and pushed:

1. pull that single reviewed branch state on the A100 server;
2. run `scripts/fetch_weights.sh` once with `SKIP_PIP=1`, the active `slotmem` environment, the existing checkpoint root, and the existing Wan2.2 root;
3. confirm the new platform manifest records the pulled commit and expected dirty-state policy;
4. run the donor CPU survey/review/freeze phase;
5. run three donor GPU jobs;
6. freeze the donor map and regenerate the formal subject run manifest;
7. execute the nine formal blocks.

The GPU gate is passed only when all three donor read checks are hits, donor payload interventions are effective, all nine target blocks accept provenance, and every `semantic_top8` mask contains exactly eight valid slot indices.
