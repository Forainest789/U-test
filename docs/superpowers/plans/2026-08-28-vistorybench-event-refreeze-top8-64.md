# ViStoryBench Event Refreeze and Top-8/64 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the donor-ineligible Gu Zhenzhen target through a reproducible reviewed selection workflow, freeze the resulting three-event benchmark, and migrate every subject-subspace consumer to the checkpoint-compatible top-8/64 protocol.

**Architecture:** Keep the frozen event JSON as the target authority and the existing donor module as the sole owner of donor eligibility. Add one target-selection module that builds on a reusable recurrence inventory and donor-rejection function, and centralize the 64-slot/top-8 geometry in `prefix_contract.py` so masks, audits, bundles, and the run harness consume one contract. Execution has two explicit human gates: female-character review before target refreeze and donor visual review after the three targets are frozen.

**Tech Stack:** Python 3.10+, standard library, PyTorch, pytest, existing SlotMem/ViStoryBench utilities; no new dependency or Conda environment.

## Global Constraints

- Dataset revision is exactly `92f845531b67e97a67ae04b256ec5d8c020e8341` and the complete 80-story tree must validate.
- Retain `vistory79_song_yuchen_s2_s8` and `vistory16_chen_father_s1_s10` unchanged.
- Remove `vistory15_gu_zhenzhen_s8_s20`; the replacement must be a reviewer-confirmed female `realistic_human` recurrence with at least one unchanged-rule donor and a different `entity_uid` from all three original target identities.
- Replacement ordering is donor count descending, `abs(horizon - 12)` ascending, event ID ascending.
- Donor matching remains exact on official tag, style class, source character count, horizon bucket, cross-story identity, ambiguity, interval validity, and reference availability.
- Memory encoder layers are exactly `0–15`; slot universe is exactly `0–63`; intervention budget is exactly `8`; budget fraction is exactly `0.125`.
- Q* remains record-only and cannot enter ranking, mask construction, donor selection, or injection.
- Legacy 32-slot artifacts fail closed; no padding, truncation, duplication, or remapping is allowed.
- Keep seeds exactly `[0, 1, 2]` and preserve the nine-block `3 events × 3 seeds` matrix.
- Reuse the active `slotmem` Conda environment and existing WAN/SlotMem checkpoints; commands must not activate or create an environment.
- Do not add shell `exit` statements. Do not overwrite existing run artifacts; use a fresh run root.

---

## File structure

- `utest/prefix_contract.py`: single authority for layer count, slot count, top-k budget, and budget fraction.
- `utest/subject_subspace.py`: build 64-slot rankings and top-8 masks from the shared contract.
- `utest/subject_subspace_audit.py`: reject any manifest or tensor that violates the shared geometry.
- `utest/subject_reappearance_harness.py`: derive complements from 64 slots and reject donor/target incompatibility before GPU work.
- `utest/vistory_donor_bundle.py`: require the frozen 64-slot geometry in all donor payloads.
- `utest/vistory_donors.py`: expose one recurrence inventory and one donor-rejection function used by both donor survey and target selection.
- `utest/vistory_target_selection.py`: survey replacement targets, create a strict review template, and freeze a deterministic three-event selection.
- `tools/refreeze_vistory_reappearance_events.py`: CLI for `survey`, `review-template`, and `freeze`.
- `utest/events/vistorybench_reappearance_v1.json`: final three-event authority, changed only after the target-review gate.
- `utest/events/vistorybench_replacement_target_survey_v1.json`: checked-in machine-generated target ranking evidence.
- `utest/events/vistorybench_replacement_target_review_v1.json`: checked-in human female-character dispositions and reviewer provenance.
- `utest/tests/test_subject_subspace.py`, `test_prefix_contract.py`, `test_subject_reappearance_harness.py`, `test_vistory_donor_bundle.py`: top-8/64 regressions.
- `utest/tests/test_vistory_donors.py`, `test_vistory_target_selection.py`, `test_vistory_reappearance.py`: unchanged donor rules and deterministic target refreeze.
- `utest/README.md`: exact active-environment server commands and human-review gates.

---

### Task 1: Centralize and enforce the top-8/64 geometry

**Files:**
- Modify: `utest/prefix_contract.py`
- Modify: `utest/subject_subspace.py`
- Modify: `utest/subject_subspace_audit.py`
- Modify: `utest/tests/test_prefix_contract.py`
- Modify: `utest/tests/test_subject_subspace.py`

**Interfaces:**
- Produces: `FROZEN_MEMORY_ENCODER_LAYERS: tuple[int, ...] = tuple(range(16))`
- Produces: `FROZEN_MEMORY_ENCODER_SLOTS: int = 64`
- Produces: `FROZEN_SUBJECT_SUBSPACE_BUDGET: int = 8`
- Produces: `FROZEN_SUBJECT_SUBSPACE_FRACTION: float = 0.125`
- Produces: `validate_slotmem_memory_encoder_geometry(frozen_args: Mapping[str, object]) -> tuple[tuple[int, ...], int]`
- Consumes: no new external dependency.

- [ ] **Step 1: Write the failing geometry tests**

Add assertions that the real checkpoint geometry is accepted and the legacy geometry is rejected:

```python
from utest.prefix_contract import (
    FROZEN_MEMORY_ENCODER_SLOTS,
    FROZEN_SUBJECT_SUBSPACE_BUDGET,
    FROZEN_SUBJECT_SUBSPACE_FRACTION,
    validate_slotmem_memory_encoder_geometry,
)


def test_frozen_geometry_matches_existing_64_slot_checkpoint() -> None:
    layers, slots = validate_slotmem_memory_encoder_geometry(
        {
            "slotmem_memory_encoder_layers": "0-15",
            "slotmem_memory_encoder_slots": "64",
        }
    )
    assert layers == tuple(range(16))
    assert slots == FROZEN_MEMORY_ENCODER_SLOTS == 64
    assert FROZEN_SUBJECT_SUBSPACE_BUDGET == 8
    assert FROZEN_SUBJECT_SUBSPACE_FRACTION == 0.125


def test_legacy_32_slot_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="actual='32'.*frozen expected='64'"):
        validate_slotmem_memory_encoder_geometry(
            {
                "slotmem_memory_encoder_layers": "0-15",
                "slotmem_memory_encoder_slots": "32",
            }
        )
```

Update the subject-subspace fixture to create `torch.arange(192).reshape(64, 3)` encoded slots and `[64, 4]` attention. Assert each full ranking equals a permutation of `range(64)`, each mask has eight indices, and `budget_fraction == 0.125`.

Extend the existing target-evidence parametrization with `{"qstar": {"score": 1.0}}` and assert `build_mask_manifest` raises `ValueError`; this keeps Q* outside ranking and mask construction.

- [ ] **Step 2: Run the focused tests and confirm the old contract fails**

Run:

```bash
python -m pytest utest/tests/test_prefix_contract.py utest/tests/test_subject_subspace.py -q
```

Expected before implementation: failures mention expected slot count 32, ranking length 32, or budget fraction 0.25.

- [ ] **Step 3: Implement the shared constants and remove core numeric duplicates**

In `prefix_contract.py`, set the four constants exactly as declared under Interfaces and update the validator docstring and error text to say `64-slot` and `frozen expected='64'`.

In `subject_subspace.py`, import the three geometry constants and replace the hard-coded validation/build values:

```python
slot_universe = set(range(FROZEN_MEMORY_ENCODER_SLOTS))
if len(semantic) != FROZEN_MEMORY_ENCODER_SLOTS or set(semantic) != slot_universe:
    raise ValueError("semantic ranking must be a full frozen slot permutation")

semantic_top8 = sorted(semantic[:FROZEN_SUBJECT_SUBSPACE_BUDGET])
random_top8 = deterministic_random_indices(
    event["event_id"],
    seed,
    group,
    FROZEN_MEMORY_ENCODER_SLOTS,
    FROZEN_SUBJECT_SUBSPACE_BUDGET,
)
```

Write `slot_count`, `budget`, and `budget_fraction` from the constants. In `subject_subspace_audit.py`, validate the same fields with the constants and pass the declared values to `_valid_mask` and tensor-shape checks.

- [ ] **Step 4: Run focused tests and scan production code**

Run:

```bash
python -m pytest utest/tests/test_prefix_contract.py utest/tests/test_subject_subspace.py -q
rg -n "range\(32\)|slot_count.?[:=].?32|budget_fraction.?[:=].?0\.25|frozen expected='32'" utest/prefix_contract.py utest/subject_subspace.py utest/subject_subspace_audit.py
```

Expected: pytest passes; `rg` returns no production-code match.

- [ ] **Step 5: Commit the independently testable geometry core**

```bash
git add utest/prefix_contract.py utest/subject_subspace.py utest/subject_subspace_audit.py utest/tests/test_prefix_contract.py utest/tests/test_subject_subspace.py
git commit -m "fix: align subject subspace with 64-slot checkpoints"
```

---

### Task 2: Migrate runtime, donor bundle, and harness consumers to 64 slots

**Files:**
- Modify: `utest/subject_reappearance_harness.py`
- Modify: `utest/vistory_donor_bundle.py`
- Modify: `utest/subject_subspace_probe.py`
- Modify: `utest/tests/test_subject_reappearance_harness.py`
- Modify: `utest/tests/test_vistory_donor_bundle.py`

**Interfaces:**
- Consumes: the four frozen constants from Task 1.
- Produces: `_expected_rows(arm: str, masks: Mapping[str, list[int]]) -> list[int]` over `range(64)`.
- Produces: donor payload validation requiring `[64, hidden_dim]` for every layer `0–15`.
- Produces: pre-GPU rejection for any target mask/donor payload geometry mismatch.

- [ ] **Step 1: Convert fixtures to valid 64-slot artifacts and add one explicit 32-slot rejection**

Change valid donor payload fixtures from `torch.zeros((32, hidden_dim))` to `torch.zeros((64, hidden_dim))`, metadata shapes from `[32, hidden_dim]` to `[64, hidden_dim]`, and full-mask fixtures from `range(32)` to `range(64)`.

Rename the bundle regression to `test_encoder_slot_count_is_frozen_to_64` and make its negative cases exact:

```python
@pytest.mark.parametrize(
    ("encoder_layers", "encoder_slots"),
    [("0-14", "64"), ("0-16", "64"), ("0-4,6-15", "64"), ("0-15", "32")],
)
def test_frozen_encoder_geometry_must_match_the_formal_mask_universe(
    tmp_path: Path, encoder_layers: str, encoder_slots: str
) -> None:
    with pytest.raises(ValueError, match="SlotMem donor protocol mismatch"):
        _completed_fixture(
            tmp_path,
            encoder_layers=encoder_layers,
            encoder_slots=encoder_slots,
        )
```

Keep a harness test whose donor tensor is self-consistent at 32 slots and assert rejection occurs before the mocked GPU call.

- [ ] **Step 2: Run the runtime tests and confirm remaining 32-slot assumptions fail**

Run:

```bash
python -m pytest utest/tests/test_subject_reappearance_harness.py utest/tests/test_vistory_donor_bundle.py -q
```

Expected before implementation: complement rows, payload shapes, or frozen geometry assertions fail.

- [ ] **Step 3: Replace runtime hard-codes with the shared contract**

Import `FROZEN_MEMORY_ENCODER_SLOTS` and `FROZEN_SUBJECT_SUBSPACE_BUDGET` where needed. Implement complements as:

```python
universe = range(FROZEN_MEMORY_ENCODER_SLOTS)
return [index for index in universe if index not in selected]
```

Require each validated layer contract to declare `slot_count == 64` and `budget == 8`. In donor-bundle payload validation, require each layer tensor's first dimension to equal `FROZEN_MEMORY_ENCODER_SLOTS`; preserve the existing cross-event hidden-dimension equality check. Update only the probe's self-check fixture geometry; do not change probe scoring semantics.

- [ ] **Step 4: Verify runtime behavior and scan all active production paths**

Run:

```bash
python -m pytest utest/tests/test_subject_reappearance_harness.py utest/tests/test_vistory_donor_bundle.py utest/tests/test_subject_subspace.py -q
rg -n "range\(32\)|slot_count.?[:=].?32|\[32," utest --glob "*.py" --glob "!tests/**"
```

Expected: tests pass; any remaining `32` match must be unrelated to subject-subspace slot geometry and documented in the review message.

- [ ] **Step 5: Commit the runtime migration**

```bash
git add utest/subject_reappearance_harness.py utest/vistory_donor_bundle.py utest/subject_subspace_probe.py utest/tests/test_subject_reappearance_harness.py utest/tests/test_vistory_donor_bundle.py
git commit -m "fix: enforce 64-slot donor runtime geometry"
```

---

### Task 3: Add reproducible replacement-target survey and freeze tooling

**Files:**
- Create: `utest/vistory_target_selection.py`
- Create: `tools/refreeze_vistory_reappearance_events.py`
- Create: `utest/tests/test_vistory_target_selection.py`
- Modify: `utest/vistory_donors.py`
- Modify: `utest/tests/test_vistory_donors.py`

**Interfaces:**
- Produces in `utest.vistory_donors`:
  - `enumerate_official_recurrences(data_root: Path) -> list[dict]`
  - `donor_rejection_reasons(target: Mapping[str, object], donor: Mapping[str, object]) -> list[str]`
- Produces in `utest.vistory_target_selection`:
  - `build_replacement_target_survey(*, data_root: Path, selection_path: Path, output_path: Path) -> dict`
  - `write_replacement_review_template(*, survey_path: Path, output_path: Path) -> dict`
  - `freeze_replacement_selection(*, data_root: Path, selection_path: Path, survey_path: Path, review_path: Path, output_path: Path) -> dict`
- Review schema:

```json
{
  "schema_version": 1,
  "dataset_commit": "92f845531b67e97a67ae04b256ec5d8c020e8341",
  "survey_sha256": "64 lowercase hexadecimal digits",
  "reviewer": "non-empty human name",
  "candidates": [
    {"candidate_id": "stable SHA-256", "female_character": true}
  ]
}
```

- CLI:
  - `survey --data-root PATH --selection PATH --output PATH`
  - `review-template --survey PATH --output PATH`
  - `freeze --data-root PATH --selection PATH --survey PATH --review PATH --output PATH`

- [ ] **Step 1: Write failing recurrence-reuse and target-ranking tests**

Create synthetic stories with these candidates:

```python
ranked = [
    {"candidate_id": "c", "eligible_donor_count": 3, "horizon": 12},
    {"candidate_id": "a", "eligible_donor_count": 4, "horizon": 10},
    {"candidate_id": "b", "eligible_donor_count": 4, "horizon": 11},
]
```

Mark all three female in the strict review and assert candidate `b` wins: donor count beats exact horizon, then distance `1` beats distance `2`. Add a second equal-count/equal-distance test asserting event ID order breaks the tie. Add tests that reject zero-donor candidates, missing review rows, duplicate rows, boolean/non-integer schema confusion, blank reviewer, stale survey hash, changed story/reference bytes, and an attempted Gu Zhenzhen selection.

In `test_vistory_donors.py`, compare old and refactored survey candidate/rejection bytes for a fixed synthetic tree so the donor rules cannot drift during extraction.

- [ ] **Step 2: Run the new tests and confirm missing interfaces fail**

Run:

```bash
python -m pytest utest/tests/test_vistory_target_selection.py utest/tests/test_vistory_donors.py -q
```

Expected before implementation: import errors for `vistory_target_selection` and the two new donor interfaces.

- [ ] **Step 3: Extract recurrence inventory and donor rejection logic once**

Move the existing recurrence construction from `_build_survey` into `enumerate_official_recurrences`. Each record must contain stable identity fields, source/read shots, horizon and bucket, official tag/style, source character count, prompts, official story path/hash, and reference path/hash.

Move only the current structural comparisons into `donor_rejection_reasons`. `_build_survey` must call both new functions and preserve the existing `candidates` and `rejections` schema and sort by `candidate_id`. No condition may be added, removed, or weakened.

- [ ] **Step 4: Implement strict target survey, review template, and freeze**

Generate target event IDs as:

```python
event_id = f"vistory{story_id}_{slug}_s{source_shot}_s{read_shot}"
```

Build `slug` with `unicodedata.normalize("NFKD", character_name)`, ASCII encoding with ignored non-ASCII marks, lowercase conversion, replacement of each non-alphanumeric run with `_`, and trimming leading/trailing `_`. If the result is empty or two candidates collide, raise `ValueError` rather than inventing an unstable ID.

For every realistic-human valid recurrence except all recurrences belonging to the three original target `entity_uid` values, count donors for which `donor_rejection_reasons` returns an empty list. Emit only candidates with `eligible_donor_count > 0`, sorted by candidate ID. Include official character description, source/read prompts, reference path/hash, story path/hash, horizon, and donor IDs.

The review template writes one row per candidate with `female_character: null` and an empty top-level reviewer. Freeze requires exactly one boolean disposition for every survey candidate and a non-empty reviewer. Select from `female_character: true` rows using:

```python
key=lambda row: (
    -row["eligible_donor_count"],
    abs(row["horizon"] - 12),
    row["event_id"],
)
```

Preserve event order as Song Yuchen, replacement, Chen Sihan's Father. Add a top-level `replacement_selection` object binding `original_event_id`, selected event/candidate IDs, donor count, horizon distance, survey/review hashes, reviewer, and dataset commit. Use `write_json_no_clobber` for all outputs.

- [ ] **Step 5: Add the three-command CLI without shell termination logic**

Dispatch argparse subcommands to the three public functions and print sorted indented JSON. Follow `tools/prepare_vistory_donors.py` for direct execution from repository root. The module may return through `main()`; do not add `exit` or `raise SystemExit`.

- [ ] **Step 6: Verify determinism, unchanged donor behavior, and CLI help**

Run:

```bash
python -m pytest utest/tests/test_vistory_target_selection.py utest/tests/test_vistory_donors.py -q
python tools/refreeze_vistory_reappearance_events.py --help
python tools/refreeze_vistory_reappearance_events.py survey --help
```

Expected: all tests pass; both help commands return status 0 and list the exact interfaces above.

- [ ] **Step 7: Commit the selection tooling**

```bash
git add utest/vistory_donors.py utest/vistory_target_selection.py tools/refreeze_vistory_reappearance_events.py utest/tests/test_vistory_donors.py utest/tests/test_vistory_target_selection.py
git commit -m "feat: add reviewed ViStoryBench target refreeze"
```

---

### Human Gate A: Select and inspect the replacement event on the A100 server

This gate runs in the already active `slotmem` environment. It creates CPU-only JSON artifacts and performs no GPU generation.

- [ ] **Step 1: Create a fresh run root and survey candidates**

```bash
cd /data/long_term_data/shixiao/videomem/U-test-vistory-8f0b728

export VM_ROOT=/data/long_term_data/shixiao/videomem
export VISTORY_REV=92f845531b67e97a67ae04b256ec5d8c020e8341
export VISTORY_DATA="$VM_ROOT/datasets/ViStoryBench-full-$VISTORY_REV/ViStoryBench"
export REFREEZE_ROOT="$PWD/runs/vistorybench_reappearance_top8_64_refreeze"

mkdir -p "$REFREEZE_ROOT"

python tools/refreeze_vistory_reappearance_events.py survey \
  --data-root "$VISTORY_DATA" \
  --selection "$PWD/utest/events/vistorybench_reappearance_v1.json" \
  --output "$REFREEZE_ROOT/target_survey.json"

python tools/refreeze_vistory_reappearance_events.py review-template \
  --survey "$REFREEZE_ROOT/target_survey.json" \
  --output "$REFREEZE_ROOT/target_review.json"
```

- [ ] **Step 2: Manually inspect every candidate reference and fill the review**

Set the top-level `reviewer` to the human reviewer's name. Replace every `female_character: null` with `true` or `false` after checking the official character description and reference image named in that row. Do not change candidate IDs, hashes, ordering, or add fields.

- [ ] **Step 3: Freeze and audit the selected replacement**

```bash
python tools/refreeze_vistory_reappearance_events.py freeze \
  --data-root "$VISTORY_DATA" \
  --selection "$PWD/utest/events/vistorybench_reappearance_v1.json" \
  --survey "$REFREEZE_ROOT/target_survey.json" \
  --review "$REFREEZE_ROOT/target_review.json" \
  --output "$REFREEZE_ROOT/vistorybench_reappearance_v1.refrozen.json"

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["REFREEZE_ROOT"])
selection = json.loads(
    (root / "vistorybench_reappearance_v1.refrozen.json").read_text(encoding="utf-8")
)
ids = [row["event_id"] for row in selection["events"]]
assert len(ids) == 3
assert ids[0] == "vistory79_song_yuchen_s2_s8"
assert ids[2] == "vistory16_chen_father_s1_s10"
assert "vistory15_gu_zhenzhen_s8_s20" not in ids
assert selection["replacement_selection"]["selected_event_id"] == ids[1]
print(json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True))
PY
```

Stop here for user review. Return `target_survey.json`, `target_review.json`, and `vistorybench_reappearance_v1.refrozen.json` before modifying the repository authority.

---

### Task 4: Check in the reviewed three-event freeze and remove duplicated target IDs

**Files:**
- Create: `utest/events/vistorybench_replacement_target_survey_v1.json`
- Create: `utest/events/vistorybench_replacement_target_review_v1.json`
- Modify: `utest/events/vistorybench_reappearance_v1.json`
- Modify: `utest/vistory_reappearance.py`
- Modify: `utest/vistory_donors.py`
- Modify: `utest/vistory_donor_bundle.py`
- Modify: `utest/vistory_donor_harness.py`
- Modify: `utest/subject_reappearance_harness.py`
- Modify: `utest/tests/test_vistory_reappearance.py`
- Modify: `utest/tests/test_vistory_donors.py`
- Modify: `utest/tests/test_vistory_donor_harness.py`
- Modify: `utest/tests/test_vistory_donor_bundle.py`
- Modify: `utest/tests/test_subject_reappearance_harness.py`

**Interfaces:**
- Consumes: the three reviewed artifacts from Human Gate A.
- Produces in `utest.vistory_reappearance`: `load_frozen_selection(path: Path | None = None) -> dict` and `frozen_target_event_ids(path: Path | None = None) -> frozenset[str]`.
- Produces in `utest.vistory_target_selection`: `validate_frozen_replacement_provenance(selection: Mapping, survey: Mapping, review: Mapping) -> None`.
- Produces: one checked-in selection whose middle event exactly matches the reviewed generated artifact.

- [ ] **Step 1: Write binding tests before replacing the frozen JSON**

Replace the old three-tuple duplication with invariant checks:

```python
selection = load_frozen_selection()
events = selection["events"]
assert len(events) == 3
assert events[0] == {
    "story_id": "79",
    "event_id": "vistory79_song_yuchen_s2_s8",
    "character_name": "Song Yuchen",
    "source_shot": 2,
    "target_shot": 8,
    "story_sha256": "4298F6EFAA5F2D4A9D69C86E169E0167CE324334F656033A6D692CAFD9484109",
}
assert events[2] == {
    "story_id": "16",
    "event_id": "vistory16_chen_father_s1_s10",
    "character_name": "Chen Sihan's Father",
    "source_shot": 1,
    "target_shot": 10,
    "story_sha256": "6B1AD31634E5DA0108ACD51B16DA2E7F29B202858FCC5D0E556F4BEDB22D005E",
}
assert events[1]["event_id"] == selection["replacement_selection"]["selected_event_id"]
assert events[1]["character_name"] != "Gu Zhenzhen"
assert "vistory15_gu_zhenzhen_s8_s20" not in frozen_target_event_ids()
```

Add a test that loads the checked-in survey and review artifacts, calls `validate_frozen_replacement_provenance`, and recomputes the selected candidate ordering key and hashes before accepting the checked-in JSON.

- [ ] **Step 2: Run the event/donor tests and confirm the old freeze fails**

Run:

```bash
python -m pytest utest/tests/test_vistory_reappearance.py utest/tests/test_vistory_donors.py utest/tests/test_vistory_donor_harness.py utest/tests/test_vistory_donor_bundle.py utest/tests/test_subject_reappearance_harness.py -q
```

Expected before replacing the selection: failures show Gu Zhenzhen remains and the replacement provenance is absent.

- [ ] **Step 3: Install the reviewed generated selection and centralize target IDs**

Copy the reviewed `target_survey.json` and `target_review.json` byte-for-byte into their two frozen audit paths. Replace `utest/events/vistorybench_reappearance_v1.json` byte-for-byte with the reviewed `vistorybench_reappearance_v1.refrozen.json`. Implement `load_frozen_selection` with UTF-8 JSON parsing and strict checks for schema version 1, exact dataset/evaluator commits, seeds `[0, 1, 2]`, exact event count 3, retained events at positions 0 and 2, and absence of all three original identities from the replacement position. Implement `validate_frozen_replacement_provenance` so it verifies the two artifact hashes, exact review coverage, selected ordering key, dataset revision, reviewer, and selected event binding.

Derive `TARGET_EVENT_IDS` and harness `FROZEN_EVENTS` from `load_frozen_selection()` instead of maintaining a second literal set. Keep the public `TARGET_EVENT_IDS` name for current callers.

- [ ] **Step 4: Run focused and full CPU tests**

Run:

```bash
python -m pytest utest/tests/test_vistory_reappearance.py utest/tests/test_vistory_donors.py utest/tests/test_vistory_donor_harness.py utest/tests/test_vistory_donor_bundle.py utest/tests/test_subject_reappearance_harness.py -q
python -m pytest utest/tests -q
```

Expected: all tests pass and the matrix remains exactly nine blocks.

- [ ] **Step 5: Commit the reviewed target freeze**

```bash
git add utest/events/vistorybench_reappearance_v1.json utest/events/vistorybench_replacement_target_survey_v1.json utest/events/vistorybench_replacement_target_review_v1.json utest/vistory_reappearance.py utest/vistory_target_selection.py utest/vistory_donors.py utest/vistory_donor_bundle.py utest/vistory_donor_harness.py utest/subject_reappearance_harness.py utest/tests/test_vistory_reappearance.py utest/tests/test_vistory_target_selection.py utest/tests/test_vistory_donors.py utest/tests/test_vistory_donor_harness.py utest/tests/test_vistory_donor_bundle.py utest/tests/test_subject_reappearance_harness.py
git commit -m "data: refreeze donor-eligible ViStoryBench targets"
```

---

### Task 5: Document and verify the post-freeze donor-review workflow

**Files:**
- Modify: `utest/README.md`
- Test: existing `utest/tests` suite and CLI self-checks.

**Interfaces:**
- Consumes: final checked-in three-event selection and top-8/64 constants.
- Produces: exact server commands that stop before donor approval and GPU execution.
- Produces: a donor review template that uses the existing strict review schema.

- [ ] **Step 1: Replace obsolete 32-slot operational guidance**

Remove the requirement for an unproven 32-slot checkpoint/config and the `SLOTMEM32_BASE_ARGS_YAML` path. Document that `runs/m0a_slotmem_stage2/inference_args.yaml` must normalize to layers `0-15` and slots `64`, and that the formal mask contract is top-8/64 with fraction `0.125`.

Update the README verification assertion to:

```python
assert all(row["slot_count"] == 64 for row in mask["layers"])
assert all(row["budget"] == 8 for row in mask["layers"])
assert all(len(row["semantic_top8"]) == 8 for row in mask["layers"])
```

- [ ] **Step 2: Document fresh target preparation and donor survey commands**

Use a new output root and the active environment:

```bash
cd /data/long_term_data/shixiao/videomem/U-test-vistory-8f0b728

export VM_ROOT=/data/long_term_data/shixiao/videomem
export VISTORY_REV=92f845531b67e97a67ae04b256ec5d8c020e8341
export VISTORY_DATA="$VM_ROOT/datasets/ViStoryBench-full-$VISTORY_REV/ViStoryBench"
export EXP_ROOT="$PWD/runs/vistorybench_reappearance_top8_64_v1"

mkdir -p "$EXP_ROOT/config" "$EXP_ROOT/donors"

python tools/prepare_slotmem_vistory_reappearance.py \
  --data-root "$VISTORY_DATA" \
  --output-root "$EXP_ROOT/inputs"

python tools/prepare_vistory_donors.py survey \
  --data-root "$VISTORY_DATA" \
  --targets "$EXP_ROOT/inputs/manifest.json" \
  --output "$EXP_ROOT/donors/survey.json"
```

Document generation of a review JSON row for every candidate with `approved: false` initially. The human reviewer must fill presentation class, dominant colour, both visibility fields, reviewer name, approval, and tie group under the existing schema. No command may auto-approve a donor.

- [ ] **Step 3: Add the post-review freeze and dry-run commands**

After manual donor review:

```bash
python tools/prepare_vistory_donors.py freeze \
  --data-root "$VISTORY_DATA" \
  --targets "$EXP_ROOT/inputs/manifest.json" \
  --survey "$EXP_ROOT/donors/survey.json" \
  --review "$EXP_ROOT/donors/review.json" \
  --output-root "$EXP_ROOT/donors/selection"
```

The documentation must stop before donor GPU generation until the user has inspected and approved the frozen donor selection. After donor payload generation and bundle freeze, the existing harness dry-run command must report nine blocks, seeds `[0, 1, 2]`, `preflight`/`full` no longer blocked for missing donors, and Q* as record-only availability status.

- [ ] **Step 4: Verify docs, protocol literals, and the complete test suite**

Run:

```bash
rg -n "32-slot|top-8/32|SLOTMEM32|slot_count.*32|budget_fraction.*0\.25" utest/README.md utest --glob "*.py" --glob "*.md"
python -m pytest utest/tests -q
python -m utest.subject_subspace_audit --self-check
python -m utest.eligibility --self-check
```

Expected: no active-protocol 32-slot references remain; historical documents may retain clearly labeled historical values. All tests and both self-checks pass.

- [ ] **Step 5: Commit documentation and final verification changes**

```bash
git add utest/README.md
git commit -m "docs: add top-8/64 refreeze workflow"
```

---

## Review and execution order

Each implementation task uses a fresh implementation subagent. After its commit, dispatch a different fresh reviewer for:

1. specification compliance and scope review;
2. code quality, fail-closed behavior, and test-evidence review.

The primary agent independently inspects the diff and reruns the task's verification command before accepting it. Do not begin Task 4 until Human Gate A artifacts have been reviewed by the user. Do not begin donor GPU generation until the user completes the donor review produced after Task 4.

Final acceptance requires:

```bash
python -m pytest utest/tests -q
python -m utest.subject_subspace_audit --self-check
python -m utest.eligibility --self-check
git diff --check 9127892..HEAD
```

Commit `9127892` is the approved design baseline; the final diff check therefore covers every implementation and review-fix commit created by this plan.
