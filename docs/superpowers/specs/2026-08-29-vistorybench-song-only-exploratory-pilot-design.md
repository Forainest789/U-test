# ViStoryBench Song-Only Exploratory Pilot Design

Status: approved design; awaiting written-spec review

## Goal

Generate one controlled visual pilot for the frozen Song Yuchen reappearance event using
seed `0` and the four existing preflight arms:

- `full_correct`
- `no_memory`
- `zero_path`
- `wrong_subject`

This pilot answers whether the current SlotMem subject-memory path produces a visible,
identity-specific effect before spending GPU time on a replacement three-event primary
benchmark.

## Non-goals

- Do not modify the checked-in three-event target authority.
- Do not approve a visually mismatched donor for Bella or Chen Sihan's Father.
- Do not relax presentation-class, dominant-colour, visibility, geometry, or provenance
  checks.
- Do not treat the pilot as a three-event primary result or include it in formal aggregate
  statistics.
- Do not use Q* for selection, ranking, masking, injection, or pass/fail decisions.

## Frozen Human-Gate Decision

The full donor survey remains the immutable source artifact. The exploratory review covers
exactly the three eligible Song Yuchen rows and records a disposition for every one:

| Donor | Candidate ID | Presentation | Dominant colour | Source visible | Read visible | Decision |
|---|---|---|---|---|---|---|
| Chen Sihan's Father | `2e08901266442994503aa6c94b30cb8c75617d70266647a0a70425cc6dcfbc55` | `male` | `brown` | `true` | `true` | reject |
| Mr. Fogg | `6cde40c6ee47af0ec9fb2b0596a0088316c4510eea0f8a203dd40544e220fa13` | `male` | `blue-grey` | `true` | `true` | reject |
| Colonel Cromarty | `ad04290d1a7ddd4691b8337c3a71afca0e8daee38a706be8c5a40f83bb725938` | `male` | `black` | `true` | `true` | approve |

Song Yuchen is classified as `male / black`. Colonel Cromarty is the only candidate that
matches both frozen visual attributes while remaining a clearly different identity.

## Architecture

Reuse the existing donor selection, donor generation, bundle, and target harnesses. Add an
explicit `exploratory_single_event` protocol scope at the three boundaries that currently
require exactly the frozen three targets:

1. donor selection freeze;
2. donor generation run manifest and completion validation;
3. validated donor-map bundle publication.

The formal default remains unchanged and continues to require exactly all three frozen
target event IDs. Single-event behavior is reachable only through an explicit CLI option
that names one event from the frozen authority. The produced artifacts record the scope
and selected event ID so exploratory artifacts cannot be mistaken for formal ones.

The target harness retains its frozen nine-block manifest. It already accepts a partial
donor map and already supports selecting one `--event-id` and one `--seed` during execution.
Bella and Chen Sihan's Father therefore remain visible as `blocked_missing_donor`; only the
Song Yuchen seed-0 block is executed.

## Data Flow

1. Read the complete checked-in target input manifest and the complete 26-candidate donor
   survey.
2. Create a new Song-only review artifact containing exactly the three Song candidate rows.
   It remains bound to the raw SHA-256 of the complete survey.
3. Freeze one reviewed donor selection with:
   - `protocol_scope: exploratory_single_event`;
   - `target_event_ids: ["vistory79_song_yuchen_s2_s8"]`;
   - the existing dataset, target-input, survey, review, reference, and story hashes.
4. Build and execute one donor job for Colonel Cromarty using donor seed `0` and the frozen
   top-8/64 geometry.
5. Freeze a partial donor map containing only the Song entry and the exploratory scope.
6. Build the normal nine-block target dry-run using that partial map.
7. Execute only `vistory79_song_yuchen_s2_s8 / seed_0` through prefix, source qualification,
   probe, and four-arm preflight.
8. Stop after the four decoded outputs and their validation records. Do not aggregate the
   result as a primary benchmark.

## Interfaces

### Donor freeze

Add an optional CLI argument:

```text
prepare_vistory_donors.py freeze --exploratory-target-event-id EVENT_ID
```

When omitted, existing exact-three behavior is byte-for-byte compatible. When supplied,
the event ID must occur in the checked-in frozen authority and the review must cover
exactly all survey candidates for that event, with exactly one approved donor after normal
strict review validation.

### Donor harness

The selection artifact is the authority for expected event IDs. Formal selections must
contain the frozen three; exploratory selections must contain exactly the declared single
event. Job counts, completion validation, and CLI success derive from that validated set
instead of a numeric literal.

### Donor bundle

The bundle continues to validate the complete frozen target input manifest. It publishes
only the event IDs authorized by the validated donor selection and records the same scope.
Formal bundle behavior remains exact-three.

### Target execution

No new target-harness mode is needed. Use the existing partial-donor behavior and existing
event/seed selectors. The run manifest remains nine blocks, but only Song seed `0` may be
executed for this pilot.

## Fail-Closed Rules

- Reject exploratory mode without exactly one explicit frozen event ID.
- Reject a non-frozen or duplicated target event ID.
- Reject review rows for other targets, missing Song rows, duplicate rows, blank reviewer,
  stale survey hash, changed references, or more/less than one approved Song donor.
- Reject any selection, run, completion record, or bundle whose protocol scope or selected
  event IDs disagree.
- Reject legacy 32-slot payloads and any layer set other than `0–15`.
- Preserve no-clobber output behavior and use a fresh Song-only run root.
- Do not add `exit` or `SystemExit` control flow.
- Do not let Q* enter any decision path.

## Verification

Add focused regressions proving:

- the formal default still rejects one- or two-event selections;
- exploratory mode accepts exactly Song Yuchen and no other implicit subset;
- the review must cover exactly Song's three candidates and select Colonel Cromarty once;
- scope/event mismatches fail at freeze, donor run, completion validation, and bundle
  publication;
- the partial donor map makes only the Song blocks donor-ready;
- event/seed selection executes only Song seed `0`;
- the preflight arm set is exactly the existing four arms and all reuse one prefix snapshot
  and target seed;
- Q* remains descriptive/unavailable and cannot change execution or verdicts.

Run the focused suites for donor selection, donor harness, donor bundle, and subject
reappearance harness, then the complete CPU suite and the existing subject-subspace audit
and eligibility self-checks. A different fresh subagent performs independent review before
the implementation is accepted.

## Execution Stop

The first GPU run ends after the four Song seed-0 preflight outputs are validated and made
available for visual inspection. Expansion to three seeds, eight arms, or a refrozen
three-event primary requires a separate explicit decision based on that inspection.
