# ViStoryBench Event Refreeze and Top-8/64 Protocol Design

Date: 2026-08-28
Status: approved for specification
Scope: replace the ineligible Gu Zhenzhen target event, migrate the frozen subject-subspace geometry from top-8/32 to top-8/64, then regenerate and manually review donor candidates.

## Objective

Keep the existing subject-reappearance experiment comparable while making every target event donor-eligible under the already frozen structural matching rules and aligning the probe/injection protocol with the available 64-slot SlotMem checkpoints.

The final target set contains exactly three events:

1. `vistory79_song_yuchen_s2_s8` (Song Yuchen), unchanged.
2. `vistory16_chen_father_s1_s10` (Chen Sihan's Father), unchanged.
3. One deterministic replacement for `vistory15_gu_zhenzhen_s8_s20`.

The replacement must be a realistic-human female character with a real source–absence–first-reappearance interval and at least one eligible donor under the unchanged donor matcher.

## Non-goals

- Do not relax or reinterpret donor matching conditions.
- Do not make Q* select slots or affect generation; Q* remains record-only.
- Do not support arbitrary checkpoint geometries. This protocol is frozen to the existing 64-slot checkpoints.
- Do not automatically approve a donor or infer visual compatibility from text.
- Do not change the three seeds `[0, 1, 2]` or the two retained events.

## Replacement-event selection

### Candidate enumeration

Scan all 80 stories from dataset revision `92f845531b67e97a67ae04b256ec5d8c020e8341`. For every official character tagged `realistic_human`, enumerate consecutive appearances `(source_shot, target_shot)`. A pair is a valid reappearance interval only when the character appears exactly once where present in the inclusive interval, is present in both endpoints, and is absent from every intermediate shot.

Exclude the two retained target events and the original Gu Zhenzhen event from replacement eligibility. Existing reference-image, path-containment, story-completeness, identity-ambiguity, and raw-story SHA-256 checks remain fail-closed.

### Donor support

For each valid target candidate, count eligible donors by applying the current donor matcher without modification. Its structural conditions remain:

- donor comes from another story and is not the target identity;
- exact official character tag match;
- exact style-class match;
- exact source-shot character-count match;
- equal existing horizon bucket (`5–7`, `8–10`, or `11–13`);
- unambiguous identity, valid absence interval, and present official reference.

A replacement candidate with zero eligible donors is rejected.

### Female-character review and deterministic choice

ViStoryBench does not expose a trusted structured gender field. The pipeline therefore emits a replacement-candidate review artifact containing the official character description, reference path and hash, interval prompts, eligible donor count, horizon, and stable event ID. A reviewer explicitly records whether the candidate is a female character; the implementation must not infer this from the name.

Among reviewer-confirmed female candidates, select exactly one by this total ordering:

1. eligible donor count, descending;
2. absolute distance from the original Gu Zhenzhen horizon `12`, ascending;
3. event ID, ascending.

The frozen selection records the complete ranking inputs, review provenance, official story SHA-256, dataset revision, and selected event specification. The same dataset and review artifact must reproduce the same selected event byte-for-byte.

## Top-8/64 protocol

The protocol geometry is:

- memory encoder layers: `0–15`;
- memory slots per layer: `64`;
- subject-subspace intervention budget: `8` slots;
- budget fraction: `8 / 64 = 0.125`.

One shared frozen slot-count constant and one shared top-k budget constant define this geometry. Production code must not retain independent 32-slot ranges or checks.

Semantic ranking, reference ranking, random controls, masks, complements, donor tensors, probe artifacts, bundle manifests, and run manifests use the slot universe `0–63`. Semantic and reference rankings must each be a permutation of all 64 slots. A top-8 selection must contain eight distinct in-range indices. Random controls remain deterministic for the existing seed contract and also select eight distinct indices from `0–63`.

Artifacts declaring or encoding 32 slots are incompatible with this protocol and must fail before GPU inference. The implementation must not pad, duplicate, truncate, or remap a 32-slot artifact into 64 slots.

Q* records may reference the generated run and probe artifacts, but Q* values cannot enter slot ranking, mask construction, donor selection, or injection.

## Freeze and review flow

The workflow is deliberately split at human-review boundaries:

1. Validate the pinned complete ViStoryBench tree.
2. Survey replacement target candidates and their structural donor counts.
3. Generate the replacement-target review artifact.
4. A reviewer confirms female-character status for every candidate considered by the ranking.
5. Deterministically select the replacement and freeze the new three-event specification.
6. Regenerate the prepared target inputs from that specification.
7. Rerun the unchanged donor survey for all three targets.
8. Generate a donor-review template containing every structurally eligible donor.
9. A reviewer evaluates presentation class, dominant colour, source visibility, and read-check visibility using the existing review schema.
10. Freeze exactly one approved donor per target under the existing tie-group rules.

No command automatically marks a target as female or a donor as approved. Survey and template generation may overwrite only a newly chosen output path; frozen outputs retain the repository's no-clobber behavior.

## Components and ownership

- The frozen event JSON remains the authority for the three target identities and intervals.
- Target-candidate discovery owns enumeration, structural donor-count calculation, deterministic ranking, and its audit artifact.
- The existing donor module remains the sole authority for donor eligibility; the replacement survey reuses that logic instead of duplicating the conditions.
- The prefix/geometry contract owns the fixed 64-slot and top-8 invariants.
- Subject-subspace, audit, donor-bundle, and run-harness modules consume the shared geometry contract.
- Human review artifacts own judgments not present in official structured metadata.

## Failure behavior

The workflow fails closed when:

- the official 80-story tree or any frozen hash is incomplete or mismatched;
- the retained events change or the frozen target count is not three;
- Gu Zhenzhen remains in the refrozen target set;
- no reviewer-confirmed female replacement has an eligible donor;
- ranking inputs are missing, duplicated, stale, or inconsistent with the survey;
- any probe, mask, donor bundle, checkpoint declaration, or run manifest is not exactly 64-slot/top-8 compatible;
- any donor candidate was not manually dispositioned before donor-map freezing;
- a target has zero approved donors or ambiguous approvals outside the existing tie-group rule.

Errors identify the offending event, artifact, or geometry and return normally through Python exceptions or nonzero command status. New shell helpers must not contain `exit`, because the commands are intended to be run in the user's active Conda terminal.

## Verification

The smallest sufficient verification set includes:

- retained-event and exact-three-event assertions;
- a negative assertion that the Gu Zhenzhen event is absent;
- synthetic candidate tests covering absence validity, zero-donor rejection, reviewer-confirmed female filtering, and all ranking tie-breakers;
- a reproducibility test proving identical survey plus review bytes yield identical frozen selection bytes;
- donor-matcher regression tests proving the structural conditions did not change;
- 64-slot positive tests for rankings, masks, complements, random controls, probe artifacts, donor tensors, bundles, and run manifests;
- 32-slot negative tests proving legacy artifacts are rejected rather than adapted;
- an assertion that Q* is not read by selection or injection paths;
- the full repository test suite after focused tests pass.

The first server-side run stops after producing the new three-event inputs and donor-review template. GPU donor generation and the full 3-event × 3-seed experiment remain gated on the user's manual donor review and frozen donor map.
