# ViStoryBench Donor and Source-Semantic Producer Design

**Status:** frozen design; implementation plan approved
**Parent experiment:** `vistorybench_subject_reappearance_v1`  
**Dataset commit:** `92f845531b67e97a67ae04b256ec5d8c020e8341`  
**Evaluator commit:** `b44ec9108668cc2bcc8c5280886b235e9fb8bea9`

## Goal

Supply the two missing, pre-target inputs required by the existing nine-block
subject-reappearance harness:

1. one validated matched-wrong SlotMem donor per evaluated event, shared across
   target seeds `0,1,2`; and
2. one deterministic, source-only `semantic_scores.json` per event×seed capture.

This design does not alter the three evaluated events, the eight causal arms,
the primary `semantic_top8` budget, or the rule that Q* is descriptive and
`not_available` without an independent teacher.

## Frozen decisions

- Evaluated events remain exactly stories `79`, `15`, and `16` from the parent
  experiment.
- Donors may come from other official ViStoryBench stories, but never from the
  target story or target identity.
- Each target event receives one fixed donor payload generated with donor seed
  `0`; that payload is reused for all three target seeds.
- `semantic_top8` is the primary subject-subspace mask.
- Primary semantic scores use only the source capture's role metadata and
  MemoryEncoder attention. They do not use target prompts, target frames,
  decoded arms, CIDS, Q*, or target losses.
- Visual counterfactual and reference-alignment rankings remain robustness
  methods and cannot replace the primary mask.
- No new runtime dependency or learned probe is introduced.

## Component 1: frozen donor selection

### Inputs

- Official ViStoryBench `story.json` files and character reference images.
- The three frozen target event manifests.
- A reviewed donor annotation file containing only pre-generation information.

### Matching contract

Each donor must satisfy all hard constraints:

- `donor_story_id != target_story_id`;
- `donor_entity_uid != target_entity_uid`;
- the donor appears visibly in its source shot and in its selected read-check
  shot;
- official character `tag` and realistic/non-realistic story style match the
  target;
- source-shot character count matches;
- recurrence horizon is in the same frozen bucket: `5-7`, `8-10`, or `11-13`;
- the reviewed annotation records presentation/gender class and dominant
  clothing/body colour, and both match the target;
- the selected source and read-check interval contains no ambiguous duplicate
  identity.

The selector rejects zero or multiple reviewed matches. A deterministic
`selection_seed=0` breaks only an explicitly declared tie; it never searches
generated outcomes. The frozen selection records every candidate considered,
the rejection reason, official input hashes, and the selected pair.

### Donor production

For each of the three selected donors:

1. convert the official interval into the existing event contract;
2. generate an immutable prefix with donor seed `0`;
3. run `utest.event_harness dump-donor` at the read-check shot;
4. require a target read hit and an effective intervention;
5. write `slotmem_donor_payload_v2`, `donor_payload_info.json`, and the existing
   matched-pair manifest fields; and
6. build one donor-map event entry that is reused by target seeds `0,1,2`.

The formal nine-block run manifest is created only after all three bundles pass
`validate_donor_bundle`. Donor-generation outputs are never evaluated as target
arms.

## Component 2: source-only semantic-score producer

### Inputs

For one event×seed block the producer reads only:

- the frozen block event JSON;
- `source_capture.pt` written at source chunk `0`;
- the derived source story JSON and official source reference hashes already
  bound by the capture; and
- the checked-out code and model identities already present in capture
  provenance.

It refuses any key beginning with `target_`, `qstar`, `cids`, or `decoded_` and
sets `target_evidence_read=false`.

### Token group scores

For raw source token `i`, define from existing `raw_token_meta`:

```text
is_target(i) = 1 when char_id equals the frozen subject, otherwise 0
inside(i)    = 1 only when is_target(i)=1 and inside_box=true
centre(i)    = inside(i) * exp(-tau_local(i)^2 / 2)
is_other(i)  = 1 for a non-empty, non-target char_id, otherwise 0
outside(i)   = 1 when is_target(i)=1 and inside_box=false, otherwise 0
```

The four vectors required by the existing score contract are:

```text
identity_name     = is_target
stable_attributes = centre
other_characters  = is_other
action_scene      = outside
```

These vectors are projected through the already-captured MemoryEncoder
attention by the existing `semantic_slot_scores` implementation. Member-layer
scores are averaged within the frozen groups `0-4`, `5-10`, and `11-15`; the
top eight of 32 slots becomes `semantic_top8`.

The producer fails closed if any required layer is absent, token metadata and
attention disagree, no target token is present, no target token lies inside a
valid box, `tau_local` is missing/non-finite, or any provenance hash differs.
It never substitutes all-ones, random, prompt-only, or target-derived scores.

### Output

The output uses the existing `build_semantic_score_artifact` schema and adds no
parallel format. It binds:

- event ID, subject, seed, bank, and layer;
- source-capture file and canonical hashes;
- the exact source semantic vocabulary;
- formula name/version and source-only marker;
- code/model/source hashes; and
- four finite vectors with exactly the raw-token length.

Publishing is atomic and no-clobber. Identical inputs reproduce identical JSON
bytes; changed inputs require a fresh output path.

## End-to-end execution order

1. Download or synchronize the official donor candidate stories.
2. Produce and review the pre-generation donor annotation/selection manifest.
3. Generate and validate the three fixed donor bundles.
4. Build a fresh formal 3-event×3-seed run manifest with the frozen donor map.
5. Generate one prefix per formal block, producing `source_capture.pt`.
6. Produce and validate that block's `semantic_scores.json`.
7. Freeze `subject_subspace_manifest.json` with primary `semantic_top8`.
8. Run four-arm preflight, then the full eight arms only after preflight passes.
9. Export CIDS inputs and aggregate event-clustered results; record Q* as
   `not_available` unless an independent teacher is later frozen.

## Error handling and provenance

- Official files are hash-checked before parsing.
- Same-story, same-identity, mismatched-style, mismatched-gap, and ambiguous
  donors are rejected before GPU work.
- Missing or invalid donor bundles keep preflight/full blocked.
- Missing or invalid source boxes keep probe/full blocked; no fallback ranking
  is emitted.
- Existing output artifacts are never overwritten.
- Every formal artifact records the repository commit and dirty flag; ignored
  editable-install metadata is not experimental state.
- Preliminary no-donor dry-run outputs cannot be relabelled as formal outputs.

## Verification

Minimum automated checks:

- deterministic donor selection and `selection_seed=0` tie behavior;
- rejection of same-story/same-identity and every matching-field mismatch;
- exactly three selected donors and one donor per target event;
- donor-map reuse across all three target seeds;
- exact token-vector formula on a small fixture;
- deterministic centre weighting and finite-value checks;
- fail-closed behavior for missing boxes, missing layers, mismatched token counts,
  target-derived keys, tampered hashes, and existing outputs;
- compatibility with `validate_semantic_scores`, `freeze_subject_subspace`, and
  `validate_donor_bundle`;
- dry-run proof of nine blocks, available preflight/full commands, primary
  `semantic_top8`, and Q* `not_available`; and
- the full zero-GPU regression suite.

## Explicit non-goals

- No target-conditioned slot search or alpha sweep.
- No trained subject classifier or learned probe.
- No external CLIP/DINO/face model in the primary estimator.
- No claim that the three-event pilot estimates all 80 ViStoryBench stories.
- No claim that donor matching proves visual identity by itself; specificity is
  established only by decoded wrong-subject intervention relative to the other
  fixed arms.
