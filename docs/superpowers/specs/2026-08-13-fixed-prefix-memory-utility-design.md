# Fixed-Prefix SlotMem Memory Utility Validation

Date: 2026-08-13  
Status: approved design, pending implementation

## Objective

Build a causal validation harness on the frozen Wan2.2-I2V-A14B + SlotMem Stage-2
platform. For each eligible NarraStream recurrence event, generate the prefix exactly
once, branch all memory interventions from the same immutable pre-target snapshot, and
measure the decoded marginal effect of the target memory read.

The confirmatory arms are:

- `no_memory`: the target read returns no payload;
- `zero`: payload structure and metadata remain intact, while token values are zero;
- `correct`: the native same-story, same-entity payload passes through unchanged;
- `wrong`: token values come from a pre-frozen, matched, different-entity donor;
- `random`: deterministic Gaussian tokens matched to the correct payload's shape and
  per-layer, per-feature-channel mean and standard deviation.

`scramble` is removed from the confirmatory interface. A row permutation of an encoded
K/V set is permutation-invariant and may only return as an explicitly exploratory arm
after a decoded-output self-check proves that it is not a no-op.

## Non-goals

- Do not modify or train Wan2.2, SlotMem LoRA, Memory Encoder, Memory Writer, or the
  Character-Wise Cross-Attention modules.
- Do not train the utility predictor or router before E0, M1, and M2 pass.
- Do not collapse the decoded outcome vector into an invented scalar utility score.
- Do not call an incomplete metric vector `memory_utility`.
- Do not use `wrong` as a deployment action.

## Architecture

### Event selection

`utest.eligibility` remains the zero-GPU source of recurrence events. Its real-data
report is written to `runs/e0.json`. A separate deterministic split step consumes that
report and writes story-disjoint `dev-metric`, `dev-M2`, validation, train/calibration,
and formal-test manifests according to the frozen research-plan thresholds.

Each event record contains at least `story_id`, `entity_uid`, `character_name`,
`memory_chunk_idx`, `target_chunk_idx`, `gap_chunks`, source JSON, and reference asset.
Formal-test events are not opened by the fixed-prefix development runner.

### Prefix preparation

The harness reuses SlotMem's existing `resume_state` support instead of adding a second
state format. For target chunk `k`, it runs native correct inference over chunks
`0..k-1`, saves the state with `next_chunk_idx=k`, and never regenerates that prefix for
another arm.

The saved state already contains the memory bank, memory metadata, first-appearance
state, final local conditioning frames, and prior runtime records. Target noise is
reconstructed from the frozen `seed_base + k` rule. The target run uses the original
full story JSON and resumes at `k`.

The harness writes `prefix_contract.json` beside the snapshot. It records:

- snapshot SHA256 and byte size;
- code commit and dirty flag;
- `platform.manifest.json` SHA256;
- input JSON, target prompt bytes, reference asset, and negative-prompt hashes;
- target chunk and entity identifiers;
- seed, sampler, timestep, CFG, resolution, and model/injection settings;
- checkpoint paths and hashes from the platform manifest;
- normalized inference arguments whose values must be identical across arms.

Before and after every arm, the snapshot SHA256 must equal the contract. The harness
also compares each arm's runtime contract against the frozen fields. A mismatch fails
the entire event; it is not an exclusion that can be silently retried with a changed
prefix.

### Arm execution

All arms use one read-point wrapper around
`RoleWiseSlotMemoryBank.get_memory_payload`. The wrapper is installed only in the
post-prefix process, so prefix generation is native and arm-independent.

The wrapper records every attempted read with chunk, character, bank, payload presence,
slot count, layer count, tensor shape, dtype, payload hash, and transformed hash. It
separates `payload_layers_seen` from `layers_transformed`, so a native `correct` read is
observable even though its token values are not rewritten.

`no_memory` returns `None` at the reader boundary. It does not globally disable the
writer or alter model construction. This keeps every non-read component identical and
still records the read attempt.

`zero` uses `zeros_like` on every tensor payload. `correct` returns the original payload.
`random` generates values with a generator seeded from the frozen event seed, arm seed,
and stable layer identifier. For each layer and feature channel it uses the correct
tokens' mean and population standard deviation. Zero-variance channels remain at their
mean. Shape, dtype, device, token metadata, query, prompt, and injection settings remain
unchanged.

`wrong` requires a donor manifest entry. The donor must have a different `entity_uid`,
preferably a different story/source, and a compatible feature shape. The pairing table
also freezes coarse class, colour description, character count, source visibility, gap
bucket, slot shape, selection seed, payload path, and payload SHA256. Missing keys or an
incompatible donor fail loudly. There is no fallback to the first payload in a file.

The primary output is the target chunk. The same run may continue through target+1 and
records that chunk as a secondary delayed-effect endpoint, never as part of the primary
endpoint.

### Intervention contract

For `correct`, `zero`, `wrong`, and `random`, the target character must resolve to at
least one non-empty slot in every expected layer. For `no_memory`, the read attempt must
be recorded and the returned payload must be absent. The contract also records writer
update count, finite residual norm, and before/after memory-bank hashes on eligible
Stage-2 update events.

Decoded intervention effectiveness is not inferred from a fired hook. The harness first
reruns one arm from the same snapshot at the same seed to measure the technical repeat
floor. It then computes aligned per-frame L1 distances. `correct` versus `no_memory` and
each transformed arm versus `correct` are reported against that floor. Cross-seed
variation is never used as the technical repeat floor.

An event group fails M1 when:

- a frozen contract field or snapshot hash differs;
- a required payload is absent or addresses the wrong character;
- an arm did not perform its defined operation;
- donor provenance or shape validation fails;
- decoded `correct` and `no_memory` do not diverge above the technical repeat floor;
- a required writer update is non-finite, numerically zero, or leaves the bank hash
  unchanged.

Failed groups remain in a failure ledger with a reason; they cannot become neutral
utility examples.

## Measurement and utility output

The decoded evaluator writes one normalized record per
`(story, event, arm, seed, endpoint)` with:

`C_id`, `A_prompt`, `Q_bg`, Motion Smoothness, Dynamic Degree, `Q_flicker`,
`Q_boundary`, `Q_anatomy`, and `Q_non-target`.

The implementations, versions, masks, directions, missing-value rules, repeatability
margins, Dynamic Degree absolute floor, and human anchors come from the frozen metric
card. `Q_anatomy` may be N/A only under the card's visibility rule.

`utest.memory_utility` computes arm-minus-`no_memory` decoded deltas, the
helpful/neutral/harmful label, both the all-eligible and independently Gate-A-qualified
populations, and story-cluster intervals. It reports `memory_utility` only after M2 has
established that `correct` separates from matched `wrong`; otherwise it reports
`memory_presence_effect`. If any required metric or frozen margin is missing, the result
is `measurement_incomplete` and has no utility label.

The initial five-arm validation reports every arm. The later primary decoded census uses
`correct` versus `no_memory`; `wrong`, `zero`, and `random` are mechanism/content
controls.

## E0 and M0 stage runner

A remote Bash entry point accepts explicit environment variables for the real
NarraStream input root, Wan2.2 directory, SlotMem checkpoint root, output root, conda
environment, and optional official benchmark repositories/API credentials. It never
guesses a dataset path.

The runner performs:

1. E0 on the real converted NarraStream inputs and writes the full inclusion/exclusion
   report to `runs/e0.json`.
2. Platform provenance generation, including repository commit/dirty state, checkpoint
   SHA256s, Wan2.2 file manifest, runtime versions, GPU identity, and command arguments.
3. M0a on the unmodified official seven-chunk sample. It records all seven chunk
   artifacts, non-empty native memory reads, writer/read statistics, total and per-chunk
   wall time, and peak allocated/reserved VRAM.
4. M0a validation. `completed_chunk_count` must equal seven, every expected video and
   metadata file must exist, checkpoint domains must be loaded, and at least one
   post-first-appearance memory read must be non-empty.
5. M0b when the paper-comparable NarraStream benchmark inputs, preprocessing, checkpoint,
   and evaluator are all explicitly supplied. It compares Subject Consistency with
   `0.8771`, passing when the absolute difference is at most `0.02` or the bootstrap 95%
   interval covers `0.8771`.
6. If any comparability prerequisite is absent, it writes an M0b report with status
   `non-comparable`, enumerates the missing prerequisites, and makes no reproduction
   claim. An official single sample can never substitute for M0b.

The 14B dual-expert M0a run is a remote-server test. The local 6 GB GPU is used only for
zero-GPU/unit checks and cannot be presented as M0 evidence.

## Interfaces and artifacts

The implementation will expose commands equivalent to:

```text
python -m utest.event_harness prepare-prefix --event "$EVENT_JSON" --output "$PREFIX_DIR" -- $SLOTMEM_ARGS
python -m utest.event_harness run-arms --prefix "$PREFIX_DIR" --donor-manifest "$DONOR_MANIFEST" --arms no_memory,zero,correct,wrong,random
python -m utest.event_harness validate --event-run "$EVENT_RUN_DIR"
bash scripts/run_slotmem_stage_gates.sh
```

Each event directory contains the immutable snapshot and contract, one directory and
audit report per arm, a technical-repeat report, an intervention-contract report, a
failure ledger entry when applicable, normalized metric records, and the final census.

The stage runner writes machine-readable `e0.json`, `platform.manifest.json`,
`m0a_report.json`, `m0b_report.json`, commands, logs, and a summary that distinguishes
`passed`, `failed`, `blocked`, and `non-comparable`.

## Error handling

- Existing outputs are never silently overwritten; the caller must choose a fresh run
  directory or explicitly resume a matching run.
- Snapshot and donor files are resolved to absolute paths and hash-checked before use.
- Partial arm output does not count as a completed group.
- CUDA OOM, missing weights, absent datasets, missing evaluator dependencies, and API
  failures are recorded separately from scientific null results.
- Formal-test access requires an explicit frozen manifest and is refused by development
  commands.
- A failed M0a stops arm generation. E0 below 128 permits M0/M1 deployment work but
  blocks single-source controller label production.

## Verification

Unit tests cover:

- correct/zero/random/wrong tensor semantics for flat and layer-wise payloads;
- deterministic random output, moment matching, zero-variance channels, dtype and shape;
- strict donor identity, manifest, hash, and shape validation;
- `no_memory` read attempts and required addressing hits;
- snapshot hash immutability and cross-arm contract comparison;
- technical-repeat-floor and decoded-divergence decisions;
- incomplete metric refusal, Gate A seed separation, population reporting, and utility
  naming;
- M0a and M0b report validation with synthetic artifact trees.

The remote quick test runs one real recurrence event through all five arms from one
snapshot. The remote full test runs real E0 and complete seven-chunk M0a before any M2
budget is released. Completion claims require fresh unit-test output and machine-readable
remote reports; logs or hook counts alone are insufficient.
