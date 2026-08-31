# SlotMem Base-Inference Layerwise Memory Repair Design

Status: approved in conversation; pending written-spec review

## Goal

Repair the existing SlotMem base inference path so one normal DiT forward captures raw
features from configured MemoryEncoder layers `0-15`, selects the existing subject-token
positions without changing their semantics, encodes each layer independently into 64
slots, stores a complete layerwise payload, and injects layer `i` only into DiT layer `i`.

The repair must preserve the current `two_role_diff` character-token selection and shared
query-mask behavior. It must not copy one layer across all layers, add an extra DiT
forward, or reinterpret an existing flat donor payload as layerwise memory.

## Root Cause

The base inference loop currently registers one `AttentionOutputFeatureTap` at
`sparse_role_memory_layer_idx`. That tensor is used both to build the shared query payload
and to extract online memory. Although runtime arguments declare layerwise binding,
MemoryEncoder layers `0-15`, and 64 slots, the online-memory writer receives one flat
tensor. It therefore takes the flat branch and encodes only `layer_idx=0`, producing one
`[64, hidden_dim]` tensor.

The donor harness currently validates the payload file, keys, reported shapes, audit, and
hashes, but does not require a complete layerwise geometry. It can therefore publish a
completion record for a flat payload. The donor bundle has the stronger check and rejects
the artifact later.

The observed Song Yuchen donor demonstrates the failure exactly:

- configured layers: `0-15`;
- configured slots: `64`;
- binding mode: `layerwise`;
- stored payload: one flat `Tensor[64, 5120]`;
- audit `payload_layers_seen`: `1`.

## Architecture

### One forward, two consumers

The normal base-inference forward will register feature taps for every configured
MemoryEncoder layer. After the forward, it will separate the captured features into two
consumers:

1. **Query consumer:** retain the existing shared query semantics by selecting only the
   tensor at `sparse_role_memory_layer_idx` when building the next-step query payload.
2. **Memory consumer:** package every configured layer tensor as a layerwise container and
   pass it to the existing online-memory extraction path.

Both consumers use the same forward result. No additional DiT call is introduced.

### Token selection

The existing attention maps, `two_role_diff` suppression, selected token indices, limits,
and ordering remain unchanged. `_extract_memory_from_step_maps` already applies one frozen
set of selected indices to every provided layer's own raw feature tensor and returns a
layerwise token container with matching layerwise metadata. The repair reuses that code.

### Encoding and storage

The existing layerwise branch of `_stage2_prepare_payload_for_bank` remains authoritative.
For every layer `i`:

1. take only raw features captured from DiT layer `i`;
2. call the existing layer-specific SlotMem MemoryEncoder with `layer_idx=i`;
3. produce exactly 64 finite floating-point slots;
4. preserve the result under string key `str(i)`.

The stored token payload is:

```text
{
  "__layerwise__": true,
  "layers": {
    "0": Tensor[64, D],
    ...,
    "15": Tensor[64, D]
  }
}
```

The existing RoleWiseSlotMemoryBank layerwise storage and per-layer injection path remain
unchanged. DiT layer `i` continues to select memory layer `i`.

## Fail-Closed Rules

- Derive expected capture layers from the active configured MemoryEncoder layer set.
- When layerwise MemoryEncoder mode is active, reject a capture missing any expected
  layer, containing an extra layer, or returning a non-2D tensor.
- Never fall back from an incomplete layerwise capture to the flat path.
- Keep the shared query payload based on `sparse_role_memory_layer_idx`; reject the capture
  if that required query layer is absent.
- Do not synthesize missing layers by copying, tiling, or re-encoding another layer's raw
  features.
- At donor completion, require exactly the declared single-bank target-character payload
  key and validate the complete layerwise geometry.
- Require exact string layer keys, layers `0-15`, 64 rows per layer, finite floating-point
  2D tensors, and one shared hidden dimension.
- Preserve payload, audit, command, repository, checkpoint, and runtime hashes.
- Preserve no-clobber behavior. Existing failed Song v1 artifacts remain immutable.

## Shared Validation

Introduce one small payload-geometry validator at the existing input-contract boundary.
It accepts a payload plus expected layer IDs and slot count, validates the complete
layerwise tensor contract, and returns canonical shapes. The donor harness and donor
bundle both call it so completion and publication enforce the same invariant.

This helper is limited to structural tensor validation. Event identity, donor identity,
scope, provenance, and audit checks remain in their existing owners.

## Interfaces and Compatibility

- No CLI argument changes.
- No change to `role_token_selection_mode`, query mask, target seed, donor seed, memory
  bank mode, or arm definitions.
- No change to formal target authority or Song-only exploratory scope.
- Valid existing 16-layer/64-slot artifacts remain compatible.
- Flat or incomplete artifacts that previously received a completion record become
  invalid under the strengthened completion validator.
- The repair applies to the shared base inference runtime, not only Song Yuchen.

## Testing Strategy

Use vertical TDD slices:

1. Add a regression proving base inference separates one shared query feature tensor from
   a complete layerwise memory capture produced by the same forward.
2. Add capture-boundary cases for missing configured layers and a missing query layer.
3. Add donor-completion cases rejecting a flat tensor, empty layers, missing/extra layers,
   63 slots, non-floating or non-finite tensors, and inconsistent hidden dimensions.
4. Prove a complete layerwise 0-15/64 payload passes both donor completion and bundle
   publication.
5. Preserve existing teacher-forced single-layer query tests to ensure Q* query semantics
   do not drift.

Run focused inference-hotpath, donor-harness, donor-bundle, and subject-reappearance tests,
then the full CPU suite and existing subject-subspace self-checks. An independent agent
reviews the final diff before acceptance.

## Operational Recovery

The current `vistorybench_song_yuchen_exploratory_v1` donor is retained as failed evidence.
After the repaired code is committed, pushed, and pulled onto the A100 host, build a clean
donor run under `vistorybench_song_yuchen_exploratory_v2`. Reuse the immutable reviewed
selection, regenerate the donor prefix and dump, verify completion now reports all 16
layers with 64 slots, then freeze the donor bundle and continue the Song seed-0 target
pilot.

## Acceptance Criteria

- Base inference captures raw tensors for every configured layer in one normal forward.
- Query payload behavior and selected token indices are unchanged.
- Stored donor memory contains exactly layers `0-15`, each shaped `[64, D]`.
- Layer `i` is encoded and injected only through layer `i`.
- Invalid flat or incomplete payloads cannot receive or reuse a valid completion record.
- Complete payloads pass donor completion, donor bundle publication, and target-harness
  donor compatibility checks.
- No additional model forward, dependency, or Song-specific runtime branch is added.
