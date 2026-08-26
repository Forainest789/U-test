# SlotMem Fusion Verification Design

## Objective

Determine whether SlotMem's identity-memory residual genuinely improves the frozen
conditional denoising target, merely has excessive strength, or conflicts with the
current host representation. The experiment must reuse the existing SlotMem model,
teacher-forced identity probe, sparse-memory layer scales, and token diagnostics. It
must not add a router, change weights, decode videos, or emit identity-token labels.

The result is a mechanism diagnosis for a frozen event, noise seed, timestep, and
layer group. `representation_competition_candidate` is deliberately weaker than a
proof of representation competition; proving that mechanism would require a later
host-identity ablation.

## Scope and Execution Boundary

Add an opt-in `--verify-fusion` mode to `utest.identity_token_probe`.

- The normal smoke, S0/S1, S2, Q*, and S3 paths remain unchanged.
- Verification mode runs the existing 25-arm S0/S1 screen.
- Verification mode never enters S2 or S3, even if content causality passes.
- V0 adds no model calls.
- V1 runs at most eight alpha arms for each of at most two selected cells.
- The hard verification-mode measured-arm ceiling is 41: 25 S0/S1 plus 16 V1.
- CUDA warm-up and truncated semantic prepasses remain reported separately from
  measured arms.
- The existing conditional-only identity path remains active, so V1 performs no
  unconditional DiT calls.

This is a diagnostic extension of the existing probe, not a new model component.

## V0: Prediction-Space Error Decomposition

For every S0/S1 cell with `no_memory`, `correct`, `wrong`, and where available
`zero`, keep the conditional predictions in process only long enough to compute the
following scalars. Raw predictions are not written to disk.

Let

\[
v_0 = v_{\text{no-memory}},\quad e_0=v_0-y,\quad
\Delta_a=v_a-v_0
\]

for arm \(a\). Report

\[
A_a=2\operatorname{mean}(e_0\Delta_a),\qquad
B_a=\operatorname{mean}(\Delta_a^2)
\]

and verify

\[
L_a-L_0=A_a+B_a.
\]

The per-arm record contains:

- `loss_delta_from_no_memory`
- `directional_alignment` (the signed linear term \(A_a\))
- `delta_energy` (the non-negative quadratic term \(B_a\))
- `decomposition_residual`
- `predicted_optimal_alpha`
- `predicted_optimal_gain`

If \(B_a>0\), the locally linear residual model predicts

\[
\alpha^*=\operatorname{clip}(-A_a/(2B_a),0,1)
\]

and

\[
G^*=-(\alpha^*A_a+(\alpha^*)^2B_a).
\]

`predicted_optimal_alpha` and `predicted_optimal_gain` are diagnostic predictions,
not observed effects. They are null when the quadratic term is zero or non-finite.

The decomposition must reconstruct the observed loss difference within

```text
max(1e-8, 1e-5 * max(abs(loss_delta), abs(A) + abs(B), 1e-12))
```

or the run fails. All values must be finite.

## V1 Trigger and Deterministic Cell Selection

A cell becomes a V1 candidate only when its correct-memory decomposition satisfies:

```text
directional_alignment < 0
0 < predicted_optimal_alpha < 1
predicted_optimal_gain > trigger_floor
```

where

```text
trigger_floor = max(repeat_loss_floor, configured benefit_margin, 0)
```

This trigger means the correct-memory direction is potentially useful but the
full-strength residual may be poorly calibrated. Cells with non-negative directional
alignment do not consume alpha-sweep forwards because reducing a locally harmful
direction is not evidence that the memory can supplement the target.

Candidates are sorted deterministically by:

1. descending predicted optimal gain;
2. descending observed `q_content`;
3. ascending timestep index;
4. lexicographic layer group.

The first row is the primary verification cell. A second row is chosen only if
available, preferring a different timestep, then a different layer group, before
falling back to the next sorted candidate. No more than two cells are selected.

If no cell qualifies, V1 is skipped and the experiment terminates with a valid V0
mechanism result.

## V1: Matched Alpha Sweep

For every selected cell, run the following eight teacher-forced conditional arms:

```text
correct alpha = 0.00, 0.25, 0.50, 1.00
wrong   alpha = 0.00, 0.25, 0.50, 1.00
```

Alpha is applied only through the existing
`engine.sparse_role_memory_layer_scales` entries for the selected layer group. The
original scale mapping is copied before the sweep and restored in a `finally` block.
No checkpoint, module parameter, payload, query mask, noisy latent, target, or model
weight is mutated.

Alpha zero is intentionally distinct from `no_memory`: it executes the matched memory
and query path but multiplies the sparse residual by zero. This isolates path/routing
effects from the effective memory residual.

All V1 arms enable the existing `capture_sparse_token_diagnostics`. The alpha-one
correct and wrong predictions must reproduce the corresponding S0/S1 conditional
prediction SHA exactly. A mismatch fails the runtime-contract gate because diagnostic
capture must not alter the estimand.

## Host-Feature Diagnostics

Existing per-layer token diagnostics already expose `flat_idx`, `host_features`,
`raw_delta_features`, and `effective_delta_features`. V1 aggregates them online and
discards the feature tensors before writing output.

For each arm, layer, and alpha, report:

- selected query and memory token counts;
- mean host norm and effective delta norm;
- mean and maximum delta/host norm ratio;
- mean cosine between host and effective delta;
- mean cosine of the current host against the matched alpha-zero host;
- mean host norm drift relative to alpha zero;
- mean cosine between correct and wrong effective deltas on shared token indices.

Missing or misaligned token indices, non-finite features, or incompatible feature
dimensions fail the verification cell rather than silently dropping diagnostics.
Only aggregate scalars and counts are persisted.

## Mechanism Classification

Each V1 cell receives evidence flags and one conservative primary classification.
The configured trigger floor is used for all loss comparisons.

### `supplement_candidate`

- an observed correct-memory alpha has loss below matched correct alpha zero by more
  than the floor; and
- the same alpha is better than matched wrong memory by more than the floor.

Two cells with the same direction strengthen the result, but do not generalize beyond
the frozen event without independent videos and seeds.

### `representation_competition_candidate`

- a sub-unit correct alpha improves over correct alpha zero;
- correct alpha one is worse than the best sub-unit alpha by more than the floor; and
- host drift or delta/host ratio increases with alpha.

This is evidence consistent with over-strong fusion or representation competition,
not a definitive distinction between the two.

### `direction_mismatch`

- V0 correct directional alignment is non-negative; or
- no tested correct alpha improves over its alpha-zero control.

This indicates that reducing scale alone does not rescue the current memory direction.

### `path_or_routing_confound`

The correct or wrong alpha-zero loss differs from `no_memory` by more than the floor.
This flag may accompany another classification and means that matched memory/query
path effects must be separated from the effective residual.

### `no_authority`

Delta energy and observed loss changes remain at or below the configured floors across
the sweep. This is a null result, not evidence for supplementation or competition.

The report preserves all component flags so a single label cannot hide mixed evidence.

## Outputs and Report Contract

Verification mode adds:

```text
fusion_verification.json
```

and embeds the same JSON-safe object under `fusion_verification` in
`identity_probe_report.json`.

The object contains:

- `schema_version`
- `mode`
- `v0_cells`
- `trigger_floor`
- `trigger_candidates`
- `selected_cells`
- `v1_records`
- `alpha_curves`
- `host_feature_diagnostics`
- `mechanism_classification`
- `measured_forward_count`
- `measured_forward_budget`
- `gates`

`screening_cells.jsonl` also gains the V0 decomposition for each available arm. The
normal report files remain backward compatible. Verification mode leaves token score,
token group, and intervention outputs empty and marks the identity-set gate pending
with reason `fusion verification mode does not run S2`.

Count reconciliation remains mandatory:

- `forward_count == measured_arm_count`
- `measured_arm_count <= 41`
- `unconditional_dit_count == 0`
- `raw_dit_invocation_count` equals semantic prepasses plus conditional and
  unconditional DiT counts.

## Failure Handling and Reproducibility

- The existing fixed prefix, donor payload, teacher target, prompt, noise seed, and
  input hashes remain frozen.
- The engine and model are loaded once.
- Every alpha arm uses the same noisy latent and flow target for its cell.
- Layer scales are restored even when an arm raises.
- Alpha-one SHA parity is a hard gate.
- Decomposition reconstruction is a hard gate.
- V1 selection and tie-breaking are deterministic.
- A missing V1 candidate is a valid completed result.
- A blocked content gate remains scientifically valid and does not cause token labels
  to be emitted.

## Implementation Surface

- `utest/identity_token_probe.py`: pure decomposition, trigger selection, alpha sweep,
  host aggregation, classification, CLI flag, report integration, and budget gate.
- `utest/tests/test_identity_token_probe.py`: scalar decomposition, deterministic
  selection, classification, scale restoration, call budget, and report tests.
- `utest/README.md`: A100 verification command, field definitions, and interpretation.

No change is required in `infer_slotmem.py`, `reference_inference_runtime.py`, model
weights, payload formats, or Q*.

## Verification Plan

Implementation will follow TDD in independently reviewable slices:

1. Pure V0 decomposition and reconstruction tests.
2. Deterministic V1 trigger and two-cell selection tests.
3. Existing-scale alpha sweep with restoration and alpha-one SHA parity tests.
4. Host-feature scalar aggregation and conservative classification tests.
5. End-to-end fake-engine verification-mode budget and report tests.
6. Existing identity, Q*, hot-path, runner, self-check, and Python compilation
   regressions.
7. A100 smoke-scale verification before the full 41-arm run.

