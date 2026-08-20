# SlotMem Q* Seven-Arm Memory Utility Evaluation

**Date:** 2026-08-20  
**Status:** Approved for implementation review  
**Scope:** Evaluation and inference only. Model weights, attention math, sampler math, and training configuration remain unchanged.

## 1. Objective

Measure whether one historical SlotMem item `m_j` reduces prediction error for a frozen future target under identical conditions:

\[
Q^*_{j,\Delta,\tau}
= L_{den}(M_t \setminus m_j) - L_{den}(M_t).
\]

The Wan flow-matching target is:

\[
v^* = \epsilon - x_{t+\Delta},
\qquad
L_{den}(M)=\operatorname{MSE}(\hat v(M),v^*).
\]

Positive `Q*` means the correct historical memory lowers teacher-forced denoising error. Negative `Q*` means it is harmful. A prediction change without a loss improvement is influence, not utility.

`C_id` is an auxiliary rollout-validation score. It is not the Q* loss and is not used to define denoising utility.

## 2. Seven Runs

The experiment produces seven named runs:

| Run | Memory intervention | Role |
|---|---|---|
| `correct` | Native target memory | Q* reference |
| `correct_repeat` | Identical native target memory | Technical repeat floor |
| `no_memory` | Remove only target `m_j` | Primary ablation |
| `zero` | Preserve addressing/shape, zero target payload | Mechanism control |
| `random` | Deterministic per-layer, per-channel moment-matched payload | Content-free control |
| `wrong` | Exact-shape payload from a frozen matched donor | Content-specificity control |
| `native` | Base Wan path without SlotMem/LoRA memory path | Diagnostic only |

The first six form the confirmatory paired Q* experiment. `native` is reported separately and never substitutes for `correct_repeat` or enters the primary Q* estimand.

## 3. Identical-Condition Contract

Every confirmatory arm must share:

- one immutable prefix snapshot and snapshot SHA256;
- one future clean target latent `x_(t+Delta)` from an arm-independent target video;
- one Gaussian noise tensor `epsilon` and tensor SHA256;
- one noisy latent `z_tau` and tensor SHA256;
- one scheduler timestep `tau` and noise-domain assignment;
- identical prompt, text embedding, reference condition, CFG, solver, checkpoint, dtype, and model arguments;
- identical target character, memory key, bank, slot shape, and target-region mask when a mask is available.

The future target must be a held-out real clip or an independently acquired teacher clip that was not produced by the evaluated model or any of its seven arms. It is frozen before arm execution. A `correct`-arm rollout may not be used as its own teacher target.

The contract is rebuilt after the final prefix target is selected. It records the actual target prompt and actual target seed; stale inherited prompt or seed values fail before GPU work begins. A dirty source tree is recorded and rejected by default, with an explicit development-only override.

## 4. Q* Probe

Add a teacher-forced probe that loads the frozen model once, prepares the future latent/noise/timestep once, then evaluates cloned immutable memory payloads for the seven run definitions.

The probe reuses the existing pure intervention functions in `utest.content_audit`; it does not create a second implementation of zero/random/wrong semantics. It performs no optimizer step, no parameter mutation, no bank write, and no sampler update.

For every `(event, memory_id, Delta, tau, arm)`, write:

- global flow-matching MSE;
- optional target-region flow MSE when a frozen mask exists;
- `Q*_arm = L_arm - L_correct` for controls and `Q* = L_no_memory - L_correct` for the primary estimand;
- normalized prediction influence `||v_correct - v_arm|| / ||v_correct||`;
- correct-repeat loss and prediction floors;
- prompt, target, noise, noisy-latent, prefix, payload, and runtime hashes;
- memory read hit, slot/layer counts, injection delta norm, dtype, device, and finite checks;
- `memory_regime`, either `static_prefix` or `dynamic_writer`.

The output is `qstar_report.json`; raw per-cell rows are also written as `qstar_records.jsonl` for later aggregation.

### Primary interpretation

- `influence_above_floor=false`: memory did not measurably affect the prediction.
- `influence_above_floor=true` and `Q* <= repeat_margin`: memory affected the prediction but did not show benefit.
- `Q* > repeat_margin`: correct memory improved teacher-forced prediction for that `(Delta, tau)` cell.
- `L_correct < L_wrong` and `L_correct < L_random`: evidence that content, not merely payload presence or magnitude, matters.

## 5. Horizon and Timestep Design

The existing sample-5 `Delta=5` event remains the short regression/pilot. It must be re-frozen with the actual chunk-5 prompt and seed.

The long-sequence primary event contains nine chunks:

- chunk 0: establish the target person with a clear face, stable clothing, and one identity cue;
- chunks 1-7: target absent; scene, action, camera, and similar-person distractors vary;
- chunk 8: target person reappears with a new action and composition;
- target prompt does not repeat the complete appearance description;
- background recurrence is not required, keeping person memory as the manipulated content.

The initial pilot evaluates `Delta=5`. The long event evaluates `Delta=8`. After both pass contract checks, the runner may accept additional independently frozen events at `Delta=2` and `Delta=5`; one story with repeated intermediate target appearances is not used to emulate multiple horizons because those appearances can update memory and confound `m_j`.

Timestep values come from actual scheduler indices, not hand-written continuous values. The default pilot grid samples five regions spanning both experts: one high-noise cell and four low-noise cells from early to late denoising. The resolved timestep values and scheduler percentages are frozen in the report.

## 6. Full Rollout Validation

Q* probes are cheap local measurements. Selected events also run the seven full rollouts from the same immutable prefix:

- confirmatory arms run in isolated processes;
- `correct` and `correct_repeat` are adjacent;
- every arm uses the same target seed and runtime contract;
- snapshot hash is checked before and after every process;
- decoded videos must have aligned frame count, resolution, and FPS;
- intervention L1 must exceed the correct-repeat technical floor;
- `C_id` and quality metrics are auxiliary outcomes for testing whether Q* predicts rollout behavior.

The initial Q* claim is limited to denoising utility. Q* is promoted to an online memory-utility monitor only after event-level Q* predicts independently measured rollout utility on held-out events.

## 7. Writer Regime

Writer residual and bank-hash evidence remain visible but are not falsified or repaired by changing the model.

- Positive finite writer residual plus target-scope bank hash change: `dynamic_writer`.
- Zero residual or unchanged target-scope bank: `static_prefix`.

A `static_prefix` run may establish reader-memory denoising utility. It may not be described as dynamic-memory utility. A runner option that explicitly requires dynamic writer behavior fails closed when the checkpoint is static.

## 8. Runner and Outputs

Create one strict entry point, `scripts/run_slotmem_qstar_event.sh`, that performs:

1. CPU self-checks for intervention semantics and Q* arithmetic;
2. event, target-video, donor-manifest, model-contract, and clean-tree preflight;
3. long prefix generation and immutable contract creation;
4. seven-run teacher-forced Q* probe;
5. strict seven-run rollout when rollout validation is enabled;
6. contract validation and report aggregation;
7. optional external `C_id`/quality scoring, required when rollout utility validation is requested.

Required outputs:

```text
<event-run>/
  prefix/
    event.json
    prefix_state.pt
    prefix_contract.json
  qstar/
    qstar_report.json
    qstar_records.jsonl
    probe.log
  arms/
    correct/
    correct_repeat/
    no_memory/
    zero/
    random/
    wrong/
    native/
    intervention_contract.json
    failure_ledger.json
    utility_report.json
  run_manifest.json
```

The legacy `scripts/run_slotmem_utest.sh` is not a scientific entry point. It should delegate to the strict runner or exit with the exact strict command instead of producing a weaker success message.

## 9. Failure Policy

The runner stops before subsequent expensive work when any of these occur:

- target clip, donor, checkpoint, prefix, prompt, reference, noise, or timestep hash mismatch;
- target character is outside the SlotMem read window;
- required payload layer/slot is absent or has a mismatched shape;
- `wrong` donor provenance is incomplete;
- `random` moment checks fail;
- source, target, noisy latent, or noise differs across confirmatory arms;
- loss, prediction, injection statistic, or writer statistic is non-finite;
- correct-repeat exceeds the frozen tolerance;
- rollout frame alignment fails;
- required Q*/rollout evaluator output is missing.

Failures remain machine-readable and never become neutral utility observations.

## 10. Tests and Verification

Minimum runnable checks:

- unit tests for Q* sign and arm-delta arithmetic;
- exact shared-input hash checks across arms;
- correct-repeat floor behavior;
- finite global and masked flow loss;
- deterministic random-arm output independent of layer iteration order;
- exact donor shape/provenance rejection;
- stale prompt/seed contract rejection;
- static/dynamic writer classification;
- seven-run command construction and ordering;
- synthetic end-to-end probe using a small fake model;
- strict runner dry-run snapshot of the complete command chain;
- existing `utest` test suite.

GPU verification uses one short `Delta=5` event first. The nine-chunk `Delta=8` run is allowed only after the short event produces a complete `qstar_report.json` and passes all contract checks.

## 11. User-Facing Command Shape

The final remote command will be an evaluation/inference command, not a training command:

```bash
EVENT_JSON=/data/events/sample_5_qstar.json \
FUTURE_TARGET_VIDEO=/data/targets/sample_5_chunk_005.mp4 \
BASE_INFERENCE_ARGS=/data/runs/stage_gates/slotmem_m0_001/m0a/inference_args.yaml \
PLATFORM_MANIFEST=/data/runs/stage_gates/slotmem_m0_001/platform.manifest.json \
DONOR_PAYLOAD=/data/events/donor_payload.pt \
DONOR_MANIFEST=/data/events/donor_manifest.json \
EVENT_RUN_ROOT=/data/runs/qstar_sample_5 \
QSTAR_TIMESTEP_INDICES=0,12,25,37,49 \
RUN_ROLLOUT=1 \
CID_SCORER=/data/videomem/scripts/score_identity.py \
UTEST_ENV=utest \
bash scripts/run_slotmem_qstar_event.sh
```

The implemented runner prints every resolved seven-run probe and rollout command before GPU execution and writes the same commands to `run_manifest.json`.
