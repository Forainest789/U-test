# U-test — counterfactual utility measurement on frozen SlotMem

This repo is a SlotMem fork plus one package. Nothing here trains anything or edits the
generator: `utest` intervenes on the memory condition of the surrounding fork and scores
what comes out.

Why on someone else's frozen system: the question — *when does reading a memory help or
hurt the generated video* — presupposes that memory does something. Measured on our own
untrained injector it did not, and the nulls could never separate "memory is useless" from
"our injection is inert". On a published, trained, frozen memory system the intervention on
memory content is the only thing that varies, so whatever is observed is attributable to it.

Full protocol in [`docs/research-plan.md`](../docs/research-plan.md); the gate that decides
what may be claimed is [`docs/stage-gates.md`](../docs/stage-gates.md).

## Order

The order is the point. Each step's failure is cheap; skipping it makes the next step's
result uninterpretable.

| | Step | GPU | Gate |
|---|---|---|---|
| **E0** | count stories that contain a real recurrence event | none | `N_e >= 128` or the single-source controller line stops |
| **M0a** | run the official sample on Stage-2 | 1 card | it runs; record wall time and peak VRAM |
| **M0b** | reproduce the paper number | 1 card | NarraStream Subject Consistency `0.8771` within 0.02, else mark non-comparable |
| **M1** | prove the four arms change only memory | 1 card | hashes equal, decoded output actually diverges, target character resolves to a slot |
| **M2** | content causality on 12 development stories | 1 card | `correct` separates from matched `wrong`, and the no-memory arm is itself coherent |
| **M3** | decoded utility census | 1 card | helpful, neutral and harmful all occur beyond metric noise |

`E0` needs no weights and no GPU, and it can kill the plan. Run it first.

```bash
python -m utest.eligibility --data-root <narrastream-scripts> --out runs/e0.json
```

## Arms

| Arm | Memory | Answers |
|---|---|---|
| `no_memory` | reader disabled | baseline |
| `zero` | payload shape kept, tokens zeroed | is the path an additive/positional bias? |
| `correct` | same story, same `entity_uid` | does correct content do anything? |
| `wrong` | matched different-entity donor, query unchanged | does the effect depend on *which* memory? |

All arms share prompt, prefix snapshot, initial noise, seed, sampler and injection layers;
only the memory payload differs. `wrong` and `zero` are evaluated, never trained on.

Row-permuting already-encoded slots is excluded: cross-attention is permutation invariant
over a K/V set, so it is a mathematical no-op rather than a structure control.

```bash
python -m utest.content_audit --arm correct --dump-donor donor.pt -- <infer_slotmem args>
python -m utest.content_audit --arm wrong --donor donor.pt -- <infer_slotmem args>
```

Two more arms need no patch — pass them to the launcher directly:
`--no-enable_sparse_role_memory_attn` (memory off) and `--native_wan_inference` (base Wan2.2).

## Scoring

`utest.memory_utility` turns per-(story, event, arm, seed) outcome vectors into the
three-way census. Three rules are in the code rather than in a document, because one of
them was written down once before and lost in a rewrite:

- **The estimand is named by evidence.** Without a content-causality verdict the report
  says `memory_presence_effect`, never utility — a content-generic prior reproduced that
  number twice, and calling it the utility of a specific memory is the error this whole
  design exists to avoid.
- **Qualification seeds overlapping formal seeds raises.** Filtering stories on the formal
  draw's own baseline conditions on the minuend of `correct - none`, drops exactly the
  events whose baseline draw was poor, and biases the harm rate.
- **Dynamic degree is gated on the arm's absolute value.** A frozen clip maxes out
  smoothness, which is how a collapse into pure smoothing once passed as an improvement.

Statistics are clustered on story: recurrence events and seeds are repeated measurements
nested inside it, never independent samples.

## Remote stage test

Run E0 on real converted NarraStream inputs and the complete seven-chunk M0a on a
14B-capable server:

```bash
cd /data/long_term_data/shixiao/videomem/U-test
NARRASTREAM_INPUT_ROOT=/data/benchmarks/narrastream/slotmem_inputs \
WAN22_DIR=/data/long_term_data/shixiao/videomem/wan_models/Wan2.2-I2V-A14B \
CKPT_ROOT=/data/long_term_data/shixiao/videomem/U-test \
RUN_ROOT=/data/long_term_data/shixiao/videomem/U-test/runs/stage_gates \
UTEST_ENV=utest \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_slotmem_stage_gates.sh
```

For a roughly 60 GB GPU, keep the default `DUAL_EXPERT_LOAD_MODE=active`. `standard`
needs both approximately 28 GB experts plus VAE/text/SlotMem working memory and is likely
to OOM; managed layer offload is safer but slower. The stage runner keeps the official
50 denoising steps and bf16, writes `m0a/resume_state.pt` after each chunk, and can resume
without repeating completed chunks or rehashing checkpoints:

```bash
RUN_ID=20260813T120000Z RESUME_RUN=1 \
NARRASTREAM_INPUT_ROOT=/data/benchmarks/narrastream/slotmem_inputs \
WAN22_DIR=/data/long_term_data/shixiao/videomem/wan_models/Wan2.2-I2V-A14B \
CKPT_ROOT=/data/long_term_data/shixiao/videomem/U-test \
RUN_ROOT=/data/long_term_data/shixiao/videomem/U-test/runs/stage_gates \
UTEST_ENV=utest CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_slotmem_stage_gates.sh
```

The run writes `e0.json`, `platform.manifest.json`, all seven M0a chunks,
`efficiency.json`, `m0a_report.json`, `m0b_report.json`, and `stage_summary.json` under
one timestamped directory. M0b is deliberately `non-comparable` unless all four official
comparability flags and a normalized metric JSON are supplied:

```bash
M0B_OFFICIAL_INPUTS=1 \
M0B_OFFICIAL_PREPROCESSING=1 \
M0B_OFFICIAL_CHECKPOINT=1 \
M0B_OFFICIAL_EVALUATOR=1 \
M0B_METRIC_JSON=/data/results/official_narrastream_subject_consistency.json \
bash scripts/run_slotmem_stage_gates.sh
```

The normalized metric file contains `subject_consistency` and optional bootstrap bounds:

```json
{"subject_consistency": 0.8771, "ci_low": 0.861, "ci_high": 0.889}
```

## Fixed-prefix memory tests

`utest/events/person_reappearance_delta8_story.json` is the long person-memory fixture:
chunk 0 establishes Mara, chunks 1–7 remove her while varying similar-person, action,
camera, and lighting distractors, and chunk 8 brings her back in a new action and wide
composition. Its paired event is `utest/events/person_reappearance_delta8.json`. Replace
the event's `reference_path` with the frozen chunk-0 reference on the execution host and
supply an independently acquired 81-frame chunk-8 teacher video; an arm rollout is not a
valid teacher target. The existing sample-5 chunk-0→chunk-5 recurrence remains the short
`Delta=5` pilot, while this fixture is the `Delta=8` primary event.

### Strict Q* seven-run command

This is evaluation/inference, not weight training. It freezes one clean future target,
one Gaussian noise tensor, one noisy latent, and the actual scheduler timestep for every
confirmatory run. The primary value is `Q* = L_no_memory - L_correct`; positive Q* above
the adjacent `correct_repeat` floor means the historical target memory reduced current
flow-matching prediction error. `C_id` remains an optional rollout outcome.

Three things about that loss are load-bearing, because each was wrong once:

- **It is scored on the conditional velocity, not the CFG composite.** The sampler steps
  with `uncond + cfg_scale * (cond - uncond)`, and with `cfg_uncond_with_memory` the
  memory enters both branches, so the memory term in the composite is
  `5*delta_cond - 4*delta_uncond` and can cancel or flip. The flow target is defined
  against the unguided conditional output, so that is what `qstar_records.jsonl` scores;
  the composite survives only as `cfg_prediction_sha256`.
- **Every confirmatory arm runs the same forward.** `no_memory` carries no payload and
  would otherwise fall back to the stock DiT forward while the other five run
  `_memory_aware_dit_forward`, putting a forward-implementation term inside
  `L_no_memory - L_correct`. The probe passes `force_memory_path`, so the memory-off arm
  takes the same custom forward with an empty payload. `native` is exempt: it is base
  Wan and diagnostic-only.
- **The repeat floor and the benefit margin are separate numbers in separate units.**
  `--repeat-loss-tolerance` bounds an absolute MSE difference, `--repeat-influence-tolerance`
  bounds a unitless relative L2 ratio, and `--benefit-margin` is the smallest Q* that may
  be called `beneficial`. A deterministic repeat gives `repeat_loss_floor == 0`, so
  without an explicit margin any positive Q* clears the bar; the report says so in
  `benefit_margin_degenerate`.

```bash
EVENT_JSON="$PWD/utest/events/person_reappearance_delta8.json" \
FUTURE_TARGET_VIDEO=/data/targets/person_reappearance_delta8_chunk_008_teacher.mp4 \
FUTURE_TARGET_MANIFEST=/data/targets/person_reappearance_delta8_chunk_008_teacher.manifest.json \
BASE_INFERENCE_ARGS=/data/runs/stage_gates/slotmem_m0_001/m0a/inference_args.yaml \
PLATFORM_MANIFEST=/data/runs/stage_gates/slotmem_m0_001/platform.manifest.json \
DONOR_PAYLOAD=/data/events/donor_payload.pt \
DONOR_MANIFEST=/data/events/donor_manifest.json \
EVENT_RUN_ROOT=/data/runs/qstar_person_reappearance_delta8 \
QSTAR_TIMESTEP_INDICES=0,12,25,37,49 \
QSTAR_NOISE_SEED=0 \
RUN_ROLLOUT=1 \
CID_SCORER=/data/videomem/scripts/score_identity.py \
SLOTMEM_OFFLOAD_MODELS=0 \
UTEST_ENV=utest \
bash scripts/run_slotmem_qstar_event.sh
```

The teacher provenance manifest is mandatory; a filename and SHA alone cannot prove that
the target was not copied from an arm rollout:

```json
{
  "story_id": "person_reappearance_delta8",
  "target_chunk_idx": 8,
  "video_path": "/data/targets/person_reappearance_delta8_chunk_008_teacher.mp4",
  "video_sha256": "64-character SHA256",
  "source_type": "held_out_real",
  "generated_by_arm": false,
  "generated_by_evaluated_model": false
}
```

`source_type` may also be `independent_teacher`. Before any prefix GPU work, the runner
cross-checks this manifest and validates that the donor manifest identity, SHA, payload
key, exact slot shape, and embedded payload event all describe the same donor. Updating
only a donor SHA is intentionally rejected.

Both provenance documents are inputs, never outputs. `scripts/run_slotmem_delta8_cloud.sh`
exits before any work unless `FUTURE_TARGET_MANIFEST` and `DONOR_MANIFEST` both point at
existing files, and it writes neither: a `generated_by_arm: false` the runner stamped on a
video it never traced attests nothing, and a `slot_shape` the runner read out of the donor
payload it is about to compare against is a tautology. The donor row's matched-pair fields
(`coarse_class`, `colour`, `character_count`, `gap_bucket`, `selection_seed`) must be
chosen for *this* target; copying a row from another event's manifest and rewriting only
`target_story_id` / `target_entity_uid` passes every check and silently imports the other
event's matching decision.

Omit `CID_SCORER` when only Q* is required. Set `RUN_ROLLOUT=0` for the cheap
teacher-forced stage, and set `DRY_RUN=1` to write/inspect `run_manifest.json` without
loading weights. The runner rejects dirty source by default; `ALLOW_DIRTY_SOURCE=1` is a
development-only override. `REQUIRE_DYNAMIC_WRITER=1` is optional: without it, a zero
writer residual is truthfully reported as `static_prefix` and reader-memory Q* remains
valid.

The run writes `qstar/qstar_report.json`, `qstar/qstar_records.jsonl`, seven rollout
directories under `arms/`, `arms/intervention_contract.json`, and `run_manifest.json`.
The seven names are `correct`, `correct_repeat`, `no_memory`, `zero`, `random`, `wrong`,
and diagnostic-only `native`.

### Fast identity-token causal probe on A100 80 GB

This probe asks a narrower question than Q*: after content causality is established, can
a small set of video-token positions be identified that is sufficient for, and more
necessary to, Mara's identity recovery than equal-budget random or low-score controls?
It does not claim that a token has a context-free semantic type. An `identity_core`
label is an operational result for this frozen event, timestep, layer group, prompt, and
model; action and scene scores are diagnostics for separating plausible confounds.

The fast smoke run performs five measured teacher-forced arms at timestep 25 and the
middle layer group. Identity arms return the conditional flow velocity directly and skip
the unused unconditional CFG DiT; Q* and normal generation retain their existing CFG
path. Memory-bearing arms may additionally invoke one truncated semantic/query prepass.
Smoke verifies determinism, path influence, and correct-versus-wrong content direction,
then stops before token classification:

```bash
EVENT_JSON="$PWD/utest/events/person_reappearance_delta8.json" \
FUTURE_TARGET_VIDEO=/data/targets/person_reappearance_delta8_chunk_008_teacher.mp4 \
FUTURE_TARGET_MANIFEST=/data/targets/person_reappearance_delta8_chunk_008_teacher.manifest.json \
BASE_INFERENCE_ARGS=/data/runs/stage_gates/slotmem_m0_001/m0a/inference_args.yaml \
PLATFORM_MANIFEST=/data/runs/stage_gates/slotmem_m0_001/platform.manifest.json \
DONOR_PAYLOAD=/data/events/donor_payload.pt \
DONOR_MANIFEST=/data/events/donor_manifest.json \
EVENT_RUN_ROOT=/data/runs/identity_probe_smoke \
IDENTITY_SMOKE=1 RUN_DECODED_VALIDATION=0 \
UTEST_ENV=utest CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_slotmem_identity_probe.sh
```

After the optimized smoke reproduces the prior conditional losses and prediction hashes,
use a different, fresh `EVENT_RUN_ROOT` for the full S0/S1 grid. A blocked single-cell
smoke is still a valid null result for that cell; S2 remains gated on a positive
content-specific cell in the full grid:

```bash
EVENT_JSON="$PWD/utest/events/person_reappearance_delta8.json" \
FUTURE_TARGET_VIDEO=/data/targets/person_reappearance_delta8_chunk_008_teacher.mp4 \
FUTURE_TARGET_MANIFEST=/data/targets/person_reappearance_delta8_chunk_008_teacher.manifest.json \
BASE_INFERENCE_ARGS=/data/runs/stage_gates/slotmem_m0_001/m0a/inference_args.yaml \
PLATFORM_MANIFEST=/data/runs/stage_gates/slotmem_m0_001/platform.manifest.json \
DONOR_PAYLOAD=/data/events/donor_payload.pt \
DONOR_MANIFEST=/data/events/donor_manifest.json \
EVENT_RUN_ROOT=/data/runs/identity_probe_full \
IDENTITY_SMOKE=0 RUN_DECODED_VALIDATION=0 \
UTEST_ENV=utest CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_slotmem_identity_probe.sh
```

The five-arm smoke and 50-arm ceiling are measured-arm budgets, not raw DiT invocation
counts. The full path runs 25 S0/S1 screening arms, at most 25 measured S2 arms, two
truncated semantic-attention captures, and one unmeasured CUDA warm-up. S2 scores name,
stable attributes, correct-versus-wrong persistence, action, and scene channels; proposes
spatiotemporal groups; then applies group knockout and equal-budget identity/random/low
controls. It emits token labels only when the set-level gates pass: identity fraction at
most 25%, identity-only retention at least 80%, identity knockout stronger than controls,
correct content better than matched-wrong content, and the same causal direction in a
held-out cell. A failure blocks the identity claim instead of selecting the best-looking
tokens anyway.

The report reconciles work with `measured_arm_count`, `warmup_arm_count`,
`semantic_prepass_count`, `conditional_dit_count`, `unconditional_dit_count`, and
`raw_dit_invocation_count`; `actual_model_forward_count` is a compatibility alias for
the raw total. A semantic prepass is truncated after the required layer and is reported
separately rather than treated as a full-forward equivalent. If the full S0/S1 content
gate is blocked, stop there and do not infer token kinds or advance to S2.

To diagnose whether an identity residual supplies useful denoising direction or instead
becomes harmful at full strength, reuse the frozen prefix and arms in fusion-verification
mode. This mode runs the 25-arm S0/S1 grid, performs a zero-forward prediction-error
decomposition, and only when that decomposition predicts a rescuable sub-unit scale runs
matched `correct`/`wrong` sweeps at alpha `0,0.25,0.5,1` for at most two cells:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_2 \
SLOTMEM_OFFLOAD_MODELS=0 CUDA_VISIBLE_DEVICES=0 \
python -m utest.identity_token_probe \
  --prefix "$PREFIX_USED" \
  --future-target-video "$FUTURE_TARGET_VIDEO" \
  --arms-root "$ARMS_USED" \
  --donor "$DONOR_PAYLOAD" \
  --donor-manifest "$DONOR_MANIFEST" \
  --output "$FUSION_VERIFY_OUTPUT" \
  --timestep-indices 0,25,49 \
  --layer-groups 0-4,5-10,11-15 \
  --noise-seed 0 \
  --benefit-margin 0 \
  --verify-fusion
```

The mode never enters S2/S3, never emits identity-token labels, and has a hard ceiling
of 41 measured arms (`25 + 2*8`). Results are written to
`fusion_verification.json`. `representation_competition_candidate` means that a
sub-unit correct alpha helps, alpha one is worse, and host drift or delta/host ratio
grows with alpha; it is evidence consistent with over-strong fusion or representation
competition, not proof of the latter. Alpha-zero is a matched memory/query-path control
and is intentionally distinct from `no_memory`.

Only after S2 passes, an optional fresh run may add four decoded rollouts for external
identity and motion scoring:

```bash
EVENT_JSON="$PWD/utest/events/person_reappearance_delta8.json" \
FUTURE_TARGET_VIDEO=/data/targets/person_reappearance_delta8_chunk_008_teacher.mp4 \
FUTURE_TARGET_MANIFEST=/data/targets/person_reappearance_delta8_chunk_008_teacher.manifest.json \
BASE_INFERENCE_ARGS=/data/runs/stage_gates/slotmem_m0_001/m0a/inference_args.yaml \
PLATFORM_MANIFEST=/data/runs/stage_gates/slotmem_m0_001/platform.manifest.json \
DONOR_PAYLOAD=/data/events/donor_payload.pt \
DONOR_MANIFEST=/data/events/donor_manifest.json \
EVENT_RUN_ROOT=/data/runs/identity_probe_decoded \
IDENTITY_SMOKE=0 RUN_DECODED_VALIDATION=1 \
UTEST_ENV=utest CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_slotmem_identity_probe.sh
```

The 50-forward ceiling is for the teacher-forced S0-S2 probe. The four S3 diffusion
rollouts are deliberately outside that budget and are disabled by default. The runner
requires one A100 with at least 75 GiB visible memory, bf16, model offload disabled, and
the actual DiffSynth backend resolved to FlashAttention 2. `ALLOW_ATTENTION_FALLBACK=1`
exists only for development diagnostics and makes the run non-comparable to the fast A100
protocol. `DRY_RUN=1` records the complete command chain without loading weights.

Primary outputs are `identity_probe/identity_probe_report.json`,
`identity_probe/token_scores.parquet` (or its explicitly reported fallback),
`identity_probe/interventions.jsonl`, `identity_probe/summary.md`, diagnostic figures,
and the top-level `run_manifest.json`. Read the gate states before the rankings: a high
token score under a blocked content or identity gate is a candidate, not causal evidence.

### Legacy fixed-prefix five-arm rollout

Copy one event from `e0.json` into its own JSON file. Use a different eligible story to
prepare a donor prefix, then dump the native correct payload:

```bash
python -m utest.event_harness prepare-prefix \
  --event /data/events/donor_event.json \
  --output /data/runs/donor_prefix \
  --platform-manifest /data/runs/stage/platform.manifest.json \
  --inference-args-file /data/runs/stage/m0a/inference_args.yaml

python -m utest.event_harness dump-donor \
  --prefix /data/runs/donor_prefix \
  --output /data/runs/donor_dump \
  --donor-payload /data/runs/donor_payload.pt
```

Freeze the target/donor pairing as JSON. `payload_key` must be one of the keys written to
`donor_payload_info.json`; no arbitrary fallback is allowed:

```json
{
  "pairs": [{
    "target_story_id": "story_017",
    "target_entity_uid": "story_017::the white sedan",
    "donor_story_id": "story_042",
    "donor_entity_uid": "story_042::the white sedan",
    "payload_path": "/data/runs/donor_payload.pt",
    "payload_sha256": "64-character SHA256 from donor_payload_info.json",
    "payload_key": "the white sedan|0",
    "coarse_class": "car",
    "colour": "white",
    "character_count": 1,
    "source_visible": true,
    "gap_bucket": "1-2",
    "slot_shape": {"0": [64, 128]},
    "selection_seed": 0
  }]
}
```

Copy the exact `slot_shape` mapping for the selected `payload_key` from
`donor_payload_info.json.payload_slot_shapes`; do not infer it from another donor.

Run all arms from one immutable target prefix:

```bash
EVENT_JSON=/data/events/target_event.json \
BASE_INFERENCE_ARGS=/data/runs/stage/m0a/inference_args.yaml \
PLATFORM_MANIFEST=/data/runs/stage/platform.manifest.json \
DONOR_PAYLOAD=/data/runs/donor_payload.pt \
DONOR_MANIFEST=/data/events/donor_manifest.json \
EVENT_RUN_ROOT=/data/runs/fixed_prefix_story_017 \
UTEST_ENV=utest \
bash scripts/run_fixed_prefix_event_test.sh
```

The scientific gate is `prefix/arms/intervention_contract.json`. It contains snapshot
hash checks, target addressing, read/transform counts, the same-condition technical
repeat floor, and decoded frame-L1 contrasts. Hook counts alone are not a passing result.
The adjacent `utility_report.json` is `measurement_incomplete` until normalized decoded
outcome records and frozen rule JSON are supplied as `OUTCOME_RECORDS` and
`UTILITY_RULES`; missing evaluator dimensions are never treated as zero.

Each outcome record has `story_id`, `event_id`, `arm`, `seed`, and an `outcomes` object
containing `C_id`, `A_prompt`, `Q_bg`, `Q_motion_smoothness`,
`Q_motion_dynamic_degree`, `Q_flicker`, `Q_boundary`, `Q_anatomy`, and `Q_non_target`.
The frozen rules file has this shape (replace the numeric margins with W2 values):

```json
{
  "delta_id": 0.01,
  "quality_margins": {
    "A_prompt": 0.02,
    "Q_bg": 0.02,
    "Q_motion_smoothness": 0.02,
    "Q_flicker": 0.02,
    "Q_boundary": 0.02,
    "Q_anatomy": 0.02,
    "Q_non_target": 0.02
  },
  "dynamic_degree_floor": 0.2,
  "gate_a_floors": {"C_id": 0.3, "Q_motion_dynamic_degree": 0.2},
  "qualification_seeds": [1],
  "formal_seeds": [7],
  "content_causal": true,
  "n_boot": 10000,
  "bootstrap_seed": 0
}
```

### ViStoryBench subject reappearance

This is a local SlotMem video slicing protocol over the three frozen ViStoryBench
source–absence–reappearance intervals. CIDS is computed later with the frozen official
ViStoryBench evaluator; the derived video protocol itself is not the official image
sequence protocol.

Run this only from the already-active `slotmem` Conda environment. Do not create or
activate another environment. The commands below reuse the existing Wan2.2 directory and
the existing Stage-1/Stage-2 low/high checkpoints. All frozen producers are no-clobber.

```bash
cd /data/long_term_data/shixiao/videomem/U-test-vistory-8f0b728

export REPO_ROOT="$PWD"
export VM_ROOT=/data/long_term_data/shixiao/videomem
export WAN22_DIR="$VM_ROOT/wan_models/Wan2.2-I2V-A14B"
export CKPT_ROOT="$VM_ROOT/U-test"
export STAGE1_LOW_CKPT_PATH="$CKPT_ROOT/ckpt/stage1/stage1_low.pt"
export STAGE1_HIGH_CKPT_PATH="$CKPT_ROOT/ckpt/stage1/stage1_high.pt"
export STAGE2_LOW_CKPT_PATH="$CKPT_ROOT/ckpt/stage2/stage2_low.pt"
export STAGE2_HIGH_CKPT_PATH="$CKPT_ROOT/ckpt/stage2/stage2_high.pt"
export PLATFORM_MANIFEST="$REPO_ROOT/platform.manifest.json"
export VISTORY_REV=92f845531b67e97a67ae04b256ec5d8c020e8341
```

After the final reviewed commit is pulled, refresh the platform manifest once without
installing packages or changing the active environment:

```bash
SKIP_PIP=1 CKPT_ROOT="$CKPT_ROOT" WAN22_DIR="$WAN22_DIR" \
  bash scripts/fetch_weights.sh
```

The formal line must synchronize the frozen Hugging Face revision into a new directory.
The earlier custom server directory is not revision-attested and is excluded even if a
structural completeness check would pass. Hugging Face may reuse already cached blobs,
but `--local-dir` itself must not exist before this command. Leave every older directory
untouched:

```bash
export VISTORY_SNAPSHOT_ROOT="$VM_ROOT/datasets/ViStoryBench-full-$VISTORY_REV"

python - <<'PY' &&
import os
from pathlib import Path

root = Path(os.environ["VISTORY_SNAPSHOT_ROOT"])
assert not root.exists(), f"refusing to overwrite existing snapshot root: {root}"
PY
hf download ViStoryBench/ViStoryBench \
  --repo-type dataset \
  --revision "$VISTORY_REV" \
  --local-dir "$VISTORY_SNAPSHOT_ROOT" &&
export VISTORY_DATA="$VISTORY_SNAPSHOT_ROOT/ViStoryBench"
```

Then run the zero-GPU completeness gate before creating any survey. It requires exactly
the official story IDs `01..80`, parses every `story.json`, and reads every character
reference file:

```bash
python - <<'PY'
import os
from pathlib import Path

from utest.vistory_donors import validate_frozen_vistory_tree

root = Path(os.environ["VISTORY_DATA"])
validate_frozen_vistory_tree(root)
print("complete frozen ViStoryBench tree:", root.resolve())
PY
```

The measured M0 argv is the checkpoint-compatible base configuration. The zero-GPU gate
below proves that its last-option-wins values are MemoryEncoder layers `0..15` and 64
slots, and that the formal subject-subspace contract is top-8/64 with fraction `0.125`.
It also publishes the exact argv as JSON under a fresh experiment root. Do not edit the
recorded argv to manufacture a different geometry.

```bash
export M0_BASE_ARGS_YAML="$REPO_ROOT/runs/m0a_slotmem_stage2/inference_args.yaml"
export EXP_ROOT="$REPO_ROOT/runs/vistorybench_reappearance_top8_64_v1"
export BASE_ARGS_JSON="$EXP_ROOT/config/base_inference_args.json"

python - <<'PY'
import json
import os
from pathlib import Path
import yaml

from utest.prefix_contract import (
    FROZEN_MEMORY_ENCODER_LAYERS,
    FROZEN_MEMORY_ENCODER_SLOTS,
    FROZEN_SUBJECT_SUBSPACE_BUDGET,
    FROZEN_SUBJECT_SUBSPACE_FRACTION,
    normalized_frozen_args,
    validate_slotmem_memory_encoder_geometry,
)

source = Path(os.environ["M0_BASE_ARGS_YAML"])
root = Path(os.environ["EXP_ROOT"])
target = Path(os.environ["BASE_ARGS_JSON"])
assert source.is_file(), source
assert not root.exists(), f"refusing to reuse formal output root: {root}"
payload = yaml.safe_load(source.read_text(encoding="utf-8"))
argv = payload["argv"]
frozen_args = normalized_frozen_args(argv)
layers, slots = validate_slotmem_memory_encoder_geometry(frozen_args)
assert layers == FROZEN_MEMORY_ENCODER_LAYERS == tuple(range(16))
assert slots == FROZEN_MEMORY_ENCODER_SLOTS == 64
assert FROZEN_SUBJECT_SUBSPACE_BUDGET == 8
assert FROZEN_SUBJECT_SUBSPACE_FRACTION == 0.125
target.parent.mkdir(parents=True)
with target.open("x", encoding="utf-8") as handle:
    json.dump({"argv": argv}, handle, indent=2)
    handle.write("\n")
print("base argv:", target.resolve())
print("MemoryEncoder layers:", layers)
print("MemoryEncoder slots:", slots)
print(
    "subject-subspace budget/fraction:",
    FROZEN_SUBJECT_SUBSPACE_BUDGET,
    FROZEN_SUBJECT_SUBSPACE_FRACTION,
)
PY
```

Prepare the official three target events, then survey all structurally eligible official
donors. This phase is CPU-only:

```bash
python tools/prepare_slotmem_vistory_reappearance.py \
  --data-root "$VISTORY_DATA" \
  --output-root "$EXP_ROOT/inputs"

python tools/prepare_vistory_donors.py survey \
  --data-root "$VISTORY_DATA" \
  --targets "$EXP_ROOT/inputs/manifest.json" \
  --output "$EXP_ROOT/donors/survey.json"
```

Inspect every candidate's two prompts and official reference before making a decision:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["VISTORY_DATA"])
survey = json.loads(
    (Path(os.environ["EXP_ROOT"]) / "donors/survey.json").read_text(encoding="utf-8")
)
for row in survey["candidates"]:
    print("\n", row["target_event_id"], row["candidate_id"])
    print("reference:", root / row["reference"]["path"])
    print("source:", row["source_prompt"])
    print("read-check:", row["read_prompt"])
PY
```

Create the strict review skeleton once. It contains one row for every surveyed candidate
and deliberately sets every `approved` value to `false`; this command never approves a
donor. The four required presentation/colour strings and `reviewer` start blank so the
strict validator cannot mistake placeholders for completed human review. A human must
fill all five strings and set both visibility booleans from the official reference and
prompts. Keep `tie_group` as JSON `null` for a
single approval; multiple approvals for one target require the same explicitly reviewed,
non-empty tie-group string. Approve exactly one candidate per target unless declaring
that tie. The strict freeze rejects missing rows, blank required strings,
presentation/colour mismatches, invisible approved donors, or stale survey provenance.

```bash
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["EXP_ROOT"]) / "donors"
survey_path = root / "survey.json"
survey = json.loads(survey_path.read_text(encoding="utf-8"))
review = {
    "schema_version": 1,
    "dataset_commit": survey["dataset_commit"],
    "survey_sha256": hashlib.sha256(survey_path.read_bytes()).hexdigest(),
    "reviews": [
        {
            "target_event_id": row["target_event_id"],
            "candidate_id": row["candidate_id"],
            "target_presentation_class": "",
            "donor_presentation_class": "",
            "target_dominant_colour": "",
            "donor_dominant_colour": "",
            "donor_source_visible": False,
            "donor_read_check_visible": False,
            "approved": False,
            "tie_group": None,
            "reviewer": "",
        }
        for row in survey["candidates"]
    ],
}
required_strings = (
    "target_presentation_class",
    "donor_presentation_class",
    "target_dominant_colour",
    "donor_dominant_colour",
    "reviewer",
)
assert len(review["reviews"]) == len(survey["candidates"])
assert all(row["approved"] is False for row in review["reviews"])
assert all(row["donor_source_visible"] is False for row in review["reviews"])
assert all(row["donor_read_check_visible"] is False for row in review["reviews"])
assert all(row["tie_group"] is None for row in review["reviews"])
assert all(
    row[field] == "" for row in review["reviews"] for field in required_strings
)
path = root / "review.json"
with path.open("x", encoding="utf-8") as handle:
    json.dump(review, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write("\n")
print(path.resolve())
PY
```

**STOP for manual donor review.** Inspect every candidate, fill every required string and
visibility decision, and set approvals/tie groups according to the rules above. Do not
paste or run the freeze command until the completed `review.json` has been inspected.

After that manual review is complete, freeze the CPU-only donor selection:

```bash
python tools/prepare_vistory_donors.py freeze \
  --data-root "$VISTORY_DATA" \
  --targets "$EXP_ROOT/inputs/manifest.json" \
  --survey "$EXP_ROOT/donors/survey.json" \
  --review "$EXP_ROOT/donors/review.json" \
  --output-root "$EXP_ROOT/donors/selection"
```

**STOP after this CPU-only freeze.** Inspect the complete review and all three frozen
selection bundles, then obtain explicit user approval of the frozen donor selection.
Do not build a donor run manifest and do not launch GPU donor generation before that
approval. Everything below is post-approval only.

After approval, build the three-job donor manifest. This mandatory zero-GPU protocol gate
rejects a missing option, layers other than `0..15`, or an effective slot count other
than 64 with the actual and expected values in the error:

```bash
export DONOR_RUN_ROOT="$EXP_ROOT/donors/run"
export DONOR_RUN_MANIFEST="$DONOR_RUN_ROOT/run_manifest.json"
export SLOTMEM_OFFLOAD_MODELS=0

python -m utest.vistory_donor_harness dry-run \
  --selection "$EXP_ROOT/donors/selection/selection.json" \
  --output "$DONOR_RUN_ROOT" \
  --base-inference-args "$BASE_ARGS_JSON" \
  --platform-manifest "$PLATFORM_MANIFEST"

python - <<'PY'
import json
import os
from pathlib import Path

run = json.loads(Path(os.environ["DONOR_RUN_MANIFEST"]).read_text(encoding="utf-8"))
assert len(run["jobs"]) == 3
assert {job["donor_seed"] for job in run["jobs"]} == {0}
print("donor jobs:", len(run["jobs"]), "donor seeds:", [0])
PY
```

Run all three seed-0 donors on one GPU. The donor harness has no `full` subcommand:
`prefix` plus `dump` is its complete run, and each successful dump writes the immutable
completion record. From a completely fresh donor run directory, `resume` may be used
instead as the combined prefix-plus-dump driver.

```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DUAL_EXPERT_LOAD_MODE=active
export DUAL_EXPERT_MANAGE_AUX_MODELS=1

python -m utest.vistory_donor_harness prefix \
  --manifest "$DONOR_RUN_MANIFEST"

python -m utest.vistory_donor_harness dump \
  --manifest "$DONOR_RUN_MANIFEST"
```

Donor `resume` is suitable only as the combined driver from a fresh dry-run or when a
job already has its complete immutable completion record. It deliberately refuses a
prefix-only/partial job. If `prefix` or `dump` fails midway, preserve the entire failed
run directory by atomically renaming it to a timestamped sibling, then rebuild the same
clean path with `dry-run` and rerun. Do not delete the failed evidence:

```bash
python - <<'PY'
import os
from datetime import datetime, timezone
from pathlib import Path

source = Path(os.environ["DONOR_RUN_ROOT"])
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
failed = source.with_name(f"{source.name}.failed-{stamp}")
assert source.is_dir(), source
assert not failed.exists(), failed
source.rename(failed)
print("preserved partial donor run:", failed)
PY

python -m utest.vistory_donor_harness dry-run \
  --selection "$EXP_ROOT/donors/selection/selection.json" \
  --output "$DONOR_RUN_ROOT" \
  --base-inference-args "$BASE_ARGS_JSON" \
  --platform-manifest "$PLATFORM_MANIFEST"

python -m utest.vistory_donor_harness resume \
  --manifest "$DONOR_RUN_MANIFEST"
```

Freeze one validated event-level donor map, shared by target seeds 0, 1, and 2. This
command revalidates all three completion records and payloads before publishing:

```bash
export DONOR_BUNDLE_ROOT="$EXP_ROOT/donors/bundle"
export DONOR_MAP="$DONOR_BUNDLE_ROOT/donor_map.json"

python tools/freeze_vistory_donor_map.py \
  --targets "$EXP_ROOT/inputs/manifest.json" \
  --selection "$EXP_ROOT/donors/selection/selection.json" \
  --donor-run-manifest "$DONOR_RUN_MANIFEST" \
  --output-root "$DONOR_BUNDLE_ROOT"
```

Only now build the formal immutable 3-event × 3-seed target plan:

```bash
export TARGET_RUN_ROOT="$EXP_ROOT/formal"
export TARGET_RUN_MANIFEST="$TARGET_RUN_ROOT/run_manifest.json"

python -m utest.subject_reappearance_harness dry-run \
  --inputs "$EXP_ROOT/inputs/manifest.json" \
  --output "$TARGET_RUN_ROOT" \
  --base-inference-args "$BASE_ARGS_JSON" \
  --platform-manifest "$PLATFORM_MANIFEST" \
  --donor-map "$DONOR_MAP"

python - <<'PY'
import json
import os
from pathlib import Path

run = json.loads(Path(os.environ["TARGET_RUN_MANIFEST"]).read_text(encoding="utf-8"))
blocks = run["blocks"]
assert len(blocks) == 9
assert sorted({block["seed"] for block in blocks}) == [0, 1, 2]
assert {block["commands"]["preflight"]["status"] for block in blocks} == {
    "deferred_until_prefix"
}
assert {block["commands"]["full"]["status"] for block in blocks} == {
    "deferred_until_prefix"
}
assert {block["qstar"]["status"] for block in blocks} == {"not_available"}
assert {block["commands"]["qstar"]["status"] for block in blocks} == {
    "not_available"
}
print("formal target blocks: 9; seeds: [0, 1, 2]")
print("preflight/full: donor-ready and deferred until prefix")
print("Q*: record-only, independent teacher not available")
PY
```

`--base-inference-args` accepts JSON argv only: either `["--flag","value"]` or
`{"argv":["--flag","value"]}`. Shell command text is
rejected. With the donor payloads and bundle already frozen, dry-run must produce nine
blocks for seeds `[0, 1, 2]`; `preflight` and `full` are donor-ready but deferred until a
prefix exists. Q* remains record-only: without an independently frozen teacher map its
availability status is `not_available`, and it never enters donor selection, ranking,
mask construction, or injection. After a validated prefix exists, the harness atomically
freezes real stage argv in each block's `stage_commands.json` and executes only that
artifact.

Generate all nine target prefixes/captures. Then inspect each block's source chunk and
write `source_qualification.json` only when the named subject is visible,
distinguishable, and unambiguous. The following publisher is intentionally a separate,
manual post-review step; do not run it before reviewing all nine source videos.

```bash
python -m utest.subject_reappearance_harness prefix \
  --manifest "$TARGET_RUN_MANIFEST"

python - <<'PY'
import json
import os
from pathlib import Path

run = json.loads(Path(os.environ["TARGET_RUN_MANIFEST"]).read_text(encoding="utf-8"))
for block in run["blocks"]:
    event = json.loads(Path(block["event_json"]).read_text(encoding="utf-8"))
    source_video = (
        Path(block["block_dir"])
        / "prefix/prefix_generation"
        / f"chunk_{int(event['source_chunk_idx']):03d}.mp4"
    )
    assert source_video.is_file(), source_video
    print(block["event_id"], "seed", block["seed"], source_video)
PY

python - <<'PY'
import json
import os
from pathlib import Path

run = json.loads(Path(os.environ["TARGET_RUN_MANIFEST"]).read_text(encoding="utf-8"))
for block in run["blocks"]:
    path = Path(block["source_qualification"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump({"status": "passed"}, handle)
        handle.write("\n")
print("published reviewed source qualifications:", len(run["blocks"]))
PY
```

The `probe` stage first produces and validates source-only `semantic_scores.json`, then
freezes the three layer-group masks. It never reads target frames, decoded arms, CIDS, or
Q*. Run the CPU probe, the four-arm GPU preflight, and only then the full eight arms:

```bash
python -m utest.subject_reappearance_harness probe \
  --manifest "$TARGET_RUN_MANIFEST"

python -m utest.subject_reappearance_harness preflight \
  --manifest "$TARGET_RUN_MANIFEST"

python -m utest.subject_reappearance_harness full \
  --manifest "$TARGET_RUN_MANIFEST"
```

Preflight order is `full_correct,no_memory,zero_path,wrong_subject`; the full order is
the fixed eight-arm subject-subspace table. All arms reuse one immutable pre-target
snapshot and target seed, use `max_memory_characters=4`, clear `target_character`, and
use `fixed_reference_scope=source_only`. With no validated teacher map, Q* is recorded
as `not_available` and no teacher command exists: do not run the `qstar` stage and do not
reinterpret decoded CIDS as Q*. With a separately frozen independent teacher map, build
a fresh formal run with `--teacher-map` and only then run the real `qstar` subcommand.
With a teacher but no donor, Q* is `blocked_missing_donor`. `resume` revalidates completed
prefix, probe, arm, decoded preflight, and Q* artifacts; it skips only intact outputs and
archives failed/partial attempts beside their replacement so logs are retained.

```bash
# Run only when this manifest was built with a validated independent --teacher-map.
python -m utest.subject_reappearance_harness qstar \
  --manifest "$TARGET_RUN_MANIFEST"
```

Export the completed arms into the frozen eight-frame official CIDS adapter, then execute
the exact evaluator argv recorded in that adapter. `VISTORY_EVALUATOR_ROOT` must be the
checkout at evaluator commit `b44ec9108668cc2bcc8c5280886b235e9fb8bea9`.

```bash
export CIDS_ROOT="$EXP_ROOT/cids"
export VISTORY_EVALUATOR_ROOT="$VM_ROOT/datasets/ViStoryBench-evaluator-b44ec910"

python tools/prepare_vistory_cids_inputs.py \
  --run-manifest "$TARGET_RUN_MANIFEST" \
  --output "$CIDS_ROOT"

python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

evaluator_root = Path(os.environ["VISTORY_EVALUATOR_ROOT"])
commit = subprocess.check_output(
    ["git", "-C", str(evaluator_root), "rev-parse", "HEAD"], text=True
).strip()
assert commit == "b44ec9108668cc2bcc8c5280886b235e9fb8bea9", commit
manifest = json.loads(
    (Path(os.environ["CIDS_ROOT"]) / "cids_input_manifest.json").read_text(
        encoding="utf-8"
    )
)
subprocess.run(
    manifest["official_cids"]["cli_argv"],
    cwd=evaluator_root,
    check=True,
)
PY
```

The final report also requires the separately normalized source-continuity, VBench
quality, and prompt-alignment result files; this repository does not fabricate them.
Once those evaluators have produced the following files, aggregate the event-clustered
report. Q* remains a descriptive field and is `not_available` without the independent
teacher contract.

```bash
export CONTINUITY_RESULTS="$EXP_ROOT/metrics/source_continuity.json"
export QUALITY_RESULTS="$EXP_ROOT/metrics/vbench_quality.json"
export PROMPT_RESULTS="$EXP_ROOT/metrics/prompt_alignment.json"
export IDENTITY_REPEAT_FLOOR_JSON="$EXP_ROOT/metrics/cids_repeat_floor.json"
export FINAL_REPORT_ROOT="$EXP_ROOT/report"

export CIDS_ITEMS="$(python - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(
    (Path(os.environ["CIDS_ROOT"]) / "cids_input_manifest.json").read_text(
        encoding="utf-8"
    )
)
official = manifest["official_cids"]
items = Path(official["result_path"]) / official["items_relative_path"]
assert items.is_file(), items
print(items)
PY
)"

export IDENTITY_REPEAT_FLOOR="$(python - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(
    Path(os.environ["IDENTITY_REPEAT_FLOOR_JSON"]).read_text(encoding="utf-8")
)
value = payload["repeat_floor"]
assert isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
print(value)
PY
)"

python tools/analyze_subject_reappearance.py \
  --run-manifest "$TARGET_RUN_MANIFEST" \
  --cids-input-manifest "$CIDS_ROOT/cids_input_manifest.json" \
  --cids-items "$CIDS_ITEMS" \
  --continuity-results "$CONTINUITY_RESULTS" \
  --quality-results "$QUALITY_RESULTS" \
  --prompt-results "$PROMPT_RESULTS" \
  --quality-rules "$REPO_ROOT/utest/events/vistorybench_reappearance_quality_rules.json" \
  --repeat-floor "$IDENTITY_REPEAT_FLOOR" \
  --n-boot 10000 \
  --bootstrap-seed 0 \
  --output "$FINAL_REPORT_ROOT"
```

After `probe` (and again after the complete run), this zero-GPU audit checks the frozen
matrix, the donor geometry, the source-only semantic producer, mask budget, and Q* status:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

donor_map = json.loads(Path(os.environ["DONOR_MAP"]).read_text(encoding="utf-8"))
run = json.loads(Path(os.environ["TARGET_RUN_MANIFEST"]).read_text(encoding="utf-8"))
assert len(donor_map["events"]) == 3
assert len(run["blocks"]) == 9
assert sorted({block["seed"] for block in run["blocks"]}) == [0, 1, 2]
assert {block["commands"]["preflight"]["status"] for block in run["blocks"]} == {
    "deferred_until_prefix"
}

for donor in donor_map["events"].values():
    pair = json.loads(Path(donor["manifest"]).read_text(encoding="utf-8"))["pairs"][0]
    shapes = pair["slot_shape"]
    assert set(shapes) == {str(layer) for layer in range(16)}
    assert {shape[0] for shape in shapes.values()} == {64}
    assert pair["payload_key"].endswith("|0")

for block in run["blocks"]:
    scores = json.loads(Path(block["semantic_scores"]).read_text(encoding="utf-8"))
    mask = json.loads(
        Path(block["subject_subspace_manifest"]).read_text(encoding="utf-8")
    )
    assert scores["producer"]["kind"] == "slotmem_source_semantic_token_scores"
    assert scores["target_evidence_read"] is False
    assert mask["target_evidence_read"] is False
    assert mask["primary_mask"] == "semantic_top8"
    assert {row["bank"] for row in mask["layers"]} == {0}
    assert {row["layer_group"] for row in mask["layers"]} == {
        "0-4", "5-10", "11-15"
    }
    assert all(row["slot_count"] == 64 for row in mask["layers"])
    assert all(row["budget"] == 8 for row in mask["layers"])
    assert all(row["budget_fraction"] == 0.125 for row in mask["layers"])
    assert all(len(row["semantic_top8"]) == 8 for row in mask["layers"])

qstar = {block["qstar"]["status"] for block in run["blocks"]}
assert qstar <= {"available", "not_available", "blocked_missing_donor"}
print("donor jobs:                 3")
print("donor seeds:                [0]")
print("formal target blocks:       9")
print("target seeds:               [0, 1, 2]")
print("donor preflight statuses:   ready")
print("semantic producer kind:     slotmem_source_semantic_token_scores")
print("semantic mask cardinality:  8 / 64 (0.125)")
print("semantic layer groups:      0-4, 5-10, 11-15")
print("Q*:                         descriptive or unavailable", sorted(qstar))
PY
```

## Legacy development setup

This is not part of the ViStoryBench workflow above, which reuses the active `slotmem`
environment.

```bash
conda create -n utest python=3.10 -y
```

```bash
UTEST_ENV=utest bash scripts/fetch_weights.sh
```

Fetches Wan2.2-I2V-A14B (~126 GB) and the SlotMem Stage-2 checkpoints (~21 GB), asserts
Stage-2 is present, and writes `platform.manifest.json` with checkpoint SHA256s plus the
repo commit and a dirty flag — a commit hash describes the tree only when the tree is clean.

Inference VRAM: one 14B expert resident is ~28 GB, so `DUAL_EXPERT_LOAD_MODE=active` peaks
around 36–42 GB with both experts still in the pipeline, swapping once per chunk at the 0.9
noise-domain boundary. Both experts resident is ~64 GB. Dropping an expert is not a cost
knob — it runs a denoiser outside the noise range it was trained for.
