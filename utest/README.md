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

## Setup

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
