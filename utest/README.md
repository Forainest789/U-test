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

## Fixed-prefix five-arm event test

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
    "slot_shape": "from donor payload audit",
    "selection_seed": 0
  }]
}
```

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
