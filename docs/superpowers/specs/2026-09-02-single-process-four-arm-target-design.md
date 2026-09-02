# Single-Process Four-Arm Target Generation Design

Status: approved

## Goal

Generate the frozen subject-reappearance preflight arms `full_correct`, `no_memory`,
`zero_path`, and `wrong_subject` in one Python process with one loaded
`SlotMemInferenceEngine`. Each arm generates only the target chunk. This removes four
independent model cold starts and the existing target+1 generation without changing
SlotMem model or inference-runtime code.

## Constraints

- Do not modify `infer_slotmem.py`, `reference_inference_runtime.py`, model code, checkpoint
  loading, memory injection, semantic selection, or denoising behavior.
- Keep the four arm definitions, frozen prefix snapshot, prompt, target seed, reference
  frames, donor provenance, subject mask, resolution, frame count, and inference-step
  count unchanged.
- Run arms sequentially on one GPU; do not parallelize GPU forwards.
- Disable whole-model between-run offload with the existing `--no-offload_models` setting.
- Retain the current active dual-expert residency policy. It may move the active expert as
  required by the noise-domain schedule; loading both 14B experts permanently is outside
  this change and may exceed the available VRAM.
- Treat `no_memory` as the normal no-memory baseline. Do not add a native-Wan arm.

## Non-goals

- Do not optimize the 16-layer feature-capture hot path.
- Do not change the scientific meaning of correct, zero, wrong-subject, or no-memory.
- Do not generate or score target+1 delayed-effect outputs in this preflight mode.
- Do not change the existing multi-process path used by other experiments.
- Do not build a persistent GPU daemon or an inter-process protocol.

## Architecture

Add an opt-in single-process target-only execution path to the subject-reappearance
harness. Reuse the established decoded-arm pattern in `identity_token_probe.run_s3`:
construct one `SlotMemInferenceEngine`, build frozen arm payload bundles, and call
`engine.generate_chunk()` once per arm in a local loop.

The new path belongs entirely to `utest`; the SlotMem entry point and shared runtime remain
unchanged. The current multi-process command artifacts remain authoritative for existing
runs, while the new run artifact records that it used the single-process target-only mode.

## Data Flow

1. Validate the existing run manifest, selected event and seed, passed source
   qualification, frozen prefix contract, subject-subspace manifest, and donor bundle.
2. Parse the frozen inference arguments and require `offload_models=false`.
3. Construct one `SlotMemInferenceEngine` from those arguments.
4. Load the prefix snapshot once on CPU and reconstruct the exact target reference frames.
5. Build four complete memory bundles from the frozen prefix:
   - `full_correct`: unchanged target payload;
   - `no_memory`: omit the target payload;
   - `zero_path`: preserve target payload shapes and metadata but zero its values;
   - `wrong_subject`: replace only the frozen subject rows with the validated donor rows.
   Non-target character payloads remain unchanged in every arm.
6. For each arm, reset run-local engine diagnostics, use the same prompt, seed and
   references, call `generate_chunk()` once, and save the returned target video under the
   arm's existing output directory.
7. Pass `online_memory_chars=[]` and `online_memory_bank_percents=[]`. Target-only
   preflight has no downstream chunk, so writer output is neither needed nor allowed to
   become cross-arm state.
8. Write per-arm audit, timing and hash records, then run four-arm decoded validation.

## State Isolation

Model weights and the loaded pipeline are the only state intentionally shared across
arms. Every value that can affect an arm result is recreated or reset before its forward:

- arm memory tensors are derived independently from the immutable prefix snapshot;
- the sampler seed is identical and `generate_chunk()` creates fresh initial noise;
- scheduler timesteps and query caches are initialized inside each `generate_chunk()`;
- reference-frame lists passed to an arm are fresh list objects;
- engine diagnostic accumulators and last-forward statistics are cleared;
- no online memory is collected or written;
- the audit reader patch is installed and restored within one arm's scope.

The harness hashes the prefix snapshot before the first arm and after every arm. Any
change, missing reset, invalid output, or failed audit stops the loop before the next arm.

## Output Contract

The target-only run produces exactly four videos in this order:

1. `full_correct`
2. `no_memory`
3. `zero_path`
4. `wrong_subject`

Each arm directory also contains:

- an audit record proving the selected payload and target read behavior;
- wall-clock generation time;
- frame count and video SHA-256;
- the shared prefix SHA-256, target seed and runtime-contract identity.

The phase-level validation records:

- `execution_mode: single_process_target_only`;
- `engine_initialization_count: 1`;
- the frozen arm order;
- exactly one target video per arm;
- no target+1 output;
- decoded equivalence of `zero_path` and `no_memory` where required by the existing gate;
- non-zero divergence for correct and wrong-subject where required by the existing gate.

Existing output directories are never overwritten. Resume may skip an arm only when its
video, hashes, audit and runtime identity all validate. Because a resumed process must load
the engine again, `engine_initialization_count: 1` applies per invocation, not across
separate resume invocations.

## Failure Handling

- Fail before engine initialization when frozen inputs, donor geometry, or provenance are
  invalid.
- Fail before the next arm when generation, saving, audit, snapshot integrity, or decoded
  validation fails.
- Preserve completed arm artifacts for diagnosis.
- Never silently fall back to the old multi-process path after the single-process mode has
  started.

## Verification

CPU tests must prove:

- the mode accepts exactly the four frozen preflight arms in the frozen order;
- target-only execution schedules one `generate_chunk()` call per arm;
- all four calls receive the same seed, prompt, references and immutable prefix identity;
- payload transformations match the existing subject-subspace implementation;
- non-target payloads are byte-identical across arms;
- online memory collection is disabled;
- diagnostics are reset and a failed arm prevents later calls;
- existing multi-process command construction remains unchanged.

One A100 smoke run must prove:

- only one engine initialization appears in the process log;
- exactly four target videos are produced and no target+1 video exists;
- all four audits and the existing decoded gate pass;
- peak VRAM remains within the current host limit;
- total wall time is lower than the previous four-arm preflight for the same frozen event,
  excluding already-completed prefix, qualification and probe stages.

The performance claim is limited to removal of redundant cold starts and target+1 work.
No claim is made that a memory-bearing target forward becomes faster.
