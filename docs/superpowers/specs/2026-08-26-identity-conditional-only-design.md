# Identity Probe Conditional-Only Optimization Design

**Date:** 2026-08-26
**Status:** Approved in conversation; pending written-spec review

## Evidence and problem

The A100 smoke run was scientifically deterministic and operationally valid:
`correct` and `correct_repeat` had identical loss and prediction SHA, all intervention
arms had distinct prediction SHAs, and the frozen prefix was created from clean commit
`b06c7b9`. The selected cell nevertheless blocked content causality because both matched
wrong memory and no target memory achieved lower conditional flow loss than correct
memory.

The run also exposed an accounting and efficiency defect. Each identity teacher-forced
arm currently performs a truncated semantic/query-localization prepass, a full
conditional DiT forward, and a full unconditional DiT forward. Identity scoring reads
only `prediction_cond`, so the unconditional result cannot affect any identity loss,
candidate score, or gate. The report calls a `generate_chunk` request one “actual model
forward,” which understates the number of DiT invocations.

## Scope

Only identity teacher-forced probes become conditional-only. The paired Q* probe keeps
its current conditional score and CFG-composite hash diagnostic. Normal diffusion
generation and optional decoded S3 validation keep both CFG branches. Prefix generation,
frozen inputs, memory payload construction, query localization, conditional memory
injection, loss equations, thresholds, and gates do not change.

No dependency or new CUDA kernel is required. The optimization removes an unused PyTorch
DiT branch; FlashAttention 2 and BF16 behavior remain unchanged.

## Runtime contract

`teacher_forced_probe` gains an opt-in boolean `conditional_only`, defaulting to false.
Identity S0/S1 and S2 measured forwards set it to true. Semantic-capture-only calls
already return before conditional and unconditional denoising and therefore need no new
flag. Q* does not set the flag.

For a conditional-only request, the runtime:

1. performs the same semantic/query prepass when the arm has memory;
2. performs the same conditional memory-aware DiT forward;
3. snapshots the same conditional sparse and writer diagnostics;
4. skips the unconditional DiT forward and CFG composition;
5. returns `prediction_cond` unchanged and exposes `prediction` as the same conditional
   tensor with explicit `prediction_semantics="conditional"` and
   `cfg_composite_available=false` metadata.

The default path returns the existing CFG composite and marks
`prediction_semantics="cfg_composite"`, `cfg_composite_available=true`.

## Auditable counts

Each teacher-forced result reports a `dit_forward_counts` mapping:

- `semantic_prepass`: truncated semantic/query DiT invocations;
- `conditional`: full conditional DiT invocations;
- `unconditional`: full unconditional DiT invocations.

The identity report adds:

- `measured_arm_count`;
- `warmup_arm_count`;
- `semantic_prepass_count`;
- `conditional_dit_count`;
- `unconditional_dit_count`;
- `raw_dit_invocation_count`.

`forward_count` remains a compatibility alias for measured arms and `forward_budget`
continues to bound measured arms, not raw DiT invocations. The ambiguous
`actual_model_forward_count` is retained for one schema version as an alias of
`raw_dit_invocation_count`, with its meaning corrected in the report summary.
Truncated prepasses are counted separately rather than represented as equivalent full
forwards.

## Correctness gates

The optimization is accepted only if:

- identity calls set `conditional_only=true` while Q* calls do not;
- conditional-only executes no unconditional forward;
- the returned conditional prediction and conditional diagnostics match the legacy path
  before its unconditional branch;
- smoke still schedules five measured arms plus one warm-up;
- report counts reconcile exactly with per-call counts;
- existing Q*, identity scoring, inference-hotpath, and runner tests pass.

The completed negative smoke result remains evidence for timestep 25 and layers 5–10; it
is not rerun or relabeled by this optimization. After verification, the optimized code
may run the full S0/S1 layer-by-timestep screen. S2 remains conditional on a positive,
content-specific cell.

## Expected performance

Memory-bearing identity arms remove one full DiT invocation each. Wall-clock speedup will
be smaller than 2x because text/image preparation and the truncated semantic prepass
remain. Performance claims will use measured A100 timings from the next run, not a
theoretical multiplier.
