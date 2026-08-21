# Q* Teacher-Forced Current-Step Injection Fix

**Date:** 2026-08-21  
**Status:** Approved in conversation  
**Scope:** Evaluation runtime and its fail-closed checks only. Prefix snapshots, model weights, memory interventions, sampler math, and rollout behavior remain unchanged.

## Problem

The Q* probe evaluates one scheduler timestep at a time. In `two_role_diff` mode, the runtime currently captures role-query features during the measured conditional forward and caches them for the next denoising step. A teacher-forced probe returns after that first forward, so it never consumes the captured query features. Payload reads succeed, but sparse memory injection remains disabled and all confirmatory arms produce identical predictions.

## Considered Approaches

1. **Current-step query prepass (selected).** In teacher-forced mode only, run the existing semantic probe on the fixed noisy latent before the measured forward, build the current role-query payload, then execute the unchanged measured conditional and unconditional forwards. This matches the probe's single-step contract without advancing the scheduler.
2. Run a synthetic preceding scheduler step. Rejected because it changes the fixed `z_t` contract and introduces sampler history into a local causal measurement.
3. Force `layer7_single` selection for Q*. Rejected because it would evaluate a different runtime configuration from the rollout under test.

## Data Flow

For each frozen `(target, noise, timestep, arm)` cell:

1. Load the immutable arm payload and cloned noisy latent.
2. Resolve roles from the same memory metadata used by rollout inference.
3. In teacher-forced `two_role_diff`, perform an unmeasured semantic-query prepass on that noisy latent.
4. Build current-step role boxes and query features using the existing extraction helpers.
5. Run the normal memory-aware conditional/unconditional prediction with those current-step features.
6. Return the prediction and actual sparse-injection diagnostics without a scheduler step or memory-bank write.

Normal multi-step rollout keeps its existing one-step query cache and is not changed.

## Failure Policy

The production Q* predictor must fail closed when a confirmatory arm reports a target payload read but the measured forward does not report an enabled sparse-memory injection with selected memory tokens and a positive finite effective delta in at least one injection layer. `no_memory` and diagnostic `native` remain exempt because absence of the target payload is their intended intervention.

## Verification

- A focused runtime test must show that a single teacher-forced `two_role_diff` call passes a non-empty current query payload into its measured memory-aware forward.
- A Q* predictor check must reject a target payload whose measured injection diagnostics remain disabled or zero.
- Existing Q* arithmetic, intervention, runner, and inference tests must remain green.
- The existing prefix snapshot is reused for the GPU rerun; only the Q* probe is rerun before any remaining rollout arms.
