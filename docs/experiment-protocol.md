# Counterfactual Utility Control Experiment Protocol

Protocol date: 2026-08-09  
Primary design: within-story paired intervention with a frozen prefix  
Primary model: Wan2.2-I2V-A14B + frozen SlotMem Stage-2  
Primary unit: independent story/source

## 1. Confirmatory Questions

1. Does SlotMem transmit instance-specific memory content under correct/wrong/zero/none intervention?
2. Does an identity-matched memory have heterogeneous decoded effects across target recurrence events?
3. Does fixed-trajectory attribution add predictive information beyond relevance, attention, recency, and horizon?
4. Does a decoded-utility router reduce harmful reads or improve paired outcome over SlotMem all-memory while satisfying quality non-inferiority?

The unique primary policy comparison is `utility-router vs all-memory`. Other comparisons are secondary and corrected for multiplicity.

## 2. Data and Split

### 2.1 Eligibility

Run a deterministic zero-GPU audit over NarraStream scripts in W1, in parallel with native reproduction and before M1–M3 expansion. Include a story only when it contains a traceable character occurrence, at least one full absent chunk, and an explicit reappearance. Save every included and excluded story with a reason code. If fewer than 128 stories are eligible, stop the single-source controller/distribution-claim path until a second training source is added.

### 2.2 Split

- Development: two disjoint subsets. `dev-M2` is the fixed 12 stories that run the four arms; `dev-metric` is a separate set that calibrates margins, ruler range, and human anchors. They must not overlap — the D3 criterion is that the correct-minus-wrong median exceeds the frozen repeatability margin, so estimating that margin from the same stories fits the threshold on the data it tests. Neither subset enters formal test.
- If at least 192 stories are eligible, reserve 64 for formal test and 32 for validation; use the remainder for train/calibration.
- If 128–191 stories are eligible, reserve 48 for formal test and 24 for validation; explicitly weaken harmful-rate precision claims.
- If fewer than 128 stories are eligible, add a second training source before fitting a controller rather than splitting a small recurrence subset three ways.
- No character reference, source video, prompt template instance, or donor may cross train/test boundaries.
- Donor pairing is constructed within split and frozen before generation.

### 2.3 External Distribution

Use ViStoryBench or ST-Bench recurrence cases without tuning formal thresholds. If task adaptation is required, freeze it on development examples only.

## 3. Frozen Generation Contract

For each target recurrence chunk:

1. Generate prefix chunks once.
2. Snapshot final image condition, local context, memory bank, sampler state, prompt embeddings, and reference assets.
3. Draw target initial noise once per generation seed.
4. Restore the exact snapshot for every arm.
5. Change only memory retrieval.

Assert equality across arms for:

- target and negative prompt bytes and SHA256;
- conditional and unconditional text embeddings;
- reference image and prefix state;
- target initial noise;
- sampler, timestep schedule, CFG, resolution, and checkpoint;
- target character name/query and injection layers.

Full rollout trajectories are expected to diverge after the intervention. Only the fixed-trajectory probe holds every latent state constant.

Two assertions make "the intervention happened" checkable at the level that matters:

1. **Decoded divergence.** Correct must differ from no-memory by more than the technical repeat floor, obtained by rerunning one arm at the same seed, condition, and snapshot; run a determinism self-check first, and if that rerun is bit-identical the floor is zero. Cross-seed variance measures generation stochasticity, is far larger, and must not be used here — it belongs to the M3 uncertainty estimate. A fired hook is not a changed output: on 2026-08-08 the injection path was active throughout while zero and no-memory were bit-identical on all ten metrics.
2. **Addressing hit.** For every recurrence event and every arm that should carry memory, the target character resolves to at least one slot; record the matched role and slot count. On an addressing miss a character-addressable reader returns an empty payload, so the correct arm silently degenerates into the zero arm while still being counted as correct.

Zero-versus-no-memory equality is a recorded result, not a contract condition: all-zero K/V leaves a zero residual, so equality is expected and a difference is evidence of an additive or positional bias.

## 4. Interventions

| Arm | Definition | Confirmatory role |
|---|---|---|
| no-memory | Disable SlotMem reader | Baseline |
| zero | Preserve payload structure and zero token values | Mechanism/null control |
| correct | Read the same-story, same-entity payload | Treatment |
| wrong | Replace values with a matched different-entity donor while preserving target query/prompt | Content control |

Row permutation of already encoded slots is excluded from confirmatory analysis because cross-attention is permutation invariant over a K/V set. Any future scramble must demonstrate a non-zero end-to-end output change in a model-level self-check.

## 5. Instrumentation

### 5.1 Provenance

Record code and checkpoint hashes, story/chunk/entity/donor IDs, all condition hashes, memory reads, transformed layers, seeds, hardware, runtime, and output artifacts.

### 5.2 Memory and Read Path

Record slot count/norm/cosine, writer update delta, attention entropy, layer/step hidden delta, target/background energy ratio, and non-target spill.

On a Stage-2 `memory -> update -> target` sample, require a positive writer update count, a finite residual above numerical epsilon, and at least one changed bank hash. Otherwise label the system a static-bank read audit and do not claim dynamic memory.

### 5.3 Fixed-Trajectory Attribution

Cache a no-memory latent trajectory. Re-evaluate the frozen states with correct, wrong, zero, and none conditions. Store per-step/layer/patch denoising-output deltas. The unsigned norm measures influence magnitude only and cannot determine harm/helpfulness. Add a correct-wrong directional projection for development diagnostics and a GT-signed denoising-improvement check on Long-RVOS. These values are attribution features, not utility labels.

## 6. Decoded Outcomes

Primary target outcome: reference-conditioned target identity consistency on the reappearance chunk. Restricting the primary endpoint to the intervened chunk is deliberate — the estimand is the marginal effect of this one read — but harm compounds across chunks, so the same outcome vector on the target+1 chunk is reported as a secondary readout giving a lower bound on delayed harm. If it shows a materially higher harmful rate, the conclusion becomes "per-chunk decisions understate the cumulative cost" rather than the primary number.

Hard quality outcomes:

- prompt/action alignment;
- background preservation;
- motion/dynamic degree;
- temporal flicker;
- chunk-boundary smoothness;
- anatomy/structure quality;
- non-target preservation.

Freeze a W2 metric card before labels: Grounding-DINO referring-expression detection plus SAM2.1 tracking supplies entity masks; DINOv2-L/14 masked-crop cosine measures target identity; GPT-4.1 at temperature zero with a fixed JSON rubric and recorded exact returned model ID measures action/scene/entity-count prompt alignment, with at least 20% blinded-human validation; mean DINOv2 cosine on common valid background patches measures background consistency; VBench measures Motion Smoothness/Dynamic Degree, Temporal Flickering, and Human Anatomy; NarraStream measures Boundary Smoothness; and per-entity masked DINOv2 plus count preservation measures non-targets. Also report official VBench Background Consistency for benchmark comparability. Anatomy is N/A outside valid human-visible cases, never imputed. A VLM may be primary for prompt alignment but may not be the sole source of an overall harmful label.

Dynamic Degree is gated separately from Motion Smoothness and carries an absolute floor, not non-inferiority: a baseline that is already static would let "frozen together" pass, and a frozen clip maximizes smoothness. On 2026-08-05 injection degraded to pure smoothing 62/62 while smoothness metrics improved; SlotMem's own VBench Dynamic Degree spans 0.3913 to 0.9130 across methods, so freezing is a live failure mode in this family.

Before any label exists, establish the identity ruler's RANGE, not only its threshold: the constructive ceiling (reference against itself), the achievable ceiling under real appearance change (a real video's later chunk against its own first frame), and the repeat-noise floor. Report every identity delta and margin as a fraction of that range; without it a delta of 0.01 has no interpretable scale. A component whose range is the same order as its repeat noise is replaced or demoted at W2 rather than explained after the fact. The same positive controls apply to background and non-target preservation.

Freeze smallest effects of interest and non-inferiority margins using `dev-metric` repeatability and blinded human anchors. Label each event helpful, neutral, or harmful using the frozen identity and quality rules.

## 7. Utility Predictor

### 7.1 Inputs

The deployed student uses only pre-target observables already available in native inference: SlotMem slots, target query features, prompt embeddings, attention/relevance summaries, horizon, and memory age/update count. Fixed-trajectory summaries are reserved for an offline/online-teacher ablation because acquiring them costs extra condition forwards.

### 7.2 Outputs

Predict the outcome-delta vector, harmful probability, and predictive uncertainty.

### 7.3 Loss

Use standardized multi-output Huber regression, harmful-event BCE, within-target ranking, and Brier calibration. Wrong donors are a pre-outcome, story-hash-selected 25% stratified hard-negative subset; ranking is training-only and wrong memory is not a deployment action. Fixed-trajectory or latent quantities may be teacher inputs but may not replace decoded targets.

### 7.4 Policy

The confirmatory method is a conservative chunk-level READ/NOOP policy. READ only when the lower confidence bound of identity gain exceeds its smallest effect of interest and every predicted quality delta satisfies non-inferiority.

## 8. Fair Baselines

All baselines operate on the same frozen SlotMem generator and memory bank:

- no-memory;
- all-memory;
- random read with matched read rate;
- recent;
- prompt similarity;
- attention mass;
- character-address-only;
- decoded oracle upper bound;
- online fixed-trajectory teacher;
- distilled utility router.

## 9. Analysis

- Independent unit: story/source.
- Seeds and recurrence events are nested repeated measurements.
- Gate A (no-memory baseline coherence) is decided on frozen qualification seeds that are disjoint from the formal comparison seeds. Deciding it on the formal seed conditions on the realized minuend, removes exactly the events whose no-memory draw was poor, and biases the harmful rate — selection on the dependent variable rather than sample cleaning. With an independent seed it is a story-level pre-stratification variable instead.
- Report both the full eligible population and the Gate-A-qualified population; every harmful rate states which conditional distribution it belongs to, and the two are never merged into one number. Disqualified stories are listed with counts and reasons.
- Primary estimates: story-cluster paired bootstrap 95% CI and paired effect size.
- Proportions: Wilson or Jeffreys 95% CI.
- Secondary outcomes: Holm correction.
- Report raw story counts, missing/failed runs, read rate, and failure reason.
- M2 uses exactly 12 valid development stories and passes screening only if at least 10/12 primary correct-minus-wrong identity contrasts have the favorable sign and the median exceeds the frozen repeatability margin. Passing is not confirmatory evidence.
- A successful policy must exceed the policy SESOI and close at least 25% of the valid decoded-oracle gain over all-memory; report a cluster-bootstrap interval for the closure ratio.
- Formal power: Monte Carlo simulation after the decoded pilot supplies paired SD, event rate, and seed ICC; target 80% power, two-sided alpha 0.05.
- Do not report observed power.

## 10. Efficiency

Measure wall time, peak VRAM, disk/cache size, per-story label cost, inference overhead, and speedup versus full paired counterfactual rollout. Report probe-forward time separately and include it in online-teacher end-to-end latency; the deployed student must not hide a P3 forward. W5 label generation is conditional on the W1 measured per-story and probe costs. Cache all reusable prefixes, slots, embeddings, attribution features, and evaluator inputs.

## 11. Robustness and Ablations

- fixed-trajectory features on/off;
- decoded labels versus latent proxy labels;
- uncertainty/LCB on/off;
- multi-outcome constraints versus identity-only;
- matched versus easy wrong donors;
- native versus identity-light prompts;
- global router versus layer/timestep extension only after global success;
- second data distribution;
- inference-only audit on a second memory system when available.

## 12. Blinding and Human Evaluation

Randomize arm order and left/right presentation. Raters separately judge identity, instruction adherence, motion, visual quality, and non-target preservation. Record inter-rater agreement and ties. Do not use a single VLM as prompt author, sample selector, and final judge.

## 13. Exclusion and Missingness

Exclude only by predeclared technical reasons: corrupted output, failed provenance assertion, absent target character, addressing miss, unreadable reference, or intervention not executed at the decoded level. Keep a failure ledger. Do not exclude a valid run because its effect is small, negative, or visually unattractive.

Gate A disqualification is not an exclusion: those stories stay in the eligible population, are reported alongside the qualified one, and only leave the primary delta statistics.

## 14. Claim Rules

The method claim requires content causality, utility heterogeneity, held-out predictive calibration, policy improvement over all-memory/relevance, quality non-inferiority, and external-distribution direction consistency. Without those, report the strongest lower claim permitted by `plan/stage-gates.md`.
