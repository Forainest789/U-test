# CVPR Stage Gates

## D0 — Protocol Locked

Required:

- eligible-story audit and exclusion codes;
- source-disjoint split and donor table;
- development budget split into two DISJOINT subsets: `dev-metric` (calibrates margins, ruler range, human anchors) and `dev-M2` (the fixed 12 stories that run the four arms). Sharing them fits the threshold on the data it then tests: D3 passes when the correct-minus-wrong median exceeds the frozen repeatability margin, and that margin would be estimated from the same stories' variability;
- qualification seeds for Gate A, frozen and disjoint from the formal comparison seeds;
- prompt/prefix/noise/checkpoint hash schema;
- primary endpoint, margins, seeds, baselines, and analysis unit;
- hardware/software manifest;
- no formal-test access.

Eligibility is run in W1 before M1–M3 expansion. Fewer than 128 eligible stories blocks the single-source controller plus formal-distribution claim until a second training source is secured.

## D1 — Native SlotMem Reproduced

Required:

- official sample output;
- non-empty memory reads;
- checkpoint and commit hashes;
- measured wall time, VRAM, and disk footprint.
- before execution, the paper anchor is frozen as NarraStream Subject Consistency 0.8771; a numeric reproduction claim requires the same evaluation inputs/protocol and either absolute deviation at most 0.02 or a 95% CI covering 0.8771. Otherwise mark it non-comparable rather than passing on a sample video.

Failure blocks all later GPU work.

## D2 — Intervention Contract Valid

Required:

- all non-memory hashes identical across arms;
- correct/wrong/zero transformations executed;
- decoded output actually diverges: correct differs from no-memory by more than the TECHNICAL repeat floor, measured by rerunning one arm at the same seed, condition and snapshot. A fired hook is not a changed output — on 2026-08-08 the injection path was active for every step while zero and no-memory were bit-identical on all ten metrics. Cross-seed variance is generation stochasticity and must not be used as this floor;
- target character resolves to at least one slot in every arm that should carry memory, recorded per event. A character-addressable reader returns an empty payload on an addressing miss, which silently turns the correct arm into the zero arm while it is still counted as correct;
- zero-versus-no-memory equality is recorded as a measured result, not required: all-zero K/V leaves a zero residual, so equality is the analytic expectation and a difference is evidence of an additive/positional bias;
- no-memory reader disabled;
- end-to-end self-check detects zero/wrong changes;
- row-scramble removed from confirmatory gates unless proven non-NOOP.
- on an eligible Stage-2 update sample, writer update count is positive, residual is finite and above numerical epsilon, and the bank hash changes at least once.

## D2.5 — Measurement Range Established

Frozen in W2, before any label is defined. Required:

- the eight-component metric card: implementation, version, preprocessing, masks, direction, and missing-value rule;
- identity ruler RANGE, not just a threshold: the constructive ceiling (reference against itself), the achievable ceiling under real appearance change (a real video's later chunk against its own first frame), and the repeat-noise floor. Every identity delta and margin is reported as a fraction of that range. A component whose range is the same order as its repeat noise lacks the resolution to be a primary endpoint and is replaced or demoted now, not explained away after M3;
- Dynamic Degree carries its own ABSOLUTE floor, separate from Motion Smoothness. Smoothness may use non-inferiority; dynamic degree may not, because a baseline that is already static makes "frozen together" pass. A frozen clip maxes out smoothness — on 2026-08-05 injection degraded to pure smoothing 62/62 while smoothness metrics improved, and SlotMem's own VBench Dynamic Degree spans 0.3913 to 0.9130 across methods;
- blinded human anchors on at least 20% of `dev-metric` pairs, with agreement reported.

## D3 — Content-Causal Read Path

Required on exactly 12 valid `dev-M2` development stories:

- Gate A — the no-memory arm is itself a legitimate baseline. On the target chunk it must clear absolute quality floors: subject detectable, dynamic degree above the frozen floor, flicker and anatomy within frozen ranges, no subject substitution inside the chunk. A delta against a collapsed baseline is not utility; MVP1 recorded that no-memory is a degenerate floor, and a degenerate floor inflates every delta while hiding the harmful tail.
  Gate A is decided on the frozen QUALIFICATION seeds, never on the formal outcome seed. Deciding it on the same seed conditions on the realized minuend, drops exactly the events whose no-memory draw was poor, and distorts the harm rate — selection on the dependent variable, not sample cleaning. Report both the full eligible population and the Gate-A-qualified population, label every harmful rate with which conditional distribution it belongs to, and list the disqualified stories with reasons rather than dropping them silently;
- non-zero localized hidden/output effect;
- correct distinguishable from matched wrong;
- target-region concentration above background;
- decoded direction not reducible to zero/null effect;
- at least 10/12 primary correct-minus-wrong identity contrasts have the favorable sign and their median exceeds the frozen repeatability margin.

This is a one-way screening gate: failure stops utility learning; success is not confirmatory evidence and development stories do not enter formal results.

## D4 — Decoded Utility Estimand Exists

Required:

- helpful and harmful/neutral cases are both present beyond metric noise;
- all-memory and no-memory do not trivially dominate every story;
- automatic harm labels agree with blinded human anchors;
- pilot variance supports a formal sample-size simulation.

## D5 — Utility Predictor Generalizes

Required:

- held-out story/source evaluation;
- harm calibration better than relevance baselines;
- outcome-delta ranking and error reported with CI;
- no donor/story/template leakage;
- model and threshold frozen before formal test.

## D6 — Policy Improves Over All-Memory

Required:

- primary paired CI favors utility routing or harmful rate is significantly reduced;
- all quality non-inferiority gates pass;
- better than random/recent/similarity/attention routing;
- non-degenerate READ rate;
- latency/VRAM reported.
- primary improvement exceeds the frozen policy SESOI;
- student closes at least 25% of a valid decoded-oracle gain over all-memory, with a story-cluster bootstrap interval.

This is the minimum CVPR method gate.

## D7 — Generalization

Required:

- direction-consistent result on one external narrative distribution;
- second memory-system audit, or explicit claim restriction to SlotMem/Wan2.2;
- failure cases and distribution limitations reported.

## D8 — Table/Figure Data Contract

Every result family has raw logs, aggregation code, table schema, figure-manifest entry, and caption. Mock planning values are forbidden in submission prose.

## D9 — Claim and Paper Review

Required:

- contribution-to-evidence traceability complete;
- no latent quantity named utility;
- no unsupported “first” or universal memory claim;
- formal-test results reproduced from frozen manifests;
- internal review of main paper and supplement.
