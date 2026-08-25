# Fast Identity-Token Causal Probe for SlotMem

**Date:** 2026-08-25
**Status:** Approved conversational design; awaiting written-spec review
**Scope:** Evaluation and inference tooling only. Do not change SlotMem weights, sampler math, training configuration, or the normal rollout path.

## 1. Objective

Build a fast, fail-closed experiment on the existing SlotMem Q* runtime that answers two primary questions for a long-gap character reappearance event:

1. Does correct historical character memory have a content-specific causal effect beyond no memory and matched-wrong memory?
2. Can a small subset of current video query tokens be identified as identity-bearing by semantic, cross-view, and keep/drop counterfactual evidence?

The primary event is `person_reappearance_delta8`: Mara appears in chunk 0, is absent from chunks 1 through 7, and reappears in chunk 8 under a new action, view, composition, and lighting condition.

Fast inference is an instrument, not the contribution. FlashAttention 2, caching, and skipped decoding make the probe affordable; the scientific outputs are content causality, layer/timestep localization, and a causally validated identity-token subset.

## 2. Claims and Non-Claims

### 2.1 Allowed claims after the corresponding gates pass

- Correct memory has content-specific teacher-forced utility in the tested event and cell.
- Memory utility varies across the tested layer groups and timesteps.
- A stated query-token subset is sufficient for a stated fraction of full correct-memory benefit.
- Dropping that subset is more harmful than dropping equal-budget controls.
- Tokens in a causally positive group are `identity-core candidates` under the operational definition in this spec.

### 2.2 Claims not supported by this experiment

- SlotMem is globally better or worse than Wan2.2.
- Attention weight alone identifies identity.
- A token contains one exclusive human-interpretable concept.
- A single delta-8 event establishes population-level generalization.
- FlashAttention 2 is a research contribution.
- The experiment has trained a deployable memory router.

## 3. Experimental Unit and Frozen Inputs

The independent replication unit is the recurrence event/story. Tokens, latent frames, layers, timesteps, and repeated probes are nested measurements, not independent samples.

A token instance is identified by:

\[
e_i=(\text{event},\tau,\ell,t,h,w).
\]

Every confirmatory comparison freezes and hashes:

- prefix snapshot and memory bank;
- target prompt and text embeddings;
- target image condition and image embeddings;
- held-out clean target latent;
- Gaussian noise, noisy latent, and scheduler timestep;
- target character and matched-wrong donor payload;
- model weights, dtype, attention backend, and runtime configuration.

The held-out target remains arm-independent. No generated arm may serve as its own teacher.

## 4. Primary Estimands

For conditional flow-matching loss `L`:

\[
Q_{\mathrm{presence}}=L_{\mathrm{no-memory}}-L_{\mathrm{correct}},
\]

\[
Q_{\mathrm{content}}=L_{\mathrm{wrong}}-L_{\mathrm{correct}}.
\]

`Q_presence` asks whether memory helps. `Q_content` asks whether the benefit depends on the correct character content. `Q_content` is the load-bearing identity estimand.

Prediction change without loss improvement is influence, not utility. Correct and wrong memory that yield indistinguishable outcomes do not establish identity-specific reading.

## 5. Sequential Experiment

The experiment stops at the first failed gate.

### 5.1 S0: Runtime contract and acceleration calibration

Load the model once. Run the middle timestep with all-layer injection for:

- `correct`;
- `correct_repeat`;
- `zero`;
- `no_memory`.

Require:

- exact frozen-input hashes across arms;
- correct-repeat within the existing loss and influence tolerances;
- target payload hit for payload-bearing arms;
- at least one enabled injection layer with selected memory tokens and positive finite effective delta;
- no-memory taking the same custom forward with an empty payload;
- finite loss and prediction.

Record wall time after one warm-up forward, peak allocated/reserved VRAM, and effective attention implementation.

### 5.2 S1: Content causality and coarse layer/timestep screening

Use scheduler indices `0,25,49`, resolved against the frozen 50-step schedule:

- high noise: index 0;
- middle noise: index 25;
- low noise: index 49.

Use injection layer groups:

- early: `0-4`;
- middle: `5-10`;
- late: `11-15`.

For every timestep, compute `no_memory` once. For every layer group, compute `correct` and `matched-wrong`. At the middle timestep, also retain the current all-layer `correct/wrong` reference. S0 results are reused where conditions are identical.

The intended S0+S1 budget is about 25 measured forwards, excluding warm-up. Do not decode video and do not run a scheduler step.

For every cell record:

- conditional loss and prediction hash;
- `Q_presence` and `Q_content`;
- normalized prediction influence;
- selected query and memory token counts;
- raw and effective injection delta norm;
- host token norm and effective-delta/host ratio;
- attention entropy and winner counts;
- per-layer diagnostics and timing.

Rank cells primarily by positive `Q_content` above the repeat/benefit margin, then by localized effective-delta/host ratio. Select one primary cell and at most one validation cell. If no cell has content-specific influence above the technical floor, stop before token typing.

### 5.3 S2: Identity-token proposal and causal confirmation

Run full proposal and group knockout only in the primary cell. In the validation cell, rerun only the final equal-budget identity set and controls.

The primary cell performs:

- original role probe plus a Mara-token-drop branch;
- one diagnostic semantic capture with stable attributes, action groups, and scene groups;
- action-token-drop, scene-token-drop, and equal-count random-text-drop controls;
- up to eight spatial group knockouts;
- equal-budget identity keep/drop and control interventions.

The validation cell does not repeat diagnostic decomposition or eight group knockouts. It tests whether the frozen identity set has the same directional effect.

The intended total through S2 is no more than about 50 single-step measured forwards plus a small number of unmeasured diagnostic forwards.

### 5.4 S3: Decoded validation only after S2 passes

Run the target chunk only for:

- `full_correct`;
- `no_memory`;
- `identity_only`;
- `drop_identity`.

Reuse the same prefix, prompt, target seed, and runtime contract. Decoded identity is primary; prompt/action, motion, anatomy, flicker, and non-target/background preservation are hard gates. S3 is optional during development and required before a decoded identity claim.

## 6. Token Evidence

All raw token values are converted to robust percentile ranks in `[0,1]` within one `event x timestep x layer` cell. Use deterministic average ranks for ties; map a singleton finite value to `0.5`. Raw values are retained. Scores are not compared across layers before this normalization.

### 6.1 Name specificity

With the same noisy latent and prompt sequence layout, compare the normal role probe to a branch in which only the `Mara` text positions are masked from cross-attention:

\[
r_i=\operatorname{ReLU}(A_i^{\mathrm{normal}}-A_i^{\mathrm{drop-Mara}}),
\]

\[
s_{\mathrm{name},i}=\operatorname{PercentileRank}(r_i).
\]

Keep sequence length and position IDs fixed. Rewriting the prompt is not an equivalent intervention.

### 6.2 Stable-attribute evidence

The frozen diagnostic prompt contains Mara's source description:

- short copper bob;
- teal scarf;
- crescent-shaped scar above the left eyebrow;
- mustard raincoat.

Capture all attribute token groups in one forward. Normalize each group by its number of text tokens. Define:

\[
s_{\mathrm{attr},i}=\operatorname{rank}\left(\operatorname{median}_{a\in\mathcal A} A_i^a\right).
\]

The diagnostic prompt proposes candidates only. It never supplies the measured Q* outcome.

### 6.3 Action evidence

Split the target action into three groups:

- locomotion core: `runs`, `two steps`;
- interaction core: `catches`, `closing`; context: `tram door`, `one hand`;
- gaze core: `looks up`, `toward camera`.

For group `g`, average over heads and divide by the number of text positions:

\[
A_i^g=\frac{1}{H|T_g|}\sum_{h,r\in T_g}\operatorname{Attn}_{h,i,r}.
\]

The attention proposal is:

\[
a_i^{\mathrm{attn}}=\max_g\operatorname{rank}(A_i^g).
\]

Keep the prompt layout fixed and mask action-core text positions to obtain:

\[
D_i^{\mathrm{action}}=
\frac{\|h_i^{\mathrm{full}}-h_i^{\mathrm{drop-action}}\|_2}
{\|h_i^{\mathrm{full}}\|_2+\epsilon}.
\]

Compute equal-count scene-drop and frozen random-text-drop controls:

\[
D_i^{\mathrm{net-action}}=
\operatorname{ReLU}\left[D_i^{\mathrm{action}}-
\max(D_i^{\mathrm{scene}},D_i^{\mathrm{random}})\right].
\]

Then:

\[
s_{\mathrm{action},i}=
\sqrt{\operatorname{rank}(a_i^{\mathrm{attn}})
\operatorname{rank}(D_i^{\mathrm{net-action}})}.
\]

Optionally record temporal concentration over latent frames, but do not use it in the hard label because camera and subject motion can break fixed-position correspondence.

### 6.4 Scene evidence

Use the diagnostic text groups `platform`, `tram`, `rain`, `commuters`, and `dusk`. Normalize by text-token count and store the maximum ranked group response as `s_scene`. Scene evidence is a competing/mixed label, not a subtraction from identity evidence.

### 6.5 Cross-view persistence

Use both the raw feature space and the learned read space:

\[
p_i^{\mathrm{raw}}=
\max_j\cos(h_i,S_j^{\mathrm{Mara}})-
\max_j\cos(h_i,S_j^{\mathrm{wrong}}),
\]

\[
p_i^{\mathrm{read}}=
\operatorname{LSE}(q_iK_{\mathrm{Mara}}^\top)-
\operatorname{LSE}(q_iK_{\mathrm{wrong}}^\top).
\]

Define `s_persist` as the mean of the two percentile ranks. Also store content-specific injection influence:

\[
d_i=
\frac{\|\delta_i^{\mathrm{correct}}-\delta_i^{\mathrm{wrong}}\|_2}
{\|h_i\|_2+\epsilon}.
\]

`d_i` is an influence diagnostic and a tie-breaker, not utility.

### 6.6 Group-causal evidence

For token group `g`:

\[
C_g=
\frac{L_{\mathrm{drop}(g)}-L_{\mathrm{full-correct}}}
{L_{\mathrm{no-memory}}-L_{\mathrm{full-correct}}+\epsilon}.
\]

All tokens in a tested group receive the group's causal score. The schema and plots must call it `group_causal_score`; they must not imply per-token Shapley attribution.

## 7. Proposal Score and Candidate Universe

Before causal confirmation:

\[
S_i^{\mathrm{pre}}=
\sqrt[3]{s_{\mathrm{name},i}s_{\mathrm{attr},i}s_{\mathrm{persist},i}}.
\]

Use `d_i` only to break equal ranks. Do not subtract action or scene evidence because tokens may be genuinely mixed.

The candidate universe is:

\[
\mathcal U=\mathcal M_{\mathrm{probe}}\cup
\operatorname{Top10\%}(s_{\mathrm{persist}}).
\]

This union allows the experiment to discover tokens missed by the current semantic mask.

## 8. Deterministic Candidate Groups

Map each candidate to `(latent_t,h,w)`. Build spatial-temporal adjacency with:

- eight-neighborhood connections within a latent frame;
- the same position and one-cell neighborhood in adjacent latent frames.

Create connected components. First merge undersized components into the nearest centroid. If more than eight components remain, repeatedly merge the smallest component into its nearest centroid until eight remain. If fewer than eight remain, recursively split the largest splittable component at the median of its widest normalized `t/h/w` coordinate, stopping at eight groups or when another split would create a group with fewer than four tokens. Resolve all ties by lexicographic `(t,h,w)` order. Group construction uses coordinates only, never Q* outcomes or final causal scores.

Archive exact membership, split decisions, configuration, and a SHA256 hash.

## 9. Equal-Budget Query Interventions

Let the original SlotMem query mask contain `N` tokens and set:

\[
K=\max(4,\lceil0.25N\rceil).
\]

The S2 confirmatory intervention universe is the frozen expanded candidate universe `U`, and `full_correct` means correct memory injected at every position in `U`. Recompute this S2 full baseline once; do not silently reuse an S1 full baseline with a different query mask. `U` contains the original mask, while the budget remains tied to the original system footprint `N`.

Freeze:

- `identity_top_K`: highest `S_pre`;
- `random_K`: seeded random tokens matched to the identity set's per-frame counts;
- `low_score_K`: lowest scores with the same per-frame counts;
- `drop_identity_K`: the full candidate universe without identity top-K;
- `wrong_identity_K`: identity top-K positions reading matched-wrong memory;
- `drop_random_K` and `drop_low_K`: equal-budget necessity controls.

The four size-`K` keep/read arms must have the same count and frame histogram. The three drop arms must each remove exactly `K` positions with the same removal histogram from `U`. A mismatch is a contract failure, not a warning.

## 10. Labels and Gates

### 10.1 Full-memory prerequisite

\[
B_{\mathrm{full}}=L_{\mathrm{no}}-L_{\mathrm{full-correct}}
\]

must exceed the frozen benefit margin. Otherwise no causal token labels are emitted.

### 10.2 Set-level sufficiency and necessity

\[
R_{\mathrm{keep}}=
\frac{L_{\mathrm{no}}-L_{\mathrm{identity-only}}}
{L_{\mathrm{no}}-L_{\mathrm{full-correct}}},
\]

\[
R_{\mathrm{drop}}=
\frac{L_{\mathrm{drop-id}}-L_{\mathrm{full-correct}}}
{L_{\mathrm{no}}-L_{\mathrm{full-correct}}}.
\]

The development success gate is:

- identity set uses at most 25% of the original query-token count;
- `R_keep >= 0.8`;
- identity drop is more harmful than random and low-score drops beyond the repeat margin;
- correct identity-only is better than wrong identity-only beyond the benefit margin;
- the direction agrees in the validation cell.

These are screening thresholds, not universal constants.

### 10.3 Token/group labels

`identity-core candidate` requires:

- `s_name >= 0.75`;
- `s_attr >= 0.75`;
- `s_persist >= 0.75`;
- its group-causal score exceeds equal-budget controls and the repeat margin;
- correct top-K beats wrong top-K;
- direction agrees in the validation cell.

`identity-associated` has at least two high identity proposal channels but lacks causal confirmation.

`attention-only/redundant` has high name or action attention but low content-specific delta and no group knockout effect, or correct and wrong memory are indistinguishable.

`action-associated` requires `s_action >= 0.75`, action-drop influence above scene/random text-drop, and a concrete action-core group rather than context nouns alone.

`scene-associated` has dominant scene evidence without identity causality.

`mixed` is multi-label, for example `identity-core candidate + action-associated`. The first experiment makes a formal claim only about identity labels; other labels remain diagnostic until separately validated.

After causal confirmation, define the non-negative group display term

\[
s_{\mathrm{causal},g}=\operatorname{rank}(\operatorname{ReLU}(C_g)),
\]

and a display score may be computed as:

\[
S_i^{\mathrm{final}}=\sqrt{S_i^{\mathrm{pre}}s_{\mathrm{causal},g(i)}}.
\]

Hard labels continue to use the conjunctive rules above, not the display score alone.

## 11. Runtime Design

### 11.1 Server target

- NVIDIA A100 80GB;
- BF16 inference;
- `torch.inference_mode()`;
- no model offload for the timed path;
- no gradients or optimizer state;
- no VAE decode in S0-S2.

### 11.2 FlashAttention 2

The vendored Wan attention already selects FlashAttention 2 when the package is installed. The runner sets:

```text
DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_2
SLOTMEM_OFFLOAD_MODELS=0
```

and fails closed if the runtime manifest does not report FlashAttention 2, unless an explicit development-only fallback flag is supplied.

The custom `CharacterWiseCrossAttention` currently uses explicit softmax/einsum. Do not rewrite it in the first implementation: its memory axis is small, while model loading and repeated Wan forwards dominate. Profile it and reconsider only if it exceeds 10% of measured probe wall time.

### 11.3 Cache boundaries

Safe immutable caches:

- prefix and memory payloads;
- target clean/noisy latents and noise;
- prompt and image embeddings;
- one current-step query payload per frozen timestep and probe configuration;
- stable-attribute/action/scene diagnostic token indices;
- source and wrong-donor projected memory K/V per layer;
- baseline no-memory result per timestep;
- S1 cells reused by S2 when hashes match.

Do not initially add resumable mid-DiT hidden-state caching. It changes the forward boundary and creates a larger correctness burden than its likely first-pass speed benefit. Do not cache across different prompts, timesteps, layers, weights, or attention backends.

Every cache item carries a versioned key and input hash. A mismatch recomputes rather than silently reusing.

## 12. Components and Interfaces

Reuse existing Q* predictor, prefix contract, arm payloads, and memory-aware forward. Add the minimum bounded components:

1. A pure token-scoring/grouping module with no model ownership.
2. A probe orchestrator that loads the model once, schedules S0-S2, and writes records.
3. A thin shell runner for the server and dry-run contract.
4. A small extension to expose per-query raw/normalized delta and the hidden/token maps required by scoring.

The proposed command surface is:

```text
python -m utest.identity_token_probe \
  --prefix PREFIX \
  --future-target-video FUTURE_TARGET_VIDEO \
  --donor DONOR_PAYLOAD \
  --donor-manifest DONOR_MANIFEST \
  --output OUTPUT \
  --timestep-indices 0,25,49 \
  --layer-groups 0-4,5-10,11-15 \
  --max-groups 8 \
  --identity-budget 0.25 \
  --run-decoded-validation false
```

The server shell wrapper accepts the same frozen inputs as `run_slotmem_qstar_event.sh`, requires a fresh output root, records every argv, and supports `DRY_RUN=1`.

## 13. Outputs

Write a versioned, self-contained run directory:

```text
identity_probe/
  runtime_manifest.json
  input_contract.json
  screening_cells.jsonl
  selected_cells.json
  diagnostic_prompt_manifest.json
  token_scores.jsonl
  token_groups.json
  interventions.jsonl
  identity_probe_report.json
  figures/
    layer_timestep_qcontent.png
    token_type_maps.png
    group_causal_map.png
```

Each token record contains:

- event/timestep/layer and flat/`t,h,w` indices;
- raw and ranked semantic channels;
- raw/read persistence margins;
- content-specific delta/host ratio;
- proposal score;
- group ID and group-causal score;
- multi-label result and every threshold decision.

The summary report includes explicit PASS/BLOCK/PENDING gates, forward counts, cache hits, timing, peak VRAM, and reasons. It must remain useful without the figures.

## 14. Failure Policy

Fail closed when:

- any frozen input hash differs across a confirmatory comparison;
- a payload-bearing arm does not perform positive finite measured injection;
- the matched-wrong donor is schema-incompatible;
- losses, hidden states, or scores are non-finite;
- token masks differ in count/frame histogram beyond the declared matching rule;
- the original query count or candidate count is below four;
- a cached artifact key does not match its current inputs;
- fewer than three stable attributes have valid text token positions;
- full correct-memory benefit does not exceed the frozen margin.

If attributes are missing, output `identity-candidate` at most, never `identity-core candidate`. If correct and wrong are indistinguishable, output a content-insensitive result and stop.

## 15. Verification

Add focused runnable checks for:

- percentile normalization and all score equations;
- name/action/scene text-position masking with unchanged sequence length;
- action score subtracting scene/random controls;
- deterministic spatial-temporal grouping and group hashes;
- recursive group splitting and minimum group size;
- equal-budget and per-frame mask matching;
- group-causal, `R_keep`, and `R_drop` arithmetic;
- label boundaries and missing-channel downgrades;
- cache-key invalidation;
- S0/S1 forward schedule and reuse count;
- fail-closed injection diagnostics;
- shell dry-run command resolution;
- S0-S2 never invoking VAE decode or a scheduler step.

Run existing Q* and inference-hotpath tests as regression checks. The first GPU smoke test uses one middle timestep, one layer group, four groups, and no decoded validation. Only after it passes does the server run the full S0-S2 matrix.

## 16. Server Execution Sequence

1. Freeze source commit, weights, input manifests, and environment package versions.
2. Verify FlashAttention 2 import and runtime selection on the A100.
3. Run CPU/unit checks and shell dry-run.
4. Run the one-cell GPU smoke test.
5. Inspect contract, repeatability, injection, timing, and VRAM.
6. Run S0-S1.
7. Stop if no content-specific cell passes.
8. Run primary-cell S2 and validation-cell set confirmation.
9. Stop if identity sufficiency/necessity fails.
10. Run four decoded S3 arms only when requested.
11. Archive raw records, report, commands, and environment manifest together.

## 17. Extension Boundary

If identity-core candidates pass S2/S3 across multiple independent events, a later design may extend the diagnostic vocabulary and causal controls to pose, action, background, camera, and transition tokens, then train a lightweight deployed router. That work is outside this implementation.
