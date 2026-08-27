# Weekly Lab Meeting 8.27 presentation design

Date: 2026-08-27  
Status: approved narrative; ready for production

## Communication job

By the end of the lab meeting, the research group should understand that this week's work produced one bounded mechanism candidate and, more importantly, converted the investigation into a reproducible multi-event causal validation pipeline with a clear GPU execution gate.

## Source and visual contract

- Use `C:/Users/93547/Desktop/Weekly Lab Meeting 0806.pptx` as the exact visual reference.
- Preserve its 8-slide, English, conclusion-first presentation style.
- Duplicate source slides and edit inherited elements in place; preserve typography, spacing, palette, and master/layout hierarchy.
- Use only local repository evidence and frozen run artifacts. Do not present planned GPU runs as completed results.

## Narrative

1. **Weekly Lab Meeting — 8.27**: minimal title slide.
2. **This week in one sentence**: one bounded identity-memory mechanism candidate plus a ready multi-event validation pipeline.
3. **The mechanism candidate is narrow**: summarize the 41-arm fusion verification and the timestep 25 / layers 11–15 correct-memory candidate.
4. **What the candidate means—and does not mean**: contrast the timestep 25 candidate with timestep 49 overload and state the single-event, single-seed, teacher-forced evidence boundary.
5. **The evaluation moved from one event to a 3 × 3 matrix**: introduce the three frozen ViStoryBench reappearance events and three seeds.
6. **The new pipeline tests subject causality, not correlation**: show the eight frozen subject-subspace interventions, source-only mask discovery, and reference-leakage controls.
7. **Implementation is ready; experiment evidence is not yet complete**: report 51 commits, 54 changed files, and the focused 128/128 passing tests, while separating these engineering gates from pending GPU execution.
8. **Next step: execute sequentially and stop on failed gates**: run nine-block source qualification and preflight, then full eight-arm decoded validation and CIDS, and decide by sufficiency, necessity, specificity, and quality non-inferiority.

## Evidence boundaries

- The fusion verification result is a mechanism candidate, not a decoded identity improvement claim.
- Timestep 49 supports an overload interpretation but not confirmed identity-specific representation competition.
- The ViStoryBench workflow is implemented and zero-GPU tested; the 9-block GPU matrix and decoded CIDS results are still pending.
- Q* remains descriptive and unavailable without an independently sourced teacher video.

## Primary local sources

- `docs/2026-08-26-slotmem-fusion-verification-results.md`
- `docs/2026-08-26-slotmem-injection-stability-research.md`
- `docs/2026-08-27-reappearance-benchmark-candidates.md`
- `docs/superpowers/specs/2026-08-27-vistorybench-subject-reappearance-design.md`
- `utest/README.md`
- `runs/identity_fusion_verify_20260826T155804Z/`
- Git history from 2026-08-21 through 2026-08-27

## Acceptance checks

- Eight slides, with every output slide mapped to a source slide.
- No unsupported result claim or unresolved placeholder.
- No unintended overlap, clipping, or title wrapping.
- Final deck renders cleanly slide by slide and passes template-fidelity and overflow checks.
