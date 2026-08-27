# Weekly Lab Meeting 8.27 presentation design

Date: 2026-08-27  
Status: approved narrative; Mara update pending production

## Communication job

By the end of the lab meeting, the research group should understand that the candidate memory path can measurably influence generation, while correct-character specificity and identity benefit remain unproven, and should agree on the next experiments needed to separate these claims.

## Source and visual contract

- Use `C:/Users/93547/Desktop/Weekly Lab Meeting 0806.pptx` as the exact visual reference.
- Preserve its 8-slide, English, conclusion-first presentation style.
- Duplicate source slides and edit inherited elements in place; preserve typography, spacing, palette, and master/layout hierarchy.
- Use only local repository evidence, frozen run artifacts, and the Mara/Evan dual-experiment report. Do not present descriptive output differences as identity improvement.
- Remove engineering progress, implementation readiness, commit counts, test counts, and environment issues from the audience-facing deck.

## Narrative

1. **Weekly Lab Meeting — 8.27**: minimal title slide.
2. **The memory path works, but identity utility is not yet established**: combine the existing mechanism candidate with the new Mara result and state the bounded conclusion.
3. **Mechanism candidate: timestep 25, layers 11–15**: retain the local candidate as a hypothesis about where correct-memory information may enter the model.
4. **Mara-Delta8 uses a seven-arm frozen-control design**: show the chunk-0 memory source, the eight-chunk gap, the chunk-8 target, and the correct/repeat/no-memory/zero/random/wrong/native comparisons.
5. **Mara memory influences generation across all five timesteps**: show non-zero injection at 5/5 timesteps, increasing from 0.000404 to 0.017253, plus structured decoded-output differences.
6. **Injection is measurable; content specificity is unstable**: contrast the non-zero influence with mean Q* = -0.000668, positive Q* at 1/5 timesteps, and correct beating both wrong and random at only 1/5 timesteps.
7. **Validation must pass three separate gates**: influence, correct-content specificity, then identity utility; use the existing causal-validation work only as the experimental framework for these gates.
8. **Next experiments are defined by decision criteria**: rerun Mara under fully frozen controls, add direct identity evaluation, then expand across characters, scenes, gaps, and seeds. Mention Evan only as an inconclusive replication because it produced no measurable treatment difference.

## Evidence boundaries

- Non-zero injection and RGB/SNR differences establish influence, not correct-character use or identity improvement.
- Mara-Delta8 supports the basic injection claim but does not establish stable content specificity: mean Q* is negative and the correct arm has a joint advantage over wrong/random at only 1/5 timesteps.
- RGB and SNR have no identity direction; decoded differences remain descriptive until an independent identity metric or blinded human evaluation is added.
- Evan-Delta5 is inconclusive because the teacher-forced arms produced no measurable treatment difference; it is neither positive evidence nor a negative result about SlotMem capability.
- Dynamic memory updating was not evaluated because both experiments used frozen prefix memory.

## Primary local sources

- `docs/2026-08-26-slotmem-fusion-verification-results.md`
- `docs/2026-08-26-slotmem-injection-stability-research.md`
- `docs/2026-08-27-reappearance-benchmark-candidates.md`
- `docs/superpowers/specs/2026-08-27-vistorybench-subject-reappearance-design.md`
- `D:/xwechat_files/wxid_7663186635312_c879/msg/file/2026-08/slotmem_character_memory_injection_report.pdf`
- `utest/README.md`
- `runs/identity_fusion_verify_20260826T155804Z/`
- Git history from 2026-08-21 through 2026-08-27

## Acceptance checks

- Eight slides, with every output slide mapped to a source slide.
- Mara result is the main new evidence, with the distinction between influence, specificity, and identity utility explicit.
- Engineering progress and engineering problems are absent from visible slide content.
- No unsupported result claim or unresolved placeholder.
- No unintended overlap, clipping, or title wrapping.
- Final deck renders cleanly slide by slide and passes template-fidelity and overflow checks.

## Review revision: evidence-first summary slide

- Replace slide 2's four metric cards with one compact Mara timestep-results table and one decoded-arm keyframe comparison taken directly from the experiment report.
- Keep only a short explanation below the evidence: injection is measurable, while stable correct-content and identity benefit remain unproven.
- Preserve slides 1 and 3-8 unchanged.
