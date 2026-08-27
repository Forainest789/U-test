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

The deck follows a research-report sequence, with one claim per slide and the original experimental visuals carrying the result pages:

1. **Weekly Lab Meeting — 8.27**: minimal title slide.
2. **Hypothesis**: a valid character-memory mechanism must show measurable influence, correct-content specificity, and decoded identity benefit; timestep 25 / layers 11–15 is the current candidate locus.
3. **Experiment**: Mara-Delta8 uses seven frozen arms after an eight-chunk, 38.31-second gap, with prompt, target, noise, schedule, and teacher-forced timesteps held fixed.
4. **Result I — teacher-forced evidence**: show the five-timestep Q* and injection figure plus the editable numeric table. Memory influence is non-zero at 5/5 timesteps; correct-content specificity is positive at only 1/5.
5. **Result II — decoded evidence**: show the seven-arm keyframes beside the original RGB/RMSE/SNR table supplied by the user. The arms diverge visibly and numerically, but these measures have no identity direction.
6. **Conclusion**: historical memory enters and affects generation, while stable correct-character use and identity benefit remain unproven.
7. **Discussion**: separate observation from interpretation; emphasize that output change is not identity correctness, a single character/seed limits inference, and Evan-Delta5 is inconclusive because its teacher-forced arms did not separate.
8. **Next plan**: rerun Mara under frozen controls and predefined specificity criteria, add independent decoded identity evaluation, then expand characters, scenes, gaps, and seeds before testing dynamic memory.

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

## Review revision: paper-like, evidence-first deck

- Remove the card/module presentation pattern across the deck; use flat figure/table layouts and connected research statements.
- Place the original experiment table and decoded-arm image directly on the result slide instead of translating them into summary cards.
- Keep result-slide prose to one summary sentence, with detailed interpretation deferred to conclusion and discussion.
- Use the sequence Hypothesis → Experiment → Results → Conclusion → Discussion → Next Plan.
