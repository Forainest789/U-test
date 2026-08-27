# Weekly Lab Meeting 8.27 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an eight-slide English weekly lab meeting deck that inherits the 0806 deck exactly and explains the mechanism candidate, the new Mara-Delta8 findings, their evidence boundary, and the next experiments.

**Architecture:** Reuse the existing duplicate-only starter deck and edit inherited text, table, and image elements in place with `@oai/artifact-tool`. Use the frozen fusion-verification run and the Mara/Evan dual-experiment PDF as evidence. Render and inspect every slide before delivery.

**Tech Stack:** PowerPoint `.pptx`, JavaScript ES modules, `@oai/artifact-tool`, bundled presentation inspection/fidelity helpers.

## Global Constraints

- Use `C:/Users/93547/Desktop/Weekly Lab Meeting 0806.pptx` as the visual source.
- Keep exactly eight slides and preserve the source master, layouts, typography, spacing, and palette.
- Visible slide copy is English, simple, scientific, and audience-facing.
- Remove implementation metrics, engineering readiness, and engineering problems from visible content.
- Separate three claims: measurable influence, correct-content specificity, and decoded identity utility.
- Do not claim identity improvement from non-zero injection, Q*, RGB differences, or SNR alone.
- Use a flat, evidence-first research sequence: Hypothesis → Experiment → Results → Conclusion → Discussion → Next Plan.
- Remove card/module layouts from the audience-facing deck; result slides are dominated by original tables and figures with one summary sentence.

---

### Task 1: Update the evidence ledger and slide mapping

**Files:**
- Create: `.codex/weekly-report-2026-08-27/template-audit.txt`
- Create: `.codex/weekly-report-2026-08-27/template-frame-map.json`
- Create: `.codex/weekly-report-2026-08-27/source-notes.txt`

**Interfaces:**
- Consumes: the inspected source deck, existing frame map, fusion run, and Mara/Evan report.
- Produces: an exact source-slide mapping and claim-to-source ledger for authoring.

- [ ] Preserve the established mapping to source slides 1, 2, 4, 5, 7, 6, 7, and 8 unless a Mara visual cannot fit an inherited frame.
- [ ] Record the Mara design, five-timestep injection values, Q* summary, RGB/SNR limitations, and next-step gates in the source ledger.
- [ ] Mark the engineering-readiness slide content for complete audience-facing replacement.
- [ ] Validate the frame map with the bundled template starter tool.

### Task 2: Prepare the inherited edit base

**Files:**
- Create: `.codex/weekly-report-2026-08-27/template-starter.pptx`

**Interfaces:**
- Consumes: `template-frame-map.json` and the 0806 source deck.
- Produces: an eight-slide duplicate-only authoring base.

- [ ] Run the artifact-operation marker exactly once for one PPTX edit.
- [ ] Reuse the validated duplicate-only starter deck; regenerate it only if the mapping changes.
- [ ] Verify every output slide still matches its mapped source pattern before content edits.

### Task 3: Edit inherited elements and export the weekly report

**Files:**
- Create: `.codex/weekly-report-2026-08-27/build-weekly-report.mjs`
- Create: `Weekly Lab Meeting 0827.pptx`

**Interfaces:**
- Consumes: `template-starter.pptx`, inherited element IDs, and local evidence.
- Produces: the final editable eight-slide PowerPoint deck.

- [ ] Import the starter deck with `PresentationFile.importPptx`.
- [ ] Rewrite only mapped inherited text, table, and image elements to implement the approved eight-slide Mara narrative.
- [ ] Use the report's Mara figures or cropped report visuals only where they directly support the slide claim.
- [ ] Add `[Sources]` speaker-note blocks for the fusion candidate, Mara design/results, evidence boundaries, and next-step gates.
- [ ] Export a new copy named `Weekly Lab Meeting 0827 Mara.pptx` with `PresentationFile.exportPptx`.

### Task 4: Verify every slide and template fidelity

**Files:**
- Create: `.codex/weekly-report-2026-08-27/final-render/`
- Create: `.codex/weekly-report-2026-08-27/qa-ledger.txt`

**Interfaces:**
- Consumes: the final PPTX, starter PPTX, frame map, and slide renders.
- Produces: a checked final deck with no unresolved placeholders or unintended layout defects.

- [ ] Render all eight final slides and inspect each at full size.
- [ ] Run overflow, placeholder, and template-fidelity checks.
- [ ] Confirm no engineering metrics or engineering problems remain in visible content.
- [ ] Fix clipping, wrapping, overlap, unsupported claims, or source-note gaps.
- [ ] Re-render and repeat checks until all gates pass.

### Task 5: Apply the evidence-first slide 2 review revision

**Files:**
- Modify: `.codex/weekly-report-2026-08-27-mara/build-weekly-report-mara.mjs`
- Modify: `Weekly Lab Meeting 0827 Mara.pptx`

**Interfaces:**
- Consumes: Mara Table 3 and Figure 4 from the experiment report, plus the existing slide 2 title and explanation frames.
- Produces: a slide 2 led by the original experimental table and decoded-arm image rather than metric cards.

- [ ] Crop the Mara rows from Table 3 and the seven-arm decoded keyframe figure without redrawing either source.
- [ ] Delete the inherited metric-card content inside slide 2's evidence region and place the table and image in bounded source-layout zones.
- [ ] Reduce the explanation to two evidence-bound sentences.
- [ ] Re-export, restore the 0806 theme, render all eight slides, and rerun overflow, placeholder, source-note, and template-fidelity checks.

### Task 6: Rebuild the complete deck as a scientific report

**Files:**
- Modify: `.codex/weekly-report-2026-08-27-mara/build-weekly-report-mara.mjs`
- Modify: `Weekly Lab Meeting 0827 Mara.pptx`

**Interfaces:**
- Consumes: the approved narrative, Mara teacher-forced figure/table, seven-arm decoded image, and the user-supplied complete-video RGB/RMSE/SNR table.
- Produces: an eight-slide evidence-first weekly report without disconnected card modules.

- [ ] Reorganize slides as title, hypothesis, experiment, two result slides, conclusion, discussion, and next plan.
- [ ] Use the user-supplied table image and the decoded-arm image together on Result II.
- [ ] Use the Q*/injection figure and a concise numeric table on Result I.
- [ ] Limit every result slide to one summary statement and keep conclusion strength proportional to the evidence.
- [ ] Render and inspect all eight slides individually, then rerun structural, source-note, overflow, and template-fidelity checks.
