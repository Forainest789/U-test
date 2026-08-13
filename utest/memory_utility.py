"""Decoded-outcome scoring for the frozen-SlotMem four-arm protocol.

Consumes per-(story, event, arm, seed) decoded outcome vectors and produces the three-way
helpful/neutral/harmful census with story-level cluster intervals. This is the aggregation
half of the protocol; ``scripts/slotmem_content_audit.py`` is the intervention half.

Three things here exist because the project already paid for them:

* **The estimand is named by evidence, not by intention.** Until a run supplies proof that
  ``correct`` separates from ``wrong``, this reports a ``memory_presence_effect``, not
  utility. 2026-07-15 and 2026-08-08 both measured a real, repeatable effect of "memory is
  present" and it was content-generic -- wrong-video memory reproduced it. Calling that
  number the utility of a particular memory is the specific error the whole plan is built
  to avoid, so the report refuses to use the word on its own.
* **Gate A is decided on qualification seeds only**, asserted disjoint from the formal
  seeds. Filtering stories on the formal seed's own no-memory draw conditions on the
  minuend of ``correct - none``: it removes exactly the events whose baseline draw was
  poor and biases the harm rate. Selection on the dependent variable, not cleaning.
* **Quality is a conjunction with one absolute floor.** A frozen clip maximises smoothness
  and flicker scores; 2026-08-05 watched injection collapse into pure smoothing 62/62
  while those metrics improved. Dynamic degree is therefore gated on its own absolute
  value, never on non-inferiority against a baseline that may itself be static.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from .bootstrap import cluster_bootstrap_mean_ci, wilson_interval

IDENTITY = "C_id"
PROMPT_ALIGNMENT = "A_prompt"
BACKGROUND = "Q_bg"
MOTION_SMOOTHNESS = "Q_motion_smoothness"
DYNAMIC_DEGREE = "Q_motion_dynamic_degree"
FLICKER = "Q_flicker"
BOUNDARY = "Q_boundary"
ANATOMY = "Q_anatomy"
NON_TARGET = "Q_non_target"
"""Gated on its own absolute value -- see the module docstring."""

REQUIRED_OUTCOMES = (
    IDENTITY,
    PROMPT_ALIGNMENT,
    BACKGROUND,
    MOTION_SMOOTHNESS,
    DYNAMIC_DEGREE,
    FLICKER,
    BOUNDARY,
    ANATOMY,
    NON_TARGET,
)
QUALITY_MARGIN_OUTCOMES = (
    PROMPT_ALIGNMENT,
    BACKGROUND,
    MOTION_SMOOTHNESS,
    FLICKER,
    BOUNDARY,
    ANATOMY,
    NON_TARGET,
)

HELPFUL, NEUTRAL, HARMFUL = "helpful", "neutral", "harmful"


def measurement_completeness(outcomes: Mapping[str, float | None]) -> tuple[bool, list[str]]:
    """Require the frozen decoded vector; anatomy may be N/A only when declared."""
    missing: list[str] = []
    for name in REQUIRED_OUTCOMES:
        if name not in outcomes:
            missing.append(name)
        elif outcomes[name] is None and not (
            name == ANATOMY and outcomes.get("Q_anatomy_applicable") is False
        ):
            missing.append(name)
    return not missing, missing


def label_event(
    delta: Mapping[str, float],
    arm_outcome: Mapping[str, float],
    *,
    delta_id: float,
    quality_margins: Mapping[str, float],
    dynamic_degree_floor: float,
) -> tuple[str, list[str]]:
    """Three-way label for one event, plus the reasons that decided it.

    ``quality_margins[j]`` is the non-inferiority allowance for component ``j``: the arm
    may lose up to that much before the read counts as harmful. Identity uses its own
    two-sided ``delta_id``. Dynamic degree is checked on ``arm_outcome``, not ``delta``.
    """
    reasons: list[str] = []
    d_id = float(delta.get(IDENTITY, 0.0))

    breached = [
        name for name, margin in quality_margins.items()
        if float(delta.get(name, 0.0)) < -abs(float(margin))
    ]
    reasons.extend(f"quality_breach:{name}" for name in breached)

    frozen = float(arm_outcome.get(DYNAMIC_DEGREE, float("inf"))) < float(dynamic_degree_floor)
    if frozen:
        reasons.append("dynamic_degree_below_floor")

    if d_id < -abs(delta_id):
        reasons.append("identity_loss")
    if reasons:
        return HARMFUL, reasons
    if d_id > abs(delta_id):
        return HELPFUL, ["identity_gain"]
    return NEUTRAL, ["within_margin"]


def gate_a_pass(
    baseline_outcome: Mapping[str, float], *, floors: Mapping[str, float]
) -> tuple[bool, list[str]]:
    """Is the no-memory arm itself a legitimate baseline on this story?

    ``floors`` are ABSOLUTE minima on the baseline's own outcome, not deltas. MVP1
    recorded that no-memory is a degenerate floor: when the baseline is incoherent every
    delta against it inflates and the harmful tail becomes unobservable, because the
    outcome is already at the bottom.
    """
    failed = [
        name for name, floor in floors.items()
        if float(baseline_outcome.get(name, float("-inf"))) < float(floor)
    ]
    return (not failed), failed


def _delta(arm: Mapping[str, float], none: Mapping[str, float]) -> dict[str, float]:
    return {
        key: float(arm[key]) - float(none[key])
        for key in REQUIRED_OUTCOMES
        if arm.get(key) is not None and none.get(key) is not None
    }


def utility_census(
    records: Iterable[Mapping],
    *,
    delta_id: float,
    quality_margins: Mapping[str, float],
    dynamic_degree_floor: float,
    gate_a_floors: Mapping[str, float],
    qualification_seeds: Sequence[int],
    formal_seeds: Sequence[int],
    treatment_arm: str = "correct",
    baseline_arm: str = "no_memory",
    content_causal: bool | None = None,
    comparison_arms: Sequence[str] = ("correct", "wrong", "zero", "random"),
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """Three-way census over both populations, clustered on story.

    ``records`` are mappings with ``story_id``, ``event_id``, ``arm``, ``seed`` and
    ``outcomes`` (the component vector). ``content_causal`` is the M2 verdict: leave it
    ``None`` before M2 has run -- the report will name the estimand accordingly and will
    not call anything utility.
    """
    qualification = set(int(s) for s in qualification_seeds)
    formal = set(int(s) for s in formal_seeds)
    overlap = qualification & formal
    if overlap:
        raise ValueError(
            f"qualification seeds {sorted(overlap)} also appear in the formal seeds: "
            "deciding Gate A on the formal draw conditions on the minuend of "
            "correct-minus-none and biases the harm rate"
        )

    by_key: dict[tuple[str, str, int], dict[str, Mapping[str, float]]] = defaultdict(dict)
    for record in records:
        key = (str(record["story_id"]), str(record["event_id"]), int(record["seed"]))
        by_key[key][str(record["arm"])] = record["outcomes"]

    # Gate A on the qualification seeds only, decided per story: a story qualifies when
    # every qualification-seed baseline it has clears the floors.
    story_gate: dict[str, tuple[bool, list[str]]] = {}
    for (story, _event, seed_id), arms in by_key.items():
        if seed_id not in qualification or baseline_arm not in arms:
            continue
        passed, failed = gate_a_pass(arms[baseline_arm], floors=gate_a_floors)
        was_ok, prior = story_gate.get(story, (True, []))
        story_gate[story] = (was_ok and passed, prior + failed)

    missing_margins = sorted(set(QUALITY_MARGIN_OUTCOMES) - set(quality_margins))

    def label_arm(arm_name: str) -> list[dict]:
        labelled: list[dict] = []
        for (story, event, seed_id), arms in sorted(by_key.items()):
            if seed_id not in formal or arm_name not in arms or baseline_arm not in arms:
                continue
            _arm_complete, arm_missing = measurement_completeness(arms[arm_name])
            _base_complete, base_missing = measurement_completeness(arms[baseline_arm])
            missing = sorted(set(arm_missing + base_missing + [f"margin:{x}" for x in missing_margins]))
            base = {
                "story_id": story,
                "event_id": event,
                "seed": seed_id,
                "arm": arm_name,
                "gate_a": story_gate.get(story, (True, []))[0],
            }
            if missing:
                labelled.append({
                    **base,
                    "status": "measurement_incomplete",
                    "missing_metrics": missing,
                })
                continue
            delta = _delta(arms[arm_name], arms[baseline_arm])
            label, reasons = label_event(
                delta,
                arms[arm_name],
                delta_id=delta_id,
                quality_margins=quality_margins,
                dynamic_degree_floor=dynamic_degree_floor,
            )
            labelled.append({
                **base,
                "status": "complete",
                "delta": delta,
                "label": label,
                "reasons": reasons,
            })
        return labelled

    arm_rows = {arm: label_arm(arm) for arm in comparison_arms}
    labelled = arm_rows.get(treatment_arm, label_arm(treatment_arm))

    populations = {
        "all_eligible": labelled,
        "gate_a_qualified": [row for row in labelled if row["gate_a"]],
    }
    report = {
        "estimand": "memory_utility" if content_causal else "memory_presence_effect",
        "content_causal": content_causal,
        "populations": {
            name: _population_summary(rows, n_boot=n_boot, seed=seed)
            for name, rows in populations.items()
        },
        "arm_populations": {
            arm: {
                "all_eligible": _population_summary(rows, n_boot=n_boot, seed=seed),
                "gate_a_qualified": _population_summary(
                    [row for row in rows if row["gate_a"]], n_boot=n_boot, seed=seed
                ),
            }
            for arm, rows in arm_rows.items()
        },
        "gate_a_disqualified": sorted(
            {story: reasons for story, (ok, reasons) in story_gate.items() if not ok}.items()
        ),
        "events": labelled,
    }
    if not content_causal:
        report["naming_note"] = (
            "correct-minus-none without established content causality is a "
            "memory-PRESENCE effect: a content-generic prior reproduces it (2026-07-15, "
            "2026-08-08). Do not report it as the utility of a specific memory."
        )
    return report


def _population_summary(rows: Sequence[Mapping], *, n_boot: int, seed: int) -> dict:
    if not rows:
        return {"n_stories": 0, "n_events": 0}
    complete = [row for row in rows if row.get("status", "complete") == "complete"]
    if not complete:
        return {
            "n_stories": len({str(row["story_id"]) for row in rows}),
            "n_events": len(rows),
            "n_complete": 0,
            "n_incomplete": len(rows),
            "status": "measurement_incomplete",
        }
    by_story: dict[str, list[float]] = defaultdict(list)
    for row in complete:
        by_story[row["story_id"]].append(float(row["delta"].get(IDENTITY, 0.0)))
    mean, lo, hi = cluster_bootstrap_mean_ci(
        list(by_story.values()), n_boot=n_boot, seed=seed,
    )
    n_stories = len(by_story)
    # Rates are story-level too: a story contributes its own fraction, so a story with
    # many events cannot dominate the proportion.
    story_labels: dict[str, list[str]] = defaultdict(list)
    for row in complete:
        story_labels[row["story_id"]].append(row["label"])
    summary = {"n_stories": n_stories, "n_events": len(rows),
               "n_complete": len(complete), "n_incomplete": len(rows) - len(complete),
               "delta_identity": {"mean": mean, "lo": lo, "hi": hi}}
    for label in (HELPFUL, NEUTRAL, HARMFUL):
        # A story counts toward a label when that label is its majority outcome across
        # its own events and seeds.
        hits = sum(
            1 for labels in story_labels.values()
            if labels.count(label) * 2 > len(labels)
        )
        rate, rate_lo, rate_hi = wilson_interval(hits, n_stories)
        summary[f"{label}_rate"] = {"rate": rate, "lo": rate_lo, "hi": rate_hi, "n": hits}
    return summary
