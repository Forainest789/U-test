"""The aggregation half of the four-arm protocol: labels, Gate A, populations, naming."""
from __future__ import annotations

import pytest

from utest.bootstrap import cluster_bootstrap_mean_ci, wilson_interval
from utest.memory_utility import (
    ANATOMY, BACKGROUND, BOUNDARY, DYNAMIC_DEGREE, FLICKER, HARMFUL, HELPFUL,
    IDENTITY, MOTION_SMOOTHNESS, NEUTRAL, NON_TARGET, PROMPT_ALIGNMENT,
    gate_a_pass, label_event, utility_census,
)

MARGINS = {
    PROMPT_ALIGNMENT: 0.02,
    BACKGROUND: 0.02,
    MOTION_SMOOTHNESS: 0.02,
    FLICKER: 0.02,
    BOUNDARY: 0.02,
    ANATOMY: 0.02,
    NON_TARGET: 0.02,
}
FLOORS = {IDENTITY: 0.3, DYNAMIC_DEGREE: 0.2}


def _outcomes(identity: float, dynamic: float = 0.8, **rest) -> dict[str, float]:
    return {
        IDENTITY: identity,
        PROMPT_ALIGNMENT: 0.9,
        BACKGROUND: 0.9,
        MOTION_SMOOTHNESS: 0.9,
        DYNAMIC_DEGREE: dynamic,
        FLICKER: 0.9,
        BOUNDARY: 0.9,
        ANATOMY: 0.9,
        NON_TARGET: 0.9,
        **rest,
    }


def _record(story, event, arm, seed, identity, dynamic=0.8, **rest):
    return {"story_id": story, "event_id": event, "arm": arm, "seed": seed,
            "outcomes": _outcomes(identity, dynamic, **rest)}


def test_identity_gain_alone_is_not_helpful_when_quality_breaks() -> None:
    delta = {IDENTITY: +0.10, "Q_bg": -0.05}
    label, reasons = label_event(
        delta, _outcomes(0.8), delta_id=0.01, quality_margins=MARGINS,
        dynamic_degree_floor=0.2,
    )
    assert label == HARMFUL and "quality_breach:Q_bg" in reasons


def test_frozen_output_cannot_buy_identity() -> None:
    """The 2026-08-05 attractor: smoothing raises identity while the video stops moving."""
    delta = {IDENTITY: +0.10, "Q_bg": 0.0, "Q_flicker": +0.05}
    label, reasons = label_event(
        delta, _outcomes(0.9, dynamic=0.05), delta_id=0.01, quality_margins=MARGINS,
        dynamic_degree_floor=0.2,
    )
    assert label == HARMFUL and "dynamic_degree_below_floor" in reasons
    # ...and the floor is absolute: a baseline that is ALSO frozen must not rescue it,
    # which a non-inferiority check on the delta would have done.
    assert float(delta.get(DYNAMIC_DEGREE, 0.0)) == 0.0


def test_three_way_labels() -> None:
    common = dict(delta_id=0.01, quality_margins=MARGINS, dynamic_degree_floor=0.2)
    assert label_event({IDENTITY: +0.05}, _outcomes(0.9), **common)[0] == HELPFUL
    assert label_event({IDENTITY: 0.0}, _outcomes(0.9), **common)[0] == NEUTRAL
    assert label_event({IDENTITY: -0.05}, _outcomes(0.9), **common)[0] == HARMFUL


def test_gate_a_rejects_a_degenerate_baseline() -> None:
    ok, failed = gate_a_pass(_outcomes(0.9), floors=FLOORS)
    assert ok and not failed
    ok, failed = gate_a_pass(_outcomes(0.1), floors=FLOORS)
    assert not ok and failed == [IDENTITY]


def test_gate_a_on_formal_seeds_is_refused() -> None:
    """Deciding Gate A on the formal draw conditions on the minuend -- refuse, loudly."""
    with pytest.raises(ValueError, match="conditions on the minuend"):
        utility_census(
            [], delta_id=0.01, quality_margins=MARGINS, dynamic_degree_floor=0.2,
            gate_a_floors=FLOORS, qualification_seeds=[1, 2], formal_seeds=[2, 3],
        )


def test_census_reports_both_populations_and_refuses_the_word_utility() -> None:
    records = []
    # story A: healthy baseline, memory helps
    records += [_record("A", "e0", "no_memory", 7, 0.60), _record("A", "e0", "correct", 7, 0.70)]
    # story B: healthy baseline, memory hurts
    records += [_record("B", "e0", "no_memory", 7, 0.60), _record("B", "e0", "correct", 7, 0.50)]
    # story C: COLLAPSED baseline -- the delta looks huge but the floor was at the bottom
    records += [_record("C", "e0", "no_memory", 7, 0.05), _record("C", "e0", "correct", 7, 0.55)]
    # qualification-seed baselines decide Gate A
    records += [_record("A", "e0", "no_memory", 1, 0.62), _record("B", "e0", "no_memory", 1, 0.61),
                _record("C", "e0", "no_memory", 1, 0.06)]

    report = utility_census(
        records, delta_id=0.01, quality_margins=MARGINS, dynamic_degree_floor=0.2,
        gate_a_floors=FLOORS, qualification_seeds=[1], formal_seeds=[7], n_boot=200,
    )

    assert report["estimand"] == "memory_presence_effect"
    assert "naming_note" in report
    every = report["populations"]["all_eligible"]
    gated = report["populations"]["gate_a_qualified"]
    assert every["n_stories"] == 3 and gated["n_stories"] == 2
    # C's +0.50 against a collapsed baseline inflates the unconditional mean; excluding
    # it is the whole point of Gate A, and both numbers must be visible.
    assert every["delta_identity"]["mean"] > gated["delta_identity"]["mean"]
    assert gated["helpful_rate"]["n"] == 1 and gated["harmful_rate"]["n"] == 1
    assert [story for story, _ in report["gate_a_disqualified"]] == ["C"]

    # naming flips only when M2 supplied the verdict
    proven = utility_census(
        records, delta_id=0.01, quality_margins=MARGINS, dynamic_degree_floor=0.2,
        gate_a_floors=FLOORS, qualification_seeds=[1], formal_seeds=[7],
        content_causal=True, n_boot=200,
    )
    assert proven["estimand"] == "memory_utility" and "naming_note" not in proven


def test_cluster_bootstrap_does_not_treat_seeds_as_independent() -> None:
    """Ten seeds of one story are one story, so the interval must not shrink."""
    one_story = [[0.1] * 10]
    ten_stories = [[0.1]] * 10
    _, lo_one, hi_one = cluster_bootstrap_mean_ci(one_story, n_boot=500)
    _, lo_ten, hi_ten = cluster_bootstrap_mean_ci(ten_stories, n_boot=500)
    assert (hi_one - lo_one) == pytest.approx(0.0, abs=1e-9)  # a single cluster
    assert (hi_ten - lo_ten) == pytest.approx(0.0, abs=1e-9)  # identical values
    varied = cluster_bootstrap_mean_ci([[0.0], [0.2]] * 5, n_boot=500)
    assert varied[2] > varied[1]  # real between-story spread survives
    # a story's own event count must not weight it
    mean, _, _ = cluster_bootstrap_mean_ci([[1.0] * 100, [0.0]], n_boot=100)
    assert mean == pytest.approx(0.5)


def test_wilson_interval_stays_inside_the_unit_range() -> None:
    rate, lo, hi = wilson_interval(0, 8)
    assert rate == 0.0 and lo >= 0.0 and hi > 0.0  # Wald would give width 0 here
    rate, lo, hi = wilson_interval(8, 8)
    assert rate == 1.0 and hi <= 1.0 and lo < 1.0
    rate, lo, hi = wilson_interval(4, 8)
    assert lo < 0.5 < hi


def test_incomplete_metric_vector_has_no_utility_label() -> None:
    records = [
        {"story_id": "A", "event_id": "e0", "arm": "no_memory", "seed": 7,
         "outcomes": {IDENTITY: 0.5, DYNAMIC_DEGREE: 0.8}},
        {"story_id": "A", "event_id": "e0", "arm": "correct", "seed": 7,
         "outcomes": {IDENTITY: 0.6, DYNAMIC_DEGREE: 0.8}},
    ]
    report = utility_census(
        records, delta_id=0.01, quality_margins=MARGINS,
        dynamic_degree_floor=0.2, gate_a_floors=FLOORS,
        qualification_seeds=[1], formal_seeds=[7], content_causal=True, n_boot=20,
    )
    row = report["events"][0]
    assert row["status"] == "measurement_incomplete"
    assert "label" not in row
    assert PROMPT_ALIGNMENT in row["missing_metrics"]


def test_all_control_arms_are_reported_against_no_memory() -> None:
    records = []
    for arm, identity in {
        "no_memory": 0.50,
        "correct": 0.60,
        "wrong": 0.45,
        "zero": 0.51,
        "random": 0.48,
    }.items():
        records.append(_record("A", "e0", arm, 7, identity))
    records.append(_record("A", "e0", "no_memory", 1, 0.50))
    report = utility_census(
        records, delta_id=0.01, quality_margins=MARGINS,
        dynamic_degree_floor=0.2, gate_a_floors=FLOORS,
        qualification_seeds=[1], formal_seeds=[7], content_causal=True, n_boot=20,
    )
    assert set(report["arm_populations"]) == {"correct", "wrong", "zero", "random"}
    assert report["arm_populations"]["wrong"]["all_eligible"]["delta_identity"]["mean"] < 0
