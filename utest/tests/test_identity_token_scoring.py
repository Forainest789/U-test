from __future__ import annotations

import math

import pytest

from utest.identity_token_scoring import (
    build_candidate_groups,
    build_intervention_masks,
    causal_metrics,
    classify_token,
    group_membership_sha256,
    percentile_rank,
    score_token_channels,
)


def test_percentile_rank_is_deterministic_for_ties_and_singletons() -> None:
    assert percentile_rank([3.0]) == [0.5]
    assert percentile_rank([1.0, 1.0, 3.0]) == pytest.approx([0.25, 0.25, 1.0])
    with pytest.raises(ValueError, match="finite"):
        percentile_rank([0.0, math.inf])


def test_identity_and_action_scores_follow_frozen_equations() -> None:
    rows = score_token_channels([
        {
            "flat_idx": 4,
            "name_raw": 9.0,
            "attribute_raw": 8.0,
            "persistence_raw_margin": 7.0,
            "persistence_read_margin": 6.0,
            "action_attention_raw": 6.0,
            "action_hidden_raw": 5.0,
            "scene_hidden_raw": 1.0,
            "random_hidden_raw": 2.0,
            "scene_raw": 3.0,
        },
        {
            "flat_idx": 7,
            "name_raw": 1.0,
            "attribute_raw": 2.0,
            "persistence_raw_margin": 3.0,
            "persistence_read_margin": 2.0,
            "action_attention_raw": 1.0,
            "action_hidden_raw": 1.0,
            "scene_hidden_raw": 1.0,
            "random_hidden_raw": 1.0,
            "scene_raw": 8.0,
        },
    ])

    assert rows[0]["s_pre"] == pytest.approx(1.0)
    assert rows[0]["action_hidden_net_raw"] == pytest.approx(3.0)
    assert rows[0]["s_action"] > rows[1]["s_action"]


def test_causal_metrics_and_identity_label_require_correct_content() -> None:
    metrics = causal_metrics({
        "no_memory": 1.0,
        "full_correct": 0.5,
        "identity_only": 0.6,
        "drop_identity": 0.9,
        "drop_random": 0.55,
        "drop_low": 0.52,
        "wrong_identity": 0.85,
    })
    assert metrics["r_keep"] == pytest.approx(0.8)
    assert metrics["r_drop"] == pytest.approx(0.8)
    row = {
        "s_name": 0.9,
        "s_attr": 0.8,
        "s_persist": 0.85,
        "s_action": 0.2,
        "s_scene": 0.1,
        "group_causal_score": 0.7,
        "group_control_floor": 0.1,
        "content_delta": 0.25,
    }
    assert "identity-core candidate" in classify_token(
        row, repeat_margin=0.01, benefit_margin=0.01, validation_direction=True
    )
    no_content_row = {**row, "content_delta": 0.0}
    assert "identity-core candidate" not in classify_token(
        no_content_row, repeat_margin=0.01, benefit_margin=0.01, validation_direction=True
    )


def test_grouping_is_coordinate_only_bounded_and_deterministic() -> None:
    indices = [0, 1, 4, 5, 16, 17, 20, 21, 32, 33, 36, 37]
    first = build_candidate_groups(
        indices, height=4, width=4, max_groups=3, min_group_size=4
    )
    second = build_candidate_groups(
        list(reversed(indices)), height=4, width=4, max_groups=3, min_group_size=4
    )
    assert first == second
    assert 1 <= len(first) <= 3
    assert sorted(index for group in first for index in group["indices"]) == sorted(indices)
    assert all(len(group["indices"]) >= 4 for group in first)
    assert group_membership_sha256(first) == group_membership_sha256(second)


def test_interventions_match_count_and_per_frame_histogram() -> None:
    original = [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51]
    universe = original + [4, 20, 36, 52]
    scores = {index: float(index % 16) for index in universe}
    masks = build_intervention_masks(
        original,
        universe,
        scores,
        budget_fraction=0.25,
        seed=7,
        height=4,
        width=4,
    )
    keep_names = ["identity_top", "random", "low_score", "wrong_identity"]
    assert {len(masks[name]) for name in keep_names} == {4}
    histogram = lambda values: [
        sum(index // 16 == frame for index in values) for frame in range(4)
    ]
    assert {tuple(histogram(masks[name])) for name in keep_names} == {
        tuple(histogram(masks["identity_top"]))
    }
    assert all(
        len(universe) - len(masks[name]) == 4
        for name in ("drop_identity", "drop_random", "drop_low")
    )
