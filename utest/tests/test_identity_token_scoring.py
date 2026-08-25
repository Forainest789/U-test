from __future__ import annotations

import math

import pytest

from utest.identity_token_scoring import (
    causal_metrics,
    classify_token,
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
    assert "identity-core candidate" not in classify_token(
        row, repeat_margin=0.01, benefit_margin=0.3, validation_direction=True
    )
