from __future__ import annotations

import math

import pytest

from utest.qstar import (
    classify_memory_regime,
    classify_qstar,
    masked_mse,
    normalized_influence,
    qstar_deltas,
)


def test_qstar_sign_and_control_deltas() -> None:
    deltas = qstar_deltas(
        {
            "correct": 0.25,
            "correct_repeat": 0.2500001,
            "no_memory": 0.40,
            "zero": 0.36,
            "random": 0.34,
            "wrong": 0.39,
            "native": 0.31,
        }
    )

    assert deltas["qstar"] == pytest.approx(0.15)
    assert deltas["arm_deltas"]["wrong"] == pytest.approx(0.14)
    assert "native" not in deltas["arm_deltas"]
    assert deltas["repeat_loss_floor"] == pytest.approx(1e-7)


def test_qstar_classification_separates_influence_from_utility() -> None:
    assert classify_qstar(qstar=0.2, influence=0.3, repeat_margin=1e-6, influence_floor=1e-6) == "beneficial"
    assert classify_qstar(qstar=0.0, influence=0.3, repeat_margin=1e-6, influence_floor=1e-6) == "influence_without_benefit"
    assert classify_qstar(qstar=-0.2, influence=0.3, repeat_margin=1e-6, influence_floor=1e-6) == "influence_without_benefit"
    assert classify_qstar(qstar=0.2, influence=0.0, repeat_margin=1e-6, influence_floor=1e-6) == "no_measurable_influence"


def test_normalized_influence_and_masked_mse() -> None:
    assert normalized_influence([2.0, 0.0], [1.0, 0.0]) == pytest.approx(0.5)
    assert masked_mse([1.0, 3.0, 9.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="mask selects no values"):
        masked_mse([1.0], [0.0], [0.0])


def test_non_finite_values_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite"):
        qstar_deltas({"correct": 1.0, "correct_repeat": 1.0, "no_memory": math.inf})
    with pytest.raises(ValueError, match="finite"):
        normalized_influence([math.nan], [0.0])


def test_writer_regime_is_explicit() -> None:
    assert classify_memory_regime({"positive_residual_count": 1, "bank_hash_change_count": 1}) == "dynamic_writer"
    assert classify_memory_regime({"positive_residual_count": 0, "bank_hash_change_count": 1}) == "static_prefix"
    assert classify_memory_regime({"positive_residual_count": 2, "bank_hash_change_count": 0}) == "static_prefix"
