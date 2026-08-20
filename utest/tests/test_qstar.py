from __future__ import annotations

import math
import json
from pathlib import Path

import pytest

from utest.qstar import (
    classify_memory_regime,
    classify_qstar,
    masked_mse,
    normalized_influence,
    qstar_deltas,
)
from utest.qstar_probe import (
    ProbeCell,
    evaluate_probe_cell,
    validate_probe_runtime,
    write_probe_outputs,
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


def test_synthetic_seven_run_probe_is_paired_and_excludes_native(tmp_path: Path) -> None:
    cell = ProbeCell(
        event_id="e1",
        memory_id="ana|0",
        horizon=8,
        timestep_index=12,
        timestep=750.0,
        clean_target=[0.0, 0.0],
        noise=[1.0, 1.0],
        noisy_latent=[0.75, 0.75],
        flow_target=[1.0, 1.0],
        input_hashes={
            "prefix": "prefix-hash",
            "target": "target-hash",
            "noise": "noise-hash",
            "noisy_latent": "noisy-hash",
            "prompt": "prompt-hash",
        },
    )
    payloads = {
        "correct": 0.9,
        "no_memory": 0.5,
        "zero": 0.55,
        "random": 0.6,
        "wrong": 0.52,
    }
    calls = []

    def predictor(arm, shared_cell, payload, native):
        calls.append((arm, id(shared_cell), payload, native))
        value = 0.0 if native else float(payload or 0.0)
        return {
            "prediction": [value, value],
            "injection_delta_norm": 0.0 if native else abs(value),
            "memory_read_hit": arm not in {"no_memory", "native"},
        }

    report, records = evaluate_probe_cell(cell, payloads, predictor)

    assert [row[0] for row in calls] == [
        "correct", "correct_repeat", "no_memory", "zero", "random", "wrong", "native"
    ]
    assert calls[0][2] == calls[1][2] == payloads["correct"]
    assert calls[-1][3] is True
    assert len({row[1] for row in calls}) == 1
    assert len(records) == 7
    assert all(row["input_hashes"] == cell.input_hashes for row in records)
    assert report["qstar"] > 0
    assert report["repeat_loss_floor"] == 0
    assert "native" not in report["arm_deltas"]

    write_probe_outputs(tmp_path, [report], records)
    written = json.loads((tmp_path / "qstar_report.json").read_text(encoding="utf-8"))
    assert written["status"] == "passed"
    assert len((tmp_path / "qstar_records.jsonl").read_text(encoding="utf-8").splitlines()) == 7


def test_probe_runtime_rejects_stale_prompt_and_seed() -> None:
    frozen = {"target_prompt_sha256": "prompt-a", "target_seed": 47}
    with pytest.raises(ValueError, match="target_prompt_sha256_mismatch"):
        validate_probe_runtime(frozen, {**frozen, "target_prompt_sha256": "prompt-b"})
    with pytest.raises(ValueError, match="target_seed_mismatch"):
        validate_probe_runtime(frozen, {**frozen, "target_seed": 99})
