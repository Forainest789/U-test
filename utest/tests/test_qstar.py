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
    prepare_flow_cell,
    validate_measured_injection,
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


def _constant_cell() -> ProbeCell:
    return ProbeCell(
        event_id="e1",
        memory_id="mara|0",
        horizon=8,
        timestep_index=0,
        timestep=999.0,
        clean_target=[0.0],
        noise=[1.0],
        noisy_latent=[1.0],
        flow_target=[1.0],
        input_hashes={"prefix": "p"},
    )


def _predictor_for(values: dict):
    def predictor(arm, cell, payload, native):
        return {"prediction": [values[arm]], "memory_read_hit": not native}

    return predictor


def test_repeat_loss_and_influence_tolerances_are_separate_units() -> None:
    # one perturbation, two readings two orders apart: 0.01 as a relative L2 ratio,
    # 1e-4 as an absolute MSE difference. One scalar cannot gate both.
    values = {
        "correct": 1.0, "correct_repeat": 1.01, "no_memory": 2.0,
        "zero": 1.5, "random": 1.4, "wrong": 1.6, "native": 1.2,
    }
    with pytest.raises(ValueError, match="correct_repeat loss"):
        evaluate_probe_cell(
            _constant_cell(), {k: 0.0 for k in ("correct", "no_memory", "zero", "random", "wrong")},
            _predictor_for(values), repeat_loss_tolerance=0.0, repeat_influence_tolerance=1.0,
        )
    with pytest.raises(ValueError, match="correct_repeat prediction"):
        evaluate_probe_cell(
            _constant_cell(), {k: 0.0 for k in ("correct", "no_memory", "zero", "random", "wrong")},
            _predictor_for(values), repeat_loss_tolerance=1.0, repeat_influence_tolerance=0.0,
        )


def test_zero_benefit_margin_is_flagged_and_liftable() -> None:
    payloads = {k: 0.0 for k in ("correct", "no_memory", "zero", "random", "wrong")}
    # deterministic repeat -> repeat_loss_floor == 0, so a 1e-9 Q* would pass unguarded
    values = {
        "correct": 1.0, "correct_repeat": 1.0, "no_memory": 1.0 - 5e-10,
        "zero": 1.0, "random": 1.0, "wrong": 1.0, "native": 1.0,
    }
    report, _ = evaluate_probe_cell(_constant_cell(), payloads, _predictor_for(values))
    assert report["repeat_loss_floor"] == 0.0
    assert report["benefit_margin_degenerate"] is True
    assert report["qstar"] > 0 and report["classification"] == "beneficial"

    report, _ = evaluate_probe_cell(
        _constant_cell(), payloads, _predictor_for(values), benefit_margin=1e-6
    )
    assert report["repeat_margin"] == 1e-6
    assert report["benefit_margin_degenerate"] is False
    assert report["classification"] == "influence_without_benefit"


def test_probe_runtime_rejects_stale_prompt_and_seed() -> None:
    frozen = {"target_prompt_sha256": "prompt-a", "target_seed": 47}
    with pytest.raises(ValueError, match="target_prompt_sha256_mismatch"):
        validate_probe_runtime(frozen, {**frozen, "target_prompt_sha256": "prompt-b"})
    with pytest.raises(ValueError, match="target_seed_mismatch"):
        validate_probe_runtime(frozen, {**frozen, "target_seed": 99})


def test_present_payload_requires_measured_sparse_injection() -> None:
    disabled = {
        "sparse_role_memory_stats_by_layer": {
            "15": {"enabled": 0.0, "selected_memory_tokens": 0, "effective_delta_norm": 0.0}
        }
    }
    with pytest.raises(ValueError, match="measured sparse injection is absent"):
        validate_measured_injection("correct", True, disabled)

    measured = {
        "sparse_role_memory_stats_by_layer": {
            "15": {"enabled": 1.0, "selected_memory_tokens": 64, "effective_delta_norm": 0.0125}
        }
    }
    assert validate_measured_injection("correct", True, measured) == pytest.approx(0.0125)
    assert validate_measured_injection("no_memory", False, disabled) == 0.0


def test_flow_cell_uses_scheduler_targets_without_advancing_sampler() -> None:
    class FakeScheduler:
        timesteps = [900.0, 500.0]

        def __init__(self):
            self.calls = []

        def add_noise(self, clean, noise, timestep):
            self.calls.append(("add_noise", timestep))
            return clean + noise

        def training_target(self, clean, noise, timestep):
            self.calls.append(("training_target", timestep))
            return noise - clean

        def step(self, *args):
            raise AssertionError("teacher-forced probe must not advance the sampler")

    scheduler = FakeScheduler()
    noisy, target, timestep = prepare_flow_cell(scheduler, 2.0, 5.0, 1)

    assert (noisy, target, timestep) == (7.0, 3.0, 500.0)
    assert scheduler.calls == [("add_noise", 500.0), ("training_target", 500.0)]
