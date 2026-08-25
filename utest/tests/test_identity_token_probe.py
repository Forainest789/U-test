from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from utest.identity_token_probe import build_screening_schedule, run_probe, select_cells
from utest.qstar_probe import ProbeCell


def test_s0_s1_schedule_has_25_unique_measured_forwards() -> None:
    schedule = build_screening_schedule()
    keys = {(row.timestep_index, row.layer_group, row.arm) for row in schedule}

    assert len(schedule) == len(keys) == 25
    assert sum(row.arm == "no_memory" for row in schedule) == 3
    assert any(
        row.arm == "correct_repeat" and row.timestep_index == 25 for row in schedule
    )
    assert any(
        row.arm == "wrong" and len(row.layer_group) == 16 for row in schedule
    )


def test_cell_selection_prioritizes_positive_content_delta() -> None:
    selected = select_cells(
        [
            {
                "timestep_index": 0,
                "layer_group": [0, 1, 2, 3, 4],
                "q_content": -0.1,
                "delta_host_ratio": 9.0,
            },
            {
                "timestep_index": 25,
                "layer_group": [5, 6, 7, 8, 9, 10],
                "q_content": 0.2,
                "delta_host_ratio": 0.1,
            },
            {
                "timestep_index": 49,
                "layer_group": [11, 12, 13, 14, 15],
                "q_content": 0.1,
                "delta_host_ratio": 0.5,
            },
        ],
        repeat_margin=0.01,
        influence_floor=0.0,
    )

    assert selected["primary"]["timestep_index"] == 25
    assert selected["validation"]["timestep_index"] == 49


def _fake_context(engine) -> dict:
    cells = [
        ProbeCell(
            event_id="delta8",
            memory_id="Mara|0",
            horizon=8,
            timestep_index=timestep,
            timestep=float(timestep),
            clean_target=[0.0],
            noise=[0.0],
            noisy_latent=[0.0],
            flow_target=[0.0],
            input_hashes={"cell": str(timestep)},
        )
        for timestep in (0, 25, 49)
    ]
    bundle = lambda value, hit: {
        "memory_tokens": value,
        "memory_bank_tokens": None,
        "memory_bank_percents": [],
        "memory_bank_token_meta": None,
        "memory_token_lengths_per_character": None,
        "target_read_hit": hit,
        "target_payload_sha256": str(value),
        "target_payload_summary": {"layers": 1, "slots": 4},
    }
    return {
        "engine": engine,
        "event": {"event_id": "delta8", "character_name": "Mara"},
        "contract": {"snapshot": {"sha256": "prefix"}},
        "prompt": "Mara returns",
        "target_seed": 7,
        "reference_frames": None,
        "fixed_reference": None,
        "cells": cells,
        "payloads": {
            "correct": bundle(1.0, True),
            "zero": bundle(0.0, True),
            "no_memory": bundle(None, False),
            "wrong": bundle(-1.0, True),
        },
        "state": {},
    }


def test_orchestrator_loads_once_and_stops_before_s2_without_content_signal(
    tmp_path: Path,
) -> None:
    calls = {"loads": 0, "forwards": 0}

    class FakeEngine:
        device = "cpu"
        sparse_role_memory_injection_layers = list(range(16))

        def generate_chunk(self, **kwargs):
            calls["forwards"] += 1
            assert kwargs["teacher_forced_probe"]["force_memory_path"] is True
            no_memory = kwargs.get("memory_tokens") is None
            return {
                "prediction_cond": [0.0],
                "prediction": [0.0],
                "forced_memory_path": no_memory,
                "sparse_role_memory_stats_by_layer": {} if no_memory else {
                    "0": {
                        "enabled": 1,
                        "selected_query_tokens": 4,
                        "selected_memory_tokens": 4,
                        "effective_delta_norm": 0.1,
                        "host_token_norm": 1.0,
                    }
                },
                "attention_implementation": "flash_attention_2",
            }

    def load(args, *, include_native):
        calls["loads"] += 1
        assert include_native is False
        return _fake_context(FakeEngine())

    args = SimpleNamespace(
        repeat_loss_tolerance=0.0,
        repeat_influence_tolerance=0.0,
        benefit_margin=0.0,
        influence_floor=0.0,
        allow_attention_fallback=False,
        output=tmp_path,
    )
    report = run_probe(args, context_loader=load)

    assert calls == {"loads": 1, "forwards": 25}
    assert report["gates"]["content_causality"]["status"] == "BLOCK"
    assert report["forward_count"] == 25
