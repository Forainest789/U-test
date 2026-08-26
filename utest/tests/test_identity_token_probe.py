from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

from utest.identity_token_probe import (
    _sum_dit_forward_counts,
    _v0_cells,
    build_screening_schedule,
    cache_key,
    classify_fusion_mechanism,
    finalize_s2,
    prediction_error_decomposition,
    run_probe,
    run_fusion_alpha_sweep,
    run_s3,
    s2_forward_budget,
    select_fusion_verification_cells,
    select_cells,
    semantic_group_manifest,
    write_outputs,
)
from utest.qstar_probe import ProbeCell
from utest.qstar_probe import tensor_sha256


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


def test_dit_forward_counts_reconcile_measured_warmup_and_semantic_calls() -> None:
    records = [
        {"dit_forward_counts": {"semantic_prepass": 1, "conditional": 1, "unconditional": 0}},
        {"dit_forward_counts": {"semantic_prepass": 0, "conditional": 1, "unconditional": 0}},
        {"dit_forward_counts": {"semantic_prepass": 1, "conditional": 0, "unconditional": 0}},
    ]

    assert _sum_dit_forward_counts(records) == {
        "semantic_prepass": 2,
        "conditional": 2,
        "unconditional": 0,
        "raw": 4,
    }


def test_prediction_error_decomposition_reconstructs_loss_and_predicts_rescue() -> None:
    neutral = prediction_error_decomposition(
        prediction=[1.5, 0.5], baseline=[1.0, 1.0], target=[0.0, 0.0]
    )
    assert neutral["loss_delta_from_no_memory"] == 0.25
    assert neutral["directional_alignment"] == 0.0
    assert neutral["delta_energy"] == 0.25
    assert abs(neutral["decomposition_residual"]) < 1e-12
    assert neutral["predicted_optimal_alpha"] == 0.0

    rescue = prediction_error_decomposition(
        prediction=[0.0, 2.0], baseline=[1.0, 2.0], target=[0.0, 0.0]
    )
    assert rescue["directional_alignment"] < 0.0
    assert 0.0 < rescue["predicted_optimal_alpha"] <= 1.0
    assert rescue["predicted_optimal_gain"] > 0.0


def test_prediction_error_decomposition_avoids_fp32_cancellation() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    baseline = torch.tensor([-1.2904350757598877], dtype=torch.float32)
    prediction = torch.tensor([-1.290442943572998], dtype=torch.float32)

    result = prediction_error_decomposition(
        prediction=prediction,
        baseline=baseline,
        target=torch.zeros_like(baseline),
    )

    assert abs(result["decomposition_residual"]) < 1e-8


def test_v0_attaches_available_arm_decompositions_without_new_forwards() -> None:
    screening = [
        {"timestep_index": 25, "layer_group": [5, 6], "q_content": 0.1},
        {"timestep_index": 25, "layer_group": [0, 1], "q_content": 0.2},
    ]
    records = [
        {"timestep_index": 25, "layer_group": list(range(16)), "arm": "no_memory", "_prediction": [1.0, 2.0]},
        {"timestep_index": 25, "layer_group": [5, 6], "arm": "correct", "_prediction": [0.0, 2.0]},
        {"timestep_index": 25, "layer_group": [5, 6], "arm": "wrong", "_prediction": [1.5, 2.0]},
        {"timestep_index": 25, "layer_group": [0, 1], "arm": "correct", "_prediction": [1.0, 1.0]},
        {"timestep_index": 25, "layer_group": [0, 1], "arm": "wrong", "_prediction": [2.0, 2.0]},
        {"timestep_index": 25, "layer_group": [0, 1], "arm": "zero", "_prediction": [1.0, 1.5]},
    ]
    cells = [SimpleNamespace(timestep_index=25, flow_target=[0.0, 0.0])]

    result = _v0_cells(screening, records, cells)

    assert set(result[0]["error_decomposition"]) == {"correct", "wrong"}
    assert set(result[1]["error_decomposition"]) == {"correct", "wrong", "zero"}
    assert result[0]["error_decomposition"]["correct"]["directional_alignment"] < 0.0
    assert "error_decomposition" not in screening[0]


def test_fusion_selection_requires_rescuable_direction_and_prefers_distinct_cell() -> None:
    def cell(timestep, layers, alignment, alpha, gain, q_content):
        return {
            "timestep_index": timestep,
            "layer_group": layers,
            "q_content": q_content,
            "error_decomposition": {
                "correct": {
                    "directional_alignment": alignment,
                    "predicted_optimal_alpha": alpha,
                    "predicted_optimal_gain": gain,
                }
            },
        }

    result = select_fusion_verification_cells(
        [
            cell(25, [5, 6], -0.4, 0.5, 0.10, 0.01),
            cell(25, [0, 1], -0.3, 0.5, 0.09, 0.02),
            cell(49, [5, 6], -0.2, 0.5, 0.08, 0.03),
            cell(0, [0, 1], 0.1, 0.5, 0.50, 0.50),
            cell(49, [0, 1], -0.2, 1.0, 0.50, 0.50),
        ],
        trigger_floor=0.0,
    )

    assert len(result["trigger_candidates"]) == 3
    assert [row["timestep_index"] for row in result["selected_cells"]] == [25, 49]
    assert result["selected_cells"][0]["layer_group"] == [5, 6]


def test_fusion_alpha_sweep_uses_existing_scales_and_restores_engine() -> None:
    calls = []

    class Engine:
        device = "cpu"
        sparse_role_memory_injection_layers = [0, 1]
        sparse_role_memory_layer_scales = {5: 0.8, 6: 0.9, 12: 0.7}

        def generate_chunk(self, **kwargs):
            assert kwargs["teacher_forced_probe"]["capture_sparse_token_diagnostics"] is True
            alpha = self.sparse_role_memory_layer_scales[5]
            assert self.sparse_role_memory_layer_scales[6] == alpha
            memory = kwargs["memory_tokens"]
            prediction = float(memory) * alpha
            calls.append((alpha, memory, list(self.sparse_role_memory_injection_layers)))
            layer_stats = {
                str(layer): {
                    "enabled": 1,
                    "selected_query_tokens": 1,
                    "selected_memory_tokens": 1,
                    "effective_delta_norm": abs(prediction),
                    "host_token_norm": 1.0,
                    "token_diagnostics": {
                        "flat_idx": [0, 1],
                        "host_features": [[1.0, 0.0], [0.0, 1.0]],
                        "effective_delta_features": [
                            [prediction, 0.0],
                            [0.0, prediction],
                        ],
                    },
                }
                for layer in (5, 6)
            }
            return {
                "prediction_cond": [prediction],
                "prediction": [prediction],
                "forced_memory_path": False,
                "sparse_role_memory_stats_by_layer": layer_stats,
                "attention_implementation": "flash_attention_2",
                "dit_forward_counts": {
                    "semantic_prepass": 1,
                    "conditional": 1,
                    "unconditional": 0,
                },
            }

    engine = Engine()
    context = _fake_context(engine)
    selected = [{"timestep_index": 25, "layer_group": [5, 6]}]
    screening = [
        {
            "timestep_index": 25,
            "layer_group": [5, 6],
            "arm": "correct",
            "prediction_sha256": tensor_sha256([1.0]),
        },
        {
            "timestep_index": 25,
            "layer_group": [5, 6],
            "arm": "wrong",
            "prediction_sha256": tensor_sha256([-1.0]),
        },
    ]

    result = run_fusion_alpha_sweep(context, selected, screening)

    assert [(row["alpha"], row["arm"]) for row in result["records"]] == [
        (alpha, arm)
        for alpha in (0.0, 0.25, 0.5, 1.0)
        for arm in ("correct", "wrong")
    ]
    assert len(calls) == result["measured_forward_count"] == 8
    half_correct = next(
        row for row in result["records"]
        if row["alpha"] == 0.5 and row["arm"] == "correct"
    )
    half_wrong = next(
        row for row in result["records"]
        if row["alpha"] == 0.5 and row["arm"] == "wrong"
    )
    assert half_correct["host_feature_diagnostics"]["5"]["mean_host_delta_cosine"] == 1.0
    assert half_wrong["host_feature_diagnostics"]["5"]["mean_host_delta_cosine"] == -1.0
    assert half_wrong["host_feature_diagnostics"]["5"]["mean_correct_wrong_delta_cosine"] == -1.0
    assert "_result" not in half_correct and "_prediction" not in half_correct
    assert engine.sparse_role_memory_injection_layers == [0, 1]
    assert engine.sparse_role_memory_layer_scales == {5: 0.8, 6: 0.9, 12: 0.7}


def test_fusion_mechanism_classifies_competition_without_claiming_proof() -> None:
    v0 = {
        "loss_no_memory": 1.0,
        "error_decomposition": {
            "correct": {"directional_alignment": -0.2, "delta_energy": 0.3}
        },
    }
    records = []
    losses = {
        ("correct", 0.0): 1.0,
        ("correct", 0.25): 0.9,
        ("correct", 0.5): 0.95,
        ("correct", 1.0): 1.1,
        ("wrong", 0.0): 1.0,
        ("wrong", 0.25): 0.98,
        ("wrong", 0.5): 1.0,
        ("wrong", 1.0): 1.05,
    }
    for (arm, alpha), loss in losses.items():
        records.append({
            "arm": arm,
            "alpha": alpha,
            "loss": loss,
            "host_feature_diagnostics": {
                "5": {
                    "mean_host_norm_drift": alpha * 0.1,
                    "mean_delta_host_ratio": alpha * 0.2,
                }
            },
        })

    result = classify_fusion_mechanism(v0, records, trigger_floor=0.0)

    assert result["primary"] == "representation_competition_candidate"
    assert result["flags"]["supplement_candidate"] is True
    assert result["flags"]["representation_competition_candidate"] is True
    assert "proven" not in json.dumps(result).lower()


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
            assert kwargs["teacher_forced_probe"]["conditional_only"] is True
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
                "dit_forward_counts": {
                    "semantic_prepass": 0 if no_memory else 1,
                    "conditional": 1,
                    "unconditional": 0,
                },
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


def test_fusion_verification_stops_after_v0_when_no_cell_can_be_rescued(
    tmp_path: Path,
) -> None:
    calls = {"forwards": 0}

    class Engine:
        device = "cpu"
        sparse_role_memory_injection_layers = list(range(16))
        sparse_role_memory_layer_scales = {layer: 1.0 for layer in range(16)}

        def generate_chunk(self, **kwargs):
            calls["forwards"] += 1
            no_memory = kwargs.get("memory_tokens") is None
            return {
                "prediction_cond": [0.0],
                "prediction": [0.0],
                "forced_memory_path": no_memory,
                "sparse_role_memory_stats_by_layer": {} if no_memory else {
                    "0": {
                        "enabled": 1,
                        "selected_query_tokens": 1,
                        "selected_memory_tokens": 1,
                        "effective_delta_norm": 0.1,
                        "host_token_norm": 1.0,
                    }
                },
                "attention_implementation": "flash_attention_2",
                "dit_forward_counts": {
                    "semantic_prepass": 0 if no_memory else 1,
                    "conditional": 1,
                    "unconditional": 0,
                },
            }

    args = SimpleNamespace(
        repeat_loss_tolerance=0.0,
        repeat_influence_tolerance=0.0,
        benefit_margin=0.0,
        influence_floor=0.0,
        allow_attention_fallback=False,
        run_decoded_validation=False,
        verify_fusion=True,
        output=tmp_path,
    )
    report = run_probe(
        args, context_loader=lambda args, include_native: _fake_context(Engine())
    )

    assert calls["forwards"] == 25
    assert report["forward_count"] == 25
    assert report["forward_budget"] == 41
    assert report["fusion_verification"]["selected_cells"] == []
    assert report["fusion_verification"]["measured_forward_count"] == 0
    assert "s2" not in report
    assert report["gates"]["identity_set"] == {
        "status": "PENDING",
        "reasons": ["fusion verification mode does not run S2"],
    }


def test_fusion_verification_runs_bounded_v1_and_never_enters_s2(
    tmp_path: Path,
) -> None:
    calls = {"forwards": 0}

    class Engine:
        device = "cpu"
        sparse_role_memory_injection_layers = list(range(16))
        sparse_role_memory_layer_scales = {layer: 1.0 for layer in range(16)}

        def generate_chunk(self, **kwargs):
            calls["forwards"] += 1
            memory = kwargs.get("memory_tokens")
            layers = list(self.sparse_role_memory_injection_layers)
            alpha = self.sparse_role_memory_layer_scales[layers[0]] if memory is not None else 0.0
            prediction = 1.0 if memory is None else 1.0 - 1.5 * float(memory) * alpha
            capture = kwargs["teacher_forced_probe"].get("capture_sparse_token_diagnostics", False)
            by_layer = {}
            if memory is not None:
                for layer in layers:
                    row = {
                        "enabled": 1,
                        "selected_query_tokens": 1,
                        "selected_memory_tokens": 1,
                        "effective_delta_norm": max(
                            abs(1.5 * float(memory) * alpha),
                            0.01 if float(memory) == 0.0 else 0.0,
                        ),
                        "host_token_norm": 1.0,
                    }
                    if capture:
                        row["token_diagnostics"] = {
                            "flat_idx": [0],
                            "host_features": [[1.0, 0.0]],
                            "effective_delta_features": [[-1.5 * float(memory) * alpha, 0.0]],
                        }
                    by_layer[str(layer)] = row
            return {
                "prediction_cond": [prediction],
                "prediction": [prediction],
                "forced_memory_path": memory is None,
                "sparse_role_memory_stats_by_layer": by_layer,
                "attention_implementation": "flash_attention_2",
                "dit_forward_counts": {
                    "semantic_prepass": 0 if memory is None else 1,
                    "conditional": 1,
                    "unconditional": 0,
                },
            }

    args = SimpleNamespace(
        repeat_loss_tolerance=0.0,
        repeat_influence_tolerance=0.0,
        benefit_margin=0.0,
        influence_floor=0.0,
        allow_attention_fallback=False,
        run_decoded_validation=False,
        verify_fusion=True,
        output=tmp_path,
    )
    report = run_probe(
        args, context_loader=lambda args, include_native: _fake_context(Engine())
    )

    fusion = report["fusion_verification"]
    assert calls["forwards"] == report["forward_count"] == 41
    assert len(fusion["selected_cells"]) == 2
    assert fusion["measured_forward_count"] == 16
    assert len(fusion["mechanism_classification"]) == 2
    assert all(
        row["classification"]["primary"] == "representation_competition_candidate"
        for row in fusion["mechanism_classification"]
    )
    assert report["unconditional_dit_count"] == 0
    assert "s2" not in report


def test_delta8_semantic_manifest_separates_identity_action_and_scene() -> None:
    root = Path(__file__).resolve().parents[2]
    story = json.loads(
        (root / "utest/events/person_reappearance_delta8_story.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = semantic_group_manifest(
        story, {"character_name": "Mara", "target_chunk_idx": 8}
    )

    assert manifest["identity_name"] == ["Mara"]
    assert len(manifest["stable_attributes"]) == 4
    assert manifest["action_core"] == [
        "runs",
        "two steps",
        "catches",
        "closing",
        "looks up",
        "toward camera",
    ]
    assert set(manifest["scene"]) == {
        "platform",
        "tram",
        "rain",
        "commuters",
        "dusk",
    }


def test_s2_never_exceeds_declared_measured_budget() -> None:
    assert s2_forward_budget(max_groups=8, has_validation=True) <= 25
    assert 25 + s2_forward_budget(max_groups=8, has_validation=True) <= 50


def test_finalize_s2_emits_identity_core_only_after_set_level_pass() -> None:
    token_rows = [
        {
            "flat_idx": 4,
            "s_name": 0.9,
            "s_attr": 0.8,
            "s_persist": 0.85,
            "s_action": 0.2,
            "s_scene": 0.1,
            "group_causal_score": 0.7,
            "content_delta": 0.25,
        }
    ]
    losses = {
        "no_memory": 1.0,
        "full_correct": 0.5,
        "identity_only": 0.6,
        "drop_identity": 0.9,
        "drop_random": 0.55,
        "drop_low": 0.52,
        "wrong_identity": 0.85,
    }

    result = finalize_s2(
        token_rows,
        losses,
        identity_fraction=0.25,
        repeat_margin=0.01,
        benefit_margin=0.01,
        validation_direction=True,
    )

    assert result["gate"]["status"] == "PASS"
    assert result["metrics"]["r_keep"] >= 0.8
    assert "identity-core candidate" in result["token_rows"][0]["labels"]


def test_production_s2_wires_expansion_knockouts_and_validation() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "utest/identity_token_probe.py").read_text(encoding="utf-8")
    body = source[source.index("def run_s2("):source.index("def run_probe(")]

    assert '"all_token_diagnostic_correct"' in body
    assert '"all_token_diagnostic_wrong"' in body
    assert "build_candidate_groups(" in body
    assert 'f"drop_group:{group[\'group_id\']}"' in body
    assert '"validation_wrong_identity"' in body
    assert "S2 measured-forward budget exceeded" in body


def test_identity_opts_into_conditional_only_without_changing_qstar() -> None:
    root = Path(__file__).resolve().parents[2]
    identity = (root / "utest/identity_token_probe.py").read_text(encoding="utf-8")
    qstar = (root / "utest/qstar_probe.py").read_text(encoding="utf-8")
    screening = identity[
        identity.index("def _run_screening_forward("):identity.index("def _screening_cells(")
    ]
    semantic_capture = identity[
        identity.index("def _semantic_capture("):identity.index("def _s2_model_forward(")
    ]
    s2_forward = identity[
        identity.index("def _s2_model_forward("):identity.index("def _text_positions(")
    ]

    assert '"conditional_only": True' in screening
    assert '"conditional_only": True' in s2_forward
    assert '"conditional_only": True' not in semantic_capture
    assert identity.count('"conditional_only": True') == 2
    assert '"conditional_only"' not in qstar


def test_cache_key_changes_for_every_frozen_boundary() -> None:
    base = {
        "prefix": "a",
        "prompt": "b",
        "timestep": 25,
        "layers": [5, 6],
        "backend": "fa2",
    }
    original = cache_key("cell", base)
    for key, value in (
        ("prompt", "c"),
        ("timestep", 49),
        ("layers", [7]),
        ("backend", "sdpa"),
    ):
        changed = dict(base)
        changed[key] = value
        assert cache_key("cell", changed) != original


def test_output_report_is_complete_without_reading_figures(tmp_path: Path) -> None:
    result = {
        "schema_version": 1,
        "forward_count": 25,
        "forward_budget": 50,
        "gates": {"runtime_contract": {"status": "PASS"}},
        "runtime": {"attention_implementation": "flash_attention_2"},
        "timing": {"total_wall_time_s": 1.0},
        "input_contract": {"prefix": "abc"},
        "screening_records": [],
        "screening_cells": [
            {
                "timestep_index": 25,
                "layer_group": [5, 6, 7, 8, 9, 10],
                "q_content": 0.2,
            }
        ],
        "selected_cells": {"primary": None, "validation": None},
        "s2": {
            "semantic_manifest": {"identity_name": ["Mara"]},
            "diagnostic_prompt": "Mara runs",
            "token_rows": [],
            "groups": [],
            "interventions": [],
        },
        "fusion_verification": {
            "schema_version": 1,
            "mode": "verify_fusion",
            "selected_cells": [],
            "v1_records": [],
        },
    }
    write_outputs(tmp_path, result)

    expected = {
        "runtime_manifest.json",
        "input_contract.json",
        "screening_cells.jsonl",
        "selected_cells.json",
        "diagnostic_prompt_manifest.json",
        "token_scores.jsonl",
        "token_groups.json",
        "interventions.jsonl",
        "identity_probe_report.json",
        "fusion_verification.json",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    report = json.loads(
        (tmp_path / "identity_probe_report.json").read_text(encoding="utf-8")
    )
    assert report["gates"] and report["timing"] and report["forward_count"] <= 50


def test_s3_runs_exactly_four_frozen_query_arms(tmp_path: Path) -> None:
    calls = []

    class Engine:
        def generate_chunk(self, **kwargs):
            calls.append(kwargs)
            return ["frame"], "latents", {}

    context = _fake_context(Engine())
    context["event"]["character_name"] = "Mara"
    s2 = {
        "candidate_universe": [1, 2, 3, 4, 5, 6],
        "masks": {
            "identity_top": [1, 2],
            "drop_identity": [3, 4, 5, 6],
        },
    }
    saved = []
    result = run_s3(
        context,
        s2,
        output=tmp_path,
        save_video_fn=lambda frames, path, fps: saved.append((frames, Path(path), fps)),
    )

    assert [row["arm"] for row in result["arms"]] == [
        "full_correct",
        "no_memory",
        "identity_only",
        "drop_identity",
    ]
    assert len(calls) == len(saved) == 4
    assert calls[0]["query_indices_by_role"] == {"Mara": [1, 2, 3, 4, 5, 6]}
    assert calls[1]["query_indices_by_role"] is None
    assert calls[2]["query_indices_by_role"] == {"Mara": [1, 2]}


def test_smoke_mode_runs_one_cell_and_stops_before_s2(tmp_path: Path) -> None:
    calls = {"forwards": 0}

    class Engine:
        device = "cpu"
        sparse_role_memory_injection_layers = list(range(16))

        def generate_chunk(self, **kwargs):
            calls["forwards"] += 1
            assert kwargs["teacher_forced_probe"]["conditional_only"] is True
            memory = kwargs.get("memory_tokens")
            no_memory = memory is None
            prediction = 1.0 if memory == -1.0 else 0.0
            return {
                "prediction_cond": [prediction],
                "prediction": [prediction],
                "forced_memory_path": no_memory,
                "sparse_role_memory_stats_by_layer": {} if no_memory else {
                    "5": {
                        "enabled": 1,
                        "selected_query_tokens": 4,
                        "selected_memory_tokens": 4,
                        "effective_delta_norm": 0.1,
                        "host_token_norm": 1.0,
                    }
                },
                "attention_implementation": "flash_attention_2",
                "dit_forward_counts": {
                    "semantic_prepass": 0 if no_memory else 1,
                    "conditional": 1,
                    "unconditional": 0,
                },
            }

    args = SimpleNamespace(
        repeat_loss_tolerance=0.0,
        repeat_influence_tolerance=0.0,
        benefit_margin=0.0,
        influence_floor=0.0,
        allow_attention_fallback=False,
        output=tmp_path,
        smoke=True,
    )
    result = run_probe(
        args, context_loader=lambda args, include_native: _fake_context(Engine())
    )

    assert calls["forwards"] == 5
    assert result["gates"]["content_causality"]["status"] == "PASS"
    assert "s2" not in result
    assert result["forward_count"] == 5
    assert result["measured_arm_count"] == 5
    assert result["warmup_arm_count"] == 0
    assert result["semantic_prepass_count"] == 4
    assert result["conditional_dit_count"] == 5
    assert result["unconditional_dit_count"] == 0
    assert result["raw_dit_invocation_count"] == 9
    assert result["actual_model_forward_count"] == result["raw_dit_invocation_count"]
