from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from utest.event_harness import (
    _writer_evidence,
    build_arm_commands,
    build_prefix_inference_args,
    score_event,
    validate_audit_group,
    validate_runtime_reports,
)
from utest.qstar import classify_memory_regime
from utest.memory_utility import REQUIRED_OUTCOMES


def test_arm_commands_share_snapshot_seed_and_target_window(tmp_path: Path) -> None:
    snapshot = tmp_path / "prefix.pt"
    contract = {
        "snapshot": {"path": str(snapshot), "sha256": "abc"},
        "event": {"target_chunk_idx": 4},
        "arm_seed": 17,
        "base_inference_args": [
            "--json_path", "story.json", "--seed_base", "42",
            "--output_path", "old", "--max_chunks", "4",
            "--target_seed_override", "99",
        ],
    }
    commands = build_arm_commands(
        contract,
        output_root=tmp_path / "arms",
        event_json=tmp_path / "event.json",
        arms=("no_memory", "correct", "random"),
        python="python",
    )

    assert set(commands) == {"no_memory", "correct", "random", "correct_repeat"}
    assert list(commands) == ["no_memory", "correct", "correct_repeat", "random"]
    for run_name, command in commands.items():
        assert command[command.index("--resume_state_path") + 1] == str(snapshot)
        assert command[command.index("--save_state_path") + 1] == str(
            (tmp_path / "arms" / run_name / "resume_state.pt").resolve()
        )
        assert command[command.index("--max_chunks") + 1] == "6"
        assert command[command.index("--seed") + 1] == "17"
        assert "--target_seed_override" not in command


def test_prefix_generation_saves_new_state_without_resuming(tmp_path: Path) -> None:
    source = tmp_path / "story.json"
    reference = tmp_path / "reference.jpg"
    event = {
        "source_json_path": str(source),
        "reference_path": str(reference),
        "character_name": "luca",
        "target_chunk_idx": 4,
    }

    args = build_prefix_inference_args(
        event,
        tmp_path,
        [
            "--resume_state_path", "stale.pt",
            "--start_chunk_idx", "2",
            "--target_seed_override", "99",
            "--output_path", "old",
        ],
    )

    assert "--resume_state_path" not in args
    assert "--start_chunk_idx" not in args
    assert "--target_seed_override" not in args
    assert args[args.index("--save_state_path") + 1] == str(tmp_path / "prefix_state.pt")
    assert args[args.index("--max_chunks") + 1] == "4"
    assert args[args.index("--json_path") + 1] == str(source.resolve())
    assert args[args.index("--ref_image_path") + 1] == str(reference.resolve())
    assert "--target_character" not in args


def test_prefix_contract_can_freeze_the_future_target_seed_without_changing_source_seed_base(
    tmp_path: Path,
) -> None:
    event = {
        "source_json_path": str(tmp_path / "story.json"),
        "target_chunk_idx": 6,
    }
    args = build_prefix_inference_args(
        event,
        tmp_path,
        ["--seed_base", "2", "--target_seed_override", "stale"],
        target_seed_override=2,
    )

    assert args[args.index("--seed_base") + 1] == "2"
    assert args[args.index("--target_seed_override") + 1] == "2"


def test_arm_commands_apply_one_target_seed_override(tmp_path: Path) -> None:
    contract = {
        "snapshot": {"path": str(tmp_path / "prefix.pt"), "sha256": "abc"},
        "event": {"target_chunk_idx": 4},
        "arm_seed": 17,
        "base_inference_args": ["--seed_base", "42"],
    }

    commands = build_arm_commands(
        contract,
        output_root=tmp_path / "arms",
        event_json=tmp_path / "event.json",
        arms=("correct", "random"),
        target_seed_override=271,
    )

    for command in commands.values():
        assert command[command.index("--target_seed_override") + 1] == "271"
        assert command[command.index("--start_chunk_idx") + 1] == "4"


def test_seven_run_commands_keep_repeat_adjacent_and_native_last(tmp_path: Path) -> None:
    snapshot = tmp_path / "prefix.pt"
    contract = {
        "snapshot": {"path": str(snapshot), "sha256": "abc"},
        "event": {"target_chunk_idx": 8},
        "arm_seed": 17,
        "base_inference_args": ["--json_path", "story.json", "--seed_base", "42"],
    }

    commands = build_arm_commands(
        contract,
        output_root=tmp_path / "arms",
        event_json=tmp_path / "event.json",
        arms=("correct", "no_memory", "zero", "random", "wrong"),
        donor=tmp_path / "donor.pt",
        donor_manifest=tmp_path / "donor.json",
        include_native=True,
    )

    assert list(commands) == [
        "correct", "correct_repeat", "no_memory", "zero", "random", "wrong", "native"
    ]
    native = commands["native"]
    assert "utest.content_audit" not in native
    assert "--native_wan_inference" in native
    assert native[native.index("--resume_state_path") + 1] == str(snapshot)
    assert native[native.index("--start_chunk_idx") + 1] == "8"


def test_writer_regime_allows_static_prefix_unless_explicitly_required() -> None:
    static = {"update_count": 1, "positive_residual_count": 0, "bank_hash_change_count": 0}
    dynamic = {"update_count": 1, "positive_residual_count": 1, "bank_hash_change_count": 1}

    assert classify_memory_regime(static) == "static_prefix"
    assert classify_memory_regime(dynamic) == "dynamic_writer"


def test_prefix_generation_removes_stale_reference_when_event_has_none(
    tmp_path: Path,
) -> None:
    event = {"source_json_path": str(tmp_path / "story.json"), "target_chunk_idx": 1}

    args = build_prefix_inference_args(
        event,
        tmp_path,
        ["--ref_image_path", "stale.jpg"],
    )

    assert "--ref_image_path" not in args


def test_audit_group_requires_real_target_hits_and_transformations() -> None:
    reports = {
        "no_memory": {
            "attempted_reads": 1, "source_non_null_reads": 1,
            "returned_non_null_reads": 3, "target_read_hits": 1,
            "target_source_non_null_reads": 1,
            "target_returned_non_null_reads": 0,
            "intervention_effective": True,
        },
        "correct": {
            "source_non_null_reads": 1, "target_read_hits": 1,
            "payload_layers_seen": 16, "intervention_effective": True,
        },
        "zero": {"target_read_hits": 1, "layers_transformed": 16, "intervention_effective": True},
        "wrong": {"target_read_hits": 1, "layers_transformed": 16, "intervention_effective": True},
        "random": {"target_read_hits": 1, "layers_transformed": 16, "intervention_effective": True},
    }
    assert validate_audit_group(reports) == []

    reports["wrong"]["target_read_hits"] = 0
    assert validate_audit_group(reports) == ["wrong:target_address_miss"]

    reports["wrong"]["target_read_hits"] = 1
    reports["random"]["native_read_mismatches"] = 1
    assert validate_audit_group(reports) == ["random:native_read_changed"]

    reports["random"]["native_read_mismatches"] = 0
    reports["correct"]["target_read_mismatches"] = 1
    assert validate_audit_group(reports) == ["correct:passthrough_changed"]


def test_runtime_validation_uses_each_arm_actual_report(tmp_path: Path) -> None:
    snapshot = tmp_path / "prefix.pt"
    snapshot.write_bytes(b"prefix")
    runtime = {
        "frozen_args": {"cfg_scale": "5.0"},
        "source_json_sha256": "source",
        "target_prompt_sha256": "prompt",
        "reference_sha256": None,
        "target_seed": 17,
    }
    contract = {
        "snapshot": {
            "path": str(snapshot),
            "sha256": hashlib.sha256(b"prefix").hexdigest(),
        },
        "runtime_contract": runtime,
    }
    reports = {
        name: {"runtime_contract": dict(runtime)}
        for name in ("no_memory", "zero", "correct", "wrong", "random", "correct_repeat")
    }
    reports["random"]["runtime_contract"] = {
        **runtime,
        "frozen_args": {"cfg_scale": "6.0"},
    }

    assert validate_runtime_reports(contract, snapshot, reports) == [
        "random:frozen_args_mismatch"
    ]


def test_writer_evidence_is_scoped_to_target_and_requires_residual() -> None:
    efficiency = {
        "chunks": [
            {
                "chunk_idx": 1,
                "writer_updates": [{"stats": {"residual_norm": 3.0}}],
                "memory_bank_hash_changed": True,
            },
            {
                "chunk_idx": 4,
                "writer_updates": [{"stats": {"layers": {"0": {"residual_norm": 0.2}}}}],
                "memory_bank_hash_changed": True,
            },
        ]
    }
    assert _writer_evidence(efficiency, 4) == {
        "update_count": 1,
        "positive_residual_count": 1,
        "bank_hash_change_count": 1,
    }

    efficiency["chunks"][1]["writer_updates"][0]["stats"] = {
        "residual_norm": float("inf")
    }
    assert _writer_evidence(efficiency, 4)["positive_residual_count"] == 0


def test_score_command_writes_complete_five_arm_report(tmp_path: Path) -> None:
    event_run = tmp_path / "arms"
    event_run.mkdir()
    records = []
    for seed, arms in {
        1: {"no_memory": 0.5},
        7: {"no_memory": 0.5, "correct": 0.6, "wrong": 0.4, "zero": 0.5, "random": 0.48},
    }.items():
        for arm, identity in arms.items():
            outcomes = {name: 0.9 for name in REQUIRED_OUTCOMES}
            outcomes["C_id"] = identity
            outcomes["Q_motion_dynamic_degree"] = 0.8
            records.append({
                "story_id": "s1", "event_id": "e1", "arm": arm,
                "seed": seed, "outcomes": outcomes,
            })
    quality = {
        name: 0.02
        for name in REQUIRED_OUTCOMES
        if name not in {"C_id", "Q_motion_dynamic_degree"}
    }
    records_path = tmp_path / "records.json"
    rules_path = tmp_path / "rules.json"
    records_path.write_text(json.dumps(records), encoding="utf-8")
    rules_path.write_text(json.dumps({
        "delta_id": 0.01,
        "quality_margins": quality,
        "dynamic_degree_floor": 0.2,
        "gate_a_floors": {"C_id": 0.3, "Q_motion_dynamic_degree": 0.2},
        "qualification_seeds": [1],
        "formal_seeds": [7],
        "content_causal": True,
        "n_boot": 20,
    }), encoding="utf-8")
    args = argparse.Namespace(event_run=event_run, records=records_path, rules=rules_path)

    assert score_event(args) == 0
    report = json.loads((event_run / "utility_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert set(report["arm_populations"]) == {"correct", "wrong", "zero", "random"}
