from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import utest.event_harness as event_harness
from utest.event_harness import (
    _writer_evidence,
    build_arm_commands,
    build_prefix_inference_args,
    dump_donor,
    load_event,
    prepare_prefix,
    score_event,
    validate_audit_group,
    validate_runtime_reports,
)
from utest.prefix_contract import build_runtime_contract
from utest.qstar import classify_memory_regime
from utest.memory_utility import REQUIRED_OUTCOMES


def test_event_parent_paths_resolve_from_the_event_file(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    event_path = bundle / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "path_resolution": "event_parent",
                "source_json_path": "story.json",
                "reference_path": "reference.jpg",
            }
        ),
        encoding="utf-8",
    )

    event = load_event(event_path)

    assert event["source_json_path"] == str((bundle / "story.json").resolve())
    assert event["reference_path"] == str((bundle / "reference.jpg").resolve())
    assert "path_resolution" not in event


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


def test_donor_command_builder_derives_seed_zero_runtime_from_correct_branch(
    tmp_path: Path,
) -> None:
    story = tmp_path / "story.json"
    reference = tmp_path / "reference.jpg"
    story.write_text(
        json.dumps({"chunks": [{"content": f"chunk {index}"} for index in range(5)]}),
        encoding="utf-8",
    )
    reference.write_bytes(b"reference")
    event = {
        "source_json_path": str(story),
        "reference_path": str(reference),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "character_name": "person",
        "target_chunk_idx": 4,
    }
    contract = {
        "snapshot": {"path": str(tmp_path / "prefix.pt"), "sha256": "abc"},
        "event": event,
        "arm_seed": 0,
        "base_inference_args": [
            "--json_path",
            str(story),
            "--ref_image_path",
            str(reference),
            "--seed_base",
            "0",
        ],
    }

    command = build_arm_commands(
        contract,
        output_root=tmp_path / "dump",
        event_json=tmp_path / "event.json",
        arms=("correct",),
        dump_correct_donor=tmp_path / "payload.pt",
        target_seed_override=0,
        offload_models=False,
    )["correct"]
    inference_args = command[command.index("--") + 1 :]
    runtime = build_runtime_contract(event, inference_args)

    assert inference_args[inference_args.index("--target_seed_override") + 1] == "0"
    assert "--no-offload_models" in inference_args
    assert runtime["target_seed"] == 0


def test_prepare_prefix_still_returns_zero_and_writes_contract(
    tmp_path: Path, monkeypatch,
) -> None:
    import torch

    story = tmp_path / "story.json"
    reference = tmp_path / "reference.jpg"
    platform = tmp_path / "platform.json"
    event_path = tmp_path / "event.json"
    story.write_text(
        json.dumps({"chunks": [{"content": f"chunk {index}"} for index in range(5)]}),
        encoding="utf-8",
    )
    reference.write_bytes(b"reference")
    platform.write_text("{}", encoding="utf-8")
    event_path.write_text(
        json.dumps(
            {
                "source_json_path": str(story),
                "reference_path": str(reference),
                "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                "character_name": "person",
                "target_chunk_idx": 4,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "prefix"

    def materialize_snapshot(_command, _log_path):
        output.mkdir(parents=True, exist_ok=True)
        torch.save({"next_chunk_idx": 4}, output / "prefix_state.pt")

    monkeypatch.setattr(event_harness, "_run", materialize_snapshot)
    args = argparse.Namespace(
        event=event_path,
        output=output,
        inference_args_file=None,
        inference_args=["--seed_base", "0"],
        target_seed_override=0,
        python=sys.executable,
        platform_manifest=platform,
        arm_seed=0,
        future_target_video=None,
        future_target_manifest=None,
        timestep_indices="0",
        arms_root=None,
        allow_dirty_source=True,
    )

    assert prepare_prefix(args) == 0
    assert (output / "prefix_contract.json").is_file()


def test_dump_donor_defaults_to_strict_frozen_prefix_target_seed(
    tmp_path: Path, monkeypatch,
) -> None:
    import torch

    prefix = tmp_path / "prefix"
    prefix.mkdir()
    snapshot = prefix / "prefix_state.pt"
    snapshot.write_bytes(b"snapshot")
    event_json = prefix / "event.json"
    event_json.write_text("{}", encoding="utf-8")
    (prefix / "prefix_contract.json").write_text(
        json.dumps(
            {
                "snapshot": {
                    "path": str(snapshot),
                    "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                },
                "event": {},
                "event_json": str(event_json),
                "runtime_contract": {"target_seed": 271},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "dump"
    payload = tmp_path / "payload.pt"
    captured = []

    def capture_builder(*args, target_seed_override, **kwargs):
        captured.append(target_seed_override)
        return {"correct": ["fake-command"]}

    def materialize_dump(_command, _log_path):
        torch.save(
            {"format": "slotmem_donor_payload_v2", "payloads": {}},
            payload,
        )
        audit = output / "correct" / "audit.json"
        audit.parent.mkdir(parents=True)
        audit.write_text(
            json.dumps({"target_read_hits": 1, "intervention_effective": True}),
            encoding="utf-8",
        )

    monkeypatch.setattr(event_harness, "build_arm_commands", capture_builder)
    monkeypatch.setattr(event_harness, "_run", materialize_dump)
    args = argparse.Namespace(
        prefix=prefix,
        output=output,
        donor_payload=payload,
        target_seed_override=None,
        python=sys.executable,
    )

    assert dump_donor(args) == 0
    assert captured == [271]


@pytest.mark.parametrize("value", [True, 271.0])
def test_dump_donor_rejects_non_integer_frozen_prefix_target_seed(
    tmp_path: Path, value: object,
) -> None:
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    (prefix / "prefix_contract.json").write_text(
        json.dumps(
            {
                "snapshot": {"path": str(prefix / "prefix_state.pt")},
                "runtime_contract": {"target_seed": value},
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        prefix=prefix,
        output=tmp_path / "dump",
        donor_payload=tmp_path / "payload.pt",
        target_seed_override=None,
        python=sys.executable,
    )

    with pytest.raises(ValueError, match="target seed override"):
        dump_donor(args)


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


def test_cli_propagates_handler_status_without_system_exit(tmp_path: Path) -> None:
    event_run = tmp_path / "arms"
    records_path = tmp_path / "records.json"
    rules_path = tmp_path / "rules.json"
    records_path.write_text("[]", encoding="utf-8")
    rules_path.write_text(
        json.dumps(
            {
                "delta_id": 0.01,
                "quality_margins": {},
                "dynamic_degree_floor": 0.2,
                "gate_a_floors": {},
                "qualification_seeds": [1],
                "formal_seeds": [7],
                "n_boot": 20,
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "utest.event_harness",
            "score",
            "--event-run",
            str(event_run),
            "--records",
            str(records_path),
            "--rules",
            str(rules_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "event harness failed with status 2" in result.stderr
