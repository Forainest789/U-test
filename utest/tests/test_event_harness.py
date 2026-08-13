from __future__ import annotations

import argparse
import json
from pathlib import Path

from utest.event_harness import (
    _writer_evidence,
    build_arm_commands,
    score_event,
    validate_audit_group,
)
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
    for command in commands.values():
        assert command[command.index("--resume_state_path") + 1] == str(snapshot)
        assert command[command.index("--max_chunks") + 1] == "6"
        assert command[command.index("--seed") + 1] == "17"


def test_audit_group_requires_real_target_hits_and_transformations() -> None:
    reports = {
        "no_memory": {
            "attempted_reads": 1, "source_non_null_reads": 1,
            "returned_non_null_reads": 0, "target_read_hits": 1,
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
