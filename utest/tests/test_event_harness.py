from __future__ import annotations

from pathlib import Path

from utest.event_harness import build_arm_commands, validate_audit_group


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
