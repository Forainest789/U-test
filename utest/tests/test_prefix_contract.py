from __future__ import annotations

import json
from pathlib import Path

from utest.prefix_contract import (
    build_contract,
    build_runtime_contract,
    sha256_file,
    validate_contract,
)


def _fixture(tmp_path: Path):
    snapshot = tmp_path / "prefix.pt"
    snapshot.write_bytes(b"prefix-v1")
    source = tmp_path / "story.json"
    source.write_text(
        json.dumps({"chunks": [{"content": "prefix"}, {"content": "target"}]}),
        encoding="utf-8",
    )
    reference = tmp_path / "frame.jpg"
    reference.write_bytes(b"image")
    manifest = tmp_path / "platform.manifest.json"
    manifest.write_text(json.dumps({"repo_commit": "abc", "repo_dirty": False}), encoding="utf-8")
    event = {
        "story_id": "s1",
        "entity_uid": "s1::ana",
        "character_name": "ana",
        "target_chunk_idx": 1,
        "source_json_path": str(source),
        "reference_path": str(reference),
    }
    args = [
        "--json_path", str(source), "--ref_image_path", str(reference),
        "--seed_base", "42", "--cfg_scale", "5.0", "--output_path", "ignored",
        "--resume_state_path", str(snapshot), "--save_state_path", "arm.pt",
        "--start_chunk_idx", "1", "--max_chunks", "1",
    ]
    return snapshot, manifest, event, args


def test_snapshot_mutation_is_rejected(tmp_path: Path) -> None:
    snapshot, manifest, event, args = _fixture(tmp_path)
    contract = build_contract(event, snapshot, args, manifest)
    assert contract["snapshot"]["sha256"] == sha256_file(snapshot)

    snapshot.write_bytes(b"prefix-v2")
    assert validate_contract(contract, snapshot, contract["runtime_contract"]) == [
        "snapshot_sha256_mismatch"
    ]


def test_runtime_frozen_argument_mismatch_is_rejected(tmp_path: Path) -> None:
    snapshot, manifest, event, args = _fixture(tmp_path)
    contract = build_contract(event, snapshot, args, manifest)
    runtime = {**contract["runtime_contract"], "target_seed": 99}
    assert validate_contract(contract, snapshot, runtime) == ["target_seed_mismatch"]


def test_runtime_contract_is_derived_from_actual_arguments(tmp_path: Path) -> None:
    _, _, event, args = _fixture(tmp_path)

    runtime = build_runtime_contract(event, [*args, "--cfg_scale", "6.0"])

    assert runtime["source_json_sha256"] == sha256_file(
        Path(event["source_json_path"])
    )
    assert runtime["target_seed"] == 43
    assert runtime["frozen_args"]["cfg_scale"] == "6.0"


def test_target_seed_override_changes_only_runtime_seed(tmp_path: Path) -> None:
    _, _, event, args = _fixture(tmp_path)

    runtime = build_runtime_contract(event, [*args, "--target_seed_override", "271"])

    assert runtime["target_seed"] == 271
    assert "target_seed_override" not in runtime["frozen_args"]


def test_output_and_resume_paths_are_not_frozen_generation_args(tmp_path: Path) -> None:
    snapshot, manifest, event, args = _fixture(tmp_path)
    contract = build_contract(event, snapshot, args, manifest)
    frozen = contract["runtime_contract"]["frozen_args"]
    assert "output_path" not in frozen
    assert "resume_state_path" not in frozen
    assert "save_state_path" not in frozen
    assert "start_chunk_idx" not in frozen
    assert "max_chunks" not in frozen
    assert frozen["cfg_scale"] == "5.0"
