from __future__ import annotations

import json
from pathlib import Path

import pytest

from utest.prefix_contract import (
    FROZEN_MEMORY_ENCODER_SLOTS,
    FROZEN_SUBJECT_SUBSPACE_BUDGET,
    FROZEN_SUBJECT_SUBSPACE_FRACTION,
    build_contract,
    build_runtime_contract,
    normalized_frozen_args,
    sha256_file,
    validate_contract,
    validate_slotmem_memory_encoder_geometry,
    write_bytes_no_clobber,
    write_json_no_clobber,
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


def test_frozen_geometry_matches_existing_64_slot_checkpoint() -> None:
    layers, slots = validate_slotmem_memory_encoder_geometry(
        {
            "slotmem_memory_encoder_layers": "0-15",
            "slotmem_memory_encoder_slots": "64",
        }
    )

    assert layers == tuple(range(16))
    assert slots == FROZEN_MEMORY_ENCODER_SLOTS == 64
    assert FROZEN_SUBJECT_SUBSPACE_BUDGET == 8
    assert FROZEN_SUBJECT_SUBSPACE_FRACTION == 0.125


def test_inline_geometry_matches_existing_64_slot_checkpoint() -> None:
    layers, slots = validate_slotmem_memory_encoder_geometry(
        normalized_frozen_args([
            "--slotmem_memory_encoder_layers=0-15",
            "--slotmem_memory_encoder_slots=64",
        ])
    )

    assert layers == tuple(range(16))
    assert slots == 64


def test_paired_64_then_inline_32_uses_the_last_geometry_value() -> None:
    with pytest.raises(ValueError, match="actual='32'.*frozen expected='64'"):
        validate_slotmem_memory_encoder_geometry(normalized_frozen_args([
            "--slotmem_memory_encoder_layers", "0-15",
            "--slotmem_memory_encoder_slots", "64",
            "--slotmem_memory_encoder_slots=32",
        ]))


def test_inline_32_then_paired_64_uses_the_last_geometry_value() -> None:
    layers, slots = validate_slotmem_memory_encoder_geometry(
        normalized_frozen_args([
            "--slotmem_memory_encoder_layers=0-15",
            "--slotmem_memory_encoder_slots=32",
            "--slotmem_memory_encoder_slots", "64",
        ])
    )

    assert layers == tuple(range(16))
    assert slots == 64


def test_legacy_32_slot_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="actual='32'.*frozen expected='64'"):
        validate_slotmem_memory_encoder_geometry(
            {
                "slotmem_memory_encoder_layers": "0-15",
                "slotmem_memory_encoder_slots": "32",
            }
        )


def test_json_publication_is_deterministic_and_no_clobber(tmp_path: Path) -> None:
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    value = {"z": [3, 2, 1], "subject": "宋雨辰"}

    write_json_no_clobber(first, value)
    write_json_no_clobber(second, value)

    assert first.read_bytes() == second.read_bytes()
    original = first.read_bytes()
    with pytest.raises(FileExistsError):
        write_json_no_clobber(first, {"changed": True})
    assert first.read_bytes() == original


def test_bytes_publication_is_atomic_and_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "artifact.bin"

    write_bytes_no_clobber(output, b"frozen")

    assert output.read_bytes() == b"frozen"
    with pytest.raises(FileExistsError):
        write_bytes_no_clobber(output, b"changed")
    assert output.read_bytes() == b"frozen"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


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


def test_default_fixed_reference_scope_is_frozen(tmp_path: Path) -> None:
    snapshot, manifest, event, args = _fixture(tmp_path)

    contract = build_contract(event, snapshot, args, manifest)

    assert contract["runtime_contract"]["frozen_args"]["fixed_reference_scope"] == "all_chunks"


def test_fixed_reference_scope_mismatch_is_rejected(tmp_path: Path) -> None:
    snapshot, manifest, event, args = _fixture(tmp_path)
    contract = build_contract(event, snapshot, args, manifest)

    runtime = build_runtime_contract(
        event, [*args, "--fixed_reference_scope", "source_only"]
    )

    assert validate_contract(contract, snapshot, runtime) == ["frozen_args_mismatch"]


def test_historical_contract_without_reference_scope_defaults_only_to_all_chunks(
    tmp_path: Path,
) -> None:
    snapshot, manifest, event, args = _fixture(tmp_path)
    historical = build_contract(event, snapshot, args, manifest)
    historical["runtime_contract"]["frozen_args"].pop("fixed_reference_scope")

    current_all_chunks = build_runtime_contract(event, args)
    current_source_only = build_runtime_contract(
        event, [*args, "--fixed_reference_scope", "source_only"]
    )

    assert validate_contract(historical, snapshot, current_all_chunks) == []
    assert validate_contract(historical, snapshot, current_source_only) == [
        "frozen_args_mismatch"
    ]


def test_qstar_contract_freezes_future_target_horizon_and_timestep_grid(tmp_path: Path) -> None:
    snapshot, manifest, event, args = _fixture(tmp_path)
    future = tmp_path / "future.mp4"
    future.write_bytes(b"held-out-teacher")
    event["source_chunk_idx"] = 0
    event["horizon"] = 1
    teacher_manifest = tmp_path / "teacher.json"
    teacher_manifest.write_text(
        json.dumps(
            {
                "story_id": "s1",
                "target_chunk_idx": 1,
                "video_path": str(future),
                "video_sha256": sha256_file(future),
                "source_type": "held_out_real",
                "generated_by_arm": False,
                "generated_by_evaluated_model": False,
            }
        ),
        encoding="utf-8",
    )

    contract = build_contract(
        event,
        snapshot,
        args,
        manifest,
        future_target_video=future,
        future_target_manifest=teacher_manifest,
        timestep_indices=(0, 12, 25, 37, 49),
    )

    assert contract["inputs"]["future_target_video_sha256"] == sha256_file(future)
    assert contract["inputs"]["future_target_manifest_sha256"] == sha256_file(teacher_manifest)
    assert contract["qstar"]["horizon"] == 1
    assert contract["qstar"]["timestep_indices"] == [0, 12, 25, 37, 49]


def test_qstar_contract_rejects_missing_future_target(tmp_path: Path) -> None:
    snapshot, manifest, event, args = _fixture(tmp_path)
    missing = tmp_path / "missing.mp4"

    try:
        build_contract(
            event,
            snapshot,
            args,
            manifest,
            future_target_video=missing,
            timestep_indices=(0,),
        )
    except FileNotFoundError as error:
        assert "future target" in str(error)
    else:
        raise AssertionError("missing future target must fail")


def test_qstar_contract_rejects_target_without_provenance_manifest(tmp_path: Path) -> None:
    snapshot, manifest, event, args = _fixture(tmp_path)
    future = tmp_path / "future.mp4"
    future.write_bytes(b"unproven teacher")

    try:
        build_contract(
            event,
            snapshot,
            args,
            manifest,
            future_target_video=future,
            timestep_indices=(0,),
        )
    except ValueError as error:
        assert "provenance manifest" in str(error)
    else:
        raise AssertionError("teacher without provenance manifest must fail")


def test_long_reappearance_fixture_has_one_establishment_and_one_return() -> None:
    root = Path(__file__).resolve().parents[2]
    event = json.loads(
        (root / "utest/events/person_reappearance_delta8.json").read_text(encoding="utf-8")
    )
    story = json.loads(
        (root / "utest/events/person_reappearance_delta8_story.json").read_text(encoding="utf-8")
    )
    chunks = story["chunks"]
    target = event["character_name"]

    assert len(chunks) == 9
    assert event["source_chunk_idx"] == 0
    assert event["target_chunk_idx"] == 8
    assert event["horizon"] == 8
    assert target in chunks[0]["character_list"]
    assert all(target not in chunk["character_list"] for chunk in chunks[1:8])
    assert target in chunks[8]["character_list"]
    assert chunks[0]["content"] != chunks[8]["content"]
