from __future__ import annotations

from pathlib import Path

import json
import hashlib
import sys
import pytest
import torch

from utest.subject_reappearance_harness import (
    FULL_ARMS,
    PREFLIGHT_ARMS,
    build_block_commands,
    build_matrix,
    qstar_contract_or_status,
    _run_logged,
    _recover_partial_prefix,
    _resume_completed_arm,
    _ensure_semantic_scores,
    _execute_stage,
    _expected_rows,
    _command_artifact_payload,
    _validate_qstar_report,
    main,
    validate_block,
)
from utest.subject_subspace import FROZEN_LAYER_GROUPS, build_mask_manifest
from utest.subject_subspace import capture_tensor_sha256
from utest.content_audit import LAYERS_KEY, _payload_sha256, transform_slot_rows
from utest.prefix_contract import (
    MemoryGeometryError,
    build_runtime_contract,
    normalized_frozen_args,
    sha256_file,
)


VALID_BASE_ARGV = [
    "--slotmem_memory_encoder_layers",
    "0-15",
    "--slotmem_memory_encoder_slots",
    "64",
]


def test_matrix_is_exactly_three_events_by_three_seeds() -> None:
    selection = {
        "schema_version": 1,
        "seeds": [0, 1, 2],
        "events": [
            {"event_id": "e79"},
            {"event_id": "e15"},
            {"event_id": "e16"},
        ],
    }

    matrix = build_matrix(selection, seeds=(0, 1, 2))

    assert len(matrix) == 9
    assert {(row.event_id, row.seed) for row in matrix} == {
        (event, seed) for event in ("e79", "e15", "e16") for seed in (0, 1, 2)
    }


def test_expected_rows_use_the_frozen_64_slot_universe() -> None:
    masks = {"semantic": list(range(8)), "random": list(range(8, 16))}

    assert _expected_rows("full_correct", masks) == list(range(64))
    assert _expected_rows("drop_subject", masks) == list(range(8, 64))


def test_teacher_absence_serializes_qstar_not_available() -> None:
    assert qstar_contract_or_status(event={"event_id": "e79"}, teacher=None) == {
        "status": "not_available",
        "reason": "independent_teacher_missing",
    }


def _block(tmp_path: Path) -> dict:
    event = {
        "event_id": "e79",
        "story_id": "79",
        "entity_uid": "79::Ana",
        "character_name": "Ana",
        "source_chunk_idx": 0,
        "target_chunk_idx": 6,
        "source_seed": 0,
        "target_seed": 0,
    }
    return {
        "event": event,
        "event_json": tmp_path / "event.json",
        "subject_subspace_manifest": tmp_path / "subject_subspace_manifest.json",
        "output": tmp_path / "e79" / "seed_0",
        "target_seed": 0,
        "donor": tmp_path / "donor.pt",
        "donor_manifest": tmp_path / "donors.json",
        "contract": {
            "event": event,
            "snapshot": {"path": str(tmp_path / "prefix_state.pt"), "sha256": "a" * 64},
            "base_inference_args": [
                "--json_path", str(tmp_path / "story.json"),
                "--max_memory_characters", "99",
                "--target_character", "leaky",
                "--fixed_reference_scope", "all_chunks",
            ],
        },
    }


def test_arm_orders_are_frozen_and_every_arm_reuses_snapshot_and_target_seed(tmp_path: Path) -> None:
    assert PREFLIGHT_ARMS == ("full_correct", "no_memory", "zero_path", "wrong_subject")
    assert FULL_ARMS == (
        "full_correct", "no_memory", "zero_path", "subject_only", "drop_subject",
        "random_only", "drop_random", "wrong_subject",
    )

    commands = build_block_commands(_block(tmp_path), python="python")

    assert tuple(commands) == FULL_ARMS
    snapshots = {command[command.index("--resume_state_path") + 1] for command in commands.values()}
    seeds = {command[command.index("--target_seed_override") + 1] for command in commands.values()}
    assert snapshots == {str(tmp_path / "prefix_state.pt")}
    assert seeds == {"0"}
    for command in commands.values():
        assert command[command.index("--max_memory_characters") + 1] == "4"
        assert command[command.index("--fixed_reference_scope") + 1] == "source_only"
        assert "--target_character" not in command
        assert "--subject_subspace_capture_path" not in command


def test_source_capture_path_is_a_read_only_runtime_output_not_a_frozen_arm_arg() -> None:
    assert normalized_frozen_args(["--fixed_reference_scope", "source_only"]) == normalized_frozen_args([
        "--fixed_reference_scope", "source_only",
        "--subject_subspace_capture_path", "source_capture.pt",
    ])


def _prepared_inputs(tmp_path: Path) -> Path:
    inputs = tmp_path / "inputs"
    records = []
    specs = (
        ("vistory79_song_yuchen_s2_s8", "79", "Song Yuchen", 2, 8, 6),
        ("vistory15_gu_zhenzhen_s8_s20", "15", "Gu Zhenzhen", 8, 20, 12),
        ("vistory16_chen_father_s1_s10", "16", "Chen Sihan's Father", 1, 10, 9),
    )
    for event_id, story_id, character, source_shot, target_shot, target_idx in specs:
        root = inputs / event_id
        root.mkdir(parents=True)
        event = {
            "schema_version": 1,
            "event_id": event_id,
            "story_id": story_id,
            "entity_uid": f"{story_id}::{character}",
            "character_name": character,
            "source_chunk_idx": 0,
            "target_chunk_idx": target_idx,
            "max_memory_characters": 4,
            "source_json_path": str(root / "story.json"),
            "reference_path": str(root / "00.jpg"),
        }
        (root / "story.json").write_text(json.dumps({
            "chunks": [{"content": f"chunk {index}", "character_list": [character]} for index in range(target_idx + 1)]
        }), encoding="utf-8")
        (root / "00.jpg").write_bytes(b"reference")
        event["reference_sha256"] = sha256_file(root / "00.jpg")
        event_path = root / "event.json"
        event_path.write_text(json.dumps(event), encoding="utf-8")
        event_manifest = {
            "schema_version": 1,
            "task_id": "vistorybench_subject_reappearance_v1",
            "dataset_commit": "92f845531b67e97a67ae04b256ec5d8c020e8341",
            "evaluator_commit": "b44ec9108668cc2bcc8c5280886b235e9fb8bea9",
            "seeds": [0, 1, 2],
            "event_id": event_id,
            "story_id": story_id,
            "character_name": character,
            "source_shot": source_shot,
            "target_shot": target_shot,
            "official_story": {
                "path": f"official/{story_id}.json",
                "sha256": next(
                    row["story_sha256"]
                    for row in json.loads(
                        (Path(__file__).parents[1] / "events" / "vistorybench_reappearance_v1.json").read_text(
                            encoding="utf-8"
                        )
                    )["events"]
                    if row["event_id"] == event_id
                ).casefold(),
            },
            "reference_path": f"{event_id}/00.jpg",
            "reference_sha256": sha256_file(root / "00.jpg"),
            "outputs": {"event": {
                "path": f"{event_id}/event.json",
                "sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(),
            }, "story": {
                "path": f"{event_id}/story.json",
                "sha256": sha256_file(root / "story.json"),
            }},
        }
        event_manifest_path = root / "manifest.json"
        event_manifest_path.write_text(json.dumps(event_manifest), encoding="utf-8")
        records.append({
            "event_id": event_id,
            "manifest_path": f"{event_id}/manifest.json",
            "manifest_sha256": hashlib.sha256(event_manifest_path.read_bytes()).hexdigest(),
        })
    manifest = inputs / "manifest.json"
    manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "task_id": "vistorybench_subject_reappearance_v1",
            "dataset_commit": "92f845531b67e97a67ae04b256ec5d8c020e8341",
            "evaluator_commit": "b44ec9108668cc2bcc8c5280886b235e9fb8bea9",
            "seeds": [0, 1, 2],
            "events": records,
        }),
        encoding="utf-8",
    )
    return manifest


def test_dry_run_writes_one_immutable_nine_block_manifest_without_gpu_or_qstar_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _prepared_inputs(tmp_path)
    base = tmp_path / "args.json"
    base.write_text(
        json.dumps({"argv": [*VALID_BASE_ARGV, "--seed_base", "42"]}),
        encoding="utf-8",
    )
    platform = tmp_path / "platform.json"
    platform.write_text("{}", encoding="utf-8")
    output = tmp_path / "run"

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("GPU launched")))
    assert main([
        "dry-run", "--inputs", str(inputs), "--output", str(output),
        "--base-inference-args", str(base), "--platform-manifest", str(platform),
        "--python", "python",
    ]) == 0

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["python"] == "python"
    assert len(manifest["blocks"]) == 9
    assert all(block["full_arms"] == list(FULL_ARMS) for block in manifest["blocks"])
    assert all(block["qstar"] == {"status": "not_available", "reason": "independent_teacher_missing"} for block in manifest["blocks"])
    assert all("qstar_command" not in block["commands"] for block in manifest["blocks"])
    assert all(block["commands"]["preflight"]["status"] == "blocked_missing_donor" for block in manifest["blocks"])
    assert all(block["commands"]["full"]["arm_order"] == list(FULL_ARMS) for block in manifest["blocks"])
    for block in manifest["blocks"]:
        semantic = block["commands"]["semantic_scores"]
        assert semantic[:3] == [
            str(Path(sys.executable).resolve()),
            "-m",
            "utest.source_semantic_scores",
        ]
        assert block["source_capture"] in semantic
        assert block["semantic_scores"] in semantic
        assert block["event_json"] in semantic
        assert not any(
            "target_latent" in arg or "target_frame" in arg for arg in semantic
        )
        assert "semantic_scores" not in block["required_external_inputs"]
        assert block["logs"]["semantic_scores_stdout"].endswith(
            "semantic_scores.stdout.log"
        )
        assert block["logs"]["semantic_scores_stderr"].endswith(
            "semantic_scores.stderr.log"
        )
        event = json.loads(Path(block["event_json"]).read_text(encoding="utf-8"))
        preview = block["commands"]["prefix_inference_args"]
        assert event["source_seed"] == event["target_seed"] == block["target_seed"]
        assert preview[preview.index("--target_seed_override") + 1] == str(block["target_seed"])
        assert build_runtime_contract(event, preview)["target_seed"] == block["target_seed"]
    with pytest.raises(FileExistsError):
        main([
            "dry-run", "--inputs", str(inputs), "--output", str(output),
            "--base-inference-args", str(base), "--platform-manifest", str(platform),
        ])


def test_dry_run_rejects_missing_memory_geometry_without_gpu_or_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _prepared_inputs(tmp_path)
    base = tmp_path / "args.json"
    base.write_text("[]", encoding="utf-8")
    platform = tmp_path / "platform.json"
    platform.write_text("{}", encoding="utf-8")
    output = tmp_path / "run"
    calls = []
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="actual=None.*frozen expected='0-15'"):
        main([
            "dry-run", "--inputs", str(inputs), "--output", str(output),
            "--base-inference-args", str(base), "--platform-manifest", str(platform),
        ])

    assert calls == []
    assert not (output / "run_manifest.json").exists()


def test_dry_run_rejects_legacy_32_slot_geometry_without_gpu_or_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _prepared_inputs(tmp_path)
    base = tmp_path / "args.json"
    base.write_text(json.dumps({"argv": [
        "--slotmem_memory_encoder_layers", "0-15",
        "--slotmem_memory_encoder_slots", "32",
    ]}), encoding="utf-8")
    platform = tmp_path / "platform.json"
    platform.write_text("{}", encoding="utf-8")
    output = tmp_path / "run"
    calls = []
    monkeypatch.setattr(
        "subprocess.run", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    with pytest.raises(ValueError, match="actual='32'.*frozen expected='64'"):
        main([
            "dry-run", "--inputs", str(inputs), "--output", str(output),
            "--base-inference-args", str(base), "--platform-manifest", str(platform),
        ])

    assert calls == []
    assert not (output / "run_manifest.json").exists()


def test_dry_run_accepts_32_then_64_duplicate_geometry_by_last_value(
    tmp_path: Path,
) -> None:
    inputs = _prepared_inputs(tmp_path)
    base = tmp_path / "args.json"
    base.write_text(json.dumps({"argv": [
        "--slotmem_memory_encoder_layers", "0-15",
        "--slotmem_memory_encoder_slots", "32",
        "--slotmem_memory_encoder_slots", "64",
    ]}), encoding="utf-8")
    platform = tmp_path / "platform.json"
    platform.write_text("{}", encoding="utf-8")
    output = tmp_path / "run"

    assert main([
        "dry-run", "--inputs", str(inputs), "--output", str(output),
        "--base-inference-args", str(base), "--platform-manifest", str(platform),
    ]) == 0
    assert (output / "run_manifest.json").is_file()


def test_dry_run_rejects_64_then_32_duplicate_geometry_by_last_value(
    tmp_path: Path,
) -> None:
    inputs = _prepared_inputs(tmp_path)
    base = tmp_path / "args.json"
    base.write_text(json.dumps({"argv": [
        "--slotmem_memory_encoder_layers", "0-15",
        "--slotmem_memory_encoder_slots", "64",
        "--slotmem_memory_encoder_slots", "32",
    ]}), encoding="utf-8")
    platform = tmp_path / "platform.json"
    platform.write_text("{}", encoding="utf-8")
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="actual='32'.*frozen expected='64'"):
        main([
            "dry-run", "--inputs", str(inputs), "--output", str(output),
            "--base-inference-args", str(base), "--platform-manifest", str(platform),
        ])
    assert not (output / "run_manifest.json").exists()


def test_dry_run_rejects_shell_text_base_arguments_without_partial_outputs(tmp_path: Path) -> None:
    inputs = _prepared_inputs(tmp_path)
    base = tmp_path / "args.txt"
    base.write_text("python infer.py --seed_base 42", encoding="utf-8")
    platform = tmp_path / "platform.json"
    platform.write_text("{}", encoding="utf-8")
    output = tmp_path / "run"

    with pytest.raises(json.JSONDecodeError):
        main([
            "dry-run", "--inputs", str(inputs), "--output", str(output),
            "--base-inference-args", str(base), "--platform-manifest", str(platform),
        ])
    assert not output.exists()


def _dry_run_manifest(tmp_path: Path) -> tuple[Path, dict]:
    inputs = _prepared_inputs(tmp_path)
    base = tmp_path / "args.json"
    base.write_text(json.dumps({"argv": VALID_BASE_ARGV}), encoding="utf-8")
    platform = tmp_path / "platform.json"
    platform.write_text("{}", encoding="utf-8")
    output = tmp_path / "run"
    assert main([
        "dry-run", "--inputs", str(inputs), "--output", str(output),
        "--base-inference-args", str(base), "--platform-manifest", str(platform),
        "--python", "python",
    ]) == 0
    path = output / "run_manifest.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _materialize_existing_prefix_contract(row: dict, *, slot_count: int = 64) -> Path:
    snapshot = Path(row["prefix_snapshot"])
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(f"existing {slot_count}-slot prefix".encode())
    event = json.loads(Path(row["event_json"]).read_text(encoding="utf-8"))
    inference_args = [
        *row["commands"]["prefix_inference_args"],
        f"--slotmem_memory_encoder_slots={slot_count}",
    ]
    contract = {
        "event": event,
        "snapshot": {
            "path": str(snapshot.resolve()),
            "sha256": sha256_file(snapshot),
        },
        "runtime_contract": build_runtime_contract(event, inference_args),
        "base_inference_args": inference_args,
    }
    contract_path = snapshot.parent / "prefix_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return contract_path


def test_loader_rejects_v1_manifest_with_clear_semantic_migration_message(
    tmp_path: Path,
) -> None:
    path, manifest = _dry_run_manifest(tmp_path)
    manifest["schema_version"] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="predates generated semantic scores.*rerun dry-run",
    ):
        main(["probe", "--manifest", str(path)])


def test_existing_manifest_rejects_rehashed_legacy_32_base_before_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest = _dry_run_manifest(tmp_path)
    base_path = Path(manifest["base_inference_args"])
    base_path.write_text(json.dumps({"argv": [
        "--slotmem_memory_encoder_layers", "0-15",
        "--slotmem_memory_encoder_slots", "32",
    ]}), encoding="utf-8")
    manifest["base_inference_args_sha256"] = sha256_file(base_path)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._run_logged",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="actual='32'.*frozen expected='64'"):
        main(["prefix", "--manifest", str(path)])

    assert calls == []


@pytest.mark.parametrize(
    "tamper",
    ("preview_missing_geometry", "prefix_missing_separator", "tail_missing_geometry", "synchronized_32"),
)
def test_existing_manifest_rejects_invalid_block_geometry_before_gpu(
    tamper: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest = _dry_run_manifest(tmp_path)
    commands = manifest["blocks"][0]["commands"]
    preview, prefix = commands["prefix_inference_args"], commands["prefix"]
    if tamper == "prefix_missing_separator":
        prefix.remove("--")
    elif tamper == "synchronized_32":
        preview.extend(["--slotmem_memory_encoder_slots", "32"])
        prefix.extend(["--slotmem_memory_encoder_slots", "32"])
    else:
        target = preview if tamper == "preview_missing_geometry" else prefix
        index = target.index("--slotmem_memory_encoder_slots")
        del target[index : index + 2]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._run_logged",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="SlotMem donor protocol mismatch|separator"):
        main(["prefix", "--manifest", str(path)])

    assert calls == []


def test_existing_manifest_accepts_block_geometry_duplicates_32_then_64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest = _dry_run_manifest(tmp_path)
    commands = manifest["blocks"][0]["commands"]
    for argv in (commands["prefix_inference_args"], commands["prefix"]):
        argv.extend([
            "--slotmem_memory_encoder_slots", "32",
            "--slotmem_memory_encoder_slots", "64",
        ])
    path.write_text(json.dumps(manifest), encoding="utf-8")
    calls = []

    def gpu_boundary(*args, **kwargs) -> None:
        calls.append((args, kwargs))
        raise RuntimeError("gpu boundary reached")

    monkeypatch.setattr(
        "utest.subject_reappearance_harness._run_logged", gpu_boundary
    )

    with pytest.raises(RuntimeError, match="gpu boundary reached"):
        main(["prefix", "--manifest", str(path), "--seed", "0"])

    assert len(calls) == 1


def test_existing_manifest_rejects_block_geometry_duplicates_64_then_32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest = _dry_run_manifest(tmp_path)
    commands = manifest["blocks"][0]["commands"]
    for argv in (commands["prefix_inference_args"], commands["prefix"]):
        argv.extend([
            "--slotmem_memory_encoder_slots", "64",
            "--slotmem_memory_encoder_slots", "32",
        ])
    path.write_text(json.dumps(manifest), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._run_logged",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="actual='32'.*frozen expected='64'"):
        main(["prefix", "--manifest", str(path), "--seed", "0"])

    assert calls == []


@pytest.mark.parametrize("stage", ("prefix", "resume"))
def test_existing_32_slot_prefix_contract_is_never_run_or_archived(
    stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest = _dry_run_manifest(tmp_path)
    row = manifest["blocks"][0]
    contract_path = _materialize_existing_prefix_contract(row, slot_count=32)
    calls = []
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._run_logged",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(MemoryGeometryError, match="actual='32'.*frozen expected='64'"):
        main([
            stage,
            "--manifest", str(path),
            "--event-id", row["event_id"],
            "--seed", str(row["seed"]),
        ])

    assert calls == []
    assert contract_path.is_file()
    assert not contract_path.parent.with_name("prefix.failed_1").exists()


@pytest.mark.parametrize("stage", ("prefix", "resume"))
def test_second_block_32_slot_prefix_contract_stops_the_whole_batch_before_gpu(
    stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest = _dry_run_manifest(tmp_path)
    bad_row = manifest["blocks"][1]
    contract_path = _materialize_existing_prefix_contract(bad_row, slot_count=32)
    calls = []
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._run_logged",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(MemoryGeometryError, match="actual='32'.*frozen expected='64'"):
        main([stage, "--manifest", str(path)])

    assert calls == []
    assert contract_path.is_file()
    assert not contract_path.parent.with_name("prefix.failed_1").exists()


def test_non_geometry_partial_prefix_resume_still_recovers_before_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest = _dry_run_manifest(tmp_path)
    row = manifest["blocks"][0]
    prefix = Path(row["prefix_snapshot"]).parent
    contract_path = _materialize_existing_prefix_contract(row)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["event"]["character_name"] = "corrupt non-geometry identity"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    calls = []

    def rerun_boundary(*args, **kwargs) -> None:
        calls.append((args, kwargs))
        raise RuntimeError("prefix rerun reached")

    monkeypatch.setattr(
        "utest.subject_reappearance_harness._run_logged", rerun_boundary
    )

    with pytest.raises(RuntimeError, match="prefix rerun reached"):
        main([
            "resume",
            "--manifest", str(path),
            "--event-id", row["event_id"],
            "--seed", str(row["seed"]),
        ])

    assert len(calls) == 1
    assert not prefix.exists()
    assert (prefix.with_name("prefix.failed_1") / "prefix_contract.json").is_file()


@pytest.mark.parametrize("missing", ("runtime_contract", "frozen_args"))
def test_missing_prefix_runtime_structure_is_archived_and_rerun_on_resume(
    missing: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest = _dry_run_manifest(tmp_path)
    row = manifest["blocks"][0]
    prefix = Path(row["prefix_snapshot"]).parent
    contract_path = _materialize_existing_prefix_contract(row)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if missing == "runtime_contract":
        del contract["runtime_contract"]
    else:
        del contract["runtime_contract"]["frozen_args"]
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    calls = []

    def rerun_boundary(*args, **kwargs) -> None:
        calls.append((args, kwargs))
        raise RuntimeError("prefix rerun reached")

    monkeypatch.setattr(
        "utest.subject_reappearance_harness._run_logged", rerun_boundary
    )

    with pytest.raises(RuntimeError, match="prefix rerun reached"):
        main([
            "resume",
            "--manifest", str(path),
            "--event-id", row["event_id"],
            "--seed", str(row["seed"]),
        ])

    assert len(calls) == 1
    assert not prefix.exists()
    assert (prefix.with_name("prefix.failed_1") / "prefix_contract.json").is_file()


@pytest.mark.parametrize("malformation", ("missing", "mapping", "non_string"))
def test_malformed_prefix_base_args_are_archived_and_rerun_on_resume(
    malformation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest = _dry_run_manifest(tmp_path)
    row = manifest["blocks"][0]
    prefix = Path(row["prefix_snapshot"]).parent
    contract_path = _materialize_existing_prefix_contract(row)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if malformation == "missing":
        del contract["base_inference_args"]
    elif malformation == "mapping":
        contract["base_inference_args"] = {"argv": VALID_BASE_ARGV}
    else:
        contract["base_inference_args"].append(64)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    calls = []

    def rerun_boundary(*args, **kwargs) -> None:
        calls.append((args, kwargs))
        raise RuntimeError("prefix rerun reached")

    monkeypatch.setattr(
        "utest.subject_reappearance_harness._run_logged", rerun_boundary
    )

    with pytest.raises(RuntimeError, match="prefix rerun reached"):
        main([
            "resume",
            "--manifest", str(path),
            "--event-id", row["event_id"],
            "--seed", str(row["seed"]),
        ])

    assert len(calls) == 1
    assert not prefix.exists()
    assert (prefix.with_name("prefix.failed_1") / "prefix_contract.json").is_file()


@pytest.mark.parametrize(
    "tamper",
    ("module", "target_arg", "output", "external", "logs", "python"),
)
def test_loader_rejects_tampered_semantic_producer_contract(
    tamper: str, tmp_path: Path
) -> None:
    path, manifest = _dry_run_manifest(tmp_path)
    block = manifest["blocks"][0]
    if tamper == "module":
        block["commands"]["semantic_scores"][2] = "utest.untrusted_producer"
    elif tamper == "target_arg":
        block["commands"]["semantic_scores"].extend(["--target-frame", "future.pt"])
    elif tamper == "output":
        changed = str(Path(block["block_dir"]) / "elsewhere" / "semantic.json")
        block["semantic_scores"] = changed
        command = block["commands"]["semantic_scores"]
        command[command.index("--output") + 1] = changed
    elif tamper == "external":
        block["required_external_inputs"]["semantic_scores"] = block["semantic_scores"]
    elif tamper == "logs":
        block["logs"]["semantic_scores_stdout"] = str(tmp_path / "redirected.log")
    else:
        block["commands"]["semantic_scores"][0] = str(
            tmp_path / "wrapper" / Path(sys.executable).name
        )
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="semantic|Python interpreter"):
        main(["probe", "--manifest", str(path)])


def test_loader_rejects_self_consistent_block_relocation(tmp_path: Path) -> None:
    path, manifest = _dry_run_manifest(tmp_path)
    block = manifest["blocks"][0]
    original = block["block_dir"]
    relocated = str(path.parent / "relocated" / block["event_id"] / f"seed_{block['seed']}")
    block["block_dir"] = relocated
    for key in (
        "event_json", "prefix_snapshot", "command_artifact", "source_qualification",
        "subject_subspace_manifest", "source_capture", "semantic_scores",
    ):
        block[key] = block[key].replace(original, relocated)
    for key, value in block["logs"].items():
        block["logs"][key] = value.replace(original, relocated)
    for command_name in ("prefix", "semantic_scores", "probe"):
        block["commands"][command_name] = [
            arg.replace(original, relocated)
            for arg in block["commands"][command_name]
        ]
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="block directory contract"):
        main(["probe", "--manifest", str(path)])


@pytest.mark.parametrize(
    "tamper",
    ("event", "source", "semantic", "output", "extra", "python"),
)
def test_loader_rejects_tampered_probe_command_contract(
    tamper: str, tmp_path: Path
) -> None:
    path, manifest = _dry_run_manifest(tmp_path)
    command = manifest["blocks"][0]["commands"]["probe"]
    if tamper == "python":
        command[0] = str(tmp_path / "wrapper" / Path(sys.executable).name)
    elif tamper == "extra":
        command.extend(["--target-frame", "future.pt"])
    else:
        option = {
            "event": "--event",
            "source": "--source-capture",
            "semantic": "--semantic-scores",
            "output": "--output",
        }[tamper]
        command[command.index(option) + 1] = str(tmp_path / f"changed-{tamper}")
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="probe command contract"):
        main(["probe", "--manifest", str(path)])


def _validated_block(tmp_path: Path) -> tuple[dict, Path]:
    block = _block(tmp_path)
    root = Path(block["output"])
    root.mkdir(parents=True)
    Path(block["event_json"]).write_text(json.dumps(block["event"]), encoding="utf-8")
    event_file_sha = sha256_file(Path(block["event_json"]))
    source_layers = {
        str(layer): torch.arange(192, dtype=torch.float32).reshape(64, 3) + layer
        for layer in range(16)
    }
    capture_path = root / "subspace" / "source_capture.pt"
    capture_path.parent.mkdir(parents=True)
    torch.save({"captures": [
        {"character": "Ana", "bank": 0, "layer": layer, "encoded_slots": source_layers[str(layer)]}
        for layer in range(16)
    ]}, capture_path)
    block["source_capture"] = str(capture_path)
    rankings = {
        f"bank_0/group_{group}": {
            "bank": 0,
            "layer_group": group,
            "member_layers": list(members),
            "source_payload_sha256_by_layer": {
                str(layer): capture_tensor_sha256(source_layers[str(layer)]) for layer in members
            },
            "semantic": list(range(64)),
            "visual_cf": None,
            "reference": None,
        }
        for group, members in FROZEN_LAYER_GROUPS.items()
    }
    mask = build_mask_manifest(
        inputs={"source_capture_sha256": sha256_file(capture_path)},
        rankings=rankings,
        event=block["event"],
        seed=0,
    )
    mask_path = Path(block["subject_subspace_manifest"])
    mask_path.write_text(json.dumps(mask), encoding="utf-8")
    mask_sha = mask["mask_manifest_sha256"]
    manifest_file_sha = sha256_file(mask_path)
    layer_masks = {}
    source_hashes = {}
    for row in mask["layers"]:
        for layer in row["member_layers"]:
            layer_masks[str(layer)] = row
            source_hashes[str(layer)] = capture_tensor_sha256(source_layers[str(layer)])
    donor_layers = {layer: torch.full_like(tensor, 99.0) for layer, tensor in source_layers.items()}
    donor_path = root / "donor.pt"
    torch.save({
        "format": "slotmem_donor_payload_v2",
        "event": {"story_id": "donor", "entity_uid": "donor::Other", "character_name": "Other"},
        "payloads": {"Other|0": {"__layerwise__": True, LAYERS_KEY: donor_layers}},
    }, donor_path)
    donor_manifest_path = root / "donor_manifest.json"
    donor_manifest_path.write_text(json.dumps({"pairs": [{
        "target_story_id": "79", "target_entity_uid": "79::Ana",
        "donor_story_id": "donor", "donor_entity_uid": "donor::Other",
        "payload_path": str(donor_path.resolve()), "payload_sha256": sha256_file(donor_path),
        "payload_key": "Other|0", "coarse_class": "person", "colour": "unknown",
        "character_count": 1, "source_visible": True, "gap_bucket": "long",
        "slot_shape": {layer: [64, 3] for layer in donor_layers}, "selection_seed": 0,
    }]}), encoding="utf-8")
    block["donor"] = {
        "payload_path": str(donor_path.resolve()),
        "manifest_path": str(donor_manifest_path.resolve()),
        "payload_key": "Other|0",
    }
    (root / "source_qualification.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    for arm in PREFLIGHT_ARMS:
        arm_dir = root / "arms" / arm
        arm_dir.mkdir(parents=True)
        selected = {
            layer: (
                [] if arm == "no_memory"
                else row["semantic_top8"] if arm == "wrong_subject"
                else list(range(64))
            )
            for layer, row in layer_masks.items()
        }
        returned_layers = {}
        for layer, row in layer_masks.items():
            source = source_layers[layer]
            if arm == "no_memory":
                continue
            if arm == "full_correct":
                transformed = source
            elif arm == "zero_path":
                transformed = torch.zeros_like(source)
            else:
                transformed = transform_slot_rows(source, arm, row, donor_layers[layer])
            returned_layers[layer] = transformed
        returned_payload = (
            None
            if arm == "no_memory"
            else _payload_sha256({"tokens": {"__layerwise__": True, LAYERS_KEY: returned_layers}})
        )
        audit = {
            "target_read_hits": 1,
            "target_source_non_null_reads": 1,
            "target_returned_non_null_reads": 0 if arm == "no_memory" else 1,
            "subject_subspace_contract": {
                "event_id": "e79",
                "seed": 0,
                "target_evidence_read": False,
                "mask_manifest_sha256": mask_sha,
                "source_capture_sha256": sha256_file(capture_path),
                "manifest_file_sha256": manifest_file_sha,
                "event_file_sha256": event_file_sha,
            },
            "read_records": [{
                "chunk_idx": 6,
                "character": "Ana",
                "bank": 0,
                "source_manifest_sha256_by_layer": source_hashes,
                "selected_indices_by_layer": selected,
                "returned_sha256": returned_payload,
                "returned_manifest_sha256_by_layer": (
                    {layer: capture_tensor_sha256(tensor) for layer, tensor in returned_layers.items()}
                ),
            }],
        }
        (arm_dir / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
        (arm_dir / "chunk_006.metadata.json").write_text(json.dumps({
            "reference_conditioning": {
                "fixed_reference_scope": "source_only",
                "fixed_reference_used": False,
                "random_reference_source": "prior_chunk_tail",
            }
        }), encoding="utf-8")
        (arm_dir / "efficiency.json").write_text(json.dumps({
            "chunks": [{
                "chunk_idx": 6,
                "last_sparse_role_memory_stats_by_layer": {
                    "15": {
                        "enabled": 1.0 if arm == "full_correct" else 0.0,
                        "selected_memory_tokens": 8 if arm == "full_correct" else 0,
                        "effective_delta_norm": 0.1 if arm == "full_correct" else 0.0,
                    }
                },
            }]
        }), encoding="utf-8")
    import imageio.v3 as iio
    import numpy as np

    frames = {
        "no_memory": np.zeros((2, 16, 16, 3), dtype=np.uint8),
        "zero_path": np.zeros((2, 16, 16, 3), dtype=np.uint8),
        "full_correct": np.full((2, 16, 16, 3), 40, dtype=np.uint8),
        "wrong_subject": np.full((2, 16, 16, 3), 80, dtype=np.uint8),
    }
    for arm, video in frames.items():
        iio.imwrite(root / "arms" / arm / "chunk_006.mp4", video, fps=4)
    return block, root


def test_preflight_validation_accepts_only_measured_source_only_contract(tmp_path: Path) -> None:
    block, _ = _validated_block(tmp_path)
    report = validate_block(block, arms=PREFLIGHT_ARMS)
    assert report["status"] == "passed"
    assert report["arms"] == list(PREFLIGHT_ARMS)
    assert report["decoded_preflight"]["path_equivalent"] is True
    assert report["decoded_preflight"]["content_influence_measured"] is True


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("qualification", "qualification"),
        ("target_read", "target memory"),
        ("mask_sha", "mask contract"),
        ("reference", "initial reference"),
        ("injection", "measured injection"),
        ("injection_nan", "not finite"),
    ],
)
def test_block_validation_fails_closed(fault: str, message: str, tmp_path: Path) -> None:
    block, root = _validated_block(tmp_path)
    if fault == "qualification":
        path = root / "source_qualification.json"
        value = {"status": "failed"}
    elif fault == "target_read":
        path = root / "arms" / "full_correct" / "audit.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["target_read_hits"] = 0
    elif fault == "mask_sha":
        path = root / "arms" / "full_correct" / "audit.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["subject_subspace_contract"]["mask_manifest_sha256"] = "f" * 64
    elif fault == "reference":
        path = root / "arms" / "full_correct" / "chunk_006.metadata.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["reference_conditioning"]["fixed_reference_used"] = True
    elif fault == "injection":
        path = root / "arms" / "full_correct" / "efficiency.json"
        value = {"chunks": [{"chunk_idx": 6, "last_sparse_role_memory_stats_by_layer": {
            "15": {"enabled": 1, "selected_memory_tokens": 8, "effective_delta_norm": 0.0}
        }}]}
    else:
        path = root / "arms" / "full_correct" / "efficiency.json"
        value = {"chunks": [{"chunk_idx": 6, "last_sparse_role_memory_stats_by_layer": {
            "15": {"enabled": 1, "selected_memory_tokens": 8, "effective_delta_norm": float("nan")}
        }}]}
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_block(block, arms=PREFLIGHT_ARMS)


def test_source_chunk_injection_cannot_satisfy_target_measurement(tmp_path: Path) -> None:
    block, root = _validated_block(tmp_path)
    path = root / "arms" / "full_correct" / "efficiency.json"
    path.write_text(json.dumps({
        "chunks": [
            {"chunk_idx": 0, "last_sparse_role_memory_stats_by_layer": {
                "15": {"enabled": 1, "selected_memory_tokens": 8, "effective_delta_norm": 1.0}
            }},
            {"chunk_idx": 6, "last_sparse_role_memory_stats_by_layer": {
                "15": {"enabled": 1, "selected_memory_tokens": 8, "effective_delta_norm": 0.0}
            }},
        ]
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="measured injection"):
        validate_block(block, arms=PREFLIGHT_ARMS)


@pytest.mark.parametrize(
    "target_stats",
    [
        {"injection": {"raw_delta_norm": 2.0, "effective_delta_norm": 0.0}},
        {"last_jigsaw_stage2_writer_stats": {"residual_norm": 2.0}},
        {"last_sparse_role_memory_stats_by_layer": {
            "15": {"enabled": 1, "selected_memory_tokens": 8, "effective_delta_norm": float("inf")}
        }},
    ],
)
def test_measured_injection_rejects_non_runtime_or_nonfinite_evidence(
    target_stats: dict, tmp_path: Path
) -> None:
    block, root = _validated_block(tmp_path)
    path = root / "arms" / "full_correct" / "efficiency.json"
    path.write_text(json.dumps({"chunks": [{"chunk_idx": 6, **target_stats}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="measured injection|not finite"):
        validate_block(block, arms=PREFLIGHT_ARMS)


def test_returned_payload_hash_must_be_recomputed_from_frozen_source(tmp_path: Path) -> None:
    block, root = _validated_block(tmp_path)
    path = root / "arms" / "full_correct" / "audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    audit["read_records"][0]["returned_sha256"] = "d" * 64
    path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError, match="returned payload SHA differs"):
        validate_block(block, arms=PREFLIGHT_ARMS)


@pytest.mark.parametrize(
    ("arm", "value", "message"),
    [
        ("zero_path", 10, "path-equivalent"),
        ("wrong_subject", 40, "content influence"),
    ],
)
def test_decoded_preflight_mechanical_gate_fails_closed(
    arm: str, value: int, message: str, tmp_path: Path
) -> None:
    block, root = _validated_block(tmp_path)
    import imageio.v3 as iio
    import numpy as np

    iio.imwrite(
        root / "arms" / arm / "chunk_006.mp4",
        np.full((2, 16, 16, 3), value, dtype=np.uint8),
        fps=4,
    )
    with pytest.raises(ValueError, match=message):
        validate_block(block, arms=PREFLIGHT_ARMS)


def test_probe_stage_materializes_and_validates_semantic_scores_before_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _prepared_inputs(tmp_path)
    base = tmp_path / "args.json"
    base.write_text(json.dumps({"argv": VALID_BASE_ARGV}), encoding="utf-8")
    platform = tmp_path / "platform.json"
    platform.write_text("{}", encoding="utf-8")
    output = tmp_path / "run"
    assert main([
        "dry-run", "--inputs", str(inputs), "--output", str(output),
        "--base-inference-args", str(base), "--platform-manifest", str(platform),
    ]) == 0
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    block = manifest["blocks"][0]
    Path(block["source_qualification"]).write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    capture = Path(block["block_dir"]) / "subspace" / "source_capture.pt"
    capture.parent.mkdir(parents=True, exist_ok=True)
    capture.write_bytes(b"capture")
    calls = []

    def fake_run(command, **_kwargs):
        module = command[2]
        calls.append(f"run:{module}")
        if module == "utest.source_semantic_scores":
            Path(block["semantic_scores"]).write_text("{}", encoding="utf-8")

    def fake_validate(**kwargs):
        assert kwargs == {
            "event_path": Path(block["event_json"]),
            "source_capture_path": Path(block["source_capture"]),
            "scores_path": Path(block["semantic_scores"]),
            "repo_root": Path(__file__).parents[2].resolve(),
        }
        assert kwargs["scores_path"].is_file()
        calls.append("validate")
        return {}

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "utest.subject_reappearance_harness.validate_source_semantic_scores_file",
        fake_validate,
    )

    assert main([
        "probe", "--manifest", str(output / "run_manifest.json"),
        "--event-id", block["event_id"], "--seed", str(block["seed"]),
    ]) == 0

    assert calls == [
        "run:utest.source_semantic_scores",
        "validate",
        "run:utest.subject_subspace_probe",
    ]
    assert Path(block["logs"]["semantic_scores_stdout"]).is_file()
    assert Path(block["logs"]["semantic_scores_stderr"]).is_file()


def _semantic_runtime_row(tmp_path: Path) -> dict:
    root = tmp_path.resolve()
    subspace = root / "subspace"
    subspace.mkdir(parents=True, exist_ok=True)
    event = root / "event.json"
    capture = subspace / "source_capture.pt"
    scores = subspace / "semantic_scores.json"
    mask = subspace / "subject_subspace_manifest.json"
    return {
        "block_dir": str(root),
        "seed": 0,
        "event_json": str(event),
        "source_capture": str(capture),
        "semantic_scores": str(scores),
        "subject_subspace_manifest": str(mask),
        "required_external_inputs": {},
        "commands": {
            "semantic_scores": [
                str(Path(sys.executable).resolve()),
                "-m", "utest.source_semantic_scores",
                "--event", str(event),
                "--source-capture", str(capture),
                "--output", str(scores),
                "--repo-root", str(Path(__file__).parents[2].resolve()),
            ],
            "probe": [
                str(Path(sys.executable).resolve()),
                "-m", "utest.subject_subspace_probe",
                "--event", str(event),
                "--source-capture", str(capture),
                "--semantic-scores", str(scores),
                "--output", str(mask),
                "--seed", "0",
            ],
        },
        "logs": {
            "semantic_scores_stdout": str(subspace / "semantic_scores.stdout.log"),
            "semantic_scores_stderr": str(subspace / "semantic_scores.stderr.log"),
            "probe_stdout": str(subspace / "stdout.log"),
            "probe_stderr": str(subspace / "stderr.log"),
        },
    }


def test_existing_semantic_scores_are_validated_without_being_reproduced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _semantic_runtime_row(tmp_path)
    capture = Path(row["source_capture"])
    scores = Path(row["semantic_scores"])
    capture.write_bytes(b"capture")
    scores.write_bytes(b"frozen scores")
    calls = []

    monkeypatch.setattr(
        "utest.subject_reappearance_harness._run_logged",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("existing scores were reproduced")
        ),
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness.validate_source_semantic_scores_file",
        lambda **_kwargs: calls.append("validated"),
    )

    _ensure_semantic_scores(row)

    assert calls == ["validated"]
    assert scores.read_bytes() == b"frozen scores"


def test_invalid_existing_semantic_scores_are_rejected_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _semantic_runtime_row(tmp_path)
    capture = Path(row["source_capture"])
    scores = Path(row["semantic_scores"])
    capture.write_bytes(b"capture")
    scores.write_bytes(b"tampered scores")

    monkeypatch.setattr(
        "utest.subject_reappearance_harness._run_logged",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid scores were overwritten")
        ),
    )

    def reject(**_kwargs):
        raise ValueError("semantic score provenance mismatch")

    monkeypatch.setattr(
        "utest.subject_reappearance_harness.validate_source_semantic_scores_file",
        reject,
    )

    with pytest.raises(ValueError, match="provenance mismatch"):
        _ensure_semantic_scores(row)

    assert scores.read_bytes() == b"tampered scores"
    assert not Path(row["logs"]["semantic_scores_stdout"]).exists()
    assert not Path(row["logs"]["semantic_scores_stderr"]).exists()


def test_probe_resume_validates_scores_before_skipping_a_completed_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _semantic_runtime_row(tmp_path)
    capture = Path(row["source_capture"])
    scores = Path(row["semantic_scores"])
    event = Path(row["event_json"])
    qualification = tmp_path / "source_qualification.json"
    mask = Path(row["subject_subspace_manifest"])
    capture.write_bytes(b"capture")
    scores.write_text("{}", encoding="utf-8")
    event.write_text("{}", encoding="utf-8")
    qualification.write_text('{"status": "passed"}', encoding="utf-8")
    mask.write_text("{}", encoding="utf-8")
    calls = []
    row.update({
        "event_id": "event",
        "seed": 0,
        "source_qualification": str(qualification),
        "subject_subspace_manifest": str(mask),
    })

    monkeypatch.setattr(
        "utest.subject_reappearance_harness.validate_source_semantic_scores_file",
        lambda **_kwargs: calls.append("semantic"),
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness.validate_subject_subspace_manifest",
        lambda *_args, **_kwargs: calls.append("mask"),
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._load_source_slots",
        lambda *_args, **_kwargs: calls.append("slots"),
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._run_logged",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed probe was rerun")
        ),
    )

    _execute_stage({"blocks": [row]}, "probe", resume=True)

    assert calls == ["semantic", "mask", "slots"]


def test_failed_preflight_validation_leaves_no_partial_block_events(tmp_path: Path) -> None:
    inputs = _prepared_inputs(tmp_path)
    selection = json.loads(inputs.read_text(encoding="utf-8"))
    selection["events"][-1]["manifest_sha256"] = "0" * 64
    inputs.write_text(json.dumps(selection), encoding="utf-8")
    base = tmp_path / "args.json"
    base.write_text(json.dumps({"argv": VALID_BASE_ARGV}), encoding="utf-8")
    platform = tmp_path / "platform.json"
    platform.write_text("{}", encoding="utf-8")
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="manifest SHA-256"):
        main([
            "dry-run", "--inputs", str(inputs), "--output", str(output),
            "--base-inference-args", str(base), "--platform-manifest", str(platform),
        ])
    assert not list(output.rglob("event.json"))


def test_prepared_manifest_cannot_replace_the_frozen_official_story_hash(tmp_path: Path) -> None:
    inputs = _prepared_inputs(tmp_path)
    selection = json.loads(inputs.read_text(encoding="utf-8"))
    item = selection["events"][0]
    event_manifest_path = inputs.parent / item["manifest_path"]
    event_manifest = json.loads(event_manifest_path.read_text(encoding="utf-8"))
    event_manifest["official_story"]["sha256"] = "0" * 64
    event_manifest_path.write_text(json.dumps(event_manifest), encoding="utf-8")
    item["manifest_sha256"] = sha256_file(event_manifest_path)
    inputs.write_text(json.dumps(selection), encoding="utf-8")
    base, platform = tmp_path / "args.json", tmp_path / "platform.json"
    base.write_text(json.dumps({"argv": VALID_BASE_ARGV}), encoding="utf-8")
    platform.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="official story"):
        main([
            "dry-run", "--inputs", str(inputs), "--output", str(tmp_path / "run"),
            "--base-inference-args", str(base), "--platform-manifest", str(platform),
        ])


def test_failed_command_logs_are_retained_without_blocking_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = iter((RuntimeError("failed"), None))

    def fake_run(*args, **kwargs):
        result = next(calls)
        if result:
            raise result

    monkeypatch.setattr("subprocess.run", fake_run)
    stdout, stderr = tmp_path / "stdout.log", tmp_path / "stderr.log"
    with pytest.raises(RuntimeError, match="failed"):
        _run_logged(["python"], stdout, stderr)
    _run_logged(["python"], stdout, stderr)

    assert stdout.is_file() and stderr.is_file()
    assert (tmp_path / "stdout.retry_1.log").is_file()
    assert (tmp_path / "stderr.retry_1.log").is_file()


def test_partial_prefix_is_preserved_and_cleared_for_resume(tmp_path: Path) -> None:
    block = tmp_path / "event" / "seed_0"
    prefix = block / "prefix"
    prefix.mkdir(parents=True)
    (prefix / "partial.bin").write_bytes(b"partial")
    capture = block / "subspace" / "source_capture.pt"
    capture.parent.mkdir(parents=True)
    capture.write_bytes(b"partial capture")
    row = {
        "block_dir": str(block),
        "source_capture": str(capture),
        "command_artifact": str(block / "stage_commands.json"),
    }

    _recover_partial_prefix(row)

    assert not prefix.exists() and not capture.exists()
    assert (block / "prefix.failed_1" / "partial.bin").read_bytes() == b"partial"
    assert (block / "subspace" / "source_capture.pt.failed_1").read_bytes() == b"partial capture"


def test_resume_accepts_an_existing_64_slot_prefix_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path, manifest = _dry_run_manifest(tmp_path)
    row = manifest["blocks"][0]
    snapshot = Path(row["prefix_snapshot"])
    _materialize_existing_prefix_contract(row)
    calls = []
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._run_logged",
        lambda *args, **kwargs: calls.append("gpu"),
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._freeze_or_load_command_artifact",
        lambda actual_row, actual_contract: calls.append(
            (actual_row["event_id"], actual_contract["snapshot"]["sha256"])
        ),
    )

    _execute_stage(
        manifest,
        "prefix",
        event_id=row["event_id"],
        seed=int(row["seed"]),
        resume=True,
    )

    assert calls == [(row["event_id"], sha256_file(snapshot))]


def test_partial_arm_resume_skips_valid_and_archives_invalid_outputs(tmp_path: Path) -> None:
    block, root = _validated_block(tmp_path)
    valid = root / "arms" / "full_correct"
    assert _resume_completed_arm(block, "full_correct", valid) is True
    assert valid.is_dir()

    invalid = root / "arms" / "wrong_subject"
    assert _resume_completed_arm(block, "wrong_subject", invalid) is True
    (invalid / "stdout.log").write_text("old attempt", encoding="utf-8")
    audit_path = invalid / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["read_records"][0]["returned_sha256"] = "d" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    assert _resume_completed_arm(block, "wrong_subject", invalid) is False
    assert not invalid.exists()
    assert (root / "arms" / "wrong_subject.failed_1" / "stdout.log").read_text(
        encoding="utf-8"
    ) == "old attempt"


def test_full_revalidates_preflight_with_the_validated_prefix_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    block_dir = tmp_path / "block"
    block_dir.mkdir()
    qualification = block_dir / "source_qualification.json"
    qualification.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    preflight = block_dir / "preflight" / "validation.json"
    preflight.parent.mkdir()
    preflight.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    (block_dir / "event.json").write_text("{}", encoding="utf-8")
    contract = {"runtime_contract": {"target_seed": 0}, "snapshot": {"path": "unused"}}
    row = {
        "event_id": "event", "seed": 0, "target_seed": 0,
        "block_dir": str(block_dir), "source_qualification": str(qualification),
        "commands": {"full": {"status": "deferred_until_prefix", "arm_order": list(FULL_ARMS)}},
        "event_json": str(block_dir / "event.json"),
        "subject_subspace_manifest": str(block_dir / "mask.json"),
        "source_capture": str(block_dir / "capture.pt"), "donor": {},
    }
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._validated_prefix_contract", lambda actual: contract
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._validate_donor_target_compatibility",
        lambda _row: (_ for _ in ()).throw(
            AssertionError("full must rely on its passed preflight")
        ),
    )

    def checked(block: dict, **kwargs):
        assert block["contract"] is contract
        raise RuntimeError("preflight runtime checked")

    monkeypatch.setattr("utest.subject_reappearance_harness.validate_block", checked)
    with pytest.raises(RuntimeError, match="runtime checked"):
        _execute_stage({"blocks": [row]}, "full")


def _preflight_execution_row(tmp_path: Path) -> tuple[dict, Path]:
    block, block_dir = _validated_block(tmp_path)
    row = {
        "event_id": block["event"]["event_id"],
        "seed": 0,
        "target_seed": 0,
        "block_dir": str(block_dir),
        "source_qualification": str(block_dir / "source_qualification.json"),
        "commands": {
            "preflight": {
                "status": "deferred_until_prefix",
                "arm_order": list(PREFLIGHT_ARMS),
            }
        },
        "event_json": str(block["event_json"]),
        "subject_subspace_manifest": str(block["subject_subspace_manifest"]),
        "source_capture": str(block["source_capture"]),
        "donor": block["donor"],
    }
    return row, block_dir


def _rewrite_execution_donor(
    row: dict,
    *,
    hidden_dimension: int | None = None,
    slot_count: int | None = None,
    missing_layer: str | None = None,
    donor_bank: int | None = None,
) -> None:
    donor_path = Path(row["donor"]["payload_path"])
    artifact = torch.load(donor_path, map_location="cpu", weights_only=True)
    payload_key = next(iter(artifact["payloads"]))
    layers = artifact["payloads"][payload_key][LAYERS_KEY]
    if hidden_dimension is not None or slot_count is not None:
        layers = {
            layer: torch.zeros(
                (
                    slot_count if slot_count is not None else tensor.shape[0],
                    hidden_dimension if hidden_dimension is not None else tensor.shape[1],
                ),
                dtype=tensor.dtype,
            )
            for layer, tensor in layers.items()
        }
        artifact["payloads"][payload_key][LAYERS_KEY] = layers
    if missing_layer is not None:
        layers.pop(missing_layer)
    if donor_bank is not None:
        selected = artifact["payloads"].pop(payload_key)
        payload_key = f'{payload_key.rsplit("|", 1)[0]}|{donor_bank}'
        artifact["payloads"][payload_key] = selected
        row["donor"]["payload_key"] = payload_key
    torch.save(artifact, donor_path)
    donor_manifest_path = Path(row["donor"]["manifest_path"])
    donor_manifest = json.loads(donor_manifest_path.read_text(encoding="utf-8"))
    donor_manifest["pairs"][0]["payload_sha256"] = sha256_file(donor_path)
    donor_manifest["pairs"][0]["payload_key"] = payload_key
    donor_manifest["pairs"][0]["slot_shape"] = {
        layer: list(tensor.shape) for layer, tensor in layers.items()
    }
    donor_manifest_path.write_text(json.dumps(donor_manifest), encoding="utf-8")


def test_preflight_rejects_self_consistent_donor_with_incompatible_target_shape_before_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row, block_dir = _preflight_execution_row(tmp_path)
    _rewrite_execution_donor(row, hidden_dimension=4)
    calls = []
    contract = {"snapshot": {"path": str(block_dir / "snapshot.pt")}}
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._validated_prefix_contract",
        lambda _row: contract,
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._freeze_or_load_command_artifact",
        lambda _row, _contract: {"preflight": {"full_correct": ["gpu"]}},
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness.validate_contract", lambda *_args: []
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._run_logged",
        lambda *_args: calls.append("gpu"),
    )

    with pytest.raises(ValueError, match="donor.*target.*shape"):
        _execute_stage({"blocks": [row]}, "preflight")

    assert calls == []


def test_preflight_rejects_self_consistent_legacy_32_slot_donor_before_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row, block_dir = _preflight_execution_row(tmp_path)
    _rewrite_execution_donor(row, slot_count=32)
    calls = []
    contract = {"snapshot": {"path": str(block_dir / "snapshot.pt")}}
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._validated_prefix_contract",
        lambda _row: contract,
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._freeze_or_load_command_artifact",
        lambda _row, _contract: {"preflight": {"full_correct": ["gpu"]}},
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness.validate_contract", lambda *_args: []
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._run_logged",
        lambda *_args: calls.append("gpu"),
    )

    with pytest.raises(ValueError, match="donor.*target.*shape"):
        _execute_stage({"blocks": [row]}, "preflight")

    assert calls == []


def test_preflight_accepts_compatible_donor_before_reaching_gpu_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row, block_dir = _preflight_execution_row(tmp_path)
    calls = []
    contract = {"snapshot": {"path": str(block_dir / "snapshot.pt")}}
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._validated_prefix_contract",
        lambda _row: contract,
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._freeze_or_load_command_artifact",
        lambda _row, _contract: {"preflight": {"full_correct": ["gpu"]}},
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness.validate_contract", lambda *_args: []
    )

    def gpu_boundary(*_args) -> None:
        calls.append("gpu")
        raise RuntimeError("gpu boundary reached")

    monkeypatch.setattr(
        "utest.subject_reappearance_harness._run_logged", gpu_boundary
    )

    with pytest.raises(RuntimeError, match="gpu boundary reached"):
        _execute_stage({"blocks": [row]}, "preflight")

    assert calls == ["gpu"]


def test_preflight_rejects_self_consistent_donor_with_missing_layer_before_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row, block_dir = _preflight_execution_row(tmp_path)
    _rewrite_execution_donor(row, missing_layer="15")
    calls = []
    contract = {"snapshot": {"path": str(block_dir / "snapshot.pt")}}
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._validated_prefix_contract",
        lambda _row: contract,
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._freeze_or_load_command_artifact",
        lambda _row, _contract: {"preflight": {"full_correct": ["gpu"]}},
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness.validate_contract", lambda *_args: []
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._run_logged",
        lambda *_args: calls.append("gpu"),
    )

    with pytest.raises(ValueError, match="donor-target layer sets"):
        _execute_stage({"blocks": [row]}, "preflight")

    assert calls == []


def test_preflight_rejects_self_consistent_donor_with_missing_target_bank_before_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row, block_dir = _preflight_execution_row(tmp_path)
    _rewrite_execution_donor(row, donor_bank=1)
    calls = []
    contract = {"snapshot": {"path": str(block_dir / "snapshot.pt")}}
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._validated_prefix_contract",
        lambda _row: contract,
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._freeze_or_load_command_artifact",
        lambda _row, _contract: {"preflight": {"full_correct": ["gpu"]}},
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness.validate_contract", lambda *_args: []
    )
    monkeypatch.setattr(
        "utest.subject_reappearance_harness._run_logged",
        lambda *_args: calls.append("gpu"),
    )

    with pytest.raises(ValueError, match="donor-target bank sets"):
        _execute_stage({"blocks": [row]}, "preflight")

    assert calls == []


@pytest.mark.parametrize("descendant", ["story_path", "reference_path"])
def test_run_revalidates_prepared_descendants_before_gpu(
    descendant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _prepared_inputs(tmp_path)
    base = tmp_path / "args.json"
    base.write_text(json.dumps({"argv": VALID_BASE_ARGV}), encoding="utf-8")
    platform = tmp_path / "platform.json"
    platform.write_text("{}", encoding="utf-8")
    output = tmp_path / "run"
    assert main([
        "dry-run", "--inputs", str(inputs), "--output", str(output),
        "--base-inference-args", str(base), "--platform-manifest", str(platform),
    ]) == 0
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    Path(manifest["blocks"][0]["prepared_provenance"][descendant]).write_bytes(b"tampered")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("GPU launched")))

    with pytest.raises(ValueError, match="provenance|SHA-256"):
        main(["prefix", "--manifest", str(output / "run_manifest.json")])


def test_valid_teacher_freezes_a_qstar_stage_command_and_requires_donor(
    tmp_path: Path,
) -> None:
    inputs = _prepared_inputs(tmp_path)
    selection = json.loads(inputs.read_text(encoding="utf-8"))
    event_id = selection["events"][0]["event_id"]
    event = json.loads((inputs.parent / event_id / "event.json").read_text(encoding="utf-8"))
    donor_layers = {str(layer): torch.zeros(64, 3) for layer in range(16)}
    donor_path = tmp_path / "donor.pt"
    torch.save({
        "format": "slotmem_donor_payload_v2",
        "event": {"story_id": "donor", "entity_uid": "donor::Other", "character_name": "Other"},
        "payloads": {"Other|0": {"__layerwise__": True, LAYERS_KEY: donor_layers}},
    }, donor_path)
    donor_manifest = tmp_path / "donor.json"
    donor_manifest.write_text(json.dumps({"pairs": [{
        "target_story_id": event["story_id"], "target_entity_uid": event["entity_uid"],
        "donor_story_id": "donor", "donor_entity_uid": "donor::Other",
        "payload_path": str(donor_path.resolve()), "payload_sha256": sha256_file(donor_path),
        "payload_key": "Other|0", "coarse_class": "person", "colour": "unknown",
        "character_count": 1, "source_visible": True, "gap_bucket": "long",
        "slot_shape": {layer: [64, 3] for layer in donor_layers}, "selection_seed": 0,
    }]}), encoding="utf-8")
    donor_map = tmp_path / "donor_map.json"
    donor_map.write_text(json.dumps({"events": {event_id: {
        "payload": str(donor_path), "manifest": str(donor_manifest),
    }}}), encoding="utf-8")
    teacher_video = tmp_path / "teacher.mp4"
    teacher_video.write_bytes(b"independent teacher")
    teacher_manifest = tmp_path / "teacher.json"
    teacher_manifest.write_text(json.dumps({
        "story_id": event["story_id"], "target_chunk_idx": event["target_chunk_idx"],
        "video_path": str(teacher_video.resolve()), "video_sha256": sha256_file(teacher_video),
        "source_type": "independent_teacher", "generated_by_arm": False,
        "generated_by_evaluated_model": False,
    }), encoding="utf-8")
    teacher_map = tmp_path / "teacher_map.json"
    teacher_map.write_text(json.dumps({"events": {event_id: {
        "video": str(teacher_video), "manifest": str(teacher_manifest),
    }}}), encoding="utf-8")
    base = tmp_path / "args.json"
    base.write_text(json.dumps({"argv": VALID_BASE_ARGV}), encoding="utf-8")
    platform = tmp_path / "platform.json"
    platform.write_text("{}", encoding="utf-8")
    output = tmp_path / "run"
    assert main([
        "dry-run", "--inputs", str(inputs), "--output", str(output),
        "--base-inference-args", str(base), "--platform-manifest", str(platform),
        "--donor-map", str(donor_map), "--teacher-map", str(teacher_map),
    ]) == 0
    run = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    row = next(block for block in run["blocks"] if block["event_id"] == event_id)
    assert row["qstar"]["status"] == "available"
    assert row["commands"]["qstar"]["status"] == "deferred_until_prefix"
    snapshot = Path(row["prefix_snapshot"])
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(b"snapshot")
    frozen_event = json.loads(Path(row["event_json"]).read_text(encoding="utf-8"))
    contract = {
        "event": frozen_event,
        "snapshot": {"path": str(snapshot), "sha256": sha256_file(snapshot)},
        "base_inference_args": row["commands"]["prefix_inference_args"],
        "qstar": {"timestep_indices": [0, 12, 25, 37, 49]},
    }
    artifact = _command_artifact_payload(row, contract)
    assert artifact["qstar"][0:3] == [sys.executable, "-m", "utest.qstar_probe"]
    assert artifact["qstar"][artifact["qstar"].index("--noise-seed") + 1] == "0"


@pytest.mark.parametrize(
    "fault",
    [
        None, "loss_nan", "prediction_sha", "noise_sha", "cell_qstar", "classification",
        "payload_deleted", "empty_path_semantics", "repeat_payload",
    ],
)
def test_qstar_resume_requires_a_complete_frozen_teacher_report(
    fault: str | None, tmp_path: Path
) -> None:
    root = tmp_path / "block"
    qstar = root / "qstar"
    qstar.mkdir(parents=True)
    snapshot_sha, teacher_sha = "a" * 64, "b" * 64
    row = {
        "block_dir": str(root),
        "event_id": "event",
        "qstar": {"video_sha256": teacher_sha},
    }
    contract = {
        "snapshot": {"sha256": snapshot_sha},
        "event": {"character_name": "Ana"},
        "runtime_contract": {"target_prompt_sha256": "c" * 64},
        "qstar": {"timestep_indices": [0], "horizon": 6},
    }
    input_hashes = {
        "prefix": snapshot_sha,
        "target_video": teacher_sha,
        "target_latent": "d" * 64,
        "noise": "e" * 64,
        "noisy_latent": "f" * 64,
        "flow_target": "1" * 64,
        "prompt": "c" * 64,
    }
    report = {
        "schema_version": 1,
        "status": "passed",
        "target_video_sha256": teacher_sha,
        "prefix_sha256": snapshot_sha,
        "model_weights_changed": False,
        "native_is_diagnostic": True,
        "memory_regime": "static_prefix",
        "thresholds": {
            "repeat_loss_tolerance": 0.0, "repeat_influence_tolerance": 0.0,
            "benefit_margin": 0.0, "influence_floor": 0.0,
        },
        "cells": [{
            "event_id": "event", "memory_id": "Ana|0", "horizon": 6,
            "timestep_index": 0, "timestep": 999.0, "input_hashes": input_hashes,
            "losses": {arm: 1.0 for arm in ("correct", "correct_repeat", "no_memory", "zero", "random", "wrong", "native")},
            "qstar": 0.0, "arm_deltas": {"zero": 0.0, "random": 0.0, "wrong": 0.0},
            "repeat_loss_floor": 0.0, "repeat_prediction_floor": 0.0,
            "primary_influence": 1.0, "repeat_margin": 0.0,
            "benefit_margin_degenerate": True,
            "classification": "influence_without_benefit",
        }],
    }
    records = [
        {
            "event_id": "event", "memory_id": "Ana|0", "horizon": 6,
            "timestep_index": 0, "timestep": 999.0, "arm": arm,
            "role": "diagnostic" if arm == "native" else "confirmatory",
            "loss": 1.0, "masked_loss": None, "prediction_sha256": "2" * 64,
            "input_hashes": input_hashes, "memory_read_hit": arm not in {"no_memory", "native"},
            "injection_delta_norm": 0.1 if arm not in {"no_memory", "native"} else 0.0,
            "payload_sha256": (
                None if arm in {"no_memory", "native"}
                else "4" * 64 if arm in {"correct", "correct_repeat"}
                else {"zero": "5", "random": "6", "wrong": "7"}[arm] * 64
            ),
            "payload_layers": 16 if arm not in {"no_memory", "native"} else 0,
            "payload_slots": 8 if arm not in {"no_memory", "native"} else 0,
            "cfg_prediction_sha256": None if arm == "native" else "3" * 64,
            "forced_memory_path": arm != "native", "diagnostics": {},
        }
        for arm in ("correct", "correct_repeat", "no_memory", "zero", "random", "wrong", "native")
    ]
    if fault == "loss_nan":
        records[0]["loss"] = float("nan")
    elif fault == "prediction_sha":
        records[0]["prediction_sha256"] = None
    elif fault == "noise_sha":
        records[0]["input_hashes"] = {**input_hashes, "noise": "not-a-sha"}
    elif fault == "cell_qstar":
        report["cells"][0]["qstar"] = float("inf")
    elif fault == "classification":
        report["cells"][0]["classification"] = "beneficial"
    elif fault == "payload_deleted":
        del records[0]["payload_sha256"]
    elif fault == "empty_path_semantics":
        records[0].update({
            "memory_read_hit": False, "forced_memory_path": False,
            "injection_delta_norm": 0.0, "payload_sha256": None,
            "payload_layers": 0, "payload_slots": 0,
        })
    elif fault == "repeat_payload":
        records[1]["payload_sha256"] = "8" * 64
    (qstar / "qstar_report.json").write_text(json.dumps(report), encoding="utf-8")
    records_path = qstar / "qstar_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    if fault is None:
        assert _validate_qstar_report(row, contract)["status"] == "passed"
    else:
        with pytest.raises(ValueError, match=r"Q\* report"):
            _validate_qstar_report(row, contract)
