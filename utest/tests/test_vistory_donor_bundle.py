from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import torch

import utest.prefix_contract as prefix_contract
import utest.vistory_donor_harness as donor_harness
from utest.input_contract import validate_donor_bundle
from utest.prefix_contract import sha256_file
from utest.tests.test_subject_reappearance_harness import _prepared_inputs
from utest.tests.test_vistory_donor_harness import (
    _inputs,
    _materialize_completion,
    _materialize_execution,
    _materialize_prefix,
    _exploratory_selection,
    _selection,
    _write_json,
)
from utest.subject_reappearance_harness import build_run_manifest as build_subject_run_manifest
from utest.subject_subspace_audit import validate_frozen_donor_artifact
from utest.vistory_donor_bundle import (
    build_validated_event_donor_map,
    freeze_vistory_donor_map,
    validate_target_inputs,
)
from utest.vistory_donor_harness import build_donor_run_manifest


SONG_EVENT_ID = "vistory79_song_yuchen_s2_s8"


@pytest.fixture(autouse=True)
def _clean_repository_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    state = lambda _repo: ("frozen-bundle-test-commit", False)
    monkeypatch.setattr(donor_harness, "_git_state", state)
    monkeypatch.setattr(prefix_contract, "_git_state", state)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_events(inputs: Path) -> dict[str, dict]:
    top = _read(inputs)
    events = {}
    for row in top["events"]:
        manifest = _read(inputs.parent / row["manifest_path"])
        event = _read(inputs.parent / manifest["outputs"]["event"]["path"])
        events[row["event_id"]] = event
    return events


def _completed_fixture(
    tmp_path: Path,
    *,
    selected_target_ids: set[str] | None = None,
    memory_bank_mode: str | None = None,
    loose_candidate_story_type: bool = False,
    loose_donor_story_type: bool = False,
    encoder_layers: str | None = "0-15",
    encoder_slots: str | None = "64",
) -> tuple[Path, Path, Path, dict[str, dict]]:
    targets = _prepared_inputs(tmp_path / "targets")
    target_events = _target_events(targets)
    if selected_target_ids is None:
        selection_path = _selection(tmp_path / "donors")
    else:
        assert len(selected_target_ids) == 1
        selection_path = _exploratory_selection(
            tmp_path / "donors", next(iter(selected_target_ids))
        )
    selection = _read(selection_path)
    selection["target_inputs_sha256"] = sha256_file(targets)
    for row in selection["events"]:
        manifest_path = selection_path.parent / row["manifest_path"]
        manifest = _read(manifest_path)
        event_path = selection_path.parent / manifest["outputs"]["event"]["path"]
        event = _read(event_path)
        if loose_donor_story_type:
            row["donor_story_id"] = int(str(row["donor_story_id"]).rsplit("-", 1)[1])
            event["story_id"] = row["donor_story_id"]
        event["entity_uid"] = row["donor_entity_uid"]
        _write_json(event_path, event)
        target = target_events[row["target_event_id"]]
        manifest["candidate"] = {
            "target_event_id": row["target_event_id"],
            "candidate_id": row["candidate_id"],
            "target_story_id": (
                int(target["story_id"]) if loose_candidate_story_type else target["story_id"]
            ),
            "target_entity_uid": target["entity_uid"],
            "donor_story_id": row["donor_story_id"],
            "donor_entity_uid": row["donor_entity_uid"],
            "donor_char_id": event["character_name"],
            "source_character_count": 1,
            "gap_bucket": "5-7",
        }
        manifest["review"] = {
            "target_event_id": row["target_event_id"],
            "candidate_id": row["candidate_id"],
            "target_presentation_class": "male",
            "donor_presentation_class": "male",
            "target_dominant_colour": "black",
            "donor_dominant_colour": "black",
            "donor_source_visible": True,
            "donor_read_check_visible": True,
            "approved": True,
            "tie_group": None,
            "reviewer": "fixture",
        }
        manifest["outputs"]["event"]["sha256"] = sha256_file(event_path)
        _write_json(manifest_path, manifest)
        row["manifest_sha256"] = sha256_file(manifest_path)
    _write_json(selection_path, selection)

    base, platform = _inputs(tmp_path / "donors")
    base_value = _read(base)
    for option in (
        "--slotmem_memory_encoder_layers",
        "--slotmem_memory_encoder_slots",
    ):
        base_value["argv"] = donor_harness._set_option(
            base_value["argv"], option, None
        )
    if encoder_layers is not None:
        base_value["argv"].extend(["--slotmem_memory_encoder_layers", encoder_layers])
    if encoder_slots is not None:
        base_value["argv"].extend(["--slotmem_memory_encoder_slots", encoder_slots])
    if memory_bank_mode is not None:
        base_value["argv"].extend(["--slotmem_memory_bank_mode", memory_bank_mode])
    _write_json(base, base_value)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "donor_run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=donor_harness.sys.executable,
    )
    for job in run["jobs"]:
        _materialize_prefix(job, platform)
        payload = Path(job["donor_payload"])
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload_key = f'{job["event"]["character_name"]}|0'
        layers = {
            str(layer): torch.zeros((64, 3), dtype=torch.float16)
            for layer in range(16)
        }
        torch.save(
            {
                "format": "slotmem_donor_payload_v2",
                "event": job["event"],
                "payloads": {
                    payload_key: {"__layerwise__": True, "layers": layers}
                },
            },
            payload,
        )
        _write_json(
            Path(job["donor_payload_info"]),
            {
                "format": "slotmem_donor_payload_v2",
                "payload_path": str(payload.resolve()),
                "payload_sha256": sha256_file(payload),
                "payload_keys": [payload_key],
                "payload_slot_shapes": {
                    payload_key: {str(layer): [64, 3] for layer in range(16)}
                },
                "event": job["event"],
            },
        )
        _write_json(
            Path(job["donor_audit"]),
            {
                "arm": "correct",
                "seed": 0,
                "target_character": job["event"]["character_name"],
                "target_chunk_idx": job["event"]["target_chunk_idx"],
                "target_read_hits": 1,
                "intervention_effective": True,
                "donor_dumped": str(payload.resolve()),
                "donor_sha256": sha256_file(payload),
                "runtime_contract": job["dump_runtime_contract"],
            },
        )
        _materialize_execution(job, "dump")
        _materialize_completion(job, run)
    run_manifest = Path(run["output_root"]) / "run_manifest.json"
    _write_json(run_manifest, run)
    return targets, selection_path, run_manifest, target_events


def test_exploratory_bundle_keeps_complete_targets_but_publishes_only_song(
    tmp_path: Path,
) -> None:
    targets, selection, run, target_events = _completed_fixture(
        tmp_path,
        selected_target_ids={SONG_EVENT_ID},
    )

    validated_targets = validate_target_inputs(targets)
    assert {row["event_id"] for row in validated_targets["events"]} == set(
        target_events
    )
    result = freeze_vistory_donor_map(
        target_inputs_path=targets,
        selection_path=selection,
        donor_run_manifest_path=run,
        output_root=tmp_path / "bundle",
    )

    assert result["protocol_scope"] == "exploratory_single_event"
    assert result["target_event_ids"] == [SONG_EVENT_ID]
    assert set(result["events"]) == {SONG_EVENT_ID}
    pair = _read(Path(result["events"][SONG_EVENT_ID]["manifest"]))
    assert pair["provenance"]["selection"] == {
        "path": str(selection.resolve()),
        "sha256": sha256_file(selection),
    }
    assert pair["provenance"]["donor_run_manifest"] == {
        "path": str(run.resolve()),
        "sha256": sha256_file(run),
    }


def test_bundle_rejects_run_scope_that_differs_from_validated_selection(
    tmp_path: Path,
) -> None:
    target_path, selection_path, run_path, _ = _completed_fixture(
        tmp_path,
        selected_target_ids={SONG_EVENT_ID},
    )
    targets = validate_target_inputs(target_path)
    selection = donor_harness.validate_frozen_selection(selection_path)
    run = donor_harness.validate_completed_donor_run(run_path, selection)
    run.update(
        donor_run_manifest_path=str(run_path.resolve()),
        donor_run_manifest_sha256=sha256_file(run_path),
        target_event_ids=["vistory42_bella_s15_s21"],
    )

    with pytest.raises(ValueError, match="run scope"):
        build_validated_event_donor_map(
            targets,
            selection,
            run,
            tmp_path / "bundle",
        )


def test_exploratory_bundle_rejects_selection_event_outside_declared_scope(
    tmp_path: Path,
) -> None:
    targets, selection, run, _ = _completed_fixture(
        tmp_path,
        selected_target_ids={SONG_EVENT_ID},
    )
    value = _read(selection)
    value["events"][0]["target_event_id"] = "vistory42_bella_s15_s21"
    _write_json(selection, value)
    output = tmp_path / "bundle"

    with pytest.raises(ValueError, match="selection target event IDs"):
        freeze_vistory_donor_map(
            target_inputs_path=targets,
            selection_path=selection,
            donor_run_manifest_path=run,
            output_root=output,
        )

    assert not output.exists()


@pytest.mark.parametrize("mutation", ["missing", "extra", "cross_wired"])
def test_exploratory_bundle_rejects_jobs_that_do_not_exactly_match_scope(
    tmp_path: Path,
    mutation: str,
) -> None:
    targets, selection, run_path, _ = _completed_fixture(
        tmp_path,
        selected_target_ids={SONG_EVENT_ID},
    )
    run = _read(run_path)
    if mutation == "missing":
        run["jobs"].clear()
    elif mutation == "extra":
        run["jobs"].append(dict(run["jobs"][0]))
    else:
        run["jobs"][0]["target_event_id"] = "vistory42_bella_s15_s21"
    _write_json(run_path, run)
    output = tmp_path / "bundle"

    with pytest.raises(ValueError):
        freeze_vistory_donor_map(
            target_inputs_path=targets,
            selection_path=selection,
            donor_run_manifest_path=run_path,
            output_root=output,
        )

    assert not output.exists()


def test_target_inputs_remain_strictly_the_complete_frozen_three(tmp_path: Path) -> None:
    targets = _prepared_inputs(tmp_path / "targets")
    value = _read(targets)
    value["events"].pop()
    _write_json(targets, value)

    with pytest.raises(ValueError, match="exactly three"):
        validate_target_inputs(targets)


def test_freeze_emits_exactly_three_valid_event_level_donor_pairs(tmp_path: Path) -> None:
    targets, selection, donor_run, target_events = _completed_fixture(tmp_path)

    result = freeze_vistory_donor_map(
        target_inputs_path=targets,
        selection_path=selection,
        donor_run_manifest_path=donor_run,
        output_root=tmp_path / "bundle",
    )

    assert set(result["events"]) == set(target_events)
    assert len(result["events"]) == 3
    assert "protocol_scope" not in result
    assert "target_event_ids" not in result
    assert not any("seed" in key for key in result["events"])
    for event_id, entry in result["events"].items():
        assert set(entry) == {"payload", "manifest"}
        report = validate_donor_bundle(
            target_events[event_id],
            Path(entry["payload"]),
            Path(entry["manifest"]),
            loader=lambda path: torch.load(path, map_location="cpu", weights_only=True),
        )
        assert report["status"] == "passed"
        pair = _read(Path(entry["manifest"]))
        frozen = pair["pairs"][0]
        artifact = torch.load(entry["payload"], map_location="cpu", weights_only=True)
        mask_banks = {
            0: {str(layer): {"slot_count": 64} for layer in range(16)}
        }
        for _seed in (0, 1, 2):
            validate_frozen_donor_artifact(artifact, frozen, banks=mask_banks)
        assert frozen["donor_seed"] == 0
        assert frozen["slot_count"] == {str(layer): 64 for layer in range(16)}
        assert frozen["slot_shape"] == {
            str(layer): [64, 3] for layer in range(16)
        }
        assert frozen["payload_dtype"] == {
            str(layer): "float16" for layer in range(16)
        }
        assert pair["provenance"]["source_event"]["sha256"] == sha256_file(
            Path(pair["provenance"]["source_event"]["path"])
        )
        assert pair["provenance"]["repository"] == _read(donor_run)["repository"]
        assert pair["provenance"]["target_inputs"] == {
            "path": str(targets.resolve()),
            "sha256": sha256_file(targets),
        }
        assert pair["provenance"]["selection"] == {
            "path": str(selection.resolve()),
            "sha256": sha256_file(selection),
        }
        assert pair["provenance"]["donor_run_manifest"] == {
            "path": str(donor_run.resolve()),
            "sha256": sha256_file(donor_run),
        }
        assert pair["provenance"]["platform_manifest"]["sha256"] == _read(donor_run)[
            "platform_manifest_sha256"
        ]
    assert _read(tmp_path / "bundle" / "donor_map.json") == result


def test_bundle_is_event_level_and_ready_for_all_nine_target_blocks(tmp_path: Path) -> None:
    targets, selection, donor_run, _ = _completed_fixture(tmp_path)
    bundle_root = tmp_path / "bundle"
    freeze_vistory_donor_map(
        target_inputs_path=targets,
        selection_path=selection,
        donor_run_manifest_path=donor_run,
        output_root=bundle_root,
    )
    donor_run_value = _read(donor_run)

    subject = build_subject_run_manifest(
        inputs=targets,
        output=tmp_path / "subject_run",
        base_inference_args=Path(donor_run_value["base_inference_args"]),
        platform_manifest=Path(donor_run_value["platform_manifest"]),
        python=sys.executable,
        donor_map=bundle_root / "donor_map.json",
    )

    assert len(subject["blocks"]) == 9
    assert all(
        block["commands"]["preflight"]["status"] == "deferred_until_prefix"
        and block["commands"]["full"]["status"] == "deferred_until_prefix"
        for block in subject["blocks"]
    )
    for event_id in sorted({block["event_id"] for block in subject["blocks"]}):
        rows = [block for block in subject["blocks"] if block["event_id"] == event_id]
        assert {block["seed"] for block in rows} == {0, 1, 2}
        assert len({block["donor"]["payload_sha256"] for block in rows}) == 1


def test_existing_bundle_is_never_overwritten(tmp_path: Path) -> None:
    targets, selection, donor_run, _ = _completed_fixture(tmp_path)
    output = tmp_path / "bundle"
    first = freeze_vistory_donor_map(
        target_inputs_path=targets,
        selection_path=selection,
        donor_run_manifest_path=donor_run,
        output_root=output,
    )
    frozen_bytes = (output / "donor_map.json").read_bytes()

    with pytest.raises(FileExistsError):
        freeze_vistory_donor_map(
            target_inputs_path=targets,
            selection_path=selection,
            donor_run_manifest_path=donor_run,
            output_root=output,
        )

    assert (output / "donor_map.json").read_bytes() == frozen_bytes
    assert _read(output / "donor_map.json") == first


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_missing_or_extra_donor_job_is_rejected_atomically(
    tmp_path: Path, mutation: str
) -> None:
    targets, selection, donor_run, _ = _completed_fixture(tmp_path)
    run = _read(donor_run)
    if mutation == "missing":
        run["jobs"].pop()
    else:
        run["jobs"].append(dict(run["jobs"][0]))
    _write_json(donor_run, run)
    output = tmp_path / "bundle"

    with pytest.raises(ValueError):
        freeze_vistory_donor_map(
            target_inputs_path=targets,
            selection_path=selection,
            donor_run_manifest_path=donor_run,
            output_root=output,
        )

    assert not output.exists()


def _refresh_payload_bindings(run: dict) -> None:
    for job in run["jobs"]:
        payload = Path(job["donor_payload"])
        info_path = Path(job["donor_payload_info"])
        info = _read(info_path)
        info["payload_sha256"] = sha256_file(payload)
        _write_json(info_path, info)
        audit_path = Path(job["donor_audit"])
        audit = _read(audit_path)
        audit["donor_sha256"] = sha256_file(payload)
        _write_json(audit_path, audit)
        completion_path = Path(job["completion"])
        completion = _read(completion_path)
        completion["donor_payload"]["sha256"] = sha256_file(payload)
        completion["donor_payload_info"]["sha256"] = sha256_file(info_path)
        completion["donor_audit"]["sha256"] = sha256_file(audit_path)
        _write_json(completion_path, completion)


def test_flat_tensor_payload_is_rejected_before_freeze(tmp_path: Path) -> None:
    targets, selection, donor_run, _ = _completed_fixture(tmp_path)
    run = _read(donor_run)
    job = run["jobs"][0]
    payload_path = Path(job["donor_payload"])
    artifact = torch.load(payload_path, map_location="cpu", weights_only=True)
    payload_key = next(iter(artifact["payloads"]))
    artifact["payloads"][payload_key] = torch.zeros((64, 3), dtype=torch.float16)
    torch.save(artifact, payload_path)
    info_path = Path(job["donor_payload_info"])
    info = _read(info_path)
    info["payload_slot_shapes"][payload_key] = {"0": [64, 3]}
    _write_json(info_path, info)
    _refresh_payload_bindings(run)

    with pytest.raises(ValueError, match="layerwise"):
        freeze_vistory_donor_map(
            target_inputs_path=targets,
            selection_path=selection,
            donor_run_manifest_path=donor_run,
            output_root=tmp_path / "bundle",
        )


def test_empty_layerwise_payload_is_rejected_before_freeze(tmp_path: Path) -> None:
    targets, selection, donor_run, _ = _completed_fixture(tmp_path)
    run = _read(donor_run)
    job = run["jobs"][0]
    payload_path = Path(job["donor_payload"])
    artifact = torch.load(payload_path, map_location="cpu", weights_only=True)
    payload_key = next(iter(artifact["payloads"]))
    artifact["payloads"][payload_key] = {"__layerwise__": True, "layers": {}}
    torch.save(artifact, payload_path)
    info_path = Path(job["donor_payload_info"])
    info = _read(info_path)
    info["payload_slot_shapes"][payload_key] = {}
    _write_json(info_path, info)
    _refresh_payload_bindings(run)

    with pytest.raises(ValueError, match="non-empty layerwise"):
        freeze_vistory_donor_map(
            target_inputs_path=targets,
            selection_path=selection,
            donor_run_manifest_path=donor_run,
            output_root=tmp_path / "bundle",
        )


@pytest.mark.parametrize(
    "malformation",
    [
        "integer",
        "missing_tensor",
        "marker_int",
        "integer_layer_key",
        "missing_layer",
        "extra_layer",
        "wrong_slots",
        "hidden_dim",
    ],
)
def test_layerwise_payload_requires_complete_floating_tensor_layers(
    tmp_path: Path, malformation: str
) -> None:
    targets, selection, donor_run, _ = _completed_fixture(tmp_path)
    run = _read(donor_run)
    job = run["jobs"][0]
    payload_path = Path(job["donor_payload"])
    artifact = torch.load(payload_path, map_location="cpu", weights_only=True)
    payload_key = next(iter(artifact["payloads"]))
    layers = artifact["payloads"][payload_key]["layers"]
    if malformation == "integer":
        layers["0"] = torch.zeros((64, 3), dtype=torch.int64)
    elif malformation == "missing_tensor":
        layers["missing"] = "not-a-tensor"
    elif malformation == "marker_int":
        artifact["payloads"][payload_key]["__layerwise__"] = 1
    elif malformation == "integer_layer_key":
        layers[0] = layers.pop("0")
    elif malformation == "missing_layer":
        layers.pop("15")
    elif malformation == "extra_layer":
        layers["16"] = torch.zeros((64, 3), dtype=torch.float16)
    elif malformation == "wrong_slots":
        layers["0"] = torch.zeros((63, 3), dtype=torch.float16)
    else:
        layers["15"] = torch.zeros((64, 4), dtype=torch.float16)
    torch.save(artifact, payload_path)
    if malformation in {"missing_layer", "extra_layer", "wrong_slots", "hidden_dim"}:
        info_path = Path(job["donor_payload_info"])
        info = _read(info_path)
        info["payload_slot_shapes"][payload_key] = {
            str(layer): list(tensor.shape)
            for layer, tensor in layers.items()
            if isinstance(tensor, torch.Tensor)
        }
        _write_json(info_path, info)
    _refresh_payload_bindings(run)

    with pytest.raises(ValueError, match="layerwise|layers 0-15|64-slot|hidden dimension"):
        freeze_vistory_donor_map(
            target_inputs_path=targets,
            selection_path=selection,
            donor_run_manifest_path=donor_run,
            output_root=tmp_path / "bundle",
        )


def test_multiple_payload_keys_are_rejected_as_a_single_bank_contract_violation(tmp_path: Path) -> None:
    targets, selection, donor_run, _ = _completed_fixture(tmp_path)
    run = _read(donor_run)
    job = run["jobs"][0]
    payload_path = Path(job["donor_payload"])
    artifact = torch.load(payload_path, map_location="cpu", weights_only=True)
    bank_zero = next(iter(artifact["payloads"]))
    bank_one = bank_zero.rsplit("|", 1)[0] + "|1"
    artifact["payloads"][bank_one] = {
        "__layerwise__": True,
        "layers": {"0": torch.ones((2, 3), dtype=torch.float16)},
    }
    torch.save(artifact, payload_path)
    info_path = Path(job["donor_payload_info"])
    info = _read(info_path)
    info["payload_keys"] = sorted([bank_zero, bank_one])
    info["payload_slot_shapes"][bank_one] = {"0": [2, 3]}
    _write_json(info_path, info)
    _refresh_payload_bindings(run)

    with pytest.raises(ValueError, match="single-bank"):
        freeze_vistory_donor_map(
            target_inputs_path=targets,
            selection_path=selection,
            donor_run_manifest_path=donor_run,
            output_root=tmp_path / "bundle",
        )


def test_legacy_multi_bank_dump_runtime_is_rejected(tmp_path: Path) -> None:
    targets, selection, donor_run, _ = _completed_fixture(
        tmp_path, memory_bank_mode="legacy_multi"
    )

    with pytest.raises(ValueError, match="single-bank mode"):
        freeze_vistory_donor_map(
            target_inputs_path=targets,
            selection_path=selection,
            donor_run_manifest_path=donor_run,
            output_root=tmp_path / "bundle",
        )


def test_encoder_slot_count_is_frozen_to_64(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="actual=None.*frozen expected='64'"):
        _completed_fixture(tmp_path, encoder_slots=None)


@pytest.mark.parametrize(
    ("encoder_layers", "encoder_slots"),
    [("0-14", "64"), ("0-16", "64"), ("0-4,6-15", "64"), ("0-15", "32")],
)
def test_frozen_encoder_geometry_must_match_the_formal_mask_universe(
    tmp_path: Path, encoder_layers: str, encoder_slots: str
) -> None:
    with pytest.raises(ValueError, match="SlotMem donor protocol mismatch"):
        _completed_fixture(
            tmp_path,
            encoder_layers=encoder_layers,
            encoder_slots=encoder_slots,
        )


def test_encoder_layer_range_must_be_explicitly_frozen_to_0_through_15(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="actual=None.*frozen expected='0-15'"):
        _completed_fixture(tmp_path, encoder_layers=None, encoder_slots="64")


def test_all_three_payloads_must_share_one_platform_hidden_dimension(tmp_path: Path) -> None:
    targets, selection, donor_run, _ = _completed_fixture(tmp_path)
    run = _read(donor_run)
    job = run["jobs"][0]
    payload_path = Path(job["donor_payload"])
    artifact = torch.load(payload_path, map_location="cpu", weights_only=True)
    payload_key = next(iter(artifact["payloads"]))
    artifact["payloads"][payload_key]["layers"] = {
        str(layer): torch.zeros((64, 4), dtype=torch.float16)
        for layer in range(16)
    }
    torch.save(artifact, payload_path)
    info_path = Path(job["donor_payload_info"])
    info = _read(info_path)
    info["payload_slot_shapes"][payload_key] = {
        str(layer): [64, 4] for layer in range(16)
    }
    _write_json(info_path, info)
    _refresh_payload_bindings(run)

    with pytest.raises(ValueError, match="platform hidden dimension"):
        freeze_vistory_donor_map(
            target_inputs_path=targets,
            selection_path=selection,
            donor_run_manifest_path=donor_run,
            output_root=tmp_path / "bundle",
        )


def test_cross_wired_payload_is_rejected_even_when_hash_claims_are_refreshed(tmp_path: Path) -> None:
    targets, selection, donor_run, _ = _completed_fixture(tmp_path)
    run = _read(donor_run)
    left, right = (Path(job["donor_payload"]) for job in run["jobs"][:2])
    left_bytes, right_bytes = left.read_bytes(), right.read_bytes()
    left.write_bytes(right_bytes)
    right.write_bytes(left_bytes)
    _refresh_payload_bindings(run)
    output = tmp_path / "bundle"

    with pytest.raises(ValueError, match="frozen v2 event payload"):
        freeze_vistory_donor_map(
            target_inputs_path=targets,
            selection_path=selection,
            donor_run_manifest_path=donor_run,
            output_root=output,
        )

    assert not output.exists()


def test_payload_tamper_is_rejected_without_partial_bundle(tmp_path: Path) -> None:
    targets, selection, donor_run, _ = _completed_fixture(tmp_path)
    payload = Path(_read(donor_run)["jobs"][0]["donor_payload"])
    with payload.open("ab") as handle:
        handle.write(b"tamper")
    output = tmp_path / "bundle"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        freeze_vistory_donor_map(
            target_inputs_path=targets,
            selection_path=selection,
            donor_run_manifest_path=donor_run,
            output_root=output,
        )

    assert not output.exists()


def test_boolean_donor_seed_is_rejected(tmp_path: Path) -> None:
    targets, selection, donor_run, _ = _completed_fixture(tmp_path)
    frozen = _read(selection)
    frozen["donor_seed"] = False
    _write_json(selection, frozen)

    with pytest.raises(ValueError, match="integer 0"):
        freeze_vistory_donor_map(
            target_inputs_path=targets,
            selection_path=selection,
            donor_run_manifest_path=donor_run,
            output_root=tmp_path / "bundle",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", True), ("seeds", [False, True, 2])],
)
def test_target_top_level_json_discriminators_are_type_strict(
    tmp_path: Path, field: str, value: object
) -> None:
    targets = _prepared_inputs(tmp_path / "targets")
    top = _read(targets)
    top[field] = value
    _write_json(targets, top)

    with pytest.raises(ValueError, match="target inputs"):
        validate_target_inputs(targets)


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", True), ("seeds", [False, True, 2])],
)
def test_target_event_manifest_discriminators_are_type_strict(
    tmp_path: Path, field: str, value: object
) -> None:
    targets = _prepared_inputs(tmp_path / "targets")
    top = _read(targets)
    row = top["events"][0]
    manifest_path = targets.parent / row["manifest_path"]
    manifest = _read(manifest_path)
    manifest[field] = value
    _write_json(manifest_path, manifest)
    row["manifest_sha256"] = sha256_file(manifest_path)
    _write_json(targets, top)

    with pytest.raises(ValueError, match="target event manifest"):
        validate_target_inputs(targets)


def test_target_event_identity_fields_are_nonempty_strings(tmp_path: Path) -> None:
    targets = _prepared_inputs(tmp_path / "targets")
    top = _read(targets)
    row = top["events"][0]
    manifest_path = targets.parent / row["manifest_path"]
    manifest = _read(manifest_path)
    event_path = targets.parent / manifest["outputs"]["event"]["path"]
    event = _read(event_path)
    event["story_id"] = int(event["story_id"])
    _write_json(event_path, event)
    manifest["outputs"]["event"]["sha256"] = sha256_file(event_path)
    _write_json(manifest_path, manifest)
    row["manifest_sha256"] = sha256_file(manifest_path)
    _write_json(targets, top)

    with pytest.raises(ValueError, match="non-empty string"):
        validate_target_inputs(targets)


def test_candidate_binding_does_not_coerce_story_identity_types(tmp_path: Path) -> None:
    targets, selection, donor_run, _ = _completed_fixture(
        tmp_path, loose_candidate_story_type=True
    )

    with pytest.raises(ValueError, match="cross-wired"):
        freeze_vistory_donor_map(
            target_inputs_path=targets,
            selection_path=selection,
            donor_run_manifest_path=donor_run,
            output_root=tmp_path / "bundle",
        )


def test_job_binding_does_not_coerce_donor_story_types(tmp_path: Path) -> None:
    targets, selection, donor_run, _ = _completed_fixture(
        tmp_path, loose_donor_story_type=True
    )

    with pytest.raises(ValueError, match="non-empty string"):
        freeze_vistory_donor_map(
            target_inputs_path=targets,
            selection_path=selection,
            donor_run_manifest_path=donor_run,
            output_root=tmp_path / "bundle",
        )


def test_dangling_output_symlink_is_rejected_before_resolution(tmp_path: Path) -> None:
    targets, selection, donor_run, _ = _completed_fixture(tmp_path)
    output = tmp_path / "bundle-link"
    try:
        os.symlink(tmp_path / "missing-bundle", output, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    assert output.is_symlink() and not output.exists()

    with pytest.raises(FileExistsError, match="output"):
        freeze_vistory_donor_map(
            target_inputs_path=targets,
            selection_path=selection,
            donor_run_manifest_path=donor_run,
            output_root=output,
        )


def test_donor_job_ancestor_symlink_escape_is_rejected_before_artifact_reads(
    tmp_path: Path,
) -> None:
    targets, selection, donor_run, _ = _completed_fixture(tmp_path)
    job_root = Path(_read(donor_run)["jobs"][0]["prefix_dir"]).parent
    outside = tmp_path / "outside-job"
    shutil.move(job_root, outside)
    try:
        os.symlink(outside, job_root, target_is_directory=True)
    except OSError as error:
        shutil.move(outside, job_root)
        pytest.skip(f"symlink creation unavailable: {error}")

    with pytest.raises(ValueError, match="symlink|escapes"):
        freeze_vistory_donor_map(
            target_inputs_path=targets,
            selection_path=selection,
            donor_run_manifest_path=donor_run,
            output_root=tmp_path / "bundle",
        )


def test_donor_job_ancestor_link_check_runs_on_every_frozen_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    targets, selection, donor_run, _ = _completed_fixture(tmp_path)
    job_root = Path(_read(donor_run)["jobs"][0]["prefix_dir"]).parent
    real_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path == job_root or real_is_symlink(path),
    )

    with pytest.raises(ValueError, match="symlink ancestor"):
        freeze_vistory_donor_map(
            target_inputs_path=targets,
            selection_path=selection,
            donor_run_manifest_path=donor_run,
            output_root=tmp_path / "bundle",
        )
