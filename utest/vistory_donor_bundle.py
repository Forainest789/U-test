"""Freeze completed ViStoryBench donor jobs for the subject harness."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch

from .content_audit import LAYERS_KEY, _is_layerwise
from .input_contract import payload_slot_shapes, validate_donor_bundle
from .prefix_contract import (
    FROZEN_MEMORY_ENCODER_SLOTS,
    sha256_file,
    validate_slotmem_memory_encoder_geometry,
    write_json_no_clobber,
)
from .subject_reappearance_harness import _prepared_events
from .vistory_donor_harness import (
    _json_equal_strict,
    validate_completed_donor_run,
    validate_frozen_selection,
)
from .vistory_donors import (
    TARGET_EVENT_IDS,
    _publish_directory_no_clobber,
    donor_selection_event_ids,
)


def _read_object(path: Path, label: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return dict(value)


def _strict_zero(value: object, label: str) -> None:
    if type(value) is not int or value != 0:
        raise ValueError(f"{label} must be integer 0")


def _nonempty_string(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _strict_schema_and_seeds(value: Mapping, label: str) -> None:
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError(f"{label} schema_version must be integer 1")
    seeds = value.get("seeds")
    if (
        not isinstance(seeds, list)
        or seeds != [0, 1, 2]
        or any(type(seed) is not int for seed in seeds)
    ):
        raise ValueError(f"{label} seeds must be the integer list [0, 1, 2]")


def _dump_geometry(job: Mapping) -> tuple[tuple[int, ...], int]:
    runtime = job.get("dump_runtime_contract")
    frozen_args = runtime.get("frozen_args") if isinstance(runtime, Mapping) else None
    if not isinstance(frozen_args, Mapping):
        raise ValueError("donor dump runtime is missing frozen args")
    return validate_slotmem_memory_encoder_geometry(frozen_args)


def validate_target_inputs(target_inputs_path: Path) -> dict:
    """Validate the frozen three target events and resolve their provenance."""
    target_inputs_path = Path(target_inputs_path).resolve()
    targets = _read_object(target_inputs_path, "target inputs")
    _strict_schema_and_seeds(targets, "target inputs")
    for field in ("task_id", "dataset_commit", "evaluator_commit"):
        _nonempty_string(targets.get(field), f"target inputs {field}")
    raw_rows = targets.get("events")
    if not isinstance(raw_rows, list) or len(raw_rows) != 3:
        raise ValueError("target inputs events must contain exactly three objects")
    inputs_root = target_inputs_path.parent.resolve()
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError("target inputs event row must be an object")
        for field in ("event_id", "manifest_path", "manifest_sha256"):
            _nonempty_string(raw_row.get(field), f"target inputs event {field}")
        manifest_relative = Path(raw_row["manifest_path"])
        manifest_path = (inputs_root / manifest_relative).resolve()
        if manifest_relative.is_absolute() or not manifest_path.is_relative_to(inputs_root):
            raise ValueError("target event manifest escapes the prepared input root")
        manifest = _read_object(manifest_path, "target event manifest")
        _strict_schema_and_seeds(manifest, "target event manifest")
        for field in (
            "task_id",
            "dataset_commit",
            "evaluator_commit",
            "event_id",
            "story_id",
            "character_name",
        ):
            _nonempty_string(manifest.get(field), f"target event manifest {field}")
        for field in ("source_shot", "target_shot"):
            if type(manifest.get(field)) is not int:
                raise ValueError(f"target event manifest {field} must be an integer")
        outputs = manifest.get("outputs")
        event_output = outputs.get("event") if isinstance(outputs, Mapping) else None
        if not isinstance(event_output, Mapping):
            raise ValueError("target event manifest output event must be an object")
        event_relative = Path(
            _nonempty_string(event_output.get("path"), "target event output path")
        )
        event_path = (inputs_root / event_relative).resolve()
        if event_relative.is_absolute() or not event_path.is_relative_to(inputs_root):
            raise ValueError("target event output escapes the prepared input root")
        event = _read_object(event_path, "target event")
        if type(event.get("schema_version")) is not int or event["schema_version"] != 1:
            raise ValueError("target event schema_version must be integer 1")
        for field in ("event_id", "story_id", "entity_uid", "character_name"):
            _nonempty_string(event.get(field), f"target event {field}")
        for field in ("source_chunk_idx", "target_chunk_idx"):
            if type(event.get(field)) is not int:
                raise ValueError(f"target event {field} must be an integer")
    rows = _prepared_events(target_inputs_path, targets)
    if len(rows) != 3 or {row["event_id"] for row in rows} != TARGET_EVENT_IDS:
        raise ValueError("target inputs must contain exactly the frozen three events")
    return {
        **targets,
        "target_inputs_path": str(target_inputs_path),
        "target_inputs_sha256": sha256_file(target_inputs_path),
        "events": rows,
    }


def _payload_metadata(job: Mapping) -> tuple[str, dict, dict, dict]:
    runtime = job.get("dump_runtime_contract")
    frozen_args = runtime.get("frozen_args") if isinstance(runtime, Mapping) else None
    if not isinstance(frozen_args, Mapping) or frozen_args.get(
        "slotmem_memory_bank_mode", "single"
    ) != "single":
        raise ValueError("donor dump runtime must prove SlotMem single-bank mode")
    expected_layers, expected_slots = _dump_geometry(job)
    payload_path = Path(str(job["donor_payload"])).resolve()
    info_path = Path(str(job["donor_payload_info"])).resolve()
    info = _read_object(info_path, "donor payload info")
    artifact = torch.load(payload_path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("payloads"), Mapping):
        raise ValueError("donor payload must contain the validated v2 payload mapping")
    keys = info.get("payload_keys")
    if (
        not isinstance(keys, list)
        or len(keys) != 1
        or type(keys[0]) is not str
    ):
        raise ValueError("frozen donor artifact must contain exactly one single-bank payload key")
    event = job.get("event")
    character = str(event.get("character_name", "")) if isinstance(event, Mapping) else ""
    bank_zero = [
        key
        for key in keys
        if key.rsplit("|", 1)[0].casefold() == character.casefold()
        and key.rsplit("|", 1)[-1] == "0"
    ]
    if len(bank_zero) != 1:
        raise ValueError("frozen donor payload must contain exactly one target-character bank 0 key")
    key = bank_zero[0]
    payload = artifact["payloads"].get(key)
    shapes = payload_slot_shapes(payload, key)
    if not _json_equal_strict(info.get("payload_slot_shapes", {}).get(key), shapes):
        raise ValueError("donor payload shape differs from payload info")

    if (
        not _is_layerwise(payload)
        or payload.get("__layerwise__") is not True
        or not payload[LAYERS_KEY]
    ):
        raise ValueError("selected donor payload must contain non-empty layerwise tensors")
    tensors = payload[LAYERS_KEY]
    expected_layer_keys = {str(layer) for layer in expected_layers}
    if (
        any(type(layer) is not str or not layer for layer in tensors)
        or set(tensors) != expected_layer_keys
        or set(shapes) != expected_layer_keys
        or any(
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 2
            or tensor.shape[0] != expected_slots
            or not tensor.is_floating_point()
            or not torch.isfinite(tensor).all()
            for tensor in tensors.values()
        )
        or len({int(tensor.shape[1]) for tensor in tensors.values()}) != 1
    ):
        raise ValueError(
            "selected donor payload must contain exactly layers 0-15 as finite floating "
            f"{FROZEN_MEMORY_ENCODER_SLOTS}-slot 2D tensors with one hidden dimension"
        )
    dtypes = {
        layer: str(tensor.dtype).removeprefix("torch.") for layer, tensor in tensors.items()
    }
    slot_counts = {layer: shape[0] for layer, shape in shapes.items()}
    return key, shapes, dtypes, slot_counts


def _selection_manifest(selected: Mapping) -> dict:
    manifest_path = Path(str(selected["manifest_path"])).resolve()
    manifest = _read_object(manifest_path, "reviewed donor event manifest")
    candidate, review = manifest.get("candidate"), manifest.get("review")
    if not isinstance(candidate, Mapping) or not isinstance(review, Mapping):
        raise ValueError("reviewed donor event manifest is missing candidate/review")
    if review.get("approved") is not True:
        raise ValueError("donor candidate is not approved")
    if (
        review.get("target_presentation_class") != review.get("donor_presentation_class")
        or review.get("target_dominant_colour") != review.get("donor_dominant_colour")
        or review.get("donor_source_visible") is not True
        or review.get("donor_read_check_visible") is not True
    ):
        raise ValueError("reviewed donor matching constraints are not satisfied")
    return {"path": manifest_path, "candidate": candidate, "review": review}


def _pair_entry(target: Mapping, selected: Mapping, job: Mapping) -> tuple[dict, dict]:
    _strict_zero(selected.get("donor_seed"), "selection event donor_seed")
    _strict_zero(job.get("donor_seed"), "donor job donor_seed")
    for field in (
        "target_event_id",
        "candidate_id",
        "donor_story_id",
        "donor_entity_uid",
        "event_path",
    ):
        _nonempty_string(selected.get(field), f"selection event {field}")
    reviewed = _selection_manifest(selected)
    candidate, review = reviewed["candidate"], reviewed["review"]
    donor_event = job.get("event")
    if not isinstance(donor_event, Mapping):
        raise ValueError("donor job event is missing")
    for field in ("event_id", "story_id", "entity_uid", "character_name"):
        _nonempty_string(donor_event.get(field), f"donor job event {field}")
    target_event = target["event"]
    if job.get("target_event_id") != target_event["event_id"]:
        raise ValueError("donor job target_event_id is cross-wired")
    expected = {
        "target_event_id": target_event["event_id"],
        "candidate_id": selected["candidate_id"],
        "target_story_id": target_event["story_id"],
        "target_entity_uid": target_event["entity_uid"],
        "donor_story_id": selected["donor_story_id"],
        "donor_entity_uid": selected["donor_entity_uid"],
        "donor_char_id": donor_event.get("character_name"),
    }
    if any(
        type(candidate.get(field)) is not type(value) or candidate.get(field) != value
        for field, value in expected.items()
    ):
        raise ValueError("reviewed donor candidate is cross-wired to target/selection/job")
    if (
        review.get("target_event_id") != target_event["event_id"]
        or review.get("candidate_id") != selected["candidate_id"]
    ):
        raise ValueError("reviewed donor decision is cross-wired to its candidate")
    for field in (
        "target_presentation_class",
        "donor_presentation_class",
        "target_dominant_colour",
        "donor_dominant_colour",
    ):
        _nonempty_string(review.get(field), f"review {field}")
    if type(candidate.get("source_character_count")) is not int or candidate[
        "source_character_count"
    ] <= 0:
        raise ValueError("reviewed donor source_character_count must be a positive integer")
    _nonempty_string(candidate.get("gap_bucket"), "reviewed donor gap_bucket")
    if (
        donor_event.get("story_id") != selected["donor_story_id"]
        or donor_event.get("entity_uid") != selected["donor_entity_uid"]
        or target_event["story_id"] == selected["donor_story_id"]
        or target_event["entity_uid"] == selected["donor_entity_uid"]
    ):
        raise ValueError("target and donor story/identity must differ and remain cross-bound")

    payload_path = Path(str(job["donor_payload"])).resolve()
    info_path = Path(str(job["donor_payload_info"])).resolve()
    audit_path = Path(str(job["donor_audit"])).resolve()
    completion_path = Path(str(job["completion"])).resolve()
    payload_key, shapes, dtypes, slot_counts = _payload_metadata(job)
    entry = {
        "target_story_id": target_event["story_id"],
        "target_entity_uid": target_event["entity_uid"],
        "donor_story_id": selected["donor_story_id"],
        "donor_entity_uid": selected["donor_entity_uid"],
        "payload_path": str(payload_path),
        "payload_sha256": sha256_file(payload_path),
        "coarse_class": review["donor_presentation_class"],
        "colour": review["donor_dominant_colour"],
        "character_count": candidate["source_character_count"],
        "source_visible": review["donor_source_visible"],
        "gap_bucket": candidate["gap_bucket"],
        "slot_shape": shapes,
        "selection_seed": 0,
        "payload_key": payload_key,
        "donor_seed": 0,
        "read_check_visible": review["donor_read_check_visible"],
        "payload_dtype": dtypes,
        "slot_count": slot_counts,
    }
    provenance = {
        "target_event": target["prepared_provenance"],
        "selection_event_manifest": {
            "path": str(reviewed["path"]),
            "sha256": sha256_file(reviewed["path"]),
        },
        "source_event": {
            "path": selected["event_path"],
            "sha256": sha256_file(Path(str(selected["event_path"]))),
        },
        "donor_payload_info": {"path": str(info_path), "sha256": sha256_file(info_path)},
        "donor_audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        "donor_completion": {
            "path": str(completion_path),
            "sha256": sha256_file(completion_path),
        },
    }
    return entry, provenance


def build_validated_event_donor_map(
    targets: Mapping, selection: Mapping, donor_run: Mapping, output_root: Path
) -> dict[str, object]:
    target_by_id = {row["event_id"]: row for row in targets["events"]}
    selected_by_id = {row["target_event_id"]: row for row in selection["events"]}
    jobs_by_id = {job["target_event_id"]: job for job in donor_run["jobs"]}
    expected_ids = donor_selection_event_ids(selection)
    expected_scope = (
        {
            "protocol_scope": selection["protocol_scope"],
            "target_event_ids": sorted(expected_ids),
        }
        if expected_ids != TARGET_EVENT_IDS
        else {}
    )
    run_scope = {
        field: donor_run[field]
        for field in ("protocol_scope", "target_event_ids")
        if field in donor_run
    }
    if not _json_equal_strict(run_scope, expected_scope):
        raise ValueError("donor run scope does not match selection scope")
    if set(target_by_id) != TARGET_EVENT_IDS or len(targets["events"]) != len(
        TARGET_EVENT_IDS
    ):
        raise ValueError("target inputs must contain exactly the frozen three events")
    if (
        set(selected_by_id) != expected_ids
        or len(selection["events"]) != len(expected_ids)
        or set(jobs_by_id) != expected_ids
        or len(donor_run["jobs"]) != len(expected_ids)
    ):
        raise ValueError("selection and completed donor jobs do not match their scope")

    staging_root = Path(output_root)
    events = {}
    platform_hidden_dimension = None
    for event_id in sorted(expected_ids):
        target = target_by_id[event_id]
        selected = selected_by_id[event_id]
        job = jobs_by_id[event_id]
        if not _json_equal_strict(job.get("selection_event"), {
            key: value for key, value in selected.items() if key != "event"
        }):
            raise ValueError("donor job selection event is cross-wired")
        entry, provenance = _pair_entry(target, selected, job)
        hidden_dimension = next(iter({shape[1] for shape in entry["slot_shape"].values()}))
        if platform_hidden_dimension is None:
            platform_hidden_dimension = hidden_dimension
        elif hidden_dimension != platform_hidden_dimension:
            raise ValueError("donor payloads do not share one platform hidden dimension")
        pair_path = staging_root / event_id / "matched_pair.json"
        pair = {
            "schema_version": 1,
            "pairs": [entry],
            "provenance": {
                **provenance,
                "target_inputs": {
                    "path": targets["target_inputs_path"],
                    "sha256": targets["target_inputs_sha256"],
                },
                "selection": {
                    "path": selection["selection_path"],
                    "sha256": sha256_file(Path(str(selection["selection_path"]))),
                },
                "donor_run_manifest": {
                    "path": donor_run["donor_run_manifest_path"],
                    "sha256": donor_run["donor_run_manifest_sha256"],
                },
                "repository": donor_run["repository"],
                "platform_manifest": {
                    "path": donor_run["platform_manifest"],
                    "sha256": donor_run["platform_manifest_sha256"],
                },
            },
        }
        write_json_no_clobber(pair_path, pair)
        validate_donor_bundle(
            target["event"],
            Path(str(job["donor_payload"])),
            pair_path,
            loader=lambda path: torch.load(path, map_location="cpu", weights_only=True),
        )
        events[event_id] = {
            "payload": str(Path(str(job["donor_payload"])).resolve()),
            "manifest": str(pair_path.resolve()),
        }
    return {
        "schema_version": 1,
        "selection_sha256": sha256_file(Path(str(selection["selection_path"]))),
        "donor_run_manifest_sha256": donor_run["donor_run_manifest_sha256"],
        **expected_scope,
        "events": events,
    }


def freeze_vistory_donor_map(
    *,
    target_inputs_path: Path,
    selection_path: Path,
    donor_run_manifest_path: Path,
    output_root: Path,
) -> dict[str, object]:
    """Validate completed scoped donor jobs and atomically publish one event-level map."""
    target_inputs_path = Path(target_inputs_path).resolve()
    selection_path = Path(selection_path).resolve()
    donor_run_manifest_path = Path(donor_run_manifest_path).resolve()
    raw_output_root = Path(output_root)
    if os.path.lexists(str(raw_output_root)) or raw_output_root.is_symlink():
        raise FileExistsError(f"donor bundle output already exists: {raw_output_root}")
    output_root = raw_output_root.resolve()
    targets = validate_target_inputs(target_inputs_path)
    selection = validate_frozen_selection(selection_path)
    if selection.get("target_inputs_sha256") != sha256_file(target_inputs_path):
        raise ValueError("donor selection target-input provenance mismatch")
    donor_run = validate_completed_donor_run(donor_run_manifest_path, selection)
    donor_run = {
        **donor_run,
        "donor_run_manifest_path": str(donor_run_manifest_path),
        "donor_run_manifest_sha256": sha256_file(donor_run_manifest_path),
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(dir=output_root.parent, prefix=f".{output_root.name}.", suffix=".tmp")
    ).resolve()
    try:
        result = build_validated_event_donor_map(targets, selection, donor_run, staging_root)
        # Rewrite the staging paths to their immutable published locations.
        for event_id, entry in result["events"].items():
            entry["manifest"] = str((output_root / event_id / "matched_pair.json").resolve())
        write_json_no_clobber(staging_root / "donor_map.json", result)
        _publish_directory_no_clobber(staging_root, output_root)
    finally:
        if staging_root.exists():
            if not staging_root.is_relative_to(output_root.parent):
                raise RuntimeError("refusing to clean donor bundle staging outside output parent")
            shutil.rmtree(staging_root)
    return result
