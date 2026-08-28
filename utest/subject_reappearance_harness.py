"""Orchestrate the frozen ViStoryBench subject-reappearance experiment."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .content_audit import LAYERS_KEY, _payload_sha256, transform_slot_rows
from .event_harness import (
    _branch_inference_args,
    _frame_l1_median,
    _set_option,
    build_prefix_inference_args,
)
from .prefix_contract import (
    FROZEN_MEMORY_ENCODER_SLOTS,
    build_runtime_contract,
    normalized_frozen_args,
    sha256_file,
    validate_contract,
    validate_slotmem_memory_encoder_geometry,
)
from .subject_subspace import canonical_json_sha256, capture_tensor_sha256
from .subject_subspace_audit import SUBSPACE_ARMS
from .subject_subspace_audit import validate_subject_subspace_manifest
from .source_semantic_scores import validate_source_semantic_scores_file
from .qstar import SEVEN_RUNS, classify_qstar, qstar_deltas


PREFLIGHT_ARMS = ("full_correct", "no_memory", "zero_path", "wrong_subject")
FULL_ARMS = SUBSPACE_ARMS
_FROZEN_SELECTION = json.loads(
    (Path(__file__).parent / "events" / "vistorybench_reappearance_v1.json").read_text(
        encoding="utf-8"
    )
)
TASK_ID = _FROZEN_SELECTION["task_id"]
DATASET_COMMIT = _FROZEN_SELECTION["dataset_commit"]
EVALUATOR_COMMIT = _FROZEN_SELECTION["evaluator_commit"]
FROZEN_EVENTS = {row["event_id"]: row for row in _FROZEN_SELECTION["events"]}
REPO_ROOT = Path(__file__).parents[1].resolve()
RUN_MANIFEST_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Block:
    event_id: str
    seed: int
    event: Mapping


def _validated_inference_python(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("run manifest Python interpreter is invalid")
    return value


def _semantic_scores_command(
    *, event_path: Path, source_capture: Path, semantic_scores: Path
) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        "-m",
        "utest.source_semantic_scores",
        "--event",
        str(event_path),
        "--source-capture",
        str(source_capture),
        "--output",
        str(semantic_scores),
        "--repo-root",
        str(REPO_ROOT),
    ]


def _validate_semantic_scores_contract(
    row: Mapping, *, block_dir: Path | None = None
) -> list[str]:
    block_dir = (
        Path(str(row.get("block_dir", ""))).resolve()
        if block_dir is None
        else block_dir.resolve()
    )
    event_path = block_dir / "event.json"
    source_capture = block_dir / "subspace" / "source_capture.pt"
    semantic_scores = block_dir / "subspace" / "semantic_scores.json"
    expected_command = _semantic_scores_command(
        event_path=event_path,
        source_capture=source_capture,
        semantic_scores=semantic_scores,
    )
    commands = row.get("commands")
    external = row.get("required_external_inputs")
    logs = row.get("logs")
    if (
        str(row.get("event_json", "")) != str(event_path)
        or str(row.get("source_capture", "")) != str(source_capture)
        or str(row.get("semantic_scores", "")) != str(semantic_scores)
    ):
        raise ValueError("block semantic score artifact path contract is invalid")
    if not isinstance(commands, Mapping) or commands.get("semantic_scores") != expected_command:
        raise ValueError("block semantic score command contract is invalid")
    if not isinstance(external, Mapping) or "semantic_scores" in external:
        raise ValueError("semantic scores must be generated, not external input")
    if (
        not isinstance(logs, Mapping)
        or logs.get("semantic_scores_stdout")
        != str(block_dir / "subspace" / "semantic_scores.stdout.log")
        or logs.get("semantic_scores_stderr")
        != str(block_dir / "subspace" / "semantic_scores.stderr.log")
    ):
        raise ValueError("block semantic score log contract is invalid")
    return expected_command


def _probe_command(
    *,
    event_path: Path,
    source_capture: Path,
    semantic_scores: Path,
    output: Path,
    seed: int,
) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        "-m",
        "utest.subject_subspace_probe",
        "--event",
        str(event_path),
        "--source-capture",
        str(source_capture),
        "--semantic-scores",
        str(semantic_scores),
        "--output",
        str(output),
        "--seed",
        str(seed),
    ]


def _validate_probe_contract(
    row: Mapping, *, block_dir: Path | None = None
) -> list[str]:
    block_dir = (
        Path(str(row.get("block_dir", ""))).resolve()
        if block_dir is None
        else block_dir.resolve()
    )
    mask = block_dir / "subspace" / "subject_subspace_manifest.json"
    expected = _probe_command(
        event_path=block_dir / "event.json",
        source_capture=block_dir / "subspace" / "source_capture.pt",
        semantic_scores=block_dir / "subspace" / "semantic_scores.json",
        output=mask,
        seed=int(row.get("seed", -1)),
    )
    commands = row.get("commands")
    logs = row.get("logs")
    if str(row.get("subject_subspace_manifest", "")) != str(mask):
        raise ValueError("block probe output path contract is invalid")
    if not isinstance(commands, Mapping) or commands.get("probe") != expected:
        raise ValueError("block probe command contract is invalid")
    if (
        not isinstance(logs, Mapping)
        or logs.get("probe_stdout") != str(block_dir / "subspace" / "stdout.log")
        or logs.get("probe_stderr") != str(block_dir / "subspace" / "stderr.log")
    ):
        raise ValueError("block probe log contract is invalid")
    return expected


def build_matrix(selection: Mapping, seeds: Sequence[int] = (0, 1, 2)) -> list[Block]:
    """Return the preregistered 3-event x 3-seed block matrix."""
    events = list(selection.get("events", ()))
    seeds = tuple(seeds)
    if len(events) != 3 or len({str(row.get("event_id", "")) for row in events}) != 3:
        raise ValueError("selection must contain exactly three unique events")
    if seeds != (0, 1, 2) or list(selection.get("seeds", seeds)) != [0, 1, 2]:
        raise ValueError("seeds must be exactly (0, 1, 2)")
    return [Block(str(event["event_id"]), seed, event) for event in events for seed in seeds]


def qstar_contract_or_status(*, event: Mapping, teacher: Mapping | None) -> dict:
    """Keep Q* unavailable unless an independent teacher is explicitly frozen."""
    if teacher is None:
        return {"status": "not_available", "reason": "independent_teacher_missing"}
    return {**dict(teacher), "status": "available", "event_id": str(event["event_id"])}


def build_block_commands(
    block: Mapping, *, python: str, arms: Sequence[str] = FULL_ARMS
) -> dict[str, list[str]]:
    """Build target-arm commands from one frozen prefix contract."""
    requested = tuple(str(arm) for arm in arms)
    if any(arm not in FULL_ARMS for arm in requested):
        raise ValueError("unknown subject-reappearance arm")
    contract = dict(block["contract"])
    base = list(contract["base_inference_args"])
    base = _set_option(base, "--max_memory_characters", "4")
    base = _set_option(base, "--target_character", None)
    base = _set_option(base, "--fixed_reference_scope", "source_only")
    base = _set_option(base, "--subject_subspace_capture_path", None)
    contract["base_inference_args"] = base
    output = Path(block["output"])
    event_json = Path(block["event_json"]).resolve()
    mask = Path(block["subject_subspace_manifest"]).resolve()
    target_idx = int(contract["event"]["target_chunk_idx"])
    snapshot = str(contract["snapshot"]["path"])
    target_seed = int(block["target_seed"])
    commands: dict[str, list[str]] = {}
    for arm in requested:
        arm_dir = (output / "arms" / arm).resolve()
        inference_args = _branch_inference_args(
            contract, arm_dir, target_idx, snapshot, target_seed
        )
        command = [
            python,
            "-m",
            "utest.subject_subspace_audit",
            "--arm",
            arm,
            "--seed",
            str(target_seed),
            "--manifest",
            str(mask),
            "--event-json",
            str(event_json),
            "--report",
            str(arm_dir / "audit.json"),
        ]
        if arm == "wrong_subject":
            if not block.get("donor") or not block.get("donor_manifest"):
                raise ValueError("wrong_subject requires donor and donor_manifest")
            command.extend(
                [
                    "--donor",
                    str(Path(block["donor"]).resolve()),
                    "--donor-manifest",
                    str(Path(block["donor_manifest"]).resolve()),
                ]
            )
        commands[arm] = [*command, "--", *inference_args]
    return commands


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _json_bytes(payload: Mapping) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _read_bytes_json(path: Path) -> tuple[object, str]:
    data = path.read_bytes()
    return json.loads(data.decode("utf-8-sig")), hashlib.sha256(data).hexdigest()


def _parse_args(text: str) -> list[str]:
    value = json.loads(text)
    if isinstance(value, Mapping):
        value = value.get("argv")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("base inference args must be a JSON argv list or object")
    return list(value[1:] if value and not value[0].startswith("--") else value)


def _write_json_exclusive(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _json_bytes(payload)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _prepared_events(inputs: Path, selection: Mapping) -> list[dict]:
    if (
        selection.get("schema_version") != 1
        or selection.get("task_id") != TASK_ID
        or selection.get("dataset_commit") != DATASET_COMMIT
        or selection.get("evaluator_commit") != EVALUATOR_COMMIT
        or selection.get("seeds") != [0, 1, 2]
        or {row.get("event_id") for row in selection.get("events", ())} != set(FROZEN_EVENTS)
    ):
        raise ValueError("prepared top-level provenance is not the frozen ViStoryBench selection")
    rows = []
    inputs_root = inputs.parent.resolve()
    for item in selection["events"]:
        event_id = str(item["event_id"])
        frozen = FROZEN_EVENTS[event_id]
        story_id = str(frozen["story_id"])
        character = str(frozen["character_name"])
        source_shot, target_shot = int(frozen["source_shot"]), int(frozen["target_shot"])
        target_idx = target_shot - source_shot
        event_manifest_path = (inputs.parent / str(item["manifest_path"])).resolve()
        if not event_manifest_path.is_relative_to(inputs_root):
            raise ValueError("prepared event manifest escapes the prepared input root")
        event_manifest, event_manifest_sha = _read_bytes_json(event_manifest_path)
        if event_manifest_sha.casefold() != str(
            item.get("manifest_sha256", "")
        ).casefold():
            raise ValueError("prepared event manifest SHA-256 mismatch")
        event_output = event_manifest["outputs"]["event"]
        story_output = event_manifest["outputs"]["story"]
        event_path = (inputs.parent / str(event_output["path"])).resolve()
        story_path = (inputs.parent / str(story_output["path"])).resolve()
        if not event_path.is_relative_to(inputs_root) or not story_path.is_relative_to(inputs_root):
            raise ValueError("prepared event/story output escapes the prepared input root")
        event, event_sha = _read_bytes_json(event_path)
        if event_sha.casefold() != str(event_output.get("sha256", "")).casefold():
            raise ValueError("prepared event JSON SHA-256 mismatch")
        story_bytes = story_path.read_bytes()
        reference_path = Path(str(event["reference_path"])).resolve()
        reference_sha = sha256_file(reference_path)
        if str(event_manifest.get("official_story", {}).get("sha256", "")).casefold() != str(
            frozen["story_sha256"]
        ).casefold():
            raise ValueError("prepared official story SHA-256 differs from frozen selection")
        expected = {
            "event_id": event_id,
            "story_id": story_id,
            "character_name": character,
            "source_shot": source_shot,
            "target_shot": target_shot,
        }
        if (
            event_manifest.get("schema_version") != 1
            or event_manifest.get("task_id") != TASK_ID
            or event_manifest.get("dataset_commit") != DATASET_COMMIT
            or event_manifest.get("evaluator_commit") != EVALUATOR_COMMIT
            or event_manifest.get("seeds") != [0, 1, 2]
            or any(str(event_manifest.get(key)) != str(value) for key, value in expected.items())
            or event.get("event_id") != event_id
            or str(event.get("story_id")) != story_id
            or event.get("character_name") != character
            or event.get("source_chunk_idx") != 0
            or event.get("target_chunk_idx") != target_idx
            or Path(str(event["source_json_path"])).resolve() != story_path
            or hashlib.sha256(story_bytes).hexdigest() != story_output.get("sha256")
            or reference_sha != event.get("reference_sha256")
            or reference_sha != event_manifest.get("reference_sha256")
            or not reference_path.as_posix().endswith(str(event_manifest.get("reference_path", "")))
        ):
            raise ValueError("prepared event/story/reference provenance mismatch")
        rows.append(
            {
                **item,
                "event": event,
                "event_path": event_path,
                "prepared_provenance": {
                    "event_manifest_path": str(event_manifest_path),
                    "event_manifest_sha256": event_manifest_sha,
                    "event_path": str(event_path),
                    "event_sha256": event_sha,
                    "story_path": str(story_path),
                    "story_sha256": hashlib.sha256(story_bytes).hexdigest(),
                    "reference_path": str(reference_path),
                    "reference_sha256": reference_sha,
                },
            }
        )
    return rows


def _map_entry(payload: object, event_id: str, seed: int) -> Mapping | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("donor/teacher map must be a JSON object")
    rows = payload.get("blocks", payload.get("events", payload))
    if not isinstance(rows, Mapping):
        raise ValueError("donor/teacher map entries must be an object")
    value = rows.get(f"{event_id}/seed_{seed}", rows.get(event_id))
    return value if isinstance(value, Mapping) else None


def _validated_donor(entry: Mapping | None, event: Mapping) -> dict | None:
    if entry is None:
        return None
    donor_value = entry.get("payload", entry.get("donor"))
    manifest_value = entry.get("manifest", entry.get("donor_manifest"))
    if not donor_value or not manifest_value:
        raise ValueError("donor map entry requires payload and manifest")
    donor, manifest = Path(str(donor_value)), Path(str(manifest_value))
    import torch
    from .input_contract import validate_donor_bundle

    return validate_donor_bundle(
        event,
        donor,
        manifest,
        loader=lambda path: torch.load(path, map_location="cpu", weights_only=True),
    )


def _validated_teacher(entry: Mapping | None, event: Mapping, arms_root: Path) -> dict | None:
    if entry is None:
        return None
    video_value = entry.get("video", entry.get("future_target_video"))
    manifest_value = entry.get("manifest", entry.get("teacher_manifest"))
    if not video_value or not manifest_value:
        raise ValueError("teacher map entry requires video and manifest")
    video, manifest = Path(str(video_value)), Path(str(manifest_value))
    from .input_contract import validate_teacher_bundle

    return validate_teacher_bundle(event, video, manifest, arms_root=arms_root)


def build_run_manifest(
    *,
    inputs: Path,
    output: Path,
    base_inference_args: Path,
    platform_manifest: Path,
    python: str = sys.executable,
    donor_map: Path | None = None,
    teacher_map: Path | None = None,
) -> dict:
    """Materialize a zero-GPU, immutable command plan for all nine blocks."""
    python = _validated_inference_python(python)
    selection, inputs_sha = _read_bytes_json(inputs)
    matrix = build_matrix(selection)
    by_id = {row["event_id"]: row for row in _prepared_events(inputs, selection)}
    base_bytes = base_inference_args.read_bytes()
    base = _parse_args(base_bytes.decode("utf-8-sig"))
    validate_slotmem_memory_encoder_geometry(normalized_frozen_args(base))
    platform_bytes = platform_manifest.read_bytes()
    donors, donor_map_sha = _read_bytes_json(donor_map) if donor_map else (None, None)
    teachers, teacher_map_sha = _read_bytes_json(teacher_map) if teacher_map else (None, None)
    blocks = []
    event_files: list[tuple[Path, dict]] = []
    for block in matrix:
        prepared = by_id[block.event_id]
        block_dir = (output / block.event_id / f"seed_{block.seed}").resolve()
        event_path = block_dir / "event.json"
        event = {
            **prepared["event"],
            "source_seed": block.seed,
            "target_seed": block.seed,
        }
        event_files.append((event_path, event))
        prefix_dir = block_dir / "prefix"
        source_capture = block_dir / "subspace" / "source_capture.pt"
        semantic_scores = block_dir / "subspace" / "semantic_scores.json"
        mask = block_dir / "subspace" / "subject_subspace_manifest.json"
        donor = _validated_donor(_map_entry(donors, block.event_id, block.seed), event)
        teacher = _validated_teacher(
            _map_entry(teachers, block.event_id, block.seed), event, block_dir / "full" / "arms"
        )
        frozen_base = _set_option(base, "--seed_base", str(block.seed))
        frozen_base = _set_option(frozen_base, "--max_memory_characters", "4")
        frozen_base = _set_option(frozen_base, "--target_character", None)
        frozen_base = _set_option(frozen_base, "--fixed_reference_scope", "source_only")
        frozen_base = _set_option(
            frozen_base, "--subject_subspace_capture_path", str(source_capture)
        )
        prefix_preview = build_prefix_inference_args(
            event, prefix_dir, frozen_base, target_seed_override=block.seed
        )
        prefix_command = [
            python,
            "-m",
            "utest.event_harness",
            "prepare-prefix",
            "--event",
            str(event_path),
            "--output",
            str(prefix_dir),
            "--platform-manifest",
            str(platform_manifest.resolve()),
            "--arm-seed",
            str(block.seed),
            "--target-seed-override",
            str(block.seed),
            "--python",
            python,
        ]
        if teacher:
            prefix_command.extend(
                [
                    "--future-target-video",
                    teacher["video_path"],
                    "--future-target-manifest",
                    teacher["manifest_path"],
                    "--arms-root",
                    str(block_dir / "full" / "arms"),
                ]
            )
        prefix_command.extend(["--", *frozen_base])
        semantic_scores_command = _semantic_scores_command(
            event_path=event_path,
            source_capture=source_capture,
            semantic_scores=semantic_scores,
        )
        probe_command = _probe_command(
            event_path=event_path,
            source_capture=source_capture,
            semantic_scores=semantic_scores,
            output=mask,
            seed=block.seed,
        )
        if donor:
            phase_commands = {
                "preflight": {"status": "deferred_until_prefix", "arm_order": list(PREFLIGHT_ARMS)},
                "full": {"status": "deferred_until_prefix", "arm_order": list(FULL_ARMS)},
            }
        else:
            phase_commands = {
                "preflight": {"status": "blocked_missing_donor", "arm_order": list(PREFLIGHT_ARMS)},
                "full": {"status": "blocked_missing_donor", "arm_order": list(FULL_ARMS)},
            }
        commands = {
            "prefix": prefix_command,
            "prefix_inference_args": prefix_preview,
            "semantic_scores": semantic_scores_command,
            "probe": probe_command,
            **phase_commands,
        }
        qstar = qstar_contract_or_status(event=event, teacher=teacher)
        if teacher and not donor:
            qstar = {**qstar, "status": "blocked_missing_donor"}
        commands["qstar"] = {
            "status": (
                "deferred_until_prefix" if teacher and donor
                else "blocked_missing_donor" if teacher
                else "not_available"
            )
        }
        blocks.append(
            {
                "event_id": block.event_id,
                "seed": block.seed,
                "block_dir": str(block_dir),
                "event_json": str(event_path),
                "event_json_sha256": hashlib.sha256(_json_bytes(event)).hexdigest(),
                "prepared_provenance": prepared["prepared_provenance"],
                "prefix_snapshot": str(prefix_dir / "prefix_state.pt"),
                "command_artifact": str(block_dir / "stage_commands.json"),
                "source_qualification": str(block_dir / "source_qualification.json"),
                "subject_subspace_manifest": str(mask),
                "source_capture": str(source_capture),
                "semantic_scores": str(semantic_scores),
                "required_external_inputs": {
                    "wrong_subject_donor_map": None if donor else "missing",
                },
                "donor": donor,
                "preflight_arms": list(PREFLIGHT_ARMS),
                "full_arms": list(FULL_ARMS),
                "target_seed": block.seed,
                "fixed_reference_scope": "source_only",
                "qstar": qstar,
                "commands": commands,
                "logs": {
                    "prefix_stdout": str(block_dir / "logs" / "prefix.stdout.log"),
                    "prefix_stderr": str(block_dir / "logs" / "prefix.stderr.log"),
                    "semantic_scores_stdout": str(
                        block_dir / "subspace" / "semantic_scores.stdout.log"
                    ),
                    "semantic_scores_stderr": str(
                        block_dir / "subspace" / "semantic_scores.stderr.log"
                    ),
                    "probe_stdout": str(block_dir / "subspace" / "stdout.log"),
                    "probe_stderr": str(block_dir / "subspace" / "stderr.log"),
                },
            }
        )
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "task_id": "vistorybench_subject_reappearance_v1",
        "python": python,
        "inputs_manifest": str(inputs.resolve()),
        "inputs_manifest_sha256": inputs_sha,
        "base_inference_args": str(base_inference_args.resolve()),
        "base_inference_args_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "platform_manifest": str(platform_manifest.resolve()),
        "platform_manifest_sha256": hashlib.sha256(platform_bytes).hexdigest(),
        "donor_map": str(donor_map.resolve()) if donor_map else None,
        "donor_map_sha256": donor_map_sha,
        "teacher_map": str(teacher_map.resolve()) if teacher_map else None,
        "teacher_map_sha256": teacher_map_sha,
        "preflight_arms": list(PREFLIGHT_ARMS),
        "full_arms": list(FULL_ARMS),
        "blocks": blocks,
    }
    for event_path, event in event_files:
        _write_json_exclusive(event_path, event)
    return manifest


def _positive_measured_injection(chunk: Mapping) -> bool:
    if "last_sparse_role_memory_stats_by_layer" in chunk:
        by_layer = chunk["last_sparse_role_memory_stats_by_layer"]
        if not isinstance(by_layer, Mapping):
            raise ValueError("layerwise injection diagnostic is not an object")
        rows = list(by_layer.values())
    elif "last_sparse_role_memory_stats" in chunk:
        rows = [chunk["last_sparse_role_memory_stats"]]
    else:
        return False
    measured = False
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("injection diagnostic layer is not an object")
        try:
            enabled = float(row.get("enabled", 0.0) or 0.0)
            selected = float(row.get("selected_memory_tokens", 0.0) or 0.0)
            effective = float(row.get("effective_delta_norm", 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("injection diagnostic is not numeric") from exc
        if not all(math.isfinite(value) for value in (enabled, selected, effective)):
            raise ValueError("injection diagnostic is not finite")
        measured = measured or (enabled > 0 and selected > 0 and effective > 0)
    return measured


def _expected_rows(arm: str, masks: Mapping[str, list[int]]) -> list[int]:
    if arm == "no_memory":
        return []
    if arm in {"subject_only", "wrong_subject"}:
        return list(masks["semantic"])
    if arm == "drop_subject":
        return [
            index
            for index in range(FROZEN_MEMORY_ENCODER_SLOTS)
            if index not in masks["semantic"]
        ]
    if arm == "random_only":
        return list(masks["random"])
    if arm == "drop_random":
        return [
            index
            for index in range(FROZEN_MEMORY_ENCODER_SLOTS)
            if index not in masks["random"]
        ]
    return list(range(FROZEN_MEMORY_ENCODER_SLOTS))


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value.casefold()) <= set(
        "0123456789abcdef"
    )


def _load_source_slots(block: Mapping, manifest: Mapping) -> dict[int, dict[str, object]]:
    import torch

    path = Path(block.get("source_capture", Path(block["subject_subspace_manifest"]).parent / "source_capture.pt"))
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest["inputs"]["source_capture_sha256"]:
        raise ValueError("source capture SHA-256 differs from frozen mask contract")
    artifact = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    slots: dict[int, dict[str, object]] = {}
    for row in artifact.get("captures", ()):
        if str(row.get("character", "")).casefold() == str(block["event"]["character_name"]).casefold():
            bank, layer = int(row["bank"]), str(row["layer"])
            if layer in slots.setdefault(bank, {}):
                raise ValueError(f"source capture repeats bank {bank} layer {layer}")
            slots[bank][layer] = row["encoded_slots"]
    if not slots:
        raise ValueError("source capture has no subject slots")
    return slots


def _load_donor_slots(block: Mapping) -> dict[int, dict[str, object]]:
    import torch

    donor = block.get("donor")
    if not isinstance(donor, Mapping):
        raise ValueError("wrong_subject requires validated donor provenance")
    validated = _validated_donor(
        {"payload": donor["payload_path"], "manifest": donor["manifest_path"]},
        block["event"],
    )
    data = Path(validated["payload_path"]).read_bytes()
    artifact = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    selected = artifact["payloads"][validated["payload_key"]]
    bank = int(str(validated["payload_key"]).rsplit("|", 1)[1])
    return {bank: {str(layer): tensor for layer, tensor in selected[LAYERS_KEY].items()}}


def _validate_donor_target_compatibility(row: Mapping) -> None:
    """Reject a frozen donor that cannot be transformed into this target's slots."""
    import torch

    event = _read_json(Path(row["event_json"]))
    manifest = _read_json(Path(row["subject_subspace_manifest"]))
    seed = int(row.get("target_seed", row["seed"]))
    banks = validate_subject_subspace_manifest(manifest, event, seed=seed)
    block = {**row, "event": event}
    source_slots = _load_source_slots(block, manifest)
    donor_slots = _load_donor_slots(block)
    expected_banks = set(banks)
    if set(source_slots) != expected_banks or set(donor_slots) != expected_banks:
        raise ValueError("donor-target bank sets do not match the frozen mask contract")
    for bank, layers in banks.items():
        expected_layers = set(layers)
        source_layers = source_slots[bank]
        donor_layers = donor_slots[bank]
        if set(source_layers) != expected_layers or set(donor_layers) != expected_layers:
            raise ValueError(
                f"donor-target layer sets for bank {bank} do not match the frozen mask contract"
            )
        for layer, layer_contract in layers.items():
            source = source_layers[layer]
            donor = donor_layers[layer]
            expected_slots = layer_contract["slot_count"]
            if (
                expected_slots != FROZEN_MEMORY_ENCODER_SLOTS
                or not isinstance(source, torch.Tensor)
                or not isinstance(donor, torch.Tensor)
                or source.ndim != 2
                or donor.ndim != 2
                or source.shape[0] != expected_slots
                or donor.shape[0] != expected_slots
                or tuple(source.shape) != tuple(donor.shape)
            ):
                raise ValueError(
                    f"donor-target tensor shape mismatch at bank {bank} layer {layer}"
                )


def _expected_returned_hashes(
    arm: str,
    bank: int,
    bank_layers: Mapping[str, Mapping],
    source_slots: Mapping[int, Mapping[str, object]],
    donor_slots: Mapping[int, Mapping[str, object]] | None,
) -> tuple[str | None, dict[str, str]]:
    import torch

    if arm == "no_memory":
        return None, {}
    output = {}
    for layer, contract in bank_layers.items():
        source = source_slots.get(bank, {}).get(layer)
        if not isinstance(source, torch.Tensor):
            raise ValueError(f"source capture is missing bank {bank} layer {layer}")
        if capture_tensor_sha256(source) != contract["source_payload_sha256"]:
            raise ValueError("source capture payload differs from frozen mask contract")
        if arm == "full_correct":
            transformed = source
        elif arm == "zero_path":
            transformed = torch.zeros_like(source)
        else:
            donor = donor_slots.get(bank, {}).get(layer) if donor_slots else None
            transformed = transform_slot_rows(source, arm, contract, donor)
        output[layer] = transformed
    payload = {"tokens": {"__layerwise__": True, LAYERS_KEY: output}}
    return _payload_sha256(payload), {
        layer: capture_tensor_sha256(tensor) for layer, tensor in output.items()
    }


def validate_block(
    block: Mapping,
    *,
    arms: Sequence[str],
    decoded_gate: bool = True,
    require_measured: bool = True,
) -> dict:
    """Fail closed over one completed preflight or full-arm block."""
    root = Path(block["output"])
    qualification = _read_json(
        Path(block.get("source_qualification", root / "source_qualification.json"))
    )
    if qualification.get("status") != "passed":
        raise ValueError("source qualification is not passed")
    manifest = _read_json(Path(block["subject_subspace_manifest"]))
    seed = int(block.get("target_seed", block.get("seed", 0)))
    banks = validate_subject_subspace_manifest(manifest, block["event"], seed=seed)
    mask_sha = str(manifest["mask_manifest_sha256"])
    manifest_file_sha = sha256_file(Path(block["subject_subspace_manifest"]))
    event_file = Path(block["event_json"])
    event_file_sha = sha256_file(event_file)
    target_idx = int(block["event"]["target_chunk_idx"])
    subject = str(block["event"].get("character_name", ""))
    source_slots = _load_source_slots(block, manifest)
    donor_slots = _load_donor_slots(block) if "wrong_subject" in arms else None
    measured = False
    video_paths = {}
    for arm in arms:
        arm_dir = Path(block.get("arms_root", root / "arms")) / arm
        audit = _read_json(arm_dir / "audit.json")
        if int(audit.get("target_read_hits", 0)) <= 0 or int(
            audit.get("target_source_non_null_reads", 0)
        ) <= 0:
            raise ValueError(f"{arm}: target memory was not read")
        provenance = audit.get("subject_subspace_contract", {})
        if (
            provenance.get("mask_manifest_sha256") != mask_sha
            or provenance.get("manifest_file_sha256") != manifest_file_sha
            or provenance.get("source_capture_sha256") != manifest["inputs"]["source_capture_sha256"]
            or provenance.get("event_file_sha256") != event_file_sha
            or provenance.get("event_id") != block["event"].get("event_id")
            or provenance.get("seed") != seed
            or provenance.get("target_evidence_read") is not False
        ):
            raise ValueError(f"{arm}: returned payload mask contract mismatch")
        contract = block.get("contract")
        if isinstance(contract, Mapping) and "runtime_contract" in contract:
            runtime_errors = validate_contract(
                contract, Path(contract["snapshot"]["path"]), audit.get("runtime_contract")
            )
            if runtime_errors:
                raise ValueError(f"{arm}: runtime contract mismatch: {','.join(runtime_errors)}")
        target_records = [
            row
            for row in audit.get("read_records", ())
            if row.get("chunk_idx") == target_idx
            and str(row.get("character", "")).casefold() == subject.casefold()
        ]
        if not target_records:
            raise ValueError(f"{arm}: target payload record missing")
        for record in target_records:
            bank_layers = banks.get(int(record.get("bank", 0)))
            if bank_layers is None:
                raise ValueError(f"{arm}: target payload bank is absent from frozen mask contract")
            source_hashes = {
                layer: value["source_payload_sha256"] for layer, value in bank_layers.items()
            }
            masks = {
                layer: {
                    "semantic": value["semantic_top8"],
                    "random": value["random_top8"],
                }
                for layer, value in bank_layers.items()
            }
            if record.get("source_manifest_sha256_by_layer") != source_hashes:
                raise ValueError(f"{arm}: source payload SHA differs from frozen mask contract")
            expected = {layer: _expected_rows(arm, value) for layer, value in masks.items()}
            if record.get("selected_indices_by_layer") != expected:
                raise ValueError(f"{arm}: returned payload selection differs from frozen mask contract")
            expected_payload, expected_layers = _expected_returned_hashes(
                arm, int(record.get("bank", 0)), bank_layers, source_slots, donor_slots
            )
            if (
                record.get("returned_sha256") != expected_payload
                or record.get("returned_manifest_sha256_by_layer") != expected_layers
            ):
                raise ValueError(f"{arm}: returned payload SHA differs from frozen transform")
            if arm == "no_memory" and int(audit.get("target_returned_non_null_reads", 0)) != 0:
                raise ValueError("no_memory returned a payload")
        metadata = _read_json(arm_dir / f"chunk_{target_idx:03d}.metadata.json")
        reference = metadata.get("reference_conditioning", {})
        if reference.get("fixed_reference_scope") != "source_only" or reference.get(
            "fixed_reference_used"
        ) is not False:
            raise ValueError(f"{arm}: initial reference was used after source")
        efficiency = _read_json(arm_dir / "efficiency.json")
        if arm not in {"no_memory", "zero_path"}:
            target_chunks = [
                row for row in efficiency.get("chunks", ()) if int(row.get("chunk_idx", -1)) == target_idx
            ]
            measured = measured or any(_positive_measured_injection(row) for row in target_chunks)
        video = arm_dir / f"chunk_{target_idx:03d}.mp4"
        if not video.is_file():
            raise ValueError(f"{arm}: decoded target video missing")
        video_paths[arm] = video
    if require_measured and not measured:
        raise ValueError("no non-zero arm has measured injection")
    decoded = None
    if decoded_gate and set(PREFLIGHT_ARMS).issubset(video_paths):
        distances = {
            "zero_path_vs_no_memory": _frame_l1_median(video_paths["zero_path"], video_paths["no_memory"]),
            "full_correct_vs_no_memory": _frame_l1_median(video_paths["full_correct"], video_paths["no_memory"]),
            "full_correct_vs_wrong_subject": _frame_l1_median(video_paths["full_correct"], video_paths["wrong_subject"]),
        }
        if not all(math.isfinite(value) for value in distances.values()):
            raise ValueError("decoded preflight L1 is not finite")
        if distances["zero_path_vs_no_memory"] != 0:
            raise ValueError("decoded zero_path and no_memory are not path-equivalent")
        if distances["full_correct_vs_no_memory"] <= 0 or distances["full_correct_vs_wrong_subject"] <= 0:
            raise ValueError("decoded preflight has no measurable content influence")
        decoded = {
            "video_sha256": {arm: sha256_file(path) for arm, path in video_paths.items()},
            "median_frame_l1": distances,
            "path_equivalent": True,
            "content_influence_measured": True,
        }
    return {"status": "passed", "arms": list(arms), "decoded_preflight": decoded}


def _load_run_manifest(path: Path) -> dict:
    manifest = _read_json(path)
    run_root = path.parent.resolve()
    if manifest.get("schema_version") == 1:
        raise ValueError(
            "run manifest predates generated semantic scores; rerun dry-run"
        )
    if (
        manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION
        or manifest.get("task_id") != TASK_ID
    ):
        raise ValueError("run manifest schema/task is not frozen")
    python = _validated_inference_python(manifest.get("python"))
    for path_key, hash_key in (
        ("inputs_manifest", "inputs_manifest_sha256"),
        ("base_inference_args", "base_inference_args_sha256"),
        ("platform_manifest", "platform_manifest_sha256"),
        ("donor_map", "donor_map_sha256"),
        ("teacher_map", "teacher_map_sha256"),
    ):
        value = manifest.get(path_key)
        if value and sha256_file(Path(value)) != manifest.get(hash_key):
            raise ValueError(f"{path_key} SHA-256 mismatch")
    blocks = manifest.get("blocks", ())
    addresses = {(row.get("event_id"), row.get("seed")) for row in blocks}
    event_ids = {event_id for event_id, _ in addresses}
    if (
        len(blocks) != 9
        or len(addresses) != 9
        or len(event_ids) != 3
        or event_ids != set(FROZEN_EVENTS)
        or any({seed for candidate, seed in addresses if candidate == event_id} != {0, 1, 2} for event_id in event_ids)
        or manifest.get("preflight_arms") != list(PREFLIGHT_ARMS)
        or manifest.get("full_arms") != list(FULL_ARMS)
    ):
        raise ValueError("run manifest must contain exactly nine unique frozen blocks")
    for row in blocks:
        expected_block_dir = (
            run_root / str(row.get("event_id", "")) / f"seed_{int(row.get('seed', -1))}"
        ).resolve()
        if str(row.get("block_dir", "")) != str(expected_block_dir):
            raise ValueError("block directory contract is invalid")
        _validate_semantic_scores_contract(row, block_dir=expected_block_dir)
        _validate_probe_contract(row, block_dir=expected_block_dir)
        commands = row.get("commands", {})
        if (
            not isinstance(commands.get("prefix"), list)
            or not commands["prefix"]
            or commands["prefix"][0] != python
        ):
            raise ValueError("block inference Python interpreter contract is invalid")
        if (
            row.get("preflight_arms") != list(PREFLIGHT_ARMS)
            or row.get("full_arms") != list(FULL_ARMS)
            or row.get("fixed_reference_scope") != "source_only"
            or int(row.get("target_seed", -1)) != int(row["seed"])
        ):
            raise ValueError("block arm, reference, or seed contract is invalid")
        if sha256_file(Path(row["event_json"])) != row.get("event_json_sha256"):
            raise ValueError("block event JSON SHA-256 mismatch")
    selection = _read_json(Path(manifest["inputs_manifest"]))
    prepared = {
        row["event_id"]: row for row in _prepared_events(Path(manifest["inputs_manifest"]), selection)
    }
    validated_donors = {}
    validated_teachers = {}
    for row in blocks:
        event = _read_json(Path(row["event_json"]))
        expected_event = {
            **prepared[row["event_id"]]["event"],
            "source_seed": int(row["seed"]),
            "target_seed": int(row["seed"]),
        }
        if (
            event != expected_event
            or Path(row["event_json"]).resolve()
            != (Path(row["block_dir"]) / "event.json").resolve()
        ):
            raise ValueError("block event differs from the frozen prepared event")
        if row.get("prepared_provenance") != prepared[row["event_id"]]["prepared_provenance"]:
            raise ValueError("block prepared provenance differs from current descendants")
        donor = row.get("donor")
        if donor:
            key = (row["event_id"], donor["payload_path"], donor["manifest_path"])
            if key not in validated_donors:
                validated_donors[key] = _validated_donor(
                    {"payload": donor["payload_path"], "manifest": donor["manifest_path"]},
                    event,
                )
            if donor != validated_donors[key]:
                raise ValueError("block donor provenance mismatch")
        teacher = row.get("qstar")
        if isinstance(teacher, Mapping) and teacher.get("video_path"):
            key = (row["event_id"], teacher["video_path"], teacher["manifest_path"])
            if key not in validated_teachers:
                validated_teachers[key] = _validated_teacher(
                    {"video": teacher["video_path"], "manifest": teacher["manifest_path"]},
                    event,
                    Path(row["block_dir"]) / "full" / "arms",
                )
            expected_teacher = qstar_contract_or_status(
                event=event, teacher=validated_teachers[key]
            )
            if not donor:
                expected_teacher["status"] = "blocked_missing_donor"
            if teacher != expected_teacher:
                raise ValueError("block teacher provenance mismatch")
        qstar_status = row.get("qstar", {}).get("status")
        command_status = row.get("commands", {}).get("qstar", {}).get("status")
        expected_command_status = (
            "deferred_until_prefix" if qstar_status == "available"
            else "blocked_missing_donor" if qstar_status == "blocked_missing_donor"
            else "not_available"
        )
        if command_status != expected_command_status:
            raise ValueError("block Q* status/command contract mismatch")
    return manifest


def _run_logged(command: Sequence[str], stdout_path: Path, stderr_path: Path) -> None:
    import subprocess

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    original_stdout, original_stderr = stdout_path, stderr_path
    while stdout_path.exists() or stderr_path.exists():
        attempt += 1
        stdout_path = original_stdout.with_name(
            f"{original_stdout.stem}.retry_{attempt}{original_stdout.suffix}"
        )
        stderr_path = original_stderr.with_name(
            f"{original_stderr.stem}.retry_{attempt}{original_stderr.suffix}"
        )
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open(
        "x", encoding="utf-8"
    ) as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr, text=True, check=True)


def _ensure_semantic_scores(row: Mapping) -> None:
    """Produce once, validate always, and never overwrite an invalid artifact."""
    command = _validate_semantic_scores_contract(row)
    capture = Path(row["source_capture"])
    if not capture.is_file():
        raise FileNotFoundError("source capture artifact missing")
    scores = Path(row["semantic_scores"])
    if scores.exists() and not scores.is_file():
        raise ValueError("semantic score artifact is not a file")
    if not scores.exists():
        _run_logged(
            command,
            Path(row["logs"]["semantic_scores_stdout"]),
            Path(row["logs"]["semantic_scores_stderr"]),
        )
    validate_source_semantic_scores_file(
        event_path=Path(row["event_json"]),
        source_capture_path=capture,
        scores_path=scores,
        repo_root=REPO_ROOT,
    )


def _selected_blocks(manifest: Mapping, event_id: str | None, seed: int | None) -> list[dict]:
    rows = [
        row
        for row in manifest["blocks"]
        if (event_id is None or row["event_id"] == event_id)
        and (seed is None or int(row["seed"]) == seed)
    ]
    if not rows:
        raise ValueError("no run-manifest block matches the requested event/seed")
    return rows


def _archive_path(path: Path) -> Path:
    attempt = 1
    while True:
        archived = path.with_name(f"{path.name}.failed_{attempt}")
        if not archived.exists():
            path.rename(archived)
            return archived
        attempt += 1


def _validated_prefix_contract(row: Mapping) -> dict:
    contract = _read_json(Path(row["block_dir"]) / "prefix" / "prefix_contract.json")
    snapshot = Path(contract["snapshot"]["path"])
    errors = validate_contract(contract, snapshot)
    event = _read_json(Path(row["event_json"]))
    runtime = build_runtime_contract(event, row["commands"]["prefix_inference_args"])
    if (
        errors
        or str(snapshot.resolve()) != str(Path(row["prefix_snapshot"]).resolve())
        or contract.get("event") != event
        or runtime.get("target_seed") != int(row["target_seed"])
        or contract.get("runtime_contract") != runtime
    ):
        raise ValueError("prefix contract does not match the frozen block")
    return contract


def _validate_qstar_report(row: Mapping, contract: Mapping) -> dict:
    root = Path(row["block_dir"]) / "qstar"
    report = _read_json(root / "qstar_report.json")
    records_path = root / "qstar_records.jsonl"
    try:
        records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("Q* records are missing or invalid") from exc
    indices = [int(value) for value in contract.get("qstar", {}).get("timestep_indices", ())]
    event_id = str(row["event_id"])
    cells = report.get("cells")
    invalid = (
        report.get("schema_version") != 1
        or report.get("status") != "passed"
        or report.get("target_video_sha256") != row["qstar"]["video_sha256"]
        or report.get("prefix_sha256") != contract["snapshot"]["sha256"]
        or report.get("model_weights_changed") is not False
        or report.get("native_is_diagnostic") is not True
        or report.get("memory_regime") not in {"static_prefix", "dynamic_writer"}
        or not indices
        or not isinstance(cells, list)
        or len(cells) != len(indices)
        or len(records) != len(indices) * len(SEVEN_RUNS)
    )
    thresholds = report.get("thresholds", {})
    if not isinstance(thresholds, Mapping) or not all(
        _finite_number(thresholds.get(key))
        for key in (
            "repeat_loss_tolerance", "repeat_influence_tolerance",
            "benefit_margin", "influence_floor",
        )
    ):
        invalid = True
    by_cell = {
        int(cell.get("timestep_index", -1)): cell
        for cell in cells or ()
        if isinstance(cell, Mapping)
    }
    if set(by_cell) != set(indices):
        invalid = True
    for index in indices:
        cell = by_cell.get(index, {})
        input_hashes = cell.get("input_hashes", {})
        expected_identity = (
            str(cell.get("event_id")) == event_id
            and cell.get("memory_id") == f"{contract['event']['character_name']}|0"
            and int(cell.get("horizon", -1)) == int(contract["qstar"]["horizon"])
            and _finite_number(cell.get("timestep"))
        )
        hashes_valid = _valid_qstar_input_hashes(input_hashes, row, contract)
        losses = cell.get("losses", {})
        losses_valid = (
            isinstance(losses, Mapping)
            and set(losses) == set(SEVEN_RUNS)
            and all(_finite_number(value) for value in losses.values())
        )
        metrics_valid = all(
            _finite_number(cell.get(key))
            for key in (
                "qstar", "repeat_loss_floor", "repeat_prediction_floor",
                "primary_influence", "repeat_margin",
            )
        )
        if not (expected_identity and hashes_valid and losses_valid and metrics_valid):
            invalid = True
            continue
        deltas = qstar_deltas(losses)
        classification = classify_qstar(
            qstar=cell["qstar"],
            influence=cell["primary_influence"],
            repeat_margin=cell["repeat_margin"],
            influence_floor=thresholds["influence_floor"],
        )
        if (
            cell.get("qstar") != deltas["qstar"]
            or cell.get("repeat_loss_floor") != deltas["repeat_loss_floor"]
            or cell.get("arm_deltas") != deltas["arm_deltas"]
            or cell.get("classification") != classification
            or cell.get("benefit_margin_degenerate") is not (cell["repeat_margin"] <= 0)
        ):
            invalid = True
        cell_records = [record for record in records if record.get("timestep_index") == index]
        if {record.get("arm") for record in cell_records} != set(SEVEN_RUNS) or len(
            cell_records
        ) != len(SEVEN_RUNS):
            invalid = True
            continue
        records_by_arm = {str(record["arm"]): record for record in cell_records}
        for record in cell_records:
            arm = str(record.get("arm"))
            masked_loss = record.get("masked_loss")
            if (
                record.get("event_id") != event_id
                or record.get("memory_id") != cell.get("memory_id")
                or record.get("horizon") != cell.get("horizon")
                or record.get("timestep") != cell.get("timestep")
                or record.get("role") != ("diagnostic" if arm == "native" else "confirmatory")
                or not _finite_number(record.get("loss"))
                or record.get("loss") != losses[arm]
                or (masked_loss is not None and not _finite_number(masked_loss))
                or not _is_sha256(record.get("prediction_sha256"))
                or record.get("input_hashes") != input_hashes
                or not _finite_number(record.get("injection_delta_norm"))
                or not isinstance(record.get("memory_read_hit"), bool)
                or not isinstance(record.get("forced_memory_path"), bool)
                or not isinstance(record.get("diagnostics"), Mapping)
                or not _valid_qstar_arm_record(arm, record)
                or (
                    arm != "native"
                    and not _is_sha256(record.get("cfg_prediction_sha256"))
                )
            ):
                invalid = True
        correct_identity = tuple(
            records_by_arm["correct"].get(key)
            for key in ("payload_sha256", "payload_layers", "payload_slots")
        )
        repeat_identity = tuple(
            records_by_arm["correct_repeat"].get(key)
            for key in ("payload_sha256", "payload_layers", "payload_slots")
        )
        if correct_identity != repeat_identity:
            invalid = True
    if invalid:
        raise ValueError("Q* report does not match the frozen teacher/prefix")
    return report


def _finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _valid_qstar_input_hashes(
    value: object, row: Mapping, contract: Mapping
) -> bool:
    keys = {
        "prefix", "target_video", "target_latent", "noise",
        "noisy_latent", "flow_target", "prompt",
    }
    return (
        isinstance(value, Mapping)
        and set(value) == keys
        and all(_is_sha256(item) for item in value.values())
        and value["prefix"] == contract["snapshot"]["sha256"]
        and value["target_video"] == row["qstar"]["video_sha256"]
        and value["prompt"] == contract["runtime_contract"]["target_prompt_sha256"]
    )


def _valid_qstar_arm_record(arm: str, record: Mapping) -> bool:
    required = {
        "memory_read_hit", "forced_memory_path", "injection_delta_norm",
        "payload_sha256", "payload_layers", "payload_slots", "cfg_prediction_sha256",
    }
    if not required.issubset(record):
        return False
    injection = float(record["injection_delta_norm"])
    layers, slots = record["payload_layers"], record["payload_slots"]
    counts_valid = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (layers, slots)
    )
    if not counts_valid:
        return False
    if arm == "native":
        cfg = record["cfg_prediction_sha256"]
        return (
            record["memory_read_hit"] is False
            and record["forced_memory_path"] is False
            and injection == 0
            and record["payload_sha256"] is None
            and layers == slots == 0
            and (cfg is None or _is_sha256(cfg))
        )
    if arm == "no_memory":
        return (
            record["memory_read_hit"] is False
            and record["forced_memory_path"] is True
            and injection == 0
            and record["payload_sha256"] is None
            and layers == slots == 0
            and _is_sha256(record["cfg_prediction_sha256"])
        )
    return (
        record["memory_read_hit"] is True
        and record["forced_memory_path"] is True
        and injection > 0
        and _is_sha256(record["payload_sha256"])
        and layers > 0
        and slots > 0
        and _is_sha256(record["cfg_prediction_sha256"])
    )


def _command_artifact_payload(row: Mapping, contract: Mapping) -> dict:
    event = _read_json(Path(row["event_json"]))
    base = {
        "event": event,
        "event_json": Path(row["event_json"]),
        "subject_subspace_manifest": Path(row["subject_subspace_manifest"]),
        "target_seed": int(row["target_seed"]),
        "contract": contract,
        **(
            {
                "donor": row["donor"]["payload_path"],
                "donor_manifest": row["donor"]["manifest_path"],
            }
            if row.get("donor")
            else {}
        ),
    }
    python = row["commands"]["prefix"][0]
    phases = {}
    if row.get("donor"):
        phases = {
            "preflight": build_block_commands(
                {**base, "output": Path(row["block_dir"]) / "preflight"},
                python=python,
                arms=PREFLIGHT_ARMS,
            ),
            "full": build_block_commands(
                {**base, "output": Path(row["block_dir"]) / "full"}, python=python
            ),
        }
    qstar = None
    if row.get("qstar", {}).get("status") == "available" and row.get("donor"):
        qstar = [
            python, "-m", "utest.qstar_probe",
            "--prefix", str(Path(row["block_dir"]) / "prefix"),
            "--future-target-video", row["qstar"]["video_path"],
            "--output", str(Path(row["block_dir"]) / "qstar"),
            "--arms-root", str(Path(row["block_dir"]) / "full" / "arms"),
            "--donor", row["donor"]["payload_path"],
            "--donor-manifest", row["donor"]["manifest_path"],
            "--noise-seed", str(row["target_seed"]),
            "--timestep-indices", ",".join(
                str(value) for value in contract["qstar"]["timestep_indices"]
            ),
        ]
    artifact = {
        "schema_version": 1,
        "event_id": row["event_id"],
        "seed": int(row["seed"]),
        "snapshot": dict(contract["snapshot"]),
        "preflight": phases.get("preflight"),
        "full": phases.get("full"),
        "qstar": qstar,
    }
    artifact["artifact_sha256"] = canonical_json_sha256(artifact)
    return artifact


def _freeze_or_load_command_artifact(row: Mapping, contract: Mapping) -> dict:
    path = Path(row["command_artifact"])
    expected = _command_artifact_payload(row, contract)
    if path.is_file():
        actual = _read_json(path)
        if actual != expected or actual.get("artifact_sha256") != canonical_json_sha256(
            {key: value for key, value in actual.items() if key != "artifact_sha256"}
        ):
            raise ValueError("stage command artifact differs from the frozen prefix")
        return actual
    _write_json_exclusive(path, expected)
    return expected


def _recover_partial_prefix(row: Mapping) -> None:
    prefix = Path(row["block_dir"]) / "prefix"
    if prefix.exists():
        _archive_path(prefix)
    capture = Path(row["source_capture"])
    if capture.exists():
        _archive_path(capture)
    artifact = Path(row["command_artifact"])
    if artifact.exists():
        _archive_path(artifact)


def _resume_completed_arm(block: Mapping, arm: str, arm_dir: Path) -> bool:
    if not arm_dir.exists():
        return False
    try:
        validate_block(
            block,
            arms=(arm,),
            decoded_gate=False,
            require_measured=False,
        )
        return True
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        _archive_path(arm_dir)
        return False


def _execute_stage(
    manifest: Mapping,
    stage: str,
    *,
    event_id: str | None = None,
    seed: int | None = None,
    resume: bool = False,
) -> None:
    for row in _selected_blocks(manifest, event_id, seed):
        block_dir = Path(row["block_dir"])
        if stage == "prefix":
            completed = block_dir / "prefix" / "prefix_contract.json"
            if resume and completed.is_file():
                try:
                    contract = _validated_prefix_contract(row)
                    _freeze_or_load_command_artifact(row, contract)
                    continue
                except (FileNotFoundError, KeyError, TypeError, ValueError):
                    _recover_partial_prefix(row)
            elif resume and (block_dir / "prefix").exists():
                _recover_partial_prefix(row)
            _run_logged(
                row["commands"]["prefix"],
                Path(row["logs"]["prefix_stdout"]),
                Path(row["logs"]["prefix_stderr"]),
            )
            contract = _validated_prefix_contract(row)
            _freeze_or_load_command_artifact(row, contract)
            continue
        if stage == "probe":
            qualification = _read_json(Path(row["source_qualification"]))
            if qualification.get("status") != "passed":
                raise ValueError("source qualification is not passed")
            _ensure_semantic_scores(row)
            probe_command = _validate_probe_contract(row)
            completed = Path(row["subject_subspace_manifest"])
            if resume and completed.is_file():
                try:
                    event = _read_json(Path(row["event_json"]))
                    frozen = _read_json(completed)
                    validate_subject_subspace_manifest(frozen, event, seed=int(row["seed"]))
                    _load_source_slots({**row, "event": event}, frozen)
                    continue
                except (FileNotFoundError, KeyError, TypeError, ValueError):
                    _archive_path(completed)
            _run_logged(
                probe_command,
                Path(row["logs"]["probe_stdout"]),
                Path(row["logs"]["probe_stderr"]),
            )
            continue
        if stage == "qstar":
            if resume and row["commands"]["qstar"]["status"] == "not_available":
                continue
            if row["commands"]["qstar"]["status"] not in {"deferred_until_prefix"}:
                raise ValueError(f"qstar is {row['commands']['qstar']['status']}")
            contract = _validated_prefix_contract(row)
            artifact = _freeze_or_load_command_artifact(row, contract)
            if artifact["qstar"] is None:
                raise ValueError("qstar command is unavailable")
            report = block_dir / "qstar" / "qstar_report.json"
            if resume and report.parent.exists():
                try:
                    _validate_qstar_report(row, contract)
                    continue
                except (FileNotFoundError, KeyError, TypeError, ValueError):
                    _archive_path(report.parent)
            _run_logged(
                artifact["qstar"],
                block_dir / "qstar" / "stdout.log",
                block_dir / "qstar" / "stderr.log",
            )
            _validate_qstar_report(row, contract)
            continue
        phase = row["commands"][stage]
        if phase["status"] not in {"deferred_until_prefix"}:
            raise ValueError(f"{stage} is {phase['status']}")
        qualification = _read_json(Path(row["source_qualification"]))
        if qualification.get("status") != "passed":
            raise ValueError("source qualification is not passed")
        contract = _validated_prefix_contract(row)
        if stage == "preflight":
            _validate_donor_target_compatibility(row)
        elif stage == "full":
            preflight_validation = block_dir / "preflight" / "validation.json"
            if not preflight_validation.is_file():
                raise ValueError("full arms require a passed preflight")
            preflight_report = validate_block(
                {
                    "event": _read_json(Path(row["event_json"])),
                    "event_json": Path(row["event_json"]),
                    "subject_subspace_manifest": Path(row["subject_subspace_manifest"]),
                    "output": block_dir / "preflight",
                    "target_seed": int(row["target_seed"]),
                    "contract": contract,
                    "source_capture": row["source_capture"],
                    "donor": row.get("donor"),
                    "source_qualification": row["source_qualification"],
                    "arms_root": block_dir / "preflight" / "arms",
                },
                arms=PREFLIGHT_ARMS,
            )
            if _read_json(preflight_validation) != preflight_report:
                raise ValueError("full arms require an intact passed preflight")
        validation_path = block_dir / stage / "validation.json"
        snapshot = Path(contract["snapshot"]["path"])
        command_block = {
            "event": _read_json(Path(row["event_json"])),
            "event_json": Path(row["event_json"]),
            "subject_subspace_manifest": Path(row["subject_subspace_manifest"]),
            "output": block_dir / stage,
            "target_seed": int(row["target_seed"]),
            "contract": contract,
            **(
                {
                    "donor": row["donor"]["payload_path"],
                    "donor_manifest": row["donor"]["manifest_path"],
                }
                if row.get("donor")
                else {}
            ),
        }
        artifact = _freeze_or_load_command_artifact(row, contract)
        commands = artifact[stage]
        if commands is None:
            raise ValueError(f"{stage} commands are blocked")
        if resume and validation_path.is_file():
            try:
                validate_block(
                    {
                        **command_block,
                        "source_capture": row["source_capture"],
                        "donor": row.get("donor"),
                        "source_qualification": row["source_qualification"],
                        "arms_root": block_dir / stage / "arms",
                    },
                    arms=phase["arm_order"],
                )
                continue
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                _archive_path(validation_path)
        for arm, command in commands.items():
            arm_dir = block_dir / stage / "arms" / arm
            resume_block = {
                **command_block,
                "source_capture": row["source_capture"],
                "donor": row.get("donor"),
                "source_qualification": row["source_qualification"],
                "arms_root": block_dir / stage / "arms",
            }
            if resume and _resume_completed_arm(resume_block, arm, arm_dir):
                continue
            errors = validate_contract(contract, snapshot)
            if errors:
                raise ValueError(f"{arm}: " + ",".join(errors))
            _run_logged(command, arm_dir / "stdout.log", arm_dir / "stderr.log")
            errors = validate_contract(contract, snapshot)
            if errors:
                raise ValueError(f"{arm}: " + ",".join(errors))
        report = validate_block(
            {
                **command_block,
                "source_capture": row["source_capture"],
                "donor": row.get("donor"),
                "source_qualification": row["source_qualification"],
                "arms_root": block_dir / stage / "arms",
            },
            arms=phase["arm_order"],
        )
        _write_json_exclusive(validation_path, report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dry = sub.add_parser("dry-run")
    dry.add_argument("--inputs", type=Path, required=True)
    dry.add_argument("--output", type=Path, required=True)
    dry.add_argument("--base-inference-args", type=Path, required=True)
    dry.add_argument("--platform-manifest", type=Path, required=True)
    dry.add_argument("--donor-map", type=Path)
    dry.add_argument("--teacher-map", type=Path)
    dry.add_argument("--python", default=sys.executable)
    for name in ("prefix", "probe", "preflight", "full", "qstar", "resume"):
        stage = sub.add_parser(name)
        stage.add_argument("--manifest", type=Path, required=True)
        stage.add_argument("--event-id")
        stage.add_argument("--seed", type=int, choices=(0, 1, 2))
    args = parser.parse_args(argv)
    if args.command != "dry-run":
        manifest = _load_run_manifest(args.manifest.resolve())
        stages = ("prefix", "probe", "preflight", "full", "qstar") if args.command == "resume" else (args.command,)
        for stage in stages:
            _execute_stage(
                manifest,
                stage,
                event_id=args.event_id,
                seed=args.seed,
                resume=args.command == "resume",
            )
        return 0
    if (args.output / "run_manifest.json").exists():
        raise FileExistsError(args.output / "run_manifest.json")
    manifest = build_run_manifest(
        inputs=args.inputs.resolve(),
        output=args.output.resolve(),
        base_inference_args=args.base_inference_args.resolve(),
        platform_manifest=args.platform_manifest.resolve(),
        python=args.python,
        donor_map=args.donor_map.resolve() if args.donor_map else None,
        teacher_map=args.teacher_map.resolve() if args.teacher_map else None,
    )
    _write_json_exclusive(args.output / "run_manifest.json", manifest)
    print(json.dumps({"blocks": len(manifest["blocks"]), "manifest": str((args.output / "run_manifest.json").resolve())}))
    return 0


if __name__ == "__main__":
    main()
