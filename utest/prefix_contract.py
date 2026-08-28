"""Immutable pre-target snapshot contracts for fixed-prefix SlotMem experiments."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


RUNTIME_ONLY_ARGS = {
    "output_path",
    "resume_state_path",
    "save_state_path",
    "target_seed_override",
    "start_chunk_idx",
    "max_chunks",
    "efficiency_metrics_path",
    "efficiency_runtime_log",
    "merge_chunks",
    "merged_output_name",
    "subject_subspace_capture_path",
}
FROZEN_MEMORY_ENCODER_LAYERS: tuple[int, ...] = tuple(range(16))
FROZEN_MEMORY_ENCODER_SLOTS: int = 64
FROZEN_SUBJECT_SUBSPACE_BUDGET: int = 8
FROZEN_SUBJECT_SUBSPACE_FRACTION: float = 0.125


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bytes_no_clobber(path: Path, data: bytes) -> None:
    """Atomically publish bytes without replacing an existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_no_clobber(path: Path, value: object) -> None:
    """Atomically publish deterministic JSON without replacing an artifact."""
    write_bytes_no_clobber(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _arguments(argv: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    index = 0
    while index < len(argv):
        token = str(argv[index])
        if not token.startswith("--"):
            index += 1
            continue
        name = token[2:].replace("-", "_")
        if index + 1 < len(argv) and not str(argv[index + 1]).startswith("--"):
            parsed[name] = str(argv[index + 1])
            index += 2
        else:
            parsed[name] = "true"
            index += 1
    return parsed


def normalized_frozen_args(argv: Sequence[str]) -> dict[str, str]:
    frozen = {
        key: value
        for key, value in sorted(_arguments(argv).items())
        if key not in RUNTIME_ONLY_ARGS
    }
    frozen.setdefault("fixed_reference_scope", "all_chunks")
    return dict(sorted(frozen.items()))


def validate_slotmem_memory_encoder_geometry(
    frozen_args: Mapping[str, object],
) -> tuple[tuple[int, ...], int]:
    """Require the frozen 16-layer, 64-slot protocol without rewriting config."""
    raw_layers = frozen_args.get("slotmem_memory_encoder_layers")
    layers: list[int] = []
    if type(raw_layers) is str and raw_layers.strip():
        for part in raw_layers.split(","):
            bounds = [item.strip() for item in part.strip().split("-")]
            if len(bounds) not in {1, 2} or any(not item.isdigit() for item in bounds):
                layers = []
                break
            first, last = int(bounds[0]), int(bounds[-1])
            if last < first:
                layers = []
                break
            layers.extend(range(first, last + 1))
    if tuple(layers) != FROZEN_MEMORY_ENCODER_LAYERS or len(layers) != len(set(layers)):
        raise ValueError(
            "SlotMem donor protocol mismatch: --slotmem_memory_encoder_layers "
            f"actual={raw_layers!r}, frozen expected='0-15'; use a 64-slot-compatible "
            "checkpoint/config rather than changing an unproven checkpoint geometry"
        )

    raw_slots = frozen_args.get("slotmem_memory_encoder_slots")
    try:
        slots = int(raw_slots) if type(raw_slots) is str else -1
    except ValueError:
        slots = -1
    if slots != FROZEN_MEMORY_ENCODER_SLOTS:
        raise ValueError(
            "SlotMem donor protocol mismatch: --slotmem_memory_encoder_slots "
            f"actual={raw_slots!r}, frozen expected='64'; use a 64-slot-compatible "
            "checkpoint/config rather than changing an unproven checkpoint geometry"
        )
    return FROZEN_MEMORY_ENCODER_LAYERS, FROZEN_MEMORY_ENCODER_SLOTS


def _git_state(repo: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    )
    return commit, dirty


def build_runtime_contract(event: Mapping, inference_args: Sequence[str]) -> dict:
    """Resolve the contract fields from the arguments an arm will actually run."""
    parsed = _arguments(inference_args)
    source = Path(event.get("source_json_path") or parsed.get("json_path", "")).resolve()
    reference_text = event.get("reference_path") or parsed.get("ref_image_path", "")
    reference = Path(reference_text).resolve() if reference_text else None
    target_idx = int(event["target_chunk_idx"])
    story = json.loads(source.read_text(encoding="utf-8"))
    chunks = story.get("chunks", story) if isinstance(story, dict) else story
    target_prompt = str(
        chunks[target_idx].get("content") or chunks[target_idx].get("caption") or ""
    )
    return {
        "frozen_args": normalized_frozen_args(inference_args),
        "source_json_sha256": sha256_file(source),
        "target_prompt_sha256": hashlib.sha256(
            target_prompt.encode("utf-8")
        ).hexdigest(),
        "reference_sha256": (
            sha256_file(reference) if reference and reference.is_file() else None
        ),
        "target_seed": int(
            parsed.get(
                "target_seed_override",
                int(parsed.get("seed_base", 42)) + target_idx,
            )
        ),
    }


def build_contract(
    event: Mapping,
    snapshot: Path,
    inference_args: Sequence[str],
    platform_manifest: Path,
    *,
    arm_seed: int = 0,
    future_target_video: Path | None = None,
    future_target_manifest: Path | None = None,
    timestep_indices: Sequence[int] = (),
    arms_root: Path | None = None,
) -> dict:
    snapshot = snapshot.resolve()
    platform_manifest = platform_manifest.resolve()
    parsed = _arguments(inference_args)
    source = Path(event.get("source_json_path") or parsed.get("json_path", "")).resolve()
    reference_text = event.get("reference_path") or parsed.get("ref_image_path", "")
    reference = Path(reference_text).resolve() if reference_text else None
    target_idx = int(event["target_chunk_idx"])

    story = json.loads(source.read_text(encoding="utf-8"))
    chunks = story.get("chunks", story) if isinstance(story, dict) else story
    target_prompt = str(chunks[target_idx].get("content") or chunks[target_idx].get("caption") or "")
    repo = Path(__file__).resolve().parents[1]
    code_commit, code_dirty = _git_state(repo)

    inputs = {
        "source_json_path": str(source),
        "source_json_sha256": sha256_file(source),
        "target_prompt_sha256": hashlib.sha256(target_prompt.encode("utf-8")).hexdigest(),
        "target_prompt": target_prompt,
        "reference_path": str(reference) if reference else None,
        "reference_sha256": sha256_file(reference) if reference and reference.is_file() else None,
    }
    qstar = None
    if future_target_video is not None:
        future_target_video = future_target_video.resolve()
        if not future_target_video.is_file():
            raise FileNotFoundError(f"future target video not found: {future_target_video}")
        if future_target_manifest is None:
            raise ValueError("future target requires a provenance manifest")
        from .input_contract import validate_teacher_bundle

        # arms_root matters here: this is where the teacher is frozen into the contract,
        # so it is the point at which a target copied out of an arm rollout must be caught.
        teacher = validate_teacher_bundle(
            event, future_target_video, future_target_manifest, arms_root=arms_root
        )
        indices = [int(value) for value in timestep_indices]
        if not indices or any(value < 0 for value in indices) or len(indices) != len(set(indices)):
            raise ValueError("Q* timestep indices must be unique non-negative integers")
        source_idx = int(event.get("source_chunk_idx", 0))
        horizon = int(event.get("horizon", target_idx - source_idx))
        if horizon != target_idx - source_idx or horizon <= 0:
            raise ValueError("event horizon does not match source/target chunk indices")
        inputs.update(
            future_target_video_path=str(future_target_video),
            future_target_video_sha256=sha256_file(future_target_video),
            future_target_manifest_path=teacher["manifest_path"],
            future_target_manifest_sha256=teacher["manifest_sha256"],
            future_target_source_type=teacher["source_type"],
        )
        qstar = {
            "source_chunk_idx": source_idx,
            "target_chunk_idx": target_idx,
            "horizon": horizon,
            "timestep_indices": indices,
        }
    runtime_contract = build_runtime_contract(event, inference_args)
    if event.get("target_seed") is not None and int(event["target_seed"]) != int(
        runtime_contract["target_seed"]
    ):
        raise ValueError("event target_seed does not match actual target seed")
    contract = {
        "schema_version": 2 if qstar is not None else 1,
        "event": dict(event),
        "snapshot": {
            "path": str(snapshot),
            "bytes": snapshot.stat().st_size,
            "sha256": sha256_file(snapshot),
        },
        "platform_manifest": {
            "path": str(platform_manifest),
            "sha256": sha256_file(platform_manifest),
        },
        "code": {"commit": code_commit, "dirty": code_dirty},
        "inputs": inputs,
        "runtime_contract": runtime_contract,
        "base_inference_args": [str(value) for value in inference_args],
        "arm_seed": int(arm_seed),
    }
    if qstar is not None:
        contract["qstar"] = qstar
    return contract


def _frozen_args_for_comparison(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    normalized = dict(value)
    normalized.setdefault("fixed_reference_scope", "all_chunks")
    return normalized


def validate_contract(
    contract: Mapping, snapshot: Path, runtime: Mapping | None = None
) -> list[str]:
    errors: list[str] = []
    snapshot = snapshot.resolve()
    expected_snapshot = contract["snapshot"]
    if not snapshot.is_file() or sha256_file(snapshot) != expected_snapshot["sha256"]:
        errors.append("snapshot_sha256_mismatch")
    if runtime is None:
        return errors
    expected_runtime = contract["runtime_contract"]
    for key in (
        "frozen_args",
        "source_json_sha256",
        "target_prompt_sha256",
        "reference_sha256",
        "target_seed",
    ):
        actual = runtime.get(key)
        expected = expected_runtime.get(key)
        if key == "frozen_args":
            actual = _frozen_args_for_comparison(actual)
            expected = _frozen_args_for_comparison(expected)
        if actual != expected:
            errors.append(f"{key}_mismatch")
    return errors
