#!/usr/bin/env python3
"""Build the frozen derived eight-shot ViStoryBench CIDS adapter."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utest.subject_reappearance_harness import (  # noqa: E402
    EVALUATOR_COMMIT,
    FROZEN_EVENTS,
    FULL_ARMS,
    _load_run_manifest,
    _validated_prefix_contract,
    validate_block,
)

OFFICIAL_METHOD = "slotmem_subject_reappearance"
OFFICIAL_MODE = "derived_8frame"
OFFICIAL_LANGUAGE = "en"
OFFICIAL_TIMESTAMP = "20260827_000000"
OFFICIAL_CIDS_CONFIG = {
    "ref_mode": "origin",
    "use_multi_face_encoder": True,
    "ensemble_method": "average",
    "detection": {"dino": {"box_threshold": 0.25, "text_threshold": 0.25}},
    "encoders": {"clip": {"model_id": "openai/clip-vit-large-patch14"}},
    "matching": {"superfluous_threshold": 0.8, "topk_per_nochar": 5},
    "ensemble_weights": {"arcface": 0.4, "adaface": 0.4, "facenet": 0.2},
}
OFFICIAL_CONFIG_DOCUMENT = {
    "core": {"runtime": {"device": "cuda"}},
    "evaluators": {"cids": OFFICIAL_CIDS_CONFIG},
}


def _canonical_sha(value: Mapping) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


OFFICIAL_CIDS_CONTRACT_SHA256 = _canonical_sha(OFFICIAL_CIDS_CONFIG)


def _config_yaml_bytes() -> bytes:
    # Keep this dependency-free; the frozen evaluator already supplies PyYAML to read it.
    return (
        "core:\n"
        "  runtime:\n"
        "    device: cuda\n"
        "evaluators:\n"
        "  cids:\n"
        "    ref_mode: origin\n"
        "    use_multi_face_encoder: true\n"
        "    ensemble_method: average\n"
        "    detection:\n"
        "      dino:\n"
        "        box_threshold: 0.25\n"
        "        text_threshold: 0.25\n"
        "    encoders:\n"
        "      clip:\n"
        "        model_id: openai/clip-vit-large-patch14\n"
        "    matching:\n"
        "      superfluous_threshold: 0.8\n"
        "      topk_per_nochar: 5\n"
        "    ensemble_weights:\n"
        "      arcface: 0.4\n"
        "      adaface: 0.4\n"
        "      facenet: 0.2\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_hashed_json(path: Path) -> tuple[dict, str]:
    data = path.read_bytes()
    value = json.loads(data.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value, hashlib.sha256(data).hexdigest()


def _json_bytes(value: Mapping) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _safe_component(value: object, label: str) -> str:
    text = str(value)
    if not text or text in {".", ".."} or re.fullmatch(r"[A-Za-z0-9_.-]+", text) is None:
        raise ValueError(f"unsafe {label} path component: {text!r}")
    return text


def _inside(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("derived output escapes staging root")
    return path


def uniform_frame_indices(frame_count: int, count: int = 8) -> list[int]:
    """Integer-rounded linspace including both endpoints."""
    if count < 2 or frame_count < count:
        raise ValueError(f"need at least {count} distinct video frames")
    indices = [round(index * (frame_count - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise ValueError(f"video does not yield {count} distinct frame indices")
    return indices


def _export_video_frames(video: Path, destination: Path) -> list[dict]:
    import cv2

    video = video.resolve()
    before = _sha256(video)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video}")
    indices = uniform_frame_indices(int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        for shot_index, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"cannot read frame {frame_index} from {video}")
            output = destination / f"shot_{shot_index:02d}.png"
            if not cv2.imwrite(str(output), frame):
                raise ValueError(f"cannot write extracted frame: {output}")
            rows.append({
                "frame_index": frame_index,
                "derived_shot_index": shot_index,
                "path": str(output.resolve()),
                "sha256": _sha256(output),
                "source_video": str(video),
                "source_video_sha256": before,
            })
    finally:
        capture.release()
    if _sha256(video) != before:
        raise ValueError(f"video SHA-256 changed while decoding: {video}")
    return rows


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    if _sha256(source) != _sha256(destination):
        raise ValueError("reference bytes changed while materializing derived dataset")


def _official_story(event: Mapping) -> tuple[Path, dict, str]:
    reference = Path(str(event["reference_path"])).resolve()
    story_path = reference.parents[2] / "story.json"
    story, story_sha = _read_hashed_json(story_path)
    frozen_sha = str(FROZEN_EVENTS[str(event["event_id"])]["story_sha256"]).casefold()
    if story_sha != frozen_sha:
        raise ValueError("official story SHA-256 differs from frozen selection")
    return story_path, story, story_sha


def _character_key(story: Mapping, character_name: str) -> str:
    matches = []
    for key, row in story.get("Characters", {}).items():
        if not isinstance(row, Mapping):
            continue
        names = [str(row[field]) for field in ("name", "name_en") if field in row]
        if len(set(names)) > 1:
            raise ValueError(f"conflicting official character name aliases for {key}")
        if names and names[0] == character_name:
            matches.append(str(key))
    if len(matches) != 1:
        raise ValueError("official character name must resolve to exactly one character_key")
    return matches[0]


def _target_shot(story: Mapping, event_id: str) -> dict:
    index = int(FROZEN_EVENTS[event_id]["target_shot"])
    matches = [row for row in story.get("Shots", ()) if int(row.get("index", -1)) == index]
    if len(matches) != 1:
        raise ValueError("official target shot must resolve exactly once")
    return dict(matches[0])


def _validate_completed_block(row: Mapping) -> dict:
    event, event_sha = _read_hashed_json(Path(str(row["event_json"])))
    if event_sha != str(row["event_json_sha256"]).casefold():
        raise ValueError("block event JSON SHA-256 mismatch")
    contract = _validated_prefix_contract(row)
    block_dir = Path(str(row["block_dir"])).resolve()
    validation_path = block_dir / "full" / "validation.json"
    stored, validation_sha = _read_hashed_json(validation_path)
    checked = validate_block(
        {
            "event": event,
            "event_json": Path(str(row["event_json"])),
            "subject_subspace_manifest": Path(str(row["subject_subspace_manifest"])),
            "output": block_dir / "full",
            "target_seed": int(row["target_seed"]),
            "contract": contract,
            "source_capture": row["source_capture"],
            "donor": row.get("donor"),
            "source_qualification": row["source_qualification"],
            "arms_root": block_dir / "full" / "arms",
        },
        arms=FULL_ARMS,
    )
    if stored != checked:
        raise ValueError("stored full validation differs from recomputed Task6 validation")
    return {
        "event": event,
        "contract": contract,
        "validation_path": str(validation_path),
        "validation_sha256": validation_sha,
        "validation": checked,
    }


def _derived_story(official: Mapping, target_shot: Mapping, derived_story_id: str) -> dict:
    result = copy.deepcopy(dict(official))
    result["Shots"] = [{**copy.deepcopy(dict(target_shot)), "index": index} for index in range(8)]
    result["derived_story_id"] = derived_story_id
    result["derivation"] = "one frozen official target shot cloned over eight uniform video frames"
    return result


def _publish_lock(output: Path) -> tuple[int, Path]:
    lock = output.with_name(f".{output.name}.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(f"output publication is already locked: {lock}") from exc
    if output.exists():
        os.close(descriptor)
        lock.unlink()
        raise FileExistsError(output)
    return descriptor, lock


def _replace_root(value, old: Path, new: Path):
    if isinstance(value, dict):
        return {key: _replace_root(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_root(item, old, new) for item in value]
    if isinstance(value, str) and (value == str(old) or value.startswith(str(old) + os.sep)):
        return str(new) + value[len(str(old)):]
    return value


def _cleanup_partial_block(staging: Path, event_id: str, seed: int) -> None:
    roots = [_inside(staging, "source_frames", event_id, f"seed_{seed}")]
    for arm_index, _ in enumerate(FULL_ARMS):
        derived = str(900000 + list(FROZEN_EVENTS).index(event_id) * 100 + seed * 10 + arm_index)
        roots.extend([
            _inside(staging, "official_outputs", OFFICIAL_METHOD, OFFICIAL_MODE,
                    OFFICIAL_LANGUAGE, OFFICIAL_TIMESTAMP, derived),
            _inside(staging, "derived_dataset", "ViStory", derived),
        ])
    for root in roots:
        if root.exists():
            shutil.rmtree(root)


def _failure_row(
    base: Mapping, exc: Exception, *, stage: str, event_path: Path,
    event_sha: str, completed: Mapping | None = None,
) -> dict:
    provenance = {"event_json": {"path": str(event_path.resolve()), "sha256": event_sha}}
    if completed is not None:
        provenance["full_validation"] = {
            "path": str(completed["validation_path"]),
            "sha256": str(completed["validation_sha256"]),
        }
    return {
        **base,
        "status": "incomplete" if isinstance(exc, FileNotFoundError) else "failed",
        "reason": str(exc),
        "failure_stage": stage,
        "failure_type": type(exc).__name__,
        "failure_provenance": provenance,
        "failure_provenance_sha256": _canonical_sha(provenance),
    }


def _materialize_block(
    row: Mapping, event: Mapping, base: Mapping, staging: Path, completed: Mapping,
) -> dict:
    event_id, story_id, seed = str(base["event_id"]), str(base["story_id"]), int(base["seed"])
    story_path, official, story_sha = _official_story(event)
    character_key = _character_key(official, str(event["character_name"]))
    target_shot = _target_shot(official, event_id)
    source_video = (
        Path(str(row["block_dir"])) / "prefix" / "prefix_generation"
        / f"chunk_{int(event['source_chunk_idx']):03d}.mp4"
    )
    source_frames = _export_video_frames(
        source_video, _inside(staging, "source_frames", event_id, f"seed_{seed}", "shots", story_id)
    )
    source_frames = [
        {**frame, "event_id": event_id, "seed": seed, "arm": "shared_prefix"}
        for frame in source_frames
    ]
    arm_rows = {}
    for arm_index, arm in enumerate(FULL_ARMS):
        derived_story_id = str(900000 + list(FROZEN_EVENTS).index(event_id) * 100 + seed * 10 + arm_index)
        output_story = _inside(
            staging, "official_outputs", OFFICIAL_METHOD, OFFICIAL_MODE,
            OFFICIAL_LANGUAGE, OFFICIAL_TIMESTAMP, derived_story_id,
        )
        frames = _export_video_frames(
            Path(str(row["block_dir"])) / "full" / "arms" / arm
            / f"chunk_{int(event['target_chunk_idx']):03d}.mp4",
            _inside(output_story, "shots"),
        )
        frames = [{**frame, "event_id": event_id, "seed": seed, "arm": arm} for frame in frames]
        dataset_story = _inside(staging, "derived_dataset", "ViStory", derived_story_id)
        derived_story_path = _inside(dataset_story, "story.json")
        derived_story_path.parent.mkdir(parents=True, exist_ok=False)
        derived_story_path.write_bytes(_json_bytes(_derived_story(official, target_shot, derived_story_id)))
        references = []
        official_images = story_path.parent / "image"
        for official_key in official.get("Characters", {}):
            source_dir = official_images / str(official_key)
            if not source_dir.is_dir():
                continue
            for reference in sorted(path for path in source_dir.rglob("*") if path.is_file()):
                relative = reference.relative_to(source_dir)
                copied = _inside(dataset_story, "image", str(official_key), *relative.parts)
                _link_or_copy(reference, copied)
                references.append({
                    "character_key": str(official_key),
                    "official_path": str(reference.resolve()),
                    "derived_path": str(copied),
                    "sha256": _sha256(reference),
                })
        if not any(reference["character_key"] == character_key for reference in references):
            raise ValueError("official character reference folder is empty")
        arm_rows[arm] = {
            "derived_story_id": derived_story_id,
            "character_key": character_key,
            "original_story_id": story_id,
            "original_target_shot": int(FROZEN_EVENTS[event_id]["target_shot"]),
            "derived_story_json": str(derived_story_path),
            "derived_story_sha256": _sha256(derived_story_path),
            "official_output_story_dir": str(output_story),
            "reference_images": references,
            "target_frames": frames,
        }
    return {
        **base,
        "status": "passed",
        "event_json": str(Path(str(row["event_json"])).resolve()),
        "event_json_sha256": str(row["event_json_sha256"]),
        "official_story": {"path": str(story_path), "sha256": story_sha},
        "full_validation": completed["validation_path"],
        "full_validation_sha256": completed["validation_sha256"],
        "source_frames": source_frames,
        "arms": arm_rows,
    }


def prepare_cids_inputs(run_manifest: Path, output: Path) -> dict:
    """Validate nine blocks and atomically materialize eligible derived stories."""
    run_manifest = run_manifest.resolve()
    run = _load_run_manifest(run_manifest)
    if set(row["event_id"] for row in run["blocks"]) != set(FROZEN_EVENTS):
        raise ValueError("run event IDs differ from the frozen three-event selection")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, lock = _publish_lock(output)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)).resolve()
    try:
        config_path = _inside(staging, "config.yaml")
        config_path.write_bytes(_config_yaml_bytes())
        _inside(
            staging, "cids_results", OFFICIAL_METHOD, OFFICIAL_MODE,
            OFFICIAL_LANGUAGE, OFFICIAL_TIMESTAMP,
        ).mkdir(parents=True)
        blocks = []
        sorted_rows = sorted(
            run["blocks"], key=lambda row: (list(FROZEN_EVENTS).index(row["event_id"]), int(row["seed"]))
        )
        for row in sorted_rows:
            event_id = _safe_component(row["event_id"], "event_id")
            seed = int(row["seed"])
            event_path = Path(str(row["event_json"]))
            event, event_sha = _read_hashed_json(event_path)
            story_id = _safe_component(event["story_id"], "story_id")
            base = {
                "event_id": event_id,
                "story_id": story_id,
                "character_name": str(event["character_name"]),
                "seed": seed,
            }
            try:
                completed = _validate_completed_block(row)
            except Exception as exc:
                _cleanup_partial_block(staging, event_id, seed)
                blocks.append(_failure_row(
                    base, exc, stage="task6_validation", event_path=event_path,
                    event_sha=event_sha,
                ))
                continue
            try:
                blocks.append(_materialize_block(row, event, base, staging, completed))
            except Exception as exc:
                _cleanup_partial_block(staging, event_id, seed)
                blocks.append(_failure_row(
                    base, exc, stage="task7_materialization", event_path=event_path,
                    event_sha=event_sha, completed=completed,
                ))
        manifest = {
            "schema_version": 2,
            "task_id": "vistorybench_subject_reappearance_v1",
            "run_manifest": str(run_manifest),
            "run_manifest_sha256": _sha256(run_manifest),
            "evaluator_commit": EVALUATOR_COMMIT,
            "frame_count_per_video": 8,
            "protocol_boundary": {
                "identity_endpoint": "official_evaluator_derived",
                "identity_note": "derived eight-shot dataset; not the original ViStoryBench image-sequence protocol",
                "source_continuity": "local_diagnostic",
            },
            "official_cids": {
                "config": OFFICIAL_CIDS_CONFIG,
                "config_document": OFFICIAL_CONFIG_DOCUMENT,
                "config_file": {
                    "path": str(config_path),
                    "sha256": _sha256(config_path),
                },
                "detector_matcher_contract_sha256": OFFICIAL_CIDS_CONTRACT_SHA256,
                "method": OFFICIAL_METHOD,
                "mode": OFFICIAL_MODE,
                "language": OFFICIAL_LANGUAGE,
                "timestamp": OFFICIAL_TIMESTAMP,
                "dataset_path": str(_inside(staging, "derived_dataset")),
                "outputs_path": str(_inside(staging, "official_outputs")),
                "result_path": str(_inside(staging, "cids_results")),
                "items_relative_path": f"{OFFICIAL_METHOD}/{OFFICIAL_MODE}/en/{OFFICIAL_TIMESTAMP}/metrics/cids/items.jsonl",
                "cli_argv": [
                    "python", "bench_run.py", "--config", str(config_path),
                    "--method", OFFICIAL_METHOD,
                    "--mode", OFFICIAL_MODE, "--language", "en", "--split", "full",
                    "--metrics", "cids", "--timestamp", OFFICIAL_TIMESTAMP,
                    "--dataset_path", str(_inside(staging, "derived_dataset")),
                    "--outputs_path", str(_inside(staging, "official_outputs")),
                    "--result_path", str(_inside(staging, "cids_results")), "--resume",
                ],
            },
            "blocks": blocks,
        }
        manifest = _replace_root(manifest, staging, output)
        (staging / "cids_input_manifest.json").write_bytes(_json_bytes(manifest))
        os.rename(staging, output)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        os.close(descriptor)
        if lock.exists():
            lock.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare_cids_inputs(args.run_manifest, args.output)
    print(json.dumps({
        "manifest": str((args.output / "cids_input_manifest.json").resolve()),
        "passed": sum(row["status"] == "passed" for row in result["blocks"]),
        "ineligible": sum(row["status"] != "passed" for row in result["blocks"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
