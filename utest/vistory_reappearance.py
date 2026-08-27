"""Deterministic ViStoryBench-to-SlotMem event conversion."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from .prefix_contract import sha256_file


def convert_event(official: Mapping, spec: Mapping) -> tuple[dict, dict]:
    """Slice one frozen source-to-first-reappearance interval."""
    subject = str(spec["character_name"])
    source = int(spec["source_shot"])
    target = int(spec["target_shot"])
    if target - source < 2:
        raise ValueError("source and target must contain a nonempty absence interval")
    by_index = {int(row["index"]): row for row in official["Shots"]}
    selected = [by_index[index] for index in range(source, target + 1)]
    if subject not in selected[0]["Characters Appearing"]["en"]:
        raise ValueError("subject absent from source")
    if subject not in selected[-1]["Characters Appearing"]["en"]:
        raise ValueError("subject absent from first reappearance")
    if any(
        subject in row["Characters Appearing"]["en"] for row in selected[1:-1]
    ):
        raise ValueError("full absence interval contains subject")

    chunks = [
        {
            "chunk_idx": chunk_idx,
            "official_shot_idx": int(row["index"]),
            "content": (
                f'{row["Setting Description"]["en"]} '
                f'{row["Shot Perspective Design"]["en"]}. '
                f'{row["Static Shot Description"]["en"]}'
            ),
            "character_list": list(row["Characters Appearing"]["en"]),
        }
        for chunk_idx, row in enumerate(selected)
    ]
    derived = {
        "schema_version": 1,
        "story_id": str(spec["story_id"]),
        "characters": {
            name: row["prompt_en"] for name, row in official["Characters"].items()
        },
        "chunks": chunks,
    }
    event = {
        "schema_version": 1,
        "story_id": str(spec["story_id"]),
        "event_id": str(spec["event_id"]),
        "entity_uid": f'{spec["story_id"]}::{subject}',
        "character_name": subject,
        "source_chunk_idx": 0,
        "target_chunk_idx": len(chunks) - 1,
        "horizon": len(chunks) - 1,
        "target_seed": None,
        "max_memory_characters": 4,
    }
    return derived, event


def prepare_dataset(
    data_root: Path, output_root: Path, selection: Mapping
) -> dict:
    """Validate and convert the selected official stories."""
    data_root = Path(data_root).resolve()
    output_root = Path(output_root).resolve()
    if int(selection.get("schema_version", -1)) != 1:
        raise ValueError("selection schema_version must be 1")
    if list(selection.get("seeds", [])) != [0, 1, 2]:
        raise ValueError("selection seeds must be exactly [0, 1, 2]")

    prepared = []
    seen_event_ids: set[str] = set()
    for spec in selection["events"]:
        event_id = str(spec["event_id"])
        if event_id in seen_event_ids:
            raise ValueError(f"duplicate event_id: {event_id}")
        seen_event_ids.add(event_id)

        expected_sha = str(spec["story_sha256"])
        if re.fullmatch(r"[0-9A-Fa-f]{64}", expected_sha) is None:
            raise ValueError("story_sha256 must contain exactly 64 hexadecimal digits")
        story_path = data_root / str(spec["story_id"]) / "story.json"
        actual_sha = sha256_file(story_path)
        if actual_sha.lower() != expected_sha.lower():
            raise ValueError(
                f"story SHA-256 mismatch for {story_path}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        official = json.loads(story_path.read_text(encoding="utf-8-sig"))
        derived, event = convert_event(official, spec)

        reference_dir = (
            data_root
            / str(spec["story_id"])
            / "image"
            / str(spec["character_name"])
        )
        reference = reference_dir / "00.jpg"
        if not reference.is_file():
            raise FileNotFoundError(f"official reference image missing: {reference}")
        reference_images = [path for path in sorted(reference_dir.iterdir()) if path.is_file()]
        reference_records = [
            {
                "path": path.relative_to(data_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in reference_images
        ]
        event_root = output_root / event_id
        story_output = event_root / "story.json"
        event_output = event_root / "event.json"
        event.update(
            {
                "source_json_path": str(story_output),
                "reference_path": str(reference),
                "reference_sha256": sha256_file(reference),
            }
        )
        prepared.append(
            {
                "spec": dict(spec),
                "derived": derived,
                "event": event,
                "event_root": event_root,
                "story_path": story_path,
                "story_sha256": actual_sha,
                "reference": reference,
                "reference_records": reference_records,
            }
        )

    report = {
        "schema_version": 1,
        "task_id": str(selection["task_id"]),
        "dataset_commit": str(selection["dataset_commit"]),
        "evaluator_commit": str(selection["evaluator_commit"]),
        "seeds": [0, 1, 2],
        "events": [],
    }
    for item in prepared:
        event_root = item["event_root"]
        event_root.mkdir(parents=True, exist_ok=True)
        story_output = event_root / "story.json"
        event_output = event_root / "event.json"
        manifest_output = event_root / "manifest.json"
        _write_json(story_output, item["derived"])
        _write_json(event_output, item["event"])

        spec = item["spec"]
        manifest = {
            "schema_version": 1,
            "task_id": report["task_id"],
            "dataset_commit": report["dataset_commit"],
            "evaluator_commit": report["evaluator_commit"],
            "seeds": report["seeds"],
            "event_id": str(spec["event_id"]),
            "story_id": str(spec["story_id"]),
            "character_name": str(spec["character_name"]),
            "source_shot": int(spec["source_shot"]),
            "target_shot": int(spec["target_shot"]),
            "official_story": {
                "path": item["story_path"].relative_to(data_root).as_posix(),
                "sha256": item["story_sha256"],
            },
            "reference_path": item["reference"].relative_to(data_root).as_posix(),
            "reference_sha256": sha256_file(item["reference"]),
            "reference_images": item["reference_records"],
            "chunk_mapping": [
                {
                    "chunk_idx": chunk["chunk_idx"],
                    "official_shot_idx": chunk["official_shot_idx"],
                }
                for chunk in item["derived"]["chunks"]
            ],
            "field_sources": {
                "characters": "Characters[*].prompt_en",
                "character_list": "Shots[*].Characters Appearing.en",
                "content": [
                    "Shots[*].Setting Description.en",
                    "Shots[*].Shot Perspective Design.en",
                    "Shots[*].Static Shot Description.en",
                ],
            },
            "outputs": {
                "story": {
                    "path": story_output.relative_to(output_root).as_posix(),
                    "sha256": sha256_file(story_output),
                },
                "event": {
                    "path": event_output.relative_to(output_root).as_posix(),
                    "sha256": sha256_file(event_output),
                },
            },
        }
        _write_json(manifest_output, manifest)
        report["events"].append(
            {
                "event_id": str(spec["event_id"]),
                "manifest_path": manifest_output.relative_to(output_root).as_posix(),
                "manifest_sha256": sha256_file(manifest_output),
            }
        )
    _write_json(output_root / "manifest.json", report)
    return report


def _write_json(path: Path, payload: Mapping) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
