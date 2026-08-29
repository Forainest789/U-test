"""Deterministic ViStoryBench-to-SlotMem event conversion."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from .prefix_contract import sha256_file


FROZEN_SELECTION_PATH = (
    Path(__file__).parent / "events" / "vistorybench_reappearance_v1.json"
)
VISTORY_DATASET_COMMIT = "92f845531b67e97a67ae04b256ec5d8c020e8341"
VISTORY_EVALUATOR_COMMIT = "b44ec9108668cc2bcc8c5280886b235e9fb8bea9"
VISTORY_TASK_ID = "vistorybench_subject_reappearance_v1"
_EVENT_FIELDS = {
    "story_id",
    "event_id",
    "character_name",
    "source_shot",
    "target_shot",
    "story_sha256",
}
_REPLACEMENT_FIELDS = {
    "original_event_id",
    "selected_event_id",
    "selected_candidate_id",
    "eligible_donor_count",
    "horizon",
    "horizon_distance",
    "survey_sha256",
    "review_sha256",
    "reviewer",
    "dataset_commit",
}
_RETAINED_EVENTS = {
    0: {
        "story_id": "79",
        "event_id": "vistory79_song_yuchen_s2_s8",
        "character_name": "Song Yuchen",
        "source_shot": 2,
        "target_shot": 8,
        "story_sha256": "4298F6EFAA5F2D4A9D69C86E169E0167CE324334F656033A6D692CAFD9484109",
    },
    2: {
        "story_id": "16",
        "event_id": "vistory16_chen_father_s1_s10",
        "character_name": "Chen Sihan's Father",
        "source_shot": 1,
        "target_shot": 10,
        "story_sha256": "6B1AD31634E5DA0108ACD51B16DA2E7F29B202858FCC5D0E556F4BEDB22D005E",
    },
}
_ORIGINAL_IDENTITIES = {
    ("79", "Song Yuchen", "vistory79_song_yuchen_s2_s8"),
    ("15", "Gu Zhenzhen", "vistory15_gu_zhenzhen_s8_s20"),
    ("16", "Chen Sihan's Father", "vistory16_chen_father_s1_s10"),
}


def _require_exact_fields(value: Mapping, expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{label} schema mismatch: missing={missing}, extra={extra}")


def _require_sha256(value: object, label: str, *, uppercase: bool) -> str:
    pattern = r"[0-9A-F]{64}" if uppercase else r"[0-9a-f]{64}"
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        case = "uppercase" if uppercase else "lowercase"
        raise ValueError(f"{label} must be 64 {case} hexadecimal digits")
    return value


def load_frozen_selection(path: Path | None = None) -> dict:
    """Load and fail closed on the checked-in three-event authority."""
    selection_path = FROZEN_SELECTION_PATH if path is None else Path(path)
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid frozen target selection: {selection_path}: {error}") from error
    if not isinstance(selection, dict):
        raise ValueError("frozen target selection must be a JSON object")
    _require_exact_fields(
        selection,
        {
            "schema_version",
            "task_id",
            "dataset_commit",
            "evaluator_commit",
            "seeds",
            "events",
            "replacement_selection",
        },
        "frozen target selection",
    )
    if type(selection["schema_version"]) is not int or selection["schema_version"] != 1:
        raise ValueError("frozen target selection schema_version must be integer 1")
    if selection["task_id"] != VISTORY_TASK_ID:
        raise ValueError("frozen target selection task_id mismatch")
    if selection["dataset_commit"] != VISTORY_DATASET_COMMIT:
        raise ValueError("frozen target selection dataset_commit mismatch")
    if selection["evaluator_commit"] != VISTORY_EVALUATOR_COMMIT:
        raise ValueError("frozen target selection evaluator_commit mismatch")
    if selection["seeds"] != [0, 1, 2] or any(
        type(seed) is not int for seed in selection["seeds"]
    ):
        raise ValueError("frozen target selection seeds must be integer [0, 1, 2]")

    events = selection["events"]
    if not isinstance(events, list) or len(events) != 3:
        raise ValueError("frozen target selection must contain exactly three events")
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError(f"frozen event {index} must be a JSON object")
        _require_exact_fields(event, _EVENT_FIELDS, f"frozen event {index}")
        if not all(isinstance(event[field], str) and event[field] for field in (
            "story_id", "event_id", "character_name"
        )):
            raise ValueError(f"frozen event {index} identity fields must be non-empty strings")
        if type(event["source_shot"]) is not int or type(event["target_shot"]) is not int:
            raise ValueError(f"frozen event {index} shots must be integers")
        if event["target_shot"] - event["source_shot"] < 2:
            raise ValueError(f"frozen event {index} must have a nonempty absence interval")
        _require_sha256(event["story_sha256"], f"frozen event {index} story_sha256", uppercase=True)
    for index, expected in _RETAINED_EVENTS.items():
        if events[index] != expected:
            raise ValueError(f"retained frozen event changed at position {index}")
    replacement_identity = (events[1]["story_id"], events[1]["character_name"])
    original_entity_uids = {(story_id, name) for story_id, name, _ in _ORIGINAL_IDENTITIES}
    original_event_ids = {event_id for _, _, event_id in _ORIGINAL_IDENTITIES}
    if (
        replacement_identity in original_entity_uids
        or events[1]["event_id"] in original_event_ids
    ):
        raise ValueError("replacement must exclude all three original target identities")
    event_ids = [event["event_id"] for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("frozen target event_ids must be unique")

    replacement = selection["replacement_selection"]
    if not isinstance(replacement, Mapping):
        raise ValueError("replacement_selection must be a JSON object")
    _require_exact_fields(replacement, _REPLACEMENT_FIELDS, "replacement_selection")
    if replacement["original_event_id"] != "vistory15_gu_zhenzhen_s8_s20":
        raise ValueError("replacement_selection original_event_id mismatch")
    if replacement["selected_event_id"] != events[1]["event_id"]:
        raise ValueError("replacement_selection selected event mismatch")
    if replacement["dataset_commit"] != VISTORY_DATASET_COMMIT:
        raise ValueError("replacement_selection dataset_commit mismatch")
    if not isinstance(replacement["reviewer"], str) or not replacement["reviewer"].strip():
        raise ValueError("replacement_selection reviewer must be non-empty")
    for field in ("selected_candidate_id", "survey_sha256", "review_sha256"):
        _require_sha256(replacement[field], f"replacement_selection {field}", uppercase=False)
    for field in ("eligible_donor_count", "horizon", "horizon_distance"):
        if type(replacement[field]) is not int:
            raise ValueError(f"replacement_selection {field} must be an integer")
    if replacement["eligible_donor_count"] <= 0:
        raise ValueError("replacement_selection must have an eligible donor")
    horizon = events[1]["target_shot"] - events[1]["source_shot"]
    if replacement["horizon"] != horizon or replacement["horizon_distance"] != abs(horizon - 12):
        raise ValueError("replacement_selection horizon binding mismatch")
    return selection


def frozen_target_event_ids(path: Path | None = None) -> frozenset[str]:
    """Return target IDs derived from the validated frozen authority."""
    return frozenset(event["event_id"] for event in load_frozen_selection(path)["events"])


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
