"""Reviewed, deterministic replacement selection for ViStoryBench targets."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path

from .prefix_contract import sha256_file, write_json_no_clobber
from .vistory_donors import (
    VISTORY_DATASET_COMMIT,
    donor_rejection_reasons,
    enumerate_official_recurrences,
    validate_frozen_vistory_tree,
)
from .vistory_reappearance import convert_event


EVALUATOR_COMMIT = "b44ec9108668cc2bcc8c5280886b235e9fb8bea9"
TASK_ID = "vistorybench_subject_reappearance_v1"
ORIGINAL_EVENT_ID = "vistory15_gu_zhenzhen_s8_s20"
ORIGINAL_EVENTS = (
    ("vistory79_song_yuchen_s2_s8", "79", "Song Yuchen", 2, 8),
    (ORIGINAL_EVENT_ID, "15", "Gu Zhenzhen", 8, 20),
    ("vistory16_chen_father_s1_s10", "16", "Chen Sihan's Father", 1, 10),
)
SURVEY_FIELDS = {
    "schema_version",
    "dataset_commit",
    "selection_sha256",
    "original_event_id",
    "excluded_entity_uids",
    "candidates",
}
CANDIDATE_FIELDS = {
    "candidate_id",
    "event_id",
    "story_id",
    "character_name",
    "entity_uid",
    "source_shot",
    "target_shot",
    "horizon",
    "gap_bucket",
    "official_tag",
    "style_class",
    "source_character_count",
    "official_character_description",
    "official_story",
    "reference",
    "source_prompt",
    "read_prompt",
    "eligible_donor_count",
    "eligible_donor_ids",
}
FROZEN_SURVEY_SHA256 = "af44edabe4d70845869ca49fefb18a1c9199c37dd500cf276c37a9cc3c166562"
FROZEN_REVIEW_SHA256 = "32ded0398440d4b582ecce4c126a6bea8b838a3376d76ddeaaf0ea9bbaecba65"


def _read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_sha256(value: object) -> str:
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_exact_fields(value: Mapping, expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{label} schema mismatch: missing={missing}, extra={extra}")


def _require_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return value


def _slug(character_name: str) -> str:
    ascii_name = (
        unicodedata.normalize("NFKD", character_name)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_name).strip("_")
    if not slug:
        raise ValueError(f"empty stable slug for character: {character_name!r}")
    return slug


def _load_original_selection(data_root: Path, selection_path: Path) -> dict:
    selection = _read_json(selection_path, "frozen target selection")
    _require_exact_fields(
        selection,
        {"schema_version", "task_id", "dataset_commit", "evaluator_commit", "seeds", "events"},
        "frozen target selection",
    )
    if _require_int(selection["schema_version"], "selection schema_version") != 1:
        raise ValueError("selection schema_version must be integer 1")
    if selection["dataset_commit"] != VISTORY_DATASET_COMMIT:
        raise ValueError("selection dataset_commit does not match the pinned revision")
    if selection["task_id"] != TASK_ID:
        raise ValueError("selection task_id does not match the frozen task")
    if selection["evaluator_commit"] != EVALUATOR_COMMIT:
        raise ValueError("selection evaluator_commit does not match the frozen evaluator")
    if selection["seeds"] != [0, 1, 2] or any(
        type(seed) is not int for seed in selection["seeds"]
    ):
        raise ValueError("selection seeds must be exactly integer [0, 1, 2]")
    events = selection["events"]
    if not isinstance(events, list) or len(events) != 3:
        raise ValueError("selection must contain exactly three original events")
    for event, expected in zip(events, ORIGINAL_EVENTS):
        if not isinstance(event, Mapping):
            raise ValueError("selection event must be a JSON object")
        _require_exact_fields(
            event,
            {
                "story_id",
                "event_id",
                "character_name",
                "source_shot",
                "target_shot",
                "story_sha256",
            },
            "selection event",
        )
        event_id, story_id, character_name, source_shot, target_shot = expected
        actual = (
            event["event_id"],
            event["story_id"],
            event["character_name"],
            event["source_shot"],
            event["target_shot"],
        )
        if actual != expected:
            raise ValueError(f"original target event changed: expected={expected}, actual={actual}")
        if type(event["source_shot"]) is not int or type(event["target_shot"]) is not int:
            raise ValueError(f"selection event shots must be integers: {event_id}")
        expected_story_sha = str(event["story_sha256"])
        if re.fullmatch(r"[0-9A-Fa-f]{64}", expected_story_sha) is None:
            raise ValueError(f"selection story_sha256 is invalid: {event_id}")
        story_path = (data_root / story_id / "story.json").resolve()
        if not story_path.is_relative_to(data_root):
            raise ValueError(f"selection story path escapes data root: {story_id}")
        if sha256_file(story_path).lower() != expected_story_sha.lower():
            raise ValueError(f"selection story SHA-256 mismatch: {event_id}")
        official = _read_json(story_path, f"official story {story_id}")
        names = [str(name).strip().casefold() for name in official["Characters"]]
        if names.count(character_name.strip().casefold()) != 1:
            raise ValueError(f"selection identity is missing or ambiguous: {event_id}")
        convert_event(official, event)
        reference = (data_root / story_id / "image" / character_name / "00.jpg").resolve()
        if not reference.is_relative_to(data_root) or not reference.is_file():
            raise ValueError(f"selection reference is missing or escapes data root: {event_id}")
    return selection


def _build_replacement_target_survey(
    *, data_root: Path, selection_path: Path
) -> dict:
    data_root = Path(data_root).resolve()
    selection_path = Path(selection_path).resolve()
    validate_frozen_vistory_tree(data_root)
    selection = _load_original_selection(data_root, selection_path)
    recurrences = enumerate_official_recurrences(data_root)
    excluded = {f"{story_id}::{name}" for _, story_id, name, _, _ in ORIGINAL_EVENTS}
    event_ids = {
        str(selection["events"][0]["event_id"]),
        str(selection["events"][2]["event_id"]),
    }
    candidates: list[dict] = []
    for recurrence in recurrences:
        if recurrence["entity_uid"] in excluded:
            continue
        if (
            recurrence["official_tag"] != "realistic_human"
            or recurrence["style_class"] != "realistic"
            or recurrence["gap_bucket"] is None
            or recurrence["ambiguous_name"]
            or recurrence["ambiguous_presence"]
            or not recurrence["interval_valid"]
            or recurrence["reference"]["sha256"] is None
        ):
            continue
        event_id = (
            f'vistory{recurrence["story_id"]}_{_slug(recurrence["character_name"])}'
            f'_s{recurrence["source_shot"]}_s{recurrence["read_shot"]}'
        )
        if event_id in event_ids:
            raise ValueError(f"colliding stable replacement event_id: {event_id}")
        event_ids.add(event_id)
        # ponytail: the benchmark is frozen at 80 stories; index matcher keys if expanded.
        eligible = [
            donor
            for donor in recurrences
            if not donor_rejection_reasons(recurrence, donor)
        ]
        if not eligible:
            continue
        identity = {
            "dataset_commit": VISTORY_DATASET_COMMIT,
            "event_id": event_id,
            "entity_uid": recurrence["entity_uid"],
            "source_shot": recurrence["source_shot"],
            "target_shot": recurrence["read_shot"],
            "official_story_sha256": recurrence["official_story"]["sha256"],
            "reference_sha256": recurrence["reference"]["sha256"],
        }
        candidates.append(
            {
                "candidate_id": _canonical_sha256(identity),
                "event_id": event_id,
                "story_id": recurrence["story_id"],
                "character_name": recurrence["character_name"],
                "entity_uid": recurrence["entity_uid"],
                "source_shot": recurrence["source_shot"],
                "target_shot": recurrence["read_shot"],
                "horizon": recurrence["horizon"],
                "gap_bucket": recurrence["gap_bucket"],
                "official_tag": recurrence["official_tag"],
                "style_class": recurrence["style_class"],
                "source_character_count": recurrence["source_character_count"],
                "official_character_description": recurrence[
                    "official_character_description"
                ],
                "official_story": recurrence["official_story"],
                "reference": recurrence["reference"],
                "source_prompt": recurrence["source_prompt"],
                "read_prompt": recurrence["read_prompt"],
                "eligible_donor_count": len(eligible),
                "eligible_donor_ids": sorted(donor["recurrence_id"] for donor in eligible),
            }
        )
    return {
        "schema_version": 1,
        "dataset_commit": selection["dataset_commit"],
        "selection_sha256": sha256_file(selection_path),
        "original_event_id": ORIGINAL_EVENT_ID,
        "excluded_entity_uids": sorted(excluded),
        "candidates": sorted(candidates, key=lambda row: row["candidate_id"]),
    }


def build_replacement_target_survey(
    *, data_root: Path, selection_path: Path, output_path: Path
) -> dict:
    survey = _build_replacement_target_survey(
        data_root=Path(data_root), selection_path=Path(selection_path)
    )
    write_json_no_clobber(Path(output_path).resolve(), survey)
    return survey


def _validate_survey(survey: Mapping) -> list[dict]:
    _require_exact_fields(survey, SURVEY_FIELDS, "replacement survey")
    if _require_int(survey["schema_version"], "survey schema_version") != 1:
        raise ValueError("survey schema_version must be integer 1")
    if survey["dataset_commit"] != VISTORY_DATASET_COMMIT:
        raise ValueError("survey dataset_commit does not match the pinned revision")
    _require_sha256(survey["selection_sha256"], "survey selection_sha256")
    if survey["original_event_id"] != ORIGINAL_EVENT_ID:
        raise ValueError("survey original_event_id mismatch")
    expected_excluded = sorted(
        f"{story_id}::{name}" for _, story_id, name, _, _ in ORIGINAL_EVENTS
    )
    if survey["excluded_entity_uids"] != expected_excluded:
        raise ValueError("survey excluded identities mismatch")
    if not isinstance(survey["candidates"], list):
        raise ValueError("survey candidates must be a list")
    candidates: list[dict] = []
    seen: set[str] = set()
    for raw in survey["candidates"]:
        if not isinstance(raw, Mapping):
            raise ValueError("survey candidate must be a JSON object")
        _require_exact_fields(raw, CANDIDATE_FIELDS, "survey candidate")
        row = dict(raw)
        candidate_id = _require_sha256(row["candidate_id"], "candidate_id")
        if candidate_id in seen:
            raise ValueError(f"duplicate survey candidate_id: {candidate_id}")
        seen.add(candidate_id)
        if row["entity_uid"] in expected_excluded or row["event_id"] == ORIGINAL_EVENT_ID:
            raise ValueError(
                f"survey candidate uses an original target identity: {row['entity_uid']}"
            )
        for field in ("source_shot", "target_shot", "horizon", "source_character_count", "eligible_donor_count"):
            _require_int(row[field], f"candidate {field}")
        if row["eligible_donor_count"] <= 0:
            raise ValueError(f"candidate has zero eligible donors: {candidate_id}")
        if not isinstance(row["eligible_donor_ids"], list) or any(
            not isinstance(value, str) for value in row["eligible_donor_ids"]
        ):
            raise ValueError("candidate eligible_donor_ids must be a list of strings")
        if len(row["eligible_donor_ids"]) != row["eligible_donor_count"]:
            raise ValueError(f"candidate donor count mismatch: {candidate_id}")
        candidates.append(row)
    if [row["candidate_id"] for row in candidates] != sorted(seen):
        raise ValueError("survey candidates are not sorted by candidate_id")
    return candidates


def write_replacement_review_template(
    *, survey_path: Path, output_path: Path
) -> dict:
    survey_path = Path(survey_path).resolve()
    survey = _read_json(survey_path, "replacement survey")
    candidates = _validate_survey(survey)
    template = {
        "schema_version": 1,
        "dataset_commit": survey["dataset_commit"],
        "survey_sha256": sha256_file(survey_path),
        "reviewer": "",
        "candidates": [
            {"candidate_id": row["candidate_id"], "female_character": None}
            for row in candidates
        ],
    }
    write_json_no_clobber(Path(output_path).resolve(), template)
    return template


def _validated_review_payload(
    review: Mapping, survey_sha256: str, candidates: list[dict]
) -> str:
    _require_exact_fields(
        review,
        {"schema_version", "dataset_commit", "survey_sha256", "reviewer", "candidates"},
        "replacement review",
    )
    if _require_int(review["schema_version"], "review schema_version") != 1:
        raise ValueError("review schema_version must be integer 1")
    if review["dataset_commit"] != VISTORY_DATASET_COMMIT:
        raise ValueError("review dataset_commit does not match the pinned revision")
    survey_sha = _require_sha256(review["survey_sha256"], "review survey_sha256")
    if survey_sha != survey_sha256:
        raise ValueError("review survey_sha256 is stale")
    reviewer = review["reviewer"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("reviewer must be a non-empty human name")
    rows = review["candidates"]
    if not isinstance(rows, list):
        raise ValueError("review candidates must be a list")
    expected = {row["candidate_id"] for row in candidates}
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("review candidate must be a JSON object")
        _require_exact_fields(raw, {"candidate_id", "female_character"}, "review candidate")
        candidate_id = _require_sha256(raw["candidate_id"], "review candidate_id")
        if candidate_id in seen:
            raise ValueError(f"duplicate review candidate_id: {candidate_id}")
        if candidate_id not in expected:
            raise ValueError(f"stale or unknown review candidate_id: {candidate_id}")
        if type(raw["female_character"]) is not bool:
            raise ValueError(f"female_character must be boolean: {candidate_id}")
        seen.add(candidate_id)
    missing = sorted(expected - seen)
    if missing:
        raise ValueError(f"missing review candidate_ids: {missing}")
    return reviewer.strip()


def _validated_review(review: Mapping, survey_path: Path, candidates: list[dict]) -> str:
    return _validated_review_payload(review, sha256_file(survey_path), candidates)


def validate_frozen_replacement_provenance(
    selection: Mapping, survey: Mapping, review: Mapping
) -> None:
    """Validate the reviewed evidence and its exact binding to the frozen replacement."""
    candidates = _validate_survey(survey)
    survey_sha = _artifact_sha256(survey)
    review_sha = _artifact_sha256(review)
    if survey_sha != FROZEN_SURVEY_SHA256:
        raise ValueError("frozen replacement survey SHA-256 mismatch")
    if review_sha != FROZEN_REVIEW_SHA256:
        raise ValueError("frozen replacement review SHA-256 mismatch")
    reviewer = _validated_review_payload(review, survey_sha, candidates)
    if len(candidates) != 47:
        raise ValueError("frozen replacement survey must contain exactly 47 candidates")
    dispositions = {
        row["candidate_id"]: row["female_character"] for row in review["candidates"]
    }
    confirmed = [row for row in candidates if dispositions[row["candidate_id"]]]
    if len(confirmed) != 6:
        raise ValueError("frozen replacement review must contain exactly six female candidates")
    selected = min(
        confirmed,
        key=lambda row: (
            -row["eligible_donor_count"],
            abs(row["horizon"] - 12),
            row["event_id"],
        ),
    )

    if not isinstance(selection, Mapping):
        raise ValueError("frozen target selection must be a JSON object")
    if selection.get("schema_version") != 1 or type(selection.get("schema_version")) is not int:
        raise ValueError("frozen target selection schema_version mismatch")
    if selection.get("task_id") != TASK_ID:
        raise ValueError("frozen target selection task_id mismatch")
    if selection.get("dataset_commit") != VISTORY_DATASET_COMMIT:
        raise ValueError("frozen target selection dataset_commit mismatch")
    if selection.get("evaluator_commit") != EVALUATOR_COMMIT:
        raise ValueError("frozen target selection evaluator_commit mismatch")
    if selection.get("seeds") != [0, 1, 2] or any(
        type(seed) is not int for seed in selection.get("seeds", ())
    ):
        raise ValueError("frozen target selection seeds mismatch")
    events = selection.get("events")
    if not isinstance(events, list) or len(events) != 3:
        raise ValueError("frozen target selection must contain exactly three events")
    if any(not isinstance(event, Mapping) for event in events):
        raise ValueError("frozen target selection events must be JSON objects")
    retained = (ORIGINAL_EVENTS[0], ORIGINAL_EVENTS[2])
    for event, expected in zip((events[0], events[2]), retained):
        event_id, story_id, character_name, source_shot, target_shot = expected
        if (
            event.get("event_id"),
            event.get("story_id"),
            event.get("character_name"),
            event.get("source_shot"),
            event.get("target_shot"),
        ) != (event_id, story_id, character_name, source_shot, target_shot):
            raise ValueError("retained frozen event binding mismatch")
    excluded = {f"{story_id}::{name}" for _, story_id, name, _, _ in ORIGINAL_EVENTS}
    if f'{events[1].get("story_id")}::{events[1].get("character_name")}' in excluded:
        raise ValueError("frozen replacement uses an original target identity")

    expected_event = {
        "story_id": selected["story_id"],
        "event_id": selected["event_id"],
        "character_name": selected["character_name"],
        "source_shot": selected["source_shot"],
        "target_shot": selected["target_shot"],
        "story_sha256": selected["official_story"]["sha256"].upper(),
    }
    if events[1] != expected_event:
        raise ValueError("frozen replacement event does not match reviewed winner")
    expected_provenance = {
        "original_event_id": ORIGINAL_EVENT_ID,
        "selected_event_id": selected["event_id"],
        "selected_candidate_id": selected["candidate_id"],
        "eligible_donor_count": selected["eligible_donor_count"],
        "horizon": selected["horizon"],
        "horizon_distance": abs(selected["horizon"] - 12),
        "survey_sha256": survey_sha,
        "review_sha256": review_sha,
        "reviewer": reviewer,
        "dataset_commit": VISTORY_DATASET_COMMIT,
    }
    if selection.get("replacement_selection") != expected_provenance:
        raise ValueError("frozen replacement provenance mismatch")


def freeze_replacement_selection(
    *,
    data_root: Path,
    selection_path: Path,
    survey_path: Path,
    review_path: Path,
    output_path: Path,
) -> dict:
    data_root = Path(data_root).resolve()
    selection_path = Path(selection_path).resolve()
    survey_path = Path(survey_path).resolve()
    review_path = Path(review_path).resolve()
    selection = _load_original_selection(data_root, selection_path)
    survey = _read_json(survey_path, "replacement survey")
    candidates = _validate_survey(survey)
    current = _build_replacement_target_survey(
        data_root=data_root, selection_path=selection_path
    )
    if survey != current:
        raise ValueError("replacement survey does not match current official inputs")
    review = _read_json(review_path, "replacement review")
    reviewer = _validated_review(review, survey_path, candidates)
    dispositions = {
        row["candidate_id"]: row["female_character"] for row in review["candidates"]
    }
    confirmed = [row for row in candidates if dispositions[row["candidate_id"]]]
    if not confirmed:
        raise ValueError("no reviewer-confirmed female candidate has an eligible donor")
    selected = min(
        confirmed,
        key=lambda row: (
            -row["eligible_donor_count"],
            abs(row["horizon"] - 12),
            row["event_id"],
        ),
    )
    if selected["event_id"] == ORIGINAL_EVENT_ID or selected["entity_uid"] == "15::Gu Zhenzhen":
        raise ValueError("Gu Zhenzhen cannot be selected as the replacement")
    replacement = {
        "story_id": selected["story_id"],
        "event_id": selected["event_id"],
        "character_name": selected["character_name"],
        "source_shot": selected["source_shot"],
        "target_shot": selected["target_shot"],
        "story_sha256": selected["official_story"]["sha256"].upper(),
    }
    frozen = {
        **selection,
        "events": [selection["events"][0], replacement, selection["events"][2]],
        "replacement_selection": {
            "original_event_id": ORIGINAL_EVENT_ID,
            "selected_event_id": selected["event_id"],
            "selected_candidate_id": selected["candidate_id"],
            "eligible_donor_count": selected["eligible_donor_count"],
            "horizon": selected["horizon"],
            "horizon_distance": abs(selected["horizon"] - 12),
            "survey_sha256": sha256_file(survey_path),
            "review_sha256": sha256_file(review_path),
            "reviewer": reviewer,
            "dataset_commit": VISTORY_DATASET_COMMIT,
        },
    }
    frozen_event_ids = [str(event["event_id"]) for event in frozen["events"]]
    if len(frozen_event_ids) != len(set(frozen_event_ids)):
        raise ValueError(f"duplicate frozen event_id: {frozen_event_ids}")
    write_json_no_clobber(Path(output_path).resolve(), frozen)
    return frozen
