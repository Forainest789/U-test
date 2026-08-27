"""Deterministic pre-generation donor selection from official ViStoryBench data."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from .prefix_contract import sha256_file, write_bytes_no_clobber, write_json_no_clobber
from .vistory_reappearance import convert_event


TARGET_EVENT_IDS = {
    "vistory79_song_yuchen_s2_s8",
    "vistory15_gu_zhenzhen_s8_s20",
    "vistory16_chen_father_s1_s10",
}
REVIEW_FIELDS = {
    "target_event_id",
    "candidate_id",
    "target_presentation_class",
    "donor_presentation_class",
    "target_dominant_colour",
    "donor_dominant_colour",
    "donor_source_visible",
    "donor_read_check_visible",
    "approved",
    "tie_group",
    "reviewer",
}
TARGET_INPUT_FIELDS = {
    "schema_version",
    "task_id",
    "dataset_commit",
    "evaluator_commit",
    "seeds",
    "events",
}
TARGET_EVENT_FIELDS = {
    "schema_version",
    "task_id",
    "dataset_commit",
    "evaluator_commit",
    "seeds",
    "event_id",
    "story_id",
    "character_name",
    "source_shot",
    "target_shot",
    "official_story",
    "reference_path",
    "reference_sha256",
    "reference_images",
    "chunk_mapping",
    "field_sources",
    "outputs",
}
TARGET_EVENT_REQUIRED_FIELDS = {
    "schema_version",
    "dataset_commit",
    "event_id",
    "story_id",
    "character_name",
    "source_shot",
    "target_shot",
    "official_story",
    "reference_path",
    "reference_sha256",
}


def horizon_bucket(horizon: int) -> str:
    for lower, upper in ((5, 7), (8, 10), (11, 13)):
        if lower <= horizon <= upper:
            return f"{lower}-{upper}"
    raise ValueError(f"unsupported donor horizon: {horizon}")


def _canonical_sha256(value: object) -> str:
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _style_class(tag: str) -> str:
    if tag == "realistic_human":
        return "realistic"
    if tag in {"unrealistic_human", "non_human"}:
        return "non_realistic"
    raise ValueError(f"unsupported official character tag: {tag}")


def _shot_prompt(shot: Mapping[str, object]) -> str:
    return (
        f'{shot["Setting Description"]["en"]} '
        f'{shot["Shot Perspective Design"]["en"]}. '
        f'{shot["Static Shot Description"]["en"]}'
    )


def _resolve_within(root: Path, relative: object, label: str) -> Path:
    path = (root / str(relative)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes {root.name.replace('_', ' ')} root: {relative}")
    return path


def _validate_fields(
    value: object,
    *,
    required: set[str],
    allowed: set[str],
    label: str,
) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} type must be a JSON object")
    missing = sorted(required - set(value))
    extra = sorted(set(value) - allowed)
    if missing or extra:
        raise ValueError(f"{label} schema mismatch: missing={missing}, extra={extra}")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} type must be a non-empty string")
    return value


def _require_sha256(value: object, label: str) -> str:
    text = _require_string(value, label)
    if re.fullmatch(r"[0-9A-Fa-f]{64}", text) is None:
        raise ValueError(f"{label} type must be a 64-digit SHA-256")
    return text.lower()


def _require_int_discriminator(value: object, expected: int, label: str) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(f"{label} must be integer {expected}")
    return value


def _linux_rename_directory_no_clobber(source: Path, destination: Path) -> None:
    """Publish a directory atomically with Linux RENAME_NOREPLACE."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOTSUP, "renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error, os.strerror(error), str(destination))
    raise OSError(error, os.strerror(error), str(destination))


def _publish_directory_no_clobber(source: Path, destination: Path) -> None:
    if os.name == "nt":
        os.rename(source, destination)
        return
    if sys.platform.startswith("linux"):
        _linux_rename_directory_no_clobber(source, destination)
        return
    raise OSError(
        errno.ENOTSUP,
        "atomic no-replace directory publication is unsupported on this platform",
    )


def _load_target_inputs(data_root: Path, target_inputs_path: Path) -> tuple[dict, list[dict]]:
    top = dict(
        _validate_fields(
            _read_json(target_inputs_path, "target inputs"),
            required={"schema_version", "dataset_commit", "events"},
            allowed=TARGET_INPUT_FIELDS,
            label="target inputs",
        )
    )
    _require_int_discriminator(
        top.get("schema_version"), 1, "target inputs type schema_version"
    )
    _require_string(top["dataset_commit"], "target inputs dataset_commit")
    if not isinstance(top["events"], list):
        raise ValueError("target inputs type events must be a list")
    if "seeds" in top and (
        not isinstance(top["seeds"], list)
        or any(type(seed) is not int for seed in top["seeds"])
    ):
        raise ValueError("target inputs type seeds must be a list of integers")
    for field in ("task_id", "evaluator_commit"):
        if field in top:
            _require_string(top[field], f"target inputs {field}")
    targets: list[dict] = []
    seen_event_ids: set[str] = set()
    for raw_entry in top["events"]:
        entry = _validate_fields(
            raw_entry,
            required={"event_id", "manifest_path", "manifest_sha256"},
            allowed={"event_id", "manifest_path", "manifest_sha256"},
            label="target inputs event entry",
        )
        event_id = _require_string(entry["event_id"], "target inputs event_id")
        manifest_relative = _require_string(
            entry["manifest_path"], "target inputs manifest_path"
        )
        manifest_sha = _require_sha256(
            entry["manifest_sha256"], "target inputs manifest_sha256"
        )
        if event_id in seen_event_ids:
            raise ValueError(f"duplicate target event_id: {event_id}")
        seen_event_ids.add(event_id)
        manifest_path = _resolve_within(
            target_inputs_path.parent, manifest_relative, "target manifest path"
        )
        if sha256_file(manifest_path).lower() != manifest_sha:
            raise ValueError(f"target event manifest SHA-256 mismatch: {manifest_path}")
        manifest = _validate_fields(
            _read_json(manifest_path, "target event manifest"),
            required=TARGET_EVENT_REQUIRED_FIELDS,
            allowed=TARGET_EVENT_FIELDS,
            label="target event",
        )
        _require_int_discriminator(
            manifest.get("schema_version"), 1, "target event type schema_version"
        )
        for field in (
            "dataset_commit",
            "event_id",
            "story_id",
            "character_name",
            "reference_path",
            "reference_sha256",
        ):
            _require_string(manifest[field], f"target event {field}")
        if type(manifest["source_shot"]) is not int or type(manifest["target_shot"]) is not int:
            raise ValueError("target event type source_shot/target_shot must be integers")
        official_story = _validate_fields(
            manifest["official_story"],
            required={"path", "sha256"},
            allowed={"path", "sha256"},
            label="target event official_story",
        )
        official_story_path = _require_string(
            official_story["path"], "target event official_story path"
        )
        official_story_sha = _require_sha256(
            official_story["sha256"], "target event official_story sha256"
        )
        reference_sha = _require_sha256(
            manifest["reference_sha256"], "target event reference_sha256"
        )
        for field in ("task_id", "evaluator_commit"):
            if field in manifest:
                _require_string(manifest[field], f"target event {field}")
        if "seeds" in manifest and (
            not isinstance(manifest["seeds"], list)
            or any(type(seed) is not int for seed in manifest["seeds"])
        ):
            raise ValueError("target event type seeds must be a list of integers")
        for field in ("reference_images", "chunk_mapping"):
            if field in manifest and not isinstance(manifest[field], list):
                raise ValueError(f"target event type {field} must be a list")
        for field in ("field_sources", "outputs"):
            if field in manifest and not isinstance(manifest[field], Mapping):
                raise ValueError(f"target event type {field} must be an object")
        if manifest.get("dataset_commit") != top["dataset_commit"]:
            raise ValueError("target event dataset_commit does not match target inputs")
        if manifest.get("event_id") != event_id:
            raise ValueError("target event_id does not match target inputs entry")
        story_path = _resolve_within(
            data_root, official_story_path, "official story path"
        )
        if sha256_file(story_path).lower() != official_story_sha:
            raise ValueError(f"official target story SHA-256 mismatch: {story_path}")
        reference_path = _resolve_within(
            data_root, manifest["reference_path"], "official reference path"
        )
        if sha256_file(reference_path).lower() != reference_sha:
            raise ValueError(f"official target reference SHA-256 mismatch: {reference_path}")
        official = json.loads(story_path.read_text(encoding="utf-8-sig"))
        character = manifest["character_name"]
        if character not in official["Characters"]:
            raise ValueError("target event schema character_name is absent from official story")
        normalized_names = [str(name).strip().casefold() for name in official["Characters"]]
        if normalized_names.count(str(character).strip().casefold()) != 1:
            raise ValueError("target event schema character identity is ambiguous")
        character_row = official["Characters"][character]
        shots_by_index = {int(shot["index"]): shot for shot in official["Shots"]}
        if manifest["source_shot"] not in shots_by_index or manifest["target_shot"] not in shots_by_index:
            raise ValueError("target event schema source_shot/target_shot is absent")
        source = shots_by_index[manifest["source_shot"]]
        try:
            convert_event(official, manifest)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"target event schema interval is invalid: {error}") from error
        tag = str(character_row["tag"])
        targets.append(
            {
                "event_id": str(manifest["event_id"]),
                "story_id": str(manifest["story_id"]),
                "character_name": str(character),
                "entity_uid": f'{manifest["story_id"]}::{character}',
                "official_tag": tag,
                "style_class": _style_class(tag),
                "source_character_count": len(source["Characters Appearing"]["en"]),
                "gap_bucket": horizon_bucket(
                    int(manifest["target_shot"]) - int(manifest["source_shot"])
                ),
                "official_story": {
                    "path": Path(official_story_path).as_posix(),
                    "sha256": sha256_file(story_path),
                },
                "reference": {
                    "path": Path(manifest["reference_path"]).as_posix(),
                    "sha256": sha256_file(reference_path),
                },
                "target_manifest": {
                    "path": Path(manifest_relative).as_posix(),
                    "sha256": sha256_file(manifest_path),
                },
            }
        )
    return top, targets


def _candidate_id(candidate: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {
            "target_event_id": candidate["target_event_id"],
            "donor_story_id": candidate["donor_story_id"],
            "donor_char_id": candidate["donor_char_id"],
            "source_shot": candidate["source_shot"],
            "read_shot": candidate["read_shot"],
        }
    )


def _build_survey(data_root: Path, target_inputs_path: Path) -> dict[str, object]:
    top, targets = _load_target_inputs(data_root, target_inputs_path)
    candidates: list[dict] = []
    rejections: list[dict] = []
    for discovered_path in sorted(
        data_root.glob("*/story.json"), key=lambda path: path.parent.name
    ):
        story_path = discovered_path.resolve()
        if not story_path.is_relative_to(data_root):
            raise ValueError(f"official story path escapes data root: {discovered_path}")
        story_id = story_path.parent.name
        official = json.loads(story_path.read_text(encoding="utf-8-sig"))
        by_index = {int(shot["index"]): shot for shot in official["Shots"]}
        normalized_names = [str(name).strip().casefold() for name in official["Characters"]]
        ambiguous_names = {
            name
            for name in official["Characters"]
            if normalized_names.count(str(name).strip().casefold()) > 1
        }
        for target in targets:
            for character, character_row in sorted(official["Characters"].items()):
                appearances = [
                    index
                    for index, shot in sorted(by_index.items())
                    if character in shot["Characters Appearing"]["en"]
                ]
                for source_shot, read_shot in zip(appearances, appearances[1:]):
                    core = {
                        "target_event_id": target["event_id"],
                        "target_story_id": target["story_id"],
                        "target_entity_uid": target["entity_uid"],
                        "donor_story_id": story_id,
                        "donor_char_id": character,
                        "donor_entity_uid": f"{story_id}::{character}",
                        "source_shot": source_shot,
                        "read_shot": read_shot,
                    }
                    reasons: list[str] = []
                    if story_id == target["story_id"]:
                        reasons.append("same_story")
                    if core["donor_entity_uid"] == target["entity_uid"]:
                        reasons.append("same_identity")
                    if character in ambiguous_names:
                        reasons.append("ambiguous_duplicate_identity")
                    if any(
                        shot["Characters Appearing"]["en"].count(character) != 1
                        for index, shot in by_index.items()
                        if source_shot <= index <= read_shot
                        and character in shot["Characters Appearing"]["en"]
                    ):
                        reasons.append("ambiguous_duplicate_identity")
                    horizon = read_shot - source_shot
                    try:
                        bucket = horizon_bucket(horizon)
                    except ValueError:
                        reasons.append("unsupported_horizon")
                        bucket = None
                    tag = str(character_row["tag"])
                    try:
                        style = _style_class(tag)
                    except ValueError:
                        reasons.append("unsupported_official_tag")
                        style = None
                    source_characters = by_index[source_shot]["Characters Appearing"]["en"]
                    if tag != target["official_tag"]:
                        reasons.append("official_tag_mismatch")
                    if style != target["style_class"]:
                        reasons.append("style_class_mismatch")
                    if len(source_characters) != target["source_character_count"]:
                        reasons.append("source_character_count_mismatch")
                    if bucket != target["gap_bucket"]:
                        reasons.append("gap_bucket_mismatch")
                    reference = _resolve_within(
                        data_root,
                        Path(story_id) / "image" / character / "00.jpg",
                        "donor reference path",
                    )
                    if not reference.is_file():
                        reasons.append("missing_reference")
                    try:
                        convert_event(
                            official,
                            {
                                "story_id": story_id,
                                "event_id": f"survey_{_candidate_id(core)[:16]}",
                                "character_name": character,
                                "source_shot": source_shot,
                                "target_shot": read_shot,
                            },
                        )
                    except (KeyError, TypeError, ValueError):
                        reasons.append("invalid_absence_interval")
                    candidate_id = _candidate_id(core)
                    record = {
                        "candidate_id": candidate_id,
                        **core,
                        "horizon": horizon,
                        "gap_bucket": bucket,
                        "official_tag": tag,
                        "style_class": style,
                        "source_character_count": len(source_characters),
                        "official_story": {
                            "path": story_path.relative_to(data_root).as_posix(),
                            "sha256": sha256_file(story_path),
                        },
                        "reference": {
                            "path": reference.relative_to(data_root).as_posix(),
                            "sha256": sha256_file(reference) if reference.is_file() else None,
                        },
                        "source_prompt": _shot_prompt(by_index[source_shot]),
                        "read_prompt": _shot_prompt(by_index[read_shot]),
                    }
                    if reasons:
                        rejections.append({**record, "reasons": reasons})
                        continue
                    candidates.append(record)
    return {
        "schema_version": 1,
        "dataset_commit": top["dataset_commit"],
        "target_inputs_sha256": sha256_file(target_inputs_path),
        "selection_seed": 0,
        "targets": targets,
        "candidates": sorted(candidates, key=lambda row: row["candidate_id"]),
        "rejections": sorted(rejections, key=lambda row: row["candidate_id"]),
    }


def build_donor_candidate_survey(
    *, data_root: Path, target_inputs_path: Path, output_path: Path
) -> dict[str, object]:
    survey = _build_survey(Path(data_root).resolve(), Path(target_inputs_path).resolve())
    write_json_no_clobber(Path(output_path).resolve(), survey)
    return survey


def _read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _selected_reviews(
    review: Mapping, survey: Mapping
) -> tuple[list[tuple[dict, dict]], list[dict]]:
    candidates = {row["candidate_id"]: row for row in survey["candidates"]}
    reviews_by_target: dict[str, list[tuple[dict, dict]]] = {
        target["event_id"]: [] for target in survey["targets"]
    }
    seen_candidate_ids: set[str] = set()
    dispositions: list[dict] = []
    for raw_review in review.get("reviews", []):
        if not isinstance(raw_review, Mapping):
            raise ValueError("review row must be a JSON object")
        reviewed = dict(raw_review)
        if set(reviewed) != REVIEW_FIELDS:
            raise ValueError("review row fields do not match the strict schema")
        for field in (
            "target_event_id",
            "candidate_id",
            "target_presentation_class",
            "donor_presentation_class",
            "target_dominant_colour",
            "donor_dominant_colour",
            "reviewer",
        ):
            if not isinstance(reviewed[field], str) or not reviewed[field].strip():
                raise ValueError(f"review {field} must be a non-empty string")
        if not isinstance(reviewed["approved"], bool):
            raise ValueError("review approved must be boolean")
        if not isinstance(reviewed["donor_source_visible"], bool) or not isinstance(
            reviewed["donor_read_check_visible"], bool
        ):
            raise ValueError("review visibility decisions must be boolean")
        if reviewed["tie_group"] is not None and (
            not isinstance(reviewed["tie_group"], str)
            or not reviewed["tie_group"].strip()
        ):
            raise ValueError("review tie_group must be null or a non-empty string")
        candidate_id = reviewed.get("candidate_id")
        if candidate_id in seen_candidate_ids:
            raise ValueError(f"duplicate review candidate_id: {candidate_id}")
        seen_candidate_ids.add(candidate_id)
        if candidate_id not in candidates:
            raise ValueError(f"stale or unknown candidate_id: {candidate_id}")
        candidate = candidates[candidate_id]
        if reviewed.get("target_event_id") != candidate["target_event_id"]:
            raise ValueError("review target_event_id does not match survey candidate")
        reasons: list[str] = []
        if reviewed["target_presentation_class"] != reviewed["donor_presentation_class"]:
            reasons.append("presentation_mismatch")
        if reviewed["target_dominant_colour"] != reviewed["donor_dominant_colour"]:
            reasons.append("colour_mismatch")
        if reviewed["donor_source_visible"] is not True:
            reasons.append("source_not_visible")
        if reviewed["donor_read_check_visible"] is not True:
            reasons.append("read_not_visible")
        if reviewed["approved"] is True and reasons:
            messages = {
                "presentation_mismatch": "approved review presentation class mismatch",
                "colour_mismatch": "approved review dominant colour mismatch",
                "source_not_visible": "approved review donor is not visible at source",
                "read_not_visible": "approved review donor is not visible at read-check",
            }
            raise ValueError(messages[reasons[0]])
        if reviewed["approved"] is False:
            reasons.append("human_rejected")
        dispositions.append(
            {
                "target_event_id": candidate["target_event_id"],
                "candidate_id": candidate_id,
                "approved": reviewed["approved"],
                "tie_group": reviewed["tie_group"],
                "reviewer": reviewed["reviewer"],
                "reasons": reasons,
                "review": reviewed,
            }
        )
        if reviewed.get("approved") is True:
            reviews_by_target[candidate["target_event_id"]].append(
                (candidate, reviewed)
            )

    missing_reviews = sorted(set(candidates) - seen_candidate_ids)
    if missing_reviews:
        raise ValueError(
            "unreviewed eligible candidate IDs: " + ",".join(missing_reviews)
        )
    selected: list[tuple[dict, dict]] = []
    generator = random.Random(0)
    for target_event_id in sorted(reviews_by_target):
        approved = reviews_by_target[target_event_id]
        if not approved:
            raise ValueError(f"zero approved donor candidates for {target_event_id}")
        if len(approved) > 1:
            tie_groups = {review.get("tie_group") for _, review in approved}
            if len(tie_groups) != 1 or not next(iter(tie_groups)):
                raise ValueError(
                    f"multiple approved donor candidates without one declared tie_group: "
                    f"{target_event_id}"
                )
            approved = sorted(approved, key=lambda item: item[0]["candidate_id"])
            selected.append(generator.choice(approved))
        else:
            selected.append(approved[0])
    return selected, sorted(dispositions, key=lambda row: row["candidate_id"])


def freeze_donor_selection(
    *,
    data_root: Path,
    target_inputs_path: Path,
    survey_path: Path,
    review_path: Path,
    output_root: Path,
) -> dict[str, object]:
    data_root = Path(data_root).resolve()
    target_inputs_path = Path(target_inputs_path).resolve()
    survey_path = Path(survey_path).resolve()
    review_path = Path(review_path).resolve()
    output_root = Path(output_root).resolve()
    if os.path.lexists(output_root):
        raise FileExistsError(f"output root already exists: {output_root}")

    survey = _read_json(survey_path, "donor survey")
    _require_int_discriminator(
        survey.get("schema_version"), 1, "survey schema_version"
    )
    _require_int_discriminator(
        survey.get("selection_seed"), 0, "survey selection_seed"
    )
    expected_survey = _build_survey(data_root, target_inputs_path)
    if survey != expected_survey:
        raise ValueError("donor survey does not match current official inputs")
    review = _read_json(review_path, "donor review")
    if set(review) != {"schema_version", "dataset_commit", "survey_sha256", "reviews"}:
        raise ValueError("review document fields do not match the strict schema")
    _require_int_discriminator(
        review.get("schema_version"), 1, "review schema_version"
    )
    if review.get("dataset_commit") != survey.get("dataset_commit"):
        raise ValueError("review dataset_commit does not match survey")
    if review.get("survey_sha256") != sha256_file(survey_path):
        raise ValueError("review survey_sha256 does not match survey")
    if not isinstance(review.get("reviews"), list):
        raise ValueError("review reviews must be a list")

    selected, review_dispositions = _selected_reviews(review, survey)
    selected_target_ids = {candidate["target_event_id"] for candidate, _ in selected}
    if selected_target_ids != TARGET_EVENT_IDS:
        raise ValueError(
            "frozen donor selection must contain exactly the three target event IDs"
        )
    dispositions_by_id = {
        row["candidate_id"]: row for row in review_dispositions
    }
    candidate_audit = []
    for candidate in survey["candidates"]:
        disposition = dispositions_by_id.get(candidate["candidate_id"])
        if disposition is None:
            raise RuntimeError("validated review coverage became incomplete")
        candidate_audit.append(
            {
                **candidate,
                "structural_status": "accepted",
                "review_status": (
                    "approved" if disposition["approved"] else "rejected"
                ),
                "approved": disposition["approved"],
                "review": disposition["review"],
                "reasons": disposition["reasons"],
            }
        )
    candidate_audit.extend(
        {
            **{key: value for key, value in row.items() if key != "reasons"},
            "structural_status": "rejected",
            "review_status": "not_applicable",
            "approved": False,
            "review": None,
            "reasons": row["reasons"],
        }
        for row in survey["rejections"]
    )
    candidate_audit.sort(
        key=lambda row: (row["candidate_id"], row["structural_status"])
    )

    prepared: list[dict] = []
    for candidate, reviewed in selected:
        story_path = _resolve_within(
            data_root, candidate["official_story"]["path"], "donor story path"
        )
        if sha256_file(story_path) != candidate["official_story"]["sha256"]:
            raise ValueError(f"official donor story SHA-256 mismatch: {story_path}")
        reference_path = _resolve_within(
            data_root, candidate["reference"]["path"], "donor reference path"
        )
        if sha256_file(reference_path) != candidate["reference"]["sha256"]:
            raise ValueError(f"official donor reference SHA-256 mismatch: {reference_path}")
        official = json.loads(story_path.read_text(encoding="utf-8-sig"))
        donor_event_id = f'donor_{candidate["candidate_id"][:16]}'
        derived, event = convert_event(
            official,
            {
                "story_id": candidate["donor_story_id"],
                "event_id": donor_event_id,
                "character_name": candidate["donor_char_id"],
                "source_shot": candidate["source_shot"],
                "target_shot": candidate["read_shot"],
            },
        )
        event_root = output_root / candidate["target_event_id"]
        event.update(
            {
                "source_json_path": "story.json",
                "reference_path": "reference.jpg",
                "reference_sha256": candidate["reference"]["sha256"],
                "donor_seed": 0,
                "path_resolution": "event_parent",
            }
        )
        prepared.append(
            {
                "candidate": candidate,
                "review": reviewed,
                "derived": derived,
                "event": event,
                "relative_root": event_root.relative_to(output_root),
                "reference_bytes": reference_path.read_bytes(),
            }
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            dir=output_root.parent,
            prefix=f".{output_root.name}.",
            suffix=".tmp",
        )
    ).resolve()
    try:
        selection_events: list[dict] = []
        for item in prepared:
            candidate = item["candidate"]
            event_root = staging_root / item["relative_root"]
            story_output = event_root / "story.json"
            event_output = event_root / "event.json"
            reference_output = event_root / "reference.jpg"
            manifest_output = event_root / "manifest.json"
            write_json_no_clobber(story_output, item["derived"])
            write_json_no_clobber(event_output, item["event"])
            write_bytes_no_clobber(reference_output, item["reference_bytes"])
            manifest = {
                "schema_version": 1,
                "target_event_id": candidate["target_event_id"],
                "candidate": candidate,
                "review": item["review"],
                "donor_seed": 0,
                "path_resolution": "selection_parent",
                "outputs": {
                    "story": {
                        "path": story_output.relative_to(staging_root).as_posix(),
                        "sha256": sha256_file(story_output),
                    },
                    "event": {
                        "path": event_output.relative_to(staging_root).as_posix(),
                        "sha256": sha256_file(event_output),
                    },
                    "reference": {
                        "path": reference_output.relative_to(staging_root).as_posix(),
                        "sha256": sha256_file(reference_output),
                    },
                },
            }
            write_json_no_clobber(manifest_output, manifest)
            selection_events.append(
                {
                    "target_event_id": candidate["target_event_id"],
                    "candidate_id": candidate["candidate_id"],
                    "donor_story_id": candidate["donor_story_id"],
                    "donor_entity_uid": candidate["donor_entity_uid"],
                    "donor_seed": 0,
                    "manifest_path": manifest_output.relative_to(
                        staging_root
                    ).as_posix(),
                    "manifest_sha256": sha256_file(manifest_output),
                }
            )
        selection = {
            "schema_version": 1,
            "dataset_commit": survey["dataset_commit"],
            "selection_seed": 0,
            "donor_seed": 0,
            "path_contract": {
                "selection_paths_relative_to": "selection_parent",
                "event_paths_relative_to": "event_parent",
            },
            "target_inputs_sha256": survey["target_inputs_sha256"],
            "survey_sha256": sha256_file(survey_path),
            "review_sha256": sha256_file(review_path),
            "candidate_audit": candidate_audit,
            "events": sorted(selection_events, key=lambda row: row["target_event_id"]),
        }
        write_json_no_clobber(staging_root / "selection.json", selection)
        _publish_directory_no_clobber(staging_root, output_root)
    finally:
        if staging_root.exists():
            if not staging_root.is_relative_to(output_root.parent):
                raise RuntimeError("refusing to clean staging outside output parent")
            shutil.rmtree(staging_root)
    return selection
