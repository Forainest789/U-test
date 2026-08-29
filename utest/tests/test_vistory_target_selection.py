from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import utest.vistory_target_selection as target_selection_module
import utest.vistory_reappearance as reappearance_module
from utest.prefix_contract import sha256_file
from utest.vistory_target_selection import (
    build_replacement_target_survey,
    freeze_replacement_selection,
    validate_frozen_replacement_provenance,
    write_replacement_review_template,
)
from utest.vistory_reappearance import load_frozen_selection


DATASET_COMMIT = "92f845531b67e97a67ae04b256ec5d8c020e8341"
EVALUATOR_COMMIT = "b44ec9108668cc2bcc8c5280886b235e9fb8bea9"
EVENTS_ROOT = Path(__file__).parents[1] / "events"


def test_checked_in_replacement_provenance_is_exactly_bound() -> None:
    selection = load_frozen_selection()
    survey_path = EVENTS_ROOT / "vistorybench_replacement_target_survey_v1.json"
    review_path = EVENTS_ROOT / "vistorybench_replacement_target_review_v1.json"
    survey = json.loads(survey_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))

    validate_frozen_replacement_provenance(selection, survey, review)

    assert sha256_file(survey_path) == "af44edabe4d70845869ca49fefb18a1c9199c37dd500cf276c37a9cc3c166562"
    assert sha256_file(review_path) == "32ded0398440d4b582ecce4c126a6bea8b838a3376d76ddeaaf0ea9bbaecba65"
    assert len(survey["candidates"]) == len(review["candidates"]) == 47
    dispositions = {
        row["candidate_id"]: row["female_character"] for row in review["candidates"]
    }
    confirmed = [
        row for row in survey["candidates"] if dispositions[row["candidate_id"]]
    ]
    assert len(confirmed) == 6
    selected = min(
        confirmed,
        key=lambda row: (
            -row["eligible_donor_count"],
            abs(row["horizon"] - 12),
            row["event_id"],
        ),
    )
    assert selected["event_id"] == "vistory42_bella_s15_s21"
    assert selected["candidate_id"] == selection["replacement_selection"]["selected_candidate_id"]


def _checked_in_replacement_artifacts() -> tuple[dict, dict, dict]:
    return (
        load_frozen_selection(),
        json.loads(
            (EVENTS_ROOT / "vistorybench_replacement_target_survey_v1.json").read_text(
                encoding="utf-8"
            )
        ),
        json.loads(
            (EVENTS_ROOT / "vistorybench_replacement_target_review_v1.json").read_text(
                encoding="utf-8"
            )
        ),
    )


def test_frozen_provenance_rejects_a_changed_human_disposition() -> None:
    selection, survey, review = _checked_in_replacement_artifacts()
    review["candidates"][0]["female_character"] = not review["candidates"][0][
        "female_character"
    ]

    with pytest.raises(ValueError, match="review SHA-256 mismatch"):
        validate_frozen_replacement_provenance(selection, survey, review)


def test_frozen_provenance_rejects_a_changed_selected_event() -> None:
    selection, survey, review = _checked_in_replacement_artifacts()
    selection["events"][1]["event_id"] = "forged"

    with pytest.raises(ValueError, match="selected event mismatch"):
        validate_frozen_replacement_provenance(selection, survey, review)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda selection: selection["events"][2].__setitem__("source_shot", True),
            "shots must be integers",
        ),
        (
            lambda selection: selection["events"][0].__setitem__("story_sha256", "bad"),
            "story_sha256 must be 64 uppercase hexadecimal digits",
        ),
        (
            lambda selection: selection["events"][0].__setitem__("unexpected", True),
            "schema mismatch",
        ),
    ],
)
def test_frozen_provenance_reuses_the_strict_selection_payload_gate(
    mutation, message: str
) -> None:
    selection, survey, review = _checked_in_replacement_artifacts()
    mutation(selection)

    with pytest.raises(ValueError, match=message):
        validate_frozen_replacement_provenance(selection, survey, review)


@pytest.mark.parametrize(
    "path_attribute",
    [
        "FROZEN_SELECTION_PATH",
        "FROZEN_SURVEY_PATH",
        "FROZEN_REVIEW_PATH",
    ],
)
def test_frozen_provenance_rejects_compact_rewritten_authority_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_attribute: str,
) -> None:
    selection, survey, review = _checked_in_replacement_artifacts()
    copied: dict[str, Path] = {}
    for source_name, attribute in (
        ("vistorybench_reappearance_v1.json", "FROZEN_SELECTION_PATH"),
        ("vistorybench_replacement_target_survey_v1.json", "FROZEN_SURVEY_PATH"),
        ("vistorybench_replacement_target_review_v1.json", "FROZEN_REVIEW_PATH"),
    ):
        target = tmp_path / source_name
        shutil.copyfile(EVENTS_ROOT / source_name, target)
        copied[attribute] = target
        monkeypatch.setattr(reappearance_module, attribute, target)
    rewritten = copied[path_attribute]
    rewritten.write_text(
        json.dumps(json.loads(rewritten.read_text(encoding="utf-8")), separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checked-in .* SHA-256 mismatch"):
        validate_frozen_replacement_provenance(selection, survey, review)


def _story(characters: dict[str, str], appearances: dict[int, list[str]]) -> dict:
    return {
        "Characters": {
            name: {"prompt_en": f"portrait of {name}", "tag": tag}
            for name, tag in characters.items()
        },
        "Shots": [
            {
                "index": index,
                "Characters Appearing": {"en": appearances.get(index, [])},
                "Setting Description": {"en": f"setting {index}"},
                "Shot Perspective Design": {"en": f"perspective {index}"},
                "Static Shot Description": {"en": f"static {index}"},
            }
            for index in range(1, 21)
        ],
    }


def _write_story(
    data_root: Path,
    story_id: str,
    story: dict,
    *,
    reference_names: tuple[str, ...],
) -> Path:
    story_path = data_root / story_id / "story.json"
    story_path.parent.mkdir(parents=True, exist_ok=True)
    story_path.write_text(json.dumps(story), encoding="utf-8")
    for name in reference_names:
        reference = data_root / story_id / "image" / name / "00.jpg"
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_bytes(f"reference:{story_id}:{name}".encode())
    return story_path


def _frozen_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "official"
    originals = {
        "15": ("Gu Zhenzhen", 8, 20),
        "16": ("Chen Sihan's Father", 1, 10),
        "79": ("Song Yuchen", 2, 8),
    }
    for number in range(1, 81):
        story_id = f"{number:02d}"
        if story_id in originals:
            name, source, target = originals[story_id]
            story = _story({name: "realistic_human"}, {source: [name], target: [name]})
            references = (name,)
        elif story_id == "20":
            name = "Alice Example"
            story = _story({name: "realistic_human"}, {1: [name], 12: [name]})
            references = (name,)
        elif story_id == "21":
            name = "Donor Example"
            story = _story({name: "realistic_human"}, {1: [name], 12: [name]})
            references = (name,)
        else:
            name = f"Character{story_id}"
            story = _story({name: "realistic_human"}, {})
            references = (name,)
        _write_story(data_root, story_id, story, reference_names=references)

    events = []
    for story_id, (name, source, target) in (
        ("79", originals["79"]),
        ("15", originals["15"]),
        ("16", originals["16"]),
    ):
        slug = {
            "79": "song_yuchen",
            "15": "gu_zhenzhen",
            "16": "chen_father",
        }[story_id]
        events.append(
            {
                "story_id": story_id,
                "event_id": f"vistory{int(story_id)}_{slug}_s{source}_s{target}",
                "character_name": name,
                "source_shot": source,
                "target_shot": target,
                "story_sha256": sha256_file(data_root / story_id / "story.json").upper(),
            }
        )
    selection = {
        "schema_version": 1,
        "task_id": "vistorybench_subject_reappearance_v1",
        "dataset_commit": DATASET_COMMIT,
        "evaluator_commit": EVALUATOR_COMMIT,
        "seeds": [0, 1, 2],
        "events": events,
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    return data_root, selection_path


def _review_candidate(
    review_path: Path, template: dict, candidate_id: str, *, reviewer: str = "Reviewer"
) -> None:
    review = json.loads(json.dumps(template))
    review["reviewer"] = reviewer
    for row in review["candidates"]:
        row["female_character"] = row["candidate_id"] == candidate_id
    review_path.write_text(json.dumps(review), encoding="utf-8")


def _replace_recurrence(
    data_root: Path,
    story_id: str,
    name: str,
    horizon: int,
    *,
    source_character_count: int = 1,
) -> None:
    extra = f"{name} Extra"
    characters = {name: "realistic_human"}
    source = [name]
    references = [name]
    if source_character_count == 2:
        characters[extra] = "realistic_human"
        source.append(extra)
        references.append(extra)
    _write_story(
        data_root,
        story_id,
        _story(characters, {1: source, 1 + horizon: [name]}),
        reference_names=tuple(references),
    )


def _write_review(
    path: Path, template: dict, female_names: set[str], survey: dict
) -> None:
    names_by_id = {
        row["candidate_id"]: row["character_name"] for row in survey["candidates"]
    }
    review = json.loads(json.dumps(template))
    review["reviewer"] = "Reviewer"
    for row in review["candidates"]:
        row["female_character"] = names_by_id[row["candidate_id"]] in female_names
    path.write_text(json.dumps(review), encoding="utf-8")


def test_survey_review_and_freeze_select_a_reviewed_female_candidate(
    tmp_path: Path,
) -> None:
    data_root, selection_path = _frozen_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_replacement_target_survey(
        data_root=data_root,
        selection_path=selection_path,
        output_path=survey_path,
    )
    assert all(
        row["entity_uid"] not in {"79::Song Yuchen", "15::Gu Zhenzhen", "16::Chen Sihan's Father"}
        for row in survey["candidates"]
    )
    selected = next(
        row for row in survey["candidates"] if row["character_name"] == "Alice Example"
    )
    assert selected["event_id"] == "vistory20_alice_example_s1_s12"
    assert selected["eligible_donor_count"] > 0

    review_path = tmp_path / "review.json"
    template = write_replacement_review_template(
        survey_path=survey_path, output_path=review_path
    )
    assert template["reviewer"] == ""
    assert {row["female_character"] for row in template["candidates"]} == {None}
    _review_candidate(review_path, template, selected["candidate_id"])

    output_path = tmp_path / "refrozen.json"
    frozen = freeze_replacement_selection(
        data_root=data_root,
        selection_path=selection_path,
        survey_path=survey_path,
        review_path=review_path,
        output_path=output_path,
    )
    assert [row["event_id"] for row in frozen["events"]] == [
        "vistory79_song_yuchen_s2_s8",
        "vistory20_alice_example_s1_s12",
        "vistory16_chen_father_s1_s10",
    ]
    assert frozen["replacement_selection"]["selected_candidate_id"] == selected["candidate_id"]
    assert frozen["replacement_selection"]["reviewer"] == "Reviewer"


def test_freeze_ranks_by_donor_count_then_horizon_distance(tmp_path: Path) -> None:
    data_root, selection_path = _frozen_fixture(tmp_path)
    _replace_recurrence(data_root, "20", "Rank A", 10)
    _replace_recurrence(data_root, "21", "Rank B", 11)
    _replace_recurrence(data_root, "22", "Rank C", 12, source_character_count=2)
    for story_id in ("23", "24", "25", "26"):
        _replace_recurrence(data_root, story_id, f"A Donor {story_id}", 10)
    for story_id in ("27", "28", "29", "30"):
        _replace_recurrence(data_root, story_id, f"B Donor {story_id}", 11)
    for story_id in ("31", "32", "33"):
        _replace_recurrence(
            data_root, story_id, f"C Donor {story_id}", 12, source_character_count=2
        )

    survey_path = tmp_path / "ranked-survey.json"
    survey = build_replacement_target_survey(
        data_root=data_root,
        selection_path=selection_path,
        output_path=survey_path,
    )
    ranked = {
        row["character_name"]: row
        for row in survey["candidates"]
        if row["character_name"] in {"Rank A", "Rank B", "Rank C"}
    }
    assert ranked["Rank A"]["eligible_donor_count"] == 5
    assert ranked["Rank B"]["eligible_donor_count"] == 5
    assert ranked["Rank C"]["eligible_donor_count"] == 3

    review_path = tmp_path / "ranked-review.json"
    template = write_replacement_review_template(
        survey_path=survey_path, output_path=review_path
    )
    _write_review(review_path, template, set(ranked), survey)
    frozen = freeze_replacement_selection(
        data_root=data_root,
        selection_path=selection_path,
        survey_path=survey_path,
        review_path=review_path,
        output_path=tmp_path / "ranked-selection.json",
    )
    assert frozen["events"][1]["character_name"] == "Rank B"


def test_freeze_uses_event_id_as_the_final_tie_breaker(tmp_path: Path) -> None:
    data_root, selection_path = _frozen_fixture(tmp_path)
    _replace_recurrence(data_root, "20", "Zulu", 11)
    _replace_recurrence(data_root, "21", "Alpha", 11)
    survey_path = tmp_path / "tie-survey.json"
    survey = build_replacement_target_survey(
        data_root=data_root,
        selection_path=selection_path,
        output_path=survey_path,
    )
    review_path = tmp_path / "tie-review.json"
    template = write_replacement_review_template(
        survey_path=survey_path, output_path=review_path
    )
    _write_review(review_path, template, {"Zulu", "Alpha"}, survey)

    frozen = freeze_replacement_selection(
        data_root=data_root,
        selection_path=selection_path,
        survey_path=survey_path,
        review_path=review_path,
        output_path=tmp_path / "tie-selection.json",
    )
    assert frozen["events"][1]["event_id"] == "vistory20_zulu_s1_s12"


def _review_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict, dict]:
    data_root, selection_path = _frozen_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_replacement_target_survey(
        data_root=data_root,
        selection_path=selection_path,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    review = write_replacement_review_template(
        survey_path=survey_path, output_path=review_path
    )
    selected = next(
        row for row in survey["candidates"] if row["character_name"] == "Alice Example"
    )
    _review_candidate(review_path, review, selected["candidate_id"])
    return data_root, selection_path, survey_path, survey, review_path


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing review candidate_ids"),
        ("duplicate", "duplicate review candidate_id"),
        ("non_boolean", "female_character must be boolean"),
        ("boolean_schema", "schema_version must be an integer"),
        ("blank_reviewer", "reviewer must be a non-empty human name"),
        ("stale_hash", "survey_sha256 is stale"),
    ],
)
def test_freeze_rejects_incomplete_or_malformed_human_review(
    tmp_path: Path, mutation: str, message: str
) -> None:
    data_root, selection_path, survey_path, _, review_path = _review_fixture(tmp_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        review["candidates"].pop()
    elif mutation == "duplicate":
        review["candidates"].append(dict(review["candidates"][0]))
    elif mutation == "non_boolean":
        review["candidates"][0]["female_character"] = 1
    elif mutation == "boolean_schema":
        review["schema_version"] = True
    elif mutation == "blank_reviewer":
        review["reviewer"] = "  "
    elif mutation == "stale_hash":
        review["survey_sha256"] = "0" * 64
    review_path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        freeze_replacement_selection(
            data_root=data_root,
            selection_path=selection_path,
            survey_path=survey_path,
            review_path=review_path,
            output_path=tmp_path / "invalid-selection.json",
        )


def test_survey_omits_a_valid_recurrence_with_zero_eligible_donors(
    tmp_path: Path,
) -> None:
    data_root, selection_path = _frozen_fixture(tmp_path)
    _replace_recurrence(
        data_root, "20", "Unsupported Target", 12, source_character_count=2
    )
    _write_story(
        data_root,
        "21",
        _story({"No Recurrence": "realistic_human"}, {}),
        reference_names=("No Recurrence",),
    )
    survey = build_replacement_target_survey(
        data_root=data_root,
        selection_path=selection_path,
        output_path=tmp_path / "zero-donor-survey.json",
    )
    assert "Unsupported Target" not in {
        row["character_name"] for row in survey["candidates"]
    }


@pytest.mark.parametrize("changed", ["story", "reference"])
def test_freeze_rejects_changed_official_candidate_bytes(
    tmp_path: Path, changed: str
) -> None:
    data_root, selection_path, survey_path, survey, review_path = _review_fixture(tmp_path)
    candidate = next(
        row for row in survey["candidates"] if row["character_name"] == "Alice Example"
    )
    path = data_root / (
        candidate["official_story"]["path"]
        if changed == "story"
        else candidate["reference"]["path"]
    )
    if changed == "story":
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        path.write_bytes(b"changed-reference")

    with pytest.raises(ValueError, match="does not match current official inputs"):
        freeze_replacement_selection(
            data_root=data_root,
            selection_path=selection_path,
            survey_path=survey_path,
            review_path=review_path,
            output_path=tmp_path / "changed-selection.json",
        )


def test_freeze_is_byte_reproducible_for_identical_survey_and_review(
    tmp_path: Path,
) -> None:
    data_root, selection_path, survey_path, _, review_path = _review_fixture(tmp_path)
    first = tmp_path / "first-selection.json"
    second = tmp_path / "second-selection.json"
    freeze_replacement_selection(
        data_root=data_root,
        selection_path=selection_path,
        survey_path=survey_path,
        review_path=review_path,
        output_path=first,
    )
    freeze_replacement_selection(
        data_root=data_root,
        selection_path=selection_path,
        survey_path=survey_path,
        review_path=review_path,
        output_path=second,
    )
    assert first.read_bytes() == second.read_bytes()


def test_survey_rejects_empty_or_colliding_event_slugs(tmp_path: Path) -> None:
    data_root, selection_path = _frozen_fixture(tmp_path)
    _replace_recurrence(data_root, "20", "人物", 11)
    with pytest.raises(ValueError, match="empty stable slug"):
        build_replacement_target_survey(
            data_root=data_root,
            selection_path=selection_path,
            output_path=tmp_path / "empty-slug.json",
        )

    _write_story(
        data_root,
        "20",
        _story(
            {"A B": "realistic_human", "A-B": "realistic_human"},
            {1: ["A B", "A-B"], 12: ["A B", "A-B"]},
        ),
        reference_names=("A B", "A-B"),
    )
    with pytest.raises(ValueError, match="colliding stable replacement event_id"):
        build_replacement_target_survey(
            data_root=data_root,
            selection_path=selection_path,
            output_path=tmp_path / "colliding-slug.json",
        )


def test_survey_omits_an_adjacent_non_absence_interval(tmp_path: Path) -> None:
    data_root, selection_path = _frozen_fixture(tmp_path)
    _replace_recurrence(data_root, "20", "Adjacent", 1)
    survey = build_replacement_target_survey(
        data_root=data_root,
        selection_path=selection_path,
        output_path=tmp_path / "adjacent.json",
    )
    assert "Adjacent" not in {row["character_name"] for row in survey["candidates"]}


def test_pinned_survey_rejects_empty_shots_in_a_non_retained_story(
    tmp_path: Path,
) -> None:
    data_root, selection_path = _frozen_fixture(tmp_path)
    story_path = data_root / "22" / "story.json"
    story = json.loads(story_path.read_text(encoding="utf-8"))
    story["Shots"] = []
    story_path.write_text(json.dumps(story), encoding="utf-8")
    output_path = tmp_path / "empty-shots-survey.json"

    with pytest.raises(ValueError, match=r"story 22 Shots must not be empty"):
        build_replacement_target_survey(
            data_root=data_root,
            selection_path=selection_path,
            output_path=output_path,
        )

    assert not output_path.exists()


def test_survey_event_id_preserves_the_official_story_id(tmp_path: Path) -> None:
    data_root, selection_path = _frozen_fixture(tmp_path)
    _replace_recurrence(data_root, "01", "Low ID", 11)
    survey = build_replacement_target_survey(
        data_root=data_root,
        selection_path=selection_path,
        output_path=tmp_path / "low-id.json",
    )
    low_id = next(row for row in survey["candidates"] if row["character_name"] == "Low ID")
    assert low_id["event_id"] == "vistory01_low_id_s1_s12"


def test_survey_rejects_a_replacement_event_id_colliding_with_a_retained_event(
    tmp_path: Path,
) -> None:
    data_root, selection_path = _frozen_fixture(tmp_path)
    _write_story(
        data_root,
        "79",
        _story(
            {
                "Song Yuchen": "realistic_human",
                "Song-Yuchen": "realistic_human",
            },
            {
                2: ["Song Yuchen", "Song-Yuchen"],
                8: ["Song Yuchen", "Song-Yuchen"],
            },
        ),
        reference_names=("Song Yuchen", "Song-Yuchen"),
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["events"][0]["story_sha256"] = sha256_file(
        data_root / "79" / "story.json"
    ).upper()
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    with pytest.raises(ValueError, match="colliding stable replacement event_id"):
        build_replacement_target_survey(
            data_root=data_root,
            selection_path=selection_path,
            output_path=tmp_path / "retained-collision.json",
        )


def test_freeze_rejects_a_forged_gu_zhenzhen_candidate(tmp_path: Path) -> None:
    data_root, selection_path, survey_path, survey, review_path = _review_fixture(tmp_path)
    forged = dict(survey["candidates"][0])
    forged.update(
        candidate_id="0" * 64,
        event_id="vistory15_gu_zhenzhen_s8_s20",
        story_id="15",
        character_name="Gu Zhenzhen",
        entity_uid="15::Gu Zhenzhen",
    )
    survey["candidates"].append(forged)
    survey["candidates"].sort(key=lambda row: row["candidate_id"])
    survey_path.write_text(json.dumps(survey), encoding="utf-8")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["survey_sha256"] = sha256_file(survey_path)
    review["candidates"].append(
        {"candidate_id": forged["candidate_id"], "female_character": True}
    )
    review_path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError, match="original target identity"):
        freeze_replacement_selection(
            data_root=data_root,
            selection_path=selection_path,
            survey_path=survey_path,
            review_path=review_path,
            output_path=tmp_path / "forged-gu.json",
        )


def test_freeze_requires_at_least_one_reviewed_female_candidate(tmp_path: Path) -> None:
    data_root, selection_path, survey_path, _, review_path = _review_fixture(tmp_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    for row in review["candidates"]:
        row["female_character"] = False
    review_path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="no reviewer-confirmed female"):
        freeze_replacement_selection(
            data_root=data_root,
            selection_path=selection_path,
            survey_path=survey_path,
            review_path=review_path,
            output_path=tmp_path / "no-female.json",
        )


def test_freeze_rejects_duplicate_final_event_ids_at_the_output_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, selection_path, survey_path, survey, review_path = _review_fixture(tmp_path)
    candidate = next(
        row for row in survey["candidates"] if row["character_name"] == "Alice Example"
    )
    candidate["event_id"] = "vistory79_song_yuchen_s2_s8"
    survey_path.write_text(json.dumps(survey), encoding="utf-8")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["survey_sha256"] = sha256_file(survey_path)
    review_path.write_text(json.dumps(review), encoding="utf-8")
    monkeypatch.setattr(
        target_selection_module,
        "_build_replacement_target_survey",
        lambda **_: survey,
    )
    output_path = tmp_path / "duplicate-final-event.json"

    with pytest.raises(ValueError, match="duplicate frozen event_id"):
        freeze_replacement_selection(
            data_root=data_root,
            selection_path=selection_path,
            survey_path=survey_path,
            review_path=review_path,
            output_path=output_path,
        )
    assert not output_path.exists()


def test_target_refreeze_outputs_never_clobber_existing_files(tmp_path: Path) -> None:
    data_root, selection_path, survey_path, _, review_path = _review_fixture(tmp_path)
    survey_bytes = survey_path.read_bytes()
    with pytest.raises(FileExistsError):
        build_replacement_target_survey(
            data_root=data_root,
            selection_path=selection_path,
            output_path=survey_path,
        )
    assert survey_path.read_bytes() == survey_bytes

    review_bytes = review_path.read_bytes()
    with pytest.raises(FileExistsError):
        write_replacement_review_template(
            survey_path=survey_path, output_path=review_path
        )
    assert review_path.read_bytes() == review_bytes

    output_path = tmp_path / "refrozen-output.json"
    freeze_replacement_selection(
        data_root=data_root,
        selection_path=selection_path,
        survey_path=survey_path,
        review_path=review_path,
        output_path=output_path,
    )
    frozen_bytes = output_path.read_bytes()
    with pytest.raises(FileExistsError):
        freeze_replacement_selection(
            data_root=data_root,
            selection_path=selection_path,
            survey_path=survey_path,
            review_path=review_path,
            output_path=output_path,
        )
    assert output_path.read_bytes() == frozen_bytes


@pytest.mark.parametrize(
    "arguments",
    [["--help"], ["survey", "--help"], ["review-template", "--help"], ["freeze", "--help"]],
)
def test_target_refreeze_cli_help_is_directly_executable(arguments: list[str]) -> None:
    tool = Path(__file__).parents[2] / "tools" / "refreeze_vistory_reappearance_events.py"
    result = subprocess.run(
        [sys.executable, str(tool), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
