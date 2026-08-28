from __future__ import annotations

import json
import errno
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

import utest.vistory_donors as donor_module
from utest.prefix_contract import sha256_file
from utest.vistory_donors import (
    build_donor_candidate_survey,
    donor_rejection_reasons,
    enumerate_official_recurrences,
    freeze_donor_selection,
    horizon_bucket,
    validate_frozen_vistory_tree,
)


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
            for index in range(1, 14)
        ],
    }


def _write_official_story(
    data_root: Path,
    story_id: str,
    story: dict,
    *,
    reference_names: tuple[str, ...],
) -> Path:
    story_path = data_root / story_id / "story.json"
    story_path.parent.mkdir(parents=True)
    story_path.write_text(json.dumps(story), encoding="utf-8")
    for name in reference_names:
        reference = data_root / story_id / "image" / name / "00.jpg"
        reference.parent.mkdir(parents=True)
        reference.write_bytes(f"reference:{story_id}:{name}".encode())
    return story_path


def _write_target_inputs(
    root: Path,
    data_root: Path,
    *,
    story_id: str,
    character: str,
    source_shot: int,
    target_shot: int,
    event_id: str = "target_event",
    dataset_commit: str = "dataset-commit",
) -> Path:
    story_path = data_root / story_id / "story.json"
    reference = data_root / story_id / "image" / character / "00.jpg"
    event_root = root / event_id
    event_root.mkdir(parents=True)
    event_manifest = {
        "schema_version": 1,
        "dataset_commit": dataset_commit,
        "event_id": event_id,
        "story_id": story_id,
        "character_name": character,
        "source_shot": source_shot,
        "target_shot": target_shot,
        "official_story": {
            "path": f"{story_id}/story.json",
            "sha256": sha256_file(story_path),
        },
        "reference_path": f"{story_id}/image/{character}/00.jpg",
        "reference_sha256": sha256_file(reference),
    }
    manifest_path = event_root / "manifest.json"
    manifest_path.write_text(json.dumps(event_manifest), encoding="utf-8")
    top = {
        "schema_version": 1,
        "dataset_commit": dataset_commit,
        "events": [
            {
                "event_id": event_id,
                "manifest_path": f"{event_id}/manifest.json",
                "manifest_sha256": sha256_file(manifest_path),
            }
        ],
    }
    target_inputs = root / "manifest.json"
    target_inputs.write_text(json.dumps(top), encoding="utf-8")
    return target_inputs


def _write_frozen_official_tree(data_root: Path) -> None:
    for story_number in range(1, 81):
        story_id = f"{story_number:02d}"
        if story_id == "15":
            character = "Target"
            appearances = {2: [character], 8: [character]}
        elif story_id == "20":
            character = "Donor"
            appearances = {3: [character], 9: [character]}
        else:
            character = f"Character{story_id}"
            appearances = {}
        _write_official_story(
            data_root,
            story_id,
            _story({character: "realistic_human"}, appearances),
            reference_names=(character,),
        )


def test_official_survey_accepts_and_publishes_a_complete_frozen_tree(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "official"
    _write_frozen_official_tree(data_root)
    targets = _write_target_inputs(
        tmp_path / "targets",
        data_root,
        story_id="15",
        character="Target",
        source_shot=2,
        target_shot=8,
        dataset_commit="92f845531b67e97a67ae04b256ec5d8c020e8341",
    )
    output = tmp_path / "survey.json"

    validate_frozen_vistory_tree(data_root)
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=output,
    )

    assert output.is_file()
    assert any(row["donor_story_id"] == "20" for row in survey["candidates"])


def test_frozen_tree_validation_accepts_an_official_story_with_utf8_bom(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "official"
    _write_frozen_official_tree(data_root)
    story_path = data_root / "01" / "story.json"
    story_path.write_text(story_path.read_text(encoding="utf-8"), encoding="utf-8-sig")

    validate_frozen_vistory_tree(data_root)


def test_official_survey_rejects_a_missing_non_target_story(tmp_path: Path) -> None:
    data_root = tmp_path / "official"
    _write_frozen_official_tree(data_root)
    (data_root / "42" / "story.json").unlink()
    targets = _write_target_inputs(
        tmp_path / "targets",
        data_root,
        story_id="15",
        character="Target",
        source_shot=2,
        target_shot=8,
        dataset_commit="92f845531b67e97a67ae04b256ec5d8c020e8341",
    )
    output = tmp_path / "survey.json"

    with pytest.raises(ValueError, match=r"official ViStoryBench tree.*42"):
        build_donor_candidate_survey(
            data_root=data_root,
            target_inputs_path=targets,
            output_path=output,
        )

    assert not output.exists()


def test_official_survey_rejects_a_missing_non_target_reference(tmp_path: Path) -> None:
    data_root = tmp_path / "official"
    _write_frozen_official_tree(data_root)
    (data_root / "42" / "image" / "Character42" / "00.jpg").unlink()
    targets = _write_target_inputs(
        tmp_path / "targets",
        data_root,
        story_id="15",
        character="Target",
        source_shot=2,
        target_shot=8,
        dataset_commit="92f845531b67e97a67ae04b256ec5d8c020e8341",
    )

    with pytest.raises(ValueError, match=r"official primary reference missing.*42"):
        build_donor_candidate_survey(
            data_root=data_root,
            target_inputs_path=targets,
            output_path=tmp_path / "survey.json",
        )


def _write_target_inputs_many(
    root: Path, data_root: Path, specs: list[dict[str, object]]
) -> Path:
    entries = []
    for spec in specs:
        event_id = str(spec["event_id"])
        story_id = str(spec["story_id"])
        character = str(spec["character_name"])
        story_path = data_root / story_id / "story.json"
        reference = data_root / story_id / "image" / character / "00.jpg"
        event_root = root / event_id
        event_root.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "dataset_commit": "dataset-commit",
            **spec,
            "official_story": {
                "path": f"{story_id}/story.json",
                "sha256": sha256_file(story_path),
            },
            "reference_path": f"{story_id}/image/{character}/00.jpg",
            "reference_sha256": sha256_file(reference),
        }
        path = event_root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        entries.append(
            {
                "event_id": event_id,
                "manifest_path": f"{event_id}/manifest.json",
                "manifest_sha256": sha256_file(path),
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    target_inputs = root / "manifest.json"
    target_inputs.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_commit": "dataset-commit",
                "events": entries,
            }
        ),
        encoding="utf-8",
    )
    return target_inputs


def _three_target_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "official"
    target_specs: list[dict[str, object]] = []
    target_ids = (
        "vistory79_song_yuchen_s2_s8",
        "vistory15_gu_zhenzhen_s8_s20",
        "vistory16_chen_father_s1_s10",
    )
    for offset, (horizon, event_id) in enumerate(zip((6, 9, 12), target_ids)):
        target_story_id = str(10 + offset)
        target_name = f"Target{offset}"
        _write_official_story(
            data_root,
            target_story_id,
            _story(
                {target_name: "realistic_human"},
                {1: [target_name], 1 + horizon: [target_name]},
            ),
            reference_names=(target_name,),
        )
        donor_story_id = str(20 + offset)
        donor_name = f"Donor{offset}"
        _write_official_story(
            data_root,
            donor_story_id,
            _story(
                {donor_name: "realistic_human"},
                {1: [donor_name], 1 + horizon: [donor_name]},
            ),
            reference_names=(donor_name,),
        )
        target_specs.append(
            {
                "event_id": event_id,
                "story_id": target_story_id,
                "character_name": target_name,
                "source_shot": 1,
                "target_shot": 1 + horizon,
            }
        )
    return data_root, _write_target_inputs_many(
        tmp_path / "targets", data_root, target_specs
    )


def _write_review_for_survey(path: Path, survey_path: Path, survey: dict) -> dict:
    review = {
        "schema_version": 1,
        "dataset_commit": "dataset-commit",
        "survey_sha256": sha256_file(survey_path),
        "reviews": [
            {
                "target_event_id": candidate["target_event_id"],
                "candidate_id": candidate["candidate_id"],
                "target_presentation_class": "male",
                "donor_presentation_class": "male",
                "target_dominant_colour": "black",
                "donor_dominant_colour": "black",
                "donor_source_visible": True,
                "donor_read_check_visible": True,
                "approved": True,
                "tie_group": None,
                "reviewer": "human",
            }
            for candidate in survey["candidates"]
        ],
    }
    path.write_text(json.dumps(review), encoding="utf-8")
    return review


def test_survey_enumerates_a_matching_official_recurrence(tmp_path: Path) -> None:
    data_root = tmp_path / "official"
    target_story = _story(
        {"Target": "realistic_human"}, {2: ["Target"], 8: ["Target"]}
    )
    _write_official_story(data_root, "10", target_story, reference_names=("Target",))
    donor_story = _story(
        {"Donor": "realistic_human"}, {3: ["Donor"], 9: ["Donor"]}
    )
    donor_path = _write_official_story(
        data_root, "20", donor_story, reference_names=("Donor",)
    )
    targets = _write_target_inputs(
        tmp_path / "targets",
        data_root,
        story_id="10",
        character="Target",
        source_shot=2,
        target_shot=8,
    )

    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=tmp_path / "survey.json",
    )

    assert survey["selection_seed"] == 0
    assert survey["candidates"] == [
        {
            "candidate_id": survey["candidates"][0]["candidate_id"],
            "target_event_id": "target_event",
            "target_story_id": "10",
            "target_entity_uid": "10::Target",
            "donor_story_id": "20",
            "donor_char_id": "Donor",
            "donor_entity_uid": "20::Donor",
            "source_shot": 3,
            "read_shot": 9,
            "horizon": 6,
            "gap_bucket": "5-7",
            "official_tag": "realistic_human",
            "style_class": "realistic",
            "source_character_count": 1,
            "official_story": {
                "path": "20/story.json",
                "sha256": sha256_file(donor_path),
            },
            "reference": {
                "path": "20/image/Donor/00.jpg",
                "sha256": sha256_file(
                    data_root / "20" / "image" / "Donor" / "00.jpg"
                ),
            },
            "source_prompt": "setting 3 perspective 3. static 3",
            "read_prompt": "setting 9 perspective 9. static 9",
        }
    ]


def test_extracted_recurrences_preserve_donor_survey_bytes(tmp_path: Path) -> None:
    data_root = tmp_path / "official"
    _write_official_story(
        data_root,
        "10",
        _story({"Target": "realistic_human"}, {2: ["Target"], 8: ["Target"]}),
        reference_names=("Target",),
    )
    _write_official_story(
        data_root,
        "20",
        _story({"Donor": "realistic_human"}, {3: ["Donor"], 9: ["Donor"]}),
        reference_names=("Donor",),
    )
    targets = _write_target_inputs(
        tmp_path / "targets",
        data_root,
        story_id="10",
        character="Target",
        source_shot=2,
        target_shot=8,
    )

    recurrences = enumerate_official_recurrences(data_root)
    target = next(row for row in recurrences if row["entity_uid"] == "10::Target")
    donor = next(row for row in recurrences if row["entity_uid"] == "20::Donor")
    assert donor_rejection_reasons(target, donor) == []

    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=tmp_path / "survey.json",
    )
    frozen_bytes = json.dumps(
        {"candidates": survey["candidates"], "rejections": survey["rejections"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(frozen_bytes).hexdigest() == (
        "3be60100a9362957bcf1dcbd7cd5e76ffae24cf968b9f63fe948c2dd1a4bf5bb"
    )


def test_survey_rejects_ambiguous_duplicate_character_presence(tmp_path: Path) -> None:
    data_root = tmp_path / "official"
    _write_official_story(
        data_root,
        "10",
        _story({"Target": "realistic_human"}, {2: ["Target"], 8: ["Target"]}),
        reference_names=("Target",),
    )
    donor = _story(
        {"Donor": "realistic_human"},
        {3: ["Donor", "Donor"], 9: ["Donor"]},
    )
    _write_official_story(data_root, "20", donor, reference_names=("Donor",))
    targets = _write_target_inputs(
        tmp_path / "targets",
        data_root,
        story_id="10",
        character="Target",
        source_shot=2,
        target_shot=8,
    )

    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=tmp_path / "survey.json",
    )

    assert survey["candidates"] == []
    donor_rows = [
        row for row in survey["rejections"] if row["donor_char_id"] == "Donor"
    ]
    assert "ambiguous_duplicate_identity" in donor_rows[0]["reasons"]


def test_survey_records_same_story_candidates_as_rejected(tmp_path: Path) -> None:
    data_root = tmp_path / "official"
    story = _story(
        {"Target": "realistic_human", "Wrong": "realistic_human"},
        {
            2: ["Target"],
            3: ["Wrong"],
            8: ["Target"],
            9: ["Wrong"],
        },
    )
    _write_official_story(
        data_root, "10", story, reference_names=("Target", "Wrong")
    )
    targets = _write_target_inputs(
        tmp_path / "targets",
        data_root,
        story_id="10",
        character="Target",
        source_shot=2,
        target_shot=8,
    )

    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=tmp_path / "survey.json",
    )

    wrong = [row for row in survey["rejections"] if row["donor_char_id"] == "Wrong"]
    assert wrong[0]["reasons"] == ["same_story"]


def test_survey_rejects_target_manifest_path_escape(tmp_path: Path) -> None:
    data_root = tmp_path / "official"
    story_path = _write_official_story(
        data_root,
        "10",
        _story({"Target": "realistic_human"}, {2: ["Target"], 8: ["Target"]}),
        reference_names=("Target",),
    )
    escaped = tmp_path / "escaped.json"
    escaped.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_commit": "dataset-commit",
                "event_id": "target_event",
                "story_id": "10",
                "character_name": "Target",
                "source_shot": 2,
                "target_shot": 8,
                "official_story": {
                    "path": "10/story.json",
                    "sha256": sha256_file(story_path),
                },
                "reference_path": "10/image/Target/00.jpg",
                "reference_sha256": sha256_file(
                    data_root / "10" / "image" / "Target" / "00.jpg"
                ),
            }
        ),
        encoding="utf-8",
    )
    target_root = tmp_path / "targets"
    target_root.mkdir()
    target_inputs = target_root / "manifest.json"
    target_inputs.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_commit": "dataset-commit",
                "events": [
                    {
                        "event_id": "target_event",
                        "manifest_path": "../escaped.json",
                        "manifest_sha256": sha256_file(escaped),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target manifest path escapes .* root"):
        build_donor_candidate_survey(
            data_root=data_root,
            target_inputs_path=target_inputs,
            output_path=tmp_path / "survey.json",
        )


def test_survey_rejects_official_story_path_escape(tmp_path: Path) -> None:
    data_root = tmp_path / "official"
    outside_story = tmp_path / "story.json"
    outside_story.write_text(
        json.dumps(
            _story(
                {"Target": "realistic_human"},
                {2: ["Target"], 8: ["Target"]},
            )
        ),
        encoding="utf-8",
    )
    reference = data_root / "10" / "image" / "Target" / "00.jpg"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"reference")
    target_root = tmp_path / "targets"
    event_root = target_root / "target_event"
    event_root.mkdir(parents=True)
    event_manifest = {
        "schema_version": 1,
        "dataset_commit": "dataset-commit",
        "event_id": "target_event",
        "story_id": "10",
        "character_name": "Target",
        "source_shot": 2,
        "target_shot": 8,
        "official_story": {
            "path": "../story.json",
            "sha256": sha256_file(outside_story),
        },
        "reference_path": "10/image/Target/00.jpg",
        "reference_sha256": sha256_file(reference),
    }
    event_manifest_path = event_root / "manifest.json"
    event_manifest_path.write_text(json.dumps(event_manifest), encoding="utf-8")
    target_inputs = target_root / "manifest.json"
    target_inputs.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_commit": "dataset-commit",
                "events": [
                    {
                        "event_id": "target_event",
                        "manifest_path": "target_event/manifest.json",
                        "manifest_sha256": sha256_file(event_manifest_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="official story path escapes .* root"):
        build_donor_candidate_survey(
            data_root=data_root,
            target_inputs_path=target_inputs,
            output_path=tmp_path / "survey.json",
        )


def test_freeze_materializes_exactly_three_seed_zero_donor_events(
    tmp_path: Path,
) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    assert len(survey["candidates"]) == 3
    review_path = tmp_path / "review.json"
    _write_review_for_survey(review_path, survey_path, survey)

    selection = freeze_donor_selection(
        data_root=data_root,
        target_inputs_path=targets,
        survey_path=survey_path,
        review_path=review_path,
        output_root=tmp_path / "selection",
    )

    assert selection["donor_seed"] == 0
    assert len(selection["events"]) == 3
    assert {row["target_event_id"] for row in selection["events"]} == {
        "vistory79_song_yuchen_s2_s8",
        "vistory15_gu_zhenzhen_s8_s20",
        "vistory16_chen_father_s1_s10",
    }
    for row in selection["events"]:
        audit = next(
            item
            for item in selection["candidate_audit"]
            if item["candidate_id"] == row["candidate_id"]
        )
        assert audit["approved"] is True
        assert audit["review"]["donor_source_visible"] is True
        assert audit["review"]["donor_read_check_visible"] is True
        event_root = tmp_path / "selection" / row["target_event_id"]
        assert (event_root / "story.json").is_file()
        assert (event_root / "event.json").is_file()
        assert (event_root / "reference.jpg").is_file()
        assert (event_root / "manifest.json").is_file()


def test_freeze_rejects_mismatched_presentation_class(tmp_path: Path) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    review = _write_review_for_survey(review_path, survey_path, survey)
    review["reviews"][0]["donor_presentation_class"] = "female"
    review_path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError, match="presentation class"):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=tmp_path / "selection",
        )


def test_freeze_rejects_mismatched_dominant_colour(tmp_path: Path) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    review = _write_review_for_survey(review_path, survey_path, survey)
    review["reviews"][0]["donor_dominant_colour"] = "blue"
    review_path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError, match="dominant colour"):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=tmp_path / "selection",
        )


@pytest.mark.parametrize(
    "field", ["donor_source_visible", "donor_read_check_visible"]
)
def test_freeze_requires_reviewed_visibility(field: str, tmp_path: Path) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    review = _write_review_for_survey(review_path, survey_path, survey)
    review["reviews"][0][field] = False
    review_path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError, match="visible"):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=tmp_path / "selection",
        )


def test_freeze_rejects_nonzero_selection_seed_before_materializing(
    tmp_path: Path,
) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    survey["selection_seed"] = 1
    survey_path.write_text(json.dumps(survey), encoding="utf-8")
    review_path = tmp_path / "review.json"
    _write_review_for_survey(review_path, survey_path, survey)

    with pytest.raises(ValueError, match="selection_seed"):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=tmp_path / "selection",
        )
    assert not (tmp_path / "selection").exists()


def test_freeze_rejects_zero_approved_candidates_for_a_target(tmp_path: Path) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    review = _write_review_for_survey(review_path, survey_path, survey)
    review["reviews"][0]["approved"] = False
    review_path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError, match="zero approved donor candidates"):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=tmp_path / "selection",
        )


def test_freeze_rejects_undeclared_multiple_matches(tmp_path: Path) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    _write_official_story(
        data_root,
        "30",
        _story(
            {"ExtraDonor": "realistic_human"},
            {1: ["ExtraDonor"], 7: ["ExtraDonor"]},
        ),
        reference_names=("ExtraDonor",),
    )
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    _write_review_for_survey(review_path, survey_path, survey)

    with pytest.raises(ValueError, match="multiple approved.*tie_group"):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=tmp_path / "selection",
        )


def test_freeze_breaks_only_an_explicit_tie_deterministically(tmp_path: Path) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    _write_official_story(
        data_root,
        "30",
        _story(
            {"ExtraDonor": "realistic_human"},
            {1: ["ExtraDonor"], 7: ["ExtraDonor"]},
        ),
        reference_names=("ExtraDonor",),
    )
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    review = _write_review_for_survey(review_path, survey_path, survey)
    tied_target = "vistory79_song_yuchen_s2_s8"
    tied_ids = []
    for row in review["reviews"]:
        if row["target_event_id"] == tied_target:
            row["tie_group"] = "reviewed_visual_tie"
            tied_ids.append(row["candidate_id"])
    assert len(tied_ids) == 2
    review_path.write_text(json.dumps(review), encoding="utf-8")

    first = freeze_donor_selection(
        data_root=data_root,
        target_inputs_path=targets,
        survey_path=survey_path,
        review_path=review_path,
        output_root=tmp_path / "selection-a",
    )
    second = freeze_donor_selection(
        data_root=data_root,
        target_inputs_path=targets,
        survey_path=survey_path,
        review_path=review_path,
        output_root=tmp_path / "selection-b",
    )

    def chosen(selection: dict) -> str:
        return next(
            row["candidate_id"]
            for row in selection["events"]
            if row["target_event_id"] == tied_target
        )

    assert chosen(first) == chosen(second)
    assert chosen(first) in tied_ids


def test_freeze_requires_named_human_reviewer(tmp_path: Path) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    review = _write_review_for_survey(review_path, survey_path, survey)
    review["reviews"][0]["reviewer"] = ""
    review_path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError, match="reviewer"):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=tmp_path / "selection",
        )


@pytest.mark.parametrize(
    ("target_tag", "donor_tag", "donor_appearances", "reference", "reason"),
    [
        (
            "realistic_human",
            "unrealistic_human",
            {3: ["Donor"], 9: ["Donor"]},
            True,
            "style_class_mismatch",
        ),
        (
            "unrealistic_human",
            "non_human",
            {3: ["Donor"], 9: ["Donor"]},
            True,
            "official_tag_mismatch",
        ),
        (
            "realistic_human",
            "realistic_human",
            {3: ["Donor", "Extra"], 9: ["Donor"]},
            True,
            "source_character_count_mismatch",
        ),
        (
            "realistic_human",
            "realistic_human",
            {3: ["Donor"], 12: ["Donor"]},
            True,
            "gap_bucket_mismatch",
        ),
        (
            "realistic_human",
            "realistic_human",
            {3: ["Donor"], 9: ["Donor"]},
            False,
            "missing_reference",
        ),
    ],
)
def test_survey_records_each_hard_constraint_rejection(
    target_tag: str,
    donor_tag: str,
    donor_appearances: dict[int, list[str]],
    reference: bool,
    reason: str,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "official"
    _write_official_story(
        data_root,
        "10",
        _story({"Target": target_tag}, {2: ["Target"], 8: ["Target"]}),
        reference_names=("Target",),
    )
    donor_characters = {"Donor": donor_tag}
    if any("Extra" in names for names in donor_appearances.values()):
        donor_characters["Extra"] = donor_tag
    _write_official_story(
        data_root,
        "20",
        _story(donor_characters, donor_appearances),
        reference_names=("Donor",) if reference else (),
    )
    targets = _write_target_inputs(
        tmp_path / "targets",
        data_root,
        story_id="10",
        character="Target",
        source_shot=2,
        target_shot=8,
    )

    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=tmp_path / "survey.json",
    )

    donor_rows = [
        row for row in survey["rejections"] if row["donor_char_id"] == "Donor"
    ]
    assert reason in donor_rows[0]["reasons"]


def test_survey_bytes_and_candidate_order_are_deterministic(tmp_path: Path) -> None:
    data_root = tmp_path / "official"
    _write_official_story(
        data_root,
        "10",
        _story({"Target": "realistic_human"}, {2: ["Target"], 8: ["Target"]}),
        reference_names=("Target",),
    )
    for story_id, name in (("30", "Zulu"), ("20", "Alpha")):
        _write_official_story(
            data_root,
            story_id,
            _story({name: "realistic_human"}, {3: [name], 9: [name]}),
            reference_names=(name,),
        )
    targets = _write_target_inputs(
        tmp_path / "targets",
        data_root,
        story_id="10",
        character="Target",
        source_shot=2,
        target_shot=8,
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first = build_donor_candidate_survey(
        data_root=data_root, target_inputs_path=targets, output_path=first_path
    )
    second = build_donor_candidate_survey(
        data_root=data_root, target_inputs_path=targets, output_path=second_path
    )

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first == second
    ids = [row["candidate_id"] for row in first["candidates"]]
    assert ids == sorted(ids)


def test_freeze_rejects_tampered_survey_candidate(tmp_path: Path) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    survey["candidates"][0]["source_prompt"] = "tampered"
    survey_path.write_text(json.dumps(survey), encoding="utf-8")
    review_path = tmp_path / "review.json"
    _write_review_for_survey(review_path, survey_path, survey)

    with pytest.raises(ValueError, match="does not match current official inputs"):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=tmp_path / "selection",
        )


def test_freeze_rejects_changed_official_reference(tmp_path: Path) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    _write_review_for_survey(review_path, survey_path, survey)
    candidate = survey["candidates"][0]
    (data_root / candidate["reference"]["path"]).write_bytes(b"changed")

    with pytest.raises(ValueError, match="does not match current official inputs"):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=tmp_path / "selection",
        )


def test_survey_and_freeze_never_overwrite_existing_outputs(tmp_path: Path) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    survey_bytes = survey_path.read_bytes()
    with pytest.raises(FileExistsError):
        build_donor_candidate_survey(
            data_root=data_root,
            target_inputs_path=targets,
            output_path=survey_path,
        )
    assert survey_path.read_bytes() == survey_bytes

    review_path = tmp_path / "review.json"
    _write_review_for_survey(review_path, survey_path, survey)
    output_root = tmp_path / "selection"
    freeze_donor_selection(
        data_root=data_root,
        target_inputs_path=targets,
        survey_path=survey_path,
        review_path=review_path,
        output_root=output_root,
    )
    selection_path = output_root / "selection.json"
    selection_bytes = selection_path.read_bytes()
    with pytest.raises(FileExistsError):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=output_root,
        )
    assert selection_path.read_bytes() == selection_bytes


def test_horizon_buckets_are_frozen() -> None:
    assert [horizon_bucket(value) for value in (5, 6, 7)] == ["5-7"] * 3
    assert [horizon_bucket(value) for value in (8, 9, 10)] == ["8-10"] * 3
    assert [horizon_bucket(value) for value in (11, 12, 13)] == ["11-13"] * 3
    with pytest.raises(ValueError, match="unsupported donor horizon"):
        horizon_bucket(4)
    with pytest.raises(ValueError, match="unsupported donor horizon"):
        horizon_bucket(14)


def test_survey_rejects_casefold_duplicate_character_names(tmp_path: Path) -> None:
    data_root = tmp_path / "official"
    _write_official_story(
        data_root,
        "10",
        _story({"Target": "realistic_human"}, {2: ["Target"], 8: ["Target"]}),
        reference_names=("Target",),
    )
    donor = _story(
        {"Donor": "realistic_human", "donor": "realistic_human"},
        {3: ["Donor"], 9: ["Donor"]},
    )
    _write_official_story(data_root, "20", donor, reference_names=("Donor",))
    targets = _write_target_inputs(
        tmp_path / "targets",
        data_root,
        story_id="10",
        character="Target",
        source_shot=2,
        target_shot=8,
    )

    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=tmp_path / "survey.json",
    )

    donor_rows = [
        row for row in survey["rejections"] if row["donor_char_id"] == "Donor"
    ]
    assert "ambiguous_duplicate_identity" in donor_rows[0]["reasons"]


@pytest.mark.parametrize("arguments", [["--help"], ["survey", "--help"], ["freeze", "--help"]])
def test_prepare_vistory_donors_cli_help_is_directly_executable(
    arguments: list[str],
) -> None:
    repository_root = Path(__file__).parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(repository_root / "tools" / "prepare_vistory_donors.py"),
            *arguments,
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_freeze_rejects_review_fields_outside_strict_schema(tmp_path: Path) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    review = _write_review_for_survey(review_path, survey_path, survey)
    review["post_generation_score"] = 0.99
    review_path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError, match="review document fields"):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=tmp_path / "selection",
        )


def test_freeze_rejects_list_of_pairs_as_review_row(tmp_path: Path) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    review = _write_review_for_survey(review_path, survey_path, survey)
    review["reviews"][0] = list(review["reviews"][0].items())
    review_path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError, match="review row must be a JSON object"):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=tmp_path / "selection",
        )


def test_rejected_candidates_keep_official_hashes_and_review_prompts(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "official"
    _write_official_story(
        data_root,
        "10",
        _story({"Target": "realistic_human"}, {2: ["Target"], 8: ["Target"]}),
        reference_names=("Target",),
    )
    donor_story = _story(
        {"Donor": "unrealistic_human"}, {3: ["Donor"], 9: ["Donor"]}
    )
    donor_story_path = _write_official_story(
        data_root, "20", donor_story, reference_names=("Donor",)
    )
    donor_reference = data_root / "20" / "image" / "Donor" / "00.jpg"
    targets = _write_target_inputs(
        tmp_path / "targets",
        data_root,
        story_id="10",
        character="Target",
        source_shot=2,
        target_shot=8,
    )

    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=tmp_path / "survey.json",
    )

    row = next(row for row in survey["rejections"] if row["donor_story_id"] == "20")
    assert row["official_story"]["sha256"] == sha256_file(donor_story_path)
    assert row["reference"]["sha256"] == sha256_file(donor_reference)
    assert row["source_prompt"] == "setting 3 perspective 3. static 3"
    assert row["read_prompt"] == "setting 9 perspective 9. static 9"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("dataset_commit", "dataset_commit"),
        ("survey_sha256", "survey_sha256"),
        ("candidate_id", "candidate_id"),
    ],
)
def test_freeze_rejects_stale_review_provenance(
    mutation: str, message: str, tmp_path: Path
) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    review = _write_review_for_survey(review_path, survey_path, survey)
    if mutation == "candidate_id":
        review["reviews"][0]["candidate_id"] = "0" * 64
    else:
        review[mutation] = "stale"
    review_path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=tmp_path / "selection",
        )


def test_freeze_rejects_duplicate_review_rows(tmp_path: Path) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    review = _write_review_for_survey(review_path, survey_path, survey)
    duplicate = dict(review["reviews"][0])
    duplicate["tie_group"] = "declared"
    review["reviews"][0]["tie_group"] = "declared"
    review["reviews"].append(duplicate)
    review_path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate review candidate_id"):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=tmp_path / "selection",
        )


def test_freeze_rejects_an_existing_output_root_without_changing_it(
    tmp_path: Path,
) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    _write_review_for_survey(review_path, survey_path, survey)
    output_root = tmp_path / "selection"
    output_root.mkdir()
    sentinel = output_root / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output root already exists"):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=output_root,
        )

    assert list(output_root.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("failure_point", ["second_bundle", "selection"])
def test_freeze_write_failure_publishes_nothing_and_cleans_staging(
    failure_point: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    _write_review_for_survey(review_path, survey_path, survey)
    output_root = tmp_path / "selection"
    original = donor_module.write_json_no_clobber
    event_writes = 0

    def failing_write(path: Path, value: object) -> None:
        nonlocal event_writes
        if path.name == "event.json":
            event_writes += 1
        if failure_point == "second_bundle" and event_writes == 2:
            raise OSError("injected second bundle failure")
        if failure_point == "selection" and path.name == "selection.json":
            raise OSError("injected selection failure")
        original(path, value)

    monkeypatch.setattr(donor_module, "write_json_no_clobber", failing_write)

    with pytest.raises(OSError, match="injected"):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=output_root,
        )

    assert not output_root.exists()
    assert list(tmp_path.glob(".selection.*.tmp")) == []


def test_freeze_records_unapproved_review_reasons_without_selecting_it(
    tmp_path: Path,
) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    _write_official_story(
        data_root,
        "30",
        _story(
            {"RejectedDonor": "realistic_human"},
            {1: ["RejectedDonor"], 7: ["RejectedDonor"]},
        ),
        reference_names=("RejectedDonor",),
    )
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    rejected_candidate = next(
        row for row in survey["candidates"] if row["donor_story_id"] == "30"
    )
    review_path = tmp_path / "review.json"
    review = _write_review_for_survey(review_path, survey_path, survey)
    rejected_review = next(
        row
        for row in review["reviews"]
        if row["candidate_id"] == rejected_candidate["candidate_id"]
    )
    rejected_review.update(
        {
            "donor_presentation_class": "female",
            "donor_dominant_colour": "blue",
            "donor_source_visible": False,
            "donor_read_check_visible": False,
            "approved": False,
        }
    )
    review_path.write_text(json.dumps(review), encoding="utf-8")

    selection = freeze_donor_selection(
        data_root=data_root,
        target_inputs_path=targets,
        survey_path=survey_path,
        review_path=review_path,
        output_root=tmp_path / "selection",
    )

    disposition = next(
        row
        for row in selection["candidate_audit"]
        if row["candidate_id"] == rejected_candidate["candidate_id"]
    )
    assert disposition["approved"] is False
    assert disposition["reasons"] == [
        "presentation_mismatch",
        "colour_mismatch",
        "source_not_visible",
        "read_not_visible",
        "human_rejected",
    ]
    assert rejected_candidate["candidate_id"] not in {
        row["candidate_id"] for row in selection["events"]
    }
    assert len(selection["candidate_audit"]) == len(survey["candidates"]) + len(
        survey["rejections"]
    )
    assert all(isinstance(row["reasons"], list) for row in selection["candidate_audit"])
    assert not {
        "candidates",
        "rejections",
        "review_dispositions",
    }.intersection(selection)


def test_freeze_is_byte_deterministic_across_output_roots(tmp_path: Path) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    _write_review_for_survey(review_path, survey_path, survey)
    first_root = tmp_path / "selection-a"
    second_root = tmp_path / "selection-b"

    freeze_donor_selection(
        data_root=data_root,
        target_inputs_path=targets,
        survey_path=survey_path,
        review_path=review_path,
        output_root=first_root,
    )
    freeze_donor_selection(
        data_root=data_root,
        target_inputs_path=targets,
        survey_path=survey_path,
        review_path=review_path,
        output_root=second_root,
    )

    first_files = {
        path.relative_to(first_root).as_posix(): path.read_bytes()
        for path in first_root.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second_root).as_posix(): path.read_bytes()
        for path in second_root.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files
    event_path = next(first_root.glob("*/event.json"))
    event = json.loads(event_path.read_text(encoding="utf-8"))
    assert event["path_resolution"] == "event_parent"
    assert event["source_json_path"] == "story.json"
    assert event["reference_path"] == "reference.jpg"


@pytest.mark.parametrize(
    ("scope", "mutation"),
    [
        ("top", "missing"),
        ("top", "extra"),
        ("top", "wrong_type"),
        ("top", "schema_bool"),
        ("event", "missing"),
        ("event", "extra"),
        ("event", "wrong_type"),
        ("event", "schema_bool"),
        ("event", "missing_shot"),
        ("event", "unknown_character"),
        ("event", "bad_seeds"),
    ],
)
def test_survey_rejects_invalid_target_schema_as_value_error(
    scope: str, mutation: str, tmp_path: Path
) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    top = json.loads(targets.read_text(encoding="utf-8"))
    if scope == "top":
        if mutation == "missing":
            del top["events"]
        elif mutation == "extra":
            top["unexpected"] = True
        elif mutation == "schema_bool":
            top["schema_version"] = True
        else:
            top["events"] = {}
        targets.write_text(json.dumps(top), encoding="utf-8")
    else:
        entry = top["events"][0]
        event_path = targets.parent / entry["manifest_path"]
        event = json.loads(event_path.read_text(encoding="utf-8"))
        if mutation == "missing":
            del event["character_name"]
        elif mutation == "extra":
            event["unexpected"] = True
        elif mutation == "wrong_type":
            event["source_shot"] = "1"
        elif mutation == "schema_bool":
            event["schema_version"] = True
        elif mutation == "missing_shot":
            event["source_shot"] = 99
        elif mutation == "unknown_character":
            event["character_name"] = "Unknown"
        else:
            event["seeds"] = [0, True, 2]
        event_path.write_text(json.dumps(event), encoding="utf-8")
        entry["manifest_sha256"] = sha256_file(event_path)
        targets.write_text(json.dumps(top), encoding="utf-8")

    with pytest.raises(ValueError, match="target .* (schema|type)"):
        build_donor_candidate_survey(
            data_root=data_root,
            target_inputs_path=targets,
            output_path=tmp_path / "survey.json",
        )


def test_linux_publish_maps_atomic_race_to_file_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    class FakeRenameAt2:
        argtypes = None
        restype = None

        def __call__(self, *args: object) -> int:
            calls.append(args)
            return -1

    class FakeLibC:
        renameat2 = FakeRenameAt2()

    monkeypatch.setattr(donor_module.ctypes, "CDLL", lambda *args, **kwargs: FakeLibC())
    monkeypatch.setattr(donor_module.ctypes, "get_errno", lambda: errno.EEXIST)

    with pytest.raises(FileExistsError):
        donor_module._linux_rename_directory_no_clobber(
            tmp_path / "staging", tmp_path / "concurrent-empty-root"
        )

    assert calls[0][-1] == 1


def test_freeze_publish_race_preserves_concurrent_empty_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    _write_review_for_survey(review_path, survey_path, survey)
    output_root = tmp_path / "selection"

    def concurrent_publish(source: Path, destination: Path) -> None:
        destination.mkdir()
        raise FileExistsError("concurrent output root")

    monkeypatch.setattr(
        donor_module, "_publish_directory_no_clobber", concurrent_publish
    )

    with pytest.raises(FileExistsError, match="concurrent output root"):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=output_root,
        )

    assert output_root.is_dir()
    assert list(output_root.iterdir()) == []
    assert list(tmp_path.glob(".selection.*.tmp")) == []


def test_freeze_rejects_every_unreviewed_eligible_candidate(tmp_path: Path) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    _write_official_story(
        data_root,
        "30",
        _story(
            {"Unreviewed": "realistic_human"},
            {1: ["Unreviewed"], 7: ["Unreviewed"]},
        ),
        reference_names=("Unreviewed",),
    )
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    unreviewed = next(
        row for row in survey["candidates"] if row["donor_story_id"] == "30"
    )
    review_path = tmp_path / "review.json"
    review = _write_review_for_survey(review_path, survey_path, survey)
    review["reviews"] = [
        row
        for row in review["reviews"]
        if row["candidate_id"] != unreviewed["candidate_id"]
    ]
    review_path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError, match=unreviewed["candidate_id"]):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=tmp_path / "selection",
        )


def test_freeze_rejects_boolean_review_schema_version(tmp_path: Path) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    review_path = tmp_path / "review.json"
    review = _write_review_for_survey(review_path, survey_path, survey)
    review["schema_version"] = True
    review_path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError, match="review schema_version"):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=tmp_path / "selection",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", True), ("selection_seed", False)],
)
def test_freeze_rejects_boolean_survey_discriminators(
    field: str, value: bool, tmp_path: Path
) -> None:
    data_root, targets = _three_target_fixture(tmp_path)
    survey_path = tmp_path / "survey.json"
    survey = build_donor_candidate_survey(
        data_root=data_root,
        target_inputs_path=targets,
        output_path=survey_path,
    )
    survey[field] = value
    survey_path.write_text(json.dumps(survey), encoding="utf-8")
    review_path = tmp_path / "review.json"
    _write_review_for_survey(review_path, survey_path, survey)

    with pytest.raises(ValueError, match=field):
        freeze_donor_selection(
            data_root=data_root,
            target_inputs_path=targets,
            survey_path=survey_path,
            review_path=review_path,
            output_root=tmp_path / "selection",
        )
