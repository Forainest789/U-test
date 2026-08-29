from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import utest.vistory_reappearance as reappearance_module
from tools.prepare_slotmem_vistory_reappearance import main
from utest.prefix_contract import sha256_file
from utest.vistory_reappearance import (
    convert_event,
    frozen_target_event_ids,
    load_frozen_selection,
    prepare_dataset,
)


def _fake_story(character: str, appearances: tuple[int, ...], shot_count: int) -> dict:
    return {
        "Characters": {character: {"prompt_en": f"portrait of {character}"}},
        "Shots": [
            {
                "index": index,
                "Characters Appearing": {
                    "en": [character] if index in appearances else []
                },
                "Setting Description": {"en": f"setting {index}"},
                "Shot Perspective Design": {"en": f"perspective {index}"},
                "Static Shot Description": {"en": f"static {index}"},
            }
            for index in range(1, shot_count + 1)
        ],
    }


def _event_spec(character: str, source: int, target: int) -> dict:
    return {
        "story_id": "fixture",
        "event_id": "fixture_event",
        "character_name": character,
        "source_shot": source,
        "target_shot": target,
    }


def test_slice_reindexes_source_absence_target() -> None:
    official = _fake_story("Ana", appearances=(2, 5), shot_count=5)
    derived, event = convert_event(official, _event_spec("Ana", 2, 5))

    assert [chunk["official_shot_idx"] for chunk in derived["chunks"]] == [2, 3, 4, 5]
    assert event["source_chunk_idx"] == 0
    assert event["target_chunk_idx"] == 3
    assert event["horizon"] == 3
    assert derived["chunks"][0]["character_list"] == ["Ana"]
    assert all("Ana" not in chunk["character_list"] for chunk in derived["chunks"][1:3])
    assert derived["chunks"][3]["character_list"] == ["Ana"]


def test_conversion_rejects_presence_inside_absence() -> None:
    official = _fake_story("Ana", appearances=(2, 4, 5), shot_count=5)

    with pytest.raises(ValueError, match="full absence"):
        convert_event(official, _event_spec("Ana", 2, 5))


def test_conversion_preserves_character_order_and_prompt_fields() -> None:
    official = _fake_story("Ana", appearances=(1, 3), shot_count=3)
    official["Characters"]["Bo"] = {"prompt_en": "portrait of Bo"}
    official["Shots"][2]["Characters Appearing"]["en"] = ["Bo", "Ana"]

    derived, _ = convert_event(official, _event_spec("Ana", 1, 3))

    assert derived["chunks"][2]["character_list"] == ["Bo", "Ana"]
    assert derived["chunks"][2]["content"] == "setting 3 perspective 3. static 3"


@pytest.mark.parametrize(
    ("appearances", "message"),
    [((5,), "source"), ((2,), "first reappearance")],
)
def test_conversion_requires_subject_at_interval_endpoints(
    appearances: tuple[int, ...], message: str
) -> None:
    official = _fake_story("Ana", appearances=appearances, shot_count=5)

    with pytest.raises(ValueError, match=message):
        convert_event(official, _event_spec("Ana", 2, 5))


def test_conversion_requires_nonempty_absence_interval() -> None:
    official = _fake_story("Ana", appearances=(2, 3), shot_count=3)

    with pytest.raises(ValueError, match="absence interval"):
        convert_event(official, _event_spec("Ana", 2, 3))


def test_prepare_rejects_story_hash_before_parsing(tmp_path) -> None:
    story_path = tmp_path / "data" / "fixture" / "story.json"
    story_path.parent.mkdir(parents=True)
    story_path.write_text("not valid json", encoding="utf-8")
    selection = {
        "schema_version": 1,
        "task_id": "fixture_task",
        "dataset_commit": "dataset",
        "evaluator_commit": "evaluator",
        "seeds": [0, 1, 2],
        "events": [
            {
                **_event_spec("Ana", 2, 5),
                "story_sha256": "0" * 64,
            }
        ],
    }

    with pytest.raises(ValueError, match="story SHA-256 mismatch"):
        prepare_dataset(tmp_path / "data", tmp_path / "output", selection)


def test_prepare_rejects_malformed_story_hash_before_reading_file(tmp_path) -> None:
    selection = {
        "schema_version": 1,
        "task_id": "fixture_task",
        "dataset_commit": "dataset",
        "evaluator_commit": "evaluator",
        "seeds": [0, 1, 2],
        "events": [
            {
                **_event_spec("Ana", 2, 5),
                "story_sha256": "A" * 63,
            }
        ],
    }

    with pytest.raises(ValueError, match="64 hexadecimal"):
        prepare_dataset(tmp_path / "missing", tmp_path / "output", selection)


def test_prepare_accepts_bom_story_when_raw_bytes_hash_matches(tmp_path) -> None:
    data_root = tmp_path / "data"
    story_path = data_root / "fixture" / "story.json"
    story_path.parent.mkdir(parents=True)
    story_path.write_bytes(
        json.dumps(_fake_story("Ana", appearances=(2, 5), shot_count=5)).encode(
            "utf-8-sig"
        )
    )
    reference = data_root / "fixture" / "image" / "Ana" / "00.jpg"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"reference")
    selection = {
        "schema_version": 1,
        "task_id": "fixture_task",
        "dataset_commit": "dataset",
        "evaluator_commit": "evaluator",
        "seeds": [0, 1, 2],
        "events": [
            {
                **_event_spec("Ana", 2, 5),
                "story_sha256": sha256_file(story_path),
            }
        ],
    }

    report = prepare_dataset(data_root, tmp_path / "output", selection)

    assert report["events"][0]["event_id"] == "fixture_event"


def test_prepare_writes_hashed_portable_event_bundle(tmp_path) -> None:
    data_root = tmp_path / "data"
    story_path = data_root / "fixture" / "story.json"
    story_path.parent.mkdir(parents=True)
    story_path.write_text(
        json.dumps(_fake_story("Ana", appearances=(2, 5), shot_count=5)),
        encoding="utf-8",
    )
    reference_dir = data_root / "fixture" / "image" / "Ana"
    reference_dir.mkdir(parents=True)
    (reference_dir / "00.jpg").write_bytes(b"initial reference")
    (reference_dir / "01.jpg").write_bytes(b"second reference")
    selection = {
        "schema_version": 1,
        "task_id": "fixture_task",
        "dataset_commit": "dataset",
        "evaluator_commit": "evaluator",
        "seeds": [0, 1, 2],
        "events": [
            {
                **_event_spec("Ana", 2, 5),
                "story_sha256": sha256_file(story_path),
            }
        ],
    }

    report = prepare_dataset(data_root, tmp_path / "output", selection)

    event_root = tmp_path / "output" / "fixture_event"
    top_level_manifest = json.loads(
        (tmp_path / "output" / "manifest.json").read_text(encoding="utf-8")
    )
    derived = json.loads((event_root / "story.json").read_text(encoding="utf-8"))
    event = json.loads((event_root / "event.json").read_text(encoding="utf-8"))
    manifest = json.loads((event_root / "manifest.json").read_text(encoding="utf-8"))
    assert [chunk["official_shot_idx"] for chunk in derived["chunks"]] == [2, 3, 4, 5]
    assert event["source_chunk_idx"] == 0
    assert event["reference_path"] == str((reference_dir / "00.jpg").resolve())
    assert manifest["official_story"] == {
        "path": "fixture/story.json",
        "sha256": sha256_file(story_path),
    }
    assert manifest["reference_path"] == "fixture/image/Ana/00.jpg"
    assert manifest["reference_sha256"] == sha256_file(reference_dir / "00.jpg")
    assert manifest["reference_images"] == [
        {
            "path": "fixture/image/Ana/00.jpg",
            "sha256": sha256_file(reference_dir / "00.jpg"),
        },
        {
            "path": "fixture/image/Ana/01.jpg",
            "sha256": sha256_file(reference_dir / "01.jpg"),
        },
    ]
    assert manifest["outputs"]["story"]["sha256"] == sha256_file(event_root / "story.json")
    assert manifest["outputs"]["event"]["sha256"] == sha256_file(event_root / "event.json")
    assert report["events"] == [
        {
            "event_id": "fixture_event",
            "manifest_path": "fixture_event/manifest.json",
            "manifest_sha256": sha256_file(event_root / "manifest.json"),
        }
    ]
    assert top_level_manifest == report


def test_frozen_selection_keeps_approved_events_shots_and_hashes() -> None:
    selection = load_frozen_selection()
    events = selection["events"]

    assert selection["dataset_commit"] == "92f845531b67e97a67ae04b256ec5d8c020e8341"
    assert selection["evaluator_commit"] == "b44ec9108668cc2bcc8c5280886b235e9fb8bea9"
    assert selection["seeds"] == [0, 1, 2]
    assert len(events) == 3
    assert events[0] == {
        "story_id": "79",
        "event_id": "vistory79_song_yuchen_s2_s8",
        "character_name": "Song Yuchen",
        "source_shot": 2,
        "target_shot": 8,
        "story_sha256": "4298F6EFAA5F2D4A9D69C86E169E0167CE324334F656033A6D692CAFD9484109",
    }
    assert events[2] == {
        "story_id": "16",
        "event_id": "vistory16_chen_father_s1_s10",
        "character_name": "Chen Sihan's Father",
        "source_shot": 1,
        "target_shot": 10,
        "story_sha256": "6B1AD31634E5DA0108ACD51B16DA2E7F29B202858FCC5D0E556F4BEDB22D005E",
    }
    assert events[1]["event_id"] == selection["replacement_selection"]["selected_event_id"]
    assert events[1]["character_name"] != "Gu Zhenzhen"
    assert "vistory15_gu_zhenzhen_s8_s20" not in frozen_target_event_ids()


def test_frozen_selection_story_hashes_are_64_digit_uppercase_hex() -> None:
    selection = load_frozen_selection()

    assert all(
        re.fullmatch(r"[0-9A-F]{64}", event["story_sha256"])
        for event in selection["events"]
    )


def _write_mutated_frozen_selection(tmp_path: Path, selection: dict) -> Path:
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    return path


def test_frozen_selection_rejects_a_changed_retained_position(tmp_path: Path) -> None:
    selection = load_frozen_selection()
    selection["events"][0]["target_shot"] = 9

    with pytest.raises(ValueError, match="retained frozen event changed"):
        load_frozen_selection(_write_mutated_frozen_selection(tmp_path, selection))


def test_frozen_selection_rejects_an_original_replacement_identity(tmp_path: Path) -> None:
    selection = load_frozen_selection()
    selection["events"][1].update(
        story_id="15",
        event_id="vistory15_gu_zhenzhen_s8_s20",
        character_name="Gu Zhenzhen",
        source_shot=8,
        target_shot=20,
    )
    selection["replacement_selection"]["selected_event_id"] = selection["events"][1]["event_id"]
    selection["replacement_selection"]["horizon"] = 12
    selection["replacement_selection"]["horizon_distance"] = 0

    with pytest.raises(ValueError, match="exclude all three original target identities"):
        load_frozen_selection(_write_mutated_frozen_selection(tmp_path, selection))


def test_frozen_selection_rejects_stale_replacement_binding(tmp_path: Path) -> None:
    selection = load_frozen_selection()
    selection["replacement_selection"]["selected_event_id"] = "stale"

    with pytest.raises(ValueError, match="selected event mismatch"):
        load_frozen_selection(_write_mutated_frozen_selection(tmp_path, selection))


@pytest.mark.parametrize(
    "path_attribute",
    [
        "FROZEN_SELECTION_PATH",
        "FROZEN_SURVEY_PATH",
        "FROZEN_REVIEW_PATH",
    ],
)
def test_default_authority_rejects_compact_rewritten_checked_in_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_attribute: str,
) -> None:
    authority_root = Path(__file__).parents[1] / "events"
    copied: dict[str, Path] = {}
    for source_name, attribute in (
        ("vistorybench_reappearance_v1.json", "FROZEN_SELECTION_PATH"),
        ("vistorybench_replacement_target_survey_v1.json", "FROZEN_SURVEY_PATH"),
        ("vistorybench_replacement_target_review_v1.json", "FROZEN_REVIEW_PATH"),
    ):
        target = tmp_path / source_name
        shutil.copyfile(authority_root / source_name, target)
        copied[attribute] = target
        monkeypatch.setattr(reappearance_module, attribute, target, raising=False)
    rewritten = copied[path_attribute]
    rewritten.write_text(
        json.dumps(json.loads(rewritten.read_text(encoding="utf-8")), separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checked-in .* SHA-256 mismatch"):
        load_frozen_selection()


def test_default_authority_hashes_and_parses_the_same_byte_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority_root = Path(__file__).parents[1] / "events"
    copied: dict[str, Path] = {}
    for source_name, attribute in (
        ("vistorybench_reappearance_v1.json", "FROZEN_SELECTION_PATH"),
        ("vistorybench_replacement_target_survey_v1.json", "FROZEN_SURVEY_PATH"),
        ("vistorybench_replacement_target_review_v1.json", "FROZEN_REVIEW_PATH"),
    ):
        target = tmp_path / source_name
        shutil.copyfile(authority_root / source_name, target)
        copied[attribute] = target
        monkeypatch.setattr(reappearance_module, attribute, target)

    selection_path = copied["FROZEN_SELECTION_PATH"]
    authority_paths = set(copied.values())
    read_counts = {path: 0 for path in authority_paths}
    attack_executed = False
    original_read_bytes = Path.read_bytes

    def read_then_swap(path: Path) -> bytes:
        nonlocal attack_executed
        raw = original_read_bytes(path)
        if path in authority_paths:
            read_counts[path] += 1
        if path == selection_path and read_counts[path] == 1:
            forged = json.loads(raw.decode("utf-8"))
            forged["events"][1].update(
                story_id="41",
                event_id="vistory41_mallory_s15_s21",
                character_name="Mallory",
            )
            forged["replacement_selection"]["selected_event_id"] = forged["events"][1][
                "event_id"
            ]
            selection_path.write_text(json.dumps(forged), encoding="utf-8")
            attack_executed = True
        return raw

    monkeypatch.setattr(Path, "read_bytes", read_then_swap)

    selection = load_frozen_selection()

    assert attack_executed is True
    assert read_counts == {path: 1 for path in authority_paths}
    assert selection["events"][1]["event_id"] == "vistory42_bella_s15_s21"


def test_cli_prepares_dataset_from_explicit_selection(tmp_path, capsys) -> None:
    data_root = tmp_path / "data"
    story_path = data_root / "fixture" / "story.json"
    story_path.parent.mkdir(parents=True)
    story_path.write_text(
        json.dumps(_fake_story("Ana", appearances=(2, 5), shot_count=5)),
        encoding="utf-8",
    )
    reference = data_root / "fixture" / "image" / "Ana" / "00.jpg"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"reference")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "fixture_task",
                "dataset_commit": "dataset",
                "evaluator_commit": "evaluator",
                "seeds": [0, 1, 2],
                "events": [
                    {
                        **_event_spec("Ana", 2, 5),
                        "story_sha256": sha256_file(story_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    main(
        [
            "--data-root",
            str(data_root),
            "--output-root",
            str(tmp_path / "output"),
            "--selection",
            str(selection_path),
        ]
    )

    assert json.loads(capsys.readouterr().out)["task_id"] == "fixture_task"
    assert (tmp_path / "output" / "fixture_event" / "manifest.json").is_file()


def test_cli_is_directly_executable_from_repository_root() -> None:
    repository_root = Path(__file__).parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(repository_root / "tools" / "prepare_slotmem_vistory_reappearance.py"),
            "--help",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
