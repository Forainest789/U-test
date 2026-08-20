from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from utest.input_contract import validate_donor_bundle, validate_teacher_bundle


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _donor_fixture(tmp_path: Path):
    donor = tmp_path / "donor.pt"
    donor.write_bytes(b"serialized donor")
    event = {"story_id": "sample_5", "entity_uid": "sample_5::evan", "target_chunk_idx": 5}
    entry = {
        "target_story_id": "sample_5",
        "target_entity_uid": "sample_5::evan",
        "donor_story_id": "sample_77",
        "donor_entity_uid": "sample_77::luca",
        "payload_path": str(donor),
        "payload_sha256": _sha(donor),
        "payload_key": "Luca|0",
        "coarse_class": "person",
        "colour": "brown hair",
        "character_count": 1,
        "source_visible": True,
        "gap_bucket": "5+",
        "slot_shape": {"0": [64, 128]},
        "selection_seed": 0,
    }
    manifest = tmp_path / "donor_manifest.json"
    manifest.write_text(json.dumps({"pairs": [entry]}), encoding="utf-8")
    return donor, manifest, event


def test_donor_bundle_rejects_manifest_luca_with_payload_evan(tmp_path: Path) -> None:
    donor, manifest, event = _donor_fixture(tmp_path)
    payload = {
        "format": "slotmem_donor_payload_v2",
        "event": {
            "story_id": "sample_77",
            "entity_uid": "sample_77::evan",
            "character_name": "Evan",
        },
        "payloads": {"Evan|0": {"shape": [64, 128]}},
    }

    with pytest.raises(ValueError, match="donor_entity_uid.*payload event"):
        validate_donor_bundle(event, donor, manifest, loader=lambda _: payload)


def test_donor_bundle_binds_hash_identity_key_and_shape(tmp_path: Path) -> None:
    donor, manifest, event = _donor_fixture(tmp_path)

    class FakeTensor:
        shape = (64, 128)

    payload = {
        "format": "slotmem_donor_payload_v2",
        "event": {
            "story_id": "sample_77",
            "entity_uid": "sample_77::luca",
            "character_name": "Luca",
        },
        "payloads": {"Luca|0": FakeTensor()},
    }

    report = validate_donor_bundle(event, donor, manifest, loader=lambda _: payload)

    assert report["payload_sha256"] == _sha(donor)
    assert report["donor_entity_uid"] == "sample_77::luca"
    assert report["payload_key"] == "Luca|0"
    assert report["slot_shapes"] == {"0": [64, 128]}


def test_teacher_bundle_rejects_arm_rollout_and_accepts_held_out_video(tmp_path: Path) -> None:
    event = {"story_id": "sample_5", "target_chunk_idx": 5}
    arms = tmp_path / "arms"
    arm_video = arms / "correct" / "chunk_005.mp4"
    arm_video.parent.mkdir(parents=True)
    arm_video.write_bytes(b"arm output")
    manifest = tmp_path / "teacher.json"
    manifest.write_text(
        json.dumps(
            {
                "story_id": "sample_5",
                "target_chunk_idx": 5,
                "video_path": str(arm_video),
                "video_sha256": _sha(arm_video),
                "source_type": "held_out_real",
                "generated_by_arm": False,
                "generated_by_evaluated_model": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inside evaluated arm outputs"):
        validate_teacher_bundle(event, arm_video, manifest, arms_root=arms)

    teacher = tmp_path / "teachers" / "sample_5_chunk_005.mp4"
    teacher.parent.mkdir()
    teacher.write_bytes(b"independent teacher")
    manifest.write_text(
        json.dumps(
            {
                "story_id": "sample_5",
                "target_chunk_idx": 5,
                "video_path": str(teacher),
                "video_sha256": _sha(teacher),
                "source_type": "independent_teacher",
                "generated_by_arm": False,
                "generated_by_evaluated_model": False,
            }
        ),
        encoding="utf-8",
    )

    report = validate_teacher_bundle(event, teacher, manifest, arms_root=arms)
    assert report["video_sha256"] == _sha(teacher)
    assert report["source_type"] == "independent_teacher"
