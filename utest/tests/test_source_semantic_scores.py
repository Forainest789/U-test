from __future__ import annotations

import math
import json
from pathlib import Path

import pytest
import torch

from utest.prefix_contract import sha256_file
from utest.source_semantic_scores import (
    SOURCE_SEMANTIC_FORMULA,
    produce_source_semantic_scores,
    validate_source_semantic_scores_file,
)
from utest.subject_subspace import source_metadata_semantic_groups
from utest.subject_subspace import canonical_json_sha256, capture_tensor_sha256


def _producer_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo = Path(__file__).resolve().parents[2]
    story = tmp_path / "story.json"
    story.write_text(
        json.dumps(
            {
                "characters": {"Ana": "red coat", "Bo": "blue hat"},
                "chunks": [
                    {"content": "station", "character_list": ["Ana", "Bo"]}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reference, model = tmp_path / "00.jpg", tmp_path / "model.pt"
    reference.write_bytes(b"reference")
    model.write_bytes(b"model")
    event = {
        "event_id": "event",
        "character_name": "Ana",
        "source_chunk_idx": 0,
        "target_chunk_idx": 1,
        "seed": 7,
        "source_seed": 7,
        "target_seed": 7,
        "source_json_path": str(story.resolve()),
        "reference_path": str(reference.resolve()),
        "reference_sha256": sha256_file(reference),
    }
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    raw = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    slots = torch.arange(96, dtype=torch.float32).reshape(32, 3)
    metadata = [
        {"char_id": "Ana", "inside_box": True, "tau_local": 0.0},
        {"char_id": "Ana", "inside_box": True, "tau_local": 1.0},
        {"char_id": "Ana", "inside_box": False, "tau_local": 0.0},
        {"char_id": "Bo", "inside_box": True, "tau_local": 0.0},
    ]
    attention = {
        "Ana": torch.full((24, 3), 1.0 / 3.0),
        "Bo": torch.ones((8, 1)),
    }
    rows = []
    for layer in range(16):
        row = {
            "character": "Ana",
            "bank": 0,
            "layer": layer,
            "raw_tokens": raw,
            "raw_token_meta": metadata,
            "encoded_slots": slots,
            "attention": attention,
        }
        row["tensor_shapes"] = {
            "raw_tokens": [4, 3],
            "encoded_slots": [32, 3],
            "attention": {"Ana": [24, 3], "Bo": [8, 1]},
        }
        row["sha256"] = {
            "raw_tokens": capture_tensor_sha256(raw),
            "raw_token_meta": canonical_json_sha256(metadata),
            "encoded_slots": capture_tensor_sha256(slots),
            "attention": capture_tensor_sha256(attention),
        }
        rows.append(row)
    provenance = {
        "source_json_path": str(story.resolve()),
        "source_json_sha256": sha256_file(story),
        "reference_file_sha256": sha256_file(reference),
        "fixed_reference_scope": "source_only",
        "source_seed": 7,
        "code_identity": {
            "infer_slotmem_sha256": sha256_file(repo / "infer_slotmem.py"),
            "mem_encoder_utils_sha256": sha256_file(repo / "mem_encoder_utils.py"),
        },
        "runtime_identity": {
            "python_version": "3.11",
            "torch_version": str(torch.__version__),
            "inference_args_sha256": "1" * 64,
        },
        "model_identity": {
            "high_noise": [{"path": str(model.resolve()), "sha256": sha256_file(model)}],
            "low_noise": [],
        },
    }
    summary = [
        {key: row[key] for key in ("character", "bank", "layer", "tensor_shapes", "sha256")}
        for row in rows
    ]
    canonical = {
        "schema_version": 1,
        "source_chunk_idx": 0,
        "target_evidence_read": False,
        "provenance": provenance,
        "captures": summary,
    }
    capture = {
        **canonical,
        "captures": rows,
        "canonical_artifact_sha256": canonical_json_sha256(canonical),
    }
    capture_path = tmp_path / "source_capture.pt"
    torch.save(capture, capture_path)
    return event_path, capture_path, tmp_path / "semantic_scores.json", repo


def _rewrite_capture(path: Path, capture: dict) -> None:
    canonical = {
        "schema_version": 1,
        "source_chunk_idx": 0,
        "target_evidence_read": False,
        "provenance": capture["provenance"],
        "captures": [
            {
                key: row[key]
                for key in ("character", "bank", "layer", "tensor_shapes", "sha256")
            }
            for row in capture["captures"]
        ],
    }
    capture["canonical_artifact_sha256"] = canonical_json_sha256(canonical)
    torch.save(capture, path)


def test_producer_writes_source_only_semantic_artifact(tmp_path: Path) -> None:
    event, capture, output, repo = _producer_fixture(tmp_path)

    artifact = produce_source_semantic_scores(
        event_path=event,
        source_capture_path=capture,
        output_path=output,
        repo_root=repo,
    )

    assert output.is_file()
    assert artifact["target_evidence_read"] is False
    assert artifact["producer"]["kind"] == "slotmem_source_semantic_token_scores"
    assert artifact["producer"]["formula"] == SOURCE_SEMANTIC_FORMULA
    assert artifact["producer"]["subject_char_id"] == "Ana"
    assert artifact["producer"]["source_seed"] == 7
    assert artifact["producer"]["source_json_sha256"] == sha256_file(
        Path(json.loads(event.read_text())["source_json_path"])
    )
    assert artifact["source_capture_sha256"] == sha256_file(capture)
    assert len(artifact["captures"]) == 16
    first = artifact["captures"][0]
    assert (first["bank"], first["layer"]) == (0, 0)
    assert first["groups"]["identity_name"] == [1.0, 1.0, 1.0, 0.0]
    assert first["groups"]["stable_attributes"] == pytest.approx(
        [1.0, math.exp(-0.5), 0.0, 0.0]
    )


def test_producer_refuses_to_overwrite_existing_scores(tmp_path: Path) -> None:
    event, capture, output, repo = _producer_fixture(tmp_path)
    output.write_bytes(b"frozen")

    with pytest.raises(FileExistsError):
        produce_source_semantic_scores(
            event_path=event,
            source_capture_path=capture,
            output_path=output,
            repo_root=repo,
        )

    assert output.read_bytes() == b"frozen"


def test_producer_rejects_incomplete_frozen_layers(tmp_path: Path) -> None:
    event, capture_path, output, repo = _producer_fixture(tmp_path)
    capture = torch.load(capture_path, map_location="cpu", weights_only=True)
    capture["captures"].pop()
    _rewrite_capture(capture_path, capture)

    with pytest.raises(ValueError, match="frozen layers"):
        produce_source_semantic_scores(
            event_path=event,
            source_capture_path=capture_path,
            output_path=output,
            repo_root=repo,
        )


def test_producer_rejects_raw_token_metadata_count_mismatch(tmp_path: Path) -> None:
    event, capture_path, output, repo = _producer_fixture(tmp_path)
    capture = torch.load(capture_path, map_location="cpu", weights_only=True)
    row = capture["captures"][0]
    row["raw_token_meta"] = row["raw_token_meta"][:-1]
    row["sha256"]["raw_token_meta"] = canonical_json_sha256(row["raw_token_meta"])
    _rewrite_capture(capture_path, capture)

    with pytest.raises(ValueError, match="shape mismatch"):
        produce_source_semantic_scores(
            event_path=event,
            source_capture_path=capture_path,
            output_path=output,
            repo_root=repo,
        )


def test_producer_rejects_changed_source_story(tmp_path: Path) -> None:
    event_path, capture, output, repo = _producer_fixture(tmp_path)
    event = json.loads(event_path.read_text(encoding="utf-8"))
    Path(event["source_json_path"]).write_text(
        json.dumps(
            {
                "characters": {"Ana": "green coat", "Bo": "blue hat"},
                "chunks": [
                    {"content": "station", "character_list": ["Ana", "Bo"]}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provenance"):
        produce_source_semantic_scores(
            event_path=event_path,
            source_capture_path=capture,
            output_path=output,
            repo_root=repo,
        )


def test_producer_rejects_changed_model_provenance(tmp_path: Path) -> None:
    event, capture_path, output, repo = _producer_fixture(tmp_path)
    capture = torch.load(capture_path, map_location="cpu", weights_only=True)
    model_path = Path(capture["provenance"]["model_identity"]["high_noise"][0]["path"])
    model_path.write_bytes(b"changed-model")

    with pytest.raises(ValueError, match="provenance SHA-256"):
        produce_source_semantic_scores(
            event_path=event,
            source_capture_path=capture_path,
            output_path=output,
            repo_root=repo,
        )


def test_producer_rejects_event_seed_mismatch(tmp_path: Path) -> None:
    event_path, capture, output, repo = _producer_fixture(tmp_path)
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["target_seed"] = 8
    event_path.write_text(json.dumps(event), encoding="utf-8")

    with pytest.raises(ValueError, match="seed"):
        produce_source_semantic_scores(
            event_path=event_path,
            source_capture_path=capture,
            output_path=output,
            repo_root=repo,
        )


@pytest.mark.parametrize(
    "forbidden_key", ["target_latents", "qstar_scores", "cids", "decoded_video"]
)
def test_producer_rejects_target_derived_capture_keys(
    tmp_path: Path, forbidden_key: str
) -> None:
    event, capture_path, output, repo = _producer_fixture(tmp_path)
    capture = torch.load(capture_path, map_location="cpu", weights_only=True)
    capture[forbidden_key] = "forbidden"
    torch.save(capture, capture_path)

    with pytest.raises(ValueError, match="target evidence"):
        produce_source_semantic_scores(
            event_path=event,
            source_capture_path=capture_path,
            output_path=output,
            repo_root=repo,
        )


def test_produced_file_validates_against_source_inputs(tmp_path: Path) -> None:
    event, capture, output, repo = _producer_fixture(tmp_path)
    produce_source_semantic_scores(
        event_path=event,
        source_capture_path=capture,
        output_path=output,
        repo_root=repo,
    )

    rows = validate_source_semantic_scores_file(
        event_path=event,
        source_capture_path=capture,
        scores_path=output,
        repo_root=repo,
    )

    assert set(rows) == {(0, layer) for layer in range(16)}


def test_scores_validator_rejects_wrong_vector_length(tmp_path: Path) -> None:
    event, capture, output, repo = _producer_fixture(tmp_path)
    artifact = produce_source_semantic_scores(
        event_path=event,
        source_capture_path=capture,
        output_path=output,
        repo_root=repo,
    )
    artifact["captures"][0]["groups"]["identity_name"].pop()
    artifact["canonical_artifact_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in artifact.items()
            if key != "canonical_artifact_sha256"
        }
    )
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="token count"):
        validate_source_semantic_scores_file(
            event_path=event,
            source_capture_path=capture,
            scores_path=tampered,
            repo_root=repo,
        )


@pytest.mark.parametrize(
    ("group", "index", "value"),
    [
        ("identity_name", 0, 0.0),
        ("stable_attributes", 1, 0.25),
        ("other_characters", 3, 0.0),
    ],
)
def test_scores_validator_recomputes_groups_from_source_metadata(
    tmp_path: Path, group: str, index: int, value: float
) -> None:
    event, capture, output, repo = _producer_fixture(tmp_path)
    artifact = produce_source_semantic_scores(
        event_path=event,
        source_capture_path=capture,
        output_path=output,
        repo_root=repo,
    )
    artifact["captures"][0]["groups"][group][index] = value
    artifact["canonical_artifact_sha256"] = canonical_json_sha256(
        {
            key: item
            for key, item in artifact.items()
            if key != "canonical_artifact_sha256"
        }
    )
    output.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="source metadata"):
        validate_source_semantic_scores_file(
            event_path=event,
            source_capture_path=capture,
            scores_path=output,
            repo_root=repo,
        )


def test_producer_is_byte_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    event, capture, first, repo = _producer_fixture(tmp_path)
    second = tmp_path / "semantic_scores_second.json"

    produce_source_semantic_scores(
        event_path=event,
        source_capture_path=capture,
        output_path=first,
        repo_root=repo,
    )
    produce_source_semantic_scores(
        event_path=event,
        source_capture_path=capture,
        output_path=second,
        repo_root=repo,
    )

    assert first.read_bytes() == second.read_bytes()


def test_producer_refuses_to_publish_an_invalid_subject_address(tmp_path: Path) -> None:
    event, capture_path, output, repo = _producer_fixture(tmp_path)
    capture = torch.load(capture_path, map_location="cpu", weights_only=True)
    capture["captures"][0]["character"] = "ana"
    _rewrite_capture(capture_path, capture)

    with pytest.raises(ValueError, match="character"):
        produce_source_semantic_scores(
            event_path=event,
            source_capture_path=capture_path,
            output_path=output,
            repo_root=repo,
        )

    assert not output.exists()


def test_source_metadata_groups_follow_frozen_formula() -> None:
    metadata = [
        {"char_id": "alice", "inside_box": True, "tau_local": 0.0},
        {"char_id": "alice", "inside_box": True, "tau_local": 1.0},
        {"char_id": "alice", "inside_box": False, "tau_local": 0.0},
        {"char_id": "bob", "inside_box": True, "tau_local": 0.0},
        {"char_id": "", "inside_box": False, "tau_local": 0.0},
    ]

    groups = source_metadata_semantic_groups(metadata, "alice")

    assert groups["identity_name"] == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert groups["stable_attributes"] == pytest.approx(
        [1.0, math.exp(-0.5), 0.0, 0.0, 0.0]
    )
    assert groups["other_characters"] == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert groups["action_scene"] == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_source_metadata_groups_reject_empty_subject() -> None:
    with pytest.raises(ValueError, match="subject"):
        source_metadata_semantic_groups(
            [{"char_id": "alice", "inside_box": True, "tau_local": 0.0}], ""
        )


@pytest.mark.parametrize(
    "metadata",
    [
        [{}],
        [{"char_id": 7, "inside_box": True, "tau_local": 0.0}],
        [{"char_id": "alice", "inside_box": 1, "tau_local": 0.0}],
        [{"char_id": "alice", "inside_box": True, "tau_local": "centre"}],
    ],
)
def test_source_metadata_groups_reject_malformed_metadata(metadata: list[dict]) -> None:
    with pytest.raises(ValueError, match="metadata"):
        source_metadata_semantic_groups(metadata, "alice")


@pytest.mark.parametrize("tau_local", [math.nan, math.inf, -math.inf])
def test_source_metadata_groups_reject_nonfinite_tau_local(tau_local: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        source_metadata_semantic_groups(
            [{"char_id": "alice", "inside_box": True, "tau_local": tau_local}],
            "alice",
        )


def test_source_metadata_groups_reject_missing_target_tokens() -> None:
    with pytest.raises(ValueError, match="target token"):
        source_metadata_semantic_groups(
            [{"char_id": "bob", "inside_box": True, "tau_local": 0.0}], "alice"
        )


def test_source_metadata_groups_reject_target_without_box_evidence() -> None:
    with pytest.raises(ValueError, match="inside-box"):
        source_metadata_semantic_groups(
            [{"char_id": "alice", "inside_box": False, "tau_local": 0.0}],
            "alice",
        )


def test_source_metadata_groups_accept_inside_box_when_centre_underflows() -> None:
    groups = source_metadata_semantic_groups(
        [{"char_id": "alice", "inside_box": True, "tau_local": 40.0}],
        "alice",
    )

    assert groups["identity_name"] == [1.0]
    assert groups["stable_attributes"] == [0.0]
