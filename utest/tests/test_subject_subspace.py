from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch

from utest.content_audit import transform_slot_rows
from utest.prefix_contract import sha256_file
from utest.subject_subspace import (
    FROZEN_LAYER_GROUPS,
    SEMANTIC_GROUPS,
    aggregate_semantic_slot_scores,
    build_mask_manifest,
    build_semantic_score_artifact,
    canonical_json_sha256,
    capture_tensor_sha256,
    consensus_rank_indices,
    deterministic_random_indices,
    reference_agreement_scores,
    semantic_slot_scores,
    slot_attention_matrix,
    source_only_semantic_group_manifest,
    top_fraction_indices,
    validate_semantic_scores,
    validate_source_capture,
    visual_counterfactual_scores,
)
from utest.subject_subspace_audit import (
    install_subject_subspace,
    main as audit_main,
    validate_frozen_donor_artifact,
    validate_subject_payload,
    validate_subject_subspace_manifest,
)
from utest.subject_subspace_probe import freeze_subject_subspace, main as probe_main


def _reader_contract() -> tuple[dict, dict, dict]:
    event = {
        "event_id": "event",
        "character_name": "Ana",
        "source_chunk_idx": 0,
        "target_chunk_idx": 6,
        "target_seed": 0,
    }
    layers = {
        str(layer): torch.full((32, 3), float(layer), dtype=torch.float32)
        for layer in range(16)
    }
    rankings = {}
    for group, members in FROZEN_LAYER_GROUPS.items():
        rankings[f"bank_0/group_{group}"] = {
            "bank": 0,
            "layer_group": group,
            "member_layers": list(members),
            "source_payload_sha256_by_layer": {
                str(layer): capture_tensor_sha256(layers[str(layer)]) for layer in members
            },
            "semantic": list(range(32)),
            "visual_cf": None,
            "reference": None,
        }
    manifest = build_mask_manifest(
        inputs={"source_capture_sha256": "a" * 64},
        rankings=rankings,
        event=event,
        seed=0,
    )
    payload = {
        "tokens": {"__layerwise__": True, "layers": layers},
        "token_meta": {
            "__layerwise__": True,
            "layers": {
                layer: [{"slot": index} for index in range(32)] for layer in layers
            },
        },
    }
    return event, manifest, payload


def test_reader_contract_validates_manifest_and_source_payload_hashes() -> None:
    event, manifest, payload = _reader_contract()
    layers = validate_subject_subspace_manifest(manifest, event, seed=0)

    assert set(layers[0]) == {str(layer) for layer in range(16)}
    validate_subject_payload(payload, bank_idx=0, layers=layers[0])

    payload["tokens"]["layers"]["0"][0, 0] = -1
    with pytest.raises(ValueError, match="source payload SHA-256"):
        validate_subject_payload(payload, bank_idx=0, layers=layers[0])


def test_reader_manifest_rejects_target_evidence_even_with_a_valid_canonical_hash() -> None:
    event, manifest, _ = _reader_contract()
    manifest["inputs"]["target_loss_sha256"] = "f" * 64
    manifest["mask_manifest_sha256"] = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "mask_manifest_sha256"}
    )

    with pytest.raises(ValueError, match="target evidence"):
        validate_subject_subspace_manifest(manifest, event, seed=0)


def test_reader_manifest_rejects_nested_target_ranking_and_unbound_source_capture() -> None:
    event, manifest, _ = _reader_contract()
    manifest["layers"][0]["rankings"]["target_loss"] = [0]
    manifest["mask_manifest_sha256"] = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "mask_manifest_sha256"}
    )
    with pytest.raises(ValueError, match="target evidence"):
        validate_subject_subspace_manifest(manifest, event, seed=0)

    _, manifest, _ = _reader_contract()
    manifest["inputs"]["source_capture_sha256"] = "not-a-sha"
    manifest["mask_manifest_sha256"] = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "mask_manifest_sha256"}
    )
    with pytest.raises(ValueError, match="source capture SHA-256"):
        validate_subject_subspace_manifest(manifest, event, seed=0)


def test_reader_manifest_rejects_non_hex_payload_hash_and_missing_group() -> None:
    event, manifest, _ = _reader_contract()
    manifest["layers"][0]["source_payload_sha256_by_layer"]["0"] = "z" * 64
    manifest["mask_manifest_sha256"] = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "mask_manifest_sha256"}
    )
    with pytest.raises(ValueError, match="source payload SHA-256"):
        validate_subject_subspace_manifest(manifest, event, seed=0)

    _, manifest, _ = _reader_contract()
    manifest["layers"].pop()
    manifest["mask_manifest_sha256"] = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "mask_manifest_sha256"}
    )
    with pytest.raises(ValueError, match="missing a frozen layer group"):
        validate_subject_subspace_manifest(manifest, event, seed=0)


def test_subject_only_is_applied_only_after_live_payload_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event, manifest, payload = _reader_contract()

    class FakeBank:
        def get_memory_payload_for_read(self, char_id, bank_idx=0):
            return payload

    monkeypatch.setitem(
        sys.modules,
        "infer_slotmem",
        types.SimpleNamespace(RoleWiseSlotMemoryBank=FakeBank),
    )
    report = tmp_path / "audit.json"
    flush = install_subject_subspace(
        arm="subject_only",
        seed=0,
        manifest=manifest,
        event=event,
        report_path=report,
        event_file_sha256="e" * 64,
        manifest_file_sha256="f" * 64,
    )
    bank = FakeBank()

    bank.current_chunk_idx = 5
    assert bank.get_memory_payload_for_read("Ana") is payload
    bank.current_chunk_idx = 6
    selected = bank.get_memory_payload_for_read("Ana")
    assert selected["tokens"]["layers"]["0"].shape == (8, 3)
    assert [row["slot"] for row in selected["token_meta"]["layers"]["0"]] == list(range(8))
    assert bank.get_memory_payload_for_read("Bo") is payload
    flush()

    audit = json.loads(report.read_text(encoding="utf-8"))
    target = next(row for row in audit["read_records"] if row["chunk_idx"] == 6 and row["character"] == "Ana")
    assert audit["arm"] == "subject_only"
    assert audit["subject_subspace_contract"] == {
        "event_file_sha256": "e" * 64,
        "event_id": "event",
        "manifest_file_sha256": "f" * 64,
        "mask_manifest_sha256": manifest["mask_manifest_sha256"],
        "seed": 0,
        "source_capture_sha256": "a" * 64,
        "target_evidence_read": False,
    }
    assert target["selected_indices_by_layer"]["0"] == list(range(8))
    assert target["source_manifest_sha256_by_layer"]["0"] == manifest["layers"][0]["source_payload_sha256_by_layer"]["0"]


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        ("full_correct", list(range(32))),
        ("no_memory", []),
        ("zero_path", list(range(32))),
    ],
)
def test_baseline_arm_reports_have_explicit_layer_selection(
    arm: str, expected: list[int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event, manifest, payload = _reader_contract()

    class FakeBank:
        def get_memory_payload_for_read(self, char_id, bank_idx=0):
            return payload

    monkeypatch.setitem(
        sys.modules,
        "infer_slotmem",
        types.SimpleNamespace(RoleWiseSlotMemoryBank=FakeBank),
    )
    report = tmp_path / f"{arm}.json"
    flush = install_subject_subspace(
        arm=arm, seed=0, manifest=manifest, event=event, report_path=report
    )
    bank = FakeBank()
    bank.current_chunk_idx = 6
    bank.get_memory_payload_for_read("Ana")
    flush()

    target = json.loads(report.read_text(encoding="utf-8"))["read_records"][0]
    assert target["selected_indices_by_layer"]["0"] == expected
    assert "returned_sha256" in target


def test_subject_subspace_audit_self_check(capsys: pytest.CaptureFixture[str]) -> None:
    assert audit_main(["--self-check"]) == 0
    assert "self-check OK" in capsys.readouterr().out


def test_subject_subspace_report_is_immutable_before_reader_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event, manifest, _ = _reader_contract()

    class FakeBank:
        def get_memory_payload_for_read(self, char_id, bank_idx=0):
            return None

    original = FakeBank.get_memory_payload_for_read
    monkeypatch.setitem(
        sys.modules,
        "infer_slotmem",
        types.SimpleNamespace(RoleWiseSlotMemoryBank=FakeBank),
    )
    report = tmp_path / "audit.json"
    report.write_text("frozen", encoding="utf-8")

    with pytest.raises(FileExistsError):
        install_subject_subspace(
            arm="full_correct",
            seed=0,
            manifest=manifest,
            event=event,
            report_path=report,
        )
    assert FakeBank.get_memory_payload_for_read is original
    assert report.read_text(encoding="utf-8") == "frozen"


def test_wrong_subject_requires_a_frozen_validated_donor_before_reader_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event, manifest, _ = _reader_contract()

    class FakeBank:
        def get_memory_payload_for_read(self, char_id, bank_idx=0):
            return None

    original = FakeBank.get_memory_payload_for_read
    monkeypatch.setitem(
        sys.modules,
        "infer_slotmem",
        types.SimpleNamespace(RoleWiseSlotMemoryBank=FakeBank),
    )

    with pytest.raises(ValueError, match="frozen donor"):
        install_subject_subspace(
            arm="wrong_subject",
            seed=0,
            manifest=manifest,
            event=event,
            report_path=tmp_path / "wrong.json",
        )
    assert FakeBank.get_memory_payload_for_read is original


def test_wrong_subject_reader_uses_donor_values_and_target_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event, manifest, payload = _reader_contract()
    donor_tokens = {
        "__layerwise__": True,
        "layers": {
            layer: torch.full_like(tensor, 99.0)
            for layer, tensor in payload["tokens"]["layers"].items()
        },
    }
    artifact = {
        "format": "slotmem_donor_payload_v2",
        "event": {
            "story_id": "donor-story",
            "entity_uid": "donor::ana",
            "character_name": "Other Ana",
        },
        "payloads": {"Other Ana|0": donor_tokens},
    }
    entry = {
        "donor_story_id": "donor-story",
        "donor_entity_uid": "donor::ana",
        "payload_key": "Other Ana|0",
        "slot_shape": {layer: [32, 3] for layer in donor_tokens["layers"]},
    }

    class FakeBank:
        def get_memory_payload_for_read(self, char_id, bank_idx=0):
            return payload

    monkeypatch.setitem(sys.modules, "infer_slotmem", types.SimpleNamespace(RoleWiseSlotMemoryBank=FakeBank))
    flush = install_subject_subspace(
        arm="wrong_subject",
        seed=0,
        manifest=manifest,
        event=event,
        report_path=tmp_path / "wrong.json",
        donor_entry=entry,
        donor_artifact=artifact,
    )
    bank = FakeBank()
    bank.current_chunk_idx = 6
    output = bank.get_memory_payload_for_read("Ana")
    flush()

    assert torch.all(output["tokens"]["layers"]["0"] == 99)
    assert output["token_meta"]["layers"]["0"] == payload["token_meta"]["layers"]["0"][:8]


@pytest.mark.parametrize(
    ("arm", "expected_rows"),
    [
        ("subject_only", [1, 3]),
        ("drop_subject", [0, 2]),
        ("random_only", [0, 2]),
        ("drop_random", [1, 3]),
    ],
)
def test_slot_mask_arms_select_exact_rows(arm: str, expected_rows: list[int]) -> None:
    tokens = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    masks = {"semantic_top8": [1, 3], "random_top8": [0, 2]}

    assert torch.equal(transform_slot_rows(tokens, arm, masks), tokens[expected_rows])


def test_wrong_subject_applies_subject_rows_after_exact_shape_validation() -> None:
    correct = torch.zeros(4, 3)
    donor = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    assert torch.equal(
        transform_slot_rows(correct, "wrong_subject", {"semantic_top8": [1, 3]}, donor),
        donor[[1, 3]],
    )
    with pytest.raises(ValueError, match="exact shape"):
        transform_slot_rows(correct, "wrong_subject", {"semantic_top8": [1, 3]}, donor[:3])


@pytest.mark.parametrize("invalid", [[[1], 2], [3, 1], [1, 1]])
def test_slot_masks_reject_nested_unsorted_and_duplicate_indices(invalid: list) -> None:
    with pytest.raises(ValueError, match="unique ascending"):
        transform_slot_rows(torch.zeros(4, 2), "subject_only", {"semantic_top8": invalid})


def test_frozen_wrong_donor_bundle_binds_embedded_identity_key_and_shape() -> None:
    event, manifest, _ = _reader_contract()
    banks = validate_subject_subspace_manifest(manifest, event, seed=0)
    donor = {
        "__layerwise__": True,
        "layers": {str(layer): torch.zeros(32, 3) for layer in range(16)},
    }
    artifact = {
        "format": "slotmem_donor_payload_v2",
        "event": {
            "story_id": "donor-story",
            "entity_uid": "donor::ana",
            "character_name": "Other Ana",
        },
        "payloads": {"Other Ana|0": donor},
    }
    entry = {
        "donor_story_id": "donor-story",
        "donor_entity_uid": "donor::ana",
        "payload_key": "Other Ana|0",
        "slot_shape": {str(layer): [32, 3] for layer in range(16)},
    }

    assert validate_frozen_donor_artifact(artifact, entry, banks=banks) is donor
    entry["slot_shape"]["0"] = [31, 3]
    with pytest.raises(ValueError, match="slot_shape"):
        validate_frozen_donor_artifact(artifact, entry, banks=banks)

    entry["slot_shape"]["0"] = [32, 3]
    donor["layers"]["0"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_frozen_donor_artifact(artifact, entry, banks=banks)


def test_semantic_slot_score_penalizes_context() -> None:
    attention = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
    groups = {
        "identity_name": torch.tensor([1.0, 0.0]),
        "stable_attributes": torch.tensor([0.8, 0.0]),
        "other_characters": torch.tensor([0.0, 0.7]),
        "action_scene": torch.tensor([0.1, 1.0]),
    }
    assert semantic_slot_scores(attention, groups)[0] > semantic_slot_scores(attention, groups)[1]


def test_rankings_are_stable_and_source_only_methods_are_exact() -> None:
    assert top_fraction_indices(torch.arange(32, dtype=torch.float32), 0.25) == list(range(24, 32))
    assert top_fraction_indices(torch.ones(4), 0.5) == [0, 1]
    source = torch.eye(2)
    assert visual_counterfactual_scores(source, torch.tensor([[0.0, 0.0], [0.0, 1.0]])).tolist() == [1.0, 0.0]
    assert reference_agreement_scores(source, torch.tensor([[1.0, 0.0], [1.0, 0.0]])).tolist() == [1.0, 0.0]


def test_random_control_is_deterministic_equal_budget_and_addressed() -> None:
    first = deterministic_random_indices("event", 1, "0-4", 32, 8)
    assert first == deterministic_random_indices("event", 1, "0-4", 32, 8)
    assert len(first) == len(set(first)) == 8
    assert first != deterministic_random_indices("event", 2, "0-4", 32, 8)


def test_consensus_does_not_impute_missing_methods() -> None:
    rankings = {"semantic": [0, 1, 2, 3], "visual_cf": None, "reference": [1, 0, 3, 2]}
    assert consensus_rank_indices(rankings, 2) == [0, 1]
    with pytest.raises(ValueError, match="permutation"):
        consensus_rank_indices({"semantic": [0, 1], "reference": [0]}, 1)


@pytest.mark.parametrize("inputs", [
    {"nested": {"target_loss_sha256": "abc"}},
    {"nested": [{"qstar_score": 1.0}]},
    {"cids": "abc"},
    {"decoded_video": "abc"},
])
def test_manifest_rejects_target_derived_inputs_recursively(inputs: dict) -> None:
    with pytest.raises(ValueError, match="target evidence"):
        build_mask_manifest(inputs=inputs, rankings={}, event={}, seed=0)


def test_manifest_freezes_primary_consensus_and_random_masks() -> None:
    frozen = {"0-4": list(range(5)), "5-10": list(range(5, 11)), "11-15": list(range(11, 16))}
    manifest = build_mask_manifest(
        inputs={"source_capture_sha256": "a" * 64},
        rankings={f"bank_0/group_{group}": {"bank": 0, "layer_group": group, "member_layers": members, "source_payload_sha256_by_layer": {str(layer): "b" * 64 for layer in members}, "semantic": list(range(31, -1, -1)), "visual_cf": None, "reference": list(range(32))} for group, members in frozen.items()},
        event={"event_id": "event", "character_name": "Ana", "source_chunk_idx": 0},
        seed=1,
    )
    layer = manifest["layers"][0]
    assert manifest["primary_mask"] == "semantic_top8"
    assert layer["semantic_top8"] == list(range(24, 32))
    assert layer["visual_cf_top8"]["status"] == "not_available"
    assert layer["reference_top8"] == list(range(8))
    assert len(layer["consensus_top8"]) == len(layer["random_top8"]) == 8
    assert manifest["target_evidence_read"] is False
    assert layer["member_layers"] == list(range(5))
    with pytest.raises(ValueError, match="canonical"):
        build_mask_manifest(inputs={"source_capture_sha256": "a" * 64}, rankings={"alias": manifest["layers"][0]}, event={"event_id": "event", "character_name": "Ana", "source_chunk_idx": 0}, seed=1)


def test_semantic_vocabulary_reads_source_chunk_only() -> None:
    story = {"characters": {"Ana": "red coat", "Bo": "blue hat"}, "chunks": [
        {"content": "source station", "character_list": ["Bo", "Ana"]},
        {"content": "TARGET SECRET", "character_list": ["Ana"]},
    ]}
    groups = source_only_semantic_group_manifest(story, {"character_name": "Ana", "source_chunk_idx": 0, "target_chunk_idx": 1})
    assert groups == {"identity_name": ["Ana"], "stable_attributes": ["red coat"], "other_characters": ["Bo"], "action_scene": ["source station"]}
    assert "TARGET SECRET" not in repr(groups)


def _artifacts(tmp_path: Path) -> tuple[dict, dict, Path, dict]:
    repo = Path(__file__).resolve().parents[2]
    story, reference, model = tmp_path / "story.json", tmp_path / "00.jpg", tmp_path / "model.pt"
    story.write_text(json.dumps({"characters": {"Ana": "red coat"}, "chunks": [{"content": "station", "character_list": ["Ana"]}]}) + "\n", encoding="utf-8")
    reference.write_bytes(b"reference")
    model.write_bytes(b"model")
    event = {"event_id": "event", "character_name": "Ana", "source_chunk_idx": 0, "target_chunk_idx": 1, "source_json_path": str(story.resolve()), "reference_path": str(reference.resolve()), "reference_sha256": sha256_file(reference)}
    raw, slots = torch.arange(12, dtype=torch.float32).reshape(4, 3), torch.arange(96, dtype=torch.float32).reshape(32, 3)
    meta, attention = [{"char_id": "Ana", "inside_box": index < 2} for index in range(4)], {"Ana": torch.full((32, 4), 0.25)}
    rows = []
    for layer in range(16):
        row = {"character": "Ana", "bank": 0, "layer": layer, "raw_tokens": raw, "raw_token_meta": meta, "encoded_slots": slots, "attention": attention}
        row["tensor_shapes"] = {"raw_tokens": [4, 3], "encoded_slots": [32, 3], "attention": {"Ana": [32, 4]}}
        row["sha256"] = {"raw_tokens": capture_tensor_sha256(raw), "raw_token_meta": canonical_json_sha256(meta), "encoded_slots": capture_tensor_sha256(slots), "attention": capture_tensor_sha256(attention)}
        rows.append(row)
    provenance = {"source_json_path": str(story.resolve()), "source_json_sha256": sha256_file(story), "reference_file_sha256": sha256_file(reference), "fixed_reference_scope": "source_only", "source_seed": 0, "code_identity": {"infer_slotmem_sha256": sha256_file(repo / "infer_slotmem.py"), "mem_encoder_utils_sha256": sha256_file(repo / "mem_encoder_utils.py")}, "runtime_identity": {"python_version": "3.11", "torch_version": str(torch.__version__), "inference_args_sha256": "1" * 64}, "model_identity": {"high_noise": [{"path": str(model.resolve()), "sha256": sha256_file(model)}], "low_noise": []}}
    canonical = {"schema_version": 1, "source_chunk_idx": 0, "target_evidence_read": False, "provenance": provenance, "captures": [{key: row[key] for key in ("character", "bank", "layer", "tensor_shapes", "sha256")} for row in rows]}
    capture = {**canonical, "captures": rows, "canonical_artifact_sha256": canonical_json_sha256(canonical)}
    capture_path = tmp_path / "source_capture.pt"
    torch.save(capture, capture_path)
    vocabulary = {"identity_name": ["Ana"], "stable_attributes": ["red coat"], "other_characters": [], "action_scene": ["station"]}
    score_rows = [{"character": "Ana", "bank": 0, "layer": layer, "groups": {name: [1.0, 0.0, 0.0, 0.0] for name in SEMANTIC_GROUPS}} for layer in range(16)]
    scores = build_semantic_score_artifact(event_id="event", source_capture_sha256=sha256_file(capture_path), source_capture_canonical_artifact_sha256=capture["canonical_artifact_sha256"], semantic_manifest=vocabulary, source_provenance=provenance, captures=score_rows)
    return capture, event, capture_path, scores


def test_capture_and_semantic_artifacts_fail_closed(tmp_path: Path) -> None:
    capture, event, capture_path, scores = _artifacts(tmp_path)
    repo = Path(__file__).resolve().parents[2]
    source_hash = capture["provenance"]["source_json_sha256"]
    subject = validate_source_capture(capture, event, repo_root=repo, source_json_sha256=source_hash)
    vocabulary = scores["semantic_manifest"]
    capture_hash = sha256_file(capture_path)
    validated = validate_semantic_scores(scores, event=event, source_capture_sha256=capture_hash, source_capture=capture, subject_captures=subject, expected_semantic_manifest=vocabulary)
    assert validated[(0, 4)]["character"] == "Ana"
    Path(event["source_json_path"]).write_text("changed after frozen read", encoding="utf-8")
    assert validate_source_capture(capture, event, repo_root=repo, source_json_sha256=source_hash)
    with pytest.raises(ValueError, match="provenance"):
        validate_source_capture(capture, event, repo_root=repo, source_json_sha256=sha256_file(Path(event["source_json_path"])))
    bad_scores = json.loads(json.dumps(scores))
    bad_scores["captures"][0]["groups"]["identity_name"] = [1.0]
    bad_scores["canonical_artifact_sha256"] = canonical_json_sha256({key: value for key, value in bad_scores.items() if key != "canonical_artifact_sha256"})
    with pytest.raises(ValueError, match="token count"):
        validate_semantic_scores(bad_scores, event=event, source_capture_sha256=capture_hash, source_capture=capture, subject_captures=subject, expected_semantic_manifest=vocabulary)
    bad_producer = json.loads(json.dumps(scores))
    bad_producer["producer"]["kind"] = "unbound"
    bad_producer["canonical_artifact_sha256"] = canonical_json_sha256({key: value for key, value in bad_producer.items() if key != "canonical_artifact_sha256"})
    with pytest.raises(ValueError, match="producer"):
        validate_semantic_scores(bad_producer, event=event, source_capture_sha256=capture_hash, source_capture=capture, subject_captures=subject, expected_semantic_manifest=vocabulary)
    wrong_character = json.loads(json.dumps(scores))
    wrong_character["captures"][0]["character"] = "ana"
    wrong_character["canonical_artifact_sha256"] = canonical_json_sha256({key: value for key, value in wrong_character.items() if key != "canonical_artifact_sha256"})
    with pytest.raises(ValueError, match="character"):
        validate_semantic_scores(wrong_character, event=event, source_capture_sha256=capture_hash, source_capture=capture, subject_captures=subject, expected_semantic_manifest=vocabulary)
    capture["captures"][0]["raw_tokens"][0, 0] = -1
    with pytest.raises(ValueError, match="raw_tokens"):
        validate_source_capture(capture, event, repo_root=repo, source_json_sha256=source_hash)
    scores["provenance"] = {"target_loss_sha256": "c" * 64}
    with pytest.raises(ValueError, match="target evidence"):
        validate_semantic_scores(scores, event=event, source_capture_sha256=capture_hash, source_capture=capture, subject_captures=subject, expected_semantic_manifest=vocabulary)


def test_slot_attention_scatter_preserves_raw_positions() -> None:
    capture = {"raw_tokens": torch.zeros(3, 2), "encoded_slots": torch.zeros(4, 2), "raw_token_meta": [{"char_id": "Bo"}, {"char_id": "Ana"}, {"char_id": "Bo"}], "attention": {"Ana": torch.tensor([[1.0], [1.0]]), "Bo": torch.tensor([[0.25, 0.75], [0.5, 0.5]])}}
    assert slot_attention_matrix(capture).tolist() == [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.25, 0.0, 0.75], [0.5, 0.0, 0.5]]


def test_layer_group_scores_are_the_mean_of_member_layers() -> None:
    base = {"bank": 0, "raw_tokens": torch.zeros(2, 2), "encoded_slots": torch.zeros(2, 2), "raw_token_meta": [{"char_id": "Ana"}, {"char_id": "Ana"}]}
    captures = [{**base, "layer": 0, "attention": {"Ana": torch.eye(2)}}, {**base, "layer": 1, "attention": {"Ana": torch.flip(torch.eye(2), dims=(0,))}}]
    groups = {"identity_name": [1.0, 0.0], "stable_attributes": [1.0, 0.0], "other_characters": [0.0, 0.0], "action_scene": [0.0, 0.0]}
    scores = {(0, layer): {"groups": groups} for layer in (0, 1)}
    assert aggregate_semantic_slot_scores(captures, scores).tolist() == [0.5, 0.5]


def test_probe_freezes_exact_budget_and_reference_capture_fails_honestly(tmp_path: Path) -> None:
    capture, event, capture_path, scores = _artifacts(tmp_path)
    event_path, scores_path, output = tmp_path / "event.json", tmp_path / "scores.json", tmp_path / "manifest.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    scores_path.write_text(json.dumps(scores), encoding="utf-8")
    manifest = freeze_subject_subspace(event_path=event_path, source_capture_path=capture_path, semantic_scores_path=scores_path, output_path=output, seed=0, repo_root=Path(__file__).resolve().parents[2])
    assert [row["layer_group"] for row in manifest["layers"]] == ["0-4", "5-10", "11-15"]
    assert all(len(row["semantic_top8"]) == len(row["random_top8"]) == 8 for row in manifest["layers"])
    assert all(row["visual_cf_top8"]["status"] == row["reference_top8"]["status"] == "not_available" for row in manifest["layers"])
    frozen_output = output.read_bytes()
    with pytest.raises(FileExistsError):
        freeze_subject_subspace(event_path=event_path, source_capture_path=capture_path, semantic_scores_path=scores_path, output_path=output, seed=0, repo_root=Path(__file__).resolve().parents[2])
    assert output.read_bytes() == frozen_output
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
    for invalid_seed in (True, 0.0):
        with pytest.raises(ValueError, match="seed"):
            freeze_subject_subspace(event_path=event_path, source_capture_path=capture_path, semantic_scores_path=scores_path, output_path=tmp_path / f"invalid-{invalid_seed!r}.json", seed=invalid_seed, repo_root=Path(__file__).resolve().parents[2])
    with pytest.raises(ValueError, match="seed"):
        freeze_subject_subspace(event_path=event_path, source_capture_path=capture_path, semantic_scores_path=scores_path, output_path=tmp_path / "wrong-seed.json", seed=2, repo_root=Path(__file__).resolve().parents[2])
    for event_seed in (True, 0.0, 1):
        event["target_seed"] = event_seed
        event_path.write_text(json.dumps(event), encoding="utf-8")
        with pytest.raises(ValueError, match="seed"):
            freeze_subject_subspace(event_path=event_path, source_capture_path=capture_path, semantic_scores_path=scores_path, output_path=tmp_path / f"wrong-event-seed-{event_seed!r}.json", seed=0, repo_root=Path(__file__).resolve().parents[2])
    dry_run = tmp_path / "reference.json"
    probe_main(["capture-reference", "--reference-image", event["reference_path"], "--output", str(dry_run), "--dry-run"])
    assert json.loads(dry_run.read_text())["status"] == "not_available"
    with pytest.raises(FileExistsError):
        probe_main(["capture-reference", "--reference-image", event["reference_path"], "--output", str(dry_run), "--dry-run"])
    with pytest.raises(RuntimeError, match="fabricate"):
        probe_main(["capture-reference", "--reference-image", event["reference_path"], "--output", str(tmp_path / "real.pt")])


def test_probe_rejects_incomplete_frozen_layer_group(tmp_path: Path) -> None:
    capture, event, capture_path, scores = _artifacts(tmp_path)
    capture["captures"] = capture["captures"][:-1]
    capture["canonical_artifact_sha256"] = canonical_json_sha256({
        "schema_version": 1, "source_chunk_idx": 0, "target_evidence_read": False,
        "provenance": capture["provenance"],
        "captures": [{key: row[key] for key in ("character", "bank", "layer", "tensor_shapes", "sha256")} for row in capture["captures"]],
    })
    torch.save(capture, capture_path)
    scores["source_capture_sha256"] = sha256_file(capture_path)
    scores["source_capture_canonical_artifact_sha256"] = capture["canonical_artifact_sha256"]
    scores["captures"] = scores["captures"][:-1]
    scores["canonical_artifact_sha256"] = canonical_json_sha256({key: value for key, value in scores.items() if key != "canonical_artifact_sha256"})
    event_path, scores_path = tmp_path / "event.json", tmp_path / "scores.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    scores_path.write_text(json.dumps(scores), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen layer group"):
        freeze_subject_subspace(event_path=event_path, source_capture_path=capture_path, semantic_scores_path=scores_path, output_path=tmp_path / "manifest.json", seed=0, repo_root=Path(__file__).resolve().parents[2])
