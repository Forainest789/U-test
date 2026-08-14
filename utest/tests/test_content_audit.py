from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from utest.content_audit import (
    LAYERS_KEY,
    LAYERWISE_MARKER,
    install,
    intervention_applies,
    stable_transform_seed,
    transform_payload,
    transform_tokens,
    validate_donor_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_confirmatory_arm_semantics() -> None:
    tokens = torch.arange(48, dtype=torch.float32).reshape(8, 6)
    generator = torch.Generator().manual_seed(7)

    assert transform_tokens(tokens, "correct", generator) is tokens
    assert torch.count_nonzero(transform_tokens(tokens, "zero", generator)) == 0
    assert transform_tokens(tokens, "no_memory", generator) is None


def test_intervention_applies_only_to_frozen_character_and_chunk() -> None:
    event = {"character_name": "ana", "target_chunk_idx": 4}

    assert intervention_applies(event, "ana", 4)
    assert not intervention_applies(event, "bob", 4)
    assert not intervention_applies(event, "ana", 5)
    assert not intervention_applies(event, "ana", None)


def test_random_seed_is_stable_per_read_and_layer() -> None:
    event = {"story_id": "s1", "event_id": "e4"}

    seed = stable_transform_seed(event, 4, "ana", 0, 17, "7")

    assert seed == stable_transform_seed(event, 4, "ana", 0, 17, "7")
    assert seed != stable_transform_seed(event, 4, "ana", 0, 17, "8")
    assert seed != stable_transform_seed(event, 4, "ana", 1, 17, "7")


def test_random_is_deterministic_and_channel_moment_matched() -> None:
    tokens = torch.arange(48, dtype=torch.float32).reshape(8, 6)
    a = transform_tokens(tokens, "random", torch.Generator().manual_seed(7))
    b = transform_tokens(tokens, "random", torch.Generator().manual_seed(7))

    assert torch.equal(a, b)
    assert a.shape == tokens.shape and a.dtype == tokens.dtype
    assert torch.allclose(a.mean(0), tokens.mean(0), atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        a.std(0, correction=0), tokens.std(0, correction=0), atol=1e-5, rtol=1e-5
    )


def test_random_keeps_constant_channels_constant() -> None:
    tokens = torch.tensor([[1.0, 5.0], [3.0, 5.0], [7.0, 5.0]])
    random = transform_tokens(
        tokens, "random", torch.Generator().manual_seed(11)
    )
    assert torch.all(random[:, 1] == 5.0)


def test_layerwise_random_preserves_metadata_and_counts_layers() -> None:
    tokens = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    metadata = [{"char_id": "ana"}] * 6
    payload = {
        "tokens": {
            LAYERWISE_MARKER: True,
            LAYERS_KEY: {"0": tokens, "7": tokens * 2},
        },
        "token_meta": metadata,
    }
    output, layers = transform_payload(
        payload, "random", torch.Generator().manual_seed(13)
    )

    assert layers == 2
    assert output["token_meta"] is metadata
    assert output["tokens"][LAYERS_KEY]["0"].shape == tokens.shape


def test_layerwise_random_is_independent_of_layer_iteration_order() -> None:
    tokens = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    event = {"story_id": "s1", "event_id": "e4"}

    def factory(layer: str):
        return torch.Generator().manual_seed(
            stable_transform_seed(event, 4, "ana", 0, 17, layer)
        )

    forward, _ = transform_payload(
        {
            "tokens": {
                LAYERWISE_MARKER: True,
                LAYERS_KEY: {"0": tokens, "7": tokens * 2},
            }
        },
        "random",
        None,
        generator_for_layer=factory,
    )
    reverse, _ = transform_payload(
        {
            "tokens": {
                LAYERWISE_MARKER: True,
                LAYERS_KEY: {"7": tokens * 2, "0": tokens},
            }
        },
        "random",
        None,
        generator_for_layer=factory,
    )

    for layer in ("0", "7"):
        assert torch.equal(
            forward["tokens"][LAYERS_KEY][layer],
            reverse["tokens"][LAYERS_KEY][layer],
        )


def test_wrong_requires_every_layer_and_never_falls_back() -> None:
    tokens = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    payload = {
        "tokens": {
            LAYERWISE_MARKER: True,
            LAYERS_KEY: {"0": tokens, "7": tokens},
        }
    }
    donor = {
        LAYERWISE_MARKER: True,
        LAYERS_KEY: {"0": torch.ones(6, 4)},
    }

    with pytest.raises(ValueError, match="layer 7"):
        transform_payload(
            payload, "wrong", torch.Generator().manual_seed(0), donor
        )


def test_wrong_rejects_any_non_exact_tensor_shape() -> None:
    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape
            self.device = "cpu"
            self.dtype = "float32"

        def to(self, *args):
            return self

        def clone(self):
            return self

    target = FakeTensor((6, 4))

    for donor_shape in ((5, 4), (7, 4), (6, 8)):
        with pytest.raises(ValueError, match="exact shape"):
            transform_tokens(target, "wrong", None, FakeTensor(donor_shape))


def test_audit_report_contains_actual_runtime_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeBank:
        def get_memory_payload_for_read(self, char_id, bank_idx=0):
            return None

    monkeypatch.setitem(
        sys.modules,
        "infer_slotmem",
        types.SimpleNamespace(RoleWiseSlotMemoryBank=FakeBank),
    )
    report = tmp_path / "audit.json"
    runtime = {"target_seed": 271, "frozen_args": {"cfg_scale": "5.0"}}

    flush = install(
        "correct",
        7,
        None,
        None,
        str(report),
        event={"character_name": "ana", "target_chunk_idx": 4},
        runtime_contract=runtime,
    )
    flush()

    assert json.loads(report.read_text(encoding="utf-8"))["runtime_contract"] == runtime


def test_no_memory_suppresses_only_the_target_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DummyTensor:
        shape = (2, 3)

    payload = {"tokens": DummyTensor()}

    class FakeBank:
        def get_memory_payload_for_read(self, char_id, bank_idx=0):
            return payload

    monkeypatch.setattr(torch, "Tensor", DummyTensor, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "infer_slotmem",
        types.SimpleNamespace(RoleWiseSlotMemoryBank=FakeBank),
    )
    report = tmp_path / "audit.json"
    flush = install(
        "no_memory",
        7,
        None,
        None,
        str(report),
        event={"character_name": "ana", "target_chunk_idx": 4},
        runtime_contract={},
    )
    bank = FakeBank()

    bank.current_chunk_idx = 3
    assert bank.get_memory_payload_for_read("ana") is payload
    bank.current_chunk_idx = 4
    assert bank.get_memory_payload_for_read("ana") is None
    assert bank.get_memory_payload_for_read("bob") is payload
    bank.current_chunk_idx = 5
    assert bank.get_memory_payload_for_read("ana") is payload
    flush()

    audit = json.loads(report.read_text(encoding="utf-8"))
    assert audit["target_read_hits"] == 1
    assert audit["target_source_non_null_reads"] == 1
    assert audit["target_returned_non_null_reads"] == 0
    assert [row["chunk_idx"] for row in audit["read_records"]] == [3, 4, 4, 5]


def test_donor_manifest_rejects_same_entity_story_and_bad_hash(tmp_path: Path) -> None:
    donor = tmp_path / "donor.pt"
    donor.write_bytes(b"payload")
    event = {"story_id": "target", "entity_uid": "target::ana"}
    base = {
        "target_story_id": "target",
        "target_entity_uid": "target::ana",
        "donor_story_id": "donor",
        "donor_entity_uid": "donor::ana",
        "payload_path": str(donor),
        "payload_sha256": _sha256(donor),
        "payload_key": "ana|0",
        "coarse_class": "person",
        "colour": "brown hair",
        "character_count": 1,
        "source_visible": True,
        "gap_bucket": "1-2",
        "slot_shape": [4],
        "selection_seed": 0,
    }

    same_entity = {**base, "donor_entity_uid": "target::ana"}
    with pytest.raises(ValueError, match="different entity_uid"):
        validate_donor_manifest(same_entity, event, donor)

    same_story = {**base, "donor_story_id": "target"}
    with pytest.raises(ValueError, match="different story"):
        validate_donor_manifest(same_story, event, donor)

    bad_hash = {**base, "payload_sha256": "0" * 64}
    with pytest.raises(ValueError, match="SHA256"):
        validate_donor_manifest(bad_hash, event, donor)

    missing_key = dict(base)
    del missing_key["payload_key"]
    with pytest.raises(ValueError, match="payload_key"):
        validate_donor_manifest(missing_key, event, donor)

    assert validate_donor_manifest(base, event, donor)["payload_sha256"] == _sha256(donor)
