from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from utest.content_audit import (
    LAYERS_KEY,
    LAYERWISE_MARKER,
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


def test_random_is_deterministic_and_channel_moment_matched() -> None:
    tokens = torch.arange(48, dtype=torch.float32).reshape(8, 6)
    a = transform_tokens(tokens, "random", torch.Generator().manual_seed(7))
    b = transform_tokens(tokens, "random", torch.Generator().manual_seed(7))

    assert torch.equal(a, b)
    assert a.shape == tokens.shape and a.dtype == tokens.dtype
    assert torch.allclose(a.mean(0), tokens.mean(0), atol=1e-5)
    assert torch.allclose(
        a.std(0, correction=0), tokens.std(0, correction=0), atol=1e-5
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

    assert validate_donor_manifest(base, event, donor)["payload_sha256"] == _sha256(donor)
