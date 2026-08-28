"""Source-only scoring, validation, and frozen slot-mask primitives."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path

import torch

from .prefix_contract import (
    FROZEN_MEMORY_ENCODER_SLOTS,
    FROZEN_SUBJECT_SUBSPACE_BUDGET,
    FROZEN_SUBJECT_SUBSPACE_FRACTION,
    sha256_file,
)

SEMANTIC_GROUPS = ("identity_name", "stable_attributes", "other_characters", "action_scene")
SOURCE_SEMANTIC_FORMULA = {"name": "source_role_box_centre", "version": 1}
FROZEN_LAYER_GROUPS = {
    "0-4": tuple(range(0, 5)),
    "5-10": tuple(range(5, 11)),
    "11-15": tuple(range(11, 16)),
}
FORBIDDEN_INPUT_PREFIXES = ("target_", "qstar", "cids", "decoded_")


def source_metadata_semantic_groups(
    raw_token_metadata: Sequence[Mapping[str, object]],
    subject_char_id: str,
) -> dict[str, list[float]]:
    """Return frozen source-only semantic vectors in raw-token order."""
    if not isinstance(subject_char_id, str) or not subject_char_id.strip():
        raise ValueError("source semantic groups require a nonempty subject character ID")
    groups = {name: [] for name in SEMANTIC_GROUPS}
    has_target = False
    has_inside_target = False
    for item in raw_token_metadata:
        if (
            not isinstance(item, Mapping)
            or not {"char_id", "inside_box", "tau_local"}.issubset(item)
            or not isinstance(item["char_id"], str)
            or not isinstance(item["inside_box"], bool)
            or isinstance(item["tau_local"], bool)
            or not isinstance(item["tau_local"], Real)
        ):
            raise ValueError("source semantic token metadata is malformed")
        char_id = item["char_id"]
        inside_box = item["inside_box"]
        tau_local = float(item["tau_local"])
        if not math.isfinite(tau_local):
            raise ValueError("source semantic tau_local must be finite")
        is_target = float(char_id == subject_char_id)
        inside = is_target * float(inside_box)
        has_target = has_target or bool(is_target)
        has_inside_target = has_inside_target or bool(inside)
        groups["identity_name"].append(is_target)
        groups["stable_attributes"].append(
            inside * math.exp(-(tau_local ** 2) / 2.0)
        )
        groups["other_characters"].append(
            float(bool(char_id) and char_id != subject_char_id)
        )
        groups["action_scene"].append(is_target * float(not inside_box))
    if not has_target:
        raise ValueError("source semantic metadata contains no target token")
    if not has_inside_target:
        raise ValueError("source semantic metadata contains no inside-box target token")
    return groups


def canonical_json_sha256(value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def capture_tensor_sha256(value: object) -> str:
    """Match Task 3's nested-tensor hash contract."""
    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().cpu()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                update(item[key])
        else:
            raise ValueError("capture hash accepts tensors and tensor mappings only")

    update(value)
    return digest.hexdigest()


def _has_target_evidence(value: object, *, allow_marker: bool = False) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            forbidden = str(key).casefold().startswith(FORBIDDEN_INPUT_PREFIXES)
            if forbidden and not (allow_marker and key == "target_evidence_read"):
                return True
            if _has_target_evidence(item, allow_marker=allow_marker):
                return True
        return False
    return isinstance(value, (list, tuple)) and any(_has_target_evidence(item) for item in value)


def semantic_slot_scores(attention: torch.Tensor, groups: Mapping[str, torch.Tensor]) -> torch.Tensor:
    attention = torch.as_tensor(attention).float()
    vectors = {name: torch.as_tensor(groups[name], device=attention.device).float() for name in SEMANTIC_GROUPS if name in groups}
    if set(vectors) != set(SEMANTIC_GROUPS) or attention.ndim != 2 or any(vector.ndim != 1 or vector.numel() != attention.shape[1] for vector in vectors.values()):
        raise ValueError("semantic score vectors must match attention token count")
    identity = torch.stack([attention @ vectors["identity_name"], attention @ vectors["stable_attributes"]]).mean(0)
    nuisance = torch.stack([attention @ vectors["other_characters"], attention @ vectors["action_scene"]]).amax(0)
    return identity - nuisance


def _rank(scores: torch.Tensor) -> list[int]:
    values = torch.as_tensor(scores).detach().float().cpu()
    if values.ndim != 1 or not values.numel() or not torch.isfinite(values).all():
        raise ValueError("slot scores must be a nonempty finite vector")
    return sorted(range(values.numel()), key=lambda index: (-float(values[index]), index))


def top_fraction_indices(scores: torch.Tensor, fraction: float) -> list[int]:
    if not 0 < float(fraction) <= 1:
        raise ValueError("top fraction must be in (0, 1]")
    ranked = _rank(scores)
    return sorted(ranked[: max(1, int(len(ranked) * float(fraction)))])


def deterministic_random_indices(event_id: str, seed: int, layer_group: str, universe: int, count: int) -> list[int]:
    if isinstance(seed, bool) or not 0 < int(count) <= int(universe):
        raise ValueError("random mask requires a valid seed, universe, and count")
    address = json.dumps([str(event_id), int(seed), str(layer_group), "random-control"], ensure_ascii=False, separators=(",", ":")).encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(address).digest(), "big"))
    return sorted(rng.sample(range(int(universe)), int(count)))


def _matching_slots(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left, right = torch.as_tensor(left).float(), torch.as_tensor(right).float()
    if left.ndim != 2 or left.shape != right.shape or not left.numel() or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("slot payloads must have identical finite nonempty 2D shapes")
    return left, right


def visual_counterfactual_scores(source: torch.Tensor, removed: torch.Tensor) -> torch.Tensor:
    # ponytail: feature-space removal depends on the existing role box; upgrade to
    # a pixel-space counterfactual if box quality proves insufficient.
    source, removed = _matching_slots(source, removed)
    return torch.linalg.vector_norm(source - removed, dim=-1)


def reference_agreement_scores(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    source, reference = _matching_slots(source, reference)
    return torch.nn.functional.cosine_similarity(source, reference, dim=-1)


def consensus_rank_indices(rankings: Mapping[str, list[int] | None], count: int) -> list[int]:
    available = [values for values in rankings.values() if values is not None]
    if not available:
        raise ValueError("consensus requires an available ranking")
    universe, expected = len(available[0]), set(range(len(available[0])))
    if any(len(values) != universe or set(values) != expected for values in available):
        raise ValueError("each available ranking must be a full slot permutation")
    if not 0 < int(count) <= universe:
        raise ValueError("consensus count exceeds slot universe")
    positions = [{slot: rank for rank, slot in enumerate(values)} for values in available]
    ordered = sorted(expected, key=lambda slot: (sum(position[slot] for position in positions) / len(positions), slot))
    return sorted(ordered[: int(count)])


def source_only_semantic_group_manifest(story: Mapping, event: Mapping) -> dict[str, list[str]]:
    character, source_idx = str(event.get("character_name", "")).strip(), int(event.get("source_chunk_idx", -1))
    chunks, characters = story.get("chunks", []), story.get("characters", {})
    if not character or source_idx != 0 or source_idx >= len(chunks):
        raise ValueError("semantic groups require frozen source chunk 0")
    source = chunks[source_idx]
    attributes, scene = str(characters.get(character, "")).strip(), str(source.get("content", "")).strip()
    appearing = list(source.get("character_list", []))
    if not attributes or not scene or character not in appearing:
        raise ValueError("source semantic fields are incomplete")
    return {
        "identity_name": [character],
        "stable_attributes": [attributes],
        "other_characters": [str(name) for name in appearing if str(name).casefold() != character.casefold()],
        "action_scene": [scene],
    }


def slot_attention_matrix(capture: Mapping) -> torch.Tensor:
    raw, slots, meta, attention = (capture[key] for key in ("raw_tokens", "encoded_slots", "raw_token_meta", "attention"))
    role_indices: dict[str, list[int]] = {}
    for index, item in enumerate(meta):
        role_indices.setdefault(str(item.get("char_id") or "0"), []).append(index)
    rows = []
    for role in sorted(role_indices):
        local, indices = attention.get(role), role_indices[role]
        if not isinstance(local, torch.Tensor) or local.ndim != 2 or local.shape[1] != len(indices):
            raise ValueError("capture attention does not match raw token metadata")
        if not torch.isfinite(local).all() or torch.any(local < 0) or not torch.allclose(local.float().sum(-1), torch.ones(local.shape[0]), atol=1e-5, rtol=1e-5):
            raise ValueError("capture attention is not a probability matrix")
        scattered = torch.zeros((local.shape[0], raw.shape[0]), dtype=local.dtype)
        scattered[:, indices] = local.cpu()
        rows.append(scattered)
    result = torch.cat(rows)
    if result.shape[0] != slots.shape[0]:
        raise ValueError("capture attention rows do not match encoded slots")
    return result


def aggregate_semantic_slot_scores(captures: list[Mapping], score_rows: Mapping[tuple[int, int], Mapping]) -> torch.Tensor:
    """Mean per-layer semantic slot scores for one frozen bank/layer group."""
    values = []
    for row in captures:
        address = (int(row["bank"]), int(row["layer"]))
        groups = {name: torch.tensor(score_rows[address]["groups"][name]) for name in SEMANTIC_GROUPS}
        values.append(semantic_slot_scores(slot_attention_matrix(row), groups))
    if not values or any(value.shape != values[0].shape for value in values):
        raise ValueError("frozen layer group semantic slot shapes do not match")
    return torch.stack(values).mean(0)


def validate_source_capture(artifact: Mapping, event: Mapping, *, repo_root: Path, source_json_sha256: str) -> list[Mapping]:
    """Verify Task 3 canonical/payload hashes and files named by provenance."""
    if _has_target_evidence(artifact, allow_marker=True) or artifact.get("target_evidence_read") is not False:
        raise ValueError("source capture contains target evidence")
    captures, provenance = artifact.get("captures"), artifact.get("provenance")
    if artifact.get("schema_version") != 1 or artifact.get("source_chunk_idx") != 0 or not captures or not isinstance(provenance, Mapping):
        raise ValueError("source capture schema is invalid")
    required = {"source_json_path", "source_json_sha256", "reference_file_sha256", "fixed_reference_scope", "source_seed", "code_identity", "runtime_identity", "model_identity"}
    runtime = provenance.get("runtime_identity", {})
    if not required.issubset(provenance) or provenance["fixed_reference_scope"] not in {"all_chunks", "source_only"} or isinstance(provenance["source_seed"], bool) or not isinstance(provenance["source_seed"], int) or not all(str(runtime.get(key, "")).strip() for key in ("python_version", "torch_version", "inference_args_sha256")):
        raise ValueError("source capture provenance is incomplete")
    summary = [{key: row[key] for key in ("character", "bank", "layer", "tensor_shapes", "sha256")} for row in captures]
    canonical = {"schema_version": 1, "source_chunk_idx": 0, "target_evidence_read": False, "provenance": dict(provenance), "captures": summary}
    if artifact.get("canonical_artifact_sha256") != canonical_json_sha256(canonical):
        raise ValueError("source capture canonical artifact SHA-256 mismatch")
    source_path = Path(provenance["source_json_path"]).resolve()
    files = [
        (Path(event["reference_path"]).resolve(), provenance["reference_file_sha256"]),
        (Path(repo_root) / "infer_slotmem.py", provenance["code_identity"]["infer_slotmem_sha256"]),
        (Path(repo_root) / "mem_encoder_utils.py", provenance["code_identity"]["mem_encoder_utils_sha256"]),
    ] + [(Path(row["path"]).resolve(), row["sha256"]) for domain in ("high_noise", "low_noise") for row in provenance["model_identity"][domain]]
    if Path(event["source_json_path"]).resolve() != source_path or source_json_sha256.casefold() != str(provenance["source_json_sha256"]).casefold() or str(event["reference_sha256"]).casefold() != str(files[0][1]).casefold():
        raise ValueError("source capture provenance does not match event")
    if len(files) <= 3 or any(not path.is_file() or sha256_file(path).casefold() != str(expected).casefold() for path, expected in files):
        raise ValueError("source capture provenance SHA-256 mismatch")
    selected, seen = [], set()
    for row in captures:
        raw, slots, meta, attention = (row[key] for key in ("raw_tokens", "encoded_slots", "raw_token_meta", "attention"))
        hashes = {"raw_tokens": capture_tensor_sha256(raw), "raw_token_meta": canonical_json_sha256(meta), "encoded_slots": capture_tensor_sha256(slots), "attention": capture_tensor_sha256(attention)}
        shapes = {"raw_tokens": list(raw.shape), "encoded_slots": list(slots.shape), "attention": {str(key): list(value.shape) for key, value in attention.items()}}
        if row["sha256"] != hashes or row["tensor_shapes"] != shapes or len(meta) != raw.shape[0] or raw.shape[1] != slots.shape[1]:
            differing = next((name for name in hashes if row["sha256"].get(name) != hashes[name]), "payload")
            raise ValueError(f"source capture {differing} SHA-256 or shape mismatch")
        slot_attention_matrix(row)
        if str(row["character"]).casefold() == str(event["character_name"]).casefold():
            address = (int(row["bank"]), int(row["layer"]))
            if address in seen:
                raise ValueError("source capture subject layer is duplicated")
            seen.add(address)
            selected.append(row)
    if not selected:
        raise ValueError("source capture contains no subject payload")
    return selected


def build_semantic_score_artifact(
    *,
    event_id: str,
    source_capture_sha256: str,
    source_capture_canonical_artifact_sha256: str,
    semantic_manifest: Mapping,
    source_provenance: Mapping,
    captures: list[Mapping],
    formula: Mapping,
    subject_char_id: str,
    source_seed: int,
) -> dict:
    producer = {
        "kind": "slotmem_source_semantic_token_scores",
        "version": 1,
        "source_chunk_idx": 0,
        "target_evidence_read": False,
        "formula": dict(formula),
        "subject_char_id": subject_char_id,
        "source_seed": source_seed,
        "source_json_sha256": source_provenance["source_json_sha256"],
        "semantic_vocabulary_sha256": canonical_json_sha256(semantic_manifest),
        "code_identity": source_provenance["code_identity"],
        "model_identity": source_provenance["model_identity"],
    }
    rows = [{**dict(row), "address": f'bank_{int(row["bank"])}/layer_{int(row["layer"])}'} for row in captures]
    artifact = {"schema_version": 1, "event_id": event_id, "source_chunk_idx": 0, "target_evidence_read": False, "source_capture_sha256": source_capture_sha256, "source_capture_canonical_artifact_sha256": source_capture_canonical_artifact_sha256, "semantic_manifest": dict(semantic_manifest), "producer": producer, "captures": rows}
    artifact["canonical_artifact_sha256"] = canonical_json_sha256(artifact)
    return artifact


def validate_semantic_scores(artifact: Mapping, *, event: Mapping, source_capture_sha256: str, source_capture: Mapping, subject_captures: list[Mapping], expected_semantic_manifest: Mapping) -> dict[tuple[int, int], Mapping]:
    if _has_target_evidence(artifact, allow_marker=True) or artifact.get("target_evidence_read") is not False:
        raise ValueError("semantic scores contain target evidence")
    canonical = {key: value for key, value in artifact.items() if key != "canonical_artifact_sha256"}
    if artifact.get("schema_version") != 1 or artifact.get("canonical_artifact_sha256") != canonical_json_sha256(canonical):
        raise ValueError("semantic score canonical artifact SHA-256 mismatch")
    if artifact.get("event_id") != event.get("event_id") or artifact.get("source_chunk_idx") != 0 or artifact.get("semantic_manifest") != expected_semantic_manifest:
        raise ValueError("semantic score provenance does not match source event")
    if artifact.get("source_capture_sha256") != source_capture_sha256 or artifact.get("source_capture_canonical_artifact_sha256") != source_capture.get("canonical_artifact_sha256"):
        raise ValueError("semantic score source capture SHA-256 mismatch")
    provenance, producer = source_capture["provenance"], artifact.get("producer")
    expected_producer = {
        "kind": "slotmem_source_semantic_token_scores",
        "version": 1,
        "source_chunk_idx": 0,
        "target_evidence_read": False,
        "formula": SOURCE_SEMANTIC_FORMULA,
        "subject_char_id": event["character_name"],
        "source_seed": provenance["source_seed"],
        "source_json_sha256": provenance["source_json_sha256"],
        "semantic_vocabulary_sha256": canonical_json_sha256(expected_semantic_manifest),
        "code_identity": provenance["code_identity"],
        "model_identity": provenance["model_identity"],
    }
    if producer != expected_producer:
        raise ValueError("semantic score producer contract does not match source capture")
    expected = {(int(row["bank"]), int(row["layer"])): row for row in subject_captures}
    found = {}
    for row in artifact.get("captures", []):
        address, groups = (int(row["bank"]), int(row["layer"])), row.get("groups")
        source = expected.get(address)
        canonical_address = f"bank_{address[0]}/layer_{address[1]}"
        if source is None or address in found or row.get("address") != canonical_address or row.get("character") != source.get("character") or row.get("character") != event.get("character_name"):
            raise ValueError("semantic score character or capture address does not match")
        if set(groups or {}) != set(SEMANTIC_GROUPS):
            raise ValueError("semantic score capture address or groups are invalid")
        if any(len(groups[name]) != source["raw_tokens"].shape[0] or not torch.isfinite(torch.tensor(groups[name], dtype=torch.float32)).all() for name in SEMANTIC_GROUPS):
            raise ValueError("semantic score vector token count or values are invalid")
        expected_groups = source_metadata_semantic_groups(
            source["raw_token_meta"], event["character_name"]
        )
        if any(groups[name] != expected_groups[name] for name in SEMANTIC_GROUPS):
            raise ValueError("semantic score groups do not match source metadata")
        found[address] = row
    if set(found) != set(expected):
        raise ValueError("semantic scores do not cover every subject payload")
    return found


def build_mask_manifest(*, inputs: Mapping, rankings: Mapping[str, Mapping], event: Mapping, seed: int) -> dict:
    if _has_target_evidence(inputs):
        raise ValueError("target evidence is forbidden in subject subspace inputs")
    layers, seen = [], set()
    for address, row in sorted(rankings.items()):
        bank, group = int(row["bank"]), str(row["layer_group"])
        canonical_address = f"bank_{bank}/group_{group}"
        if address != canonical_address or canonical_address in seen:
            raise ValueError("layer group address must be canonical and unique")
        seen.add(canonical_address)
        members = list(FROZEN_LAYER_GROUPS.get(group, ()))
        if list(row.get("member_layers", [])) != members or set(row.get("source_payload_sha256_by_layer", {})) != {str(layer) for layer in members}:
            raise ValueError("frozen layer group membership or payload hashes are invalid")
        semantic = row["semantic"]
        slot_universe = set(range(FROZEN_MEMORY_ENCODER_SLOTS))
        if len(semantic) != FROZEN_MEMORY_ENCODER_SLOTS or set(semantic) != slot_universe:
            raise ValueError("semantic ranking must be a full frozen slot permutation")
        methods = {name: row.get(name) for name in ("semantic", "visual_cf", "reference")}
        masks = {
            "semantic_top8": sorted(semantic[:FROZEN_SUBJECT_SUBSPACE_BUDGET]),
            "visual_cf_top8": ({"status": "not_available", "reason": "validated_source_counterfactual_payload_missing"} if methods["visual_cf"] is None else sorted(methods["visual_cf"][:FROZEN_SUBJECT_SUBSPACE_BUDGET])),
            "reference_top8": ({"status": "not_available", "reason": "validated_reference_only_payload_missing"} if methods["reference"] is None else sorted(methods["reference"][:FROZEN_SUBJECT_SUBSPACE_BUDGET])),
            "consensus_top8": consensus_rank_indices(methods, FROZEN_SUBJECT_SUBSPACE_BUDGET),
            "random_top8": deterministic_random_indices(event["event_id"], seed, canonical_address, FROZEN_MEMORY_ENCODER_SLOTS, FROZEN_SUBJECT_SUBSPACE_BUDGET),
        }
        layers.append({"bank": bank, "layer_group": group, "member_layers": members, "source_payload_sha256_by_layer": dict(row["source_payload_sha256_by_layer"]), "address": canonical_address, "slot_count": FROZEN_MEMORY_ENCODER_SLOTS, "budget": FROZEN_SUBJECT_SUBSPACE_BUDGET, "rankings": methods, **masks, "mask_sha256": {name: canonical_json_sha256(value) for name, value in masks.items() if isinstance(value, list)}})
    if not layers or event.get("source_chunk_idx") != 0:
        raise ValueError("subject subspace requires source chunk 0 layer rankings")
    banks = {row["bank"] for row in layers}
    if any({row["layer_group"] for row in layers if row["bank"] == bank} != set(FROZEN_LAYER_GROUPS) for bank in banks):
        raise ValueError("each bank requires all frozen layer groups")
    group_order = {name: index for index, name in enumerate(FROZEN_LAYER_GROUPS)}
    layers.sort(key=lambda row: (row["bank"], group_order[row["layer_group"]]))
    manifest = {"schema_version": 1, "event_id": event["event_id"], "character_name": event["character_name"], "seed": int(seed), "source_chunk_idx": 0, "target_evidence_read": False, "primary_mask": "semantic_top8", "budget_fraction": FROZEN_SUBJECT_SUBSPACE_FRACTION, "inputs": dict(inputs), "layers": layers}
    manifest["mask_manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest
