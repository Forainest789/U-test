"""CPU preflight for donor identity and independent teacher provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_donor_entry(manifest: Mapping | list, event: Mapping) -> dict:
    entries = manifest.get("pairs", manifest) if isinstance(manifest, Mapping) else manifest
    if isinstance(entries, Mapping):
        entries = [entries]
    matches = [
        dict(entry)
        for entry in list(entries or [])
        if str(entry.get("target_story_id")) == str(event.get("story_id"))
        and str(entry.get("target_entity_uid")) == str(event.get("entity_uid"))
    ]
    if len(matches) != 1:
        raise ValueError(f"donor manifest must contain exactly one pair for this event, found {len(matches)}")
    return matches[0]


def validate_donor_entry(entry: Mapping, event: Mapping, donor_path: Path) -> dict:
    required = {
        "target_story_id", "target_entity_uid", "donor_story_id", "donor_entity_uid",
        "payload_path", "payload_sha256", "coarse_class", "colour", "character_count",
        "source_visible", "gap_bucket", "slot_shape", "selection_seed", "payload_key",
    }
    missing = sorted(required - set(entry))
    if missing:
        raise ValueError(f"donor manifest missing keys: {missing}")
    if str(entry["target_story_id"]) != str(event.get("story_id")):
        raise ValueError("donor manifest target_story_id does not match event")
    if str(entry["target_entity_uid"]) != str(event.get("entity_uid")):
        raise ValueError("donor manifest target_entity_uid does not match event")
    if str(entry["donor_entity_uid"]) == str(event.get("entity_uid")):
        raise ValueError("wrong donor must have a different entity_uid")
    if str(entry["donor_story_id"]) == str(event.get("story_id")):
        raise ValueError("wrong donor must come from a different story")
    resolved = donor_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"donor payload not found: {resolved}")
    if Path(str(entry["payload_path"])).resolve() != resolved:
        raise ValueError("donor manifest payload_path does not match --donor")
    actual_hash = sha256_file(resolved)
    if str(entry["payload_sha256"]).lower() != actual_hash:
        raise ValueError("donor payload SHA256 does not match manifest")
    return {**dict(entry), "payload_path": str(resolved), "payload_sha256": actual_hash}


def payload_slot_shapes(tokens, payload_key: str) -> dict[str, list[int]]:
    if hasattr(tokens, "shape"):
        return {str(payload_key).rsplit("|", 1)[-1]: [int(value) for value in tokens.shape]}
    if (
        isinstance(tokens, Mapping)
        and bool(tokens.get("__layerwise__", False))
        and isinstance(tokens.get("layers"), Mapping)
    ):
        return {
            str(layer): [int(value) for value in tensor.shape]
            for layer, tensor in tokens["layers"].items()
            if hasattr(tensor, "shape")
        }
    raise ValueError("selected donor payload has no tensor slot shape")


def validate_layerwise_slot_payload(
    payload: object,
    *,
    expected_layers: Sequence[int],
    expected_slots: int,
) -> dict[str, list[int]]:
    import torch

    expected_keys = tuple(str(int(layer)) for layer in expected_layers)
    if not isinstance(payload, Mapping) or payload.get("__layerwise__") is not True:
        raise ValueError("selected donor payload must be a layerwise tensor payload")
    layers = payload.get("layers")
    if not isinstance(layers, Mapping) or not layers:
        raise ValueError("selected donor payload layers must not be empty")
    if any(type(layer) is not str or not layer for layer in layers):
        raise ValueError("selected donor payload layers must use non-empty string keys")
    if set(layers) != set(expected_keys):
        raise ValueError(f"selected donor payload layers must be exactly {list(expected_keys)}")

    shapes: dict[str, list[int]] = {}
    hidden_dims: set[int] = set()
    for layer in expected_keys:
        tensor = layers[layer]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"selected donor payload layer {layer} must be a tensor")
        if tensor.ndim != 2:
            raise ValueError(f"selected donor payload layer {layer} must be a 2D tensor")
        if int(tensor.shape[0]) != int(expected_slots):
            raise ValueError(
                f"selected donor payload layer {layer} must be a {expected_slots}-slot tensor"
            )
        if not tensor.is_floating_point():
            raise ValueError(f"selected donor payload layer {layer} must be floating point")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"selected donor payload layer {layer} must be finite")
        shapes[layer] = [int(value) for value in tensor.shape]
        hidden_dims.add(int(tensor.shape[1]))
    if len(hidden_dims) != 1:
        raise ValueError("selected donor payload layers must share one hidden dimension")
    return shapes


def validate_donor_bundle(
    event: Mapping,
    donor_path: Path,
    manifest_path: Path,
    *,
    loader: Callable[[Path], object] | None = None,
) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = validate_donor_entry(select_donor_entry(manifest, event), event, donor_path)
    if loader is None:
        import torch

        loader = lambda path: torch.load(path, map_location="cpu", weights_only=False)
    artifact = loader(donor_path.resolve())
    if not isinstance(artifact, Mapping) or artifact.get("format") != "slotmem_donor_payload_v2":
        raise ValueError("donor payload must use slotmem_donor_payload_v2")
    payload_event = artifact.get("event")
    payloads = artifact.get("payloads")
    if not isinstance(payload_event, Mapping) or not isinstance(payloads, Mapping):
        raise ValueError("donor payload is missing embedded event or payloads")
    for field in ("story_id", "entity_uid"):
        expected = entry[f"donor_{field}"]
        if str(payload_event.get(field)) != str(expected):
            raise ValueError(f"donor_{field} does not match payload event")
    payload_key = str(entry["payload_key"])
    if payload_key not in payloads:
        raise ValueError(f"donor payload_key not found: {payload_key}")
    character = str(payload_event.get("character_name", "")).strip()
    if not character or payload_key.rsplit("|", 1)[0].casefold() != character.casefold():
        raise ValueError("donor payload_key character does not match payload event")
    shapes = payload_slot_shapes(payloads[payload_key], payload_key)
    frozen_shape = entry["slot_shape"]
    if isinstance(frozen_shape, Mapping):
        normalized = {str(key): [int(value) for value in shape] for key, shape in frozen_shape.items()}
        if normalized != shapes:
            raise ValueError(f"donor slot_shape mismatch: manifest={normalized} payload={shapes}")
    elif isinstance(frozen_shape, list) and len(shapes) == 1:
        if [int(value) for value in frozen_shape] != next(iter(shapes.values())):
            raise ValueError("donor slot_shape does not match payload")
    else:
        raise ValueError("donor slot_shape must be an exact list or layer-to-shape mapping")
    return {
        "status": "passed",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "payload_path": entry["payload_path"],
        "payload_sha256": entry["payload_sha256"],
        "target_story_id": entry["target_story_id"],
        "target_entity_uid": entry["target_entity_uid"],
        "donor_story_id": entry["donor_story_id"],
        "donor_entity_uid": entry["donor_entity_uid"],
        "payload_key": payload_key,
        "slot_shapes": shapes,
    }


def validate_teacher_bundle(
    event: Mapping,
    video_path: Path,
    manifest_path: Path,
    *,
    arms_root: Path | None = None,
) -> dict:
    video_path = video_path.resolve()
    manifest_path = manifest_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"future target video not found: {video_path}")
    if arms_root is not None:
        try:
            video_path.relative_to(arms_root.resolve())
        except ValueError:
            pass
        else:
            raise ValueError("future target video is inside evaluated arm outputs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "story_id", "target_chunk_idx", "video_path", "video_sha256", "source_type",
        "generated_by_arm", "generated_by_evaluated_model",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"teacher manifest missing keys: {missing}")
    if str(manifest["story_id"]) != str(event.get("story_id")):
        raise ValueError("teacher story_id does not match event")
    if int(manifest["target_chunk_idx"]) != int(event.get("target_chunk_idx")):
        raise ValueError("teacher target_chunk_idx does not match event")
    if Path(str(manifest["video_path"])).resolve() != video_path:
        raise ValueError("teacher manifest video_path does not match target video")
    actual_hash = sha256_file(video_path)
    if str(manifest["video_sha256"]).lower() != actual_hash:
        raise ValueError("teacher video SHA256 does not match manifest")
    if str(manifest["source_type"]) not in {"held_out_real", "independent_teacher"}:
        raise ValueError("teacher source_type must be held_out_real or independent_teacher")
    if manifest["generated_by_arm"] is not False or manifest["generated_by_evaluated_model"] is not False:
        raise ValueError("teacher manifest must declare an arm-independent target")
    return {
        "status": "passed",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "video_path": str(video_path),
        "video_sha256": actual_hash,
        "source_type": str(manifest["source_type"]),
        "story_id": str(manifest["story_id"]),
        "target_chunk_idx": int(manifest["target_chunk_idx"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--donor-manifest", type=Path, required=True)
    parser.add_argument("--future-target-video", type=Path, required=True)
    parser.add_argument("--future-target-manifest", type=Path, required=True)
    parser.add_argument("--arms-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    event = json.loads(args.event.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "status": "passed",
        "donor": validate_donor_bundle(event, args.donor, args.donor_manifest),
        "teacher": validate_teacher_bundle(
            event,
            args.future_target_video,
            args.future_target_manifest,
            arms_root=args.arms_root,
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
