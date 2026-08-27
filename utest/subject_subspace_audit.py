"""Validate and apply frozen subject-slot masks at SlotMem's reader boundary."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

import torch

from .content_audit import LAYERS_KEY, _is_layerwise, _slot_row_indices, install, transform_slot_rows
from .input_contract import payload_slot_shapes, select_donor_entry, validate_donor_entry
from .prefix_contract import build_runtime_contract
from .subject_subspace import FROZEN_LAYER_GROUPS, canonical_json_sha256, capture_tensor_sha256


SUBSPACE_ARMS = (
    "full_correct",
    "no_memory",
    "zero_path",
    "subject_only",
    "drop_subject",
    "random_only",
    "drop_random",
    "wrong_subject",
)


def _contains_target_evidence(value: object, *, allow_marker: bool = False) -> bool:
    if isinstance(value, Mapping):
        return any(
            (
                str(key).casefold().startswith(("target_", "qstar", "cids", "decoded_"))
                and not (allow_marker and key == "target_evidence_read")
            )
            or _contains_target_evidence(item)
            for key, item in value.items()
        )
    return isinstance(value, (list, tuple)) and any(_contains_target_evidence(item) for item in value)


def _valid_mask(value: object, *, slots: int, budget: int) -> bool:
    if not isinstance(value, list) or len(value) != budget:
        return False
    try:
        _slot_row_indices(slots, "subject_only", {"semantic_top8": value})
    except ValueError:
        return False
    return True


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value.casefold()) <= set("0123456789abcdef")


def validate_subject_subspace_manifest(manifest: Mapping, event: Mapping, *, seed: int) -> dict[int, dict[str, dict]]:
    """Return a bank/layer lookup only after the frozen mask contract validates."""
    canonical = {key: value for key, value in manifest.items() if key != "mask_manifest_sha256"}
    event_seeds = [event[key] for key in ("seed", "source_seed", "target_seed") if event.get(key) is not None]
    if (
        manifest.get("schema_version") != 1
        or manifest.get("mask_manifest_sha256") != canonical_json_sha256(canonical)
        or manifest.get("event_id") != event.get("event_id")
        or manifest.get("character_name") != event.get("character_name")
        or manifest.get("source_chunk_idx") != 0
        or event.get("source_chunk_idx") != 0
        or manifest.get("target_evidence_read") is not False
        or manifest.get("primary_mask") != "semantic_top8"
        or manifest.get("budget_fraction") != 0.25
    ):
        raise ValueError("subject subspace manifest provenance is invalid")
    if _contains_target_evidence(manifest, allow_marker=True):
        raise ValueError("subject subspace manifest contains target evidence")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping) or not _is_sha256(inputs.get("source_capture_sha256")):
        raise ValueError("subject subspace source capture SHA-256 is invalid")
    all_seeds = [seed, manifest.get("seed"), *event_seeds]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in all_seeds) or any(value != seed for value in all_seeds):
        raise ValueError("subject subspace seed does not match event")

    banks: dict[int, dict[str, dict]] = {}
    seen_groups = set()
    rows = manifest.get("layers")
    if not isinstance(rows, list):
        raise ValueError("subject subspace layers must be a list")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("subject subspace layer group must be an object")
        bank, group = row.get("bank"), str(row.get("layer_group", ""))
        if isinstance(bank, bool) or not isinstance(bank, int) or group not in FROZEN_LAYER_GROUPS:
            raise ValueError("subject subspace bank or layer group is invalid")
        address = f"bank_{bank}/group_{group}"
        if row.get("address") != address or (bank, group) in seen_groups:
            raise ValueError("subject subspace layer group address is invalid or duplicated")
        seen_groups.add((bank, group))
        members = list(FROZEN_LAYER_GROUPS[group])
        hashes = row.get("source_payload_sha256_by_layer", {})
        if not isinstance(hashes, Mapping):
            raise ValueError("subject subspace payload hashes must be an object")
        if row.get("member_layers") != members or set(hashes) != {str(layer) for layer in members}:
            raise ValueError("subject subspace member layers are incomplete")
        if row.get("slot_count") != 32 or row.get("budget") != 8:
            raise ValueError("subject subspace slot count or budget is invalid")
        masks = {name: row.get(name) for name in ("semantic_top8", "random_top8")}
        if any(not _valid_mask(value, slots=32, budget=8) for value in masks.values()):
            raise ValueError("subject subspace mask indices are invalid")
        expected_mask_hashes = {name: canonical_json_sha256(value) for name, value in masks.items()}
        actual_mask_hashes = row.get("mask_sha256", {})
        if any(actual_mask_hashes.get(name) != digest for name, digest in expected_mask_hashes.items()):
            raise ValueError("subject subspace mask SHA-256 is invalid")
        for layer in members:
            digest = hashes[str(layer)]
            if not _is_sha256(digest):
                raise ValueError("source payload SHA-256 is invalid")
            layer_key = str(layer)
            if layer_key in banks.setdefault(bank, {}):
                raise ValueError(f"subject subspace layer {layer_key} is duplicated")
            banks[bank][layer_key] = {
                **masks,
                "layer_group": group,
                "slot_count": 32,
                "source_payload_sha256": digest,
            }
    if not banks or any({group for candidate_bank, group in seen_groups if candidate_bank == bank} != set(FROZEN_LAYER_GROUPS) for bank in banks):
        raise ValueError("subject subspace bank is missing a frozen layer group")
    return banks


def validate_subject_payload(payload: Mapping, *, bank_idx: int, layers: Mapping[str, Mapping]) -> None:
    """Fail closed unless one live target payload exactly matches its source capture."""
    tokens, metadata = payload.get("tokens"), payload.get("token_meta")
    if not _is_layerwise(tokens) or not _is_layerwise(metadata):
        raise ValueError(f"bank {bank_idx} requires layerwise tokens and token metadata")
    token_layers, meta_layers = tokens[LAYERS_KEY], metadata[LAYERS_KEY]
    if set(token_layers) != set(layers) or set(meta_layers) != set(layers):
        raise ValueError(f"bank {bank_idx} payload layers do not match the mask manifest")
    for layer, contract in layers.items():
        tensor, meta = token_layers[layer], meta_layers[layer]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[0] != contract["slot_count"]:
            raise ValueError(f"bank {bank_idx} layer {layer} slot count is invalid")
        if not isinstance(meta, list) or len(meta) != tensor.shape[0]:
            raise ValueError(f"bank {bank_idx} layer {layer} token metadata is invalid")
        if capture_tensor_sha256(tensor) != contract["source_payload_sha256"]:
            raise ValueError(f"bank {bank_idx} layer {layer} source payload SHA-256 mismatch")


def validate_frozen_donor_artifact(artifact: Mapping, entry: Mapping, *, banks: Mapping[int, Mapping]):
    """Validate the already-loaded donor bytes and return the selected token payload."""
    if not isinstance(artifact, Mapping) or artifact.get("format") != "slotmem_donor_payload_v2":
        raise ValueError("donor payload must use slotmem_donor_payload_v2")
    payload_event, payloads = artifact.get("event"), artifact.get("payloads")
    if not isinstance(payload_event, Mapping) or not isinstance(payloads, Mapping):
        raise ValueError("donor payload is missing embedded event or payloads")
    for field in ("story_id", "entity_uid"):
        if str(payload_event.get(field)) != str(entry[f"donor_{field}"]):
            raise ValueError(f"donor_{field} does not match payload event")
    payload_key = str(entry["payload_key"])
    if payload_key not in payloads:
        raise ValueError(f"donor payload_key not found: {payload_key}")
    character = str(payload_event.get("character_name", "")).strip()
    if not character or payload_key.rsplit("|", 1)[0].casefold() != character.casefold():
        raise ValueError("donor payload_key character does not match payload event")
    try:
        bank = int(payload_key.rsplit("|", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError("donor payload_key must end with a bank index") from exc
    if bank not in banks:
        raise ValueError(f"donor payload_key bank {bank} is absent from the mask manifest")
    selected = payloads[payload_key]
    shapes = payload_slot_shapes(selected, payload_key)
    frozen_shape = entry["slot_shape"]
    if not isinstance(frozen_shape, Mapping) or {
        str(layer): [int(value) for value in shape] for layer, shape in frozen_shape.items()
    } != shapes:
        raise ValueError("donor slot_shape does not match payload")
    if not _is_layerwise(selected) or set(selected[LAYERS_KEY]) != set(banks[bank]):
        raise ValueError("donor payload layers do not match the mask manifest")
    for layer, contract in banks[bank].items():
        tensor = selected[LAYERS_KEY][layer]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 2
            or tensor.shape[0] != contract["slot_count"]
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"donor layer {layer} must be a finite 2D 32-slot tensor")
    return selected


def install_subject_subspace(
    *,
    arm: str,
    seed: int,
    manifest: Mapping,
    event: Mapping,
    report_path: Path,
    donor_path: Path | None = None,
    donor_entry: Mapping | None = None,
    donor_artifact: Mapping | None = None,
    runtime_contract: Mapping | None = None,
    event_file_sha256: str | None = None,
    manifest_file_sha256: str | None = None,
    donor_provenance: Mapping | None = None,
):
    """Install one validated eight-arm intervention using the existing audit seam."""
    if arm not in SUBSPACE_ARMS:
        raise ValueError(f"unknown subject-subspace arm: {arm}")
    banks = validate_subject_subspace_manifest(manifest, event, seed=seed)
    if arm == "wrong_subject":
        if not isinstance(donor_artifact, Mapping) or not isinstance(donor_entry, Mapping):
            raise ValueError("wrong_subject requires a frozen donor artifact and entry")
        validate_frozen_donor_artifact(donor_artifact, donor_entry, banks=banks)
    elif donor_artifact is not None or donor_entry is not None or donor_path is not None:
        raise ValueError("donor inputs are only valid for wrong_subject")
    report_path = Path(report_path)
    if report_path.exists():
        raise FileExistsError(report_path)
    mapped = {"full_correct": "correct", "zero_path": "zero"}.get(arm, arm)
    provenance = {
        "event_id": event["event_id"],
        "seed": seed,
        "target_evidence_read": False,
        "mask_manifest_sha256": manifest["mask_manifest_sha256"],
        "source_capture_sha256": manifest["inputs"]["source_capture_sha256"],
        "event_file_sha256": event_file_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        **dict(donor_provenance or {}),
    }
    return install(
        mapped,
        seed,
        str(donor_path) if donor_path else None,
        None,
        str(report_path),
        event=dict(event),
        donor_entry=dict(donor_entry) if donor_entry else None,
        runtime_contract=dict(runtime_contract or {}),
        subject_contract={
            "arm": arm,
            "banks": banks,
            "provenance": provenance,
            **({"donor_artifact": donor_artifact} if donor_artifact is not None else {}),
        },
    )


def _self_check() -> None:
    tokens = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    masks = {"semantic_top8": [1, 3], "random_top8": [0, 2]}
    assert torch.equal(transform_slot_rows(tokens, "subject_only", masks), tokens[[1, 3]])
    assert torch.equal(transform_slot_rows(tokens, "drop_random", masks), tokens[[1, 3]])
    print("[subject-subspace-audit] self-check OK")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=SUBSPACE_ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--event-json", type=Path)
    parser.add_argument("--donor", type=Path)
    parser.add_argument("--donor-manifest", type=Path)
    parser.add_argument("--report", type=Path, default=Path("subject_subspace_audit.json"))
    parser.add_argument("--slotmem-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.self_check:
        _self_check()
        return 0
    if any(value is None for value in (args.arm, args.seed, args.manifest, args.event_json)):
        parser.error("runs require --arm --seed --manifest and --event-json")
    if args.arm == "wrong_subject" and not (args.donor and args.donor_manifest):
        parser.error("--arm wrong_subject requires --donor and --donor-manifest")
    if args.arm != "wrong_subject" and (args.donor or args.donor_manifest):
        parser.error("donor inputs are only valid for --arm wrong_subject")

    event_bytes, manifest_bytes = args.event_json.read_bytes(), args.manifest.read_bytes()
    event = json.loads(event_bytes.decode("utf-8-sig"))
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    if not isinstance(event, dict) or not isinstance(manifest, dict):
        raise ValueError("event and subject subspace manifest must be JSON objects")
    banks = validate_subject_subspace_manifest(manifest, event, seed=args.seed)
    donor_entry = donor_artifact = donor_provenance = None
    if args.arm == "wrong_subject":
        donor_manifest_bytes = args.donor_manifest.read_bytes()
        donor_manifest = json.loads(donor_manifest_bytes.decode("utf-8-sig"))
        donor_entry = validate_donor_entry(
            select_donor_entry(donor_manifest, event), event, args.donor
        )
        donor_bytes = args.donor.read_bytes()
        if hashlib.sha256(donor_bytes).hexdigest() != donor_entry["payload_sha256"]:
            raise ValueError("donor payload changed after manifest validation")
        donor_artifact = torch.load(io.BytesIO(donor_bytes), map_location="cpu", weights_only=True)
        validate_frozen_donor_artifact(donor_artifact, donor_entry, banks=banks)
        donor_provenance = {
            "donor_manifest_sha256": hashlib.sha256(donor_manifest_bytes).hexdigest(),
            "donor_payload_sha256": donor_entry["payload_sha256"],
            "donor_payload_key": donor_entry["payload_key"],
            "donor_story_id": donor_entry["donor_story_id"],
            "donor_entity_uid": donor_entry["donor_entity_uid"],
        }
    slotmem_dir = args.slotmem_dir.resolve()
    if not (slotmem_dir / "infer_slotmem.py").is_file():
        parser.error(f"no infer_slotmem.py under {slotmem_dir}")
    sys.path.insert(0, str(slotmem_dir))
    rest = list(args.rest[1:] if args.rest[:1] == ["--"] else args.rest)
    flush = install_subject_subspace(
        arm=args.arm,
        seed=args.seed,
        manifest=manifest,
        event=event,
        report_path=args.report,
        donor_path=args.donor,
        donor_entry=donor_entry,
        donor_artifact=donor_artifact,
        runtime_contract=build_runtime_contract(event, rest),
        event_file_sha256=hashlib.sha256(event_bytes).hexdigest(),
        manifest_file_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        donor_provenance=donor_provenance,
    )
    import infer_slotmem

    sys.argv = ["infer_slotmem.py", *rest]
    try:
        infer_slotmem.main()
    finally:
        flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
