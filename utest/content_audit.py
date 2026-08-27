#!/usr/bin/env python3
"""Content-causality audit for SlotMem from an already-frozen prefix.

SlotMem reports consistency scores only, which cannot separate "the memory of THIS
character is doing the work" from "any memory-shaped side signal smooths the video".
This is the FU-MD five-arm apparatus pointed at the frozen SlotMem checkpoint.

It does not fork their code. Generation reads funnel through the dedicated
``RoleWiseSlotMemoryBank.get_memory_payload_for_read`` boundary, while writer updates
continue using the unmodified bank accessor. We patch only the reader boundary and call
their ``main()`` with its own argv.

Arms:
  no_memory return no payload at the reader boundary.
  correct   passthrough. With --dump-donor, saves payloads for a later wrong arm.
  zero      tokens zeroed, shapes and metadata kept.
  wrong     tokens replaced by one pre-frozen, manifest-validated donor payload.
  random    deterministic Gaussian tokens matching each feature channel's moments.

Usage (on the server, after scripts/fetch_weights.sh):
  python -m utest.content_audit --arm correct --dump-donor donor_3_271.pt \
      -- --ckpt_dir $CKPT_DIR --json_path ... (everything test_slotmem_stage1.sh passes)
  python -m utest.content_audit --arm wrong --donor donor_3_271.pt -- ...

  python -m utest.content_audit --self-check   # no GPU, no weights, no clone
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Callable, Mapping

try:
    import torch
except ModuleNotFoundError:  # Command planning and contract checks do not need tensors.
    torch = None

from .prefix_contract import build_runtime_contract
from .input_contract import (
    select_donor_entry,
    validate_donor_bundle,
    validate_donor_entry,
)

ARMS = ("no_memory", "zero", "correct", "wrong", "random")

LAYERWISE_MARKER = "__layerwise__"
LAYERS_KEY = "layers"


def intervention_applies(
    event: Mapping, character: object, chunk_idx: object
) -> bool:
    """Return whether this read is the one frozen treatment address."""
    try:
        return (
            str(character).strip().casefold() == str(event["character_name"]).strip().casefold()
            and int(chunk_idx) == int(event["target_chunk_idx"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def stable_transform_seed(
    event: Mapping,
    target_idx: int,
    character: object,
    bank_idx: int,
    arm_seed: int,
    layer: object,
) -> int:
    """Low 64 bits of a canonical SHA256 transform address."""
    canonical = json.dumps(
        [
            event.get("story_id"),
            event.get("event_id"),
            int(target_idx),
            str(character),
            int(bank_idx),
            int(arm_seed),
            str(layer),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[-8:], "big")


def _is_layerwise(value) -> bool:
    return (
        isinstance(value, dict)
        and bool(value.get(LAYERWISE_MARKER, False))
        and isinstance(value.get(LAYERS_KEY, None), dict)
    )


def _match_rows(donor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Require the pre-frozen donor tensor to match the target exactly."""
    if tuple(donor.shape) != tuple(target.shape):
        raise ValueError(
            f"wrong donor must have exact shape {tuple(target.shape)}, "
            f"got {tuple(donor.shape)}"
        )
    return donor


def _moment_matched_gaussian(tokens: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Gaussian values with exact source moments over the slot dimension."""
    source = tokens.detach().to(device="cpu", dtype=torch.float32)
    if source.ndim < 2:
        raise ValueError(f"memory tokens must have a slot and feature dimension, got {tuple(source.shape)}")
    noise = torch.randn(source.shape, generator=gen, dtype=torch.float32, device="cpu")
    source_mean = source.mean(dim=0, keepdim=True)
    source_std = source.std(dim=0, correction=0, keepdim=True)
    if int(source.shape[0]) <= 1:
        matched = source_mean.expand_as(source)
    else:
        noise_mean = noise.mean(dim=0, keepdim=True)
        noise_std = noise.std(dim=0, correction=0, keepdim=True)
        normalized = (noise - noise_mean) / noise_std.clamp_min(torch.finfo(torch.float32).eps)
        matched = normalized * source_std + source_mean
        matched = torch.where(source_std > 0, matched, source_mean.expand_as(matched))
    if not torch.allclose(matched.mean(dim=0), source_mean.squeeze(0), atol=1e-5, rtol=1e-5):
        raise RuntimeError("random arm failed float32 channel-mean matching")
    if not torch.allclose(
        matched.std(dim=0, correction=0), source_std.squeeze(0), atol=1e-5, rtol=1e-5
    ):
        raise RuntimeError("random arm failed float32 channel-std matching")
    return matched.to(device=tokens.device, dtype=tokens.dtype)


def transform_tokens(
    tokens: torch.Tensor, arm: str, gen: torch.Generator | None, donor=None
) -> torch.Tensor | None:
    if arm == "no_memory":
        return None
    if arm == "correct":
        return tokens
    if arm == "zero":
        return torch.zeros_like(tokens)
    if arm == "random":
        if gen is None:
            raise ValueError("random arm requires a generator")
        return _moment_matched_gaussian(tokens, gen)
    if arm == "wrong":
        if donor is None:
            raise ValueError("wrong arm reached a layer with no donor tensor")
        return _match_rows(donor.to(tokens.device, tokens.dtype), tokens).clone()
    raise ValueError(f"unknown arm: {arm}")


def _slot_row_indices(rows: int, arm: str, masks: Mapping[str, list[int]]) -> list[int]:
    mask_name = "random_top8" if arm in {"random_only", "drop_random"} else "semantic_top8"
    if arm not in {"subject_only", "drop_subject", "random_only", "drop_random", "wrong_subject"}:
        raise ValueError(f"unknown subject-subspace arm: {arm}")
    indices = masks.get(mask_name)
    if not isinstance(indices, list) or not indices:
        raise ValueError(f"{mask_name} must contain unique ascending in-range slot indices")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError(f"{mask_name} must contain unique ascending in-range slot indices")
    if indices != sorted(indices) or len(indices) != len(set(indices)) or indices[0] < 0 or indices[-1] >= rows:
        raise ValueError(f"{mask_name} must contain unique ascending in-range slot indices")
    selected = indices if arm.endswith("_only") or arm == "wrong_subject" else [index for index in range(rows) if index not in set(indices)]
    if not selected:
        raise ValueError(f"{arm} selected an empty slot payload")
    return selected


def _select_slot_rows(tokens: torch.Tensor, arm: str, selected: list[int], donor=None) -> torch.Tensor:
    if arm == "wrong_subject":
        if donor is None:
            raise ValueError("wrong_subject requires donor tokens")
        tokens = _match_rows(donor.to(tokens.device, tokens.dtype), tokens)
    return tokens[selected]


def transform_slot_rows(
    tokens: torch.Tensor,
    arm: str,
    masks: Mapping[str, list[int]],
    donor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one frozen equal-budget slot mask to a 2D payload tensor."""
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or not tokens.shape[0]:
        raise ValueError("slot payload must be a nonempty 2D tensor")
    selected = _slot_row_indices(int(tokens.shape[0]), arm, masks)
    return _select_slot_rows(tokens, arm, selected, donor)


def transform_slot_payload(payload, arm: str, masks_by_layer: Mapping[str, Mapping], donor_tokens=None):
    """Select layerwise token and metadata rows together at the reader boundary."""
    tokens = payload.get("tokens") if isinstance(payload, Mapping) else None
    metadata = payload.get("token_meta") if isinstance(payload, Mapping) else None
    if not _is_layerwise(tokens) or not _is_layerwise(metadata):
        raise ValueError("subject-subspace arms require layerwise tokens and token_meta")
    token_layers, meta_layers = tokens[LAYERS_KEY], metadata[LAYERS_KEY]
    if set(token_layers) != set(meta_layers):
        raise ValueError("layerwise token metadata does not match token layers")
    donor_layers = donor_tokens.get(LAYERS_KEY, {}) if _is_layerwise(donor_tokens) else {}
    if arm == "wrong_subject" and set(donor_layers) != set(token_layers):
        raise ValueError("wrong_subject donor layers must exactly match target layers")
    output_tokens, output_meta, selected = {}, {}, {}
    for layer, layer_tokens in token_layers.items():
        key = str(layer)
        masks = masks_by_layer.get(key)
        if masks is None:
            raise ValueError(f"mask manifest has no layer {key}")
        layer_meta = meta_layers[layer]
        if not isinstance(layer_meta, list) or len(layer_meta) != int(layer_tokens.shape[0]):
            raise ValueError(f"token metadata row count does not match layer {key}")
        donor = donor_layers.get(layer)
        rows = _slot_row_indices(int(layer_tokens.shape[0]), arm, masks)
        output_tokens[layer] = _select_slot_rows(layer_tokens, arm, rows, donor)
        output_meta[layer] = [layer_meta[index] for index in rows]
        selected[key] = list(rows)
    return {
        **payload,
        "tokens": {LAYERWISE_MARKER: True, LAYERS_KEY: output_tokens},
        "token_meta": {LAYERWISE_MARKER: True, LAYERS_KEY: output_meta},
    }, len(output_tokens), selected


def transform_payload(
    payload,
    arm: str,
    gen: torch.Generator | None,
    donor_tokens=None,
    *,
    generator_for_layer: Callable[[str], torch.Generator] | None = None,
):
    """Pure function over a get_memory_payload() return value. Returns (new_payload, n_layers)."""
    if payload is None or arm == "correct":
        return payload, 0
    if arm == "no_memory":
        return None, 0
    tokens = payload.get("tokens")

    if isinstance(tokens, torch.Tensor):
        donor = donor_tokens if isinstance(donor_tokens, torch.Tensor) else None
        layer_gen = generator_for_layer("shared") if generator_for_layer else gen
        return {**payload, "tokens": transform_tokens(tokens, arm, layer_gen, donor)}, 1

    if _is_layerwise(tokens):
        donor_layers = donor_tokens.get(LAYERS_KEY, {}) if _is_layerwise(donor_tokens) else {}
        out, n = {}, 0
        for layer, t in tokens.get(LAYERS_KEY, {}).items():
            if not isinstance(t, torch.Tensor):
                out[layer] = t
                continue
            layer_donor = donor_layers.get(layer)
            if arm == "wrong" and layer_donor is None:
                raise ValueError(f"wrong arm donor has no tensor for layer {layer}")
            layer_gen = generator_for_layer(str(layer)) if generator_for_layer else gen
            out[layer] = transform_tokens(t, arm, layer_gen, layer_donor)
            n += 1
        return {**payload, "tokens": {LAYERWISE_MARKER: True, LAYERS_KEY: out}}, n

    return payload, 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_tensor_artifact(path: Path | str):
    return torch.load(path, map_location="cpu", weights_only=True)


def validate_donor_manifest(entry: dict, event: dict, donor_path: Path) -> dict:
    """Validate one pre-frozen target/donor pair before loading its tensors."""
    return validate_donor_entry(entry, event, donor_path)


def _tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().to(device="cpu").view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _payload_sha256(payload) -> str | None:
    tokens = payload.get("tokens") if isinstance(payload, dict) else None
    if isinstance(tokens, torch.Tensor):
        return _tensor_sha256(tokens) if hasattr(tokens, "detach") else None
    if _is_layerwise(tokens):
        layer_hashes = {
            str(layer): _tensor_sha256(tensor)
            for layer, tensor in tokens.get(LAYERS_KEY, {}).items()
            if isinstance(tensor, torch.Tensor) and hasattr(tensor, "detach")
        }
        encoded = json.dumps(layer_hashes, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return None


def _payload_summary(payload) -> dict:
    tokens = payload.get("tokens") if isinstance(payload, dict) else None
    if isinstance(tokens, torch.Tensor):
        return {
            "layers": 1,
            "slots": int(tokens.shape[0]),
            "shapes": {"shared": list(tokens.shape)},
            "dtypes": {"shared": str(getattr(tokens, "dtype", "unknown"))},
        }
    if _is_layerwise(tokens):
        tensors = {
            str(layer): tensor
            for layer, tensor in tokens.get(LAYERS_KEY, {}).items()
            if isinstance(tensor, torch.Tensor)
        }
        return {
            "layers": len(tensors),
            "slots": sum(int(tensor.shape[0]) for tensor in tensors.values()),
            "shapes": {layer: list(tensor.shape) for layer, tensor in tensors.items()},
            "dtypes": {layer: str(tensor.dtype) for layer, tensor in tensors.items()},
        }
    return {"layers": 0, "slots": 0, "shapes": {}, "dtypes": {}}


def _layer_tensor_hashes(payload, hasher: Callable[[torch.Tensor], str]) -> dict[str, str]:
    tokens = payload.get("tokens") if isinstance(payload, Mapping) else None
    if not _is_layerwise(tokens):
        return {}
    return {
        str(layer): hasher(tensor)
        for layer, tensor in tokens[LAYERS_KEY].items()
        if isinstance(tensor, torch.Tensor)
    }


def _write_report(path: Path, stats: Mapping, *, exclusive: bool) -> None:
    data = json.dumps(stats, indent=2).encode("utf-8")
    if not exclusive:
        path.write_bytes(data)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    created = False
    try:
        with temporary.open("xb") as handle:
            created = True
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if created and temporary.exists():
            temporary.unlink()


def _cpu_clone(tokens):
    if isinstance(tokens, torch.Tensor):
        return tokens.detach().cpu().clone()
    if _is_layerwise(tokens):
        return {
            LAYERWISE_MARKER: True,
            LAYERS_KEY: {
                k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v)
                for k, v in tokens.get(LAYERS_KEY, {}).items()
            },
        }
    return None


def install(
    arm: str,
    seed: int,
    donor_path: str | None,
    dump_path: str | None,
    report_path: str,
    *,
    event: dict | None = None,
    donor_entry: dict | None = None,
    runtime_contract: dict | None = None,
    subject_contract: Mapping | None = None,
):
    """Patch SlotMem's single memory read point. Call before infer_slotmem.main()."""
    import infer_slotmem

    gen = None
    loaded_donor = (
        subject_contract["donor_artifact"]
        if subject_contract and "donor_artifact" in subject_contract
        else _load_tensor_artifact(donor_path) if donor_path else {}
    )
    donor = (
        loaded_donor.get("payloads", {})
        if isinstance(loaded_donor, dict) and loaded_donor.get("format") == "slotmem_donor_payload_v2"
        else loaded_donor
    )
    donor_key = str(donor_entry.get("payload_key")) if donor_entry and donor_entry.get("payload_key") else None
    if arm in {"wrong", "wrong_subject"}:
        if donor_key is None or not isinstance(donor, dict) or donor_key not in donor:
            raise ValueError("wrong donor manifest must select one existing payload_key")
        selected_donor = donor[donor_key]
    else:
        selected_donor = None
    dumped: dict = {}
    target_character = str((event or {}).get("character_name", ""))
    stats = {
        "arm": str(subject_contract.get("arm", arm)) if subject_contract else arm,
        "seed": seed,
        "attempted_reads": 0,
        "source_non_null_reads": 0,
        "returned_non_null_reads": 0,
        "target_source_non_null_reads": 0,
        "target_returned_non_null_reads": 0,
        "payload_layers_seen": 0,
        "layers_transformed": 0,
        "target_character": target_character or None,
        "target_chunk_idx": int((event or {}).get("target_chunk_idx", -1)),
        "target_read_hits": 0,
        "native_read_mismatches": 0,
        "target_read_mismatches": 0,
        "read_records": [],
        "runtime_contract": dict(runtime_contract or {}),
    }
    if subject_contract:
        stats["subject_subspace_contract"] = dict(subject_contract["provenance"])

    reader_class = infer_slotmem.RoleWiseSlotMemoryBank
    original = reader_class.get_memory_payload_for_read

    def patched(self, char_id, bank_idx=0):
        payload = original(self, char_id, bank_idx)
        stats["attempted_reads"] += 1
        chunk_idx = getattr(self, "current_chunk_idx", None)
        is_target = intervention_applies(event or {}, char_id, chunk_idx)
        summary = _payload_summary(payload)
        record = {
            "chunk_idx": int(chunk_idx) if chunk_idx is not None else None,
            "character": str(char_id),
            "bank": int(bank_idx),
            "source_present": payload is not None and int(summary["layers"]) > 0,
            "source_sha256": _payload_sha256(payload),
            **summary,
        }
        if is_target:
            stats["target_read_hits"] += 1
        if payload is None:
            record["returned_present"] = False
            record["layers_transformed"] = 0
            stats["read_records"].append(record)
            return None
        stats["source_non_null_reads"] += 1
        if is_target:
            stats["target_source_non_null_reads"] += 1
            stats["payload_layers_seen"] += int(summary["layers"])
        key = f"{char_id}|{bank_idx}"
        if dump_path is not None and is_target:
            dumped.setdefault(key, _cpu_clone(payload.get("tokens")))
        if is_target:
            if subject_contract:
                from .subject_subspace import capture_tensor_sha256
                from .subject_subspace_audit import validate_subject_payload

                layers = subject_contract["banks"].get(int(bank_idx))
                if layers is None:
                    raise ValueError(f"mask manifest has no bank {bank_idx}")
                validate_subject_payload(payload, bank_idx=int(bank_idx), layers=layers)
                record["source_manifest_sha256_by_layer"] = _layer_tensor_hashes(
                    payload, capture_tensor_sha256
                )
                if arm not in {
                    "subject_only", "drop_subject", "random_only", "drop_random", "wrong_subject"
                }:
                    keep = arm != "no_memory"
                    record["selected_indices_by_layer"] = {
                        layer: list(range(contract["slot_count"])) if keep else []
                        for layer, contract in layers.items()
                    }
            generator_for_layer = None
            if arm == "random":
                generator_for_layer = lambda layer: torch.Generator().manual_seed(
                    stable_transform_seed(
                        event or {},
                        int(chunk_idx),
                        char_id,
                        int(bank_idx),
                        int(seed),
                        layer,
                    )
                )
            if subject_contract and arm in {
                "subject_only", "drop_subject", "random_only", "drop_random", "wrong_subject"
            }:
                new_payload, n, selected = transform_slot_payload(
                    payload, arm, layers, selected_donor
                )
                record["selected_indices_by_layer"] = selected
            else:
                new_payload, n = transform_payload(
                    payload,
                    arm,
                    gen,
                    selected_donor,
                    generator_for_layer=generator_for_layer,
                )
        else:
            new_payload, n = payload, 0
        stats["layers_transformed"] += n
        record["layers_transformed"] = int(n)
        returned = _payload_summary(new_payload)
        record["returned_present"] = new_payload is not None and int(returned["layers"]) > 0
        record["returned_shapes"] = returned["shapes"]
        record["returned_dtypes"] = returned["dtypes"]
        record["returned_sha256"] = _payload_sha256(new_payload)
        if is_target and subject_contract:
            from .subject_subspace import capture_tensor_sha256

            record["returned_manifest_sha256_by_layer"] = _layer_tensor_hashes(
                new_payload, capture_tensor_sha256
            )
        if not is_target and record["source_sha256"] != record["returned_sha256"]:
            stats["native_read_mismatches"] += 1
        if (
            is_target
            and arm == "correct"
            and record["source_sha256"] != record["returned_sha256"]
        ):
            stats["target_read_mismatches"] += 1
        if record["returned_present"]:
            stats["returned_non_null_reads"] += 1
            if is_target:
                stats["target_returned_non_null_reads"] += 1
        stats["read_records"].append(record)
        return new_payload

    reader_class.get_memory_payload_for_read = patched

    flushed = False
    def flush():
        nonlocal flushed
        if flushed:
            return
        flushed = True
        try:
            if dump_path is not None and dumped:
                torch.save(
                    {
                        "format": "slotmem_donor_payload_v2",
                        "event": dict(event or {}),
                        "payloads": dumped,
                    },
                    dump_path,
                )
                stats["donor_dumped"] = str(dump_path)
                stats["donor_sha256"] = sha256_file(Path(dump_path))
            if arm == "correct":
                effective = stats["target_source_non_null_reads"] > 0
            elif arm == "no_memory":
                effective = (
                    stats["target_source_non_null_reads"] > 0
                    and stats["target_returned_non_null_reads"] == 0
                )
            else:
                effective = stats["layers_transformed"] > 0
            if target_character:
                effective = effective and stats["target_read_hits"] > 0
            stats["intervention_effective"] = bool(effective)
            stats["reads"] = stats["attempted_reads"]
            stats["reads_none"] = stats["attempted_reads"] - stats["source_non_null_reads"]
            _write_report(Path(report_path), stats, exclusive=subject_contract is not None)
            print(f"[audit] {json.dumps(stats)}", flush=True)
        finally:
            if reader_class.get_memory_payload_for_read is patched:
                reader_class.get_memory_payload_for_read = original

    return flush


def self_check():
    gen = torch.Generator().manual_seed(0)
    tok = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    meta = [{"char_id": "A"}] * 6
    layerwise = {"tokens": {LAYERWISE_MARKER: True, LAYERS_KEY: {"0": tok, "7": tok * 2}}, "token_meta": meta}

    out, _ = transform_payload(layerwise, "zero", gen)
    assert torch.count_nonzero(out["tokens"][LAYERS_KEY]["7"]) == 0

    out, n = transform_payload(layerwise, "random", torch.Generator().manual_seed(0))
    assert n == 2
    random_tokens = out["tokens"][LAYERS_KEY]["0"]
    assert torch.allclose(random_tokens.mean(0), tok.mean(0), atol=1e-5)
    assert torch.allclose(random_tokens.std(0, correction=0), tok.std(0, correction=0), atol=1e-5)
    assert out["token_meta"] is meta

    donor = {LAYERWISE_MARKER: True, LAYERS_KEY: {"0": torch.full((6, 4), 9.0), "7": torch.full((6, 4), 5.0)}}
    out, _ = transform_payload(layerwise, "wrong", gen, donor)
    assert out["tokens"][LAYERS_KEY]["0"].shape == (6, 4)
    assert torch.all(out["tokens"][LAYERS_KEY]["0"] == 9.0)
    assert out["tokens"][LAYERS_KEY]["7"].shape == (6, 4)

    out, n = transform_payload({"tokens": tok, "token_meta": meta}, "zero", gen)
    assert n == 1 and torch.count_nonzero(out["tokens"]) == 0, "flat (non-layerwise) payload"

    assert transform_payload(None, "zero", gen) == (None, 0)
    assert transform_payload(layerwise, "no_memory", gen) == (None, 0)

    try:
        transform_tokens(tok, "wrong", gen, torch.zeros(6, 8))
    except ValueError:
        pass
    else:
        raise AssertionError("non-exact donor shape must raise")

    print("[audit] self-check OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, default="correct")
    ap.add_argument("--seed", type=int, default=0, help="random-arm seed; NOT the sampler seed")
    ap.add_argument("--donor", help="payload dump from a --dump-donor run of a different video")
    ap.add_argument("--donor-manifest", help="frozen JSON donor pair for --arm wrong")
    ap.add_argument("--event-json", help="JSON recurrence event used for addressing and donor validation")
    ap.add_argument("--dump-donor", help="write this run's payloads here for a later wrong arm")
    ap.add_argument("--report", default="slotmem_audit.json")
    # infer_slotmem.py sits at the repo root here: this repo IS the SlotMem fork, so the
    # audit imports it directly instead of pointing at a vendored copy.
    ap.add_argument("--slotmem-dir", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--self-check", action="store_true", help="verify arm semantics; no GPU, no weights")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- then infer_slotmem.py's own args")
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return 0
    if not args.event_json:
        ap.error("all arm runs need --event-json for frozen target addressing")
    if args.arm == "wrong" and not (args.donor and args.donor_manifest and args.event_json):
        ap.error("--arm wrong needs --donor, --donor-manifest, and --event-json")

    event = json.loads(Path(args.event_json).read_text(encoding="utf-8")) if args.event_json else {}
    donor_entry = None
    if args.arm == "wrong":
        manifest = json.loads(Path(args.donor_manifest).read_text(encoding="utf-8"))
        validate_donor_bundle(
            event,
            Path(args.donor),
            Path(args.donor_manifest),
            loader=_load_tensor_artifact,
        )
        donor_entry = validate_donor_manifest(
            select_donor_entry(manifest, event), event, Path(args.donor)
        )

    slotmem_dir = Path(args.slotmem_dir)
    if not (slotmem_dir / "infer_slotmem.py").exists():
        ap.error(f"no infer_slotmem.py under {slotmem_dir}; run this from the repo root")
    sys.path.insert(0, str(slotmem_dir))

    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
    actual_runtime = build_runtime_contract(event, rest)
    flush = install(
        args.arm,
        args.seed,
        args.donor,
        args.dump_donor,
        args.report,
        event=event,
        donor_entry=donor_entry,
        runtime_contract=actual_runtime,
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
