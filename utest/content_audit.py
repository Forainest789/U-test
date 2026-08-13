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
import sys
from pathlib import Path

import torch

ARMS = ("no_memory", "zero", "correct", "wrong", "random")

LAYERWISE_MARKER = "__layerwise__"
LAYERS_KEY = "layers"


def _is_layerwise(value) -> bool:
    return (
        isinstance(value, dict)
        and bool(value.get(LAYERWISE_MARKER, False))
        and isinstance(value.get(LAYERS_KEY, None), dict)
    )


def _match_rows(donor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Reshape a donor's slot block to the target's row count. Feature dim must agree."""
    if donor.shape[1:] != target.shape[1:]:
        raise ValueError(
            f"donor feature shape {tuple(donor.shape[1:])} != target {tuple(target.shape[1:])}; "
            "the donor was dumped under a different memory-encoder config"
        )
    n_d, n_t = int(donor.shape[0]), int(target.shape[0])
    if n_d == n_t:
        return donor
    if n_d > n_t:
        return donor[:n_t]
    reps = -(-n_t // n_d)  # ceil
    return donor.repeat(reps, *([1] * (donor.ndim - 1)))[:n_t]


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
    return matched.to(device=tokens.device, dtype=tokens.dtype)


def transform_tokens(
    tokens: torch.Tensor, arm: str, gen: torch.Generator, donor=None
) -> torch.Tensor | None:
    if arm == "no_memory":
        return None
    if arm == "correct":
        return tokens
    if arm == "zero":
        return torch.zeros_like(tokens)
    if arm == "random":
        return _moment_matched_gaussian(tokens, gen)
    if arm == "wrong":
        if donor is None:
            raise ValueError("wrong arm reached a layer with no donor tensor")
        return _match_rows(donor.to(tokens.device, tokens.dtype), tokens).clone()
    raise ValueError(f"unknown arm: {arm}")


def transform_payload(payload, arm: str, gen: torch.Generator, donor_tokens=None):
    """Pure function over a get_memory_payload() return value. Returns (new_payload, n_layers)."""
    if payload is None or arm == "correct":
        return payload, 0
    if arm == "no_memory":
        return None, 0
    tokens = payload.get("tokens")

    if isinstance(tokens, torch.Tensor):
        donor = donor_tokens if isinstance(donor_tokens, torch.Tensor) else None
        return {**payload, "tokens": transform_tokens(tokens, arm, gen, donor)}, 1

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
            out[layer] = transform_tokens(t, arm, gen, layer_donor)
            n += 1
        return {**payload, "tokens": {LAYERWISE_MARKER: True, LAYERS_KEY: out}}, n

    return payload, 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_donor_manifest(entry: dict, event: dict, donor_path: Path) -> dict:
    """Validate one pre-frozen target/donor pair before loading its tensors."""
    required = {
        "target_story_id", "target_entity_uid", "donor_story_id", "donor_entity_uid",
        "payload_path", "payload_sha256", "coarse_class", "colour", "character_count",
        "source_visible", "gap_bucket", "slot_shape", "selection_seed",
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
    if Path(entry["payload_path"]).resolve() != resolved:
        raise ValueError("donor manifest payload_path does not match --donor")
    actual_hash = sha256_file(resolved)
    if str(entry["payload_sha256"]).lower() != actual_hash:
        raise ValueError("donor payload SHA256 does not match manifest")
    return {**entry, "payload_path": str(resolved), "payload_sha256": actual_hash}


def _tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().to(device="cpu").view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _payload_summary(payload) -> dict:
    tokens = payload.get("tokens") if isinstance(payload, dict) else None
    if isinstance(tokens, torch.Tensor):
        return {"layers": 1, "slots": int(tokens.shape[0]), "shapes": {"shared": list(tokens.shape)}}
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
        }
    return {"layers": 0, "slots": 0, "shapes": {}}


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
):
    """Patch SlotMem's single memory read point. Call before infer_slotmem.main()."""
    import infer_slotmem

    gen = torch.Generator().manual_seed(seed)
    loaded_donor = torch.load(donor_path, map_location="cpu", weights_only=False) if donor_path else {}
    donor = (
        loaded_donor.get("payloads", {})
        if isinstance(loaded_donor, dict) and loaded_donor.get("format") == "slotmem_donor_payload_v2"
        else loaded_donor
    )
    donor_key = str(donor_entry.get("payload_key")) if donor_entry and donor_entry.get("payload_key") else None
    if arm == "wrong":
        if donor_key is None and isinstance(donor, dict) and len(donor) == 1:
            donor_key = str(next(iter(donor)))
        if donor_key is None or not isinstance(donor, dict) or donor_key not in donor:
            raise ValueError("wrong donor manifest must select one existing payload_key")
        selected_donor = donor[donor_key]
    else:
        selected_donor = None
    dumped: dict = {}
    target_character = str((event or {}).get("character_name", ""))
    stats = {
        "arm": arm,
        "seed": seed,
        "attempted_reads": 0,
        "source_non_null_reads": 0,
        "returned_non_null_reads": 0,
        "payload_layers_seen": 0,
        "layers_transformed": 0,
        "target_character": target_character or None,
        "target_read_hits": 0,
        "read_records": [],
    }

    original = infer_slotmem.RoleWiseSlotMemoryBank.get_memory_payload_for_read

    def patched(self, char_id, bank_idx=0):
        payload = original(self, char_id, bank_idx)
        stats["attempted_reads"] += 1
        summary = _payload_summary(payload)
        record = {
            "character": str(char_id),
            "bank": int(bank_idx),
            "source_present": payload is not None and int(summary["layers"]) > 0,
            **summary,
        }
        if payload is None:
            record["returned_present"] = False
            stats["read_records"].append(record)
            return None
        stats["source_non_null_reads"] += 1
        stats["payload_layers_seen"] += int(summary["layers"])
        if target_character and str(char_id) == target_character:
            stats["target_read_hits"] += 1
        key = f"{char_id}|{bank_idx}"
        if dump_path is not None:
            dumped.setdefault(key, _cpu_clone(payload.get("tokens")))
        is_target = not target_character or str(char_id) == target_character
        if arm == "no_memory" or is_target:
            new_payload, n = transform_payload(payload, arm, gen, selected_donor)
        else:
            new_payload, n = payload, 0
        stats["layers_transformed"] += n
        returned = _payload_summary(new_payload)
        record["returned_present"] = new_payload is not None and int(returned["layers"]) > 0
        record["returned_shapes"] = returned["shapes"]
        if record["returned_present"]:
            stats["returned_non_null_reads"] += 1
        stats["read_records"].append(record)
        return new_payload

    infer_slotmem.RoleWiseSlotMemoryBank.get_memory_payload_for_read = patched

    def flush():
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
            effective = stats["source_non_null_reads"] > 0
        elif arm == "no_memory":
            effective = stats["source_non_null_reads"] > 0 and stats["returned_non_null_reads"] == 0
        else:
            effective = stats["layers_transformed"] > 0
        if target_character:
            effective = effective and stats["target_read_hits"] > 0
        stats["intervention_effective"] = bool(effective)
        stats["reads"] = stats["attempted_reads"]
        stats["reads_none"] = stats["attempted_reads"] - stats["source_non_null_reads"]
        Path(report_path).write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"[audit] {json.dumps(stats)}", flush=True)

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

    donor = {LAYERWISE_MARKER: True, LAYERS_KEY: {"0": torch.full((3, 4), 9.0), "7": torch.full((9, 4), 5.0)}}
    out, _ = transform_payload(layerwise, "wrong", gen, donor)
    assert out["tokens"][LAYERS_KEY]["0"].shape == (6, 4), "short donor must tile up"
    assert torch.all(out["tokens"][LAYERS_KEY]["0"] == 9.0)
    assert out["tokens"][LAYERS_KEY]["7"].shape == (6, 4), "long donor must truncate"

    out, n = transform_payload({"tokens": tok, "token_meta": meta}, "zero", gen)
    assert n == 1 and torch.count_nonzero(out["tokens"]) == 0, "flat (non-layerwise) payload"

    assert transform_payload(None, "zero", gen) == (None, 0)
    assert transform_payload(layerwise, "no_memory", gen) == (None, 0)

    try:
        transform_tokens(tok, "wrong", gen, torch.zeros(6, 8))
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched donor feature dim must raise, not silently pad")

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
    if args.arm == "wrong" and not (args.donor and args.donor_manifest and args.event_json):
        ap.error("--arm wrong needs --donor, --donor-manifest, and --event-json")

    event = json.loads(Path(args.event_json).read_text(encoding="utf-8")) if args.event_json else {}
    donor_entry = None
    if args.arm == "wrong":
        manifest = json.loads(Path(args.donor_manifest).read_text(encoding="utf-8"))
        entries = manifest.get("pairs", manifest) if isinstance(manifest, dict) else manifest
        if isinstance(entries, dict):
            entries = [entries]
        matches = [
            entry for entry in entries
            if str(entry.get("target_story_id")) == str(event.get("story_id"))
            and str(entry.get("target_entity_uid")) == str(event.get("entity_uid"))
        ]
        if len(matches) != 1:
            ap.error(f"donor manifest must contain exactly one pair for this event, found {len(matches)}")
        donor_entry = validate_donor_manifest(matches[0], event, Path(args.donor))

    slotmem_dir = Path(args.slotmem_dir)
    if not (slotmem_dir / "infer_slotmem.py").exists():
        ap.error(f"no infer_slotmem.py under {slotmem_dir}; run this from the repo root")
    sys.path.insert(0, str(slotmem_dir))

    flush = install(
        args.arm,
        args.seed,
        args.donor,
        args.dump_donor,
        args.report,
        event=event,
        donor_entry=donor_entry,
    )

    import infer_slotmem

    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
    sys.argv = ["infer_slotmem.py", *rest]
    try:
        infer_slotmem.main()
    finally:
        flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
