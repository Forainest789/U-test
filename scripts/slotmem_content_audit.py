#!/usr/bin/env python3
"""Content-causality audit for SlotMem: is the slot memory a content channel, or a
generic prior that any tensor of the right shape would have supplied?

SlotMem publishes consistency scores only, which cannot separate "the memory of THIS
character is doing the work" from "any memory-shaped side signal smooths the video".
This is the FU-MD correct/wrong/zero apparatus pointed at their published checkpoint.

It changes no line of the generation path. SlotMem keeps its two bank reads apart
already: generation calls ``RoleWiseSlotMemoryBank.get_memory_payload_for_read`` and the
writer's read-modify-write calls ``get_memory_payload``. We patch the two separately --
the reader's return value is the intervention, the writer's is only counted -- and then
call their own ``main()`` with their own argv.

Arms (identical seed_base => common random numbers, so arms are paired):
  correct    passthrough. With --dump-donor, saves the TARGET's payload for a wrong arm.
  no_memory  the target's read returns None. This character has no memory; others keep theirs.
  zero       the target's tokens zeroed, shape kept. "Present, but says nothing."
  random     the target's tokens replaced by Gaussian noise matched to their per-dim
             mean/std. "A tensor of the right shape and scale, carrying no content."
  scramble   the target's slots permuted. Content kept, ordering destroyed.
  wrong      the target's tokens replaced by ANOTHER STORY's entity (--donor).
             correct > wrong is the claim SlotMem never tests.

One more arm needs no patch at all -- pass it to their launcher directly:
  base Wan2.2   --native_wan_inference   (same prefix frames, no LoRA, no memory path)

Three failure modes this file exists to prevent. Each has already cost a run:

1. ADDRESS MISS. SlotMem reads ``character_list[:max_memory_characters]``
   (infer_slotmem.py ~3639) -- plain prefix truncation, no ranking. A target that
   sits below the cap is NEVER READ, the intervention lands on a payload nobody opens,
   and the arm silently degrades into baseline. The 2026-08-19 sample_5 run lost two
   arms this way: target ``evan`` was 3rd of 3 in chunk 3 under a cap of 2.
   ``--preflight`` is CPU-only and picks an event whose target is inside the cap.

2. WRITER-PATH CONTAMINATION. SlotMem's writer reads the bank through the same method,
   from ``_stage2_prepare_payload_for_bank`` (infer_slotmem.py ~3482), and branches on
   whether the prior is non-empty. Transforming that call rewrites the bank instead of
   the frame -- a different experiment -- and flips writer_update into
   initial_slot_extract. Those reads are counted and passed through untouched. The writer
   runs after generation (~3905), so an intervention lands on the frame, never the bank.

3. VACUOUS GATE. ``intervention_effective`` used to be true-by-definition on the correct
   arm, so a correct arm that never read its target still looked healthy (sample_77,
   2026-08-19: intervention_effective=true with layers_transformed=0). Now EVERY arm,
   correct included, must show ``target_reads > 0``, and every non-correct arm must also
   show ``target_transforms > 0``. Otherwise this script exits non-zero, so the queue
   stops instead of burning the next twelve GPU-hours.

Usage:
  python scripts/slotmem_content_audit.py --self-check      # no GPU, no weights, no clone

  python scripts/slotmem_content_audit.py --preflight \
      --json-path .../narrastream_slotmem/sample_5/rewrite_caption.json \
      --max-memory-characters 2 --out runs/utest/sample_5/event.json

  python scripts/slotmem_content_audit.py --arm zero --event runs/utest/sample_5/event.json \
      --report runs/utest/sample_5/arms/zero/audit.json -- <infer_slotmem's own args>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # --preflight is pure JSON on purpose: it must be runnable
    torch = None             # on a laptop, before anyone books a GPU for twelve hours.

ARMS = ("correct", "no_memory", "zero", "random", "scramble", "wrong")
NEEDS_TARGET = tuple(a for a in ARMS if a != "correct")

LAYERWISE_MARKER = "__layerwise__"
LAYERS_KEY = "layers"
DONOR_FORMAT = "slotmem_donor_v3"

# SlotMem already separates the two reads: generation calls get_memory_payload_for_read,
# the writer's read-modify-write calls get_memory_payload directly. Patching them
# separately is what keeps an intervention out of the bank -- no call-stack inspection,
# no depth limit that silently stops matching when the stack changes shape.
READER_SEAM = "get_memory_payload_for_read"
WRITER_FRAME = "_stage2_prepare_payload_for_bank"
# main() stamps the chunk index onto the manager before any read of that chunk, so the
# reads carry their own address instead of having it guessed off the frames above them.
CHUNK_ATTR = "current_chunk_idx"
# Everything the patch and the readability model assume about their source. Checked on the
# CPU in --preflight and again before any arm: a rename fails in seconds, not 70 minutes in.
SOURCE_ANCHORS = (
    f"def {READER_SEAM}",
    f"def {WRITER_FRAME}",
    f"mem_manager.{CHUNK_ATTR} = int(chunk_idx)",
    "chars[: int(args.max_memory_characters)]",
)


# ---------------------------------------------------------------- payload plumbing


def _is_layerwise(value) -> bool:
    return (
        isinstance(value, dict)
        and bool(value.get(LAYERWISE_MARKER, False))
        and isinstance(value.get(LAYERS_KEY, None), dict)
    )


def _iter_tensors(tokens):
    """Yield (label, tensor) for either carrier shape SlotMem hands back.

    SlotMem returns a layerwise container only when the bank holds per-layer entries;
    otherwise ``get_memory_payload`` falls through to the flat base payload, and the
    tokens are one plain Tensor. Both shapes have been observed on the frozen platform
    -- every 2026-08-19 read was flat -- so neither may be the only one handled.
    """
    if isinstance(tokens, torch.Tensor):
        yield "shared", tokens
    elif _is_layerwise(tokens):
        for layer, t in tokens.get(LAYERS_KEY, {}).items():
            if isinstance(t, torch.Tensor):
                yield str(layer), t


def _sha(tokens) -> str | None:
    """Content hash over whatever carrier this is. float32 projection: deterministic,
    and injective from fp16/bf16, which is all a change-detector needs."""
    items = list(_iter_tensors(tokens))
    if not items:
        return None
    h = hashlib.sha256()
    for label, t in sorted(items, key=lambda kv: kv[0]):  # never compare tensors
        h.update(label.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def _summarize(tokens) -> dict:
    items = list(_iter_tensors(tokens))
    return {
        "carrier": "layerwise" if _is_layerwise(tokens) else ("flat" if items else "none"),
        "entries": len(items),
        "shapes": {k: list(t.shape) for k, t in items},
        "dtypes": {k: str(t.dtype) for k, t in items},
        "slots": sum(int(t.shape[0]) for _, t in items),
        "sha256": _sha(tokens),
    }


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
    reps = -(-n_t // n_d)
    return donor.repeat(reps, *([1] * (donor.ndim - 1)))[:n_t]


def transform_tokens(tokens: torch.Tensor, arm: str, gen: torch.Generator, donor=None) -> torch.Tensor:
    if arm == "zero":
        return torch.zeros_like(tokens)
    if arm == "scramble":
        perm = torch.randperm(int(tokens.shape[0]), generator=gen)
        return tokens[perm.to(tokens.device)].clone()
    if arm == "random":
        # Match the first two moments per feature dim: same shape, same scale, no content.
        # An unmatched N(0,1) would also change the injection's magnitude, which confounds
        # "content matters" with "amplitude matters".
        flat = tokens.reshape(int(tokens.shape[0]), -1).to(torch.float32)
        mu = flat.mean(dim=0, keepdim=True)
        sd = flat.std(dim=0, keepdim=True).clamp_min(1e-6)
        noise = torch.randn(flat.shape, generator=gen).to(flat.device)
        return (mu + sd * noise).reshape(tokens.shape).to(tokens.dtype)
    if arm == "wrong":
        if donor is None:
            raise ValueError("wrong arm reached a tensor with no donor counterpart")
        return _match_rows(donor.to(tokens.device, tokens.dtype), tokens).clone()
    raise ValueError(f"unknown token-level arm: {arm}")


def transform_payload(payload, arm: str, gen: torch.Generator, donor_tokens=None):
    """Pure function over a get_memory_payload() return value -> (new_payload, n_changed)."""
    if payload is None or arm == "correct":
        return payload, 0
    if arm == "no_memory":
        # The reader treats None as "this character has no memory yet" and routes it to
        # first_roles. That is exactly the counterfactual, and it is one changed read.
        return None, 1
    tokens = payload.get("tokens")

    if isinstance(tokens, torch.Tensor):
        donor = None
        if isinstance(donor_tokens, torch.Tensor):
            donor = donor_tokens
        elif _is_layerwise(donor_tokens):
            layers = donor_tokens.get(LAYERS_KEY, {})
            donor = next(iter(layers.values())) if layers else None
        return {**payload, "tokens": transform_tokens(tokens, arm, gen, donor)}, 1

    if _is_layerwise(tokens):
        donor_layers = donor_tokens.get(LAYERS_KEY, {}) if _is_layerwise(donor_tokens) else {}
        donor_flat = donor_tokens if isinstance(donor_tokens, torch.Tensor) else None
        out, n = {}, 0
        for layer, t in tokens.get(LAYERS_KEY, {}).items():
            if not isinstance(t, torch.Tensor):
                out[layer] = t
                continue
            out[layer] = transform_tokens(t, arm, gen, donor_layers.get(layer, donor_flat))
            n += 1
        return {**payload, "tokens": {LAYERWISE_MARKER: True, LAYERS_KEY: out}}, n

    return payload, 0


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


# ---------------------------------------------------------------- donor


def load_donor(path: str, target_story_id: str | None) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or blob.get("format") != DONOR_FORMAT:
        raise SystemExit(
            f"[audit] {path} is not a {DONOR_FORMAT} dump. Re-dump it with "
            "--arm correct --dump-donor on a DIFFERENT story."
        )
    # Never key a donor by character name. NarraStream reuses one roster across stories
    # (sample_5 and sample_77 are both [Dario, Evan, Luca, Milan]), so name-keyed lookup
    # exact-matches the target story's own same-named character: that is a same-name
    # different-person swap, not a wrong-identity swap.
    if target_story_id and str(blob.get("story_id")) == str(target_story_id):
        raise SystemExit(
            f"[audit] donor story_id == target story_id ({target_story_id}); a wrong arm "
            "needs another story's entity"
        )
    if blob.get("tokens") is None:
        raise SystemExit(f"[audit] donor {path} carries no tokens")
    return blob


def dump_donor(path: str, tokens, event: dict) -> dict:
    blob = {
        "format": DONOR_FORMAT,
        "story_id": event.get("story_id"),
        "entity_uid": event.get("entity_uid"),
        "character": event.get("target_character"),
        "chunk_idx": event.get("target_chunk_idx"),
        "tokens": tokens,
    }
    torch.save(blob, path)
    return {
        "donor_dumped": str(path),
        "donor_sha256": _sha(tokens),
        "donor_entity_uid": blob["entity_uid"],
    }


# ---------------------------------------------------------------- the patch


def install(
    arm: str,
    seed: int,
    event: dict,
    donor_path: str | None,
    dump_path: str | None,
    report_path: str,
):
    """Patch SlotMem's single memory read point. Call before infer_slotmem.main()."""
    import infer_slotmem

    gen = torch.Generator().manual_seed(seed)
    target_char = str(event["target_character"]).strip().lower()
    target_chunk = int(event["target_chunk_idx"])
    story_id = event.get("story_id")
    donor = load_donor(donor_path, story_id) if donor_path else None
    if arm == "wrong" and donor is None:
        raise SystemExit("[audit] --arm wrong needs --donor")

    dumped = {"tokens": None}
    stats = {
        "arm": arm,
        "seed": seed,
        "story_id": story_id,
        "target_character": event["target_character"],
        "target_chunk_idx": target_chunk,
        "reader_reads": 0,
        "writer_path_reads": 0,
        "reads_none": 0,
        "target_reads": 0,
        "target_transforms": 0,
        "off_target_chunk_skips": 0,
        "read_records": [],
    }

    bank = infer_slotmem.RoleWiseSlotMemoryBank
    original_get = bank.get_memory_payload
    original_read = bank.get_memory_payload_for_read
    # ponytail: a plain depth counter, not a threading.local. Chunk reads happen inline in
    # main()'s loop on one thread; make the reads concurrent and this needs to become one.
    in_reader = {"depth": 0}

    def patched_writer_read(self, char_id, bank_idx=0):
        """get_memory_payload reached from anywhere but the reader: the writer's
        read-modify-write. Recording it is the point; touching it is a different
        experiment (it rewrites the bank, and flips writer_update into
        initial_slot_extract when the prior comes back empty)."""
        if in_reader["depth"] == 0:
            stats["writer_path_reads"] += 1
        return original_get(self, char_id, bank_idx)

    def patched(self, char_id, bank_idx=0):
        # Run SlotMem's own reader body, with the writer counter muted so the reader's
        # inner call to get_memory_payload is not counted as a write.
        in_reader["depth"] += 1
        try:
            payload = original_read(self, char_id, bank_idx)
        finally:
            in_reader["depth"] -= 1
        chunk_idx = getattr(self, CHUNK_ATTR, None)
        chunk_idx = int(chunk_idx) if isinstance(chunk_idx, int) else None

        stats["reader_reads"] += 1
        if payload is None:
            stats["reads_none"] += 1
            return None

        record = {
            "chunk_idx": chunk_idx,
            "character": str(char_id),
            "bank": int(bank_idx),
            "source": _summarize(payload.get("tokens")),
        }
        on_target = str(char_id).strip().lower() == target_char
        # When the chunk resolves and disagrees, refuse rather than intervene on the wrong
        # chunk. The runner also pins the run to the target chunk with --max_chunks, so a
        # miss here fails the gate instead of faking a result.
        if on_target and chunk_idx is not None and chunk_idx != target_chunk:
            stats["off_target_chunk_skips"] += 1
            on_target = False

        if not on_target:
            record["on_target"] = False
            stats["read_records"].append(record)
            return payload

        stats["target_reads"] += 1
        if dump_path is not None and dumped["tokens"] is None:
            dumped["tokens"] = _cpu_clone(payload.get("tokens"))

        donor_tokens = donor.get("tokens") if donor else None
        new_payload, n = transform_payload(payload, arm, gen, donor_tokens)
        stats["target_transforms"] += n
        record.update(
            on_target=True,
            transformed=n,
            returned=_summarize(new_payload.get("tokens") if new_payload else None),
        )
        stats["read_records"].append(record)
        return new_payload

    bank.get_memory_payload = patched_writer_read
    bank.get_memory_payload_for_read = patched

    def flush() -> int:
        if dump_path is not None:
            if dumped["tokens"] is None:
                stats["donor_dump_failed"] = "the target was never read; nothing to dump"
            else:
                stats.update(dump_donor(dump_path, dumped["tokens"], event))

        # A no-op intervention and a null result look identical in the output video. This
        # is how you tell them apart BEFORE reading any metric -- and on the correct arm
        # too, which used to be exempt and therefore unfalsifiable.
        reached = stats["target_reads"] > 0
        changed = stats["target_transforms"] > 0
        stats["intervention_effective"] = reached and (arm == "correct" or changed)
        if not reached:
            stats["gate_failure"] = (
                f"target '{event['target_character']}' was never read at chunk {target_chunk}. "
                f"SlotMem reads character_list[:max_memory_characters] "
                f"(infer_slotmem.py ~3639); re-run --preflight and pick a readable event."
            )
        elif not stats["intervention_effective"]:
            stats["gate_failure"] = (
                f"arm '{arm}' read its target {stats['target_reads']}x but changed nothing. "
                "Check that transform_payload handles this carrier: "
                f"{stats['read_records'][-1]['source']['carrier'] if stats['read_records'] else 'unknown'}"
            )

        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(stats, indent=2), encoding="utf-8")
        printable = {k: v for k, v in stats.items() if k != "read_records"}
        print(f"[audit] {json.dumps(printable)}", flush=True)
        if not stats["intervention_effective"]:
            print(f"[audit] GATE FAILED: {stats['gate_failure']}", file=sys.stderr, flush=True)
            return 1
        return 0

    return flush


# ---------------------------------------------------------------- preflight


def _load_chunks(json_path: str) -> list[dict]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    chunks = data.get("chunks", data) if isinstance(data, dict) else data
    if not isinstance(chunks, list) or not chunks:
        raise SystemExit(f"[audit] no chunks in {json_path}")
    return chunks


def _sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _assert_slotmem_contract(slotmem_dir: Path) -> dict:
    """The preflight's model of readability is only as good as SlotMem's source. Check
    both anchors here, on the CPU, instead of discovering a rename 70 minutes in."""
    src_path = slotmem_dir / "infer_slotmem.py"
    if not src_path.exists():
        raise SystemExit(f"[audit] no infer_slotmem.py under {slotmem_dir}; run scripts/setup_slotmem.sh")
    src = src_path.read_text(encoding="utf-8", errors="replace")
    missing = [a for a in SOURCE_ANCHORS if a not in src]
    if missing:
        raise SystemExit(
            f"[audit] SlotMem source no longer contains {missing}. The reader/writer seam, the "
            "chunk address, and/or the readability model are stale; re-read infer_slotmem.py "
            "before running arms."
        )
    return {"infer_slotmem_sha256": _sha_file(str(src_path))}


def preflight(args) -> int:
    chunks = _load_chunks(args.json_path)
    cap = int(args.max_memory_characters)
    contract = _assert_slotmem_contract(Path(args.slotmem_dir))

    first_seen: dict[str, int] = {}
    for i, chunk in enumerate(chunks):
        for name in chunk.get("character_list", []):
            first_seen.setdefault(str(name), i)

    readable, table = {}, []
    for i, chunk in enumerate(chunks):
        names = [str(n) for n in chunk.get("character_list", [])]
        keep = names[:cap] if cap > 0 else names
        readable[i] = keep
        table.append({"chunk_idx": i, "character_list": names, "readable": keep,
                      "dropped_by_cap": names[len(keep):]})

    candidates = []
    for i, keep in readable.items():
        if i < int(args.min_target_chunk):
            continue
        for rank, name in enumerate(keep):
            gap = i - first_seen[name]
            if gap < int(args.min_gap):
                continue
            candidates.append({
                "target_character": name,
                "target_chunk_idx": i,
                "first_appearance_chunk": first_seen[name],
                "gap_chunks": gap,
                "rank_in_character_list": rank,
                "co_readable": [n for n in keep if n != name],
            })
    # Longest gap first: the further the last sighting, the more the arm is asking memory
    # rather than the overlap frames to carry the identity. Break ties on the earliest
    # chunk, which is the cheapest to reach from the prefix.
    candidates.sort(key=lambda c: (-c["gap_chunks"], c["target_chunk_idx"]))

    if args.target_character:
        wanted = args.target_character.strip().lower()
        candidates = [c for c in candidates if c["target_character"].strip().lower() == wanted]
    if args.target_chunk_idx is not None:
        candidates = [c for c in candidates if c["target_chunk_idx"] == int(args.target_chunk_idx)]

    out = {
        "schema_version": 2,
        "story_id": args.story_id or Path(args.json_path).parent.name,
        "json_path": str(args.json_path),
        "json_sha256": _sha_file(args.json_path),
        "max_memory_characters": cap,
        "chunk_count": len(chunks),
        "readability_table": table,
        "candidates": candidates,
        "slotmem_contract": contract,
    }

    if not candidates:
        out["gate_failure"] = (
            "no event has its target inside character_list[:max_memory_characters] with "
            f"gap >= {args.min_gap}. Raise --max-memory-characters (this changes the frozen "
            "platform for EVERY arm, so record it) or pick another story."
        )
        _write_event(args.out, out)
        print(f"[preflight] GATE FAILED: {out['gate_failure']}", file=sys.stderr)
        return 1

    chosen = candidates[0]
    out.update(chosen)
    out["entity_uid"] = f"{out['story_id']}::{chosen['target_character'].strip().lower()}"
    # Stop the run at the target chunk: SlotMem slices chunks[:max_chunks] BEFORE resuming
    # (infer_slotmem.py ~3548), so this is an absolute bound, and it makes "the target
    # chunk" the only chunk any arm generates. Halves the bill and removes the chunk-
    # scoping question from the intervention entirely.
    out["max_chunks"] = chosen["target_chunk_idx"] + 1
    out["prefix_max_chunks"] = chosen["target_chunk_idx"]
    _write_event(args.out, out)
    print(
        f"[preflight] OK story={out['story_id']} target={chosen['target_character']} "
        f"chunk={chosen['target_chunk_idx']} rank={chosen['rank_in_character_list']}/{cap} "
        f"gap={chosen['gap_chunks']} co_readable={chosen['co_readable']} "
        f"-> --max_chunks {out['max_chunks']}",
        flush=True,
    )
    if len(candidates) > 1:
        print(f"[preflight] {len(candidates) - 1} other readable event(s) in {args.out}", flush=True)
    return 0


def _write_event(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- self-check


def self_check() -> None:
    gen = torch.Generator().manual_seed(0)
    flat = {"tokens": torch.arange(24, dtype=torch.float32).reshape(6, 4), "token_meta": ["m"] * 6}

    # The carrier every 2026-08-19 read actually used: a flat tensor, not a layerwise dict.
    out, n = transform_payload(flat, "zero", gen)
    assert n == 1 and torch.count_nonzero(out["tokens"]) == 0, "flat carrier must be reachable"
    assert out["token_meta"] == flat["token_meta"], "positions must survive a content arm"

    out, n = transform_payload(flat, "scramble", gen)
    assert n == 1 and sorted(out["tokens"].flatten().tolist()) == sorted(
        flat["tokens"].flatten().tolist()
    ), "scramble moves content, never invents it"
    # One seed can legitimately draw the identity permutation (1 in 6! here), so ask
    # across a few instead of pinning the check to whatever seed 0 happens to give.
    moved = [transform_payload(flat, "scramble", torch.Generator().manual_seed(s))[0]["tokens"]
             for s in range(4)]
    assert any(not torch.equal(t, flat["tokens"]) for t in moved), "scramble must permute"

    out, n = transform_payload(flat, "random", gen)
    assert n == 1 and out["tokens"].shape == flat["tokens"].shape
    assert not torch.equal(out["tokens"], flat["tokens"]), "random must change content"

    assert transform_payload(flat, "no_memory", gen) == (None, 1), "no_memory returns a null read"
    assert transform_payload(flat, "correct", gen) == (flat, 0), "correct is passthrough"
    assert transform_payload(None, "zero", gen) == (None, 0), "a null read stays null"

    layerwise = {"tokens": {LAYERWISE_MARKER: True, LAYERS_KEY: {
        "0": torch.ones(6, 4), "7": torch.ones(6, 4) * 2}}}
    out, n = transform_payload(layerwise, "zero", gen)
    assert n == 2 and torch.count_nonzero(out["tokens"][LAYERS_KEY]["7"]) == 0, "layerwise carrier"

    donor = {"tokens": {LAYERWISE_MARKER: True, LAYERS_KEY: {
        "0": torch.full((3, 4), 9.0), "7": torch.full((9, 4), 5.0)}}}
    out, n = transform_payload(layerwise, "wrong", gen, donor["tokens"])
    assert out["tokens"][LAYERS_KEY]["0"].shape == (6, 4), "short donor tiles up"
    assert torch.equal(out["tokens"][LAYERS_KEY]["7"], torch.full((6, 4), 5.0)), "long donor truncates"

    # A flat target with a layerwise donor, and vice versa: the 2026-08-19 platform
    # produced flat reads while the dump format allows either, so neither may raise.
    out, n = transform_payload(flat, "wrong", gen, donor["tokens"])
    assert n == 1 and out["tokens"].shape == (6, 4), "layerwise donor into a flat target"
    out, n = transform_payload(layerwise, "wrong", gen, torch.full((6, 4), 7.0))
    assert n == 2 and torch.equal(out["tokens"][LAYERS_KEY]["0"], torch.full((6, 4), 7.0)), \
        "flat donor into a layerwise target"

    try:
        transform_tokens(torch.zeros(6, 4), "wrong", gen, torch.zeros(6, 8))
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched donor feature dim must raise, not silently pad")

    assert _sha(flat["tokens"]) == _sha(flat["tokens"].clone()), "hash is content-only"
    assert _sha(flat["tokens"]) != _sha(torch.zeros(6, 4)), "hash separates zeroed content"
    assert _sha(None) is None

    _wiring_check()
    print("[audit] self-check OK")


def _wiring_check() -> None:
    """Install the patch on a stub bank and check where it lands.

    transform_payload being correct proves nothing about which reads reach it, and that is
    the half that has broken twice: once on the address, once on the writer path. This runs
    the real install() against a stand-in that has SlotMem's two-method shape.
    """
    import tempfile
    import types

    class StubBank:  # SlotMem's shape: a reader seam delegating to the writer's method
        def __init__(self):
            self.current_chunk_idx = 7

        def get_memory_payload(self, char_id, bank_idx=0):
            return {"tokens": torch.ones(4, 3), "token_meta": ["m"] * 4}

        def get_memory_payload_for_read(self, char_id, bank_idx=0):
            return self.get_memory_payload(char_id, bank_idx)

    fake = types.ModuleType("infer_slotmem")
    fake.RoleWiseSlotMemoryBank = StubBank
    saved = sys.modules.get("infer_slotmem")
    sys.modules["infer_slotmem"] = fake
    try:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "audit.json"
            flush = install(
                "zero", 0,
                {"target_character": "Evan", "target_chunk_idx": 7, "story_id": "stub"},
                None, None, str(report),
            )
            bank = StubBank()
            live = lambda payload: int(torch.count_nonzero(payload["tokens"]))

            assert live(bank.get_memory_payload("Evan", 0)) == 12,                 "the writer's read-modify-write must come back untouched"
            assert live(bank.get_memory_payload_for_read("Evan", 0)) == 0,                 "the target's generation read must be transformed"
            assert live(bank.get_memory_payload_for_read("Luca", 0)) == 12,                 "a bystander's read must come back untouched"
            bank.current_chunk_idx = 8
            assert live(bank.get_memory_payload_for_read("Evan", 0)) == 12,                 "the target read off the target chunk must be refused, not intervened on"
            code = flush()
            stats = json.loads(report.read_text(encoding="utf-8"))
    finally:
        sys.modules.pop("infer_slotmem", None)
        if saved is not None:
            sys.modules["infer_slotmem"] = saved

    assert stats["writer_path_reads"] == 1, f"writer reads miscounted: {stats}"
    assert stats["reader_reads"] == 3, f"reader reads miscounted: {stats}"
    assert stats["target_reads"] == 1 and stats["target_transforms"] == 1, stats
    assert stats["off_target_chunk_skips"] == 1, stats
    assert code == 0 and stats["intervention_effective"], stats


# ---------------------------------------------------------------- cli


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, default="correct")
    ap.add_argument("--seed", type=int, default=0, help="arm-transform seed; NOT the sampler seed")
    ap.add_argument("--event", help="event.json from --preflight: the target address")
    ap.add_argument("--donor", help="a slotmem_donor_v3 dump from a DIFFERENT story")
    ap.add_argument("--dump-donor", help="write this run's target payload here, for a later wrong arm")
    ap.add_argument("--report", default="slotmem_audit.json")
    # This file ships inside the fork it patches, so the repo root is the default platform.
    # Upstream YilaiLiu-HKU/SlotMem is NOT that platform -- it has neither the reader seam nor
    # the chunk stamp this patch needs -- and the anchor check below refuses it by name.
    ap.add_argument("--slotmem-dir",
                    default=os.environ.get("UTEST") or str(Path(__file__).resolve().parents[1]),
                    help="the SlotMem checkout that froze the platform (this fork, not upstream)")
    ap.add_argument("--self-check", action="store_true", help="verify arm semantics; no GPU, no weights")

    pf = ap.add_argument_group("preflight (CPU only): pick an address the reader will actually open")
    pf.add_argument("--preflight", action="store_true")
    pf.add_argument("--json-path", help="the story's rewrite_caption.json")
    pf.add_argument("--max-memory-characters", type=int, default=2, help="must match the run's value")
    pf.add_argument("--min-gap", type=int, default=1, help="chunks between first sighting and target")
    pf.add_argument("--min-target-chunk", type=int, default=1)
    pf.add_argument("--target-character", help="pin the target instead of taking the widest gap")
    pf.add_argument("--target-chunk-idx", type=int)
    pf.add_argument("--story-id", help="defaults to the json's parent directory name")
    pf.add_argument("--out", default="event.json")

    ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- then infer_slotmem.py's own args")
    args = ap.parse_args()

    if args.preflight:
        if not args.json_path:
            ap.error("--preflight needs --json-path")
        return preflight(args)
    if torch is None:
        ap.error("torch is required for everything except --preflight; "
                 "run this inside the slotmem env (see scripts/setup_slotmem.sh)")
    if args.self_check:
        self_check()
        return 0

    if not args.event:
        ap.error("every arm needs --event from a --preflight run: without a verified target "
                 "address an arm cannot be distinguished from baseline")
    event = json.loads(Path(args.event).read_text(encoding="utf-8"))
    if "target_character" not in event:
        ap.error(f"{args.event} has no target_character; the preflight gate failed. "
                 f"{event.get('gate_failure', '')}")
    if args.arm == "wrong" and not args.donor:
        ap.error("--arm wrong needs --donor from a --dump-donor run of a DIFFERENT story")
    if args.arm in NEEDS_TARGET and args.dump_donor:
        ap.error("--dump-donor only makes sense on --arm correct: a donor must be untouched memory")

    slotmem_dir = Path(args.slotmem_dir)
    _assert_slotmem_contract(slotmem_dir)
    sys.path.insert(0, str(slotmem_dir))

    flush = install(args.arm, args.seed, event, args.donor, args.dump_donor, args.report)

    import infer_slotmem

    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
    sys.argv = ["infer_slotmem.py", *rest]
    try:
        infer_slotmem.main()
    finally:
        gate = flush()
    return gate


if __name__ == "__main__":
    raise SystemExit(main())
