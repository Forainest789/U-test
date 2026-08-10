#!/usr/bin/env python3
"""Content-causality audit for SlotMem: is the slot memory a content channel or a generic prior?

SlotMem reports consistency scores only, which cannot separate "the memory of THIS
character is doing the work" from "any memory-shaped side signal smooths the video".
This is the FU-MD correct/wrong/scramble apparatus pointed at their checkpoint.

It does not fork their code. All memory reads funnel through one method --
``RoleWiseSlotMemoryBank.get_memory_payload`` -- so we patch that and call their
``main()`` with their own argv.

Arms (identical seed_base => common random numbers across arms):
  correct   passthrough. With --dump-donor, saves the payloads for a later wrong arm.
  scramble  permute slots within each layer; token_meta untouched, so positions stay
            put and only content moves. Delta vs correct == content ordering matters.
  zero      tokens zeroed, shapes kept. Isolates "memory present" from "memory says X".
  wrong     tokens replaced from --donor (another video's dump), shape-matched.
            correct > wrong is the claim SlotMem never tests.

Two more arms need no patch at all -- pass these to their launcher directly:
  memory off   --no-enable_sparse_role_memory_attn
  base Wan2.2  --native_wan_inference

Usage (on the server, after scripts/fetch_weights.sh):
  python -m utest.content_audit --arm correct --dump-donor donor_3_271.pt \
      -- --ckpt_dir $CKPT_DIR --json_path ... (everything test_slotmem_stage1.sh passes)
  python -m utest.content_audit --arm wrong --donor donor_3_271.pt -- ...

  python -m utest.content_audit --self-check   # no GPU, no weights, no clone
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ARMS = ("correct", "scramble", "zero", "wrong")

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


def transform_tokens(tokens: torch.Tensor, arm: str, gen: torch.Generator, donor=None) -> torch.Tensor:
    if arm == "correct":
        return tokens
    if arm == "zero":
        return torch.zeros_like(tokens)
    if arm == "scramble":
        perm = torch.randperm(int(tokens.shape[0]), generator=gen)
        return tokens[perm].clone()
    if arm == "wrong":
        if donor is None:
            raise ValueError("wrong arm reached a layer with no donor tensor")
        return _match_rows(donor.to(tokens.device, tokens.dtype), tokens).clone()
    raise ValueError(f"unknown arm: {arm}")


def transform_payload(payload, arm: str, gen: torch.Generator, donor_tokens=None):
    """Pure function over a get_memory_payload() return value. Returns (new_payload, n_layers)."""
    if payload is None or arm == "correct":
        return payload, 0
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
            out[layer] = transform_tokens(t, arm, gen, donor_layers.get(layer))
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


def install(arm: str, seed: int, donor_path: str | None, dump_path: str | None, report_path: str):
    """Patch SlotMem's single memory read point. Call before infer_slotmem.main()."""
    import infer_slotmem

    gen = torch.Generator().manual_seed(seed)
    donor = torch.load(donor_path, map_location="cpu", weights_only=False) if donor_path else {}
    dumped: dict = {}
    stats = {"arm": arm, "seed": seed, "reads": 0, "reads_none": 0, "layers_transformed": 0}

    original = infer_slotmem.RoleWiseSlotMemoryBank.get_memory_payload

    def patched(self, char_id, bank_idx=0):
        payload = original(self, char_id, bank_idx)
        stats["reads"] += 1
        if payload is None:
            stats["reads_none"] += 1
            return None
        key = f"{char_id}|{bank_idx}"
        if dump_path is not None:
            dumped.setdefault(key, _cpu_clone(payload.get("tokens")))
        # Donor lookup falls back to any dumped character: a wrong arm must never
        # silently degrade into a correct one just because the char_id differs.
        donor_tokens = donor.get(key) if isinstance(donor, dict) else None
        if arm == "wrong" and donor_tokens is None and isinstance(donor, dict) and donor:
            donor_tokens = next(iter(donor.values()))
        new_payload, n = transform_payload(payload, arm, gen, donor_tokens)
        stats["layers_transformed"] += n
        return new_payload

    infer_slotmem.RoleWiseSlotMemoryBank.get_memory_payload = patched

    def flush():
        if dump_path is not None and dumped:
            torch.save(dumped, dump_path)
            stats["donor_dumped"] = str(dump_path)
        # A no-op intervention and a null result look the same in the output video.
        # This file is how you tell them apart before reading any metric.
        stats["intervention_effective"] = arm == "correct" or stats["layers_transformed"] > 0
        Path(report_path).write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"[audit] {json.dumps(stats)}", flush=True)

    return flush


def self_check():
    gen = torch.Generator().manual_seed(0)
    tok = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    meta = [{"char_id": "A"}] * 6
    layerwise = {"tokens": {LAYERWISE_MARKER: True, LAYERS_KEY: {"0": tok, "7": tok * 2}}, "token_meta": meta}

    out, n = transform_payload(layerwise, "scramble", gen)
    assert n == 2, n
    s0 = out["tokens"][LAYERS_KEY]["0"]
    assert s0.shape == tok.shape
    assert not torch.equal(s0, tok), "scramble did not move anything"
    assert torch.equal(s0.sum(0), tok.sum(0)), "scramble must permute rows, not alter them"
    assert out["token_meta"] is meta, "scramble must leave token_meta (positions) alone"

    out, _ = transform_payload(layerwise, "zero", gen)
    assert torch.count_nonzero(out["tokens"][LAYERS_KEY]["7"]) == 0

    donor = {LAYERWISE_MARKER: True, LAYERS_KEY: {"0": torch.full((3, 4), 9.0), "7": torch.full((9, 4), 5.0)}}
    out, _ = transform_payload(layerwise, "wrong", gen, donor)
    assert out["tokens"][LAYERS_KEY]["0"].shape == (6, 4), "short donor must tile up"
    assert torch.all(out["tokens"][LAYERS_KEY]["0"] == 9.0)
    assert out["tokens"][LAYERS_KEY]["7"].shape == (6, 4), "long donor must truncate"

    out, n = transform_payload({"tokens": tok, "token_meta": meta}, "zero", gen)
    assert n == 1 and torch.count_nonzero(out["tokens"]) == 0, "flat (non-layerwise) payload"

    assert transform_payload(None, "zero", gen) == (None, 0)

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
    ap.add_argument("--seed", type=int, default=0, help="scramble permutation seed; NOT the sampler seed")
    ap.add_argument("--donor", help="payload dump from a --dump-donor run of a different video")
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
    if args.arm == "wrong" and not args.donor:
        ap.error("--arm wrong needs --donor from a prior --dump-donor run of a DIFFERENT video")

    slotmem_dir = Path(args.slotmem_dir)
    if not (slotmem_dir / "infer_slotmem.py").exists():
        ap.error(f"no infer_slotmem.py under {slotmem_dir}; run this from the repo root")
    sys.path.insert(0, str(slotmem_dir))

    flush = install(args.arm, args.seed, args.donor, args.dump_donor, args.report)

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
