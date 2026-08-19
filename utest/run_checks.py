"""Read-only sanity checks for a completed SlotMem run directory.

These checks are GPU-free and depend only on :mod:`json`. They catch the two silent
failures a ``status: completed`` prefix otherwise hides:

* the event's target character was never addressed by the reader -- evicted by
  ``max_memory_characters``, absent from the prompt, or a case mismatch -- so the
  experiment would measure a character that never entered memory; and
* the recurrent writer reported a non-zero gate but a zero slot residual (the
  ``MemoryWriter.delta_mlp`` output branch is effectively zero), which is a static-bank
  read rather than a dynamic-memory write.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _casefold(value: object) -> str:
    return str(value or "").strip().casefold()


def validate_target_read(efficiency: Mapping, event: Mapping) -> list[str]:
    """Return error strings if the target character was not addressed at its memory chunk."""
    target_raw = str(event.get("character_name") or "").strip()
    target = _casefold(target_raw)
    if not target:
        return ["event_missing_character_name"]
    memory_idx = event.get("memory_chunk_idx")
    chunks = [row for row in list(efficiency.get("chunks") or []) if isinstance(row, dict)]
    if memory_idx is None:
        rows = chunks
    else:
        rows = [row for row in chunks if row.get("chunk_idx") == memory_idx]
        if not rows:
            return ["memory_chunk_not_in_prefix"]
    addressed = [
        str(role)
        for row in rows
        for role in list((row.get("memory_read") or {}).get("attempted_roles") or [])
    ]
    if not any(_casefold(role) == target for role in addressed):
        return ["target_character_not_addressed"]
    # ponytail: no exact-case check. eligibility.normalize() lowercases every event name
    # while the runtime bank keeps source casing, so exact match would fail on every event.
    return []


def writer_delta_status(efficiency: Mapping) -> dict:
    """Classify recurrent writer updates: did the slot actually move, or only the gate?"""
    updates = list(
        (efficiency.get("runtime_evidence") or {}).get("writer_updates") or []
    )
    recurrent = [
        update for update in updates
        if isinstance(update, dict)
        and (update.get("stats") or {}).get("mode") == "writer_update"
    ]
    positive = 0
    zero_with_gate = 0
    for update in recurrent:
        stats = update.get("stats") or {}
        residual = float(stats.get("residual_norm") or 0.0)
        gate = float(stats.get("mean_gate") or 0.0)
        if residual > 0.0:
            positive += 1
        elif gate > 0.0:
            zero_with_gate += 1
    return {
        "recurrent_update_count": len(recurrent),
        "positive_residual_count": positive,
        "zero_residual_with_positive_gate_count": zero_with_gate,
        "writer_delta_branch_zero": bool(recurrent and not positive and zero_with_gate),
    }


def analyze_run(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    event = _read_json(run_dir / "event.json")
    efficiency_path = run_dir / "prefix_generation" / "efficiency.json"
    if not efficiency_path.is_file():
        efficiency_path = run_dir / "efficiency.json"
    efficiency = _read_json(efficiency_path)
    times = [
        float(row.get("time_s") or 0.0)
        for row in list(efficiency.get("chunks") or [])
        if isinstance(row, dict)
    ]
    positive = [value for value in times if value > 0.0]
    return {
        "run_dir": str(run_dir),
        "status": efficiency.get("status"),
        "target_read_errors": validate_target_read(efficiency, event),
        "writer": writer_delta_status(efficiency),
        "chunk_time_s": times,
        "slowdown_last_over_first": (positive[-1] / positive[0]) if len(positive) > 1 else None,
    }


def _self_check() -> None:
    efficiency = {
        "chunks": [
            {"chunk_idx": 0, "memory_read": {"attempted_roles": ["Evan", "Luca"]}},
            {"chunk_idx": 1, "memory_read": {"attempted_roles": ["Milan", "Evan"]}},
        ],
        "runtime_evidence": {
            "writer_updates": [
                {"character": "Evan", "stats": {"mode": "writer_update", "residual_norm": 0.0, "mean_gate": 0.47}},
            ]
        },
    }
    assert validate_target_read(efficiency, {"character_name": "luca", "memory_chunk_idx": 1}) == [
        "target_character_not_addressed"
    ]
    assert validate_target_read(efficiency, {"character_name": "Luca", "memory_chunk_idx": 1}) == [
        "target_character_not_addressed"
    ]
    assert validate_target_read(efficiency, {"character_name": "Luca", "memory_chunk_idx": 0}) == []
    assert validate_target_read(efficiency, {"character_name": "luca", "memory_chunk_idx": 0}) == []
    assert validate_target_read(efficiency, {"character_name": "luca", "memory_chunk_idx": 2}) == [
        "memory_chunk_not_in_prefix"
    ]
    status = writer_delta_status(efficiency)
    assert status["writer_delta_branch_zero"] is True
    assert status["positive_residual_count"] == 0
    print("self-check ok")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return 0
    if args.run_dir is None:
        parser.error("one of --run-dir or --self-check is required")
    print(json.dumps(analyze_run(args.run_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
