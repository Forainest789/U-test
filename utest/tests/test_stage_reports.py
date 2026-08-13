from __future__ import annotations

import json
from pathlib import Path

from utest.stage_reports import evaluate_m0b, validate_m0a


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_m0_tree(root: Path, *, chunks: int, memory_reads: int) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(chunks):
        (root / f"chunk_{index:03d}.mp4").write_bytes(b"video")
        _write_json(root / f"chunk_{index:03d}.metadata.json", {"chunk_idx": index})
        rows.append({"chunk_idx": index, "known_roles": ["ana"] if index and memory_reads else []})
    _write_json(
        root / "slotmem_inference_metadata.json",
        {
            "chunk_count": 7,
            "completed_chunk_count": chunks,
            "chunks": rows,
            "runtime_evidence": {
                "nonempty_memory_reads": memory_reads,
                "loaded_checkpoint_domains": ["high_noise", "low_noise"],
                "writer_bank_hash_changes": 1,
                "writer_positive_residual_count": 1,
            },
        },
    )
    _write_json(root / "inference_args.yaml", {"train_stage": "stage2"})
    efficiency = root / "efficiency.json"
    _write_json(
        efficiency,
        {
            "total_elapsed_s": 100.0,
            "peak_allocated_gb": 31.0,
            "peak_reserved_gb": 35.0,
            "measured_chunks": chunks,
        },
    )
    platform = root / "platform.manifest.json"
    _write_json(platform, {"repo_commit": "abc", "checkpoints": {"stage2/stage2_low.pt": {}}})
    return efficiency, platform


def test_m0a_requires_all_seven_chunks_and_nonempty_read(tmp_path: Path) -> None:
    efficiency, platform = _make_m0_tree(tmp_path, chunks=2, memory_reads=0)
    report = validate_m0a(tmp_path, efficiency_path=efficiency, platform_manifest=platform)
    assert report["status"] == "failed"
    assert "completed_chunks:2/7" in report["reasons"]
    assert "no_nonempty_memory_read" in report["reasons"]


def test_m0a_passes_only_with_complete_runtime_evidence(tmp_path: Path) -> None:
    efficiency, platform = _make_m0_tree(tmp_path, chunks=7, memory_reads=3)
    report = validate_m0a(tmp_path, efficiency_path=efficiency, platform_manifest=platform)
    assert report["status"] == "passed"
    assert report["evidence"]["peak_reserved_gb"] == 35.0


def test_m0b_missing_official_inputs_is_non_comparable() -> None:
    report = evaluate_m0b(
        None,
        {
            "official_inputs": False,
            "official_preprocessing": True,
            "official_checkpoint": True,
            "official_evaluator": True,
        },
    )
    assert report["status"] == "non-comparable"
    assert report["missing"] == ["official_inputs"]


def test_m0b_accepts_anchor_tolerance_or_covering_interval() -> None:
    prerequisites = {
        "official_inputs": True,
        "official_preprocessing": True,
        "official_checkpoint": True,
        "official_evaluator": True,
    }
    close = evaluate_m0b({"subject_consistency": 0.86}, prerequisites)
    assert close["status"] == "passed"
    covered = evaluate_m0b(
        {"subject_consistency": 0.84, "ci_low": 0.83, "ci_high": 0.88}, prerequisites
    )
    assert covered["status"] == "passed"
    far = evaluate_m0b({"subject_consistency": 0.80}, prerequisites)
    assert far["status"] == "failed"
