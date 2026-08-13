"""Machine-readable E0/M0 gate reports for the remote SlotMem platform run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


M0B_ANCHOR = 0.8771
M0B_TOLERANCE = 0.02
M0B_PREREQUISITES = (
    "official_inputs",
    "official_preprocessing",
    "official_checkpoint",
    "official_evaluator",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_m0a(
    output_dir: Path,
    *,
    efficiency_path: Path | None = None,
    platform_manifest: Path | None = None,
    expected_chunks: int = 7,
) -> dict:
    output_dir = output_dir.resolve()
    reasons: list[str] = []
    manifest_path = output_dir / "slotmem_inference_metadata.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    completed = int(manifest.get("completed_chunk_count", 0) or 0)
    if completed != expected_chunks:
        reasons.append(f"completed_chunks:{completed}/{expected_chunks}")

    missing_videos = [
        index for index in range(expected_chunks)
        if not (output_dir / f"chunk_{index:03d}.mp4").is_file()
    ]
    missing_metadata = [
        index for index in range(expected_chunks)
        if not (output_dir / f"chunk_{index:03d}.metadata.json").is_file()
    ]
    if missing_videos:
        reasons.append(f"missing_chunk_videos:{','.join(map(str, missing_videos))}")
    if missing_metadata:
        reasons.append(f"missing_chunk_metadata:{','.join(map(str, missing_metadata))}")

    runtime = manifest.get("runtime_evidence", {}) if isinstance(manifest, dict) else {}
    nonempty_reads = int(runtime.get("nonempty_memory_reads", 0) or 0)
    if nonempty_reads <= 0:
        reasons.append("no_nonempty_memory_read")
    loaded_domains = sorted(str(value) for value in runtime.get("loaded_checkpoint_domains", []))
    if loaded_domains != ["high_noise", "low_noise"]:
        reasons.append("checkpoint_domains_not_both_loaded")
    writer_changes = int(runtime.get("writer_bank_hash_changes", 0) or 0)
    if writer_changes <= 0:
        reasons.append("no_writer_bank_hash_change")
    writer_residuals = int(runtime.get("writer_positive_residual_count", 0) or 0)
    if writer_residuals <= 0:
        reasons.append("no_positive_writer_residual")

    args_path = output_dir / "inference_args.yaml"
    inference_args = _read_json(args_path) if args_path.is_file() else {}
    if isinstance(inference_args.get("args"), dict):
        inference_args = inference_args["args"]
    if str(inference_args.get("train_stage", "")) != "stage2":
        reasons.append("not_stage2_inference")

    efficiency = _read_json(efficiency_path) if efficiency_path and efficiency_path.is_file() else {}
    total_elapsed = float(efficiency.get("total_elapsed_s", 0.0) or 0.0)
    peak_allocated = float(efficiency.get("peak_allocated_gb", 0.0) or 0.0)
    peak_reserved = float(efficiency.get("peak_reserved_gb", 0.0) or 0.0)
    if total_elapsed <= 0:
        reasons.append("wall_time_missing")
    if peak_allocated <= 0 or peak_reserved <= 0:
        reasons.append("vram_evidence_missing")

    platform = _read_json(platform_manifest) if platform_manifest and platform_manifest.is_file() else {}
    checkpoints = platform.get("checkpoints", {}) if isinstance(platform, dict) else {}
    if not any("stage2" in str(key).lower() for key in checkpoints):
        reasons.append("platform_manifest_missing_stage2")

    return {
        "schema_version": 1,
        "gate": "M0a",
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "output_dir": str(output_dir),
        "evidence": {
            "completed_chunks": completed,
            "expected_chunks": expected_chunks,
            "nonempty_memory_reads": nonempty_reads,
            "writer_bank_hash_changes": writer_changes,
            "writer_positive_residual_count": writer_residuals,
            "loaded_checkpoint_domains": loaded_domains,
            "total_elapsed_s": total_elapsed,
            "peak_allocated_gb": peak_allocated,
            "peak_reserved_gb": peak_reserved,
            "platform_manifest": str(platform_manifest.resolve()) if platform_manifest else None,
        },
    }


def evaluate_m0b(metric_result: Mapping | None, prerequisites: Mapping[str, bool]) -> dict:
    missing = sorted(key for key in M0B_PREREQUISITES if not bool(prerequisites.get(key, False)))
    if missing:
        return {
            "schema_version": 1,
            "gate": "M0b",
            "status": "non-comparable",
            "missing": missing,
            "anchor": M0B_ANCHOR,
            "claim_allowed": False,
        }
    if metric_result is None or "subject_consistency" not in metric_result:
        return {
            "schema_version": 1,
            "gate": "M0b",
            "status": "failed",
            "reasons": ["subject_consistency_missing"],
            "anchor": M0B_ANCHOR,
            "claim_allowed": False,
        }
    value = float(metric_result["subject_consistency"])
    ci_low = metric_result.get("ci_low")
    ci_high = metric_result.get("ci_high")
    within_tolerance = abs(value - M0B_ANCHOR) <= M0B_TOLERANCE
    interval_covers = (
        ci_low is not None
        and ci_high is not None
        and float(ci_low) <= M0B_ANCHOR <= float(ci_high)
    )
    passed = within_tolerance or interval_covers
    return {
        "schema_version": 1,
        "gate": "M0b",
        "status": "passed" if passed else "failed",
        "subject_consistency": value,
        "anchor": M0B_ANCHOR,
        "absolute_error": abs(value - M0B_ANCHOR),
        "tolerance": M0B_TOLERANCE,
        "ci_low": float(ci_low) if ci_low is not None else None,
        "ci_high": float(ci_high) if ci_high is not None else None,
        "within_tolerance": within_tolerance,
        "interval_covers_anchor": interval_covers,
        "claim_allowed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    m0a = sub.add_parser("m0a")
    m0a.add_argument("--output-dir", type=Path, required=True)
    m0a.add_argument("--efficiency", type=Path, required=True)
    m0a.add_argument("--platform-manifest", type=Path, required=True)
    m0a.add_argument("--out", type=Path, required=True)
    m0a.add_argument("--expected-chunks", type=int, default=7)
    m0b = sub.add_parser("m0b")
    m0b.add_argument("--prerequisites", type=Path, required=True)
    m0b.add_argument("--metric-json", type=Path)
    m0b.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "m0a":
        report = validate_m0a(
            args.output_dir,
            efficiency_path=args.efficiency,
            platform_manifest=args.platform_manifest,
            expected_chunks=args.expected_chunks,
        )
    else:
        prerequisites = _read_json(args.prerequisites)
        metrics = _read_json(args.metric_json) if args.metric_json and args.metric_json.is_file() else None
        report = evaluate_m0b(metrics, prerequisites)
    _write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] in ("passed", "non-comparable") else 2


if __name__ == "__main__":
    raise SystemExit(main())
