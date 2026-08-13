"""Prepare one SlotMem prefix and branch fixed-prefix memory interventions from it."""
from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .content_audit import ARMS
from .prefix_contract import build_contract, sha256_file, validate_contract


def _set_option(argv: Sequence[str], name: str, value: str) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == name:
            index += 2 if index + 1 < len(argv) and not argv[index + 1].startswith("--") else 1
            continue
        output.append(str(argv[index]))
        index += 1
    output.extend([name, str(value)])
    return output


def _write_json(path: Path, payload: Mapping | Sequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_arm_commands(
    contract: Mapping,
    *,
    output_root: Path,
    event_json: Path,
    arms: Iterable[str],
    python: str = sys.executable,
    donor: Path | None = None,
    donor_manifest: Path | None = None,
    dump_correct_donor: Path | None = None,
) -> dict[str, list[str]]:
    requested = tuple(str(arm) for arm in arms)
    unknown = sorted(set(requested) - set(ARMS))
    if unknown:
        raise ValueError(f"unknown confirmatory arms: {unknown}")
    target_idx = int(contract["event"]["target_chunk_idx"])
    snapshot = str(contract["snapshot"]["path"])
    arm_seed = str(int(contract.get("arm_seed", 0)))
    commands: dict[str, list[str]] = {}
    for run_name, arm in [*((arm, arm) for arm in requested), ("correct_repeat", "correct")]:
        arm_dir = (output_root / run_name).resolve()
        inference_args = list(contract["base_inference_args"])
        inference_args = _set_option(inference_args, "--resume_state_path", snapshot)
        inference_args = _set_option(inference_args, "--max_chunks", str(target_idx + 2))
        inference_args = _set_option(inference_args, "--output_path", str(arm_dir))
        inference_args = _set_option(
            inference_args, "--efficiency_metrics_path", str(arm_dir / "efficiency.json")
        )
        command = [
            python,
            "-m",
            "utest.content_audit",
            "--arm",
            arm,
            "--seed",
            arm_seed,
            "--event-json",
            str(event_json.resolve()),
            "--report",
            str(arm_dir / "audit.json"),
        ]
        if arm == "correct" and run_name == "correct" and dump_correct_donor is not None:
            command.extend(["--dump-donor", str(dump_correct_donor.resolve())])
        if arm == "wrong":
            if donor is None or donor_manifest is None:
                raise ValueError("wrong arm requires donor and donor_manifest")
            command.extend(
                ["--donor", str(donor.resolve()), "--donor-manifest", str(donor_manifest.resolve())]
            )
        command.extend(["--", *inference_args])
        commands[run_name] = command
    return commands


def validate_audit_group(reports: Mapping[str, Mapping]) -> list[str]:
    errors: list[str] = []
    for arm in ARMS:
        report = reports.get(arm)
        if report is None:
            errors.append(f"{arm}:missing_report")
            continue
        if int(report.get("target_read_hits", 0)) <= 0:
            errors.append(f"{arm}:target_address_miss")
            continue
        if not bool(report.get("intervention_effective", False)):
            errors.append(f"{arm}:intervention_not_effective")
            continue
        if arm == "no_memory":
            if int(report.get("attempted_reads", 0)) <= 0:
                errors.append("no_memory:no_read_attempt")
            if int(report.get("returned_non_null_reads", 0)) != 0:
                errors.append("no_memory:reader_returned_payload")
        elif arm == "correct":
            if int(report.get("payload_layers_seen", 0)) <= 0:
                errors.append("correct:no_payload_layers")
        elif int(report.get("layers_transformed", 0)) <= 0:
            errors.append(f"{arm}:no_layers_transformed")
    return errors


def _frame_l1_median(left: Path, right: Path) -> float:
    import numpy as np
    import imageio.v3 as iio

    distances = []
    for a, b in zip(iio.imiter(left), iio.imiter(right)):
        if a.shape != b.shape:
            raise ValueError(f"video frame shape mismatch: {a.shape} != {b.shape}")
        distances.append(float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))))
    if not distances:
        raise ValueError("no aligned video frames")
    return float(np.median(np.asarray(distances, dtype=np.float64)))


def _has_positive_residual(value) -> bool:
    if isinstance(value, dict):
        if float(value.get("residual_norm", 0.0) or 0.0) > 0.0:
            return True
        return any(_has_positive_residual(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_positive_residual(item) for item in value)
    return False


def _writer_evidence(efficiency: Mapping, target_idx: int) -> dict:
    chunks = [
        row for row in list(efficiency.get("chunks", []) or [])
        if int(row.get("chunk_idx", -1)) >= target_idx
    ]
    updates = [
        update for row in chunks for update in list(row.get("writer_updates", []) or [])
    ]
    return {
        "update_count": len(updates),
        "positive_residual_count": sum(
            1 for update in updates if _has_positive_residual(update.get("stats", {}))
        ),
        "bank_hash_change_count": sum(
            1 for row in chunks if bool(row.get("memory_bank_hash_changed", False))
        ),
    }


def validate_event_run(event_run: Path) -> dict:
    contract_path = event_run / "prefix_contract.json"
    if not contract_path.is_file():
        contract_path = event_run.parent / "prefix_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    snapshot = Path(contract["snapshot"]["path"])
    errors = validate_contract(contract, snapshot, contract["runtime_contract"])
    reports: dict[str, dict] = {}
    writer_evidence: dict[str, dict] = {}
    for arm in ARMS:
        report_path = event_run / arm / "audit.json"
        if report_path.is_file():
            reports[arm] = json.loads(report_path.read_text(encoding="utf-8"))
    errors.extend(validate_audit_group(reports))

    target_idx = int(contract["event"]["target_chunk_idx"])
    for arm in ARMS:
        efficiency_path = event_run / arm / "efficiency.json"
        if not efficiency_path.is_file():
            errors.append(f"{arm}:efficiency_missing")
            continue
        efficiency = json.loads(efficiency_path.read_text(encoding="utf-8"))
        evidence = _writer_evidence(efficiency, target_idx)
        writer_evidence[arm] = evidence
        if evidence["update_count"] <= 0:
            errors.append(f"{arm}:writer_update_missing")
        if evidence["positive_residual_count"] <= 0:
            errors.append(f"{arm}:writer_residual_not_positive")
        if evidence["bank_hash_change_count"] <= 0:
            errors.append(f"{arm}:bank_hash_unchanged")
    filename = f"chunk_{target_idx:03d}.mp4"
    video_paths = {name: event_run / name / filename for name in (*ARMS, "correct_repeat")}
    decoded: dict[str, object] = {}
    if all(video_paths[name].is_file() for name in ("correct", "correct_repeat", "no_memory")):
        repeat_floor = _frame_l1_median(video_paths["correct"], video_paths["correct_repeat"])
        correct_none = _frame_l1_median(video_paths["correct"], video_paths["no_memory"])
        decoded.update(
            technical_repeat_floor=repeat_floor,
            correct_vs_no_memory=correct_none,
            intervention_diverged=correct_none > repeat_floor,
        )
        if correct_none <= repeat_floor:
            errors.append("decoded_intervention_below_repeat_floor")
        for arm in ("zero", "wrong", "random"):
            if video_paths[arm].is_file():
                decoded[f"{arm}_vs_correct"] = _frame_l1_median(
                    video_paths[arm], video_paths["correct"]
                )
    else:
        errors.append("decoded_videos_missing")

    report = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "snapshot_sha256": sha256_file(snapshot) if snapshot.is_file() else None,
        "errors": errors,
        "decoded_l1": decoded,
        "writer_evidence": writer_evidence,
        "arms": reports,
    }
    _write_json(event_run / "intervention_contract.json", report)
    _write_json(event_run / "failure_ledger.json", {"failures": errors})
    return report


def _run(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, check=True)


def prepare_prefix(args: argparse.Namespace) -> int:
    event = json.loads(args.event.read_text(encoding="utf-8"))
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"prefix output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    event_copy = output / "event.json"
    _write_json(event_copy, event)
    snapshot = output / "prefix_state.pt"
    prefix_output = output / "prefix_generation"
    if args.inference_args_file:
        saved = json.loads(args.inference_args_file.read_text(encoding="utf-8"))
        inference_args = list(saved.get("argv", []))
        if inference_args and not str(inference_args[0]).startswith("--"):
            inference_args = inference_args[1:]
    else:
        inference_args = list(args.inference_args)
    if inference_args[:1] == ["--"]:
        inference_args = inference_args[1:]
    inference_args = _set_option(inference_args, "--json_path", str(Path(event["source_json_path"]).resolve()))
    if event.get("reference_path"):
        inference_args = _set_option(
            inference_args, "--ref_image_path", str(Path(event["reference_path"]).resolve())
        )
    inference_args = _set_option(inference_args, "--max_chunks", str(int(event["target_chunk_idx"])))
    inference_args = _set_option(inference_args, "--resume_state_path", str(snapshot))
    inference_args = _set_option(inference_args, "--output_path", str(prefix_output))
    inference_args = _set_option(
        inference_args, "--efficiency_metrics_path", str(prefix_output / "efficiency.json")
    )
    repo = Path(__file__).resolve().parents[1]
    _run([args.python, "-u", str(repo / "infer_slotmem.py"), *inference_args], output / "prepare.log")
    if not snapshot.is_file():
        raise RuntimeError("native inference did not write the prefix resume state")
    import torch

    state = torch.load(snapshot, map_location="cpu", weights_only=False)
    if int(state.get("next_chunk_idx", -1)) != int(event["target_chunk_idx"]):
        raise RuntimeError("prefix resume state next_chunk_idx does not equal target_chunk_idx")
    contract = build_contract(
        event, snapshot, inference_args, args.platform_manifest, arm_seed=args.arm_seed
    )
    contract["event_json"] = str(event_copy)
    _write_json(output / "prefix_contract.json", contract)
    os.chmod(snapshot, stat.S_IREAD)
    return 0


def run_arms(args: argparse.Namespace) -> int:
    prefix = args.prefix.resolve()
    contract = json.loads((prefix / "prefix_contract.json").read_text(encoding="utf-8"))
    snapshot = Path(contract["snapshot"]["path"])
    event_json = Path(contract.get("event_json", prefix / "event.json"))
    output = (args.output or (prefix / "arms")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "prefix_contract.json", contract)
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    commands = build_arm_commands(
        contract,
        output_root=output,
        event_json=event_json,
        arms=arms,
        python=args.python,
        donor=args.donor,
        donor_manifest=args.donor_manifest,
        dump_correct_donor=args.dump_correct_donor,
    )
    expected_hash = contract["snapshot"]["sha256"]
    for name, command in commands.items():
        if sha256_file(snapshot) != expected_hash:
            raise RuntimeError(f"snapshot changed before {name}")
        _run(command, output / name / "run.log")
        if sha256_file(snapshot) != expected_hash:
            raise RuntimeError(f"snapshot changed during {name}")
    report = validate_event_run(output)
    return 0 if report["status"] == "passed" else 2


def dump_donor(args: argparse.Namespace) -> int:
    prefix = args.prefix.resolve()
    contract = json.loads((prefix / "prefix_contract.json").read_text(encoding="utf-8"))
    snapshot = Path(contract["snapshot"]["path"])
    event_json = Path(contract.get("event_json", prefix / "event.json"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    commands = build_arm_commands(
        contract,
        output_root=output,
        event_json=event_json,
        arms=("correct",),
        python=args.python,
        dump_correct_donor=args.donor_payload,
    )
    expected_hash = contract["snapshot"]["sha256"]
    _run(commands["correct"], output / "correct" / "run.log")
    if sha256_file(snapshot) != expected_hash:
        raise RuntimeError("snapshot changed while dumping donor")
    report = json.loads((output / "correct" / "audit.json").read_text(encoding="utf-8"))
    if int(report.get("target_read_hits", 0)) <= 0 or not report.get("intervention_effective"):
        raise RuntimeError("donor correct run did not resolve the target character")
    import torch

    payload = torch.load(args.donor_payload, map_location="cpu", weights_only=False)
    keys = sorted(str(key) for key in payload.get("payloads", {}))
    info = {
        "format": payload.get("format"),
        "payload_path": str(args.donor_payload.resolve()),
        "payload_sha256": sha256_file(args.donor_payload),
        "payload_keys": keys,
        "event": contract["event"],
    }
    _write_json(output / "donor_payload_info.json", info)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-prefix")
    prepare.add_argument("--event", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--platform-manifest", type=Path, required=True)
    prepare.add_argument("--inference-args-file", type=Path)
    prepare.add_argument("--arm-seed", type=int, default=0)
    prepare.add_argument("--python", default=sys.executable)
    prepare.add_argument("inference_args", nargs=argparse.REMAINDER)
    prepare.set_defaults(handler=prepare_prefix)

    run = sub.add_parser("run-arms")
    run.add_argument("--prefix", type=Path, required=True)
    run.add_argument("--output", type=Path)
    run.add_argument("--arms", default=",".join(ARMS))
    run.add_argument("--donor", type=Path)
    run.add_argument("--donor-manifest", type=Path)
    run.add_argument("--dump-correct-donor", type=Path)
    run.add_argument("--python", default=sys.executable)
    run.set_defaults(handler=run_arms)

    donor = sub.add_parser("dump-donor")
    donor.add_argument("--prefix", type=Path, required=True)
    donor.add_argument("--output", type=Path, required=True)
    donor.add_argument("--donor-payload", type=Path, required=True)
    donor.add_argument("--python", default=sys.executable)
    donor.set_defaults(handler=dump_donor)

    validate = sub.add_parser("validate")
    validate.add_argument("--event-run", type=Path, required=True)
    validate.set_defaults(handler=lambda ns: 0 if validate_event_run(ns.event_run)["status"] == "passed" else 2)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
