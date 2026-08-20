"""Prepare one SlotMem prefix and branch fixed-prefix memory interventions from it."""
from __future__ import annotations

import argparse
import json
import math
import os
import stat
import subprocess
import sys
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .content_audit import ARMS
from .memory_utility import REQUIRED_OUTCOMES, utility_census
from .prefix_contract import build_contract, sha256_file, validate_contract
from .qstar import classify_memory_regime


def _set_option(argv: Sequence[str], name: str, value: str | None) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == name:
            index += 2 if index + 1 < len(argv) and not argv[index + 1].startswith("--") else 1
            continue
        output.append(str(argv[index]))
        index += 1
    if value is not None:
        output.extend([name, str(value)])
    return output


def _write_json(path: Path, payload: Mapping | Sequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_prefix_inference_args(
    event: Mapping, output: Path, argv: Sequence[str]
) -> list[str]:
    """Build native inference args that create, rather than load, a prefix state."""
    result = _set_option(
        argv, "--json_path", str(Path(str(event["source_json_path"])).resolve())
    )
    if event.get("reference_path"):
        result = _set_option(
            result,
            "--ref_image_path",
            str(Path(str(event["reference_path"])).resolve()),
        )
    else:
        result = _set_option(result, "--ref_image_path", None)
    result = _set_option(result, "--max_chunks", str(int(event["target_chunk_idx"])))
    # ponytail: never pin the target to the front of character_list. That reorders the
    # frozen platform's read window for every chunk; preflight picks a naturally-readable
    # event instead. Clear it explicitly so an inherited argv cannot smuggle it back in.
    result = _set_option(result, "--target_character", None)
    result = _set_option(result, "--resume_state_path", None)
    result = _set_option(result, "--start_chunk_idx", None)
    result = _set_option(result, "--target_seed_override", None)
    result = _set_option(result, "--save_state_path", str(output / "prefix_state.pt"))
    result = _set_option(result, "--output_path", str(output / "prefix_generation"))
    return _set_option(
        result,
        "--efficiency_metrics_path",
        str(output / "prefix_generation" / "efficiency.json"),
    )


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
    target_seed_override: int | None = None,
    include_native: bool = False,
) -> dict[str, list[str]]:
    requested = tuple(str(arm) for arm in arms)
    unknown = sorted(set(requested) - set(ARMS))
    if unknown:
        raise ValueError(f"unknown confirmatory arms: {unknown}")
    target_idx = int(contract["event"]["target_chunk_idx"])
    snapshot = str(contract["snapshot"]["path"])
    arm_seed = str(int(contract.get("arm_seed", 0)))
    commands: dict[str, list[str]] = {}
    scheduled: list[tuple[str, str]] = []
    for arm in requested:
        scheduled.append((arm, arm))
        if arm == "correct":
            scheduled.append(("correct_repeat", "correct"))
    if "correct" not in requested:
        scheduled.append(("correct_repeat", "correct"))
    for run_name, arm in scheduled:
        arm_dir = (output_root / run_name).resolve()
        inference_args = _branch_inference_args(
            contract,
            arm_dir,
            target_idx,
            snapshot,
            target_seed_override,
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
    if include_native:
        arm_dir = (output_root / "native").resolve()
        inference_args = _branch_inference_args(
            contract,
            arm_dir,
            target_idx,
            snapshot,
            target_seed_override,
        )
        inference_args = _set_option(inference_args, "--no-native_wan_inference", None)
        inference_args = _set_option(inference_args, "--native_wan_inference", None)
        inference_args.append("--native_wan_inference")
        repo = Path(__file__).resolve().parents[1]
        commands["native"] = [python, "-u", str(repo / "infer_slotmem.py"), *inference_args]
    return commands


def _branch_inference_args(
    contract: Mapping,
    arm_dir: Path,
    target_idx: int,
    snapshot: str,
    target_seed_override: int | None,
) -> list[str]:
    inference_args = list(contract["base_inference_args"])
    inference_args = _set_option(inference_args, "--offload_models", None)
    inference_args = _set_option(inference_args, "--no-offload_models", None)
    offload_models = os.environ.get("SLOTMEM_OFFLOAD_MODELS", "0").strip().lower()
    inference_args.append(
        "--offload_models"
        if offload_models in ("1", "true", "yes", "on")
        else "--no-offload_models"
    )
    inference_args = _set_option(inference_args, "--resume_state_path", snapshot)
    inference_args = _set_option(inference_args, "--start_chunk_idx", str(target_idx))
    inference_args = _set_option(
        inference_args, "--save_state_path", str(arm_dir / "resume_state.pt")
    )
    inference_args = _set_option(
        inference_args,
        "--target_seed_override",
        str(target_seed_override) if target_seed_override is not None else None,
    )
    inference_args = _set_option(inference_args, "--max_chunks", str(target_idx + 2))
    inference_args = _set_option(inference_args, "--output_path", str(arm_dir))
    return _set_option(
        inference_args, "--efficiency_metrics_path", str(arm_dir / "efficiency.json")
    )


def validate_audit_group(reports: Mapping[str, Mapping]) -> list[str]:
    errors: list[str] = []
    for arm in ARMS:
        report = reports.get(arm)
        if report is None:
            errors.append(f"{arm}:missing_report")
            continue
        if int(report.get("native_read_mismatches", 0)) != 0:
            errors.append(f"{arm}:native_read_changed")
            continue
        if int(report.get("target_read_hits", 0)) <= 0:
            errors.append(f"{arm}:target_address_miss")
            continue
        if not bool(report.get("intervention_effective", False)):
            errors.append(f"{arm}:intervention_not_effective")
            continue
        if arm == "no_memory":
            if int(report.get("target_source_non_null_reads", 0)) <= 0:
                errors.append("no_memory:no_read_attempt")
            if int(report.get("target_returned_non_null_reads", 0)) != 0:
                errors.append("no_memory:reader_returned_payload")
        elif arm == "correct":
            if int(report.get("payload_layers_seen", 0)) <= 0:
                errors.append("correct:no_payload_layers")
            elif int(report.get("target_read_mismatches", 0)) != 0:
                errors.append("correct:passthrough_changed")
        elif int(report.get("layers_transformed", 0)) <= 0:
            errors.append(f"{arm}:no_layers_transformed")
    return errors


def validate_runtime_reports(
    contract: Mapping, snapshot: Path, reports: Mapping[str, Mapping]
) -> list[str]:
    """Compare every executed arm's reported runtime with the frozen expectation."""
    errors = validate_contract(contract, snapshot)
    for run_name in (*ARMS, "correct_repeat"):
        report = reports.get(run_name)
        if report is None:
            errors.append(f"{run_name}:missing_report")
            continue
        runtime = report.get("runtime_contract")
        if not isinstance(runtime, Mapping):
            errors.append(f"{run_name}:runtime_contract_missing")
            continue
        errors.extend(
            f"{run_name}:{error}"
            for error in validate_contract(contract, snapshot, runtime)
            if error != "snapshot_sha256_mismatch"
        )
    return errors


def _frame_l1_median(left: Path, right: Path) -> float:
    import numpy as np
    import imageio.v3 as iio

    distances = []
    missing = object()
    for a, b in zip_longest(iio.imiter(left), iio.imiter(right), fillvalue=missing):
        if a is missing or b is missing:
            raise ValueError("video frame count mismatch")
        if a.shape != b.shape:
            raise ValueError(f"video frame shape mismatch: {a.shape} != {b.shape}")
        distances.append(float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))))
    if not distances:
        raise ValueError("no aligned video frames")
    return float(np.median(np.asarray(distances, dtype=np.float64)))


def _has_positive_residual(value) -> bool:
    if isinstance(value, dict):
        residual = float(value.get("residual_norm", 0.0) or 0.0)
        if math.isfinite(residual) and residual > 0.0:
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


def validate_event_run(
    event_run: Path,
    *,
    require_dynamic_writer: bool = False,
    require_native: bool = False,
) -> dict:
    contract_path = event_run / "prefix_contract.json"
    if not contract_path.is_file():
        contract_path = event_run.parent / "prefix_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    snapshot = Path(contract["snapshot"]["path"])
    reports: dict[str, dict] = {}
    writer_evidence: dict[str, dict] = {}
    writer_regimes: dict[str, str] = {}
    for run_name in (*ARMS, "correct_repeat"):
        report_path = event_run / run_name / "audit.json"
        if report_path.is_file():
            reports[run_name] = json.loads(report_path.read_text(encoding="utf-8"))
    errors = validate_runtime_reports(contract, snapshot, reports)
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
        regime = classify_memory_regime(evidence)
        writer_regimes[arm] = regime
        if require_dynamic_writer and regime != "dynamic_writer":
            errors.append(f"{arm}:dynamic_writer_required")
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
        native_path = event_run / "native" / filename
        if native_path.is_file():
            decoded["native_vs_correct"] = _frame_l1_median(
                native_path, video_paths["correct"]
            )
        elif require_native:
            errors.append("native:decoded_video_missing")
    else:
        errors.append("decoded_videos_missing")

    report = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "snapshot_sha256": sha256_file(snapshot) if snapshot.is_file() else None,
        "errors": errors,
        "decoded_l1": decoded,
        "writer_evidence": writer_evidence,
        "writer_regimes": writer_regimes,
        "memory_regime": (
            "dynamic_writer"
            if writer_regimes and all(value == "dynamic_writer" for value in writer_regimes.values())
            else "static_prefix"
        ),
        "arms": reports,
    }
    _write_json(event_run / "intervention_contract.json", report)
    _write_json(event_run / "failure_ledger.json", {"failures": errors})
    utility_path = event_run / "utility_report.json"
    if not utility_path.is_file():
        _write_json(
            utility_path,
            {
                "status": "measurement_incomplete",
                "reason": "decoded_outcome_records_not_provided",
                "required_outcomes": list(REQUIRED_OUTCOMES),
                "utility_label_emitted": False,
            },
        )
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
    inference_args = build_prefix_inference_args(event, output, inference_args)
    repo = Path(__file__).resolve().parents[1]
    _run([args.python, "-u", str(repo / "infer_slotmem.py"), *inference_args], output / "prepare.log")
    if not snapshot.is_file():
        raise RuntimeError("native inference did not write the prefix resume state")
    import torch

    state = torch.load(snapshot, map_location="cpu", weights_only=False)
    if int(state.get("next_chunk_idx", -1)) != int(event["target_chunk_idx"]):
        raise RuntimeError("prefix resume state next_chunk_idx does not equal target_chunk_idx")
    from .run_checks import validate_target_read, writer_delta_status

    efficiency_path = output / "prefix_generation" / "efficiency.json"
    if efficiency_path.is_file():
        efficiency = json.loads(efficiency_path.read_text(encoding="utf-8"))
        target_errors = validate_target_read(efficiency, event)
        if target_errors:
            raise RuntimeError("prefix target read check failed: " + ",".join(target_errors))
        writer = writer_delta_status(efficiency)
        if writer["writer_delta_branch_zero"]:
            print(
                "[prefix] writer delta branch is zero -> static-bank read, not dynamic memory: "
                + json.dumps(writer, ensure_ascii=False),
                flush=True,
            )
    timestep_indices = tuple(
        int(value.strip())
        for value in str(args.timestep_indices).split(",")
        if value.strip()
    )
    contract = build_contract(
        event,
        snapshot,
        inference_args,
        args.platform_manifest,
        arm_seed=args.arm_seed,
        future_target_video=args.future_target_video,
        timestep_indices=timestep_indices,
    )
    if bool(contract["code"]["dirty"]) and not bool(args.allow_dirty_source):
        raise RuntimeError(
            "source tree is dirty; commit the experiment code or pass --allow-dirty-source for development only"
        )
    contract["event_json"] = str(event_copy)
    _write_json(output / "prefix_contract.json", contract)
    os.chmod(snapshot, stat.S_IREAD)
    return 0


def run_arms(args: argparse.Namespace) -> int:
    prefix = args.prefix.resolve()
    contract = json.loads((prefix / "prefix_contract.json").read_text(encoding="utf-8"))
    if args.target_seed_override is not None:
        contract = {
            **contract,
            "runtime_contract": {
                **contract["runtime_contract"],
                "target_seed": int(args.target_seed_override),
            },
        }
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
        target_seed_override=args.target_seed_override,
        include_native=bool(args.include_native),
    )
    expected_hash = contract["snapshot"]["sha256"]
    for name, command in commands.items():
        if sha256_file(snapshot) != expected_hash:
            raise RuntimeError(f"snapshot changed before {name}")
        _run(command, output / name / "run.log")
        if sha256_file(snapshot) != expected_hash:
            raise RuntimeError(f"snapshot changed during {name}")
        if name == "no_memory":
            audit_path = output / "no_memory" / "audit.json"
            if audit_path.is_file():
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if int(audit.get("target_read_hits", 0)) <= 0:
                    raise RuntimeError(
                        "no_memory target read missed; the target character was not addressed "
                        "at target_chunk_idx, so the five-arm comparison would be invalid. "
                        "Pick an event whose target character is actually read at the target chunk."
                    )
    report = validate_event_run(
        output,
        require_dynamic_writer=bool(args.require_dynamic_writer),
        require_native=bool(args.include_native),
    )
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


def score_event(args: argparse.Namespace) -> int:
    payload = json.loads(args.records.read_text(encoding="utf-8"))
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    rules = json.loads(args.rules.read_text(encoding="utf-8"))
    report = utility_census(
        records,
        delta_id=float(rules["delta_id"]),
        quality_margins=rules["quality_margins"],
        dynamic_degree_floor=float(rules["dynamic_degree_floor"]),
        gate_a_floors=rules["gate_a_floors"],
        qualification_seeds=rules["qualification_seeds"],
        formal_seeds=rules["formal_seeds"],
        content_causal=rules.get("content_causal"),
        n_boot=int(rules.get("n_boot", 10000)),
        seed=int(rules.get("bootstrap_seed", 0)),
    )
    rows = list(report.get("events", []))
    incomplete = [row for row in rows if row.get("status") == "measurement_incomplete"]
    report["status"] = "measurement_incomplete" if incomplete or not rows else "complete"
    report["utility_label_emitted"] = bool(rows and not incomplete)
    _write_json(args.event_run / "utility_report.json", report)
    return 0 if report["status"] == "complete" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-prefix")
    prepare.add_argument("--event", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--platform-manifest", type=Path, required=True)
    prepare.add_argument("--inference-args-file", type=Path)
    prepare.add_argument("--arm-seed", type=int, default=0)
    prepare.add_argument("--future-target-video", type=Path)
    prepare.add_argument("--timestep-indices", default="0,12,25,37,49")
    prepare.add_argument("--allow-dirty-source", action="store_true")
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
    run.add_argument("--target-seed-override", type=int)
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--include-native", action="store_true")
    run.add_argument("--require-dynamic-writer", action="store_true")
    run.set_defaults(handler=run_arms)

    donor = sub.add_parser("dump-donor")
    donor.add_argument("--prefix", type=Path, required=True)
    donor.add_argument("--output", type=Path, required=True)
    donor.add_argument("--donor-payload", type=Path, required=True)
    donor.add_argument("--python", default=sys.executable)
    donor.set_defaults(handler=dump_donor)

    validate = sub.add_parser("validate")
    validate.add_argument("--event-run", type=Path, required=True)
    validate.add_argument("--require-native", action="store_true")
    validate.add_argument("--require-dynamic-writer", action="store_true")
    validate.set_defaults(
        handler=lambda ns: 0
        if validate_event_run(
            ns.event_run,
            require_native=bool(ns.require_native),
            require_dynamic_writer=bool(ns.require_dynamic_writer),
        )["status"]
        == "passed"
        else 2
    )

    score = sub.add_parser("score")
    score.add_argument("--event-run", type=Path, required=True)
    score.add_argument("--records", type=Path, required=True)
    score.add_argument("--rules", type=Path, required=True)
    score.set_defaults(handler=score_event)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
