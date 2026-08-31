"""Orchestrate frozen seed-zero ViStoryBench donor jobs for the declared scope."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from .event_harness import (
    _set_option,
    build_arm_commands,
    build_prefix_inference_args,
    load_event,
    offload_models_from_environment,
)
from .input_contract import validate_layerwise_slot_payload
from .prefix_contract import (
    _git_state,
    build_runtime_contract,
    normalized_frozen_args,
    sha256_file,
    validate_contract,
    validate_slotmem_memory_encoder_geometry,
    write_json_no_clobber,
)
from .vistory_donors import donor_selection_event_ids, donor_selection_scope_fields


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_object(path: Path, label: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return dict(value)


def _strict_zero(value: object, label: str) -> None:
    if type(value) is not int or value != 0:
        raise ValueError(f"{label} must be integer 0")


def _json_equal_strict(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _json_equal_strict(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal_strict(a, b) for a, b in zip(left, right)
        )
    return left == right


def _resolve_relative(root: Path, value: object, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError(f"{label} must be relative to the selection parent")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} escapes the selection parent")
    return resolved


def _base_argv(path: Path) -> list[str]:
    value: object = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, Mapping):
        value = value.get("argv")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("base inference args must be a JSON argv list or object")
    return list(value[1:] if value and not value[0].startswith("--") else value)


def validate_frozen_selection(selection_path: Path) -> dict:
    """Validate and resolve the Task-3 selection without changing its contents."""
    selection_path = Path(selection_path).resolve()
    selection = _read_object(selection_path, "donor selection")
    if type(selection.get("schema_version")) is not int or selection["schema_version"] != 1:
        raise ValueError("selection schema_version must be integer 1")
    _strict_zero(selection.get("selection_seed"), "selection_seed")
    _strict_zero(selection.get("donor_seed"), "donor_seed")
    if selection.get("path_contract") != {
        "selection_paths_relative_to": "selection_parent",
        "event_paths_relative_to": "event_parent",
    }:
        raise ValueError("selection does not declare the frozen portable path contract")
    expected_event_ids = donor_selection_event_ids(selection)
    events = selection.get("events")
    if not isinstance(events, list) or len(events) != len(expected_event_ids):
        raise ValueError("selection event count does not match its protocol scope")
    if (
        any(not isinstance(row, Mapping) for row in events)
        or {row.get("target_event_id") for row in events} != expected_event_ids
    ):
        raise ValueError("selection target event IDs do not match its protocol scope")

    root = selection_path.parent
    resolved_events = []
    for row_value in events:
        if not isinstance(row_value, Mapping):
            raise ValueError("selection event must be an object")
        row = dict(row_value)
        _strict_zero(row.get("donor_seed"), "event donor_seed")
        manifest_path = _resolve_relative(root, row.get("manifest_path"), "manifest_path")
        if sha256_file(manifest_path) != row.get("manifest_sha256"):
            raise ValueError(f"donor event manifest SHA-256 mismatch: {manifest_path}")
        manifest = _read_object(manifest_path, "donor event manifest")
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
            raise ValueError("event manifest schema_version must be integer 1")
        _strict_zero(manifest.get("donor_seed"), "event manifest donor_seed")
        if manifest.get("target_event_id") != row.get("target_event_id"):
            raise ValueError("donor event manifest target_event_id mismatch")
        if manifest.get("path_resolution") != "selection_parent":
            raise ValueError("event manifest must use selection_parent path resolution")
        outputs = manifest.get("outputs")
        if not isinstance(outputs, Mapping):
            raise ValueError("donor event manifest outputs must be an object")
        resolved_outputs = {}
        for name in ("story", "event", "reference"):
            output = outputs.get(name)
            if not isinstance(output, Mapping):
                raise ValueError(f"donor event manifest output {name} is invalid")
            output_path = _resolve_relative(root, output.get("path"), f"{name} path")
            if sha256_file(output_path) != output.get("sha256"):
                raise ValueError(f"donor {name} SHA-256 mismatch: {output_path}")
            resolved_outputs[name] = str(output_path)
        raw_event = _read_object(Path(resolved_outputs["event"]), "portable donor event")
        if raw_event.get("path_resolution") != "event_parent":
            raise ValueError("donor event must use path_resolution=event_parent")
        for field in ("source_chunk_idx", "target_chunk_idx", "horizon"):
            if type(raw_event.get(field)) is not int:
                raise ValueError(f"donor event {field} must be an integer")
        event = load_event(Path(resolved_outputs["event"]))
        _strict_zero(event.get("donor_seed"), "portable event donor_seed")
        if Path(event["source_json_path"]) != Path(resolved_outputs["story"]):
            raise ValueError("portable event source_json_path mismatch")
        if Path(event["reference_path"]) != Path(resolved_outputs["reference"]):
            raise ValueError("portable event reference_path mismatch")
        if event.get("reference_sha256") != sha256_file(Path(resolved_outputs["reference"])):
            raise ValueError("portable event reference SHA-256 mismatch")
        resolved_events.append(
            {
                **row,
                "manifest_path": str(manifest_path),
                "event_path": resolved_outputs["event"],
                "event": event,
            }
        )
    return {**selection, "selection_path": str(selection_path), "events": resolved_events}


def _canonical_jobs(
    selection: Mapping,
    output_root: Path,
    base_inference_args_path: Path,
    platform_manifest_path: Path,
    offload_models: bool,
) -> list[dict]:
    base = _base_argv(base_inference_args_path)
    validate_slotmem_memory_encoder_geometry(normalized_frozen_args(base))
    base = _set_option(base, "--seed_base", "0")
    jobs = []
    for selected in sorted(selection["events"], key=lambda row: row["target_event_id"]):
        target_event_id = str(selected["target_event_id"])
        job_root = output_root / "jobs" / target_event_id
        prefix = job_root / "prefix"
        dump = job_root / "dump"
        payload = job_root / "donor_payload.pt"
        audit = dump / "correct" / "audit.json"
        prefix_inference_args = build_prefix_inference_args(
            selected["event"], prefix, base, target_seed_override=0
        )
        dump_commands = build_arm_commands(
            {
                "event": selected["event"],
                "snapshot": {"path": str(prefix / "prefix_state.pt")},
                "arm_seed": 0,
                "base_inference_args": prefix_inference_args,
            },
            output_root=dump,
            event_json=prefix / "event.json",
            arms=("correct",),
            python=sys.executable,
            dump_correct_donor=payload,
            target_seed_override=0,
            offload_models=offload_models,
        )
        correct_command = dump_commands["correct"]
        dump_inference_args = correct_command[correct_command.index("--") + 1 :]
        prefix_argv = [
            sys.executable,
            "-m",
            "utest.event_harness",
            "prepare-prefix",
            "--event",
            selected["event_path"],
            "--output",
            str(prefix),
            "--platform-manifest",
            str(platform_manifest_path),
            "--arm-seed",
            "0",
            "--target-seed-override",
            "0",
            "--python",
            sys.executable,
            "--",
            *base,
        ]
        dump_argv = [
            sys.executable,
            "-m",
            "utest.event_harness",
            "dump-donor",
            "--prefix",
            str(prefix),
            "--output",
            str(dump),
            "--donor-payload",
            str(payload),
            "--target-seed-override",
            "0",
            "--python",
            sys.executable,
        ]
        jobs.append(
            {
                "target_event_id": target_event_id,
                "donor_seed": 0,
                "selection_event": {key: value for key, value in selected.items() if key != "event"},
                "event": selected["event"],
                "prefix_dir": str(prefix),
                "prefix_snapshot": str(prefix / "prefix_state.pt"),
                "prefix_contract": str(prefix / "prefix_contract.json"),
                "dump_dir": str(dump),
                "donor_payload": str(payload),
                "donor_payload_info": str(dump / "donor_payload_info.json"),
                "donor_audit": str(audit),
                "completion": str(job_root / "completion.json"),
                "prefix_inference_args": prefix_inference_args,
                "dump_runtime_contract": build_runtime_contract(
                    selected["event"], dump_inference_args
                ),
                "commands": {
                    "prefix": {
                        "argv": prefix_argv,
                        "claim": str(job_root / "claims" / "prefix.claim.json"),
                        "stdout": str(job_root / "logs" / "prefix.stdout.log"),
                        "stderr": str(job_root / "logs" / "prefix.stderr.log"),
                    },
                    "dump": {
                        "argv": dump_argv,
                        "claim": str(job_root / "claims" / "dump.claim.json"),
                        "stdout": str(job_root / "logs" / "dump.stdout.log"),
                        "stderr": str(job_root / "logs" / "dump.stderr.log"),
                    },
                },
            }
        )
    return jobs


def build_donor_run_manifest(
    *,
    selection_path: Path,
    output_root: Path,
    base_inference_args_path: Path,
    platform_manifest_path: Path,
    python_executable: str,
) -> dict[str, object]:
    """Build the immutable zero-GPU command plan for the frozen donor selection."""
    if str(python_executable) != sys.executable:
        raise ValueError("donor harness python must be the current sys.executable")
    selection_path = Path(selection_path).resolve()
    output_root = Path(output_root).resolve()
    base_inference_args_path = Path(base_inference_args_path).resolve()
    platform_manifest_path = Path(platform_manifest_path).resolve()
    selection = validate_frozen_selection(selection_path)
    offload_models = offload_models_from_environment()
    jobs = _canonical_jobs(
        selection,
        output_root,
        base_inference_args_path,
        platform_manifest_path,
        offload_models,
    )
    commit, dirty = _git_state(REPO_ROOT)
    scope = donor_selection_scope_fields(selection)
    return {
        "schema_version": 1,
        "task_id": "vistorybench_donor_generation_v1",
        "donor_seed": 0,
        "python": sys.executable,
        "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "base_inference_args": str(base_inference_args_path),
        "base_inference_args_sha256": sha256_file(base_inference_args_path),
        "platform_manifest": str(platform_manifest_path),
        "platform_manifest_sha256": sha256_file(platform_manifest_path),
        "repository": {"commit": commit, "dirty": dirty},
        "runtime_environment": {"slotmem_offload_models": offload_models},
        "output_root": str(output_root),
        "jobs": jobs,
        **scope,
    }


def validate_donor_run_manifest(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path).resolve()
    run = _read_object(manifest_path, "donor run manifest")
    if type(run.get("schema_version")) is not int or run["schema_version"] != 1:
        raise ValueError("donor run schema_version must be integer 1")
    _strict_zero(run.get("donor_seed"), "run donor_seed")
    output_root = Path(str(run.get("output_root", ""))).resolve()
    if manifest_path.parent != output_root:
        raise ValueError("donor run manifest must be inside its canonical output_root")
    if run.get("python") != sys.executable:
        raise ValueError("donor run python is not the current sys.executable")
    repository = run.get("repository")
    if (
        not isinstance(repository, Mapping)
        or not isinstance(repository.get("commit"), str)
        or not repository["commit"]
        or type(repository.get("dirty")) is not bool
    ):
        raise ValueError("donor run repository provenance is invalid")
    current_commit, current_dirty = _git_state(REPO_ROOT)
    if current_dirty or (current_commit, current_dirty) != (
        repository["commit"],
        repository["dirty"],
    ):
        raise ValueError("current repository does not match clean donor run provenance")
    runtime_environment = run.get("runtime_environment")
    if (
        not isinstance(runtime_environment, Mapping)
        or set(runtime_environment) != {"slotmem_offload_models"}
        or type(runtime_environment.get("slotmem_offload_models")) is not bool
    ):
        raise ValueError("donor run runtime_environment is invalid")
    for path_key, hash_key in (
        ("selection", "selection_sha256"),
        ("base_inference_args", "base_inference_args_sha256"),
        ("platform_manifest", "platform_manifest_sha256"),
    ):
        path = Path(str(run.get(path_key, ""))).resolve()
        if not path.is_file() or sha256_file(path) != run.get(hash_key):
            raise ValueError(f"donor run {path_key} provenance mismatch")
    selection_path = Path(run["selection"]).resolve()
    base_path = Path(run["base_inference_args"]).resolve()
    platform_path = Path(run["platform_manifest"]).resolve()
    selection = validate_frozen_selection(selection_path)
    expected = {
        "schema_version": 1,
        "task_id": "vistorybench_donor_generation_v1",
        "donor_seed": 0,
        "python": sys.executable,
        "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "base_inference_args": str(base_path),
        "base_inference_args_sha256": sha256_file(base_path),
        "platform_manifest": str(platform_path),
        "platform_manifest_sha256": sha256_file(platform_path),
        "repository": dict(repository),
        "runtime_environment": dict(runtime_environment),
        "output_root": str(output_root),
        "jobs": _canonical_jobs(
            selection,
            output_root,
            base_path,
            platform_path,
            runtime_environment["slotmem_offload_models"],
        ),
        **donor_selection_scope_fields(selection),
    }
    if not _json_equal_strict(run, expected):
        raise ValueError("donor run manifest does not match its canonical derivation")
    return expected


def _run_logged(command: Mapping) -> subprocess.CompletedProcess[str]:
    claim_path = Path(str(command["claim"]))
    stdout_path = Path(str(command["stdout"]))
    stderr_path = Path(str(command["stderr"]))
    for path in (claim_path, stdout_path, stderr_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    with claim_path.open("x", encoding="utf-8") as claim:
        json.dump(dict(command), claim, ensure_ascii=False, indent=2, sort_keys=True)
        claim.write("\n")
        claim.flush()
        os.fsync(claim.fileno())
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open(
        "x", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            list(command["argv"]),
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    return completed


def _validate_prefix(job: Mapping, run: Mapping) -> dict:
    contract_path = Path(str(job["prefix_contract"]))
    snapshot = Path(str(job["prefix_snapshot"]))
    if not contract_path.is_file() or not snapshot.is_file():
        raise FileNotFoundError(f"valid prefix artifacts are missing: {contract_path}")
    contract = _read_object(contract_path, "prefix contract")
    if type(contract.get("schema_version")) is not int or contract["schema_version"] != 1:
        raise ValueError("donor prefix contract schema_version must be integer 1")
    _strict_zero(contract.get("arm_seed"), "prefix arm_seed")
    if not _json_equal_strict(contract.get("event"), job.get("event")):
        raise ValueError("donor prefix event does not match its frozen job")
    if not _json_equal_strict(
        contract.get("base_inference_args"), job.get("prefix_inference_args")
    ):
        raise ValueError("donor prefix inference args do not match its frozen job")
    if not _json_equal_strict(contract.get("code"), run.get("repository")):
        raise ValueError("donor prefix repository provenance mismatch")
    platform = contract.get("platform_manifest")
    if not isinstance(platform, Mapping) or platform.get("sha256") != run.get(
        "platform_manifest_sha256"
    ):
        raise ValueError("donor prefix platform provenance mismatch")
    runtime = build_runtime_contract(contract["event"], contract["base_inference_args"])
    if not _json_equal_strict(contract.get("runtime_contract"), runtime):
        raise ValueError("donor prefix runtime contract is not type-exact")
    errors = validate_contract(contract, snapshot, runtime)
    if errors:
        raise ValueError("invalid donor prefix: " + ",".join(errors))
    return contract


def _validate_payload(job: Mapping) -> dict:
    payload_path = Path(str(job["donor_payload"])).resolve()
    info_path = Path(str(job["donor_payload_info"])).resolve()
    if not payload_path.is_file() or not info_path.is_file():
        raise FileNotFoundError(f"donor payload artifacts are missing: {info_path}")
    info = _read_object(info_path, "donor payload info")
    if info.get("format") != "slotmem_donor_payload_v2":
        raise ValueError("donor payload info is not slotmem_donor_payload_v2")
    if Path(str(info.get("payload_path", ""))).resolve() != payload_path:
        raise ValueError("donor payload info path mismatch")
    if info.get("payload_sha256") != sha256_file(payload_path):
        raise ValueError("donor payload SHA-256 mismatch")
    import torch

    try:
        artifact = torch.load(payload_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("donor payload artifact cannot be loaded") from error
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("format") != "slotmem_donor_payload_v2"
        or not _json_equal_strict(artifact.get("event"), job.get("event"))
        or not isinstance(artifact.get("payloads"), Mapping)
    ):
        raise ValueError("donor payload artifact is not the frozen v2 event payload")
    if not _json_equal_strict(info.get("event"), job.get("event")):
        raise ValueError("donor payload event does not match its frozen job")
    event = job.get("event")
    if not isinstance(event, Mapping):
        raise ValueError("donor job event is missing")
    expected_key = f'{event.get("character_name", "")}|0'
    keys = info.get("payload_keys")
    payloads = artifact["payloads"]
    if keys != [expected_key] or set(payloads) != {expected_key}:
        raise ValueError(
            "donor payload must contain exactly the target-character bank 0 key"
        )
    runtime = job.get("dump_runtime_contract")
    frozen_args = runtime.get("frozen_args") if isinstance(runtime, Mapping) else None
    if not isinstance(frozen_args, Mapping):
        raise ValueError("donor dump runtime is missing frozen args")
    expected_layers, expected_slots = validate_slotmem_memory_encoder_geometry(
        frozen_args
    )
    actual_shapes = validate_layerwise_slot_payload(
        payloads[expected_key],
        expected_layers=expected_layers,
        expected_slots=expected_slots,
    )
    shapes = info.get("payload_slot_shapes")
    if not isinstance(shapes, Mapping) or set(shapes) != {expected_key}:
        raise ValueError("donor payload keys/shapes are invalid")
    if not _json_equal_strict(shapes[expected_key], actual_shapes):
        raise ValueError("donor payload slot shapes do not match payload info")
    audit_path = Path(str(job["donor_audit"])).resolve()
    if not audit_path.is_file():
        raise FileNotFoundError(f"donor audit is missing: {audit_path}")
    audit = _read_object(audit_path, "donor audit")
    if not _json_equal_strict(
        audit.get("runtime_contract"), job.get("dump_runtime_contract")
    ):
        raise ValueError("donor audit runtime contract mismatch")
    if (
        audit.get("arm") != "correct"
        or type(audit.get("seed")) is not int
        or audit["seed"] != 0
        or audit.get("target_character") != event.get("character_name")
        or type(audit.get("target_chunk_idx")) is not int
        or audit["target_chunk_idx"] != event.get("target_chunk_idx")
        or Path(str(audit.get("donor_dumped", ""))).resolve() != payload_path
        or audit.get("donor_sha256") != sha256_file(payload_path)
        or type(audit.get("target_read_hits")) is not int
        or audit["target_read_hits"] <= 0
        or audit.get("intervention_effective") is not True
    ):
        raise ValueError("donor audit does not bind the correct frozen dump")
    return {
        "info": info,
        "artifact": artifact,
        "audit": audit,
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": sha256_file(audit_path),
    }


def _execution_record(job: Mapping, stage: str) -> dict:
    command = job["commands"][stage]
    paths = {key: Path(str(command[key])).resolve() for key in ("claim", "stdout", "stderr")}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"donor execution record is missing for {stage}")
    if not _json_equal_strict(
        _read_object(paths["claim"], f"{stage} execution claim"), command
    ):
        raise ValueError(f"{stage} execution claim does not match canonical command")
    return {
        key: {"path": command[key], "sha256": sha256_file(path)}
        for key, path in paths.items()
    }


def validate_donor_run_paths(run: Mapping) -> None:
    """Fail closed unless every donor artifact stays below real, non-link ancestors."""
    root = Path(str(run["output_root"]))
    resolved_root = root.resolve()
    is_link = lambda path: path.is_symlink() or (
        hasattr(path, "is_junction") and path.is_junction()
    )
    if is_link(root):
        raise ValueError(f"donor output root is a symlink or junction: {root}")
    for job in run["jobs"]:
        paths = [
            job[key]
            for key in (
                "prefix_dir",
                "prefix_snapshot",
                "prefix_contract",
                "dump_dir",
                "donor_payload",
                "donor_payload_info",
                "donor_audit",
                "completion",
            )
        ]
        paths.extend(
            job["commands"][stage][key]
            for stage in ("prefix", "dump")
            for key in ("claim", "stdout", "stderr")
        )
        for value in paths:
            path = Path(str(value))
            try:
                relative = path.relative_to(root)
            except ValueError as error:
                raise ValueError(f"donor write path escapes output root: {value}") from error
            cursor = root
            for part in relative.parts:
                cursor /= part
                if is_link(cursor):
                    raise ValueError(f"donor write path has a symlink ancestor: {value}")
            if not path.resolve().is_relative_to(resolved_root):
                raise ValueError(f"donor write path has a symlink escape: {value}")


def _completion_record(job: Mapping, run: Mapping) -> dict:
    return {
        "schema_version": 1,
        "target_event_id": job["target_event_id"],
        "donor_seed": 0,
        "prefix_snapshot": {
            "path": job["prefix_snapshot"],
            "sha256": sha256_file(Path(job["prefix_snapshot"])),
        },
        "prefix_contract": {
            "path": job["prefix_contract"],
            "sha256": sha256_file(Path(job["prefix_contract"])),
        },
        "donor_payload": {
            "path": job["donor_payload"],
            "sha256": sha256_file(Path(job["donor_payload"])),
        },
        "donor_payload_info": {
            "path": job["donor_payload_info"],
            "sha256": sha256_file(Path(job["donor_payload_info"])),
        },
        "donor_audit": {
            "path": job["donor_audit"],
            "sha256": sha256_file(Path(job["donor_audit"])),
        },
        "repository": run["repository"],
        "platform_manifest": {
            "path": run["platform_manifest"],
            "sha256": run["platform_manifest_sha256"],
        },
        "dump_runtime_contract": job["dump_runtime_contract"],
        "execution": {
            stage: _execution_record(job, stage) for stage in ("prefix", "dump")
        },
        **donor_selection_scope_fields(run),
    }


def _write_completion(job: Mapping, run: Mapping) -> dict:
    _validate_prefix(job, run)
    _validate_payload(job)
    completion = _completion_record(job, run)
    write_json_no_clobber(Path(job["completion"]), completion)
    return completion


def _validate_completion(job: Mapping, run: Mapping) -> dict:
    completion_path = Path(str(job["completion"]))
    if not completion_path.is_file():
        raise FileNotFoundError(f"donor completion is missing: {completion_path}")
    _validate_prefix(job, run)
    _validate_payload(job)
    completion = _read_object(completion_path, "donor completion")
    if not _json_equal_strict(completion, _completion_record(job, run)):
        raise ValueError("donor completion does not match current artifact hashes")
    return completion


def validate_completed_donor_run(
    manifest_path: Path, selection: Mapping
) -> dict:
    """Return a donor run only after every scoped frozen job validates."""
    run = validate_donor_run_manifest(manifest_path)
    validate_donor_run_paths(run)
    selection_path = Path(str(selection.get("selection_path", ""))).resolve()
    if not selection_path.is_file() or sha256_file(selection_path) != run.get(
        "selection_sha256"
    ):
        raise ValueError("completed donor run selection provenance mismatch")
    canonical_selection = validate_frozen_selection(selection_path)
    if not _json_equal_strict(selection, canonical_selection):
        raise ValueError("completed donor run requires the canonical selection")
    for job in run["jobs"]:
        _validate_completion(job, run)
    return run


def _execute_job_command(
    job: Mapping, run: Mapping, stage: str, command_name: str
) -> dict:
    command = job["commands"][command_name]
    completed = _run_logged(command)
    result = {
        "target_event_id": job["target_event_id"],
        "stage": stage,
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout": command["stdout"],
        "stderr": command["stderr"],
    }
    if stage == "resume":
        result["command"] = command_name
    if completed.returncode == 0:
        try:
            if command_name == "prefix":
                _validate_prefix(job, run)
                _execution_record(job, "prefix")
            else:
                _write_completion(job, run)
        except (FileNotFoundError, ValueError) as error:
            result.update(status="failed_validation", error=str(error))
    return result


def run_stage(stage: str, manifest_path: Path) -> dict[str, object]:
    if stage not in {"prefix", "dump", "resume"}:
        raise ValueError(f"unsupported donor run stage: {stage}")
    run = validate_donor_run_manifest(manifest_path)
    if (
        offload_models_from_environment()
        is not run["runtime_environment"]["slotmem_offload_models"]
    ):
        raise ValueError("current offload environment does not match donor run")
    validate_donor_run_paths(run)
    results = []
    for job in run["jobs"]:
        if stage == "resume":
            if os.path.lexists(str(job["completion"])):
                _validate_completion(job, run)
                results.append(
                    {
                        "target_event_id": job["target_event_id"],
                        "stage": stage,
                        "status": "skipped_valid",
                        "donor_payload": job["donor_payload"],
                        "donor_payload_info": job["donor_payload_info"],
                    }
                )
                continue
            state_paths = (
                job["prefix_dir"],
                job["prefix_snapshot"],
                job["prefix_contract"],
                job["dump_dir"],
                job["donor_payload"],
                job["donor_payload_info"],
                job["donor_audit"],
                job["commands"]["prefix"]["claim"],
                job["commands"]["prefix"]["stdout"],
                job["commands"]["prefix"]["stderr"],
                job["commands"]["dump"]["claim"],
                job["commands"]["dump"]["stdout"],
                job["commands"]["dump"]["stderr"],
            )
            if any(os.path.lexists(str(path)) for path in state_paths):
                raise ValueError(
                    f"partial donor job state refuses resume: {job['target_event_id']}"
                )
            prefix_result = _execute_job_command(job, run, stage, "prefix")
            if prefix_result["status"] != "completed":
                results.append(prefix_result)
                break
            result = _execute_job_command(job, run, stage, "dump")
            results.append(result)
            if result["status"] != "completed":
                break
            continue

        if stage == "prefix" and any(
            os.path.lexists(str(job[key]))
            for key in (
                "prefix_dir",
                "prefix_snapshot",
                "prefix_contract",
                "dump_dir",
                "donor_payload",
                "donor_payload_info",
                "donor_audit",
                "completion",
            )
        ):
            raise FileExistsError(
                f"donor prefix output already exists: {job['prefix_dir']}"
            )
        if stage == "dump":
            _validate_prefix(job, run)
            _execution_record(job, "prefix")
            if any(
                os.path.lexists(str(job[key]))
                for key in (
                    "donor_payload",
                    "donor_payload_info",
                    "donor_audit",
                    "completion",
                    "dump_dir",
                )
            ):
                raise FileExistsError(
                    f"donor dump output already exists: {job['dump_dir']}"
                )
        result = _execute_job_command(job, run, stage, stage)
        results.append(result)
        if result["status"] != "completed":
            break
    return {"stage": stage, "results": results}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dry = sub.add_parser("dry-run")
    dry.add_argument("--selection", type=Path, required=True)
    dry.add_argument("--output", type=Path, required=True)
    dry.add_argument("--base-inference-args", type=Path, required=True)
    dry.add_argument("--platform-manifest", type=Path, required=True)
    for stage in ("prefix", "dump", "resume"):
        stage_parser = sub.add_parser(stage)
        stage_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "dry-run":
        run = build_donor_run_manifest(
            selection_path=args.selection,
            output_root=args.output,
            base_inference_args_path=args.base_inference_args,
            platform_manifest_path=args.platform_manifest,
            python_executable=sys.executable,
        )
        manifest_path = args.output.resolve() / "run_manifest.json"
        write_json_no_clobber(manifest_path, run)
        print(json.dumps({"jobs": len(run["jobs"]), "manifest": str(manifest_path)}))
        return 0
    expected_jobs = len(validate_donor_run_manifest(args.manifest)["jobs"])
    result = run_stage(args.command, args.manifest)
    print(json.dumps(result, ensure_ascii=False))
    statuses = {row.get("status") for row in result["results"]}
    return (
        0
        if len(result["results"]) == expected_jobs
        and statuses <= {"completed", "skipped_valid"}
        else 2
    )


if __name__ == "__main__":
    status = main()
    if status != 0:
        raise RuntimeError(f"donor harness failed with status {status}")
