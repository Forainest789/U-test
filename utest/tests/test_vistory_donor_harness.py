from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

import utest.prefix_contract as prefix_contract
import utest.vistory_donor_harness as donor_harness
from utest.prefix_contract import build_contract
from utest.vistory_donor_harness import (
    build_donor_run_manifest,
    main,
    run_stage,
    validate_completed_donor_run,
    validate_frozen_selection,
)
from utest.vistory_donors import (
    EXPLORATORY_SINGLE_EVENT_SCOPE,
    TARGET_EVENT_IDS,
    donor_selection_scope_fields,
)


TARGET_IDS = tuple(sorted(TARGET_EVENT_IDS))


@pytest.fixture(autouse=True)
def _clean_repository_provenance(monkeypatch) -> None:
    state = lambda _repo: ("frozen-test-commit", False)
    monkeypatch.setattr(donor_harness, "_git_state", state)
    monkeypatch.setattr(prefix_contract, "_git_state", state)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialize_execution(job: dict, stage: str) -> None:
    command = job["commands"][stage]
    claim = Path(command["claim"])
    if not claim.exists():
        _write_json(claim, command)
    for key in ("stdout", "stderr"):
        log = Path(command[key])
        if not log.exists():
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("", encoding="utf-8")


def _selection(tmp_path: Path) -> Path:
    root = tmp_path / "selection"
    events = []
    for index, target_id in enumerate(reversed(TARGET_IDS), start=1):
        event_root = root / target_id
        story = event_root / "story.json"
        reference = event_root / "reference.jpg"
        event = event_root / "event.json"
        _write_json(story, {"chunks": [{"content": "source"}, {"content": "read"}]})
        reference.write_bytes(f"reference-{index}".encode())
        _write_json(
            event,
            {
                "event_id": f"donor-{index}",
                "story_id": f"story-{index}",
                "character_name": f"person-{index}",
                "source_chunk_idx": 0,
                "target_chunk_idx": 1,
                "horizon": 1,
                "donor_seed": 0,
                "path_resolution": "event_parent",
                "source_json_path": "story.json",
                "reference_path": "reference.jpg",
                "reference_sha256": _sha256(reference),
            },
        )
        manifest = event_root / "manifest.json"
        _write_json(
            manifest,
            {
                "schema_version": 1,
                "target_event_id": target_id,
                "donor_seed": 0,
                "path_resolution": "selection_parent",
                "outputs": {
                    "story": {"path": story.relative_to(root).as_posix(), "sha256": _sha256(story)},
                    "event": {"path": event.relative_to(root).as_posix(), "sha256": _sha256(event)},
                    "reference": {
                        "path": reference.relative_to(root).as_posix(),
                        "sha256": _sha256(reference),
                    },
                },
            },
        )
        events.append(
            {
                "target_event_id": target_id,
                "candidate_id": f"candidate-{index}",
                "donor_story_id": f"story-{index}",
                "donor_entity_uid": f"entity-{index}",
                "donor_seed": 0,
                "manifest_path": manifest.relative_to(root).as_posix(),
                "manifest_sha256": _sha256(manifest),
            }
        )
    selection = root / "selection.json"
    _write_json(
        selection,
        {
            "schema_version": 1,
            "dataset_commit": "dataset-commit",
            "selection_seed": 0,
            "donor_seed": 0,
            "path_contract": {
                "selection_paths_relative_to": "selection_parent",
                "event_paths_relative_to": "event_parent",
            },
            "target_inputs_sha256": "a" * 64,
            "survey_sha256": "b" * 64,
            "review_sha256": "c" * 64,
            "candidate_audit": [],
            "events": events,
        },
    )
    return selection


def _exploratory_selection(tmp_path: Path, target_id: str = TARGET_IDS[0]) -> Path:
    selection = _selection(tmp_path)
    value = json.loads(selection.read_text(encoding="utf-8"))
    value.update(
        protocol_scope=EXPLORATORY_SINGLE_EVENT_SCOPE,
        target_event_ids=[target_id],
        events=[
            event for event in value["events"] if event["target_event_id"] == target_id
        ],
    )
    _write_json(selection, value)
    return selection


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "base.json"
    platform = tmp_path / "platform.json"
    _write_json(
        base,
        {
            "argv": [
                "python",
                "--seed_base",
                "99",
                "--resume_state_path",
                "stale.pt",
                "--output_path",
                "stale",
                "--slotmem_memory_encoder_layers",
                "0-15",
                "--slotmem_memory_encoder_slots",
                "64",
            ]
        },
    )
    _write_json(platform, {"repo_commit": "platform", "repo_dirty": False})
    return base, platform


def _materialize_prefix(job: dict, platform: Path) -> None:
    snapshot = Path(job["prefix_snapshot"])
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(b"frozen-prefix")
    contract = build_contract(
        job["event"],
        snapshot,
        job["prefix_inference_args"],
        platform,
        arm_seed=0,
    )
    contract["event_json"] = str(snapshot.parent / "event.json")
    _write_json(Path(job["prefix_contract"]), contract)
    _materialize_execution(job, "prefix")


def _materialize_payload(job: dict) -> None:
    payload = Path(job["donor_payload"])
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload_key = f'{job["event"]["character_name"]}|0'
    layer_shapes = {str(layer): [64, 3] for layer in range(16)}
    torch.save(
        {
            "format": "slotmem_donor_payload_v2",
            "event": job["event"],
            "payloads": {
                payload_key: {
                    "__layerwise__": True,
                    "layers": {
                        str(layer): torch.zeros((64, 3), dtype=torch.float16)
                        for layer in range(16)
                    },
                }
            },
        },
        payload,
    )
    _write_json(
        Path(job["donor_payload_info"]),
        {
            "format": "slotmem_donor_payload_v2",
            "payload_path": str(payload.resolve()),
            "payload_sha256": _sha256(payload),
            "payload_keys": [payload_key],
            "payload_slot_shapes": {payload_key: layer_shapes},
            "event": job["event"],
        },
    )
    _write_json(
        Path(job["dump_dir"]) / "correct" / "audit.json",
        {
            "arm": "correct",
            "seed": 0,
            "target_character": job["event"]["character_name"],
            "target_chunk_idx": job["event"]["target_chunk_idx"],
            "target_read_hits": 1,
            "intervention_effective": True,
            "donor_dumped": str(payload.resolve()),
            "donor_sha256": _sha256(payload),
            "runtime_contract": job["dump_runtime_contract"],
        },
    )
    _materialize_execution(job, "dump")


def _mutate_payload_geometry(job: dict, malformation: str) -> None:
    payload_path = Path(job["donor_payload"])
    artifact = torch.load(payload_path, map_location="cpu", weights_only=True)
    payload_key = next(iter(artifact["payloads"]))
    payload = artifact["payloads"][payload_key]
    if malformation == "flat":
        payload = torch.zeros((64, 3), dtype=torch.float16)
    else:
        layers = payload["layers"]
        if malformation == "empty":
            layers.clear()
        elif malformation == "missing":
            layers.pop("15")
        elif malformation == "extra":
            layers["16"] = torch.zeros((64, 3), dtype=torch.float16)
        elif malformation == "wrong_slots":
            layers["0"] = torch.zeros((63, 3), dtype=torch.float16)
        elif malformation == "integer_tensor":
            layers["0"] = torch.zeros((64, 3), dtype=torch.int64)
        elif malformation == "nonfinite":
            layers["0"][0, 0] = float("nan")
        elif malformation == "hidden_dim":
            layers["15"] = torch.zeros((64, 4), dtype=torch.float16)
        else:
            raise ValueError(f"unsupported payload malformation: {malformation}")
    artifact["payloads"][payload_key] = payload
    torch.save(artifact, payload_path)

    if isinstance(payload, torch.Tensor):
        shapes = {"0": [int(value) for value in payload.shape]}
    else:
        shapes = {
            layer: [int(value) for value in tensor.shape]
            for layer, tensor in payload["layers"].items()
        }
    info_path = Path(job["donor_payload_info"])
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["payload_sha256"] = _sha256(payload_path)
    info["payload_slot_shapes"] = {payload_key: shapes}
    _write_json(info_path, info)

    audit_path = Path(job["donor_audit"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["donor_sha256"] = _sha256(payload_path)
    _write_json(audit_path, audit)


def _read_runtime_contract(job: dict) -> dict:
    path = Path(job["prefix_contract"])
    if not path.is_file():
        return {}
    contract = json.loads(path.read_text(encoding="utf-8"))
    return contract["runtime_contract"]


def _materialize_completion(job: dict, run: dict) -> None:
    _write_json(
        Path(job["completion"]),
        {
            "schema_version": 1,
            "target_event_id": job["target_event_id"],
            "donor_seed": 0,
            "prefix_snapshot": {
                "path": job["prefix_snapshot"],
                "sha256": _sha256(Path(job["prefix_snapshot"])),
            },
            "prefix_contract": {
                "path": job["prefix_contract"],
                "sha256": _sha256(Path(job["prefix_contract"])),
            },
            "donor_payload": {
                "path": job["donor_payload"],
                "sha256": _sha256(Path(job["donor_payload"])),
            },
            "donor_payload_info": {
                "path": job["donor_payload_info"],
                "sha256": _sha256(Path(job["donor_payload_info"])),
            },
            "donor_audit": {
                "path": job["donor_audit"],
                "sha256": _sha256(Path(job["donor_audit"])),
            },
            "repository": run["repository"],
            "platform_manifest": {
                "path": run["platform_manifest"],
                "sha256": run["platform_manifest_sha256"],
            },
            "dump_runtime_contract": job["dump_runtime_contract"],
            "execution": {
                stage: {
                    key: {
                        "path": job["commands"][stage][key],
                        "sha256": _sha256(Path(job["commands"][stage][key])),
                    }
                    for key in ("claim", "stdout", "stderr")
                }
                for stage in ("prefix", "dump")
            },
            **donor_selection_scope_fields(run),
        },
    )


def test_dry_run_builds_exactly_three_seed_zero_event_harness_jobs(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    base, platform = _inputs(tmp_path)

    run = build_donor_run_manifest(
        selection_path=selection,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )

    assert [job["target_event_id"] for job in run["jobs"]] == list(TARGET_IDS)
    assert {job["donor_seed"] for job in run["jobs"]} == {0}
    assert all(
        job["commands"]["prefix"]["argv"][1:4]
        == ["-m", "utest.event_harness", "prepare-prefix"]
        for job in run["jobs"]
    )
    assert all(
        job["commands"]["dump"]["argv"][1:4]
        == ["-m", "utest.event_harness", "dump-donor"]
        for job in run["jobs"]
    )
    assert all("--resume_state_path" not in job["prefix_inference_args"] for job in run["jobs"])
    assert all(job["prefix_inference_args"][job["prefix_inference_args"].index("--seed_base") + 1] == "0" for job in run["jobs"])
    assert run["selection_sha256"] == _sha256(selection)
    assert run["base_inference_args_sha256"] == _sha256(base)
    assert run["platform_manifest_sha256"] == _sha256(platform)
    assert isinstance(run["repository"]["commit"], str) and run["repository"]["commit"]
    assert type(run["repository"]["dirty"]) is bool
    assert "protocol_scope" not in run
    assert "target_event_ids" not in run
    frozen = prefix_contract.normalized_frozen_args(
        run["jobs"][0]["prefix_inference_args"]
    )
    assert {
        key: frozen[key]
        for key in (
            "slotmem_memory_encoder_layers",
            "slotmem_memory_encoder_slots",
        )
    } == {
        "slotmem_memory_encoder_layers": "0-15",
        "slotmem_memory_encoder_slots": "64",
    }


def test_dry_run_builds_explicit_exploratory_single_event_job(tmp_path: Path) -> None:
    selection = _exploratory_selection(tmp_path)
    base, platform = _inputs(tmp_path)

    run = build_donor_run_manifest(
        selection_path=selection,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )

    assert run["protocol_scope"] == EXPLORATORY_SINGLE_EVENT_SCOPE
    assert run["target_event_ids"] == [TARGET_IDS[0]]
    assert [job["target_event_id"] for job in run["jobs"]] == [TARGET_IDS[0]]
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    assert donor_harness.validate_donor_run_manifest(manifest) == run


def test_donor_run_rejects_exploratory_scope_tamper(tmp_path: Path) -> None:
    selection = _exploratory_selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    run["protocol_scope"] = "formal"
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)

    with pytest.raises(ValueError, match="canonical derivation"):
        donor_harness.validate_donor_run_manifest(manifest)


@pytest.mark.parametrize(
    "scope_fields,error",
    [
        ({"protocol_scope": None, "target_event_ids": None}, "unsupported"),
        ({"protocol_scope": EXPLORATORY_SINGLE_EVENT_SCOPE}, "exactly one"),
        ({"target_event_ids": [TARGET_IDS[0]]}, "unsupported"),
        (
            {"protocol_scope": "formal", "target_event_ids": [TARGET_IDS[0]]},
            "unsupported",
        ),
    ],
)
def test_selection_scope_fields_fail_closed(
    tmp_path: Path, scope_fields: dict, error: str,
) -> None:
    selection = _selection(tmp_path)
    value = json.loads(selection.read_text(encoding="utf-8"))
    value.update(scope_fields)
    _write_json(selection, value)

    with pytest.raises(ValueError, match=error):
        validate_frozen_selection(selection)


@pytest.mark.parametrize("job_change", ["missing", "extra"])
def test_donor_run_rejects_jobs_outside_exploratory_scope(
    tmp_path: Path, job_change: str,
) -> None:
    selection = _exploratory_selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    if job_change == "missing":
        run["jobs"] = []
    else:
        run["jobs"].append(json.loads(json.dumps(run["jobs"][0])))
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)

    with pytest.raises(ValueError, match="canonical derivation"):
        donor_harness.validate_donor_run_manifest(manifest)


def test_dry_run_rejects_legacy_32_slot_base_config_before_gpu(
    tmp_path: Path, monkeypatch,
) -> None:
    selection = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    value = json.loads(base.read_text(encoding="utf-8"))
    value["argv"].extend(["--slotmem_memory_encoder_slots", "32"])
    _write_json(base, value)
    calls = []
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(
        ValueError,
        match=r"actual='32'.*frozen expected='64'.*64-slot-compatible checkpoint/config",
    ):
        main(
            [
                "dry-run",
                "--selection",
                str(selection),
                "--output",
                str(tmp_path / "run"),
                "--base-inference-args",
                str(base),
                "--platform-manifest",
                str(platform),
            ]
        )

    assert calls == []
    assert not (tmp_path / "run" / "run_manifest.json").exists()


def test_dry_run_uses_last_duplicate_memory_geometry_option(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    value = json.loads(base.read_text(encoding="utf-8"))
    value["argv"][-2:-2] = ["--slotmem_memory_encoder_slots", "63"]
    _write_json(base, value)

    run = build_donor_run_manifest(
        selection_path=selection,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )

    assert len(run["jobs"]) == 3
    frozen = prefix_contract.normalized_frozen_args(
        run["jobs"][0]["prefix_inference_args"]
    )
    assert frozen["slotmem_memory_encoder_slots"] == "64"


def test_dry_run_pins_dump_target_seed_override_to_zero(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    base, platform = _inputs(tmp_path)

    run = build_donor_run_manifest(
        selection_path=selection,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )

    for job in run["jobs"]:
        command = job["commands"]["dump"]["argv"]
        assert command[command.index("--target-seed-override") + 1] == "0"


def test_prefix_failure_stops_later_jobs_and_preserves_complete_logs(
    tmp_path: Path, monkeypatch,
) -> None:
    selection = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write("prefix output\n")
        kwargs["stderr"].write("prefix error\n")
        return subprocess.CompletedProcess(command, 17)

    monkeypatch.setattr("utest.vistory_donor_harness.subprocess.run", fail)

    result = run_stage("prefix", manifest)

    assert len(calls) == 1
    assert set(result) == {"stage", "results"}
    assert result["results"] == [
        {
            "target_event_id": TARGET_IDS[0],
            "stage": "prefix",
            "status": "failed",
            "returncode": 17,
            "stdout": run["jobs"][0]["commands"]["prefix"]["stdout"],
            "stderr": run["jobs"][0]["commands"]["prefix"]["stderr"],
        }
    ]
    assert Path(result["results"][0]["stdout"]).read_text() == "prefix output\n"
    assert Path(result["results"][0]["stderr"]).read_text() == "prefix error\n"


def test_dump_refuses_to_run_before_a_valid_prefix(tmp_path: Path, monkeypatch) -> None:
    selection = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    calls = []
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda *args, **kwargs: calls.append(args) or subprocess.CompletedProcess(args, 0, "", ""),
    )

    with pytest.raises(FileNotFoundError, match="prefix"):
        run_stage("dump", manifest)

    assert calls == []


def test_prefix_success_status_requires_validated_artifacts(tmp_path: Path, monkeypatch) -> None:
    selection = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    calls = []
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda command, **kwargs: calls.append(command)
        or subprocess.CompletedProcess(command, 0, "apparently okay\n", ""),
    )

    result = run_stage("prefix", manifest)

    assert len(calls) == 1
    assert result["results"][0]["status"] == "failed_validation"
    assert "valid prefix artifacts are missing" in result["results"][0]["error"]


def test_dump_success_status_requires_validated_v2_payload_info(tmp_path: Path, monkeypatch) -> None:
    selection = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    for job in run["jobs"]:
        _materialize_prefix(job, platform)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    calls = []
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda command, **kwargs: calls.append(command)
        or subprocess.CompletedProcess(command, 0, "apparently okay\n", ""),
    )

    result = run_stage("dump", manifest)

    assert len(calls) == 1
    assert result["results"][0]["status"] == "failed_validation"
    assert "donor payload artifacts are missing" in result["results"][0]["error"]


def test_resume_skips_three_fully_valid_jobs_without_subprocess(tmp_path: Path, monkeypatch) -> None:
    selection = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    for job in run["jobs"]:
        _materialize_prefix(job, platform)
        _materialize_payload(job)
        _materialize_completion(job, run)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    calls = []
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda *args, **kwargs: calls.append(args),
    )

    result = run_stage("resume", manifest)

    assert calls == []
    assert [row["target_event_id"] for row in result["results"]] == list(TARGET_IDS)
    assert {row["status"] for row in result["results"]} == {"skipped_valid"}


def test_fresh_resume_runs_prefix_then_dump_for_each_job_in_stable_order(
    tmp_path: Path, monkeypatch,
) -> None:
    selection = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    by_prefix = {job["prefix_dir"]: job for job in run["jobs"]}
    by_dump = {job["dump_dir"]: job for job in run["jobs"]}
    calls = []
    real_run = subprocess.run

    def materialize(command, **kwargs):
        if command[0] == "git":
            return real_run(command, **kwargs)
        action = command[3]
        calls.append((action, command))
        if action == "prepare-prefix":
            job = by_prefix[command[command.index("--output") + 1]]
            _materialize_prefix(job, platform)
        else:
            job = by_dump[command[command.index("--output") + 1]]
            _materialize_payload(job)
        return subprocess.CompletedProcess(command, 0, f"{action} output\n", "")

    monkeypatch.setattr("utest.vistory_donor_harness.subprocess.run", materialize)

    result = run_stage("resume", manifest)

    assert [action for action, _ in calls] == ["prepare-prefix", "dump-donor"] * 3
    assert [row["target_event_id"] for row in result["results"]] == list(TARGET_IDS)
    assert {row["status"] for row in result["results"]} == {"completed"}


def test_dump_rejects_existing_invalid_output_without_overwrite_or_subprocess(
    tmp_path: Path, monkeypatch,
) -> None:
    selection = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    for job in run["jobs"]:
        _materialize_prefix(job, platform)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    payload = Path(run["jobs"][0]["donor_payload"])
    payload.write_bytes(b"do-not-overwrite")
    calls = []
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(FileExistsError, match="donor dump output"):
        run_stage("dump", manifest)

    assert calls == []
    assert payload.read_bytes() == b"do-not-overwrite"


def test_dry_run_cli_writes_manifest_without_exposing_a_seed_option(
    tmp_path: Path, capsys,
) -> None:
    selection = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    output = tmp_path / "run"

    status = main(
        [
            "dry-run",
            "--selection",
            str(selection),
            "--output",
            str(output),
            "--base-inference-args",
            str(base),
            "--platform-manifest",
            str(platform),
        ]
    )

    assert status == 0
    written = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert len(written["jobs"]) == 3
    assert "seed" not in capsys.readouterr().out.casefold()
    source = Path("utest/vistory_donor_harness.py").read_text(encoding="utf-8")
    assert "raise SystemExit" not in source
    assert "exit()" not in source


@pytest.mark.parametrize("scope", ["selection", "event_row", "event_manifest", "event_json"])
def test_boolean_donor_seed_is_rejected_at_every_frozen_boundary(
    tmp_path: Path, scope: str,
) -> None:
    selection_path = _selection(tmp_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    row = selection["events"][0]
    manifest_path = selection_path.parent / row["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if scope == "selection":
        selection["donor_seed"] = True
    elif scope == "event_row":
        row["donor_seed"] = True
    elif scope == "event_manifest":
        manifest["donor_seed"] = True
        _write_json(manifest_path, manifest)
        row["manifest_sha256"] = _sha256(manifest_path)
    else:
        event_path = selection_path.parent / manifest["outputs"]["event"]["path"]
        event = json.loads(event_path.read_text(encoding="utf-8"))
        event["donor_seed"] = True
        _write_json(event_path, event)
        manifest["outputs"]["event"]["sha256"] = _sha256(event_path)
        _write_json(manifest_path, manifest)
        row["manifest_sha256"] = _sha256(manifest_path)
    _write_json(selection_path, selection)
    base, platform = _inputs(tmp_path)

    with pytest.raises(ValueError, match="donor_seed"):
        build_donor_run_manifest(
            selection_path=selection_path,
            output_root=tmp_path / "run",
            base_inference_args_path=base,
            platform_manifest_path=platform,
            python_executable=sys.executable,
        )


def test_dump_validates_three_existing_v2_payload_outputs(tmp_path: Path, monkeypatch) -> None:
    selection = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    for job in run["jobs"]:
        _materialize_prefix(job, platform)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    by_dump = {job["dump_dir"]: job for job in run["jobs"]}
    calls = []

    def materialize(command, **kwargs):
        job = by_dump[command[command.index("--output") + 1]]
        calls.append(job["target_event_id"])
        _materialize_payload(job)
        return subprocess.CompletedProcess(command, 0, "dump output\n", "")

    monkeypatch.setattr("utest.vistory_donor_harness.subprocess.run", materialize)

    result = run_stage("dump", manifest)

    assert calls == list(TARGET_IDS)
    assert {row["status"] for row in result["results"]} == {"completed"}
    assert all(Path(job["donor_payload"]).is_file() for job in run["jobs"])
    assert all(Path(job["donor_payload_info"]).is_file() for job in run["jobs"])
    assert all(
        (Path(job["dump_dir"]).parent / "completion.json").is_file()
        for job in run["jobs"]
    )


def test_resume_rejects_invalid_existing_payload_without_subprocess(
    tmp_path: Path, monkeypatch,
) -> None:
    selection = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    for job in run["jobs"]:
        _materialize_prefix(job, platform)
        _materialize_payload(job)
        _materialize_completion(job, run)
    first_info = Path(run["jobs"][0]["donor_payload_info"])
    info = json.loads(first_info.read_text(encoding="utf-8"))
    info["payload_sha256"] = "0" * 64
    _write_json(first_info, info)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    calls = []
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_stage("resume", manifest)

    assert calls == []


def test_resume_rejects_stale_completion_with_wrong_slot_geometry_without_subprocess(
    tmp_path: Path, monkeypatch,
) -> None:
    selection = _exploratory_selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    job = run["jobs"][0]
    _materialize_prefix(job, platform)
    _materialize_payload(job)
    _materialize_completion(job, run)
    _mutate_payload_geometry(job, "wrong_slots")
    completion_path = Path(job["completion"])
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    for field in ("donor_payload", "donor_payload_info", "donor_audit"):
        completion[field]["sha256"] = _sha256(Path(job[field]))
    _write_json(completion_path, completion)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    calls = []
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(ValueError, match="64-slot"):
        run_stage("resume", manifest)

    assert calls == []


def test_selection_requires_task3_event_parent_path_contract(tmp_path: Path) -> None:
    selection_path = _selection(tmp_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    row = selection["events"][0]
    manifest_path = selection_path.parent / row["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    event_path = selection_path.parent / manifest["outputs"]["event"]["path"]
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event.pop("path_resolution")
    event["source_json_path"] = str(
        (selection_path.parent / manifest["outputs"]["story"]["path"]).resolve()
    )
    event["reference_path"] = str(
        (selection_path.parent / manifest["outputs"]["reference"]["path"]).resolve()
    )
    _write_json(event_path, event)
    manifest["outputs"]["event"]["sha256"] = _sha256(event_path)
    _write_json(manifest_path, manifest)
    row["manifest_sha256"] = _sha256(manifest_path)
    _write_json(selection_path, selection)
    base, platform = _inputs(tmp_path)

    with pytest.raises(ValueError, match="event_parent"):
        build_donor_run_manifest(
            selection_path=selection_path,
            output_root=tmp_path / "run",
            base_inference_args_path=base,
            platform_manifest_path=platform,
            python_executable=sys.executable,
        )


def test_completed_run_gate_returns_only_three_fully_valid_jobs(
    tmp_path: Path, capsys,
) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    for job in run["jobs"]:
        _materialize_prefix(job, platform)
        _materialize_payload(job)
        _materialize_completion(job, run)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)

    completed = validate_completed_donor_run(
        manifest, validate_frozen_selection(selection_path)
    )

    assert [job["target_event_id"] for job in completed["jobs"]] == list(TARGET_IDS)
    completion = json.loads(
        Path(run["jobs"][0]["completion"]).read_text(encoding="utf-8")
    )
    assert set(completion) == {
        "schema_version",
        "target_event_id",
        "donor_seed",
        "prefix_snapshot",
        "prefix_contract",
        "donor_payload",
        "donor_payload_info",
        "donor_audit",
        "repository",
        "platform_manifest",
        "dump_runtime_contract",
        "execution",
    }
    assert main(["resume", "--manifest", str(manifest)]) == 0
    assert set(json.loads(capsys.readouterr().out)) == {"stage", "results"}


@pytest.mark.parametrize(
    ("malformation", "message"),
    [
        ("flat", "layerwise"),
        ("empty", "layers must not be empty"),
        ("missing", "layers"),
        ("extra", "layers"),
        ("wrong_slots", "64-slot"),
        ("integer_tensor", "floating point"),
        ("nonfinite", "finite"),
        ("hidden_dim", "hidden dimension"),
    ],
)
def test_completion_rejects_invalid_geometry_before_record_is_written(
    tmp_path: Path, malformation: str, message: str,
) -> None:
    selection_path = _exploratory_selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    job = run["jobs"][0]
    _materialize_prefix(job, platform)
    _materialize_payload(job)
    _mutate_payload_geometry(job, malformation)

    with pytest.raises(ValueError, match=message):
        donor_harness._write_completion(job, run)

    assert not Path(job["completion"]).exists()


def test_completed_run_accepts_one_fully_valid_exploratory_job(
    tmp_path: Path, capsys,
) -> None:
    selection_path = _exploratory_selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    job = run["jobs"][0]
    _materialize_prefix(job, platform)
    _materialize_payload(job)
    _materialize_completion(job, run)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)

    completed = validate_completed_donor_run(
        manifest, validate_frozen_selection(selection_path)
    )

    assert [row["target_event_id"] for row in completed["jobs"]] == [TARGET_IDS[0]]
    completion = json.loads(Path(job["completion"]).read_text(encoding="utf-8"))
    assert completion["protocol_scope"] == EXPLORATORY_SINGLE_EVENT_SCOPE
    assert completion["target_event_ids"] == [TARGET_IDS[0]]
    assert main(["resume", "--manifest", str(manifest)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert set(result) == {"stage", "results"}
    assert result["results"][0]["status"] == "skipped_valid"


@pytest.mark.parametrize("mutation", ["missing", "null", "wrong"])
def test_exploratory_completion_scope_fails_closed(
    tmp_path: Path, mutation: str,
) -> None:
    selection_path = _exploratory_selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    job = run["jobs"][0]
    _materialize_prefix(job, platform)
    _materialize_payload(job)
    _materialize_completion(job, run)
    completion_path = Path(job["completion"])
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        completion.pop("protocol_scope")
        completion.pop("target_event_ids")
    elif mutation == "null":
        completion["protocol_scope"] = None
    else:
        completion["target_event_ids"] = [TARGET_IDS[1]]
    _write_json(completion_path, completion)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)

    with pytest.raises(ValueError, match="donor completion"):
        validate_completed_donor_run(
            manifest, validate_frozen_selection(selection_path)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_id", "unreviewed-candidate"),
        ("manifest_path", "unreviewed/manifest.json"),
    ],
)
def test_completed_run_rejects_tampered_validated_selection_mapping(
    tmp_path: Path, field: str, value: str,
) -> None:
    selection_path = _exploratory_selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    job = run["jobs"][0]
    _materialize_prefix(job, platform)
    _materialize_payload(job)
    _materialize_completion(job, run)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    selection = validate_frozen_selection(selection_path)
    tampered = json.loads(json.dumps(selection))
    tampered["events"][0][field] = value

    with pytest.raises(ValueError, match="canonical selection"):
        validate_completed_donor_run(manifest, tampered)


def test_run_manifest_rejects_non_boolean_dirty_before_subprocess(
    tmp_path: Path, monkeypatch,
) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    run["repository"]["dirty"] = "false"
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    calls = []
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(ValueError, match="repository"):
        run_stage("prefix", manifest)

    assert calls == []


def test_run_manifest_rejects_tampered_command_before_subprocess(
    tmp_path: Path, monkeypatch,
) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    run["jobs"][0]["commands"]["prefix"]["argv"] = [
        sys.executable,
        "-c",
        "print('must-not-run')",
    ]
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    calls = []
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(ValueError, match="canonical"):
        run_stage("prefix", manifest)

    assert calls == []


def test_stage_rejects_changed_or_dirty_repository_before_subprocess(
    tmp_path: Path, monkeypatch,
) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    calls = []
    monkeypatch.setattr(donor_harness, "_git_state", lambda _repo: ("changed", True))
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(ValueError, match="repository"):
        run_stage("prefix", manifest)

    assert calls == []


def test_completed_run_rejects_fake_payload_bytes_even_with_matching_sidecar(
    tmp_path: Path,
) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    for job in run["jobs"]:
        _materialize_prefix(job, platform)
        _materialize_payload(job)
        _materialize_completion(job, run)
    fake = Path(run["jobs"][0]["donor_payload"])
    fake.write_bytes(b"not-a-torch-payload")
    info_path = Path(run["jobs"][0]["donor_payload_info"])
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["payload_sha256"] = _sha256(fake)
    _write_json(info_path, info)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)

    with pytest.raises(ValueError, match="payload artifact"):
        validate_completed_donor_run(
            manifest, validate_frozen_selection(selection_path)
        )


def test_resume_rejects_partial_artifacts_without_completion_or_subprocess(
    tmp_path: Path, monkeypatch,
) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    for job in run["jobs"]:
        _materialize_prefix(job, platform)
        _materialize_payload(job)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    calls = []
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(ValueError, match="partial"):
        run_stage("resume", manifest)

    assert calls == []


def test_concurrent_stage_uses_one_permanent_claim_before_subprocess(
    tmp_path: Path, monkeypatch,
) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    calls = []

    def fail_after_overlap(command, **kwargs):
        calls.append(command)
        time.sleep(0.15)
        return subprocess.CompletedProcess(command, 17, "failed\n", "error\n")

    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run", fail_after_overlap
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_stage, "prefix", manifest) for _ in range(2)]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except FileExistsError:
                outcomes.append("claimed")

    assert len(calls) == 1
    assert "claimed" in outcomes
    assert Path(run["jobs"][0]["commands"]["prefix"]["claim"]).is_file()


def test_module_entrypoint_propagates_nonzero_status_without_system_exit() -> None:
    source = Path("utest/vistory_donor_harness.py").read_text(encoding="utf-8")

    assert 'raise RuntimeError(f"donor harness failed with status {status}")' in source
    assert "raise SystemExit" not in source
    assert "exit()" not in source


@pytest.mark.parametrize(
    "field",
    ["event", "artifact_path", "log_path", "output_root", "python", "extra_job_field"],
)
def test_canonical_manifest_rejects_every_job_or_path_tamper_before_subprocess(
    tmp_path: Path, monkeypatch, field: str,
) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    if field == "event":
        run["jobs"][0]["event"]["event_id"] = "tampered"
    elif field == "artifact_path":
        run["jobs"][0]["prefix_dir"] = str(tmp_path / "outside-prefix")
    elif field == "log_path":
        run["jobs"][0]["commands"]["prefix"]["stdout"] = str(tmp_path / "outside.log")
    elif field == "output_root":
        run["output_root"] = str(tmp_path / "elsewhere")
    elif field == "python":
        run["python"] = "python"
    else:
        run["jobs"][0]["extra"] = "forbidden"
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    calls = []
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(ValueError):
        run_stage("prefix", manifest)

    assert calls == []


@pytest.mark.parametrize("state", ["prefix_only", "payload_only"])
def test_resume_rejects_either_direction_of_partial_state(
    tmp_path: Path, monkeypatch, state: str,
) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    first = run["jobs"][0]
    if state == "prefix_only":
        _materialize_prefix(first, platform)
    else:
        _materialize_payload(first)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    calls = []
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(ValueError, match="partial"):
        run_stage("resume", manifest)

    assert calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_read_hits", 0),
        ("intervention_effective", False),
        ("arm", "wrong"),
        ("seed", True),
        ("seed", 0.0),
        ("target_character", "another-person"),
        ("target_chunk_idx", True),
        ("target_chunk_idx", 1.0),
        ("donor_dumped", "another-payload.pt"),
        ("donor_sha256", "0" * 64),
        ("runtime_contract", {"target_seed": 0}),
    ],
)
def test_completed_run_rejects_tampered_audit_even_when_completion_hash_matches(
    tmp_path: Path, field: str, value: object,
) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    for job in run["jobs"]:
        _materialize_prefix(job, platform)
        _materialize_payload(job)
        _materialize_completion(job, run)
    first = run["jobs"][0]
    audit_path = Path(first["donor_audit"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit[field] = value
    _write_json(audit_path, audit)
    completion_path = Path(first["completion"])
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["donor_audit"]["sha256"] = _sha256(audit_path)
    _write_json(completion_path, completion)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)

    with pytest.raises(ValueError, match="audit"):
        validate_completed_donor_run(
            manifest, validate_frozen_selection(selection_path)
        )


def test_completed_run_rejects_audit_that_copies_prefix_runtime(
    tmp_path: Path,
) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    for job in run["jobs"]:
        _materialize_prefix(job, platform)
        _materialize_payload(job)
        audit_path = Path(job["donor_audit"])
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["runtime_contract"] = _read_runtime_contract(job)
        _write_json(audit_path, audit)
        _materialize_completion(job, run)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)

    with pytest.raises(ValueError, match="runtime"):
        validate_completed_donor_run(
            manifest, validate_frozen_selection(selection_path)
        )


def test_prefix_rejects_existing_artifacts_even_if_claim_was_deleted(
    tmp_path: Path, monkeypatch,
) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    first = run["jobs"][0]
    _materialize_prefix(first, platform)
    for key in ("claim", "stdout", "stderr"):
        Path(first["commands"]["prefix"][key]).unlink(missing_ok=True)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    calls = []
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(FileExistsError, match="prefix output"):
        run_stage("prefix", manifest)

    assert calls == []


def test_completed_run_rejects_deleted_execution_claim(tmp_path: Path) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    for job in run["jobs"]:
        _materialize_prefix(job, platform)
        _materialize_payload(job)
        _materialize_completion(job, run)
    Path(run["jobs"][0]["commands"]["prefix"]["claim"]).unlink(missing_ok=True)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)

    with pytest.raises(FileNotFoundError, match="execution"):
        validate_completed_donor_run(
            manifest, validate_frozen_selection(selection_path)
        )


def test_dump_rejects_deleted_prefix_claim_before_subprocess(
    tmp_path: Path, monkeypatch,
) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    first = run["jobs"][0]
    _materialize_prefix(first, platform)
    Path(first["commands"]["prefix"]["claim"]).unlink()
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    calls = []
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(FileNotFoundError, match="execution"):
        run_stage("dump", manifest)

    assert calls == []


def test_completed_run_rejects_sidecar_shape_tamper_with_updated_completion_hash(
    tmp_path: Path,
) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    for job in run["jobs"]:
        _materialize_prefix(job, platform)
        _materialize_payload(job)
        _materialize_completion(job, run)
    first = run["jobs"][0]
    info_path = Path(first["donor_payload_info"])
    info = json.loads(info_path.read_text(encoding="utf-8"))
    payload_key = info["payload_keys"][0]
    info["payload_slot_shapes"][payload_key]["0"] = [99, 99]
    _write_json(info_path, info)
    completion_path = Path(first["completion"])
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["donor_payload_info"]["sha256"] = _sha256(info_path)
    _write_json(completion_path, completion)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)

    with pytest.raises(ValueError, match="slot shapes"):
        validate_completed_donor_run(
            manifest, validate_frozen_selection(selection_path)
        )


def test_build_rejects_alternate_python_executable(tmp_path: Path) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)

    with pytest.raises(ValueError, match="sys.executable"):
        build_donor_run_manifest(
            selection_path=selection_path,
            output_root=tmp_path / "run",
            base_inference_args_path=base,
            platform_manifest_path=platform,
            python_executable="python",
        )


def test_run_manifest_rejects_float_job_seed_before_subprocess(
    tmp_path: Path, monkeypatch,
) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    run["jobs"][0]["donor_seed"] = 0.0
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    calls = []
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(ValueError, match="canonical"):
        run_stage("prefix", manifest)

    assert calls == []


def test_completed_run_rejects_float_payload_shape_dimensions(tmp_path: Path) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    for job in run["jobs"]:
        _materialize_prefix(job, platform)
        _materialize_payload(job)
        _materialize_completion(job, run)
    first = run["jobs"][0]
    info_path = Path(first["donor_payload_info"])
    info = json.loads(info_path.read_text(encoding="utf-8"))
    payload_key = info["payload_keys"][0]
    info["payload_slot_shapes"][payload_key]["0"] = [64.0, 3.0]
    _write_json(info_path, info)
    completion_path = Path(first["completion"])
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["donor_payload_info"]["sha256"] = _sha256(info_path)
    _write_json(completion_path, completion)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)

    with pytest.raises(ValueError, match="slot shapes"):
        validate_completed_donor_run(
            manifest, validate_frozen_selection(selection_path)
        )


def test_selection_rejects_float_event_chunk_discriminator(tmp_path: Path) -> None:
    selection_path = _selection(tmp_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    row = selection["events"][0]
    manifest_path = selection_path.parent / row["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    event_path = selection_path.parent / manifest["outputs"]["event"]["path"]
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["target_chunk_idx"] = 1.0
    _write_json(event_path, event)
    manifest["outputs"]["event"]["sha256"] = _sha256(event_path)
    _write_json(manifest_path, manifest)
    row["manifest_sha256"] = _sha256(manifest_path)
    _write_json(selection_path, selection)

    with pytest.raises(ValueError, match="target_chunk_idx"):
        validate_frozen_selection(selection_path)


def test_dump_rejects_float_runtime_seed_before_subprocess(
    tmp_path: Path, monkeypatch,
) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    first = run["jobs"][0]
    _materialize_prefix(first, platform)
    contract_path = Path(first["prefix_contract"])
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["runtime_contract"]["target_seed"] = 0.0
    _write_json(contract_path, contract)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    calls = []
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(ValueError, match="runtime"):
        run_stage("dump", manifest)

    assert calls == []


@pytest.mark.parametrize("linked_directory", ["logs", "claims", "prefix", "dump"])
def test_prefix_rejects_write_ancestor_symlink_escape_before_any_write_or_subprocess(
    tmp_path: Path, monkeypatch, linked_directory: str,
) -> None:
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    first = run["jobs"][0]
    job_root = Path(first["prefix_dir"]).parent
    job_root.mkdir(parents=True)
    outside = tmp_path / f"outside-{linked_directory}"
    outside.mkdir()
    linked = job_root / linked_directory
    if os.name == "nt":
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(linked), str(outside)],
            capture_output=True,
            text=True,
            check=False,
        )
        if created.returncode != 0:
            pytest.skip(f"directory links unavailable: {created.stderr}")
    else:
        linked.symlink_to(outside, target_is_directory=True)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)
    calls = []
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(ValueError, match="symlink"):
        run_stage("prefix", manifest)

    assert calls == []
    assert list(outside.iterdir()) == []


def test_logged_subprocess_uses_file_streams_without_mock_only_capture_branch() -> None:
    source = Path("utest/vistory_donor_harness.py").read_text(encoding="utf-8")

    assert "completed.stdout" not in source
    assert "completed.stderr" not in source


def test_offload_mode_is_frozen_for_completion_and_checked_only_before_execution(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("SLOTMEM_OFFLOAD_MODELS", "1")
    selection_path = _selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    assert run["runtime_environment"] == {"slotmem_offload_models": True}
    assert all(job["dump_runtime_contract"]["target_seed"] == 0 for job in run["jobs"])
    assert all(
        "offload_models" in job["dump_runtime_contract"]["frozen_args"]
        for job in run["jobs"]
    )
    for job in run["jobs"]:
        _materialize_prefix(job, platform)
        _materialize_payload(job)
        _materialize_completion(job, run)
    manifest = tmp_path / "run" / "run_manifest.json"
    _write_json(manifest, run)

    monkeypatch.setenv("SLOTMEM_OFFLOAD_MODELS", "0")
    completed = validate_completed_donor_run(
        manifest, validate_frozen_selection(selection_path)
    )
    assert completed["runtime_environment"]["slotmem_offload_models"] is True
    calls = []
    monkeypatch.setattr(
        "utest.vistory_donor_harness.subprocess.run",
        lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(ValueError, match="offload environment"):
        run_stage("resume", manifest)

    assert calls == []
