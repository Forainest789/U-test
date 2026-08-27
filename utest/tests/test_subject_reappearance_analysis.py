import json
import hashlib
import copy
import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
import tools.analyze_subject_reappearance as analysis_module
import tools.prepare_vistory_cids_inputs as prepare_module

from tools.prepare_vistory_cids_inputs import (
    OFFICIAL_CIDS_CONFIG,
    OFFICIAL_CIDS_CONTRACT_SHA256,
    OFFICIAL_CONFIG_DOCUMENT,
    OFFICIAL_METHOD,
    OFFICIAL_MODE,
    OFFICIAL_TIMESTAMP,
    _config_yaml_bytes,
    prepare_cids_inputs,
    uniform_frame_indices,
)
from tools.analyze_subject_reappearance import (
    analyze_inputs,
    classify_block,
    summarize_event,
    write_analysis_outputs,
)
from utest.memory_utility import REQUIRED_OUTCOMES, label_event
from utest.subject_reappearance_harness import EVALUATOR_COMMIT, FROZEN_EVENTS, FULL_ARMS


RULES = Path(__file__).parents[1] / "events" / "vistorybench_reappearance_quality_rules.json"


def _outcomes(identity: float, anatomy: float) -> dict[str, float]:
    return {
        name: identity if name == "C_id" else anatomy if name == "Q_anatomy" else 0.9
        for name in REQUIRED_OUTCOMES
    }


def _passing_blocks(qstar: float | None) -> list[dict]:
    scores = {
        "full_correct": 0.80,
        "subject_only": 0.79,
        "random_only": 0.70,
        "drop_subject": 0.60,
        "drop_random": 0.77,
        "wrong_subject": 0.68,
    }
    outcomes = {
        arm: _outcomes(score, 0.9)
        for arm, score in scores.items()
    }
    return [
        {
            "event_id": "event", "seed": seed, "scores": dict(scores),
            "outcomes": copy.deepcopy(outcomes), "qstar": qstar,
        }
        for seed in (0, 1, 2)
    ]


def test_eight_frame_indices_include_ends_without_duplicates() -> None:
    assert uniform_frame_indices(frame_count=81, count=8) == [0, 11, 23, 34, 46, 57, 69, 80]


def test_primary_causal_gate_requires_sufficiency_necessity_and_specificity() -> None:
    scores = {
        "full_correct": 0.80,
        "subject_only": 0.79,
        "random_only": 0.70,
        "drop_subject": 0.60,
        "drop_random": 0.77,
        "wrong_subject": 0.68,
    }

    assert classify_block(scores, repeat_floor=0.01) == {
        "sufficiency": True,
        "necessity": True,
        "specificity": True,
        "passed": True,
    }


def test_quality_failure_is_separate_from_identity_failure() -> None:
    rules = json.loads(RULES.read_text(encoding="utf-8"))
    full = _outcomes(identity=0.8, anatomy=0.9)
    arm = _outcomes(identity=0.9, anatomy=0.85)

    label, reasons = label_event(
        {name: arm[name] - full[name] for name in REQUIRED_OUTCOMES},
        arm,
        delta_id=rules["delta_id"],
        quality_margins=rules["quality_margins"],
        dynamic_degree_floor=rules["dynamic_degree_floor"],
    )

    assert label == "harmful"
    assert reasons == ["quality_breach:Q_anatomy"]


def test_qstar_is_descriptive_and_cannot_change_passed() -> None:
    rules = json.loads(RULES.read_text(encoding="utf-8"))
    without_qstar = summarize_event(_passing_blocks(qstar=None), rules=rules, repeat_floor=0.01)
    with_qstar = summarize_event(_passing_blocks(qstar=-10.0), rules=rules, repeat_floor=0.01)

    assert without_qstar["passed"] == with_qstar["passed"] is True
    assert without_qstar == with_qstar


def test_quality_gate_checks_every_causal_arm_not_only_subject_only() -> None:
    rules = json.loads(RULES.read_text(encoding="utf-8"))
    blocks = _passing_blocks(qstar=None)
    blocks[0]["outcomes"]["wrong_subject"] = _outcomes(identity=0.68, anatomy=0.1)

    event = summarize_event(blocks, rules=rules, repeat_floor=0.01)

    assert event["blocks"][0]["identity_gate"]["passed"] is True
    assert event["blocks"][0]["quality_gate"]["passed"] is False
    assert event["blocks"][0]["quality_gate"]["per_arm"]["wrong_subject"]["reasons"] == [
        "quality_breach:Q_anatomy"
    ]
    assert event["supporting_seed_count"] == 2


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frame_export_materializes_official_derived_eight_shot_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = []
    for event_id, frozen in FROZEN_EVENTS.items():
        for seed in (0, 1, 2):
            block = tmp_path / "run" / event_id / f"seed_{seed}"
            reference_dir = tmp_path / "ViStory" / str(frozen["story_id"]) / "image" / "subject_key"
            reference_dir.mkdir(parents=True, exist_ok=True)
            (reference_dir / "00.jpg").write_bytes(b"official reference")
            event = {
                "event_id": event_id, "story_id": str(frozen["story_id"]),
                "character_name": frozen["character_name"], "source_chunk_idx": 0,
                "target_chunk_idx": int(frozen["target_shot"]) - int(frozen["source_shot"]),
                "reference_path": str(reference_dir / "00.jpg"),
            }
            event_path = _write_json(block / "event.json", event)
            rows.append({
                "event_id": event_id, "seed": seed, "block_dir": str(block),
                "event_json": str(event_path), "event_json_sha256": _sha(event_path),
            })
    passed = rows[0]
    passed_event = json.loads(Path(passed["event_json"]).read_text())
    prefix = Path(passed["block_dir"]) / "prefix" / "prefix_generation"
    prefix.mkdir(parents=True)
    iio.imwrite(
        prefix / "chunk_000.mp4",
        np.arange(9 * 16 * 16 * 3, dtype=np.uint8).reshape(9, 16, 16, 3), fps=4,
    )
    for arm in FULL_ARMS:
        arm_dir = Path(passed["block_dir"]) / "full" / "arms" / arm
        arm_dir.mkdir(parents=True)
        iio.imwrite(
            arm_dir / f"chunk_{passed_event['target_chunk_idx']:03d}.mp4",
            np.zeros((81, 16, 16, 3), dtype=np.uint8), fps=4,
        )
    validation = _write_json(Path(passed["block_dir"]) / "full" / "validation.json", {"status": "passed"})
    run = _write_json(tmp_path / "run" / "run_manifest.json", {
        "schema_version": 1, "task_id": "vistorybench_subject_reappearance_v1", "blocks": rows,
    })
    official = {
        "Characters": {"subject_key": {"name": passed_event["character_name"], "tag": "realistic_human"}},
        "Shots": [{"index": int(FROZEN_EVENTS[passed_event["event_id"]]["target_shot"]), "script": "target"}],
    }
    official_path = Path(passed_event["reference_path"]).parents[2] / "story.json"
    _write_json(official_path, official)
    monkeypatch.setattr("tools.prepare_vistory_cids_inputs._load_run_manifest", lambda path: json.loads(path.read_text()))

    def completed(row: dict) -> dict:
        if (row["event_id"], row["seed"]) != (rows[0]["event_id"], rows[0]["seed"]):
            raise FileNotFoundError("full rollout missing")
        return {
            "validation_path": str(validation), "validation_sha256": _sha(validation),
            "validation": {"status": "passed"}, "event": passed_event, "contract": {},
        }

    monkeypatch.setattr("tools.prepare_vistory_cids_inputs._validate_completed_block", completed)
    monkeypatch.setattr(
        "tools.prepare_vistory_cids_inputs._official_story",
        lambda event: (official_path, official, _sha(official_path)),
    )

    manifest = prepare_cids_inputs(run, tmp_path / "prepared")

    row = manifest["blocks"][0]
    assert manifest["protocol_boundary"]["identity_endpoint"] == "official_evaluator_derived"
    assert manifest["protocol_boundary"]["source_continuity"] == "local_diagnostic"
    assert [frame["frame_index"] for frame in row["arms"]["full_correct"]["target_frames"]] == [0, 11, 23, 34, 46, 57, 69, 80]
    arm = row["arms"]["full_correct"]
    assert Path(arm["official_output_story_dir"]).parts[-6:] == (
        "official_outputs", OFFICIAL_METHOD, OFFICIAL_MODE, "en", OFFICIAL_TIMESTAMP,
        arm["derived_story_id"],
    )
    derived = json.loads(Path(arm["derived_story_json"]).read_text())
    assert [shot["index"] for shot in derived["Shots"]] == list(range(8))
    assert arm["character_key"] == "subject_key"
    assert _sha(Path(arm["reference_images"][0]["derived_path"])) == arm["reference_images"][0]["sha256"]
    assert all(_sha(Path(frame["path"])) == frame["sha256"] for frame in row["source_frames"])
    assert sum(block["status"] == "incomplete" for block in manifest["blocks"]) == 8
    assert manifest["official_cids"]["config"] == OFFICIAL_CIDS_CONFIG
    assert manifest["official_cids"]["config_document"] == OFFICIAL_CONFIG_DOCUMENT
    config_file = Path(manifest["official_cids"]["config_file"]["path"])
    assert config_file.read_bytes() == _config_yaml_bytes()
    assert _sha(config_file) == manifest["official_cids"]["config_file"]["sha256"]
    assert manifest["official_cids"]["detector_matcher_contract_sha256"] == OFFICIAL_CIDS_CONTRACT_SHA256
    result_language = (
        tmp_path / "prepared" / "cids_results" / OFFICIAL_METHOD / OFFICIAL_MODE / "en"
    )
    assert [path.name for path in result_language.iterdir()] == [OFFICIAL_TIMESTAMP]
    assert manifest["official_cids"]["cli_argv"][-3:] == [
        "--result_path", str(tmp_path / "prepared" / "cids_results"), "--resume"
    ]
    assert "--config" in manifest["official_cids"]["cli_argv"]
    assert "--split" in manifest["official_cids"]["cli_argv"]
    with pytest.raises(FileExistsError):
        prepare_cids_inputs(run, tmp_path / "prepared")


def test_prepare_isolates_materialization_failure_and_cleans_partial_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = []
    for event_id, frozen in FROZEN_EVENTS.items():
        for seed in (0, 1, 2):
            event = _write_json(tmp_path / "events" / event_id / f"{seed}.json", {
                "event_id": event_id, "story_id": frozen["story_id"],
                "character_name": frozen["character_name"],
            })
            rows.append({"event_id": event_id, "seed": seed, "event_json": str(event)})
    run = _write_json(tmp_path / "run.json", {"blocks": rows})
    monkeypatch.setattr(prepare_module, "_load_run_manifest", lambda path: json.loads(path.read_text()))
    validation = _write_json(tmp_path / "validation.json", {"status": "passed"})
    monkeypatch.setattr(prepare_module, "_validate_completed_block", lambda row: {
        "validation_path": str(validation), "validation_sha256": _sha(validation),
    })
    first = (rows[0]["event_id"], rows[0]["seed"])

    def materialize(
        row: dict, event: dict, base: dict, staging: Path, completed: dict,
    ) -> dict:
        if (row["event_id"], row["seed"]) == first:
            partial = staging / "source_frames" / row["event_id"] / f"seed_{row['seed']}" / "shots"
            partial.mkdir(parents=True)
            (partial / "shot_00.png").write_bytes(b"partial")
            derived = staging / "derived_dataset" / "ViStory" / "900000"
            derived.mkdir(parents=True)
            (derived / "story.json").write_bytes(b"partial")
            raise ValueError("bad target video")
        return {**base, "status": "passed", "arms": {}}

    monkeypatch.setattr(prepare_module, "_materialize_block", materialize)
    output = tmp_path / "prepared"
    manifest = prepare_cids_inputs(run, output)

    assert len(manifest["blocks"]) == 9
    assert manifest["blocks"][0]["status"] == "failed"
    assert manifest["blocks"][1]["status"] == "passed"
    assert not (output / "source_frames" / first[0] / "seed_0").exists()
    assert not (output / "derived_dataset" / "ViStory" / "900000").exists()


def test_prepare_materialization_failure_is_accepted_and_preserved_by_analyzer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = []
    for event_id, frozen in FROZEN_EVENTS.items():
        for seed in (0, 1, 2):
            event = _write_json(tmp_path / "events" / event_id / f"{seed}.json", {
                "event_id": event_id, "story_id": frozen["story_id"],
                "character_name": frozen["character_name"],
            })
            rows.append({
                "event_id": event_id, "seed": seed, "event_json": str(event),
                "event_json_sha256": _sha(event), "block_dir": str(tmp_path / "blocks" / event_id / str(seed)),
            })
    run = _write_json(tmp_path / "run.json", {"blocks": rows})
    validation = _write_json(tmp_path / "validation.json", {"status": "passed"})
    first = (rows[0]["event_id"], rows[0]["seed"])

    def completed(row: dict) -> dict:
        if (row["event_id"], row["seed"]) != first:
            raise FileNotFoundError("Task6 block missing")
        return {"validation_path": str(validation), "validation_sha256": _sha(validation)}

    monkeypatch.setattr(prepare_module, "_load_run_manifest", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(prepare_module, "_validate_completed_block", completed)
    monkeypatch.setattr(
        prepare_module, "_materialize_block",
        lambda row, event, base, staging, checked: (_ for _ in ()).throw(ValueError("decode failed")),
    )
    output = tmp_path / "prepared"
    manifest = prepare_cids_inputs(run, output)
    failed = manifest["blocks"][0]
    assert failed["failure_stage"] == "task7_materialization"
    assert failed["failure_provenance"]["full_validation"]["sha256"] == _sha(validation)

    monkeypatch.setattr(analysis_module, "_load_run_manifest", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(analysis_module, "_validate_completed_block", completed)
    monkeypatch.setattr(
        analysis_module, "_materialize_block",
        lambda row, event, base, staging, checked: (_ for _ in ()).throw(ValueError("decode failed")),
    )
    manifest_path = output / "cids_input_manifest.json"
    input_sha = _sha(manifest_path)
    run_sha = _sha(run)
    cids_items = _write_jsonl(
        output / "cids_results" / OFFICIAL_METHOD / OFFICIAL_MODE / "en"
        / OFFICIAL_TIMESTAMP / "metrics" / "cids" / "items.jsonl", [],
    )
    provenance = {"run_manifest_sha256": run_sha, "cids_input_manifest_sha256": input_sha}
    continuity = _write_json(tmp_path / "continuity.json", {
        "schema_version": 1, "provenance": {
            **provenance, "kind": "local_target_to_source_crop_continuity",
            "evaluator_commit": EVALUATOR_COMMIT,
            "detector_matcher_contract_sha256": OFFICIAL_CIDS_CONTRACT_SHA256,
        }, "records": [],
    })
    quality = _write_json(tmp_path / "quality.json", {
        "schema_version": 1,
        "provenance": {**provenance, "kind": "vbench_and_local_quality"}, "records": [],
    })
    prompt = _write_json(tmp_path / "prompt.json", {
        "schema_version": 1,
        "provenance": {**provenance, "kind": "frozen_prompt_alignment"}, "records": [],
    })
    records, report = analyze_inputs(
        run, manifest_path, cids_items, continuity, quality, prompt, RULES,
        repeat_floor=0.01, n_boot=20,
    )
    retained = next(row for row in records if (row["event_id"], row["seed"]) == first)
    assert retained["status"] == "failed"
    assert retained["reason"] == "decode failed"
    assert report["failed_blocks"]


def test_analyzer_rejects_passed_block_forged_as_task7_materialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _analysis_inputs(tmp_path)
    _patch_task6_validation(monkeypatch)
    manifest = json.loads(paths[1].read_text(encoding="utf-8"))
    original = manifest["blocks"][0]
    event_path = Path(original["event_json"])
    completed = analysis_module._validate_completed_block(
        json.loads(paths[0].read_text(encoding="utf-8"))["blocks"][0]
    )
    base = {
        "event_id": original["event_id"], "story_id": original["story_id"],
        "character_name": original["character_name"], "seed": original["seed"],
    }
    manifest["blocks"][0] = prepare_module._failure_row(
        base, ValueError("forged failure"), stage="task7_materialization",
        event_path=event_path, event_sha=_sha(event_path), completed=completed,
    )
    _write_json(paths[1], manifest)
    monkeypatch.setattr(
        analysis_module, "_materialize_block",
        lambda row, event, base, staging, checked: {**base, "status": "passed"},
    )

    with pytest.raises(ValueError, match="now materializes successfully"):
        analyze_inputs(*paths, repeat_floor=0.01, n_boot=20)


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _analysis_inputs(
    tmp_path: Path, failed: set[tuple[str, int]] | None = None,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    failed = failed or set()
    run_path = tmp_path / "run.json"
    blocks, input_blocks = [], []
    cids_rows, continuity_rows, quality_rows, prompt_rows = [], [], [], []
    arm_scores = {
        "full_correct": 0.80, "no_memory": 0.60, "zero_path": 0.60,
        "subject_only": 0.79, "random_only": 0.70, "drop_subject": 0.60,
        "drop_random": 0.77, "wrong_subject": 0.68,
    }
    cids_root = tmp_path / "cids_adapter"
    for event_number, (event_id, frozen) in enumerate(FROZEN_EVENTS.items()):
        story_id = str(frozen["story_id"])
        for seed in range(3):
            block_dir = tmp_path / "blocks" / event_id / f"seed_{seed}"
            event = {
                "event_id": event_id, "story_id": story_id,
                "character_name": frozen["character_name"], "source_chunk_idx": 0,
                "target_chunk_idx": int(frozen["target_shot"]) - int(frozen["source_shot"]),
            }
            event_path = _write_json(block_dir / "event.json", event)
            validation = _write_json(block_dir / "full" / "validation.json", {"status": "passed"})
            blocks.append({
                "event_id": event_id, "seed": seed, "block_dir": str(block_dir),
                "event_json": str(event_path), "event_json_sha256": _sha(event_path),
                "qstar": {"status": "not_available", "reason": "independent_teacher_missing"},
            })
            if (event_id, seed) in failed:
                failure_provenance = {
                    "event_json": {"path": str(event_path), "sha256": _sha(event_path)},
                }
                input_blocks.append({
                    "event_id": event_id, "story_id": story_id,
                    "character_name": frozen["character_name"], "seed": seed,
                    "status": "failed", "reason": "frozen Task6 validation failed",
                    "failure_stage": "task6_validation", "failure_type": "ValueError",
                    "failure_provenance": failure_provenance,
                    "failure_provenance_sha256": prepare_module._canonical_sha(failure_provenance),
                })
                continue
            source_file = cids_root / "source" / event_id / f"seed_{seed}" / "source.png"
            source_file.parent.mkdir(parents=True, exist_ok=True)
            source_file.write_bytes(b"source")
            source_frames = [{
                "frame_index": index, "path": str(source_file), "sha256": _sha(source_file),
                "source_video": str(source_file), "source_video_sha256": _sha(source_file),
                "event_id": event_id, "seed": seed, "arm": "shared_prefix",
            } for index in range(8)]
            official_story = block_dir / "official_story.json"
            target_shot = {
                "index": int(frozen["target_shot"]),
                "Characters Appearing": {"en": ["subject_key"]},
                "script": "frozen target",
            }
            official_value = {
                "Characters": {"subject_key": {"name": frozen["character_name"]}},
                "Shots": [target_shot],
            }
            _write_json(official_story, official_value)
            reference_file = block_dir / "image" / "subject_key" / "00.jpg"
            reference_file.parent.mkdir(parents=True, exist_ok=True)
            reference_file.write_bytes(b"reference")
            arm_inputs = {}
            for arm, score in arm_scores.items():
                arm_index = list(FULL_ARMS).index(arm)
                derived_id = str(900000 + event_number * 100 + seed * 10 + arm_index)
                output_story = (
                    cids_root / "official_outputs" / OFFICIAL_METHOD / OFFICIAL_MODE / "en"
                    / OFFICIAL_TIMESTAMP / derived_id
                )
                derived_story = cids_root / "derived_dataset" / "ViStory" / derived_id / "story.json"
                _write_json(
                    derived_story,
                    prepare_module._derived_story(official_value, target_shot, derived_id),
                )
                derived_reference = derived_story.parent / "image" / "subject_key" / "00.jpg"
                derived_reference.parent.mkdir(parents=True)
                derived_reference.write_bytes(reference_file.read_bytes())
                target_frames = []
                for frame_index in range(8):
                    frame_file = output_story / "shots" / f"shot_{frame_index:02d}.png"
                    frame_file.parent.mkdir(parents=True, exist_ok=True)
                    frame_file.write_bytes(f"{arm}-{frame_index}".encode())
                    frame = {
                        "frame_index": frame_index, "path": str(frame_file),
                        "sha256": _sha(frame_file), "source_video": str(frame_file),
                        "source_video_sha256": _sha(frame_file),
                        "event_id": event_id, "seed": seed, "arm": arm,
                        "derived_shot_index": frame_index,
                    }
                    target_frames.append(frame)
                    cids_rows.append({
                        "metric": {"name": "cids", "submetric": "cross_sim"},
                        "scope": {
                            "level": "item", "story_id": derived_id,
                            "shot_index": frame_index, "character_key": "subject_key",
                        },
                        "value": score, "unit": "cosine_similarity", "status": "complete",
                        "extras": {"box": None},
                    })
                    continuity_rows.append({
                        "event_id": event_id, "seed": seed, "arm": arm,
                        "derived_shot_index": frame_index, "similarity": score - 0.01,
                    })
                arm_inputs[arm] = {
                    "derived_story_id": derived_id, "character_key": "subject_key",
                    "original_story_id": story_id,
                    "original_target_shot": int(frozen["target_shot"]),
                    "derived_story_json": str(derived_story),
                    "derived_story_sha256": _sha(derived_story),
                    "official_output_story_dir": str(output_story),
                    "reference_images": [{
                        "character_key": "subject_key",
                        "official_path": str(reference_file),
                        "derived_path": str(derived_reference), "sha256": _sha(reference_file),
                    }],
                    "target_frames": target_frames,
                }
                quality_rows.append({
                    "event_id": event_id, "seed": seed, "arm": arm,
                    "background_consistency": 0.9, "motion_smoothness": 0.9,
                    "dynamic_degree": 0.5, "temporal_flickering": 0.9,
                    "boundary_smoothness": 0.9, "human_anatomy": 0.9,
                    "non_target_consistency": 0.9,
                })
                prompt_rows.append({"event_id": event_id, "seed": seed, "arm": arm, "prompt_alignment": 0.9})
            input_blocks.append({
                "event_id": event_id, "story_id": story_id,
                "character_name": frozen["character_name"],
                "seed": seed, "status": "passed",
                "event_json": str(event_path), "event_json_sha256": _sha(event_path),
                "official_story": {"path": str(official_story), "sha256": _sha(official_story)},
                "full_validation": str(validation), "full_validation_sha256": _sha(validation),
                "source_frames": source_frames, "arms": arm_inputs,
            })
    run = {"schema_version": 1, "task_id": "vistorybench_subject_reappearance_v1", "blocks": blocks}
    _write_json(run_path, run)
    config_path = cids_root / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(_config_yaml_bytes())
    result_root = cids_root / "cids_results"
    cids_manifest = _write_json(cids_root / "cids_input_manifest.json", {
        "schema_version": 2, "task_id": run["task_id"], "run_manifest": str(run_path),
        "run_manifest_sha256": _sha(run_path), "evaluator_commit": EVALUATOR_COMMIT,
        "protocol_boundary": {"identity_endpoint": "official_evaluator_derived", "source_continuity": "local_diagnostic"},
        "official_cids": {
            "config": OFFICIAL_CIDS_CONFIG,
            "config_document": OFFICIAL_CONFIG_DOCUMENT,
            "config_file": {"path": str(config_path), "sha256": _sha(config_path)},
            "detector_matcher_contract_sha256": OFFICIAL_CIDS_CONTRACT_SHA256,
            "method": OFFICIAL_METHOD, "mode": OFFICIAL_MODE, "language": "en",
            "timestamp": OFFICIAL_TIMESTAMP,
            "dataset_path": str(cids_root / "derived_dataset"),
            "outputs_path": str(cids_root / "official_outputs"),
            "result_path": str(result_root),
            "items_relative_path": f"{OFFICIAL_METHOD}/{OFFICIAL_MODE}/en/{OFFICIAL_TIMESTAMP}/metrics/cids/items.jsonl",
            "cli_argv": [
                "python", "bench_run.py", "--config", str(config_path),
                "--method", OFFICIAL_METHOD, "--mode", OFFICIAL_MODE,
                "--language", "en", "--split", "full", "--metrics", "cids",
                "--timestamp", OFFICIAL_TIMESTAMP, "--dataset_path", str(cids_root / "derived_dataset"),
                "--outputs_path", str(cids_root / "official_outputs"),
                "--result_path", str(result_root), "--resume",
            ],
        },
        "blocks": input_blocks,
    })
    provenance = {"run_manifest_sha256": _sha(run_path)}
    cids = _write_jsonl(
        result_root / OFFICIAL_METHOD / OFFICIAL_MODE / "en" / OFFICIAL_TIMESTAMP
        / "metrics" / "cids" / "items.jsonl",
        cids_rows,
    )
    continuity = _write_json(tmp_path / "continuity.json", {
        "schema_version": 1,
        "provenance": {
            **provenance, "kind": "local_target_to_source_crop_continuity",
            "cids_input_manifest_sha256": _sha(cids_manifest),
            "evaluator_commit": EVALUATOR_COMMIT,
            "detector_matcher_contract_sha256": OFFICIAL_CIDS_CONTRACT_SHA256,
        },
        "records": continuity_rows,
    })
    quality = _write_json(tmp_path / "quality.json", {
        "schema_version": 1, "provenance": {
            **provenance, "kind": "vbench_and_local_quality",
            "cids_input_manifest_sha256": _sha(cids_manifest),
        }, "records": quality_rows,
    })
    prompt = _write_json(tmp_path / "prompt.json", {
        "schema_version": 1, "provenance": {
            **provenance, "kind": "frozen_prompt_alignment",
            "cids_input_manifest_sha256": _sha(cids_manifest),
        }, "records": prompt_rows,
    })
    return run_path, cids_manifest, cids, continuity, quality, prompt, RULES


def _patch_task6_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        analysis_module, "_load_run_manifest", lambda path: json.loads(path.read_text(encoding="utf-8")),
    )
    frozen = copy.deepcopy(FROZEN_EVENTS)
    monkeypatch.setattr(analysis_module, "FROZEN_EVENTS", frozen)

    def completed(row: dict) -> dict:
        validation = Path(row["block_dir"]) / "full" / "validation.json"
        official_story = Path(row["block_dir"]) / "official_story.json"
        frozen[row["event_id"]]["story_sha256"] = _sha(official_story)
        return {
            "validation_path": str(validation.resolve()),
            "validation_sha256": _sha(validation),
            "validation": json.loads(validation.read_text(encoding="utf-8")),
        }

    monkeypatch.setattr(analysis_module, "_validate_completed_block", completed)


def test_analysis_aggregates_seeds_before_event_bootstrap_and_writes_atomic_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _analysis_inputs(tmp_path)
    _patch_task6_validation(monkeypatch)

    records, report = analyze_inputs(*paths, repeat_floor=0.01, n_boot=200)

    assert len(records) == 9
    assert report["passed"] is True
    assert report["passed_event_count"] == 3
    assert len(report["event_values"]) == 3
    assert report["cluster_bootstrap"]["sufficiency"]["n_event_clusters"] == 3
    assert all(event["supporting_seed_count"] == 3 for event in report["events"])
    assert all(record["qstar"]["status"] == "not_available" for record in records)
    assert report["analysis_contract"]["quality_rules"]["content"] == json.loads(RULES.read_text())
    assert report["analysis_contract"]["parameters"] == {
        "repeat_floor": 0.01, "n_boot": 200, "bootstrap_seed": 0,
    }
    output = tmp_path / "analysis"
    write_analysis_outputs(records, report, output)
    assert {path.name for path in output.iterdir()} == {
        "subject_reappearance_records.jsonl",
        "subject_reappearance_report.json",
        "subject_reappearance_table.csv",
    }
    with pytest.raises(FileExistsError):
        write_analysis_outputs(records, report, output)

    frame = Path(json.loads(paths[1].read_text(encoding="utf-8"))["blocks"][0]["arms"]["full_correct"]["target_frames"][0]["path"])
    frame.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        analyze_inputs(*paths, repeat_floor=0.01, n_boot=20)


def test_failed_block_is_retained_and_event_needs_two_valid_supporting_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_event = next(iter(FROZEN_EVENTS))
    paths = _analysis_inputs(tmp_path, failed={(first_event, 0)})
    _patch_task6_validation(monkeypatch)
    original = analysis_module._validate_completed_block

    def validation(row: dict) -> dict:
        if (row["event_id"], int(row["seed"])) == (first_event, 0):
            raise ValueError("frozen Task6 validation failed")
        return original(row)

    monkeypatch.setattr(analysis_module, "_validate_completed_block", validation)

    records, report = analyze_inputs(*paths, repeat_floor=0.01, n_boot=20)

    failed = next(row for row in records if row["event_id"] == first_event and row["seed"] == 0)
    event = next(row for row in report["events"] if row["event_id"] == first_event)
    assert failed["status"] == "failed"
    assert event["valid_seed_count"] == event["supporting_seed_count"] == 2
    assert event["passed"] is True
    assert len(report["failed_blocks"]) == 1


def test_single_valid_seed_event_is_descriptive_but_excluded_from_cluster_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_event = next(iter(FROZEN_EVENTS))
    paths = _analysis_inputs(tmp_path, failed={(first_event, 0), (first_event, 1)})
    _patch_task6_validation(monkeypatch)
    original = analysis_module._validate_completed_block

    def validation(row: dict) -> dict:
        if row["event_id"] == first_event and int(row["seed"]) in {0, 1}:
            raise ValueError("frozen Task6 validation failed")
        return original(row)

    monkeypatch.setattr(analysis_module, "_validate_completed_block", validation)
    _, report = analyze_inputs(*paths, repeat_floor=0.01, n_boot=20)

    event = next(row for row in report["events"] if row["event_id"] == first_event)
    assert event["valid_seed_count"] == 1
    assert event["mean_scores"]  # retained as descriptive evidence
    assert first_event not in {row["event_id"] for row in report["event_values"]}
    assert report["cluster_bootstrap"]["sufficiency"]["n_event_clusters"] == 2


def test_qstar_is_loaded_only_after_frozen_classification_and_stays_descriptive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _analysis_inputs(tmp_path)
    _patch_task6_validation(monkeypatch)
    calls = []

    def qstar(row: dict) -> dict:
        calls.append((row["event_id"], int(row["seed"])))
        return {"status": "available", "label": "independent-teacher-relative Q*", "mean": -10.0}

    monkeypatch.setattr(analysis_module, "_qstar_for_block", qstar)
    records, report = analyze_inputs(*paths, repeat_floor=0.01, n_boot=20)
    first = records[0]
    assert report["passed"] is True
    assert first["qstar"]["label"] == "independent-teacher-relative Q*"
    assert first["qstar"]["mean"] == -10.0
    assert len(calls) == 9
    assert report["classification_snapshot_sha256_before_qstar"]


def test_qstar_loader_reuses_task6_full_report_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = tmp_path / "block"
    report_path = _write_json(block / "qstar" / "qstar_report.json", {"placeholder": True})
    records_path = _write_jsonl(block / "qstar" / "qstar_records.jsonl", [{"x": 1}])
    row = {"block_dir": str(block), "qstar": {"status": "available"}}
    monkeypatch.setattr(analysis_module, "_validated_prefix_contract", lambda value: {"prefix": "ok"})
    called = []

    def validate(value: dict, contract: dict) -> dict:
        called.append((value, contract))
        return {"cells": [{"timestep_index": 0, "qstar": -3.0, "arm_deltas": {"x": -1.0}}]}

    monkeypatch.setattr(analysis_module, "_validate_qstar_report", validate)
    result = analysis_module._qstar_for_block(row)
    assert called == [(row, {"prefix": "ok"})]
    assert result["mean"] == -3.0
    assert result["report"]["sha256"] == _sha(report_path)
    assert result["records"]["sha256"] == _sha(records_path)


def test_conflicting_metric_alias_and_unknown_official_join_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _analysis_inputs(tmp_path)
    _patch_task6_validation(monkeypatch)
    quality = json.loads(paths[4].read_text(encoding="utf-8"))
    quality["records"][0]["Q_bg"] = 0.1
    _write_json(paths[4], quality)
    with pytest.raises(ValueError, match="conflicting aliases"):
        analyze_inputs(*paths, repeat_floor=0.01, n_boot=20)


def test_official_jsonl_ignores_non_cids_items_but_rejects_malformed_cids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _analysis_inputs(tmp_path)
    _patch_task6_validation(monkeypatch)
    rows = [json.loads(line) for line in paths[2].read_text().splitlines()]
    rows.append({
        "metric": {"name": "prompt_align", "submetric": "single_action"},
        "scope": {"level": "item", "story_id": "external"},
        "value": 0.5, "status": "complete", "extras": {},
    })
    _write_jsonl(paths[2], rows)
    records, report = analyze_inputs(*paths, repeat_floor=0.01, n_boot=20)
    assert len(records) == 9
    assert report["passed"] is True

    rows[0]["unit"] = "not_cosine"
    _write_jsonl(paths[2], rows)
    with pytest.raises(ValueError, match="official CIDS item schema mismatch"):
        analyze_inputs(*paths, repeat_floor=0.01, n_boot=20)


def test_analysis_rejects_shape_valid_but_unregistered_quality_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = list(_analysis_inputs(tmp_path))
    _patch_task6_validation(monkeypatch)
    rules = json.loads(RULES.read_text(encoding="utf-8"))
    rules["quality_margins"]["Q_anatomy"] = 0.99
    paths[-1] = _write_json(tmp_path / "tampered_rules.json", rules)

    with pytest.raises(ValueError, match="not the preregistered contract"):
        analyze_inputs(*paths, repeat_floor=0.01, n_boot=20)


def test_analysis_rebinds_target_character_and_full_derived_story_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _analysis_inputs(tmp_path)
    _patch_task6_validation(monkeypatch)
    manifest = json.loads(paths[1].read_text(encoding="utf-8"))
    arm = manifest["blocks"][0]["arms"]["full_correct"]
    derived_path = Path(arm["derived_story_json"])
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    derived["Shots"][0]["script"] = "tampered but rehashed"
    _write_json(derived_path, derived)
    arm["derived_story_sha256"] = _sha(derived_path)
    _write_json(paths[1], manifest)

    input_sha = _sha(paths[1])
    for metric_path in paths[3:6]:
        payload = json.loads(metric_path.read_text(encoding="utf-8"))
        payload["provenance"]["cids_input_manifest_sha256"] = input_sha
        _write_json(metric_path, payload)
    with pytest.raises(ValueError, match="frozen target-shot clone"):
        analyze_inputs(*paths, repeat_floor=0.01, n_boot=20)


@pytest.mark.parametrize("tamper", ["drop", "target_frame_substitution"])
def test_analysis_requires_complete_exact_official_reference_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str,
) -> None:
    paths = _analysis_inputs(tmp_path)
    _patch_task6_validation(monkeypatch)
    manifest = json.loads(paths[1].read_text(encoding="utf-8"))
    arm = manifest["blocks"][0]["arms"]["full_correct"]
    if tamper == "drop":
        arm["reference_images"] = []
    else:
        frame = arm["target_frames"][0]
        arm["reference_images"][0]["derived_path"] = frame["path"]
        arm["reference_images"][0]["sha256"] = frame["sha256"]
    _write_json(paths[1], manifest)

    with pytest.raises(ValueError, match="reference"):
        analyze_inputs(*paths, repeat_floor=0.01, n_boot=20)


def test_official_cids_item_requires_box_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _analysis_inputs(tmp_path)
    _patch_task6_validation(monkeypatch)
    rows = [json.loads(line) for line in paths[2].read_text().splitlines()]
    del rows[0]["extras"]["box"]
    _write_jsonl(paths[2], rows)
    with pytest.raises(ValueError, match="official CIDS item schema mismatch"):
        analyze_inputs(*paths, repeat_floor=0.01, n_boot=20)


def test_official_cids_value_must_be_a_cosine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _analysis_inputs(tmp_path)
    _patch_task6_validation(monkeypatch)
    rows = [json.loads(line) for line in paths[2].read_text().splitlines()]
    rows[0]["value"] = 1.01
    _write_jsonl(paths[2], rows)
    with pytest.raises(ValueError, match="cosine range"):
        analyze_inputs(*paths, repeat_floor=0.01, n_boot=20)


@pytest.mark.parametrize(
    ("metric_index", "field", "value"),
    [(5, "prompt_alignment", 50.0), (4, "human_anatomy", 1.01)],
)
def test_prompt_and_quality_inputs_must_be_normalized_unit_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    metric_index: int, field: str, value: float,
) -> None:
    paths = _analysis_inputs(tmp_path)
    _patch_task6_validation(monkeypatch)
    payload = json.loads(paths[metric_index].read_text(encoding="utf-8"))
    payload["records"][0][field] = value
    _write_json(paths[metric_index], payload)
    with pytest.raises(ValueError, match=r"normalized to \[0,1\]"):
        analyze_inputs(*paths, repeat_floor=0.01, n_boot=20)

    paths = _analysis_inputs(tmp_path / "join")
    _patch_task6_validation(monkeypatch)
    rows = [json.loads(line) for line in paths[2].read_text().splitlines()]
    rows[0]["scope"]["story_id"] = "123"
    _write_jsonl(paths[2], rows)
    with pytest.raises(ValueError, match="unknown derived story"):
        analyze_inputs(*paths, repeat_floor=0.01, n_boot=20)


@pytest.mark.parametrize("script", ["prepare_vistory_cids_inputs.py", "analyze_subject_reappearance.py"])
def test_cli_is_directly_executable_from_repository_root(script: str) -> None:
    root = Path(__file__).parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "tools" / script), "--help"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
