#!/usr/bin/env python3
"""Aggregate the frozen ViStoryBench subject-reappearance causal endpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from statistics import median
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prepare_vistory_cids_inputs import (  # noqa: E402
    OFFICIAL_CIDS_CONFIG,
    OFFICIAL_CIDS_CONTRACT_SHA256,
    OFFICIAL_CONFIG_DOCUMENT,
    OFFICIAL_LANGUAGE,
    OFFICIAL_METHOD,
    OFFICIAL_MODE,
    OFFICIAL_TIMESTAMP,
    _character_key,
    _config_yaml_bytes,
    _derived_story,
    _failure_row,
    _materialize_block,
    _publish_lock,
    _target_shot,
    _validate_completed_block,
)
from utest.bootstrap import cluster_bootstrap_mean_ci  # noqa: E402
from utest.memory_utility import REQUIRED_OUTCOMES, label_event  # noqa: E402
from utest.subject_reappearance_harness import (  # noqa: E402
    EVALUATOR_COMMIT,
    FROZEN_EVENTS,
    FULL_ARMS,
    _load_run_manifest,
    _validate_qstar_report,
    _validated_prefix_contract,
)

CAUSAL_ARMS = (
    "full_correct", "subject_only", "random_only", "drop_subject", "drop_random",
    "wrong_subject",
)
QUALITY_ALIASES = {
    "Q_bg": ("Q_bg", "background_consistency"),
    "Q_motion_smoothness": ("Q_motion_smoothness", "motion_smoothness"),
    "Q_motion_dynamic_degree": ("Q_motion_dynamic_degree", "dynamic_degree"),
    "Q_flicker": ("Q_flicker", "temporal_flickering", "temporal_flicker"),
    "Q_boundary": ("Q_boundary", "boundary_smoothness"),
    "Q_anatomy": ("Q_anatomy", "human_anatomy"),
    "Q_non_target": ("Q_non_target", "non_target_consistency"),
}
PREREGISTERED_QUALITY_RULES_SHA256 = "5f88ccafe9878f82c7df5201af7cd333bae12a1d4b15ca15cbc4eef356ef5696"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> tuple[dict, str]:
    data = path.resolve().read_bytes()
    value = json.loads(data.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value, hashlib.sha256(data).hexdigest()


def _read_jsonl(path: Path) -> tuple[list[dict], str]:
    data = path.resolve().read_bytes()
    try:
        rows = [json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL: {path}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("JSONL rows must be objects")
    return rows, hashlib.sha256(data).hexdigest()


def _finite(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _alias_value(record: Mapping, aliases: Sequence[str], name: str) -> float:
    found = [(alias, _finite(record[alias], name)) for alias in aliases if alias in record]
    if not found:
        raise ValueError(f"missing metric {name}")
    if any(value != found[0][1] for _, value in found[1:]):
        raise ValueError(f"conflicting aliases for {name}: {[alias for alias, _ in found]}")
    return found[0][1]


def _unit_interval(value: float, name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be normalized to [0,1]")
    return value


def classify_block(scores: Mapping[str, float], repeat_floor: float) -> dict[str, bool]:
    """Apply the preregistered sufficiency, necessity, and specificity contrasts."""
    missing = sorted(set(CAUSAL_ARMS) - set(scores))
    if missing:
        raise ValueError(f"invalid causal scores; missing={missing}")
    values = [_finite(scores[name], name) for name in CAUSAL_ARMS]
    floor = _finite(repeat_floor, "repeat_floor")
    if floor < 0:
        raise ValueError("repeat_floor must be non-negative")
    sufficiency = values[1] - values[2] > floor
    necessity = values[4] - values[3] > floor
    specificity = values[1] - values[5] > floor
    return {
        "sufficiency": sufficiency,
        "necessity": necessity,
        "specificity": specificity,
        "passed": sufficiency and necessity and specificity,
    }


def _quality_gate(outcomes: Mapping[str, Mapping[str, float]], rules: Mapping) -> dict:
    try:
        full = {name: _finite(outcomes["full_correct"][name], name) for name in REQUIRED_OUTCOMES}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("complete full_correct outcomes are required") from exc
    per_arm = {}
    for arm in CAUSAL_ARMS:
        try:
            values = {name: _finite(outcomes[arm][name], f"{arm}:{name}") for name in REQUIRED_OUTCOMES}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"complete {arm} outcomes are required") from exc
        label, reasons = label_event(
            {name: values[name] - full[name] for name in REQUIRED_OUTCOMES},
            values,
            delta_id=float(rules["delta_id"]),
            quality_margins=rules["quality_margins"],
            dynamic_degree_floor=float(rules["dynamic_degree_floor"]),
        )
        quality_reasons = [
            reason for reason in reasons
            if reason.startswith("quality_breach:") or reason == "dynamic_degree_below_floor"
        ]
        per_arm[arm] = {"passed": not quality_reasons, "label": label, "reasons": quality_reasons}
    return {
        "passed": all(row["passed"] for row in per_arm.values()),
        "per_arm": per_arm,
        "reasons": [
            f"{arm}:{reason}" for arm, row in per_arm.items() for reason in row["reasons"]
        ],
    }


def summarize_event(
    blocks: list[Mapping], *, rules: Mapping, repeat_floor: float,
    failed_blocks: Sequence[Mapping] = (), event_id: str | None = None,
) -> dict:
    """Aggregate eligible seeds inside one event before assigning its verdict."""
    if not blocks:
        return {
            "event_id": str(event_id), "valid_seed_count": 0, "supporting_seed_count": 0,
            "passed": False, "status": "insufficient_valid_seeds",
            "failed_blocks": list(failed_blocks), "blocks": [],
        }
    event_ids = {str(block.get("event_id", "")) for block in blocks}
    seeds = [int(block["seed"]) for block in blocks]
    if len(event_ids) != 1 or len(seeds) != len(set(seeds)):
        raise ValueError("event blocks must have one event_id and unique seeds")
    block_rows = []
    for block in sorted(blocks, key=lambda row: int(row["seed"])):
        identity = classify_block(block["scores"], repeat_floor)
        quality = _quality_gate(block["outcomes"], rules)
        block_rows.append({
            "seed": int(block["seed"]), "identity_gate": identity, "quality_gate": quality,
            "passed": identity["passed"] and quality["passed"],
        })
    mean_scores = {
        arm: sum(float(block["scores"][arm]) for block in blocks) / len(blocks)
        for arm in CAUSAL_ARMS
    }
    identity_gate = classify_block(mean_scores, repeat_floor)
    supporting = sum(row["passed"] for row in block_rows)
    quality_support = sum(row["quality_gate"]["passed"] for row in block_rows)
    valid_seed_count = len(block_rows)
    return {
        "event_id": event_ids.pop(),
        "valid_seed_count": valid_seed_count,
        "supporting_seed_count": supporting,
        "mean_scores": mean_scores,
        "identity_gate": identity_gate,
        "quality_gate": {"passed": quality_support >= 2, "supporting_seed_count": quality_support},
        "passed": valid_seed_count >= 2 and identity_gate["passed"] and supporting >= 2,
        "status": "complete" if valid_seed_count >= 2 else "insufficient_valid_seeds",
        "failed_blocks": list(failed_blocks),
        "blocks": block_rows,
    }


def _validate_rules(rules: Mapping) -> None:
    margins = rules.get("quality_margins")
    expected = {name for name in REQUIRED_OUTCOMES if name not in {"C_id", "Q_motion_dynamic_degree"}}
    if (
        rules.get("schema_version") != 1
        or not isinstance(margins, Mapping)
        or set(margins) != expected
        or _finite(rules.get("delta_id"), "delta_id") < 0
        or _finite(rules.get("dynamic_degree_floor"), "dynamic_degree_floor") < 0
        or any(_finite(value, name) < 0 for name, value in margins.items())
    ):
        raise ValueError("invalid frozen quality rules")


def _validate_artifact(
    path_value: object, expected_sha: object, label: str, cache: dict[Path, str]
) -> None:
    path = Path(str(path_value)).resolve()
    if not path.is_file():
        raise ValueError(f"{label} SHA-256 mismatch")
    actual = cache.get(path)
    if actual is None:
        actual = cache[path] = _sha256(path)
    if actual != str(expected_sha).casefold():
        raise ValueError(f"{label} SHA-256 mismatch")


def _require_envelope(
    payload: Mapping, *, kind: str, run_sha: str, input_sha: str,
    contract_sha: str | None = None,
) -> None:
    provenance = payload.get("provenance")
    expected = {
        "kind": kind,
        "run_manifest_sha256": run_sha,
        "cids_input_manifest_sha256": input_sha,
    }
    if contract_sha is not None:
        expected.update({
            "evaluator_commit": EVALUATOR_COMMIT,
            "detector_matcher_contract_sha256": contract_sha,
        })
    if (
        payload.get("schema_version") != 1
        or not isinstance(provenance, Mapping)
        or any(str(provenance.get(key)).casefold() != str(value).casefold() for key, value in expected.items())
    ):
        raise ValueError(f"{kind} provenance mismatch")


def _metric_key(row: Mapping) -> tuple[str, int, str]:
    return str(row["event_id"]), int(row["seed"]), str(row["arm"])


def _indexed(rows: object) -> dict:
    if not isinstance(rows, list):
        raise ValueError("metric records must be a list")
    output = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("metric record must be an object")
        key = _metric_key(row)
        if key in output:
            raise ValueError(f"duplicate metric record: {key}")
        output[key] = row
    return output


def _validate_input_manifest(
    manifest: Mapping, manifest_path: Path, run: Mapping, run_sha: str
) -> tuple[dict, list[dict]]:
    root = manifest_path.resolve().parent
    official = manifest.get("official_cids", {})
    expected_cli = [
        "python", "bench_run.py", "--config", str((root / "config.yaml").resolve()),
        "--method", OFFICIAL_METHOD, "--mode", OFFICIAL_MODE,
        "--language", "en", "--split", "full", "--metrics", "cids",
        "--timestamp", OFFICIAL_TIMESTAMP,
        "--dataset_path", str((root / "derived_dataset").resolve()),
        "--outputs_path", str((root / "official_outputs").resolve()),
        "--result_path", str((root / "cids_results").resolve()), "--resume",
    ]
    if (
        manifest.get("schema_version") != 2
        or manifest.get("task_id") != "vistorybench_subject_reappearance_v1"
        or Path(str(manifest.get("run_manifest", ""))).resolve() != Path(str(run["_path"])).resolve()
        or str(manifest.get("run_manifest_sha256", "")).casefold() != run_sha
        or manifest.get("evaluator_commit") != EVALUATOR_COMMIT
        or manifest.get("protocol_boundary", {}).get("identity_endpoint") != "official_evaluator_derived"
        or manifest.get("protocol_boundary", {}).get("source_continuity") != "local_diagnostic"
        or official.get("config") != OFFICIAL_CIDS_CONFIG
        or official.get("config_document") != OFFICIAL_CONFIG_DOCUMENT
        or official.get("detector_matcher_contract_sha256") != OFFICIAL_CIDS_CONTRACT_SHA256
        or official.get("method") != OFFICIAL_METHOD
        or official.get("mode") != OFFICIAL_MODE
        or official.get("language") != OFFICIAL_LANGUAGE
        or official.get("timestamp") != OFFICIAL_TIMESTAMP
        or Path(str(official.get("dataset_path", ""))).resolve() != (root / "derived_dataset").resolve()
        or Path(str(official.get("outputs_path", ""))).resolve() != (root / "official_outputs").resolve()
        or Path(str(official.get("result_path", ""))).resolve() != (root / "cids_results").resolve()
        or official.get("items_relative_path") != f"{OFFICIAL_METHOD}/{OFFICIAL_MODE}/en/{OFFICIAL_TIMESTAMP}/metrics/cids/items.jsonl"
        or official.get("cli_argv") != expected_cli
    ):
        raise ValueError("CIDS input manifest provenance/config mismatch")
    config_file = official.get("config_file", {})
    expected_config = (root / "config.yaml").resolve()
    if (
        Path(str(config_file.get("path", ""))).resolve() != expected_config
        or not expected_config.is_relative_to(root)
        or expected_config.read_bytes() != _config_yaml_bytes()
        or str(config_file.get("sha256", "")).casefold() != _sha256(expected_config)
    ):
        raise ValueError("frozen official CIDS config file mismatch")
    run_blocks = {(row["event_id"], int(row["seed"])): row for row in run["blocks"]}
    blocks, ineligible, derived_ids = {}, [], set()
    hashes: dict[Path, str] = {}
    rows = list(manifest.get("blocks", ()))
    if len(rows) != 9:
        raise ValueError("CIDS input manifest must retain all nine blocks")
    for block in rows:
        key = str(block["event_id"]), int(block["seed"])
        if key not in run_blocks or key in blocks or key[0] not in FROZEN_EVENTS:
            raise ValueError("CIDS block identity mismatch")
        run_row = run_blocks[key]
        if block.get("status") != "passed":
            provenance = block.get("failure_provenance")
            if (
                block.get("status") not in {"failed", "incomplete"}
                or not str(block.get("reason", ""))
                or block.get("failure_stage") not in {"task6_validation", "task7_materialization"}
                or not str(block.get("failure_type", ""))
                or not isinstance(provenance, Mapping)
                or str(block.get("failure_provenance_sha256", "")).casefold() != _canonical_sha(provenance)
            ):
                raise ValueError("ineligible block needs status and reason")
            event_provenance = provenance.get("event_json", {})
            if Path(str(event_provenance.get("path", ""))).resolve() != Path(str(run_row["event_json"])).resolve():
                raise ValueError("failure event provenance path mismatch")
            _validate_artifact(
                event_provenance.get("path"), event_provenance.get("sha256"),
                "failure event JSON", hashes,
            )
            if block["failure_stage"] == "task7_materialization":
                completed = _validate_completed_block(run_row)
                validation = provenance.get("full_validation", {})
                if (
                    Path(str(validation.get("path", ""))).resolve()
                    != Path(str(completed["validation_path"])).resolve()
                    or str(validation.get("sha256", "")).casefold() != completed["validation_sha256"]
                ):
                    raise ValueError("Task7 failure validation provenance mismatch")
                _validate_artifact(
                    validation.get("path"), validation.get("sha256"),
                    "Task7 failure validation", hashes,
                )
                event_path = Path(str(run_row["event_json"])).resolve()
                event, event_sha = _read_json(event_path)
                base = {
                    "event_id": str(run_row["event_id"]),
                    "story_id": str(event["story_id"]),
                    "character_name": str(event["character_name"]),
                    "seed": int(run_row["seed"]),
                }
                with tempfile.TemporaryDirectory(prefix="subject-reappearance-replay-") as directory:
                    try:
                        _materialize_block(
                            run_row, event, base, Path(directory).resolve(), completed,
                        )
                    except Exception as exc:
                        expected_failure = _failure_row(
                            base, exc, stage="task7_materialization",
                            event_path=event_path, event_sha=event_sha, completed=completed,
                        )
                    else:
                        raise ValueError(
                            "Task7-ineligible block now materializes successfully; regenerate CIDS inputs"
                        )
                failure_fields = (
                    "event_id", "story_id", "character_name", "seed", "status", "reason",
                    "failure_stage", "failure_type", "failure_provenance",
                    "failure_provenance_sha256",
                )
                if any(block.get(field) != expected_failure[field] for field in failure_fields):
                    raise ValueError("Task7 materialization failure replay mismatch")
            else:
                try:
                    _validate_completed_block(run_row)
                except (FileNotFoundError, KeyError, TypeError, ValueError):
                    pass
                else:
                    raise ValueError("Task6-ineligible block is now complete; regenerate CIDS inputs")
            ineligible.append(dict(block))
            continue
        completed = _validate_completed_block(run_row)
        expected_validation = Path(str(run_row["block_dir"])).resolve() / "full" / "validation.json"
        if (
            Path(str(block["full_validation"])).resolve() != expected_validation
            or str(block["full_validation_sha256"]).casefold() != completed["validation_sha256"]
        ):
            raise ValueError("full validation path/SHA mismatch")
        if Path(str(block["event_json"])).resolve() != Path(str(run_row["event_json"])).resolve():
            raise ValueError("run/CIDS event path mismatch")
        _validate_artifact(block["event_json"], block["event_json_sha256"], "event JSON", hashes)
        _validate_artifact(block["official_story"]["path"], block["official_story"]["sha256"], "official story", hashes)
        event, _ = _read_json(Path(str(block["event_json"])))
        frozen = FROZEN_EVENTS[key[0]]
        official_story, official_sha = _read_json(Path(str(block["official_story"]["path"])))
        if (
            str(event.get("event_id")) != key[0]
            or str(event.get("story_id")) != str(frozen["story_id"])
            or str(event.get("character_name")) != str(frozen["character_name"])
            or str(block.get("story_id")) != str(event["story_id"])
            or str(block.get("character_name")) != str(event["character_name"])
            or official_sha != str(frozen["story_sha256"]).casefold()
        ):
            raise ValueError("passed block is not bound to its frozen event/official story")
        character_key = _character_key(official_story, str(event["character_name"]))
        target_shot = _target_shot(official_story, key[0])
        official_references = {}
        official_image_root = Path(str(block["official_story"]["path"])).resolve().parent / "image"
        for official_key in official_story["Characters"]:
            reference_root = official_image_root / str(official_key)
            if not reference_root.is_dir():
                continue
            for reference_path in sorted(path for path in reference_root.rglob("*") if path.is_file()):
                reference_key = str(official_key), reference_path.relative_to(reference_root).as_posix()
                official_references[reference_key] = (reference_path.resolve(), _sha256(reference_path))
        if not any(key_[0] == character_key for key_ in official_references):
            raise ValueError("frozen target character has no official references")
        source_frames = block.get("source_frames", ())
        if len(source_frames) != 8 or len({int(row["frame_index"]) for row in source_frames}) != 8:
            raise ValueError("passed block needs eight distinct source frames")
        for frame in source_frames:
            _validate_artifact(frame["path"], frame["sha256"], "source frame", hashes)
            _validate_artifact(frame["source_video"], frame["source_video_sha256"], "source video", hashes)
        arms = block.get("arms", {})
        if set(arms) != set(FULL_ARMS):
            raise ValueError("passed block needs all eight arms")
        for arm, arm_row in arms.items():
            derived = str(arm_row["derived_story_id"])
            if not derived.isdecimal() or derived in derived_ids:
                raise ValueError("derived_story_id must be unique and numeric")
            derived_ids.add(derived)
            expected_output = (
                root / "official_outputs" / OFFICIAL_METHOD / OFFICIAL_MODE / OFFICIAL_LANGUAGE
                / OFFICIAL_TIMESTAMP / derived
            ).resolve()
            expected_story = (root / "derived_dataset" / "ViStory" / derived / "story.json").resolve()
            if (
                Path(str(arm_row["official_output_story_dir"])).resolve() != expected_output
                or Path(str(arm_row["derived_story_json"])).resolve() != expected_story
                or not expected_output.is_relative_to(root)
                or not expected_story.is_relative_to(root)
            ):
                raise ValueError("derived CIDS path escapes or violates official hierarchy")
            _validate_artifact(expected_story, arm_row["derived_story_sha256"], "derived story", hashes)
            story, _ = _read_json(expected_story)
            if (
                str(arm_row.get("character_key")) != character_key
                or str(arm_row.get("original_story_id")) != str(event["story_id"])
                or int(arm_row.get("original_target_shot", -1)) != int(frozen["target_shot"])
                or story != _derived_story(official_story, target_shot, derived)
            ):
                raise ValueError("derived story is not the frozen target-shot clone")
            references = arm_row.get("reference_images", ())
            indexed_references = {}
            for reference in references:
                official_path = Path(str(reference.get("official_path"))).resolve()
                reference_key = None
                for candidate, (expected_path_, _) in official_references.items():
                    if official_path == expected_path_:
                        reference_key = candidate
                        break
                if reference_key is None or reference_key in indexed_references:
                    raise ValueError("derived reference set differs from frozen official references")
                expected_official, expected_sha = official_references[reference_key]
                expected_derived = (
                    expected_story.parent / "image" / reference_key[0]
                    / Path(reference_key[1])
                ).resolve()
                if (
                    str(reference.get("character_key")) != reference_key[0]
                    or official_path != expected_official
                    or Path(str(reference.get("derived_path"))).resolve() != expected_derived
                    or str(reference.get("sha256", "")).casefold() != expected_sha
                ):
                    raise ValueError("derived reference mirror mismatch")
                _validate_artifact(reference["official_path"], reference["sha256"], "official reference", hashes)
                _validate_artifact(reference["derived_path"], reference["sha256"], "derived reference", hashes)
                if not Path(str(reference["derived_path"])).resolve().is_relative_to(root):
                    raise ValueError("derived reference escapes CIDS input root")
                indexed_references[reference_key] = reference
            if set(indexed_references) != set(official_references):
                raise ValueError("derived reference set differs from frozen official references")
            frames = arm_row.get("target_frames", ())
            if (
                len(frames) != 8
                or [int(frame["derived_shot_index"]) for frame in frames] != list(range(8))
                or len({int(frame["frame_index"]) for frame in frames}) != 8
            ):
                raise ValueError("derived arm needs eight ordered distinct frames")
            for frame in frames:
                _validate_artifact(frame["path"], frame["sha256"], "target frame", hashes)
                _validate_artifact(frame["source_video"], frame["source_video_sha256"], "target video", hashes)
                if not Path(str(frame["path"])).resolve().is_relative_to(expected_output):
                    raise ValueError("target frame escapes derived official output story")
        blocks[key] = block
    if set(run_blocks) != set(blocks) | {(row["event_id"], int(row["seed"])) for row in ineligible}:
        raise ValueError("CIDS manifest does not cover the frozen nine-block matrix")
    return blocks, ineligible


def _official_cids_scores(blocks: Mapping, items: Sequence[Mapping]) -> dict:
    expected = {}
    story_ids = set()
    for (event_id, seed), block in blocks.items():
        for arm, row in block["arms"].items():
            story = str(row["derived_story_id"])
            story_ids.add(story)
            for shot_index in range(8):
                expected[story, shot_index, str(row["character_key"])] = (event_id, seed, arm)
    values = defaultdict(list)
    seen = set()
    for item in items:
        metric, scope = item.get("metric"), item.get("scope")
        if isinstance(metric, Mapping) and metric.get("name") != "cids":
            continue
        if (
            not isinstance(metric, Mapping)
            or metric.get("name") != "cids"
            or metric.get("submetric") != "cross_sim"
            or not isinstance(scope, Mapping)
            or scope.get("level") != "item"
            or item.get("unit") != "cosine_similarity"
            or item.get("status") != "complete"
            or not isinstance(item.get("extras"), Mapping)
            or "box" not in item.get("extras", {})
        ):
            raise ValueError("official CIDS item schema mismatch")
        story = str(scope.get("story_id"))
        if story not in story_ids:
            raise ValueError("official CIDS items contain an unknown derived story")
        key = story, int(scope.get("shot_index", -1)), str(scope.get("character_key"))
        if key not in expected:
            continue  # official evaluator may emit other characters from the cloned shot
        if key in seen:
            raise ValueError("duplicate official CIDS target item")
        seen.add(key)
        value = _finite(item.get("value"), "CIDS cross_sim")
        if not -1.0 <= value <= 1.0:
            raise ValueError("CIDS cross_sim must be in cosine range [-1,1]")
        values[expected[key]].append(value)
    if seen != set(expected):
        raise ValueError("official CIDS target items are incomplete")
    return {key: float(median(scores)) for key, scores in values.items()}


def _continuity_scores(blocks: Mapping, payload: Mapping) -> dict:
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ValueError("continuity records must be a list")
    expected, indexed = {}, {}
    for (event_id, seed), block in blocks.items():
        for arm, arm_row in block["arms"].items():
            for frame in arm_row["target_frames"]:
                key = event_id, seed, arm, int(frame["derived_shot_index"])
                expected[key] = frame
    for row in rows:
        key = str(row["event_id"]), int(row["seed"]), str(row["arm"]), int(row["derived_shot_index"])
        if key in indexed:
            raise ValueError("duplicate continuity record")
        indexed[key] = row
    if set(indexed) != set(expected):
        raise ValueError("continuity records do not exactly match eligible derived shots")
    by_block = defaultdict(list)
    for key, row in indexed.items():
        by_block[key[:3]].append(_alias_value(row, ("similarity", "score"), "continuity"))
    return {key: float(median(values)) for key, values in by_block.items()}


def _normalize_quality(record: Mapping) -> dict[str, float]:
    return {
        canonical: _unit_interval(_alias_value(record, aliases, canonical), canonical)
        for canonical, aliases in QUALITY_ALIASES.items()
    }


def _event_contrasts(event: Mapping) -> dict[str, float] | None:
    scores = event.get("mean_scores")
    if not isinstance(scores, Mapping):
        return None
    return {
        "sufficiency": scores["subject_only"] - scores["random_only"],
        "necessity": scores["drop_random"] - scores["drop_subject"],
        "specificity": scores["subject_only"] - scores["wrong_subject"],
    }


def _qstar_for_block(row: Mapping) -> dict:
    report_path = Path(str(row["block_dir"])) / "qstar" / "qstar_report.json"
    if not report_path.is_file():
        if row.get("qstar", {}).get("status") == "available":
            raise ValueError("frozen independent teacher is missing its Q* report")
        return {"status": "not_available", "reason": "independent_teacher_missing"}
    contract = _validated_prefix_contract(row)
    report = _validate_qstar_report(row, contract)
    records_path = report_path.with_name("qstar_records.jsonl")
    cells = [{
        "timestep_index": int(cell["timestep_index"]),
        "qstar": float(cell["qstar"]),
        "arm_deltas": {name: float(value) for name, value in cell["arm_deltas"].items()},
    } for cell in report["cells"]]
    return {
        "status": "available",
        "label": "independent-teacher-relative Q*",
        "mean": sum(cell["qstar"] for cell in cells) / len(cells),
        "cells": cells,
        "report": {"path": str(report_path.resolve()), "sha256": _sha256(report_path)},
        "records": {"path": str(records_path.resolve()), "sha256": _sha256(records_path)},
    }


def analyze_inputs(
    run_manifest: Path,
    cids_input_manifest: Path,
    cids_items: Path,
    continuity_results: Path,
    quality_results: Path,
    prompt_results: Path,
    quality_rules: Path,
    *,
    repeat_floor: float,
    n_boot: int = 10000,
    bootstrap_seed: int = 0,
) -> tuple[list[dict], dict]:
    """Validate inputs, freeze decoded classifications, then append descriptive Q*."""
    rules, rules_sha = _read_json(quality_rules)
    _validate_rules(rules)
    if _canonical_sha(rules) != PREREGISTERED_QUALITY_RULES_SHA256:
        raise ValueError("quality rules are not the preregistered contract")
    floor = _finite(repeat_floor, "repeat_floor")
    if floor < 0 or n_boot <= 0:
        raise ValueError("repeat_floor/n_boot are invalid")
    run = _load_run_manifest(run_manifest.resolve())
    run = {**run, "_path": str(run_manifest.resolve())}
    run_sha = _sha256(run_manifest.resolve())
    input_manifest, input_sha = _read_json(cids_input_manifest)
    passed_blocks, ineligible = _validate_input_manifest(
        input_manifest, cids_input_manifest, run, run_sha
    )
    official = input_manifest["official_cids"]
    expected_items = (
        Path(str(official["result_path"])) / str(official["items_relative_path"])
    ).resolve()
    if cids_items.resolve() != expected_items or not expected_items.is_relative_to(cids_input_manifest.resolve().parent):
        raise ValueError("official CIDS items path mismatch")
    cids_rows, cids_sha = _read_jsonl(cids_items)
    continuity, continuity_sha = _read_json(continuity_results)
    quality, quality_sha = _read_json(quality_results)
    prompt, prompt_sha = _read_json(prompt_results)
    _require_envelope(
        continuity, kind="local_target_to_source_crop_continuity", run_sha=run_sha,
        input_sha=input_sha, contract_sha=OFFICIAL_CIDS_CONTRACT_SHA256,
    )
    _require_envelope(
        quality, kind="vbench_and_local_quality", run_sha=run_sha, input_sha=input_sha,
    )
    _require_envelope(
        prompt, kind="frozen_prompt_alignment", run_sha=run_sha, input_sha=input_sha,
    )
    cids_scores = _official_cids_scores(passed_blocks, cids_rows)
    continuity_scores = _continuity_scores(passed_blocks, continuity)
    quality_rows, prompt_rows = _indexed(quality.get("records")), _indexed(prompt.get("records"))
    expected = {(event, seed, arm) for event, seed in passed_blocks for arm in FULL_ARMS}
    if set(quality_rows) != expected or set(prompt_rows) != expected:
        raise ValueError("quality/prompt records do not exactly match eligible blocks")
    inputs_contract = {
        "cids_items": {"path": str(cids_items.resolve()), "sha256": cids_sha},
        "continuity_results": {"path": str(continuity_results.resolve()), "sha256": continuity_sha},
        "quality_results": {"path": str(quality_results.resolve()), "sha256": quality_sha},
        "prompt_results": {"path": str(prompt_results.resolve()), "sha256": prompt_sha},
        "quality_rules": {
            "path": str(quality_rules.resolve()), "sha256": rules_sha, "content": rules,
        },
        "parameters": {
            "repeat_floor": floor, "n_boot": int(n_boot), "bootstrap_seed": int(bootstrap_seed),
        },
    }
    run_blocks = {(row["event_id"], int(row["seed"])): row for row in run["blocks"]}
    records = []
    for event_id in FROZEN_EVENTS:
        for seed in (0, 1, 2):
            key = event_id, seed
            if key not in passed_blocks:
                failed = next(row for row in ineligible if (row["event_id"], int(row["seed"])) == key)
                records.append({
                    "event_id": event_id, "story_id": failed["story_id"], "seed": seed,
                    "status": failed["status"], "reason": failed["reason"],
                    "failure_stage": failed["failure_stage"],
                    "failure_type": failed["failure_type"],
                    "failure_provenance": failed["failure_provenance"],
                    "failure_provenance_sha256": failed["failure_provenance_sha256"],
                    "analysis_contract": inputs_contract,
                })
                continue
            block = passed_blocks[key]
            scores, local, outcomes = {}, {}, {}
            for arm in FULL_ARMS:
                metric_key = event_id, seed, arm
                scores[arm] = cids_scores[metric_key]
                local[arm] = continuity_scores[metric_key]
                outcomes[arm] = {
                    "C_id": scores[arm],
                    "A_prompt": _unit_interval(_alias_value(
                        prompt_rows[metric_key], ("A_prompt", "prompt_alignment"), "A_prompt"
                    ), "A_prompt"),
                    **_normalize_quality(quality_rows[metric_key]),
                }
            identity, quality_gate = classify_block(scores, floor), _quality_gate(outcomes, rules)
            records.append({
                "event_id": event_id,
                "story_id": block["story_id"],
                "character_name": block["character_name"],
                "seed": seed,
                "status": "classified",
                "identity_endpoint": "official_evaluator_derived",
                "identity_scores": scores,
                "source_continuity_diagnostic": local,
                "outcomes": outcomes,
                "identity_gate": identity,
                "quality_gate": quality_gate,
                "passed": identity["passed"] and quality_gate["passed"],
                "analysis_contract": inputs_contract,
            })
    classification_sha = _canonical_sha(records)
    # Q* is intentionally loaded only after every decoded classification is frozen.
    for record in records:
        if record["status"] != "classified":
            record["qstar"] = {"status": "not_applicable", "reason": "block_not_classified"}
        else:
            record["qstar"] = _qstar_for_block(run_blocks[record["event_id"], int(record["seed"])])
    events = []
    for event_id in FROZEN_EVENTS:
        eligible = [
            {
                "event_id": event_id, "seed": row["seed"], "scores": row["identity_scores"],
                "outcomes": row["outcomes"],
            }
            for row in records if row["event_id"] == event_id and row["status"] == "classified"
        ]
        failed = [
            {"seed": row["seed"], "status": row["status"], "reason": row["reason"]}
            for row in records if row["event_id"] == event_id and row["status"] != "classified"
        ]
        events.append(summarize_event(
            eligible, rules=rules, repeat_floor=floor, failed_blocks=failed, event_id=event_id,
        ))
    event_values = []
    for event in events:
        contrasts = _event_contrasts(event)
        if contrasts is not None and int(event["valid_seed_count"]) >= 2:
            event_values.append({"event_id": event["event_id"], **contrasts})
    bootstrap = {}
    for name in ("sufficiency", "necessity", "specificity"):
        values = [[row[name]] for row in event_values]
        mean, low, high = cluster_bootstrap_mean_ci(
            values, n_boot=n_boot, seed=bootstrap_seed,
        )
        bootstrap[name] = {
            "mean": mean, "low": low, "high": high,
            "n_event_clusters": len(values),
        }
    passed_events = sum(event["passed"] for event in events)
    report = {
        "schema_version": 2,
        "task_id": "vistorybench_subject_reappearance_v1",
        "identity_endpoint": "official_evaluator_derived",
        "source_continuity": "local_diagnostic_only",
        "qstar_role": "descriptive_only",
        "classification_snapshot_sha256_before_qstar": classification_sha,
        "run_manifest": {"path": str(run_manifest.resolve()), "sha256": run_sha},
        "cids_input_manifest": {"path": str(cids_input_manifest.resolve()), "sha256": input_sha},
        "analysis_contract": inputs_contract,
        "event_values": event_values,
        "cluster_bootstrap": bootstrap,
        "events": events,
        "failed_blocks": [row for row in records if row["status"] != "classified"],
        "passed_event_count": passed_events,
        "passed": passed_events >= 2,
    }
    return records, report


def write_analysis_outputs(records: Sequence[Mapping], report: Mapping, output: Path) -> None:
    """Atomically create the immutable records/report/table bundle."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, lock = _publish_lock(output)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)).resolve()
    try:
        (staging / "subject_reappearance_records.jsonl").write_text(
            "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in records),
            encoding="utf-8",
        )
        (staging / "subject_reappearance_report.json").write_text(
            json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        with (staging / "subject_reappearance_table.csv").open("x", encoding="utf-8", newline="") as handle:
            fields = [
                "event_id", "story_id", "seed", "status", "identity_passed", "quality_passed",
                "sufficiency", "necessity", "specificity", "qstar_status",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in records:
                scores = row.get("identity_scores", {})
                writer.writerow({
                    "event_id": row["event_id"], "story_id": row["story_id"], "seed": row["seed"],
                    "status": row["status"],
                    "identity_passed": row.get("identity_gate", {}).get("passed", ""),
                    "quality_passed": row.get("quality_gate", {}).get("passed", ""),
                    "sufficiency": scores.get("subject_only", 0) - scores.get("random_only", 0) if scores else "",
                    "necessity": scores.get("drop_random", 0) - scores.get("drop_subject", 0) if scores else "",
                    "specificity": scores.get("subject_only", 0) - scores.get("wrong_subject", 0) if scores else "",
                    "qstar_status": row["qstar"]["status"],
                })
        os.rename(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        os.close(descriptor)
        if lock.exists():
            lock.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--cids-input-manifest", type=Path, required=True)
    parser.add_argument("--cids-items", type=Path, required=True)
    parser.add_argument("--continuity-results", type=Path, required=True)
    parser.add_argument("--quality-results", type=Path, required=True)
    parser.add_argument("--prompt-results", type=Path, required=True)
    parser.add_argument("--quality-rules", type=Path, required=True)
    parser.add_argument("--repeat-floor", type=float, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    records, report = analyze_inputs(
        args.run_manifest, args.cids_input_manifest, args.cids_items,
        args.continuity_results, args.quality_results, args.prompt_results,
        args.quality_rules, repeat_floor=args.repeat_floor, n_boot=args.n_boot,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_analysis_outputs(records, report, args.output)
    print(json.dumps({"records": len(records), "events": len(report["events"]), "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
