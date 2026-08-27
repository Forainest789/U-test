"""Produce immutable source-only semantic token scores from a SlotMem capture."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from .prefix_contract import write_json_no_clobber
from .subject_subspace import (
    FROZEN_LAYER_GROUPS,
    SOURCE_SEMANTIC_FORMULA,
    build_semantic_score_artifact,
    source_metadata_semantic_groups,
    source_only_semantic_group_manifest,
    validate_semantic_scores,
    validate_source_capture,
)


def _json_object(data: bytes, label: object) -> dict:
    value = json.loads(data.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {label}")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validated_inputs(
    event_path: Path, source_capture_path: Path, repo_root: Path
) -> tuple[dict, dict, list[Mapping], dict, str]:
    event_bytes = event_path.read_bytes()
    capture_bytes = source_capture_path.read_bytes()
    event = _json_object(event_bytes, event_path)
    source_path = Path(event["source_json_path"]).resolve()
    source_bytes = source_path.read_bytes()
    story = _json_object(source_bytes, source_path)
    capture = torch.load(io.BytesIO(capture_bytes), map_location="cpu", weights_only=True)
    subject_rows = validate_source_capture(
        capture,
        event,
        repo_root=repo_root,
        source_json_sha256=_sha256(source_bytes),
    )
    seed = capture["provenance"]["source_seed"]
    event_seeds = [
        event[key] for key in ("seed", "source_seed", "target_seed") if key in event
    ]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value != seed
        for value in event_seeds
    ):
        raise ValueError("source capture seed does not match event provenance")
    required_layers = {
        layer for members in FROZEN_LAYER_GROUPS.values() for layer in members
    }
    by_bank: dict[int, set[int]] = {}
    for row in subject_rows:
        by_bank.setdefault(int(row["bank"]), set()).add(int(row["layer"]))
    if not by_bank or any(layers != required_layers for layers in by_bank.values()):
        raise ValueError("source capture frozen layers are incomplete or contain extras")
    return event, capture, subject_rows, story, _sha256(capture_bytes)


def produce_source_semantic_scores(
    *,
    event_path: Path,
    source_capture_path: Path,
    output_path: Path,
    repo_root: Path,
) -> dict[str, object]:
    event_path = Path(event_path).resolve()
    source_capture_path = Path(source_capture_path).resolve()
    repo_root = Path(repo_root).resolve()
    event, capture, subject_rows, story, capture_sha256 = _validated_inputs(
        event_path, source_capture_path, repo_root
    )
    score_rows = [
        {
            "character": row["character"],
            "bank": int(row["bank"]),
            "layer": int(row["layer"]),
            "groups": source_metadata_semantic_groups(
                row["raw_token_meta"], event["character_name"]
            ),
        }
        for row in subject_rows
    ]
    semantic_manifest = source_only_semantic_group_manifest(story, event)
    artifact = build_semantic_score_artifact(
        event_id=event["event_id"],
        source_capture_sha256=capture_sha256,
        source_capture_canonical_artifact_sha256=capture["canonical_artifact_sha256"],
        semantic_manifest=semantic_manifest,
        source_provenance=capture["provenance"],
        captures=score_rows,
        formula=SOURCE_SEMANTIC_FORMULA,
        subject_char_id=event["character_name"],
        source_seed=capture["provenance"]["source_seed"],
    )
    validate_semantic_scores(
        artifact,
        event=event,
        source_capture_sha256=capture_sha256,
        source_capture=capture,
        subject_captures=subject_rows,
        expected_semantic_manifest=semantic_manifest,
    )
    write_json_no_clobber(Path(output_path).resolve(), artifact)
    return artifact


def validate_source_semantic_scores_file(
    *,
    event_path: Path,
    source_capture_path: Path,
    scores_path: Path,
    repo_root: Path,
) -> dict[tuple[int, int], Mapping[str, object]]:
    event, capture, subject_rows, story, capture_sha256 = _validated_inputs(
        Path(event_path).resolve(),
        Path(source_capture_path).resolve(),
        Path(repo_root).resolve(),
    )
    scores_path = Path(scores_path).resolve()
    scores = _json_object(scores_path.read_bytes(), scores_path)
    return validate_semantic_scores(
        scores,
        event=event,
        source_capture_sha256=capture_sha256,
        source_capture=capture,
        subject_captures=subject_rows,
        expected_semantic_manifest=source_only_semantic_group_manifest(story, event),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    produce_source_semantic_scores(
        event_path=args.event,
        source_capture_path=args.source_capture,
        output_path=args.output,
        repo_root=args.repo_root,
    )
    return 0


if __name__ == "__main__":
    main()
