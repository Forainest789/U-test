"""Freeze source-only subject-slot rankings from validated capture artifacts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
from pathlib import Path
from typing import Sequence

import torch

from .prefix_contract import sha256_file, write_bytes_no_clobber
from .source_semantic_scores import produce_source_semantic_scores
from .subject_subspace import (
    FROZEN_LAYER_GROUPS,
    aggregate_semantic_slot_scores,
    build_mask_manifest,
    canonical_json_sha256,
    capture_tensor_sha256,
    source_only_semantic_group_manifest,
    validate_semantic_scores,
    validate_source_capture,
)


def _json_bytes(data: bytes, label: object) -> dict:
    value = json.loads(data.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {label}")
    return value


def _bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _output(path: Path) -> Path:
    return path if path.suffix.casefold() == ".json" else path / "subject_subspace_manifest.json"


def freeze_subject_subspace(
    *, event_path: Path, source_capture_path: Path, semantic_scores_path: Path,
    output_path: Path, seed: int, repo_root: Path,
) -> dict:
    paths = [Path(path).resolve() for path in (event_path, source_capture_path, semantic_scores_path)]
    event_path, source_capture_path, semantic_scores_path = paths
    output_path = _output(Path(output_path)).resolve()
    event_bytes, capture_bytes, score_bytes = (path.read_bytes() for path in paths)
    event, scores = _json_bytes(event_bytes, event_path), _json_bytes(score_bytes, semantic_scores_path)
    source_path = Path(event["source_json_path"]).resolve()
    source_bytes = source_path.read_bytes()
    capture = torch.load(io.BytesIO(capture_bytes), map_location="cpu", weights_only=True)
    subject = validate_source_capture(capture, event, repo_root=Path(repo_root).resolve(), source_json_sha256=_bytes_sha256(source_bytes))
    capture_seed = capture["provenance"]["source_seed"]
    event_seeds = [event[key] for key in ("seed", "source_seed", "target_seed") if event.get(key) is not None]
    all_seeds = [seed, capture_seed, *event_seeds]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in all_seeds) or any(seed != value for value in all_seeds[1:]):
        raise ValueError("requested seed does not match source capture/event seed")
    semantic_manifest = source_only_semantic_group_manifest(_json_bytes(source_bytes, source_path), event)
    score_rows = validate_semantic_scores(
        scores, event=event, source_capture_sha256=_bytes_sha256(capture_bytes),
        source_capture=capture, subject_captures=subject,
        expected_semantic_manifest=semantic_manifest,
    )
    by_bank = {}
    for row in subject:
        by_bank.setdefault(int(row["bank"]), {})[int(row["layer"])] = row
    rankings = {}
    for bank, by_layer in sorted(by_bank.items()):
        configured_layers = {layer for members in FROZEN_LAYER_GROUPS.values() for layer in members}
        if set(by_layer) != configured_layers:
            raise ValueError(f"frozen layer groups are incomplete or contain extra layers for bank {bank}")
        for group, members_tuple in FROZEN_LAYER_GROUPS.items():
            members = list(members_tuple)
            if any(layer not in by_layer for layer in members):
                raise ValueError(f"frozen layer group {group} is incomplete for bank {bank}")
            values = aggregate_semantic_slot_scores([by_layer[layer] for layer in members], score_rows)
            semantic = sorted(range(values.numel()), key=lambda index: (-float(values[index]), index))
            address = f"bank_{bank}/group_{group}"
            rankings[address] = {
                "bank": bank, "layer_group": group, "member_layers": members,
                "source_payload_sha256_by_layer": {str(layer): by_layer[layer]["sha256"]["encoded_slots"] for layer in members},
                "semantic": semantic, "visual_cf": None, "reference": None,
            }
    inputs = {
        "event_sha256": _bytes_sha256(event_bytes),
        "source_capture_sha256": _bytes_sha256(capture_bytes),
        "source_capture_canonical_artifact_sha256": capture["canonical_artifact_sha256"],
        "semantic_scores_sha256": _bytes_sha256(score_bytes),
        "semantic_scores_canonical_artifact_sha256": scores["canonical_artifact_sha256"],
        "semantic_manifest_sha256": canonical_json_sha256(semantic_manifest),
        "source_json_sha256": _bytes_sha256(source_bytes),
        "visual_counterfactual_payload_sha256": None,
        "reference_payload_sha256": None,
    }
    manifest = build_mask_manifest(inputs=inputs, rankings=rankings, event=event, seed=seed)
    write_bytes_no_clobber(output_path, (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return manifest


def _capture_reference(args: argparse.Namespace) -> None:
    reference, output = Path(args.reference_image).resolve(), Path(args.output).resolve()
    reference_bytes = reference.read_bytes()
    record = {
        "status": "not_available",
        "reason": "reference_only_source_extraction_path_missing",
        "reference_path": str(reference),
        "reference_sha256": _bytes_sha256(reference_bytes),
        "target_evidence_read": False,
        "inference_args_sha256": canonical_json_sha256(list(args.inference_args)),
    }
    if not args.dry_run:
        raise RuntimeError("capture-reference unavailable: refusing to fabricate reference tokens")
    write_bytes_no_clobber(output, (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(record, sort_keys=True))


def _self_check_fixture(root: Path, repo: Path) -> tuple[Path, Path, Path, Path]:
    story, reference, model = root / "story.json", root / "00.jpg", root / "model.pt"
    story.write_text(json.dumps({"characters": {"Ana": "red coat"}, "chunks": [{"content": "station", "character_list": ["Ana"]}]}) + "\n", encoding="utf-8")
    reference.write_bytes(b"reference")
    model.write_bytes(b"model")
    event = {"event_id": "self_check", "character_name": "Ana", "source_chunk_idx": 0, "target_chunk_idx": 1, "source_json_path": str(story.resolve()), "reference_path": str(reference.resolve()), "reference_sha256": sha256_file(reference)}
    event_path = root / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    raw, slots = torch.arange(12, dtype=torch.float32).reshape(4, 3), torch.arange(96, dtype=torch.float32).reshape(32, 3)
    meta, attention = [
        {"char_id": "Ana", "inside_box": index < 2, "tau_local": float(index)}
        for index in range(4)
    ], {"Ana": torch.full((32, 4), 0.25)}
    rows = []
    for layer in range(16):
        row = {"character": "Ana", "bank": 0, "layer": layer, "raw_tokens": raw, "raw_token_meta": meta, "encoded_slots": slots, "attention": attention}
        row["tensor_shapes"] = {"raw_tokens": [4, 3], "encoded_slots": [32, 3], "attention": {"Ana": [32, 4]}}
        row["sha256"] = {"raw_tokens": capture_tensor_sha256(raw), "raw_token_meta": canonical_json_sha256(meta), "encoded_slots": capture_tensor_sha256(slots), "attention": capture_tensor_sha256(attention)}
        rows.append(row)
    provenance = {"source_json_path": str(story.resolve()), "source_json_sha256": sha256_file(story), "reference_file_sha256": sha256_file(reference), "fixed_reference_scope": "source_only", "source_seed": 0, "code_identity": {"infer_slotmem_sha256": sha256_file(repo / "infer_slotmem.py"), "mem_encoder_utils_sha256": sha256_file(repo / "mem_encoder_utils.py")}, "runtime_identity": {"python_version": "self-check", "torch_version": str(torch.__version__), "inference_args_sha256": "1" * 64}, "model_identity": {"high_noise": [{"path": str(model.resolve()), "sha256": sha256_file(model)}], "low_noise": []}}
    canonical = {"schema_version": 1, "source_chunk_idx": 0, "target_evidence_read": False, "provenance": provenance, "captures": [{key: row[key] for key in ("character", "bank", "layer", "tensor_shapes", "sha256")} for row in rows]}
    capture = {**canonical, "captures": rows, "canonical_artifact_sha256": canonical_json_sha256(canonical)}
    capture_path = root / "source_capture.pt"
    torch.save(capture, capture_path)
    scores_path = root / "scores.json"
    produce_source_semantic_scores(
        event_path=event_path,
        source_capture_path=capture_path,
        output_path=scores_path,
        repo_root=repo,
    )
    return event_path, capture_path, scores_path, root / "manifest.json"


def _self_check() -> None:
    with tempfile.TemporaryDirectory() as directory:
        repo, root = Path(__file__).resolve().parents[1], Path(directory)
        event, capture, scores, output = _self_check_fixture(root, repo)
        manifest = freeze_subject_subspace(event_path=event, source_capture_path=capture, semantic_scores_path=scores, output_path=output, seed=0, repo_root=repo)
        assert len(manifest["layers"][0]["semantic_top8"]) == 8
        assert len(manifest["layers"][0]["random_top8"]) == 8
        assert manifest["target_evidence_read"] is False
    print("[subject-subspace] self-check OK")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    for name in ("event", "source-capture", "semantic-scores", "output"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--seed", type=int)
    subparsers = parser.add_subparsers(dest="command")
    reference = subparsers.add_parser("capture-reference")
    reference.add_argument("--reference-image", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument("--dry-run", action="store_true")
    reference.add_argument("inference_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.self_check:
        return _self_check()
    if args.command == "capture-reference":
        return _capture_reference(args)
    required = (args.event, args.source_capture, args.semantic_scores, args.output, args.seed)
    if any(value is None for value in required):
        parser.error("freeze requires --event --source-capture --semantic-scores --output --seed")
    result = freeze_subject_subspace(event_path=args.event, source_capture_path=args.source_capture, semantic_scores_path=args.semantic_scores, output_path=args.output, seed=args.seed, repo_root=Path(__file__).resolve().parents[1])
    print(json.dumps({"output": str(_output(args.output).resolve()), "mask_manifest_sha256": result["mask_manifest_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
