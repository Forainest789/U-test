#!/usr/bin/env python3
"""Prepare SlotMem inference outputs for NarraStream-Bench.

Input layout:
  <case-root>/<sample>/merged_chunks.mp4

Output layout:
  <output-root>/<case-name>/video/sample_0.mp4
  <output-root>/<case-name>/prompt.jsonl
  <output-root>/<case-name>/manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def read_prompts(data_root: Path, sample: str) -> list[str]:
    path = data_root / sample / "rewrite_caption.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", payload) if isinstance(payload, dict) else payload
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"Cannot read chunks from {path}")
    prompts = []
    for chunk in chunks:
        text = chunk.get("content") if isinstance(chunk, dict) else None
        if not text:
            raise ValueError(f"Missing content in {path}: {chunk!r}")
        prompts.append(str(text))
    return prompts


def discover_samples(data_root: Path) -> list[str]:
    samples = []
    for path in sorted(data_root.iterdir()):
        if not path.is_dir() or path.name == "subject_references":
            continue
        if (path / "rewrite_caption.json").is_file():
            samples.append(path.name)
    return samples


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SlotMem inference outputs for NarraStream-Bench."
    )
    parser.add_argument(
        "--case-root",
        required=True,
        type=Path,
        help="Directory containing one generated sample subdirectory per case.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Directory containing matching sample rewrite_caption.json files.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Destination root for NarraStream-Bench video, prompt, and manifest files.",
    )
    parser.add_argument("--case-name", default=None, help="Output case name; defaults to --case-root basename.")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--allow-missing", action="store_true", help="Write manifest entries for missing videos instead of failing immediately.")
    args = parser.parse_args()

    case_root = args.case_root.resolve()
    case_name = args.case_name or case_root.name
    out_root = (args.output_root / case_name).resolve()
    video_dir = out_root / "video"
    prompt_path = out_root / "prompt.jsonl"
    samples = discover_samples(args.data_root)
    records = []
    missing = []

    out_root.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    with prompt_path.open("w", encoding="utf-8") as prompt_file:
        for index, sample in enumerate(samples):
            src = case_root / sample / "merged_chunks.mp4"
            if not src.is_file():
                missing.append({"sample": sample, "path": str(src)})
                if not args.allow_missing:
                    continue
            prompts = read_prompts(args.data_root, sample)
            dst = video_dir / f"sample_{index}.mp4"
            input_video = None
            if src.is_file():
                link_or_copy(src, dst, args.mode)
                input_video = str(dst)
            prompt_file.write(json.dumps({"prompts": prompts}, ensure_ascii=False) + "\n")
            records.append(
                {
                    "index": index,
                    "sample": sample,
                    "source_video": str(src),
                    "input_video": input_video,
                    "prompt_count": len(prompts),
                }
            )

    manifest = {
        "case_name": case_name,
        "case_root": str(case_root),
        "data_root": str(args.data_root.resolve()),
        "output_root": str(out_root),
        "video_dir": str(video_dir),
        "prompts": str(prompt_path),
        "sample_count": len(samples),
        "prepared_count": sum(1 for r in records if r["input_video"]),
        "missing": missing,
        "records": records,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if missing and not args.allow_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
