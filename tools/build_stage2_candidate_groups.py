#!/usr/bin/env python3
"""Build explicit stage2 target/memory/update candidate triplets from character-list JSON."""

import argparse
import csv
import json
import os
from pathlib import Path


def _character_set(clip):
    chars = clip.get("characters") or []
    return {str(x).strip() for x in chars if str(x).strip()}


def _clip_index(clip):
    try:
        return int(clip.get("clip_index"))
    except (TypeError, ValueError):
        return None


def _video_exists(video_root, video_id, group_index, clip_index):
    if not video_root:
        return True
    path = Path(video_root) / video_id / f"group_{group_index}" / f"clip{clip_index}.mp4"
    return path.is_file()


def build_rows(character_lists_dir, video_root=None, require_videos=False):
    json_paths = sorted(Path(character_lists_dir).glob("*.json"))
    for json_path in json_paths:
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        video_id = str(data.get("video_id") or json_path.stem)
        for shot in data.get("shots") or []:
            try:
                group_index = int(shot.get("group_index"))
            except (TypeError, ValueError):
                continue

            clips = {}
            for clip in shot.get("clips") or []:
                idx = _clip_index(clip)
                if idx is not None:
                    clips[idx] = clip
            if len(clips) < 3:
                continue

            for target_idx, target_clip in clips.items():
                target_chars = _character_set(target_clip)
                if not target_chars:
                    continue
                if require_videos and not _video_exists(video_root, video_id, group_index, target_idx):
                    continue

                overlaps = []
                for raw in target_clip.get("overlapping_clip_indices") or []:
                    try:
                        overlap_idx = int(raw)
                    except (TypeError, ValueError):
                        continue
                    if overlap_idx != target_idx and overlap_idx in clips:
                        overlaps.append(overlap_idx)
                overlaps = sorted(set(overlaps))
                if len(overlaps) < 2:
                    continue

                for memory_idx in overlaps:
                    memory_clip = clips[memory_idx]
                    memory_chars = _character_set(memory_clip)
                    if not (target_chars & memory_chars):
                        continue
                    if require_videos and not _video_exists(video_root, video_id, group_index, memory_idx):
                        continue

                    for update_idx in overlaps:
                        if update_idx == memory_idx:
                            continue
                        update_clip = clips[update_idx]
                        shared_chars = sorted(target_chars & memory_chars & _character_set(update_clip))
                        if not shared_chars:
                            continue
                        if require_videos and not _video_exists(video_root, video_id, group_index, update_idx):
                            continue

                        candidate_clips = "|".join(str(x) for x in sorted({target_idx, memory_idx, update_idx}))
                        yield {
                            "video_id": video_id,
                            "group_index": group_index,
                            "candidate_clips": candidate_clips,
                            "target_clip": target_idx,
                            "memory_clip": memory_idx,
                            "update_memory_clip": update_idx,
                            "shared_characters": ";".join(shared_chars),
                        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--character-lists-dir",
        required=True,
    )
    parser.add_argument(
        "--output-csv",
        required=True,
    )
    parser.add_argument("--video-root", default="")
    parser.add_argument(
        "--require-videos",
        action="store_true",
        help="Drop triplets whose target/memory/update mp4 files are missing.",
    )
    args = parser.parse_args()

    rows = list(build_rows(
        args.character_lists_dir,
        video_root=args.video_root,
        require_videos=args.require_videos,
    ))

    fieldnames = [
        "video_id",
        "group_index",
        "candidate_clips",
        "target_clip",
        "memory_clip",
        "update_memory_clip",
        "shared_characters",
    ]
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    unique_groups = {(r["video_id"], r["group_index"]) for r in rows}
    unique_targets = {(r["video_id"], r["group_index"], r["target_clip"]) for r in rows}
    print(
        f"wrote {len(rows)} stage2 triplets, {len(unique_targets)} targets, "
        f"{len(unique_groups)} groups -> {output_path}"
    )


if __name__ == "__main__":
    main()
