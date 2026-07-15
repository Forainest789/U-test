#!/usr/bin/env python3
"""Build SlotMem character-list annotations from caption JSON files."""
import argparse
import csv
import json
import os
import re
from pathlib import Path


def sort_key(value):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value or "").replace("`", "").lower()).strip()


def extract_group_labels(group_caption):
    labels = []
    for match in re.finditer(r"(?:^|;\s*)([^:;\n]{1,100}):", group_caption or ""):
        label = match.group(1).strip().strip("`'\"")
        if label and label not in labels:
            labels.append(label)
    return labels


def infer_clip_characters(clip_caption, candidate_labels):
    caption_norm = normalize_text(clip_caption)
    found = []
    for label in candidate_labels:
        label_norm = normalize_text(label)
        if label_norm and label_norm in caption_norm and label not in found:
            found.append(label)
    return found


def attach_character_lists(caption_data):
    video_id = str(caption_data.get("video_id") or "")
    out = {
        "video_id": video_id,
        "complete": bool(caption_data.get("complete", True)),
        "selected_group_indices": list(caption_data.get("selected_group_indices") or []),
        "shots": [],
    }
    if not out["selected_group_indices"]:
        out["selected_group_indices"] = [
            shot.get("group_index")
            for shot in sorted(caption_data.get("shots", []), key=lambda item: sort_key(item.get("group_index")))
        ]

    for shot in sorted(caption_data.get("shots", []), key=lambda item: sort_key(item.get("group_index"))):
        group_caption = str(shot.get("group_caption") or "")
        group_characters = [str(c) for c in shot.get("characters") or []]
        if not group_characters:
            group_characters = extract_group_labels(group_caption)

        clips = []
        for clip in sorted(shot.get("clips", []), key=lambda item: sort_key(item.get("clip_index"))):
            clip_characters = [str(c) for c in clip.get("characters") or []]
            if not clip_characters:
                clip_characters = infer_clip_characters(clip.get("caption") or "", group_characters)
            clips.append({
                "clip_index": clip.get("clip_index"),
                "caption": clip.get("caption", ""),
                "error": clip.get("error", ""),
                "characters": clip_characters,
                "overlapping_clip_indices": [],
            })

        for clip in clips:
            own = set(normalize_text(c) for c in clip["characters"])
            overlaps = []
            for other in clips:
                if other["clip_index"] == clip["clip_index"]:
                    continue
                other_chars = set(normalize_text(c) for c in other["characters"])
                if own and own.intersection(other_chars):
                    overlaps.append(other["clip_index"])
            clip["overlapping_clip_indices"] = sorted(overlaps, key=sort_key)

        candidate_clips = [
            clip["clip_index"]
            for clip in clips
            if clip["characters"] and clip["overlapping_clip_indices"]
        ]
        out["shots"].append({
            "group_index": shot.get("group_index"),
            "group_caption": group_caption,
            "clips": clips,
            "characters": group_characters,
            "candidate_clips": sorted(candidate_clips, key=sort_key),
        })

    return out


def iter_caption_files(caption_dir):
    for path in sorted(Path(caption_dir).glob("*.json")):
        if path.parent.name == "character_lists":
            continue
        yield path


def main():
    parser = argparse.ArgumentParser(description="Build character_lists JSONs and candidate_groups.csv from caption JSONs.")
    parser.add_argument("--caption_dir", type=str, required=True,
                        help="Directory containing caption JSON files such as Top001.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output character_lists directory. Default: caption_dir/character_lists")
    parser.add_argument("--candidate_groups_csv", type=str, default=None,
                        help="Output candidate CSV. Default: caption_dir/candidate_groups.csv")
    args = parser.parse_args()

    caption_dir = Path(args.caption_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else caption_dir / "character_lists"
    candidate_csv = Path(args.candidate_groups_csv).resolve() if args.candidate_groups_csv else caption_dir / "candidate_groups.csv"
    if not caption_dir.is_dir():
        raise FileNotFoundError(f"caption_dir not found: {caption_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = []
    n_files = 0
    for src_path in iter_caption_files(caption_dir):
        with src_path.open("r", encoding="utf-8-sig") as f:
            caption_data = json.load(f)
        enriched = attach_character_lists(caption_data)
        video_id = enriched.get("video_id") or src_path.stem
        dst_path = output_dir / f"{video_id}.json"
        with dst_path.open("w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)
        n_files += 1

        for shot in enriched.get("shots", []):
            candidates = shot.get("candidate_clips") or []
            if candidates:
                candidate_rows.append({
                    "video_id": video_id,
                    "group_index": shot.get("group_index"),
                    "candidate_clips": "|".join(str(x) for x in candidates),
                })

    os.makedirs(candidate_csv.parent, exist_ok=True)
    with candidate_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "group_index", "candidate_clips"])
        writer.writeheader()
        writer.writerows(candidate_rows)

    print(f"Wrote {n_files} character-list JSON files to {output_dir}")
    print(f"Wrote {len(candidate_rows)} candidate groups to {candidate_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
