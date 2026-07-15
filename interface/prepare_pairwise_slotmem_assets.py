"""Prepare SlotMem-vs-baseline assets for blind pairwise evaluation.

The frontend intentionally consumes only the manifest and normalized videos.
All source-specific alignment decisions are handled here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent

DEFAULT_OUTPUT_ROOT = ROOT / "pairwise_slotmem_assets"
PRIMARY_METHOD = {
    "key": "slotmem",
    "display_name": "SlotMem",
    "variant": "both_ckpt_high1100_low1700_chunks4_trainaligned",
}


@dataclass(frozen=True)
class BaselineSpec:
    key: str
    display_name: str
    relative_template: str

    def source_path(self, sample_id: str, baseline_root: Path) -> Path:
        return baseline_root / self.relative_template.format(sample=sample_id)


BASELINES = {
    "wan22_native": BaselineSpec(
        key="wan22_native",
        display_name="WAN 2.2 Native",
        relative_template="StoryMem/results/WAN2.2/{sample}/story.mp4",
    ),
    "storymem": BaselineSpec(
        key="storymem",
        display_name="StoryMem",
        relative_template="StoryMem/results/testdata_storymem/{sample}/merged_chunks_reencoded_for_eval.mp4",
    ),
    "storydiffusion": BaselineSpec(
        key="storydiffusion",
        display_name="StoryDiffusion",
        relative_template="StoryMem/results/Storydiffusion/{sample}/story.mp4",
    ),
    "iamflow_i2v_kvselfattn": BaselineSpec(
        key="iamflow_i2v_kvselfattn",
        display_name="IAMFlow",
        relative_template="IAMFlow/inference_outputs/iamflow_wan22_i2v_testdata_20260614_kvselfattn_full/{sample}/merged_chunks.mp4",
    ),
}


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def available_ffmpeg_encoders() -> set[str]:
    output = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True, stderr=subprocess.STDOUT)
    encoders: set[str] = set()
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def choose_encoder(requested: str) -> str:
    if requested != "auto":
        return requested
    encoders = available_ffmpeg_encoders()
    for candidate in ("libx264", "libopenh264", "mpeg4"):
        if candidate in encoders:
            return candidate
    raise RuntimeError("No supported ffmpeg encoder found; tried libx264, libopenh264, mpeg4")


def encoder_args(encoder: str, crf: int) -> list[str]:
    if encoder == "libx264":
        return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", str(crf)]
    if encoder == "libopenh264":
        return ["-c:v", "libopenh264", "-pix_fmt", "yuv420p", "-b:v", "5000k"]
    if encoder == "mpeg4":
        return ["-c:v", "mpeg4", "-pix_fmt", "yuv420p", "-q:v", "3"]
    return ["-c:v", encoder, "-pix_fmt", "yuv420p"]


def stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def probe_video(path: Path, fps: int) -> dict[str, float | int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration,nb_frames,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(cmd, text=True))
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found: {path}")
    stream = streams[0]

    duration = float(stream.get("duration") or 0.0)
    raw_frames = stream.get("nb_frames")
    try:
        frames = int(raw_frames)
    except (TypeError, ValueError):
        frames = max(1, round(duration * fps))
    if duration <= 0 and frames > 0:
        duration = frames / float(fps)

    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration_s": duration,
        "frames": frames,
    }


def chunk_sources_for_baseline(baseline: BaselineSpec, sample_id: str, baseline_root: Path) -> list[Path]:
    if baseline.key == "wan22_native":
        sample_root = baseline_root / "StoryMem" / "results" / "WAN2.2" / sample_id
        for subdir in ("merge_ready", "clips"):
            files = sorted((sample_root / subdir).glob("chunk_*.mp4"))
            if files:
                return files
    elif baseline.key == "storydiffusion":
        sample_root = baseline_root / "StoryMem" / "results" / "Storydiffusion" / sample_id
        for subdir in ("merge_ready", "clips"):
            files = sorted((sample_root / subdir).glob("chunk_*.mp4"))
            if files:
                return files
    elif baseline.key == "storymem":
        sample_root = baseline_root / "StoryMem" / "results" / "testdata_storymem" / sample_id
        files = sorted(sample_root.glob("01_*.mp4"))
        if files:
            return files
    elif baseline.key == "iamflow_i2v_kvselfattn":
        files = sorted(
            (
                baseline_root
                / "IAMFlow"
                / "inference_outputs"
                / "iamflow_wan22_i2v_testdata_20260614_kvselfattn_full"
                / sample_id
            ).glob("chunk_*.mp4")
        )
        if files:
            return files
    return []


def load_prompt_chunks(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", data) if isinstance(data, dict) else data
    if not isinstance(chunks, list):
        raise ValueError(f"Cannot find prompt chunks in {path}")

    normalized = []
    for idx, chunk in enumerate(chunks):
        if isinstance(chunk, str):
            text = chunk
            character_list: list[str] = []
        elif isinstance(chunk, dict):
            text = (
                chunk.get("content")
                or chunk.get("prompt")
                or chunk.get("caption")
                or chunk.get("text")
                or ""
            )
            raw_characters = chunk.get("character_list") or chunk.get("characters") or []
            character_list = [str(item) for item in raw_characters] if isinstance(raw_characters, list) else []
        else:
            text = str(chunk)
            character_list = []

        normalized.append(
            {
                "index": idx,
                "text": " ".join(str(text).split()),
                "character_list": character_list,
            }
        )
    return normalized


def resolve_slot_video(sample_dir: Path) -> Path:
    merged = sample_dir / "merged_chunks.mp4"
    if merged.is_file():
        return merged
    chunk = sample_dir / "chunk_000.mp4"
    if chunk.is_file():
        return chunk
    raise FileNotFoundError(f"No SlotMem video found under {sample_dir}")


def slot_chunk_files(sample_dir: Path) -> list[Path]:
    files = sorted(sample_dir.glob("chunk_*.mp4"))
    if files:
        return files
    return [resolve_slot_video(sample_dir)]


def build_prompt_segments(
    prompt_chunks: list[dict[str, Any]],
    chunk_frames: list[int],
    fps: int,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    cursor_frames = 0
    target_frames = sum(chunk_frames)
    for idx, frames in enumerate(chunk_frames):
        if idx >= len(prompt_chunks):
            break
        start_frame = cursor_frames
        end_frame = min(cursor_frames + int(frames), target_frames)
        cursor_frames += int(frames)
        if end_frame <= start_frame:
            continue
        segments.append(
            {
                "start_s": round(start_frame / fps, 4),
                "end_s": round(end_frame / fps, 4),
                "text": prompt_chunks[idx]["text"],
                "character_list": prompt_chunks[idx]["character_list"],
            }
        )
        if end_frame >= target_frames:
            break
    if segments:
        segments[-1]["end_s"] = round(target_frames / fps, 4)
    return segments


def normalize_video(
    src: Path,
    dst: Path,
    *,
    frames: int,
    width: int,
    height: int,
    fps: int,
    crf: int,
    encoder: str,
    retime: bool,
    force: bool,
    dry_run: bool,
) -> None:
    if dst.is_file() and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    filters: list[str] = []
    if retime:
        source_frames = max(1, int(probe_video(src, fps)["frames"]))
        filters.append(f"setpts={frames / source_frames:.12f}*PTS")
    filters.extend(
        [
            f"fps={fps}",
            f"scale={width}:{height}:flags=bicubic",
            "setsar=1",
            "tpad=stop_mode=clone:stop_duration=2",
        ]
    )

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if force else "-n",
        "-i",
        str(src),
        "-an",
        "-vf",
        ",".join(filters),
        "-frames:v",
        str(frames),
        *encoder_args(encoder, crf),
        "-movflags",
        "+faststart",
        str(dst),
    ]
    if dry_run:
        print(" ".join(cmd))
        return
    run(cmd)


def concat_videos(
    chunk_paths: list[Path],
    dst: Path,
    *,
    encoder: str,
    force: bool,
    dry_run: bool,
) -> None:
    if dst.is_file() and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    concat_list = dst.with_suffix(".concat.txt")
    concat_list.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in chunk_paths),
        encoding="utf-8",
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if force else "-n",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-an",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    if dry_run:
        print(" ".join(cmd))
        return
    try:
        run(cmd)
    except subprocess.CalledProcessError:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if force else "-n",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-an",
            *encoder_args(encoder, 18),
            "-movflags",
            "+faststart",
            str(dst),
        ]
        run(cmd)


def normalize_chunk_sequence(
    sources: list[Path],
    dst: Path,
    *,
    chunk_frames: list[int],
    width: int,
    height: int,
    fps: int,
    crf: int,
    encoder: str,
    force: bool,
    dry_run: bool,
) -> None:
    if len(sources) < len(chunk_frames):
        raise ValueError(f"Need {len(chunk_frames)} chunk sources for {dst}, got {len(sources)}")
    if dst.is_file() and not force:
        return

    tmp_dir = dst.parent / f".{dst.stem}_chunks"
    if force and tmp_dir.exists() and not dry_run:
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    normalized_chunks: list[Path] = []
    for idx, target_frames in enumerate(chunk_frames):
        out_chunk = tmp_dir / f"chunk_{idx:03d}.mp4"
        normalize_video(
            sources[idx],
            out_chunk,
            frames=int(target_frames),
            width=width,
            height=height,
            fps=fps,
            crf=crf,
            encoder=encoder,
            retime=True,
            force=force,
            dry_run=dry_run,
        )
        normalized_chunks.append(out_chunk)
    concat_videos(normalized_chunks, dst, encoder=encoder, force=force, dry_run=dry_run)


def build_task(
    *,
    sample_id: str,
    baseline: BaselineSpec,
    slot_root: Path,
    data_root: Path,
    baseline_root: Path,
    output_root: Path,
    width: int,
    height: int,
    fps: int,
    crf: int,
    encoder: str,
    force: bool,
    dry_run: bool,
    seed: str,
) -> dict[str, Any]:
    sample_slot_dir = slot_root / sample_id
    baseline_source = baseline.source_path(sample_id, baseline_root)
    if not baseline_source.is_file():
        raise FileNotFoundError(f"Baseline video not found for {baseline.key}/{sample_id}: {baseline_source}")

    prompt_chunks = load_prompt_chunks(data_root / sample_id / "rewrite_caption.json")
    slot_chunks = slot_chunk_files(sample_slot_dir)
    baseline_chunks = chunk_sources_for_baseline(baseline, sample_id, baseline_root)
    if not baseline_chunks:
        raise FileNotFoundError(f"Baseline chunk videos not found for {baseline.key}/{sample_id}")

    used_chunks = min(len(slot_chunks), len(prompt_chunks))
    chunk_frames = [int(probe_video(path, fps)["frames"]) for path in slot_chunks[:used_chunks]]
    target_frames = sum(chunk_frames)
    if target_frames <= 0:
        raise RuntimeError(f"Invalid target frame count for {baseline.key}/{sample_id}")

    video_dir = output_root / "videos" / baseline.key / sample_id
    slot_output = video_dir / "A_source_primary.mp4"
    baseline_output = video_dir / "B_source_baseline.mp4"
    normalize_chunk_sequence(
        slot_chunks,
        slot_output,
        chunk_frames=chunk_frames,
        width=width,
        height=height,
        fps=fps,
        crf=crf,
        encoder=encoder,
        force=force,
        dry_run=dry_run,
    )
    normalize_chunk_sequence(
        baseline_chunks,
        baseline_output,
        chunk_frames=chunk_frames,
        width=width,
        height=height,
        fps=fps,
        crf=crf,
        encoder=encoder,
        force=force,
        dry_run=dry_run,
    )

    options = [
        {
            "method_key": PRIMARY_METHOD["key"],
            "method_display_name": PRIMARY_METHOD["display_name"],
            "video_path": str(slot_output),
        },
        {
            "method_key": baseline.key,
            "method_display_name": baseline.display_name,
            "video_path": str(baseline_output),
        },
    ]
    rng = random.Random(stable_int(f"{seed}:{baseline.key}:{sample_id}:options"))
    rng.shuffle(options)
    for label, option in zip(("A", "B"), options):
        option["label"] = label

    segments = build_prompt_segments(prompt_chunks, chunk_frames, fps)
    return {
        "task_id": f"{baseline.key}_{sample_id}",
        "sample_id": sample_id,
        "duration_s": round(target_frames / fps, 4),
        "frames": target_frames,
        "fps": fps,
        "prompt_segments": segments,
        "options": options,
    }


def build_manifest(
    *,
    baseline: BaselineSpec,
    samples: list[str],
    slot_root: Path,
    data_root: Path,
    baseline_root: Path,
    output_root: Path,
    width: int,
    height: int,
    fps: int,
    crf: int,
    encoder: str,
    force: bool,
    dry_run: bool,
    seed: str,
) -> Path:
    tasks = [
        build_task(
            sample_id=sample_id,
            baseline=baseline,
            slot_root=slot_root,
            data_root=data_root,
            baseline_root=baseline_root,
            output_root=output_root,
            width=width,
            height=height,
            fps=fps,
            crf=crf,
            encoder=encoder,
            force=force,
            dry_run=dry_run,
            seed=seed,
        )
        for sample_id in samples
    ]
    random.Random(stable_int(f"{seed}:{baseline.key}:schedule")).shuffle(tasks)

    manifest = {
        "dataset_id": f"slotmem_pairwise_{baseline.key}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "primary_method": {
            "key": PRIMARY_METHOD["key"],
            "display_name": PRIMARY_METHOD["display_name"],
        },
        "baseline": {
            "key": baseline.key,
            "display_name": baseline.display_name,
        },
        "canvas": {
            "width": width,
            "height": height,
            "fps": fps,
        },
        "task_count": len(tasks),
        "tasks": tasks,
    }
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{baseline.key}.json"
    if dry_run:
        print(f"[dry-run] write manifest: {manifest_path}")
        return manifest_path
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def discover_samples(data_root: Path, slot_root: Path) -> list[str]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not slot_root.is_dir():
        raise FileNotFoundError(f"SlotMem root not found: {slot_root}")

    samples = [
        path.name
        for path in sorted(data_root.iterdir())
        if path.is_dir()
        and path.name != "subject_references"
        and (path / "rewrite_caption.json").is_file()
        and has_slot_video(slot_root / path.name)
    ]
    if not samples:
        raise RuntimeError(
            f"No samples found with rewrite_caption.json under {data_root} "
            f"and matching SlotMem outputs under {slot_root}"
        )
    return samples


def has_slot_video(sample_dir: Path) -> bool:
    if not sample_dir.is_dir():
        return False
    if (sample_dir / "merged_chunks.mp4").is_file():
        return True
    return any(path.is_file() for path in sample_dir.glob("chunk_*.mp4"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        choices=[*BASELINES.keys(), "all"],
        default="wan22_native",
        help="Baseline manifest to prepare.",
    )
    parser.add_argument("--slot-root", type=Path, required=True, help="Root containing SlotMem generated sample outputs.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument(
        "--encoder",
        default="auto",
        help="ffmpeg video encoder. auto prefers libx264, then libopenh264, then mpeg4.",
    )
    parser.add_argument("--seed", default="slotmem-pairwise-v1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baselines = BASELINES.values() if args.baseline == "all" else [BASELINES[args.baseline]]
    slot_root = args.slot_root.expanduser()
    data_root = args.data_root.expanduser()
    samples = discover_samples(data_root, slot_root)
    encoder = choose_encoder(args.encoder)
    print(f"[Init] ffmpeg encoder: {encoder}")
    print(f"[Init] samples: {len(samples)}")

    for baseline in baselines:
        manifest = build_manifest(
            baseline=baseline,
            samples=samples,
            slot_root=slot_root,
            data_root=data_root,
            baseline_root=args.baseline_root.expanduser(),
            output_root=args.output_root.expanduser(),
            width=args.width,
            height=args.height,
            fps=args.fps,
            crf=args.crf,
            encoder=encoder,
            force=args.force,
            dry_run=args.dry_run,
            seed=args.seed,
        )
        print(f"[OK] {baseline.key}: {manifest}")


if __name__ == "__main__":
    main()
