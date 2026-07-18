"""Export clip-record JSON into grouped MP4 files.

Each group uses one ffmpeg invocation: input-side seek skips preceding media,
``-t`` bounds decoding, and ``filter_complex`` trims contiguous clip outputs.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# CUDA decoding is optional and selected explicitly with ``--ffmpeg_hwaccel``;
# encoding always uses CPU-based ``libx264``.
FFMPEG_DECODE = []
# Batch size retained for the per-clip fallback path.
BATCH_SIZE = 20


def _find_video_for_id(video_dir, video_id):
    """Find the MP4 whose filename prefix matches ``video_id``."""
    if not os.path.isdir(video_dir):
        return None
    for f in os.listdir(video_dir):
        if not f.lower().endswith(".mp4"):
            continue
        prefix = f.split(".")[0]
        if prefix == video_id:
            return os.path.join(video_dir, f)
    return None


def _extract_single_clip(video_path, start_frame, end_frame, output_path, fps, gpu_id=None):
    """Extract one clip using input-side seek to avoid decoding preceding frames."""
    start_sec = start_frame / fps
    duration_sec = (end_frame - start_frame + 1) / fps
    
    tmp_path = output_path + ".tmp"
    
    # Place ``-ss`` before ``-i`` for demuxer-level seeking.
    cmd = (
        ["ffmpeg", "-y"]
        + FFMPEG_DECODE
        + ["-ss", f"{start_sec:.6f}"]
        + ["-i", video_path]
        + ["-t", f"{duration_sec:.6f}"]
        + ["-c:v", "libx264", "-crf", "18", "-preset", "fast"]
        + ["-an", "-movflags", "+faststart", "-f", "mp4"]
        + [tmp_path]
    )
    
    # Select the decode GPU through the environment.
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    ret = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if ret.returncode != 0:
        # Remove a failed temporary output.
        for p in (tmp_path, output_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        raise RuntimeError(f"ffmpeg 切片失败 {output_path}: {ret.stderr or ret.returncode}")
    
    # Rename the completed temporary output atomically.
    try:
        os.replace(tmp_path, output_path)
    except OSError as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise RuntimeError(f"重命名失败 {tmp_path} -> {output_path}: {e}")


def _extract_clips_batch(video_path, clips_info, fps, gpu_id=None):
    """Extract clips individually with input-side seek."""
    for start_f, end_f, out_path in clips_info:
        _extract_single_clip(video_path, start_f, end_f, out_path, fps, gpu_id)


def _extract_group_clips(video_path, group_clips, shot_range, fps, gpu_id=None):
    """Extract every clip in one group with a single bounded ffmpeg decode.

    Input-side seek and a group-level duration avoid decoding unrelated media;
    trims are relative to the seek point. Returns the number of clips produced.
    """
    if not group_clips:
        return 0
    
    n_clips = len(group_clips)
    group_start_frame = shot_range[0]
    group_end_frame = shot_range[1]
    
    # Compute the group seek point and bounded decode duration.
    seek_sec = group_start_frame / fps
    duration_sec = (group_end_frame - group_start_frame + 1) / fps
    
    # Build trims relative to the group seek point.
    filter_parts = []
    for i, clip_info in enumerate(group_clips):
        start_f = clip_info["start_frame"]
        end_f = clip_info["end_frame"]
        
        # Times are in seconds relative to the seek point.
        rel_start_sec = (start_f - group_start_frame) / fps
        rel_end_sec = (end_f - group_start_frame + 1) / fps
        
        # Trim and reset timestamps for each output stream.
        filter_parts.append(
            f"[0:v]trim=start={rel_start_sec:.6f}:end={rel_end_sec:.6f},setpts=PTS-STARTPTS[v{i}]"
        )
    
    filter_complex = "; ".join(filter_parts)
    
    # Build the ffmpeg command.
    cmd = ["ffmpeg", "-y"]
    cmd += FFMPEG_DECODE
    cmd += ["-ss", f"{seek_sec:.6f}"]  # Input-side seek.
    cmd += ["-i", video_path]
    cmd += ["-t", f"{duration_sec:.6f}"]  # Bound the decode range.
    cmd += ["-filter_complex", filter_complex]
    
    # Add stream mapping and encoding options for each output.
    tmp_paths = []
    for i, clip_info in enumerate(group_clips):
        out_path = clip_info["output_path"]
        tmp_path = out_path + ".tmp"
        tmp_paths.append((tmp_path, out_path))
        
        cmd += [
            "-map", f"[v{i}]",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-an", "-movflags", "+faststart", "-f", "mp4",
            tmp_path
        ]
    
    # Select the decode GPU through the environment.
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Run ffmpeg.
    ret = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if ret.returncode != 0:
        # Remove every temporary output after failure.
        for tmp_path, out_path in tmp_paths:
            for p in (tmp_path, out_path):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
        raise RuntimeError(
            f"ffmpeg group 切片失败 (group has {n_clips} clips): "
            f"{ret.stderr or ret.returncode}"
        )
    
    # Atomically rename completed temporary outputs.
    for tmp_path, out_path in tmp_paths:
        try:
            os.replace(tmp_path, out_path)
        except OSError as e:
            # Remove outputs created before the rename failure.
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise RuntimeError(f"重命名失败 {tmp_path} -> {out_path}: {e}")
    
    return n_clips


def main():
    parser = argparse.ArgumentParser(description="按 clips_record 的 JSON 快速导出视频切片")
    parser.add_argument("--record_dir", type=str, required=True,
                        help="clips_record 目录，内含 *_clips.json")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="源视频所在目录，文件名须为 {video_id}.xxx.mp4")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出根目录，将生成 output_dir/{video_id}/group_{i}/clip*.mp4")
    parser.add_argument("--video_id", type=str, default=None,
                        help="只处理指定 video_id 的 JSON；不指定则处理 record_dir 下所有 *_clips.json")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行处理 movie 的 worker 数，默认 10")
    parser.add_argument("--ffmpeg_hwaccel", type=str, default="none",
                        choices=["none", "cuda"],
                        help="ffmpeg 硬件解码方式，默认 none；有 NVIDIA 环境时可设 cuda")
    args = parser.parse_args()

    global FFMPEG_DECODE
    FFMPEG_DECODE = [] if args.ffmpeg_hwaccel == "none" else ["-hwaccel", args.ffmpeg_hwaccel]

    record_dir = os.path.abspath(args.record_dir)
    video_dir = os.path.abspath(args.video_dir)
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(record_dir):
        print(f"错误: record_dir 不存在: {record_dir}", file=sys.stderr)
        return 1
    if not os.path.isdir(video_dir):
        print(f"错误: video_dir 不存在: {video_dir}", file=sys.stderr)
        return 1
    os.makedirs(output_dir, exist_ok=True)

    if args.video_id:
        json_path = os.path.join(record_dir, f"{args.video_id}_clips.json")
        if not os.path.isfile(json_path):
            print(f"错误: 未找到 {json_path}", file=sys.stderr)
            return 1
        json_files = [(args.video_id, json_path)]
    else:
        json_files = []
        for f in sorted(os.listdir(record_dir)):
            if f.endswith("_clips.json"):
                vid = f.replace("_clips.json", "")
                json_files.append((vid, os.path.join(record_dir, f)))

    # Collect movie tasks; GPU assignment happens later.
    movie_tasks_raw = []
    for video_id, json_path in json_files:
        video_path = _find_video_for_id(video_dir, video_id)
        if not video_path:
            # Video IDs match the filename prefix before the first dot.
            mp4_in_dir = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")] if os.path.isdir(video_dir) else []
            prefixes = [f.split(".")[0] for f in mp4_in_dir[:5]]
            print(f"[Skip] 未找到视频: {video_id} (需要 prefix=={repr(video_id)}); "
                  f"video_dir={video_dir}, 前几个 mp4 的 prefix: {prefixes}", file=sys.stderr)
            continue
        movie_tasks_raw.append((video_id, video_path, json_path))

    if not movie_tasks_raw:
        print("无 movie 需要导出。")
        return 0

    # Resolve available GPU IDs.
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_devices:
        gpu_list = [g.strip() for g in cuda_devices.split(",") if g.strip()]
    else:
        # Default to GPU 0 when CUDA_VISIBLE_DEVICES is unset.
        gpu_list = ["0"]
    
    n_gpus = len(gpu_list)
    n_workers = min(args.workers, len(movie_tasks_raw))
    
    # Assign tasks to GPUs round-robin.
    movie_tasks = []
    for i, (video_id, video_path, json_path) in enumerate(movie_tasks_raw):
        gpu_id = gpu_list[i % n_gpus]
        movie_tasks.append((video_id, video_path, json_path, gpu_id))
    
    print(f"[Parallel] workers={n_workers}, GPUs={gpu_list}, 共 {len(movie_tasks)} 个 movie (按 group 处理)")

    def do_movie(task):
        video_id, video_path, json_path, gpu_id = task
        with open(json_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        fps = float(data.get("fps", 24))
        video_out = os.path.join(output_dir, video_id)
        groups = data.get("groups", [])

        if not groups:
            return video_id, 0, 0, "skipped_empty", 0, 0

        # Inspect completed groups for resume state.
        last_group_idx = max(g.get("group_index", 0) for g in groups)
        last_group_dir = os.path.join(video_out, f"group_{last_group_idx}")

        # A last_clip in the final group marks the movie complete.
        if os.path.isdir(last_group_dir):
            last_clip_path = os.path.join(last_group_dir, "last_clip.mp4")
            if os.path.isfile(last_clip_path):
                n_groups = len(groups)
                n_clips = sum(len(g.get("clips", [])) for g in groups)
                return video_id, n_groups, n_clips, "skipped_complete", n_clips, 0

        # Existing group_0 indicates resumable output.
        group_0_dir = os.path.join(video_out, "group_0")
        resume_from_group = 0

        if os.path.isdir(group_0_dir):
            # Resume after the last group containing last_clip.
            sorted_groups = sorted(groups, key=lambda g: g.get("group_index", 0))
            last_complete_group = -1

            for group in sorted_groups:
                g_idx = group.get("group_index", 0)
                group_dir = os.path.join(video_out, f"group_{g_idx}")
                last_clip_path = os.path.join(group_dir, "last_clip.mp4")

                if os.path.isdir(group_dir) and os.path.isfile(last_clip_path):
                    last_complete_group = g_idx
                else:
                    break

            resume_from_group = last_complete_group + 1

        # Count groups still requiring work.
        sorted_groups = sorted(groups, key=lambda g: g.get("group_index", 0))
        groups_to_process = []
        skipped_clips = 0
        skipped_groups = 0
        total_clips_to_process = 0
        
        for group in sorted_groups:
            g_idx = group.get("group_index", 0)
            group_dir = os.path.join(video_out, f"group_{g_idx}")
            
            # Skip groups marked complete by last_clip.
            if g_idx < resume_from_group:
                skipped_clips += len(group.get("clips", []))
                skipped_groups += 1
                continue
            
            # Read group boundaries.
            shot_range = group.get("shot_range", [0, 0])
            clips_in_group = group.get("clips", [])
            
            if not clips_in_group:
                continue
            
            # Build group clip descriptors.
            group_clips = []
            for clip in clips_in_group:
                start_f = clip["start_frame"]
                end_f = clip["end_frame"]
                name = clip.get("name", "clip")
                out_name = f"{name}.mp4"
                out_path = os.path.join(group_dir, out_name)
                
                group_clips.append({
                    "name": name,
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "output_path": out_path
                })
            
            groups_to_process.append({
                "group_index": g_idx,
                "group_dir": group_dir,
                "shot_range": shot_range,
                "clips": group_clips
            })
            total_clips_to_process += len(group_clips)
        
        n_groups_to_process = len(groups_to_process)
        
        if n_groups_to_process == 0:
            # Nothing remains for this movie.
            return video_id, 0, 0, "skipped_empty", skipped_clips, 0
        
        # Process each group with one ffmpeg invocation.
        start_time = time.time()
        processed_clips = 0
        processed_groups = 0
        
        for group_info in groups_to_process:
            g_idx = group_info["group_index"]
            group_dir = group_info["group_dir"]
            shot_range = group_info["shot_range"]
            group_clips = group_info["clips"]
            
            # Create the group output directory.
            os.makedirs(group_dir, exist_ok=True)
            
            group_start_time = time.time()
            
            # Extract the group in one bounded decode.
            clips_processed = _extract_group_clips(
                video_path, group_clips, shot_range, fps, gpu_id
            )
            
            group_end_time = time.time()
            group_duration = group_end_time - group_start_time
            
            processed_clips += clips_processed
            processed_groups += 1
            
            # Estimate throughput and remaining time.
            elapsed = group_end_time - start_time
            clips_per_sec = processed_clips / elapsed if elapsed > 0 else 0
            remaining_clips = total_clips_to_process - processed_clips
            eta_sec = remaining_clips / clips_per_sec if clips_per_sec > 0 else 0
            
            # Report group-level progress.
            group_speed = clips_processed / group_duration if group_duration > 0 else 0
            print(f"    [{video_id}@GPU{gpu_id}] group {processed_groups}/{n_groups_to_process} (g{g_idx}): "
                  f"{processed_clips}/{total_clips_to_process} clips ({100*processed_clips/total_clips_to_process:.1f}%), "
                  f"本组 {clips_processed} clips in {group_duration:.1f}s ({group_speed:.2f} clips/s), "
                  f"平均 {clips_per_sec:.2f} clips/s, "
                  f"ETA {eta_sec:.0f}s", flush=True)
        
        status = "resumed" if skipped_groups > 0 else "new"
        total_time = time.time() - start_time
        return video_id, n_groups_to_process, processed_clips, status, skipped_clips, total_time

    success_movies = 0
    skipped_movies = 0
    resumed_movies = 0
    failed_movies = 0
    total_groups = 0
    total_clips = 0
    total_skipped_clips = 0
    total_process_time = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(do_movie, t): t for t in movie_tasks}
        for future in as_completed(futures):
            t = futures[future]
            try:
                result = future.result()
                vid = result[0]
                n_groups = result[1]
                n_clips = result[2]
                status = result[3]
                skipped_clips = result[4] if len(result) > 4 else 0
                movie_time = result[5] if len(result) > 5 else 0

                total_groups += n_groups
                total_clips += n_clips
                total_skipped_clips += skipped_clips
                total_process_time += movie_time

                if status == "skipped_complete":
                    skipped_movies += 1
                    print(f"  [SKIP] {vid} 已完成 ({n_groups} groups, {n_clips} clips)")
                elif status == "skipped_empty":
                    skipped_movies += 1
                    print(f"  [SKIP] {vid} 无 groups")
                elif status == "resumed":
                    resumed_movies += 1
                    success_movies += 1
                    avg_speed = n_clips / movie_time if movie_time > 0 else 0
                    print(f"  [RESUME] {vid} 断点续传 ({n_groups} groups, 新增 {n_clips} clips, 跳过 {skipped_clips} clips) "
                          f"耗时 {movie_time:.1f}s, 平均 {avg_speed:.2f} clips/s")
                else:
                    success_movies += 1
                    avg_speed = n_clips / movie_time if movie_time > 0 else 0
                    print(f"  [OK] {vid} ({n_groups} groups, {n_clips} clips) "
                          f"耗时 {movie_time:.1f}s, 平均 {avg_speed:.2f} clips/s")
            except Exception as e:
                failed_movies += 1
                print(f"  [FAIL] {t[0]}: {e}", file=sys.stderr)

    avg_total_speed = total_clips / total_process_time if total_process_time > 0 else 0
    print(f"\n完成: 成功 {success_movies} 个 movie (含 {resumed_movies} 个续传), "
          f"跳过 {skipped_movies} 个已完成, 失败 {failed_movies} 个 movie")
    print(f"      共 {total_groups} groups, {total_clips} 新 clips, 跳过 {total_skipped_clips} 已有 clips")
    print(f"      累计处理时间 {total_process_time:.1f}s, 平均速率 {avg_total_speed:.2f} clips/s")
    return 0 if failed_movies == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
