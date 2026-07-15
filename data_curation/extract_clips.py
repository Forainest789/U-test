"""
根据 clips_record 下的 *_clips.json 快速导出视频切片，目录结构与 batch 脚本一致：
  output_dir / {video_id} / group_{i} / clip1.mp4, clip2.mp4, ... last_clip.mp4

优化策略（按 group 处理）：
1. 对于每个 group，使用一次 ffmpeg 调用处理所有 clips
2. 使用 -ss 在 -i 之前实现 demuxer 级别 seek，跳到 group 起始位置
3. 使用 -t 限制读取范围为 group 的帧范围
4. 使用 filter_complex + trim 将解码后的帧分割成多个输出
5. 由于 group 内 clips 帧范围连续，trim 的时间范围很小，效率高
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 可选：GPU 解码（无 NVENC 时用 CPU 编码），由 --ffmpeg_hwaccel 设置。
FFMPEG_DECODE = []
# 旧版按 batch 分批时的大小（现在按 group 处理，此参数仅供 fallback 函数使用）
BATCH_SIZE = 20


def _find_video_for_id(video_dir, video_id):
    """在 video_dir 下查找文件名第一个点前为 video_id 的 mp4。"""
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
    """
    使用 ffmpeg -ss 输入端 seek 提取单个 clip。
    -ss 在 -i 之前可以实现 demuxer 级别的 seek，避免解码不需要的帧。
    
    Args:
        gpu_id: 指定使用的 GPU ID，None 表示使用默认 GPU
    """
    start_sec = start_frame / fps
    duration_sec = (end_frame - start_frame + 1) / fps
    
    tmp_path = output_path + ".tmp"
    
    # -ss 放在 -i 之前实现输入端 seek（关键！）
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
    
    # 设置环境变量指定 GPU
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    ret = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if ret.returncode != 0:
        # 清理临时文件
        for p in (tmp_path, output_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        raise RuntimeError(f"ffmpeg 切片失败 {output_path}: {ret.stderr or ret.returncode}")
    
    # 重命名临时文件
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
    """
    逐个提取 clips，每个 clip 使用 -ss 输入端 seek。
    clips_info: list of (start_frame, end_frame, output_path)
    gpu_id: 指定使用的 GPU ID
    """
    for start_f, end_f, out_path in clips_info:
        _extract_single_clip(video_path, start_f, end_f, out_path, fps, gpu_id)


def _extract_group_clips(video_path, group_clips, shot_range, fps, gpu_id=None):
    """
    使用一次 ffmpeg 调用提取一个 group 内的所有 clips。
    
    原理：
    1. 使用 -ss 跳到 group 的起始位置（demuxer 级别 seek，跳过不需要的数据）
    2. 使用 -t 限制读取范围为 group 的长度
    3. 使用 filter_complex + trim 将解码后的帧分割成多个输出
    4. 由于已经 seek 到 group 开始，trim 的时间范围很小，效率高
    
    Args:
        video_path: 源视频路径
        group_clips: list of dict，每个 dict 包含 name, start_frame, end_frame, output_path
        shot_range: [start_frame, end_frame] group 的整体帧范围
        fps: 视频帧率
        gpu_id: 指定使用的 GPU ID
    
    Returns:
        处理的 clip 数量
    """
    if not group_clips:
        return 0
    
    n_clips = len(group_clips)
    group_start_frame = shot_range[0]
    group_end_frame = shot_range[1]
    
    # 计算 seek 位置和读取时长
    seek_sec = group_start_frame / fps
    duration_sec = (group_end_frame - group_start_frame + 1) / fps
    
    # 构建 filter_complex
    # 每个 clip 的 trim 时间是相对于 seek 点（group_start_frame）的
    filter_parts = []
    for i, clip_info in enumerate(group_clips):
        start_f = clip_info["start_frame"]
        end_f = clip_info["end_frame"]
        
        # 相对于 seek 点的时间（秒）
        rel_start_sec = (start_f - group_start_frame) / fps
        rel_end_sec = (end_f - group_start_frame + 1) / fps
        
        # 构建 trim filter
        # [0:v] -> trim -> setpts 重置时间戳 -> [v{i}]
        filter_parts.append(
            f"[0:v]trim=start={rel_start_sec:.6f}:end={rel_end_sec:.6f},setpts=PTS-STARTPTS[v{i}]"
        )
    
    filter_complex = "; ".join(filter_parts)
    
    # 构建 ffmpeg 命令
    cmd = ["ffmpeg", "-y"]
    cmd += FFMPEG_DECODE
    cmd += ["-ss", f"{seek_sec:.6f}"]  # 输入端 seek（关键！）
    cmd += ["-i", video_path]
    cmd += ["-t", f"{duration_sec:.6f}"]  # 限制读取范围
    cmd += ["-filter_complex", filter_complex]
    
    # 为每个输出添加 -map 和编码参数
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
    
    # 设置环境变量指定 GPU
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 执行 ffmpeg
    ret = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if ret.returncode != 0:
        # 清理所有临时文件
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
    
    # 重命名所有临时文件
    for tmp_path, out_path in tmp_paths:
        try:
            os.replace(tmp_path, out_path)
        except OSError as e:
            # 清理已创建的文件
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

    # 收集所有 movie 任务
    # (video_id, video_path, json_path) - gpu_id 稍后分配
    movie_tasks_raw = []
    for video_id, json_path in json_files:
        video_path = _find_video_for_id(video_dir, video_id)
        if not video_path:
            # 调试信息：脚本用「文件名第一个点前的部分」匹配 video_id，便于排查路径/命名问题
            mp4_in_dir = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")] if os.path.isdir(video_dir) else []
            prefixes = [f.split(".")[0] for f in mp4_in_dir[:5]]
            print(f"[Skip] 未找到视频: {video_id} (需要 prefix=={repr(video_id)}); "
                  f"video_dir={video_dir}, 前几个 mp4 的 prefix: {prefixes}", file=sys.stderr)
            continue
        movie_tasks_raw.append((video_id, video_path, json_path))

    if not movie_tasks_raw:
        print("无 movie 需要导出。")
        return 0

    # 获取可用的 GPU 列表
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_devices:
        gpu_list = [g.strip() for g in cuda_devices.split(",") if g.strip()]
    else:
        # 如果没有设置 CUDA_VISIBLE_DEVICES，尝试检测可用 GPU 数量
        gpu_list = ["0"]  # 默认使用 GPU 0
    
    n_gpus = len(gpu_list)
    n_workers = min(args.workers, len(movie_tasks_raw))
    
    # 给每个任务分配 GPU（轮询方式）
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

        # === 断点续传检查 ===
        # 找到最后一个 group 的索引
        last_group_idx = max(g.get("group_index", 0) for g in groups)
        last_group_dir = os.path.join(video_out, f"group_{last_group_idx}")

        # 检查最后一个 group 是否存在 last_clip.mp4，存在则认为该 movie 已完成
        if os.path.isdir(last_group_dir):
            last_clip_path = os.path.join(last_group_dir, "last_clip.mp4")
            if os.path.isfile(last_clip_path):
                n_groups = len(groups)
                n_clips = sum(len(g.get("clips", [])) for g in groups)
                return video_id, n_groups, n_clips, "skipped_complete", n_clips, 0

        # 检查 group_0 是否存在，决定是否需要断点续传
        group_0_dir = os.path.join(video_out, "group_0")
        resume_from_group = 0

        if os.path.isdir(group_0_dir):
            # 找到有 last_clip.mp4 的最后一个 group，从下一个 group 开始继续
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

        # 统计需要处理的 groups
        sorted_groups = sorted(groups, key=lambda g: g.get("group_index", 0))
        groups_to_process = []
        skipped_clips = 0
        skipped_groups = 0
        total_clips_to_process = 0
        
        for group in sorted_groups:
            g_idx = group.get("group_index", 0)
            group_dir = os.path.join(video_out, f"group_{g_idx}")
            
            # 跳过已完成的 group（以 last_clip.mp4 为标志）
            if g_idx < resume_from_group:
                skipped_clips += len(group.get("clips", []))
                skipped_groups += 1
                continue
            
            # 构建 group 信息
            shot_range = group.get("shot_range", [0, 0])
            clips_in_group = group.get("clips", [])
            
            if not clips_in_group:
                continue
            
            # 构建 group_clips 列表
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
            # 没有需要处理的 group
            return video_id, 0, 0, "skipped_empty", skipped_clips, 0
        
        # 按 group 处理，每个 group 一次 ffmpeg 调用
        start_time = time.time()
        processed_clips = 0
        processed_groups = 0
        
        for group_info in groups_to_process:
            g_idx = group_info["group_index"]
            group_dir = group_info["group_dir"]
            shot_range = group_info["shot_range"]
            group_clips = group_info["clips"]
            
            # 创建 group 目录
            os.makedirs(group_dir, exist_ok=True)
            
            group_start_time = time.time()
            
            # 使用优化的 group 处理函数
            clips_processed = _extract_group_clips(
                video_path, group_clips, shot_range, fps, gpu_id
            )
            
            group_end_time = time.time()
            group_duration = group_end_time - group_start_time
            
            processed_clips += clips_processed
            processed_groups += 1
            
            # 计算速率和预计剩余时间
            elapsed = group_end_time - start_time
            clips_per_sec = processed_clips / elapsed if elapsed > 0 else 0
            remaining_clips = total_clips_to_process - processed_clips
            eta_sec = remaining_clips / clips_per_sec if clips_per_sec > 0 else 0
            
            # 输出进度（按 group 显示）
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
