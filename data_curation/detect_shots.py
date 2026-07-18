"""Detect shots and write brightness-filtered groups to CSV.

Video IDs must match ``Top`` plus three digits. The secondary filter keeps
shots whose sampled frames average at least one eighth of the dtype maximum.
Invalid or failed paths are logged separately.
"""
import os
import re
import json
import subprocess
import shutil
import sys
import traceback
import argparse
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.environ.get("SLOTMEM_CURATION_TMP", tempfile.gettempdir())
# Video IDs are the prefix before the first dot: Top plus three digits.
VIDEO_ID_PATTERN = re.compile(r"^Top\d{3}$")

# Import the official TransNetV2 inference implementation.
_TRANSNET_INFERENCE = os.path.join(SCRIPT_DIR, "TransNetV2", "inference")
if _TRANSNET_INFERENCE not in sys.path:
    sys.path.insert(0, _TRANSNET_INFERENCE)
import transnetv2


def _resolve_transnet_weights(model_path):
    if model_path and os.path.isdir(model_path):
        for root, _, files in os.walk(model_path):
            if "saved_model.pb" in files:
                return root
        fallback = os.path.join(SCRIPT_DIR, "TransNetV2", "inference", "transnetv2-weights")
        if os.path.isfile(os.path.join(fallback, "saved_model.pb")):
            return fallback
    default = os.path.join(SCRIPT_DIR, "TransNetV2", "inference", "transnetv2-weights")
    if os.path.isfile(os.path.join(default, "saved_model.pb")):
        return default
    raise FileNotFoundError("未找到 TransNetV2 权重...")

# =========================================================
# Compute mean frame luminance relative to the dtype maximum.
# =========================================================
def _max_brightness_from_frame(frame):
    """Return the theoretical maximum value for the frame dtype."""
    if np.issubdtype(frame.dtype, np.integer):
        return float(np.iinfo(frame.dtype).max)
    if np.issubdtype(frame.dtype, np.floating):
        return 1.0
    return 255.0


def _frame_brightness(rgb_frame):
    """Return mean luminance for an RGB frame shaped ``[H,W,3]``."""
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    return float(np.mean(gray))


# =========================================================
# Video processing pipeline.
# =========================================================
class VideoProcessor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self._weights_dir = _resolve_transnet_weights(model_dir)
        self._transnet = None

    def _get_model(self):
        if self._transnet is None:
            print(f"[Model] 正在从 {self._weights_dir} 加载 TransNet V2...")
            self._transnet = transnetv2.TransNetV2(model_dir=self._weights_dir)
        return self._transnet

    def _get_video_fps(self, video_path):
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            if result.returncode != 0:
                print(f"[Preprocess] ffprobe 无法读取帧率，将尝试转码: {result.stderr or result.returncode}", file=sys.stderr)
            return None
        rate = result.stdout.strip()
        if "/" in rate:
            num, den = rate.split("/", 1)
            try:
                return float(num) / float(den)
            except (ValueError, ZeroDivisionError) as exc:
                print(f"[Preprocess] 无法解析 ffprobe 帧率 {rate!r}: {exc}", file=sys.stderr)
                return None
        try:
            return float(rate)
        except ValueError as exc:
            print(f"[Preprocess] 无法解析 ffprobe 帧率 {rate!r}: {exc}", file=sys.stderr)
            return None

    def convert_to_24fps(self, video_path):
        fps = self._get_video_fps(video_path)
        if fps is not None and abs(fps - 24.0) < 0.05:
            print(f"[Preprocess] 已是 24fps (检测到 {fps:.2f})，跳过转码: {video_path}")
            return True

        print(f"[Preprocess] 正在转码为 24fps: {video_path}")
        directory = os.path.dirname(video_path)
        filename = os.path.basename(video_path)
        name, ext = os.path.splitext(filename)
        temp_output = os.path.join(directory, f"{name}_temp_24fps{ext}")
        os.makedirs(TMP_DIR, exist_ok=True)
        tmp_path = os.path.join(TMP_DIR, f"{name}_temp_24fps{ext}.tmp")
        # Decode on GPU and encode on CPU because H20 lacks NVENC.
        cmd = [
            "ffmpeg", "-y", "-hwaccel", "cuda", "-i", video_path, "-r", "24",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-f", "mp4", tmp_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[Error] 转码失败，ffmpeg 退出码 {result.returncode}", file=sys.stderr)
            print(result.stderr or "(无 stderr 输出)", file=sys.stderr)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise RuntimeError(f"24fps 转码失败: {result.stderr or result.returncode}")
        try:
            shutil.move(tmp_path, temp_output)
        except OSError as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise RuntimeError(f"从 tmp 目录移动转码文件失败: {e}")
        try:
            shutil.move(temp_output, video_path)
        except OSError as e:
            raise RuntimeError(f"转码结果覆盖原文件失败: {e}")
        print(f"[Preprocess] 转码成功，原文件已更新。")
        return True

    def predict_shots(self, video_path):
        model = self._get_model()
        print(f"[Inference] 正在分析镜头切分点...")
        _video_frames, single_frame_predictions, _all_frame_predictions = model.predict_video(video_path)
        scenes = transnetv2.TransNetV2.predictions_to_scenes(single_frame_predictions, threshold=0.5).tolist()
        filtered_scenes = [s for s in scenes if (s[1] - s[0] + 1) >= 161]
        return scenes, filtered_scenes

    def extract_shot_to_mp4(self, video_path, start_frame, end_frame, output_path, fps=24):
        start_sec = start_frame / fps
        duration_sec = (end_frame - start_frame + 1) / fps
        # Write a temporary file and rename atomically after its trailer completes.
        tmp_path = output_path + ".tmp"
        # Decode on GPU and encode on CPU because H20 lacks NVENC.
        cmd = [
            "ffmpeg", "-y", "-hwaccel", "cuda", "-i", video_path,
            "-ss", str(start_sec), "-t", str(duration_sec),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-movflags", "+faststart", "-an", "-f", "mp4", tmp_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            err = result.stderr or "(无 stderr 输出)"
            print(f"[Error] 导出镜头失败: {output_path}", file=sys.stderr)
            print(err, file=sys.stderr)
            for p in (tmp_path, output_path):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            raise RuntimeError(f"ffmpeg 导出镜头失败 (退出码 {result.returncode})\n{err}")
        try:
            os.replace(tmp_path, output_path)
        except OSError as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise RuntimeError(f"无法将临时文件重命名为 {output_path}: {e}")

    def extract_target_frames_sequentially(self, video_path, frame_indices):
        """Read the video sequentially and return requested frames."""
        print(f"[Preprocess] 正在使用 OpenCV 从视频中提取 {len(frame_indices)} 张关键帧...")
        target_indices = sorted(list(set(frame_indices)))
        if not target_indices: return {}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")
        frames_dict = {}
        current_target_idx = 0
        num_targets = len(target_indices)

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or current_target_idx >= num_targets:
                break
            if idx == target_indices[current_target_idx]:
                frames_dict[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                current_target_idx += 1
            idx += 1
        cap.release()
        return frames_dict

def _video_id_from_filename(filepath):
    """Return the filename prefix before the first dot as the video ID."""
    basename = os.path.basename(filepath)
    name_no_ext = os.path.splitext(basename)[0]
    return name_no_ext.split(".")[0] if name_no_ext else basename


def _is_valid_video_id(video_id):
    """Return whether the ID is ``Top`` followed by three digits."""
    return bool(video_id and VIDEO_ID_PATTERN.match(video_id))


def _process_one_video(video_path, processor, skip_convert_24fps=False):
    """Return detected, filtered, brightness-filtered groups, and FPS."""
    if not skip_convert_24fps:
        processor.convert_to_24fps(video_path)
        fps = 24.0
    else:
        fps = processor._get_video_fps(video_path) or 24.0
    shot_groups, filtered_shot_groups = processor.predict_shots(video_path)
    if not filtered_shot_groups:
        return shot_groups, filtered_shot_groups, [], fps

    target_frames = []
    for start_f, end_f in filtered_shot_groups:
        mid_f = (start_f + end_f) // 2
        target_frames.extend([start_f, mid_f, end_f])
    frames_dict = processor.extract_target_frames_sequentially(video_path, target_frames)
    frame_brightness = {idx: _frame_brightness(img) for idx, img in frames_dict.items()}

    if frames_dict:
        max_brightness = _max_brightness_from_frame(next(iter(frames_dict.values())))
    else:
        max_brightness = 255.0
    t_1_8 = max_brightness / 8.0  # Require at least one eighth of the dtype maximum.

    shot_avg_brightness = []
    for start_f, end_f in filtered_shot_groups:
        mid_f = (start_f + end_f) // 2
        b1 = frame_brightness.get(start_f, 0.0)
        b2 = frame_brightness.get(mid_f, 0.0)
        b3 = frame_brightness.get(end_f, 0.0)
        avg_b = (b1 + b2 + b3) / 3.0
        shot_avg_brightness.append((start_f, end_f, avg_b))

    filtered_avg_brightness_ge_1_8 = [(s, e) for s, e, avg_b in shot_avg_brightness if avg_b >= t_1_8]
    return shot_groups, filtered_shot_groups, filtered_avg_brightness_ge_1_8, fps


CLIP_FRAMES = 81  # Adjacent clips share one boundary frame.


def _build_clip_ranges(filtered_ge_1_8):
    """Serialize brightness-filtered ranges as overlapping 81-frame clips."""
    groups = []
    for g_idx, (start_f, end_f) in enumerate(filtered_ge_1_8):
        clips = []
        s = start_f
        order = 0
        while s <= end_f:
            e = min(s + CLIP_FRAMES - 1, end_f)
            num_frames = e - s + 1
            if num_frames >= CLIP_FRAMES:
                order += 1
                name = f"clip{order}"
            else:
                name = "last_clip"
            clips.append({"name": name, "start_frame": s, "end_frame": e})
            s = e  # The next clip begins on the current clip's final frame.
            if num_frames < CLIP_FRAMES:
                break
        groups.append({
            "group_index": g_idx,
            "shot_range": [start_f, end_f],
            "clips": clips,
        })
    return groups


def _export_group_clips(processor, video_path, start_f, end_f, group_dir, fps=24):
    """Export overlapping 81-frame clips and name a short remainder ``last_clip``."""
    os.makedirs(group_dir, exist_ok=True)
    order = 0
    s = start_f
    while s <= end_f:
        e = min(s + CLIP_FRAMES - 1, end_f)
        num_frames = e - s + 1
        if num_frames >= CLIP_FRAMES:
            order += 1
            out_name = f"clip{order}.mp4"
        else:
            out_name = "last_clip.mp4"
        out_path = os.path.join(group_dir, out_name)
        processor.extract_shot_to_mp4(video_path, s, e, out_path, fps=fps)
        s = e  # The next clip begins on the current clip's final frame.
        if num_frames < CLIP_FRAMES:
            break


def _process_one_video_standalone(video_path, filename, model_path, output_dir, no_convert_24fps, export_clips):
    """Process one video, returning either its row or failed path."""
    video_id = _video_id_from_filename(video_path)
    processor = VideoProcessor(model_path)
    try:
        shot_groups, filtered_shot_groups, filtered_ge_1_8, fps = _process_one_video(
            video_path, processor, skip_convert_24fps=no_convert_24fps
        )
        row = {
            "video_id": video_id,
            "shot_groups": str(shot_groups),
            "filtered_shot_groups": str([(s, e) for s, e in filtered_shot_groups]),
            "filtered_shot_groups_avg_brightness_ge_1_8": str(filtered_ge_1_8),
        }
        # Record clip boundaries regardless of whether clips are exported.
        clips_record_dir = os.path.join(output_dir, "clips_record")
        os.makedirs(clips_record_dir, exist_ok=True)
        clip_ranges = _build_clip_ranges(filtered_ge_1_8)
        record_path = os.path.join(clips_record_dir, f"{video_id}_clips.json")
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump({
                "video_id": video_id,
                "fps": fps,
                "groups": clip_ranges,
            }, f, ensure_ascii=False, indent=2)
        if export_clips and filtered_ge_1_8:
            video_large_dir = os.path.join(output_dir, video_id)
            for g_idx, (start_f, end_f) in enumerate(filtered_ge_1_8):
                group_dir = os.path.join(video_large_dir, f"group_{g_idx}")
                _export_group_clips(processor, video_path, start_f, end_f, group_dir, fps=fps)
        return (row, None)
    except Exception as e:
        print(f"[Error] {video_id} 处理失败: {e}", file=sys.stderr)
        traceback.print_exc()
        return (None, video_path)


def main():
    parser = argparse.ArgumentParser(description="批量镜头检测 + 亮度二次过滤，结果汇总 CSV")
    parser.add_argument("--path", type=str, required=True,
                        help="输入目录（其下所有 .mp4）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录，CSV 写入 output_dir/processed_shots.csv；异常路径写入 output_dir/abnormal_mp4name.log。默认与 --path 相同")
    parser.add_argument("--model_path", type=str, default=None, help="TransNetV2 权重目录")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="最多处理的视频数，不指定则处理完整文件夹中所有合法 MP4")
    parser.add_argument("--workers", type=int, default=3,
                        help="以数据(每个 MP4)为单位的并行数，默认 3")
    parser.add_argument("--start_index", type=int, default=0,
                        help="从第几个（0-based）合法视频开始处理，默认 0。例如 2 表示跳过前两个，从第三个开始")
    parser.add_argument("--no_convert_24fps", action="store_true",
                        help="不进行 24fps 转码，直接用视频当前帧率处理（用于某视频转码报错时从该视频开始跑）")
    parser.add_argument("--export_clips", action="store_true",
                        help="对最终过滤得到的每个 group 按 81 帧一段导出 clip：每视频一大文件夹(video_id)，每 group 一小文件夹，内为 clip1.mp4, clip2.mp4, ... last_clip.mp4")
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = os.path.join(SCRIPT_DIR, "TransNetV2", "inference", "transnetv2-weights")
    input_dir = os.path.abspath(args.path)
    if not os.path.isdir(input_dir):
        print(f"错误: 不是目录或不存在: {input_dir}")
        return 1

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else input_dir
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "processed_shots.csv")
    abnormal_log_path = os.path.join(output_dir, "abnormal_mp4name.log")

    mp4_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")])
    if not mp4_files:
        print(f"未在 {input_dir} 下找到 .mp4 文件")
        return 0

            # Log the full path when the video ID is invalid.
    abnormal_paths = []
    for f in mp4_files:
        video_path = os.path.join(input_dir, f)
        video_id = _video_id_from_filename(video_path)
        if not _is_valid_video_id(video_id):
            abnormal_paths.append(video_path)
    if abnormal_paths:
        with open(abnormal_log_path, "w", encoding="utf-8") as log:
            for p in abnormal_paths:
                log.write(p + "\n")
        print(f"[Abnormal] 共 {len(abnormal_paths)} 个文件标识不符合 Top{{3位数字}}，已写入: {abnormal_log_path}")

    # Process valid IDs from start_index, with no limit when max_videos is unset.
    to_process = [f for f in mp4_files if os.path.join(input_dir, f) not in abnormal_paths]
    to_process = to_process[args.start_index:]
    if args.max_videos is not None:
        to_process = to_process[: args.max_videos]
    if not to_process:
        print("无符合标识的视频可处理")
        return 0

    n_workers = min(args.workers, len(to_process))
    print(f"[Parallel] 以数据为单位并行，workers={n_workers}，共 {len(to_process)} 个视频")

    def task(item):
        i, filename = item
        video_path = os.path.join(input_dir, filename)
        return (i, _process_one_video_standalone(
            video_path, filename, args.model_path, output_dir,
            args.no_convert_24fps, args.export_clips
        ))

    results = []  # (index, row or None, failed_path or None)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(task, (i, f)): i for i, f in enumerate(to_process)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                i, (row, failed_path) = future.result()
                results.append((i, row, failed_path))
            except Exception as e:
                print(f"[Error] worker 异常: {e}", file=sys.stderr)
                traceback.print_exc()
                filename = to_process[idx]
                video_path = os.path.join(input_dir, filename)
                results.append((idx, None, video_path))

    results.sort(key=lambda x: x[0])
    rows = [r[1] for r in results if r[1] is not None]
    failed_paths = [r[2] for r in results if r[2] is not None]
    if failed_paths:
        with open(abnormal_log_path, "a", encoding="utf-8") as log:
            for p in failed_paths:
                log.write(p + "\n")
        print(f"[Abnormal] {len(failed_paths)} 个视频处理失败，已追加至: {abnormal_log_path}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"\n[Success] 共 {len(rows)} 条结果已写入: {csv_path}")
    else:
        print("\n无有效结果，未写入 CSV")
    return 0

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
