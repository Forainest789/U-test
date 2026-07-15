#!/usr/bin/env python3
"""
重新计算 clips_record 下所有 JSON 的 clips 划分，改为首尾重叠模式。
每个 clip 的首帧 = 上一个 clip 的尾帧。
输出到新的 clips_record_cross 文件夹。

可选：对 candidate_groups.csv 与 character_lists 做 81 帧过滤——
仅保留实际视频帧数为 81 的 candidate 与 overlap clip，并从 CSV 和 JSON 中移除其余项。
"""
import os
import json
import argparse
import csv
import subprocess

CLIP_FRAMES = 81


def get_video_num_frames(path):
    """ffprobe 获取视频帧数，失败或异常返回 None。"""
    if not path or not os.path.isfile(path):
        return None
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,r_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=30)
        data = json.loads(out.decode("utf-8"))
    except Exception:
        return None
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    if not streams:
        return None
    s = streams[0]
    r = s.get("r_frame_rate", "24/1")
    if "/" in r:
        num, den = r.split("/", 1)
        fps = float(num) / float(den) if float(den) != 0 else 24.0
    else:
        fps = float(r) if r else 24.0
    nb_frames = s.get("nb_frames")
    if nb_frames is not None:
        try:
            return int(nb_frames)
        except (ValueError, TypeError):
            pass
    duration = float(fmt.get("duration", 0) or 0)
    if duration > 0 and fps > 0:
        return int(round(duration * fps))
    return None


def build_valid_81_frame_set(video_root, video_ids, group_clip_tuples):
    """对 (video_id, group_index, clip_index) 列表检查视频是否为 81 帧，返回满足的集合。"""
    valid = set()
    try:
        from tqdm import tqdm
        it = tqdm(group_clip_tuples, desc="检查视频帧数")
    except ImportError:
        it = group_clip_tuples
    for (video_id, group_index, clip_index) in it:
        path = os.path.join(
            video_root, video_id, f"group_{group_index}", f"clip{clip_index}.mp4"
        )
        n = get_video_num_frames(path)
        if n == CLIP_FRAMES:
            valid.add((video_id, group_index, clip_index))
    return valid


def filter_candidate_and_overlap_by_81_frames(
    candidate_groups_csv,
    character_lists_dir,
    video_root,
    required_frames=CLIP_FRAMES,
):
    """
    从 CSV 与 character_lists JSON 中收集所有 (video_id, group_index, clip_index)，
    仅保留实际视频帧数 == required_frames 的 clip，写回 CSV 和 JSON。
    """
    video_root = os.path.abspath(video_root)
    character_lists_dir = os.path.abspath(character_lists_dir)

    # 1) 从 CSV 收集所有 (video_id, group_index, clip_index)
    csv_rows = []
    group_clip_tuples = set()
    with open(candidate_groups_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            video_id = (row.get("video_id") or "").strip()
            try:
                g = int(row.get("group_index", 0))
            except (ValueError, TypeError):
                continue
            cand = (row.get("candidate_clips") or "").strip()
            if not cand:
                continue
            indices = [int(x.strip()) for x in cand.split("|") if x.strip().isdigit()]
            csv_rows.append((video_id, g, indices))
            for ci in indices:
                group_clip_tuples.add((video_id, g, ci))

    # 2) 从 JSON 收集所有 (video_id, group_index, clip_index)
    video_ids_from_csv = {r[0] for r in csv_rows}
    json_files = [f for f in os.listdir(character_lists_dir) if f.endswith(".json")]
    for jf in json_files:
        video_id = jf.replace(".json", "")
        path = os.path.join(character_lists_dir, jf)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        for shot in data.get("shots", []):
            g = shot.get("group_index")
            if g is None:
                continue
            for c in shot.get("clips", []):
                ci = c.get("clip_index")
                if ci is not None:
                    group_clip_tuples.add((video_id, g, ci))

    # 3) 检查每个 clip 视频帧数，得到合法集合
    tuples_list = list(group_clip_tuples)
    video_ids_all = list({t[0] for t in tuples_list})
    valid_set = build_valid_81_frame_set(video_root, video_ids_all, tuples_list)
    print(f"[81帧过滤] 检查 {len(tuples_list)} 个 clip，其中 {len(valid_set)} 个为 {required_frames} 帧")

    # 4) 过滤每个 character_list JSON
    for jf in json_files:
        path = os.path.join(character_lists_dir, jf)
        video_id = jf.replace(".json", "")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        new_shots = []
        for shot in data.get("shots", []):
            g = shot.get("group_index")
            if g is None:
                continue
            # 只保留帧数合法的 clip
            new_clips = [c for c in shot.get("clips", []) if (video_id, g, c.get("clip_index")) in valid_set]
            for c in new_clips:
                # 只保留 overlap 中帧数合法的
                overlap = c.get("overlapping_clip_indices") or []
                c["overlapping_clip_indices"] = [x for x in overlap if (video_id, g, x) in valid_set]
            # candidate_clips 只保留仍存在的 clip 且帧数合法
            cand = shot.get("candidate_clips") or []
            new_cand = [x for x in cand if (video_id, g, x) in valid_set and any(c.get("clip_index") == x for c in new_clips)]
            shot["clips"] = new_clips
            shot["candidate_clips"] = new_cand
            if new_clips and new_cand:
                new_shots.append(shot)
        orig_len = len(data.get("shots", []))
        data["shots"] = new_shots
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if len(new_shots) != orig_len:
            print(f"  [JSON] {jf}: shots {orig_len} -> {len(new_shots)}")

    # 5) 过滤 CSV：每行 candidate_clips 只保留在 valid_set 中的 clip，空行删除
    new_csv_rows = []
    for (video_id, g, indices) in csv_rows:
        kept = [ci for ci in indices if (video_id, g, ci) in valid_set]
        if not kept:
            continue
        new_csv_rows.append((video_id, g, kept))

    with open(candidate_groups_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for (video_id, g, kept) in new_csv_rows:
            w.writerow([video_id, g, "|".join(str(x) for x in kept)])

    print(f"[81帧过滤] CSV 行数 {len(csv_rows)} -> {len(new_csv_rows)}")
    return 0


def build_overlapping_clips(start_f, end_f):
    """
    根据 shot_range 按首尾重叠方式切分 clips。
    clip1: start_f ~ start_f+80
    clip2: start_f+80 ~ start_f+160
    clip3: start_f+160 ~ start_f+240
    ...
    """
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
        # 首尾重叠：下一个 clip 的首帧 = 当前 clip 的尾帧
        s = e
        if num_frames < CLIP_FRAMES:
            break
    return clips


def process_json(input_path, output_path):
    """处理单个 JSON 文件，重新计算 clips 并保存到新路径。"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for group in data.get("groups", []):
        shot_range = group.get("shot_range", [])
        if len(shot_range) == 2:
            start_f, end_f = shot_range
            group["clips"] = build_overlapping_clips(start_f, end_f)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return len(data.get("groups", []))


def remove_empty_overlap_from_candidate_and_stats(candidate_groups_csv, character_lists_dir):
    """
    从 candidate_clips 中移除 overlapping_clip_indices 为空的 clip；
    按 JSON 重新生成 CSV（只保留 candidate_clips 非空的 group）；
    统计并打印处理前后可用 data group 数量。
    返回 (处理前 group 数, 处理后 group 数, 更新后的 CSV 行数)。
    """
    character_lists_dir = os.path.abspath(character_lists_dir)
    json_files = [f for f in os.listdir(character_lists_dir) if f.endswith(".json")]

    csv_rows_out = []
    total_before = 0
    total_after = 0

    for jf in sorted(json_files):
        path = os.path.join(character_lists_dir, jf)
        video_id = jf.replace(".json", "")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        for shot in data.get("shots", []):
            g = shot.get("group_index")
            if g is None:
                continue
            clips = shot.get("clips") or []
            clip_by_index = {c.get("clip_index"): c for c in clips}
            candidate_clips = shot.get("candidate_clips") or []

            if candidate_clips:
                total_before += 1

            # 只保留 overlap 非空的 clip 作为 candidate
            new_candidate_clips = [
                ci for ci in candidate_clips
                if clip_by_index.get(ci) and (clip_by_index[ci].get("overlapping_clip_indices") or [])
            ]
            shot["candidate_clips"] = new_candidate_clips

            if new_candidate_clips:
                total_after += 1
                csv_rows_out.append((video_id, g, new_candidate_clips))

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # 写回 CSV
    fieldnames = ["video_id", "group_index", "candidate_clips"]
    with open(candidate_groups_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for (video_id, g, kept) in csv_rows_out:
            w.writerow([video_id, g, "|".join(str(x) for x in kept)])

    print(f"[统计] 处理前（有 candidate_clips 的 group）: {total_before}")
    print(f"[统计] 移除 overlap 为空的 candidate 后，可用 data group 数量: {total_after}")
    print(f"[统计] CSV 行数: {len(csv_rows_out)}")
    return total_before, total_after, len(csv_rows_out)


def main():
    parser = argparse.ArgumentParser(description="重新计算 clips 划分为首尾重叠模式；可选：按 81 帧过滤 candidate/JSON")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="原始 clips_record 目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录 (首尾重叠版本)")
    parser.add_argument("--candidate_groups_csv", type=str, default=None,
                        help="candidate_groups.csv 路径；与 character_lists_dir、video_root 同时指定时执行 81 帧过滤")
    parser.add_argument("--character_lists_dir", type=str, default=None,
                        help="character_lists 目录（含 TopXXX.json）")
    parser.add_argument("--video_root", type=str, default=None,
                        help="视频根目录（含 video_id/group_N/clipM.mp4）")
    parser.add_argument("--remove_empty_overlap", action="store_false",
                        help="从 candidate_clips 中移除 overlap 为空的 clip，并统计可用 group 数")
    args = parser.parse_args()

    # 移除 overlap 为空的 candidate 并统计可用 group（仅需 csv + character_lists_dir）
    if args.remove_empty_overlap and args.candidate_groups_csv and args.character_lists_dir:
        if not os.path.isfile(args.candidate_groups_csv):
            print(f"错误: candidate_groups_csv 不存在: {args.candidate_groups_csv}")
            return 1
        if not os.path.isdir(args.character_lists_dir):
            print(f"错误: character_lists_dir 不存在: {args.character_lists_dir}")
            return 1
        print("执行：从 candidate 中移除 overlap 为空的 clip，并统计可用 group...")
        remove_empty_overlap_from_candidate_and_stats(
            args.candidate_groups_csv,
            args.character_lists_dir,
        )
        print("完成。")

    # 81 帧过滤：从 CSV 和 character_lists JSON 中忽略帧数不为 81 的 candidate 与 overlap
    if args.candidate_groups_csv and args.character_lists_dir and args.video_root:
        if not os.path.isfile(args.candidate_groups_csv):
            print(f"错误: candidate_groups_csv 不存在: {args.candidate_groups_csv}")
            return 1
        if not os.path.isdir(args.character_lists_dir):
            print(f"错误: character_lists_dir 不存在: {args.character_lists_dir}")
            return 1
        if not os.path.isdir(args.video_root):
            print(f"错误: video_root 不存在: {args.video_root}")
            return 1
        print("执行 81 帧过滤（candidate 与 overlap 仅保留 81 帧）...")
        filter_candidate_and_overlap_by_81_frames(
            args.candidate_groups_csv,
            args.character_lists_dir,
            args.video_root,
            required_frames=CLIP_FRAMES,
        )
        print("81 帧过滤完成。")

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return 1

    os.makedirs(output_dir, exist_ok=True)

    json_files = sorted([f for f in os.listdir(input_dir) if f.endswith("_clips.json")])
    print(f"找到 {len(json_files)} 个 JSON 文件")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    total_groups = 0
    for json_file in json_files:
        input_path = os.path.join(input_dir, json_file)
        output_path = os.path.join(output_dir, json_file)
        n_groups = process_json(input_path, output_path)
        total_groups += n_groups
        print(f"  [OK] {json_file} ({n_groups} groups)")

    print(f"完成: 处理了 {len(json_files)} 个文件, 共 {total_groups} 个 groups")
    return 0


if __name__ == "__main__":
    exit(main())
