"""Caption shot groups produced by the clip-extraction pipeline.

Groups contain numbered clips plus an ignored ``last_clip`` remainder. Sampled
frames are sent to an OpenAI-compatible chat-completions API and persisted as
requests complete.
"""
import os
import re
import random
import logging
import sys
import argparse
import json
import base64
import asyncio
import time
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# Sample 81-frame clips uniformly. Later clips skip frame zero because it
# duplicates the preceding clip's frame 80.
FIRST_CLIP_SAMPLE_FRAMES = [0, 40, 80]
OTHER_CLIP_SAMPLE_FRAMES = [40, 80]
# Maximum frames used for a group-level caption.
MAX_GLOBAL_FRAMES = 11
# Maximum clips per group in test mode.
MAX_CLIPS_IN_TEST = 3
# Bound production requests per group to reduce gateway 503 responses.
MAX_CLIPS_PER_GROUP = 5
# Image resize dimensions.
RESIZE_WIDTH = 672
RESIZE_HEIGHT = 384
DEFAULT_PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def _read_markdown_prompt_section(path, heading):
    """Read the first fenced text block under a markdown heading."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    pattern = rf"##\s+{re.escape(heading)}\s*```(?:text)?\s*(.*?)```"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Missing fenced prompt section '{heading}' in {path}")
    return match.group(1).strip()


def load_caption_prompts(global_prompt_path, chunk_prompt_path):
    """Load global and chunk caption prompts from markdown files."""
    return {
        "global_system": _read_markdown_prompt_section(global_prompt_path, "System Prompt"),
        "global_user": _read_markdown_prompt_section(global_prompt_path, "User Prompt"),
        "global_frame": _read_markdown_prompt_section(global_prompt_path, "Frame Message"),
        "chunk_system": _read_markdown_prompt_section(chunk_prompt_path, "System Prompt"),
        "chunk_user": _read_markdown_prompt_section(chunk_prompt_path, "User Prompt"),
        "chunk_frame": _read_markdown_prompt_section(chunk_prompt_path, "Frame Message"),
    }


def get_clip_frame_count(clip_path):
    """Return the video frame count, or zero on failure."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def extract_frames_from_clip(clip_path, frame_indices):
    """Return requested frames as base64, or ``[]`` for an unreadable MP4."""
    frames_base64 = []
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        logging.warning(f"无法打开 clip（可能不完整/损坏，如 moov atom not found）: {clip_path}")
        return frames_base64

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
            _, buffer = cv2.imencode(".jpg", frame)
            frames_base64.append(base64.b64encode(buffer).decode("utf-8"))
    cap.release()
    return frames_base64


def get_sorted_clips(group_dir):
    """Return numerically sorted clips, excluding ``last_clip``."""
    clips = []
    for f in os.listdir(group_dir):
        if not f.endswith(".mp4"):
            continue
        if f == "last_clip.mp4":
            continue
        # Extract the numeric clip index for sorting.
        match = re.match(r"clip(\d+)\.mp4", f)
        if match:
            clips.append((int(match.group(1)), os.path.join(group_dir, f)))
    clips.sort(key=lambda x: x[0])
    return [path for _, path in clips]


def get_sorted_groups(video_dir):
    """Return group directories in numeric order."""
    groups = []
    for d in os.listdir(video_dir):
        if not d.startswith("group_"):
            continue
        match = re.match(r"group_(\d+)", d)
        if match:
            groups.append((int(match.group(1)), os.path.join(video_dir, d)))
    groups.sort(key=lambda x: x[0])
    return [(idx, path) for idx, path in groups]


def process_video_clips(video_dir, test_mode=False, max_groups_per_video=50):
    """Collect sampled clip and group-caption frames for one video directory.

    Test mode limits work to the third group. Production sampling excludes edge
    groups, preserves original group indices, and caps the selected group count.
    """
    groups = get_sorted_groups(video_dir)
    all_shots_data = []

    # Test mode processes only the third group.
    if test_mode:
        groups = groups[2:3]
    else:
        # Exclude the first and last two groups.
        if len(groups) > 4:
            groups = groups[2:-2]
        else:
            groups = []
        # Sample at most the configured count while preserving original indices.
        if len(groups) > max_groups_per_video:
            groups = sorted(random.sample(groups, max_groups_per_video), key=lambda x: x[0])

    for group_idx, group_dir in groups:
        clips = get_sorted_clips(group_dir)
        if not clips:
            continue

        # Apply separate test and production clip limits.
        if test_mode:
            clips = clips[:MAX_CLIPS_IN_TEST]
        else:
            clips = clips[:MAX_CLIPS_PER_GROUP]

        clips_base64 = []
        for clip_idx, clip_path in enumerate(clips):
            # Test mode reports source and sampled frame counts.
            if test_mode:
                total_frames = get_clip_frame_count(clip_path)
                logging.info(f"[test] group_{group_idx} clip{clip_idx + 1} ({os.path.basename(clip_path)}): 视频总帧数={total_frames}")
            # Skip the duplicated first frame after the first clip.
            frame_indices = FIRST_CLIP_SAMPLE_FRAMES if clip_idx == 0 else OTHER_CLIP_SAMPLE_FRAMES
            frames = extract_frames_from_clip(clip_path, frame_indices)
            if test_mode:
                logging.info(f"[test] group_{group_idx} clip{clip_idx + 1}: 实际采样帧数={len(frames)}")
            if frames:
                clips_base64.append(frames)

        # Skip the entire group if any MP4 is corrupt or unreadable.
        if len(clips_base64) < len(clips):
            logging.warning(f"跳过 group_{group_idx}：{len(clips) - len(clips_base64)} 个 clip 无法打开，可能为损坏/不完整 MP4")
            continue
        if clips_base64:
            # Uniformly sample group-level caption frames across all valid clip frames.
            all_frames = []
            for clip_frames in clips_base64:
                all_frames.extend(clip_frames)
            if len(all_frames) <= MAX_GLOBAL_FRAMES:
                group_global_frames_base64 = all_frames
            else:
                step = len(all_frames) / MAX_GLOBAL_FRAMES
                indices = [int(i * step) for i in range(MAX_GLOBAL_FRAMES)]
                group_global_frames_base64 = [all_frames[i] for i in indices]
            all_shots_data.append({
                "group_index": group_idx,
                "clips_base64": clips_base64,
                "group_global_frames_base64": group_global_frames_base64,
            })

    return all_shots_data



def process_video_clips_for_groups(video_dir, group_indices, test_mode=False):
    """Process existing group indices in order for deterministic resume behavior."""
    all_groups = get_sorted_groups(video_dir)
    group_dir_by_idx = {idx: path for idx, path in all_groups}
    all_shots_data = []
    for group_idx in group_indices:
        if group_idx not in group_dir_by_idx:
            logging.warning(f"断点续传: group_{group_idx} 在视频目录中不存在，跳过")
            continue
        group_dir = group_dir_by_idx[group_idx]
        clips = get_sorted_clips(group_dir)
        if not clips:
            continue
        if test_mode:
            clips = clips[:MAX_CLIPS_IN_TEST]
        else:
            clips = clips[:MAX_CLIPS_PER_GROUP]
        clips_base64 = []
        for clip_idx, clip_path in enumerate(clips):
            frame_indices = FIRST_CLIP_SAMPLE_FRAMES if clip_idx == 0 else OTHER_CLIP_SAMPLE_FRAMES
            frames = extract_frames_from_clip(clip_path, frame_indices)
            if frames:
                clips_base64.append(frames)
        if len(clips_base64) < len(clips):
            logging.warning(f"跳过 group_{group_idx}：部分 clip 无法打开")
            continue
        if clips_base64:
            all_frames = []
            for clip_frames in clips_base64:
                all_frames.extend(clip_frames)
            if len(all_frames) <= MAX_GLOBAL_FRAMES:
                group_global_frames_base64 = all_frames
            else:
                step = len(all_frames) / MAX_GLOBAL_FRAMES
                indices = [int(i * step) for i in range(MAX_GLOBAL_FRAMES)]
                group_global_frames_base64 = [all_frames[i] for i in indices]
            all_shots_data.append({
                "group_index": group_idx,
                "clips_base64": clips_base64,
                "group_global_frames_base64": group_global_frames_base64,
            })
    return all_shots_data


def save_frames_base64_to_dir(frames_base64, dir_path):
    """Save base64 frames as sequential JPEG files."""
    os.makedirs(dir_path, exist_ok=True)
    for i, b64 in enumerate(frames_base64):
        raw = base64.b64decode(b64)
        path = os.path.join(dir_path, f"frame_{i:03d}.jpg")
        with open(path, "wb") as f:
            f.write(raw)


# Retry a whole group when any stored clip contains this API error.
ERROR_403_MARKER = "Error code: 403"


def _load_existing_and_groups_with_403(save_path):
    """Load resume data and return group indices requiring a 403 retry.

    Missing or invalid JSON returns ``(None, None)``. Complete data returns an
    empty retry set; stored 403 failures return their group indices for merging.
    """
    if not os.path.exists(save_path):
        return (None, None)
    try:
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return (None, None)
    shots = data.get("shots") or []
    groups_to_retry = set()
    for s in shots:
        for c in s.get("clips") or []:
            if ERROR_403_MARKER in (c.get("error") or ""):
                groups_to_retry.add(s.get("group_index"))
                break
    return (data, groups_to_retry)


def _format_api_error_detail(e, max_body_len=500):
    """Extract status and response details from an API exception."""
    parts = [str(e)]
    code = getattr(e, "status_code", None)
    if code is not None:
        parts.append(f"status_code={code}")
    body = getattr(e, "body", None)
    if body is not None:
        if isinstance(body, (dict, list)):
            body_str = json.dumps(body, ensure_ascii=False)[:max_body_len]
        else:
            body_str = str(body)[:max_body_len]
        parts.append(f"body={body_str}")
    for attr in ("message", "code", "type", "param"):
        val = getattr(e, attr, None)
        if val is not None and attr != "message":
            parts.append(f"{attr}={val}")
    return " | ".join(parts)


def _process_one_video(video_id, clips_dir, output_dir, api_key, base_url, model_name, max_groups_per_video, test, caption_prompts, request_delay=0):
    """Caption one movie with a thread-local client and optional 503 pacing."""
    video_dir = os.path.join(clips_dir, video_id)
    if not os.path.isdir(video_dir):
        logging.warning(f"跳过不存在的目录: {video_dir}")
        return (video_id, "skipped_dir")

    save_path = os.path.join(output_dir, f"{video_id}.json")
    existing_data, groups_to_retry = _load_existing_and_groups_with_403(save_path)
    if existing_data is not None and len(groups_to_retry) == 0:
        logging.info(f"跳过已存在且无 403 错误: {save_path}")
        return (video_id, "skipped_exists")
    resume_mode = existing_data is not None and len(groups_to_retry) > 0
    existing_by_group = {}
    if resume_mode:
        existing_by_group = {s["group_index"]: s for s in existing_data.get("shots") or []}
        logging.info(f"断点续传 {video_id}: 对 {len(groups_to_retry)} 个含 403 的 group 重新 caption: {sorted(groups_to_retry)}")

    if resume_mode:
        group_indices = [s["group_index"] for s in existing_data.get("shots") or []]
        all_shots_data = process_video_clips_for_groups(video_dir, group_indices, test_mode=test)
    else:
        all_shots_data = process_video_clips(
            video_dir,
            test_mode=test,
            max_groups_per_video=max_groups_per_video,
        )
    if not all_shots_data:
        logging.warning(f"跳过无有效数据: {video_id}")
        return (video_id, "skipped_no_data")

    logging.info(f"处理 {video_id}: {len(all_shots_data)} groups（每 group 独立 global caption）")
    client = OpenAI(api_key=api_key, base_url=base_url)
    frames_root = os.path.join(output_dir, "sampled_frames", video_id)

    json_content = {"video_id": video_id, "shots": []}
    local_prompt_tpl = """### Task Overview:

            Your task is to describe the video content in terms of the subjects' expression and actions, scene background, and camera movement in a single paragraph, based on the subject descriptions in the story setting '{group_caption}'. The description should not exceed 80 words.

            ### Special Notes:

            1. Identify which subjects from the story setting (the global caption) appear in the current video. Refer to those subjects by the exact subject names used in the global caption (e.g., "standing apprentice", "white car", "blonde man"). Some subjects in the story may not appear in the video.
            2. Do not describe the subject's appearance, which has been described in the story setting. Focus on subjects' expression and actions, scene background, and camera movement.
            3. First describe the subject facing the camera, then describe the other subjects. When describing the camera position, specify which subjects are facing to the camera.
            4. Subjects who appear only a few times and are not important to the storyline should be omitted.
            5. Do not describe subjects that are far away or too small or too blurry.
            6. Do not invent new names for main subjects that already exist in the story setting—use the global caption's naming exactly.
            7. Do not repeat the content in the story setting.
            8. a local caption **may include additional people or characters** that are not listed as main subjects in the global caption. When such extra, non-main characters appear, give them names like "sitting student", "walking vendors", "cyclists" and make clear they are distinct from the main subjects defined in the global caption.

            ### Expected Output Format 1:
            "standing apprentice walks through a dense forest, carrying a large plastic bag; following apprentice trails behind, holding a small flashlight; brown dog runs across the path. The forest is filled with tall, slender trees and a carpet of fallen leaves. The camera follows them from behind in a medium shot, keeping the standing apprentice centered. The camera movement is steady and smooth."

            ### Expected Output Format 2:
            Expected Output Format 2 (example: global caption's main subject is 'standing apprentice', local also shows other apprentices — name them 'sitting students'):
            "standing apprentice stands at the front of a classroom, looking down at a notebook; several sitting students occupy the desks behind, taking notes or glancing up. The camera slowly dollies in from a medium-wide to a medium shot on the standing apprentice, with shallow depth-of-field that softly blurs the sitting students in the background. The scene feels quiet and focused, with warm indoor lighting."

            """
    local_prompt_tpl = caption_prompts["chunk_user"]

    for shot_data in all_shots_data:
        group_index = shot_data["group_index"]
        if resume_mode and group_index not in groups_to_retry:
            json_content["shots"].append(existing_by_group[group_index])
            continue
        clips_base64 = shot_data["clips_base64"]
        group_global_frames_base64 = shot_data["group_global_frames_base64"]

            # Save group-caption samples under the corresponding group directory.
        group_frames_dir = os.path.join(frames_root, f"group_{group_index}")
        save_frames_base64_to_dir(group_global_frames_base64, group_frames_dir)

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": caption_prompts["global_system"]},
                    {"role": "user", "content": caption_prompts["global_user"]},
                    {"role": "user", "content": [
                        caption_prompts["global_frame"],
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, group_global_frames_base64)
                    ]},
                ],
                temperature=0,
                timeout=120.0,
            )
            raw = response.choices[0].message.content
            group_global_caption = (raw or "").replace('\n', '').replace('  ', ' ').strip()
            if request_delay > 0:
                time.sleep(request_delay)
        except Exception as e:
            detail = _format_api_error_detail(e)
            logging.error(f"Group global caption 失败 {video_id} group_{group_index}: {detail}")
            if getattr(e, "status_code", None) == 503 and "暂无可用渠道" in str(getattr(e, "body", "") or ""):
                logging.warning("503 原因：API 网关当前对该模型无可用渠道，请稍后重试或更换 --model_name")
            group_global_caption = ""

        if len(group_global_caption) < 5:
            group_global_caption = ""

        local_prompt = local_prompt_tpl.format(group_caption=group_global_caption)
        shot_result = {"group_index": group_index, "group_caption": group_global_caption, "clips": []}
        for clip_idx, clip_frames in enumerate(clips_base64):
            clip_error = ""
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": caption_prompts["chunk_system"]},
                        {"role": "user", "content": local_prompt},
                        {"role": "user", "content": [
                            caption_prompts["chunk_frame"],
                            *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, clip_frames)
                        ]},
                    ],
                    temperature=0,
                    timeout=30.0,
                )
                raw = response.choices[0].message.content
                local_caption = (raw or "").replace('\n', '').replace('  ', ' ').strip()
                if not local_caption:
                    clip_error = "Empty response from API (no exception)"
            except Exception as e:
                detail = _format_api_error_detail(e)
                logging.error(f"Clip caption 失败 {video_id} group_{group_index} clip{clip_idx + 1}: {detail}")
                if getattr(e, "status_code", None) == 503 and "暂无可用渠道" in str(getattr(e, "body", "") or ""):
                    logging.warning("503 原因：API 网关当前对该模型无可用渠道，请稍后重试或更换 --model_name")
                local_caption = ""
                clip_error = detail
            if request_delay > 0:
                time.sleep(request_delay)
            shot_result["clips"].append({"clip_index": clip_idx + 1, "caption": local_caption, "error": clip_error})
        json_content["shots"].append(shot_result)

    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(json_content, f, ensure_ascii=False, indent=4)
        logging.info(f"已保存: {save_path}")
        return (video_id, "ok")
    except Exception as e:
        logging.error(f"保存失败 {video_id}: {e}")
        return (video_id, "error_save")


# Concurrent aiohttp/asyncio request path.
# POST {BASE_URL}/chat/completions with OpenAI-compatible payload:
# headers: Authorization: Bearer {API_KEY}, Content-Type: application/json.

def _chat_completion_payload(model, messages, temperature=0, max_tokens=2048):
    """Build a chat-completions payload matching the synchronous client path."""
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }


async def _chat_completion_aio(session, base_url, api_key, model, messages, timeout_sec, semaphore):
    """Send a semaphore-limited chat-completions request with aiohttp."""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = _chat_completion_payload(model, messages, temperature=0, max_tokens=2048)
    async with semaphore:
        try:
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout_sec)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    # Retain more response body for diagnosing 5xx gateway failures.
                    max_len = 500 if resp.status >= 500 else 200
                    return ("", f"HTTP {resp.status} | body={text[:max_len]}")
                data = await resp.json()
                content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                return ((content.replace("\n", "").replace("  ", " ").strip()), "")
        except asyncio.TimeoutError:
            return ("", "Request timeout")
        except Exception as e:
            return ("", _format_api_error_detail(e))


async def _process_one_video_async(session, video_id, clips_dir, output_dir, api_key, base_url, model_name,
                                   max_groups_per_video, test, caption_prompts, semaphore, executor, loop):
    """Caption one movie asynchronously, with concurrent clip requests per group."""
    video_dir = os.path.join(clips_dir, video_id)
    if not os.path.isdir(video_dir):
        logging.warning(f"跳过不存在的目录: {video_dir}")
        return (video_id, "skipped_dir")

    save_path = os.path.join(output_dir, f"{video_id}.json")
    existing_data, groups_to_retry = _load_existing_and_groups_with_403(save_path)
    if existing_data is not None and len(groups_to_retry) == 0:
        logging.info(f"跳过已存在且无 403 错误: {save_path}")
        return (video_id, "skipped_exists")
    resume_mode = existing_data is not None and len(groups_to_retry) > 0
    existing_by_group = {}
    if resume_mode:
        existing_by_group = {s["group_index"]: s for s in existing_data.get("shots") or []}
        logging.info(f"断点续传 {video_id}: 对 {len(groups_to_retry)} 个含 403 的 group 重新 caption: {sorted(groups_to_retry)}")

    if resume_mode:
        group_indices = [s["group_index"] for s in existing_data.get("shots") or []]
        all_shots_data = await loop.run_in_executor(
            executor,
            lambda: process_video_clips_for_groups(video_dir, group_indices, test_mode=test),
        )
    else:
        all_shots_data = await loop.run_in_executor(
            executor,
            lambda: process_video_clips(video_dir, test_mode=test, max_groups_per_video=max_groups_per_video),
        )
    if not all_shots_data:
        logging.warning(f"跳过无有效数据: {video_id}")
        return (video_id, "skipped_no_data")

    logging.info(f"处理 {video_id}: {len(all_shots_data)} groups（每 group 独立 global caption）")
    frames_root = os.path.join(output_dir, "sampled_frames", video_id)
    local_prompt_tpl = """### Task Overview:

            Your task is to describe the video content in terms of the subjects' expression and actions, scene background, and camera movement in a single paragraph, based on the subject descriptions in the story setting '{group_caption}'. The description should not exceed 80 words.

            ### Special Notes:

            1. Identify which subjects from the story setting (the global caption) appear in the current video. Refer to those subjects by the exact subject names used in the global caption (e.g., "standing apprentice", "white car", "blonde man"). Some subjects in the story may not appear in the video.
            2. Do not describe the subject's appearance, which has been described in the story setting. Focus on subjects' expression and actions, scene background, and camera movement.
            3. First describe the subject facing the camera, then describe the other subjects. When describing the camera position, specify which subjects are facing to the camera.
            4. Subjects who appear only a few times and are not important to the storyline should be omitted.
            5. Do not describe subjects that are far away or too small or too blurry.
            6. Do not invent new names for main subjects that already exist in the story setting—use the global caption's naming exactly.
            7. Do not repeat the content in the story setting.
            8. a local caption **may include additional people or characters** that are not listed as main subjects in the global caption. When such extra, non-main characters appear, give them names like "sitting student", "walking vendors", "cyclists" and make clear they are distinct from the main subjects defined in the global caption.

            ### Expected Output Format 1:
            "standing apprentice walks through a dense forest, carrying a large plastic bag; following apprentice trails behind, holding a small flashlight; brown dog runs across the path. The forest is filled with tall, slender trees and a carpet of fallen leaves. The camera follows them from behind in a medium shot, keeping the standing apprentice centered. The camera movement is steady and smooth."

            ### Expected Output Format 2:
            Expected Output Format 2 (example: global caption's main subject is 'standing apprentice', local also shows other apprentices — name them 'sitting students'):
            "standing apprentice stands at the front of a classroom, looking down at a notebook; several sitting students occupy the desks behind, taking notes or glancing up. The camera slowly dollies in from a medium-wide to a medium shot on the standing apprentice, with shallow depth-of-field that softly blurs the sitting students in the background. The scene feels quiet and focused, with warm indoor lighting."

            """
    local_prompt_tpl = caption_prompts["chunk_user"]

    json_content = {"video_id": video_id, "shots": []}
    for shot_data in all_shots_data:
        group_index = shot_data["group_index"]
        if resume_mode and group_index not in groups_to_retry:
            json_content["shots"].append(existing_by_group[group_index])
            continue
        clips_base64 = shot_data["clips_base64"]
        group_global_frames_base64 = shot_data["group_global_frames_base64"]

        group_frames_dir = os.path.join(frames_root, f"group_{group_index}")
        await loop.run_in_executor(executor, lambda fd=group_frames_dir, fr=group_global_frames_base64: save_frames_base64_to_dir(fr, fd))

        messages_global = [
            {"role": "system", "content": caption_prompts["global_system"]},
            {"role": "user", "content": caption_prompts["global_user"]},
            {"role": "user", "content": [
                caption_prompts["global_frame"],
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{x}", "detail": "low"}} for x in group_global_frames_base64],
            ]},
        ]
        group_global_caption, global_err = await _chat_completion_aio(
            session, base_url, api_key, model_name, messages_global, timeout_sec=120.0, semaphore=semaphore
        )
        if global_err or len(group_global_caption or "") < 5:
            group_global_caption = ""

        local_prompt = local_prompt_tpl.format(group_caption=group_global_caption or "")
        clip_tasks = []
        for clip_frames in clips_base64:
            messages_clip = [
                {"role": "system", "content": caption_prompts["chunk_system"]},
                {"role": "user", "content": local_prompt},
                {"role": "user", "content": [
                    caption_prompts["chunk_frame"],
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{x}", "detail": "low"}} for x in clip_frames],
                ]},
            ]
            clip_tasks.append(_chat_completion_aio(
                session, base_url, api_key, model_name, messages_clip, timeout_sec=30.0, semaphore=semaphore
            ))

        clip_results = await asyncio.gather(*clip_tasks, return_exceptions=True)
        shot_result = {"group_index": group_index, "group_caption": group_global_caption or "", "clips": []}
        for k, r in enumerate(clip_results):
            if isinstance(r, BaseException):
                local_caption, clip_error = "", str(r)
            else:
                local_caption, clip_error = r
                if not local_caption and not clip_error:
                    clip_error = "Empty response from API (no exception)"
            shot_result["clips"].append({
                "clip_index": k + 1,
                "caption": local_caption,
                "error": clip_error or "",
            })
        json_content["shots"].append(shot_result)

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(json_content, f, ensure_ascii=False, indent=4)
        logging.info(f"已保存: {save_path}")
        return (video_id, "ok")
    except Exception as e:
        logging.error(f"保存失败 {video_id}: {e}")
        return (video_id, "error_save")


async def _run_async_concurrent(video_ids, clips_dir, output_dir, api_key, base_url, model_name,
                                max_groups_per_video, test, caption_prompts, max_concurrent_requests):
    """Run semaphore-limited concurrency across movies and clips."""
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=4)

    async with aiohttp.ClientSession() as session:
        tasks = [
            _process_one_video_async(
                session, vid, clips_dir, output_dir,
                api_key, base_url, model_name, max_groups_per_video, test, caption_prompts,
                semaphore, executor, loop,
            )
            for vid in video_ids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException):
                logging.error(f"Async task error: {r}")

    executor.shutdown(wait=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("shot_caption", add_help=True)
    parser.add_argument('--clips_dir', type=str, required=True,
                        help="导出的 clip 目录，结构为 clips_dir/{video_id}/group_{i}/clip*.mp4")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="输出目录，生成 {video_id}.json")
    parser.add_argument('--video_id', type=str, default=None,
                        help="只处理指定的 video_id，不指定则处理所有")
    parser.add_argument('--api_key', type=str, default=None,
                        help="OpenAI-compatible API key. If omitted, --api_key_env is used.")
    parser.add_argument('--api_key_env', type=str, default="OPENAI_API_KEY",
                        help="Environment variable that stores the API key, default OPENAI_API_KEY")
    parser.add_argument('--base_url', type=str, default=os.environ.get("OPENAI_BASE_URL"),
                        help="OpenAI-compatible API base URL. Can also be set with OPENAI_BASE_URL.")
    parser.add_argument('--test', action='store_true',
                        help="测试模式：只处理前两个 video_id 的第三个 group 的前 MAX_CLIPS_IN_TEST 个 clip")
    parser.add_argument('--test_movie', action='store_true',
                        help="测试前两个 movie 的并发（只处理前 2 个 video_id，且用并发执行）")
    parser.add_argument('--max_groups_per_video', type=int, default=50,
                        help="每个电影在排除首 2/尾 2 个 group 后最多随机选取的 group 数做 caption，默认 50")
    parser.add_argument('--workers', type=int, default=1,
                        help="并发处理的 movie 数量，默认 1 为串行")
    parser.add_argument('--max_concurrent_requests', type=int, default=50,
                        help="高并发模式下同时进行中的 API 请求数上限")
    parser.add_argument('--request_delay', type=float, default=0,
                        help="每次 API 调用后等待秒数，减轻网关 503，默认 1.5；设为 0 则无间隔")

    parser.add_argument('--model_name', type=str, default="gemini-2.5-flash",
                        help="模型名称")
    parser.add_argument('--global_prompt_path', type=str,
                        default=os.path.join(DEFAULT_PROMPT_DIR, "global_caption_prompt.md"),
                        help="global caption prompt markdown path")
    parser.add_argument('--chunk_prompt_path', type=str,
                        default=os.path.join(DEFAULT_PROMPT_DIR, "chunk_caption_prompt.md"),
                        help="chunk caption prompt markdown path")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get(args.api_key_env)
    if not api_key:
        parser.error(f"missing API key: pass --api_key or set {args.api_key_env}")
    if not args.base_url:
        parser.error("missing API base URL: pass --base_url or set OPENAI_BASE_URL")
    caption_prompts = load_caption_prompts(args.global_prompt_path, args.chunk_prompt_path)
    clips_dir = os.path.abspath(args.clips_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Collect video IDs to process.
    if args.video_id:
        video_ids = [args.video_id]
    else:
        video_ids = sorted([
            d for d in os.listdir(clips_dir)
            if os.path.isdir(os.path.join(clips_dir, d)) and d.startswith("Top")
        ])

        # Test mode processes only the first two video IDs.
    if args.test:
        video_ids = video_ids[:2]
        logging.info(f"测试模式：只处理前两个 video_id: {video_ids}")

        # Process two test movies serially to reduce gateway 503 responses.
    if args.test_movie:
        video_ids = video_ids[:2]
        logging.info(f"test_movie：测试前两个 movie（顺序）: {video_ids}")

    # Run movies concurrently while keeping requests within each movie serial.
    if args.workers > 1 or args.test_movie:
        n_workers = max(2, args.workers) if args.test_movie else args.workers
        n_workers = min(n_workers, len(video_ids))
        logging.info(f"并发处理 movies (线程): workers={n_workers}")

        def _task(vid):
            return _process_one_video(
                vid, clips_dir, output_dir,
                api_key, args.base_url, args.model_name,
                args.max_groups_per_video, args.test, caption_prompts,
                request_delay=args.request_delay,
            )

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_task, vid): vid for vid in video_ids}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
                future.result()
    else:
        def _task(vid):
            return _process_one_video(
                vid, clips_dir, output_dir,
                api_key, args.base_url, args.model_name,
                args.max_groups_per_video, args.test, caption_prompts,
                request_delay=args.request_delay,
            )
        for video_id in tqdm(video_ids, desc="Processing videos"):
            _task(video_id)
