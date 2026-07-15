#!/usr/bin/env python3
"""Evaluate SlotMem videos with ViStory-style prompt-alignment prompts.

This is a direct SlotMem adapter: it reads SlotMem inference metadata and
samples representative frames from the generated videos. It does not collect
or synthesize reference images.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_slotmem_benchmarks import build_eval_items  # noqa: E402


DIMENSIONS = ("scene", "character_action", "camera")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def extract_scores_loose(text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for dim in DIMENSIONS:
        pattern = rf"[\"']?{re.escape(dim)}[\"']?\s*:\s*\{{[^{{}}]*?[\"']?score[\"']?\s*:\s*([0-9]{{1,3}})"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            payload[dim] = {"score": int(match.group(1)), "reason": "loose_regex_parse"}
    return payload


def normalize_scores(payload: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for dim in DIMENSIONS:
        value = payload.get(dim)
        if isinstance(value, dict):
            score = value.get("score")
            reason = value.get("reason", "")
        else:
            score = value
            reason = ""
        try:
            score_i = int(round(float(score)))
        except (TypeError, ValueError):
            score_i = 0
            reason = f"invalid_score; {reason}"
        out[dim] = {"score": max(0, min(100, score_i)), "reason": str(reason)}
    return out


def mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def read_video_frame(video_path: Path, frame_idx: int, out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    used = max(0, min(frame_idx, max(0, frame_count - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, used)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"cannot read frame {used} from {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame_rgb).save(out_path, quality=95)
    return out_path


def image_data_url(path: Path, max_image_side: int, jpeg_quality: int) -> str:
    image = Image.open(path).convert("RGB")
    if max_image_side and max(image.size) > max_image_side:
        image.thumbnail((max_image_side, max_image_side))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=max(1, min(100, jpeg_quality)), optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def prompt_for_task(task: dict[str, Any]) -> str:
    return f"""
You are evaluating one generated video representative frame against the text prompt for the corresponding SlotMem chunk.

Score three ViStoryBench-style prompt-alignment dimensions from 0 to 100:
- scene: environment, setting, objects, lighting, and background match.
- character_action: the listed characters/subjects are visible when expected and their actions/relations match.
- camera: camera distance, angle, composition, and viewpoint match.

Rules:
- Use only the image and prompt below.
- Be strict but fair. A score of 100 means nearly perfect alignment; 0 means absent or contradictory.
- Return JSON only with integer scores and a short reason per dimension.

chunk_caption: {task["caption"]}
character_keys: {json.dumps(task["characters"], ensure_ascii=False)}

Output format:
{{
  "scene": {{"score": 0, "reason": "..."}},
  "character_action": {{"score": 0, "reason": "..."}},
  "camera": {{"score": 0, "reason": "..."}}
}}
""".strip()


def build_tasks(input_root: Path, frame_root: Path) -> list[dict[str, Any]]:
    tasks = []
    for item in build_eval_items(input_root):
        chunks = [row for row in item.get("chunks", []) if isinstance(row, dict)]
        if not chunks:
            chunks = [{"chunk_idx": 0, "caption": " ".join(item.get("prompts", [])), "characters": item.get("characters", [])}]
        source_video = Path(item["source_video"])
        cap = cv2.VideoCapture(str(source_video))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        for order, chunk in enumerate(chunks):
            chunk_idx = int(chunk.get("chunk_idx", order))
            frame_idx = int((order + 0.5) * max(1, frame_count) / max(1, len(chunks)))
            frame_path = frame_root / item["sample_id"] / f"chunk_{chunk_idx:03d}_frame_{frame_idx:06d}.jpg"
            read_video_frame(source_video, frame_idx, frame_path)
            characters = chunk.get("characters") or chunk.get("character_list") or item.get("characters", [])
            if isinstance(characters, dict):
                characters = list(characters.keys())
            tasks.append(
                {
                    "sample_id": item["sample_id"],
                    "kind": item["kind"],
                    "chunk_idx": chunk_idx,
                    "frame_idx": frame_idx,
                    "frame_path": str(frame_path),
                    "source_video": str(source_video.resolve()),
                    "caption": str(chunk.get("caption") or chunk.get("content") or " ".join(item.get("prompts", []))),
                    "characters": [str(x) for x in characters] if isinstance(characters, list) else [],
                }
            )
    return tasks


def make_client(args: argparse.Namespace):
    if args.provider == "api":
        from openai import OpenAI

        api_key = args.api_key
        base_url = args.base_url
        if not api_key and args.api_key_env:
            api_key = os.getenv(args.api_key_env)
        if not api_key or not base_url:
            raise RuntimeError("API provider requires --api-key or --api-key-env, plus --base-url.")
        return OpenAI(base_url=base_url, api_key=api_key), args.api_model

    narrastream_repo = args.narrastream_repo.resolve()
    if str(narrastream_repo) not in sys.path:
        sys.path.insert(0, str(narrastream_repo))
    from narrastream_bench.models.local_qwen3_vl_client import LocalQwen3VLClient

    client = LocalQwen3VLClient(model_path=args.local_qwen35_model, max_tokens=args.max_tokens, temperature=args.temperature)
    return client, args.local_qwen35_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ViStory-style prompt alignment on SlotMem outputs.")
    parser.add_argument("--infer-output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "benchmark_outputs" / "slotmem")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--provider", choices=["api", "local-qwen35"], default="api")
    parser.add_argument("--provider-id", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default=os.environ.get("VISTORY_API_KEY_ENV"))
    parser.add_argument("--base-url", default=os.environ.get("VISTORY_BASE_URL"))
    parser.add_argument("--api-model", default=os.environ.get("VISTORY_API_MODEL", "gpt-4.1"))
    parser.add_argument("--local-qwen35-model", default=os.environ.get("LOCAL_QWEN35_MODEL", "/models/Qwen3.5-4B"))
    parser.add_argument("--narrastream-repo", type=Path, default=Path(os.environ["NARRASTREAM_REPO"]) if os.environ.get("NARRASTREAM_REPO") else REPO_ROOT / "bench" / "NarraStream-Bench")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-image-side", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = args.infer_output.resolve()
    run_name = args.run_name or input_root.name
    provider_id = args.provider_id or ("gpt41" if args.provider == "api" else "qwen35")
    run_root = (args.output_root / run_name / "vistory" / provider_id).resolve()
    frame_root = run_root / "frames"
    metric_root = run_root / "metrics"
    raw_root = metric_root / "raw"
    tasks = build_tasks(input_root, frame_root)
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        raise RuntimeError(f"No SlotMem vistory tasks found under {input_root}")

    client, model = make_client(args)
    rows = []
    errors = []
    for idx, task in enumerate(tasks, start=1):
        out_path = raw_root / task["sample_id"] / f"chunk_{task['chunk_idx']:03d}.json"
        result = read_json(out_path) if out_path.exists() and not args.overwrite else None
        if not isinstance(result, dict) or result.get("status") == "error":
            print(f"[{idx}/{len(tasks)}] {task['sample_id']} chunk={task['chunk_idx']}", flush=True)
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_for_task(task)},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_data_url(
                                            Path(task["frame_path"]),
                                            max_image_side=args.max_image_side,
                                            jpeg_quality=args.jpeg_quality,
                                        )
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or ""
                try:
                    parsed = extract_json(raw)
                    parse_status = "json"
                except Exception:
                    parsed = extract_scores_loose(raw)
                    parse_status = "loose_regex"
                    if not parsed:
                        raise
                result = {**task, "raw_text": raw, "scores": normalize_scores(parsed), "status": "ok", "parse_status": parse_status}
            except Exception as exc:  # noqa: BLE001
                result = {**task, "raw_text": locals().get("raw", ""), "scores": normalize_scores({}), "status": "error", "error": repr(exc)}
                errors.append(result)
            write_json(out_path, result)

        row = {
            "sample_id": task["sample_id"],
            "chunk_idx": task["chunk_idx"],
            "frame_idx": task["frame_idx"],
            "status": result.get("status", "ok"),
            "parse_status": result.get("parse_status", ""),
            "scene": result["scores"]["scene"]["score"],
            "character_action": result["scores"]["character_action"]["score"],
            "camera": result["scores"]["camera"]["score"],
            "frame_path": task["frame_path"],
            "source_video": task["source_video"],
        }
        rows.append(row)

    chunk_fields = ["sample_id", "chunk_idx", "frame_idx", "status", "parse_status", "scene", "character_action", "camera", "frame_path", "source_video"]
    write_csv(metric_root / "chunk_scores.csv", rows, chunk_fields)

    sample_rows = []
    for sample_id in sorted({row["sample_id"] for row in rows}):
        items = [row for row in rows if row["sample_id"] == sample_id]
        sample_rows.append(
            {
                "sample_id": sample_id,
                "chunk_count": len(items),
                "ok_chunks": sum(1 for item in items if item["status"] == "ok"),
                "scene": mean([float(item["scene"]) for item in items]),
                "character_action": mean([float(item["character_action"]) for item in items]),
                "camera": mean([float(item["camera"]) for item in items]),
            }
        )
    write_csv(metric_root / "sample_scores.csv", sample_rows, ["sample_id", "chunk_count", "ok_chunks", "scene", "character_action", "camera"])

    summary = {
        "provider": args.provider,
        "provider_id": provider_id,
        "model": model,
        "task_count": len(tasks),
        "chunk_rows": len(rows),
        "sample_rows": len(sample_rows),
        "error_count": len(errors),
        "averages": {
            "scene": mean([float(row["scene"]) for row in rows]),
            "character_action": mean([float(row["character_action"]) for row in rows]),
            "camera": mean([float(row["camera"]) for row in rows]),
        },
        "outputs": {
            "chunk_scores": str(metric_root / "chunk_scores.csv"),
            "sample_scores": str(metric_root / "sample_scores.csv"),
        },
    }
    write_json(metric_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
