#!/usr/bin/env python3
"""Run SlotMem outputs through local benchmark repos.

This wrapper intentionally only checks that benchmark repos exist. It never
clones or installs benchmark dependencies.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover - reported at runtime when needed
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VBENCH_DIMS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]


def read_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def natural_chunk_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    for part in reversed(stem.replace("-", "_").split("_")):
        if part.isdigit():
            return int(part), path.name
    return 10**9, path.name


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def load_source_chunks_from_args(sample_dir: Path) -> list[dict]:
    args_path = sample_dir / "inference_args.yaml"
    payload = None
    if yaml is not None and args_path.is_file():
        try:
            payload = yaml.safe_load(args_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
    if payload is None:
        payload = read_json(args_path)
    if not isinstance(payload, dict):
        return []
    args = payload.get("args", {})
    json_path = args.get("json_path") if isinstance(args, dict) else None
    if not json_path:
        return []
    source = read_json(Path(json_path))
    if isinstance(source, dict):
        chunks = source.get("chunks", [])
    else:
        chunks = source
    return chunks if isinstance(chunks, list) else []


def load_slotmem_metadata(sample_dir: Path) -> dict:
    manifest = read_json(sample_dir / "slotmem_inference_metadata.json")
    if isinstance(manifest, dict):
        return manifest

    chunk_records = []
    source_chunks = load_source_chunks_from_args(sample_dir)
    for video_path in sorted(sample_dir.glob("chunk_*.mp4"), key=natural_chunk_key):
        idx, _ = natural_chunk_key(video_path)
        chunk = source_chunks[idx] if 0 <= idx < len(source_chunks) and isinstance(source_chunks[idx], dict) else {}
        chunk_meta = read_json(sample_dir / f"chunk_{idx:03d}.metadata.json")
        if not isinstance(chunk_meta, dict):
            chunk_meta = read_json(sample_dir / f"chunk_{idx:03d}.json")
        if not isinstance(chunk_meta, dict):
            chunk_meta = {}
        caption = chunk_meta.get("caption") or chunk_meta.get("content") or chunk.get("content") or video_path.stem
        characters = chunk_meta.get("characters") or chunk_meta.get("character_list") or chunk.get("character_list") or []
        chunk_records.append(
            {
                "chunk_idx": idx,
                "caption": str(caption),
                "characters": characters,
                "character_list": characters,
                "video_path": str(video_path.resolve()),
                "video_saved": True,
            }
        )
    return {
        "schema_version": 1,
        "format": "slotmem_inference_metadata",
        "output_path": str(sample_dir.resolve()),
        "character_list": sorted({str(c) for row in chunk_records for c in row.get("character_list", [])}),
        "chunks": chunk_records,
    }


def is_slotmem_sample_dir(path: Path) -> bool:
    return (
        (path / "slotmem_inference_metadata.json").is_file()
        or (path / "merged_chunks.mp4").is_file()
        or any(path.glob("chunk_*.mp4"))
    )


def discover_sample_dirs(input_root: Path) -> list[Path]:
    input_root = input_root.resolve()
    if is_slotmem_sample_dir(input_root):
        return [input_root]
    sample_dirs = [p for p in sorted(input_root.iterdir()) if p.is_dir() and is_slotmem_sample_dir(p)]
    if sample_dirs:
        return sample_dirs
    return [p.parent for p in sorted(input_root.rglob("slotmem_inference_metadata.json"))]


def normalize_characters(value) -> list[str]:
    if isinstance(value, dict):
        return sorted(str(k) for k in value.keys())
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict) and "name" in item:
                out.append(str(item["name"]))
            else:
                out.append(str(item))
        return out
    return []


def build_eval_items(input_root: Path) -> list[dict]:
    items = []
    for sample_dir in discover_sample_dirs(input_root):
        meta = load_slotmem_metadata(sample_dir)
        chunks = [row for row in meta.get("chunks", []) if isinstance(row, dict)] if isinstance(meta, dict) else []
        chunks = sorted(chunks, key=lambda row: int(row.get("chunk_idx", 0)))
        captions = [str(row.get("caption") or row.get("content") or f"chunk {i}") for i, row in enumerate(chunks)]
        chunk_characters = [
            normalize_characters(row.get("characters") or row.get("character_list"))
            for row in chunks
        ]
        merged = sample_dir / str(meta.get("merged_output_name", "merged_chunks.mp4") if isinstance(meta, dict) else "merged_chunks.mp4")
        merged_from_manifest = Path(str(meta.get("merged_video_path", ""))) if isinstance(meta, dict) and meta.get("merged_video_path") else None
        if merged_from_manifest and merged_from_manifest.is_file():
            merged = merged_from_manifest
        if merged.is_file():
            items.append(
                {
                    "sample_id": sample_dir.name,
                    "kind": "merged",
                    "source_video": merged,
                    "prompts": captions or [sample_dir.name],
                    "characters": normalize_characters(meta.get("characters")) or normalize_characters(meta.get("character_list")),
                    "chunks": chunks,
                }
            )
            continue
        for row, chars in zip(chunks, chunk_characters):
            video_path = Path(row.get("video_path") or "")
            if not video_path.is_file():
                video_path = sample_dir / f"chunk_{int(row.get('chunk_idx', 0)):03d}.mp4"
            if not video_path.is_file():
                continue
            items.append(
                {
                    "sample_id": f"{sample_dir.name}_chunk_{int(row.get('chunk_idx', 0)):03d}",
                    "kind": "chunk",
                    "source_video": video_path,
                    "prompts": [str(row.get("caption") or row.get("content") or video_path.stem)],
                    "characters": chars,
                    "chunks": [row],
                }
            )
    return items


def prepare_inputs(items: list[dict], prepared_root: Path, mode: str) -> dict:
    video_dir = prepared_root / "video"
    prompt_jsonl = prepared_root / "prompt.jsonl"
    vbench_prompt_json = prepared_root / "vbench_prompts.json"
    records = []
    vbench_prompts = {}
    prepared_root.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    with prompt_jsonl.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(items):
            dst = video_dir / f"sample_{idx}.mp4"
            link_or_copy(Path(item["source_video"]), dst, mode)
            prompts = [str(p) for p in item.get("prompts", []) if str(p).strip()]
            if not prompts:
                prompts = [str(item["sample_id"])]
            f.write(json.dumps({"prompts": prompts}, ensure_ascii=False) + "\n")
            vbench_prompts[dst.name] = " ".join(prompts)
            records.append(
                {
                    "index": idx,
                    "sample_id": item["sample_id"],
                    "kind": item["kind"],
                    "source_video": str(Path(item["source_video"]).resolve()),
                    "prepared_video": str(dst.resolve()),
                    "prompts": prompts,
                    "characters": item.get("characters", []),
                    "chunks": item.get("chunks", []),
                }
            )

    vbench_prompt_json.write_text(json.dumps(vbench_prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "prepared_root": str(prepared_root.resolve()),
        "video_dir": str(video_dir.resolve()),
        "prompt_jsonl": str(prompt_jsonl.resolve()),
        "vbench_prompt_json": str(vbench_prompt_json.resolve()),
        "sample_count": len(records),
        "records": records,
    }
    (prepared_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def require_repo(path: Path, marker: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{marker} repo not found: {path}. Install or place it there; this wrapper will not clone it.")


def write_narrastream_provider_config(args: argparse.Namespace, out_dir: Path) -> Path:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write the NarraStream provider config.")
    base_config = Path(args.narrastream_repo) / "configs" / "default.yaml"
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
    services = cfg.setdefault("services", {})
    if args.provider_mode == "vllm":
        provider = "openai"
        base_url = args.vllm_base_url
        api_key_env = None
        api_key = args.vllm_api_key
        model = args.vllm_model
    elif args.provider_mode == "local-qwen35":
        provider = "local_qwen3_vl"
        base_url = None
        api_key_env = None
        api_key = None
        model = args.local_qwen35_model
    else:
        provider = args.api_provider
        base_url = args.api_base_url
        api_key_env = args.api_key_env
        api_key = None
        model = args.api_model
    for name in ("mllm", "vlm", "planner"):
        service = dict(services.get(name, {}))
        service.update({"provider": provider, "base_url": base_url, "model": model})
        if api_key_env:
            service["api_key_env"] = api_key_env
            service.pop("api_key", None)
        elif api_key:
            service["api_key"] = api_key
            service.pop("api_key_env", None)
        else:
            service.pop("api_key", None)
            service.pop("api_key_env", None)
        if base_url is None:
            service.pop("base_url", None)
        services[name] = service
    out_path = out_dir / f"narrastream_{args.provider_mode}_config.yaml"
    out_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out_path


def run_command(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("+ " + " ".join(subprocess.list2cmdline([part]) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VBench + NarraStream-Bench for SlotMem inference outputs.")
    parser.add_argument("input_root", type=Path, nargs="?", help="SlotMem output folder or a parent folder containing sample output folders.")
    parser.add_argument("--infer-output", type=Path, default=None, help="Alias for input_root.")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "benchmark_outputs" / "slotmem")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--provider-mode", choices=["api", "vllm", "local-qwen35"], default=os.environ.get("PROVIDER_MODE", "api"))
    parser.add_argument("--benchmarks", nargs="+", choices=["vbench", "narrastream"], default=["vbench", "narrastream"])
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--segment-duration", default=os.environ.get("SEGMENT_DURATION", "10"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--gpu-id", default=os.environ.get("GPU_ID", ""))
    parser.add_argument("--device", default=os.environ.get("BENCH_DEVICE", "auto"))
    parser.add_argument("--vbench-repo", type=Path, default=REPO_ROOT / "bench" / "VBench")
    parser.add_argument("--vbench-dimensions", nargs="+", default=DEFAULT_VBENCH_DIMS)
    parser.add_argument("--narrastream-repo", type=Path, default=REPO_ROOT / "bench" / "NarraStream-Bench")
    parser.add_argument("--narrastream-path-config", type=Path, default=Path(os.environ["NARRASTREAM_PATH_CONFIG"]) if os.environ.get("NARRASTREAM_PATH_CONFIG") else None)
    parser.add_argument("--narrastream-metrics", nargs="+", default=None, help="Default omitted means NarraStream runs all metrics.")
    parser.add_argument("--api-provider", default=os.environ.get("NARRASTREAM_API_PROVIDER", "openai"))
    parser.add_argument("--api-base-url", default=os.environ.get("NARRASTREAM_API_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default=os.environ.get("NARRASTREAM_API_KEY_ENV", "OPENAI_API_KEY"))
    parser.add_argument("--api-model", default=os.environ.get("NARRASTREAM_API_MODEL", "gpt-4.1"))
    parser.add_argument("--vllm-base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:18030/v1"))
    parser.add_argument("--vllm-api-key", default=os.environ.get("VLLM_API_KEY", "local-qwen30b"))
    parser.add_argument("--vllm-model", default=os.environ.get("VLLM_MODEL", "qwen30b-vllm"))
    parser.add_argument("--local-qwen35-model", default=os.environ.get("LOCAL_QWEN35_MODEL", "/models/Qwen3.5-4B"))
    parser.add_argument("--api-workers", type=int, default=int(os.environ.get("API_WORKERS", "4")))
    args = parser.parse_args()
    args.input_root = args.infer_output or args.input_root
    if args.input_root is None:
        parser.error("provide input_root or --infer-output")
    return args


def main() -> None:
    args = parse_args()
    os.environ["PYTHON_BIN"] = str(args.python_bin)
    input_root = args.input_root.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"input_root not found: {input_root}")
    if "vbench" in args.benchmarks:
        require_repo(args.vbench_repo, "VBench")
        require_repo(args.vbench_repo / "evaluate.py", "VBench evaluate.py")
    if "narrastream" in args.benchmarks:
        require_repo(args.narrastream_repo, "NarraStream-Bench")
        require_repo(args.narrastream_repo / "scripts" / "run_narrastream_bench.sh", "NarraStream runner")

    run_name = args.run_name or input_root.name
    run_root = (args.output_root / run_name).resolve()
    prepared = prepare_inputs(build_eval_items(input_root), run_root / "prepared", args.link_mode)
    if prepared["sample_count"] <= 0:
        raise RuntimeError(f"No SlotMem videos discovered under {input_root}")
    print(json.dumps({"prepared": prepared}, ensure_ascii=False, indent=2), flush=True)

    if "vbench" in args.benchmarks:
        cmd = [
            args.python_bin,
            "evaluate.py",
            "--output_path",
            str(run_root / "vbench"),
            "--videos_path",
            prepared["video_dir"],
            "--dimension",
            *args.vbench_dimensions,
            "--mode",
            "custom_input",
            "--prompt_file",
            prepared["vbench_prompt_json"],
        ]
        run_command(cmd, args.vbench_repo, args.dry_run)

    if "narrastream" in args.benchmarks:
        ns_config = write_narrastream_provider_config(args, run_root)
        cmd = [
            "bash",
            "scripts/run_narrastream_bench.sh",
            "--run-name",
            run_name,
            "--video-dir",
            prepared["video_dir"],
            "--prompts",
            prepared["prompt_jsonl"],
            "--segment-duration",
            str(args.segment_duration),
            "--output-root",
            str(run_root / "narrastream"),
            "--config",
            str(ns_config),
            "--api-workers",
            str(args.api_workers),
            "--device",
            str(args.device),
        ]
        if args.narrastream_path_config:
            cmd.extend(["--path-config", str(args.narrastream_path_config)])
        if args.gpu_id:
            cmd.extend(["--gpu-id", str(args.gpu_id)])
        if args.narrastream_metrics:
            cmd.extend(["--metrics", *args.narrastream_metrics])
        run_command(cmd, args.narrastream_repo, args.dry_run)

    print(f"Benchmark wrapper complete. Output root: {run_root}", flush=True)


if __name__ == "__main__":
    main()
