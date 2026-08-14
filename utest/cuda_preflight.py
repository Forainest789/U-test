"""Cheap CUDA functionality check before loading the 14B experts."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run_preflight(
    torch_module,
    *,
    driver_version: str | None = None,
    min_free_gb: float = 36.0,
) -> dict:
    """Return machine-readable evidence that CUDA can execute the required BF16 path."""
    report = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "runtime": {
            "torch": str(getattr(torch_module, "__version__", "unknown")),
            "torch_cuda": getattr(getattr(torch_module, "version", None), "cuda", None),
            "driver": driver_version,
        },
    }
    try:
        cuda = torch_module.cuda
        if not cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        initializer = getattr(cuda, "init", None)
        if initializer:
            initializer()
        device_index = int(cuda.current_device())
        properties = cuda.get_device_properties(device_index)
        free_bytes, _ = cuda.mem_get_info(device_index)
        free_gb = float(free_bytes) / 1024**3
        report["device"] = {
            "index": device_index,
            "count": int(cuda.device_count()),
            "name": str(cuda.get_device_name(device_index)),
            "total_memory_gb": round(float(properties.total_memory) / 1024**3, 3),
            "free_memory_gb": round(free_gb, 3),
            "required_free_memory_gb": min_free_gb,
        }
        if free_gb < min_free_gb:
            raise RuntimeError(
                f"CUDA free memory {free_gb:.3f} GiB is below required {min_free_gb:.3f} GiB"
            )
        operand = torch_module.ones((256, 256), dtype=torch_module.bfloat16).to("cuda")
        checksum = float((operand @ operand).sum().item())
        cuda.synchronize()
        if not math.isfinite(checksum):
            raise RuntimeError(f"non-finite BF16 matmul checksum: {checksum}")
        report["smoke"] = {
            "operation": "cpu_to_cuda_bf16_matmul",
            "checksum": checksum,
        }
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def _driver_version() -> str | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.splitlines()[0].strip() if result.returncode == 0 and result.stdout.strip() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-free-gb", type=float, default=36.0)
    args = parser.parse_args()
    try:
        import torch

        report = run_preflight(
            torch,
            driver_version=_driver_version(),
            min_free_gb=args.min_free_gb,
        )
    except Exception as exc:
        report = {
            "schema_version": 1,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 4


if __name__ == "__main__":
    raise SystemExit(main())
