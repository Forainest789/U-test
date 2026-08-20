"""Paired seven-run teacher-forced probe orchestration for SlotMem Q*."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .qstar import SEVEN_RUNS, classify_qstar, qstar_deltas


@dataclass(frozen=True)
class ProbeCell:
    event_id: str
    memory_id: str
    horizon: int
    timestep_index: int
    timestep: float
    clean_target: object
    noise: object
    noisy_latent: object
    flow_target: object
    input_hashes: Mapping[str, str]
    target_mask: object | None = None


def _flatten(value) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        output: list[float] = []
        for item in value:
            output.extend(_flatten(item))
        return output
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("probe tensor values must be finite")
    return [number]


def tensor_sha256(value) -> str:
    """Hash tensor values plus shape/dtype when available, or canonical JSON values."""
    if hasattr(value, "detach"):
        tensor = value.detach().contiguous().cpu()
        header = json.dumps(
            {"shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = tensor.view(getattr(__import__("torch"), "uint8")).numpy().tobytes()
        return hashlib.sha256(header + b"\0" + raw).hexdigest()
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mse(prediction, target, mask=None) -> float:
    if hasattr(prediction, "detach") and hasattr(target, "detach"):
        error = (prediction.detach().float() - target.detach().float()).square()
        if mask is not None:
            selected = mask.detach().float()
            while selected.ndim < error.ndim:
                selected = selected.unsqueeze(0)
            selected = selected.expand_as(error)
            weight = selected.sum()
            if float(weight.item()) <= 0:
                raise ValueError("mask selects no values")
            result = (error * selected).sum() / weight
        else:
            result = error.mean()
        value = float(result.item())
    else:
        predicted = _flatten(prediction)
        expected = _flatten(target)
        if len(predicted) != len(expected):
            raise ValueError("prediction and target shapes differ")
        weights = [1.0] * len(predicted) if mask is None else _flatten(mask)
        if len(weights) != len(predicted):
            raise ValueError("mask shape differs from prediction")
        weight = sum(weights)
        if weight <= 0:
            raise ValueError("mask selects no values")
        value = sum(w * (a - b) ** 2 for a, b, w in zip(predicted, expected, weights)) / weight
    if not math.isfinite(value):
        raise ValueError("probe loss must be finite")
    return value


def _influence(reference, comparison) -> float:
    if hasattr(reference, "detach") and hasattr(comparison, "detach"):
        left = reference.detach().float()
        right = comparison.detach().float()
        denominator = float(left.norm().item())
        value = float((left - right).norm().item()) / max(denominator, 1e-30)
    else:
        left = _flatten(reference)
        right = _flatten(comparison)
        if len(left) != len(right):
            raise ValueError("prediction shapes differ")
        value = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right))) / max(
            math.sqrt(sum(a * a for a in left)), 1e-30
        )
    if not math.isfinite(value):
        raise ValueError("prediction influence must be finite")
    return value


def validate_probe_runtime(frozen: Mapping, actual: Mapping) -> None:
    errors = [
        f"{key}_mismatch"
        for key in ("target_prompt_sha256", "target_seed")
        if actual.get(key) != frozen.get(key)
    ]
    if errors:
        raise ValueError(",".join(errors))


def evaluate_probe_cell(
    cell: ProbeCell,
    payloads: Mapping[str, object],
    predictor: Callable[[str, ProbeCell, object, bool], Mapping],
    *,
    repeat_tolerance: float = 0.0,
    influence_floor: float = 0.0,
) -> tuple[dict, list[dict]]:
    """Evaluate one immutable cell; the caller owns payload construction and model loading."""
    missing = [name for name in ("correct", "no_memory", "zero", "random", "wrong") if name not in payloads]
    if missing:
        raise ValueError(f"missing arm payloads: {missing}")
    outputs: dict[str, Mapping] = {}
    records: list[dict] = []
    for run_name in SEVEN_RUNS:
        source_arm = "correct" if run_name == "correct_repeat" else run_name
        native = run_name == "native"
        payload = None if native else payloads[source_arm]
        result = dict(predictor(run_name, cell, payload, native))
        if "prediction" not in result:
            raise ValueError(f"{run_name}: predictor did not return prediction")
        loss = _mse(result["prediction"], cell.flow_target)
        masked_loss = (
            _mse(result["prediction"], cell.flow_target, cell.target_mask)
            if cell.target_mask is not None
            else None
        )
        outputs[run_name] = {**result, "loss": loss, "masked_loss": masked_loss}
        records.append(
            {
                "event_id": cell.event_id,
                "memory_id": cell.memory_id,
                "horizon": int(cell.horizon),
                "timestep_index": int(cell.timestep_index),
                "timestep": float(cell.timestep),
                "arm": run_name,
                "role": "diagnostic" if native else "confirmatory",
                "loss": loss,
                "masked_loss": masked_loss,
                "prediction_sha256": tensor_sha256(result["prediction"]),
                "input_hashes": dict(cell.input_hashes),
                "memory_read_hit": bool(result.get("memory_read_hit", False)),
                "injection_delta_norm": float(result.get("injection_delta_norm", 0.0) or 0.0),
                "diagnostics": dict(result.get("diagnostics", {})),
            }
        )
    losses = {name: float(output["loss"]) for name, output in outputs.items()}
    deltas = qstar_deltas(losses)
    repeat_prediction_floor = _influence(
        outputs["correct"]["prediction"], outputs["correct_repeat"]["prediction"]
    )
    if deltas["repeat_loss_floor"] > float(repeat_tolerance) or repeat_prediction_floor > float(repeat_tolerance):
        raise ValueError("correct_repeat exceeds frozen tolerance")
    primary_influence = _influence(
        outputs["correct"]["prediction"], outputs["no_memory"]["prediction"]
    )
    report = {
        "event_id": cell.event_id,
        "memory_id": cell.memory_id,
        "horizon": int(cell.horizon),
        "timestep_index": int(cell.timestep_index),
        "timestep": float(cell.timestep),
        "input_hashes": dict(cell.input_hashes),
        "losses": losses,
        **deltas,
        "repeat_prediction_floor": repeat_prediction_floor,
        "primary_influence": primary_influence,
        "classification": classify_qstar(
            qstar=deltas["qstar"],
            influence=primary_influence,
            repeat_margin=max(float(repeat_tolerance), deltas["repeat_loss_floor"]),
            influence_floor=float(influence_floor),
        ),
    }
    return report, records


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_probe_outputs(output: Path, cells: Sequence[Mapping], records: Sequence[Mapping]) -> None:
    output = output.resolve()
    report = {
        "schema_version": 1,
        "status": "passed",
        "primary_estimand": "L_no_memory - L_correct",
        "native_is_diagnostic": True,
        "cells": list(cells),
    }
    _atomic_text(output / "qstar_report.json", json.dumps(report, indent=2, ensure_ascii=False))
    _atomic_text(
        output / "qstar_records.jsonl",
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in records),
    )


def self_check() -> None:
    cell = ProbeCell(
        event_id="self-check",
        memory_id="person|0",
        horizon=1,
        timestep_index=0,
        timestep=999.0,
        clean_target=[0.0],
        noise=[1.0],
        noisy_latent=[1.0],
        flow_target=[1.0],
        input_hashes={"prefix": "self-check"},
    )
    payloads = {"correct": 0.9, "no_memory": 0.0, "zero": 0.0, "random": 0.2, "wrong": 0.1}
    report, records = evaluate_probe_cell(
        cell,
        payloads,
        lambda arm, shared, payload, native: {
            "prediction": [0.0 if native else float(payload or 0.0)],
            "memory_read_hit": not native and arm != "no_memory",
        },
    )
    assert report["qstar"] > 0 and len(records) == 7
    print("[qstar] self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    parser.error("production probe arguments are added by the SlotMem runtime integration")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
