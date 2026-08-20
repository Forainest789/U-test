"""Pure arithmetic and interpretation for SlotMem denoising utility Q*."""
from __future__ import annotations

import math
from typing import Iterable, Mapping


CONFIRMATORY_RUNS = (
    "correct",
    "correct_repeat",
    "no_memory",
    "zero",
    "random",
    "wrong",
)
SEVEN_RUNS = (*CONFIRMATORY_RUNS, "native")


def _finite(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def qstar_deltas(losses: Mapping[str, float]) -> dict:
    """Return primary Q* and confirmatory control deltas against ``correct``."""
    required = ("correct", "correct_repeat", "no_memory")
    missing = [name for name in required if name not in losses]
    if missing:
        raise ValueError(f"missing losses: {missing}")
    values = {name: _finite(value, f"loss[{name}]") for name, value in losses.items()}
    correct = values["correct"]
    return {
        "qstar": values["no_memory"] - correct,
        "arm_deltas": {
            name: values[name] - correct
            for name in CONFIRMATORY_RUNS
            if name not in {"correct", "correct_repeat", "no_memory"} and name in values
        },
        "repeat_loss_floor": abs(values["correct_repeat"] - correct),
    }


def normalized_influence(reference: Iterable[float], comparison: Iterable[float]) -> float:
    """Compute ||reference-comparison|| / ||reference|| without a tensor dependency."""
    left = [_finite(value, "reference value") for value in reference]
    right = [_finite(value, "comparison value") for value in comparison]
    if len(left) != len(right):
        raise ValueError("prediction shapes differ")
    numerator = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
    denominator = math.sqrt(sum(a * a for a in left))
    return numerator / max(denominator, float.fromhex("0x1.0p-1022"))


def masked_mse(
    prediction: Iterable[float], target: Iterable[float], mask: Iterable[float]
) -> float:
    """Compute weighted MSE for a flattened frozen target-region mask."""
    prediction_values = [_finite(value, "prediction value") for value in prediction]
    target_values = [_finite(value, "target value") for value in target]
    mask_values = [_finite(value, "mask value") for value in mask]
    if not (len(prediction_values) == len(target_values) == len(mask_values)):
        raise ValueError("prediction, target, and mask shapes differ")
    weight = sum(mask_values)
    if weight <= 0:
        raise ValueError("mask selects no values")
    return sum(
        selected * (predicted - expected) ** 2
        for predicted, expected, selected in zip(
            prediction_values, target_values, mask_values
        )
    ) / weight


def classify_qstar(
    *, qstar: float, influence: float, repeat_margin: float, influence_floor: float
) -> str:
    qstar = _finite(qstar, "qstar")
    influence = _finite(influence, "influence")
    repeat_margin = _finite(repeat_margin, "repeat_margin")
    influence_floor = _finite(influence_floor, "influence_floor")
    if influence <= influence_floor:
        return "no_measurable_influence"
    if qstar > repeat_margin:
        return "beneficial"
    return "influence_without_benefit"


def classify_memory_regime(evidence: Mapping[str, object]) -> str:
    positive = int(evidence.get("positive_residual_count", 0) or 0)
    changed = int(evidence.get("bank_hash_change_count", 0) or 0)
    return "dynamic_writer" if positive > 0 and changed > 0 else "static_prefix"
