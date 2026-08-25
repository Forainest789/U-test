"""GPU-free scoring and intervention helpers for the identity-token probe."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def percentile_rank(values: Sequence[float]) -> list[float]:
    """Return deterministic average percentile ranks in ``[0, 1]``."""
    finite = [_finite(value, "rank value") for value in values]
    if not finite:
        return []
    if len(finite) == 1:
        return [0.5]
    order = sorted(range(len(finite)), key=lambda index: (finite[index], index))
    output = [0.0] * len(finite)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and finite[order[end]] == finite[order[start]]:
            end += 1
        value = (0.5 * (start + end - 1)) / float(len(order) - 1)
        for position in range(start, end):
            output[order[position]] = value
        start = end
    return output


def score_token_channels(rows: Sequence[Mapping]) -> list[dict]:
    """Rank raw channels within one cell and compute frozen proposal scores."""
    output = [dict(row) for row in rows]
    if not output:
        return output
    raw_names = (
        "name_raw",
        "attribute_raw",
        "persistence_raw_margin",
        "persistence_read_margin",
        "action_attention_raw",
        "scene_raw",
    )
    ranked = {
        name: percentile_rank([_finite(row[name], name) for row in output])
        for name in raw_names
    }
    action_net = [
        max(
            0.0,
            _finite(row["action_hidden_raw"], "action_hidden_raw")
            - max(
                _finite(row["scene_hidden_raw"], "scene_hidden_raw"),
                _finite(row["random_hidden_raw"], "random_hidden_raw"),
            ),
        )
        for row in output
    ]
    action_net_rank = percentile_rank(action_net)
    for index, row in enumerate(output):
        row["s_name"] = ranked["name_raw"][index]
        row["s_attr"] = ranked["attribute_raw"][index]
        row["s_persist"] = 0.5 * (
            ranked["persistence_raw_margin"][index]
            + ranked["persistence_read_margin"][index]
        )
        row["action_hidden_net_raw"] = action_net[index]
        row["s_action"] = math.sqrt(
            ranked["action_attention_raw"][index] * action_net_rank[index]
        )
        row["s_scene"] = ranked["scene_raw"][index]
        row["s_pre"] = (
            row["s_name"] * row["s_attr"] * row["s_persist"]
        ) ** (1.0 / 3.0)
    return output


def causal_metrics(losses: Mapping[str, float], epsilon: float = 1e-12) -> dict:
    """Compute set-level sufficiency/necessity metrics from aligned losses."""
    required = (
        "no_memory",
        "full_correct",
        "identity_only",
        "drop_identity",
        "drop_random",
        "drop_low",
        "wrong_identity",
    )
    missing = [name for name in required if name not in losses]
    if missing:
        raise ValueError(f"missing causal losses: {missing}")
    values = {name: _finite(losses[name], f"{name} loss") for name in required}
    benefit = values["no_memory"] - values["full_correct"]
    if benefit <= float(epsilon):
        raise ValueError("full correct-memory benefit must be positive")
    return {
        "b_full": benefit,
        "r_keep": (values["no_memory"] - values["identity_only"]) / benefit,
        "r_drop": (values["drop_identity"] - values["full_correct"]) / benefit,
        "drop_random_effect": values["drop_random"] - values["full_correct"],
        "drop_low_effect": values["drop_low"] - values["full_correct"],
        "correct_vs_wrong_identity": values["wrong_identity"] - values["identity_only"],
    }


def classify_token(
    row: Mapping,
    *,
    repeat_margin: float,
    benefit_margin: float,
    validation_direction: bool,
) -> list[str]:
    """Apply conjunctive identity gates and diagnostic multi-label rules."""
    repeat_margin = max(0.0, _finite(repeat_margin, "repeat_margin"))
    benefit_margin = max(0.0, _finite(benefit_margin, "benefit_margin"))
    s_name = _finite(row.get("s_name", 0.0), "s_name")
    s_attr = _finite(row.get("s_attr", 0.0), "s_attr")
    s_persist = _finite(row.get("s_persist", 0.0), "s_persist")
    s_action = _finite(row.get("s_action", 0.0), "s_action")
    s_scene = _finite(row.get("s_scene", 0.0), "s_scene")
    causal = _finite(row.get("group_causal_score", 0.0), "group_causal_score")
    control = _finite(row.get("group_control_floor", 0.0), "group_control_floor")
    content_delta = _finite(row.get("content_delta", 0.0), "content_delta")
    high_identity = sum(value >= 0.75 for value in (s_name, s_attr, s_persist))
    causal_identity = (
        high_identity == 3
        and causal > control + repeat_margin
        and content_delta > benefit_margin
        and bool(validation_direction)
    )
    labels: list[str] = []
    if causal_identity:
        labels.append("identity-core candidate")
    elif high_identity >= 2:
        labels.append("identity-associated")
    if max(s_name, s_action) >= 0.75 and (
        content_delta <= benefit_margin or causal <= control + repeat_margin
    ):
        labels.append("attention-only/redundant")
    if s_action >= 0.75 and bool(row.get("action_core_supported", True)):
        labels.append("action-associated")
    if s_scene >= 0.75 and not causal_identity:
        labels.append("scene-associated")
    semantic_labels = [label for label in labels if label.endswith("associated") or label.startswith("identity-core")]
    if len(semantic_labels) >= 2:
        labels.append("mixed")
    return labels or ["unclassified"]
