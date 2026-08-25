"""GPU-free scoring and intervention helpers for the identity-token probe."""
from __future__ import annotations

import math
import hashlib
import json
import random
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
        and content_delta > 0.0
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


def flat_to_coord(flat_idx: int, height: int, width: int) -> tuple[int, int, int]:
    spatial = int(height) * int(width)
    if int(flat_idx) < 0 or spatial <= 0:
        raise ValueError("invalid flattened coordinate")
    index = int(flat_idx)
    return index // spatial, (index % spatial) // int(width), index % int(width)


def _centroid(group: Sequence[int], coordinates: Mapping[int, tuple[int, int, int]], scale):
    return tuple(
        sum(coordinates[index][axis] / scale[axis] for index in group) / len(group)
        for axis in range(3)
    )


def _nearest_group_index(groups, source_index, coordinates, scale) -> int:
    source = _centroid(groups[source_index], coordinates, scale)
    candidates = []
    for index, group in enumerate(groups):
        if index == source_index:
            continue
        target = _centroid(group, coordinates, scale)
        distance = sum((left - right) ** 2 for left, right in zip(source, target))
        candidates.append((distance, tuple(sorted(group)), index))
    if not candidates:
        raise ValueError("cannot merge the only candidate group")
    return min(candidates)[2]


def build_candidate_groups(
    indices: Sequence[int],
    *,
    height: int,
    width: int,
    max_groups: int = 8,
    min_group_size: int = 4,
) -> list[dict]:
    """Build deterministic coordinate-only spatiotemporal candidate groups."""
    unique = sorted({int(index) for index in indices})
    max_groups = int(max_groups)
    min_group_size = int(min_group_size)
    if len(unique) < min_group_size or max_groups <= 0 or min_group_size <= 0:
        raise ValueError("candidate count and group limits must be positive")
    coordinates = {index: flat_to_coord(index, height, width) for index in unique}

    # ponytail: the candidate universe is small in this probe. O(n^2) adjacency keeps
    # the grouping dependency-free; replace with grid buckets if it grows past ~10k.
    remaining = set(unique)
    groups: list[list[int]] = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack = [root]
        component = [root]
        while stack:
            current = stack.pop()
            ct, cy, cx = coordinates[current]
            neighbors = [
                other
                for other in sorted(remaining)
                if max(
                    abs(coordinates[other][0] - ct),
                    abs(coordinates[other][1] - cy),
                    abs(coordinates[other][2] - cx),
                ) <= 1
            ]
            for other in neighbors:
                remaining.remove(other)
                stack.append(other)
                component.append(other)
        groups.append(sorted(component))

    maxima = [max(coord[axis] for coord in coordinates.values()) for axis in range(3)]
    minima = [min(coord[axis] for coord in coordinates.values()) for axis in range(3)]
    scale = tuple(float(max(maxima[axis] - minima[axis], 1)) for axis in range(3))

    while len(groups) > 1 and any(len(group) < min_group_size for group in groups):
        source = min(
            (index for index, group in enumerate(groups) if len(group) < min_group_size),
            key=lambda index: (len(groups[index]), tuple(groups[index])),
        )
        target = _nearest_group_index(groups, source, coordinates, scale)
        merged = sorted(groups[source] + groups[target])
        for index in sorted((source, target), reverse=True):
            groups.pop(index)
        groups.append(merged)
        groups.sort(key=lambda group: tuple(group))

    while len(groups) > max_groups:
        source = min(range(len(groups)), key=lambda index: (len(groups[index]), tuple(groups[index])))
        target = _nearest_group_index(groups, source, coordinates, scale)
        merged = sorted(groups[source] + groups[target])
        for index in sorted((source, target), reverse=True):
            groups.pop(index)
        groups.append(merged)
        groups.sort(key=lambda group: tuple(group))

    while len(groups) < max_groups:
        split_candidates = []
        for group_index, group in enumerate(groups):
            if len(group) < 2 * min_group_size:
                continue
            spans = [
                (max(coordinates[index][axis] for index in group) - min(coordinates[index][axis] for index in group))
                / scale[axis]
                for axis in range(3)
            ]
            axis = max(range(3), key=lambda value: (spans[value], -value))
            ordered = sorted(group, key=lambda index: (coordinates[index][axis], coordinates[index], index))
            midpoint = len(ordered) // 2
            left, right = ordered[:midpoint], ordered[midpoint:]
            if len(left) >= min_group_size and len(right) >= min_group_size:
                split_candidates.append((-len(group), tuple(group), group_index, left, right))
        if not split_candidates:
            break
        _, _, group_index, left, right = min(split_candidates)
        groups.pop(group_index)
        groups.extend((sorted(left), sorted(right)))
        groups.sort(key=lambda group: tuple(group))

    return [
        {
            "group_id": f"g{group_index:02d}",
            "indices": group,
            "coordinates": [list(coordinates[index]) for index in group],
        }
        for group_index, group in enumerate(sorted(groups, key=lambda group: tuple(group)))
    ]


def group_membership_sha256(groups: Sequence[Mapping]) -> str:
    canonical = [
        {
            "group_id": str(group["group_id"]),
            "indices": [int(index) for index in group["indices"]],
        }
        for group in groups
    ]
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frame_histogram(indices: Sequence[int], spatial: int) -> dict[int, int]:
    output: dict[int, int] = {}
    for index in indices:
        frame = int(index) // int(spatial)
        output[frame] = output.get(frame, 0) + 1
    return output


def build_intervention_masks(
    original: Sequence[int],
    universe: Sequence[int],
    scores: Mapping[int, float],
    *,
    budget_fraction: float,
    seed: int,
    height: int,
    width: int,
) -> dict[str, list[int]]:
    """Freeze identity and matched-control masks with identical frame counts."""
    original_set = sorted({int(index) for index in original})
    universe_set = sorted({int(index) for index in universe})
    if len(original_set) < 4 or not set(original_set).issubset(universe_set):
        raise ValueError("original query mask must contain at least four universe tokens")
    fraction = _finite(budget_fraction, "budget_fraction")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("budget_fraction must be in (0, 1]")
    budget = max(4, math.ceil(fraction * len(original_set)))
    if budget > len(universe_set):
        raise ValueError("candidate universe is smaller than intervention budget")
    ranked = []
    for index in universe_set:
        if index not in scores:
            raise ValueError(f"missing score for candidate {index}")
        ranked.append((-_finite(scores[index], f"score[{index}]"), index))
    identity = sorted(index for _, index in sorted(ranked)[:budget])
    spatial = int(height) * int(width)
    if spatial <= 0:
        raise ValueError("height and width must be positive")
    target_histogram = _frame_histogram(identity, spatial)
    identity_set = set(identity)
    available_by_frame: dict[int, list[int]] = {}
    for index in universe_set:
        if index not in identity_set:
            available_by_frame.setdefault(index // spatial, []).append(index)
    low: list[int] = []
    random_control: list[int] = []
    generator = random.Random(int(seed))
    for frame, count in sorted(target_histogram.items()):
        available = available_by_frame.get(frame, [])
        if len(available) < count:
            raise ValueError("cannot match intervention frame histogram")
        low.extend(sorted(available, key=lambda index: (_finite(scores[index], "score"), index))[:count])
        random_control.extend(sorted(generator.sample(available, count)))
    low = sorted(low)
    random_control = sorted(random_control)
    if not (
        _frame_histogram(low, spatial)
        == _frame_histogram(random_control, spatial)
        == target_histogram
    ):
        raise ValueError("intervention frame histogram mismatch")
    universe_members = set(universe_set)
    return {
        "identity_top": identity,
        "random": random_control,
        "low_score": low,
        "wrong_identity": list(identity),
        "drop_identity": sorted(universe_members - set(identity)),
        "drop_random": sorted(universe_members - set(random_control)),
        "drop_low": sorted(universe_members - set(low)),
    }
