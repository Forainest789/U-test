"""Fast teacher-forced identity-token causal probe for SlotMem."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
import statistics
import time
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Sequence

from .identity_token_scoring import (
    build_candidate_groups,
    build_intervention_masks,
    causal_metrics,
    classify_token,
    percentile_rank,
    score_token_channels,
)


ALL_LAYERS = tuple(range(16))
DEFAULT_TIMESTEPS = (0, 25, 49)
DEFAULT_GROUPS = (
    tuple(range(0, 5)),
    tuple(range(5, 11)),
    tuple(range(11, 16)),
)

DELTA8_MARA_ATTRIBUTES = (
    "short copper bob",
    "teal scarf",
    "crescent-shaped scar above her left eyebrow",
    "mustard raincoat",
)
DELTA8_ACTION_CORE = (
    "runs",
    "two steps",
    "catches",
    "closing",
    "looks up",
    "toward camera",
)
DELTA8_ACTION_CONTEXT = ("tram door", "one hand")
DELTA8_SCENE = ("platform", "tram", "rain", "commuters", "dusk")


@dataclass(frozen=True)
class ScreeningRun:
    stage: str
    timestep_index: int
    layer_group: tuple[int, ...]
    arm: str


def semantic_group_manifest(story: Mapping, event: Mapping) -> dict[str, list[str]]:
    """Freeze the first-round Mara identity/action/scene diagnostic vocabulary."""
    character = str(event.get("character_name", "")).strip()
    target_index = int(event.get("target_chunk_idx", -1))
    chunks = story.get("chunks", []) if isinstance(story, Mapping) else []
    characters = story.get("characters", {}) if isinstance(story, Mapping) else {}
    if character != "Mara" or target_index < 0 or target_index >= len(chunks):
        raise ValueError("first identity probe supports the frozen delta8 Mara event")
    source_text = str(characters.get(character, ""))
    target_text = str(chunks[target_index].get("content", ""))
    missing_attributes = [
        phrase for phrase in DELTA8_MARA_ATTRIBUTES if phrase.casefold() not in source_text.casefold()
    ]
    missing_target = [
        phrase
        for phrase in (*DELTA8_ACTION_CORE, *DELTA8_ACTION_CONTEXT, *DELTA8_SCENE)
        if phrase.casefold() not in target_text.casefold()
    ]
    if missing_attributes or missing_target:
        raise ValueError(
            f"frozen semantic phrases missing: attributes={missing_attributes}, target={missing_target}"
        )
    return {
        "identity_name": [character],
        "stable_attributes": list(DELTA8_MARA_ATTRIBUTES),
        "action_core": list(DELTA8_ACTION_CORE),
        "action_context": list(DELTA8_ACTION_CONTEXT),
        "scene": list(DELTA8_SCENE),
    }


def s2_forward_budget(*, max_groups: int, has_validation: bool) -> int:
    """Return the hard measured-forward budget for S2, excluding two captures."""
    groups = min(max(0, int(max_groups)), 5 if has_validation else 8)
    all_token_diagnostics = 2
    primary_baselines = 2  # expanded full-correct and expanded wrong; no-memory is reused.
    hidden_controls = 3  # action, scene, seeded random text neutralization.
    equal_budget_arms = 7
    validation_arms = 4 if has_validation else 0
    return (
        all_token_diagnostics
        + primary_baselines
        + hidden_controls
        + groups
        + equal_budget_arms
        + validation_arms
    )


def finalize_s2(
    token_rows: Sequence[Mapping],
    losses: Mapping[str, float],
    *,
    identity_fraction: float,
    repeat_margin: float,
    benefit_margin: float,
    validation_direction: bool,
) -> dict:
    """Apply the frozen set-level gate before emitting token-level labels."""
    metrics = causal_metrics(losses)
    control_floor = max(metrics["drop_random_effect"], metrics["drop_low_effect"]) / metrics["b_full"]
    drop_identity_effect = float(losses["drop_identity"]) - float(losses["full_correct"])
    drop_control_effect = max(
        float(losses["drop_random"]) - float(losses["full_correct"]),
        float(losses["drop_low"]) - float(losses["full_correct"]),
    )
    checks = {
        "identity_fraction": float(identity_fraction) <= 0.25,
        "sufficiency": metrics["r_keep"] >= 0.8,
        "necessity": drop_identity_effect > drop_control_effect + float(repeat_margin),
        "content_specific": metrics["correct_vs_wrong_identity"] > float(benefit_margin),
        "validation_direction": bool(validation_direction),
    }
    passed = all(checks.values())
    rows = []
    for source in token_rows:
        row = dict(source)
        row["group_control_floor"] = control_floor
        row["labels"] = classify_token(
            row,
            repeat_margin=float(repeat_margin) / metrics["b_full"],
            benefit_margin=float(benefit_margin),
            validation_direction=bool(validation_direction and passed),
        )
        rows.append(row)
    return {
        "metrics": metrics,
        "checks": checks,
        "gate": {
            "status": "PASS" if passed else "BLOCK",
            "reasons": [name for name, value in checks.items() if not value],
        },
        "token_rows": rows,
    }


def build_screening_schedule(
    timesteps: Sequence[int] = DEFAULT_TIMESTEPS,
    layer_groups: Sequence[Sequence[int]] = DEFAULT_GROUPS,
) -> list[ScreeningRun]:
    """Return the deduplicated S0/S1 schedule in execution order."""
    timesteps = tuple(int(value) for value in timesteps)
    groups = tuple(tuple(int(layer) for layer in group) for group in layer_groups)
    if len(timesteps) != 3 or len(groups) != 3:
        raise ValueError("screening requires exactly three timesteps and layer groups")
    middle = timesteps[1]
    schedule = [
        ScreeningRun("S0", middle, ALL_LAYERS, arm)
        for arm in ("correct", "correct_repeat", "zero", "no_memory")
    ]
    schedule.extend(
        ScreeningRun("S1", timestep, ALL_LAYERS, "no_memory")
        for timestep in (timesteps[0], timesteps[2])
    )
    schedule.extend(
        ScreeningRun("S1", timestep, group, arm)
        for timestep in timesteps
        for group in groups
        for arm in ("correct", "wrong")
    )
    schedule.append(ScreeningRun("S1", middle, ALL_LAYERS, "wrong"))
    return schedule


def select_cells(
    records: Sequence[Mapping], *, repeat_margin: float, influence_floor: float
) -> dict:
    """Select primary/validation cells by content utility, then localized authority."""
    repeat_margin = float(repeat_margin)
    influence_floor = float(influence_floor)
    eligible = []
    for source in records:
        row = dict(source)
        q_content = float(row.get("q_content", float("nan")))
        ratio = float(row.get("delta_host_ratio", 0.0) or 0.0)
        influence = float(row.get("content_influence", ratio) or 0.0)
        if not all(math.isfinite(value) for value in (q_content, ratio, influence)):
            raise ValueError("cell selection values must be finite")
        if q_content > repeat_margin and influence > influence_floor:
            eligible.append(row)
    eligible.sort(
        key=lambda row: (
            -float(row["q_content"]),
            -float(row.get("delta_host_ratio", 0.0) or 0.0),
            int(row["timestep_index"]),
            tuple(int(layer) for layer in row["layer_group"]),
        )
    )
    return {
        "primary": eligible[0] if eligible else None,
        "validation": eligible[1] if len(eligible) > 1 else None,
        "eligible_count": len(eligible),
    }


def _layer_diagnostics(result: Mapping) -> dict:
    by_layer = result.get("sparse_role_memory_stats_by_layer", {})
    rows = [row for row in by_layer.values() if isinstance(row, Mapping)] if isinstance(by_layer, Mapping) else []
    ratios = []
    for row in rows:
        delta = float(row.get("effective_delta_norm", 0.0) or 0.0)
        host = float(row.get("host_token_norm", 0.0) or 0.0)
        if not math.isfinite(delta) or not math.isfinite(host):
            raise ValueError("sparse layer diagnostics must be finite")
        if host > 0.0:
            ratios.append(delta / host)
    return {
        "selected_query_tokens": sum(int(row.get("selected_query_tokens", 0) or 0) for row in rows),
        "selected_memory_tokens": sum(int(row.get("selected_memory_tokens", 0) or 0) for row in rows),
        "effective_delta_norm": max(
            (float(row.get("effective_delta_norm", 0.0) or 0.0) for row in rows),
            default=0.0,
        ),
        "delta_host_ratio": max(ratios, default=0.0),
        "attention_entropy": (
            sum(float(row.get("attn_entropy", 0.0) or 0.0) for row in rows) / len(rows)
            if rows else 0.0
        ),
    }


def _run_screening_forward(context: Mapping, run: ScreeningRun) -> dict:
    from .qstar_probe import _mse, tensor_sha256, validate_measured_injection

    engine = context["engine"]
    cell = next(
        cell for cell in context["cells"] if int(cell.timestep_index) == int(run.timestep_index)
    )
    source_arm = "correct" if run.arm == "correct_repeat" else run.arm
    bundle = context["payloads"][source_arm]
    noisy_latent = cell.noisy_latent
    if hasattr(noisy_latent, "to"):
        noisy_latent = noisy_latent.to(engine.device)
    started = time.perf_counter()
    result = engine.generate_chunk(
        prompt=context["prompt"],
        memory_tokens=bundle["memory_tokens"],
        memory_bank_tokens=bundle["memory_bank_tokens"],
        memory_bank_percents=bundle["memory_bank_percents"],
        memory_bank_token_meta=bundle["memory_bank_token_meta"],
        memory_token_lengths_per_character=bundle["memory_token_lengths_per_character"],
        ref_images=context["reference_frames"],
        random_ref_frame=context["fixed_reference"],
        seed=context["target_seed"],
        online_memory_chars=[],
        online_memory_bank_percents=[],
        teacher_forced_probe={
            "timestep_index": int(cell.timestep_index),
            "noisy_latents": noisy_latent,
            "force_memory_path": True,
        },
    )
    elapsed = time.perf_counter() - started
    prediction = result.get("prediction_cond")
    if prediction is None:
        raise ValueError(f"{run.arm}: predictor did not return conditional velocity")
    backend = str(result.get("attention_implementation", ""))
    validate_measured_injection(run.arm, bool(bundle["target_read_hit"]), result)
    if run.arm == "no_memory" and not bool(result.get("forced_memory_path", False)):
        raise ValueError("no_memory did not use the memory-aware forward")
    diagnostics = _layer_diagnostics(result)
    return {
        "stage": run.stage,
        "timestep_index": int(run.timestep_index),
        "timestep": float(cell.timestep),
        "layer_group": list(run.layer_group),
        "arm": run.arm,
        "loss": _mse(prediction, cell.flow_target),
        "prediction_sha256": tensor_sha256(prediction),
        "input_hashes": dict(cell.input_hashes),
        "payload_sha256": bundle.get("target_payload_sha256"),
        "attention_implementation": backend,
        "wall_time_s": elapsed,
        "diagnostics": diagnostics,
        "_prediction": prediction,
    }


def _screening_cells(records: Sequence[Mapping]) -> list[dict]:
    from .qstar_probe import _influence

    lookup = {
        (int(row["timestep_index"]), tuple(row["layer_group"]), str(row["arm"])): row
        for row in records
    }
    no_memory = {
        int(row["timestep_index"]): row
        for row in records
        if row["arm"] == "no_memory"
    }
    pairs = sorted({
        (int(row["timestep_index"]), tuple(row["layer_group"]))
        for row in records
        if row["arm"] == "correct"
    })
    output = []
    for timestep, layers in pairs:
        correct = lookup[(timestep, layers, "correct")]
        wrong = lookup.get((timestep, layers, "wrong"))
        if wrong is None:
            continue
        no = no_memory[timestep]
        output.append({
            "timestep_index": timestep,
            "layer_group": list(layers),
            "loss_correct": float(correct["loss"]),
            "loss_wrong": float(wrong["loss"]),
            "loss_no_memory": float(no["loss"]),
            "q_presence": float(no["loss"]) - float(correct["loss"]),
            "q_content": float(wrong["loss"]) - float(correct["loss"]),
            "content_influence": _influence(correct["_prediction"], wrong["_prediction"]),
            "delta_host_ratio": float(correct["diagnostics"]["delta_host_ratio"]),
        })
    return output


def _as_flat_list(value) -> list[float]:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        output = []
        for item in value:
            output.extend(_as_flat_list(item))
        return output
    return [float(value)]


def _query_indices(payload, role: str) -> list[int]:
    if not isinstance(payload, Mapping):
        return []
    if bool(payload.get("__layerwise__", False)):
        return sorted({
            index
            for layer_payload in payload.get("layers", {}).values()
            for index in _query_indices(layer_payload, role)
        })
    role_payload = payload.get(str(role), {})
    if not isinstance(role_payload, Mapping):
        return []
    return sorted({int(value) for value in _as_flat_list(role_payload.get("flat_idx", []))})


def _semantic_vector(maps: Mapping, phrase: str, layers: Sequence[int], token_count: int) -> list[float]:
    role_maps = maps.get(str(phrase), {}) if isinstance(maps, Mapping) else {}
    vectors = []
    for layer in layers:
        value = role_maps.get(layer, role_maps.get(str(layer))) if isinstance(role_maps, Mapping) else None
        if value is None:
            continue
        if hasattr(value, "detach"):
            tensor = value.detach().float().cpu()
            while tensor.ndim > 1:
                tensor = tensor.mean(dim=0)
            vector = tensor.reshape(-1).tolist()
        else:
            vector = _as_flat_list(value)
        if len(vector) >= token_count:
            vectors.append([float(item) for item in vector[:token_count]])
    if not vectors:
        raise ValueError(f"semantic map missing for {phrase!r} at selected layers")
    return [statistics.fmean(vector[index] for vector in vectors) for index in range(token_count)]


def _token_diagnostics(result: Mapping, layers: Sequence[int]) -> dict[int, dict[int, dict]]:
    by_layer = result.get("sparse_role_memory_stats_by_layer", {})
    output: dict[int, dict[int, dict]] = {}
    for layer in layers:
        stats = by_layer.get(str(layer), by_layer.get(layer, {})) if isinstance(by_layer, Mapping) else {}
        diagnostics = stats.get("token_diagnostics", {}) if isinstance(stats, Mapping) else {}
        indices = [int(value) for value in _as_flat_list(diagnostics.get("flat_idx", []))]
        if not indices:
            continue
        layer_rows = {}
        for position, index in enumerate(indices):
            row = {}
            for name in (
                "host_norm",
                "raw_delta_norm",
                "effective_delta_norm",
                "raw_cosine_max",
                "read_logsumexp",
            ):
                values = _as_flat_list(diagnostics.get(name, []))
                if position >= len(values):
                    raise ValueError(f"{name} is not aligned with flat_idx")
                row[name] = float(values[position])
            for name in (
                "host_features",
                "raw_delta_features",
                "effective_delta_features",
            ):
                value = diagnostics.get(name)
                if value is None or not hasattr(value, "__getitem__"):
                    raise ValueError(f"{name} is missing from token diagnostics")
                row[name] = value[position]
            layer_rows[index] = row
        output[int(layer)] = layer_rows
    if not output:
        raise ValueError("selected cell returned no per-token diagnostics")
    return output


def _mean_by_layer(diag: Mapping[int, Mapping[int, Mapping]], index: int, name: str) -> float:
    values = [
        float(rows[index][name])
        for rows in diag.values()
        if index in rows
    ]
    if not values:
        raise ValueError(f"token {index} missing {name} across selected layers")
    return statistics.fmean(values)


def _mean_feature_distance(left, right, index: int, name: str, torch_module) -> float:
    values = []
    for layer, left_rows in left.items():
        if index not in left_rows or layer not in right or index not in right[layer]:
            continue
        left_value = torch_module.as_tensor(left_rows[index][name]).float()
        right_value = torch_module.as_tensor(right[layer][index][name]).float()
        values.append(float((left_value - right_value).norm().item()))
    if not values:
        raise ValueError(f"token {index} missing aligned {name}")
    return statistics.fmean(values)


def _token_grid(context: Mapping, cell) -> tuple[int, int, int]:
    shape = tuple(cell.noisy_latent.shape)
    if len(shape) != 5:
        raise ValueError("S2 requires a five-dimensional noisy latent")
    patch = context["engine"].pipe.dit.patch_size
    frames = int(shape[2]) // max(int(patch[0]), 1)
    height = int(shape[3]) // max(int(patch[1]), 1)
    width = int(shape[4]) // max(int(patch[2]), 1)
    if min(frames, height, width) <= 0:
        raise ValueError("invalid token grid")
    return frames, height, width


def _semantic_capture(context: Mapping, cell, *, prompt: str, role_ids: Sequence[str]) -> dict:
    bundle = context["payloads"]["correct"]
    noisy = cell.noisy_latent.to(context["engine"].device)
    result = context["engine"].generate_chunk(
        prompt=prompt,
        memory_tokens=bundle["memory_tokens"],
        memory_bank_tokens=bundle["memory_bank_tokens"],
        memory_bank_percents=bundle["memory_bank_percents"],
        memory_bank_token_meta=bundle["memory_bank_token_meta"],
        memory_token_lengths_per_character=bundle["memory_token_lengths_per_character"],
        ref_images=context["reference_frames"],
        random_ref_frame=context["fixed_reference"],
        seed=context["target_seed"],
        online_memory_chars=[],
        online_memory_bank_percents=[],
        teacher_forced_probe={
            "timestep_index": int(cell.timestep_index),
            "noisy_latents": noisy,
            "force_memory_path": True,
            "semantic_role_ids": list(role_ids),
            "semantic_capture_only": True,
        },
    )
    maps = result.get("semantic_attention_maps", {})
    if not isinstance(maps, Mapping) or not maps:
        raise ValueError("semantic capture returned no maps")
    return dict(result)


def _s2_model_forward(
    context: Mapping,
    cell,
    *,
    arm: str,
    query_indices: Sequence[int],
    context_zero_indices: Sequence[int] | None = None,
    capture: bool = False,
) -> dict:
    from .qstar_probe import _mse, validate_measured_injection

    bundle = context["payloads"][arm]
    noisy = cell.noisy_latent.to(context["engine"].device)
    result = context["engine"].generate_chunk(
        prompt=context["prompt"],
        memory_tokens=bundle["memory_tokens"],
        memory_bank_tokens=bundle["memory_bank_tokens"],
        memory_bank_percents=bundle["memory_bank_percents"],
        memory_bank_token_meta=bundle["memory_bank_token_meta"],
        memory_token_lengths_per_character=bundle["memory_token_lengths_per_character"],
        ref_images=context["reference_frames"],
        random_ref_frame=context["fixed_reference"],
        seed=context["target_seed"],
        online_memory_chars=[],
        online_memory_bank_percents=[],
        teacher_forced_probe={
            "timestep_index": int(cell.timestep_index),
            "noisy_latents": noisy,
            "force_memory_path": True,
            "query_indices_by_role": {
                str(context["event"]["character_name"]): [int(index) for index in query_indices]
            },
            "context_zero_indices": list(context_zero_indices or []),
            "capture_sparse_token_diagnostics": bool(capture),
        },
    )
    prediction = result.get("prediction_cond")
    if prediction is None:
        raise ValueError(f"{arm}: S2 forward returned no conditional prediction")
    validate_measured_injection(arm, bool(bundle["target_read_hit"]), result)
    return {
        "arm": arm,
        "loss": _mse(prediction, cell.flow_target),
        "query_indices": [int(index) for index in query_indices],
        "context_zero_indices": [int(index) for index in (context_zero_indices or [])],
        "result": result,
    }


def _text_positions(engine, prompt: str, phrases: Sequence[str]) -> list[int]:
    configs, roles = engine._prepare_character_semantic_probe_configs(
        prompt=prompt, role_ids=[str(phrase) for phrase in phrases]
    )
    found = {
        str(role): [int(index) for index in config.get("all_token_indices", [])]
        for role, config in zip(roles, configs)
    }
    missing = [phrase for phrase in phrases if str(phrase) not in found]
    if missing:
        raise ValueError(f"text positions missing for {missing}")
    return sorted({index for values in found.values() for index in values})


def _public_intervention(record: Mapping, name: str) -> dict:
    return {
        "name": str(name),
        "arm": str(record["arm"]),
        "loss": float(record["loss"]),
        "query_indices": list(record["query_indices"]),
        "context_zero_indices": list(record["context_zero_indices"]),
    }


def run_s2(
    context: Mapping,
    selected_cells: Mapping,
    screening_records: Sequence[Mapping],
    args,
    *,
    repeat_margin: float | None = None,
) -> dict:
    """Run bounded proposal, group knockout, and equal-budget confirmation."""
    torch_module = context.get("torch")
    if torch_module is None:
        raise ValueError("production S2 context must expose torch")
    primary_spec = selected_cells.get("primary")
    validation_spec = selected_cells.get("validation")
    if not isinstance(primary_spec, Mapping):
        raise ValueError("S2 requires a selected primary cell")
    primary = next(
        cell
        for cell in context["cells"]
        if int(cell.timestep_index) == int(primary_spec["timestep_index"])
    )
    primary_layers = tuple(int(layer) for layer in primary_spec["layer_group"])
    engine = context["engine"]
    character = str(context["event"]["character_name"])
    manifest = semantic_group_manifest(context["story"], context["event"])
    source_description = str(context["story"]["characters"][character])
    diagnostic_prompt = f"{source_description} {context['prompt']}"
    frames, height, width = _token_grid(context, primary)
    token_count = frames * height * width
    all_indices = list(range(token_count))
    original_layers = list(engine.sparse_role_memory_injection_layers)
    measured: list[dict] = []
    try:
        engine.sparse_role_memory_injection_layers = list(primary_layers)
        name_capture = _semantic_capture(
            context, primary, prompt=context["prompt"], role_ids=manifest["identity_name"]
        )
        semantic_ids = (
            manifest["stable_attributes"]
            + manifest["action_core"]
            + manifest["action_context"]
            + manifest["scene"]
        )
        diagnostic_capture = _semantic_capture(
            context, primary, prompt=diagnostic_prompt, role_ids=semantic_ids
        )
        original_query = _query_indices(name_capture.get("query_feature_payload"), character)
        if len(original_query) < 4:
            raise ValueError("original identity query mask contains fewer than four tokens")

        all_correct = _s2_model_forward(
            context, primary, arm="correct", query_indices=all_indices, capture=True
        )
        all_wrong = _s2_model_forward(
            context, primary, arm="wrong", query_indices=all_indices, capture=True
        )
        measured.extend((
            _public_intervention(all_correct, "all_token_diagnostic_correct"),
            _public_intervention(all_wrong, "all_token_diagnostic_wrong"),
        ))
        all_correct_diag = _token_diagnostics(all_correct["result"], primary_layers)
        all_wrong_diag = _token_diagnostics(all_wrong["result"], primary_layers)
        raw_margins = [
            _mean_by_layer(all_correct_diag, index, "raw_cosine_max")
            - _mean_by_layer(all_wrong_diag, index, "raw_cosine_max")
            for index in all_indices
        ]
        read_margins = [
            _mean_by_layer(all_correct_diag, index, "read_logsumexp")
            - _mean_by_layer(all_wrong_diag, index, "read_logsumexp")
            for index in all_indices
        ]
        persistence = [
            0.5 * (raw_rank + read_rank)
            for raw_rank, read_rank in zip(
                percentile_rank(raw_margins), percentile_rank(read_margins)
            )
        ]
        persistence_top_count = max(1, math.ceil(0.10 * token_count))
        persistence_top = sorted(
            sorted(all_indices, key=lambda index: (-persistence[index], index))[
                :persistence_top_count
            ]
        )
        universe = sorted(set(original_query).union(persistence_top))

        full_correct = _s2_model_forward(
            context, primary, arm="correct", query_indices=universe, capture=True
        )
        full_wrong = _s2_model_forward(
            context, primary, arm="wrong", query_indices=universe, capture=True
        )
        measured.extend((
            _public_intervention(full_correct, "full_correct"),
            _public_intervention(full_wrong, "full_wrong"),
        ))
        full_diag = _token_diagnostics(full_correct["result"], primary_layers)
        wrong_diag = _token_diagnostics(full_wrong["result"], primary_layers)

        action_positions = _text_positions(engine, context["prompt"], manifest["action_core"])
        scene_positions = _text_positions(engine, context["prompt"], manifest["scene"])
        tokenizer = engine.pipe.prompter.tokenizer.tokenizer
        encoded_prompt = tokenizer.encode(context["prompt"], add_special_tokens=True)
        excluded = set(action_positions).union(scene_positions)
        random_pool = [
            index for index in range(1, max(1, len(encoded_prompt) - 1)) if index not in excluded
        ]
        if len(random_pool) < len(action_positions):
            raise ValueError("prompt has too few text positions for the random action control")
        random_positions = sorted(
            random.Random(int(getattr(args, "noise_seed", 0)) + 911).sample(
                random_pool, len(action_positions)
            )
        )
        text_drop_records = {}
        for name, positions in (
            ("drop_action_text", action_positions),
            ("drop_scene_text", scene_positions),
            ("drop_random_text", random_positions),
        ):
            record = _s2_model_forward(
                context,
                primary,
                arm="correct",
                query_indices=universe,
                context_zero_indices=positions,
                capture=True,
            )
            text_drop_records[name] = record
            measured.append(_public_intervention(record, name))
        action_diag = _token_diagnostics(
            text_drop_records["drop_action_text"]["result"], primary_layers
        )
        scene_diag = _token_diagnostics(
            text_drop_records["drop_scene_text"]["result"], primary_layers
        )
        random_diag = _token_diagnostics(
            text_drop_records["drop_random_text"]["result"], primary_layers
        )

        name_vector = _semantic_vector(
            name_capture["semantic_attention_maps"], character, primary_layers, token_count
        )
        name_token_count = len(_text_positions(engine, context["prompt"], [character]))
        name_vector = [value / max(name_token_count, 1) for value in name_vector]
        attribute_vectors = [
            [
                value / max(len(_text_positions(engine, diagnostic_prompt, [phrase])), 1)
                for value in _semantic_vector(
                    diagnostic_capture["semantic_attention_maps"], phrase, primary_layers, token_count
                )
            ]
            for phrase in manifest["stable_attributes"]
        ]
        action_vectors = [
            [
                value / max(len(_text_positions(engine, diagnostic_prompt, [phrase])), 1)
                for value in _semantic_vector(
                    diagnostic_capture["semantic_attention_maps"], phrase, primary_layers, token_count
                )
            ]
            for phrase in manifest["action_core"]
        ]
        scene_vectors = [
            [
                value / max(len(_text_positions(engine, diagnostic_prompt, [phrase])), 1)
                for value in _semantic_vector(
                    diagnostic_capture["semantic_attention_maps"], phrase, primary_layers, token_count
                )
            ]
            for phrase in manifest["scene"]
        ]
        raw_rows = []
        for index in universe:
            host_norm = _mean_by_layer(full_diag, index, "host_norm")
            raw_rows.append({
                "flat_idx": index,
                "name_raw": max(0.0, name_vector[index]),
                "attribute_raw": statistics.median(
                    vector[index] for vector in attribute_vectors
                ),
                "persistence_raw_margin": raw_margins[index],
                "persistence_read_margin": read_margins[index],
                "action_attention_raw": max(vector[index] for vector in action_vectors),
                "action_hidden_raw": _mean_feature_distance(
                    full_diag, action_diag, index, "host_features", torch_module
                ) / max(host_norm, 1e-12),
                "scene_hidden_raw": _mean_feature_distance(
                    full_diag, scene_diag, index, "host_features", torch_module
                ) / max(host_norm, 1e-12),
                "random_hidden_raw": _mean_feature_distance(
                    full_diag, random_diag, index, "host_features", torch_module
                ) / max(host_norm, 1e-12),
                "scene_raw": max(vector[index] for vector in scene_vectors),
                "content_delta": _mean_feature_distance(
                    full_diag, wrong_diag, index, "effective_delta_features", torch_module
                ) / max(host_norm, 1e-12),
            })
        token_rows = score_token_channels(raw_rows)
        score_by_index = {int(row["flat_idx"]): float(row["s_pre"]) for row in token_rows}
        has_validation = isinstance(validation_spec, Mapping)
        max_groups = min(
            int(getattr(args, "max_groups", 8)), 5 if has_validation else 8
        )
        groups = build_candidate_groups(
            universe,
            height=height,
            width=width,
            max_groups=max_groups,
            min_group_size=4,
        )
        no_memory_loss = float(next(
            row["loss"]
            for row in screening_records
            if row["arm"] == "no_memory"
            and int(row["timestep_index"]) == int(primary.timestep_index)
        ))
        full_benefit = no_memory_loss - float(full_correct["loss"])
        if full_benefit <= max(float(getattr(args, "benefit_margin", 0.0)), 0.0):
            return {
                "measured_forward_count": len(measured),
                "semantic_manifest": manifest,
                "candidate_universe": universe,
                "token_rows": [],
                "groups": groups,
                "interventions": measured,
                "gate": {"status": "BLOCK", "reasons": ["expanded full-memory benefit is non-positive"]},
            }
        group_scores = {}
        for group in groups:
            remaining = sorted(set(universe) - set(group["indices"]))
            record = _s2_model_forward(
                context, primary, arm="correct", query_indices=remaining, capture=False
            )
            name = f"drop_group:{group['group_id']}"
            measured.append(_public_intervention(record, name))
            group_scores[str(group["group_id"])] = (
                float(record["loss"]) - float(full_correct["loss"])
            ) / full_benefit
        group_by_index = {
            int(index): str(group["group_id"])
            for group in groups
            for index in group["indices"]
        }
        for row in token_rows:
            group_id = group_by_index[int(row["flat_idx"])]
            row["group_id"] = group_id
            row["group_causal_score"] = float(group_scores[group_id])

        masks = build_intervention_masks(
            original_query,
            universe,
            score_by_index,
            budget_fraction=float(getattr(args, "identity_budget", 0.25)),
            seed=int(getattr(args, "noise_seed", 0)),
            height=height,
            width=width,
        )
        equal_records = {}
        equal_specs = (
            ("identity_only", "correct", "identity_top"),
            ("random_only", "correct", "random"),
            ("low_only", "correct", "low_score"),
            ("drop_identity", "correct", "drop_identity"),
            ("drop_random", "correct", "drop_random"),
            ("drop_low", "correct", "drop_low"),
            ("wrong_identity", "wrong", "wrong_identity"),
        )
        for name, arm, mask_name in equal_specs:
            record = _s2_model_forward(
                context, primary, arm=arm, query_indices=masks[mask_name], capture=False
            )
            equal_records[name] = record
            measured.append(_public_intervention(record, name))

        validation_direction = False
        validation = {"status": "PENDING", "reasons": ["no validation cell"]}
        if has_validation:
            validation_cell = next(
                cell
                for cell in context["cells"]
                if int(cell.timestep_index) == int(validation_spec["timestep_index"])
            )
            engine.sparse_role_memory_injection_layers = [
                int(layer) for layer in validation_spec["layer_group"]
            ]
            validation_records = {}
            for name, arm, indices in (
                ("validation_full", "correct", universe),
                ("validation_identity", "correct", masks["identity_top"]),
                ("validation_drop_identity", "correct", masks["drop_identity"]),
                ("validation_wrong_identity", "wrong", masks["wrong_identity"]),
            ):
                record = _s2_model_forward(
                    context, validation_cell, arm=arm, query_indices=indices, capture=False
                )
                validation_records[name] = record
                measured.append(_public_intervention(record, name))
            validation_no = float(next(
                row["loss"]
                for row in screening_records
                if row["arm"] == "no_memory"
                and int(row["timestep_index"]) == int(validation_cell.timestep_index)
            ))
            validation_direction = (
                validation_no > float(validation_records["validation_full"]["loss"])
                and float(validation_records["validation_wrong_identity"]["loss"])
                > float(validation_records["validation_identity"]["loss"])
                and float(validation_records["validation_drop_identity"]["loss"])
                > float(validation_records["validation_full"]["loss"])
            )
            validation = {
                "status": "PASS" if validation_direction else "BLOCK",
                "losses": {name: float(record["loss"]) for name, record in validation_records.items()},
                "no_memory": validation_no,
            }

        losses = {
            "no_memory": no_memory_loss,
            "full_correct": float(full_correct["loss"]),
            "identity_only": float(equal_records["identity_only"]["loss"]),
            "drop_identity": float(equal_records["drop_identity"]["loss"]),
            "drop_random": float(equal_records["drop_random"]["loss"]),
            "drop_low": float(equal_records["drop_low"]["loss"]),
            "wrong_identity": float(equal_records["wrong_identity"]["loss"]),
        }
        effective_repeat_margin = (
            float(repeat_margin)
            if repeat_margin is not None
            else max(
                float(getattr(args, "repeat_loss_tolerance", 0.0)),
                float(getattr(args, "benefit_margin", 0.0)),
            )
        )
        final = finalize_s2(
            token_rows,
            losses,
            identity_fraction=len(masks["identity_top"]) / len(original_query),
            repeat_margin=effective_repeat_margin,
            benefit_margin=float(getattr(args, "benefit_margin", 0.0)),
            validation_direction=validation_direction,
        )
        causal_ranks = percentile_rank([
            max(0.0, float(row["group_causal_score"])) for row in final["token_rows"]
        ])
        for row, causal_rank in zip(final["token_rows"], causal_ranks):
            row["s_causal"] = causal_rank
            row["s_final"] = math.sqrt(float(row["s_pre"]) * causal_rank)
        if len(measured) > s2_forward_budget(
            max_groups=int(getattr(args, "max_groups", 8)), has_validation=has_validation
        ):
            raise RuntimeError("S2 measured-forward budget exceeded")
        return {
            **final,
            "measured_forward_count": len(measured),
            "semantic_capture_count": 2,
            "semantic_manifest": manifest,
            "diagnostic_prompt": diagnostic_prompt,
            "original_query": original_query,
            "candidate_universe": universe,
            "persistence_top": persistence_top,
            "groups": [
                {**group, "group_causal_score": group_scores[str(group["group_id"])]}
                for group in groups
            ],
            "masks": masks,
            "losses": losses,
            "validation": validation,
            "interventions": measured,
        }
    finally:
        engine.sparse_role_memory_injection_layers = original_layers


def run_probe(args, *, context_loader=None) -> dict:
    """Run S0/S1 once; later stages extend this result after a content PASS."""
    from .qstar_probe import _influence, _load_probe_context

    loader = context_loader or _load_probe_context
    context = loader(args, include_native=False)
    torch_module = context.get("torch")
    inference_context = (
        torch_module.inference_mode()
        if torch_module is not None and hasattr(torch_module, "inference_mode")
        else nullcontext()
    )
    engine = context["engine"]
    original_layers = list(engine.sparse_role_memory_injection_layers)
    records = []
    try:
        with inference_context:
            for run in build_screening_schedule():
                engine.sparse_role_memory_injection_layers = list(run.layer_group)
                records.append(_run_screening_forward(context, run))
    finally:
        engine.sparse_role_memory_injection_layers = original_layers

    requested_fallback = bool(getattr(args, "allow_attention_fallback", False))
    backends = sorted({str(row["attention_implementation"]) for row in records})
    if (backends != ["flash_attention_2"]) and not requested_fallback:
        raise ValueError(f"FlashAttention 2 required, got {backends}")
    lookup = {
        (row["timestep_index"], tuple(row["layer_group"]), row["arm"]): row
        for row in records
    }
    correct = lookup[(25, ALL_LAYERS, "correct")]
    repeat = lookup[(25, ALL_LAYERS, "correct_repeat")]
    repeat_loss_floor = abs(float(correct["loss"]) - float(repeat["loss"]))
    repeat_influence = _influence(correct["_prediction"], repeat["_prediction"])
    if repeat_loss_floor > float(args.repeat_loss_tolerance):
        raise ValueError("correct_repeat loss exceeds frozen tolerance")
    if repeat_influence > float(args.repeat_influence_tolerance):
        raise ValueError("correct_repeat prediction exceeds frozen tolerance")
    cell_records = _screening_cells(records)
    repeat_margin = max(
        float(args.repeat_loss_tolerance),
        float(args.benefit_margin),
        repeat_loss_floor,
    )
    selected = select_cells(
        cell_records,
        repeat_margin=repeat_margin,
        influence_floor=float(args.influence_floor),
    )
    public_records = [{key: value for key, value in row.items() if key != "_prediction"} for row in records]
    content_pass = selected["primary"] is not None
    s2_result = None
    if content_pass:
        s2_context = (
            torch_module.inference_mode()
            if torch_module is not None and hasattr(torch_module, "inference_mode")
            else nullcontext()
        )
        with s2_context:
            s2_result = run_s2(
                context,
                selected,
                public_records,
                args,
                repeat_margin=repeat_margin,
            )
    report = {
        "schema_version": 1,
        "forward_count": len(records) + int(
            s2_result.get("measured_forward_count", 0) if s2_result else 0
        ),
        "forward_budget": 50,
        "screening_records": public_records,
        "screening_cells": cell_records,
        "selected_cells": selected,
        "repeat_loss_floor": repeat_loss_floor,
        "repeat_influence_floor": repeat_influence,
        "gates": {
            "runtime_contract": {"status": "PASS", "reasons": []},
            "content_causality": {
                "status": "PASS" if content_pass else "BLOCK",
                "reasons": [] if content_pass else ["no positive content-specific cell"],
            },
            "identity_set": (
                dict(s2_result["gate"])
                if s2_result is not None
                else {"status": "PENDING", "reasons": ["S2 not run"]}
            ),
        },
        "runtime": {"attention_implementation": backends[0] if len(backends) == 1 else backends},
    }
    if s2_result is not None:
        report["s2"] = s2_result
    return report
