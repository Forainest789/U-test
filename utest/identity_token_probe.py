"""Fast teacher-forced identity-token causal probe for SlotMem."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Sequence


ALL_LAYERS = tuple(range(16))
DEFAULT_TIMESTEPS = (0, 25, 49)
DEFAULT_GROUPS = (
    tuple(range(0, 5)),
    tuple(range(5, 11)),
    tuple(range(11, 16)),
)


@dataclass(frozen=True)
class ScreeningRun:
    stage: str
    timestep_index: int
    layer_group: tuple[int, ...]
    arm: str


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
    return {
        "schema_version": 1,
        "forward_count": len(records),
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
            "identity_set": {"status": "PENDING", "reasons": ["S2 not run"]},
        },
        "runtime": {"attention_implementation": backends[0] if len(backends) == 1 else backends},
    }
