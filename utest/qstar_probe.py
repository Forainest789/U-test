"""Paired seven-run teacher-forced probe orchestration for SlotMem Q*."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import gc
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


def prepare_flow_cell(scheduler, clean_target, noise, timestep_index: int):
    """Prepare one immutable flow-matching cell without advancing the sampler."""
    timestep_index = int(timestep_index)
    if timestep_index < 0 or timestep_index >= len(scheduler.timesteps):
        raise IndexError("timestep index outside scheduler")
    timestep = scheduler.timesteps[timestep_index]
    call_timestep = timestep.to(device=clean_target.device).reshape(1) if hasattr(timestep, "to") else timestep
    noisy_latent = scheduler.add_noise(clean_target, noise, call_timestep)
    flow_target = scheduler.training_target(clean_target, noise, call_timestep)
    return noisy_latent, flow_target, call_timestep


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
                "payload_sha256": result.get("payload_sha256"),
                "payload_layers": int(result.get("payload_layers", 0) or 0),
                "payload_slots": int(result.get("payload_slots", 0) or 0),
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


def _json_default(value):
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return float(tensor.item()) if tensor.numel() == 1 else tensor.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_probe_outputs(
    output: Path,
    cells: Sequence[Mapping],
    records: Sequence[Mapping],
    *,
    metadata: Mapping | None = None,
) -> None:
    output = output.resolve()
    report = {
        "schema_version": 1,
        "status": "passed",
        "primary_estimand": "L_no_memory - L_correct",
        "native_is_diagnostic": True,
        "cells": list(cells),
        **dict(metadata or {}),
    }
    _atomic_text(
        output / "qstar_report.json",
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default),
    )
    _atomic_text(
        output / "qstar_records.jsonl",
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=_json_default) + "\n"
            for row in records
        ),
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


def _load_probe_context(args):
    import torch
    from PIL import Image
    from diffsynth.utils.data import VideoData
    import infer_slotmem

    prefix = args.prefix.resolve()
    contract = json.loads((prefix / "prefix_contract.json").read_text(encoding="utf-8"))
    snapshot = Path(contract["snapshot"]["path"])
    if not snapshot.is_file():
        raise FileNotFoundError(f"prefix snapshot not found: {snapshot}")
    if contract["snapshot"]["sha256"] != _sha256_file(snapshot):
        raise ValueError("snapshot_sha256_mismatch")
    event_path = Path(contract.get("event_json", prefix / "event.json"))
    event = json.loads(event_path.read_text(encoding="utf-8"))
    source = Path(event["source_json_path"]).resolve()
    story = json.loads(source.read_text(encoding="utf-8"))
    chunks = story.get("chunks", story) if isinstance(story, dict) else story
    target_idx = int(event["target_chunk_idx"])
    if target_idx < 0 or target_idx >= len(chunks):
        raise ValueError("target_chunk_idx outside source story")
    target_chunk = chunks[target_idx]
    prompt = str(target_chunk.get("content") or target_chunk.get("caption") or "")
    target_seed = int(contract["runtime_contract"]["target_seed"])
    actual_runtime = {
        "target_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "target_seed": target_seed,
    }
    validate_probe_runtime(contract["runtime_contract"], actual_runtime)
    future_target = args.future_target_video.resolve()
    if not future_target.is_file():
        raise FileNotFoundError(f"future target video not found: {future_target}")
    arms_root = (args.arms_root.resolve() if args.arms_root else prefix.parent / "arms").resolve()
    try:
        future_target.relative_to(arms_root)
    except ValueError:
        pass
    else:
        raise ValueError("future target video must not be inside evaluated arm outputs")

    inference_argv = list(contract["base_inference_args"])
    inference_argv = _set_runtime_option(inference_argv, "--json_path", str(source))
    inference_argv = _set_runtime_option(inference_argv, "--resume_state_path", str(snapshot))
    inference_argv = _set_runtime_option(inference_argv, "--start_chunk_idx", str(target_idx))
    inference_argv = _set_runtime_option(inference_argv, "--target_seed_override", str(target_seed))
    inference_argv = _set_runtime_option(inference_argv, "--output_path", str(args.output.resolve()))
    native_argv = _set_runtime_option(inference_argv, "--native_wan_inference", None)
    native_argv.append("--native_wan_inference")
    native_args = infer_slotmem.parse_args(native_argv)
    native_engine = infer_slotmem.SlotMemInferenceEngine(native_args)

    video = VideoData(
        video_file=str(future_target),
        height=int(native_args.height),
        width=int(native_args.width),
    )
    if len(video) != int(native_args.context_frames):
        raise ValueError(
            f"future target must contain exactly {int(native_args.context_frames)} frames, got {len(video)}"
        )
    frames = video.raw_data()
    native_engine.pipe.vae.to(native_engine.device)
    target_pixels = native_engine.pipe.preprocess_video(frames)
    clean_target = native_engine.pipe.vae.encode(
        target_pixels,
        device=native_engine.device,
        tiled=bool(native_args.tiled),
        tile_size=tuple(native_args.tile_size),
        tile_stride=tuple(native_args.tile_stride),
    ).to(device=native_engine.device)
    expected_shape = (
        1,
        16,
        (int(native_args.context_frames) - 1) // 4 + 1,
        int(native_args.height) // 8,
        int(native_args.width) // 8,
    )
    if tuple(clean_target.shape) != expected_shape:
        raise ValueError(
            f"future target latent shape {tuple(clean_target.shape)} does not match {expected_shape}"
        )
    native_engine.pipe.scheduler.set_timesteps(int(native_args.num_inference_steps))
    requested_indices = _parse_timestep_indices(args.timestep_indices)
    invalid = [index for index in requested_indices if index >= len(native_engine.pipe.scheduler.timesteps)]
    if invalid:
        raise ValueError(f"timestep indices outside scheduler: {invalid}")
    generator = torch.Generator(device=native_engine.device).manual_seed(int(args.noise_seed))
    noise = torch.randn(clean_target.shape, generator=generator, device=native_engine.device, dtype=clean_target.dtype)

    state = torch.load(snapshot, map_location="cpu", weights_only=False)
    reference_path = infer_slotmem.resolve_reference_image_path(
        native_args.ref_image_path, native_args.json_path, story
    )
    fixed_reference = Image.open(reference_path).convert("RGB") if reference_path else None
    reference_frames = infer_slotmem._decode_pil_frames_from_state(state.get("prev_frames_png", []))
    if not reference_frames and fixed_reference is not None:
        reference_frames = [fixed_reference]
    if getattr(native_engine.pipe.dit, "has_image_input", False) and not reference_frames:
        raise ValueError("prefix snapshot has no reference frames for I2V probe")

    cells = []
    native_predictions = {}
    target_hash = _sha256_file(future_target)
    for timestep_index in requested_indices:
        noisy_latent, flow_target, timestep = prepare_flow_cell(
            native_engine.pipe.scheduler, clean_target, noise, timestep_index
        )
        cell = ProbeCell(
            event_id=str(event.get("event_id", f"chunk-{target_idx}")),
            memory_id=f"{event['character_name']}|0",
            horizon=int(event.get("horizon", target_idx - int(event.get("source_chunk_idx", 0)))),
            timestep_index=int(timestep_index),
            timestep=float(timestep.detach().float().item()),
            clean_target=clean_target.detach().cpu(),
            noise=noise.detach().cpu(),
            noisy_latent=noisy_latent.detach().cpu(),
            flow_target=flow_target.detach().cpu(),
            input_hashes={
                "prefix": contract["snapshot"]["sha256"],
                "target_video": target_hash,
                "target_latent": tensor_sha256(clean_target),
                "noise": tensor_sha256(noise),
                "noisy_latent": tensor_sha256(noisy_latent),
                "flow_target": tensor_sha256(flow_target),
                "prompt": actual_runtime["target_prompt_sha256"],
            },
        )
        native_result = native_engine.generate_chunk(
            prompt=prompt,
            ref_images=reference_frames or None,
            random_ref_frame=fixed_reference,
            seed=target_seed,
            online_memory_chars=[],
            online_memory_bank_percents=[],
            teacher_forced_probe={
                "timestep_index": int(timestep_index),
                "noisy_latents": noisy_latent,
            },
        )
        native_predictions[int(timestep_index)] = native_result["prediction"].detach().cpu()
        cells.append(cell)

    del native_engine, target_pixels, clean_target, noise
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    slotmem_args = infer_slotmem.parse_args(
        _set_runtime_option(inference_argv, "--native_wan_inference", None)
    )
    engine = infer_slotmem.SlotMemInferenceEngine(slotmem_args)
    payloads = _build_arm_payloads(
        infer_slotmem,
        state,
        target_chunk,
        event,
        engine,
        arm_seed=int(contract.get("arm_seed", 0)),
        donor_path=args.donor,
        donor_manifest=args.donor_manifest,
    )
    return {
        "torch": torch,
        "engine": engine,
        "event": event,
        "contract": contract,
        "prompt": prompt,
        "target_seed": target_seed,
        "reference_frames": reference_frames or None,
        "fixed_reference": fixed_reference,
        "cells": cells,
        "native_predictions": native_predictions,
        "payloads": payloads,
        "state": state,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _set_runtime_option(argv: Sequence[str], name: str, value: str | None) -> list[str]:
    output = []
    index = 0
    while index < len(argv):
        if argv[index] == name:
            index += 2 if index + 1 < len(argv) and not str(argv[index + 1]).startswith("--") else 1
            continue
        output.append(str(argv[index]))
        index += 1
    if value is not None:
        output.extend([name, str(value)])
    return output


def _parse_timestep_indices(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not parsed or any(index < 0 for index in parsed) or len(parsed) != len(set(parsed)):
        raise ValueError("timestep indices must be unique non-negative integers")
    return parsed


def _build_arm_payloads(
    infer_slotmem,
    state,
    target_chunk,
    event,
    engine,
    *,
    arm_seed: int,
    donor_path: Path | None,
    donor_manifest: Path | None,
) -> dict:
    import torch
    from collections import defaultdict
    from .content_audit import (
        ARMS,
        _payload_sha256,
        _payload_summary,
        stable_transform_seed,
        transform_payload,
        validate_donor_manifest,
    )

    manager = infer_slotmem.RoleWiseSlotMemoryBank()
    manager.memory_bank = state.get("memory_bank", {}) or {}
    manager.memory_meta_bank = state.get("memory_meta_bank", {}) or {}
    manager.first_appearance = state.get("first_appearance", {}) or {}
    manager.current_chunk_idx = int(event["target_chunk_idx"])
    chars = list(target_chunk.get("character_list", []) or [])
    if int(engine.args.max_memory_characters) > 0:
        chars = chars[: int(engine.args.max_memory_characters)]
    target_name = str(event["character_name"]).casefold()
    if target_name not in {str(char).casefold() for char in chars}:
        raise ValueError("target character is outside the SlotMem read window")
    if engine._use_legacy_multi_memory_banks():
        percents = infer_slotmem._split_csv_floats(
            engine.args.memory_bank_percents, default=[0.85, 0.60, 0.35, 0.12]
        )
        bank_indices = list(range(len(percents)))
    else:
        percents = engine._single_online_memory_bank_percents()
        bank_indices = [0]

    selected_donor = None
    if donor_path is not None or donor_manifest is not None:
        if donor_path is None or donor_manifest is None:
            raise ValueError("wrong arm requires both donor payload and donor manifest")
        manifest = json.loads(donor_manifest.read_text(encoding="utf-8"))
        entries = manifest.get("pairs", manifest) if isinstance(manifest, dict) else manifest
        if isinstance(entries, dict):
            entries = [entries]
        matches = [
            entry for entry in entries
            if str(entry.get("target_story_id")) == str(event.get("story_id"))
            and str(entry.get("target_entity_uid")) == str(event.get("entity_uid"))
        ]
        if len(matches) != 1:
            raise ValueError(f"donor manifest must select exactly one pair, found {len(matches)}")
        entry = validate_donor_manifest(matches[0], event, donor_path)
        saved = torch.load(donor_path, map_location="cpu", weights_only=False)
        donors = saved.get("payloads", {}) if saved.get("format") == "slotmem_donor_payload_v2" else saved
        selected_donor = donors[str(entry["payload_key"])]
    else:
        raise ValueError("wrong arm requires a frozen donor payload and manifest")

    result = {}
    for arm in ARMS:
        layer_tokens = defaultdict(lambda: defaultdict(list))
        layer_meta = defaultdict(lambda: defaultdict(list))
        shared_tokens = defaultdict(list)
        shared_meta = defaultdict(list)
        target_source = None
        target_returned = None
        for char in chars:
            for bank_idx in bank_indices:
                payload = manager.get_memory_payload_for_read(char, bank_idx)
                if payload is None:
                    continue
                is_target = str(char).casefold() == target_name
                transformed = payload
                if is_target:
                    target_source = payload
                    generator_for_layer = None
                    if arm == "random":
                        generator_for_layer = lambda layer, c=char, b=bank_idx: torch.Generator().manual_seed(
                            stable_transform_seed(
                                event, int(event["target_chunk_idx"]), c, int(b), arm_seed, layer
                            )
                        )
                    transformed, _ = transform_payload(
                        payload,
                        arm,
                        None,
                        selected_donor,
                        generator_for_layer=generator_for_layer,
                    )
                    target_returned = transformed
                if transformed is None:
                    continue
                tokens = transformed.get("tokens")
                meta = transformed.get("token_meta", [])
                if infer_slotmem._is_layerwise_token_payload(tokens):
                    for layer, tensor in infer_slotmem._iter_layerwise_items(tokens):
                        if isinstance(tensor, torch.Tensor):
                            layer_tokens[layer][str(bank_idx)].append(tensor)
                            layer_value = infer_slotmem._select_layerwise_value(meta, layer, default=[])
                            if isinstance(layer_value, list):
                                layer_meta[layer][str(bank_idx)].extend(layer_value)
                elif isinstance(tokens, torch.Tensor):
                    shared_tokens[str(bank_idx)].append(tokens)
                    if isinstance(meta, list):
                        shared_meta[str(bank_idx)].extend(meta)
        if target_source is None:
            raise ValueError(f"{arm}: target memory payload is absent from frozen prefix")
        if layer_tokens:
            banks = infer_slotmem._make_layerwise_container({
                layer: {
                    bank: torch.cat(values, dim=0)
                    for bank, values in bank_map.items() if values
                }
                for layer, bank_map in layer_tokens.items()
            })
            metas = infer_slotmem._make_layerwise_container({
                layer: {bank: list(layer_meta[layer].get(bank, [])) for bank in bank_map}
                for layer, bank_map in layer_tokens.items()
            })
            memory_tokens = next(
                (
                    tensor for _, bank_map in infer_slotmem._iter_layerwise_items(banks)
                    for tensor in bank_map.values() if isinstance(tensor, torch.Tensor)
                ),
                None,
            )
            lengths = None
        else:
            banks = {bank: torch.cat(values, dim=0) for bank, values in shared_tokens.items() if values}
            metas = {bank: list(shared_meta.get(bank, [])) for bank in banks}
            memory_tokens = banks.get("0")
            lengths = None
        result[arm] = {
            "memory_tokens": memory_tokens,
            "memory_bank_tokens": banks or None,
            "memory_bank_percents": percents,
            "memory_bank_token_meta": metas or None,
            "memory_token_lengths_per_character": lengths,
            "target_read_hit": target_returned is not None,
            "target_payload_sha256": _payload_sha256(target_returned),
            "target_payload_summary": _payload_summary(target_returned),
        }
    return result


def run_production_probe(args) -> int:
    from .qstar import classify_memory_regime

    context = _load_probe_context(args)
    torch = context["torch"]
    engine = context["engine"]
    reports = []
    records = []

    def predictor(run_name, cell, bundle, native):
        if native:
            return {
                "prediction": context["native_predictions"][int(cell.timestep_index)],
                "memory_read_hit": False,
                "payload_sha256": None,
                "payload_layers": 0,
                "payload_slots": 0,
                "diagnostics": {"weight_regime": "base_wan"},
            }
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
                "noisy_latents": cell.noisy_latent.to(engine.device),
            },
        )
        stats = result.get("sparse_role_memory_stats", {})
        summary = bundle["target_payload_summary"]
        return {
            "prediction": result["prediction"].detach().cpu(),
            "memory_read_hit": bool(bundle["target_read_hit"]),
            "injection_delta_norm": float(stats.get("role_head_out_norm", 0.0) or 0.0),
            "payload_sha256": bundle["target_payload_sha256"],
            "payload_layers": int(summary.get("layers", 0) or 0),
            "payload_slots": int(summary.get("slots", 0) or 0),
            "diagnostics": {
                "sparse_role_memory_stats": stats,
                "writer_stats": result.get("writer_stats", {}),
            },
        }

    for cell in context["cells"]:
        report, cell_records = evaluate_probe_cell(
            cell,
            context["payloads"],
            predictor,
            repeat_tolerance=float(args.repeat_tolerance),
            influence_floor=float(args.influence_floor),
        )
        reports.append(report)
        records.extend(cell_records)
    evidence = _prefix_writer_evidence(context["state"])
    memory_regime = classify_memory_regime(evidence)
    if args.require_dynamic_writer and memory_regime != "dynamic_writer":
        raise ValueError("dynamic writer required but frozen prefix is static_prefix")
    write_probe_outputs(
        args.output,
        reports,
        records,
        metadata={
            "memory_regime": memory_regime,
            "writer_evidence": evidence,
            "target_video": str(args.future_target_video.resolve()),
            "target_video_sha256": _sha256_file(args.future_target_video.resolve()),
            "prefix_sha256": context["contract"]["snapshot"]["sha256"],
            "model_weights_changed": False,
        },
    )
    print(f"[qstar] wrote {args.output.resolve() / 'qstar_report.json'}", flush=True)
    return 0


def _prefix_writer_evidence(state: Mapping) -> dict:
    updates = [
        update
        for row in list(state.get("efficiency_chunk_records", []) or [])
        for update in list(row.get("writer_updates", []) or [])
    ]

    def positive(value) -> bool:
        if isinstance(value, Mapping):
            residual = value.get("residual_norm", 0.0)
            try:
                if math.isfinite(float(residual)) and float(residual) > 0:
                    return True
            except (TypeError, ValueError):
                pass
            return any(positive(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(positive(item) for item in value)
        return False

    rows = list(state.get("efficiency_chunk_records", []) or [])
    return {
        "update_count": len(updates),
        "positive_residual_count": sum(1 for update in updates if positive(update.get("stats", {}))),
        "bank_hash_change_count": sum(1 for row in rows if bool(row.get("memory_bank_hash_changed", False))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--future-target-video", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arms-root", type=Path)
    parser.add_argument("--donor", type=Path)
    parser.add_argument("--donor-manifest", type=Path)
    parser.add_argument("--timestep-indices", default="0,12,25,37,49")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--repeat-tolerance", type=float, default=0.0)
    parser.add_argument("--influence-floor", type=float, default=0.0)
    parser.add_argument("--require-dynamic-writer", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    required = {
        "--prefix": args.prefix,
        "--future-target-video": args.future_target_video,
        "--output": args.output,
        "--donor": args.donor,
        "--donor-manifest": args.donor_manifest,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("production probe requires " + ", ".join(missing))
    return run_production_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
