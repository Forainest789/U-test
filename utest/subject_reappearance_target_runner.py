"""Run the four subject-reappearance target arms with one loaded engine."""

from __future__ import annotations

import time
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence


TARGET_ARMS = ("full_correct", "no_memory", "zero_path", "wrong_subject")


def build_target_arm_bundles(
    infer_slotmem,
    engine,
    *,
    state: Mapping,
    target_chunk: Mapping,
    event: Mapping,
    seed: int,
    mask_manifest: Mapping,
    runtime_contract: Mapping,
    event_file_sha256: str,
    manifest_file_sha256: str,
    report_root: Path,
    audit_installer=None,
    donor_path: Path | None = None,
    donor_entry: Mapping | None = None,
    donor_artifact: Mapping | None = None,
    donor_provenance: Mapping | None = None,
) -> dict[str, dict]:
    """Reload one frozen prefix and materialize the exact target payload per arm."""
    import torch

    if audit_installer is None:
        from .subject_subspace_audit import install_subject_subspace

        audit_installer = install_subject_subspace
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

    result = {}
    for arm in TARGET_ARMS:
        manager = infer_slotmem.RoleWiseSlotMemoryBank()
        manager.memory_bank = state.get("memory_bank", {}) or {}
        manager.memory_meta_bank = state.get("memory_meta_bank", {}) or {}
        manager.first_appearance = state.get("first_appearance", {}) or {}
        manager.current_chunk_idx = int(event["target_chunk_idx"])
        report_path = Path(report_root) / "arms" / arm / "audit.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        donor_kwargs = (
            {
                "donor_path": donor_path,
                "donor_entry": donor_entry,
                "donor_artifact": donor_artifact,
                "donor_provenance": donor_provenance,
            }
            if arm == "wrong_subject"
            else {}
        )
        flush_audit = audit_installer(
            arm=arm,
            seed=int(seed),
            manifest=mask_manifest,
            event=event,
            report_path=report_path,
            runtime_contract=runtime_contract,
            event_file_sha256=event_file_sha256,
            manifest_file_sha256=manifest_file_sha256,
            **donor_kwargs,
        )
        layer_tokens = defaultdict(lambda: defaultdict(list))
        layer_meta = defaultdict(lambda: defaultdict(list))
        shared_tokens = defaultdict(list)
        shared_meta = defaultdict(list)
        try:
            for char in chars:
                for bank_idx in bank_indices:
                    payload = manager.get_memory_payload_for_read(char, bank_idx)
                    if payload is None or payload.get("tokens") is None:
                        continue
                    tokens = payload["tokens"]
                    meta = payload.get("token_meta", [])
                    if infer_slotmem._is_layerwise_token_payload(tokens):
                        for layer, tensor in infer_slotmem._iter_layerwise_items(tokens):
                            if not isinstance(tensor, torch.Tensor):
                                continue
                            layer_key = str(layer)
                            layer_tokens[layer_key][str(bank_idx)].append(tensor)
                            layer_value = infer_slotmem._select_layerwise_value(
                                meta, layer, default=[]
                            )
                            if isinstance(layer_value, list):
                                layer_meta[layer_key][str(bank_idx)].extend(layer_value)
                    elif isinstance(tokens, torch.Tensor):
                        shared_tokens[str(bank_idx)].append(tokens)
                        if isinstance(meta, list):
                            shared_meta[str(bank_idx)].extend(meta)
        finally:
            flush_audit()

        if layer_tokens:
            banks = infer_slotmem._make_layerwise_container(
                {
                    layer: {
                        bank: torch.cat(values, dim=0)
                        for bank, values in bank_map.items()
                        if values
                    }
                    for layer, bank_map in layer_tokens.items()
                }
            )
            metas = infer_slotmem._make_layerwise_container(
                {
                    layer: {
                        bank: list(layer_meta[layer].get(bank, []))
                        for bank in bank_map
                    }
                    for layer, bank_map in layer_tokens.items()
                }
            )
            memory_tokens = next(
                (
                    tensor
                    for _, bank_map in infer_slotmem._iter_layerwise_items(banks)
                    for tensor in bank_map.values()
                    if isinstance(tensor, torch.Tensor)
                ),
                None,
            )
            lengths = None
        else:
            banks = {
                bank: torch.cat(values, dim=0)
                for bank, values in shared_tokens.items()
                if values
            }
            metas = {bank: list(shared_meta.get(bank, [])) for bank in banks}
            memory_tokens = banks.get("0")
            first_bank = sorted(shared_tokens)[0] if shared_tokens else None
            lengths = (
                [tensor.shape[0] for tensor in shared_tokens[first_bank]]
                if first_bank is not None
                else None
            )
        result[arm] = {
            "memory_tokens": memory_tokens,
            "memory_bank_tokens": banks or None,
            "memory_bank_percents": list(percents),
            "memory_bank_token_meta": metas or None,
            "memory_token_lengths_per_character": lengths,
        }
    return result


def _reset_engine_diagnostics(engine) -> None:
    engine.runtime_chunk_warnings = []
    engine.runtime_role_states = []
    engine._last_sparse_role_memory_stats = {}
    engine._last_sparse_role_memory_stats_by_layer = {}
    engine._last_jigsaw_stage2_writer_stats = {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_snapshot(path: Path, expected_sha256: str) -> None:
    if _sha256_file(path) != expected_sha256:
        raise ValueError("snapshot_sha256_mismatch")


def _write_json_exclusive(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run_target_arm_loop(
    engine,
    bundles: Mapping[str, Mapping],
    *,
    prompt: str,
    seed: int,
    reference_frames: Sequence,
    fixed_reference,
    output_root: Path,
    target_chunk_idx: int,
    save_video_fn: Callable,
    after_arm: Callable[[str, Mapping], None] | None = None,
) -> dict[str, dict]:
    """Generate exactly one target video for each frozen arm, sequentially."""
    if tuple(bundles) != TARGET_ARMS:
        raise ValueError(f"target arms must be exactly {TARGET_ARMS}")
    output_root = Path(output_root)
    records = {}
    for arm in TARGET_ARMS:
        _reset_engine_diagnostics(engine)
        kwargs = {
            "memory_tokens": None,
            "memory_bank_tokens": None,
            "memory_bank_percents": [],
            "memory_bank_token_meta": None,
            "memory_token_lengths_per_character": None,
            **dict(bundles[arm]),
        }
        started = time.perf_counter()
        frames, _, _ = engine.generate_chunk(
            prompt=prompt,
            **kwargs,
            ref_images=list(reference_frames),
            random_ref_frame=fixed_reference,
            seed=int(seed),
            online_memory_chars=[],
            online_memory_bank_percents=[],
        )
        elapsed = time.perf_counter() - started
        arm_dir = output_root / "arms" / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        video = arm_dir / f"chunk_{int(target_chunk_idx):03d}.mp4"
        save_video_fn(frames, str(video), fps=16)
        records[arm] = {
            "arm": arm,
            "target_chunk_idx": int(target_chunk_idx),
            "frame_count": len(frames),
            "wall_time_s": float(elapsed),
            "video_path": str(video.resolve()),
            "last_sparse_role_memory_stats": dict(
                getattr(engine, "_last_sparse_role_memory_stats", {})
            ),
            "last_sparse_role_memory_stats_by_layer": dict(
                getattr(engine, "_last_sparse_role_memory_stats_by_layer", {})
            ),
            "last_jigsaw_stage2_writer_stats": dict(
                getattr(engine, "_last_jigsaw_stage2_writer_stats", {})
            ),
        }
        if video.is_file():
            records[arm]["video_sha256"] = _sha256_file(video)
        if after_arm is not None:
            after_arm(arm, records[arm])
    return records


def run_target_preflight(
    context: Mapping,
    *,
    engine_factory: Callable | None = None,
    save_video_fn: Callable | None = None,
    snapshot_verifier: Callable[[Path, str], None] = _verify_snapshot,
) -> dict:
    """Load SlotMem once and generate exactly the four frozen target arms."""
    from .event_harness import _set_option

    infer_slotmem = context["infer_slotmem"]
    if save_video_fn is None:
        from diffsynth.utils.data import save_video as save_video_fn

    snapshot = Path(context["snapshot_path"])
    snapshot_sha256 = str(context["snapshot_sha256"])
    snapshot_verifier(snapshot, snapshot_sha256)
    argv = _set_option(context["inference_argv"], "--offload_models", None)
    argv = _set_option(argv, "--no-offload_models", None)
    argv.append("--no-offload_models")
    runtime_args = infer_slotmem.parse_args(argv)
    if bool(getattr(runtime_args, "offload_models", True)):
        raise ValueError("target-preflight requires offload_models=false")
    factory = engine_factory or infer_slotmem.SlotMemInferenceEngine
    engine = factory(runtime_args)
    output_root = Path(context["output_root"])
    bundles = build_target_arm_bundles(
        infer_slotmem,
        engine,
        state=context["state"],
        target_chunk=context["target_chunk"],
        event=context["event"],
        seed=int(context["target_seed"]),
        mask_manifest=context["mask_manifest"],
        runtime_contract=context["runtime_contract"],
        event_file_sha256=str(context["event_file_sha256"]),
        manifest_file_sha256=str(context["manifest_file_sha256"]),
        report_root=output_root,
        audit_installer=context.get("audit_installer"),
        donor_path=context.get("donor_path"),
        donor_entry=context.get("donor_entry"),
        donor_artifact=context.get("donor_artifact"),
        donor_provenance=context.get("donor_provenance"),
    )

    target_idx = int(context["event"]["target_chunk_idx"])

    def finish_arm(arm: str, record: Mapping) -> None:
        arm_dir = output_root / "arms" / arm
        _write_json_exclusive(
            arm_dir / f"chunk_{target_idx:03d}.metadata.json",
            {
                "chunk_idx": target_idx,
                "reference_conditioning": dict(context["reference_conditioning"]),
            },
        )
        _write_json_exclusive(
            arm_dir / "efficiency.json",
            {"chunks": [{**dict(record), "chunk_idx": target_idx}]},
        )
        snapshot_verifier(snapshot, snapshot_sha256)

    records = run_target_arm_loop(
        engine,
        bundles,
        prompt=str(context["target_chunk"].get("content") or context["target_chunk"].get("caption") or ""),
        seed=int(context["target_seed"]),
        reference_frames=context["reference_frames"],
        fixed_reference=context["fixed_reference"],
        output_root=output_root,
        target_chunk_idx=target_idx,
        save_video_fn=save_video_fn,
        after_arm=finish_arm,
    )
    return {
        "execution_mode": "single_process_target_only",
        "engine_initialization_count": 1,
        "target_chunk_idx": target_idx,
        "target_plus_one_generated": False,
        "arm_order": list(TARGET_ARMS),
        "snapshot_sha256": snapshot_sha256,
        "arms": records,
    }
