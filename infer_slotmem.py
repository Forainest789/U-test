#!/usr/bin/env python3
import argparse
import hashlib
import io
import importlib.util
import json
import math
import os
import sys
import time
import types
from collections import defaultdict
from contextlib import suppress
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None
import torch
import torch.nn.functional as F
from diffsynth.models.wan_video_dit import modulate, rope_apply, sinusoidal_embedding_1d
from diffsynth.utils.data import save_video
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (SCRIPT_DIR,):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

_local_utils_dirs = [str(SCRIPT_DIR / "utils")]
_utils_pkg = types.ModuleType("utils")
_utils_pkg.__path__ = _local_utils_dirs
sys.modules["utils"] = _utils_pkg

import utils.image_process  # noqa: F401

from utest.media_metadata import generated_timing
from utest.prefix_contract import sha256_file
from utest.reference_scope import (
    build_reference_conditioning_audit,
    choose_random_reference,
    image_png_bytes,
    validate_reference_resume,
)
from reference_inference_runtime import (
    ReferenceInferenceRuntime as ReferenceInferenceEngine,
    AttentionMapExtractorV8,
    MemoryManager,
    merge_chunk_videos,
    pick_nearest_bank_by_percent,
    resolve_reference_image_path,
    save_chunk_memory_visualization,
    save_denoise_step_edge_frames_visualization,
    save_denoise_step_visualization,
    save_feature_mapping_visualization,
)
from attention_probe_utils import (
    MultiCharacterAttentionMapExtractor,
    find_token_index_in_prompt,
    process_attention_map_to_mask as process_attention_map_to_mask_v2,
    verify_target_text_is_single_token,
)
from train_slotmem import (
    AttentionOutputFeatureTap,
    ForwardStopAfterLayer,
    LearnableMemoryEmbeddings,
    CharacterWiseCrossAttention,
    StyleAwareMemoryProjector,
    _StopForwardAfterLayer,
    _aggregate_character_semantic_responses_cpu,
    _suppress_other_character_response_cpu,
    _build_parallel_character_probe_contexts,
    _convert_parallel_probe_responses_to_role_diffs,
    _install_train_lora_forward,
    _inject_train_lora_modules,
    _load_checkpoint_payload,
    _parse_layer_indices_csv,
    _run_parallel_character_semantic_probe,
    build_wan22_training_pipe,
    run_native_dit_forward,
)
from mem_encoder_utils import (
    MemoryEncoderBank,
    MemoryWriter,
    encode_role_tokens_to_slots as _memory_encode_role_tokens_to_slots,
    extract_prefixed_state_dict as _jigsaw_extract_prefixed_state_dict,
    memory_encoder_enabled as _memory_encoder_enabled,
    memory_writer_effective_mode as _memory_writer_effective_mode,
    parse_layer_groups as _jigsaw_parse_layer_groups,
    parse_layer_list as _jigsaw_parse_layer_list,
)

def _split_csv_floats(value, default):
    if value is None:
        return list(default)
    out = []
    for item in str(value).split(","):
        item = str(item).strip()
        if not item:
            continue
        try:
            out.append(float(item))
        except Exception:
            pass
    return out if out else list(default)


def parse_float_csv(value, default_list):
    return _split_csv_floats(value, default_list)


def _env_flag(name, default=False):
    value = os.environ.get(str(name), None)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in ("0", "false", "no", "off", "none")


def _env_float(name, default=0.0):
    value = os.environ.get(str(name), None)
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _low_vram_dual_expert_mode_enabled(mode):
    return str(mode or "").strip().lower() not in ("", "standard", "eager", "default", "off", "false", "0", "none")


def _normalize_lora_state_key(key):
    text = str(key)
    for prefix in (
        "pipe.dit.",
        "dit.",
        "low_noise_model.",
        "high_noise_model.",
        "module.",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.replace(".lora_A.default.weight", ".lora_A.weight")
    text = text.replace(".lora_B.default.weight", ".lora_B.weight")
    while ".module." in text:
        text = text.replace(".module.", ".")
    return text


def _is_lora_state_key(key):
    text = str(key)
    return (
        ".lora_A." in text
        or ".lora_B." in text
        or text.endswith(".lora_A.weight")
        or text.endswith(".lora_B.weight")
    )


class _WanUniPCSchedulerAdapter:
    def __init__(self, device="cuda", num_train_timesteps=1000, default_shift=3.0):
        self.device = torch.device(device)
        self.num_train_timesteps = int(num_train_timesteps)
        self.default_shift = float(default_shift)
        self.training = False
        self._scheduler_cls = self._load_scheduler_cls()
        self._scheduler = None
        self.timesteps = torch.empty(0, dtype=torch.float32)
        self.sigmas = torch.empty(0, dtype=torch.float32)

    @staticmethod
    def _load_scheduler_cls():
        scheduler_repo = os.environ.get("SLOTMEM_UNIPC_REPO_DIR") or os.environ.get("STORYMEM_REPO_DIR")
        if not scheduler_repo:
            raise RuntimeError("Set SLOTMEM_UNIPC_REPO_DIR to load the Wan UniPC scheduler")
        scheduler_path = Path(scheduler_repo) / "wan" / "utils" / "fm_solvers_unipc.py"
        if not scheduler_path.exists():
            raise FileNotFoundError(f"UniPC scheduler not found: {scheduler_path}")
        spec = importlib.util.spec_from_file_location("slotmem_wan_unipc_local", scheduler_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.FlowUniPCMultistepScheduler

    def set_timesteps(self, num_inference_steps, denoising_strength=1.0, shift=None, training=False, **kwargs):
        del denoising_strength, kwargs
        if shift is None:
            shift = self.default_shift
        self._scheduler = self._scheduler_cls(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        self._scheduler.set_timesteps(int(num_inference_steps), device=self.device, shift=float(shift))
        self.timesteps = self._scheduler.timesteps
        self.sigmas = self._scheduler.sigmas
        self.training = bool(training)
        return self.timesteps

    def get_timesteps(self, num_inference_steps, denoising_strength=1.0, shift=None):
        return self.set_timesteps(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            shift=shift,
            training=False,
        )

    def step(self, model_output, timestep, sample, **kwargs):
        del kwargs
        if self._scheduler is None:
            raise RuntimeError("set_timesteps must be called before step")
        return self._scheduler.step(model_output, timestep, sample, return_dict=False)[0]

    def add_noise(self, original_samples, noise, timestep):
        if self._scheduler is not None:
            return self._scheduler.add_noise(original_samples, noise, timestep)
        sigma = timestep.to(device=original_samples.device, dtype=original_samples.dtype) / float(self.num_train_timesteps)
        while sigma.ndim < original_samples.ndim:
            sigma = sigma.unsqueeze(-1)
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(self, sample, noise, timestep):
        del timestep
        return noise - sample

    def training_weight(self, timestep):
        return torch.ones_like(timestep, dtype=torch.float32)


def _json_safe(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return [_json_safe(x) for x in value.detach().cpu().flatten().tolist()]
    if np is not None and isinstance(value, np.ndarray):
        return [_json_safe(x) for x in value.tolist()]
    if np is not None and isinstance(value, (np.integer,)):
        return int(value)
    if np is not None and isinstance(value, (np.floating,)):
        return float(value)
    if np is not None and isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return int(value.numel()) * int(value.element_size())
    return 0


def _nested_tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return _tensor_bytes(value)
    if isinstance(value, dict):
        return sum(_nested_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nested_tensor_bytes(v) for v in value)
    return 0


def _summarize_memory_manager_bytes(mem_manager):
    payload = {
        "tensor_bytes": 0,
        "tensor_mb": 0.0,
        "characters": 0,
        "banks": 0,
        "tensors": 0,
        "tokens": 0,
    }
    bank = getattr(mem_manager, "memory_bank", None)
    if not isinstance(bank, dict):
        return payload
    payload["characters"] = int(len(bank))
    for _, bank_map in bank.items():
        if not isinstance(bank_map, dict):
            continue
        payload["banks"] += int(len(bank_map))
        for _, tokens in bank_map.items():
            if isinstance(tokens, torch.Tensor):
                payload["tensors"] += 1
                payload["tokens"] += int(tokens.shape[0]) if tokens.ndim >= 1 else 0
                payload["tensor_bytes"] += _tensor_bytes(tokens)
            else:
                payload["tensor_bytes"] += _nested_tensor_bytes(tokens)
    payload["tensor_mb"] = float(payload["tensor_bytes"] / (1024 ** 2))
    return payload


def _hash_nested_tensors(digest, value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().to(device="cpu")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=lambda item: str(item)):
            digest.update(str(key).encode("utf-8"))
            _hash_nested_tensors(digest, value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _hash_nested_tensors(digest, item)


def _memory_bank_sha256(mem_manager):
    digest = hashlib.sha256()
    _hash_nested_tensors(digest, getattr(mem_manager, "memory_bank", {}))
    return digest.hexdigest()


def _runtime_evidence(records, engine):
    reads = [row.get("memory_read", {}) for row in records if isinstance(row, dict)]
    nonempty_reads = sum(1 for row in reads if bool(row.get("nonempty", False)))
    hash_changes = sum(
        1 for row in records
        if isinstance(row, dict) and bool(row.get("memory_bank_hash_changed", False))
    )
    writer_updates = [
        update
        for row in records if isinstance(row, dict)
        for update in list(row.get("writer_updates", []) or [])
    ]
    def positive_residual(value):
        if isinstance(value, dict):
            if float(value.get("residual_norm", 0.0) or 0.0) > 0.0:
                return True
            return any(positive_residual(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(positive_residual(item) for item in value)
        return False
    return {
        "memory_read_attempts": len(reads),
        "nonempty_memory_reads": int(nonempty_reads),
        "memory_reads": reads,
        "writer_updates": writer_updates,
        "writer_update_count": len(writer_updates),
        "writer_positive_residual_count": sum(
            1 for update in writer_updates if positive_residual(update.get("stats", {}))
        ),
        "writer_bank_hash_changes": int(hash_changes),
        "loaded_checkpoint_domains": sorted(
            str(value) for value in getattr(engine, "loaded_checkpoint_domains", set())
        ),
    }


def _analytic_role_wise_slot_memory_bank_bytes(args, engine, num_characters):
    if not bool(getattr(engine, "jigsaw_extra_encoder_enabled", False)):
        return 0
    slots = int(getattr(engine, "jigsaw_extra_encoder_slots", getattr(args, "jigsaw_extra_encoder_slots", 0)) or 0)
    try:
        dim = int(engine.patch_dim)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("SlotMem engine.patch_dim must be initialized from the runtime DiT") from exc
    if dim <= 0:
        raise ValueError(f"SlotMem engine.patch_dim must be positive, got {dim}")
    layers = list(getattr(engine, "sparse_role_memory_injection_layers", []) or [])
    groups = list(getattr(engine, "jigsaw_extra_encoder_layer_groups", []) or [])
    group_count = max(1, len(groups))
    if str(getattr(engine, "memory_layer_binding_mode", "layerwise")).strip().lower() == "layerwise" and len(layers) > 0:
        layer_or_group_count = int(len(layers))
    else:
        layer_or_group_count = int(group_count)
    dtype_bytes = 2
    return int(max(0, int(num_characters)) * max(0, slots) * max(0, dim) * max(0, layer_or_group_count) * dtype_bytes)


def _public_memory_character_ids(mem_manager):
    bank = getattr(mem_manager, "memory_bank", None)
    if not isinstance(bank, dict):
        return []
    sep = getattr(mem_manager, "_SEP", None)
    public_ids = set()
    for raw_char_id, bank_map in bank.items():
        if not isinstance(bank_map, dict):
            continue
        has_payload = any(_nested_tensor_bytes(tokens) > 0 for tokens in bank_map.values())
        if not has_payload:
            continue
        public_char_id = str(raw_char_id)
        if sep and sep in public_char_id and hasattr(mem_manager, "_split_layer_char_id"):
            try:
                _, public_char_id = mem_manager._split_layer_char_id(public_char_id)
            except Exception:
                pass
        public_ids.add(str(public_char_id))
    return sorted(public_ids)


def _full_buffer_status(args, engine, mem_manager, chunk_idx, phase):
    memory_stats = _summarize_memory_manager_bytes(mem_manager)
    public_ids = _public_memory_character_ids(mem_manager)
    target_characters = int(getattr(args, "max_memory_characters", 0) or 0)
    if target_characters <= 0:
        chunk_roles = []
        try:
            chunk_roles = list(getattr(args, "_full_buffer_sample_characters", []) or [])
        except Exception:
            chunk_roles = []
        target_characters = len(chunk_roles)
    public_count = int(len(public_ids))
    analytic_chars = public_count
    if target_characters > 0:
        analytic_chars = min(public_count, target_characters)
    analytic_slot_bytes = _analytic_role_wise_slot_memory_bank_bytes(args, engine, analytic_chars)
    is_full = bool(target_characters > 0 and public_count >= target_characters and float(memory_stats.get("tensor_mb", 0.0) or 0.0) > 0.0)
    return {
        "phase": str(phase),
        "chunk_idx": int(chunk_idx),
        "is_full": bool(is_full),
        "target_characters": int(target_characters),
        "public_character_count": int(public_count),
        "public_character_ids": public_ids,
        "raw_character_keys": int(memory_stats.get("characters", 0) or 0),
        "tensor_mb": float(memory_stats.get("tensor_mb", 0.0) or 0.0),
        "slot_bank_size_mb": float(memory_stats.get("tensor_mb", 0.0) or 0.0),
        "slot_bank_analytic_mb": float(analytic_slot_bytes / (1024 ** 2)),
    }


def _write_efficiency_json(path, payload):
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2, sort_keys=True)


def _write_slotmem_inference_manifest(path, payload):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2, sort_keys=True)


def _load_json_or_none(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _append_efficiency_jsonl(path, payload):
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True) + "\n")


def _parse_layer_scale_map(value):
    out = {}
    if value is None:
        return out
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            continue
        layer_text, scale_text = item.split(":", 1)
        try:
            out[int(layer_text.strip())] = float(scale_text.strip())
        except Exception:
            pass
    return out


_LAYERWISE_MARKER = "__layerwise__"
_LAYERWISE_LAYERS_KEY = "layers"


def _layer_key(layer_idx):
    try:
        return str(int(layer_idx))
    except Exception:
        return str(layer_idx)


def _zero_context_positions(context, indices):
    """Clone an encoded prompt and neutralize fixed positions without relayout."""
    if not indices:
        return context
    selected = sorted({int(index) for index in indices})
    if selected[0] < 0 or selected[-1] >= int(context.shape[1]):
        raise ValueError("context_zero_indices outside encoded prompt")
    output = context.clone()
    output[:, selected] = 0
    return output


def _is_layerwise_container(value):
    return isinstance(value, dict) and bool(value.get(_LAYERWISE_MARKER, False)) and isinstance(value.get(_LAYERWISE_LAYERS_KEY, None), dict)


def _make_layerwise_container(layers):
    return {
        _LAYERWISE_MARKER: True,
        _LAYERWISE_LAYERS_KEY: {str(k): v for k, v in dict(layers).items()},
    }


def _iter_layerwise_items(value):
    if _is_layerwise_container(value):
        for layer, payload in value.get(_LAYERWISE_LAYERS_KEY, {}).items():
            yield _layer_key(layer), payload
        return
    if isinstance(value, dict):
        for layer, payload in value.items():
            if isinstance(payload, torch.Tensor) or isinstance(payload, (list, dict)):
                yield _layer_key(layer), payload


def _select_layerwise_value(value, layer_idx, default=None):
    if _is_layerwise_container(value):
        layers = value.get(_LAYERWISE_LAYERS_KEY, {})
        key = _layer_key(layer_idx)
        if key in layers:
            return layers[key]
        return default
    return value if value is not None else default


def _is_layerwise_token_payload(value):
    if _is_layerwise_container(value):
        return True
    if not isinstance(value, dict):
        return False
    return any(isinstance(v, torch.Tensor) for v in value.values())


def _summarize_token_meta(token_meta):
    if not isinstance(token_meta, list) or len(token_meta) <= 0:
        return {
            "count": 0,
            "inside_box_count": 0,
            "inside_box_ratio": None,
            "tau_local_mean": None,
            "tau_local_p50": None,
            "tau_local_p90": None,
            "char_counts": {},
            "source_chunk_counts": {},
        }
    taus = []
    inside = 0
    char_counts = defaultdict(int)
    source_chunk_counts = defaultdict(int)
    for item in token_meta:
        if not isinstance(item, dict):
            continue
        if bool(item.get("inside_box", False)):
            inside += 1
        try:
            tau = float(item.get("tau_local", 0.0))
            if math.isfinite(tau):
                taus.append(tau)
        except Exception:
            pass
        rid = str(item.get("char_id", "")).strip()
        if rid:
            char_counts[rid] += 1
        source_chunk = item.get("source_chunk_idx", None)
        if source_chunk is not None:
            try:
                source_chunk_counts[str(int(source_chunk))] += 1
            except Exception:
                source_chunk_counts[str(source_chunk)] += 1
    count = int(len(token_meta))
    if len(taus) > 0:
        tau_arr = np.asarray(taus, dtype=np.float32) if np is not None else None
        tau_mean = float(tau_arr.mean()) if tau_arr is not None else float(sum(taus) / len(taus))
        tau_p50 = float(np.percentile(tau_arr, 50)) if tau_arr is not None else None
        tau_p90 = float(np.percentile(tau_arr, 90)) if tau_arr is not None else None
    else:
        tau_mean = tau_p50 = tau_p90 = None
    return {
        "count": count,
        "inside_box_count": int(inside),
        "inside_box_ratio": float(inside / max(count, 1)),
        "tau_local_mean": tau_mean,
        "tau_local_p50": tau_p50,
        "tau_local_p90": tau_p90,
        "char_counts": dict(char_counts),
        "source_chunk_counts": dict(source_chunk_counts),
    }


class RoleWiseSlotMemoryBank(MemoryManager):
    """MemoryManager wrapper that stores role-wise slot memory banks.

    The base manager is still used for normalization and CPU storage. Layer-wise
    entries are stored under hidden character keys, while public retrieval returns
    a layer-wise payload compatible with the inference forward wrapper.
    """

    _SEP = "::__layer__"

    def _layer_char_id(self, char_id, layer_idx):
        return f"{str(char_id)}{self._SEP}{_layer_key(layer_idx)}"

    def _split_layer_char_id(self, layer_char_id):
        text = str(layer_char_id)
        if self._SEP not in text:
            return None, None
        char_id, layer = text.rsplit(self._SEP, 1)
        return char_id, _layer_key(layer)

    @staticmethod
    def _is_role_wise_slot_meta(token_meta):
        if not isinstance(token_meta, list) or len(token_meta) <= 0:
            return False
        return any(isinstance(m, dict) and bool(m.get("is_jigsaw_extra_encoder_slot", False)) for m in token_meta)

    def _restore_role_wise_slot_meta(self, char_id, bank_idx, token_meta, source_chunk_idx=None):
        if not self._is_role_wise_slot_meta(token_meta):
            return
        char_id = str(char_id)
        bank_key = str(bank_idx)
        restored = []
        for item in token_meta:
            if not isinstance(item, dict):
                continue
            copied = dict(item)
            copied["char_id"] = str(copied.get("char_id", char_id))
            copied["bank_idx"] = int(bank_idx)
            if source_chunk_idx is not None:
                copied["source_chunk_idx"] = int(source_chunk_idx)
            restored.append(copied)
        if len(restored) <= 0:
            return
        if char_id not in self.memory_meta_bank or not isinstance(self.memory_meta_bank[char_id], dict):
            self.memory_meta_bank[char_id] = {}
        self.memory_meta_bank[char_id][bank_key] = restored

    def add_memory(self, char_id, tokens, bank_idx=0, token_meta=None, source_chunk_idx=None, source_video_frames=None,
                   first_appearance_only=False):
        if _is_layerwise_token_payload(tokens):
            for layer, layer_tokens in _iter_layerwise_items(tokens):
                if not isinstance(layer_tokens, torch.Tensor):
                    continue
                layer_meta = _select_layerwise_value(token_meta, layer, default=[])
                super().add_memory(
                    self._layer_char_id(char_id, layer),
                    layer_tokens,
                    bank_idx=bank_idx,
                    token_meta=layer_meta if isinstance(layer_meta, list) else [],
                    source_chunk_idx=source_chunk_idx,
                    source_video_frames=source_video_frames,
                    first_appearance_only=first_appearance_only,
                )
                self._restore_role_wise_slot_meta(
                    self._layer_char_id(char_id, layer),
                    bank_idx,
                    layer_meta if isinstance(layer_meta, list) else [],
                    source_chunk_idx=source_chunk_idx,
                )
            return
        super().add_memory(
            char_id,
            tokens,
            bank_idx=bank_idx,
            token_meta=token_meta,
            source_chunk_idx=source_chunk_idx,
            source_video_frames=source_video_frames,
            first_appearance_only=first_appearance_only,
        )
        self._restore_role_wise_slot_meta(
            char_id,
            bank_idx,
            token_meta if isinstance(token_meta, list) else [],
            source_chunk_idx=source_chunk_idx,
        )

    def get_memory_payload(self, char_id, bank_idx=0):
        char_id = str(char_id)
        layer_tokens = {}
        layer_meta = {}
        prefix = f"{char_id}{self._SEP}"
        for stored_char in list(getattr(self, "memory_bank", {}).keys()):
            if not str(stored_char).startswith(prefix):
                continue
            _, layer = self._split_layer_char_id(stored_char)
            if layer is None:
                continue
            payload = super().get_memory_payload(stored_char, bank_idx)
            if payload is None:
                continue
            tokens = payload.get("tokens", None)
            if isinstance(tokens, torch.Tensor) and tokens.ndim >= 2 and int(tokens.shape[0]) > 0:
                layer_tokens[layer] = tokens
                layer_meta[layer] = payload.get("token_meta", [])
        if len(layer_tokens) > 0:
            return {
                "tokens": _make_layerwise_container(layer_tokens),
                "token_meta": _make_layerwise_container(layer_meta),
            }
        return super().get_memory_payload(char_id, bank_idx)

    def get_memory_payload_for_read(self, char_id, bank_idx=0):
        """Reader-only boundary used by counterfactual interventions.

        Writer updates intentionally keep calling get_memory_payload() so changing one
        target read cannot rewrite the bank state that the writer consumes.
        """
        return self.get_memory_payload(char_id, bank_idx)

class _NullImageEncoder:
    def to(self, *args, **kwargs):
        return self


def _patch_inference_lora_lazy_device(model):
    patched = 0
    for module in model.modules():
        if not (
            hasattr(module, "_original_forward_before_lora")
            and hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
        ):
            continue
        if bool(getattr(module, "_jigsaw_inference_lazy_lora_patched", False)):
            continue

        def _lazy_lora_forward(this, x, *args, **kwargs):
            out = this._original_forward_before_lora(x, *args, **kwargs)
            if bool(getattr(this, "disable_adapters", False)):
                return out
            lora_a = getattr(this, "lora_A", None)
            lora_b = getattr(this, "lora_B", None)
            if lora_a is None or lora_b is None:
                return out
            compute_dtype = x.dtype
            a_weight = lora_a.weight.to(device=x.device, dtype=compute_dtype)
            b_weight = lora_b.weight.to(device=x.device, dtype=compute_dtype)
            a_bias = None if lora_a.bias is None else lora_a.bias.to(device=x.device, dtype=compute_dtype)
            b_bias = None if lora_b.bias is None else lora_b.bias.to(device=x.device, dtype=compute_dtype)
            lora_out = F.linear(F.linear(x, a_weight, a_bias), b_weight, b_bias)
            return out + lora_out.to(dtype=out.dtype) * float(getattr(this, "lora_scale", 1.0))

        module.forward = types.MethodType(_lazy_lora_forward, module)
        module._jigsaw_inference_lazy_lora_patched = True
        patched += 1
    return patched


def _inject_inference_lora_modules(model, target_modules, lora_rank, lora_alpha, init_lora_weights=True, adapter_name=None):
    injected = 0
    target_modules = [str(x).strip() for x in target_modules if str(x).strip()]
    for module_name, module in model.named_modules():
        normalized_name = _normalize_lora_state_key(module_name)
        matched = False
        for target in target_modules:
            if normalized_name == target or normalized_name.endswith(f".{target}"):
                matched = True
                break
        if not matched:
            continue
        _install_train_lora_forward(
            module,
            rank=lora_rank,
            alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            adapter_name=adapter_name,
        )
        injected += 1
    if injected <= 0:
        raise RuntimeError(f"No modules matched LoRA targets: {target_modules}")
    return model


def _remap_lora_state_dict_for_model(model, lora_sd):
    actual_lora_keys = {}
    for key in model.state_dict().keys():
        if _is_lora_state_key(key):
            actual_lora_keys[_normalize_lora_state_key(key)] = key
    remapped = {}
    missed = []
    for key, value in lora_sd.items():
        normalized = _normalize_lora_state_key(key)
        actual_key = actual_lora_keys.get(normalized)
        if actual_key is None:
            missed.append(key)
            actual_key = key
        remapped[actual_key] = value
    return remapped, missed


class SlotMemInferenceEngine(ReferenceInferenceEngine):
    def __init__(self, args):
        self.args = args
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.native_wan_inference = bool(getattr(args, "native_wan_inference", False))
        self.train_noise_domain = str(getattr(args, "train_noise_domain", "low_noise")).strip().lower()
        self.train_stage = str(getattr(args, "train_stage", "stage1")).strip().lower()
        self.noise_domain_boundary_ratio = float(getattr(args, "noise_domain_boundary_ratio", 0.9))
        self.use_projector = bool(getattr(args, "use_projector", False))
        self._effective_use_segment_embed = bool(getattr(args, "use_segment_embed", False))
        self._effective_use_learnable_memory_pos = bool(getattr(args, "use_learnable_memory_pos", False))
        self.cfg_scale_extract = float(
            getattr(args, "cfg_scale_extraction", None)
            if getattr(args, "cfg_scale_extraction", None) is not None
            else getattr(args, "cfg_scale_extract", 5.0)
        )
        self.sample_shift = float(getattr(args, "sample_shift", 5.0))
        self.ref_pad_cfg = bool(getattr(args, "ref_pad_cfg", False))
        self.num_overlap_frame = max(0, int(getattr(args, "num_overlap_frame", 0) or 0))
        self.num_motion_frames = max(1, int(getattr(args, "num_motion_frames", 1) or 1))
        if self.num_overlap_frame > 0:
            self.num_motion_frames = max(self.num_motion_frames, self.num_overlap_frame)
        self.num_motion_latent = getattr(args, "num_motion_latent", None)

        self.pipe = build_wan22_training_pipe(
            ckpt_dir=args.ckpt_dir,
            device=self.device,
            torch_dtype=self.dtype,
            task="i2v-A14B",
            train_noise_domain=self.train_noise_domain,
            load_both_noise_models=True,
            dual_expert_load_mode=getattr(args, "dual_expert_load_mode", "active"),
            dual_expert_offload_dtype=getattr(args, "dual_expert_offload_dtype", "bfloat16"),
            dual_expert_vram_limit=getattr(args, "dual_expert_vram_limit", 0.0),
            dual_expert_manage_aux_models=bool(getattr(args, "dual_expert_manage_aux_models", False)),
        )
        if getattr(self.pipe, "image_encoder", None) is None:
            self.pipe.image_encoder = _NullImageEncoder()
        if self.num_motion_latent is not None:
            setattr(self.pipe, "default_num_motion_latent", self.num_motion_latent)
        self.sample_solver = str(getattr(args, "sample_solver", "flow_euler")).strip().lower()
        if self.sample_solver == "unipc":
            self.pipe.scheduler = _WanUniPCSchedulerAdapter(
                device=self.device,
                num_train_timesteps=getattr(self.pipe.scheduler, "num_train_timesteps", 1000),
                default_shift=self.sample_shift,
            )
            print("[Init] sample_solver=unipc (Wan FlowUniPCMultistepScheduler)", flush=True)
        self._install_pipe_compat()
        runtime_dit = self.pipe.active_denoising_model() if hasattr(self.pipe, "active_denoising_model") else self.pipe.denoising_model()
        try:
            self.patch_dim = int(runtime_dit.dim)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError("Runtime DiT must expose an integer-compatible 'dim' attribute") from exc
        if self.patch_dim <= 0:
            raise ValueError(f"Runtime DiT dim must be positive, got {self.patch_dim}")

        latent_dim = int(getattr(args, "latent_dim", 16))
        self.jigsaw_memory_bank_mode = str(getattr(args, "jigsaw_memory_bank_mode", "single")).strip().lower()
        if self.jigsaw_memory_bank_mode not in ("single", "legacy_multi"):
            self.jigsaw_memory_bank_mode = "single"
        if self.jigsaw_memory_bank_mode == "legacy_multi":
            self.memory_bank_percents = _split_csv_floats(
                getattr(args, "memory_bank_percents", "0.85,0.60,0.35,0.12"),
                default=[0.85, 0.60, 0.35, 0.12],
            )
        else:
            self.memory_bank_percents = []
        self.memory_bank_selection_mode = str(getattr(args, "memory_bank_selection_mode", "percent")).strip().lower()
        if self.memory_bank_selection_mode not in ("percent", "fixed"):
            self.memory_bank_selection_mode = "percent"
        self.fixed_memory_bank_idx = max(0, int(getattr(args, "fixed_memory_bank_idx", 0)))
        self.memory_fusion = None
        self.memory_time_importance = torch.tensor(1.0, dtype=self.dtype)
        self.memory_layer_importance = torch.tensor(1.0, dtype=self.dtype)
        self.memory_layer_profile = torch.ones(len(runtime_dit.blocks), dtype=torch.float32)

        self.memory_projector = None
        if self.use_projector:
            self.memory_projector = StyleAwareMemoryProjector(
                dim=self.patch_dim,
                time_embed_dim=self.patch_dim,
                latent_dim=latent_dim,
                bottleneck_dim=int(getattr(args, "projector_bottleneck", 256)),
                dropout=0.1,
            ).to(self.device, dtype=self.dtype)

        self.memory_embeddings = None
        self.memory_pos_embed = None
        self.segment_embed = None
        if self._effective_use_segment_embed or self._effective_use_learnable_memory_pos:
            self.memory_embeddings = LearnableMemoryEmbeddings(
                patch_dim=self.patch_dim,
                max_memory_characters=int(getattr(args, "max_memory_characters", 2)),
                max_total_memory_tokens=2048 * int(getattr(args, "max_memory_characters", 2)),
                use_segment_embed=self._effective_use_segment_embed,
                use_learnable_memory_pos=self._effective_use_learnable_memory_pos,
            ).to(self.device, dtype=self.dtype)
            self.memory_pos_embed = getattr(self.memory_embeddings, "pos_embed", None)
            self.segment_embed = getattr(self.memory_embeddings, "segment_embed", None)

        self.enable_sparse_role_memory_attn = bool(getattr(args, "enable_sparse_role_memory_attn", True))
        self.sparse_role_memory_layer_idx = int(getattr(args, "sparse_role_memory_layer_idx", 3))
        self.sparse_role_memory_injection_layers = _parse_layer_indices_csv(
            getattr(args, "sparse_role_memory_injection_layers", None),
            fallback_idx=self.sparse_role_memory_layer_idx,
        )
        self.memory_layer_binding_mode = str(getattr(args, "memory_layer_binding_mode", "layerwise")).strip().lower()
        if self.memory_layer_binding_mode not in ("layerwise", "shared"):
            self.memory_layer_binding_mode = "layerwise"
        self.enable_bank_alignment_diagnostics = bool(getattr(args, "enable_bank_alignment_diagnostics", False))
        self.sparse_role_memory_num_heads = int(getattr(args, "sparse_role_memory_num_heads", 8))
        self.sparse_role_memory_head_dim = int(getattr(args, "sparse_role_memory_head_dim", 128))
        self.sparse_role_memory_rope_dim = int(getattr(args, "sparse_role_memory_rope_dim", 256))
        self.sparse_role_memory_use_half_role_heads = bool(getattr(args, "sparse_role_memory_use_half_role_heads", True))
        self.sparse_role_memory_feature_source = str(getattr(args, "sparse_role_memory_feature_source", "attn_out"))
        self.sparse_role_memory_init_scale = float(getattr(args, "sparse_role_memory_init_scale", 0.1))
        self.sparse_role_memory_time_gate = bool(getattr(args, "sparse_role_memory_time_gate", True))
        self.sparse_role_memory_query_chunk_size = max(0, int(getattr(args, "sparse_role_memory_query_chunk_size", 128)))
        self.sparse_role_memory_layer_scales = _parse_layer_scale_map(getattr(args, "sparse_role_memory_layer_scales", ""))
        self.debug_sparse_role_memory_attn = bool(getattr(args, "debug_sparse_role_memory_attn", False))
        self.jigsaw_extra_encoder_enabled = _memory_encoder_enabled(getattr(args, "jigsaw_extra_encoder_mode", "off"))
        self.jigsaw_extra_encoder_layers = _jigsaw_parse_layer_list(getattr(args, "jigsaw_extra_encoder_layers", "0-15"))
        self.jigsaw_extra_encoder_layer_groups = _jigsaw_parse_layer_groups(getattr(args, "jigsaw_extra_encoder_layer_groups", "0-4,5-10,11-15"))
        self.jigsaw_extra_encoder_slots = max(1, int(getattr(args, "jigsaw_extra_encoder_slots", 32)))
        self.jigsaw_extra_encoder_dim = max(1, int(getattr(args, "jigsaw_extra_encoder_dim", 512)))
        self.jigsaw_extra_encoder_hidden_dim = max(1, int(getattr(args, "jigsaw_extra_encoder_hidden_dim", 1024)))
        self.jigsaw_extra_encoder_use_t_embed = bool(getattr(args, "jigsaw_extra_encoder_use_t_embed", False))
        self.jigsaw_memory_encoder_t_embed_source = str(
            getattr(args, "jigsaw_memory_encoder_t_embed_source", "current")
        ).strip().lower()
        self.jigsaw_extra_encoder_use_slot_index_embed = bool(getattr(args, "jigsaw_extra_encoder_use_slot_index_embed", False))
        self.jigsaw_stage2_writer_mode = str(getattr(args, "jigsaw_stage2_writer_mode", "auto")).strip().lower()
        self.memory_writer_effective_mode = _memory_writer_effective_mode(self.train_stage, self.jigsaw_stage2_writer_mode)
        self.memory_writer_enabled = bool(
            self.jigsaw_extra_encoder_enabled and self.memory_writer_effective_mode == "residual"
        )
        self.jigsaw_stage2_writer_hidden_dim = max(1, int(getattr(args, "jigsaw_stage2_writer_hidden_dim", 1024)))
        self.jigsaw_stage2_writer_init_scale = float(getattr(args, "jigsaw_stage2_writer_init_scale", 0.1))
        self.jigsaw_stage2_writer_precision_tau = float(getattr(args, "jigsaw_stage2_writer_precision_tau", 0.3))
        self.jigsaw_stage2_writer_precision_scale = float(getattr(args, "jigsaw_stage2_writer_precision_scale", 10.0))
        self.jigsaw_stage2_writer_max_delta_ratio = float(getattr(args, "jigsaw_stage2_writer_max_delta_ratio", 0.0))
        self.jigsaw_stage2_writer_max_delta_norm = float(getattr(args, "jigsaw_stage2_writer_max_delta_norm", 0.0))
        self.jigsaw_stage2_writer_detach_c_short = bool(getattr(args, "jigsaw_stage2_writer_detach_c_short", True))
        self.jigsaw_disable_memory_side_rope = bool(getattr(args, "jigsaw_disable_memory_side_rope", True))
        self.char_attn_noise_scope = str(
            getattr(args, "char_attn_noise_scope", self.train_noise_domain)
        ).strip().lower()
        requested_sparse_domains = set()
        high_expert_ckpt_path = getattr(args, "high_expert_checkpoint_path", None)
        low_expert_ckpt_path = getattr(args, "low_expert_checkpoint_path", None)
        if isinstance(high_expert_ckpt_path, str) and high_expert_ckpt_path.strip():
            requested_sparse_domains.add("high_noise")
        if isinstance(low_expert_ckpt_path, str) and low_expert_ckpt_path.strip():
            requested_sparse_domains.add("low_noise")
        if len(requested_sparse_domains) <= 0:
            if self.char_attn_noise_scope in ("high_noise", "low_noise"):
                requested_sparse_domains.add(self.char_attn_noise_scope)
            else:
                requested_sparse_domains.add(self.train_noise_domain)
        self.sparse_role_memory_attn_low_noise = None
        self.sparse_role_memory_attn_high_noise = None
        if self.enable_sparse_role_memory_attn:
            def _build_sparse_module():
                return CharacterWiseCrossAttention(
                    dim=self.patch_dim,
                    num_heads=self.sparse_role_memory_num_heads,
                    head_dim=self.sparse_role_memory_head_dim,
                    rope_dim=self.sparse_role_memory_rope_dim,
                    use_half_role_heads=self.sparse_role_memory_use_half_role_heads,
                    max_query_tokens_per_role=int(getattr(args, "max_memory_tokens_per_character", 0)),
                    query_chunk_size=self.sparse_role_memory_query_chunk_size,
                    use_memory_side_rope=not bool(self.jigsaw_disable_memory_side_rope),
                    add_rope_center_to_value=not bool(self.jigsaw_disable_memory_side_rope),
                    init_scale=self.sparse_role_memory_init_scale,
                    time_gate=self.sparse_role_memory_time_gate,
                    debug=self.debug_sparse_role_memory_attn,
                ).to(self.device, dtype=self.dtype)

            if "low_noise" in requested_sparse_domains:
                self.sparse_role_memory_attn_low_noise = _build_sparse_module()
            if "high_noise" in requested_sparse_domains:
                self.sparse_role_memory_attn_high_noise = _build_sparse_module()
            self.sparse_role_memory_attn = self._get_character_wise_cross_attention_for_domain(self.train_noise_domain)
        else:
            self.sparse_role_memory_attn = None
        self.jigsaw_extra_encoder_low_noise = None
        self.jigsaw_extra_encoder_high_noise = None
        if self.jigsaw_extra_encoder_enabled:
            def _build_extra_encoder():
                return MemoryEncoderBank(
                    dim=int(self.patch_dim),
                    layer_groups=self.jigsaw_extra_encoder_layer_groups,
                    slots=int(self.jigsaw_extra_encoder_slots),
                    encoder_dim=int(self.jigsaw_extra_encoder_dim),
                    hidden_dim=int(self.jigsaw_extra_encoder_hidden_dim),
                    use_t_embed=bool(self.jigsaw_extra_encoder_use_t_embed),
                    use_slot_index_embed=bool(self.jigsaw_extra_encoder_use_slot_index_embed),
                    time_embed_dim=int(self.patch_dim),
                ).to(self.device, dtype=self.dtype)

            if "low_noise" in requested_sparse_domains:
                self.jigsaw_extra_encoder_low_noise = _build_extra_encoder()
            if "high_noise" in requested_sparse_domains:
                self.jigsaw_extra_encoder_high_noise = _build_extra_encoder()
            self.jigsaw_extra_encoder = self.jigsaw_extra_encoder_low_noise or self.jigsaw_extra_encoder_high_noise
            wide_layer_list = list(self.jigsaw_extra_encoder_layers)
            wide_layers_csv = ",".join(str(x) for x in wide_layer_list)
            setattr(self.args, "extract_layers", wide_layer_list)
            setattr(self.args, "sparse_role_memory_injection_layers", wide_layers_csv)
            self.sparse_role_memory_injection_layers = list(wide_layer_list)
            print(
                f"[MemoryEncoder][Init] enabled=1 layers={wide_layer_list} groups={self.jigsaw_extra_encoder_layer_groups} "
                f"slots={self.jigsaw_extra_encoder_slots} "
                f"use_t_embed={bool(self.jigsaw_extra_encoder_use_t_embed)} "
                f"use_slot_index_embed={bool(self.jigsaw_extra_encoder_use_slot_index_embed)} "
                f"memory_side_rope={not bool(self.jigsaw_disable_memory_side_rope)}",
                flush=True,
            )
        else:
            self.jigsaw_extra_encoder = None
        self.jigsaw_stage2_writer_low_noise = None
        self.jigsaw_stage2_writer_high_noise = None
        self.jigsaw_stage2_writer = None
        if bool(getattr(self, "memory_writer_enabled", False)):
            def _build_stage2_writer():
                return MemoryWriter(
                    dim=int(self.patch_dim),
                    hidden_dim=int(self.jigsaw_stage2_writer_hidden_dim),
                    init_scale=float(self.jigsaw_stage2_writer_init_scale),
                    precision_tau=float(self.jigsaw_stage2_writer_precision_tau),
                    precision_scale=float(self.jigsaw_stage2_writer_precision_scale),
                    max_delta_ratio=float(self.jigsaw_stage2_writer_max_delta_ratio),
                    max_delta_norm=float(self.jigsaw_stage2_writer_max_delta_norm),
                    detach_c_short=bool(self.jigsaw_stage2_writer_detach_c_short),
                ).to(self.device)

            if "low_noise" in requested_sparse_domains:
                self.jigsaw_stage2_writer_low_noise = _build_stage2_writer()
            if "high_noise" in requested_sparse_domains:
                self.jigsaw_stage2_writer_high_noise = _build_stage2_writer()
            self.jigsaw_stage2_writer = self.jigsaw_stage2_writer_low_noise or self.jigsaw_stage2_writer_high_noise
            print(
                f"[MemoryWriter][Init] enabled=1 mode={self.memory_writer_effective_mode} "
                f"stage={self.train_stage} hidden={self.jigsaw_stage2_writer_hidden_dim} "
                f"init_scale={self.jigsaw_stage2_writer_init_scale} tau={self.jigsaw_stage2_writer_precision_tau} "
                f"scale={self.jigsaw_stage2_writer_precision_scale}",
                flush=True,
            )
        elif str(getattr(self, "memory_writer_effective_mode", "off")) == "residual":
            print("[MemoryWriter][Init] requested but disabled because jigsaw_extra_encoder is off", flush=True)

        self._last_sparse_role_memory_stats = {
            "enabled": 0.0,
            "selected_query_tokens": 0,
            "selected_memory_tokens": 0,
            "winner_counts": {},
            "role_head_out_norm": 0.0,
            "plain_head_out_norm": 0.0,
            "attn_entropy": 0.0,
        }
        self._last_sparse_role_memory_stats_by_layer = {}
        self._last_jigsaw_stage2_writer_stats = {"enabled": 0.0, "input_slots": 0, "updated_slots": 0}
        self._jigsaw_stage2_writer_print_count = 0
        self.runtime_chunk_warnings = []
        self.runtime_role_states = []
        self.weights_loaded = False
        self.loaded_checkpoint_domains = set()
        if self.native_wan_inference:
            print("[Init] native_wan_inference=True, skip checkpoint loading and memory injection path")
            self.enable_sparse_role_memory_attn = False
            self.weights_loaded = True
        elif getattr(args, "defer_lora_until_after_first_chunk", False):
            print("[Init] defer_lora_until_after_first_chunk=True, start with base WAN for chunk 0")
        else:
            self.load_trained_weights_if_needed()

    def _install_pipe_compat(self):
        def _encode_prompt_dict(pipe_obj, prompt, positive=True):
            context = pipe_obj.prompter.encode_prompt(prompt, positive=positive, device=self.device)
            return {"context": context}

        def _set_timesteps_for_inference(scheduler_obj, num_inference_steps, shift=None):
            if shift is None:
                shift = float(getattr(self, "sample_shift", 5.0))
            return scheduler_obj.get_timesteps(
                num_inference_steps=num_inference_steps,
                denoising_strength=1.0,
                shift=shift,
            )

        self.pipe.encode_prompt = types.MethodType(_encode_prompt_dict, self.pipe)
        if not isinstance(self.pipe.scheduler, _WanUniPCSchedulerAdapter):
            self.pipe.scheduler.set_timesteps = types.MethodType(_set_timesteps_for_inference, self.pipe.scheduler)

    def _offload(self):
        if not bool(getattr(self.args, "offload_models", True)):
            return
        torch.cuda.empty_cache()
        for attr in ("dit", "dit2", "high_noise_model", "low_noise_model", "vae", "text_encoder", "image_encoder"):
            module = getattr(self.pipe, attr, None)
            if module is not None:
                if hasattr(module, "force_to"):
                    module.force_to("cpu")
                else:
                    module.to("cpu")
        prompter = getattr(self.pipe, "prompter", None)
        prompter_text_encoder = getattr(prompter, "text_encoder", None)
        if prompter_text_encoder is not None:
            prompter_text_encoder.to("cpu")
        torch.cuda.empty_cache()

    def load_trained_weights_if_needed(self):
        if self.native_wan_inference or self.weights_loaded:
            return
        loaded_any = False
        high_ckpt = getattr(self.args, "high_expert_checkpoint_path", None)
        low_ckpt = getattr(self.args, "low_expert_checkpoint_path", None)
        if isinstance(high_ckpt, str) and high_ckpt.strip():
            self._load_trained_weights(high_ckpt, target_noise_domain="high_noise")
            loaded_any = True
        if isinstance(low_ckpt, str) and low_ckpt.strip():
            self._load_trained_weights(low_ckpt, target_noise_domain="low_noise")
            loaded_any = True
        if not loaded_any:
            raise RuntimeError("high_expert_checkpoint_path or low_expert_checkpoint_path is required")
        self._disable_learned_memory_for_unloaded_domains()
        self.weights_loaded = True

    def _disable_learned_memory_for_unloaded_domains(self):
        """Keep one-sided expert ablations from using random memory modules."""
        for domain, sparse_attr, encoder_attr, writer_attr in (
            (
                "high_noise",
                "sparse_role_memory_attn_high_noise",
                "jigsaw_extra_encoder_high_noise",
                "jigsaw_stage2_writer_high_noise",
            ),
            (
                "low_noise",
                "sparse_role_memory_attn_low_noise",
                "jigsaw_extra_encoder_low_noise",
                "jigsaw_stage2_writer_low_noise",
            ),
        ):
            if domain in self.loaded_checkpoint_domains:
                continue
            disabled = []
            if getattr(self, sparse_attr, None) is not None:
                setattr(self, sparse_attr, None)
                disabled.append("sparse_role_memory_attn")
            if getattr(self, encoder_attr, None) is not None:
                setattr(self, encoder_attr, None)
                disabled.append("jigsaw_extra_encoder")
            if getattr(self, writer_attr, None) is not None:
                setattr(self, writer_attr, None)
                disabled.append("jigsaw_stage2_writer")
            if disabled:
                print(
                    f"[Init] Disabled unloaded {domain} learned memory modules: {', '.join(disabled)}",
                    flush=True,
                )

        current_domain = getattr(self.pipe, "current_noise_domain", self.train_noise_domain)
        self.sparse_role_memory_attn = self._get_character_wise_cross_attention_for_domain(current_domain)
        self.jigsaw_extra_encoder = self.jigsaw_extra_encoder_low_noise or self.jigsaw_extra_encoder_high_noise
        self.jigsaw_stage2_writer = self.jigsaw_stage2_writer_low_noise or self.jigsaw_stage2_writer_high_noise

    def _get_num_train_timesteps(self):
        scheduler = getattr(self.pipe, "scheduler", None)
        return max(int(getattr(scheduler, "num_train_timesteps", 1000) or 1000), 1)

    def _use_legacy_multi_memory_banks(self):
        return str(getattr(self, "jigsaw_memory_bank_mode", "single")).strip().lower() == "legacy_multi"

    def _select_single_bank_key(self, bank_map, require_tensor=False):
        if not isinstance(bank_map, dict) or len(bank_map) <= 0:
            return None
        keys = []
        for key, value in bank_map.items():
            if require_tensor and not (isinstance(value, torch.Tensor) and value.ndim >= 2 and int(value.shape[0]) > 0):
                continue
            keys.append(str(key))
        if "0" in keys:
            return "0"
        if len(keys) == 1:
            return keys[0]
        return None

    def _bank_idx_from_key(self, bank_key):
        try:
            return int(bank_key)
        except Exception:
            return 0

    def _get_bank_map_value(self, bank_map, bank_key, default=None):
        if not isinstance(bank_map, dict):
            return default
        if bank_key in bank_map:
            return bank_map.get(bank_key, default)
        key_str = str(bank_key)
        if key_str in bank_map:
            return bank_map.get(key_str, default)
        try:
            key_int = int(bank_key)
        except Exception:
            key_int = None
        if key_int is not None and key_int in bank_map:
            return bank_map.get(key_int, default)
        return default

    def _single_online_memory_bank_percents(self):
        scheduler = getattr(self.pipe, "scheduler", None)
        if scheduler is None:
            return [0.0]
        try:
            scheduler.set_timesteps(
                int(getattr(self.args, "num_inference_steps", 50)),
                shift=float(getattr(self.args, "sample_shift", 5.0)),
            )
        except TypeError:
            scheduler.set_timesteps(int(getattr(self.args, "num_inference_steps", 50)))
        timesteps = getattr(scheduler, "timesteps", None)
        if isinstance(timesteps, torch.Tensor) and timesteps.numel() > 0:
            final_t = timesteps[-1]
            return [float(final_t.detach().float().item()) / float(self._get_num_train_timesteps())]
        if isinstance(timesteps, (list, tuple)) and len(timesteps) > 0:
            return [float(timesteps[-1]) / float(self._get_num_train_timesteps())]
        return [0.0]

    def _timestep_percent(self, timestep):
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() <= 0:
                return 0.0
            return float(timestep.detach().float().mean().item()) / float(self._get_num_train_timesteps())
        try:
            return float(timestep) / float(self._get_num_train_timesteps())
        except Exception:
            return 0.0

    def _resolve_inference_noise_domain_from_timestep(self, timestep, boundary_ratio=None):
        boundary = float(self.noise_domain_boundary_ratio if boundary_ratio is None else boundary_ratio)
        p = self._timestep_percent(timestep)
        return "high_noise" if p >= boundary else "low_noise"

    def _set_inference_noise_domain_from_timestep(self, timestep, boundary_ratio=None):
        domain = self._resolve_inference_noise_domain_from_timestep(timestep, boundary_ratio=boundary_ratio)
        if hasattr(self.pipe, "set_active_noise_domain"):
            self.pipe.set_active_noise_domain(domain)
        else:
            setattr(self.pipe, "current_noise_domain", domain)
        return domain

    def _resolve_noise_domain_from_timestep(self, timestep):
        return self._resolve_inference_noise_domain_from_timestep(timestep)

    def _get_character_wise_cross_attention_for_domain(self, noise_domain):
        if not bool(getattr(self, "enable_sparse_role_memory_attn", True)):
            return None
        domain = str(noise_domain).strip().lower()
        if domain == "low_noise":
            return getattr(self, "sparse_role_memory_attn_low_noise", None)
        if domain == "high_noise":
            return getattr(self, "sparse_role_memory_attn_high_noise", None)
        return None

    def _get_jigsaw_extra_encoder_for_domain(self, noise_domain):
        if not bool(getattr(self, "jigsaw_extra_encoder_enabled", False)):
            return None
        domain = str(noise_domain).strip().lower()
        if domain == "low_noise":
            return getattr(self, "jigsaw_extra_encoder_low_noise", None)
        if domain == "high_noise":
            return getattr(self, "jigsaw_extra_encoder_high_noise", None)
        return getattr(self, "jigsaw_extra_encoder", None)

    def _get_memory_writer_for_domain(self, noise_domain):
        if not bool(getattr(self, "memory_writer_enabled", False)):
            return None
        domain = str(noise_domain).strip().lower()
        if domain == "low_noise":
            return getattr(self, "jigsaw_stage2_writer_low_noise", None)
        if domain == "high_noise":
            return getattr(self, "jigsaw_stage2_writer_high_noise", None)
        return getattr(self, "jigsaw_stage2_writer", None)

    def _memory_meta_is_encoded_slots(self, token_meta):
        if not isinstance(token_meta, list) or len(token_meta) <= 0:
            return False
        valid = [m for m in token_meta if isinstance(m, dict)]
        if len(valid) <= 0:
            return False
        return any(bool(m.get("is_jigsaw_extra_encoder_slot", False)) for m in valid)

    def _representative_t_embed_for_stage2_slot_update(self, noise_domain):
        domain = str(noise_domain or self.train_noise_domain).strip().lower()
        dit = self.pipe.get_noise_model(domain) if hasattr(self.pipe, "get_noise_model") else self.pipe.denoising_model()
        if dit is None:
            raise RuntimeError(f"Missing DiT model for stage2 slot update domain={domain}")
        was_training = bool(getattr(dit, "training", False))
        if hasattr(dit, "force_to"):
            dit.force_to(self.device)
        else:
            dit.to(self.device)
        dit.eval()
        num_train_timesteps = float(max(int(getattr(self.pipe.scheduler, "num_train_timesteps", 1000)), 1))
        boundary = float(getattr(self, "noise_domain_boundary_ratio", 0.9))
        if domain == "high_noise":
            p = min(0.995, max(boundary + (1.0 - boundary) * 0.5, boundary))
        else:
            p = min(max(boundary * 0.5, 0.05), 0.85)
        t_value = torch.tensor([p * num_train_timesteps], device=self.device, dtype=torch.float32)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            t_embed = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, t_value).float().to(device=self.device))
        if was_training:
            dit.train()
        return t_embed

    @torch.no_grad()
    def _encode_memory_payload_to_stage2_slots(self, tokens, token_meta, noise_domain=None, layer_idx=0):
        if self._memory_meta_is_encoded_slots(token_meta):
            return tokens, list(token_meta or []), {
                "enabled": 0.0,
                "already_encoded": 1.0,
                "input_tokens": int(tokens.shape[0]) if isinstance(tokens, torch.Tensor) else 0,
                "output_slots": int(tokens.shape[0]) if isinstance(tokens, torch.Tensor) else 0,
            }
        if not (isinstance(tokens, torch.Tensor) and tokens.ndim >= 2 and int(tokens.shape[0]) > 0):
            return tokens, [], {"enabled": 0.0, "input_tokens": 0, "output_slots": 0}
        domain = str(noise_domain or self.train_noise_domain).strip().lower()
        encoder = self._get_jigsaw_extra_encoder_for_domain(domain)
        if encoder is None:
            raise RuntimeError(f"Stage2 slot update requires SlotMem memory encoder for domain={domain}")
        encoder.to(self.device)
        model_dtype = next(encoder.parameters()).dtype
        t_embed = self._representative_t_embed_for_stage2_slot_update(domain)
        slots, slot_meta, _lengths, stats = _memory_encode_role_tokens_to_slots(
            encoder,
            tokens.to(device=self.device, dtype=model_dtype),
            list(token_meta or []) if isinstance(token_meta, list) else [],
            layer_idx=layer_idx,
            t_embed=t_embed,
        )
        return slots.detach().cpu(), list(slot_meta), dict(stats)

    def _role_payload_from_slot_meta(self, token_meta, max_tokens, device):
        payload = {}
        if not isinstance(token_meta, list) or int(max_tokens) <= 0:
            return payload
        role_to_indices = defaultdict(list)
        for i, item in enumerate(token_meta[:int(max_tokens)]):
            if not isinstance(item, dict):
                continue
            rid = str(item.get("char_id", "")).strip()
            if rid:
                role_to_indices[rid].append(int(i))
        for rid, indices in role_to_indices.items():
            if indices:
                payload[str(rid)] = {
                    "flat_idx": torch.tensor(indices, device=device, dtype=torch.long),
                }
        return payload

    @torch.no_grad()
    def stage2_update_slot_payload(self, old_tokens, old_meta, update_tokens, update_meta, noise_domain=None, layer_idx=0):
        domain = str(noise_domain or self.train_noise_domain).strip().lower()
        old_slots, old_slot_meta, old_stats = self._encode_memory_payload_to_stage2_slots(
            old_tokens,
            old_meta,
            noise_domain=domain,
            layer_idx=layer_idx,
        )
        update_slots, update_slot_meta, update_stats = self._encode_memory_payload_to_stage2_slots(
            update_tokens,
            update_meta,
            noise_domain=domain,
            layer_idx=layer_idx,
        )
        writer = self._get_memory_writer_for_domain(domain)
        if writer is None:
            return old_slots, old_slot_meta, {
                "enabled": 0.0,
                "writer_missing": 1.0,
                "old_encode": old_stats,
                "update_encode": update_stats,
            }
        if not (
            isinstance(old_slots, torch.Tensor)
            and old_slots.ndim == 2
            and int(old_slots.shape[0]) > 0
            and isinstance(update_slots, torch.Tensor)
            and update_slots.ndim == 2
            and int(update_slots.shape[0]) > 0
        ):
            return old_slots, old_slot_meta, {
                "enabled": 0.0,
                "empty_slots": 1.0,
                "old_encode": old_stats,
                "update_encode": update_stats,
            }
        writer_device = next(writer.parameters()).device
        if writer_device.type == "cpu":
            writer.to(self.device)
            writer_device = torch.device(self.device)
        writer_dtype = next(writer.parameters()).dtype
        update_payload = self._role_payload_from_slot_meta(
            update_slot_meta,
            int(update_slots.shape[0]),
            writer_device,
        )
        updated_slots, writer_stats = writer(
            old_slots.to(device=writer_device, dtype=writer_dtype),
            old_slot_meta,
            update_payload,
            update_slots.to(device=writer_device, dtype=writer_dtype).unsqueeze(0),
        )
        stats = dict(writer_stats)
        stats["residual_norm"] = float(
            (updated_slots.detach().float() - old_slots.to(device=updated_slots.device).detach().float())
            .norm(dim=-1)
            .mean()
            .item()
        )
        stats["old_encode"] = old_stats
        stats["update_encode"] = update_stats
        return updated_slots.detach().cpu(), old_slot_meta, stats

    def _normalize_model_key(self, key):
        return _normalize_lora_state_key(key)

    def _load_trained_weights(self, ckpt_path, target_noise_domain=None):
        if not isinstance(ckpt_path, str) or ckpt_path.strip() == "":
            raise RuntimeError("expert checkpoint path is required")
        target_noise_domain = str(target_noise_domain or self.train_noise_domain).strip().lower()
        if target_noise_domain not in ("low_noise", "high_noise"):
            raise RuntimeError(f"Unsupported target_noise_domain: {target_noise_domain}")
        print(f"[Init] Loading checkpoint for {target_noise_domain} from {ckpt_path}...")
        full_sd = {}
        for path in [p.strip() for p in ckpt_path.split(",") if p.strip()]:
            shard = _load_checkpoint_payload(path)
            if isinstance(shard, dict) and "state_dict" in shard:
                shard = shard["state_dict"]
            if not isinstance(shard, dict):
                raise RuntimeError(f"Invalid checkpoint payload: {path}")
            full_sd.update(shard)
        print(f"  > Checkpoint keys: {len(full_sd)}")

        target_dit_model = self.pipe.get_noise_model(target_noise_domain)
        if target_dit_model is None:
            raise RuntimeError(f"Missing denoising model for target_noise_domain={target_noise_domain}")
        managed_dual_expert = bool(getattr(self.pipe, "dual_expert_vram_management_enabled", False))
        active_dual_expert = bool(getattr(self.pipe, "dual_expert_active_offload_enabled", False))
        low_vram_dual_expert = bool(managed_dual_expert or active_dual_expert)
        dit_model = target_dit_model if low_vram_dual_expert else target_dit_model.to(self.device)
        lora_targets = str(getattr(self.args, "lora_target_modules", "q,k,v,o,ffn.0,ffn.2")).split(",")
        if managed_dual_expert:
            _inject_inference_lora_modules(
                dit_model,
                target_modules=lora_targets,
                lora_rank=int(getattr(self.args, "lora_rank", 128)),
                lora_alpha=float(getattr(self.args, "lora_alpha", 128.0)),
                init_lora_weights=bool(str(getattr(self.args, "init_lora_weights", "kaiming")).strip().lower() == "kaiming"),
                adapter_name=target_noise_domain,
            )
        else:
            _inject_train_lora_modules(
                dit_model,
                target_modules=lora_targets,
                lora_rank=int(getattr(self.args, "lora_rank", 128)),
                lora_alpha=float(getattr(self.args, "lora_alpha", 128.0)),
                init_lora_weights=bool(str(getattr(self.args, "init_lora_weights", "kaiming")).strip().lower() == "kaiming"),
                adapter_name=target_noise_domain,
            )
        if managed_dual_expert:
            patched_lora = _patch_inference_lora_lazy_device(dit_model)
            print(
                f"  > Patched low-VRAM lazy LoRA forwards: {patched_lora} modules "
                f"(domain={target_noise_domain})",
                flush=True,
            )
        lora_sd = {}
        for key, value in full_sd.items():
            norm_key = self._normalize_model_key(key)
            if _is_lora_state_key(norm_key):
                lora_sd[norm_key] = value
        if len(lora_sd) <= 0:
            raise RuntimeError("Checkpoint contains no custom LoRA keys.")
        if managed_dual_expert:
            lora_sd, missed_lora_keys = _remap_lora_state_dict_for_model(dit_model, lora_sd)
            if missed_lora_keys:
                print(
                    f"  > [LowVRAM LoRA][Warning] could not remap {len(missed_lora_keys)} LoRA keys; "
                    f"first={missed_lora_keys[0]}",
                    flush=True,
                )
        missing, unexpected = dit_model.load_state_dict(lora_sd, strict=False)
        print(f"  > Loaded LoRA keys: {len(lora_sd)} | missing={len(missing)} unexpected={len(unexpected)}")
        if bool(getattr(self.args, "strict_weight_loading", False)) and (len(missing) > 0 or len(unexpected) > 0):
            raise RuntimeError(f"Strict LoRA loading failed: missing={len(missing)}, unexpected={len(unexpected)}")

        if self.use_projector and self.memory_projector is not None:
            projector_sd = {}
            for key, value in full_sd.items():
                if str(key).startswith("memory_projector."):
                    projector_sd[str(key)[len("memory_projector."):]] = value
            if projector_sd:
                self.memory_projector.load_state_dict(projector_sd, strict=True)
                print(f"  > Loaded memory_projector params: {len(projector_sd)}")

        if self.memory_embeddings is not None:
            embed_sd = {}
            pos = full_sd.get("memory_pos_embed", None)
            if pos is not None and getattr(self.memory_embeddings, "pos_embed", None) is not None:
                if tuple(pos.shape) == tuple(self.memory_embeddings.pos_embed.shape):
                    embed_sd["pos_embed"] = pos.to(dtype=self.dtype)
                else:
                    print(f"  > Skip memory_pos_embed due to shape mismatch: ckpt={tuple(pos.shape)} current={tuple(self.memory_embeddings.pos_embed.shape)}")
            seg = full_sd.get("memory_segment_embed", None)
            if seg is not None and getattr(self.memory_embeddings, "segment_embed", None) is not None:
                if tuple(seg.shape) == tuple(self.memory_embeddings.segment_embed.shape):
                    embed_sd["segment_embed"] = seg.to(dtype=self.dtype)
                else:
                    print(f"  > Skip memory_segment_embed due to shape mismatch: ckpt={tuple(seg.shape)} current={tuple(self.memory_embeddings.segment_embed.shape)}")
            if embed_sd:
                self.memory_embeddings.load_state_dict(embed_sd, strict=False)
                self.memory_pos_embed = getattr(self.memory_embeddings, "pos_embed", None)
                self.segment_embed = getattr(self.memory_embeddings, "segment_embed", None)
                print(f"  > Loaded memory embedding params: {sorted(embed_sd.keys())}")

        sparse_role_memory_module = self._get_character_wise_cross_attention_for_domain(target_noise_domain)
        if self.enable_sparse_role_memory_attn and sparse_role_memory_module is not None:
            sparse_sd = {}
            domain_prefix = f"sparse_role_memory_attn_{target_noise_domain}."
            alt_prefixes = [domain_prefix, "sparse_role_memory_attn."]
            for key, value in full_sd.items():
                for prefix in alt_prefixes:
                    if str(key).startswith(prefix):
                        sparse_sd[str(key)[len(prefix):]] = value
                        break
            if len(sparse_sd) <= 0:
                if bool(getattr(self.args, "strict_weight_loading", False)):
                    raise RuntimeError(f"No sparse role memory keys found for domain {target_noise_domain}")
                print(
                    f"  > No sparse_role_memory_attn keys found for domain {target_noise_domain}; "
                    f"disable sparse role memory path for this checkpoint",
                    flush=True,
                )
                if target_noise_domain == "high_noise":
                    self.sparse_role_memory_attn_high_noise = None
                else:
                    self.sparse_role_memory_attn_low_noise = None
                self.runtime_chunk_warnings.append(
                    f"sparse_role_memory_attn disabled: checkpoint has no sparse role-memory weights for domain {target_noise_domain}"
                )
            else:
                sparse_role_memory_module.load_state_dict(sparse_sd, strict=True)
                print(f"  > Loaded sparse_role_memory_attn params: {len(sparse_sd)}")
        jigsaw_module = self._get_jigsaw_extra_encoder_for_domain(target_noise_domain)
        if self.jigsaw_extra_encoder_enabled and jigsaw_module is None:
            raise RuntimeError(f"SlotMem memory encoder is enabled but no module exists for domain {target_noise_domain}")
        if self.jigsaw_extra_encoder_enabled and jigsaw_module is not None:
            jigsaw_sd = {}
            for prefix in (f"jigsaw_extra_encoder_{target_noise_domain}.", "jigsaw_extra_encoder."):
                jigsaw_sd = _jigsaw_extract_prefixed_state_dict(full_sd, prefix)
                if jigsaw_sd:
                    break
            if len(jigsaw_sd) <= 0:
                raise RuntimeError(f"No SlotMem memory encoder keys found for domain {target_noise_domain}")
            else:
                missing, unexpected = jigsaw_module.load_state_dict(jigsaw_sd, strict=False)
                print(
                    f"  > Loaded jigsaw_extra_encoder_{target_noise_domain} params: {len(jigsaw_sd)} "
                    f"missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )
        writer_module = self._get_memory_writer_for_domain(target_noise_domain)
        if self.memory_writer_enabled and writer_module is None:
            raise RuntimeError(f"SlotMem stage2 memory writer is enabled but no module exists for domain {target_noise_domain}")
        if self.memory_writer_enabled and writer_module is not None:
            writer_sd = {}
            for prefix in (f"jigsaw_stage2_writer_{target_noise_domain}.", "jigsaw_stage2_writer."):
                writer_sd = _jigsaw_extract_prefixed_state_dict(full_sd, prefix)
                if writer_sd:
                    break
            if len(writer_sd) <= 0:
                warning = (
                    f"jigsaw_stage2_writer no-op: checkpoint has no writer weights for domain {target_noise_domain}"
                )
                print(f"  > [MemoryWriter][Warning] {warning}", flush=True)
                self.runtime_chunk_warnings.append(warning)
                if target_noise_domain == "high_noise":
                    self.jigsaw_stage2_writer_high_noise = None
                else:
                    self.jigsaw_stage2_writer_low_noise = None
                self.jigsaw_stage2_writer = (
                    self.jigsaw_stage2_writer_low_noise or self.jigsaw_stage2_writer_high_noise
                )
            else:
                missing, unexpected = writer_module.load_state_dict(writer_sd, strict=False)
                print(
                    f"  > Loaded jigsaw_stage2_writer_{target_noise_domain} params: {len(writer_sd)} "
                    f"missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )
        self.loaded_checkpoint_domains.add(target_noise_domain)
        current_domain = getattr(self.pipe, "current_noise_domain", self.train_noise_domain)
        self.sparse_role_memory_attn = self._get_character_wise_cross_attention_for_domain(current_domain)
        if managed_dual_expert and hasattr(self.pipe, "offload_vram_managed_models"):
            self.pipe.offload_vram_managed_models()
        if active_dual_expert:
            if hasattr(dit_model, "force_to"):
                dit_model.force_to("cpu")
            else:
                dit_model.to("cpu")

    def _get_runtime_dit_for_timestep(self, timestep):
        current_noise_domain = self._set_inference_noise_domain_from_timestep(timestep)
        dit_model = self.pipe.get_noise_model(current_noise_domain) if hasattr(self.pipe, "get_noise_model") else self.pipe.denoising_model()
        if dit_model is None:
            raise RuntimeError(f"Missing denoising model for timestep domain={current_noise_domain}")
        return dit_model

    def _sync_probe_dit_for_timestep(self, timestep):
        dit_model = self._get_runtime_dit_for_timestep(timestep)
        self.pipe.dit = dit_model
        return dit_model

    def _make_probe_pipe(self, dit_model):
        return types.SimpleNamespace(dit=dit_model)

    @torch.no_grad()
    def _prepare_character_semantic_probe_configs(self, prompt, role_ids):
        if not isinstance(prompt, str) or prompt.strip() == "":
            return [], []
        if not isinstance(role_ids, (list, tuple)):
            return [], []
        char_configs = []
        ordered_roles = []
        for role_id in role_ids:
            role_id = str(role_id).strip()
            if role_id == "":
                continue
            character_name_candidates = []
            for name_candidate in (
                role_id.replace("_", " ").strip(),
                role_id.replace("_", " ").replace("-", " ").strip(),
            ):
                if name_candidate and name_candidate not in character_name_candidates:
                    character_name_candidates.append(name_candidate)
            parts = role_id.replace("-", "_").split("_", 1)
            prefix_text = parts[0]
            suffix_text = parts[1].replace("_", " ") if len(parts) > 1 else None
            full_indices = []
            for character_name in character_name_candidates:
                token_ids, token_texts, _ = verify_target_text_is_single_token(self.pipe, character_name)
                if not token_ids:
                    continue
                full_indices = find_token_index_in_prompt(self.pipe, prompt, character_name, token_ids, token_texts)
                if full_indices:
                    break
            if not full_indices:
                continue
            prefix_token_ids, _, _ = verify_target_text_is_single_token(self.pipe, prefix_text)
            num_prefix_tokens = len(prefix_token_ids) if prefix_token_ids else 0
            if suffix_text:
                suffix_token_ids, _, _ = verify_target_text_is_single_token(self.pipe, suffix_text)
                num_suffix_tokens = len(suffix_token_ids) if suffix_token_ids else 0
            else:
                num_suffix_tokens = 0
            prefix_indices = full_indices[:num_prefix_tokens]
            suffix_indices = full_indices[num_prefix_tokens:num_prefix_tokens + num_suffix_tokens] if num_suffix_tokens > 0 else []
            char_configs.append(
                {
                    "target_token_indices": prefix_indices,
                    "suffix_token_indices": suffix_indices,
                    "all_token_indices": list(full_indices),
                    "suffix_scale": float(getattr(self.args, "suffix_attention_scale", 1.0)),
                    "token_weight": float(getattr(self.args, "token_weight", 1.0)),
                }
            )
            ordered_roles.append(role_id)
        return char_configs, ordered_roles

    @torch.no_grad()
    def _get_role_token_selection_mode(self):
        return str(getattr(self.args, "role_token_selection_mode", "baseline")).strip().lower()

    @torch.no_grad()
    def _use_two_role_difference_selection(self):
        return self._get_role_token_selection_mode() == "two_role_diff"

    @torch.no_grad()
    def _use_role_contrast_selection(self):
        return self._get_role_token_selection_mode() in ("two_role_diff", "one_vs_rest")

    @torch.no_grad()
    def _aggregate_character_semantic_responses_for_layer(self, step_maps, layer_idx):
        if not isinstance(step_maps, dict) or len(step_maps) == 0:
            return None
        layer_key = _layer_key(layer_idx)
        layer_map = None
        if layer_idx in step_maps:
            layer_map = step_maps.get(layer_idx)
        elif layer_key in step_maps:
            layer_map = step_maps.get(layer_key)
        if isinstance(layer_map, torch.Tensor):
            return self._aggregate_character_semantic_responses({layer_idx: layer_map})
        return self._aggregate_character_semantic_responses(step_maps)

    @torch.no_grad()
    def _suppress_other_character_responses(self, agg_maps):
        if not self._use_role_contrast_selection():
            return agg_maps
        if not isinstance(agg_maps, list) or len(agg_maps) < 2:
            return agg_maps
        out = list(agg_maps)
        for idx, primary in enumerate(agg_maps):
            if not isinstance(primary, torch.Tensor):
                continue
            negatives = [x.reshape(-1).to(device="cpu", dtype=torch.float32) for j, x in enumerate(agg_maps) if j != idx and isinstance(x, torch.Tensor)]
            if len(negatives) <= 0:
                continue
            primary_flat = primary.reshape(-1).to(device="cpu", dtype=torch.float32)
            min_len = min([int(primary_flat.numel())] + [int(x.numel()) for x in negatives])
            if min_len <= 0:
                continue
            neg_stack = torch.stack([x[:min_len] for x in negatives], dim=0)
            negative_flat = neg_stack.max(dim=0).values
            out[idx] = (primary_flat[:min_len] - negative_flat).clamp(min=0)
        return out

    @torch.no_grad()
    def _build_query_boxes_from_payloads(self, payloads_by_role_or_layer, h_patch, w_patch):
        role_to_indices = defaultdict(list)
        if _is_layerwise_container(payloads_by_role_or_layer):
            for _, layer_payloads in _iter_layerwise_items(payloads_by_role_or_layer):
                if not isinstance(layer_payloads, dict):
                    continue
                for role_id, payload in layer_payloads.items():
                    if not isinstance(payload, dict):
                        continue
                    flat_idx = payload.get("flat_idx", None)
                    if isinstance(flat_idx, torch.Tensor):
                        role_to_indices[str(role_id)].extend([int(x) for x in flat_idx.detach().cpu().reshape(-1).tolist()])
        elif isinstance(payloads_by_role_or_layer, dict):
            for role_id, payload in payloads_by_role_or_layer.items():
                if not isinstance(payload, dict):
                    continue
                flat_idx = payload.get("flat_idx", None)
                if isinstance(flat_idx, torch.Tensor):
                    role_to_indices[str(role_id)].extend([int(x) for x in flat_idx.detach().cpu().reshape(-1).tolist()])
        query_role_boxes = {}
        for role_id, indices in role_to_indices.items():
            unique_idx = sorted(set([int(x) for x in indices if int(x) >= 0]))
            if len(unique_idx) <= 0:
                continue
            boxes = self._build_query_boxes_from_selected_indices(unique_idx, h_patch, w_patch)
            if len(boxes) > 0:
                query_role_boxes[str(role_id)] = boxes
        return query_role_boxes

    @torch.no_grad()
    def _expand_batch_kwargs_for_cfg(self, kwargs, base_batch):
        out = {}
        if not isinstance(kwargs, dict):
            return out
        base_batch = int(base_batch)
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0 and int(value.shape[0]) == base_batch:
                out[key] = torch.cat([value, value], dim=0)
            else:
                out[key] = value
        return out

    @torch.no_grad()
    def _repeat_batch_kwargs(self, kwargs, repeat_factor, base_batch):
        out = {}
        if not isinstance(kwargs, dict):
            return out
        repeat_factor = max(1, int(repeat_factor))
        base_batch = int(base_batch)
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0 and int(value.shape[0]) == base_batch:
                out[key] = torch.cat([value] * repeat_factor, dim=0)
            else:
                out[key] = value
        return out

    @torch.no_grad()
    def _aggregate_character_semantic_responses(self, step_maps):
        return _aggregate_character_semantic_responses_cpu(step_maps)

    @torch.no_grad()
    def _suppress_single_other_character_response(self, primary_map, negative_map):
        return _suppress_other_character_response_cpu(primary_map, negative_map)

    @torch.no_grad()
    def _select_character_sensitive_token_indices(self, agg_map, h_patch, w_patch):
        if not isinstance(agg_map, torch.Tensor) or agg_map.numel() <= 0:
            return []
        h_patch = int(h_patch)
        w_patch = int(w_patch)
        if h_patch <= 0 or w_patch <= 0:
            return []
        total_spatial = max(1, h_patch * w_patch)
        num_frames = max(1, int(agg_map.shape[0]) // total_spatial)
        _, _, selected_indices, _ = process_attention_map_to_mask_v2(
            agg_map,
            threshold=float(getattr(self.args, "top_visual_tokens", -1)),
            top_k_per_head=int(getattr(self.args, "top_visual_tokens_per_head", 0)),
            spatial_shape=(h_patch, w_patch),
            num_frames=num_frames,
            otsu_scope=str(getattr(self.args, "otsu_scope", "frame")),
            neighbor_filter_kernel=int(getattr(self.args, "neighbor_filter_kernel", 3)),
            neighbor_filter_any_window=bool(getattr(self.args, "neighbor_filter_any_window", True)),
        )
        if isinstance(selected_indices, set):
            selected_indices = list(selected_indices)
        elif not isinstance(selected_indices, (list, tuple)):
            selected_indices = []
        return sorted(list(set([int(x) for x in selected_indices if int(x) >= 0])))

    @torch.no_grad()
    def _collect_character_semantic_probe_role_ids(self, collect_chars, memory_bank_token_meta, memory_bank_percents, timestep):
        self._sync_probe_dit_for_timestep(timestep)
        probe_role_ids = []
        def _append_role_ids(meta_for_step):
            if isinstance(meta_for_step, list):
                for item in meta_for_step:
                    if isinstance(item, dict):
                        rid = str(item.get("char_id", "")).strip()
                        if rid:
                            probe_role_ids.append(rid)

        if isinstance(memory_bank_token_meta, dict) and len(memory_bank_token_meta) > 0:
            if self._use_legacy_multi_memory_banks():
                num_train_timesteps = float(max(int(getattr(self.pipe.scheduler, "num_train_timesteps", 1000)), 1))
                p_step = float((timestep.detach().float() / num_train_timesteps).clamp(0.0, 1.0).mean().item())
                candidate_percents_step = memory_bank_percents if isinstance(memory_bank_percents, (list, tuple)) and len(memory_bank_percents) > 0 else self.memory_bank_percents
                selected_bank_idx_step = pick_nearest_bank_by_percent(p_step, candidate_percents_step)
                if _is_layerwise_container(memory_bank_token_meta):
                    for _, layer_meta_map in _iter_layerwise_items(memory_bank_token_meta):
                        if isinstance(layer_meta_map, dict):
                            _append_role_ids(self._get_bank_map_value(layer_meta_map, selected_bank_idx_step, None))
                else:
                    _append_role_ids(self._get_bank_map_value(memory_bank_token_meta, selected_bank_idx_step, None))
            elif _is_layerwise_container(memory_bank_token_meta):
                for _, layer_meta_map in _iter_layerwise_items(memory_bank_token_meta):
                    if isinstance(layer_meta_map, dict):
                        bank_key = self._select_single_bank_key(layer_meta_map, require_tensor=False)
                        if bank_key is not None:
                            _append_role_ids(self._get_bank_map_value(layer_meta_map, bank_key, None))
            else:
                bank_key = self._select_single_bank_key(memory_bank_token_meta, require_tensor=False)
                if bank_key is not None:
                    _append_role_ids(self._get_bank_map_value(memory_bank_token_meta, bank_key, None))
        if isinstance(collect_chars, list) and len(collect_chars) > 0:
            probe_role_ids.extend([str(x) for x in collect_chars])
        return sorted(list(set(probe_role_ids)))

    @torch.no_grad()
    def _get_negative_character_semantic_responses(self, role_to_maps, role_id):
        if not self._use_two_role_difference_selection():
            return None
        if not isinstance(role_to_maps, dict):
            return None
        valid_roles = [str(k) for k, v in role_to_maps.items() if isinstance(v, dict) and len(v) > 0]
        valid_roles = sorted(list(set(valid_roles)))
        if len(valid_roles) != 2:
            return None
        role_id = str(role_id)
        if role_id not in valid_roles:
            return None
        other_role = valid_roles[1] if valid_roles[0] == role_id else valid_roles[0]
        return role_to_maps.get(other_role, None)

    @torch.no_grad()
    def _run_character_semantic_probe(self, noisy_latents, timestep, prompt, role_ids, cond_context, uncond_context, image_emb_for_denoising, extra_input=None):
        self._last_teacher_forced_semantic_prepass_count = 0
        # Multi-step inference normally captures these features during the
        # preceding step. A one-step teacher-forced probe has no preceding
        # step, so it also uses this explicit prepass.
        del uncond_context
        probe_char_configs, probe_ordered_roles = self._prepare_character_semantic_probe_configs(prompt=prompt, role_ids=role_ids)
        if len(probe_char_configs) <= 0:
            return [], [], None
        dit_model = self._get_runtime_dit_for_timestep(timestep)
        probe_pipe = self._make_probe_pipe(dit_model)
        use_feature_tokens = str(getattr(self, "sparse_role_memory_feature_source", "attn_out")).strip().lower() in ("attn_out", "self_attn_out")
        base_batch = int(noisy_latents.shape[0])
        context_input, ablation_branch_by_role = _build_parallel_character_probe_contexts(
            positive_context=cond_context,
            char_configs=probe_char_configs,
            ordered_roles=probe_ordered_roles,
        )
        if context_input is None or len(ablation_branch_by_role) == 0:
            return [], [], None
        repeat_factor = int(context_input.shape[0] // max(base_batch, 1))
        noisy_input = torch.cat([noisy_latents] * repeat_factor, dim=0)
        image_cond_kwargs = self._repeat_batch_kwargs(image_emb_for_denoising if isinstance(image_emb_for_denoising, dict) else {}, repeat_factor=repeat_factor, base_batch=base_batch)
        extra_forward_kwargs = self._repeat_batch_kwargs(extra_input if isinstance(extra_input, dict) else {}, repeat_factor=repeat_factor, base_batch=base_batch)
        map_layer_idx = int(self.sparse_role_memory_layer_idx)
        feature_layer_indices = sorted(
            set(
                int(x) for x in getattr(self, "sparse_role_memory_injection_layers", [map_layer_idx])
                if int(x) >= 0
            )
        )
        if len(feature_layer_indices) <= 0:
            feature_layer_indices = [map_layer_idx]
        probe_layer_indices = sorted(set([map_layer_idx] + feature_layer_indices))
        stop_layer_idx = max(probe_layer_indices)
        probe_extractor = MultiCharacterAttentionMapExtractor(probe_pipe, probe_layer_indices, probe_char_configs, cfg_scale=1.0)
        probe_extractor.register_hooks()
        feature_taps = []
        if use_feature_tokens:
            # Match the normal inference path: capture query features only at
            # sparse_role_memory_layer_idx, so the query payload stays shared
            # (multi-step inference taps a single layer, not all injection layers).
            for feature_layer_idx in (map_layer_idx,):
                feature_tap = AttentionOutputFeatureTap(
                    dit_model=dit_model,
                    layer_idx=int(feature_layer_idx),
                    keep_device="cpu",
                    keep_dtype=torch.bfloat16,
                    source=str(getattr(self, "sparse_role_memory_feature_source", "attn_out")),
                    batch_index=0,
                )
                feature_tap.register()
                feature_taps.append((int(feature_layer_idx), feature_tap))
        forward_stopper = ForwardStopAfterLayer(dit_model, int(stop_layer_idx))
        forward_stopper.register()
        try:
            try:
                self._last_teacher_forced_semantic_prepass_count = 1
                run_native_dit_forward(
                    dit_model,
                    x=noisy_input,
                    timestep=timestep,
                    context=context_input,
                    **image_cond_kwargs,
                    **extra_forward_kwargs,
                )
            except _StopForwardAfterLayer:
                pass
            per_char_step_maps = probe_extractor.get_attention_maps_per_character()
            per_char_step_maps = _convert_parallel_probe_responses_to_role_diffs(per_char_step_maps=per_char_step_maps, ordered_roles=probe_ordered_roles, ablation_branch_by_role=ablation_branch_by_role)
        finally:
            forward_stopper.remove()
            captured_by_layer = {}
            for feature_layer_idx, feature_tap in feature_taps:
                captured_tokens = feature_tap.pop_tokens()
                feature_tap.remove()
                if isinstance(captured_tokens, torch.Tensor) and captured_tokens.dim() == 2:
                    captured_by_layer[_layer_key(feature_layer_idx)] = captured_tokens
            if len(captured_by_layer) > 1:
                captured_layer_tokens = _make_layerwise_container(captured_by_layer)
            elif len(captured_by_layer) == 1:
                captured_layer_tokens = next(iter(captured_by_layer.values()))
            else:
                captured_layer_tokens = None
            probe_extractor.remove_hooks()
        return per_char_step_maps, probe_ordered_roles, captured_layer_tokens

    @torch.no_grad()
    def _prepare_teacher_forced_query_payload(
        self,
        *,
        noisy_latents,
        timestep,
        prompt,
        role_ids,
        cond_context,
        uncond_context,
        image_emb_for_denoising,
        extra_input=None,
    ):
        per_char_step_maps, ordered_roles, layer_tokens = self._run_character_semantic_probe(
            noisy_latents=noisy_latents,
            timestep=timestep,
            prompt=prompt,
            role_ids=role_ids,
            cond_context=cond_context,
            uncond_context=uncond_context,
            image_emb_for_denoising=image_emb_for_denoising,
            extra_input=extra_input,
        )
        self._last_teacher_forced_semantic_maps = {
            str(role): maps
            for role, maps in zip(ordered_roles, per_char_step_maps)
            if isinstance(maps, dict)
        }
        _, _, _, h_lat, w_lat = noisy_latents.shape
        patch_size = self.pipe.dit.patch_size
        return self._build_character_mask_payload_from_probe(
            per_char_step_maps=per_char_step_maps,
            ordered_roles=ordered_roles,
            h_patch=h_lat // patch_size[1],
            w_patch=w_lat // patch_size[2],
            layer_tokens=layer_tokens,
        )

    @torch.no_grad()
    def _override_query_indices(self, payload, overrides, num_tokens):
        """Replace only role ``flat_idx`` fields, preserving captured features."""
        if overrides is None:
            return payload
        if not isinstance(overrides, dict):
            raise TypeError("query_indices_by_role must be a dict")
        if _is_layerwise_container(payload):
            return _make_layerwise_container({
                layer: self._override_query_indices(layer_payload, overrides, num_tokens)
                for layer, layer_payload in _iter_layerwise_items(payload)
            })
        output = {
            str(role): dict(role_payload)
            for role, role_payload in (payload.items() if isinstance(payload, dict) else [])
            if isinstance(role_payload, dict)
        }
        for role, indices in overrides.items():
            selected = sorted({int(index) for index in indices})
            if any(index < 0 or index >= int(num_tokens) for index in selected):
                raise ValueError("query index outside current video token grid")
            role_payload = dict(output.get(str(role), {}))
            role_payload["flat_idx"] = torch.tensor(selected, dtype=torch.long, device="cpu")
            output[str(role)] = role_payload
        return output

    @staticmethod
    def _zero_context_positions(context, indices):
        return _zero_context_positions(context, indices)

    @torch.no_grad()
    def _build_character_mask_payload_from_probe(self, per_char_step_maps, ordered_roles, h_patch, w_patch, layer_tokens=None):
        query_feature_payload = {}
        query_feature_payload_by_layer = {}
        if not isinstance(per_char_step_maps, list) or not isinstance(ordered_roles, list):
            return None, None
        layer_tokens_by_layer = dict(_iter_layerwise_items(layer_tokens)) if _is_layerwise_container(layer_tokens) else None
        use_n = min(len(per_char_step_maps), len(ordered_roles))

        if isinstance(layer_tokens_by_layer, dict) and len(layer_tokens_by_layer) > 0:
            for layer, layer_token_tensor in layer_tokens_by_layer.items():
                layer_payloads = {}
                layer_agg_maps = [
                    self._aggregate_character_semantic_responses_for_layer(per_char_step_maps[idx], layer)
                    for idx in range(use_n)
                ]
                layer_agg_maps = self._suppress_other_character_responses(layer_agg_maps)
                for char_idx in range(use_n):
                    role_id = str(ordered_roles[char_idx])
                    agg_map = layer_agg_maps[char_idx]
                    if not isinstance(agg_map, torch.Tensor):
                        continue
                    sel_idx = self._select_character_sensitive_token_indices(agg_map, h_patch, w_patch)
                    if len(sel_idx) <= 0:
                        continue
                    idx_cpu = torch.tensor(sel_idx, dtype=torch.long, device="cpu")
                    layer_payload = {"flat_idx": idx_cpu}
                    if isinstance(layer_token_tensor, torch.Tensor) and layer_token_tensor.dim() == 2:
                        valid = idx_cpu < int(layer_token_tensor.shape[0])
                        if bool(valid.any()):
                            layer_idx_cpu = idx_cpu[valid]
                            layer_payload["flat_idx"] = layer_idx_cpu
                            layer_payload["feature"] = layer_token_tensor.index_select(
                                0,
                                layer_idx_cpu.to(device=layer_token_tensor.device),
                            ).to(device="cpu", dtype=layer_token_tensor.dtype)
                    layer_payloads[role_id] = layer_payload
                if len(layer_payloads) > 0:
                    query_feature_payload_by_layer[_layer_key(layer)] = layer_payloads
            if len(query_feature_payload_by_layer) > 0:
                layerwise_payload = _make_layerwise_container(query_feature_payload_by_layer)
                query_role_boxes = self._build_query_boxes_from_payloads(layerwise_payload, h_patch, w_patch)
                if bool(getattr(self.args, "debug_sparse_role_memory_attn", False)):
                    debug_counts = {}
                    for layer, payloads in _iter_layerwise_items(layerwise_payload):
                        if isinstance(payloads, dict):
                            debug_counts[str(layer)] = {
                                str(role_id): int(payload.get("flat_idx", torch.empty(0)).numel())
                                for role_id, payload in payloads.items()
                                if isinstance(payload, dict)
                            }
                    print(f"[QueryPayloadDebug] mode=layerwise counts={debug_counts}", flush=True)
                return (
                    query_role_boxes if len(query_role_boxes) > 0 else None,
                    layerwise_payload,
                )

        agg_maps = [self._aggregate_character_semantic_responses(per_char_step_maps[idx]) for idx in range(use_n)]
        agg_maps = self._suppress_other_character_responses(agg_maps)
        for char_idx in range(use_n):
            role_id = str(ordered_roles[char_idx])
            agg_map = agg_maps[char_idx]
            if not isinstance(agg_map, torch.Tensor):
                continue
            sel_idx = self._select_character_sensitive_token_indices(agg_map, h_patch, w_patch)
            if len(sel_idx) <= 0:
                continue
            idx_cpu = torch.tensor(sel_idx, dtype=torch.long, device="cpu")
            payload = {"flat_idx": idx_cpu}
            if isinstance(layer_tokens, torch.Tensor) and layer_tokens.dim() == 2:
                valid = idx_cpu < int(layer_tokens.shape[0])
                if bool(valid.any()):
                    idx_cpu = idx_cpu[valid]
                    payload["flat_idx"] = idx_cpu
                    payload["feature"] = layer_tokens.index_select(0, idx_cpu.to(device=layer_tokens.device)).to(device="cpu", dtype=layer_tokens.dtype)
            query_feature_payload[role_id] = payload
        if bool(getattr(self.args, "debug_sparse_role_memory_attn", False)):
            debug_counts = {
                str(role_id): {
                    "tokens": int(payload.get("flat_idx", torch.empty(0)).numel()),
                    "has_feature": bool(isinstance(payload.get("feature", None), torch.Tensor)),
                }
                for role_id, payload in query_feature_payload.items()
                if isinstance(payload, dict)
            }
            print(f"[QueryPayloadDebug] mode=shared counts={debug_counts}", flush=True)
        query_role_boxes = self._build_query_boxes_from_payloads(query_feature_payload, h_patch, w_patch)
        return (query_role_boxes if len(query_role_boxes) > 0 else None, query_feature_payload if len(query_feature_payload) > 0 else None)

    @torch.no_grad()
    def _extract_memory_from_step_maps(self, step_maps, noisy_latents, char_id, negative_step_maps=None, char_latent_boxes=None, return_positions=False, return_token_meta=False, return_selected_indices=False, token_source_override=None):
        agg_map = self._aggregate_character_semantic_responses(step_maps)
        if agg_map is None:
            return None
        if self._use_two_role_difference_selection():
            agg_map = self._suppress_single_other_character_response(agg_map, self._aggregate_character_semantic_responses(negative_step_maps))

        _, _, F_lat, H_lat, W_lat = noisy_latents.shape
        dit_model = self.pipe.active_denoising_model() if hasattr(self.pipe, "active_denoising_model") else self.pipe.denoising_model()
        patch_size = dit_model.patch_size
        H_patch = H_lat // patch_size[1]
        W_patch = W_lat // patch_size[2]
        total_spatial = H_patch * W_patch
        F_actual = max(1, int(agg_map.shape[0]) // max(1, total_spatial))
        selected_indices = self._select_character_sensitive_token_indices(agg_map, H_patch, W_patch)
        if len(selected_indices) == 0:
            return None

        token_source = token_source_override
        layer_token_sources = dict(_iter_layerwise_items(token_source)) if _is_layerwise_container(token_source) else None
        if isinstance(layer_token_sources, dict) and len(layer_token_sources) > 0:
            token_source = next((v for v in layer_token_sources.values() if isinstance(v, torch.Tensor) and v.dim() == 2), None)
        if not (isinstance(token_source, torch.Tensor) and token_source.dim() == 2 and int(token_source.shape[0]) > 0):
            if bool(getattr(dit_model, "has_image_input", False)):
                in_ch = getattr(dit_model.patch_embedding, "weight", torch.empty(0, 16)).shape[1]
                if in_ch > int(noisy_latents.shape[1]):
                    pad = torch.zeros(
                        noisy_latents.shape[0], in_ch - int(noisy_latents.shape[1]), noisy_latents.shape[2], noisy_latents.shape[3], noisy_latents.shape[4],
                        device=noisy_latents.device, dtype=noisy_latents.dtype,
                    )
                    x_for_patch = torch.cat([noisy_latents, pad], dim=1)
                else:
                    x_for_patch = noisy_latents
            else:
                x_for_patch = noisy_latents
            patched, _ = dit_model.patchify(x_for_patch)
            token_source = patched.squeeze(0)

        max_valid = int(token_source.shape[0]) - 1
        valid_indices = sorted([idx for idx in selected_indices if idx <= max_valid])
        if len(valid_indices) == 0:
            return None

        max_tokens = int(getattr(self.args, "max_memory_tokens_per_character", 0))
        if max_tokens > 0 and len(valid_indices) > max_tokens:
            if bool(getattr(self.args, "use_attn_score_selection", False)):
                scores = [(idx, float(agg_map[idx].item())) for idx in valid_indices if idx < int(agg_map.shape[0])]
                scores.sort(key=lambda x: x[1], reverse=True)
                valid_indices = [x[0] for x in scores[:max_tokens]]
            else:
                rand_idx = torch.randperm(len(valid_indices))[:max_tokens].tolist()
                valid_indices = [valid_indices[i] for i in rand_idx]
            valid_indices = sorted(valid_indices)

        selected_idx_tensor_cpu = torch.tensor(valid_indices, dtype=torch.long, device="cpu")
        memory_tokens = token_source[valid_indices].detach().cpu()
        if isinstance(layer_token_sources, dict) and len(layer_token_sources) > 0:
            layer_memory_tokens = {}
            for layer, layer_source in layer_token_sources.items():
                if not (isinstance(layer_source, torch.Tensor) and layer_source.dim() == 2):
                    continue
                layer_valid_indices = [idx for idx in valid_indices if idx < int(layer_source.shape[0])]
                if len(layer_valid_indices) <= 0:
                    continue
                layer_memory_tokens[_layer_key(layer)] = layer_source[layer_valid_indices].detach().cpu()
            if len(layer_memory_tokens) > 0:
                memory_tokens = _make_layerwise_container(layer_memory_tokens)

        token_meta = None
        if return_token_meta:
            token_meta = []
            for idx in valid_indices:
                frame_idx = idx // total_spatial
                spatial_idx = idx % total_spatial
                h_idx = spatial_idx // W_patch
                w_idx = spatial_idx % W_patch
                bbox_latent_xyxy = None
                rel_l = rel_r = rel_t = rel_b = -1.0
                inside_box = False
                u = v = tau_local = 0.0
                if isinstance(char_latent_boxes, dict):
                    bbox = char_latent_boxes.get(int(frame_idx), None)
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        try:
                            x1, y1, x2, y2 = [float(x) for x in bbox]
                            bw = max(x2 - x1, 1e-6)
                            bh = max(y2 - y1, 1e-6)
                            xc = float(w_idx) + 0.5
                            yc = float(h_idx) + 0.5
                            cx = 0.5 * (x1 + x2)
                            cy = 0.5 * (y1 + y2)
                            u = (xc - cx) / (0.5 * bw + 1e-6)
                            v = (yc - cy) / (0.5 * bh + 1e-6)
                            inside_box = bool((xc >= x1) and (xc <= x2) and (yc >= y1) and (yc <= y2))
                            rel_l = (xc - x1) / (bw + 1e-6)
                            rel_r = (x2 - xc) / (bw + 1e-6)
                            rel_t = (yc - y1) / (bh + 1e-6)
                            rel_b = (y2 - yc) / (bh + 1e-6)
                            tau_local = float(math.sqrt(max(u * u + v * v, 0.0)))
                            bbox_latent_xyxy = [x1, y1, x2, y2]
                        except Exception:
                            bbox_latent_xyxy = None
                token_meta.append(
                    {
                        "char_id": str(char_id),
                        "latent_t": int(frame_idx),
                        "latent_h": int(h_idx),
                        "latent_w": int(w_idx),
                        "h_patch": int(H_patch),
                        "w_patch": int(W_patch),
                        "source_num_latent_frames": int(F_actual),
                        "bbox_latent_xyxy": bbox_latent_xyxy,
                        "rel_l": float(rel_l),
                        "rel_r": float(rel_r),
                        "rel_t": float(rel_t),
                        "rel_b": float(rel_b),
                        "inside_box": bool(inside_box),
                        "u": float(u),
                        "v": float(v),
                        "tau_local": float(tau_local),
                    }
                )

        if isinstance(memory_tokens, dict) and token_meta is not None:
            token_meta = _make_layerwise_container({layer: list(token_meta) for layer, _ in _iter_layerwise_items(memory_tokens)})

        if not return_positions:
            if return_token_meta:
                if return_selected_indices:
                    return memory_tokens, token_meta, selected_idx_tensor_cpu
                return memory_tokens, token_meta
            if return_selected_indices:
                return memory_tokens, selected_idx_tensor_cpu
            return memory_tokens

        selected_positions = []
        for idx in valid_indices:
            frame_idx = idx // total_spatial
            spatial_idx = idx % total_spatial
            h_idx = spatial_idx // W_patch
            w_idx = spatial_idx % W_patch
            if frame_idx < F_actual:
                selected_positions.append((frame_idx, h_idx, w_idx))
        selected_positions = torch.tensor(selected_positions, dtype=torch.long) if selected_positions else torch.zeros((0, 3), dtype=torch.long)

        if return_token_meta:
            if return_selected_indices:
                return memory_tokens, selected_positions, selected_positions.clone(), (H_patch, W_patch), token_meta, selected_idx_tensor_cpu
            return memory_tokens, selected_positions, selected_positions.clone(), (H_patch, W_patch), token_meta
        if return_selected_indices:
            return memory_tokens, selected_positions, selected_positions.clone(), (H_patch, W_patch), selected_idx_tensor_cpu
        return memory_tokens, selected_positions, selected_positions.clone(), (H_patch, W_patch)

    @torch.no_grad()
    def _extract_memory_from_current_step(self, noisy_latents, timestep, prompt, char_id, cond_context, uncond_context, image_emb_for_denoising, char_latent_boxes=None, return_positions=False, return_token_meta=False, return_selected_indices=False):
        character_name = char_id.replace("_", " ")
        if character_name.lower() not in prompt.lower() and char_id.lower() not in prompt.lower():
            return None
        full_token_ids, full_token_texts, _ = verify_target_text_is_single_token(self.pipe, character_name)
        if not full_token_ids:
            return None
        full_indices = find_token_index_in_prompt(self.pipe, prompt, character_name, full_token_ids, full_token_texts)
        if not full_indices:
            return None
        parts = char_id.split("_", 1)
        prefix_text = parts[0]
        suffix_text = parts[1].replace("_", " ") if len(parts) > 1 else None
        prefix_token_ids, _, _ = verify_target_text_is_single_token(self.pipe, prefix_text)
        num_prefix = len(prefix_token_ids) if prefix_token_ids else 0
        prefix_indices = full_indices[:num_prefix] if num_prefix > 0 else []
        suffix_indices = full_indices[num_prefix:] if num_prefix < len(full_indices) else []

        dit_model = self._get_runtime_dit_for_timestep(timestep)
        probe_pipe = self._make_probe_pipe(dit_model)
        extractor = AttentionMapExtractorV8(
            probe_pipe,
            self.args.extract_layers,
            target_token_indices=prefix_indices,
            suffix_token_indices=suffix_indices,
            suffix_scale=self.args.suffix_attention_scale,
            cfg_scale=self.cfg_scale_extract,
            token_weight=self.args.token_weight,
        )
        use_feature_tokens = str(getattr(self, "sparse_role_memory_feature_source", "attn_out")).strip().lower() in ("attn_out", "self_attn_out")
        feature_layer_idx = int(getattr(self, "sparse_role_memory_layer_idx", 7))
        feature_layer_indices = list(getattr(self, "jigsaw_extra_encoder_layers", [feature_layer_idx])) if bool(getattr(self, "jigsaw_extra_encoder_enabled", False)) else [feature_layer_idx]
        feature_keep_dtype = torch.bfloat16

        if self.cfg_scale_extract > 1.0:
            noisy_input = torch.cat([noisy_latents, noisy_latents], dim=0)
            context_input = torch.cat([uncond_context, cond_context], dim=0)
            image_cond_kwargs = self._expand_batch_kwargs_for_cfg(image_emb_for_denoising, int(noisy_latents.shape[0]))
        else:
            noisy_input = noisy_latents
            context_input = cond_context
            image_cond_kwargs = image_emb_for_denoising

        extractor.register_hooks()
        feature_taps = []
        if use_feature_tokens:
            for tap_layer_idx in feature_layer_indices:
                feature_tap = AttentionOutputFeatureTap(
                    dit_model=dit_model,
                    layer_idx=int(tap_layer_idx),
                    keep_device="cpu",
                    keep_dtype=feature_keep_dtype,
                    source=str(getattr(self, "sparse_role_memory_feature_source", "attn_out")),
                )
                feature_tap.register()
                feature_taps.append((int(tap_layer_idx), feature_tap))
        try:
            run_native_dit_forward(dit_model, x=noisy_input, timestep=timestep, context=context_input, **image_cond_kwargs)
        except Exception:
            for _, feature_tap in feature_taps:
                feature_tap.remove()
            extractor.remove_hooks()
            raise
        captured_by_layer = {}
        for tap_layer_idx, feature_tap in feature_taps:
            captured_tokens = feature_tap.pop_tokens()
            feature_tap.remove()
            if isinstance(captured_tokens, torch.Tensor) and captured_tokens.dim() == 2:
                captured_by_layer[_layer_key(tap_layer_idx)] = captured_tokens
        if len(captured_by_layer) > 1 or bool(getattr(self, "jigsaw_extra_encoder_enabled", False)):
            captured_layer_tokens = _make_layerwise_container(captured_by_layer)
        elif len(captured_by_layer) == 1:
            captured_layer_tokens = next(iter(captured_by_layer.values()))
        else:
            captured_layer_tokens = None
        step_maps = extractor.get_attention_maps()
        extractor.remove_hooks()
        if not step_maps:
            return None
        return self._extract_memory_from_step_maps(
            step_maps=step_maps,
            noisy_latents=noisy_latents,
            char_id=char_id,
            negative_step_maps=None,
            char_latent_boxes=char_latent_boxes,
            return_positions=return_positions,
            return_token_meta=return_token_meta,
            return_selected_indices=return_selected_indices,
            token_source_override=captured_layer_tokens,
        )

    def _memory_aware_dit_forward(self, x, t, context, memory_tokens=None, memory_bank_tokens=None, memory_bank_percents=None, memory_bank_token_meta=None, **kwargs):
        kwargs.pop("feature_mapping_recorder", None)
        query_role_boxes = kwargs.pop("query_role_boxes", None)
        query_feature_payload = kwargs.pop("query_feature_payload", None)
        capture_sparse_token_diagnostics = bool(kwargs.pop("capture_sparse_token_diagnostics", False))
        memory_token_lengths_per_character = kwargs.get("memory_token_lengths_per_character", None)

        current_noise_domain = self._set_inference_noise_domain_from_timestep(t)
        dit = self.pipe.get_noise_model(current_noise_domain) if hasattr(self.pipe, "get_noise_model") else self.pipe.denoising_model()
        active_character_wise_cross_attention = self._get_character_wise_cross_attention_for_domain(current_noise_domain)
        active_jigsaw_extra_encoder = self._get_jigsaw_extra_encoder_for_domain(current_noise_domain)
        active_jigsaw_stage2_writer = self._get_memory_writer_for_domain(current_noise_domain)
        self.sparse_role_memory_attn = active_character_wise_cross_attention
        enable_sparse_context_only = bool(
            self.weights_loaded
            and self.enable_sparse_role_memory_attn
            and active_character_wise_cross_attention is not None
        )
        step_force_disable_injection = False
        model_dtype = dit.patch_embedding.weight.dtype
        x = x.to(device=self.device, dtype=model_dtype)
        context = context.to(device=self.device, dtype=model_dtype)
        if ("y" not in kwargs or kwargs.get("y", None) is None) and bool(getattr(dit, "has_image_input", False)):
            expected_in = int(getattr(dit, "in_dim", x.shape[1]))
            raise RuntimeError(
                f"_memory_aware_dit_forward missing y for image-input model: "
                f"noisy_latents_shape={tuple(x.shape)}, noisy_channels={int(x.shape[1])}, "
                f"expected_in_dim={expected_in}, clip_feature_is_none={kwargs.get('clip_feature', None) is None}"
            )
        if "y" in kwargs and isinstance(kwargs["y"], torch.Tensor):
            kwargs["y"] = kwargs["y"].to(device=self.device, dtype=model_dtype)
        if "clip_feature" in kwargs and isinstance(kwargs["clip_feature"], torch.Tensor):
            kwargs["clip_feature"] = kwargs["clip_feature"].to(device=self.device, dtype=model_dtype)

        num_train_timesteps = float(max(int(getattr(self.pipe.scheduler, "num_train_timesteps", 1000)), 1))
        p_cur = float((t.detach().float() / num_train_timesteps).clamp(0.0, 1.0).mean().item())
        self._last_v9_infer_fusion_meta = {
            "p_cur": float(p_cur),
            "p_fusion": float(p_cur),
            "selected_bank_idx": 0,
            "selected_bank_percent": None,
            "outside_range_blocked": 0.0,
            "inject_ratio": 0.0,
            "sim1_mean": 0.0,
            "tau_sim": 0.0,
            "sim_mode": "context_only",
            "relrope_enabled": 0.0,
            "relrope_query_valid_ratio": 0.0,
            "relrope_memory_valid_ratio": 0.0,
            "memory_disabled": False,
            "per_role_mode_enabled": 0.0,
            "per_role_winner_counts": None,
        }
        self._last_jigsaw_stage2_writer_stats = {"enabled": 0.0, "input_slots": 0, "updated_slots": 0}
        def _record_memory_writer_stats(layer_idx, writer_stats):
            stats = dict(writer_stats) if isinstance(writer_stats, dict) else {}
            self._last_jigsaw_stage2_writer_stats = {
                "enabled": float(stats.get("enabled", 0.0)),
                "layer": float(int(layer_idx)),
                "input_slots": float(stats.get("input_slots", 0.0)),
                "updated_slots": float(stats.get("updated_slots", 0.0)),
                "mean_gate": float(stats.get("mean_gate", 0.0)),
                "mean_cos": float(stats.get("mean_cos", 0.0)),
                "clipped_ratio": float(stats.get("clipped_ratio", 0.0)),
            }
            interval = max(1, int(getattr(self.args, "memory_runtime_log_every", 5) or 5))
            self._jigsaw_stage2_writer_print_count += 1
            if self._jigsaw_stage2_writer_print_count % interval == 0 or float(stats.get("updated_slots", 0.0)) > 0:
                print(
                    f"[MemoryWriter][Infer] domain={current_noise_domain} layer={int(layer_idx)} "
                    f"enabled={float(stats.get('enabled', 0.0)):.0f} input_slots={int(stats.get('input_slots', 0) or 0)} "
                    f"updated_slots={int(stats.get('updated_slots', 0) or 0)} "
                    f"mean_gate={float(stats.get('mean_gate', 0.0)):.6f} "
                    f"mean_cos={float(stats.get('mean_cos', 0.0)):.6f} "
                    f"clipped_ratio={float(stats.get('clipped_ratio', 0.0)):.6f}",
                    flush=True,
                )
        selected_bank_idx = 0
        selected_bank_key = "0"
        selected_bank_percent = None
        selected_mem = memory_tokens
        selected_meta = None
        candidate_percents = []
        if isinstance(memory_bank_percents, torch.Tensor):
            candidate_percents = memory_bank_percents.detach().float().cpu().tolist()
        elif isinstance(memory_bank_percents, np.ndarray):
            candidate_percents = memory_bank_percents.astype(np.float32).tolist()
        elif isinstance(memory_bank_percents, (list, tuple)):
            for item in memory_bank_percents:
                try:
                    candidate_percents.append(float(item))
                except Exception:
                    pass
        elif memory_bank_percents is not None:
            candidate_percents = parse_float_csv(memory_bank_percents, default_list=[])
        if len(candidate_percents) <= 0 and self._use_legacy_multi_memory_banks():
            candidate_percents = [float(x) for x in self.memory_bank_percents]
        layerwise_memory_banks = _is_layerwise_container(memory_bank_tokens)
        layerwise_query_payload = _is_layerwise_container(query_feature_payload)
        layerwise_sparse_payload = bool(layerwise_memory_banks or layerwise_query_payload)
        if bool(getattr(self.args, "debug_sparse_role_memory_attn", False)):
            if layerwise_query_payload:
                query_summary = {
                    str(layer): sorted([str(k) for k in payload.keys()]) if isinstance(payload, dict) else []
                    for layer, payload in _iter_layerwise_items(query_feature_payload)
                }
            elif isinstance(query_feature_payload, dict):
                query_summary = {"shared": sorted([str(k) for k in query_feature_payload.keys()])}
            else:
                query_summary = {}
            if layerwise_memory_banks:
                memory_summary = {
                    str(layer): sorted([str(k) for k in bank_map.keys()]) if isinstance(bank_map, dict) else []
                    for layer, bank_map in _iter_layerwise_items(memory_bank_tokens)
                }
            elif isinstance(memory_bank_tokens, dict):
                memory_summary = {"shared": sorted([str(k) for k in memory_bank_tokens.keys()])}
            else:
                memory_summary = {}
            print(
                f"[SparseInputDebug] domain={current_noise_domain} p_cur={p_cur:.6f} "
                f"layerwise_memory={layerwise_memory_banks} layerwise_query={layerwise_query_payload} "
                f"query={query_summary} memory={memory_summary}",
                flush=True,
            )
        if layerwise_memory_banks:
            layer_bank_count = 0
            for _, bank_map in _iter_layerwise_items(memory_bank_tokens):
                if isinstance(bank_map, dict):
                    layer_bank_count = max(layer_bank_count, len(bank_map))
            self._last_v9_infer_fusion_meta["memory_bank_count"] = int(layer_bank_count)
        else:
            self._last_v9_infer_fusion_meta["memory_bank_count"] = len(memory_bank_tokens) if isinstance(memory_bank_tokens, dict) else 0
        self._last_v9_infer_fusion_meta["memory_bank_percents"] = [float(x) for x in candidate_percents] if candidate_percents else None
        self._last_v9_infer_fusion_meta["memory_bank_mode"] = str(getattr(self, "jigsaw_memory_bank_mode", "single"))
        if self._use_legacy_multi_memory_banks():
            if layerwise_memory_banks:
                if str(getattr(self, "memory_bank_selection_mode", "percent")) == "fixed":
                    selected_bank_idx = int(getattr(self, "fixed_memory_bank_idx", 0))
                    selected_bank_percent = float(candidate_percents[selected_bank_idx]) if 0 <= selected_bank_idx < len(candidate_percents) else None
                elif len(candidate_percents) > 0:
                    selected_bank_idx = int(pick_nearest_bank_by_percent(p_cur, candidate_percents))
                    selected_bank_percent = float(candidate_percents[selected_bank_idx])
                selected_bank_key = str(selected_bank_idx)
                selected_mem = None
                selected_meta = None
            elif isinstance(memory_bank_tokens, dict) and len(memory_bank_tokens) > 0:
                if str(getattr(self, "memory_bank_selection_mode", "percent")) == "fixed":
                    selected_bank_idx = int(getattr(self, "fixed_memory_bank_idx", 0))
                    selected_bank_percent = float(candidate_percents[selected_bank_idx]) if 0 <= selected_bank_idx < len(candidate_percents) else None
                elif len(candidate_percents) > 0:
                    selected_bank_idx = int(pick_nearest_bank_by_percent(p_cur, candidate_percents))
                    selected_bank_percent = float(candidate_percents[selected_bank_idx])
                selected_bank_key = str(selected_bank_idx)
                selected_mem = self._get_bank_map_value(memory_bank_tokens, selected_bank_idx, None)
                if isinstance(memory_bank_token_meta, dict):
                    selected_meta = self._get_bank_map_value(memory_bank_token_meta, selected_bank_idx, None)
                if not (isinstance(selected_mem, torch.Tensor) and selected_mem.ndim >= 2 and int(selected_mem.shape[0]) > 0):
                    print(
                        f"[Warning] nearest memory bank {selected_bank_idx} has no valid tokens; disable memory injection this step",
                        flush=True,
                    )
                    if isinstance(memory_tokens, torch.Tensor):
                        selected_mem = memory_tokens[:0]
                    else:
                        selected_mem = torch.empty((0, 0), device=self.device, dtype=model_dtype)
                    selected_meta = []
                    step_force_disable_injection = True
            elif isinstance(memory_bank_token_meta, dict):
                selected_meta = memory_bank_token_meta.get("0", None)
        else:
            selected_bank_percent = float(candidate_percents[0]) if len(candidate_percents) > 0 else None
            if layerwise_memory_banks:
                selected_mem = None
                selected_meta = None
                selected_bank_key = "0"
                for _, layer_bank_map in _iter_layerwise_items(memory_bank_tokens):
                    if isinstance(layer_bank_map, dict):
                        layer_key = self._select_single_bank_key(layer_bank_map, require_tensor=True)
                        if layer_key is not None:
                            selected_bank_key = str(layer_key)
                            selected_bank_idx = self._bank_idx_from_key(selected_bank_key)
                            break
            elif isinstance(memory_bank_tokens, dict) and len(memory_bank_tokens) > 0:
                bank_key = self._select_single_bank_key(memory_bank_tokens, require_tensor=True)
                if bank_key is not None:
                    selected_bank_key = str(bank_key)
                    selected_bank_idx = self._bank_idx_from_key(selected_bank_key)
                    selected_mem = self._get_bank_map_value(memory_bank_tokens, selected_bank_key, None)
                    if isinstance(memory_bank_token_meta, dict):
                        selected_meta = self._get_bank_map_value(memory_bank_token_meta, selected_bank_key, None)
                else:
                    selected_mem = None
                if not (isinstance(selected_mem, torch.Tensor) and selected_mem.ndim >= 2 and int(selected_mem.shape[0]) > 0):
                    print(
                        "[Warning] single memory bank has no valid bank '0' or unique bank; disable memory injection this step",
                        flush=True,
                    )
                    if isinstance(memory_tokens, torch.Tensor):
                        selected_mem = memory_tokens[:0]
                    else:
                        selected_mem = torch.empty((0, 0), device=self.device, dtype=model_dtype)
                    selected_meta = []
                    step_force_disable_injection = True
            elif isinstance(memory_bank_token_meta, dict):
                meta_key = self._select_single_bank_key(memory_bank_token_meta, require_tensor=False)
                if meta_key is not None:
                    selected_bank_key = str(meta_key)
                    selected_bank_idx = self._bank_idx_from_key(selected_bank_key)
                    selected_meta = self._get_bank_map_value(memory_bank_token_meta, selected_bank_key, None)
        self._last_v9_infer_fusion_meta["selected_bank_idx"] = int(selected_bank_idx)
        self._last_v9_infer_fusion_meta["selected_bank_key"] = str(selected_bank_key)
        self._last_v9_infer_fusion_meta["selected_bank_percent"] = selected_bank_percent
        self._last_v9_infer_fusion_meta["p_fusion"] = float(selected_bank_percent) if selected_bank_percent is not None else float(p_cur)

        def _collect_active_probe_roles(payload):
            active = []
            if not isinstance(payload, dict):
                return active
            for role_id, role_payload in payload.items():
                if not isinstance(role_payload, dict):
                    continue
                flat_idx = role_payload.get("flat_idx", None)
                if isinstance(flat_idx, torch.Tensor) and flat_idx.numel() > 0:
                    role_id = str(role_id)
                    if role_id not in active:
                        active.append(role_id)
            return active

        def _prefilter_memory_bank_for_active_roles(tokens, token_meta, token_lengths, active_roles, role_boxes, role_order_hint=None):
            if not isinstance(tokens, torch.Tensor) or tokens.ndim < 2 or int(tokens.shape[0]) <= 0:
                return tokens, token_meta, token_lengths
            if len(active_roles) <= 0:
                return tokens[:0], [], []
            n_mem = int(tokens.shape[0])
            active_roles_set = {str(x) for x in active_roles}
            if isinstance(token_meta, list) and len(token_meta) > 0:
                keep = []
                for i in range(min(n_mem, len(token_meta))):
                    item = token_meta[i] if isinstance(token_meta[i], dict) else None
                    if isinstance(item, dict) and str(item.get("char_id", "")).strip() in active_roles_set:
                        keep.append(i)
                if len(keep) == 0:
                    return tokens[:0], [], []
                keep_idx = torch.tensor(keep, device=tokens.device, dtype=torch.long)
                pruned_tokens = tokens.index_select(0, keep_idx)
                pruned_meta = [token_meta[i] for i in keep]
                role_count = defaultdict(int)
                role_order = []
                for item in pruned_meta:
                    rid = str(item.get("char_id", "")).strip()
                    if rid not in role_count:
                        role_order.append(rid)
                    role_count[rid] += 1
                pruned_lengths = [int(role_count[r]) for r in role_order if int(role_count[r]) > 0]
                return pruned_tokens, pruned_meta, pruned_lengths
            if isinstance(token_lengths, (list, tuple)) and len(token_lengths) > 0:
                if isinstance(role_boxes, dict) and len(role_boxes) > 0:
                    role_ids = sorted([str(k) for k in role_boxes.keys()])
                elif isinstance(role_order_hint, (list, tuple)) and len(role_order_hint) > 0:
                    role_ids = [str(x) for x in role_order_hint]
                elif all(str(r).isdigit() for r in active_roles_set):
                    role_ids = [str(i) for i in range(len(token_lengths))]
                else:
                    return tokens, token_meta, token_lengths
                if len(role_ids) < len(token_lengths):
                    role_ids = role_ids + [str(i) for i in range(len(role_ids), len(token_lengths))]
                keep_idx_list = []
                pruned_lengths = []
                start = 0
                for role_id, seg_len in zip(role_ids, token_lengths):
                    seg_len = max(int(seg_len), 0)
                    end = min(start + seg_len, n_mem)
                    if end > start and str(role_id) in active_roles_set:
                        keep_idx_list.extend(list(range(start, end)))
                        pruned_lengths.append(int(end - start))
                    start = end
                    if start >= n_mem:
                        break
                if len(keep_idx_list) == 0:
                    return tokens[:0], [], []
                keep_idx = torch.tensor(keep_idx_list, device=tokens.device, dtype=torch.long)
                pruned_tokens = tokens.index_select(0, keep_idx)
                pruned_meta = [token_meta[i] for i in keep_idx_list if isinstance(token_meta, list) and i < len(token_meta)] if isinstance(token_meta, list) else None
                return pruned_tokens, pruned_meta, pruned_lengths
            return tokens, token_meta, token_lengths

        if enable_sparse_context_only and not layerwise_sparse_payload:
            active_roles = _collect_active_probe_roles(query_feature_payload)
            selected_mem, selected_meta, memory_token_lengths_per_character = _prefilter_memory_bank_for_active_roles(
                tokens=selected_mem,
                token_meta=selected_meta,
                token_lengths=memory_token_lengths_per_character,
                active_roles=active_roles,
                role_boxes=query_role_boxes,
                role_order_hint=(list(query_feature_payload.keys()) if isinstance(query_feature_payload, dict) else None),
            )
            if (not isinstance(selected_mem, torch.Tensor)) or int(selected_mem.shape[0]) <= 0:
                step_force_disable_injection = True
                if not isinstance(query_feature_payload, dict):
                    query_feature_payload = {}

        if layerwise_memory_banks:
            selected_mem = None
        elif not isinstance(selected_mem, torch.Tensor):
            selected_mem = torch.empty((0, 0), device=self.device, dtype=model_dtype)
            step_force_disable_injection = True
        elif selected_mem.ndim < 2:
            selected_mem = selected_mem.reshape(int(selected_mem.shape[0]), -1)[:0]
            step_force_disable_injection = True
        if (not layerwise_sparse_payload) and int(selected_mem.shape[0]) <= 0:
            self._last_v9_infer_fusion_meta["memory_disabled"] = True

        bsz = int(x.shape[0])
        t_input = t
        if t_input.dim() > 1:
            t_input = t_input.reshape(t_input.shape[0], -1)[:, 0]
        with torch.amp.autocast("cuda", dtype=torch.float32):
            t_embed = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, t_input).float().to(device=self.device))
            t_mod = dit.time_projection(t_embed).unflatten(1, (6, dit.dim))
            memory_encoder_t_embed = t_embed
        if t_mod.dtype != model_dtype:
            t_mod = t_mod.to(dtype=model_dtype)
        memory_projected = None
        if not layerwise_sparse_payload:
            if (
                active_jigsaw_extra_encoder is not None
                and isinstance(selected_mem, torch.Tensor)
                and selected_mem.ndim >= 2
                and int(selected_mem.shape[0]) > 0
                and not self._memory_meta_is_encoded_slots(selected_meta)
            ):
                selected_mem, selected_meta, memory_token_lengths_per_character, encode_stats = _memory_encode_role_tokens_to_slots(
                    active_jigsaw_extra_encoder,
                    selected_mem.to(device=self.device, dtype=model_dtype),
                    selected_meta if isinstance(selected_meta, list) else [],
                    layer_idx=0,
                    t_embed=memory_encoder_t_embed,
                )
                print(
                    f"[MemoryEncoder][Infer] domain={current_noise_domain} layer=0 group={encode_stats.get('group')} "
                    f"in={encode_stats.get('input_tokens')} slots={encode_stats.get('output_slots')} "
                    f"memory_side_rope={not bool(self.jigsaw_disable_memory_side_rope)}",
                    flush=True,
                )
            selected_mem = selected_mem.to(device=self.device, dtype=model_dtype)

        x_with_y = torch.cat([x, kwargs["y"]], dim=1) if isinstance(kwargs.get("y", None), torch.Tensor) else x
        x_patched, (f, h, w) = dit.patchify(x_with_y)
        seq_lens = torch.full((bsz,), int(x_patched.shape[1]), device=self.device, dtype=torch.long)
        grid_sizes = torch.tensor([[f, h, w]] * bsz, device=self.device, dtype=torch.long)

        if not layerwise_sparse_payload:
            if active_jigsaw_stage2_writer is not None and isinstance(selected_mem, torch.Tensor) and selected_mem.ndim >= 2 and int(selected_mem.shape[0]) > 0:
                selected_mem, writer_stats = active_jigsaw_stage2_writer(
                    selected_mem,
                    selected_meta if isinstance(selected_meta, list) else [],
                    query_feature_payload,
                    x_patched,
                )
                _record_memory_writer_stats(0, writer_stats)
            mem_input = selected_mem.unsqueeze(0).expand(bsz, -1, -1)
            if int(mem_input.shape[1]) <= 0:
                # ponytail: nothing to project. selected_mem can legitimately be the
                # (0, 0) fallback -- empty bank, or the Q* probe's forced memory-path
                # arm -- and a 0-wide tensor does not fit the projector's input Linear.
                # sparse_enabled already requires shape[1] > 0, so this stays disabled.
                memory_projected = mem_input
            elif self.use_projector and self.memory_projector is not None:
                target_dtype = self.memory_projector.input_proj.weight.dtype
                mem_input = mem_input.to(dtype=target_dtype)
                x_for_projector = x.to(dtype=target_dtype)
                t_for_projector = t_embed.mean(dim=1).to(dtype=target_dtype)
                memory_projected = self.memory_projector(
                    mem_input,
                    t_for_projector,
                    x_for_projector,
                    condition_mask=None,
                    timestep=t,
                    num_train_timesteps=int(num_train_timesteps),
                ).to(dtype=model_dtype)
            else:
                memory_projected = mem_input

            if self.memory_embeddings is not None and int(memory_projected.shape[1]) > 0:
                memory_projected = self.memory_embeddings(
                    memory_projected,
                    memory_token_lengths_per_character=memory_token_lengths_per_character,
                    add_segment=bool(self._effective_use_segment_embed),
                    add_pos=bool(self._effective_use_learnable_memory_pos),
                )

        freqs = dit.freqs
        if hasattr(dit, "_expand_freqs"):
            freqs = dit._expand_freqs(grid_sizes, self.device)
        elif torch.is_tensor(freqs) and freqs.device != x_patched.device:
            freqs = freqs.to(device=x_patched.device)

        context_emb = dit.text_embedding(context)
        if getattr(dit, "has_image_input", False) and isinstance(kwargs.get("clip_feature", None), torch.Tensor) and getattr(dit, "img_emb", None) is not None and bool(getattr(dit, "require_clip_embedding", True)):
            clip_emb = dit.img_emb(kwargs["clip_feature"].to(device=self.device, dtype=model_dtype))
            context_emb = torch.cat([clip_emb, context_emb], dim=1)
        context_lens = torch.full((bsz,), int(context_emb.shape[1]), device=self.device, dtype=torch.long)

        x_out = x_patched
        sparse_layer_indices = set(int(x) for x in self.sparse_role_memory_injection_layers if int(x) >= 0)
        if len(sparse_layer_indices) <= 0:
            sparse_layer_indices = {int(self.sparse_role_memory_layer_idx)}
        # Sparse role-memory time gate must follow the real denoise percent.
        # Bank percent is only for bank selection and logging.
        sparse_timestep_percent = float(p_cur)
        self._last_sparse_role_memory_stats = {
            "enabled": 0.0,
            "selected_query_tokens": 0,
            "selected_memory_tokens": 0,
            "winner_counts": {},
            "role_head_out_norm": 0.0,
            "plain_head_out_norm": 0.0,
            "attn_entropy": 0.0,
        }
        self._last_sparse_role_memory_stats_by_layer = {}
        def _select_bank_from_map(bank_map, meta_map):
            layer_selected_mem = None
            layer_selected_meta = None
            if isinstance(bank_map, dict):
                if self._use_legacy_multi_memory_banks():
                    layer_selected_mem = self._get_bank_map_value(bank_map, selected_bank_idx, None)
                    if isinstance(meta_map, dict):
                        layer_selected_meta = self._get_bank_map_value(meta_map, selected_bank_idx, None)
                    if layer_selected_mem is None:
                        for fallback_key, fallback_value in bank_map.items():
                            if isinstance(fallback_value, torch.Tensor):
                                layer_selected_mem = fallback_value
                                if isinstance(meta_map, dict):
                                    layer_selected_meta = self._get_bank_map_value(meta_map, fallback_key, None)
                                break
                else:
                    layer_bank_key = self._select_single_bank_key(bank_map, require_tensor=True)
                    if layer_bank_key is not None:
                        layer_selected_mem = self._get_bank_map_value(bank_map, layer_bank_key, None)
                        if isinstance(meta_map, dict):
                            layer_selected_meta = self._get_bank_map_value(meta_map, layer_bank_key, None)
            return layer_selected_mem, layer_selected_meta

        def _project_memory_for_sparse(layer_mem, layer_lengths):
            if not (isinstance(layer_mem, torch.Tensor) and layer_mem.ndim >= 2 and int(layer_mem.shape[0]) > 0):
                return None
            layer_mem = layer_mem.to(device=self.device, dtype=model_dtype)
            layer_mem_input = layer_mem.unsqueeze(0).expand(bsz, -1, -1)
            if self.use_projector and self.memory_projector is not None:
                target_dtype = self.memory_projector.input_proj.weight.dtype
                layer_mem_input = layer_mem_input.to(dtype=target_dtype)
                x_for_projector = x.to(dtype=target_dtype)
                t_for_projector = t_embed.mean(dim=1).to(dtype=target_dtype)
                layer_projected = self.memory_projector(
                    layer_mem_input,
                    t_for_projector,
                    x_for_projector,
                    condition_mask=None,
                    timestep=t,
                    num_train_timesteps=int(num_train_timesteps),
                ).to(dtype=model_dtype)
            else:
                layer_projected = layer_mem_input
            if self.memory_embeddings is not None:
                layer_projected = self.memory_embeddings(
                    layer_projected,
                    memory_token_lengths_per_character=layer_lengths,
                    add_segment=bool(self._effective_use_segment_embed),
                    add_pos=bool(self._effective_use_learnable_memory_pos),
                )
            return layer_projected

        def _query_feature_centroid(layer_query_payload):
            feats = []
            if not isinstance(layer_query_payload, dict):
                return None, 0
            for payload in layer_query_payload.values():
                if not isinstance(payload, dict):
                    continue
                feat = payload.get("feature", None)
                if isinstance(feat, torch.Tensor) and feat.ndim == 2 and int(feat.shape[0]) > 0:
                    feats.append(feat.detach().float().cpu())
            if len(feats) <= 0:
                return None, 0
            q = torch.cat(feats, dim=0)
            return F.normalize(q.mean(dim=0), dim=0, eps=1e-6), int(q.shape[0])

        def _log_bank_alignment(layer_idx, layer_query_payload, layer_bank_map):
            if not bool(getattr(self, "enable_bank_alignment_diagnostics", False)):
                return
            q_centroid, q_count = _query_feature_centroid(layer_query_payload)
            if q_centroid is None or not isinstance(layer_bank_map, dict):
                return
            bank_scores = {}
            for bank_key, bank_tokens in layer_bank_map.items():
                if not (isinstance(bank_tokens, torch.Tensor) and bank_tokens.ndim == 2 and int(bank_tokens.shape[0]) > 0):
                    continue
                mem_centroid = F.normalize(bank_tokens.detach().float().cpu().mean(dim=0), dim=0, eps=1e-6)
                if int(mem_centroid.shape[0]) != int(q_centroid.shape[0]):
                    continue
                bank_scores[str(bank_key)] = {
                    "centroid_cos": float(torch.dot(q_centroid, mem_centroid).item()),
                    "mem_tokens": int(bank_tokens.shape[0]),
                }
            if len(bank_scores) <= 0:
                return
            feature_nearest_bank = max(bank_scores.keys(), key=lambda k: bank_scores[k]["centroid_cos"])
            print(
                "[BankAlign] "
                + json.dumps(
                    {
                        "layer": int(layer_idx),
                        "p_cur": float(p_cur),
                        "selected_bank": int(selected_bank_idx),
                        "selected_bank_percent": selected_bank_percent,
                        "query_feature_count": int(q_count),
                        "feature_nearest_bank": str(feature_nearest_bank),
                        "scores": bank_scores,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        def _prepare_sparse_inputs_for_layer(layer_idx, current_x_output=None):
            layer_query_payload = _select_layerwise_value(
                query_feature_payload,
                layer_idx,
                default={} if _is_layerwise_container(query_feature_payload) else query_feature_payload,
            )
            if _is_layerwise_container(query_feature_payload) and not isinstance(layer_query_payload, dict):
                layer_query_payload = {}
            layer_role_boxes = query_role_boxes
            layer_selected_mem = selected_mem
            layer_selected_meta = selected_meta
            layer_lengths = memory_token_lengths_per_character
            if layerwise_memory_banks:
                layer_bank_map = _select_layerwise_value(memory_bank_tokens, layer_idx, default={})
                layer_meta_map = _select_layerwise_value(memory_bank_token_meta, layer_idx, default={})
                layer_selected_mem, layer_selected_meta = _select_bank_from_map(layer_bank_map, layer_meta_map)
                layer_lengths = None
                _log_bank_alignment(layer_idx, layer_query_payload, layer_bank_map)
            elif isinstance(memory_bank_tokens, dict):
                _log_bank_alignment(layer_idx, layer_query_payload, memory_bank_tokens)
            layer_disabled = bool(step_force_disable_injection)
            if enable_sparse_context_only:
                active_roles = _collect_active_probe_roles(layer_query_payload)
                layer_selected_mem, layer_selected_meta, layer_lengths = _prefilter_memory_bank_for_active_roles(
                    tokens=layer_selected_mem,
                    token_meta=layer_selected_meta,
                    token_lengths=layer_lengths,
                    active_roles=active_roles,
                    role_boxes=layer_role_boxes,
                    role_order_hint=(list(layer_query_payload.keys()) if isinstance(layer_query_payload, dict) else None),
                )
            if (
                active_jigsaw_extra_encoder is not None
                and isinstance(layer_selected_mem, torch.Tensor)
                and layer_selected_mem.ndim >= 2
                and int(layer_selected_mem.shape[0]) > 0
                and not self._memory_meta_is_encoded_slots(layer_selected_meta)
            ):
                layer_selected_mem, layer_selected_meta, layer_lengths, encode_stats = _memory_encode_role_tokens_to_slots(
                    active_jigsaw_extra_encoder,
                    layer_selected_mem.to(device=self.device, dtype=model_dtype),
                    layer_selected_meta if isinstance(layer_selected_meta, list) else [],
                    layer_idx=layer_idx,
                    t_embed=memory_encoder_t_embed,
                )
                if bool(getattr(self.args, "debug_sparse_role_memory_attn", False)):
                    print(
                        f"[MemoryEncoder][Infer] domain={current_noise_domain} layer={layer_idx} group={encode_stats.get('group')} "
                        f"in={encode_stats.get('input_tokens')} slots={encode_stats.get('output_slots')} "
                        f"memory_side_rope={not bool(self.jigsaw_disable_memory_side_rope)}",
                        flush=True,
                    )
                if active_jigsaw_stage2_writer is not None:
                    layer_selected_mem, writer_stats = active_jigsaw_stage2_writer(
                        layer_selected_mem,
                        layer_selected_meta,
                        layer_query_payload,
                        current_x_output,
                    )
                    _record_memory_writer_stats(layer_idx, writer_stats)
            if not (isinstance(layer_query_payload, dict) and len(layer_query_payload) > 0):
                layer_disabled = True
            layer_projected = _project_memory_for_sparse(layer_selected_mem, layer_lengths)
            if layer_projected is None or int(layer_projected.shape[1]) <= 0:
                layer_disabled = True
                layer_projected = None
            return (layer_projected, layer_selected_meta, layer_lengths, layer_query_payload, layer_role_boxes, layer_disabled)

        def _self_attn_for_block(block_module, attn_in, layer_idx):
            return block_module.self_attn(attn_in, seq_lens, grid_sizes, freqs)

        def _forward_block_official(block_module, x_in, e_in, layer_idx):
            with torch.amp.autocast("cuda", dtype=model_dtype):
                if not all(hasattr(block_module, name) for name in ("self_attn", "cross_attn", "ffn", "norm1", "norm2", "norm3", "modulation")):
                    return block_module(
                        x_in,
                        e=e_in,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        freqs=freqs,
                        context=context_emb,
                        context_lens=context_lens,
                    )
                has_seq = len(e_in.shape) == 4
                chunk_dim = 2 if has_seq else 1
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    block_module.modulation.to(dtype=e_in.dtype, device=e_in.device) + e_in
                ).chunk(6, dim=chunk_dim)
                if has_seq:
                    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                        shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                        shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
                    )
                input_x = modulate(block_module.norm1(x_in), shift_msa, scale_msa)
                x_mid = x_in + gate_msa * _self_attn_for_block(block_module, input_x, layer_idx)
                x_mid = x_mid + block_module.cross_attn(block_module.norm3(x_mid), context_emb, context_lens)
                input_x2 = modulate(block_module.norm2(x_mid), shift_mlp, scale_mlp)
                return x_mid + gate_mlp * block_module.ffn(input_x2)

        def _forward_block_sparse(block_module, x_in, e_in, layer_idx):
            with torch.amp.autocast("cuda", dtype=model_dtype):
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    block_module.modulation.to(dtype=e_in.dtype, device=e_in.device) + e_in
                ).chunk(6, dim=1)
                input_x = modulate(block_module.norm1(x_in), shift_msa, scale_msa)
                x_mid = x_in + gate_msa * _self_attn_for_block(block_module, input_x, layer_idx)
                x_mid = x_mid + block_module.cross_attn(block_module.norm3(x_mid), context_emb, context_lens)
                character_attn_stats = {
                    "enabled": 0.0,
                    "selected_query_tokens": 0,
                    "selected_memory_tokens": 0,
                    "winner_counts": {},
                    "role_head_out_norm": 0.0,
                    "plain_head_out_norm": 0.0,
                    "attn_entropy": 0.0,
                }
                layer_memory_projected = memory_projected
                layer_selected_meta = selected_meta
                layer_lengths = memory_token_lengths_per_character
                layer_query_payload = query_feature_payload
                layer_role_boxes = query_role_boxes
                layer_disable_sparse = bool(step_force_disable_injection)
                if layerwise_sparse_payload:
                    (
                        layer_memory_projected,
                        layer_selected_meta,
                        layer_lengths,
                        layer_query_payload,
                        layer_role_boxes,
                        layer_disable_sparse,
                    ) = _prepare_sparse_inputs_for_layer(layer_idx, current_x_output=x_mid)
                if (
                    enable_sparse_context_only
                    and (not layer_disable_sparse)
                    and isinstance(layer_query_payload, dict)
                    and isinstance(layer_memory_projected, torch.Tensor)
                    and int(layer_memory_projected.shape[1]) > 0
                ):
                    x_before_sparse = x_mid
                    x_after_sparse, character_attn_stats = active_character_wise_cross_attention(
                        tokens=x_mid,
                        memory_tokens=layer_memory_projected.to(dtype=x_mid.dtype),
                        query_role_boxes=layer_role_boxes,
                        query_feature_payload=layer_query_payload,
                        memory_bank_token_meta=layer_selected_meta,
                        memory_token_lengths_per_character=layer_lengths,
                        latent_h=int(h),
                        latent_w=int(w),
                        timestep_percent=float(sparse_timestep_percent),
                        sparse_role_memory_query_chunk_size=self.sparse_role_memory_query_chunk_size,
                        capture_token_diagnostics=capture_sparse_token_diagnostics,
                    )
                    layer_scale = float(getattr(self, "sparse_role_memory_layer_scales", {}).get(int(layer_idx), 1.0))
                    layer_scale = max(0.0, layer_scale)
                    total_layer_scale = float(layer_scale)
                    delta_sparse = (x_after_sparse - x_before_sparse) * total_layer_scale
                    x_mid = x_before_sparse + delta_sparse
                    with torch.no_grad():
                        raw_delta_norm = float((x_after_sparse - x_before_sparse).detach().float().norm(dim=-1).mean().item())
                        effective_delta_norm = float((x_mid - x_before_sparse).detach().float().norm(dim=-1).mean().item())
                        # ponytail: the delta alone cannot say whether memory has any authority
                        # over x; the ratio to the host token norm can.
                        host_token_norm = float(x_before_sparse.detach().float().norm(dim=-1).mean().item())
                    character_attn_stats = dict(character_attn_stats) if isinstance(character_attn_stats, dict) else {}
                    character_attn_stats["applied_layer_scale"] = float(total_layer_scale)
                    character_attn_stats["raw_delta_norm"] = raw_delta_norm
                    character_attn_stats["effective_delta_norm"] = effective_delta_norm
                    character_attn_stats["host_token_norm"] = host_token_norm
                    token_diagnostics = character_attn_stats.get("token_diagnostics")
                    if capture_sparse_token_diagnostics and isinstance(token_diagnostics, dict):
                        token_diagnostics = dict(token_diagnostics)
                        raw_features = token_diagnostics.get("raw_delta_features")
                        if isinstance(raw_features, torch.Tensor):
                            effective_features = raw_features.detach().float() * float(total_layer_scale)
                            token_diagnostics["effective_delta_features"] = effective_features.to(dtype=torch.bfloat16)
                            token_diagnostics["effective_delta_norm"] = effective_features.norm(dim=-1)
                        character_attn_stats["token_diagnostics"] = token_diagnostics
                input_x2 = modulate(block_module.norm2(x_mid), shift_mlp, scale_mlp)
                x_out_local = x_mid + gate_mlp * block_module.ffn(input_x2)
            return x_out_local, character_attn_stats

        sparse_enabled = bool(
            enable_sparse_context_only
            and isinstance(query_feature_payload, dict)
            and len(query_feature_payload) > 0
            and (
                layerwise_sparse_payload
                or (isinstance(memory_projected, torch.Tensor) and int(memory_projected.shape[1]) > 0)
            )
        )
        for layer_idx, block in enumerate(dit.blocks):
            is_sparse_layer = (
                sparse_enabled
                and int(layer_idx) in sparse_layer_indices
                and all(hasattr(block, name) for name in ("self_attn", "cross_attn", "ffn", "norm1", "norm2", "norm3", "modulation"))
            )
            if is_sparse_layer:
                x_out, character_attn_stats = _forward_block_sparse(block, x_out, t_mod, layer_idx)
                character_attn_stats = dict(character_attn_stats) if isinstance(character_attn_stats, dict) else {}
                character_attn_stats["layer_idx"] = int(layer_idx)
                self._last_sparse_role_memory_stats = character_attn_stats
                self._last_sparse_role_memory_stats_by_layer[str(int(layer_idx))] = character_attn_stats
            else:
                x_out = _forward_block_official(block, x_out, t_mod, layer_idx)

        with torch.amp.autocast("cuda", dtype=model_dtype):
            x_out = dit.head(x_out, t_embed.float())
        return dit.unpatchify(x_out, (f, h, w))

def _populate_slotmem_internal_aliases(args):
    for public_name, internal_name in (
        ("slotmem_memory_bank_mode", "jigsaw_memory_bank_mode"),
        ("slotmem_memory_encoder_mode", "jigsaw_extra_encoder_mode"),
        ("slotmem_memory_encoder_layers", "jigsaw_extra_encoder_layers"),
        ("slotmem_memory_encoder_layer_groups", "jigsaw_extra_encoder_layer_groups"),
        ("slotmem_memory_encoder_slots", "jigsaw_extra_encoder_slots"),
        ("slotmem_memory_encoder_dim", "jigsaw_extra_encoder_dim"),
        ("slotmem_memory_encoder_hidden_dim", "jigsaw_extra_encoder_hidden_dim"),
        ("slotmem_memory_encoder_use_t_embed", "jigsaw_extra_encoder_use_t_embed"),
        ("slotmem_memory_encoder_use_slot_index_embed", "jigsaw_extra_encoder_use_slot_index_embed"),
        ("slotmem_memory_encoder_aux_weight", "jigsaw_extra_encoder_aux_weight"),
        ("slotmem_memory_encoder_bg_tokens", "jigsaw_extra_encoder_bg_tokens"),
        ("slotmem_memory_writer_mode", "jigsaw_stage2_writer_mode"),
        ("slotmem_memory_writer_hidden_dim", "jigsaw_stage2_writer_hidden_dim"),
        ("slotmem_memory_writer_init_scale", "jigsaw_stage2_writer_init_scale"),
        ("slotmem_memory_writer_precision_tau", "jigsaw_stage2_writer_precision_tau"),
        ("slotmem_memory_writer_precision_scale", "jigsaw_stage2_writer_precision_scale"),
        ("slotmem_memory_writer_max_delta_ratio", "jigsaw_stage2_writer_max_delta_ratio"),
        ("slotmem_memory_writer_max_delta_norm", "jigsaw_stage2_writer_max_delta_norm"),
        ("slotmem_memory_writer_detach_c_short", "jigsaw_stage2_writer_detach_c_short"),
        ("slotmem_disable_memory_side_rope", "jigsaw_disable_memory_side_rope"),
    ):
        if hasattr(args, public_name):
            setattr(args, internal_name, getattr(args, public_name))
    return args


def parse_args(argv=None):
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="SlotMem Wan2.2 memory inference")
    parser.add_argument("--ckpt_dir", type=str, default="/models/Wan2.2-I2V-A14B")
    parser.add_argument("--train_noise_domain", type=str, default="low_noise", choices=["low_noise", "high_noise"])
    parser.add_argument("--train_stage", type=str, default="stage1", choices=["stage1", "stage2"])
    parser.add_argument("--noise_domain_boundary_ratio", type=float, default=0.9)
    parser.add_argument("--high_expert_checkpoint_path", type=str, default=None)
    parser.add_argument("--low_expert_checkpoint_path", type=str, default=None)
    parser.add_argument("--native_wan_inference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./outputs_slotmem")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=float, default=128.0)
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2")
    parser.add_argument("--init_lora_weights", type=str, default="kaiming")
    parser.add_argument("--projector_bottleneck", type=int, default=256)
    parser.add_argument("--use_projector", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument(
        "--offload_models",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("SLOTMEM_OFFLOAD_MODELS", False),
        help="CPU-offload whole modules between chunks. Defaults to off for the 80GB fast profile; "
        "set SLOTMEM_OFFLOAD_MODELS=1 or pass --offload_models for the low-VRAM profile.",
    )
    parser.add_argument("--defer_lora_until_after_first_chunk", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--dual_expert_load_mode",
        type=str,
        default=os.environ.get("SLOTMEM_DUAL_EXPERT_LOAD_MODE", "active"),
        choices=[
            "standard",
            "vram_management",
            "managed",
            "lazy",
            "low_vram",
            "low-vram",
            "active",
            "sequential",
            "active_expert",
            "off",
        ],
        help="Inference-only loading mode for high/low Wan2.2 experts. Use standard/off to disable DiffSynth VRAM management.",
    )
    parser.add_argument(
        "--dual_expert_offload_dtype",
        type=str,
        default=os.environ.get("SLOTMEM_DUAL_EXPERT_OFFLOAD_DTYPE", "bfloat16"),
        help="CPU offload dtype used when --dual_expert_load_mode enables VRAM management.",
    )
    parser.add_argument(
        "--dual_expert_vram_limit",
        type=float,
        default=_env_float("SLOTMEM_DUAL_EXPERT_VRAM_LIMIT", -1.0),
        help="DiffSynth VRAM limit in GiB for managed modules; negative leaves it unset.",
    )
    parser.add_argument(
        "--dual_expert_manage_aux_models",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("SLOTMEM_DUAL_EXPERT_MANAGE_AUX_MODELS", False),
        help="Also wrap text encoder and VAE with DiffSynth VRAM management in inference low-VRAM mode.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--sample_solver", type=str, default="flow_euler", choices=["flow_euler", "unipc"])
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument(
        "--target_seed_override",
        type=int,
        default=None,
        help="Override the sampler seed only for the first processed/resumed target chunk.",
    )
    parser.add_argument("--inference_latents_fp32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--cfg_uncond_with_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cfg_scale_extract", type=float, default=5.0)
    parser.add_argument("--cfg_scale_extraction", type=float, default=None)
    parser.add_argument("--strict_weight_loading", action="store_true", default=False)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--context_frames", type=int, default=81)
    parser.add_argument("--negative_prompt", type=str, default="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
    parser.add_argument("--ref_image_path", type=str, default=None)
    parser.add_argument(
        "--fixed_reference_scope",
        type=str,
        default="all_chunks",
        choices=["all_chunks", "source_only"],
        help="Use the initial fixed reference for every chunk or only for source chunk 0.",
    )
    parser.add_argument("--ref_pad_cfg", action="store_true", default=False)
    parser.add_argument("--ref_pad_num", type=int, default=0)
    parser.add_argument("--use_first_aug", action="store_true", default=False)
    parser.add_argument("--num_motion_frames", "--num_motion_frame", dest="num_motion_frames", type=int, default=0)
    parser.add_argument("--num_overlap_frame", type=int, default=5)
    parser.add_argument("--num_motion_latent", type=int, default=None)
    parser.add_argument("--tiled", action="store_true", default=False)
    parser.add_argument("--tile_size", type=int, nargs="+", default=[30, 52])
    parser.add_argument("--tile_stride", type=int, nargs="+", default=[15, 26])
    parser.add_argument("--extract_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--role_token_selection_mode", type=str, default="two_role_diff", choices=["baseline", "two_role_diff", "one_vs_rest", "layer7_single"])
    parser.add_argument("--top_visual_tokens", type=float, default=0.1)
    parser.add_argument("--top_visual_tokens_per_head", type=int, default=0)
    parser.add_argument("--otsu_scope", type=str, default="frame", choices=["clip", "frame"])
    parser.add_argument("--token_weight", type=float, default=1.0)
    parser.add_argument("--suffix_attention_scale", type=float, default=1.0)
    parser.add_argument("--max_memory_tokens_per_character", type=int, default=512)
    parser.add_argument("--use_attn_score_selection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--neighbor_filter_kernel", type=int, default=5)
    parser.add_argument("--neighbor_filter_any_window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory_enable_patch_similarity_filter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--memory_patch_similarity_threshold", type=float, default=0.85)
    parser.add_argument("--slotmem_memory_bank_mode", dest="slotmem_memory_bank_mode", type=str, default="single", choices=["single", "legacy_multi"])
    parser.add_argument(
        "--memory_bank_percents",
        type=str,
        default="0.85,0.60,0.35,0.12",
        help="Legacy-only bank percent list used when --slotmem_memory_bank_mode=legacy_multi.",
    )
    parser.add_argument("--max_memory_characters", type=int, default=2)
    parser.add_argument(
        "--target_character",
        type=str,
        default="",
        help="Keep this character inside the per-chunk memory-read window when it appears.",
    )
    parser.add_argument("--use_first_appearance_memory_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_learnable_memory_pos", action="store_true")
    parser.add_argument("--use_segment_embed", action="store_true")
    parser.add_argument("--allow_injection_outside_bank_range", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_sparse_role_memory_attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sparse_role_memory_layer_idx", type=int, default=3)
    parser.add_argument("--sparse_role_memory_injection_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--memory_layer_binding_mode", type=str, default="layerwise", choices=["layerwise", "shared"])
    parser.add_argument(
        "--memory_bank_selection_mode",
        type=str,
        default="percent",
        choices=["percent", "fixed"],
            help="Legacy-only bank selector used when --slotmem_memory_bank_mode=legacy_multi.",
    )
    parser.add_argument("--fixed_memory_bank_idx", type=int, default=0, help="Legacy-only fixed bank index.")
    parser.add_argument("--enable_bank_alignment_diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--char_attn_noise_scope", type=str, default=None, choices=["high_noise", "low_noise"])
    parser.add_argument("--sparse_role_memory_num_heads", type=int, default=8)
    parser.add_argument("--sparse_role_memory_head_dim", type=int, default=128)
    parser.add_argument("--sparse_role_memory_rope_dim", type=int, default=256)
    parser.add_argument("--sparse_role_memory_use_half_role_heads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sparse_role_memory_feature_source", type=str, default="attn_out", choices=["attn_out", "self_attn_out", "block_out"])
    parser.add_argument("--sparse_role_memory_init_scale", type=float, default=0.1)
    parser.add_argument("--sparse_role_memory_time_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sparse_role_memory_query_chunk_size", type=int, default=128)
    parser.add_argument("--sparse_role_memory_layer_scales", type=str, default="")
    parser.add_argument("--debug_sparse_role_memory_attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--slotmem_memory_encoder_mode", dest="slotmem_memory_encoder_mode", type=str, default="on",
                        choices=["off", "on", "true", "1", "extra", "extra_encoder", "slotmem_memory_encoder", "contrastive_encoder"])
    parser.add_argument("--slotmem_memory_encoder_layers", dest="slotmem_memory_encoder_layers", type=str, default="0-15")
    parser.add_argument("--slotmem_memory_encoder_layer_groups", dest="slotmem_memory_encoder_layer_groups", type=str, default="0-4,5-10,11-15")
    parser.add_argument("--slotmem_memory_encoder_slots", dest="slotmem_memory_encoder_slots", type=int, default=64)
    parser.add_argument("--slotmem_memory_encoder_dim", dest="slotmem_memory_encoder_dim", type=int, default=512)
    parser.add_argument("--slotmem_memory_encoder_hidden_dim", dest="slotmem_memory_encoder_hidden_dim", type=int, default=1024)
    parser.add_argument("--slotmem_memory_encoder_use_t_embed", dest="slotmem_memory_encoder_use_t_embed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slotmem_memory_encoder_use_slot_index_embed", dest="slotmem_memory_encoder_use_slot_index_embed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slotmem_memory_encoder_aux_weight", dest="slotmem_memory_encoder_aux_weight", type=float, default=0.05)
    parser.add_argument("--slotmem_memory_encoder_bg_tokens", dest="slotmem_memory_encoder_bg_tokens", type=int, default=64)
    parser.add_argument("--slotmem_memory_writer_mode", dest="slotmem_memory_writer_mode", type=str, default="auto",
                        choices=["auto", "off", "on", "true", "1", "false", "0", "none", "residual"])
    parser.add_argument("--slotmem_memory_writer_hidden_dim", dest="slotmem_memory_writer_hidden_dim", type=int, default=1024)
    parser.add_argument("--slotmem_memory_writer_init_scale", dest="slotmem_memory_writer_init_scale", type=float, default=0.1)
    parser.add_argument("--slotmem_memory_writer_precision_tau", dest="slotmem_memory_writer_precision_tau", type=float, default=0.3)
    parser.add_argument("--slotmem_memory_writer_precision_scale", dest="slotmem_memory_writer_precision_scale", type=float, default=10.0)
    parser.add_argument("--slotmem_memory_writer_max_delta_ratio", dest="slotmem_memory_writer_max_delta_ratio", type=float, default=0.0)
    parser.add_argument("--slotmem_memory_writer_max_delta_norm", dest="slotmem_memory_writer_max_delta_norm", type=float, default=0.0)
    parser.add_argument("--slotmem_memory_writer_detach_c_short", dest="slotmem_memory_writer_detach_c_short", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slotmem_disable_memory_side_rope", dest="slotmem_disable_memory_side_rope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory_runtime_log_every", type=int, default=1)
    parser.add_argument("--save_memory_viz", action="store_true")
    parser.add_argument("--memory_viz_dir", type=str, default=None)
    parser.add_argument("--save_feature_mapping_viz", action="store_true")
    parser.add_argument("--feature_mapping_viz_dir", type=str, default=None)
    parser.add_argument("--feature_mapping_last_ratio", type=float, default=0.05)
    parser.add_argument("--feature_mapping_num_steps", type=int, default=-1)
    parser.add_argument("--feature_mapping_draw_empty", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_denoise_step_viz", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--denoise_step_viz_dir", type=str, default=None)
    parser.add_argument("--save_denoise_step_edge_viz", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--denoise_step_edge_viz_dir", type=str, default=None)
    parser.add_argument("--denoise_step_list", type=str, default="10,20,30,40,50")
    parser.add_argument("--merge_chunks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merged_output_name", type=str, default="merged_chunks.mp4")
    parser.add_argument("--max_chunks", type=int, default=-1)
    parser.add_argument(
        "--efficiency_metrics_path",
        type=str,
        default=None,
        help="Optional JSON summary path for synchronized per-chunk timing and CUDA peak memory.",
    )
    parser.add_argument(
        "--efficiency_runtime_log",
        type=str,
        default=None,
        help="Optional JSONL path for per-chunk efficiency records.",
    )
    parser.add_argument("--resume_state_path", type=str, default=None, help="Optional torch-saved online-memory resume state to load before chunk processing.")
    parser.add_argument("--save_state_path", type=str, default=None, help="Optional torch path where online-memory resume state is saved after each chunk.")
    parser.add_argument("--start_chunk_idx", type=int, default=-1, help="Optional chunk index to start from; defaults to resume state's next_chunk_idx when resuming.")
    parser.add_argument("--stop_after_chunk_idx", type=int, default=-1, help="Stop after completing this chunk index, after writing runtime logs and resume state.")
    parser.add_argument("--smoke_stop_after_chunk_idx", type=int, default=-1)
    parser.add_argument("--smoke_stop_after_denoise_step", type=int, default=-1)
    parser.add_argument("--smoke_stop_marker_path", type=str, default=None)
    parser.add_argument("--stop_after_first_full_buffer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--full_buffer_stop_mode",
        type=str,
        default="before_generate",
        choices=["before_generate", "after_write"],
        help="Stop after the first measured chunk whose online memory is full before generation or becomes full after memory write.",
    )
    parser.add_argument("--full_buffer_marker_path", type=str, default=None)
    parser.add_argument(
        "--save_only_full_buffer_target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Benchmark-only mode: do not write warmup chunk videos before the first before-generate "
            "full-buffer chunk. Generated frames are still used for conditioning and online memory."
        ),
    )
    args = _populate_slotmem_internal_aliases(parser.parse_args(actual_argv))
    smoke_chunk_idx = int(getattr(args, "smoke_stop_after_chunk_idx", -1))
    smoke_step_idx = int(getattr(args, "smoke_stop_after_denoise_step", -1))
    if smoke_chunk_idx >= 0:
        if smoke_step_idx < 0:
            parser.error("--smoke_stop_after_denoise_step must be >=0 when --smoke_stop_after_chunk_idx is set")
        if smoke_step_idx >= int(getattr(args, "num_inference_steps", 50) or 50):
            parser.error("--smoke_stop_after_denoise_step must be smaller than --num_inference_steps")
        if not isinstance(getattr(args, "smoke_stop_marker_path", None), str) or not args.smoke_stop_marker_path.strip():
            parser.error("--smoke_stop_marker_path is required when --smoke_stop_after_chunk_idx is set")
    if not bool(getattr(args, "native_wan_inference", False)):
        has_high = isinstance(args.high_expert_checkpoint_path, str) and args.high_expert_checkpoint_path.strip() != ""
        has_low = isinstance(args.low_expert_checkpoint_path, str) and args.low_expert_checkpoint_path.strip() != ""
        if not (has_high or has_low):
            parser.error("--high_expert_checkpoint_path/--low_expert_checkpoint_path is required unless --native_wan_inference is enabled")
    if bool(getattr(args, "save_only_full_buffer_target", False)):
        if not bool(getattr(args, "stop_after_first_full_buffer", False)):
            parser.error("--save_only_full_buffer_target requires --stop_after_first_full_buffer")
        if str(getattr(args, "full_buffer_stop_mode", "before_generate")) != "before_generate":
            parser.error("--save_only_full_buffer_target currently requires --full_buffer_stop_mode before_generate")
    if "--num_motion_frames" in actual_argv and "--num_overlap_frame" in actual_argv:
        parser.error("use only one motion-frame interface: --num_overlap_frame with --num_motion_latent, or --num_motion_frames alone")
    if _memory_encoder_enabled(getattr(args, "jigsaw_extra_encoder_mode", "off")):
        wide_layers = _jigsaw_parse_layer_list(getattr(args, "jigsaw_extra_encoder_layers", "0-15"))
        args.extract_layers = wide_layers
        args.sparse_role_memory_injection_layers = ",".join(str(x) for x in wide_layers)
        writer_mode = _memory_writer_effective_mode(
            getattr(args, "train_stage", "stage1"),
            getattr(args, "jigsaw_stage2_writer_mode", "auto"),
        )
        print(
            f"[MemoryEncoder][Args] inference raw layerwise layers={wide_layers}; "
            f"stage={args.train_stage} writer_mode={writer_mode}",
            flush=True,
        )
    else:
        args.extract_layers = [int(x.strip()) for x in str(args.extract_layers).split(",") if str(x).strip()]
    args.num_overlap_frame = max(0, int(getattr(args, "num_overlap_frame", 0) or 0))
    args.num_motion_frames = max(1, int(getattr(args, "num_motion_frames", 1) or 1))
    if args.num_overlap_frame > 0:
        args.num_motion_frames = max(args.num_motion_frames, args.num_overlap_frame)
    return args


def _write_smoke_marker(args, engine, chunk_idx, output_dir):
    marker_path = str(getattr(args, "smoke_stop_marker_path", "") or "").strip()
    if not marker_path:
        return None
    payload = {
        "status": "completed",
        "semantic_stop": "requested chunk completed with smoke denoise-step settings",
        "chunk_idx": int(chunk_idx),
        "requested_denoise_step_idx": int(getattr(args, "smoke_stop_after_denoise_step", -1)),
        "num_inference_steps": int(getattr(args, "num_inference_steps", -1)),
        "max_chunks": int(getattr(args, "max_chunks", -1)),
        "output_dir": str(output_dir),
        "train_stage": str(getattr(args, "train_stage", "")),
        "train_noise_domain": str(getattr(args, "train_noise_domain", "")),
        "loaded_checkpoint_domains": sorted([str(x) for x in getattr(engine, "loaded_checkpoint_domains", set())]),
        "checkpoint_paths": {
            "high_noise": str(getattr(args, "high_expert_checkpoint_path", "") or ""),
            "low_noise": str(getattr(args, "low_expert_checkpoint_path", "") or ""),
        },
        "lrpm_settings": {
            "jigsaw_extra_encoder_enabled": bool(getattr(engine, "jigsaw_extra_encoder_enabled", False)),
            "jigsaw_extra_encoder_layers": list(getattr(engine, "jigsaw_extra_encoder_layers", [])),
            "jigsaw_extra_encoder_layer_groups": list(getattr(engine, "jigsaw_extra_encoder_layer_groups", [])),
            "jigsaw_extra_encoder_slots": int(getattr(engine, "jigsaw_extra_encoder_slots", -1)),
            "enable_sparse_role_memory_attn": bool(getattr(engine, "enable_sparse_role_memory_attn", False)),
            "sparse_role_memory_injection_layers": list(getattr(engine, "sparse_role_memory_injection_layers", [])),
            "memory_layer_binding_mode": str(getattr(engine, "memory_layer_binding_mode", "")),
            "jigsaw_stage2_writer_mode": str(getattr(engine, "jigsaw_stage2_writer_mode", "")),
            "memory_writer_effective_mode": str(getattr(engine, "memory_writer_effective_mode", "")),
            "memory_writer_enabled": bool(getattr(engine, "memory_writer_enabled", False)),
        },
        "runtime_stats": {
            "last_sparse_role_memory_stats": getattr(engine, "_last_sparse_role_memory_stats", {}),
            "last_sparse_role_memory_stats_by_layer": getattr(engine, "_last_sparse_role_memory_stats_by_layer", {}),
            "last_jigsaw_stage2_writer_stats": getattr(engine, "_last_jigsaw_stage2_writer_stats", {}),
            "runtime_chunk_warnings": list(getattr(engine, "runtime_chunk_warnings", [])),
            "runtime_role_states": list(getattr(engine, "runtime_role_states", [])),
        },
    }
    marker = Path(marker_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[Smoke] Wrote smoke marker: {marker}", flush=True)
    return str(marker)


def _write_full_buffer_marker(args, engine, chunk_idx, output_dir, save_path, stop_status, chunk_record):
    marker_path = str(getattr(args, "full_buffer_marker_path", "") or "").strip()
    if not marker_path:
        marker_path = str(Path(output_dir) / "full_buffer_stop_marker.json")
    mode = str(getattr(args, "full_buffer_stop_mode", "before_generate"))
    semantic_stop = (
        "first_chunk_generated_with_full_buffer"
        if mode == "before_generate"
        else "first_chunk_after_write_filled_buffer"
    )
    payload = {
        "status": "completed",
        "semantic_stop": semantic_stop,
        "full_buffer_stop_mode": mode,
        "chunk_idx": int(chunk_idx),
        "generated_video": str(save_path),
        "output_dir": str(output_dir),
        "efficiency_metrics_path": str(getattr(args, "efficiency_metrics_path", "") or ""),
        "efficiency_runtime_log": str(getattr(args, "efficiency_runtime_log", "") or ""),
        "num_inference_steps": int(getattr(args, "num_inference_steps", -1)),
        "max_chunks": int(getattr(args, "max_chunks", -1)),
        "max_memory_characters": int(getattr(args, "max_memory_characters", 0) or 0),
        "train_stage": str(getattr(args, "train_stage", "")),
        "train_noise_domain": str(getattr(args, "train_noise_domain", "")),
        "stop_status": stop_status,
        "chunk_efficiency_record": chunk_record,
        "loaded_checkpoint_domains": sorted([str(x) for x in getattr(engine, "loaded_checkpoint_domains", set())]),
        "runtime_stats": {
            "last_sparse_role_memory_stats": getattr(engine, "_last_sparse_role_memory_stats", {}),
            "last_sparse_role_memory_stats_by_layer": getattr(engine, "_last_sparse_role_memory_stats_by_layer", {}),
            "last_jigsaw_stage2_writer_stats": getattr(engine, "_last_jigsaw_stage2_writer_stats", {}),
            "runtime_chunk_warnings": list(getattr(engine, "runtime_chunk_warnings", [])),
            "runtime_role_states": list(getattr(engine, "runtime_role_states", [])),
        },
    }
    marker = Path(marker_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[FullBufferStop] Wrote marker: {marker}", flush=True)
    return str(marker)


def _encode_pil_frames_for_state(frames):
    encoded = []
    for frame in list(frames or []):
        if not isinstance(frame, Image.Image):
            continue
        encoded.append(image_png_bytes(frame))
    return encoded


def _decode_pil_frames_from_state(items):
    frames = []
    for payload in list(items or []):
        try:
            frames.append(Image.open(io.BytesIO(payload)).convert("RGB"))
        except Exception:
            continue
    return frames


def _save_resume_state(path, *, mem_manager, prev_frames_pil, efficiency_chunk_records, next_chunk_idx, prior_total_elapsed_s):
    if not path:
        return
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "next_chunk_idx": int(next_chunk_idx),
        "memory_bank": getattr(mem_manager, "memory_bank", {}),
        "memory_meta_bank": getattr(mem_manager, "memory_meta_bank", {}),
        "first_appearance": getattr(mem_manager, "first_appearance", {}),
        "prev_frames_png": _encode_pil_frames_for_state(prev_frames_pil),
        "efficiency_chunk_records": list(efficiency_chunk_records or []),
        "prior_total_elapsed_s": float(prior_total_elapsed_s),
    }
    torch.save(payload, state_path)
    print(f"  [ResumeState] saved next_chunk_idx={int(next_chunk_idx)} path={state_path}", flush=True)


def _load_resume_state(path):
    if not path:
        return None
    state_path = Path(path)
    if not state_path.is_file():
        print(f"[ResumeState] initialize new state at path={state_path}", flush=True)
        return None
    return torch.load(state_path, map_location="cpu")


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    resume_path = Path(str(getattr(args, "resume_state_path", "") or "").strip())
    resume_requested = bool(resume_path.is_file()) or int(getattr(args, "start_chunk_idx", -1)) > 0
    if not getattr(args, "efficiency_runtime_log", None) and getattr(args, "efficiency_metrics_path", None):
        args.efficiency_runtime_log = str(Path(args.efficiency_metrics_path).with_suffix(".jsonl"))
    if getattr(args, "efficiency_runtime_log", None):
        runtime_log = Path(args.efficiency_runtime_log)
        runtime_log.parent.mkdir(parents=True, exist_ok=True)
        if not resume_requested:
            runtime_log.write_text("", encoding="utf-8")

    if not resume_requested:
        for stale_chunk in sorted(Path(args.output_path).glob("chunk_*.mp4")):
            with suppress(Exception):
                stale_chunk.unlink()
        for stale_meta in sorted(Path(args.output_path).glob("chunk_*.metadata.json")):
            with suppress(Exception):
                stale_meta.unlink()
    merged_output_path = Path(args.output_path) / str(args.merged_output_name)
    if (not resume_requested) and merged_output_path.exists():
        with suppress(Exception):
            merged_output_path.unlink()
    bench_manifest_path = Path(args.output_path) / "slotmem_inference_metadata.json"
    if (not resume_requested) and bench_manifest_path.exists():
        with suppress(Exception):
            bench_manifest_path.unlink()

    args_dump_path = os.path.join(args.output_path, "inference_args.yaml")
    try:
        payload = {"args": vars(args), "argv": sys.argv}
        with open(args_dump_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"[Init] Saved inference args to: {args_dump_path}")
    except Exception as e:
        print(f"[Init] Warning: failed to save inference args: {e}")

    engine = SlotMemInferenceEngine(args)
    mem_manager = RoleWiseSlotMemoryBank()
    resume_state = _load_resume_state(getattr(args, "resume_state_path", None))
    if isinstance(resume_state, dict):
        mem_manager.memory_bank = resume_state.get("memory_bank", {}) or {}
        mem_manager.memory_meta_bank = resume_state.get("memory_meta_bank", {}) or {}
        mem_manager.first_appearance = resume_state.get("first_appearance", {}) or {}
        mem_manager.chunk_frames = {}
        print(
            f"[ResumeState] loaded path={getattr(args, 'resume_state_path', None)} "
            f"next_chunk_idx={int(resume_state.get('next_chunk_idx', 0) or 0)}",
            flush=True,
        )
    efficiency_chunk_records = list(resume_state.get("efficiency_chunk_records", []) if isinstance(resume_state, dict) else [])
    resume_prior_total_elapsed_s = float(resume_state.get("prior_total_elapsed_s", 0.0) if isinstance(resume_state, dict) else 0.0)
    efficiency_total_start = time.perf_counter()
    role_wise_slot_memory_bank_enabled = bool(
        getattr(engine, "memory_writer_enabled", False)
        and getattr(engine, "jigsaw_extra_encoder_enabled", False)
    )
    stage2_slot_update_domain = str(getattr(args, "train_noise_domain", "low_noise")).strip().lower()

    def _stage2_prepare_payload_for_bank(char, bank_id, mem, token_meta, chunk_idx):
        if not role_wise_slot_memory_bank_enabled:
            return mem, token_meta, {"enabled": 0.0, "mode": "raw"}
        old_payload = mem_manager.get_memory_payload(char, bank_id)
        if _is_layerwise_token_payload(mem):
            out_layers = {}
            out_meta_layers = {}
            layer_stats = {}
            old_tokens_payload = old_payload.get("tokens", None) if isinstance(old_payload, dict) else None
            old_meta_payload = old_payload.get("token_meta", None) if isinstance(old_payload, dict) else None
            for layer, update_layer_tokens in _iter_layerwise_items(mem):
                if not isinstance(update_layer_tokens, torch.Tensor) or int(update_layer_tokens.shape[0]) <= 0:
                    continue
                layer_meta = _select_layerwise_value(token_meta, layer, default=[])
                old_layer_tokens = _select_layerwise_value(old_tokens_payload, layer, default=None)
                old_layer_meta = _select_layerwise_value(old_meta_payload, layer, default=[])
                if isinstance(old_layer_tokens, torch.Tensor) and int(old_layer_tokens.shape[0]) > 0:
                    stored_tokens, stored_meta, stats = engine.stage2_update_slot_payload(
                        old_layer_tokens,
                        old_layer_meta if isinstance(old_layer_meta, list) else [],
                        update_layer_tokens,
                        layer_meta if isinstance(layer_meta, list) else [],
                        noise_domain=stage2_slot_update_domain,
                        layer_idx=layer,
                    )
                    mode = "writer_update"
                else:
                    stored_tokens, stored_meta, stats = engine._encode_memory_payload_to_stage2_slots(
                        update_layer_tokens,
                        layer_meta if isinstance(layer_meta, list) else [],
                        noise_domain=stage2_slot_update_domain,
                        layer_idx=layer,
                    )
                    mode = "initial_slot_extract"
                out_layers[_layer_key(layer)] = stored_tokens
                out_meta_layers[_layer_key(layer)] = stored_meta
                layer_stats[_layer_key(layer)] = dict(stats, mode=mode)
            return _make_layerwise_container(out_layers), _make_layerwise_container(out_meta_layers), {
                "enabled": 1.0,
                "mode": "layerwise",
                "layers": layer_stats,
            }
        old_tokens = old_payload.get("tokens", None) if isinstance(old_payload, dict) else None
        old_meta = old_payload.get("token_meta", []) if isinstance(old_payload, dict) else []
        if isinstance(old_tokens, torch.Tensor) and int(old_tokens.shape[0]) > 0:
            stored_tokens, stored_meta, stats = engine.stage2_update_slot_payload(
                old_tokens,
                old_meta if isinstance(old_meta, list) else [],
                mem,
                token_meta if isinstance(token_meta, list) else [],
                noise_domain=stage2_slot_update_domain,
                layer_idx=0,
            )
            mode = "writer_update"
        else:
            stored_tokens, stored_meta, stats = engine._encode_memory_payload_to_stage2_slots(
                mem,
                token_meta if isinstance(token_meta, list) else [],
                noise_domain=stage2_slot_update_domain,
                layer_idx=0,
            )
            mode = "initial_slot_extract"
        return stored_tokens, stored_meta, dict(stats, enabled=1.0, mode=mode)

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        chunks = data.get("chunks", data)
    max_chunks = int(getattr(args, "max_chunks", -1))
    if max_chunks > 0:
        chunks = chunks[:max_chunks]
        print(f"[Init] short mode -> processing first {len(chunks)} chunk(s)")
    else:
        print(f"[Init] full mode -> processing all {len(chunks)} chunk(s)")
    sample_characters = sorted(
        {
            str(char)
            for chunk in chunks
            if isinstance(chunk, dict)
            for char in list(chunk.get("character_list", []) or [])
        }
    )
    args._full_buffer_sample_characters = sample_characters
    print(f"[Init] sample_characters={sample_characters}", flush=True)
    top_level_characters = data.get("characters", sample_characters) if isinstance(data, dict) else sample_characters
    if isinstance(top_level_characters, dict):
        top_level_character_list = sorted(str(k) for k in top_level_characters.keys())
    elif isinstance(top_level_characters, list):
        top_level_character_list = [str(x) for x in top_level_characters]
    else:
        top_level_character_list = sample_characters
    existing_bench_manifest = _load_json_or_none(bench_manifest_path) if resume_requested else None
    bench_chunk_records = {}
    if isinstance(existing_bench_manifest, dict):
        for row in existing_bench_manifest.get("chunks", []) or []:
            if isinstance(row, dict) and row.get("chunk_idx", None) is not None:
                try:
                    bench_chunk_records[int(row["chunk_idx"])] = row
                except Exception:
                    pass
    existing_generated_timing = generated_timing(
        bench_chunk_records.values(), frames=0, fps=16
    )
    slotmem_inference_manifest = {
        "schema_version": 1,
        "format": "slotmem_inference_metadata",
        "output_path": str(Path(args.output_path).resolve()),
        "json_path": str(Path(args.json_path).resolve()),
        "ref_image_path": str(args.ref_image_path or ""),
        "resolved_reference_path": None,
        "resolved_reference_file_sha256": None,
        "fixed_reference_scope": str(args.fixed_reference_scope),
        "global_prompt": data.get("global_prompt") if isinstance(data, dict) else None,
        "characters": top_level_characters,
        "character_list": top_level_character_list,
        "chunk_count": int(len(chunks)),
        "merged_output_name": str(getattr(args, "merged_output_name", "merged_chunks.mp4")),
        "generated_duration_s": existing_generated_timing["generated_timeline_end_s"],
        "chunks": [bench_chunk_records[idx] for idx in sorted(bench_chunk_records)],
    }
    _write_slotmem_inference_manifest(bench_manifest_path, slotmem_inference_manifest)
    print(f"[Init] Saved SlotMem bench metadata manifest to: {bench_manifest_path}", flush=True)

    resolved_ref_image_path = resolve_reference_image_path(args.ref_image_path, args.json_path, data)
    if resolved_ref_image_path:
        print(f"[Init] Using reference image: {resolved_ref_image_path}")
        first_frame_pil = Image.open(resolved_ref_image_path).convert("RGB")
        prev_frames_pil = [first_frame_pil]
        fixed_random_ref_frame = first_frame_pil
        resolved_reference = Path(resolved_ref_image_path).resolve()
        slotmem_inference_manifest["resolved_reference_path"] = str(resolved_reference)
        slotmem_inference_manifest["resolved_reference_file_sha256"] = sha256_file(
            resolved_reference
        )
    else:
        if getattr(engine.pipe.dit, "has_image_input", False):
            raise ValueError("No valid reference image found for I2V inference.")
        prev_frames_pil = None
        fixed_random_ref_frame = None
    _write_slotmem_inference_manifest(bench_manifest_path, slotmem_inference_manifest)

    restored_previous_frames = False
    resume_next_chunk_idx = None
    if isinstance(resume_state, dict):
        if resume_state.get("next_chunk_idx") is not None:
            resume_next_chunk_idx = int(resume_state["next_chunk_idx"])
        resumed_prev_frames = _decode_pil_frames_from_state(resume_state.get("prev_frames_png", []))
        if resumed_prev_frames:
            prev_frames_pil = resumed_prev_frames
            restored_previous_frames = True
            print(f"[ResumeState] restored prev_frames={len(resumed_prev_frames)}", flush=True)
    start_chunk_idx = int(getattr(args, "start_chunk_idx", -1))
    if start_chunk_idx < 0 and isinstance(resume_state, dict):
        start_chunk_idx = int(resume_state.get("next_chunk_idx", 0) or 0)
    start_chunk_idx = max(0, int(start_chunk_idx))
    if start_chunk_idx >= len(chunks):
        print(f"[ResumeState] start_chunk_idx={start_chunk_idx} >= chunks={len(chunks)}; nothing to process.", flush=True)
        chunks_to_iterate = []
    else:
        chunks_to_iterate = chunks[start_chunk_idx:]
    if chunks_to_iterate:
        validate_reference_resume(
            args.fixed_reference_scope,
            start_chunk_idx=start_chunk_idx,
            has_fixed_reference=fixed_random_ref_frame is not None,
            restored_previous_frames=restored_previous_frames,
            resume_next_chunk_idx=resume_next_chunk_idx,
        )
    if start_chunk_idx > 0:
        print(f"[ResumeState] starting from chunk_idx={start_chunk_idx}", flush=True)

    for chunk_idx, chunk in enumerate(chunks_to_iterate, start=start_chunk_idx):
        mem_manager.current_chunk_idx = int(chunk_idx)
        content = chunk["content"]
        chars = [] if bool(args.native_wan_inference) else list(chunk.get("character_list", []))
        target_character = str(getattr(args, "target_character", "") or "").strip()
        if target_character:
            target_key = target_character.casefold()
            chars = sorted(
                chars,
                key=lambda char: 0 if str(char).casefold() == target_key else 1,
            )
        if int(args.max_memory_characters) > 0:
            chars = chars[: int(args.max_memory_characters)]
        print(f"\n{'=' * 60}\nChunk {chunk_idx}/{len(chunks)}: {content}\n{'=' * 60}")
        for warning_text in getattr(engine, "runtime_chunk_warnings", []):
            print(f"  [ChunkWarn] {warning_text}", flush=True)

        chunk_memory_banks = defaultdict(list)
        chunk_memory_meta_banks = defaultdict(list)
        chunk_layer_memory_banks = defaultdict(lambda: defaultdict(list))
        chunk_layer_memory_meta_banks = defaultdict(lambda: defaultdict(list))
        if engine._use_legacy_multi_memory_banks():
            bank_percents = _split_csv_floats(args.memory_bank_percents, default=[0.85, 0.60, 0.35, 0.12])
            bank_indices_to_read = list(range(len(bank_percents)))
            print(f"  [MemoryBank] mode=legacy_multi percents={bank_percents}", flush=True)
        else:
            bank_percents = engine._single_online_memory_bank_percents()
            bank_indices_to_read = [0]
            exact_percent = float(bank_percents[0]) if len(bank_percents) > 0 else 0.0
            print(f"  [MemoryBank] mode=single-bank exact-step percent={exact_percent:.6f} bank=0", flush=True)
        known_roles_for_chunk = []
        first_roles_for_chunk = []
        for char in chars:
            has_any_bank = False
            for bank_idx in bank_indices_to_read:
                payload = mem_manager.get_memory_payload_for_read(char, bank_idx)
                if payload is not None and payload.get("tokens", None) is not None:
                    has_any_bank = True
                    payload_tokens = payload["tokens"]
                    payload_meta = payload.get("token_meta", [])
                    if _is_layerwise_token_payload(payload_tokens):
                        for layer, layer_tokens in _iter_layerwise_items(payload_tokens):
                            if not isinstance(layer_tokens, torch.Tensor):
                                continue
                            chunk_layer_memory_banks[_layer_key(layer)][str(bank_idx)].append(layer_tokens)
                            layer_meta = _select_layerwise_value(payload_meta, layer, default=[])
                            if isinstance(layer_meta, list):
                                chunk_layer_memory_meta_banks[_layer_key(layer)][str(bank_idx)].extend(layer_meta)
                    else:
                        chunk_memory_banks[str(bank_idx)].append(payload_tokens)
                        chunk_memory_meta_banks[str(bank_idx)].extend(payload_meta)
            if has_any_bank:
                if engine._use_legacy_multi_memory_banks():
                    print(f"  + Injecting memory for: {char} (legacy multi-bank)")
                else:
                    print(f"  + Injecting memory for: {char} (single-bank)")
            else:
                first_roles_for_chunk.append(str(char))
                mem_manager.register_appearance(char, chunk_idx)
                continue
            known_roles_for_chunk.append(str(char))
        role_state_payload = {
            "chunk_idx": int(chunk_idx),
            "known_roles": known_roles_for_chunk,
            "first_roles": first_roles_for_chunk,
            "stage2_writer_enabled": bool(getattr(engine, "memory_writer_enabled", False)),
        }
        if hasattr(engine, "runtime_role_states"):
            engine.runtime_role_states.append(dict(role_state_payload))
        print(
            "  [RoleState] "
            + json.dumps(
                role_state_payload,
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

        if len(chunk_layer_memory_banks) > 0:
            final_memory_banks = _make_layerwise_container({
                layer: {bank: torch.cat(tensors, dim=0) for bank, tensors in bank_map.items() if len(tensors) > 0}
                for layer, bank_map in chunk_layer_memory_banks.items()
            })
            final_memory_meta_banks = _make_layerwise_container({
                layer: {bank: list(chunk_layer_memory_meta_banks[layer].get(bank, [])) for bank in bank_map.keys()}
                for layer, bank_map in chunk_layer_memory_banks.items()
            })
        else:
            final_memory_banks = {k: torch.cat(v, dim=0) for k, v in chunk_memory_banks.items() if len(v) > 0}
            final_memory_meta_banks = {k: list(chunk_memory_meta_banks.get(k, [])) for k in final_memory_banks.keys()}

        if _is_layerwise_container(final_memory_banks):
            final_memory = None
            for _, bank_map in _iter_layerwise_items(final_memory_banks):
                if isinstance(bank_map, dict):
                    final_memory = bank_map.get("0", next((v for v in bank_map.values() if isinstance(v, torch.Tensor)), None))
                if isinstance(final_memory, torch.Tensor):
                    break
            for layer, bank_map in _iter_layerwise_items(final_memory_banks):
                if isinstance(bank_map, dict):
                    token_counts = {str(bank): int(tokens.shape[0]) for bank, tokens in bank_map.items() if isinstance(tokens, torch.Tensor)}
                    print(f"  [MemoryPayload] mode=layerwise layer={layer} token_counts={token_counts}", flush=True)
                    meta_map = _select_layerwise_value(final_memory_meta_banks, layer, default={})
                    if isinstance(meta_map, dict):
                        for bank, token_meta in meta_map.items():
                            print(f"  [MemoryPurity] layer={layer} bank={bank} stats={json.dumps(_summarize_token_meta(token_meta), sort_keys=True)}", flush=True)
        else:
            final_memory = final_memory_banks.get("0", None)
            if isinstance(final_memory_banks, dict) and len(final_memory_banks) > 0:
                token_counts = {str(bank): int(tokens.shape[0]) for bank, tokens in final_memory_banks.items() if isinstance(tokens, torch.Tensor)}
                print(f"  [MemoryPayload] mode=shared token_counts={token_counts}", flush=True)
                for bank, token_meta in final_memory_meta_banks.items():
                    print(f"  [MemoryPurity] bank={bank} stats={json.dumps(_summarize_token_meta(token_meta), sort_keys=True)}", flush=True)
        memory_token_lengths_per_character = None
        if len(chunk_memory_banks) > 0 and not _is_layerwise_container(final_memory_banks):
            first_bank_key = sorted(chunk_memory_banks.keys())[0]
            memory_token_lengths_per_character = [m.shape[0] for m in chunk_memory_banks[first_bank_key]]
        if _is_layerwise_container(final_memory_banks):
            read_layer_count = 0
            read_slot_count = 0
            for _, bank_map in _iter_layerwise_items(final_memory_banks):
                layer_has_tensor = False
                if isinstance(bank_map, dict):
                    for value in bank_map.values():
                        if isinstance(value, torch.Tensor):
                            layer_has_tensor = True
                            read_slot_count += int(value.shape[0]) if value.ndim >= 1 else 0
                read_layer_count += int(layer_has_tensor)
        else:
            read_layer_count = int(any(isinstance(value, torch.Tensor) for value in final_memory_banks.values()))
            read_slot_count = sum(
                int(value.shape[0]) if value.ndim >= 1 else 0
                for value in final_memory_banks.values()
                if isinstance(value, torch.Tensor)
            )
        memory_read_record = {
            "chunk_idx": int(chunk_idx),
            "attempted_roles": [str(value) for value in chars],
            "known_roles": list(known_roles_for_chunk),
            "first_roles": list(first_roles_for_chunk),
            "nonempty": bool(read_layer_count > 0 and read_slot_count > 0),
            "payload_layers": int(read_layer_count),
            "payload_slots": int(read_slot_count),
        }
        memory_bank_hash_before_write = _memory_bank_sha256(mem_manager)

        full_buffer_status_before_generate = _full_buffer_status(
            args,
            engine,
            mem_manager,
            chunk_idx,
            phase="before_generate",
        )
        if bool(getattr(args, "stop_after_first_full_buffer", False)) or getattr(args, "efficiency_runtime_log", None):
            print(
                "  [FullBufferStatusBefore] "
                + json.dumps(full_buffer_status_before_generate, ensure_ascii=False, sort_keys=True),
                flush=True,
            )

        target_seed_override = getattr(args, "target_seed_override", None)
        chunk_seed = (
            int(target_seed_override)
            if target_seed_override is not None and chunk_idx == start_chunk_idx
            else int(getattr(args, "seed_base", 42)) + chunk_idx
        )
        print(f"  [RunSeed] chunk={chunk_idx} seed={chunk_seed}", flush=True)
        chunk_ref_images = prev_frames_pil
        chunk_random_ref_frame = choose_random_reference(
            args.fixed_reference_scope,
            chunk_idx,
            fixed_random_ref_frame,
            chunk_ref_images,
        )
        reference_conditioning = build_reference_conditioning_audit(
            scope=args.fixed_reference_scope,
            chunk_idx=chunk_idx,
            fixed_reference=fixed_random_ref_frame,
            previous_frames=chunk_ref_images,
            random_reference=chunk_random_ref_frame,
        )
        original_ref_pad_num = int(getattr(args, "ref_pad_num", 0))
        effective_ref_pad_num = original_ref_pad_num
        if int(effective_ref_pad_num) != original_ref_pad_num:
            args.ref_pad_num = int(effective_ref_pad_num)
            engine.args.ref_pad_num = int(effective_ref_pad_num)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        chunk_time_start = time.perf_counter()
        try:
            video_frames, latents, online_memory_results = engine.generate_chunk(
                prompt=content,
                memory_tokens=final_memory if (not args.native_wan_inference) else None,
                memory_bank_tokens=(final_memory_banks if len(final_memory_banks) > 0 else None) if (not args.native_wan_inference) else None,
                memory_bank_percents=bank_percents,
                memory_bank_token_meta=(final_memory_meta_banks if len(final_memory_meta_banks) > 0 else None) if (not args.native_wan_inference) else None,
                memory_token_lengths_per_character=memory_token_lengths_per_character if (not args.native_wan_inference) else None,
                ref_images=chunk_ref_images,
                random_ref_frame=chunk_random_ref_frame,
                seed=chunk_seed,
                online_memory_chars=[] if args.native_wan_inference else chars,
                online_memory_bank_percents=[] if args.native_wan_inference else bank_percents,
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            chunk_elapsed_s = time.perf_counter() - chunk_time_start
            if int(getattr(args, "ref_pad_num", 0)) != original_ref_pad_num:
                args.ref_pad_num = original_ref_pad_num
                engine.args.ref_pad_num = original_ref_pad_num

        if chunk_idx < len(chunks) - 1 and int(args.num_motion_frames) > 0:
            frames_to_save = video_frames[:-int(args.num_motion_frames)] or video_frames
        else:
            frames_to_save = video_frames
        save_path = os.path.join(args.output_path, f"chunk_{chunk_idx:03d}.mp4")
        save_current_video = True
        if bool(getattr(args, "save_only_full_buffer_target", False)):
            save_current_video = bool(full_buffer_status_before_generate.get("is_full", False))
        if save_current_video:
            save_video(frames_to_save, save_path, fps=16)
            print(f"  > Saved to {save_path} (frames={len(frames_to_save)})")
        else:
            print(
                f"  [FullBufferSave] skip warmup video save for chunk={chunk_idx}; "
                "frames still feed conditioning and online memory.",
                flush=True,
            )
        saved_frame_count = int(len(frames_to_save)) if bool(save_current_video) else 0
        media_timing = generated_timing(
            bench_chunk_records.values(), frames=saved_frame_count, fps=16
        )
        chunk_metadata = {
            "schema_version": 1,
            "sample_id": Path(args.output_path).resolve().name,
            "chunk_idx": int(chunk_idx),
            "caption": str(content),
            "content": str(content),
            "characters": [str(x) for x in list(chunk.get("character_list", []) or [])],
            "character_list": [str(x) for x in list(chunk.get("character_list", []) or [])],
            "known_roles": list(known_roles_for_chunk),
            "first_roles": list(first_roles_for_chunk),
            "role_state": role_state_payload,
            # start/end are the source-caption timeline, not the encoded media timeline.
            "start": chunk.get("start", None),
            "end": chunk.get("end", None),
            "source_timeline_start_s": chunk.get("start", None),
            "source_timeline_end_s": chunk.get("end", None),
            "seed": int(chunk_seed),
            "fps": 16,
            "frames": saved_frame_count,
            "raw_frames": int(len(video_frames)),
            "video_saved": bool(save_current_video),
            "video_path": str(Path(save_path).resolve()) if bool(save_current_video) else None,
            "source_json_path": str(Path(args.json_path).resolve()),
            "reference_conditioning": reference_conditioning,
            **media_timing,
        }
        chunk_json_path = Path(args.output_path) / f"chunk_{chunk_idx:03d}.metadata.json"
        _write_slotmem_inference_manifest(chunk_json_path, chunk_metadata)
        bench_chunk_records[int(chunk_idx)] = dict(chunk_metadata)
        slotmem_inference_manifest["chunks"] = [bench_chunk_records[idx] for idx in sorted(bench_chunk_records)]
        slotmem_inference_manifest["completed_chunk_count"] = int(len(bench_chunk_records))
        slotmem_inference_manifest["generated_duration_s"] = media_timing[
            "generated_timeline_end_s"
        ]
        _write_slotmem_inference_manifest(bench_manifest_path, slotmem_inference_manifest)
        print(f"  [BenchMeta] Saved chunk metadata: {chunk_json_path}", flush=True)

        start_idx = max(0, len(video_frames) - int(args.num_motion_frames))
        prev_frames_pil = video_frames[start_idx:]

        memory_write_enabled = True
        chunk_writer_updates = []
        if not memory_write_enabled:
            pass
        else:
            online_tokens = online_memory_results.get("tokens", {}) if isinstance(online_memory_results, dict) else {}
            online_token_meta = online_memory_results.get("token_meta", {}) if isinstance(online_memory_results, dict) else {}
            if bool(args.save_feature_mapping_viz):
                mem_manager.chunk_frames[int(chunk_idx)] = list(video_frames)
            for char, bank_dict in online_tokens.items():
                for bank_id, mem in bank_dict.items():
                    if mem is not None:
                        token_meta = []
                        if isinstance(online_token_meta, dict):
                            role_meta = online_token_meta.get(char, {})
                            if isinstance(role_meta, dict):
                                token_meta = role_meta.get(bank_id, [])
                        stored_mem, stored_meta, stage2_store_stats = _stage2_prepare_payload_for_bank(
                            char,
                            int(bank_id),
                            mem,
                            token_meta,
                            int(chunk_idx),
                        )
                        if role_wise_slot_memory_bank_enabled:
                            print(
                                "  [RoleWiseSlotMemoryBank] "
                                + json.dumps(
                                    {
                                        "chunk_idx": int(chunk_idx),
                                        "char": str(char),
                                        "bank": int(bank_id),
                                        "stats": stage2_store_stats,
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    default=str,
                                )[:2000],
                                flush=True,
                            )
                        chunk_writer_updates.append({
                            "chunk_idx": int(chunk_idx),
                            "character": str(char),
                            "bank": int(bank_id),
                            "stats": stage2_store_stats,
                        })
                        mem_manager.add_memory(
                            char,
                            stored_mem,
                            bank_idx=int(bank_id),
                            token_meta=stored_meta,
                            source_chunk_idx=int(chunk_idx),
                            source_video_frames=video_frames,
                            first_appearance_only=bool(args.use_first_appearance_memory_only),
                        )
        memory_bank_stats = _summarize_memory_manager_bytes(mem_manager)
        memory_bank_hash_after_write = _memory_bank_sha256(mem_manager)
        full_buffer_status_after_write = _full_buffer_status(
            args,
            engine,
            mem_manager,
            chunk_idx,
            phase="after_write",
        )
        analytic_slot_bytes = _analytic_role_wise_slot_memory_bank_bytes(
            args,
            engine,
            int(full_buffer_status_after_write.get("public_character_count", 0) or 0),
        )
        chunk_efficiency_record = {
            "event": "chunk",
            "chunk_idx": int(chunk_idx),
            "method_flags": {
                "native_wan_inference": bool(getattr(args, "native_wan_inference", False)),
                "jigsaw_extra_encoder_enabled": bool(getattr(engine, "jigsaw_extra_encoder_enabled", False)),
            },
            "time_s": float(chunk_elapsed_s),
            "frames": int(len(frames_to_save)),
            "raw_frames": int(len(video_frames)),
            "seed": int(chunk_seed),
            "video_saved": bool(save_current_video),
            "video_path": str(save_path) if bool(save_current_video) else None,
            "saved_frames": int(len(frames_to_save)) if bool(save_current_video) else 0,
            "peak_allocated_gb": float(torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0,
            "peak_reserved_gb": float(torch.cuda.max_memory_reserved() / (1024 ** 3)) if torch.cuda.is_available() else 0.0,
            "memory_bank": memory_bank_stats,
            "slot_bank_size_mb": float(memory_bank_stats.get("tensor_mb", 0.0)),
            "slot_bank_analytic_mb": float(analytic_slot_bytes / (1024 ** 2)),
            "full_buffer_status_before_generate": full_buffer_status_before_generate,
            "full_buffer_status_after_write": full_buffer_status_after_write,
            "last_sparse_role_memory_stats": getattr(engine, "_last_sparse_role_memory_stats", {}),
            "last_sparse_role_memory_stats_by_layer": getattr(engine, "_last_sparse_role_memory_stats_by_layer", {}),
            "last_jigsaw_stage2_writer_stats": getattr(engine, "_last_jigsaw_stage2_writer_stats", {}),
            "role_state": role_state_payload,
            "memory_read": memory_read_record,
            "writer_updates": chunk_writer_updates,
            "memory_bank_sha256_before_write": memory_bank_hash_before_write,
            "memory_bank_sha256_after_write": memory_bank_hash_after_write,
            "memory_bank_hash_changed": memory_bank_hash_before_write != memory_bank_hash_after_write,
        }
        efficiency_chunk_records.append(chunk_efficiency_record)
        slotmem_inference_manifest["runtime_evidence"] = _runtime_evidence(
            efficiency_chunk_records, engine
        )
        _write_slotmem_inference_manifest(bench_manifest_path, slotmem_inference_manifest)
        _append_efficiency_jsonl(getattr(args, "efficiency_runtime_log", None), chunk_efficiency_record)
        if getattr(args, "efficiency_runtime_log", None):
            print(
                f"  [Efficiency] chunk={chunk_idx} time_s={chunk_elapsed_s:.3f} "
                f"peak_allocated_gb={chunk_efficiency_record['peak_allocated_gb']:.3f} "
                f"slot_bank_mb={chunk_efficiency_record['slot_bank_size_mb']:.3f}",
                flush=True,
            )
        _save_resume_state(
            getattr(args, "save_state_path", None),
            mem_manager=mem_manager,
            prev_frames_pil=prev_frames_pil,
            efficiency_chunk_records=efficiency_chunk_records,
            next_chunk_idx=int(chunk_idx) + 1,
            prior_total_elapsed_s=resume_prior_total_elapsed_s + (time.perf_counter() - efficiency_total_start),
        )

        full_buffer_stop_status = None
        if bool(getattr(args, "stop_after_first_full_buffer", False)):
            stop_mode = str(getattr(args, "full_buffer_stop_mode", "before_generate"))
            if stop_mode == "before_generate" and bool(full_buffer_status_before_generate.get("is_full", False)):
                full_buffer_stop_status = full_buffer_status_before_generate
            elif stop_mode == "after_write" and bool(full_buffer_status_after_write.get("is_full", False)):
                full_buffer_stop_status = full_buffer_status_after_write
            if full_buffer_stop_status is not None:
                _write_full_buffer_marker(
                    args,
                    engine,
                    chunk_idx,
                    args.output_path,
                    save_path,
                    full_buffer_stop_status,
                    chunk_efficiency_record,
                )
                print(
                    f"[FullBufferStop] mode={stop_mode} reached at chunk={chunk_idx}; stop before remaining chunks.",
                    flush=True,
                )
                break
        if int(getattr(args, "stop_after_chunk_idx", -1)) == int(chunk_idx):
            print(f"[StopAfterChunk] reached chunk={chunk_idx}; stop before remaining chunks.", flush=True)
            break

        if bool(args.save_memory_viz):
            chunk_char_positions = online_memory_results.get("viz", {}) if isinstance(online_memory_results, dict) else {}
            spatial_shape = online_memory_results.get("spatial_shape", None) if isinstance(online_memory_results, dict) else None
            if chunk_char_positions and spatial_shape is not None:
                if args.memory_viz_dir is None:
                    args.memory_viz_dir = os.path.join(args.output_path, "memory_viz")
                t_vid = len(video_frames)
                f_lat = latents.shape[2]
                vae_stride_t = max(1, (t_vid - 1) // max(1, (f_lat - 1)))
                save_chunk_memory_visualization(
                    viz_root_dir=args.memory_viz_dir,
                    chunk_idx=chunk_idx,
                    video_frames=video_frames,
                    char_positions_dict=chunk_char_positions,
                    h_patch=int(spatial_shape[0]),
                    w_patch=int(spatial_shape[1]),
                    vae_stride_t=vae_stride_t,
                    char_boxes_dict=None,
                )

        if bool(args.save_feature_mapping_viz):
            feature_steps = online_memory_results.get("feature_mapping_steps", []) if isinstance(online_memory_results, dict) else []
            if args.feature_mapping_viz_dir is None:
                args.feature_mapping_viz_dir = os.path.join(args.output_path, "memory_viz")
            save_feature_mapping_visualization(
                viz_root_dir=args.feature_mapping_viz_dir,
                generation_chunk_idx=chunk_idx,
                generation_video_frames=video_frames,
                generation_latents=latents,
                feature_mapping_steps=feature_steps,
                memory_manager=mem_manager,
                draw_empty=bool(args.feature_mapping_draw_empty),
            )

        if bool(args.save_denoise_step_viz):
            step_records = online_memory_results.get("denoise_step_records", []) if isinstance(online_memory_results, dict) else []
            if args.denoise_step_viz_dir is None:
                args.denoise_step_viz_dir = os.path.join(args.output_path, "denoise_step_viz")
            save_denoise_step_visualization(args.denoise_step_viz_dir, chunk_idx, step_records)

        if bool(args.save_denoise_step_edge_viz):
            step_records = online_memory_results.get("denoise_step_records", []) if isinstance(online_memory_results, dict) else []
            if args.denoise_step_edge_viz_dir is None:
                args.denoise_step_edge_viz_dir = os.path.join(args.output_path, "denoise_step_edge_viz")
            save_denoise_step_edge_frames_visualization(args.denoise_step_edge_viz_dir, chunk_idx, step_records)

        if (not bool(args.native_wan_inference)) and chunk_idx == 0 and bool(getattr(args, "defer_lora_until_after_first_chunk", False)):
            print("[Switch] Chunk 0 done. Loading checkpoint for subsequent chunks...")
            engine.load_trained_weights_if_needed()

        if int(getattr(args, "smoke_stop_after_chunk_idx", -1)) == int(chunk_idx):
            _write_smoke_marker(args, engine, chunk_idx, args.output_path)
            print(
                f"[Smoke] Reached chunk {chunk_idx} with num_inference_steps={int(args.num_inference_steps)}; stop before remaining chunks.",
                flush=True,
            )
            break

    if getattr(args, "efficiency_metrics_path", None):
        total_elapsed_s = resume_prior_total_elapsed_s + (time.perf_counter() - efficiency_total_start)
        measured_frames = sum(int(row.get("frames", 0)) for row in efficiency_chunk_records)
        measured_chunks = len(efficiency_chunk_records)
        full_buffer_target_record = None
        full_buffer_stop_mode = str(getattr(args, "full_buffer_stop_mode", "before_generate"))
        full_buffer_status_key = (
            "full_buffer_status_before_generate"
            if full_buffer_stop_mode == "before_generate"
            else "full_buffer_status_after_write"
        )
        for row in efficiency_chunk_records:
            status = row.get(full_buffer_status_key, {})
            if isinstance(status, dict) and bool(status.get("is_full", False)):
                full_buffer_target_record = row
                break
        summary = {
            "status": "completed",
            "output_path": str(args.output_path),
            "json_path": str(args.json_path),
            "ref_image_path": str(args.ref_image_path or ""),
            "num_inference_steps": int(getattr(args, "num_inference_steps", 0) or 0),
            "height": int(getattr(args, "height", 0) or 0),
            "width": int(getattr(args, "width", 0) or 0),
            "context_frames": int(getattr(args, "context_frames", 0) or 0),
            "seed_base": int(getattr(args, "seed_base", 0) or 0),
            "sample_solver": str(getattr(args, "sample_solver", "")),
            "sample_shift": float(getattr(args, "sample_shift", 0.0) or 0.0),
            "jigsaw_extra_encoder_enabled": bool(getattr(engine, "jigsaw_extra_encoder_enabled", False)),
            "train_stage": str(getattr(args, "train_stage", "")),
            "train_noise_domain": str(getattr(args, "train_noise_domain", "")),
            "max_memory_characters": int(getattr(args, "max_memory_characters", 0) or 0),
            "jigsaw_extra_encoder_slots": int(getattr(engine, "jigsaw_extra_encoder_slots", getattr(args, "jigsaw_extra_encoder_slots", 0)) or 0),
            "jigsaw_extra_encoder_layer_groups": list(getattr(engine, "jigsaw_extra_encoder_layer_groups", [])),
            "sparse_role_memory_injection_layers": list(getattr(engine, "sparse_role_memory_injection_layers", [])),
            "dual_expert_load_mode": str(getattr(args, "dual_expert_load_mode", "")),
            "dual_expert_offload_dtype": str(getattr(args, "dual_expert_offload_dtype", "")),
            "total_elapsed_s": float(total_elapsed_s),
            "measured_chunks": int(measured_chunks),
            "measured_frames": int(measured_frames),
            "avg_time_per_chunk_s": float(sum(float(row.get("time_s", 0.0)) for row in efficiency_chunk_records) / max(measured_chunks, 1)),
            "fps": float(measured_frames / max(sum(float(row.get("time_s", 0.0)) for row in efficiency_chunk_records), 1e-9)),
            "peak_allocated_gb": float(max([float(row.get("peak_allocated_gb", 0.0)) for row in efficiency_chunk_records] or [0.0])),
            "peak_reserved_gb": float(max([float(row.get("peak_reserved_gb", 0.0)) for row in efficiency_chunk_records] or [0.0])),
            "slot_bank_size_mb": float(max([float(row.get("slot_bank_size_mb", 0.0)) for row in efficiency_chunk_records] or [0.0])),
            "slot_bank_analytic_mb": float(max([float(row.get("slot_bank_analytic_mb", 0.0)) for row in efficiency_chunk_records] or [0.0])),
            "full_buffer_target_status_key": full_buffer_status_key,
            "full_buffer_target_chunk_idx": int(full_buffer_target_record.get("chunk_idx", -1)) if isinstance(full_buffer_target_record, dict) else None,
            "full_buffer_target_time_s": float(full_buffer_target_record.get("time_s", 0.0)) if isinstance(full_buffer_target_record, dict) else None,
            "full_buffer_target_frames": int(full_buffer_target_record.get("frames", 0)) if isinstance(full_buffer_target_record, dict) else None,
            "full_buffer_target_video": full_buffer_target_record.get("video_path") if isinstance(full_buffer_target_record, dict) else None,
            "full_buffer_target_chunk": full_buffer_target_record,
            "runtime_evidence": _runtime_evidence(efficiency_chunk_records, engine),
            "chunks": efficiency_chunk_records,
        }
        _write_efficiency_json(getattr(args, "efficiency_metrics_path", None), summary)
        print(f"       Efficiency summary: {args.efficiency_metrics_path}", flush=True)

    print(f"\n[Done] All {len(chunks)} chunks processed.")
    print(f"       Output: {args.output_path}/chunk_*.mp4")
    if bool(args.merge_chunks):
        merged = merge_chunk_videos(
            output_dir=args.output_path,
            merged_filename=args.merged_output_name,
            pattern="chunk_*.mp4",
        )
        if merged is not None:
            print(f"       Merged: {merged}")
            slotmem_inference_manifest["merged_video_path"] = str(Path(merged).resolve())
            _write_slotmem_inference_manifest(bench_manifest_path, slotmem_inference_manifest)
        else:
            print("       Merged: skipped or failed")


if __name__ == "__main__":
    main()
