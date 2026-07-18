"""
Training script for SlotMem with inline memory extraction and training.

Current design:
- Each DDP rank runs on one GPU.
- In each rank/process, every step does: sample -> memory extraction -> training.
- No file-based buffer and no dedicated extraction worker process.
"""
import time
import json
import torch
import torch.nn.functional as F
import os
import argparse
import gc
import math
import lightning as pl
import re
from collections import defaultdict, deque
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.distributed as dist
from einops import rearrange
import imageio
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (SCRIPT_DIR,):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from utils.project_utils import *
from utils.cpu_people_pixelate import (
    DEFAULT_YOLO_SEG_CKPT,
    maybe_pixelate_condition_batch_cpu,
)
from torch.utils.data import IterableDataset
import threading
import shutil
from datetime import datetime, timedelta
import secrets
import traceback
import types
from contextlib import suppress,contextmanager
from safetensors.torch import load_file as safetensors_load_file

from wan22_train_runtime import build_wan22_training_pipe
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d, modulate
LAYERWISE_MARKER = "__layerwise__"
LAYERWISE_LAYERS_KEY = "layers"


def _layer_key_for_payload(layer_idx):
    try:
        return str(int(layer_idx))
    except Exception:
        return str(layer_idx)


def _pm_is_layerwise_container(value):
    return (
        isinstance(value, dict)
        and bool(value.get(LAYERWISE_MARKER, False))
        and isinstance(value.get(LAYERWISE_LAYERS_KEY, None), dict)
    )


def _pm_make_layerwise_container(layers):
    return {
        LAYERWISE_MARKER: True,
        LAYERWISE_LAYERS_KEY: {_layer_key_for_payload(k): v for k, v in dict(layers).items()},
    }


def _pm_iter_layerwise_items(value):
    if _pm_is_layerwise_container(value):
        for layer, payload in value.get(LAYERWISE_LAYERS_KEY, {}).items():
            yield _layer_key_for_payload(layer), payload
        return
    if isinstance(value, dict):
        for layer, payload in value.items():
            if isinstance(payload, torch.Tensor) or isinstance(payload, (list, dict)):
                yield _layer_key_for_payload(layer), payload


def _pm_select_layerwise_value(value, layer_idx, default=None):
    if _pm_is_layerwise_container(value):
        layers = value.get(LAYERWISE_LAYERS_KEY, {})
        return layers.get(_layer_key_for_payload(layer_idx), default)
    return value if value is not None else default


def _memory_count_tokens(payload):
    if isinstance(payload, torch.Tensor) and payload.ndim >= 2:
        return int(payload.shape[0])
    if _pm_is_layerwise_container(payload):
        return int(sum(_memory_count_tokens(layer_payload) for _, layer_payload in _pm_iter_layerwise_items(payload)))
    if isinstance(payload, dict):
        return int(sum(_memory_count_tokens(value) for value in payload.values()))
    return 0


def _concat_token_payloads(token_payloads, meta_payloads=None):
    token_payloads = [payload for payload in token_payloads if payload is not None]
    meta_payloads = list(meta_payloads or [])
    if not token_payloads:
        return torch.empty((0, 0)), []
    if any(_pm_is_layerwise_container(payload) for payload in token_payloads):
        all_layers = []
        for payload in token_payloads:
            if _pm_is_layerwise_container(payload):
                for layer, _ in _pm_iter_layerwise_items(payload):
                    if layer not in all_layers:
                        all_layers.append(layer)
        out_tokens = {}
        out_meta = {}
        for layer in all_layers:
            parts = []
            meta_parts = []
            for idx, payload in enumerate(token_payloads):
                layer_tokens = _pm_select_layerwise_value(payload, layer, default=None)
                if isinstance(layer_tokens, torch.Tensor) and layer_tokens.ndim >= 2 and int(layer_tokens.shape[0]) > 0:
                    parts.append(layer_tokens)
                    meta_payload = meta_payloads[idx] if idx < len(meta_payloads) else None
                    layer_meta = _pm_select_layerwise_value(meta_payload, layer, default=[])
                    if isinstance(layer_meta, list):
                        meta_parts.extend(layer_meta)
            if parts:
                out_tokens[layer] = torch.cat(parts, dim=0)
                out_meta[layer] = meta_parts
        return _pm_make_layerwise_container(out_tokens), _pm_make_layerwise_container(out_meta)
    parts = [payload for payload in token_payloads if isinstance(payload, torch.Tensor) and payload.ndim >= 2 and int(payload.shape[0]) > 0]
    if not parts:
        return torch.empty((0, 0)), []
    out_meta = []
    for item in meta_payloads:
        if isinstance(item, list):
            out_meta.extend(item)
    return torch.cat(parts, dim=0), out_meta

from mem_encoder_utils import (
    MemoryEncoderBank,
    MemoryWriter,
    encode_role_tokens_to_slots as _memory_encode_role_tokens_to_slots,
    extract_prefixed_state_dict as _jigsaw_extract_prefixed_state_dict,
    memory_encoder_contrastive_loss as _memory_encoder_contrastive_loss,
    memory_encoder_enabled as _memory_encoder_enabled,
    memory_writer_effective_mode as _memory_writer_effective_mode,
    parse_layer_groups as _jigsaw_parse_layer_groups,
    parse_layer_list as _jigsaw_parse_layer_list,
)

# Import attention extraction utilities
sys.path.insert(0, os.path.dirname(__file__))
from attention_probe_utils import (
    AttentionMapExtractor,
    MultiCharacterAttentionMapExtractor,
    process_attention_map_to_mask,
    verify_target_text_is_single_token,
    find_token_index_in_prompt
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _matches_lora_target(module_name, target_names):
    for target in target_names:
        target = str(target).strip()
        if not target:
            continue
        if module_name == target or module_name.endswith(f".{target}"):
            return True
    return False


def _install_train_lora_forward(module, rank, alpha, init_lora_weights=True, adapter_name=None):
    if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
        return module
    if not isinstance(module, torch.nn.Linear):
        raise TypeError(f"Custom train LoRA currently supports only nn.Linear, got {type(module).__name__}")

    weight_dtype = module.weight.dtype
    weight_device = module.weight.device
    rank = int(rank)
    scale = float(alpha) / float(max(rank, 1))

    lora_A = torch.nn.Linear(module.in_features, rank, bias=False, device=weight_device, dtype=weight_dtype)
    lora_B = torch.nn.Linear(rank, module.out_features, bias=False, device=weight_device, dtype=weight_dtype)
    if init_lora_weights:
        torch.nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
    else:
        torch.nn.init.normal_(lora_A.weight, std=1.0 / max(rank, 1))
    torch.nn.init.zeros_(lora_B.weight)

    module.add_module("lora_A", lora_A)
    module.add_module("lora_B", lora_B)
    module.lora_alpha = float(alpha)
    module.lora_rank = rank
    module.lora_scale = scale
    module.disable_adapters = False
    module.lora_adapter_name = str(adapter_name) if adapter_name is not None else "default"
    module._original_forward_before_lora = module.forward

    def _train_lora_forward(this, x, *args, **kwargs):
        out = this._original_forward_before_lora(x, *args, **kwargs)
        if bool(getattr(this, "disable_adapters", False)):
            return out
        lora_dtype = this.lora_A.weight.dtype
        lora_in = x if x.dtype == lora_dtype else x.to(dtype=lora_dtype)
        lora_out = this.lora_B(this.lora_A(lora_in))
        return out + lora_out.to(dtype=out.dtype) * float(this.lora_scale)

    module.forward = types.MethodType(_train_lora_forward, module)
    return module


def _inject_train_lora_modules(model, target_modules, lora_rank, lora_alpha, init_lora_weights=True, adapter_name=None):
    injected = 0
    target_modules = [str(x).strip() for x in target_modules if str(x).strip()]
    for module_name, module in model.named_modules():
        if not _matches_lora_target(module_name, target_modules):
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


def _load_checkpoint_payload(path):
    path = str(path)
    if path.endswith(".safetensors"):
        return safetensors_load_file(path)
    return torch.load(path, map_location="cpu")


def _parse_layer_indices_csv(value, fallback_idx=3):
    fallback = [int(fallback_idx)]
    if value is None:
        return fallback
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple, set)):
        raw_parts = list(value)
    else:
        raw_parts = [p.strip() for p in str(value).split(",") if str(p).strip() != ""]
    out = []
    for p in raw_parts:
        try:
            v = int(p)
        except Exception:
            continue
        if v >= 0 and v not in out:
            out.append(v)
    return out if len(out) > 0 else fallback


def _parse_layer_scale_map(value):
    out = {}
    if value is None:
        return out
    for item in str(value).split(','):
        item = item.strip()
        if not item or ':' not in item:
            continue
        layer_text, scale_text = item.split(':', 1)
        try:
            out[int(layer_text.strip())] = float(scale_text.strip())
        except Exception:
            pass
    return out


def _append_extract_debug_event(events, **payload):
    if events is None:
        return
    if isinstance(events, list):
        try:
            events.append(dict(payload))
        except Exception:
            pass


def _detach_cpu_tree(obj):
    """Recursively detach tensors and move batch payloads to CPU before yield.

    This prevents extraction-side CUDA tensors from staying alive across the
    IterableDataset -> Lightning training_step boundary.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _detach_cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_detach_cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_cpu_tree(v) for v in obj)
    return obj


def _find_cuda_tensors_in_tree(obj, prefix="batch", limit=32):
    found = []

    def _walk(x, name):
        if len(found) >= int(limit):
            return
        if torch.is_tensor(x):
            if x.is_cuda:
                try:
                    mib = float(x.numel() * x.element_size()) / (1024.0 ** 2)
                except Exception:
                    mib = 0.0
                found.append((name, tuple(x.shape), str(x.dtype), str(x.device), mib))
        elif isinstance(x, dict):
            for k, v in x.items():
                _walk(v, f"{name}.{k}" if name else str(k))
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                _walk(v, f"{name}[{i}]")

    _walk(obj, prefix)
    return found

def install_deepspeed_zero_secondary_partition_guard(enabled=True):
    """Install a runtime guard for DeepSpeed ZeRO secondary partition narrow overflow.

    This guard only intercepts the specific IndexError in _partition_param_sec
    ("start out of range") and applies a bounds-safe copy fallback.
    """
    if not enabled:
        return False
    try:
        import deepspeed.runtime.zero.partition_parameters as zpp
    except Exception as e:
        print(f"[Zero3Guard] Skip install (import failed): {e}", flush=True)
        return False

    if getattr(zpp.Init, '_secondary_partition_guard_installed', False):
        return True

    original_fn = zpp.Init._partition_param_sec

    def _guarded_partition_param_sec(self, param, buffer=None, has_been_updated=False):
        try:
            return original_fn(self, param, buffer=buffer, has_been_updated=has_been_updated)
        except IndexError as e:
            if 'start out of range' not in str(e):
                raise

            # Re-run with bounds-safe logic only for the failing edge-case.
            assert param.ds_status is not zpp.ZeroParamStatus.INFLIGHT, f" {param} Cannot partition a param in flight"
            if param.ds_status is zpp.ZeroParamStatus.AVAILABLE:
                if param.ds_secondary_tensor is not None and not has_been_updated:
                    return

                tensor_size = self._aligned_size(param)
                secondary_partition_size = int(tensor_size // self.num_ranks_in_param_group)

                if param.ds_secondary_tensor is None:
                    final_location = None
                    secondary_partitioned_tensor = torch.empty(
                        secondary_partition_size,
                        dtype=param.dtype,
                        device=self.remote_device,
                    )
                    if self.pin_memory:
                        secondary_partitioned_tensor = secondary_partitioned_tensor.pin_memory()
                    if not param.requires_grad and self.quantized_nontrainable_weights:
                        secondary_partitioned_tensor, secondary_partitioned_tensor.ds_quant_scale = self.quantizer_module.quantize(
                            secondary_partitioned_tensor
                        )
                    secondary_partitioned_tensor.requires_grad = False
                    param.ds_secondary_tensor = secondary_partitioned_tensor
                    param.ds_secondary_tensor.ds_numel = secondary_partition_size
                    param.ds_secondary_tensor.status = zpp.PartitionedParamStatus.AVAILABLE
                    param.ds_secondary_tensor.final_location = final_location

                secondary_start = int(secondary_partition_size * self.rank_in_group)
                one_dim_param = param.contiguous().view(-1)
                param_numel = int(one_dim_param.numel())
                sec_numel = max(0, min(int(param.ds_numel) - secondary_start, int(secondary_partition_size)))

                with torch.no_grad():
                    if secondary_start >= param_numel or sec_numel <= 0:
                        param.ds_secondary_tensor.zero_()
                    else:
                        safe_numel = min(sec_numel, param_numel - secondary_start)
                        if safe_numel > 0:
                            param.ds_secondary_tensor.narrow(0, 0, safe_numel).copy_(
                                one_dim_param.narrow(0, secondary_start, safe_numel)
                            )
                            remain = int(param.ds_secondary_tensor.numel()) - int(safe_numel)
                            if remain > 0:
                                param.ds_secondary_tensor.narrow(0, safe_numel, remain).zero_()
                        else:
                            param.ds_secondary_tensor.zero_()

                if not zpp.get_accelerator().resolves_data_dependency():
                    zpp.get_accelerator().current_stream().synchronize()

                if not getattr(self, '_secondary_partition_guard_warned', False):
                    print(
                        "[Zero3Guard] Caught secondary partition IndexError and applied bounds-safe fallback. "
                        "Training continues without freezing tiny trainable params.",
                        flush=True,
                    )
                    self._secondary_partition_guard_warned = True
                return
            raise

    zpp.Init._partition_param_sec = _guarded_partition_param_sec
    zpp.Init._secondary_partition_guard_installed = True
    print('[Zero3Guard] Installed secondary partition bounds guard for ZeRO-3.', flush=True)
    return True


def run_native_dit_forward(dit_module, **dit_kwargs):
    """Run DiT forward in native mode for extraction.

    If `forward` was monkey-patched on the instance, call class forward directly
    to avoid routing through training-only wrappers/new modules.
    """
    if dit_module is None:
        raise ValueError("run_native_dit_forward got None dit_module")

    class_forward = getattr(type(dit_module), 'forward', None)
    if class_forward is None:
        return dit_module(**dit_kwargs)

    if 'forward' in getattr(dit_module, '__dict__', {}):
        return class_forward(dit_module, **dit_kwargs)
    return dit_module(**dit_kwargs)


def _reset_module_runtime_caches(module):
    if module is None:
        return 0

    cleared = 0
    reset_method_names = (
        'reset_cache', 'reset_caches', 'clear_cache', 'clear_caches', '_reset_cache', '_clear_cache'
    )
    cache_attr_names = {
        'cache', '_cache', 'cache_x', '_cache_x', 'cached_x', '_cached_x',
        'causal_cache', '_causal_cache', 'conv_cache', '_conv_cache',
        'temporal_cache', '_temporal_cache', 'kv_cache', '_kv_cache',
        'prev_features', '_prev_features', 'prev_input', '_prev_input',
    }

    submodules = module.modules() if isinstance(module, torch.nn.Module) else [module]
    for submodule in submodules:
        for method_name in reset_method_names:
            reset_fn = getattr(submodule, method_name, None)
            if callable(reset_fn):
                try:
                    reset_fn()
                    cleared += 1
                    break
                except Exception:
                    pass

        for attr_name, value in list(getattr(submodule, '__dict__', {}).items()):
            attr_name_l = str(attr_name).lower()
            if attr_name in cache_attr_names or ('cache' in attr_name_l and not attr_name_l.startswith('__')):
                if torch.is_tensor(value) or isinstance(value, (list, tuple, dict)) or value is not None:
                    try:
                        setattr(submodule, attr_name, None)
                        cleared += 1
                    except Exception:
                        pass
    return cleared


def _get_module_dtype_device(module, default_device=None, default_dtype=None):
    if module is None:
        return default_dtype, default_device
    inner_model = getattr(module, 'model', None)
    if inner_model is not None:
        try:
            param = next(inner_model.parameters())
            return param.dtype, param.device
        except Exception:
            pass
    if hasattr(module, 'dtype') and hasattr(module, 'device'):
        return getattr(module, 'dtype', default_dtype), getattr(module, 'device', default_device)
    try:
        param = next(module.parameters())
        return param.dtype, param.device
    except Exception:
        return default_dtype, default_device


def _get_vae_runtime_dtype(vae, default_dtype=torch.bfloat16):
    if vae is None:
        return default_dtype
    dtype = getattr(vae, 'dtype', None)
    if dtype is not None:
        return dtype
    inner_model = getattr(vae, 'model', None)
    if inner_model is not None:
        try:
            return next(inner_model.parameters()).dtype
        except Exception:
            pass
    try:
        return next(vae.parameters()).dtype
    except Exception:
        return default_dtype


def _move_vae_runtime(vae, device=None, dtype=None):
    if vae is None:
        return vae
    target_device = torch.device(device) if device is not None else torch.device(getattr(vae, 'device', 'cpu'))
    target_dtype = dtype if dtype is not None else getattr(vae, 'dtype', torch.float32)
    if hasattr(vae, 'model'):
        _reset_module_runtime_caches(vae)
        vae.model.to(device=target_device, dtype=target_dtype)
        if hasattr(vae, 'mean') and torch.is_tensor(vae.mean):
            vae.mean = vae.mean.to(device=target_device, dtype=target_dtype)
        if hasattr(vae, 'std') and torch.is_tensor(vae.std):
            vae.std = vae.std.to(device=target_device, dtype=target_dtype)
        if hasattr(vae, 'scale'):
            vae.scale = [vae.mean, 1.0 / vae.std]
        vae.device = target_device
        vae.dtype = target_dtype
        _reset_module_runtime_caches(vae)
        return vae
    if hasattr(vae, 'to'):
        return vae.to(device=target_device, dtype=target_dtype)
    return vae


def _is_dtype_mismatch_error(exc):
    msg = str(exc)
    return (
        ('weight type' in msg and 'Input type' in msg and 'should be the same' in msg)
        or ('expected scalar type' in msg and 'but found' in msg)
    )


def _pick_alternate_fp_dtype(dtype):
    if dtype == torch.float32:
        return torch.bfloat16
    return torch.float32


def _safe_vae_encode_isolated(vae, videos, device=None, op_name='vae.encode', **encode_kwargs):
    """Run one top-level VAE encode in an isolated cache scope.

    This helper does not silently fallback, drop semantics, or retry with an
    altered execution path. It only clears runtime caches before and after an
    independent encode call so stale cross-call cache cannot leak between
    unrelated extractions/training-side condition encodes.
    """
    _reset_module_runtime_caches(vae)
    try:
        try:
            return vae.encode(videos, device=device, **encode_kwargs)
        except TypeError:
            return vae.encode(videos)
    finally:
        _reset_module_runtime_caches(vae)


def _patched_encode_images_adaptive(self, first_frames, random_ref_frame, num_frames, height, width, use_first_aug=False, ref_pad_cfg=False, ref_pad_num=None, num_motion_latent=None):
    """TP2DP2-local patch for SlotMem image conditioning.

    Keep all fixes inside tp2dp2.py instead of editing vendored pipeline code.
    Main changes versus repo version:
    - keep VAE cache-isolated encode
    - avoid in-place module dtype mutation (ZeRO-3-safe)
    - keep explicit output cast to original pipeline dtype
    """
    original_dtype = getattr(self, 'torch_dtype', torch.bfloat16)
    pipe_device = torch.device(getattr(self, 'device', 'cuda'))

    if random_ref_frame is None:
        random_ref_frame = first_frames[0]

    vae = getattr(self, 'vae', None)
    image_encoder = getattr(self, 'image_encoder', None)
    dit = getattr(self, 'dit', None)
    require_clip_embedding = bool(getattr(dit, 'require_clip_embedding', False))
    use_clip_feature = bool(image_encoder is not None and require_clip_embedding)
    if vae is None:
        raise RuntimeError('encode_images_adaptive requires vae')

    vae_dtype, vae_device = _get_module_dtype_device(vae, default_device=pipe_device, default_dtype=original_dtype)
    image_dtype, image_device = _get_module_dtype_device(image_encoder, default_device=pipe_device, default_dtype=original_dtype) if use_clip_feature else (None, pipe_device)
    if vae_dtype is None:
        vae_dtype = original_dtype
    if use_clip_feature and image_dtype is None:
        image_dtype = original_dtype
    vae_device = torch.device(vae_device) if vae_device is not None else pipe_device
    image_device = torch.device(image_device) if image_device is not None else pipe_device

    num_condition_frames = len(first_frames)
    remaining_frames = num_frames - num_condition_frames

    random_ref_tensor = self.preprocess_image(random_ref_frame.resize((width, height))).to(device=vae_device, dtype=vae_dtype)
    clip_context = None
    if use_clip_feature:
        first_frame_base = self.preprocess_image(first_frames[0].resize((width, height))).to(device=image_device)
        first_frame_tensor = first_frame_base.to(dtype=image_dtype)
        try:
            clip_context = image_encoder.encode_image([first_frame_tensor])
        except RuntimeError as e:
            if not _is_dtype_mismatch_error(e):
                raise
            retry_image_dtype = _pick_alternate_fp_dtype(first_frame_tensor.dtype)
            retry_image_dtype, retry_image_device = _get_module_dtype_device(
                image_encoder,
                default_device=image_device,
                default_dtype=retry_image_dtype,
            )
            retry_image_device = torch.device(retry_image_device) if retry_image_device is not None else image_device
            clip_context = image_encoder.encode_image([
                first_frame_base.to(device=retry_image_device, dtype=retry_image_dtype)
            ])

    msk = torch.ones(1, num_frames, height // 8, width // 8, device=vae_device, dtype=vae_dtype)
    if ref_pad_cfg:
        msk[:, len(first_frames):] = 0
    else:
        msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
    msk = msk.transpose(1, 2)[0]

    if len(first_frames) > 1:
        first_frame_list = []
        for frame in first_frames:
            first_frame_list.append(
                self.preprocess_image(frame.resize((width, height)), use_aug=use_first_aug).to(device=vae_device, dtype=vae_dtype)
            )
        vae_input_condition = torch.cat(first_frame_list, dim=0).permute(1, 0, 2, 3)
    else:
        vae_input_condition = self.preprocess_image(first_frames[0].resize((width, height)), use_aug=use_first_aug).to(device=vae_device, dtype=vae_dtype).transpose(0, 1)

    if ref_pad_num == 0:
        vae_input_pad = torch.zeros(3, remaining_frames, height, width, device=vae_device, dtype=vae_dtype)
    elif ref_pad_num is not None and ref_pad_num > 0 and ref_pad_num != -1:
        pad_imgs = []
        random_ref_for_vae = random_ref_tensor.to(device=vae_device, dtype=vae_dtype)
        for _ in range(ref_pad_num):
            pad_imgs.append(random_ref_for_vae.transpose(0, 1))
        if remaining_frames > ref_pad_num:
            pad_imgs += [torch.zeros(3, 1, height, width, device=vae_device, dtype=vae_dtype)] * (remaining_frames - ref_pad_num)
        vae_input_pad = torch.cat(pad_imgs, dim=1) if len(pad_imgs) > 0 else torch.empty(3, 0, height, width, device=vae_device, dtype=vae_dtype)
    elif ref_pad_num == -1:
        random_ref_for_vae = random_ref_tensor.to(device=vae_device, dtype=vae_dtype)
        vae_input_pad = random_ref_for_vae.transpose(0, 1).repeat(1, remaining_frames, 1, 1)
    else:
        vae_input_pad = torch.zeros(3, remaining_frames, height, width, device=vae_device, dtype=vae_dtype)

    vae_input = torch.concat([vae_input_condition, vae_input_pad], dim=1)
    try:
        y_latent = _safe_vae_encode_isolated(vae, [vae_input], device=vae_device, op_name='encode_images_adaptive.vae.encode')[0]
    except RuntimeError as e:
        if not _is_dtype_mismatch_error(e):
            raise
        retry_vae_dtype = _pick_alternate_fp_dtype(vae_input.dtype)
        retry_vae_dtype, retry_vae_device = _get_module_dtype_device(
            vae,
            default_device=vae_device,
            default_dtype=retry_vae_dtype,
        )
        retry_vae_device = torch.device(retry_vae_device) if retry_vae_device is not None else vae_device
        y_latent = _safe_vae_encode_isolated(
            vae,
            [vae_input.to(device=retry_vae_device, dtype=retry_vae_dtype)],
            device=retry_vae_device,
            op_name='encode_images_adaptive.vae.encode.retry',
        )[0]

    if num_motion_latent is not None:
        keep_motion = max(0, int(num_motion_latent))
        keep_latents = min(int(y_latent.shape[1]), 1 + keep_motion)
        if keep_latents < int(y_latent.shape[1]):
            zero_pad = torch.zeros_like(y_latent[:, keep_latents:])
            y_latent = torch.cat([y_latent[:, :keep_latents], zero_pad], dim=1)

    y = torch.concat([msk.to(device=y_latent.device, dtype=y_latent.dtype), y_latent], dim=0).unsqueeze(0)

    if clip_context is not None:
        clip_context = clip_context.to(dtype=original_dtype, device=pipe_device)
    y = y.to(dtype=original_dtype, device=pipe_device)
    return {'clip_feature': clip_context, 'y': y}


def _install_tp2dp2_pipeline_only_patch(pipe):
    if pipe is None:
        return None
    if getattr(pipe, '_tp2dp2_pipeline_only_patch_installed', False):
        return pipe
    pipe.encode_images_adaptive = types.MethodType(_patched_encode_images_adaptive, pipe)
    pipe._tp2dp2_pipeline_only_patch_installed = True
    return pipe


def _install_lightweight_pipeline_lifecycle(pipe, include_denoisers=True):
    if pipe is None:
        return None
    if getattr(pipe, "_lightweight_pipeline_lifecycle_installed", False):
        return pipe

    def _iter_modules():
        seen = set()
        names = ["vae", "image_encoder"]
        if include_denoisers:
            names = ["low_noise_model", "high_noise_model", "dit"] + names
        for name in names:
            module = getattr(pipe, name, None)
            if isinstance(module, torch.nn.Module) and id(module) not in seen:
                seen.add(id(module))
                yield module
        prompter = getattr(pipe, "prompter", None)
        text_encoder = getattr(prompter, "text_encoder", None)
        if isinstance(text_encoder, torch.nn.Module) and id(text_encoder) not in seen:
            yield text_encoder

    def _requires_grad(this, requires_grad=False):
        for module in _iter_modules():
            module.requires_grad_(requires_grad)
        return this

    def _train(this, mode=True):
        this.training = bool(mode)
        for module in _iter_modules():
            module.train(mode)
        return this

    def _eval(this):
        return _train(this, False)

    pipe.training = bool(getattr(pipe, "training", True))
    pipe.requires_grad_ = types.MethodType(_requires_grad, pipe)
    pipe.train = types.MethodType(_train, pipe)
    pipe.eval = types.MethodType(_eval, pipe)
    pipe._lightweight_pipeline_lifecycle_installed = True
    return pipe


class SharedVAEPipelineView:
    """Lightweight VAE/prompt view backed by the training pipeline.

    This avoids constructing a second full Wan2.2 runtime just to access
    prompt encoding, VAE/image conditioning, and preprocessing helpers.
    """

    def __init__(self, base_pipe, tiler_kwargs=None):
        self.base_pipe = base_pipe
        self.training = True
        self.device = getattr(base_pipe, "device", "cuda")
        self.torch_dtype = getattr(base_pipe, "torch_dtype", torch.bfloat16)
        self.prompter = getattr(base_pipe, "prompter", None)
        self.vae = getattr(base_pipe, "vae", None)
        self.image_encoder = getattr(base_pipe, "image_encoder", None)
        self.tiler_kwargs = dict(tiler_kwargs or {})

    def __getattr__(self, name):
        return getattr(self.base_pipe, name)

    def _iter_managed_modules(self):
        text_encoder = getattr(self.prompter, "text_encoder", None)
        for module in (self.vae, self.image_encoder, text_encoder):
            if isinstance(module, torch.nn.Module):
                yield module

    def requires_grad_(self, requires_grad=False):
        for module in self._iter_managed_modules():
            module.requires_grad_(requires_grad)
        return self

    def train(self, mode=True):
        self.training = bool(mode)
        for module in self._iter_managed_modules():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def denoising_model(self):
        return self.base_pipe.denoising_model()

    def encode_prompt(self, prompt, positive=True):
        self.base_pipe.device = self.device
        return self.prompter.encode_prompt(prompt, positive=positive, device=self.device)

    def preprocess_image(self, image, use_aug=False):
        del use_aug
        return self.base_pipe.preprocess_image(image)

    def prepare_extra_input(self, latents=None):
        return self.base_pipe.prepare_extra_input(latents)

    def encode_images_adaptive(
        self,
        first_frames,
        random_ref_frame,
        num_frames,
        height,
        width,
        use_first_aug=False,
        ref_pad_cfg=False,
        ref_pad_num=None,
        num_motion_latent=None,
    ):
        self.base_pipe.device = self.device
        self.base_pipe.torch_dtype = self.torch_dtype
        return self.base_pipe.encode_images_adaptive(
            first_frames=first_frames,
            random_ref_frame=random_ref_frame,
            num_frames=num_frames,
            height=height,
            width=width,
            use_first_aug=use_first_aug,
            ref_pad_cfg=ref_pad_cfg,
            ref_pad_num=ref_pad_num,
            num_motion_latent=num_motion_latent,
        )


def _sample_timestep_for_domain(sched_timesteps, train_noise_domain, num_train_timesteps, boundary_ratio):
    if sched_timesteps is None or len(sched_timesteps) == 0:
        return None
    domain = str(train_noise_domain or "low_noise").strip().lower()
    if domain not in ("low_noise", "high_noise"):
        rand_idx = random.randint(0, len(sched_timesteps) - 1)
        return float(sched_timesteps[rand_idx].item())

    threshold = float(boundary_ratio) * float(max(int(num_train_timesteps), 1))
    candidates = []
    for t in sched_timesteps:
        t_val = float(t.item())
        if domain == "low_noise" and t_val < threshold:
            candidates.append(t_val)
        elif domain == "high_noise" and t_val >= threshold:
            candidates.append(t_val)
    if len(candidates) == 0:
        rand_idx = random.randint(0, len(sched_timesteps) - 1)
        return float(sched_timesteps[rand_idx].item())
    return float(random.choice(candidates))


def _previous_scheduler_timestep_value(sched_timesteps, current_timestep):
    if sched_timesteps is None or len(sched_timesteps) == 0:
        return None
    try:
        cur = float(current_timestep.detach().float().reshape(-1)[0].item()) if isinstance(current_timestep, torch.Tensor) else float(current_timestep)
        values = [
            float(t.detach().float().reshape(-1)[0].item()) if isinstance(t, torch.Tensor) else float(t)
            for t in sched_timesteps
        ]
        if len(values) == 0:
            return None
        nearest_idx = min(range(len(values)), key=lambda idx: abs(values[idx] - cur))
        prev_idx = min(int(nearest_idx) + 1, len(values) - 1)
        return float(values[prev_idx])
    except Exception:
        return None


class LearnableMemoryEmbeddings(torch.nn.Module):
    """ZeRO-3 friendly wrapper for learnable memory embeddings.

    Keep learnable memory embeddings inside a submodule and access them through
    this module's forward so DeepSpeed can gather/shard them consistently.
    Avoid calling `.to(...)` directly on bare Parameters inside the parent
    module forward.
    """

    def __init__(self, patch_dim, max_memory_characters, max_total_memory_tokens,
                 use_segment_embed=False, use_learnable_memory_pos=False):
        super().__init__()
        self.use_segment_embed = bool(use_segment_embed)
        self.use_learnable_memory_pos = bool(use_learnable_memory_pos)
        self.max_memory_characters = int(max_memory_characters)
        self.max_total_memory_tokens = int(max_total_memory_tokens)

        if self.use_segment_embed:
            self.segment_embed = torch.nn.Parameter(
                torch.zeros(1, self.max_memory_characters, patch_dim)
            )
            torch.nn.init.normal_(self.segment_embed, mean=0.0, std=0.02)
        else:
            self.register_parameter("segment_embed", None)

        if self.use_learnable_memory_pos:
            self.pos_embed = torch.nn.Parameter(
                torch.zeros(1, self.max_total_memory_tokens, patch_dim)
            )
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_parameter("pos_embed", None)

    def _module_dtype_device(self, fallback_dtype, fallback_device):
        for p in self.parameters():
            if p is not None:
                return p.dtype, p.device
        return fallback_dtype, fallback_device

    def forward(self, memory_projected, memory_token_lengths_per_character=None,
                add_segment=False, add_pos=False):
        if memory_projected is None:
            return memory_projected
        if not add_segment and not add_pos:
            return memory_projected

        orig_dtype = memory_projected.dtype
        orig_device = memory_projected.device
        param_dtype, param_device = self._module_dtype_device(orig_dtype, orig_device)

        x = memory_projected
        if x.device != param_device or x.dtype != param_dtype:
            x = x.to(device=param_device, dtype=param_dtype)

        if add_segment and self.segment_embed is not None:
            if memory_token_lengths_per_character and len(memory_token_lengths_per_character) > 0:
                segment_ids = torch.cat([
                    torch.full((int(L),), i, device=x.device, dtype=torch.long)
                    for i, L in enumerate(memory_token_lengths_per_character)
                ], dim=0)
                n_seg = x.shape[1]
                if segment_ids.shape[0] < n_seg:
                    last_seg = max(len(memory_token_lengths_per_character) - 1, 0)
                    segment_ids = torch.cat([
                        segment_ids,
                        torch.full((n_seg - segment_ids.shape[0],), last_seg, device=x.device, dtype=torch.long)
                    ], dim=0)
                elif segment_ids.shape[0] > n_seg:
                    segment_ids = segment_ids[:n_seg]
                x = x + self.segment_embed[0, segment_ids, :].unsqueeze(0)
            else:
                x = x + self.segment_embed[:, 0:1, :]

        if add_pos and self.pos_embed is not None:
            n_actual = x.shape[1]
            max_pos = self.pos_embed.shape[1]
            if n_actual > max_pos:
                raise ValueError(f"Memory token count {n_actual} exceeds maximum supported {max_pos}.")
            x = x + self.pos_embed[:, :n_actual, :]

        if x.device != orig_device or x.dtype != orig_dtype:
            x = x.to(device=orig_device, dtype=orig_dtype)
        return x


import pandas as pd
import matplotlib.pyplot as plt

class LossLogger:
    def __init__(self, save_dir, inherit_history=True):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.csv_path = os.path.join(save_dir, "loss_history.csv")
        self.png_path = os.path.join(save_dir, "loss_curve.png")
        self.png_path_no_bbox_x2 = os.path.join(save_dir, "loss_curve_no_bbox_x2.png")
        self.history = []
        self.last_step = -1
        if inherit_history and os.path.isfile(self.csv_path):
            try:
                existing_df = pd.read_csv(self.csv_path)
                self.history = existing_df.to_dict(orient='records')
                if 'step' in existing_df.columns:
                    step_series = pd.to_numeric(existing_df['step'], errors='coerce').dropna()
                    if len(step_series) > 0:
                        self.last_step = int(step_series.max())
            except Exception:
                self.history = []
                self.last_step = -1
        
    def log(self, step, loss, used_memory=None, memory_dropped=None,
            image_condition_available=None, image_condition_used=None, image_condition_dropped=None,
            memory_condition_available=None, memory_condition_used=None,
            loss_no_bbox_x2=None,
            folder_name=None,
            sample_idx=None, video_id=None, group_index=None,
            core_clip_index=None, memory_clip_index=None, diffusion_timestep=None,
            selected_bank_idx=None, selected_bank_percent=None,
            p_cur=None, p_fusion=None,
            fusion_alpha=None, fusion_quantile=None, fusion_max_inject_ratio=None,
            fusion_inject_ratio=None, fusion_sim1_mean=None, fusion_tau_sim=None,
            memory_bank_count=None, memory_bank_percents=None):
        self.history.append({
            "step": step,
            "loss": loss,
            "diffusion_timestep": diffusion_timestep,
            "used_memory": used_memory,
            "memory_dropped": memory_dropped,
            "image_condition_available": image_condition_available,
            "image_condition_used": image_condition_used,
            "image_condition_dropped": image_condition_dropped,
            "memory_condition_available": memory_condition_available,
            "memory_condition_used": memory_condition_used,
            "loss_no_bbox_x2": loss_no_bbox_x2,
            "folder_name": folder_name,
            "sample_idx": sample_idx,
            "video_id": video_id,
            "group_index": group_index,
            "core_clip_index": core_clip_index,
            "memory_clip_index": memory_clip_index,
            "selected_bank_idx": selected_bank_idx,
            "selected_bank_percent": selected_bank_percent,
            "p_cur": p_cur,
            "p_fusion": p_fusion,
            "fusion_alpha": fusion_alpha,
            "fusion_quantile": fusion_quantile,
            "fusion_max_inject_ratio": fusion_max_inject_ratio,
            "fusion_inject_ratio": fusion_inject_ratio,
            "fusion_sim1_mean": fusion_sim1_mean,
            "fusion_tau_sim": fusion_tau_sim,
            "memory_bank_count": memory_bank_count,
            "memory_bank_percents": memory_bank_percents,
        })
        
    def save(self):
        if not self.history:
            return
        
        # 1. Save CSV
        df = pd.DataFrame(self.history)
        df.to_csv(self.csv_path, index=False)

        def _to_float(v):
            try:
                if v is None:
                    return None
                if isinstance(v, str) and v.strip() == "":
                    return None
                fv = float(v)
                if np.isnan(fv) or np.isinf(fv):
                    return None
                return fv
            except Exception:
                return None

        rows = []
        for rec in self.history:
            if not isinstance(rec, dict):
                continue
            step_v = _to_float(rec.get('step'))
            if step_v is None:
                continue
            loss_v = _to_float(rec.get('loss'))
            loss_no_bbox_v = _to_float(rec.get('loss_no_bbox_x2'))
            rows.append((step_v, loss_v, loss_no_bbox_v))

        if len(rows) == 0:
            return

        rows.sort(key=lambda x: x[0])
        x_step = np.asarray([r[0] for r in rows], dtype=np.float64)
        y_loss = np.asarray([
            min(r[1], 0.4) if r[1] is not None else np.nan
            for r in rows
        ], dtype=np.float64)
        y_loss_no_bbox = np.asarray([
            min(r[2], 0.4) if r[2] is not None else np.nan
            for r in rows
        ], dtype=np.float64)

        def _moving_avg_nan(arr, window=20):
            out = np.full_like(arr, np.nan, dtype=np.float64)
            n = int(arr.shape[0])
            for i in range(n):
                lo = max(0, i - int(window) + 1)
                seg = arr[lo:i + 1]
                valid = seg[np.isfinite(seg)]
                if valid.size > 0:
                    out[i] = float(valid.mean())
            return out

        # 2. Plot PNGs (split into two figures)
        valid_loss = np.isfinite(y_loss)
        if bool(valid_loss.any()):
            plt.figure(figsize=(10, 6))
            plt.plot(
                x_step[valid_loss],
                y_loss[valid_loss],
                label='Train Loss (bbox x2)',
                linewidth=1.1,
                color='#1f77b4',
            )
            if len(x_step) > 20:
                loss_ma = _moving_avg_nan(y_loss, window=20)
                valid_loss_ma = np.isfinite(loss_ma)
                plt.plot(
                    x_step[valid_loss_ma],
                    loss_ma[valid_loss_ma],
                    label='Train Loss (bbox x2) MA20',
                    color='#1565c0',
                    linewidth=1.5,
                    alpha=0.9,
                )
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve (bbox x2)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.png_path)
            plt.close()

        valid_loss_no_bbox = np.isfinite(y_loss_no_bbox)
        if bool(valid_loss_no_bbox.any()):
            plt.figure(figsize=(10, 6))
            plt.plot(
                x_step[valid_loss_no_bbox],
                y_loss_no_bbox[valid_loss_no_bbox],
                label='Train Loss (no bbox x2)',
                linewidth=1.1,
                color='#ff7f0e',
            )
            if len(x_step) > 20:
                loss_no_bbox_ma = _moving_avg_nan(y_loss_no_bbox, window=20)
                valid_loss_no_bbox_ma = np.isfinite(loss_no_bbox_ma)
                plt.plot(
                    x_step[valid_loss_no_bbox_ma],
                    loss_no_bbox_ma[valid_loss_no_bbox_ma],
                    label='Train Loss (no bbox x2) MA20',
                    color='#ef6c00',
                    linewidth=1.5,
                    alpha=0.9,
                )
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve (no bbox x2)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.png_path_no_bbox_x2)
            plt.close()
# =============================================================================
# Performance Timing Tracker
# =============================================================================

class TimingTracker:
    """Tracks timing statistics for extraction and training."""
    
    def __init__(self, window_size=4):
        self.window_size = window_size
        self.extraction_times = deque(maxlen=window_size)
        self.training_times = deque(maxlen=window_size)
        self.wait_times = deque(maxlen=window_size)
        self._start_times = {}
    
    def start(self, name):
        self._start_times[name] = time.time()
    
    def end(self, name):
        if name in self._start_times:
            elapsed = time.time() - self._start_times[name]
            if name == 'extraction':
                self.extraction_times.append(elapsed)
            elif name == 'training':
                self.training_times.append(elapsed)
            elif name == 'wait':
                self.wait_times.append(elapsed)
            del self._start_times[name]
            return elapsed
        return 0
    
    def get_avg(self, name):
        if name == 'extraction':
            times = self.extraction_times
        elif name == 'training':
            times = self.training_times
        elif name == 'wait':
            times = self.wait_times
        else:
            return 0
        return sum(times) / len(times) if times else 0
    
    def get_stats_string(self):
        """Get formatted stats string for progress bar."""
        ext_avg = self.get_avg('extraction') * 1000  # ms
        train_avg = self.get_avg('training') * 1000  # ms
        wait_avg = self.get_avg('wait') * 1000  # ms
        return f"ext:{ext_avg:.0f}ms train:{train_avg:.0f}ms wait:{wait_avg:.0f}ms"


# =============================================================================
# Residual Memory Projector (Zero-initialized)
# =============================================================================
class AdaLNBlock(torch.nn.Module):
    """
    Timestep-aware gating block.
    Logic: Norm(x) * (1 + Scale(t)) + Shift(t)
    """
    def __init__(self, dim, time_embed_dim):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(time_embed_dim, dim * 2)
        # Small initialization instead of zero-init for better gradient flow
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x, t_emb):
        # x: [B, N, D]
        # t_emb: [B, T_dim]
        style = self.linear(self.silu(t_emb)).unsqueeze(1) # [B, 1, 2D]
        scale, shift = style.chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + scale) + shift
        return x

class StatsAdaptiveNorm(torch.nn.Module):
    """
    Distribution-aware alignment block (AdaIN-like).
    Logic: 
      1. Extract mean/std from noisy_latent (stats_dim)
      2. Map stats to affine parameters (dim) using MLP
      3. Apply affine transform to x
    """
    def __init__(self, dim, stats_dim):
        super().__init__()
        # Elementwise_affine=False because we apply our own affine from stats
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine=False) 
        
        # Input is (mean, std) -> 2 * stats_dim
        self.stats_proj = torch.nn.Sequential(
            torch.nn.Linear(stats_dim * 2, dim),
            torch.nn.SiLU(),
            torch.nn.Linear(dim, dim * 2) # Output Scale & Shift
        )
        
        # Small initialization instead of zero-init for better gradient flow
        # First layer uses default initialization, last layer uses small init
        torch.nn.init.normal_(self.stats_proj[-1].weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.stats_proj[-1].bias)

    def forward(self, x, z_mean, z_std):
        # x: [B, N, D]
        # z_mean, z_std: [B, C_lat]
        # 确保z_mean和z_std的dtype与stats_proj的权重dtype一致
        target_dtype = self.stats_proj[0].weight.dtype
        z_mean = z_mean.to(dtype=target_dtype)
        z_std = z_std.to(dtype=target_dtype)
        stats = torch.cat([z_mean, z_std], dim=1) # [B, 2*C_lat]
        style = self.stats_proj(stats).unsqueeze(1) # [B, 1, 2D]
        scale, shift = style.chunk(2, dim=-1)
        
        x = self.norm(x)
        x = x * (1 + scale) + shift # Affine transform based on z_t stats
        return x

class StyleAwareMemoryProjector(torch.nn.Module):
    """
    V7 Projector: Semantics-First, Style-Last.
    Structure: InputProj -> AdaLN(t) -> Act -> MidProj -> AdaIN(z_t) -> OutputProj(Zero)
    [修改] 添加时间门控机制：高噪声阶段（高t）基本不调制，低噪声阶段（低t）略微调制
    """
    def __init__(self, dim=5120, time_embed_dim=256, latent_dim=16, bottleneck_dim=256, dropout=0.05):
        super().__init__()
        
        # 1. Input Projection
        self.input_proj = torch.nn.Linear(dim, bottleneck_dim)
        
        # 2. Semantic Gating (Time-Aware)
        self.adaln_block = AdaLNBlock(bottleneck_dim, time_embed_dim)
        self.act = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(dropout)
        
        # 3. Intermediate Projection
        self.mid_proj = torch.nn.Linear(bottleneck_dim, bottleneck_dim)
        
        # 4. Style Alignment (Distribution-Aware)
        self.style_norm = StatsAdaptiveNorm(bottleneck_dim, stats_dim=latent_dim)
        
        # 5. [新增] 分离的时间门控机制：AdaLN和StyleNorm使用不同的门控策略
        # AdaLN (语义门控): 高t时保持较小但不为0的值，允许语义信息在高噪声时也有贡献，保持梯度流
        # StyleNorm (分布对齐): 高t时更严格抑制，因为高噪声时分布不稳定
        # 
        # AdaLN门控初始化：scale=4.0, bias=-1.0
        # - 高t时：t_normalized≈1, gate_raw = 4*(1-1) - 1 = -1, sigmoid(-1)≈0.27 (保持一定梯度)
        # - 低t时：t_normalized≈0, gate_raw = 4*(1-0) - 1 = 3, sigmoid(3)≈0.95 (几乎全开)
        self.time_gate_scale_adaln = torch.nn.Parameter(torch.tensor(4.0))  # AdaLN门控缩放
        self.time_gate_bias_adaln = torch.nn.Parameter(torch.tensor(-1.0))  # AdaLN门控偏移
        
        # StyleNorm门控初始化：scale=6.0, bias=-3.0 (更严格)
        # - 高t时：t_normalized≈1, gate_raw = 6*(1-1) - 3 = -3, sigmoid(-3)≈0.05 (严格抑制)
        # - 低t时：t_normalized≈0, gate_raw = 6*(1-0) - 3 = 3, sigmoid(3)≈0.95 (几乎全开)
        self.time_gate_scale_style = torch.nn.Parameter(torch.tensor(6.0))  # StyleNorm门控缩放（更严格）
        self.time_gate_bias_style = torch.nn.Parameter(torch.tensor(-3.0))  # StyleNorm门控偏移（更严格）
        
        # 6. Output Projection (Zero-Initialized)
        self.output_proj = torch.nn.Linear(bottleneck_dim, dim)
        torch.nn.init.zeros_(self.output_proj.weight)
        torch.nn.init.zeros_(self.output_proj.bias)

    def forward(self, memory, t_emb, noisy_latents, condition_mask=None, timestep=None, num_train_timesteps=1000):
        """
        memory: [B, N, D]
        t_emb: [B, T_dim]
        noisy_latents: [B, C_lat, F, H, W]
        condition_mask: [B, F, H, W] or None, 1表示条件帧位置（应排除），0表示非条件帧位置（应统计）
        timestep: [B] or scalar, 原始时间步，用于计算门控
        num_train_timesteps: int, 训练时的总时间步数，默认1000
        """
        # Residual connection base
        identity = memory
        
        # A. Feature Extraction
        x = self.input_proj(memory) # [B, N, D_bot]
        
        # B. Semantic Gating (Time)
        # Selects WHAT to remember based on denoising stage
        x_before_adaln = x  # 保存AdaLN前的值，用于门控
        x = self.adaln_block(x, t_emb)
        # [新增] AdaLN时间门控：高噪声时保持较小但不为0的值，允许语义信息贡献并保持梯度流
        # gate_adaln: [B, 1]，高t时≈0.27（保持梯度），低t时≈0.95（几乎全开）
        if timestep is not None:
            # 将 timestep 归一化到 [0, 1]，高t（高噪声）时接近1，低t（低噪声）时接近0
            if isinstance(timestep, torch.Tensor):
                if timestep.dim() == 0:
                    timestep = timestep.unsqueeze(0)
                if timestep.dim() == 1 and timestep.shape[0] == 1:
                    # [1] -> [B]
                    timestep = timestep.expand(x.shape[0])
            else:
                timestep = torch.tensor([timestep], device=x.device, dtype=x.dtype).expand(x.shape[0])
            
            # 归一化：高t时接近1，低t时接近0
            t_normalized = timestep.float() / num_train_timesteps  # [B]
            # AdaLN门控：保持一定梯度流，即使在高噪声时
            gate_raw_adaln = self.time_gate_scale_adaln * (1.0 - t_normalized) + self.time_gate_bias_adaln  # [B]
            gate_adaln = torch.sigmoid(gate_raw_adaln).unsqueeze(1).unsqueeze(1).to(x.dtype)  # [B, 1, 1]
        else:
            # 如果没有提供 timestep，使用默认门控（全开，即 gate=1）
            gate_adaln = torch.ones(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
        
        # 门控混合：gate_adaln在高t时保持约0.27，允许语义信息贡献并保持梯度流
        x = gate_adaln * x + (1.0 - gate_adaln) * x_before_adaln
        
        x = self.act(x)
        x = self.dropout(x)
        # 确保 x 的数据类型与 mid_proj 权重的数据类型一致
        x = x.to(self.mid_proj.weight.dtype)
        x = self.mid_proj(x)
        
        # C. Style Alignment (Latent Stats)
        # Adapts HOW it looks based on current noise distribution
        with torch.no_grad():
            if condition_mask is not None:
                # 只统计非条件帧位置（mask=0的位置）
                # condition_mask: [B, F, H, W], 需要扩展到 [B, C_lat, F, H, W]
                B, C_lat, F, H, W = noisy_latents.shape
                # 确保mask的shape匹配
                if condition_mask.shape[1:] != (F, H, W):
                    # 如果mask的F维度不匹配，可能需要调整
                    # 假设mask的第一帧是条件帧
                    if condition_mask.shape[1] == 1:
                        # 单帧mask，扩展到所有帧
                        condition_mask = condition_mask.expand(B, F, H, W)
                    else:
                        # 截断或填充到F
                        mask_F = condition_mask.shape[1]
                        if mask_F < F:
                            # 填充：假设后续帧都是非条件帧
                            pad = torch.zeros(B, F - mask_F, H, W, 
                                             device=condition_mask.device, 
                                             dtype=condition_mask.dtype)
                            condition_mask = torch.cat([condition_mask, pad], dim=1)
                        else:
                            condition_mask = condition_mask[:, :F, :, :]
                
                # 扩展mask到C_lat维度: [B, F, H, W] -> [B, 1, F, H, W] -> [B, C_lat, F, H, W]
                mask_expanded = condition_mask.unsqueeze(1).expand(B, C_lat, F, H, W)
                # 反转mask：1表示条件帧（排除），0表示非条件帧（保留）
                # 我们想要mask=0的位置，所以用 (1 - mask)
                valid_mask = (1 - mask_expanded).bool()
                
                # 只对valid位置计算统计
                if valid_mask.any():
                    # 将invalid位置设为0，然后计算统计（但需要归一化）
                    masked_latents = noisy_latents * valid_mask.float()
                    
                    # 计算有效位置的数量：在dim=(2,3,4)上sum
                    B, C_lat = noisy_latents.shape[0], noisy_latents.shape[1]
                    valid_count = valid_mask.sum(dim=(2, 3, 4), keepdim=True)  # [B, C_lat, 1] 或 [B, C_lat, 1, 1, 1]
                    valid_count = torch.clamp(valid_count, min=1)
                    valid_count_flat = valid_count.view(B, C_lat)  # 强制reshape为[B, C_lat]
                    
                    # 计算均值：在dim=(2,3,4)上sum，得到[B, C_lat]
                    masked_sum = masked_latents.sum(dim=(2, 3, 4), keepdim=False)  # [B, C_lat]
                    mean = masked_sum / valid_count_flat  # [B, C_lat]
                    
                    # 计算方差
                    mean_expanded = mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, C_lat, 1, 1, 1]
                    centered = (noisy_latents - mean_expanded) * valid_mask.float()
                    var_sum = (centered ** 2).sum(dim=(2, 3, 4), keepdim=False)  # [B, C_lat]
                    var = var_sum / valid_count_flat  # [B, C_lat]
                    std = torch.sqrt(var + 1e-5)  # [B, C_lat]
                else:
                    # 如果没有有效位置，回退到全量统计
                    var, mean = torch.var_mean(noisy_latents, dim=(2, 3, 4), unbiased=False)
                    std = torch.sqrt(var + 1e-5)
                    # 确保shape为[B, C_lat]，即使C_lat=1也要保持2D
                    if mean.dim() == 1:
                        mean = mean.unsqueeze(-1)  # [B] -> [B, 1] (当C_lat=1时)
                    if std.dim() == 1:
                        std = std.unsqueeze(-1)  # [B] -> [B, 1] (当C_lat=1时)
            else:
                # 没有mask，使用全量统计（原始行为）
                # Calculate stats over spatial+temporal dimensions (F, H, W)
                # noisy_latents is [B, C, F, H, W] -> dims (2, 3, 4)
                var, mean = torch.var_mean(noisy_latents, dim=(2, 3, 4), unbiased=False)
                std = torch.sqrt(var + 1e-5)
                # 确保shape为[B, C_lat]，即使C_lat=1也要保持2D
                if mean.dim() == 1:
                    mean = mean.unsqueeze(-1)  # [B] -> [B, 1] (当C_lat=1时)
                if std.dim() == 1:
                    std = std.unsqueeze(-1)  # [B] -> [B, 1] (当C_lat=1时)
            
        # 确保mean和std的dtype与style_norm的权重dtype一致
        target_dtype = self.style_norm.stats_proj[0].weight.dtype
        mean = mean.to(dtype=target_dtype)
        std = std.to(dtype=target_dtype)
        
        # [新增] StyleNorm时间门控：高噪声时更严格抑制分布对齐，因为高噪声时分布不稳定
        x_before_style = x  # 保存StyleNorm前的值，用于门控
        x = self.style_norm(x, mean, std)
        # 使用独立的StyleNorm门控，比AdaLN更严格
        if timestep is not None:
            # 重新计算t_normalized（如果之前已经计算过，可以复用，但为了清晰这里重新计算）
            if isinstance(timestep, torch.Tensor):
                if timestep.dim() == 0:
                    timestep = timestep.unsqueeze(0)
                if timestep.dim() == 1 and timestep.shape[0] == 1:
                    timestep = timestep.expand(x.shape[0])
            else:
                timestep = torch.tensor([timestep], device=x.device, dtype=x.dtype).expand(x.shape[0])
            
            t_normalized = timestep.float() / num_train_timesteps  # [B]
            # StyleNorm门控：更严格抑制，高t时≈0.05（几乎关闭），低t时≈0.95（几乎全开）
            gate_raw_style = self.time_gate_scale_style * (1.0 - t_normalized) + self.time_gate_bias_style  # [B]
            gate_style = torch.sigmoid(gate_raw_style).unsqueeze(1).unsqueeze(1).to(x.dtype)  # [B, 1, 1]
        else:
            gate_style = torch.ones(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
        
        # 门控混合：gate_style在高t时≈0.05，严格抑制分布对齐
        x = gate_style * x + (1.0 - gate_style) * x_before_style
        
        # D. Output Projection (Residual)
        out = self.output_proj(x)
        
        return identity + out


def pick_nearest_bank_by_percent(current_percent, bank_percents):
    """Pick nearest bank index by |p_cur - p_bank|."""
    if not bank_percents:
        return 0
    p_cur = float(current_percent)
    best_idx = 0
    best_dist = float("inf")
    for idx, p in enumerate(bank_percents):
        d = abs(float(p) - p_cur)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    return best_idx


def parse_float_csv(value, default_list):
    if value is None:
        return [float(x) for x in default_list]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                print(f"[Warning] Invalid float list '{value}', fallback to default {default_list}")
                return [float(x) for x in default_list]
        return out
    parts = [x.strip() for x in str(value).split(',') if str(x).strip() != '']
    if len(parts) == 0:
        print(f"[Warning] Empty float csv '{value}', fallback to default {default_list}")
        return [float(x) for x in default_list]

    parsed = []
    for item in parts:
        try:
            parsed.append(float(item))
        except (TypeError, ValueError):
            print(f"[Warning] Invalid float csv '{value}', fallback to default {default_list}")
            return [float(x) for x in default_list]
    if len(parsed) == 0:
        print(f"[Warning] Empty float csv '{value}', fallback to default {default_list}")
        return [float(x) for x in default_list]
    return parsed


class CharacterWiseCrossAttention(torch.nn.Module):
    """Character-wise cross-attention from selected video tokens to role memory.

    Query selection is strictly driven by probe payload flat_idx. Role boxes are
    used only as geometric priors (uv); missing bbox does not drop selected
    query tokens.
    """

    def __init__(
        self,
        dim=5120,
        num_heads=8,
        head_dim=128,
        rope_dim=256,
        use_half_role_heads=True,
        max_query_tokens_per_role=0,
        query_chunk_size=128,
        use_memory_side_rope=True,
        add_rope_center_to_value=True,
        init_scale=0.1,
        time_gate=True,
        debug=False,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = max(1, int(num_heads))
        self.head_dim = max(1, int(head_dim))
        self.inner_dim = int(self.num_heads * self.head_dim)
        self.rope_dim = max(0, int(rope_dim))
        self.use_half_role_heads = bool(use_half_role_heads)
        # Kept only for CLI compatibility; dynamic-query path never pads.
        self.max_query_tokens_per_role = max(0, int(max_query_tokens_per_role))
        self.query_chunk_size = max(0, int(query_chunk_size))
        self.use_memory_side_rope = bool(use_memory_side_rope)
        self.add_rope_center_to_value = bool(add_rope_center_to_value)
        self.time_gate = bool(time_gate)
        self.debug = bool(debug)

        self.q_proj = torch.nn.Linear(self.dim, self.inner_dim)
        self.k_proj = torch.nn.Linear(self.dim, self.inner_dim)
        self.v_proj = torch.nn.Linear(self.dim, self.inner_dim)
        self.out_proj = torch.nn.Linear(self.inner_dim, self.dim)
        self.scale = float(self.head_dim) ** -0.5

        self.output_scale = torch.nn.Parameter(torch.tensor(float(init_scale)))
        self.time_gate_scale = torch.nn.Parameter(torch.tensor(4.0))
        self.time_gate_bias = torch.nn.Parameter(torch.tensor(-2.0))

    def _build_video_token_grid(self, num_tokens, latent_h, latent_w, device):
        if latent_h is None or latent_w is None:
            return None
        h = int(latent_h)
        w = int(latent_w)
        if h <= 0 or w <= 0:
            return None
        spatial = h * w
        if spatial <= 0:
            return None
        token_ids = torch.arange(int(num_tokens), device=device, dtype=torch.long)
        t_idx = token_ids // spatial
        sp_idx = token_ids % spatial
        y_idx = sp_idx // w
        x_idx = sp_idx % w
        return t_idx, y_idx, x_idx

    def _compute_uv_from_bbox(self, x_center, y_center, bbox_xyxy, eps=1e-6):
        if bbox_xyxy is None or len(bbox_xyxy) != 4:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        except Exception:
            return None
        bw = max(x2 - x1, eps)
        bh = max(y2 - y1, eps)
        if bw <= eps or bh <= eps:
            return None
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        u = (float(x_center) - cx) / (0.5 * bw)
        v = (float(y_center) - cy) / (0.5 * bh)
        return float(u), float(v)

    def _build_query_uv_for_role_from_selected_idx(
        self,
        role_id,
        selected_idx,
        query_role_boxes,
        bsz,
        num_video_tokens,
        latent_h,
        latent_w,
        device,
        dtype,
    ):
        uv = torch.zeros((bsz, int(selected_idx.numel()), 2), device=device, dtype=dtype)
        valid = torch.zeros((bsz, int(selected_idx.numel())), device=device, dtype=torch.bool)
        if selected_idx is None or selected_idx.numel() == 0:
            return uv, valid
        if not isinstance(query_role_boxes, dict) or len(query_role_boxes) == 0:
            return uv, valid
        role_boxes = query_role_boxes.get(str(role_id), None)
        if role_boxes is None:
            return uv, valid
        grid = self._build_video_token_grid(num_video_tokens, latent_h, latent_w, device=device)
        if grid is None:
            return uv, valid

        t_idx, y_idx, x_idx = grid
        for j in range(int(selected_idx.numel())):
            flat_i = int(selected_idx[j].item())
            if flat_i < 0 or flat_i >= int(num_video_tokens):
                continue
            lt = int(t_idx[flat_i].item())
            lh = float(y_idx[flat_i].item()) + 0.5
            lw = float(x_idx[flat_i].item()) + 0.5
            bbox = None
            if isinstance(role_boxes, dict):
                bbox = role_boxes.get(int(lt), None)
                if bbox is None:
                    bbox = role_boxes.get(str(int(lt)), None)
            parsed = self._compute_uv_from_bbox(lw, lh, bbox)
            if parsed is None:
                continue
            uv[:, j, 0] = float(parsed[0])
            uv[:, j, 1] = float(parsed[1])
            valid[:, j] = True
        return uv, valid

    def _build_memory_uv_for_role(self, role_meta, bsz, num_mem_tokens, device, dtype):
        uv = torch.zeros((bsz, int(num_mem_tokens), 2), device=device, dtype=dtype)
        valid = torch.zeros((bsz, int(num_mem_tokens)), device=device, dtype=torch.bool)
        if not isinstance(role_meta, list) or len(role_meta) <= 0:
            return uv, valid
        use_n = min(int(num_mem_tokens), len(role_meta))
        for i in range(use_n):
            m = role_meta[i]
            if not isinstance(m, dict):
                continue
            try:
                uv[:, i, 0] = float(m.get('u', 0.0))
                uv[:, i, 1] = float(m.get('v', 0.0))
                valid[:, i] = bool(m.get('inside_box', False))
            except Exception:
                continue
        return uv, valid

    def _split_memory_by_role(self, memory_tokens, memory_token_meta=None, memory_token_lengths_per_character=None, query_role_boxes=None):
        role_chunks = []
        if memory_tokens is None or memory_tokens.ndim < 3 or int(memory_tokens.shape[1]) <= 0:
            return role_chunks
        n_mem = int(memory_tokens.shape[1])

        if isinstance(memory_token_meta, list) and len(memory_token_meta) > 0:
            grouped_indices = {}
            grouped_meta = {}
            use_n = min(n_mem, len(memory_token_meta))
            for i in range(use_n):
                item = memory_token_meta[i] if isinstance(memory_token_meta[i], dict) else None
                if item is None:
                    continue
                role_id = str(item.get('char_id', '')).strip()
                if role_id == '':
                    continue
                grouped_indices.setdefault(role_id, []).append(i)
                grouped_meta.setdefault(role_id, []).append(item)
            for role_id in sorted(grouped_indices.keys()):
                idx_t = torch.tensor(grouped_indices[role_id], device=memory_tokens.device, dtype=torch.long)
                if idx_t.numel() > 0:
                    role_chunks.append((role_id, idx_t, grouped_meta.get(role_id, [])))
            if len(role_chunks) > 0:
                return role_chunks

        if isinstance(memory_token_lengths_per_character, (list, tuple)) and len(memory_token_lengths_per_character) > 0:
            role_ids = []
            if isinstance(query_role_boxes, dict) and len(query_role_boxes) > 0:
                role_ids = sorted([str(k) for k in query_role_boxes.keys()])
            else:
                role_ids = [str(i) for i in range(len(memory_token_lengths_per_character))]
            start = 0
            for i, seg_len in enumerate(memory_token_lengths_per_character):
                seg_len = max(int(seg_len), 0)
                if seg_len <= 0:
                    continue
                end = min(start + seg_len, n_mem)
                if end > start:
                    role_id = role_ids[i] if i < len(role_ids) else str(i)
                    idx_t = torch.arange(start, end, device=memory_tokens.device, dtype=torch.long)
                    role_chunks.append((str(role_id), idx_t, []))
                start = end
                if start >= n_mem:
                    break
        return role_chunks

    def _apply_rope_1d(self, x, coord, base=10000.0):
        d = int(x.shape[-1])
        if d < 2:
            return x
        orig_dtype = x.dtype
        use_d = d - (d % 2)
        x_main = x[..., :use_d].to(dtype=torch.float64)
        x_rest = x[..., use_d:].to(dtype=torch.float64)
        half = use_d // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half, device=x.device, dtype=torch.float64) / float(max(half, 1))))
        theta = coord.to(device=x.device, dtype=torch.float64).unsqueeze(-1) * inv_freq
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        x_even = x_main[..., 0::2]
        x_odd = x_main[..., 1::2]
        rot_even = x_even * cos_t - x_odd * sin_t
        rot_odd = x_even * sin_t + x_odd * cos_t
        x_rot = torch.stack([rot_even, rot_odd], dim=-1).reshape(*x_main.shape)
        if x_rest.numel() > 0:
            x_rot = torch.cat([x_rot, x_rest], dim=-1)
        return x_rot.to(dtype=orig_dtype)

    def _apply_2d_rope_on_subset(self, x, uv, rope_dim):
        if uv is None:
            return x
        d = int(x.shape[-1])
        rope_dim = int(max(0, min(int(rope_dim), d)))
        rope_dim = rope_dim - (rope_dim % 4)
        if rope_dim <= 0:
            return x
        x_rope = x[..., :rope_dim]
        x_pass = x[..., rope_dim:]
        half = rope_dim // 2
        x_u = x_rope[..., :half]
        x_v = x_rope[..., half:]
        x_u = self._apply_rope_1d(x_u, uv[..., 0])
        x_v = self._apply_rope_1d(x_v, uv[..., 1])
        out = torch.cat([x_u, x_v], dim=-1)
        if x_pass.numel() > 0:
            out = torch.cat([out, x_pass], dim=-1)
        return out

    def _build_abs_uv_from_idx(self, selected_idx, num_video_tokens, latent_h, latent_w, device, dtype):
        uv = torch.zeros((int(selected_idx.numel()), 2), device=device, dtype=dtype)
        if selected_idx is None or selected_idx.numel() == 0:
            return uv
        grid = self._build_video_token_grid(num_video_tokens, latent_h, latent_w, device=device)
        if grid is None:
            return uv
        _, y_idx, x_idx = grid
        den_h = float(max(int(latent_h) - 1, 1))
        den_w = float(max(int(latent_w) - 1, 1))
        flat = selected_idx.to(device=device, dtype=torch.long)
        valid = (flat >= 0) & (flat < int(num_video_tokens))
        if bool(valid.any()):
            flat_v = flat[valid]
            uv_valid = torch.stack([
                (x_idx.index_select(0, flat_v).to(dtype=dtype) / den_w),
                (y_idx.index_select(0, flat_v).to(dtype=dtype) / den_h),
            ], dim=-1)
            uv[valid] = uv_valid
        return uv

    def _build_abs_uv_from_memory_meta(self, role_meta, num_mem_tokens, device, dtype, latent_h=None, latent_w=None):
        uv = torch.zeros((int(num_mem_tokens), 2), device=device, dtype=dtype)
        if not isinstance(role_meta, list) or len(role_meta) <= 0:
            return uv
        den_h = float(max(int(latent_h) - 1, 1)) if latent_h is not None else 1.0
        den_w = float(max(int(latent_w) - 1, 1)) if latent_w is not None else 1.0
        use_n = min(int(num_mem_tokens), len(role_meta))
        for i in range(use_n):
            m = role_meta[i]
            if not isinstance(m, dict):
                continue
            try:
                lh = float(m.get('latent_h', 0.0))
                lw = float(m.get('latent_w', 0.0))
            except Exception:
                continue
            uv[i, 0] = float(lw / den_w)
            uv[i, 1] = float(lh / den_h)
        return uv

    def _select_probe_indices(self, query_feature_payload, device, num_video_tokens):
        role_to_idx = {}
        if not isinstance(query_feature_payload, dict):
            return role_to_idx
        for role_id, payload in query_feature_payload.items():
            if not isinstance(payload, dict):
                continue
            flat_idx = payload.get('flat_idx', None)
            if not isinstance(flat_idx, torch.Tensor):
                continue
            idx = flat_idx.to(device=device, dtype=torch.long).reshape(-1)
            idx = idx[(idx >= 0) & (idx < int(num_video_tokens))]
            if idx.numel() <= 0:
                continue
            role_to_idx[str(role_id)] = torch.unique(idx, sorted=True)
        return role_to_idx

    def _reshape_heads(self, x, bsz):
        return x.view(bsz, x.shape[1], self.num_heads, self.head_dim)

    def _iter_query_chunks(self, query_idx, chunk_size):
        if query_idx is None or query_idx.numel() <= 0:
            return
        chunk_size = max(0, int(chunk_size))
        if chunk_size <= 0 or int(query_idx.numel()) <= chunk_size:
            yield query_idx
            return
        total = int(query_idx.numel())
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            yield query_idx[start:end]

    def forward_sparse(
        self,
        tokens,
        memory_tokens,
        query_role_boxes=None,
        query_feature_payload=None,
        memory_bank_token_meta=None,
        memory_token_lengths_per_character=None,
        latent_h=None,
        latent_w=None,
        timestep_percent=1.0,
        **kwargs,
    ):
        if query_role_boxes is None:
            query_role_boxes = kwargs.get('query_role_boxes', None)
        if query_feature_payload is None:
            query_feature_payload = kwargs.get('query_feature_payload', kwargs.get('query_feature_map', None))
        if memory_bank_token_meta is None:
            memory_bank_token_meta = kwargs.get('memory_bank_token_meta', kwargs.get('memory_token_meta', None))
        if memory_token_lengths_per_character is None:
            memory_token_lengths_per_character = kwargs.get('memory_token_lengths_per_character', None)
        if latent_h is None:
            latent_h = kwargs.get('latent_h', None)
        if latent_w is None:
            latent_w = kwargs.get('latent_w', None)
        if timestep_percent is None:
            timestep_percent = kwargs.get('timestep_percent', 1.0)
        try:
            query_chunk_size = int(kwargs.get('sparse_role_memory_query_chunk_size', self.query_chunk_size))
        except Exception:
            query_chunk_size = int(self.query_chunk_size)
        query_chunk_size = max(0, query_chunk_size)
        bsz, num_video_tokens, dim = tokens.shape
        debug = {
            'enabled': 1.0,
            'selected_query_tokens': 0,
            'selected_memory_tokens': 0,
            'winner_counts': {},
            'role_head_out_norm': 0.0,
            'plain_head_out_norm': 0.0,
            'attn_entropy': 0.0,
            'query_chunk_size': int(query_chunk_size),
            'query_chunk_count': 0,
            'memory_side_rope': 1.0 if bool(self.use_memory_side_rope) else 0.0,
            'merge_add_rope_center_to_value': 1.0 if bool(self.add_rope_center_to_value) else 0.0,
        }
        if memory_tokens is None or memory_tokens.ndim < 3 or int(memory_tokens.shape[1]) <= 0:
            debug['enabled'] = 0.0
            return tokens, debug

        role_to_idx = self._select_probe_indices(query_feature_payload, tokens.device, num_video_tokens)
        if len(role_to_idx) == 0:
            debug['enabled'] = 0.0
            return tokens, debug

        role_chunks = self._split_memory_by_role(
            memory_tokens=memory_tokens,
            memory_token_meta=memory_bank_token_meta,
            memory_token_lengths_per_character=memory_token_lengths_per_character,
            query_role_boxes=query_role_boxes,
        )
        if len(role_chunks) == 0:
            debug['enabled'] = 0.0
            return tokens, debug

        selected_global = []
        for idx in role_to_idx.values():
            if isinstance(idx, torch.Tensor) and idx.numel() > 0:
                selected_global.append(idx)
        if len(selected_global) == 0:
            debug['enabled'] = 0.0
            return tokens, debug

        selected_union = torch.unique(torch.cat(selected_global, dim=0), sorted=True)
        if selected_union.numel() <= 0:
            debug['enabled'] = 0.0
            return tokens, debug

        out_selected = torch.zeros((bsz, int(selected_union.numel()), dim), device=tokens.device, dtype=tokens.dtype)
        best_selected = torch.full((bsz, int(selected_union.numel())), -1e9, device=tokens.device, dtype=tokens.dtype)
        local_index_map = torch.full((num_video_tokens,), -1, device=tokens.device, dtype=torch.long)
        local_index_map[selected_union] = torch.arange(selected_union.numel(), device=tokens.device, dtype=torch.long)

        role_heads = self.num_heads // 2 if self.use_half_role_heads else self.num_heads
        role_heads = max(1, int(role_heads))
        plain_heads = max(0, int(self.num_heads - role_heads))

        role_names = []
        role_head_norm_sum = 0.0
        plain_head_norm_sum = 0.0
        entropy_sum = 0.0
        entropy_count = 0

        for role_id, mem_idx, role_meta in role_chunks:
            role_key = str(role_id)
            q_idx_real = role_to_idx.get(role_key, None)
            if q_idx_real is None or q_idx_real.numel() <= 0:
                continue
            if mem_idx is None or mem_idx.numel() <= 0:
                continue

            mem_tok = memory_tokens.index_select(1, mem_idx)

            k = self._reshape_heads(self.k_proj(mem_tok), bsz)
            v = self._reshape_heads(self.v_proj(mem_tok), bsz)
            k_abs_uv = self._build_abs_uv_from_memory_meta(
                role_meta=role_meta,
                num_mem_tokens=int(mem_idx.numel()),
                device=tokens.device,
                dtype=torch.float64,
                latent_h=latent_h,
                latent_w=latent_w,
            )
            k_role_uv, k_role_valid = self._build_memory_uv_for_role(
                role_meta=role_meta,
                bsz=bsz,
                num_mem_tokens=int(mem_idx.numel()),
                device=tokens.device,
                dtype=torch.float64,
            )

            if role_heads > 0 and bool(self.use_memory_side_rope):
                k_role = k[:, :, :role_heads, :]
                k_role = self._apply_2d_rope_on_subset(k_role, -k_abs_uv.unsqueeze(0).unsqueeze(2), self.rope_dim)
                k_role = self._apply_2d_rope_on_subset(k_role, k_role_uv.unsqueeze(2), self.rope_dim)
                k[:, :, :role_heads, :] = k_role

            kh = k.permute(0, 2, 1, 3)
            vh = v.permute(0, 2, 1, 3)

            role_head_norm_weighted = 0.0
            plain_head_norm_weighted = 0.0
            role_norm_weight = 0.0
            role_winner_sum = 0

            for q_idx_chunk in self._iter_query_chunks(q_idx_real, query_chunk_size):
                if q_idx_chunk is None or q_idx_chunk.numel() <= 0:
                    continue
                debug['query_chunk_count'] += 1

                q_tok = tokens.index_select(1, q_idx_chunk)
                q = self._reshape_heads(self.q_proj(q_tok), bsz)
                q_abs_uv = self._build_abs_uv_from_idx(
                    selected_idx=q_idx_chunk,
                    num_video_tokens=num_video_tokens,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    device=tokens.device,
                    dtype=torch.float64,
                )
                q_role_uv, _ = self._build_query_uv_for_role_from_selected_idx(
                    role_id=role_key,
                    selected_idx=q_idx_chunk,
                    query_role_boxes=query_role_boxes,
                    bsz=bsz,
                    num_video_tokens=num_video_tokens,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    device=tokens.device,
                    dtype=torch.float64,
                )

                if role_heads > 0:
                    q_role = q[:, :, :role_heads, :]
                    q_role = self._apply_2d_rope_on_subset(q_role, -q_abs_uv.unsqueeze(0).unsqueeze(2), self.rope_dim)
                    q_role = self._apply_2d_rope_on_subset(q_role, q_role_uv.unsqueeze(2), self.rope_dim)
                    q[:, :, :role_heads, :] = q_role

                qh = q.permute(0, 2, 1, 3)
                attn_logits = torch.einsum('bhqd,bhkd->bhqk', qh, kh) * self.scale
                # BBox should only provide RoPE coordinates, not hard token filtering.
                attn = torch.softmax(attn_logits, dim=-1)

                if self.debug:
                    with torch.no_grad():
                        p = attn.clamp(min=1e-8)
                        entropy_sum += float((-(p * p.log()).sum(dim=-1).mean()).item())
                        entropy_count += 1

                out_h = torch.einsum('bhqk,bhkd->bhqd', attn, vh)
                chunk_weight = float(int(q_idx_chunk.numel()))
                role_norm_weight += chunk_weight
                if role_heads > 0:
                    role_head_norm_weighted += float(out_h[:, :role_heads].detach().float().norm(dim=-1).mean().item()) * chunk_weight
                if plain_heads > 0:
                    plain_head_norm_weighted += float(out_h[:, role_heads:].detach().float().norm(dim=-1).mean().item()) * chunk_weight

                out_role = out_h.permute(0, 2, 1, 3).reshape(bsz, int(q_idx_chunk.numel()), self.inner_dim)
                out_role = self.out_proj(out_role)
                sim_role = attn_logits.max(dim=-1).values.mean(dim=1)

                q_local = local_index_map.index_select(0, q_idx_chunk)
                prev_best = best_selected.index_select(1, q_local)
                winner = sim_role > prev_best
                winner_expand = winner.unsqueeze(-1)
                merged_prev = out_selected.index_select(1, q_local)
                merged_now = torch.where(winner_expand, out_role, merged_prev)
                out_selected.index_copy_(1, q_local, merged_now)
                best_selected.index_copy_(1, q_local, torch.where(winner, sim_role, prev_best))
                role_winner_sum += int(winner.detach().sum().item())

            if role_norm_weight <= 0.0:
                continue

            if role_heads > 0:
                role_head_norm_sum += float(role_head_norm_weighted / role_norm_weight)
            if plain_heads > 0:
                plain_head_norm_sum += float(plain_head_norm_weighted / role_norm_weight)

            role_names.append(role_key)
            debug['selected_query_tokens'] += int(q_idx_real.numel())
            debug['selected_memory_tokens'] += int(mem_idx.numel())
            debug['winner_counts'][role_key] = int(role_winner_sum)

        if len(role_names) == 0:
            debug['enabled'] = 0.0
            return tokens, debug

        gate = 1.0
        if self.time_gate:
            p = float(max(0.0, min(1.0, timestep_percent)))
            gate_raw = self.time_gate_scale * (1.0 - p) + self.time_gate_bias
            gate = torch.sigmoid(gate_raw)

        scaled_delta = (self.output_scale * gate) * out_selected
        out = torch.index_add(tokens, 1, selected_union, scaled_delta)

        role_count = float(max(len(role_names), 1))
        debug['role_head_out_norm'] = float(role_head_norm_sum / role_count)
        debug['plain_head_out_norm'] = float(plain_head_norm_sum / role_count)
        if entropy_count > 0:
            debug['attn_entropy'] = float(entropy_sum / float(entropy_count))
        return out, debug

    def forward(
        self,
        tokens,
        memory_tokens,
        query_role_boxes=None,
        query_feature_payload=None,
        memory_bank_token_meta=None,
        memory_token_lengths_per_character=None,
        latent_h=None,
        latent_w=None,
        timestep_percent=1.0,
        **kwargs,
    ):
        return self.forward_sparse(
            tokens=tokens,
            memory_tokens=memory_tokens,
            query_role_boxes=query_role_boxes,
            query_feature_payload=query_feature_payload,
            memory_bank_token_meta=memory_bank_token_meta,
            memory_token_lengths_per_character=memory_token_lengths_per_character,
            latent_h=latent_h,
            latent_w=latent_w,
            timestep_percent=timestep_percent,
            **kwargs,
        )


# Helper Functions for Data Loading
# =============================================================================

def skip_none_collate(batch):
    """
    Custom collate function to skip None values.
    Must be at module level for multiprocessing pickling.
    """
    if batch is None:
        batch = []
    else:
        batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    if len(batch) == 1:
        return batch[0]
    from torch.utils.data._utils.collate import default_collate
    return default_collate(batch)


def skip_none_keep_list_collate(batch):
    """
    Custom collate function for extraction workers.
    Keeps samples as a Python list (no default_collate stack), so extractor
    can safely support batch_size > 1 while still processing/writing per sample.
    """
    if batch is None:
        batch = []
    else:
        batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return batch


# =============================================================================
# Detailed Visualization (every N samples)
# =============================================================================
def run_detailed_viz_extraction(
    pipe, character_id, original_prompt, latents, latents_batch,
    memory_video_tensor, extract_config, device, viz_dir,
    cfg_scale=5.0, cached_conditions=None
):
    """
    Run full 20-step attention extraction across all layers for detailed visualization.
    Outputs organized by timestep and by layer subfolders.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpl_patches
    
    C, T, H, W = memory_video_tensor.shape
    C_lat, F_lat, H_lat, W_lat = latents.shape
    
    # 获取 DiT patch_size，计算实际 attention/patch 空间分辨率
    _patch_size = pipe.dit.patch_size  # e.g., (1, 2, 2)
    H_patch = H_lat // _patch_size[1]  # 60 // 2 = 30
    W_patch = W_lat // _patch_size[2]  # 104 // 2 = 52
    spatial_shape = (H_patch, W_patch)
    
    # Parse character name and find token indices（热力图查询时忽略 _ - 等特殊符号，只考虑文本 token）
    character_name = character_id.replace('_', ' ').replace('-', ' ').strip()
    parts = character_id.replace('-', '_').split('_', 1)
    prefix_text = parts[0]
    suffix_text = parts[1].replace('_', ' ') if len(parts) > 1 else None
    
    token_ids, token_texts, _ = verify_target_text_is_single_token(pipe, character_name)
    if not token_ids:
        return
    full_indices = find_token_index_in_prompt(pipe, original_prompt, character_name, token_ids, token_texts)
    if not full_indices:
        return
    
    prefix_token_ids, _, _ = verify_target_text_is_single_token(pipe, prefix_text)
    num_prefix_tokens = len(prefix_token_ids) if prefix_token_ids else 0
    if suffix_text:
        suffix_token_ids, _, _ = verify_target_text_is_single_token(pipe, suffix_text)
        num_suffix_tokens = len(suffix_token_ids) if suffix_token_ids else 0
    else:
        num_suffix_tokens = 0
    prefix_indices = full_indices[:num_prefix_tokens]
    suffix_indices = full_indices[num_prefix_tokens:num_prefix_tokens + num_suffix_tokens] if num_suffix_tokens > 0 else []
    
    suffix_attention_scale = extract_config.get('suffix_attention_scale', 1.0)
    token_weight = extract_config.get('token_weight', 1)
    
    # Encode prompt
    if cfg_scale > 1.0:
        prompt_embeds_pos = pipe.prompter.encode_prompt(prompt=original_prompt, positive=True, device=device)
        prompt_embeds_neg = pipe.prompter.encode_prompt(prompt="", positive=False, device=device)
        prompt_embeds = torch.cat([prompt_embeds_neg, prompt_embeds_pos], dim=0)
    else:
        prompt_embeds = pipe.prompter.encode_prompt(prompt=original_prompt, positive=True, device=device)
    prompt_embeds = prompt_embeds.to(device=device)
    
    # Image conditioning
    image_cond_kwargs = {}
    if hasattr(pipe.dit, 'has_image_input') and pipe.dit.has_image_input:
        if cached_conditions is not None:
            if cached_conditions.get('clip_feature', None) is not None:
                image_cond_kwargs['clip_feature'] = cached_conditions['clip_feature']
            image_cond_kwargs['y'] = cached_conditions['y']
        else:
            first_frame_tensor = memory_video_tensor[:, 0, :, :]
            first_frame_np = first_frame_tensor.permute(1, 2, 0).cpu().numpy()
            first_frame_np = ((first_frame_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            first_frame = Image.fromarray(first_frame_np)
            cond = pipe.encode_images_adaptive(
                [first_frame],
                first_frame,
                F_lat * 4 - 3,
                H,
                W,
                use_first_aug=False,
                ref_pad_cfg=False,
                ref_pad_num=0,
            )
            clip_feature = cond.get('clip_feature', None)
            y = cond.get('y', None)
            if y is None:
                raise RuntimeError("encode_images_adaptive did not return y for extraction")
            if cfg_scale > 1.0:
                if clip_feature is not None:
                    clip_feature = torch.cat([clip_feature, clip_feature], dim=0)
                y = torch.cat([y, y], dim=0)
            if clip_feature is not None:
                image_cond_kwargs['clip_feature'] = clip_feature.to(device=device, dtype=latents_batch.dtype)
            image_cond_kwargs['y'] = y.to(device=device, dtype=latents_batch.dtype)
            del cond, clip_feature, y
    
    # Get all 20 timesteps
    num_total_steps = 20
    all_timesteps = pipe.scheduler.get_timesteps(
        num_inference_steps=num_total_steps, denoising_strength=1, shift=5.0
    ).to(device=device)
    
    num_blocks = len(pipe.dit.blocks)
    all_layer_indices = list(range(num_blocks))
    
    # 解析实际提取使用的层 (与 extract_patch_embeddings_for_character 保持一致)
    configured_layers = extract_config.get('extract_layers', [-1])
    if isinstance(configured_layers, list) and -1 in configured_layers:
        extract_layer_indices = all_layer_indices
    else:
        extract_layer_indices = [l for l in configured_layers if 0 <= l < num_blocks]
    
    # Collect per-step per-layer attention maps
    # step_layer_maps[step_idx][layer_idx] = attention_map (flattened, float, cpu)
    step_layer_maps = {}
    
    for step_idx, t_val in enumerate(all_timesteps):
        noise = torch.randn_like(latents_batch)
        noisy_latents = (1 - t_val / pipe.scheduler.num_train_timesteps) * latents_batch + \
                        (t_val / pipe.scheduler.num_train_timesteps) * noise
        if cfg_scale > 1.0:
            noisy_input = torch.cat([noisy_latents, noisy_latents], dim=0)
        else:
            noisy_input = noisy_latents
        noisy_input = noisy_input.to(device=device)
        
        attn_extractor = AttentionMapExtractor(
            pipe, extract_layer_indices,
            target_token_indices=prefix_indices,
            suffix_token_indices=suffix_indices,
            suffix_scale=suffix_attention_scale,
            cfg_scale=cfg_scale,
            token_weight=token_weight
        )
        attn_extractor.register_hooks()
        try:
            if hasattr(pipe, 'set_active_noise_domain_from_timestep'):
                pipe.set_active_noise_domain_from_timestep(t_val.unsqueeze(0).to(device=device, dtype=latents_batch.dtype))
            dit_kwargs = {'x': noisy_input, 'timestep': t_val.unsqueeze(0).to(device=device, dtype=latents_batch.dtype),
                          'context': prompt_embeds, **image_cond_kwargs}
            with torch.no_grad():
                _ = run_native_dit_forward(pipe.dit, **dit_kwargs)
            del dit_kwargs
        except Exception:
            attn_extractor.remove_hooks()
            raise
        
        raw_maps = attn_extractor.get_attention_maps()
        attn_extractor.remove_hooks()
        
        step_layer_maps[step_idx] = {}
        for layer_idx, amap in raw_maps.items():
            while amap.dim() > 1 and amap.shape[0] == 1:
                amap = amap.squeeze(0)
            # 修复: 对head维度取均值 [H, N_vis] → [N_vis]，避免head维度被错误flatten到空间维度
            if amap.dim() > 1:
                amap = amap.mean(dim=0)
            step_layer_maps[step_idx][layer_idx] = amap.detach().float().cpu()
        
        del noise, noisy_latents, noisy_input, raw_maps
    
    del prompt_embeds, image_cond_kwargs
    
    # --- Helper: aggregate maps over step range and layer range ---
    def aggregate_maps(step_range, layer_range):
        agg = None
        count = 0
        for s in step_range:
            if s not in step_layer_maps:
                continue
            for l in layer_range:
                if l not in step_layer_maps[s]:
                    continue
                m = step_layer_maps[s][l]
                if agg is None:
                    agg = m.clone()
                else:
                    if agg.shape == m.shape:
                        agg += m
                count += 1
        if agg is not None and count > 0:
            agg /= count
        return agg
    
    # --- Helper: attention map -> selected positions for viz ---
    top_visual_tokens = extract_config.get('top_visual_tokens', -1)
    neighbor_filter_kernel = extract_config.get('neighbor_filter_kernel', 0)
    neighbor_filter_any_window = bool(extract_config.get('neighbor_filter_any_window', True))
    
    def map_to_positions(agg_map, label):
        if agg_map is None:
            return None
        num_tokens = agg_map.shape[0]
        F_actual = num_tokens // (H_patch * W_patch)  # 使用 patch grid 分辨率
        binary_mask, continuous_map, selected_indices, frame_info = process_attention_map_to_mask(
            agg_map, threshold=top_visual_tokens, spatial_shape=spatial_shape,
            num_frames=F_actual, otsu_scope="frame", neighbor_filter_kernel=neighbor_filter_kernel,
            neighbor_filter_any_window=neighbor_filter_any_window,
        )
        if isinstance(selected_indices, set):
            selected_indices = list(selected_indices)
        total_spatial = H_patch * W_patch  # patch 空间每帧 token 数
        positions = []
        for idx in selected_indices:
            frame_idx = idx // total_spatial
            spatial_idx = idx % total_spatial
            h = spatial_idx // W_patch
            w = spatial_idx % W_patch
            if frame_idx < F_lat:
                positions.append((frame_idx, h, w))
        return positions, frame_info
    
    # --- Helper: draw visualization ---
    frames_np = ((memory_video_tensor.permute(1, 2, 3, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    T_vid, H_vid, W_vid, _ = frames_np.shape
    # stride 需要反映 patch grid → pixel 的缩放 (patch_size 在 VAE 8x 基础上再 2x)
    stride_t = max(1, T_vid // F_lat) if F_lat > 0 else 4
    stride_h = H_vid // H_patch  # 480 // 30 = 16
    stride_w = W_vid // W_patch  # 832 // 52 = 16
    
    def draw_and_save(positions, title, save_path):
        if positions is None:
            return
        pos_array = np.array(positions) if len(positions) > 0 else np.zeros((0, 3))
        if len(pos_array) == 0:
            return
        active_frames = sorted(set(pos_array[:, 0].astype(int)))
        # Pick the frame with most patches
        best_t = max(active_frames, key=lambda t: np.sum(pos_array[:, 0] == t))
        pixel_t = int(best_t * stride_t)
        if pixel_t >= T_vid:
            pixel_t = T_vid - 1
        img = frames_np[pixel_t].copy()
        
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        ax = plt.gca()
        patches_in_frame = pos_array[pos_array[:, 0] == best_t]
        for _, lat_h, lat_w in patches_in_frame:
            y_pos = int(lat_h) * stride_h
            x_pos = int(lat_w) * stride_w
            rect = mpl_patches.Rectangle((x_pos, y_pos), stride_w, stride_h, linewidth=0.5, edgecolor='red', facecolor='red', alpha=0.4)
            ax.add_patch(rect)
        plt.axis('off')
        plt.title(f"{title} | Frame {pixel_t} | {len(patches_in_frame)} patches", fontsize=9)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=120)
        plt.close()
    
    # ======== Generate visualizations ========
    safe_char = "".join([c if c.isalnum() else "_" for c in character_id])
    char_viz_dir = os.path.join(viz_dir, safe_char)
    
    # --- by_timestep (layers follow extract_layers config) ---
    ts_dir = os.path.join(char_viz_dir, "by_timestep")
    layer_desc = "all" if extract_layer_indices == all_layer_indices else str(configured_layers)
    # 1) Each of the 20 steps (extract_layers aggregated)
    for s in range(num_total_steps):
        agg = aggregate_maps([s], extract_layer_indices)
        result = map_to_positions(agg, f"step_{s+1:02d}")
        if result:
            draw_and_save(result[0], f"Step {s+1}/{num_total_steps} (t={all_timesteps[s].item():.2f}) layers={layer_desc}",
                          os.path.join(ts_dir, f"step_{s+1:02d}.jpg"))
    
    # 2) Aggregated step ranges: 0-5, 6-12, 12-20 (0-indexed)
    step_ranges = [("steps_01-05", range(0, 5)), ("steps_06-12", range(5, 12)), ("steps_12-20", range(11, 20))]
    for name, sr in step_ranges:
        agg = aggregate_maps(sr, extract_layer_indices)
        result = map_to_positions(agg, name)
        if result:
            draw_and_save(result[0], f"Aggregated {name} (layers={layer_desc})",
                          os.path.join(ts_dir, f"agg_{name}.jpg"))
    
    # Cleanup
    del step_layer_maps
    torch.cuda.empty_cache()


# =============================================================================
# Async Patch Extractor Process
# =============================================================================
def save_extraction_visualization(
    viz_root_dir, 
    sample_id, 
    video_tensor,         # [C, T, H, W] normalized [-1, 1]
    char_positions_dict,  # {char_id: positions_tensor}
    vae_stride=(4, 8, 8)  # (T_stride, H_stride, W_stride)
):
    """
    保存提取结果的热力图可视化（分角色）。
    只保留最近的 2 个 use_memory=True 的 sample_id 文件夹。
    """
    import shutil
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import os
    import numpy as np
    
    # -----------------------------------------------------------
    # 1. 目录管理逻辑 (Check: 是否始终只保存2个)
    # -----------------------------------------------------------
    os.makedirs(viz_root_dir, exist_ok=True)
    
    # 获取当前所有已保存的样本文件夹，按名称排序 (假设 sample_id 以时间戳开头，排序即按时间)
    existing_dirs = sorted([d for d in os.listdir(viz_root_dir) if os.path.isdir(os.path.join(viz_root_dir, d))])
    
    # 如果当前样本是新的，且已存数量 >= 2，则删除最旧的
    # 逻辑验证：
    #   - 假设已有 [S1, S2]
    #   - 新来 S3
    #   - sorted 结果 [S1, S2]
    #   - 删除 S1
    #   - 保存 S3
    #   - 最终状态 [S2, S3] (始终保持2个)
    if sample_id not in existing_dirs:
        while len(existing_dirs) >= 2:
            oldest_dir = os.path.join(viz_root_dir, existing_dirs.pop(0)) # 弹出并删除最旧的
            try:
                shutil.rmtree(oldest_dir)
                #print(f"[Viz] Removed old visualization: {oldest_dir}")
            except Exception as e:
                pass
                #print(f"[Viz] Error deleting old dir {oldest_dir}: {e}")
    
    # 创建当前样本目录
    sample_dir = os.path.join(viz_root_dir, sample_id)
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    os.makedirs(sample_dir, exist_ok=True)
    
    # -----------------------------------------------------------
    # 2. 视频帧预处理
    # -----------------------------------------------------------
    # [C, T, H, W] -> [T, H, W, C] & [-1, 1] -> [0, 255]
    frames = ((video_tensor.permute(1, 2, 3, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    T, H, W, C = frames.shape
    stride_t, stride_h, stride_w = vae_stride
    
    # 颜色列表 (用于区分不同角色，虽然分了文件夹，但画图时也可以用不同色)
    colors = ['red', 'cyan', 'lime', 'yellow', 'magenta']
    
    # -----------------------------------------------------------
    # 3. 分角色绘制
    # -----------------------------------------------------------
    for i, (char_id, positions) in enumerate(char_positions_dict.items()):
        if positions is None or positions.shape[0] == 0:
            continue
            
        # 为该角色创建子目录
        # 处理 char_id 中的非法文件名字符
        safe_char_id = "".join([c if c.isalnum() else "_" for c in char_id])
        char_dir = os.path.join(sample_dir, f"role_{safe_char_id}")
        os.makedirs(char_dir, exist_ok=True)
        
        # 挑选颜色
        color = colors[i % len(colors)]
        
        # 解析位置
        pos_list = positions.cpu().numpy() # [N, 3] -> (t, h, w)
        active_latent_frames = sorted(list(set(pos_list[:, 0])))
        
        for lat_t in active_latent_frames:
            # 获取该帧该角色的所有 patch
            patches_in_frame = pos_list[pos_list[:, 0] == lat_t]
            
            # 计算对应的 Pixel Frame 索引
            start_t = int(lat_t * stride_t)
            if start_t >= T: continue
            
            # 绘图
            img = frames[start_t].copy()
            
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            ax = plt.gca()
            
            # 叠加 Patch
            for _, lat_h, lat_w in patches_in_frame:
                y = lat_h * stride_h
                x = lat_w * stride_w
                rect = patches.Rectangle((x, y), stride_w, stride_h, linewidth=1, edgecolor=color, facecolor=color, alpha=0.4)
                ax.add_patch(rect)
            
            plt.axis('off')
            plt.title(f"{char_id} | Latent Frame {int(lat_t)} | {len(patches_in_frame)} Patches")
            
            # 保存
            out_name = f"frame_{start_t:04d}.jpg"
            plt.savefig(os.path.join(char_dir, out_name), bbox_inches='tight', pad_inches=0)
            plt.close()
            
    # 保存 info.txt
    with open(os.path.join(sample_dir, "info.txt"), "w") as f:
        f.write(f"Sample ID: {sample_id}\n")
        f.write(f"Characters found: {list(char_positions_dict.keys())}\n")


def _build_bbox_relative_token_meta(char_id, positions, f_lat, similarity_scores=None):
    """Build per-token bbox-relative metadata in latent patch coordinates."""
    token_meta = []
    if positions is None or not isinstance(positions, torch.Tensor) or positions.numel() == 0:
        return token_meta

    pos_cpu = positions.detach().cpu().to(torch.long)
    score_list = None
    if similarity_scores is not None:
        if isinstance(similarity_scores, torch.Tensor):
            score_list = similarity_scores.detach().cpu().reshape(-1).tolist()
        elif isinstance(similarity_scores, (list, tuple, np.ndarray)):
            score_list = [float(x) for x in similarity_scores]
        else:
            raise TypeError(f"similarity_scores must be tensor/list/tuple/ndarray, got {type(similarity_scores)}")
        if len(score_list) != int(pos_cpu.shape[0]):
            raise ValueError(
                f"similarity_scores length mismatch: scores={len(score_list)} positions={int(pos_cpu.shape[0])}"
            )
    per_t_bbox = {}
    for idx, p in enumerate(pos_cpu.tolist()):
        if len(p) != 3:
            continue
        lt, lh, lw = int(p[0]), int(p[1]), int(p[2])
        if lt not in per_t_bbox:
            per_t_bbox[lt] = [float(lw), float(lh), float(lw + 1), float(lh + 1)]
        else:
            cur = per_t_bbox[lt]
            cur[0] = min(cur[0], float(lw))
            cur[1] = min(cur[1], float(lh))
            cur[2] = max(cur[2], float(lw + 1))
            cur[3] = max(cur[3], float(lh + 1))

    denom_t = float(max(int(f_lat) - 1, 1))
    for p in pos_cpu.tolist():
        if len(p) != 3:
            continue
        lt, lh, lw = int(p[0]), int(p[1]), int(p[2])
        xc = float(lw) + 0.5
        yc = float(lh) + 0.5
        bbox = per_t_bbox.get(lt, None)

        bbox_latent_xyxy = None
        rel_l = -1.0
        rel_r = -1.0
        rel_t = -1.0
        rel_b = -1.0
        inside_box = False
        u = 0.0
        v = 0.0

        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = [float(x) for x in bbox]
            bw = max(x2 - x1, 1e-6)
            bh = max(y2 - y1, 1e-6)
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            u = (xc - cx) / (0.5 * bw + 1e-6)
            v = (yc - cy) / (0.5 * bh + 1e-6)
            rel_l = (xc - x1) / (bw + 1e-6)
            rel_r = (x2 - xc) / (bw + 1e-6)
            rel_t = (yc - y1) / (bh + 1e-6)
            rel_b = (y2 - yc) / (bh + 1e-6)
            inside_box = bool((xc >= x1) and (xc <= x2) and (yc >= y1) and (yc <= y2))
            bbox_latent_xyxy = [x1, y1, x2, y2]

        token_meta.append({
            'char_id': str(char_id),
            'latent_t': int(lt),
            'latent_h': int(lh),
            'latent_w': int(lw),
            'bbox_latent_xyxy': bbox_latent_xyxy,
            'rel_l': float(rel_l),
            'rel_r': float(rel_r),
            'rel_t': float(rel_t),
            'rel_b': float(rel_b),
            'inside_box': bool(inside_box),
            'u': float(u),
            'v': float(v),
            'tau_local': float(float(lt) / denom_t),
            'similarity_score': float(score_list[idx]) if score_list is not None else 0.0,
        })

    return token_meta


class LayerFeatureTap:
    """Capture one DiT block output token tensor from a single forward pass."""

    def __init__(self, dit_model, layer_idx, keep_device='cpu', keep_dtype=torch.bfloat16):
        self.dit_model = dit_model
        self.layer_idx = int(layer_idx)
        self.keep_device = str(keep_device)
        self.keep_dtype = keep_dtype
        self.hook = None
        self.tokens = None

    def register(self):
        if self.hook is not None:
            return
        if self.dit_model is None or not hasattr(self.dit_model, 'blocks'):
            return
        num_blocks = len(self.dit_model.blocks)
        if self.layer_idx < 0 or self.layer_idx >= num_blocks:
            return
        block = self.dit_model.blocks[self.layer_idx]

        def _hook(_module, _inputs, output):
            x = output
            if isinstance(output, (tuple, list)) and len(output) > 0:
                x = output[0]
            if not isinstance(x, torch.Tensor) or x.dim() < 2:
                return
            # CFG extraction uses B=2 (uncond/cond). Keep conditional branch only.
            if x.dim() >= 3:
                x_keep = x[-1]
            else:
                x_keep = x
            x_keep = x_keep.detach().contiguous()
            if self.keep_device == 'cpu':
                self.tokens = x_keep.to(device='cpu', dtype=self.keep_dtype)
            else:
                self.tokens = x_keep.to(dtype=self.keep_dtype)

        self.hook = block.register_forward_hook(_hook)

    def pop_tokens(self):
        out = self.tokens
        self.tokens = None
        return out

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
        self.tokens = None


class AttentionOutputFeatureTap:
    """Capture attention output features from one DiT block for one forward pass.

    This tap prefers cross-attention output (context interaction output) and is
    used as the feature source for sparse role-memory matching.
    """

    def __init__(self, dit_model, layer_idx, keep_device='cpu', keep_dtype=torch.bfloat16, source='attn_out', batch_index=-1):
        self.dit_model = dit_model
        self.layer_idx = int(layer_idx)
        self.keep_device = str(keep_device)
        self.keep_dtype = keep_dtype
        self.source = str(source).strip().lower()
        self.batch_index = int(batch_index)
        self.hook = None
        self.tokens = None

    def register(self):
        if self.hook is not None:
            return
        if self.dit_model is None or not hasattr(self.dit_model, 'blocks'):
            return
        num_blocks = len(self.dit_model.blocks)
        if self.layer_idx < 0 or self.layer_idx >= num_blocks:
            return
        block = self.dit_model.blocks[self.layer_idx]

        if self.source == 'self_attn_out' and hasattr(block, 'self_attn'):
            target_module = block.self_attn
        elif hasattr(block, 'cross_attn'):
            target_module = block.cross_attn
        elif hasattr(block, 'self_attn'):
            target_module = block.self_attn
        else:
            return

        def _hook(_module, _inputs, output):
            x = output
            if isinstance(output, (tuple, list)) and len(output) > 0:
                x = output[0]
            if not isinstance(x, torch.Tensor) or x.dim() < 2:
                return
            if x.dim() >= 3:
                batch_index = self.batch_index
                if batch_index < 0:
                    batch_index = int(x.shape[0]) + batch_index
                if batch_index < 0 or batch_index >= int(x.shape[0]):
                    batch_index = int(x.shape[0]) - 1
                x_keep = x[batch_index]
            else:
                x_keep = x
            x_keep = x_keep.detach().contiguous()
            if self.keep_device == 'cpu':
                self.tokens = x_keep.to(device='cpu', dtype=self.keep_dtype)
            else:
                self.tokens = x_keep.to(dtype=self.keep_dtype)

        self.hook = target_module.register_forward_hook(_hook)

    def pop_tokens(self):
        out = self.tokens
        self.tokens = None
        return out

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
        self.tokens = None


class _StopForwardAfterLayer(RuntimeError):
    """Internal control-flow exception for early-stopping layer probes."""


class ForwardStopAfterLayer:
    """Raise a private exception once the target DiT block finishes forward."""

    def __init__(self, dit_model, layer_idx):
        self.dit_model = dit_model
        self.layer_idx = int(layer_idx)
        self.hook = None

    def register(self):
        if self.hook is not None:
            return
        if self.dit_model is None or not hasattr(self.dit_model, 'blocks'):
            return
        num_blocks = len(self.dit_model.blocks)
        if self.layer_idx < 0 or self.layer_idx >= num_blocks:
            return
        block = self.dit_model.blocks[self.layer_idx]

        def _hook(_module, _inputs, _output):
            raise _StopForwardAfterLayer()

        self.hook = block.register_forward_hook(_hook)

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None


def _repeat_probe_batch_tensor(value, repeat_factor, base_batch):
    if not isinstance(value, torch.Tensor):
        return value
    repeat_factor = max(1, int(repeat_factor))
    base_batch = max(1, int(base_batch))
    if repeat_factor == 1:
        return value
    if value.dim() == 0:
        return value.reshape(1).repeat(repeat_factor)
    if int(value.shape[0]) == base_batch:
        return torch.cat([value] * repeat_factor, dim=0)
    return value


def _repeat_probe_batch_kwargs(kwargs, repeat_factor, base_batch):
    out = {}
    if not isinstance(kwargs, dict):
        return out
    for key, value in kwargs.items():
        out[key] = _repeat_probe_batch_tensor(value, repeat_factor, base_batch)
    return out


def _make_single_positive_condition_kwargs(kwargs):
    out = {}
    if not isinstance(kwargs, dict):
        return out
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0 and int(value.shape[0]) > 1:
            out[key] = value[-1:].contiguous()
        else:
            out[key] = value
    return out


def _aggregate_character_semantic_responses_cpu(step_maps):
    if not isinstance(step_maps, dict) or len(step_maps) == 0:
        return None
    step_agg = None
    for _, amap in step_maps.items():
        if not isinstance(amap, torch.Tensor):
            continue
        if amap.dim() >= 3 and amap.shape[0] == 2:
            amap = (amap[1] - amap[0]).clamp(min=0)
        while amap.dim() > 1 and amap.shape[0] == 1:
            amap = amap.squeeze(0)
        if amap.dim() > 1:
            amap = amap.mean(dim=0)
        amap_cpu = amap.detach().to(device='cpu', dtype=torch.float32)
        if step_agg is None:
            step_agg = amap_cpu
        else:
            min_shape = tuple(min(a, b) for a, b in zip(step_agg.shape, amap_cpu.shape))
            step_agg = step_agg[tuple(slice(0, s) for s in min_shape)]
            amap_cpu = amap_cpu[tuple(slice(0, s) for s in min_shape)]
            step_agg += amap_cpu
    return step_agg


def _suppress_other_character_response_cpu(primary_map, negative_map):
    if not isinstance(primary_map, torch.Tensor):
        return None
    if not isinstance(negative_map, torch.Tensor):
        return primary_map
    primary_flat = primary_map.reshape(-1).to(device='cpu', dtype=torch.float32)
    negative_flat = negative_map.reshape(-1).to(device='cpu', dtype=torch.float32)
    min_len = min(int(primary_flat.numel()), int(negative_flat.numel()))
    if min_len <= 0:
        return primary_flat
    return (primary_flat[:min_len] - negative_flat[:min_len]).clamp(min=0)


def _use_two_role_difference_selection(role_token_selection_mode):
    return str(role_token_selection_mode).strip().lower() == 'two_role_diff'


def _build_parallel_character_probe_contexts(positive_context, char_configs, ordered_roles):
    if not isinstance(positive_context, torch.Tensor) or positive_context.dim() < 3:
        return None, {}
    context_branches = [positive_context]
    ablation_branch_by_role = {}
    for role_offset, (role_id, char_cfg) in enumerate(zip(ordered_roles, char_configs), start=1):
        role_indices = char_cfg.get('all_token_indices', None)
        if not isinstance(role_indices, (list, tuple)) or len(role_indices) == 0:
            role_indices = list(char_cfg.get('target_token_indices', [])) + list(char_cfg.get('suffix_token_indices', []))
        role_indices = sorted(list(set([int(x) for x in role_indices if int(x) >= 0])))
        ablated_context = positive_context.clone()
        if len(role_indices) > 0:
            max_ctx_tokens = int(ablated_context.shape[1])
            role_indices = [idx for idx in role_indices if idx < max_ctx_tokens]
            if len(role_indices) > 0:
                ablated_context[:, role_indices, :] = 0
        context_branches.append(ablated_context)
        ablation_branch_by_role[str(role_id)] = int(role_offset)
    return torch.cat(context_branches, dim=0), ablation_branch_by_role


def _convert_parallel_probe_responses_to_role_diffs(per_char_step_maps, ordered_roles, ablation_branch_by_role):
    if not isinstance(per_char_step_maps, list) or not isinstance(ordered_roles, list):
        return []
    use_n = min(len(per_char_step_maps), len(ordered_roles))
    out = []
    for char_idx in range(use_n):
        role_id = str(ordered_roles[char_idx])
        branch_idx = int(ablation_branch_by_role.get(role_id, -1))
        step_maps = per_char_step_maps[char_idx]
        role_maps = {}
        if isinstance(step_maps, dict) and branch_idx >= 1:
            for layer_idx, amap in step_maps.items():
                if not isinstance(amap, torch.Tensor) or amap.dim() < 1:
                    continue
                if int(amap.shape[0]) <= branch_idx:
                    continue
                base_map = amap[0].detach().to(device='cpu', dtype=torch.float32)
                ablated_map = amap[branch_idx].detach().to(device='cpu', dtype=torch.float32)
                if base_map.shape != ablated_map.shape:
                    min_shape = tuple(min(a, b) for a, b in zip(base_map.shape, ablated_map.shape))
                    base_map = base_map[tuple(slice(0, s) for s in min_shape)]
                    ablated_map = ablated_map[tuple(slice(0, s) for s in min_shape)]
                role_maps[int(layer_idx)] = (base_map - ablated_map).clamp(min=0)
        out.append(role_maps)
    return out


def _run_parallel_character_semantic_probe(
    probe_pipe,
    dit_model,
    x,
    timestep,
    positive_context,
    char_configs,
    ordered_roles,
    target_layer,
    extra_forward_kwargs=None,
    capture_feature_tokens=False,
    feature_source='attn_out',
    feature_keep_dtype=torch.bfloat16,
    extract_debug_events=None,
):
    context_input, ablation_branch_by_role = _build_parallel_character_probe_contexts(
        positive_context=positive_context,
        char_configs=char_configs,
        ordered_roles=ordered_roles,
    )
    if context_input is None or len(ablation_branch_by_role) == 0:
        _append_extract_debug_event(
            extract_debug_events,
            stage='parallel_layer7_probe_context_build_failed',
            target_layer=int(target_layer),
            ordered_roles=[str(x) for x in ordered_roles] if isinstance(ordered_roles, list) else None,
        )
        return [], None

    base_batch = int(x.shape[0]) if isinstance(x, torch.Tensor) and x.dim() > 0 else 1
    repeat_factor = int(context_input.shape[0] // max(base_batch, 1))
    x_input = _repeat_probe_batch_tensor(x, repeat_factor, base_batch)
    t_input = _repeat_probe_batch_tensor(timestep, repeat_factor, base_batch)
    forward_kwargs = _repeat_probe_batch_kwargs(extra_forward_kwargs if isinstance(extra_forward_kwargs, dict) else {}, repeat_factor, base_batch)

    probe_extractor = MultiCharacterAttentionMapExtractor(
        probe_pipe,
        [int(target_layer)],
        char_configs,
        cfg_scale=1.0,
    )
    probe_extractor.register_hooks()

    feature_tap = None
    if capture_feature_tokens:
        feature_tap = AttentionOutputFeatureTap(
            dit_model=dit_model,
            layer_idx=int(target_layer),
            keep_device='cpu',
            keep_dtype=feature_keep_dtype,
            source=str(feature_source),
            batch_index=0,
        )
        feature_tap.register()

    forward_stopper = ForwardStopAfterLayer(dit_model, int(target_layer))
    forward_stopper.register()
    try:
        try:
            if hasattr(probe_pipe, 'set_active_noise_domain_from_timestep'):
                probe_pipe.set_active_noise_domain_from_timestep(t_input)
            _ = run_native_dit_forward(
                dit_model,
                x=x_input,
                timestep=t_input,
                context=context_input,
                **forward_kwargs,
            )
        except _StopForwardAfterLayer:
            pass
        raw_hook_summary = {}
        if isinstance(getattr(probe_extractor, 'attention_maps', None), dict):
            for (layer_idx, char_idx), dq in probe_extractor.attention_maps.items():
                raw_hook_summary[f"char{int(char_idx)}_layer{int(layer_idx)}"] = int(len(dq))
        _append_extract_debug_event(
            extract_debug_events,
            stage='parallel_layer7_probe_forward_summary',
            target_layer=int(target_layer),
            repeat_factor=int(repeat_factor),
            raw_hook_summary=raw_hook_summary,
        )
        per_char_step_maps = probe_extractor.get_attention_maps_per_character()
        per_char_step_maps = _convert_parallel_probe_responses_to_role_diffs(
            per_char_step_maps=per_char_step_maps,
            ordered_roles=ordered_roles,
            ablation_branch_by_role=ablation_branch_by_role,
        )
    finally:
        forward_stopper.remove()
        if feature_tap is not None:
            layer_tokens = feature_tap.pop_tokens()
            feature_tap.remove()
        else:
            layer_tokens = None
        probe_extractor.remove_hooks()

    return per_char_step_maps, layer_tokens


def extract_patch_embeddings_for_character(
    pipe,
    memory_video_tensor,  # [C, T, H, W]
    character_id,
    original_prompt,  # 🔥 新增：原始完整 prompt
    extract_layers,
    # [新增] 必须在这里加上 latents 参数，否则上面调用时传不进来
    latents=None,         
    cached_conditions=None,
    top_visual_tokens=-1,
    top_visual_tokens_per_head=0,
    otsu_scope="frame",
    token_weight=1,
    max_tokens=-1,
    cfg_scale=5.0,
    device="cuda",
    debug=False,
    tiler_kwargs=None,
    suffix_attention_scale=1.0,
    **kwargs
):
    """
    Extract selected memory feature triplet for a character using attention maps.

    Returns:
        memory_feature_tokens_selected: [N, 5120] tensor
        memory_feature_positions_selected: [N, 3] tensor of (frame, h, w)
        memory_feature_token_meta_selected: list[dict]
    """
    C, T, H, W = memory_video_tensor.shape
    role_token_selection_mode = str(kwargs.get('role_token_selection_mode', 'baseline')).strip().lower()
    extract_debug_events = kwargs.get('extract_debug_events', None)
    
    if latents is None:
        # 兼容旧逻辑：如果没有传入 latents，则现场计算
        vae_dtype = _get_vae_runtime_dtype(pipe.vae, default_dtype=torch.bfloat16)
        memory_video_for_vae = memory_video_tensor.to(dtype=vae_dtype, device=device)
        
        with torch.no_grad():
            # VAE encode
            original_vae_dtype = _get_vae_runtime_dtype(pipe.vae, default_dtype=torch.bfloat16)
            pipe.vae = _move_vae_runtime(pipe.vae, dtype=vae_dtype)
            
            if tiler_kwargs is None:
                tiler_kwargs = {'tiled': False}
            latents = _safe_vae_encode_isolated(pipe.vae, [memory_video_for_vae], device=device, op_name='extract_patch_embeddings_for_character.vae.encode', **tiler_kwargs)
            latents = latents[0]  # [C_lat, F_lat, H_lat, W_lat]
            latents = latents.to(device=device)
            
            pipe.vae = _move_vae_runtime(pipe.vae, dtype=original_vae_dtype)
            
            del memory_video_for_vae
    else:
        # 使用传入的缓存 (确保在 device 上)
        latents = latents.to(device=device)
    
    C_lat, F_lat, H_lat, W_lat = latents.shape
    
    # 获取 DiT patch_size，计算实际 attention/patch 空间分辨率
    # Wan 14B patch_size=(1,2,2): 实际注意力空间为 (H_lat//2, W_lat//2)
    _patch_size = pipe.dit.patch_size  # e.g., (1, 2, 2)
    H_patch = H_lat // _patch_size[1]  # 60 // 2 = 30
    W_patch = W_lat // _patch_size[2]  # 104 // 2 = 52
    spatial_shape = (H_patch, W_patch)

    
    # Debug: Check channel count
    if debug:
        print(f"[extract_patch_embeddings] C_lat={C_lat}, F_lat={F_lat}, H_lat={H_lat}, W_lat={W_lat}")
        if hasattr(pipe.dit, 'has_image_input') and pipe.dit.has_image_input:
            print(f"[extract_patch_embeddings] has_image_input=True, expected channels for patchify: {C_lat + 4}")
    
    # Add batch dimension
    latents_batch = latents.unsqueeze(0).to(device=device, dtype=pipe.torch_dtype)  # [1, C_lat, F_lat, H_lat, W_lat
    
    # 🔥 使用原始完整 prompt 编码 context
    # Text encode with CFG
    if cfg_scale > 1.0:
        prompt_embeds_pos = pipe.prompter.encode_prompt(prompt=original_prompt, positive=True, device=device)
        prompt_embeds_neg = pipe.prompter.encode_prompt(prompt="", positive=False, device=device)
        prompt_embeds = torch.cat([prompt_embeds_neg, prompt_embeds_pos], dim=0)
        positive_prompt_embeds = prompt_embeds_pos
    else:
        prompt_embeds = pipe.prompter.encode_prompt(prompt=original_prompt, positive=True, device=device)
        positive_prompt_embeds = prompt_embeds
    
    prompt_embeds = prompt_embeds.to(device=device)
    positive_prompt_embeds = positive_prompt_embeds.to(device=device)
    
    # 🔥 解析角色名（用于查找 token）；热力图查询时忽略 _ - 等特殊符号，只考虑文本 token
    character_name = character_id.replace('_', ' ').replace('-', ' ').strip()
    parts = character_id.replace('-', '_').split('_', 1) 
    prefix_text = parts[0]
    suffix_text = parts[1].replace('_', ' ') if len(parts) > 1 else None
    
    # 🔥 新逻辑：先查找完整角色名，再根据结构划分前后缀
    # 1. 获取完整角色名的 token 信息
    token_ids, token_texts, _ = verify_target_text_is_single_token(pipe, character_name)
    if not token_ids:
        _append_extract_debug_event(
            extract_debug_events,
            char_id=str(character_id),
            stage='no_token_ids_for_character_name',
            character_name=str(character_name),
        )
        return None, None, []

    # 2. 在原始 prompt 中查找完整角色名的位置
    full_indices = find_token_index_in_prompt(pipe, original_prompt, character_name, token_ids, token_texts)
    
    # 🔥 如果角色名不在 prompt 中，返回空结果（该角色无 memory）
    if not full_indices:
        if debug:
            print(f"[extract_patch_embeddings] Character '{character_name}' not found in prompt: '{original_prompt[:100]}...'")
        _append_extract_debug_event(
            extract_debug_events,
            char_id=str(character_id),
            stage='character_not_found_in_prompt',
            character_name=str(character_name),
            prompt_preview=str(original_prompt[:160]),
        )
        return torch.zeros(0, 5120, dtype=torch.float32), torch.zeros(0, 3, dtype=torch.long), []
    
    # 3. 根据 character_id 结构划分 prefix 和 suffix
    # 计算 prefix 和 suffix 各占多少个 token
    prefix_token_ids, _, _ = verify_target_text_is_single_token(pipe, prefix_text)
    num_prefix_tokens = len(prefix_token_ids) if prefix_token_ids else 0
    
    if suffix_text:
        suffix_token_ids, _, _ = verify_target_text_is_single_token(pipe, suffix_text)
        num_suffix_tokens = len(suffix_token_ids) if suffix_token_ids else 0
    else:
        num_suffix_tokens = 0
    
    # 4. 从完整位置列表中切片得到 prefix 和 suffix 的位置
    # full_indices 是连续的 token 位置，例如 [5, 6] 表示 "woman blonde"
    prefix_indices = full_indices[:num_prefix_tokens]
    suffix_indices = full_indices[num_prefix_tokens:num_prefix_tokens + num_suffix_tokens] if num_suffix_tokens > 0 else []
    
    if debug:
        print(f"[extract_patch_embeddings] Character '{character_name}' found at positions {full_indices}")
        print(f"  Prefix '{prefix_text}' -> indices {prefix_indices}")
        print(f"  Suffix '{suffix_text}' -> indices {suffix_indices}")

    char_configs = [{
        'target_token_indices': prefix_indices,
        'suffix_token_indices': suffix_indices,
        'all_token_indices': list(full_indices),
        'suffix_scale': suffix_attention_scale,
        'token_weight': token_weight,
    }]

    # 4. 实例化 Extractor (新用法)
    # 直接将两组索引和 Scale 传入
    attn_extractor = AttentionMapExtractor(
        pipe, 
        extract_layers, 
        target_token_indices=prefix_indices,  # 前缀 (权重 1.0)
        suffix_token_indices=suffix_indices,  # 后缀 (权重由 scale 控制)
        suffix_scale=suffix_attention_scale,  # 传入参数
        cfg_scale=cfg_scale, 
        token_weight=token_weight
    )
    
    # Single-step extraction only: extraction timestep must be provided by caller.
    forced_timestep = kwargs.get('forced_timestep', None)
    if forced_timestep is None:
        raise ValueError("Single-step extraction requires forced_timestep, but got None.")
    forced_timestep = float(forced_timestep)
    extract_timesteps = torch.tensor([forced_timestep], device=device, dtype=latents_batch.dtype)
    if debug:
        print(f"[extract_patch_embeddings] Single-step extraction with forced timestep: {forced_timestep}")
    
    # Prepare image conditioning (computed once, reused across steps)
    image_cond_kwargs = {}
    if hasattr(pipe.dit, 'has_image_input') and pipe.dit.has_image_input:
        if cached_conditions is not None:
            if cached_conditions.get('clip_feature', None) is not None:
                image_cond_kwargs['clip_feature'] = cached_conditions['clip_feature']
            image_cond_kwargs['y'] = cached_conditions['y']
        else:
            first_frame_tensor = memory_video_tensor[:, 0, :, :]
            first_frame_np = first_frame_tensor.permute(1, 2, 0).cpu().numpy()
            first_frame_np = ((first_frame_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            first_frame = Image.fromarray(first_frame_np)

            cond = pipe.encode_images_adaptive(
                [first_frame],
                first_frame,
                F_lat * 4 - 3,
                H,
                W,
                use_first_aug=False,
                ref_pad_cfg=False,
                ref_pad_num=0,
            )
            clip_feature = cond.get('clip_feature', None)
            y = cond.get('y', None)
            if y is None:
                raise RuntimeError("encode_images_adaptive did not return y for extraction")

            if cfg_scale > 1.0:
                if clip_feature is not None:
                    clip_feature = torch.cat([clip_feature, clip_feature], dim=0)
                y = torch.cat([y, y], dim=0)
            if clip_feature is not None:
                image_cond_kwargs['clip_feature'] = clip_feature.to(device=device, dtype=latents_batch.dtype)
            image_cond_kwargs['y'] = y.to(device=device, dtype=latents_batch.dtype)
            del cond, clip_feature, y
    layer7_probe_image_cond_kwargs = _make_single_positive_condition_kwargs(image_cond_kwargs)
    
    use_sparse_role_feature = (
        bool(kwargs.get('enable_sparse_role_memory_attn', True))
        and str(kwargs.get('sparse_role_memory_feature_source', 'attn_out')).strip().lower() in ('attn_out', 'self_attn_out')
    )
    use_hybrid_feature = (
        str(kwargs.get('memory_similarity_mode', 'hybrid')).strip().lower() == 'hybrid_feature'
        and (not bool(kwargs.get('hybrid_feature_use_legacy_memory_key', False)))
    ) or use_sparse_role_feature
    feature_layer_idx = int(kwargs.get('feature_match_layer_idx', 7))
    jigsaw_extra_enabled = _memory_encoder_enabled(kwargs.get('jigsaw_extra_encoder_mode', 'off'))
    feature_layer_indices = [int(feature_layer_idx)]
    if jigsaw_extra_enabled:
        feature_layer_indices = _parse_layer_indices_csv(
            kwargs.get('jigsaw_extra_encoder_layers', kwargs.get('sparse_role_memory_injection_layers', None)),
            fallback_idx=feature_layer_idx,
        )
    feature_keep_dtype = torch.bfloat16 if str(kwargs.get('feature_vector_dtype', 'bfloat16')).lower() == 'bfloat16' else torch.float16

    # Single-step forward; keep accumulator structure for minimal code churn.
    running_sum = None
    step_count = 0
    last_layer_tokens = None
    for step_idx, t_val in enumerate(extract_timesteps):
        noise = torch.randn_like(latents_batch)
        timestep_tensor = t_val.unsqueeze(0).to(device=device, dtype=latents_batch.dtype)
        noisy_latents = (1 - t_val / pipe.scheduler.num_train_timesteps) * latents_batch + \
                        (t_val / pipe.scheduler.num_train_timesteps) * noise
        if cfg_scale > 1.0:
            noisy_input = torch.cat([noisy_latents, noisy_latents], dim=0)
        else:
            noisy_input = noisy_latents
        noisy_input = noisy_input.to(device=device)
        if role_token_selection_mode == 'layer7_single':
            probe_layer_idx = int(kwargs.get('sparse_role_memory_layer_idx', feature_layer_idx))
            per_char_step_maps, last_layer_tokens = _run_parallel_character_semantic_probe(
                probe_pipe=pipe,
                dit_model=pipe.dit,
                x=noisy_latents,
                timestep=timestep_tensor,
                positive_context=positive_prompt_embeds,
                char_configs=char_configs,
                ordered_roles=[str(character_id)],
                target_layer=probe_layer_idx,
                extra_forward_kwargs=layer7_probe_image_cond_kwargs,
                capture_feature_tokens=bool(use_hybrid_feature),
                feature_source=str(kwargs.get('sparse_role_memory_feature_source', 'attn_out')),
                feature_keep_dtype=feature_keep_dtype,
            )
            step_attention_maps = per_char_step_maps[0] if isinstance(per_char_step_maps, list) and len(per_char_step_maps) > 0 else {}
        else:
            attn_extractor.register_hooks()
            feature_taps = []
            if use_hybrid_feature:
                for tap_layer_idx in feature_layer_indices:
                    feature_tap = AttentionOutputFeatureTap(
                        dit_model=pipe.dit,
                        layer_idx=int(tap_layer_idx),
                        keep_device='cpu',
                        keep_dtype=feature_keep_dtype,
                        source=str(kwargs.get('sparse_role_memory_feature_source', 'attn_out')),
                    )
                    feature_tap.register()
                    feature_taps.append((int(tap_layer_idx), feature_tap))
            try:
                if hasattr(pipe, 'set_active_noise_domain_from_timestep'):
                    pipe.set_active_noise_domain_from_timestep(timestep_tensor)
                dit_kwargs = {'x': noisy_input, 'timestep': timestep_tensor, 'context': prompt_embeds, **image_cond_kwargs}
                with torch.no_grad():
                    _ = run_native_dit_forward(pipe.dit, **dit_kwargs)
            except Exception:
                for _, feature_tap in feature_taps:
                    feature_tap.remove()
                attn_extractor.remove_hooks()
                raise
            captured_by_layer = {}
            for tap_layer_idx, feature_tap in feature_taps:
                captured_tokens = feature_tap.pop_tokens()
                feature_tap.remove()
                if isinstance(captured_tokens, torch.Tensor) and captured_tokens.dim() == 2:
                    captured_by_layer[str(int(tap_layer_idx))] = captured_tokens
            if len(captured_by_layer) > 1 or jigsaw_extra_enabled:
                last_layer_tokens = _pm_make_layerwise_container(captured_by_layer)
            elif len(captured_by_layer) == 1:
                last_layer_tokens = next(iter(captured_by_layer.values()))
            step_attention_maps = attn_extractor.get_attention_maps()
            attn_extractor.remove_hooks()
        if not step_attention_maps:
            del noise, noisy_latents, noisy_input
            continue
        step_aggregated = _aggregate_character_semantic_responses_cpu(step_attention_maps)
        if step_aggregated is not None:
            if running_sum is None:
                running_sum = step_aggregated.clone()
            else:
                min_shape = tuple(min(s1, s2) for s1, s2 in zip(running_sum.shape, step_aggregated.shape))
                running_sum = running_sum[tuple(slice(0, s) for s in min_shape)]
                step_aggregated = step_aggregated[tuple(slice(0, s) for s in min_shape)]
                running_sum += step_aggregated
            step_count += 1
        del step_attention_maps, noise, noisy_latents, noisy_input
    
    if running_sum is None or step_count == 0:
        _append_extract_debug_event(
            extract_debug_events,
            char_id=str(character_id),
            stage='no_attention_maps_accumulated',
            role_token_selection_mode=str(role_token_selection_mode),
            forced_timestep=float(forced_timestep),
        )
        return None, None, []
    aggregated_map = running_sum / step_count
    del running_sum
    if debug:
        print(f"[extract_patch_embeddings] Attention aggregated over {step_count} steps")
    del image_cond_kwargs, prompt_embeds
    
    # 修复: 若仍有多维(理论上经过head均值后已是1D), 保险取均值而非flatten
    if aggregated_map.dim() > 1:
        aggregated_map = aggregated_map.mean(dim=0)
    
    # Calculate frame count from tokens (使用 patch 空间分辨率)
    num_tokens = aggregated_map.shape[0]
    spatial_per_frame = H_patch * W_patch  # 30 * 52 = 1560 (非 H_lat * W_lat)
    F_actual = num_tokens // spatial_per_frame
    
    # Select tokens
    neighbor_filter_kernel = kwargs.get('neighbor_filter_kernel', 0)
    neighbor_filter_any_window = bool(kwargs.get('neighbor_filter_any_window', True))
    binary_mask, continuous_map, selected_indices, frame_info = process_attention_map_to_mask(
        aggregated_map,
        threshold=top_visual_tokens,
        top_k_per_head=top_visual_tokens_per_head,
        spatial_shape=spatial_shape,
        num_frames=F_actual,
        otsu_scope=otsu_scope,
        neighbor_filter_kernel=neighbor_filter_kernel,
        neighbor_filter_any_window=neighbor_filter_any_window,
    )
    
    # Convert set to list for indexing operations
    if isinstance(selected_indices, set):
        selected_indices = list(selected_indices)
    
    if len(selected_indices) == 0:
        _append_extract_debug_event(
            extract_debug_events,
            char_id=str(character_id),
            stage='empty_selected_indices',
            num_tokens=int(num_tokens),
            spatial_shape=tuple(spatial_shape),
            num_frames=int(F_actual),
            top_visual_tokens=float(top_visual_tokens),
            top_visual_tokens_per_head=int(top_visual_tokens_per_head),
            otsu_scope=str(otsu_scope),
            neighbor_filter_kernel=int(neighbor_filter_kernel),
            neighbor_filter_any_window=bool(neighbor_filter_any_window),
        )
        return None, None, []
    
    # Apply max_tokens limit
    if debug:
        print(f"[DEBUG extract_patch] max_tokens={max_tokens}, len(selected_indices)={len(selected_indices)}")
    if max_tokens > 0 and len(selected_indices) > max_tokens:
        if kwargs.get("use_attn_score_selection", False):
            # 从 aggregated_map 中提取对应索引的分数
            scores = [aggregated_map[idx].item() for idx in selected_indices]
            # 按分数从高到低排序，取前 max_tokens 个
            scored_indices = sorted(zip(selected_indices, scores), key=lambda x: x[1], reverse=True)
            selected_indices = [x[0] for x in scored_indices[:max_tokens]]
        else:
            # 原有的随机筛选逻辑
            indices = torch.randperm(len(selected_indices))[:max_tokens]
            indices = indices.tolist()
            selected_indices = [selected_indices[i] for i in indices]

    # [关键] 无论何种筛选方式，最后必须按原始索引排序以对齐可学习位置编码
    selected_indices = sorted(selected_indices)
    if _pm_is_layerwise_container(last_layer_tokens):
        attention_map_extra = dict(kwargs)
        for _explicit_key in (
            'top_visual_tokens',
            'top_visual_tokens_per_head',
            'otsu_scope',
            'neighbor_filter_kernel',
            'neighbor_filter_any_window',
            'max_tokens',
            'debug',
        ):
            attention_map_extra.pop(_explicit_key, None)
        return _attention_map_to_patch_embeddings(
            aggregated_map,
            latents,
            pipe,
            device,
            spatial_shape,
            H_patch,
            W_patch,
            F_lat,
            C_lat,
            character_id,
            top_visual_tokens=top_visual_tokens,
            top_visual_tokens_per_head=top_visual_tokens_per_head,
            otsu_scope=otsu_scope,
            neighbor_filter_kernel=neighbor_filter_kernel,
            max_tokens=max_tokens,
            debug=debug,
            neighbor_filter_any_window=neighbor_filter_any_window,
            source_token_features=last_layer_tokens,
            **attention_map_extra,
        )
    # Extract memory feature vectors at selected positions.
    use_layer_feature_tokens = bool(isinstance(last_layer_tokens, torch.Tensor) and last_layer_tokens.dim() == 2)
    if not use_layer_feature_tokens:
        if debug:
            print("[extract_patch_embeddings] missing attn-out feature tokens at extraction timestep; skip memory tokens")
        _append_extract_debug_event(
            extract_debug_events,
            char_id=str(character_id),
            stage='missing_attn_out_feature_tokens',
            role_token_selection_mode=str(role_token_selection_mode),
            feature_layer_idx=int(feature_layer_idx),
            selected_count=int(len(selected_indices)),
            feature_is_tensor=bool(isinstance(last_layer_tokens, torch.Tensor)),
            feature_shape=(tuple(last_layer_tokens.shape) if isinstance(last_layer_tokens, torch.Tensor) else None),
        )
        return None, None, []
    
    # Extract selected memory features at selected positions (使用 patch grid 分辨率)
    memory_feature_tokens_selected = []
    memory_feature_positions_selected = []
    
    total_spatial = H_patch * W_patch  # patch 空间每帧 token 数 (30*52=1560)
    
    for idx in selected_indices:
        frame_idx = idx // total_spatial
        spatial_idx = idx % total_spatial
        h = spatial_idx // W_patch
        w = spatial_idx % W_patch
        
        if frame_idx >= F_lat:
            continue
        
        if idx < last_layer_tokens.shape[0]:
            patch_emb = last_layer_tokens[idx]
            memory_feature_tokens_selected.append(patch_emb.detach().cpu())
            memory_feature_positions_selected.append(torch.tensor([frame_idx, h, w]))
    
    if not memory_feature_tokens_selected:
        valid_idx_count = sum(1 for idx in selected_indices if idx < int(last_layer_tokens.shape[0]))
        _append_extract_debug_event(
            extract_debug_events,
            char_id=str(character_id),
            stage='empty_feature_tokens_after_indexing',
            selected_count=int(len(selected_indices)),
            valid_feature_index_count=int(valid_idx_count),
            feature_seq_len=int(last_layer_tokens.shape[0]),
            total_spatial=int(total_spatial),
            F_lat=int(F_lat),
        )
        return torch.zeros(0, 5120, dtype=pipe.torch_dtype, device=device), torch.zeros(0, 3, dtype=torch.long, device=device), []
    _append_extract_debug_event(
        extract_debug_events,
        char_id=str(character_id),
        stage='feature_tokens_selected',
        final_token_count=int(len(memory_feature_tokens_selected)),
        selected_count=int(len(selected_indices)),
    )
    
    memory_feature_tokens_selected = torch.stack(memory_feature_tokens_selected)  # [N, 5120]
    memory_feature_positions_selected = torch.stack(memory_feature_positions_selected)  # [N, 3]
    
    selected_scores = []
    for idx in selected_indices:
        if idx < int(aggregated_map.shape[0]):
            selected_scores.append(float(aggregated_map[idx].detach().float().item()))
        else:
            selected_scores.append(0.0)
    token_meta = _build_bbox_relative_token_meta(
        character_id,
        memory_feature_positions_selected,
        F_lat,
        similarity_scores=selected_scores,
    )
    return memory_feature_tokens_selected, memory_feature_positions_selected, token_meta


def _attention_map_to_patch_embeddings(
    aggregated_map, latents, pipe, device, spatial_shape, H_patch, W_patch, F_lat, C_lat, char_id,
    top_visual_tokens=-1, top_visual_tokens_per_head=0, otsu_scope="frame", neighbor_filter_kernel=0,
    max_tokens=-1, debug=False, neighbor_filter_any_window=True, source_token_features=None, **kwargs
):
    """从单角色聚合 attention map 生成 memory feature 选中三元组。"""
    extract_debug_events = kwargs.get('extract_debug_events', None)
    if aggregated_map.dim() > 1:
        aggregated_map = aggregated_map.mean(dim=0)
    num_tokens = aggregated_map.shape[0]
    spatial_per_frame = H_patch * W_patch
    F_actual = num_tokens // spatial_per_frame
    binary_mask, continuous_map, selected_indices, frame_info = process_attention_map_to_mask(
        aggregated_map,
        threshold=top_visual_tokens,
        top_k_per_head=top_visual_tokens_per_head,
        spatial_shape=spatial_shape,
        num_frames=F_actual,
        otsu_scope=otsu_scope,
        neighbor_filter_kernel=neighbor_filter_kernel,
        neighbor_filter_any_window=bool(neighbor_filter_any_window),
    )
    if isinstance(selected_indices, set):
        selected_indices = list(selected_indices)
    if len(selected_indices) == 0:
        _append_extract_debug_event(
            extract_debug_events,
            char_id=str(char_id),
            stage='empty_selected_indices',
            num_tokens=int(num_tokens),
            spatial_shape=tuple(spatial_shape),
            num_frames=int(F_actual),
            top_visual_tokens=float(top_visual_tokens),
            top_visual_tokens_per_head=int(top_visual_tokens_per_head),
            otsu_scope=str(otsu_scope),
            neighbor_filter_kernel=int(neighbor_filter_kernel),
            neighbor_filter_any_window=bool(neighbor_filter_any_window),
        )
        return None, None, []
    if max_tokens > 0 and len(selected_indices) > max_tokens:
        if kwargs.get("use_attn_score_selection", False):
            scores = [aggregated_map[idx].item() for idx in selected_indices]
            scored_indices = sorted(zip(selected_indices, scores), key=lambda x: x[1], reverse=True)
            selected_indices = [x[0] for x in scored_indices[:max_tokens]]
        else:
            indices = torch.randperm(len(selected_indices))[:max_tokens].tolist()
            selected_indices = [selected_indices[i] for i in indices]
    selected_indices = sorted(selected_indices)

    layer_token_sources = dict(_pm_iter_layerwise_items(source_token_features)) if _pm_is_layerwise_container(source_token_features) else None
    use_layer_feature_tokens = bool(
        (isinstance(source_token_features, torch.Tensor) and source_token_features.dim() == 2)
        or (isinstance(layer_token_sources, dict) and any(isinstance(v, torch.Tensor) and v.dim() == 2 for v in layer_token_sources.values()))
    )
    if not use_layer_feature_tokens:
        if debug:
            print("[_attention_map_to_patch_embeddings] missing attn-out feature tokens; skip this character")
        _append_extract_debug_event(
            extract_debug_events,
            char_id=str(char_id),
            stage='missing_source_token_features',
            selected_count=int(len(selected_indices)),
            feature_is_tensor=bool(isinstance(source_token_features, torch.Tensor)),
            feature_dim=(tuple(source_token_features.shape) if isinstance(source_token_features, torch.Tensor) else None),
        )
        return None, None, []
    token_source_for_bounds = source_token_features
    if isinstance(layer_token_sources, dict) and len(layer_token_sources) > 0:
        token_source_for_bounds = next((v for v in layer_token_sources.values() if isinstance(v, torch.Tensor) and v.dim() == 2), None)
    memory_feature_tokens_selected, memory_feature_positions_selected = [], []
    total_spatial = H_patch * W_patch
    for idx in selected_indices:
        frame_idx = idx // total_spatial
        spatial_idx = idx % total_spatial
        h, w = spatial_idx // W_patch, spatial_idx % W_patch
        if frame_idx >= F_lat:
            continue
        if isinstance(token_source_for_bounds, torch.Tensor) and idx < token_source_for_bounds.shape[0]:
            memory_feature_tokens_selected.append(token_source_for_bounds[idx].detach().cpu())
            memory_feature_positions_selected.append(torch.tensor([frame_idx, h, w]))
    if not memory_feature_tokens_selected:
        valid_idx_count = sum(1 for idx in selected_indices if isinstance(token_source_for_bounds, torch.Tensor) and idx < int(token_source_for_bounds.shape[0]))
        _append_extract_debug_event(
            extract_debug_events,
            char_id=str(char_id),
            stage='empty_feature_tokens_after_indexing',
            selected_count=int(len(selected_indices)),
            valid_feature_index_count=int(valid_idx_count),
            feature_seq_len=int(token_source_for_bounds.shape[0]) if isinstance(token_source_for_bounds, torch.Tensor) else -1,
            total_spatial=int(total_spatial),
            F_lat=int(F_lat),
        )
        return torch.zeros(0, 5120, dtype=pipe.torch_dtype, device=device), torch.zeros(0, 3, dtype=torch.long, device=device), []
    _append_extract_debug_event(
        extract_debug_events,
        char_id=str(char_id),
        stage='feature_tokens_selected',
        selected_count=int(len(selected_indices)),
        final_token_count=int(len(memory_feature_tokens_selected)),
    )
    positions_t = torch.stack(memory_feature_positions_selected)
    selected_scores = []
    for idx in selected_indices:
        if idx < int(aggregated_map.shape[0]):
            selected_scores.append(float(aggregated_map[idx].detach().float().item()))
        else:
            selected_scores.append(0.0)
    token_meta = _build_bbox_relative_token_meta(
        char_id,
        positions_t,
        F_lat,
        similarity_scores=selected_scores,
    )
    if isinstance(layer_token_sources, dict) and len(layer_token_sources) > 0:
        layer_memory_tokens = {}
        for layer, layer_source in layer_token_sources.items():
            if not (isinstance(layer_source, torch.Tensor) and layer_source.dim() == 2):
                continue
            layer_valid_indices = [int(idx) for idx in selected_indices if int(idx) < int(layer_source.shape[0])]
            if len(layer_valid_indices) <= 0:
                continue
            layer_memory_tokens[str(layer)] = layer_source.index_select(
                0,
                torch.tensor(layer_valid_indices, dtype=torch.long, device=layer_source.device),
            ).detach().cpu()
        if len(layer_memory_tokens) > 0:
            return (
                _pm_make_layerwise_container(layer_memory_tokens),
                positions_t,
                _pm_make_layerwise_container({layer: list(token_meta) for layer in layer_memory_tokens.keys()}),
            )
    return torch.stack(memory_feature_tokens_selected), positions_t, token_meta


def extract_patch_embeddings_for_characters_batch(
    pipe, memory_video_tensor, character_ids, original_prompt, latents,
    cached_conditions, extract_layers, device, extract_config, cfg_scale=5.0,
    suffix_attention_scale=1.0, token_weight=0.2, tiler_kwargs=None, debug=False, **kwargs
):
    """
    多角色共用同一段 step 循环，只做一次 DiT 多步 forward，再按角色聚合 map 并分别做 patch 提取。
    返回 list[(pe, pos, meta)] 与 character_ids 同序；未在 prompt 中的角色为 (None, None, []).
    """
    if isinstance(extract_layers, str):
        extract_layers = [-1] if extract_layers.strip() == '-1' else [int(x.strip()) for x in extract_layers.split(',')]
    role_token_selection_mode = str(kwargs.get('role_token_selection_mode', extract_config.get('role_token_selection_mode', 'baseline'))).strip().lower()
    extract_debug_events = kwargs.get('extract_debug_events', None)
    C, T, H, W = memory_video_tensor.shape
    if latents is None:
        with torch.no_grad():
            vae_dtype = _get_vae_runtime_dtype(pipe.vae, default_dtype=torch.bfloat16)
            memory_video_for_vae = memory_video_tensor.to(dtype=vae_dtype, device=device)
            original_vae_dtype = _get_vae_runtime_dtype(pipe.vae, default_dtype=torch.bfloat16)
            pipe.vae = _move_vae_runtime(pipe.vae, dtype=vae_dtype)
            if tiler_kwargs is None:
                tiler_kwargs = {'tiled': False}
            latents = _safe_vae_encode_isolated(pipe.vae, [memory_video_for_vae], device=device, op_name='extract_patch_embeddings_for_characters_batch.vae.encode', **tiler_kwargs)[0]
            latents = latents.to(device=device)
            pipe.vae = _move_vae_runtime(pipe.vae, dtype=original_vae_dtype)
            del memory_video_for_vae
    else:
        latents = latents.to(device=device)
    C_lat, F_lat, H_lat, W_lat = latents.shape
    _patch_size = pipe.dit.patch_size
    H_patch = H_lat // _patch_size[1]
    W_patch = W_lat // _patch_size[2]
    spatial_shape = (H_patch, W_patch)
    latents_batch = latents.unsqueeze(0).to(device=device, dtype=pipe.torch_dtype)
    if cfg_scale > 1.0:
        prompt_embeds_pos = pipe.prompter.encode_prompt(prompt=original_prompt, positive=True, device=device)
        prompt_embeds_neg = pipe.prompter.encode_prompt(prompt="", positive=False, device=device)
        prompt_embeds = torch.cat([prompt_embeds_neg, prompt_embeds_pos], dim=0)
        positive_prompt_embeds = prompt_embeds_pos
    else:
        prompt_embeds = pipe.prompter.encode_prompt(prompt=original_prompt, positive=True, device=device)
        positive_prompt_embeds = prompt_embeds
    prompt_embeds = prompt_embeds.to(device=device)
    positive_prompt_embeds = positive_prompt_embeds.to(device=device)
    use_sparse_role_feature = (
        bool(kwargs.get('enable_sparse_role_memory_attn', extract_config.get('enable_sparse_role_memory_attn', True)))
        and str(kwargs.get('sparse_role_memory_feature_source', extract_config.get('sparse_role_memory_feature_source', 'attn_out'))).strip().lower() in ('attn_out', 'self_attn_out')
    )
    use_hybrid_feature = (
        str(kwargs.get('memory_similarity_mode', extract_config.get('memory_similarity_mode', 'hybrid'))).strip().lower() == 'hybrid_feature'
    ) or use_sparse_role_feature
    feature_layer_idx = int(kwargs.get('feature_match_layer_idx', extract_config.get('feature_match_layer_idx', 7)))
    jigsaw_extra_enabled = _memory_encoder_enabled(kwargs.get('jigsaw_extra_encoder_mode', extract_config.get('jigsaw_extra_encoder_mode', 'off')))
    feature_layer_indices = [int(feature_layer_idx)]
    if jigsaw_extra_enabled:
        feature_layer_indices = _parse_layer_indices_csv(
            kwargs.get('jigsaw_extra_encoder_layers', extract_config.get('jigsaw_extra_encoder_layers', kwargs.get('sparse_role_memory_injection_layers', extract_config.get('sparse_role_memory_injection_layers', None)))),
            fallback_idx=feature_layer_idx,
        )
    feature_keep_dtype = torch.bfloat16 if str(kwargs.get('feature_vector_dtype', extract_config.get('feature_vector_dtype', 'bfloat16'))).lower() == 'bfloat16' else torch.float16

    valid_chars = []
    for character_id in character_ids:
        character_name = character_id.replace('_', ' ').replace('-', ' ').strip()
        parts = character_id.replace('-', '_').split('_', 1)
        prefix_text = parts[0]
        suffix_text = parts[1].replace('_', ' ') if len(parts) > 1 else None
        token_ids, token_texts, _ = verify_target_text_is_single_token(pipe, character_name)
        if not token_ids:
            _append_extract_debug_event(
                extract_debug_events,
                char_id=str(character_id),
                stage='no_token_ids_for_character_name',
                character_name=str(character_name),
            )
            continue
        full_indices = find_token_index_in_prompt(pipe, original_prompt, character_name, token_ids, token_texts)
        if not full_indices:
            _append_extract_debug_event(
                extract_debug_events,
                char_id=str(character_id),
                stage='character_not_found_in_prompt',
                character_name=str(character_name),
                prompt_preview=str(original_prompt[:160]),
            )
            continue
        prefix_token_ids, _, _ = verify_target_text_is_single_token(pipe, prefix_text)
        num_prefix_tokens = len(prefix_token_ids) if prefix_token_ids else 0
        if suffix_text:
            suffix_token_ids, _, _ = verify_target_text_is_single_token(pipe, suffix_text)
            num_suffix_tokens = len(suffix_token_ids) if suffix_token_ids else 0
        else:
            num_suffix_tokens = 0
        prefix_indices = full_indices[:num_prefix_tokens]
        suffix_indices = full_indices[num_prefix_tokens:num_prefix_tokens + num_suffix_tokens] if num_suffix_tokens > 0 else []
        valid_chars.append((character_id, prefix_indices, suffix_indices, full_indices))
    if not valid_chars:
        _append_extract_debug_event(
            extract_debug_events,
            stage='no_valid_characters_after_prompt_match',
            requested_characters=[str(x) for x in character_ids],
        )
        return [(None, None, [])] * len(character_ids)
    if len(valid_chars) == 1:
        char_id, pre, suf, _ = valid_chars[0]
        single_explicit = (
            'pipe', 'memory_video_tensor', 'character_id', 'original_prompt', 'latents', 'cached_conditions',
            'extract_layers', 'device', 'cfg_scale', 'suffix_attention_scale', 'token_weight', 'tiler_kwargs', 'debug'
        )
        extra = {k: v for k, v in {**kwargs, **extract_config}.items() if k not in single_explicit}
        extra['extract_debug_events'] = extract_debug_events
        single_pe, single_pos, single_meta = extract_patch_embeddings_for_character(
            pipe=pipe, memory_video_tensor=memory_video_tensor, character_id=char_id,
            original_prompt=original_prompt, latents=latents, cached_conditions=cached_conditions,
            extract_layers=extract_layers, device=device, cfg_scale=cfg_scale,
            suffix_attention_scale=suffix_attention_scale, token_weight=token_weight,
            tiler_kwargs=tiler_kwargs, debug=debug, **extra
        )
        out = [(None, None, [])] * len(character_ids)
        idx = character_ids.index(char_id)
        out[idx] = (single_pe, single_pos, single_meta)
        return out

    char_configs = [
        {
            'target_token_indices': pre,
            'suffix_token_indices': suf,
            'all_token_indices': list(full),
            'suffix_scale': suffix_attention_scale,
            'token_weight': token_weight,
        }
        for _, pre, suf, full in valid_chars
    ]
    multi_extractor = MultiCharacterAttentionMapExtractor(
        pipe, extract_layers, char_configs, cfg_scale=cfg_scale
    )
    _append_extract_debug_event(
        extract_debug_events,
        stage='multi_char_extractor_init',
        role_token_selection_mode=str(role_token_selection_mode),
        extract_layers=(list(extract_layers) if isinstance(extract_layers, (list, tuple)) else extract_layers),
        valid_characters=[str(x[0]) for x in valid_chars],
        union_token_count=int(len(getattr(multi_extractor, 'union_indices', []))),
        union_indices=(list(getattr(multi_extractor, 'union_indices', []))[:32]),
    )
    forced_timestep = kwargs.get('forced_timestep', extract_config.get('forced_timestep', None))
    if forced_timestep is None:
        raise ValueError("Single-step batch extraction requires forced_timestep, but got None.")
    forced_timestep = float(forced_timestep)
    extract_timesteps = torch.tensor([forced_timestep], device=device, dtype=latents_batch.dtype)
    image_cond_kwargs = {}
    if hasattr(pipe.dit, 'has_image_input') and pipe.dit.has_image_input:
        if cached_conditions is not None:
            if cached_conditions.get('clip_feature', None) is not None:
                image_cond_kwargs['clip_feature'] = cached_conditions['clip_feature']
            image_cond_kwargs['y'] = cached_conditions['y']
        else:
            first_frame_tensor = memory_video_tensor[:, 0, :, :]
            first_frame_np = first_frame_tensor.permute(1, 2, 0).cpu().numpy()
            first_frame_np = ((first_frame_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            first_frame = Image.fromarray(first_frame_np)
            cond = pipe.encode_images_adaptive(
                [first_frame],
                first_frame,
                F_lat * 4 - 3,
                H,
                W,
                use_first_aug=False,
                ref_pad_cfg=False,
                ref_pad_num=0,
            )
            clip_feature = cond.get('clip_feature', None)
            y = cond.get('y', None)
            if y is None:
                raise RuntimeError("encode_images_adaptive did not return y for batch extraction")
            if cfg_scale > 1.0:
                if clip_feature is not None:
                    clip_feature = torch.cat([clip_feature, clip_feature], dim=0)
                y = torch.cat([y, y], dim=0)
            if clip_feature is not None:
                image_cond_kwargs['clip_feature'] = clip_feature.to(device=device, dtype=latents_batch.dtype)
            image_cond_kwargs['y'] = y.to(device=device, dtype=latents_batch.dtype)
    layer7_probe_image_cond_kwargs = _make_single_positive_condition_kwargs(image_cond_kwargs)
    # 每步累加到每角色一个 accumulator，不保存所有 step 的中间 tensor，避免 OOM
    per_char_sum = [None] * len(valid_chars)
    per_char_count = [0] * len(valid_chars)
    shared_layer_tokens = None
    for step_idx, t_val in enumerate(extract_timesteps):
        noise = torch.randn_like(latents_batch)
        timestep_tensor = t_val.unsqueeze(0).to(device=device, dtype=latents_batch.dtype)
        noisy_latents = (1 - t_val / pipe.scheduler.num_train_timesteps) * latents_batch + \
                        (t_val / pipe.scheduler.num_train_timesteps) * noise
        if cfg_scale > 1.0:
            noisy_input = torch.cat([noisy_latents, noisy_latents], dim=0)
        else:
            noisy_input = noisy_latents
        noisy_input = noisy_input.to(device=device)
        if role_token_selection_mode == 'layer7_single':
            probe_layer_idx = int(kwargs.get('sparse_role_memory_layer_idx', extract_config.get('sparse_role_memory_layer_idx', feature_layer_idx)))
            per_char_maps, shared_layer_tokens = _run_parallel_character_semantic_probe(
                probe_pipe=pipe,
                dit_model=pipe.dit,
                x=noisy_latents,
                timestep=timestep_tensor,
                positive_context=positive_prompt_embeds,
                char_configs=char_configs,
                ordered_roles=[str(char_id) for char_id, _, _, _ in valid_chars],
                target_layer=probe_layer_idx,
                extra_forward_kwargs=layer7_probe_image_cond_kwargs,
                capture_feature_tokens=bool(use_hybrid_feature),
                feature_source=str(kwargs.get('sparse_role_memory_feature_source', 'attn_out')),
                feature_keep_dtype=feature_keep_dtype,
                extract_debug_events=extract_debug_events,
            )
        else:
            multi_extractor.register_hooks()
            feature_taps = []
            if use_hybrid_feature:
                for tap_layer_idx in feature_layer_indices:
                    feature_tap = AttentionOutputFeatureTap(
                        dit_model=pipe.dit,
                        layer_idx=int(tap_layer_idx),
                        keep_device='cpu',
                        keep_dtype=feature_keep_dtype,
                        source=str(kwargs.get('sparse_role_memory_feature_source', 'attn_out')),
                    )
                    feature_tap.register()
                    feature_taps.append((int(tap_layer_idx), feature_tap))
            try:
                if hasattr(pipe, 'set_active_noise_domain_from_timestep'):
                    pipe.set_active_noise_domain_from_timestep(timestep_tensor)
                dit_kwargs = {'x': noisy_input, 'timestep': timestep_tensor, 'context': prompt_embeds, **image_cond_kwargs}
                with torch.no_grad():
                    _ = run_native_dit_forward(pipe.dit, **dit_kwargs)
            except Exception:
                for _, feature_tap in feature_taps:
                    feature_tap.remove()
                multi_extractor.remove_hooks()
                raise
            raw_hook_summary = {}
            if isinstance(getattr(multi_extractor, 'attention_maps', None), dict):
                for (layer_idx, char_idx), dq in multi_extractor.attention_maps.items():
                    raw_hook_summary[f"{str(valid_chars[char_idx][0])}@layer{int(layer_idx)}"] = int(len(dq))
            captured_by_layer = {}
            for tap_layer_idx, feature_tap in feature_taps:
                captured_tokens = feature_tap.pop_tokens()
                feature_tap.remove()
                if isinstance(captured_tokens, torch.Tensor) and captured_tokens.dim() == 2:
                    captured_by_layer[str(int(tap_layer_idx))] = captured_tokens
            if len(captured_by_layer) > 1 or jigsaw_extra_enabled:
                shared_layer_tokens = _pm_make_layerwise_container(captured_by_layer)
            elif len(captured_by_layer) == 1:
                shared_layer_tokens = next(iter(captured_by_layer.values()))
            per_char_maps = multi_extractor.get_attention_maps_per_character()
            per_char_map_summary = {}
            for char_idx, char_maps in enumerate(per_char_maps):
                per_char_map_summary[str(valid_chars[char_idx][0])] = sorted([int(k) for k in char_maps.keys()]) if isinstance(char_maps, dict) else []
            _append_extract_debug_event(
                extract_debug_events,
                stage='multi_char_extractor_forward_summary',
                role_token_selection_mode=str(role_token_selection_mode),
                extract_layers=(list(extract_layers) if isinstance(extract_layers, (list, tuple)) else extract_layers),
                raw_hook_summary=raw_hook_summary,
                per_char_map_summary=per_char_map_summary,
                feature_shape=(tuple(shared_layer_tokens.shape) if isinstance(shared_layer_tokens, torch.Tensor) else None),
                feature_layers=(sorted(list(captured_by_layer.keys())) if len(captured_by_layer) > 0 else None),
            )
            multi_extractor.remove_hooks()
        for char_idx, layer_maps in enumerate(per_char_maps):
            step_aggregated = _aggregate_character_semantic_responses_cpu(layer_maps)
            if step_aggregated is not None:
                if per_char_sum[char_idx] is None:
                    per_char_sum[char_idx] = step_aggregated.detach().float().clone()
                else:
                    min_shape = tuple(min(s1, s2) for s1, s2 in zip(per_char_sum[char_idx].shape, step_aggregated.shape))
                    per_char_sum[char_idx] = per_char_sum[char_idx][tuple(slice(0, s) for s in min_shape)]
                    step_aggregated = step_aggregated[tuple(slice(0, s) for s in min_shape)]
                    per_char_sum[char_idx] = per_char_sum[char_idx] + step_aggregated.detach().float()
                per_char_count[char_idx] += 1
        del noise, noisy_latents, noisy_input
    del image_cond_kwargs, prompt_embeds
    out = [(None, None, [])] * len(character_ids)
    opts = {
        'top_visual_tokens': kwargs.get('top_visual_tokens', -1),
        'top_visual_tokens_per_head': kwargs.get('top_visual_tokens_per_head', 0),
        'otsu_scope': kwargs.get('otsu_scope', 'frame'),
        'neighbor_filter_kernel': kwargs.get('neighbor_filter_kernel', 0),
        'neighbor_filter_any_window': bool(kwargs.get('neighbor_filter_any_window', True)),
        'max_tokens': kwargs.get('max_tokens', -1),
        'debug': debug,
        **kwargs
    }
    aggregated_maps = [None] * len(valid_chars)
    for char_idx in range(len(valid_chars)):
        if per_char_sum[char_idx] is None or per_char_count[char_idx] == 0:
            _append_extract_debug_event(
                extract_debug_events,
                char_id=str(valid_chars[char_idx][0]),
                stage='no_attention_maps_accumulated',
                role_token_selection_mode=str(role_token_selection_mode),
                forced_timestep=float(forced_timestep),
            )
            continue
        aggregated_maps[char_idx] = per_char_sum[char_idx] / per_char_count[char_idx]
    if _use_two_role_difference_selection(role_token_selection_mode) and len(valid_chars) == 2:
        agg_map_0 = aggregated_maps[0]
        agg_map_1 = aggregated_maps[1]
        if isinstance(agg_map_0, torch.Tensor) and isinstance(agg_map_1, torch.Tensor):
            aggregated_maps[0] = _suppress_other_character_response_cpu(agg_map_0, agg_map_1)
            aggregated_maps[1] = _suppress_other_character_response_cpu(agg_map_1, agg_map_0)
    for char_idx, (char_id, _, _, _) in enumerate(valid_chars):
        aggregated_map = aggregated_maps[char_idx]
        if not isinstance(aggregated_map, torch.Tensor):
            continue
        pe, pos, meta = _attention_map_to_patch_embeddings(
            aggregated_map,
            latents,
            pipe,
            device,
            spatial_shape,
            H_patch,
            W_patch,
            F_lat,
            C_lat,
            char_id=char_id,
            source_token_features=shared_layer_tokens,
            **opts,
        )
        if pe is not None:
            out[character_ids.index(char_id)] = (pe, pos, meta)
    return out

class InlineExtractThenTrainDataset(IterableDataset):
    """
    Single-process, per-rank extract-then-train dataset.

    Design:
    - Each DDP rank keeps extraction and training on the same GPU/process.
    - For every yielded sample, extraction is done first, then training consumes it.
    - Any malformed sample / token extraction error is skipped and replaced by the next sample.
    """

    def __init__(
        self,
        dataset_config,
        model_paths,
        extract_config,
        rank=0,
        world_size=1,
        tp_size=1,
        dp_rank=0,
        tp_rank=0,
        debug=False,
        shared_train_pipe=None,
        shared_vae_pipe=None,
    ):
        super().__init__()
        self.dataset_config = dataset_config
        self.model_paths = model_paths
        self.extract_config = extract_config
        self.rank = int(rank)
        self.world_size = int(max(world_size, 1))
        self.tp_size = int(max(tp_size, 1))
        self.dp_size = max(1, self.world_size // self.tp_size)
        self.dp_rank = int(dp_rank)
        self.tp_rank = int(tp_rank)
        self.debug = bool(debug)
        self._shared_train_pipe = shared_train_pipe
        self._shared_vae_pipe = shared_vae_pipe
        self._local_vae_pipe = None
        self._local_extract_pipe = None
        self._tp_group = None
        self._tp_group_ranks = None

        self.pipe = None
        self.dataset = None
        self.device = None
        self._sample_counter = 0
        self._initialized = False
        self.use_train_weights_for_extract_and_probe = bool(
            self.extract_config.get('use_train_weights_for_extract_and_probe', False)
        )
        self.train_noise_domain = str(self.extract_config.get('train_noise_domain', 'low_noise')).strip().lower()
        self.noise_domain_boundary = float(self.extract_config.get('noise_domain_boundary_ratio', 0.9))
        self.jigsaw_memory_extract_timestep_mode = 'aligned'
        self.enable_sparse_role_memory_attn = bool(self.extract_config.get('enable_sparse_role_memory_attn', True))
        writer_effective_mode = self.extract_config.get(
            'memory_writer_effective_mode',
            'off',
        )
        self.memory_writer_enabled = bool(
            str(self.extract_config.get('train_stage', 'stage1')).strip().lower() == 'stage2'
            and str(writer_effective_mode).strip().lower() == 'residual'
        )
        self.probe_bbox_loss_weight_x2 = bool(self.extract_config.get('probe_bbox_loss_weight_x2', False))
        self.disable_lora_during_extraction = bool(self.extract_config.get('disable_lora_during_extraction', True))
        if self.use_train_weights_for_extract_and_probe:
            # Keep LoRA/adapters enabled so extraction sees current training weights.
            self.disable_lora_during_extraction = False
        self.offload_image_encoder_after_extraction = bool(self.extract_config.get('offload_image_encoder_after_extraction', True))
        self.aggressive_vram_optimization = bool(self.extract_config.get('aggressive_vram_optimization', False))
        self.offload_detached_extractor_after_extraction = bool(
            self.extract_config.get('offload_detached_extractor_after_extraction', self.aggressive_vram_optimization)
        )
        self._local_extract_pipe_on_gpu = False
        self.inline_progress_print_every = max(1, int(self.extract_config.get('inline_progress_print_every', 1) or 1))
        self.inline_progress_print_max_initial = max(0, int(self.extract_config.get('inline_progress_print_max_initial', 20) or 20))
        self.sync_before_yield = bool(self.extract_config.get('sync_before_yield', True))
        self.yield_sync_timeout_minutes = max(1, int(self.extract_config.get('yield_sync_timeout_minutes', 20) or 20))
        self.enable_rank_heartbeat = bool(self.extract_config.get('enable_rank_heartbeat', True))
        self.rank_heartbeat_interval_sec = max(5, int(self.extract_config.get('rank_heartbeat_interval_sec', 30) or 30))
        self._heartbeat_stage = 'init'
        self._heartbeat_sample_idx = None
        self._heartbeat_extra = None
        self._heartbeat_thread = None
        self._heartbeat_stop_event = threading.Event()

        self.skip_log = self.dataset_config.get('dataloader_skip_log')
        self._inline_traceback_printed = False

    def _tp_leader_global_rank(self):
        return self.dp_rank * self.tp_size

    def _ensure_tp_group(self):
        if self.tp_size <= 1:
            return None
        if (not dist.is_available()) or (not dist.is_initialized()):
            return None
        if self._tp_group is not None:
            return self._tp_group

        world = int(dist.get_world_size())
        if world != int(self.world_size):
            raise RuntimeError(
                f"Distributed world_size mismatch: dist={world}, dataset={self.world_size}"
            )
        if world % self.tp_size != 0:
            raise RuntimeError(
                f"Invalid TP topology for dataset group build: world_size={world}, tp_size={self.tp_size}"
            )

        num_dp_groups = world // self.tp_size
        cur_rank = int(dist.get_rank())
        my_group = None
        my_group_ranks = None

        # All ranks must create groups in the same order.
        for dp_idx in range(num_dp_groups):
            ranks = list(range(dp_idx * self.tp_size, (dp_idx + 1) * self.tp_size))
            group = dist.new_group(ranks=ranks)
            if cur_rank in ranks:
                my_group = group
                my_group_ranks = tuple(ranks)

        if my_group is None:
            raise RuntimeError(f"Failed to build TP group for rank={cur_rank}")

        self._tp_group = my_group
        self._tp_group_ranks = my_group_ranks
        return self._tp_group

    def _tp_broadcast_object(self, obj):
        if self.tp_size <= 1 or (not dist.is_available()) or (not dist.is_initialized()):
            return obj
        tp_group = self._ensure_tp_group()
        if tp_group is None:
            return obj
        obj_list = [obj]
        # src here is global rank even when group is provided.
        dist.broadcast_object_list(obj_list, src=self._tp_leader_global_rank(), group=tp_group)
        return obj_list[0]

    def _barrier_with_timeout(self, timeout_minutes, sync_tag, sample_idx=None, group=None):
        if not (dist.is_available() and dist.is_initialized()):
            return

        timeout_seconds = max(1.0, float(timeout_minutes) * 60.0)
        work = dist.barrier(async_op=True, group=group)
        start_t = time.time()
        while not work.is_completed():
            if (time.time() - start_t) > timeout_seconds:
                raise RuntimeError(
                    f"{sync_tag}_timeout: rank={self.rank}, sample={sample_idx}, "
                    f"timeout_min={timeout_minutes}, group={'tp' if group is not None else 'world'}"
                )
            time.sleep(0.2)
        work.wait()

    def _all_ranks_ready_with_timeout(self, local_ready: bool, timeout_minutes, sync_tag, sample_idx=None, group=None):
        if not (dist.is_available() and dist.is_initialized()):
            return bool(local_ready), int(1 if local_ready else 0), 1

        timeout_seconds = max(1.0, float(timeout_minutes) * 60.0)
        ready_flag = torch.tensor([1 if local_ready else 0], device=self.device, dtype=torch.int32)
        work = dist.all_reduce(ready_flag, op=dist.ReduceOp.SUM, group=group, async_op=True)
        start_t = time.time()
        while not work.is_completed():
            if (time.time() - start_t) > timeout_seconds:
                raise RuntimeError(
                    f"{sync_tag}_timeout: rank={self.rank}, sample={sample_idx}, "
                    f"timeout_min={timeout_minutes}, group={'tp' if group is not None else 'world'}"
                )
            time.sleep(0.2)
        work.wait()

        world_size = dist.get_world_size(group=group) if group is not None else dist.get_world_size()
        ready_count = int(ready_flag.item())
        return bool(ready_count >= int(world_size)), ready_count, int(world_size)

    @contextmanager
    def _sample_seed_scope(self, sample_seed: int):
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = None
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
        random.seed(int(sample_seed))
        np.random.seed(int(sample_seed) % (2**32 - 1))
        torch.manual_seed(int(sample_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(sample_seed))
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _log_skip(self, reason, sample_id=None, extra=None):
        if not self.skip_log:
            return
        try:
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sid = str(sample_id) if sample_id is not None else "unknown"
            ext = f" | {extra}" if extra else ""
            with open(self.skip_log, 'a') as f:
                f.write(f"[{ts}] reason={reason} sample={sid}{ext}\n")
        except Exception:
            pass

    def _should_progress_print(self, sample_index):
        if sample_index <= self.inline_progress_print_max_initial:
            return True
        return (sample_index % self.inline_progress_print_every) == 0

    def _progress_print(self, stage, sample_idx=None, extra=None, force=False):
        self._heartbeat_stage = str(stage)
        self._heartbeat_sample_idx = sample_idx
        if extra is None:
            self._heartbeat_extra = None
        else:
            extra_str = str(extra)
            self._heartbeat_extra = extra_str if len(extra_str) <= 400 else (extra_str[:397] + '...')
        should_print = force or self._should_progress_print(self._sample_counter)
        if not should_print:
            return
        prefix = f"[InlineProgress Rank {self.rank}][DP {self.dp_rank}][TP {self.tp_rank}/{self.tp_size}]"
        sid = f" sample={sample_idx}" if sample_idx is not None else ""
        msg = f"{prefix} step={self._sample_counter}{sid} stage={stage}"
        if extra:
            msg = f"{msg} | {extra}"
        print(msg, flush=True)

    def _start_rank_heartbeat(self):
        if (not self.enable_rank_heartbeat) or (self.rank_heartbeat_interval_sec <= 0):
            return
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return

        def _heartbeat_loop():
            while not self._heartbeat_stop_event.is_set():
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                stage = self._heartbeat_stage
                sample_idx = self._heartbeat_sample_idx
                extra = self._heartbeat_extra
                extra_msg = f" extra={extra}" if extra else ""
                print(
                    f"[InlineHeartbeat Rank {self.rank}][DP {self.dp_rank}][TP {self.tp_rank}/{self.tp_size}] "
                    f"ts={ts} step={self._sample_counter} sample={sample_idx} stage={stage}{extra_msg}",
                    flush=True,
                )
                self._heartbeat_stop_event.wait(timeout=float(self.rank_heartbeat_interval_sec))

        self._heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            name=f"inline-heartbeat-r{self.rank}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _aggressive_release_cuda_cache(self, sample_idx=None, stage='runtime'):
        if not self.aggressive_vram_optimization:
            return
        if not torch.cuda.is_available():
            return
        gc.collect()
        torch.cuda.empty_cache()
        self._progress_print("aggressive_cuda_release", sample_idx, extra=f"stage={stage}")

    def _set_local_extract_pipe_gpu_state(self, to_gpu: bool, sample_idx=None, reason='runtime'):
        if self._local_extract_pipe is None:
            return
        if self._local_extract_pipe_on_gpu == bool(to_gpu):
            return

        target_device = torch.device(self.device) if to_gpu else torch.device('cpu')
        if hasattr(self._local_extract_pipe, 'dit') and self._local_extract_pipe.dit is not None:
            self._local_extract_pipe.dit.to(device=target_device, dtype=torch.bfloat16)
        if hasattr(self._local_extract_pipe, 'prompter') and getattr(self._local_extract_pipe.prompter, 'text_encoder', None) is not None:
            self._local_extract_pipe.prompter.text_encoder.to(device=target_device, dtype=torch.bfloat16)
        self._local_extract_pipe.device = str(target_device)
        self._local_extract_pipe_on_gpu = bool(to_gpu)
        self._progress_print(
            "local_extract_pipe_device",
            sample_idx,
            extra=f"to={'gpu' if to_gpu else 'cpu'}, reason={reason}",
        )

    def _ensure_local_extract_pipe_on_gpu(self, sample_idx=None, reason='pre_extract'):
        if self._local_extract_pipe is None:
            return
        self._set_local_extract_pipe_gpu_state(True, sample_idx=sample_idx, reason=reason)

    def _offload_local_extract_pipe_if_needed(self, sample_idx=None, reason='post_extract'):
        if self._local_extract_pipe is None:
            return
        if not self.offload_detached_extractor_after_extraction:
            return
        self._set_local_extract_pipe_gpu_state(False, sample_idx=sample_idx, reason=reason)

    def _to_u8_hwc(self, tensor):
        if tensor is None:
            return None
        t = tensor.squeeze(0) if tensor.dim() == 4 and tensor.shape[0] == 1 else tensor
        if t.dtype.is_floating_point:
            t = ((t + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        else:
            t = t.to(torch.uint8)
        if t.dim() == 3 and t.shape[0] == 3:
            t = t.permute(1, 2, 0)
        return t.cpu().contiguous()

    def _expand_model_path_list(self, path_or_paths):
        if path_or_paths is None:
            return []
        if isinstance(path_or_paths, str):
            return [p.strip() for p in path_or_paths.split(',') if p and p.strip()]
        if isinstance(path_or_paths, (list, tuple)):
            out = []
            for item in path_or_paths:
                out.extend(self._expand_model_path_list(item))
            return out
        return [str(path_or_paths)]

    def _build_extract_pipe_from_shared(self):
        from types import SimpleNamespace

        if self._local_extract_pipe is not None:
            preprocess_image_fn = getattr(self._local_extract_pipe, 'preprocess_image', None)
            if preprocess_image_fn is None:
                raise RuntimeError(
                    "detached local extractor pipe is missing preprocess_image, which is required for extraction."
                )

            return SimpleNamespace(
                dit=self._local_extract_pipe.dit,
                scheduler=self._local_extract_pipe.scheduler,
                torch_dtype=getattr(self._local_extract_pipe, 'torch_dtype', torch.bfloat16),
                prompter=self._local_extract_pipe.prompter,
                vae=self._local_extract_pipe.vae,
                image_encoder=getattr(self._local_extract_pipe, 'image_encoder', None),
                preprocess_image=preprocess_image_fn,
                encode_images_adaptive=self._local_extract_pipe.encode_images_adaptive,
                device=self.device,
            )

        if self._shared_train_pipe is None or self._shared_vae_pipe is None:
            raise RuntimeError(
                "Inline extraction requires shared training/vae pipes, but shared pipes are missing. "
                "Please fix shared pipe construction instead of using fallback behavior."
            )

        vae_source_pipe = self._local_vae_pipe if self._local_vae_pipe is not None else self._shared_vae_pipe

        preprocess_image_fn = getattr(vae_source_pipe, 'preprocess_image', None)
        if preprocess_image_fn is None:
            raise RuntimeError(
                "shared_vae_pipe.preprocess_image is required for extraction, but it is missing. "
                "Please fix shared pipe construction instead of using fallback behavior."
            )

        return SimpleNamespace(
            dit=self._shared_train_pipe.dit,
            scheduler=self._shared_train_pipe.scheduler,
            torch_dtype=getattr(self._shared_train_pipe, 'torch_dtype', torch.bfloat16),
            prompter=vae_source_pipe.prompter,
            vae=vae_source_pipe.vae,
            image_encoder=getattr(vae_source_pipe, 'image_encoder', None),
            preprocess_image=preprocess_image_fn,
            encode_images_adaptive=vae_source_pipe.encode_images_adaptive,
            device=self.device,
        )

    def _toggle_lora_for_model(self, model, enable=True):
        """Toggle LoRA adapters in-place and return per-module previous states.

        This supports PEFT injected LoRA layers that expose `disable_adapters`.
        """
        if model is None:
            return []

        prev_states = []
        for module in model.modules():
            if hasattr(module, 'disable_adapters'):
                with suppress(Exception):
                    prev = bool(getattr(module, 'disable_adapters'))
                    setattr(module, 'disable_adapters', not bool(enable))
                    prev_states.append((module, prev))
        return prev_states

    def _restore_lora_states(self, prev_states):
        if not prev_states:
            return
        for module, prev in prev_states:
            with suppress(Exception):
                setattr(module, 'disable_adapters', prev)

    def _lazy_init(self):
        if self._initialized:
            return

        if not torch.cuda.is_available():
            raise RuntimeError("InlineExtractThenTrainDataset requires CUDA")

        local_rank = int(os.environ.get('LOCAL_RANK', str(self.rank)))
        if local_rank < torch.cuda.device_count():
            torch.cuda.set_device(local_rank)
        self.device = f"cuda:{torch.cuda.current_device()}"

        strategy_name = str(self.extract_config.get('training_strategy', '') or '').strip().lower()
        slice_mode = str(self.extract_config.get('model_slice_mode', '') or '').strip().lower()
        requested_detached_local_extractor = bool(self.extract_config.get('use_detached_local_extractor_for_zero3', False))
        allow_detached_local_extractor = (
            requested_detached_local_extractor
            and (not self.use_train_weights_for_extract_and_probe)
        )
        if (
            requested_detached_local_extractor
            and self.use_train_weights_for_extract_and_probe
            and self.debug
        ):
            print(
                f"[InlineDataset Rank {self.rank}] use_detached_local_extractor_for_zero3 requested but disabled because use_train_weights_for_extract_and_probe is enabled.",
                flush=True,
            )
        use_detached_local_extractor = (
            allow_detached_local_extractor
            and
            self.world_size > 1
            and strategy_name in ('deepspeed_stage_2', 'deepspeed_stage_3')
            and slice_mode == 'zero3'
        )
        allow_detached_local_vae = bool(self.extract_config.get('use_detached_local_vae_for_zero3', False))
        use_detached_local_vae = (
            allow_detached_local_vae
            and
            self.world_size > 1
            and strategy_name in ('deepspeed_stage_2', 'deepspeed_stage_3')
            and slice_mode == 'zero3'
        )

        if use_detached_local_extractor and self._local_extract_pipe is None:
            ckpt_dir = self.model_paths.get('ckpt_dir')
            if not ckpt_dir:
                raise RuntimeError(
                    "use_detached_local_extractor_for_zero3 is enabled but ckpt_dir is missing."
                )
            if not os.path.isdir(ckpt_dir):
                raise FileNotFoundError(f"detached local extractor ckpt_dir not found: {ckpt_dir}")
            self._local_extract_pipe = build_wan22_training_pipe(
                ckpt_dir=ckpt_dir,
                device='cpu',
                torch_dtype=torch.bfloat16,
                task="i2v-A14B",
                train_noise_domain=self.train_noise_domain,
            )
            _install_lightweight_pipeline_lifecycle(self._local_extract_pipe)
            _install_tp2dp2_pipeline_only_patch(self._local_extract_pipe)
            print(f"[InlineDataset Rank {self.rank}] Using detached local full extractor pipe (DiT+VAE+encoders) for ZeRO-3 extraction.", flush=True)

        if use_detached_local_vae and (not use_detached_local_extractor) and self._local_vae_pipe is None:
            ckpt_dir = self.model_paths.get('ckpt_dir')
            if not ckpt_dir:
                raise RuntimeError(
                    "use_detached_local_vae_for_zero3 is enabled but ckpt_dir is missing."
                )
            self._local_vae_pipe = build_wan22_training_pipe(
                ckpt_dir=ckpt_dir,
                device='cpu',
                torch_dtype=torch.bfloat16,
                task="i2v-A14B",
                train_noise_domain=self.train_noise_domain,
            )
            _install_lightweight_pipeline_lifecycle(self._local_vae_pipe)
            _install_tp2dp2_pipeline_only_patch(self._local_vae_pipe)
            print(f"[InlineDataset Rank {self.rank}] Using detached local VAE pipe for ZeRO-3 extraction.", flush=True)

        # Ensure shared pipes use the current rank device for extraction-time helper paths
        # (e.g., encode_images_adaptive internally uses self.device for input tensors).
        if self._shared_vae_pipe is not None:
            self._shared_vae_pipe.device = self.device
            if hasattr(self._shared_vae_pipe, 'image_encoder') and self._shared_vae_pipe.image_encoder is not None:
                self._shared_vae_pipe.image_encoder.to(device=self.device, dtype=torch.float32)
        if self._shared_train_pipe is not None:
            self._shared_train_pipe.device = self.device
        if self._local_vae_pipe is not None:
            self._local_vae_pipe.device = self.device
            if hasattr(self._local_vae_pipe, 'image_encoder') and self._local_vae_pipe.image_encoder is not None:
                self._local_vae_pipe.image_encoder.to(device=self.device, dtype=torch.float32)
            if hasattr(self._local_vae_pipe, 'prompter') and getattr(self._local_vae_pipe.prompter, 'text_encoder', None) is not None:
                self._local_vae_pipe.prompter.text_encoder.to(device=self.device, dtype=torch.bfloat16)
        if self._local_extract_pipe is not None:
            self._local_extract_pipe.device = self.device
            if hasattr(self._local_extract_pipe, 'dit') and self._local_extract_pipe.dit is not None:
                self._local_extract_pipe.dit.to(device=self.device, dtype=torch.bfloat16)
            if hasattr(self._local_extract_pipe, 'image_encoder') and self._local_extract_pipe.image_encoder is not None:
                self._local_extract_pipe.image_encoder.to(device=self.device, dtype=torch.float32)
            if hasattr(self._local_extract_pipe, 'prompter') and getattr(self._local_extract_pipe.prompter, 'text_encoder', None) is not None:
                self._local_extract_pipe.prompter.text_encoder.to(device=self.device, dtype=torch.bfloat16)
            self._local_extract_pipe_on_gpu = True

        base_seed = int(self.dataset_config.get('seed', -1))
        if base_seed < 0:
            base_seed = int(secrets.randbits(32))
        local_seed = int((base_seed + self.dp_rank * 100003) % (2**32 - 1))
        random.seed(local_seed)
        np.random.seed(local_seed)
        torch.manual_seed(local_seed)
        torch.cuda.manual_seed_all(local_seed)

        # Route A path: use detached local full extractor pipe when configured.
        self.pipe = self._build_extract_pipe_from_shared()
        _install_tp2dp2_pipeline_only_patch(self.pipe)
        self._start_rank_heartbeat()
        self._offload_local_extract_pipe_if_needed(reason='post_lazy_init')

        # One-time check for extractor purity.
        # In shared-model mode, LoRA/adapter params may legitimately exist on the training model.
        shared_mode = (
            self._local_extract_pipe is None
            and self._shared_train_pipe is not None
            and self._shared_vae_pipe is not None
        )
        unexpected_memory_attrs = [
            n for n in ('memory_fusion', 'memory_embeddings', 'memory_pos_embed', 'memory_segment_embed')
            if hasattr(self.pipe, n)
        ]
        if len(unexpected_memory_attrs) > 0:
            raise RuntimeError(
                f"Extraction pipe unexpectedly carries training memory modules: {unexpected_memory_attrs}. "
                "Extraction must run on native path only."
            )
        if (not shared_mode) and hasattr(self.pipe, 'dit') and self.pipe.dit is not None:
            lora_like = [
                n for n, _ in self.pipe.dit.named_parameters()
                if ('lora_' in n.lower()) or ('adapter' in n.lower())
            ]
            if len(lora_like) > 0:
                raise RuntimeError(
                    f"Extraction pipeline unexpectedly contains LoRA/adapter params (count={len(lora_like)}), "
                    "which may contaminate extraction behavior."
                )

        from types import SimpleNamespace

        skip_log = self.dataset_config.get('dataloader_skip_log')
        if self.rank == 0 and skip_log:
            try:
                open(skip_log, 'w').close()
            except Exception:
                pass

        story_root = str(self.dataset_config.get('story_root') or "").strip()
        csv_path = self.dataset_config.get('candidate_groups_csv')
        char_dir = self.dataset_config.get('character_lists_dir')
        vid_root = self.dataset_config.get('video_root')
        if story_root:
            class SlotMemDirectoryDataset:
                def __init__(self, root, num_frames, height, width):
                    self.root = root
                    self.num_frames = int(num_frames)
                    self.height = int(height)
                    self.width = int(width)
                    self.sample_dirs = sorted(
                        os.path.join(root, name)
                        for name in os.listdir(root)
                        if os.path.isdir(os.path.join(root, name)) and os.path.exists(os.path.join(root, name, "rewrite_caption.json"))
                    )

                def __len__(self):
                    return len(self.sample_dirs)

                def _load_video_tensor(self, sample_dir):
                    dir_path = os.path.join(sample_dir, "preprocessed_video")
                    mp4_path = os.path.join(sample_dir, "preprocessed_video.mp4")
                    frames = []
                    if os.path.isdir(dir_path):
                        frame_files = sorted(
                            os.path.join(dir_path, x)
                            for x in os.listdir(dir_path)
                            if x.lower().endswith((".jpg", ".jpeg", ".png"))
                        )
                        for path in frame_files[:self.num_frames]:
                            frames.append(np.array(Image.open(path).convert("RGB").resize((self.width, self.height))))
                    elif os.path.isfile(mp4_path):
                        reader = imageio.get_reader(mp4_path)
                        for frame in reader:
                            frames.append(np.array(Image.fromarray(frame).convert("RGB").resize((self.width, self.height))))
                            if len(frames) >= self.num_frames:
                                break
                    if len(frames) == 0:
                        raise FileNotFoundError(f"No preprocessed video frames found for {sample_dir}")
                    while len(frames) < self.num_frames:
                        frames.append(frames[-1].copy())
                    arr = np.stack(frames[:self.num_frames], axis=0)
                    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 127.5 - 1.0
                    return tensor

                def _load_prompt_text(self, sample_dir):
                    path = os.path.join(sample_dir, "rewrite_caption.json")
                    with open(path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    prompts = []
                    if isinstance(payload, dict):
                        for chunk in payload.get("chunks", []):
                            content = str(chunk.get("content", "")).strip()
                            if content:
                                prompts.append(content)
                    if not prompts:
                        prompts = [os.path.basename(sample_dir)]
                    return ", ".join(prompts)

                def __getitem__(self, idx):
                    sample_dir = self.sample_dirs[int(idx)]
                    frame_path = os.path.join(sample_dir, "frame.jpg")
                    if not os.path.exists(frame_path):
                        frame_path = os.path.join(sample_dir, "frame.png")
                    if not os.path.exists(frame_path):
                        raise FileNotFoundError(f"Missing frame.jpg/frame.png in {sample_dir}")
                    frame = Image.open(frame_path).convert("RGB").resize((self.width, self.height))
                    frame_tensor = torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 127.5 - 1.0
                    return {
                        "folder_name": os.path.basename(sample_dir),
                        "text": self._load_prompt_text(sample_dir),
                        "use_memory": True,
                        "video": self._load_video_tensor(sample_dir),
                        "first_ref_frames": [frame_tensor],
                        "random_ref_frame": frame_tensor,
                        "video_id": os.path.basename(sample_dir),
                        "group_index": 0,
                        "core_clip_index": 0,
                        "memory_clip_index": 0,
                    }

            ds_args = self.dataset_config.get('dataset_args', {})
            self.dataset = SlotMemDirectoryDataset(
                story_root,
                num_frames=ds_args.get('num_frames', 81),
                height=ds_args.get('height', 480),
                width=ds_args.get('width', 832),
            )
        else:
            from test_dataloading import CandidateGroupsDataset

            args_ns = SimpleNamespace(**self.dataset_config.get('dataset_args', {}))
            self.dataset = CandidateGroupsDataset(csv_path, char_dir, vid_root, args_ns, skip_log_path=skip_log)

        self._initialized = True
        print(f"[InlineDataset Rank {self.rank}] Initialized on {self.device}, dataset_size={len(self.dataset)}, dp_rank={self.dp_rank}/{self.dp_size}, tp_rank={self.tp_rank}/{self.tp_size}", flush=True)

    def _sample_extract_timestep_and_bank(self):
        train_stage = str(self.extract_config.get('train_stage', 'stage1')).strip().lower()
        memory_bank_mode = str(self.extract_config.get('jigsaw_memory_bank_mode', 'single')).strip().lower()
        sched_timesteps = getattr(self.pipe.scheduler, 'timesteps', None)

        sample_extract_timestep = None
        sample_train_timestep = None
        selected_bank_idx_for_sample = None
        selected_bank_percent_for_sample = None
        selected_bank_percents_for_sample = []

        if sched_timesteps is None or len(sched_timesteps) == 0:
            return (
                sample_extract_timestep,
                sample_train_timestep,
                selected_bank_idx_for_sample,
                selected_bank_percent_for_sample,
                selected_bank_percents_for_sample,
            )

        num_train_timesteps = float(max(int(getattr(self.pipe.scheduler, 'num_train_timesteps', 1000)), 1))
        sampled_t = _sample_timestep_for_domain(
            sched_timesteps=sched_timesteps,
            train_noise_domain=self.train_noise_domain,
            num_train_timesteps=num_train_timesteps,
            boundary_ratio=self.noise_domain_boundary,
        )
        if sampled_t is None:
            return (
                sample_extract_timestep,
                sample_train_timestep,
                selected_bank_idx_for_sample,
                selected_bank_percent_for_sample,
                selected_bank_percents_for_sample,
            )

        p_extract = float(np.clip(sampled_t / num_train_timesteps, 0.0, 1.0))
        if memory_bank_mode != 'legacy_multi':
            sample_extract_timestep = sampled_t
            sample_train_timestep = sampled_t
            selected_bank_idx_for_sample = 0
            selected_bank_percent_for_sample = p_extract
            selected_bank_percents_for_sample = [p_extract]
            return (
                sample_extract_timestep,
                sample_train_timestep,
                selected_bank_idx_for_sample,
                selected_bank_percent_for_sample,
                selected_bank_percents_for_sample,
            )

        bank_percents = parse_float_csv(
            self.extract_config.get('memory_bank_percents', '0.85,0.60,0.35,0.12'),
            default_list=[0.85, 0.60, 0.35, 0.12]
        )

        if train_stage == 'stage2' and len(bank_percents) > 0:
            p_cur = p_extract
            selected_bank_idx_for_sample = int(pick_nearest_bank_by_percent(p_cur, bank_percents))
            selected_bank_percent_for_sample = float(bank_percents[selected_bank_idx_for_sample])
            nearest_idx = min(
                range(len(sched_timesteps)),
                key=lambda i: abs(float(sched_timesteps[i].item()) / num_train_timesteps - selected_bank_percent_for_sample)
            )
            sample_extract_timestep = float(sched_timesteps[nearest_idx].item())
            selected_bank_percents_for_sample = [float(x) for x in bank_percents]
        else:
            sample_extract_timestep = sampled_t
            if len(bank_percents) > 0:
                p_cur = float(np.clip(sample_extract_timestep / num_train_timesteps, 0.0, 1.0))
                selected_bank_idx_for_sample = int(pick_nearest_bank_by_percent(p_cur, bank_percents))
                selected_bank_percent_for_sample = float(bank_percents[selected_bank_idx_for_sample])
                selected_bank_percents_for_sample = [float(x) for x in bank_percents]

        # Keep train/extract strictly aligned to the same timestep.
        sample_train_timestep = sample_extract_timestep

        return (
            sample_extract_timestep,
            sample_train_timestep,
            selected_bank_idx_for_sample,
            selected_bank_percent_for_sample,
            selected_bank_percents_for_sample,
        )

    def _build_batch_from_raw_sample(self, sample, sample_idx):
        self._ensure_local_extract_pipe_on_gpu(sample_idx=sample_idx, reason='pre_extract')
        folder_name = sample.get('folder_name')
        text = sample.get('text')
        use_memory = bool(sample.get('use_memory', False))
        need_memory_extract = bool(use_memory and (self.enable_sparse_role_memory_attn or self.probe_bbox_loss_weight_x2))
        require_memory_sample = bool(self.extract_config.get('require_memory_sample', False))
        if require_memory_sample and (not use_memory):
            self._log_skip("non_memory_sample_filtered", sample_id=folder_name, extra=f"rank={self.rank}")
            self._progress_print("skip_non_memory_sample_filtered", sample_idx, extra=f"folder={folder_name}")
            return None
        self._progress_print("build_batch_start", sample_idx, extra=f"folder={folder_name}, use_memory={use_memory}")

        video_main_tensor = sample.get('video')
        if video_main_tensor is None:
            self._log_skip("missing_video", sample_id=folder_name)
            self._progress_print("skip_missing_video", sample_idx, extra=f"folder={folder_name}")
            return None

        memory_feature_tokens_selected = torch.zeros(0, 5120)
        memory_feature_positions_selected = torch.zeros(0, 3, dtype=torch.long)
        memory_feature_token_meta_selected = []
        memory_token_lengths_per_character = []
        memory_bank_embeddings = {}
        memory_bank_token_meta = {}
        memory_bank_percents = []
        stage2_update_memory_bank_embeddings = {}
        stage2_update_memory_bank_token_meta = {}

        tiler_kwargs = self.extract_config.get('tiler_kwargs', {'tiled': False})
        trace_vae_input_shape = bool(self.extract_config.get('trace_vae_input_shape', True))

        def _normalize_video_layout_for_vae(video_tensor, encode_tag):
            v = video_tensor
            raw_shape = tuple(v.shape) if isinstance(v, torch.Tensor) else None
            if isinstance(v, torch.Tensor) and v.dim() == 5 and v.shape[0] == 1:
                v = v.squeeze(0)
            if not isinstance(v, torch.Tensor) or v.dim() != 4:
                raise RuntimeError(
                    f"Unexpected video tensor shape for VAE encode: raw={raw_shape}, after_squeeze={getattr(v, 'shape', None)}, "
                    f"sample={folder_name}, tag={encode_tag}, rank={self.rank}"
                )
            # Strictly enforce expected layout C,T,H,W. If input is T,C,H,W, transpose once.
            if v.shape[0] in (1, 3):
                v_out = v.contiguous()
            elif v.shape[1] in (1, 3):
                v_out = v.permute(1, 0, 2, 3).contiguous()
            else:
                raise RuntimeError(
                    f"Illegal VAE video layout (expect C,T,H,W or T,C,H,W): raw={raw_shape}, "
                    f"normalized_candidate={tuple(v.shape)}, sample={folder_name}, tag={encode_tag}, rank={self.rank}"
                )
            c, t, h, w = v_out.shape
            if c not in (1, 3) or min(t, h, w) <= 0:
                raise RuntimeError(
                    f"Invalid normalized VAE input shape {tuple(v_out.shape)} from raw={raw_shape}, "
                    f"sample={folder_name}, tag={encode_tag}, rank={self.rank}"
                )
            if trace_vae_input_shape:
                self._progress_print(
                    "vae_input_layout",
                    sample_idx,
                    extra=(
                        f"sample={folder_name}, tag={encode_tag}, raw_shape={raw_shape}, "
                        f"norm_shape={tuple(v_out.shape)}, dtype={v_out.dtype}, rank={self.rank}"
                    )
                )
            return v_out

        def _encode_with_tiled_fallback(video_tensor, encode_tag):
            vae_dtype = _get_vae_runtime_dtype(self.pipe.vae, default_dtype=torch.bfloat16)
            v_vae = _normalize_video_layout_for_vae(video_tensor, encode_tag).to(dtype=vae_dtype, device=self.device)
            try:
                self.pipe.vae = _move_vae_runtime(
                    self.pipe.vae,
                    device=self.device,
                    dtype=vae_dtype,
                )
                return _safe_vae_encode_isolated(self.pipe.vae, [v_vae], device=self.device, op_name='_encode_with_tiled_fallback.vae.encode', **tiler_kwargs)[0]
            finally:
                del v_vae

        shared_mode = (
            self._local_extract_pipe is None
            and self._shared_train_pipe is not None
            and self._shared_vae_pipe is not None
        )
        prev_train_pipe = None
        prev_dit_train = None
        prev_vae_pipe = None
        prev_lora_states = None
        if shared_mode:
            prev_train_pipe = bool(self._shared_train_pipe.training)
            prev_dit_train = bool(self._shared_train_pipe.denoising_model().training)
            prev_vae_pipe = bool(self._shared_vae_pipe.training)
            self._shared_train_pipe.eval()
            self._shared_train_pipe.denoising_model().eval()
            self._shared_vae_pipe.eval()
        if self.disable_lora_during_extraction:
            prev_lora_states = self._toggle_lora_for_model(getattr(self.pipe, 'dit', None), enable=False)
            if self.debug and len(prev_lora_states) > 0:
                print(f"[InlineDataset Rank {self.rank}] Temporarily disabled LoRA layers for extraction: {len(prev_lora_states)}", flush=True)

        with torch.no_grad():
            try:
                latents_main = _encode_with_tiled_fallback(video_main_tensor, encode_tag="main").cpu()
                self._progress_print("main_vae_done", sample_idx, extra=f"latents_shape={tuple(latents_main.shape)}")
                self._aggressive_release_cuda_cache(sample_idx, stage='between_main_and_memory_extract')

                sample_extract_timestep = None
                selected_bank_idx_for_sample = None
                selected_bank_percent_for_sample = None
                selected_bank_percents_for_sample = []

                if need_memory_extract:
                    extra_v_tensor = sample.get('extra_video')
                    if extra_v_tensor is None:
                        self._log_skip("missing_extra_video", sample_id=folder_name)
                        self._progress_print("skip_missing_extra_video", sample_idx, extra=f"folder={folder_name}")
                        return None

                    memory_text = sample.get('memory_text', {}) or {}
                    core_chars = memory_text.get('core_characters', [])
                    mem_chars = memory_text.get('memory_characters', [])
                    if isinstance(core_chars, str):
                        core_chars = [core_chars]
                    if isinstance(mem_chars, str):
                        mem_chars = [mem_chars]
                    stage2_update_v_tensor = None
                    stage2_update_memory_text = {}
                    update_mem_chars = []
                    if bool(getattr(self, 'memory_writer_enabled', False)):
                        stage2_update_v_tensor = sample.get('stage2_update_extra_video')
                        if stage2_update_v_tensor is None:
                            self._log_skip("missing_stage2_update_extra_video", sample_id=folder_name)
                            self._progress_print("skip_missing_stage2_update_extra_video", sample_idx, extra=f"folder={folder_name}")
                            return None
                        stage2_update_memory_text = sample.get('stage2_update_memory_text', {}) or {}
                        update_mem_chars = stage2_update_memory_text.get('memory_characters', [])
                        if isinstance(update_mem_chars, str):
                            update_mem_chars = [update_mem_chars]
                    target_chars = sorted(set(core_chars) & set(mem_chars))
                    if bool(getattr(self, 'memory_writer_enabled', False)):
                        target_chars = sorted(set(target_chars) & set(update_mem_chars))

                    if len(target_chars) == 0:
                        self._log_skip("empty_target_characters", sample_id=folder_name)
                        self._progress_print("skip_empty_target_characters", sample_idx, extra=f"folder={folder_name}")
                        return None

                    self._progress_print("memory_extract_prepare", sample_idx, extra=f"target_chars={len(target_chars)}")

                    latents_extra = _encode_with_tiled_fallback(extra_v_tensor, encode_tag="memory")
                    self._progress_print("memory_extra_vae_done", sample_idx, extra=f"latents_shape={tuple(latents_extra.shape)}")

                    (
                        sample_extract_timestep,
                        sample_train_timestep,
                        selected_bank_idx_for_sample,
                        selected_bank_percent_for_sample,
                        selected_bank_percents_for_sample,
                    ) = self._sample_extract_timestep_and_bank()
                    num_train_timesteps = float(max(int(getattr(self.pipe.scheduler, 'num_train_timesteps', 1000)), 1))
                    p_extract = float(sample_extract_timestep) / num_train_timesteps if sample_extract_timestep is not None else None
                    p_train = float(sample_train_timestep) / num_train_timesteps if sample_train_timestep is not None else None
                    self._progress_print(
                        "memory_extract_timestep_sampled",
                        sample_idx,
                        extra=(
                            f"extract_t={sample_extract_timestep}, p_extract={p_extract}, "
                            f"bank_idx={selected_bank_idx_for_sample}, bank_percent={selected_bank_percent_for_sample}, "
                            f"train_t={sample_train_timestep}, p_train={p_train}"
                        )
                    )

                    per_sample_extract_config = dict(self.extract_config)
                    per_sample_extract_config['forced_timestep'] = sample_extract_timestep

                    max_chars = int(per_sample_extract_config.get('max_memory_characters', 2))
                    chars_to_run = target_chars[:max_chars]

                    def _extract_role_tokens_for_stage2_update(video_tensor, latents_tensor, prompt_text, chars, tag):
                        update_embs, update_meta = [], []
                        update_debug_events = []
                        if len(chars) > 1:
                            self._progress_print(f"{tag}_batch_extract_start", sample_idx, extra=f"chars={len(chars)}")
                            _batch_explicit = ('extract_layers', 'cfg_scale', 'suffix_attention_scale', 'token_weight', 'extract_config')
                            batch_extra = {k: v for k, v in per_sample_extract_config.items() if k not in _batch_explicit}
                            batch_extra['extract_debug_events'] = update_debug_events
                            batch_results = extract_patch_embeddings_for_characters_batch(
                                pipe=self.pipe,
                                memory_video_tensor=video_tensor,
                                character_ids=chars,
                                original_prompt=prompt_text,
                                latents=latents_tensor,
                                cached_conditions=None,
                                extract_layers=per_sample_extract_config.get('extract_layers'),
                                device=self.device,
                                extract_config=per_sample_extract_config,
                                cfg_scale=per_sample_extract_config.get('cfg_scale', 5.0),
                                suffix_attention_scale=per_sample_extract_config.get('suffix_attention_scale', 1.0),
                                token_weight=per_sample_extract_config.get('token_weight', 0.2),
                                debug=self.debug,
                                **batch_extra,
                            )
                            for _char_id, (pe, _pos, token_meta) in zip(chars, batch_results):
                                if pe is not None and _memory_count_tokens(pe) > 0:
                                    update_embs.append(pe)
                                    update_meta.append(token_meta)
                            self._progress_print(f"{tag}_batch_extract_done", sample_idx, extra=f"chars={len(chars)}, with_tokens={len(update_embs)}")
                        else:
                            _explicit = ('pipe', 'memory_video_tensor', 'character_id', 'original_prompt', 'latents', 'cached_conditions', 'device')
                            extra = {k: v for k, v in per_sample_extract_config.items() if k not in _explicit}
                            extra['extract_debug_events'] = update_debug_events
                            for char_id in chars:
                                self._progress_print(f"{tag}_single_extract_start", sample_idx, extra=f"char={char_id}")
                                pe, _pos, token_meta = extract_patch_embeddings_for_character(
                                    pipe=self.pipe,
                                    memory_video_tensor=video_tensor,
                                    character_id=char_id,
                                    original_prompt=prompt_text,
                                    latents=latents_tensor,
                                    cached_conditions=None,
                                    device=self.device,
                                    debug=self.debug,
                                    **extra,
                                )
                                if pe is not None and _memory_count_tokens(pe) > 0:
                                    update_embs.append(pe)
                                    update_meta.append(token_meta)
                            self._progress_print(f"{tag}_single_extract_done", sample_idx, extra=f"chars={len(chars)}, with_tokens={len(update_embs)}")
                        if not update_embs:
                            return None, []
                        tokens, meta = _concat_token_payloads(update_embs, update_meta)
                        return tokens, meta

                    all_embs, all_pos, all_meta = [], [], []
                    extract_debug_events = []
                    if len(chars_to_run) > 1:
                        self._progress_print("memory_batch_extract_start", sample_idx, extra=f"chars={len(chars_to_run)}")
                        _batch_explicit = ('extract_layers', 'cfg_scale', 'suffix_attention_scale', 'token_weight', 'extract_config')
                        batch_extra = {k: v for k, v in per_sample_extract_config.items() if k not in _batch_explicit}
                        batch_extra['extract_debug_events'] = extract_debug_events
                        batch_results = extract_patch_embeddings_for_characters_batch(
                            pipe=self.pipe,
                            memory_video_tensor=extra_v_tensor,
                            character_ids=chars_to_run,
                            original_prompt=memory_text.get('prompt', text),
                            latents=latents_extra,
                            cached_conditions=None,
                            extract_layers=per_sample_extract_config.get('extract_layers'),
                            device=self.device,
                            extract_config=per_sample_extract_config,
                            cfg_scale=per_sample_extract_config.get('cfg_scale', 5.0),
                            suffix_attention_scale=per_sample_extract_config.get('suffix_attention_scale', 1.0),
                            token_weight=per_sample_extract_config.get('token_weight', 0.2),
                            debug=self.debug,
                            **batch_extra,
                        )
                        for char_id, (pe, pos, token_meta) in zip(chars_to_run, batch_results):
                            if pe is not None and _memory_count_tokens(pe) > 0:
                                all_embs.append(pe)
                                if isinstance(pos, torch.Tensor) and pos.numel() > 0:
                                    all_pos.append(pos)
                                all_meta.append(token_meta)
                        self._progress_print("memory_batch_extract_done", sample_idx, extra=f"chars={len(chars_to_run)}, with_tokens={len(all_embs)}")
                    else:
                        _explicit = ('pipe', 'memory_video_tensor', 'character_id', 'original_prompt', 'latents', 'cached_conditions', 'device')
                        extra = {k: v for k, v in per_sample_extract_config.items() if k not in _explicit}
                        extra['extract_debug_events'] = extract_debug_events
                        for char_id in chars_to_run:
                            try:
                                self._progress_print("memory_single_extract_start", sample_idx, extra=f"char={char_id}")
                                pe, pos, token_meta = extract_patch_embeddings_for_character(
                                    pipe=self.pipe,
                                    memory_video_tensor=extra_v_tensor,
                                    character_id=char_id,
                                    original_prompt=memory_text.get('prompt', text),
                                    latents=latents_extra,
                                    cached_conditions=None,
                                    device=self.device,
                                    debug=self.debug,
                                    **extra,
                                )
                                if pe is not None and _memory_count_tokens(pe) > 0:
                                    all_embs.append(pe)
                                    if isinstance(pos, torch.Tensor) and pos.numel() > 0:
                                        all_pos.append(pos)
                                    all_meta.append(token_meta)
                            except Exception as e:
                                raise RuntimeError(
                                    f"single_character_extract_error: folder={folder_name}, char={char_id}, err={e}"
                                ) from e
                        self._progress_print("memory_single_extract_done", sample_idx, extra=f"chars={len(chars_to_run)}, with_tokens={len(all_embs)}")

                    if all_embs:
                        memory_feature_tokens_selected, memory_feature_token_meta_selected = _concat_token_payloads(all_embs, all_meta)
                        if len(all_pos) > 0:
                            memory_feature_positions_selected = torch.cat(all_pos, dim=0)
                        memory_token_lengths_per_character = []
                        if isinstance(memory_feature_token_meta_selected, list) and len(memory_feature_token_meta_selected) > 0:
                            role_count = defaultdict(int)
                            role_order = []
                            for item in memory_feature_token_meta_selected:
                                if not isinstance(item, dict):
                                    continue
                                rid = str(item.get('char_id', '')).strip()
                                if rid and rid not in role_count:
                                    role_order.append(rid)
                                if rid:
                                    role_count[rid] += 1
                            memory_token_lengths_per_character = [int(role_count[r]) for r in role_order if int(role_count[r]) > 0]
                        elif isinstance(memory_feature_tokens_selected, torch.Tensor) and memory_feature_tokens_selected.ndim >= 2:
                            memory_token_lengths_per_character = [int(memory_feature_tokens_selected.shape[0])]
                    if memory_feature_tokens_selected is not None and _memory_count_tokens(memory_feature_tokens_selected) > 0:
                        memory_bank_mode = str(self.extract_config.get('jigsaw_memory_bank_mode', 'single')).strip().lower()
                        if memory_bank_mode != 'legacy_multi':
                            if _pm_is_layerwise_container(memory_feature_tokens_selected):
                                memory_bank_embeddings = _pm_make_layerwise_container({
                                    layer: {'0': layer_tokens.detach().cpu()}
                                    for layer, layer_tokens in _pm_iter_layerwise_items(memory_feature_tokens_selected)
                                    if isinstance(layer_tokens, torch.Tensor)
                                })
                                memory_bank_token_meta = _pm_make_layerwise_container({
                                    layer: {'0': _pm_select_layerwise_value(memory_feature_token_meta_selected, layer, default=[])}
                                    for layer, _ in _pm_iter_layerwise_items(memory_feature_tokens_selected)
                                })
                            else:
                                memory_bank_embeddings = {'0': memory_feature_tokens_selected.detach().cpu()}
                                memory_bank_token_meta = {'0': list(memory_feature_token_meta_selected)}
                            memory_bank_percents = [float(x) for x in selected_bank_percents_for_sample] if len(selected_bank_percents_for_sample) > 0 else []
                        elif selected_bank_idx_for_sample is not None:
                            if _pm_is_layerwise_container(memory_feature_tokens_selected):
                                memory_bank_embeddings = _pm_make_layerwise_container({
                                    layer: {str(int(selected_bank_idx_for_sample)): layer_tokens.detach().cpu()}
                                    for layer, layer_tokens in _pm_iter_layerwise_items(memory_feature_tokens_selected)
                                    if isinstance(layer_tokens, torch.Tensor)
                                })
                                memory_bank_token_meta = _pm_make_layerwise_container({
                                    layer: {str(int(selected_bank_idx_for_sample)): _pm_select_layerwise_value(memory_feature_token_meta_selected, layer, default=[])}
                                    for layer, _ in _pm_iter_layerwise_items(memory_feature_tokens_selected)
                                })
                            else:
                                memory_bank_embeddings = {str(int(selected_bank_idx_for_sample)): memory_feature_tokens_selected.detach().cpu()}
                                memory_bank_token_meta = {str(int(selected_bank_idx_for_sample)): list(memory_feature_token_meta_selected)}
                            memory_bank_percents = [float(x) for x in selected_bank_percents_for_sample] if len(selected_bank_percents_for_sample) > 0 else []
                        else:
                            if _pm_is_layerwise_container(memory_feature_tokens_selected):
                                memory_bank_embeddings = _pm_make_layerwise_container({
                                    layer: {'0': layer_tokens.detach().cpu()}
                                    for layer, layer_tokens in _pm_iter_layerwise_items(memory_feature_tokens_selected)
                                    if isinstance(layer_tokens, torch.Tensor)
                                })
                                memory_bank_token_meta = _pm_make_layerwise_container({
                                    layer: {'0': _pm_select_layerwise_value(memory_feature_token_meta_selected, layer, default=[])}
                                    for layer, _ in _pm_iter_layerwise_items(memory_feature_tokens_selected)
                                })
                            else:
                                memory_bank_embeddings = {'0': memory_feature_tokens_selected.detach().cpu()}
                                memory_bank_token_meta = {'0': list(memory_feature_token_meta_selected)}
                            memory_bank_percents = []
                        self._progress_print("memory_tokens_ready", sample_idx, extra=f"token_count={_memory_count_tokens(memory_feature_tokens_selected)}")
                    else:
                        if len(extract_debug_events) > 0:
                            reason_chunks = []
                            for item in extract_debug_events[:8]:
                                if not isinstance(item, dict):
                                    continue
                                char_id = item.get('char_id', 'global')
                                stage = item.get('stage', 'unknown')
                                detail = []
                                if 'selected_count' in item:
                                    detail.append(f"selected={item['selected_count']}")
                                if 'final_token_count' in item:
                                    detail.append(f"final={item['final_token_count']}")
                                if 'feature_shape' in item and item.get('feature_shape') is not None:
                                    detail.append(f"feature_shape={item['feature_shape']}")
                                if 'top_visual_tokens' in item:
                                    detail.append(f"thr={item['top_visual_tokens']}")
                                if 'neighbor_filter_kernel' in item:
                                    detail.append(f"kernel={item['neighbor_filter_kernel']}")
                                if 'prompt_preview' in item:
                                    detail.append(f"prompt={str(item['prompt_preview'])[:48]}")
                                if 'extract_layers' in item:
                                    detail.append(f"layers={item['extract_layers']}")
                                if 'per_char_map_summary' in item:
                                    detail.append(f"maps={item['per_char_map_summary']}")
                                if 'raw_hook_summary' in item:
                                    detail.append(f"hooks={item['raw_hook_summary']}")
                                if 'union_token_count' in item:
                                    detail.append(f"union_k={item['union_token_count']}")
                                suffix = f" ({', '.join(detail)})" if len(detail) > 0 else ""
                                reason_chunks.append(f"{char_id}:{stage}{suffix}")
                            self._progress_print(
                                "memory_extract_empty_reasons",
                                sample_idx,
                                extra=" ; ".join(reason_chunks),
                            )
                        self._log_skip(
                            reason="empty_memory_tokens_after_extraction",
                            sample_id=folder_name,
                            extra=f"rank={self.rank}, extracted_timestep={sample_extract_timestep}"
                        )
                        self._progress_print("skip_empty_memory_tokens_after_extraction", sample_idx, extra=f"t={sample_extract_timestep}")
                        return None

                    del latents_extra
                    if bool(getattr(self, 'memory_writer_enabled', False)):
                        latents_stage2_update = _encode_with_tiled_fallback(stage2_update_v_tensor, encode_tag="stage2_update_memory")
                        self._progress_print("stage2_update_memory_vae_done", sample_idx, extra=f"latents_shape={tuple(latents_stage2_update.shape)}")
                        update_tokens, update_meta = _extract_role_tokens_for_stage2_update(
                            stage2_update_v_tensor,
                            latents_stage2_update,
                            stage2_update_memory_text.get('prompt', text),
                            chars_to_run,
                            tag="stage2_update_memory",
                        )
                        del latents_stage2_update
                        if update_tokens is None or _memory_count_tokens(update_tokens) <= 0:
                            self._log_skip(
                                reason="empty_stage2_update_memory_tokens_after_extraction",
                                sample_id=folder_name,
                                extra=f"rank={self.rank}, extracted_timestep={sample_extract_timestep}"
                            )
                            self._progress_print("skip_empty_stage2_update_memory_tokens_after_extraction", sample_idx, extra=f"t={sample_extract_timestep}")
                            return None
                        if _pm_is_layerwise_container(update_tokens):
                            stage2_update_memory_bank_embeddings = _pm_make_layerwise_container({
                                layer: {'0': layer_tokens.detach().cpu()}
                                for layer, layer_tokens in _pm_iter_layerwise_items(update_tokens)
                                if isinstance(layer_tokens, torch.Tensor)
                            })
                            stage2_update_memory_bank_token_meta = _pm_make_layerwise_container({
                                layer: {'0': _pm_select_layerwise_value(update_meta, layer, default=[])}
                                for layer, _ in _pm_iter_layerwise_items(update_tokens)
                            })
                        else:
                            stage2_update_memory_bank_embeddings = {'0': update_tokens.detach().cpu()}
                            stage2_update_memory_bank_token_meta = {'0': list(update_meta)}
                        self._progress_print("stage2_update_memory_tokens_ready", sample_idx, extra=f"token_count={_memory_count_tokens(update_tokens)}")
                    self._aggressive_release_cuda_cache(sample_idx, stage='after_memory_extract')
            finally:
                if prev_lora_states is not None:
                    self._restore_lora_states(prev_lora_states)
                if shared_mode:
                    if prev_train_pipe:
                        self._shared_train_pipe.train()
                    else:
                        self._shared_train_pipe.eval()
                    if prev_dit_train:
                        self._shared_train_pipe.denoising_model().train()
                    else:
                        self._shared_train_pipe.denoising_model().eval()
                    if prev_vae_pipe:
                        self._shared_vae_pipe.train()
                    else:
                        self._shared_vae_pipe.eval()

        raw_frf = sample.get('first_ref_frames')
        max_ref_frames_for_train = int(self.extract_config.get('max_ref_frames_for_train', 5) or 5)
        if isinstance(raw_frf, list):
            safe_frf = [self._to_u8_hwc(f.squeeze(0) if f.dim() == 5 else f) for f in raw_frf]
            if max_ref_frames_for_train > 0:
                safe_frf = safe_frf[:max_ref_frames_for_train]
        elif isinstance(raw_frf, torch.Tensor):
            if raw_frf.dim() == 4:
                safe_frf = [self._to_u8_hwc(raw_frf[i]) for i in range(raw_frf.shape[0])]
                if max_ref_frames_for_train > 0:
                    safe_frf = safe_frf[:max_ref_frames_for_train]
            else:
                safe_frf = self._to_u8_hwc(raw_frf)
        else:
            safe_frf = None

        raw_rrf = sample.get('random_ref_frame')
        safe_rrf = self._to_u8_hwc(raw_rrf) if raw_rrf is not None else None

        precomputed_image_emb = None
        if self.extract_config.get('precompute_image_emb', False) and safe_frf is not None and safe_rrf is not None:
            try:
                if isinstance(safe_frf, list):
                    pil_first_ref_frames = [Image.fromarray(f.numpy()) for f in safe_frf if f is not None]
                else:
                    pil_first_ref_frames = [Image.fromarray(safe_frf.numpy())]
                pil_rand_ref_frame = Image.fromarray(safe_rrf.numpy())

                condition_frames = pil_first_ref_frames[:1]
                num_motion_frames = int(self.extract_config.get('num_motion_frames', 1) or 1)
                num_overlap_frame = int(self.extract_config.get('num_overlap_frame', 0) or 0)
                if num_overlap_frame > 0:
                    num_motion_frames = max(num_motion_frames, num_overlap_frame)
                p_motion_threshold = float(self.extract_config.get('p_motion_threshold', 0.9) or 0.9)
                repeat_first_frame = bool(self.extract_config.get('repeat_first_frame', False))
                if num_motion_frames > 1 and len(pil_first_ref_frames) > 0:
                    if random.random() < p_motion_threshold:
                        condition_frames = pil_first_ref_frames[:num_motion_frames]
                    elif repeat_first_frame:
                        condition_frames = [pil_first_ref_frames[0]] * num_motion_frames

                num_condition_frames = len(condition_frames)
                _, f_lat, h_lat, w_lat = latents_main.shape
                image_emb_tmp = self.pipe.encode_images_adaptive(
                    condition_frames,
                    pil_rand_ref_frame,
                    f_lat * 4 - 3,
                    h_lat * 8,
                    w_lat * 8,
                    use_first_aug=bool(self.extract_config.get('use_first_aug', False)),
                    ref_pad_cfg=bool(self.extract_config.get('ref_pad_cfg', False)),
                    ref_pad_num=int(self.extract_config.get('ref_pad_num', 0) or 0),
                    num_motion_latent=self.extract_config.get('num_motion_latent', None),
                )

                precomputed_image_emb = {'num_condition_frames': num_condition_frames}
                for k, v in image_emb_tmp.items():
                    if isinstance(v, torch.Tensor):
                        precomputed_image_emb[k] = v.detach().cpu()
                    else:
                        precomputed_image_emb[k] = v
                self._progress_print("precompute_image_emb_done", sample_idx, extra=f"num_condition_frames={num_condition_frames}")
            except Exception as e:
                precomputed_image_emb = None
                if self.extract_config.get('precompute_image_emb_strict', False):
                    raise RuntimeError(f"precompute_image_emb failed (rank={self.rank}, sample={sample_idx}): {e}")
                self._progress_print("precompute_image_emb_failed", sample_idx, extra=str(e))

        # Inline mode runs extraction then training on the same rank process.
        # Offload image encoder after extraction-side precompute to lower training peak memory.
        if self.offload_image_encoder_after_extraction:
            with suppress(Exception):
                if hasattr(self.pipe, 'image_encoder') and self.pipe.image_encoder is not None:
                    self.pipe.image_encoder.to(device='cpu', dtype=torch.float32)

        batch = {
            'latents': latents_main,
            'text': text,
            # Keep compatibility key but avoid duplicating bank tensor payload in batch.
            'memory_feature_tokens_selected': torch.zeros(0, 5120),
            'memory_feature_positions_selected': memory_feature_positions_selected.detach().cpu() if memory_feature_positions_selected is not None else torch.zeros(0, 3, dtype=torch.long),
            'memory_feature_token_meta_selected': (
                list(memory_feature_token_meta_selected)
                if need_memory_extract and isinstance(memory_feature_token_meta_selected, list)
                else (memory_feature_token_meta_selected if need_memory_extract else [])
            ),
            'memory_token_lengths_per_character': memory_token_lengths_per_character if need_memory_extract else None,
            'memory_feature_bank_tokens_selected': memory_bank_embeddings if need_memory_extract else None,
            'memory_feature_bank_token_meta_selected': memory_bank_token_meta if need_memory_extract else None,
            'stage2_update_memory_feature_bank_tokens_selected': stage2_update_memory_bank_embeddings if (need_memory_extract and bool(getattr(self, 'memory_writer_enabled', False))) else None,
            'stage2_update_memory_feature_bank_token_meta_selected': stage2_update_memory_bank_token_meta if (need_memory_extract and bool(getattr(self, 'memory_writer_enabled', False))) else None,
            'memory_bank_percents': memory_bank_percents if need_memory_extract else None,
            'extracted_timestep': sample_train_timestep,
            'use_memory': need_memory_extract,
            'first_ref_frames': safe_frf,
            'random_ref_frame': safe_rrf,
            'precomputed_image_emb': precomputed_image_emb,
            'folder_name': folder_name,
            'video_id': sample.get('video_id'),
            'group_index': sample.get('group_index'),
            'core_clip_index': sample.get('core_clip_index'),
            'memory_clip_index': sample.get('memory_clip_index'),
            'sample_idx': sample_idx,
        }

        mem_token_count = 0
        bank_tokens_for_log = batch.get('memory_feature_bank_tokens_selected')
        if _pm_is_layerwise_container(bank_tokens_for_log):
            mem_token_count = _memory_count_tokens(bank_tokens_for_log)
        elif isinstance(bank_tokens_for_log, dict):
            for _v in bank_tokens_for_log.values():
                if isinstance(_v, torch.Tensor) and _v.ndim >= 2 and int(_v.shape[0]) > 0:
                    mem_token_count = int(_v.shape[0])
                    break
        self._progress_print("batch_ready", sample_idx, extra=f"use_memory={use_memory}, mem_tokens={mem_token_count}")

        return batch

    def __iter__(self):
        self._lazy_init()
        self._progress_print("iterator_started", force=True)

        pending_batch = None
        pending_sample_idx = None

        while True:
            sample_idx = pending_sample_idx
            if pending_batch is None:
                self._sample_counter += 1
                sample_idx = f"inline_dp{self.dp_rank}_{self._sample_counter}"
                self._progress_print("sample_loop_start", sample_idx)
                batch = None
                try:
                    sample_plan = None
                    if self.tp_rank == 0:
                        sample_plan = {
                            'data_idx': int(random.randint(0, len(self.dataset) - 1)),
                            'sample_seed': int(secrets.randbits(32)),
                        }
                    sample_plan = self._tp_broadcast_object(sample_plan)
                    data_idx = int(sample_plan['data_idx'])
                    sample_seed = int(sample_plan['sample_seed'])

                    with self._sample_seed_scope(sample_seed):
                        raw_sample = self.dataset[data_idx]
                        if raw_sample is None:
                            self._log_skip("dataset_none_sample", sample_id=sample_idx, extra=f"data_idx={data_idx}")
                            self._progress_print("skip_dataset_none_sample", sample_idx, extra=f"data_idx={data_idx}, seed={sample_seed}")
                        else:
                            folder_name = raw_sample.get('folder_name') if isinstance(raw_sample, dict) else None
                            self._progress_print("dataset_sample_loaded", sample_idx, extra=f"data_idx={data_idx}, folder={folder_name}, seed={sample_seed}")
                            try:
                                batch = self._build_batch_from_raw_sample(raw_sample, sample_idx)
                            except Exception as e:
                                folder_name = raw_sample.get('folder_name') if isinstance(raw_sample, dict) else sample_idx
                                self._log_skip("inline_extract_exception", sample_id=folder_name, extra=str(e))
                                if not self._inline_traceback_printed:
                                    tb = traceback.format_exc()
                                    print(f"[InlineDataset Rank {self.rank}] inline_extract_exception traceback:\n{tb}", flush=True)
                                    self._inline_traceback_printed = True
                                self._progress_print("skip_inline_extract_exception", sample_idx, extra=str(e))
                                if self.debug:
                                    print(f"[InlineDataset Rank {self.rank}] extract error: {e}", flush=True)
                except Exception as e:
                    self._log_skip("dataset_getitem_error", sample_id=sample_idx, extra=str(e))
                    self._progress_print("skip_dataset_getitem_error", sample_idx, extra=str(e))

                self._offload_local_extract_pipe_if_needed(sample_idx=sample_idx, reason='post_build_batch')

                if batch is None:
                    self._progress_print("batch_none_skip", sample_idx)
                else:
                    pending_batch = batch
                    pending_sample_idx = sample_idx

            if self.sync_before_yield and self.world_size > 1 and dist.is_available() and dist.is_initialized():
                sync_sample_idx = pending_sample_idx if pending_sample_idx is not None else sample_idx
                local_ready = bool(pending_batch is not None)
                self._progress_print(
                    "pre_yield_ready_check",
                    sync_sample_idx,
                    extra=f"local_ready={1 if local_ready else 0}, tp_size={self.tp_size}",
                )
                try:
                    if self.tp_size > 1:
                        sync_group = self._ensure_tp_group()
                        all_ready, ready_count, group_world_size = self._all_ranks_ready_with_timeout(
                            local_ready=local_ready,
                            timeout_minutes=self.yield_sync_timeout_minutes,
                            sync_tag='pre_yield_ready_tp',
                            sample_idx=sync_sample_idx,
                            group=sync_group,
                        )
                        if all_ready and sync_group is not None:
                            self._progress_print(
                                "pre_yield_sync_enter",
                                sync_sample_idx,
                                extra=f"ready_count={ready_count}/{group_world_size}, tp_size={self.tp_size}",
                            )
                            self._barrier_with_timeout(
                                timeout_minutes=self.yield_sync_timeout_minutes,
                                sync_tag='pre_yield_sync_tp',
                                sample_idx=sync_sample_idx,
                                group=sync_group,
                            )
                            self._progress_print("pre_yield_sync_exit", sync_sample_idx)
                    else:
                        all_ready, ready_count, group_world_size = self._all_ranks_ready_with_timeout(
                            local_ready=local_ready,
                            timeout_minutes=self.yield_sync_timeout_minutes,
                            sync_tag='pre_yield_ready_world',
                            sample_idx=sync_sample_idx,
                            group=None,
                        )
                        if all_ready:
                            self._progress_print(
                                "pre_yield_sync_enter",
                                sync_sample_idx,
                                extra=f"ready_count={ready_count}/{group_world_size}, tp_size={self.tp_size}",
                            )
                            self._barrier_with_timeout(
                                timeout_minutes=self.yield_sync_timeout_minutes,
                                sync_tag='pre_yield_sync_world',
                                sample_idx=sync_sample_idx,
                                group=None,
                            )
                            self._progress_print("pre_yield_sync_exit", sync_sample_idx)
                except Exception as e:
                    raise RuntimeError(
                        f"pre_yield_sync_timeout: rank={self.rank}, sample={sync_sample_idx}, "
                        f"timeout_min={self.yield_sync_timeout_minutes}, err={e}"
                    ) from e
                if not all_ready:
                    if pending_batch is not None:
                        self._progress_print(
                            "pre_yield_keep_ready_batch",
                            sync_sample_idx,
                            extra=f"ready_count={ready_count}/{group_world_size}, waiting_for_peer_resample=1",
                        )
                    continue

            if pending_batch is None:
                continue

            out_batch = pending_batch
            out_sample_idx = pending_sample_idx
            pending_batch = None
            pending_sample_idx = None
            cuda_refs = _find_cuda_tensors_in_tree(out_batch, prefix="out_batch")
            if cuda_refs and bool(getattr(self, 'debug', False)):
                print(f"[YieldCudaRefsBeforeCPU] sample={out_sample_idx} refs={cuda_refs}", flush=True)
            out_batch = _detach_cpu_tree(out_batch)
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            self._progress_print("yield_batch", out_sample_idx)
            yield out_batch

# =============================================================================
# Lightning Model
# =============================================================================

class LightningModelForTrainWithMemoryV4(pl.LightningModule):
    """
    Lightning model with memory mechanism.
    """
    
    def __init__(
        self,
        ckpt_dir,
        learning_rate=1e-5,
        lora_rank=4,
        lora_alpha=4,
        train_architecture="lora",
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        init_lora_weights="kaiming",
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None,
        model_VAE=None,
        args=None,
        projector_bottleneck=256,
        latent_dim=None,
        timing_tracker=None
    ):
        super().__init__()

        runtime_device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        self.pipe = build_wan22_training_pipe(
            ckpt_dir=ckpt_dir,
            device=runtime_device,
            torch_dtype=torch.bfloat16,
            task="i2v-A14B",
            train_noise_domain=(getattr(args, 'train_noise_domain', 'low_noise') if args else 'low_noise'),
        )
        _install_lightweight_pipeline_lifecycle(self.pipe)
        self.pipe.scheduler.set_timesteps(1000, shift=1.0)
        self.use_projector = getattr(args, 'use_projector', False) if args else False
        self.train_noise_domain = str(getattr(args, 'train_noise_domain', 'low_noise') if args else 'low_noise').strip().lower()
        if self.train_noise_domain not in ('low_noise', 'high_noise'):
            raise ValueError(f"train_noise_domain must be 'low_noise' or 'high_noise', got {self.train_noise_domain!r}")
        # Register denoising backbones on the LightningModule so DDP/Lightning can
        # see LoRA trainables even when they are attached under self.pipe.*.
        self.low_noise_model = getattr(self.pipe, 'low_noise_model', None)
        self.high_noise_model = getattr(self.pipe, 'high_noise_model', None)
        self.dit = getattr(self.pipe, 'dit', None)

        dit_model_runtime = self.pipe.denoising_model()
        runtime_patch_dim = getattr(dit_model_runtime, 'dim', None)
        if runtime_patch_dim is None:
            raise ValueError("Cannot infer patch dimension: runtime DiT has no 'dim' attribute")
        try:
            patch_dim = int(runtime_patch_dim)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid runtime DiT dimension {runtime_patch_dim!r}; expected a positive integer"
            ) from exc
        if patch_dim <= 0:
            raise ValueError(f"Invalid runtime DiT dimension {patch_dim}; expected a positive integer")

        tiler_kwargs = {}
        if model_VAE is not None:
            self.pipe_VAE = model_VAE.pipe.eval()
            tiler_kwargs = getattr(model_VAE, 'tiler_kwargs', {}) or {}
        else:
            tiler_kwargs = {
                "tiled": bool(getattr(args, 'tiled', False)) if args else False,
                "tile_size": (
                    int(getattr(args, 'tile_size_height', 34)) if args else 34,
                    int(getattr(args, 'tile_size_width', 34)) if args else 34,
                ),
                "tile_stride": (
                    int(getattr(args, 'tile_stride_height', 18)) if args else 18,
                    int(getattr(args, 'tile_stride_width', 16)) if args else 16,
                ),
            }
            self.pipe_VAE = SharedVAEPipelineView(self.pipe, tiler_kwargs=tiler_kwargs).eval()
        _install_tp2dp2_pipeline_only_patch(self.pipe_VAE)
        self.tiler_kwargs = dict(tiler_kwargs)
        
        # Patch dimension (5120 for Wan model)
        self.patch_dim = patch_dim
        # Will be inferred from VAE if not provided
        # Get latent_dim from VAE encoder output channels
        # The latent_dim should match the VAE's output channels (typically 8 for standard VAE)
        latent_dim_from_vae = None
        if hasattr(self.pipe_VAE, 'vae') and hasattr(self.pipe_VAE.vae, 'encoder'):
            # Try to get from encoder config
            encoder = self.pipe_VAE.vae.encoder
            if hasattr(encoder, 'out_channels'):
                latent_dim_from_vae = encoder.out_channels
            elif hasattr(encoder, 'conv_out') and hasattr(encoder.conv_out, 'out_channels'):
                latent_dim_from_vae = encoder.conv_out.out_channels
        if latent_dim_from_vae is not None:
            latent_dim = latent_dim_from_vae
        elif latent_dim is None:
            raise ValueError("Cannot infer latent_dim from VAE. Please provide latent_dim explicitly.")
        

        
        # Residual projector is created only when enabled.
        # This avoids redundant parameter allocation when --no-use_projector is used.
        if self.use_projector:
            self.memory_projector = StyleAwareMemoryProjector(
                dim=patch_dim,
                time_embed_dim=patch_dim, # DiT usually projects time to hidden_dim
                latent_dim=latent_dim,
                bottleneck_dim=projector_bottleneck,
                dropout=0.1
            )
        else:
            self.memory_projector = None
        from collections import deque
        self.projector_weight_stats = deque(maxlen=200) 
        self.use_learnable_memory_pos = bool(getattr(args, "use_learnable_memory_pos", False))
        self.use_segment_embed = bool(getattr(args, "use_segment_embed", False))
        self.noise_domain_boundary = float(getattr(args, 'noise_domain_boundary_ratio', 0.9)) if args else 0.9
        self.char_attn_noise_scope = str(getattr(args, 'char_attn_noise_scope', self.train_noise_domain)) if args else self.train_noise_domain
        self.low_noise_lora_adapter_name = str(getattr(args, 'low_noise_lora_adapter_name', 'low_noise')) if args else 'low_noise'
        self.high_noise_lora_adapter_name = str(getattr(args, 'high_noise_lora_adapter_name', 'high_noise')) if args else 'high_noise'
        self._active_noise_lora_adapter = None
        self.enable_sparse_role_memory_attn = bool(getattr(args, 'enable_sparse_role_memory_attn', True)) if args else True
        self.sparse_role_memory_layer_idx = int(getattr(args, 'sparse_role_memory_layer_idx', 3)) if args else 3
        self.sparse_role_memory_injection_layers = _parse_layer_indices_csv(
            getattr(args, 'sparse_role_memory_injection_layers', None) if args else None,
            fallback_idx=self.sparse_role_memory_layer_idx,
        )
        self.sparse_role_memory_num_heads = int(getattr(args, 'sparse_role_memory_num_heads', 8)) if args else 8
        self.sparse_role_memory_head_dim = int(getattr(args, 'sparse_role_memory_head_dim', 128)) if args else 128
        self.sparse_role_memory_rope_dim = int(getattr(args, 'sparse_role_memory_rope_dim', 256)) if args else 256
        self.sparse_role_memory_use_half_role_heads = bool(getattr(args, 'sparse_role_memory_use_half_role_heads', True)) if args else True
        self.sparse_role_memory_feature_source = str(getattr(args, 'sparse_role_memory_feature_source', 'attn_out')) if args else 'attn_out'
        self.sparse_role_memory_init_scale = float(getattr(args, 'sparse_role_memory_init_scale', 0.1)) if args else 0.1
        self.sparse_role_memory_time_gate = bool(getattr(args, 'sparse_role_memory_time_gate', True)) if args else True
        self.sparse_role_memory_query_chunk_size = max(0, int(getattr(args, 'sparse_role_memory_query_chunk_size', 128))) if args else 128
        self.sparse_role_memory_layer_scales = _parse_layer_scale_map(getattr(args, 'sparse_role_memory_layer_scales', '')) if args else {}
        self.debug_sparse_role_memory_attn = bool(getattr(args, 'debug_sparse_role_memory_attn', False)) if args else False
        self.train_stage = str(getattr(args, 'train_stage', 'stage1') if args else 'stage1').strip().lower()
        self.jigsaw_memory_bank_mode = str(getattr(args, 'jigsaw_memory_bank_mode', 'single') if args else 'single').strip().lower()
        self.jigsaw_extra_encoder_enabled = _memory_encoder_enabled(getattr(args, 'jigsaw_extra_encoder_mode', 'off') if args else 'off')
        self.jigsaw_extra_encoder_layers = _jigsaw_parse_layer_list(getattr(args, 'jigsaw_extra_encoder_layers', '0-15') if args else '0-15')
        self.jigsaw_extra_encoder_layer_groups = _jigsaw_parse_layer_groups(getattr(args, 'jigsaw_extra_encoder_layer_groups', '0-4,5-10,11-15') if args else '0-4,5-10,11-15')
        self.jigsaw_extra_encoder_slots = max(1, int(getattr(args, 'jigsaw_extra_encoder_slots', 32))) if args else 32
        self.jigsaw_extra_encoder_dim = max(1, int(getattr(args, 'jigsaw_extra_encoder_dim', 512))) if args else 512
        self.jigsaw_extra_encoder_hidden_dim = max(1, int(getattr(args, 'jigsaw_extra_encoder_hidden_dim', 1024))) if args else 1024
        self.jigsaw_extra_encoder_use_t_embed = bool(getattr(args, 'jigsaw_extra_encoder_use_t_embed', False)) if args else False
        self.jigsaw_memory_encoder_t_embed_source = str(
            getattr(args, 'jigsaw_memory_encoder_t_embed_source', 'current') if args else 'current'
        ).strip().lower()
        self.jigsaw_extra_encoder_use_slot_index_embed = bool(getattr(args, 'jigsaw_extra_encoder_use_slot_index_embed', False)) if args else False
        self.jigsaw_extra_encoder_aux_weight = float(getattr(args, 'jigsaw_extra_encoder_aux_weight', 0.05)) if args else 0.05
        self.jigsaw_extra_encoder_bg_tokens = max(0, int(getattr(args, 'jigsaw_extra_encoder_bg_tokens', 64))) if args else 64
        self.jigsaw_stage2_writer_mode = str(getattr(args, 'jigsaw_stage2_writer_mode', 'auto') if args else 'auto').strip().lower()
        self.memory_writer_effective_mode = _memory_writer_effective_mode(self.train_stage, self.jigsaw_stage2_writer_mode)
        self.memory_writer_enabled = bool(self.jigsaw_extra_encoder_enabled and self.memory_writer_effective_mode == 'residual')
        self.jigsaw_stage2_writer_hidden_dim = max(1, int(getattr(args, 'jigsaw_stage2_writer_hidden_dim', 1024))) if args else 1024
        self.jigsaw_stage2_writer_init_scale = float(getattr(args, 'jigsaw_stage2_writer_init_scale', 0.1)) if args else 0.1
        self.jigsaw_stage2_writer_precision_tau = float(getattr(args, 'jigsaw_stage2_writer_precision_tau', 0.3)) if args else 0.3
        self.jigsaw_stage2_writer_precision_scale = float(getattr(args, 'jigsaw_stage2_writer_precision_scale', 10.0)) if args else 10.0
        self.jigsaw_stage2_writer_max_delta_ratio = float(getattr(args, 'jigsaw_stage2_writer_max_delta_ratio', 0.0)) if args else 0.0
        self.jigsaw_stage2_writer_max_delta_norm = float(getattr(args, 'jigsaw_stage2_writer_max_delta_norm', 0.0)) if args else 0.0
        self.jigsaw_stage2_writer_detach_c_short = bool(getattr(args, 'jigsaw_stage2_writer_detach_c_short', True)) if args else True
        self.jigsaw_stage2_writer_context_mode = str(getattr(args, 'jigsaw_stage2_writer_context_mode', 'mean') if args else 'mean').strip().lower()
        self.jigsaw_stage2_writer_attention_scale = float(getattr(args, 'jigsaw_stage2_writer_attention_scale', 10.0)) if args else 10.0
        self.stage2_memory_encoder_lr = float(getattr(args, 'stage2_memory_encoder_lr', 1e-5)) if args else 1e-5
        self.stage2_char_attn_lr = float(getattr(args, 'stage2_char_attn_lr', 1e-5)) if args else 1e-5
        self.stage2_stage1_branch_prob = float(getattr(args, 'stage2_stage1_branch_prob', 0.2)) if args else 0.2
        self.stage2_stage1_branch_prob = min(max(self.stage2_stage1_branch_prob, 0.0), 1.0)
        self.stage2_unfreeze_memory_encoder = bool(getattr(args, 'stage2_unfreeze_memory_encoder', True)) if args else True
        self.stage2_unfreeze_char_attn = bool(getattr(args, 'stage2_unfreeze_char_attn', True)) if args else True
        self.jigsaw_disable_memory_side_rope = bool(getattr(args, 'jigsaw_disable_memory_side_rope', True)) if args else True
        self._effective_use_segment_embed = False
        self._effective_use_learnable_memory_pos = False
        # n*2048: max_total = 2048 * max_memory_characters（固定每角色 2048）
        self._max_memory_characters = getattr(args, "max_memory_characters", 2)
        self._max_total_memory_tokens = 2048 * self._max_memory_characters
        self.memory_embeddings = None
        self.attn_response_history = deque(maxlen=200)
        # Timing tracker
        self.timing_tracker = timing_tracker
        
        # Training settings (compatible with v2/v3)
        self.use_first_aug = getattr(args, 'use_first_aug', False)
        self.ref_pad_num = getattr(args, 'ref_pad_num', 0)
        self.ref_pad_cfg = getattr(args, 'ref_pad_cfg', False)
        self.num_overlap_frame = int(getattr(args, 'num_overlap_frame', 0) or 0)
        self.num_motion_latent = getattr(args, 'num_motion_latent', None)
        self.num_motion_frames = getattr(args, 'num_motion_frames', 1)
        if self.num_overlap_frame > 0:
            self.num_motion_frames = max(int(self.num_motion_frames), int(self.num_overlap_frame))
        self.p_motion_threshold = getattr(args, 'p_motion_threshold', 0.9)
        self.repeat_first_frame = getattr(args, 'repeat_first_frame', False)
        self.y_error_num = getattr(args, 'y_error_num', 1)
        
        # Error recycling (from v2)
        self.train_memory_only = getattr(args, 'train_memory_only', False)
        self.use_error_recycling = getattr(args, 'use_error_recycling', False) if not self.train_memory_only else False
        self.use_existing_error_buffers = getattr(args, 'use_existing_error_buffers', False) if self.train_memory_only else False
        
        # Memory dropping for CFG training
        self.memory_drop_prob = getattr(args, 'memory_drop_prob', 0.1)
        self.negative_prompt = getattr(args, 'negative_prompt', "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
        self.probe_bbox_loss_weight_x2 = bool(getattr(args, 'probe_bbox_loss_weight_x2', False)) if args else False
        self.train_query_probe_timestep_mode = str(
            getattr(args, 'train_query_probe_timestep_mode', 'current') if args else 'current'
        ).strip().lower()
        
        self.error_buffer_size = getattr(args, 'error_buffer_k', 500)
        self.buffer_replacement_strategy = getattr(args, 'buffer_replacement_strategy', 'random')
        self.buffer_warmup_iter = getattr(args, 'buffer_warmup_iter', 50)
        self.timestep_grid_size = getattr(args, 'timestep_grid_size', 25)
        
        num_grids = getattr(args, 'num_grids', 40)
        self.inferece_timesteps = self.pipe.scheduler.get_timesteps(num_inference_steps=num_grids, denoising_strength=1, shift=5.0)
        self.latent_error_buffer = {i: [] for i in range(num_grids)}
        self.y_error_buffer = {i: [] for i in range(num_grids)}
        
        self.iteration_count = 0
        self.error_modulate_factor = getattr(args, 'error_modulate_factor', 0.0)
        self.y_error_sample_from_all_grids = getattr(args, 'y_error_sample_from_all_grids', False)
        
        self.noise_prob = getattr(args, 'noise_prob', 0.99)
        self.y_prob = getattr(args, 'y_prob', 0.99)
        self.latent_prob = getattr(args, 'latent_prob', 0.99)
        self.clean_prob = getattr(args, 'clean_prob', 0.1)
        self.clean_buffer_update_prob = getattr(args, 'clean_buffer_update_prob', 0.5)
        
        self.freeze_parameters()
        
        # GPU optimization flags
        self._keep_models_on_gpu = True
        self._vae_on_gpu = False
        self._image_encoder_on_gpu = False
        self.aggressive_vram_optimization = bool(getattr(args, 'aggressive_vram_optimization', False)) if args else False
        self.keep_image_encoder_on_gpu = bool(getattr(args, 'keep_image_encoder_on_gpu', False)) if args else False
        self._vae_original_dtype = _get_vae_runtime_dtype(self.pipe_VAE.vae, default_dtype=torch.float32)
        self._enable_frequent_cache_clear = False
        
        if train_architecture == "lora":
            single_noise_target = self.pipe.high_noise_model if self.train_noise_domain == 'high_noise' else self.pipe.low_noise_model
            self.add_lora_to_model(
                single_noise_target,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        elif train_architecture == "full":
            if self.pipe.low_noise_model is not None and self.train_noise_domain == 'low_noise':
                self.pipe.low_noise_model.requires_grad_(True)
            if self.pipe.high_noise_model is not None and self.train_noise_domain == 'high_noise':
                self.pipe.high_noise_model.requires_grad_(True)
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self._disable_block_checkpoint_under_zero3 = False
        if args is not None:
            _strategy_name = str(getattr(args, 'training_strategy', 'auto') or 'auto').strip().lower()
            _slice_mode = str(getattr(args, 'model_slice_mode', 'none') or 'none').strip().lower()
            if _strategy_name == 'deepspeed_stage_3' and _slice_mode == 'zero3' and self.use_gradient_checkpointing:
                self._disable_block_checkpoint_under_zero3 = True
                self.use_gradient_checkpointing = False
                self.use_gradient_checkpointing_offload = False
                print('[Init][Checkpoint] Disabled torch.utils.checkpoint block recompute under ZeRO-3 because recomputation sees empty local shards ([0]) for some 5120-d parameters.', flush=True)
        self.gradient_clip_val = getattr(args, 'gradient_clip_val', 1.0) if args else 1.0
        self.strict_loading = False
        self.args = args
        self.use_train_weights_for_extract_and_probe = bool(
            getattr(args, 'use_train_weights_for_extract_and_probe', False)
        ) if args else False
        self.tp_size = max(1, int(getattr(args, 'tp_size', 1) if args else 1))
        self._probe_tp_group = None
        self._probe_tp_group_ranks = None
        self._local_probe_pipe = None
        self._local_probe_enabled = False
        self._local_probe_disabled_reason = None
        self._local_probe_leader_only = bool(getattr(args, 'probe_on_tp_leader_only', True)) if args else True
        if args is not None:
            _strategy_name = str(getattr(args, 'training_strategy', 'auto') or 'auto').strip().lower()
            _slice_mode = str(getattr(args, 'model_slice_mode', 'none') or 'none').strip().lower()
            _requested_local_probe = bool(getattr(args, 'use_detached_local_probe_for_zero3', False))
            self._local_probe_enabled = bool(
                _requested_local_probe
            ) and (_strategy_name in ('deepspeed_stage_2', 'deepspeed_stage_3')) and (_slice_mode == 'zero3')
            if self.use_train_weights_for_extract_and_probe and _requested_local_probe:
                self._local_probe_enabled = False
                self._local_probe_disabled_reason = (
                    'use_train_weights_for_extract_and_probe enabled: probe must run on current training model weights'
                )
                print(
                    "[CharacterSemanticProbe] Disabled detached local probe because use_train_weights_for_extract_and_probe is enabled.",
                    flush=True,
                )
        self.resume_step_from_lora = self._infer_resume_step_from_lora(getattr(args, 'pretrained_lora_path', None) if args else None)
        self.resume_from_ds_ckpt = bool(getattr(args, 'resume_weights_from_ds_ckpt', None)) if args else False
        self.display_step_offset = int(self.resume_step_from_lora) if self.resume_from_ds_ckpt else 0
        self._recent_non_memory_drop_losses = deque(maxlen=2)
        self._recent_memory_drop_losses = deque(maxlen=2)
        self.loss_logger = None
        self._lora_save_thread = None
        self.enable_attn_monitor = getattr(args, 'enable_attn_monitor', False) if args else False
        self.ddp_skip_sync_timeout_minutes = max(1, int(getattr(args, 'ddp_skip_sync_timeout_minutes', 10) or 10)) if args else 10
        self.cuda_trim_interval = max(1, int(getattr(args, 'cuda_trim_interval', 10))) if args else 10
        self.cuda_trim_min_fragment_mb = max(128, int(getattr(args, 'cuda_trim_min_fragment_mb', 1536))) if args else 1536
        self._video_freqs_cache = {}
        self.memory_fusion = None
        self.sparse_role_memory_attn_low_noise = None
        self.sparse_role_memory_attn_high_noise = None
        if self.enable_sparse_role_memory_attn:
            def _build_sparse_module():
                return CharacterWiseCrossAttention(
                    dim=patch_dim,
                    num_heads=self.sparse_role_memory_num_heads,
                    head_dim=self.sparse_role_memory_head_dim,
                    rope_dim=self.sparse_role_memory_rope_dim,
                    use_half_role_heads=self.sparse_role_memory_use_half_role_heads,
                    max_query_tokens_per_role=int(getattr(args, 'max_memory_tokens_per_character', 0)) if args else 0,
                    query_chunk_size=self.sparse_role_memory_query_chunk_size,
                    use_memory_side_rope=not bool(self.jigsaw_disable_memory_side_rope),
                    add_rope_center_to_value=not bool(self.jigsaw_disable_memory_side_rope),
                    init_scale=self.sparse_role_memory_init_scale,
                    time_gate=self.sparse_role_memory_time_gate,
                    debug=self.debug_sparse_role_memory_attn,
                )

            scope_l = str(self.char_attn_noise_scope).strip().lower()
            if scope_l == 'low_noise':
                self.sparse_role_memory_attn_low_noise = _build_sparse_module()
            if scope_l == 'high_noise':
                self.sparse_role_memory_attn_high_noise = _build_sparse_module()

            # Backward-compatible alias; runtime uses per-domain selector.
            if self.sparse_role_memory_attn_low_noise is not None:
                self.sparse_role_memory_attn = self.sparse_role_memory_attn_low_noise
            else:
                self.sparse_role_memory_attn = self.sparse_role_memory_attn_high_noise
        else:
            self.sparse_role_memory_attn = None
        self.jigsaw_extra_encoder_low_noise = None
        self.jigsaw_extra_encoder_high_noise = None
        if self.jigsaw_extra_encoder_enabled:
            def _build_extra_encoder():
                return MemoryEncoderBank(
                    dim=int(patch_dim),
                    layer_groups=self.jigsaw_extra_encoder_layer_groups,
                    slots=int(self.jigsaw_extra_encoder_slots),
                    encoder_dim=int(self.jigsaw_extra_encoder_dim),
                    hidden_dim=int(self.jigsaw_extra_encoder_hidden_dim),
                    use_t_embed=bool(self.jigsaw_extra_encoder_use_t_embed),
                    use_slot_index_embed=bool(self.jigsaw_extra_encoder_use_slot_index_embed),
                    time_embed_dim=int(patch_dim),
                )

            scope_l = str(self.char_attn_noise_scope).strip().lower()
            if scope_l == 'low_noise':
                self.jigsaw_extra_encoder_low_noise = _build_extra_encoder()
            if scope_l == 'high_noise':
                self.jigsaw_extra_encoder_high_noise = _build_extra_encoder()
            self.jigsaw_extra_encoder = self.jigsaw_extra_encoder_low_noise or self.jigsaw_extra_encoder_high_noise
            if args is not None:
                wide_layer_list = list(self.jigsaw_extra_encoder_layers)
                wide_layers_csv = ",".join(str(x) for x in wide_layer_list)
                setattr(args, 'extract_layers', wide_layer_list)
                setattr(args, 'sparse_role_memory_injection_layers', wide_layers_csv)
                self.sparse_role_memory_injection_layers = list(wide_layer_list)
            print(
                f"[MemoryEncoder][Init] enabled=1 layers={self.jigsaw_extra_encoder_layers} "
                f"groups={self.jigsaw_extra_encoder_layer_groups} slots={self.jigsaw_extra_encoder_slots} "
                f"dim={self.jigsaw_extra_encoder_dim} hidden={self.jigsaw_extra_encoder_hidden_dim} "
                f"use_t_embed={bool(self.jigsaw_extra_encoder_use_t_embed)} "
                f"use_slot_index_embed={bool(self.jigsaw_extra_encoder_use_slot_index_embed)} "
                f"memory_side_rope={not bool(self.jigsaw_disable_memory_side_rope)}",
                flush=True,
            )
            self.jigsaw_stage2_writer_low_noise = None
            self.jigsaw_stage2_writer_high_noise = None
            if bool(getattr(self, 'memory_writer_enabled', False)):
                def _build_stage2_writer():
                    return MemoryWriter(
                        dim=int(patch_dim),
                        hidden_dim=int(self.jigsaw_stage2_writer_hidden_dim),
                        init_scale=float(self.jigsaw_stage2_writer_init_scale),
                        precision_tau=float(self.jigsaw_stage2_writer_precision_tau),
                        precision_scale=float(self.jigsaw_stage2_writer_precision_scale),
                        max_delta_ratio=float(self.jigsaw_stage2_writer_max_delta_ratio),
                        max_delta_norm=float(self.jigsaw_stage2_writer_max_delta_norm),
                        detach_c_short=bool(self.jigsaw_stage2_writer_detach_c_short),
                        context_mode=str(self.jigsaw_stage2_writer_context_mode),
                        attention_scale=float(self.jigsaw_stage2_writer_attention_scale),
                    )

                if scope_l == 'low_noise':
                    self.jigsaw_stage2_writer_low_noise = _build_stage2_writer()
                if scope_l == 'high_noise':
                    self.jigsaw_stage2_writer_high_noise = _build_stage2_writer()
                self.jigsaw_stage2_writer = self.jigsaw_stage2_writer_low_noise or self.jigsaw_stage2_writer_high_noise
                print(
                    f"[MemoryWriter][Init] enabled=1 mode={self.memory_writer_effective_mode} "
                    f"stage={self.train_stage} hidden={self.jigsaw_stage2_writer_hidden_dim} "
                    f"init_scale={self.jigsaw_stage2_writer_init_scale} tau={self.jigsaw_stage2_writer_precision_tau} "
                    f"scale={self.jigsaw_stage2_writer_precision_scale} "
                    f"context={self.jigsaw_stage2_writer_context_mode} attn_scale={self.jigsaw_stage2_writer_attention_scale}",
                    flush=True,
                )
            else:
                self.jigsaw_stage2_writer = None
            preload_path = getattr(args, 'pretrained_lora_path', None) if args is not None else None
            if preload_path:
                try:
                    payload = _load_checkpoint_payload(preload_path)
                    payload_sd = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
                    sparse_loaded_keys = self._load_sparse_role_memory_state_from_payload(
                        payload_sd,
                        self.train_noise_domain,
                        strict=False,
                    )
                    print(f"[CharacterWiseCrossAttention][Load] preload_path={preload_path} loaded_key_count={sparse_loaded_keys}", flush=True)
                    loaded_keys = self._load_jigsaw_extra_encoder_state_from_payload(
                        payload_sd,
                        self.train_noise_domain,
                        strict=False,
                    )
                    print(f"[MemoryEncoder][Load] preload_path={preload_path} loaded_key_count={loaded_keys}", flush=True)
                    if self.train_stage == 'stage2' and (int(sparse_loaded_keys) <= 0 or int(loaded_keys) <= 0):
                        raise RuntimeError(
                            f"Stage2 requires stage1 sparse role memory and extra encoder weights from {preload_path}; "
                            f"loaded sparse={sparse_loaded_keys}, extra={loaded_keys}"
                        )
                    writer_loaded_keys = self._load_memory_writer_state_from_payload(
                        payload_sd,
                        self.train_noise_domain,
                        strict=False,
                    )
                    if writer_loaded_keys:
                        print(f"[MemoryWriter][Load] preload_path={preload_path} loaded_key_count={writer_loaded_keys}", flush=True)
                except Exception as e:
                    print(f"[MemoryEncoder][Load] warning: failed to preload encoder from {preload_path}: {e}", flush=True)
                    if self.train_stage == 'stage2':
                        raise
            self._freeze_stage1_memory_modules_for_stage2()
        else:
            self.jigsaw_extra_encoder = None
            self.jigsaw_stage2_writer_low_noise = None
            self.jigsaw_stage2_writer_high_noise = None
            self.jigsaw_stage2_writer = None
        self.train_stage = str(getattr(args, 'train_stage', 'stage1') if args else 'stage1').strip().lower()
        self.jigsaw_memory_bank_mode = str(getattr(args, 'jigsaw_memory_bank_mode', 'single') if args else 'single').strip().lower()
        # Legacy multi-bank fallback only; single mode uses per-sample p_extract instead.
        self.memory_bank_percents = (
            parse_float_csv(getattr(args, 'memory_bank_percents', '0.85,0.60,0.35,0.12'), default_list=[0.85, 0.60, 0.35, 0.12])
            if args and self.jigsaw_memory_bank_mode == 'legacy_multi'
            else []
        )
        # Load weights from broken DeepSpeed checkpoint if specified
        resume_ds_ckpt = getattr(args, 'resume_weights_from_ds_ckpt', None) if args else None
        if resume_ds_ckpt is not None:
            self._load_weights_from_deepspeed_ckpt(resume_ds_ckpt)

    def _probe_tp_leader_global_rank(self):
        return (int(self.global_rank) // int(self.tp_size)) * int(self.tp_size)

    def _ensure_probe_tp_group(self):
        if self.tp_size <= 1:
            return None
        if (not dist.is_available()) or (not dist.is_initialized()):
            return None
        if self._probe_tp_group is not None:
            return self._probe_tp_group

        world = int(dist.get_world_size())
        if world % self.tp_size != 0:
            raise RuntimeError(
                f"Invalid TP topology for probe group build: world_size={world}, tp_size={self.tp_size}"
            )

        num_dp_groups = world // self.tp_size
        cur_rank = int(dist.get_rank())
        my_group = None
        my_group_ranks = None

        for dp_idx in range(num_dp_groups):
            ranks = list(range(dp_idx * self.tp_size, (dp_idx + 1) * self.tp_size))
            group = dist.new_group(ranks=ranks)
            if cur_rank in ranks:
                my_group = group
                my_group_ranks = tuple(ranks)

        if my_group is None:
            raise RuntimeError(f"Failed to build probe TP group for rank={cur_rank}")

        self._probe_tp_group = my_group
        self._probe_tp_group_ranks = my_group_ranks
        return self._probe_tp_group

    def _tp_broadcast_probe_object(self, obj):
        if self.tp_size <= 1 or (not dist.is_available()) or (not dist.is_initialized()):
            return obj
        tp_group = self._ensure_probe_tp_group()
        if tp_group is None:
            return obj
        obj_list = [obj]
        dist.broadcast_object_list(obj_list, src=self._probe_tp_leader_global_rank(), group=tp_group)
        return obj_list[0]

    def _get_or_init_local_probe_pipe(self):
        if not self._local_probe_enabled:
            return None
        if self._local_probe_disabled_reason is not None:
            return None
        if self._local_probe_pipe is not None:
            return self._local_probe_pipe
        if self.args is None:
            return None

        ckpt_dir = getattr(self.args, 'ckpt_dir', None)
        if not ckpt_dir:
            raise RuntimeError(
                "detached local probe requires ckpt_dir, but it is missing."
            )
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"detached local probe ckpt_dir not found: {ckpt_dir}")
        self._local_probe_pipe = build_wan22_training_pipe(
            ckpt_dir=ckpt_dir,
            device='cpu',
            torch_dtype=torch.bfloat16,
            task="i2v-A14B",
            train_noise_domain=(getattr(self.args, 'train_noise_domain', 'low_noise') if self.args is not None else 'low_noise'),
        )
        _install_lightweight_pipeline_lifecycle(self._local_probe_pipe)
        _install_tp2dp2_pipeline_only_patch(self._local_probe_pipe)

        probe_device = torch.device(self.device)
        self._local_probe_pipe.device = str(probe_device)
        try:
            if hasattr(self._local_probe_pipe, 'dit') and self._local_probe_pipe.dit is not None:
                self._local_probe_pipe.dit.to(device=probe_device, dtype=torch.bfloat16)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                self._local_probe_disabled_reason = 'probe_dit_to_gpu_oom'
                self._local_probe_pipe = None
                with suppress(Exception):
                    torch.cuda.empty_cache()
                print(
                    f"[CharacterSemanticProbe][Rank {self.global_rank}] Disable detached local probe due OOM while moving DiT to GPU.",
                    flush=True,
                )
                return None
            raise
        # Keep detached probe text encoder on CPU; tokenization uses pipe_VAE tokenizer in training path.
        print(
            f"[CharacterSemanticProbe][Rank {self.global_rank}] Using detached local full probe pipe under ZeRO-3.",
            flush=True,
        )
        return self._local_probe_pipe

    def _infer_resume_step_from_lora(self, lora_path):
        if not lora_path:
            return 0
        try:
            name = Path(str(lora_path)).name
            matched = re.search(r"step_(\d+)", name)
            if matched:
                return int(matched.group(1))
            return 0
        except Exception:
            return 0
            
    def _load_weights_from_deepspeed_ckpt(self, ckpt_path):
        raise RuntimeError(
            "Old DeepSpeed training checkpoints are not supported in the Wan2.2-native SlotMem training stack. "
            f"Got resume_weights_from_ds_ckpt={ckpt_path!r}."
        )
    
    def _get_versioned_save_dir(self):
        """获取 Lightning version 目录，确保新训练不会覆盖旧的 loss_curve/attention_response_evolution"""
        log_dir = getattr(self.trainer, 'log_dir', None)
        if log_dir and os.path.isdir(log_dir):
            return log_dir
        return self.trainer.default_root_dir

    def _maybe_bootstrap_loss_history(self, version_dir):
        """If current version has no loss_history.csv, copy the latest one from previous version dirs."""
        try:
            current_csv = os.path.join(version_dir, "loss_history.csv")
            if os.path.isfile(current_csv):
                return

            version_name = os.path.basename(os.path.normpath(version_dir))
            m = re.match(r"version_(\d+)$", version_name)
            if not m:
                return

            cur_idx = int(m.group(1))
            parent_dir = os.path.dirname(os.path.normpath(version_dir))
            if not os.path.isdir(parent_dir):
                return

            candidates = []
            for entry in os.listdir(parent_dir):
                mm = re.match(r"version_(\d+)$", entry)
                if not mm:
                    continue
                idx = int(mm.group(1))
                if idx >= cur_idx:
                    continue
                csv_path = os.path.join(parent_dir, entry, "loss_history.csv")
                if os.path.isfile(csv_path):
                    candidates.append((idx, csv_path))

            if not candidates:
                return

            candidates.sort(key=lambda x: x[0], reverse=True)
            src_csv = candidates[0][1]
            shutil.copy2(src_csv, current_csv)
            print(f"[LossLogger] Bootstrapped loss history from: {src_csv}")
        except Exception as e:
            print(f"[LossLogger] Warning: failed to bootstrap loss_history.csv: {e}")

    def _infer_resume_step_from_loss_history(self, version_dir):
        """Infer next step from existing loss_history.csv in current version dir."""
        try:
            csv_path = os.path.join(version_dir, "loss_history.csv")
            if not os.path.isfile(csv_path):
                return 0
            df = pd.read_csv(csv_path)
            if 'step' not in df.columns:
                return 0
            step_series = pd.to_numeric(df['step'], errors='coerce').dropna()
            if len(step_series) == 0:
                return 0
            return int(step_series.max()) + 1
        except Exception as e:
            print(f"[LossLogger] Warning: failed to infer resume step from loss history: {e}")
            return 0

    def on_train_start(self):
        # 默认不常驻 image encoder 到 GPU，降低训练阶段峰值显存。
        if hasattr(self.pipe_VAE, 'image_encoder') and self.pipe_VAE.image_encoder is not None:
            if self.keep_image_encoder_on_gpu:
                self.pipe_VAE.image_encoder.to(device=self.device, dtype=torch.float32)
                self._image_encoder_on_gpu = True
            else:
                self.pipe_VAE.image_encoder.to(device='cpu', dtype=torch.float32)
                self._image_encoder_on_gpu = False
        if self.aggressive_vram_optimization and hasattr(self.pipe_VAE, 'vae') and self.pipe_VAE.vae is not None:
            _move_vae_runtime(self.pipe_VAE.vae, device='cpu', dtype=self._vae_original_dtype)
            self._vae_on_gpu = False
        else:
            self._vae_on_gpu = True
        version_dir = self._get_versioned_save_dir()
        inherit_log_history = bool(getattr(self, 'resume_from_ds_ckpt', False))
        if self.global_rank == 0 and inherit_log_history:
            # 使用 version 目录，避免新训练覆盖旧的 loss_curve/attention_response_evolution
            self._maybe_bootstrap_loss_history(version_dir)
        if inherit_log_history and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        resume_step_from_history = self._infer_resume_step_from_loss_history(version_dir) if inherit_log_history else 0
        if resume_step_from_history > int(getattr(self, 'display_step_offset', 0)):
            self.display_step_offset = resume_step_from_history
            print(f"[LossLogger] Resume step inherited from CSV: {self.display_step_offset}")

        # 单进程 inline 模式：每个 rank 只显示自己的训练进度条
        my_position = self.global_rank + 1
        
        # 2. 初始化 tqdm
        # total 使用 estimated_stepping_batches 估算总步数，让进度条能显示百分比
        current_global_step = int(self.global_step)
        display_start_step = max(current_global_step, int(getattr(self, 'display_step_offset', 0)))
        self._display_step_offset_runtime = display_start_step - current_global_step
        self._display_start_step = display_start_step
        self.train_pbar = tqdm(
            desc=f"[Train Rank {self.global_rank}]",
            position=my_position,
            total=display_start_step + self.trainer.estimated_stepping_batches,
            initial=display_start_step,
            leave=True,          # 训练结束保留最后状态
            dynamic_ncols=True,  # 自适应窗口宽度
            smoothing=0.01,      # 平滑系数
            file=sys.stdout      # 确保输出到标准输出
        )
        pixelate_prob = float(getattr(self.args, 'image_condition_people_pixelate_prob', 0.0)) if hasattr(self, 'args') else 0.0
        pixelate_enabled = bool(pixelate_prob > 0.0)
        pixelate_model_path = getattr(self.args, 'image_condition_people_pixelate_model_path', DEFAULT_YOLO_SEG_CKPT) if hasattr(self, 'args') else DEFAULT_YOLO_SEG_CKPT
        pixelate_block = int(getattr(self.args, 'image_condition_people_pixelate_block_size', 12)) if hasattr(self, 'args') else 12
        pixelate_conf = float(getattr(self.args, 'image_condition_people_pixelate_conf', 0.25)) if hasattr(self, 'args') else 0.25
        print(
            f"[PixelateConfig][Rank {int(getattr(self, 'global_rank', -1))}] "
            f"enabled={pixelate_enabled} prob={pixelate_prob:.4f} conf={pixelate_conf:.3f} "
            f"block={pixelate_block} model={pixelate_model_path}",
            flush=True,
        )
        if self.global_rank == 0:
            self.loss_logger = LossLogger(version_dir, inherit_history=inherit_log_history)
    def on_train_batch_end(self, outputs, batch, batch_idx):
            # 只要进度条初始化了，所有 Rank 都会执行这里
            if hasattr(self, 'train_pbar'):
                loss = outputs['loss'].item() if isinstance(outputs, dict) else outputs.item()
                
                # 手动更新 1 步
                self.train_pbar.update(1)
                
                display_step = int(self.global_step) + int(getattr(self, '_display_step_offset_runtime', 0))
                # 设置后缀信息 (Loss, Step)
                # 使用 OrderedDict 或 dict 都可以
                self.train_pbar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "step": display_step,
                })

    # [新增] 训练结束关闭进度条
    def on_train_end(self):
        if self.global_rank == 0 and getattr(self, '_lora_save_thread', None) is not None:
            if self._lora_save_thread.is_alive():
                self._lora_save_thread.join(timeout=180)
                if self._lora_save_thread.is_alive():
                    print("[Checkpoint] Warning: LoRA save thread still running at train end.", flush=True)
        save_every = int(getattr(self.args, "checkpoint_save_every_n_steps", 0) or 0) if hasattr(self, "args") else 0
        if self.global_rank == 0 and save_every > 0:
            display_step = int(self.global_step) + int(getattr(self, '_display_step_offset_runtime', 0))
            if display_step > 0:
                final_save_path = os.path.join(self.trainer.default_root_dir, f"lora_weights_step_{display_step}.pt")
                if not os.path.exists(final_save_path):
                    try:
                        if self.use_projector:
                            lora_state_dict = {
                                n: p.detach().cpu().clone()
                                for n, p in self.named_parameters()
                                if p.requires_grad
                            }
                        else:
                            lora_state_dict = {
                                n: p.detach().cpu().clone()
                                for n, p in self.named_parameters()
                                if p.requires_grad and not n.startswith("memory_projector.")
                            }
                        denoising_lora_state_dict = self._collect_active_denoising_trainable_state_dict()
                        lora_state_dict.update(denoising_lora_state_dict)
                        if hasattr(self, "memory_embeddings") and self.memory_embeddings is not None:
                            if getattr(self.memory_embeddings, "pos_embed", None) is not None:
                                lora_state_dict["memory_pos_embed"] = self.memory_embeddings.pos_embed.data.detach().cpu().clone()
                            if getattr(self.memory_embeddings, "segment_embed", None) is not None:
                                lora_state_dict["memory_segment_embed"] = self.memory_embeddings.segment_embed.data.detach().cpu().clone()
                        torch.save(lora_state_dict, final_save_path)
                        print(f"[Checkpoint] Saved final LoRA weights: {final_save_path}", flush=True)
                    except Exception as save_err:
                        print(f"[Checkpoint] Warning: Failed to save final LoRA weights: {save_err}", flush=True)
        if hasattr(self, 'train_pbar'):
            self.train_pbar.close()
    def freeze_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        self.pipe_VAE.requires_grad_(False)
        self.pipe_VAE.eval()

    def _ddp_sync_should_skip(self, local_skip: bool) -> bool:
        """Synchronize step-skip decision across ranks.
        If any rank needs to skip, all ranks skip to keep collectives aligned.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return bool(local_skip)
        debug_sync = self._should_train_runtime_print()
        sync_t0 = None
        if debug_sync:
            sync_t0 = time.time()
            self._train_runtime_log('ddp_skip_sync_enter', local_skip=bool(local_skip))
        flag = torch.tensor([1 if local_skip else 0], device=self.device, dtype=torch.int32)
        work = dist.all_reduce(flag, op=dist.ReduceOp.MAX, async_op=True)
        timeout_seconds = max(1.0, float(getattr(self, 'ddp_skip_sync_timeout_minutes', 10)) * 60.0)
        wait_start = time.time()
        while not work.is_completed():
            if (time.time() - wait_start) > timeout_seconds:
                raise RuntimeError(
                    f"ddp_skip_sync_timeout: rank={self.global_rank}, local_skip={bool(local_skip)}, "
                    f"timeout_min={self.ddp_skip_sync_timeout_minutes}"
                )
            time.sleep(0.2)
        work.wait()
        if debug_sync:
            waited = time.time() - sync_t0 if sync_t0 is not None else 0.0
            self._train_runtime_log('ddp_skip_sync_exit', reduced_flag=int(flag.item()), wait_seconds=float(waited))
        return bool(flag.item() > 0)

    def _should_train_runtime_print(self) -> bool:
        debug_runtime_print = bool(getattr(self.args, 'debug_runtime_print', False)) if hasattr(self, 'args') else False
        debug_runtime_interval = max(int(getattr(self.args, 'debug_runtime_interval', 20)), 1) if hasattr(self, 'args') else 20
        return debug_runtime_print and (int(self.global_step) % debug_runtime_interval == 0)

    def _train_runtime_log(self, stage: str, **kwargs):
        if not self._should_train_runtime_print():
            return
        rank = int(getattr(self, 'global_rank', -1))
        step = int(getattr(self, 'global_step', -1))
        parts = [f"[DEBUG][Train][Rank {rank}] step={step} stage={stage}"]
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    try:
                        parts.append(f"{k}={float(v.detach().float().item()):.6f}")
                    except Exception:
                        parts.append(f"{k}=tensor(shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device})")
                else:
                    parts.append(f"{k}=tensor(shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device})")
            else:
                parts.append(f"{k}={v}")
        print(' | '.join(parts), flush=True)

    def _set_vae_training_residency(self, on_gpu: bool):
        if not self.aggressive_vram_optimization:
            return
        if not hasattr(self.pipe_VAE, 'vae') or self.pipe_VAE.vae is None:
            return
        if on_gpu:
            if self._vae_on_gpu:
                return
            _move_vae_runtime(self.pipe_VAE.vae, device=self.device, dtype=self._vae_original_dtype)
            self._vae_on_gpu = True
            return
        if not self._vae_on_gpu:
            return
        _move_vae_runtime(self.pipe_VAE.vae, device='cpu', dtype=self._vae_original_dtype)
        self._vae_on_gpu = False
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    def _maybe_trim_cuda_cache(self, stage: str = 'runtime'):
        """Conservative cache trim to reduce fragmentation-driven peak VRAM.

        Trigger only on interval and only when reserved-allocated gap is large,
        to avoid unnecessary throughput regression.
        """
        if not torch.cuda.is_available():
            return
        step = int(getattr(self, 'global_step', 0))
        if step % int(self.cuda_trim_interval) != 0:
            return
        try:
            device = torch.device(self.device)
        except Exception:
            return
        try:
            reserved = torch.cuda.memory_reserved(device)
            allocated = torch.cuda.memory_allocated(device)
            frag = max(0, int(reserved) - int(allocated))
            min_frag = int(self.cuda_trim_min_fragment_mb) * 1024 * 1024
            if frag < min_frag:
                return
            gc.collect()
            torch.cuda.empty_cache()
            if self._should_train_runtime_print():
                print(
                    f"[DEBUG][Train][Rank {int(getattr(self, 'global_rank', -1))}] "
                    f"step={step} stage={stage} trim_cuda_cache frag_mb={frag / (1024.0 * 1024.0):.1f}",
                    flush=True,
                )
        except Exception:
            return

    @torch.no_grad()
    def _build_query_boxes_from_selected_indices(self, selected_indices, h_patch, w_patch):
        if isinstance(selected_indices, set):
            selected_indices = list(selected_indices)
        if not isinstance(selected_indices, (list, tuple)) or len(selected_indices) == 0:
            return {}
        per_t = {}
        total_spatial = int(h_patch) * int(w_patch)
        if total_spatial <= 0:
            return {}
        for idx in selected_indices:
            i = int(idx)
            lt = i // total_spatial
            sp = i % total_spatial
            lh = sp // int(w_patch)
            lw = sp % int(w_patch)
            if lt not in per_t:
                per_t[lt] = [float(lw), float(lh), float(lw + 1), float(lh + 1)]
            else:
                box = per_t[lt]
                box[0] = min(box[0], float(lw))
                box[1] = min(box[1], float(lh))
                box[2] = max(box[2], float(lw + 1))
                box[3] = max(box[3], float(lh + 1))
        return per_t

    @torch.no_grad()
    def _expand_query_boxes_to_flat_indices(self, per_t_boxes, h_patch, w_patch):
        if not isinstance(per_t_boxes, dict) or len(per_t_boxes) == 0:
            return []
        h_patch = int(h_patch)
        w_patch = int(w_patch)
        if h_patch <= 0 or w_patch <= 0:
            return []
        spatial = h_patch * w_patch
        out = []
        for t_key, bbox in per_t_boxes.items():
            try:
                lt = int(t_key)
            except Exception:
                continue
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
            except Exception:
                continue
            ix1 = max(0, min(w_patch, int(math.floor(x1))))
            iy1 = max(0, min(h_patch, int(math.floor(y1))))
            ix2 = max(0, min(w_patch, int(math.ceil(x2))))
            iy2 = max(0, min(h_patch, int(math.ceil(y2))))
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            base_t = int(lt) * spatial
            for yy in range(iy1, iy2):
                base = base_t + yy * w_patch
                for xx in range(ix1, ix2):
                    out.append(base + xx)
        if len(out) == 0:
            return []
        return sorted(list(set(out)))

    @torch.no_grad()
    def _build_character_probe_union_mask(self, query_role_boxes, bsz, num_frames, height, width, device, scale_h=1, scale_w=1):
        mask = torch.zeros((int(bsz), int(num_frames), int(height), int(width)), device=device, dtype=torch.bool)
        if not isinstance(query_role_boxes, dict) or len(query_role_boxes) == 0:
            return mask
        if int(num_frames) <= 0 or int(height) <= 0 or int(width) <= 0:
            return mask

        sh = max(int(scale_h), 1)
        sw = max(int(scale_w), 1)
        nf = int(num_frames)
        hh = int(height)
        ww = int(width)

        for _, per_t in query_role_boxes.items():
            if not isinstance(per_t, dict):
                continue
            for t_key, bbox in per_t.items():
                try:
                    lt = int(t_key)
                except Exception:
                    continue
                if lt < 0 or lt >= nf:
                    continue
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                except Exception:
                    continue
                x1 *= float(sw)
                x2 *= float(sw)
                y1 *= float(sh)
                y2 *= float(sh)

                ix1 = max(0, min(ww, int(math.floor(x1))))
                iy1 = max(0, min(hh, int(math.floor(y1))))
                ix2 = max(0, min(ww, int(math.ceil(x2))))
                iy2 = max(0, min(hh, int(math.ceil(y2))))
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                mask[:, lt, iy1:iy2, ix1:ix2] = True
        return mask

    def _compute_loss_with_optional_character_probe_weight(self, noise_pred, target, query_role_boxes):
        pred_f = noise_pred.float()
        tgt_f = target.float()
        base_loss = torch.nn.functional.mse_loss(pred_f, tgt_f)

        if not bool(getattr(self, 'probe_bbox_loss_weight_x2', False)):
            return base_loss, base_loss, 0.0, 0.0
        if pred_f.ndim != 5:
            return base_loss, base_loss, 0.0, 0.0
        if not isinstance(query_role_boxes, dict) or len(query_role_boxes) == 0:
            return base_loss, base_loss, 0.0, 0.0

        patch_size = getattr(self.pipe.denoising_model(), 'patch_size', (1, 1, 1))
        scale_h = int(patch_size[1]) if len(patch_size) > 1 else 1
        scale_w = int(patch_size[2]) if len(patch_size) > 2 else 1

        bsz, channels, num_frames, height, width = pred_f.shape
        bbox_mask = self._build_character_probe_union_mask(
            query_role_boxes=query_role_boxes,
            bsz=bsz,
            num_frames=num_frames,
            height=height,
            width=width,
            device=pred_f.device,
            scale_h=scale_h,
            scale_w=scale_w,
        )
        if not bool(bbox_mask.any()):
            return base_loss, base_loss, 0.0, 0.0

        mask5 = bbox_mask.unsqueeze(1).expand(-1, int(channels), -1, -1, -1)
        pred_in = pred_f.masked_select(mask5)
        tgt_in = tgt_f.masked_select(mask5)
        if pred_in.numel() == 0:
            return base_loss, base_loss, 0.0, 0.0

        inside_extra = torch.nn.functional.mse_loss(pred_in, tgt_in, reduction='sum') / float(pred_f.numel())
        weighted_loss = base_loss + inside_extra
        return weighted_loss, base_loss, float(bbox_mask.float().mean().item()), 1.0

    def _run_train_character_semantic_probe(self, noisy_latents, timestep, context, text, clip_feature=None, y=None, memory_bank_token_meta=None, probe_pipe=None):
        if not isinstance(text, str) or text.strip() == "":
            return None, None
        role_token_selection_mode = str(getattr(self.args, 'role_token_selection_mode', 'baseline')).strip().lower()

        role_ids = []
        if isinstance(memory_bank_token_meta, dict):
            for _, meta_list in memory_bank_token_meta.items():
                if not isinstance(meta_list, list):
                    continue
                for m in meta_list:
                    if isinstance(m, dict):
                        cid = str(m.get('char_id', '')).strip()
                        if cid:
                            role_ids.append(cid)
        role_ids = sorted(list(set(role_ids)))
        if len(role_ids) == 0:
            return None, None

        active_probe_pipe = probe_pipe if probe_pipe is not None else self.pipe
        if hasattr(active_probe_pipe, 'denoising_model') and callable(getattr(active_probe_pipe, 'denoising_model')):
            dit_model = active_probe_pipe.denoising_model()
        elif hasattr(active_probe_pipe, 'dit'):
            dit_model = active_probe_pipe.dit
        else:
            raise AttributeError("probe pipe must provide denoising_model() or dit")
        prev_training = bool(dit_model.training)
        use_eval_mode = bool(getattr(self.args, 'train_query_bbox_probe_use_eval_mode', True))
        use_no_grad = bool(getattr(self.args, 'train_query_bbox_probe_use_no_grad', True))
        debug_probe = bool(getattr(self.args, 'train_query_bbox_probe_debug', False))
        # context_only sparse branch only needs probe flat_idx; payload feature tensor is unnecessary.
        capture_probe_feature = False
        feature_layer_idx = int(getattr(self.args, 'feature_match_layer_idx', 7)) if capture_probe_feature else None
        feature_keep_dtype = (
            torch.bfloat16
            if (capture_probe_feature and str(getattr(self.args, 'feature_vector_dtype', 'bfloat16')).lower() == 'bfloat16')
            else torch.float16
        )

        token_pipe = self.pipe_VAE if hasattr(self, 'pipe_VAE') else active_probe_pipe
        if not (
            hasattr(token_pipe, 'prompter') and
            getattr(token_pipe.prompter, 'tokenizer', None) is not None
        ):
            token_pipe = active_probe_pipe
        has_tokenizer = bool(
            hasattr(token_pipe, 'prompter') and
            getattr(token_pipe.prompter, 'tokenizer', None) is not None
        )
        if not has_tokenizer:
            raise AttributeError("train character-semantic probe requires token_pipe.prompter.tokenizer, but tokenizer is missing")

        extract_layers = getattr(self.args, 'extract_layers', [10, 20, 30])
        if isinstance(extract_layers, str):
            extract_layers = [-1] if extract_layers.strip() == '-1' else [int(x.strip()) for x in extract_layers.split(',') if x.strip() != '']

        valid_chars = []
        for char_id in role_ids:
            character_name = char_id.replace('_', ' ').replace('-', ' ').strip()
            parts = char_id.replace('-', '_').split('_', 1)
            prefix_text = parts[0]
            suffix_text = parts[1].replace('_', ' ') if len(parts) > 1 else None

            token_ids, token_texts, _ = verify_target_text_is_single_token(token_pipe, character_name)
            if not token_ids:
                continue
            full_indices = find_token_index_in_prompt(token_pipe, text, character_name, token_ids, token_texts)
            if not full_indices:
                continue

            prefix_token_ids, _, _ = verify_target_text_is_single_token(token_pipe, prefix_text)
            num_prefix_tokens = len(prefix_token_ids) if prefix_token_ids else 0
            if suffix_text:
                suffix_token_ids, _, _ = verify_target_text_is_single_token(token_pipe, suffix_text)
                num_suffix_tokens = len(suffix_token_ids) if suffix_token_ids else 0
            else:
                num_suffix_tokens = 0

            prefix_indices = full_indices[:num_prefix_tokens]
            suffix_indices = full_indices[num_prefix_tokens:num_prefix_tokens + num_suffix_tokens] if num_suffix_tokens > 0 else []
            valid_chars.append((str(char_id), prefix_indices, suffix_indices, full_indices))

        if len(valid_chars) == 0:
            return None, None

        x_probe = noisy_latents.detach()
        t_probe = timestep.detach()
        c_probe = context.detach()
        clip_probe = clip_feature.detach() if isinstance(clip_feature, torch.Tensor) else clip_feature
        y_probe = y.detach() if isinstance(y, torch.Tensor) else y

        _, _, _, h_lat, w_lat = x_probe.shape
        patch_size = dit_model.patch_size
        h_patch = h_lat // patch_size[1]
        w_patch = w_lat // patch_size[2]

        query_role_boxes = {}
        query_role_features = {}
        try:
            if use_eval_mode:
                dit_model.eval()
            grad_ctx = torch.no_grad() if use_no_grad else suppress()
            with grad_ctx:
                char_configs = []
                for _, prefix_indices, suffix_indices, full_indices in valid_chars:
                    char_configs.append({
                        'target_token_indices': prefix_indices,
                        'suffix_token_indices': suffix_indices,
                        'all_token_indices': list(full_indices),
                        'suffix_scale': float(getattr(self.args, 'suffix_attention_scale', 1.0)),
                        'token_weight': float(getattr(self.args, 'token_weight', 1.0)),
                    })
                if role_token_selection_mode == 'layer7_single':
                    probe_layer_idx = int(getattr(self.args, 'sparse_role_memory_layer_idx', 3))
                    forward_kwargs = {}
                    if clip_probe is not None:
                        forward_kwargs['clip_feature'] = clip_probe
                    if y_probe is not None:
                        forward_kwargs['y'] = y_probe
                    per_char_step_maps, layer_tokens = _run_parallel_character_semantic_probe(
                        probe_pipe=active_probe_pipe,
                        dit_model=dit_model,
                        x=x_probe,
                        timestep=t_probe,
                        positive_context=c_probe,
                        char_configs=char_configs,
                        ordered_roles=[str(char_id) for char_id, _, _, _ in valid_chars],
                        target_layer=probe_layer_idx,
                        extra_forward_kwargs=forward_kwargs,
                        capture_feature_tokens=bool(capture_probe_feature),
                        feature_source=str(getattr(self.args, 'sparse_role_memory_feature_source', 'attn_out')),
                        feature_keep_dtype=feature_keep_dtype,
                    )
                else:
                    extractor = MultiCharacterAttentionMapExtractor(
                        active_probe_pipe,
                        extract_layers,
                        char_configs,
                        cfg_scale=1.0,
                    )

                    extractor.register_hooks()
                    feature_tap = None
                    if capture_probe_feature:
                        feature_tap = AttentionOutputFeatureTap(
                            dit_model=dit_model,
                            layer_idx=feature_layer_idx,
                            keep_device='cpu',
                            keep_dtype=feature_keep_dtype,
                            source=str(getattr(self.args, 'sparse_role_memory_feature_source', 'attn_out')),
                        )
                        feature_tap.register()

                    try:
                        if hasattr(active_probe_pipe, 'set_active_noise_domain_from_timestep'):
                            active_probe_pipe.set_active_noise_domain_from_timestep(t_probe)
                        dit_kwargs = {'x': x_probe, 'timestep': t_probe, 'context': c_probe}
                        if clip_probe is not None:
                            dit_kwargs['clip_feature'] = clip_probe
                        if y_probe is not None:
                            dit_kwargs['y'] = y_probe
                        _ = run_native_dit_forward(dit_model, **dit_kwargs)
                    finally:
                        layer_tokens = None
                        if feature_tap is not None:
                            layer_tokens = feature_tap.pop_tokens()
                            feature_tap.remove()
                        per_char_step_maps = extractor.get_attention_maps_per_character()
                        extractor.remove_hooks()
                        del extractor

                agg_maps = [None] * len(valid_chars)
                for char_idx in range(len(valid_chars)):
                    step_maps = None
                    if isinstance(per_char_step_maps, list) and char_idx < len(per_char_step_maps):
                        step_maps = per_char_step_maps[char_idx]
                    if not isinstance(step_maps, dict) or len(step_maps) == 0:
                        continue
                    step_agg = _aggregate_character_semantic_responses_cpu(step_maps)
                    if step_agg is None:
                        continue
                    agg_map = step_agg
                    if agg_map.dim() > 1:
                        agg_map = agg_map.mean(dim=0)
                    agg_maps[char_idx] = agg_map

                if _use_two_role_difference_selection(role_token_selection_mode) and len(valid_chars) == 2:
                    agg_map_0 = agg_maps[0]
                    agg_map_1 = agg_maps[1]
                    if isinstance(agg_map_0, torch.Tensor) and isinstance(agg_map_1, torch.Tensor):
                        agg_maps[0] = _suppress_other_character_response_cpu(agg_map_0, agg_map_1)
                        agg_maps[1] = _suppress_other_character_response_cpu(agg_map_1, agg_map_0)

                for char_idx, (char_id, _, _, _) in enumerate(valid_chars):
                    agg_map = agg_maps[char_idx]
                    if not isinstance(agg_map, torch.Tensor):
                        continue

                    _, _, selected_indices, _ = process_attention_map_to_mask(
                        agg_map,
                        threshold=float(getattr(self.args, 'top_visual_tokens', -1)),
                        top_k_per_head=int(getattr(self.args, 'top_visual_tokens_per_head', 0)),
                        spatial_shape=(h_patch, w_patch),
                        num_frames=max(1, int(agg_map.shape[0]) // max(1, int(h_patch) * int(w_patch))),
                        otsu_scope=str(getattr(self.args, 'otsu_scope', 'frame')),
                        neighbor_filter_kernel=int(getattr(self.args, 'neighbor_filter_kernel', 3)),
                        neighbor_filter_any_window=bool(getattr(self.args, 'neighbor_filter_any_window', True)),
                    )
                    per_t_boxes = self._build_query_boxes_from_selected_indices(selected_indices, h_patch, w_patch)
                    if len(per_t_boxes) > 0:
                        query_role_boxes[str(char_id)] = per_t_boxes
                        # Sparse role-memory query must strictly follow probe-selected token indices.
                        # BBox is kept only for role-relative RoPE coordinate construction.
                        sel_idx = []
                        if isinstance(selected_indices, set):
                            sel_idx = list(selected_indices)
                        elif isinstance(selected_indices, (list, tuple)):
                            sel_idx = list(selected_indices)
                        if len(sel_idx) > 0:
                            sel_idx = sorted(list(set([int(x) for x in sel_idx if int(x) >= 0])))
                            idx_cpu = torch.tensor(sel_idx, dtype=torch.long, device='cpu')
                            payload = {'flat_idx': idx_cpu}
                            if capture_probe_feature and isinstance(layer_tokens, torch.Tensor) and layer_tokens.dim() == 2:
                                valid = idx_cpu < int(layer_tokens.shape[0])
                                if bool(valid.any()):
                                    idx_cpu = idx_cpu[valid]
                                    payload['flat_idx'] = idx_cpu
                                    payload['feature'] = layer_tokens.index_select(0, idx_cpu.to(device=layer_tokens.device)).to(device='cpu', dtype=feature_keep_dtype)
                            query_role_features[str(char_id)] = payload

                    del agg_map

                if isinstance(per_char_step_maps, list):
                    del per_char_step_maps
                if isinstance(layer_tokens, torch.Tensor):
                    del layer_tokens

            if debug_probe:
                print(f"[CharacterSemanticProbe] query_role_boxes chars={len(query_role_boxes)} roles={sorted(list(query_role_boxes.keys()))}", flush=True)
        finally:
            if use_eval_mode and prev_training:
                dit_model.train()
        # Drop probe-only tensor refs before returning to training path.
        del x_probe, t_probe, c_probe
        if isinstance(clip_probe, torch.Tensor):
            del clip_probe
        if isinstance(y_probe, torch.Tensor):
            del y_probe
        out_boxes = query_role_boxes if len(query_role_boxes) > 0 else None
        out_features = query_role_features if len(query_role_features) > 0 else None
        return out_boxes, out_features

    def _run_train_character_semantic_probe_scoped(self, noisy_latents, timestep, context, text, clip_feature=None, y=None, memory_bank_token_meta=None):
        """Run probe in a short-lived local scope to minimize temporary tensor lifetime.

        This is a lifecycle-only optimization: it does not change probe math or
        downstream training logic.
        """
        use_detached_probe = bool(getattr(self, '_local_probe_enabled', False))
        if use_detached_probe:
            run_on_this_rank = True
            if self._local_probe_leader_only and self.tp_size > 1 and dist.is_available() and dist.is_initialized():
                run_on_this_rank = (int(self.global_rank) == int(self._probe_tp_leader_global_rank()))

            local_query_payload = (None, None)
            if run_on_this_rank:
                local_probe_pipe = self._get_or_init_local_probe_pipe()
                if local_probe_pipe is not None:
                    local_query_payload = self._run_train_character_semantic_probe(
                        noisy_latents=noisy_latents,
                        timestep=timestep,
                        context=context,
                        text=text,
                        clip_feature=clip_feature,
                        y=y,
                        memory_bank_token_meta=memory_bank_token_meta,
                        probe_pipe=local_probe_pipe,
                    )
                else:
                    local_query_payload = (None, None)

            if self._local_probe_leader_only and self.tp_size > 1 and dist.is_available() and dist.is_initialized():
                local_query_payload = self._tp_broadcast_probe_object(local_query_payload)
            return local_query_payload

        def _probe_call():
            return self._run_train_character_semantic_probe(
                noisy_latents=noisy_latents,
                timestep=timestep,
                context=context,
                text=text,
                clip_feature=clip_feature,
                y=y,
                memory_bank_token_meta=memory_bank_token_meta,
                probe_pipe=self.pipe,
            )

        query_payload = _probe_call()
        return query_payload

    def _prepare_train_image_emb_scoped(self, batch, latents, allow_missing_precomputed=False):
        """Prepare image condition embeddings in a short-lived local scope.

        This is a lifecycle optimization for optimization-stage memory usage and
        preserves existing training behavior.
        """
        image_emb = {}
        precomputed_image_emb = batch.get("precomputed_image_emb")
        precompute_enabled = bool(getattr(self.args, 'precompute_image_emb', False)) if hasattr(self, 'args') else False
        precompute_strict = bool(getattr(self.args, 'precompute_image_emb_strict', False)) if hasattr(self, 'args') else False

        if precomputed_image_emb is not None:
            for k, v in precomputed_image_emb.items():
                if isinstance(v, torch.Tensor):
                    image_emb[k] = v.to(device=self.device, dtype=self._vae_original_dtype)
                else:
                    image_emb[k] = v
            self._train_runtime_log(
                'prepare_image_emb_precomputed',
                keys=sorted(list(image_emb.keys())),
                y=image_emb.get('y') if isinstance(image_emb, dict) else None,
                clip_feature=image_emb.get('clip_feature') if isinstance(image_emb, dict) else None,
            )
            return image_emb

        if "first_ref_frames" in batch and batch["first_ref_frames"] is not None:
            if precompute_enabled and precompute_strict and (not bool(allow_missing_precomputed)):
                raise RuntimeError("precompute_image_emb strict mode enabled, but batch has no precomputed_image_emb")

            frames_tensor_list = [f[0] if f.dim() == 4 else f for f in batch["first_ref_frames"]]
            if frames_tensor_list:
                first_ref_frames = []
                for frame_tensor in frames_tensor_list:
                    frame_u8 = frame_tensor.to(torch.uint8) if frame_tensor.dtype != torch.uint8 else frame_tensor
                    frame_np = frame_u8.detach().cpu().numpy()
                    if frame_np.ndim == 3 and frame_np.shape[-1] == 3:
                        pass
                    elif frame_np.ndim == 3 and frame_np.shape[0] == 3:
                        frame_np = frame_np.transpose(1, 2, 0)
                    first_ref_frames.append(Image.fromarray(frame_np))

                rand_frame = batch.get("random_ref_frame")
                rand_tensor = rand_frame[0] if rand_frame.dim() == 4 else rand_frame
                rand_u8 = rand_tensor.to(torch.uint8) if rand_tensor.dtype != torch.uint8 else rand_tensor
                rand_ref_frame_np = rand_u8.detach().cpu().numpy()
                if rand_ref_frame_np.ndim == 3 and rand_ref_frame_np.shape[-1] == 3:
                    pass
                elif rand_ref_frame_np.shape[0] == 3:
                    rand_ref_frame_np = rand_ref_frame_np.transpose(1, 2, 0)
                rand_ref_frame = Image.fromarray(rand_ref_frame_np)

                condition_frames = first_ref_frames[:1]

                num_motion_frames = int(self.num_motion_frames)
                if self.num_overlap_frame > 0:
                    num_motion_frames = max(num_motion_frames, int(self.num_overlap_frame))
                if num_motion_frames > 1:
                    if random.random() < self.p_motion_threshold:
                        condition_frames = first_ref_frames[:num_motion_frames]
                    elif self.repeat_first_frame:
                        condition_frames = [first_ref_frames[0]] * num_motion_frames

                num_condition_frames = len(condition_frames)

                if (
                    (not self._image_encoder_on_gpu)
                    and hasattr(self.pipe_VAE, 'image_encoder')
                    and self.pipe_VAE.image_encoder is not None
                ):
                    self.pipe_VAE.image_encoder.to(device=self.device, dtype=torch.float32)
                    self._image_encoder_on_gpu = True
                self._set_vae_training_residency(on_gpu=True)

                self.pipe_VAE.device = self.device

                _, _, num_frames, height, width = latents.shape

                self._train_runtime_log('before_encode_images_adaptive', num_condition_frames=num_condition_frames, latent_shape=tuple(latents.shape), ref_count=len(condition_frames))
                image_emb = self.pipe_VAE.encode_images_adaptive(
                    condition_frames,
                    rand_ref_frame,
                    num_frames * 4 - 3, height * 8, width * 8,
                    use_first_aug=self.use_first_aug, ref_pad_cfg=self.ref_pad_cfg,
                    ref_pad_num=self.ref_pad_num,
                    num_motion_latent=self.num_motion_latent,
                )

                image_emb['num_condition_frames'] = num_condition_frames
                self._train_runtime_log('after_encode_images_adaptive', y=image_emb.get('y') if isinstance(image_emb, dict) else None, clip_feature=image_emb.get('clip_feature') if isinstance(image_emb, dict) else None)
                self._train_runtime_log(
                    'prepare_image_emb_fresh',
                    keys=sorted(list(image_emb.keys())),
                    y=image_emb.get('y') if isinstance(image_emb, dict) else None,
                    clip_feature=image_emb.get('clip_feature') if isinstance(image_emb, dict) else None,
                )

                for k in image_emb:
                    if isinstance(image_emb[k], torch.Tensor):
                        image_emb[k] = image_emb[k].to(dtype=self._vae_original_dtype)

                if (not self.keep_image_encoder_on_gpu) and hasattr(self.pipe_VAE, 'image_encoder') and self.pipe_VAE.image_encoder is not None:
                    self.pipe_VAE.image_encoder.to(device='cpu', dtype=torch.float32)
                    self._image_encoder_on_gpu = False
                self._set_vae_training_residency(on_gpu=False)

        return image_emb


    def _safe_zero_touch_term(self, p):
        """
        Return a scalar zero-valued term that keeps autograd connectivity without
        inserting dtype/device copy ops on ZeRO-sharded parameters.
        Avoid p.float()/p.to(...) here because that creates ToCopyBackward on
        possibly empty local shards under ZeRO-3.
        """
        if p is None or (not getattr(p, "requires_grad", False)):
            return None
        q = p.reshape(-1)
        if q.numel() == 0:
            return torch.zeros((), device=p.device, dtype=p.dtype)
        return q[:1].sum() * 0.0

    def _build_zero_touch_loss(self, dtype=torch.float32):
        """Deprecated: dataset-side resampling now avoids step-skip/zero-touch."""
        return torch.zeros((), device=self.device, dtype=dtype)

    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4,
                          lora_target_modules="q,k,v,o,ffn.0,ffn.2",
                          init_lora_weights="kaiming", pretrained_lora_path=None,
                          adapter_name=None):
        self.lora_alpha = lora_alpha
        use_kaiming = bool(init_lora_weights == "kaiming" or init_lora_weights is True)
        model = _inject_train_lora_modules(
            model,
            target_modules=lora_target_modules.split(","),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=use_kaiming,
            adapter_name=adapter_name,
        )
        
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)
        
        if pretrained_lora_path is not None:
            state_dict = _load_checkpoint_payload(pretrained_lora_path)
            if any(k.startswith("pipe.dit.") for k in state_dict.keys()):
                raise RuntimeError(
                    f"Old training checkpoint format is not supported anymore: {pretrained_lora_path}"
                )
            
            state_dict_new = {}
            domain_model_prefix = f"{self.train_noise_domain}_model."
            for key in state_dict:
                if ".lora_" not in key:
                    continue
                mapped_key = key.replace(".lora_A.default.weight", ".lora_A.weight").replace(".lora_B.default.weight", ".lora_B.weight")
                for prefix in (domain_model_prefix, f"pipe.{domain_model_prefix}"):
                    if mapped_key.startswith(prefix):
                        mapped_key = mapped_key[len(prefix):]
                        break
                state_dict_new[mapped_key] = state_dict[key]

            missing, unexpected = model.load_state_dict(state_dict_new, strict=False)
            print(
                f"[LoRA][Load] path={pretrained_lora_path} domain={self.train_noise_domain} "
                f"loaded_lora_keys={len(state_dict_new)} missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
            if str(getattr(self, 'train_stage', 'stage1')).strip().lower() == 'stage2' and len(state_dict_new) <= 0:
                raise RuntimeError(
                    f"Stage2 requires stage1 LoRA weights from {pretrained_lora_path}; loaded_lora_keys=0"
                )
            if (
                str(getattr(self, 'train_stage', 'stage1')).strip().lower() == 'stage2'
                and len(state_dict_new) > 0
                and len(unexpected) >= len(state_dict_new)
            ):
                raise RuntimeError(
                    f"Stage2 LoRA keys from {pretrained_lora_path} did not match the {self.train_noise_domain} model; "
                    f"loaded_lora_keys={len(state_dict_new)} unexpected={len(unexpected)}"
                )

    def _collect_active_denoising_trainable_state_dict(self):
        if not hasattr(self, "pipe") or self.pipe is None:
            return {}
        state_dict = {}
        seen_params = set()
        for model_name in ("low_noise_model", "high_noise_model"):
            model = getattr(self.pipe, model_name, None)
            if model is None:
                continue
            for name, param in model.named_parameters():
                if not getattr(param, "requires_grad", False):
                    continue
                param_id = id(param)
                if param_id in seen_params:
                    continue
                seen_params.add(param_id)
                state_dict[name] = param.detach().cpu().clone()
        return state_dict

    def _resolve_noise_domain_from_timestep(self, timestep):
        del timestep
        return str(getattr(self, 'train_noise_domain', 'low_noise')).strip().lower()

    def _is_char_attn_enabled_for_domain(self, noise_domain):
        scope = str(getattr(self, 'char_attn_noise_scope', 'low_noise')).strip().lower()
        if scope in ('high_noise', 'low_noise'):
            return scope == str(noise_domain)
        return False

    def _get_character_wise_cross_attention_for_domain(self, noise_domain):
        if not bool(getattr(self, 'enable_sparse_role_memory_attn', True)):
            return None
        domain = str(noise_domain).strip().lower()
        if domain == 'low_noise':
            return getattr(self, 'sparse_role_memory_attn_low_noise', None)
        if domain == 'high_noise':
            return getattr(self, 'sparse_role_memory_attn_high_noise', None)
        return None

    def _get_jigsaw_extra_encoder_for_domain(self, noise_domain):
        if not bool(getattr(self, 'jigsaw_extra_encoder_enabled', False)):
            return None
        domain = str(noise_domain).strip().lower()
        if domain == 'low_noise':
            return getattr(self, 'jigsaw_extra_encoder_low_noise', None)
        if domain == 'high_noise':
            return getattr(self, 'jigsaw_extra_encoder_high_noise', None)
        return getattr(self, 'jigsaw_extra_encoder', None)

    def _get_memory_writer_for_domain(self, noise_domain):
        if not bool(getattr(self, 'memory_writer_enabled', False)):
            return None
        domain = str(noise_domain).strip().lower()
        if domain == 'low_noise':
            return getattr(self, 'jigsaw_stage2_writer_low_noise', None)
        if domain == 'high_noise':
            return getattr(self, 'jigsaw_stage2_writer_high_noise', None)
        return getattr(self, 'jigsaw_stage2_writer', None)

    def _load_jigsaw_extra_encoder_state_from_payload(self, full_sd, target_noise_domain, strict=False):
        module = self._get_jigsaw_extra_encoder_for_domain(target_noise_domain)
        if module is None:
            return 0
        prefixes = [
            f"jigsaw_extra_encoder_{target_noise_domain}.",
            "jigsaw_extra_encoder.",
        ]
        enc_sd = {}
        for prefix in prefixes:
            enc_sd = _jigsaw_extract_prefixed_state_dict(full_sd, prefix)
            if enc_sd:
                break
        if not enc_sd:
            if strict:
                raise RuntimeError(f"No SlotMem memory encoder keys found for domain {target_noise_domain}")
            return 0
        missing, unexpected = module.load_state_dict(enc_sd, strict=False)
        print(
            f"[MemoryEncoder][Load] domain={target_noise_domain} loaded_keys={len(enc_sd)} "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
        if strict and (missing or unexpected):
            raise RuntimeError(f"Strict SlotMem memory encoder load failed: missing={missing}, unexpected={unexpected}")
        return int(len(enc_sd))

    def _load_sparse_role_memory_state_from_payload(self, full_sd, target_noise_domain, strict=False):
        module = self._get_character_wise_cross_attention_for_domain(target_noise_domain)
        if module is None:
            return 0
        prefixes = [
            f"sparse_role_memory_attn_{target_noise_domain}.",
            "sparse_role_memory_attn.",
        ]
        sparse_sd = {}
        for prefix in prefixes:
            sparse_sd = _jigsaw_extract_prefixed_state_dict(full_sd, prefix)
            if sparse_sd:
                break
        if not sparse_sd:
            if strict:
                raise RuntimeError(f"No sparse role memory attention keys found for domain {target_noise_domain}")
            return 0
        missing, unexpected = module.load_state_dict(sparse_sd, strict=False)
        print(
            f"[CharacterWiseCrossAttention][Load] domain={target_noise_domain} loaded_keys={len(sparse_sd)} "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
        if strict and (missing or unexpected):
            raise RuntimeError(f"Strict sparse role memory load failed: missing={missing}, unexpected={unexpected}")
        return int(len(sparse_sd))

    def _freeze_stage1_memory_modules_for_stage2(self):
        if str(getattr(self, 'train_stage', 'stage1')).strip().lower() != 'stage2':
            return
        sparse_touched = []
        encoder_touched = []

        def _set_modules_trainable(attrs, trainable, touched):
            for attr in attrs:
                module = getattr(self, attr, None)
                if module is None or any(module is m for _, m in touched):
                    continue
                module.requires_grad_(bool(trainable))
                touched.append((attr, module))

        _set_modules_trainable((
            'sparse_role_memory_attn_low_noise',
            'sparse_role_memory_attn_high_noise',
            'sparse_role_memory_attn',
        ), bool(getattr(self, 'stage2_unfreeze_char_attn', True)), sparse_touched)
        _set_modules_trainable((
            'jigsaw_extra_encoder_low_noise',
            'jigsaw_extra_encoder_high_noise',
            'jigsaw_extra_encoder',
        ), bool(getattr(self, 'stage2_unfreeze_memory_encoder', True)), encoder_touched)
        self.jigsaw_extra_encoder_aux_weight = 0.0
        print(
            "[Stage2][MemoryTrainability] "
            f"char_attn_trainable={bool(getattr(self, 'stage2_unfreeze_char_attn', True))} "
            f"char_attn_lr={float(getattr(self, 'stage2_char_attn_lr', 1e-5)):.6g} "
            f"memory_encoder_trainable={bool(getattr(self, 'stage2_unfreeze_memory_encoder', True))} "
            f"memory_encoder_lr={float(getattr(self, 'stage2_memory_encoder_lr', 1e-5)):.6g} "
            f"stage1_branch_prob={float(getattr(self, 'stage2_stage1_branch_prob', 0.2)):.3f} "
            "aux_weight=0.0 "
            "char_attn_modules=" + ",".join(name for name, _ in sparse_touched)
            + " memory_encoder_modules=" + ",".join(name for name, _ in encoder_touched),
            flush=True,
        )

    def _load_memory_writer_state_from_payload(self, full_sd, target_noise_domain, strict=False):
        module = self._get_memory_writer_for_domain(target_noise_domain)
        if module is None:
            return 0
        prefixes = [
            f"jigsaw_stage2_writer_{target_noise_domain}.",
            "jigsaw_stage2_writer.",
        ]
        writer_sd = {}
        for prefix in prefixes:
            writer_sd = _jigsaw_extract_prefixed_state_dict(full_sd, prefix)
            if writer_sd:
                break
        if not writer_sd:
            if strict:
                raise RuntimeError(f"No jigsaw stage2 writer keys found for domain {target_noise_domain}")
            return 0
        missing, unexpected = module.load_state_dict(writer_sd, strict=False)
        print(
            f"[MemoryWriter][Load] domain={target_noise_domain} loaded_keys={len(writer_sd)} "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
        if strict and (missing or unexpected):
            raise RuntimeError(f"Strict jigsaw stage2 writer load failed: missing={missing}, unexpected={unexpected}")
        return int(len(writer_sd))

    def _set_active_lora_adapter(self, model, adapter_name):
        if model is None or not adapter_name:
            return False
        changed = False
        for module in model.modules():
            if hasattr(module, 'set_adapter'):
                with suppress(Exception):
                    module.set_adapter(adapter_name)
                    changed = True
            elif hasattr(module, 'lora_adapter_name'):
                with suppress(Exception):
                    module.disable_adapters = (str(getattr(module, 'lora_adapter_name', '')) != str(adapter_name))
                    changed = True
        return changed

    def _set_active_lora_for_noise_domain(self, noise_domain):
        if not bool(getattr(self, 'dual_noise_lora_enabled', False)):
            return
        domain = str(noise_domain).strip().lower()
        adapter_name = self.high_noise_lora_adapter_name if domain == 'high_noise' else self.low_noise_lora_adapter_name
        model = self.pipe.denoising_model()
        changed = self._set_active_lora_adapter(model, adapter_name)
        if changed:
            self._active_noise_lora_adapter = adapter_name
    
    def _memory_aware_dit_forward(
        self,
        noisy_latents,
        memory_feature_tokens_selected,  # [N, 5120] selected attn-out memory features
        timestep,
        context,
        memory_feature_bank_tokens_selected=None,
        memory_bank_percents=None,
        memory_feature_bank_token_meta_selected=None,
        stage2_update_memory_feature_bank_tokens_selected=None,
        stage2_update_memory_feature_bank_token_meta_selected=None,
        query_role_boxes=None,
        query_feature_payload=None,
        force_disable_injection=False,
        clip_feature=None,
        y=None,
        memory_token_lengths_per_character=None,  # list of int, length per character for segment embed
        allow_sparse_role_memory=True,
        character_wise_cross_attention=None,
        allow_jigsaw_stage2_writer=True,
        **kwargs
    ):
        """Wan2.2-native MemoryAwareDiTForward with optional SlotMem conditioning."""
        self._last_v9_fusion_meta = {
            'selected_bank_idx': None,
            'selected_bank_percent': None,
            'p_cur': None,
            'p_fusion': None,
            'outside_bank_blocked': 0.0,
            'fusion_alpha': None,
            'fusion_quantile': None,
            'fusion_max_inject_ratio': None,
            'fusion_inject_ratio': None,
            'fusion_sim1_mean': None,
            'fusion_tau_sim': None,
            'fusion_relrope_enabled': None,
            'fusion_relrope_query_valid_ratio': None,
            'fusion_relrope_memory_valid_ratio': None,
            'memory_bank_count': None,
            'memory_bank_percents': None,
        }
        self.pipe.set_active_noise_domain_from_timestep(timestep, self.noise_domain_boundary)
        dit_model = self.pipe.denoising_model()
        active_character_wise_cross_attention = character_wise_cross_attention
        if active_character_wise_cross_attention is None:
            active_character_wise_cross_attention = getattr(self, 'sparse_role_memory_attn', None)
        active_jigsaw_extra_encoder = self._get_jigsaw_extra_encoder_for_domain(
            getattr(self.pipe, "current_noise_domain", getattr(self, "train_noise_domain", "low_noise"))
        ) if bool(getattr(self, 'jigsaw_extra_encoder_enabled', False)) else None
        active_jigsaw_stage2_writer = self._get_memory_writer_for_domain(
            getattr(self.pipe, "current_noise_domain", getattr(self, "train_noise_domain", "low_noise"))
        ) if (bool(allow_jigsaw_stage2_writer) and bool(getattr(self, 'memory_writer_enabled', False))) else None
        jigsaw_aux_losses = []
        jigsaw_aux_stats_last = {'enabled': 0.0}
        jigsaw_stage2_writer_stats_last = {'enabled': 0.0, 'input_slots': 0, 'updated_slots': 0}
        enable_sparse_context_only = (
            bool(allow_sparse_role_memory)
            and bool(getattr(self, 'enable_sparse_role_memory_attn', True))
            and active_character_wise_cross_attention is not None
        )
        step_force_disable_injection = bool(force_disable_injection)
        model_dtype = dit_model.patch_embedding.weight.dtype
        device = noisy_latents.device
        noisy_latents = noisy_latents.to(device=device, dtype=model_dtype)
        context = context.to(device=device, dtype=model_dtype)
        if y is None and bool(getattr(dit_model, 'has_image_input', False)):
            expected_in = int(getattr(dit_model, 'in_dim', noisy_latents.shape[1]))
            raise RuntimeError(
                f"_memory_aware_dit_forward missing y for image-input model: "
                f"noisy_latents_shape={tuple(noisy_latents.shape)}, noisy_channels={int(noisy_latents.shape[1])}, "
                f"expected_in_dim={expected_in}, clip_feature_is_none={clip_feature is None}"
            )
        if y is not None:
            y = y.to(device=device, dtype=model_dtype)
        if clip_feature is not None and isinstance(clip_feature, torch.Tensor):
            clip_feature = clip_feature.to(device=device, dtype=model_dtype)

        t_input = timestep
        if t_input.dim() > 1:
            t_input = t_input.reshape(t_input.shape[0], -1)[:, 0]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            t_embed = dit_model.time_embedding(
                sinusoidal_embedding_1d(dit_model.freq_dim, t_input)
                .float()
                .to(device=device)
            )
            t_mod = dit_model.time_projection(t_embed).unflatten(1, (6, dit_model.dim))
            memory_encoder_t_embed = t_embed
        if t_mod.dtype != model_dtype:
            t_mod = t_mod.to(dtype=model_dtype)

        memory_bank_embeddings = memory_feature_bank_tokens_selected
        memory_bank_token_meta = memory_feature_bank_token_meta_selected
        stage2_update_memory_bank_embeddings = stage2_update_memory_feature_bank_tokens_selected
        stage2_update_memory_bank_token_meta = stage2_update_memory_feature_bank_token_meta_selected
        memory_bank_mode = str(getattr(self, 'jigsaw_memory_bank_mode', 'single')).strip().lower()
        candidate_percents = []
        if isinstance(memory_bank_percents, torch.Tensor):
            candidate_percents = memory_bank_percents.detach().float().cpu().tolist()
        elif isinstance(memory_bank_percents, np.ndarray):
            candidate_percents = memory_bank_percents.astype(np.float32).tolist()
        elif isinstance(memory_bank_percents, (list, tuple)):
            for x in memory_bank_percents:
                try:
                    candidate_percents.append(float(x))
                except Exception:
                    pass
        elif memory_bank_percents is not None:
            candidate_percents = parse_float_csv(memory_bank_percents, default_list=[])
        if len(candidate_percents) == 0 and memory_bank_mode == 'legacy_multi':
            candidate_percents = [float(x) for x in self.memory_bank_percents]
        layerwise_memory_banks = _pm_is_layerwise_container(memory_bank_embeddings)
        layerwise_stage2_update_banks = _pm_is_layerwise_container(stage2_update_memory_bank_embeddings)
        layerwise_sparse_payload = bool(layerwise_memory_banks)
        if layerwise_memory_banks:
            layer_bank_count = 0
            for _, bank_map in _pm_iter_layerwise_items(memory_bank_embeddings):
                if isinstance(bank_map, dict):
                    layer_bank_count = max(layer_bank_count, len(bank_map))
            self._last_v9_fusion_meta['memory_bank_count'] = int(layer_bank_count)
        else:
            self._last_v9_fusion_meta['memory_bank_count'] = len(memory_bank_embeddings) if isinstance(memory_bank_embeddings, dict) else 0
        self._last_v9_fusion_meta['memory_bank_percents'] = [float(x) for x in candidate_percents] if candidate_percents else None

        selected_bank_percent = None
        selected_bank_token_meta = None
        p_cur_for_bank = float((timestep.detach().float() / float(max(int(self.pipe.scheduler.num_train_timesteps), 1))).clamp(0.0, 1.0).mean().item())
        self._last_v9_fusion_meta['p_cur'] = p_cur_for_bank
        nearest_bank_idx = None
        if memory_bank_mode == 'legacy_multi' and candidate_percents:
            nearest_bank_idx = int(pick_nearest_bank_by_percent(p_cur_for_bank, candidate_percents))
            if 0 <= nearest_bank_idx < len(candidate_percents):
                selected_bank_percent = float(candidate_percents[nearest_bank_idx])
        elif memory_bank_mode != 'legacy_multi' and len(candidate_percents) == 1:
            selected_bank_percent = float(candidate_percents[0])
        if layerwise_memory_banks:
            selected_bank_idx = 0 if nearest_bank_idx is None else nearest_bank_idx
        elif isinstance(memory_bank_embeddings, dict) and len(memory_bank_embeddings) > 0:
            selected_bank_idx = 0 if nearest_bank_idx is None else nearest_bank_idx
            selected_bank_key = str(selected_bank_idx)
            if memory_bank_mode != 'legacy_multi':
                if '0' in memory_bank_embeddings:
                    selected_bank_key = '0'
                elif len(memory_bank_embeddings) == 1:
                    selected_bank_key = next(iter(memory_bank_embeddings.keys()))
                else:
                    selected_bank_key = '0'
                selected_bank_idx = int(selected_bank_key) if str(selected_bank_key).isdigit() else 0
            selected_tokens = memory_bank_embeddings.get(selected_bank_key, None)
            if isinstance(memory_bank_token_meta, dict):
                selected_bank_token_meta = memory_bank_token_meta.get(selected_bank_key, None)
            if selected_tokens is None and memory_bank_mode == 'legacy_multi':
                try:
                    fallback_key = next(iter(memory_bank_embeddings.keys()))
                    selected_tokens = memory_bank_embeddings[fallback_key]
                    if isinstance(memory_bank_token_meta, dict):
                        selected_bank_token_meta = memory_bank_token_meta.get(str(fallback_key), None)
                    if str(fallback_key).isdigit():
                        selected_bank_idx = int(fallback_key)
                except Exception:
                    selected_tokens = None
            if not (isinstance(selected_tokens, torch.Tensor) and selected_tokens.ndim >= 2 and int(selected_tokens.shape[0]) > 0):
                print(f"[Warning] selected memory bank {selected_bank_idx} has no valid tokens; disable memory injection this step", flush=True)
                selected_tokens = memory_feature_tokens_selected[:0]
                step_force_disable_injection = True
            memory_feature_tokens_selected = selected_tokens
        else:
            selected_bank_idx = 0

        def _collect_active_probe_roles(payload):
            active = []
            if not isinstance(payload, dict):
                return active
            for role_id, role_payload in payload.items():
                if not isinstance(role_payload, dict):
                    continue
                flat_idx = role_payload.get('flat_idx', None)
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
                    if isinstance(item, dict) and str(item.get('char_id', '')).strip() in active_roles_set:
                        keep.append(i)
                if len(keep) == 0:
                    return tokens[:0], [], []
                keep_idx = torch.tensor(keep, device=tokens.device, dtype=torch.long)
                pruned_tokens = tokens.index_select(0, keep_idx)
                pruned_meta = [token_meta[i] for i in keep]
                role_count = defaultdict(int)
                role_order = []
                for item in pruned_meta:
                    rid = str(item.get('char_id', '')).strip()
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
            memory_feature_tokens_selected, selected_bank_token_meta, memory_token_lengths_per_character = _prefilter_memory_bank_for_active_roles(
                tokens=memory_feature_tokens_selected,
                token_meta=selected_bank_token_meta,
                token_lengths=memory_token_lengths_per_character,
                active_roles=active_roles,
                role_boxes=query_role_boxes,
                role_order_hint=(list(query_feature_payload.keys()) if isinstance(query_feature_payload, dict) else None),
            )
            if (not isinstance(memory_feature_tokens_selected, torch.Tensor)) or int(memory_feature_tokens_selected.shape[0]) <= 0:
                step_force_disable_injection = True
                if not isinstance(query_feature_payload, dict):
                    query_feature_payload = {}

        self._last_v9_fusion_meta['selected_bank_idx'] = int(selected_bank_idx)
        if selected_bank_percent is not None:
            self._last_v9_fusion_meta['selected_bank_percent'] = float(selected_bank_percent)

        if layerwise_sparse_payload:
            selected_bank_token_meta = None
        else:
            if active_jigsaw_extra_encoder is not None and isinstance(memory_feature_tokens_selected, torch.Tensor) and memory_feature_tokens_selected.ndim >= 2 and int(memory_feature_tokens_selected.shape[0]) > 0:
                memory_feature_tokens_selected, selected_bank_token_meta, memory_token_lengths_per_character, encode_stats = _memory_encode_role_tokens_to_slots(
                    active_jigsaw_extra_encoder,
                    memory_feature_tokens_selected.to(device=device, dtype=model_dtype),
                    selected_bank_token_meta if isinstance(selected_bank_token_meta, list) else [],
                    layer_idx=0,
                    t_embed=memory_encoder_t_embed,
                )
                jigsaw_aux_stats_last.update({
                    'enabled': 1.0,
                    'layer': 0.0,
                    'group': float(encode_stats.get('group', 0)),
                    'input_tokens': float(encode_stats.get('input_tokens', 0)),
                    'output_slots': float(encode_stats.get('output_slots', 0)),
                    'slots_per_role': float(encode_stats.get('slots_per_role', 0)),
                })
            memory_feature_tokens_selected = memory_feature_tokens_selected.to(dtype=model_dtype, device=device)
        if y is not None:
            x_with_y = torch.cat([noisy_latents, y], dim=1)
        else:
            x_with_y = noisy_latents
        x_patched, (f, h, w) = dit_model.patchify(x_with_y)
        batch_size, seq_len, _ = x_patched.shape
        seq_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)
        grid_sizes = torch.tensor([[f, h, w]] * batch_size, device=device, dtype=torch.long)

        def _role_payload_from_slot_meta(slot_meta, max_tokens):
            payload = {}
            if not isinstance(slot_meta, list) or int(max_tokens) <= 0:
                return payload
            role_to_indices = defaultdict(list)
            for i, item in enumerate(slot_meta[:int(max_tokens)]):
                if not isinstance(item, dict):
                    continue
                rid = str(item.get('char_id', '')).strip()
                if rid:
                    role_to_indices[rid].append(int(i))
            for rid, indices in role_to_indices.items():
                if indices:
                    payload[str(rid)] = {
                        'flat_idx': torch.tensor(indices, device=device, dtype=torch.long),
                    }
            return payload

        if layerwise_sparse_payload:
            memory_projected = None
        else:
            if (
                active_jigsaw_stage2_writer is not None
                and active_jigsaw_extra_encoder is not None
                and isinstance(memory_feature_tokens_selected, torch.Tensor)
                and memory_feature_tokens_selected.ndim >= 2
                and int(memory_feature_tokens_selected.shape[0]) > 0
            ):
                update_selected_mem = None
                update_selected_meta = None
                if isinstance(stage2_update_memory_bank_embeddings, dict) and len(stage2_update_memory_bank_embeddings) > 0:
                    update_key = '0' if '0' in stage2_update_memory_bank_embeddings else next(iter(stage2_update_memory_bank_embeddings.keys()))
                    update_selected_mem = stage2_update_memory_bank_embeddings.get(update_key, None)
                    if isinstance(stage2_update_memory_bank_token_meta, dict):
                        update_selected_meta = stage2_update_memory_bank_token_meta.get(update_key, None)
                if isinstance(update_selected_mem, torch.Tensor) and update_selected_mem.ndim >= 2 and int(update_selected_mem.shape[0]) > 0:
                    update_slots, update_meta, _update_lengths, _update_encode_stats = _memory_encode_role_tokens_to_slots(
                        active_jigsaw_extra_encoder,
                        update_selected_mem.to(device=device, dtype=model_dtype),
                        list(update_selected_meta or []) if isinstance(update_selected_meta, list) else [],
                        layer_idx=0,
                        t_embed=memory_encoder_t_embed,
                    )
                    writer_update_payload = _role_payload_from_slot_meta(update_meta, int(update_slots.shape[0]))
                    memory_feature_tokens_selected, writer_stats = active_jigsaw_stage2_writer(
                        memory_feature_tokens_selected,
                        selected_bank_token_meta if isinstance(selected_bank_token_meta, list) else [],
                        writer_update_payload,
                        update_slots.unsqueeze(0),
                    )
                else:
                    writer_stats = active_jigsaw_stage2_writer._stats(
                        enabled=1.0,
                        input_slots=int(memory_feature_tokens_selected.shape[0]),
                    )
                jigsaw_stage2_writer_stats_last.update({
                    'enabled': float(writer_stats.get('enabled', 0.0)),
                    'layer': 0.0,
                    'input_slots': float(writer_stats.get('input_slots', 0.0)),
                    'updated_slots': float(writer_stats.get('updated_slots', 0.0)),
                    'mean_gate': float(writer_stats.get('mean_gate', 0.0)),
                    'mean_cos': float(writer_stats.get('mean_cos', 0.0)),
                    'clipped_ratio': float(writer_stats.get('clipped_ratio', 0.0)),
                })
            mem_input = memory_feature_tokens_selected.unsqueeze(0).expand(batch_size, -1, -1)
            if enable_sparse_context_only and int(mem_input.shape[1]) <= 0:
                memory_projected = mem_input
            elif self.use_projector and hasattr(self, 'memory_projector') and self.memory_projector is not None:
                target_dtype = self.memory_projector.input_proj.weight.dtype
                t_for_projector = t_embed.mean(dim=1).to(dtype=target_dtype)
                mem_input = mem_input.to(dtype=target_dtype)
                noisy_latents_for_projector = noisy_latents.to(dtype=target_dtype)
                num_train_timesteps = self.pipe.scheduler.num_train_timesteps if hasattr(self.pipe.scheduler, 'num_train_timesteps') else 1000
                memory_projected = self.memory_projector(
                    mem_input,
                    t_for_projector,
                    noisy_latents_for_projector,
                    condition_mask=None,
                    timestep=timestep,
                    num_train_timesteps=num_train_timesteps,
                )
            else:
                memory_projected = mem_input.to(dtype=model_dtype)
            if self.memory_embeddings is not None:
                memory_projected = self.memory_embeddings(
                    memory_projected,
                    memory_token_lengths_per_character=memory_token_lengths_per_character,
                    add_segment=bool(self.use_segment_embed),
                    add_pos=bool(self.use_learnable_memory_pos),
                )

        n_mem = int(memory_projected.shape[1]) if isinstance(memory_projected, torch.Tensor) else 0
        if self.training and self.global_step % 10 == 0 and n_mem > 0:
            with torch.no_grad():
                mem_norm = memory_projected.detach().norm(p=2, dim=-1).mean()
                vid_norm = x_patched.detach().norm(p=2, dim=-1).mean()
                self.log("monitor/mem_norm", mem_norm, on_step=True, prog_bar=True, logger=True)
                self.log("monitor/vid_norm", vid_norm, on_step=True, logger=True)
                self.log("monitor/mem_vid_ratio", mem_norm / (vid_norm + 1e-6), on_step=True, logger=True)

        memory_tokens_for_sparse = memory_projected

        debug_runtime_print = bool(getattr(self.args, 'debug_runtime_print', False))
        debug_runtime_interval = max(int(getattr(self.args, 'debug_runtime_interval', 20)), 1)
        should_debug_print = debug_runtime_print and (int(self.global_step) % debug_runtime_interval == 0)
        if should_debug_print:
            with torch.no_grad():
                mem_mean = memory_projected.detach().mean().item() if n_mem > 0 else 0.0
                mem_std = memory_projected.detach().std().item() if n_mem > 0 else 0.0
                mem_norm = memory_projected.detach().norm(p=2, dim=-1).mean().item() if n_mem > 0 else 0.0
                print(
                    f"[DEBUG][Train][Step {int(self.global_step)}] mem_shape={tuple(memory_projected.shape)} "
                    f"mem_mean={mem_mean:.6f} mem_std={mem_std:.6f} mem_norm={mem_norm:.6f}",
                    flush=True,
                )

        freqs = dit_model.freqs
        if hasattr(dit_model, '_expand_freqs'):
            freqs = dit_model._expand_freqs(grid_sizes, device)
        elif freqs.device != device:
            freqs = freqs.to(device)
        context_emb = dit_model.text_embedding(context)
        if (
            getattr(dit_model, 'has_image_input', False)
            and clip_feature is not None
            and getattr(dit_model, 'img_emb', None) is not None
            and bool(getattr(dit_model, 'require_clip_embedding', True))
        ):
            clip_feature = clip_feature.to(device=device, dtype=model_dtype)
            clip_emb = dit_model.img_emb(clip_feature)
            context_emb = torch.cat([clip_emb, context_emb], dim=1)
        context_lens = torch.full((batch_size,), context_emb.shape[1], device=device, dtype=torch.long)
        x_output = x_patched

        attn_stats = {}
        hooks = []
        def get_attn_hook(layer_idx):
            def hook(module, input, output):
                with torch.no_grad():
                    attn_stats[layer_idx] = (0.0, output.detach().abs().mean().item())
            return hook
        if self.enable_attn_monitor and self.training and self.global_step % 50 == 0:
            for i, block in enumerate(dit_model.blocks):
                if hasattr(block, 'self_attn'):
                    hooks.append(block.self_attn.register_forward_hook(get_attn_hook(i)))

        self._last_sparse_role_memory_stats = {
            'enabled': 0.0,
            'selected_query_tokens': 0,
            'selected_memory_tokens': 0,
            'winner_counts': {},
            'role_head_out_norm': 0.0,
            'plain_head_out_norm': 0.0,
            'attn_entropy': 0.0,
        }
        sparse_layer_indices = set(
            int(x) for x in getattr(self, 'sparse_role_memory_injection_layers', [getattr(self, 'sparse_role_memory_layer_idx', 3)])
            if int(x) >= 0
        )
        if len(sparse_layer_indices) == 0:
            sparse_layer_indices = {int(getattr(self, 'sparse_role_memory_layer_idx', 3))}
        last_sparse_layer_idx = max(sparse_layer_indices)
        sparse_timestep_percent = float((timestep.detach().float() / float(max(int(self.pipe.scheduler.num_train_timesteps), 1))).clamp(0.0, 1.0).mean().item())

        def _select_bank_from_map(bank_map, meta_map):
            layer_selected_mem = None
            layer_selected_meta = None
            if isinstance(bank_map, dict):
                selected_layer_bank_key = str(selected_bank_idx)
                if memory_bank_mode != 'legacy_multi':
                    if '0' in bank_map:
                        selected_layer_bank_key = '0'
                    elif len(bank_map) == 1:
                        selected_layer_bank_key = next(iter(bank_map.keys()))
                    else:
                        selected_layer_bank_key = '0'
                layer_selected_mem = bank_map.get(selected_layer_bank_key, None)
                if isinstance(meta_map, dict):
                    layer_selected_meta = meta_map.get(selected_layer_bank_key, None)
                if layer_selected_mem is None and memory_bank_mode == 'legacy_multi':
                    for fallback_key, fallback_value in bank_map.items():
                        if isinstance(fallback_value, torch.Tensor):
                            layer_selected_mem = fallback_value
                            if isinstance(meta_map, dict):
                                layer_selected_meta = meta_map.get(str(fallback_key), None)
                            break
            return layer_selected_mem, layer_selected_meta

        def _role_payload_from_slot_meta(slot_meta, max_tokens):
            payload = {}
            if not isinstance(slot_meta, list) or int(max_tokens) <= 0:
                return payload
            role_to_indices = defaultdict(list)
            for i, item in enumerate(slot_meta[:int(max_tokens)]):
                if not isinstance(item, dict):
                    continue
                rid = str(item.get('char_id', '')).strip()
                if rid:
                    role_to_indices[rid].append(int(i))
            for rid, indices in role_to_indices.items():
                if indices:
                    payload[str(rid)] = {
                        'flat_idx': torch.tensor(indices, device=device, dtype=torch.long),
                    }
            return payload

        def _project_memory_for_sparse(layer_mem, layer_lengths):
            if not (isinstance(layer_mem, torch.Tensor) and layer_mem.ndim >= 2 and int(layer_mem.shape[0]) > 0):
                return None
            layer_mem = layer_mem.to(device=device, dtype=model_dtype)
            layer_mem_input = layer_mem.unsqueeze(0).expand(batch_size, -1, -1)
            if self.use_projector and hasattr(self, 'memory_projector') and self.memory_projector is not None:
                target_dtype = self.memory_projector.input_proj.weight.dtype
                t_for_projector = t_embed.mean(dim=1).to(dtype=target_dtype)
                layer_mem_input = layer_mem_input.to(dtype=target_dtype)
                noisy_latents_for_projector = noisy_latents.to(dtype=target_dtype)
                num_train_timesteps = self.pipe.scheduler.num_train_timesteps if hasattr(self.pipe.scheduler, 'num_train_timesteps') else 1000
                layer_projected = self.memory_projector(
                    layer_mem_input,
                    t_for_projector,
                    noisy_latents_for_projector,
                    condition_mask=None,
                    timestep=timestep,
                    num_train_timesteps=num_train_timesteps,
                ).to(dtype=model_dtype)
            else:
                layer_projected = layer_mem_input.to(dtype=model_dtype)
            if self.memory_embeddings is not None:
                layer_projected = self.memory_embeddings(
                    layer_projected,
                    memory_token_lengths_per_character=layer_lengths,
                    add_segment=bool(self.use_segment_embed),
                    add_pos=bool(self.use_learnable_memory_pos),
                )
            return layer_projected

        def _prepare_sparse_inputs_for_layer(layer_idx, current_x_output=None):
            layer_query_payload = query_feature_payload
            if _pm_is_layerwise_container(query_feature_payload):
                layer_query_payload = _pm_select_layerwise_value(query_feature_payload, layer_idx, default={})
                if not isinstance(layer_query_payload, dict):
                    layer_query_payload = {}
            layer_selected_mem = memory_feature_tokens_selected
            layer_selected_meta = selected_bank_token_meta
            layer_lengths = memory_token_lengths_per_character
            if layerwise_memory_banks:
                layer_bank_map = _pm_select_layerwise_value(memory_bank_embeddings, layer_idx, default={})
                layer_meta_map = _pm_select_layerwise_value(memory_bank_token_meta, layer_idx, default={})
                layer_selected_mem, layer_selected_meta = _select_bank_from_map(layer_bank_map, layer_meta_map)
                layer_lengths = None
            active_roles = _collect_active_probe_roles(layer_query_payload) if enable_sparse_context_only else []
            if active_jigsaw_extra_encoder is not None and isinstance(layer_selected_mem, torch.Tensor) and layer_selected_mem.ndim >= 2 and int(layer_selected_mem.shape[0]) > 0:
                raw_mem_for_aux = layer_selected_mem.to(device=device, dtype=model_dtype)
                raw_meta_for_aux = list(layer_selected_meta or []) if isinstance(layer_selected_meta, list) else []
                layer_selected_mem, layer_selected_meta, layer_lengths, encode_stats = _memory_encode_role_tokens_to_slots(
                    active_jigsaw_extra_encoder,
                    raw_mem_for_aux,
                    raw_meta_for_aux,
                    layer_idx=layer_idx,
                    t_embed=memory_encoder_t_embed,
                )
                if self.training and float(getattr(self, 'jigsaw_extra_encoder_aux_weight', 0.0)) > 0.0:
                    aux_loss, aux_stats = _memory_encoder_contrastive_loss(
                        active_jigsaw_extra_encoder,
                        layer_selected_mem,
                        layer_selected_meta,
                        layer_idx=layer_idx,
                        x_output=current_x_output,
                        query_feature_payload=layer_query_payload,
                        bg_tokens=int(getattr(self, 'jigsaw_extra_encoder_bg_tokens', 64)),
                        t_embed=memory_encoder_t_embed,
                    )
                    if isinstance(aux_loss, torch.Tensor):
                        jigsaw_aux_losses.append(aux_loss)
                    jigsaw_aux_stats_last.update(aux_stats)
                jigsaw_aux_stats_last.update({
                    'enabled': 1.0,
                    'layer': float(int(layer_idx)),
                    'group': float(encode_stats.get('group', 0)),
                    'input_tokens': float(encode_stats.get('input_tokens', 0)),
                    'output_slots': float(encode_stats.get('output_slots', 0)),
                    'slots_per_role': float(encode_stats.get('slots_per_role', 0)),
                })
                if active_jigsaw_stage2_writer is not None:
                    update_selected_mem = None
                    update_selected_meta = None
                    if layerwise_stage2_update_banks:
                        update_bank_map = _pm_select_layerwise_value(stage2_update_memory_bank_embeddings, layer_idx, default={})
                        update_meta_map = _pm_select_layerwise_value(stage2_update_memory_bank_token_meta, layer_idx, default={})
                        update_selected_mem, update_selected_meta = _select_bank_from_map(update_bank_map, update_meta_map)
                    elif isinstance(stage2_update_memory_bank_embeddings, dict) and len(stage2_update_memory_bank_embeddings) > 0:
                        update_selected_mem, update_selected_meta = _select_bank_from_map(
                            stage2_update_memory_bank_embeddings,
                            stage2_update_memory_bank_token_meta,
                        )
                    if isinstance(update_selected_mem, torch.Tensor) and update_selected_mem.ndim >= 2 and int(update_selected_mem.shape[0]) > 0:
                        update_slots, update_meta, _update_lengths, _update_encode_stats = _memory_encode_role_tokens_to_slots(
                            active_jigsaw_extra_encoder,
                            update_selected_mem.to(device=device, dtype=model_dtype),
                            list(update_selected_meta or []) if isinstance(update_selected_meta, list) else [],
                            layer_idx=layer_idx,
                            t_embed=memory_encoder_t_embed,
                        )
                        writer_update_payload = _role_payload_from_slot_meta(update_meta, int(update_slots.shape[0]))
                        layer_selected_mem, writer_stats = active_jigsaw_stage2_writer(
                            layer_selected_mem,
                            layer_selected_meta,
                            writer_update_payload,
                            update_slots.unsqueeze(0),
                        )
                    else:
                        writer_stats = active_jigsaw_stage2_writer._stats(
                            enabled=1.0,
                            input_slots=int(layer_selected_mem.shape[0]) if isinstance(layer_selected_mem, torch.Tensor) and layer_selected_mem.ndim >= 2 else 0,
                        )
                    jigsaw_stage2_writer_stats_last.update({
                        'enabled': float(writer_stats.get('enabled', 0.0)),
                        'layer': float(int(layer_idx)),
                        'input_slots': float(writer_stats.get('input_slots', 0.0)),
                        'updated_slots': float(writer_stats.get('updated_slots', 0.0)),
                        'mean_gate': float(writer_stats.get('mean_gate', 0.0)),
                        'mean_cos': float(writer_stats.get('mean_cos', 0.0)),
                        'clipped_ratio': float(writer_stats.get('clipped_ratio', 0.0)),
                    })
            if enable_sparse_context_only:
                layer_selected_mem, layer_selected_meta, layer_lengths = _prefilter_memory_bank_for_active_roles(
                    tokens=layer_selected_mem,
                    token_meta=layer_selected_meta,
                    token_lengths=layer_lengths,
                    active_roles=active_roles,
                    role_boxes=query_role_boxes,
                    role_order_hint=(list(layer_query_payload.keys()) if isinstance(layer_query_payload, dict) else None),
                )
            layer_projected = _project_memory_for_sparse(layer_selected_mem, layer_lengths)
            layer_disabled = bool(step_force_disable_injection)
            if not (isinstance(layer_query_payload, dict) and len(layer_query_payload) > 0):
                layer_disabled = True
            if layer_projected is None or int(layer_projected.shape[1]) <= 0:
                layer_disabled = True
                layer_projected = None
            return layer_projected, layer_selected_meta, layer_lengths, layer_disabled

        def _apply_sparse_delta_controls(x_before, x_after, layer_idx, layer_meta):
            static_scale = float(getattr(self, 'sparse_role_memory_layer_scales', {}).get(int(layer_idx), 1.0)) if hasattr(self, 'sparse_role_memory_layer_scales') else 1.0
            static_scale = max(0.0, static_scale)
            total_scale = float(static_scale)
            delta = (x_after - x_before) * total_scale
            return x_before + delta, {
                'applied_layer_scale': float(total_scale),
            }

        def _forward_block_official(block_module, x_in, e_in):
            with torch.amp.autocast('cuda', dtype=model_dtype):
                return block_module(
                    x_in,
                    e=e_in,
                    seq_lens=seq_lens,
                    grid_sizes=grid_sizes,
                    freqs=freqs,
                    context=context_emb,
                    context_lens=context_lens,
                )

        def _forward_block_with_sparse_context_only(
            block_module,
            x_in,
            e_in,
            memory_tokens_current,
            query_role_boxes_current,
            query_feature_payload_current,
            selected_meta_current,
            token_lengths_current,
            layer_idx_current,
        ):
            character_attn_stats = {
                'enabled': 0.0,
                'selected_query_tokens': 0,
                'selected_memory_tokens': 0,
                'winner_counts': {},
                'role_head_out_norm': 0.0,
                'plain_head_out_norm': 0.0,
                'attn_entropy': 0.0,
            }
            with torch.amp.autocast('cuda', dtype=model_dtype):
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    block_module.modulation.to(dtype=e_in.dtype, device=e_in.device) + e_in
                ).chunk(6, dim=1)
                input_x = modulate(block_module.norm1(x_in), shift_msa, scale_msa)
                x_sa = x_in + gate_msa * block_module.self_attn(input_x, seq_lens, grid_sizes, freqs)
            del input_x

            def _cross_attn_sparse_ffn(x_mid_in):
                local_character_attn_stats = character_attn_stats
                with torch.amp.autocast('cuda', dtype=model_dtype):
                    x_ca = x_mid_in + block_module.cross_attn(block_module.norm3(x_mid_in), context_emb, context_lens)

                if (
                    enable_sparse_context_only
                    and (not step_force_disable_injection)
                    and isinstance(query_feature_payload_current, dict)
                    and memory_tokens_current is not None
                    and int(memory_tokens_current.shape[1]) > 0
                ):
                    x_before_sparse = x_ca
                    x_after_sparse, local_character_attn_stats = active_character_wise_cross_attention(
                        x_ca,
                        memory_tokens_current.to(dtype=x_ca.dtype),
                        query_role_boxes_current,
                        query_feature_payload_current,
                        selected_meta_current,
                        token_lengths_current,
                        int(h),
                        int(w),
                        sparse_timestep_percent,
                    )
                    x_ca, control_stats = _apply_sparse_delta_controls(
                        x_before_sparse,
                        x_after_sparse,
                        layer_idx=layer_idx_current,
                        layer_meta=selected_meta_current,
                    )
                    local_character_attn_stats = dict(local_character_attn_stats) if isinstance(local_character_attn_stats, dict) else {}
                    local_character_attn_stats.update(control_stats)
                    with torch.no_grad():
                        local_character_attn_stats['raw_delta_norm'] = float((x_after_sparse - x_before_sparse).detach().float().norm(dim=-1).mean().item())
                        local_character_attn_stats['effective_delta_norm'] = float((x_ca - x_before_sparse).detach().float().norm(dim=-1).mean().item())

                with torch.amp.autocast('cuda', dtype=model_dtype):
                    input_x2 = modulate(block_module.norm2(x_ca), shift_mlp, scale_mlp)
                    x_out = x_ca + gate_mlp * block_module.ffn(input_x2)
                del input_x2, x_ca
                return x_out, local_character_attn_stats

            x_out, character_attn_stats = _cross_attn_sparse_ffn(x_sa)
            del x_sa
            return x_out, character_attn_stats

        for layer_idx, block in enumerate(dit_model.blocks):
            if should_debug_print and layer_idx in {0, len(dit_model.blocks) // 2, len(dit_model.blocks) - 1}:
                print(
                    f"[DEBUG][Train][Step {int(self.global_step)}][Layer {layer_idx}] "
                    f"x_shape={tuple(x_output.shape)} text_shape={tuple(context_emb.shape)} "
                    f"mem_shape={tuple(memory_tokens_for_sparse.shape) if memory_tokens_for_sparse is not None else None}",
                    flush=True,
                )
            is_sparse_context_layer = (
                enable_sparse_context_only
                and int(layer_idx) in sparse_layer_indices
                and hasattr(block, 'self_attn')
                and hasattr(block, 'cross_attn')
                and hasattr(block, 'ffn')
                and hasattr(block, 'norm1')
                and hasattr(block, 'norm2')
                and hasattr(block, 'norm3')
                and hasattr(block, 'modulation')
            )
            if is_sparse_context_layer:
                layer_memory_tokens_for_sparse = memory_tokens_for_sparse
                layer_selected_meta = selected_bank_token_meta
                layer_lengths = memory_token_lengths_per_character
                layer_disable_sparse = bool(step_force_disable_injection)
                if layerwise_sparse_payload:
                    (
                        layer_memory_tokens_for_sparse,
                        layer_selected_meta,
                        layer_lengths,
                        layer_disable_sparse,
                    ) = _prepare_sparse_inputs_for_layer(layer_idx, current_x_output=x_output)
                if self.training and kwargs.get('use_gradient_checkpointing', False):
                    character_attn_stats_holder = {}

                    def _custom_forward_sparse(
                        x_in,
                        e_in,
                        memory_tokens_in,
                        block_module=block,
                        query_role_boxes_current=query_role_boxes,
                        query_feature_payload_current=query_feature_payload,
                        selected_meta_current=layer_selected_meta,
                        token_lengths_current=layer_lengths,
                        layer_idx_current=int(layer_idx),
                        layer_disable_current=layer_disable_sparse,
                        stats_holder=character_attn_stats_holder,
                    ):
                        if layer_disable_current:
                            return _forward_block_official(block_module, x_in, e_in)
                        x_out, local_character_attn_stats = _forward_block_with_sparse_context_only(
                            block_module,
                            x_in,
                            e_in,
                            memory_tokens_in,
                            query_role_boxes_current,
                            query_feature_payload_current,
                            selected_meta_current,
                            token_lengths_current,
                            layer_idx_current,
                        )
                        stats_holder['stats'] = local_character_attn_stats
                        return x_out

                    if kwargs.get('use_gradient_checkpointing_offload', False):
                        with torch.autograd.graph.save_on_cpu():
                            x_output = torch.utils.checkpoint.checkpoint(
                                _custom_forward_sparse,
                                x_output,
                                t_mod,
                                layer_memory_tokens_for_sparse if layer_memory_tokens_for_sparse is not None else torch.empty((batch_size, 0, int(self.patch_dim)), device=device, dtype=model_dtype),
                                use_reentrant=False,
                            )
                    else:
                        x_output = torch.utils.checkpoint.checkpoint(
                            _custom_forward_sparse,
                            x_output,
                            t_mod,
                            layer_memory_tokens_for_sparse if layer_memory_tokens_for_sparse is not None else torch.empty((batch_size, 0, int(self.patch_dim)), device=device, dtype=model_dtype),
                            use_reentrant=False,
                        )
                    character_attn_stats = character_attn_stats_holder.get('stats', self._last_sparse_role_memory_stats)
                else:
                    if layer_disable_sparse:
                        x_output = _forward_block_official(block, x_output, t_mod)
                        character_attn_stats = self._last_sparse_role_memory_stats
                    else:
                        x_output, character_attn_stats = _forward_block_with_sparse_context_only(
                            block,
                            x_output,
                            t_mod,
                            layer_memory_tokens_for_sparse,
                            query_role_boxes,
                            query_feature_payload,
                            layer_selected_meta,
                            layer_lengths,
                            int(layer_idx),
                        )
                self._last_sparse_role_memory_stats = character_attn_stats
                if int(layer_idx) >= int(last_sparse_layer_idx) and not layerwise_sparse_payload:
                    query_feature_payload = None
                    query_role_boxes = None
                    memory_tokens_for_sparse = None
                continue
            if self.training and kwargs.get('use_gradient_checkpointing', False):
                def _custom_forward(x_in, e_in, block_module=block):
                    return _forward_block_official(block_module, x_in, e_in)
                if kwargs.get('use_gradient_checkpointing_offload', False):
                    with torch.autograd.graph.save_on_cpu():
                        x_output = torch.utils.checkpoint.checkpoint(_custom_forward, x_output, t_mod, use_reentrant=False)
                else:
                    x_output = torch.utils.checkpoint.checkpoint(_custom_forward, x_output, t_mod, use_reentrant=False)
            else:
                x_output = _forward_block_official(block, x_output, t_mod)

        for hook in hooks:
            hook.remove()
        if attn_stats:
            self.attn_response_history.append({"step": self.global_step, "stats": attn_stats})
        if jigsaw_aux_losses:
            self._last_jigsaw_extra_encoder_aux_loss = torch.stack([x.float() for x in jigsaw_aux_losses]).mean()
        else:
            self._last_jigsaw_extra_encoder_aux_loss = x_output.new_zeros(())
        self._last_jigsaw_extra_encoder_stats = dict(jigsaw_aux_stats_last)
        self._last_jigsaw_stage2_writer_stats = dict(jigsaw_stage2_writer_stats_last)
        with torch.amp.autocast('cuda', dtype=model_dtype):
            x_output = dit_model.head(x_output, t_embed.float())
        return dit_model.unpatchify(x_output, (f, h, w))

    def training_step(self, batch, batch_idx):
        """
        已优化的训练步：
        1. 优先使用提取端预计算的 latents，跳过耗时的 VAE 编码。
        2. 仅在 latents 缺失时回退到实时 VAE 编码。
        3. [修复] 恢复了 Motion Frames 的概率判定逻辑。
        """
        self._train_runtime_log('training_step_enter', batch_is_none=(batch is None), batch_idx=batch_idx)
        if self._ddp_sync_should_skip(batch is None):
            raise RuntimeError("training_step received invalid batch after dataset-side resampling; expected dataset to yield only valid memory samples")

        self._train_runtime_log('training_step_after_first_sync', batch_keys=sorted(list(batch.keys())) if isinstance(batch, dict) else type(batch).__name__)
        if self.timing_tracker:
            self.timing_tracker.start('training')
        
        # --- 1. 基础数据准备 ---
        text = batch.get("text", "")
        if isinstance(text, list):
            text = text[0]
        
        use_memory = batch.get("use_memory", False)
        memory_feature_bank_tokens_selected_raw = batch.get('memory_feature_bank_tokens_selected')
        memory_feature_bank_token_meta_selected_raw = batch.get('memory_feature_bank_token_meta_selected')
        stage2_update_memory_feature_bank_tokens_selected_raw = batch.get('stage2_update_memory_feature_bank_tokens_selected')
        stage2_update_memory_feature_bank_token_meta_selected_raw = batch.get('stage2_update_memory_feature_bank_token_meta_selected')

        def _bank_has_real_tokens(bank_tokens):
            if _pm_is_layerwise_container(bank_tokens):
                return int(_memory_count_tokens(bank_tokens)) > 0
            if not isinstance(bank_tokens, dict):
                return False
            for _tok in bank_tokens.values():
                if isinstance(_tok, torch.Tensor) and _tok.ndim >= 2 and int(_tok.shape[0]) > 0:
                    return True
            return False

        def _first_bank_token_count(bank_tokens):
            if _pm_is_layerwise_container(bank_tokens):
                for _, layer_value in _pm_iter_layerwise_items(bank_tokens):
                    if isinstance(layer_value, dict):
                        for _tok in layer_value.values():
                            if isinstance(_tok, torch.Tensor) and _tok.ndim >= 2 and int(_tok.shape[0]) > 0:
                                return int(_tok.shape[0])
                    elif isinstance(layer_value, torch.Tensor) and layer_value.ndim >= 2 and int(layer_value.shape[0]) > 0:
                        return int(layer_value.shape[0])
                return 0
            if not isinstance(bank_tokens, dict):
                return 0
            for _tok in bank_tokens.values():
                if isinstance(_tok, torch.Tensor) and _tok.ndim >= 2 and int(_tok.shape[0]) > 0:
                    return int(_tok.shape[0])
            return 0
        
        # 预设标志位
        memory_dropped = False
        image_emb = {}
        
        # --- 2. 核心编码逻辑 (VAE & CLIP & Text) ---
        with torch.no_grad():
            # A. 文本提示词编码 (由于涉及 Dropout 增强，通常在训练端实时计算)
            prompt_emb = self.pipe_VAE.encode_prompt(text)
            
            # B. 视频 Latents 获取 (提速关键点)
            latents = batch.get("latents")
            
            if latents is not None:
                self._train_runtime_log('latents_prefetched_found', latents=latents)
                # [路径 1] 使用提取端缓存好的 Latents (极快)
                latents = latents.to(device=self.device, dtype=self.pipe_VAE.torch_dtype)
                if latents.dim() == 4:
                    latents = latents.unsqueeze(0)
            else:
                latents = None

            self._train_runtime_log('before_latents_sync', latents_is_none=(latents is None))
            if self._ddp_sync_should_skip(latents is None):
                raise RuntimeError("latents missing after dataset-side resampling; expected prefetched valid latents for every yielded batch")

            # C. 图像特征编码 (Image Encoder / CLIP)
            self._train_runtime_log('after_latents_sync', latents=latents)
            # DDP-shared trigger: decide once per training step (not extraction step).
            pixelate_prob = float(getattr(self.args, 'image_condition_people_pixelate_prob', 0.0)) if hasattr(self, 'args') else 0.0
            pixelate_enabled = bool(pixelate_prob > 0.0)
            has_image_refs = bool(
                isinstance(batch, dict)
                and (batch.get("first_ref_frames") is not None)
                and (batch.get("random_ref_frame") is not None)
            )
            original_precomputed_image_emb = batch.get("precomputed_image_emb") if isinstance(batch, dict) else None
            had_precomputed_before = bool(isinstance(batch, dict) and (batch.get("precomputed_image_emb") is not None))
            shared_pixelate_trigger = False
            if pixelate_prob > 0.0 and has_image_refs:
                if dist.is_available() and dist.is_initialized():
                    pixelate_tensor = torch.zeros(1, device=self.device, dtype=torch.int32)
                    if self.global_rank == 0:
                        pixelate_tensor.fill_(1 if random.random() < pixelate_prob else 0)
                    dist.broadcast(pixelate_tensor, src=0)
                    shared_pixelate_trigger = bool(pixelate_tensor.item())
                else:
                    shared_pixelate_trigger = bool(random.random() < pixelate_prob)

            allow_missing_precomputed = False
            pixelate_applied = False
            if shared_pixelate_trigger and has_image_refs:
                batch, pixelate_applied = maybe_pixelate_condition_batch_cpu(
                    batch=batch,
                    enabled=True,
                    model_path=getattr(self.args, 'image_condition_people_pixelate_model_path', DEFAULT_YOLO_SEG_CKPT),
                    conf=float(getattr(self.args, 'image_condition_people_pixelate_conf', 0.25)),
                    pixel_block_size=int(getattr(self.args, 'image_condition_people_pixelate_block_size', 12)),
                    mask_dilate_kernel=int(getattr(self.args, 'image_condition_people_pixelate_mask_dilate_kernel', 9)),
                )
                # If any condition frame changed, precomputed embedding must be regenerated in this step.
                allow_missing_precomputed = bool(pixelate_applied)
                self._train_runtime_log(
                    'shared_condition_pixelate',
                    trigger=bool(shared_pixelate_trigger),
                    applied=bool(pixelate_applied),
                    prob=float(pixelate_prob),
                )

            print(
                f"[PixelateStep][Rank {int(getattr(self, 'global_rank', -1))}] "
                f"step={int(getattr(self, 'global_step', -1))} "
                f"enabled={pixelate_enabled} prob={pixelate_prob:.4f} has_refs={has_image_refs} "
                f"trigger={bool(shared_pixelate_trigger)} applied={bool(pixelate_applied)} "
                f"precomputed_before={had_precomputed_before} allow_missing_precomputed={bool(allow_missing_precomputed)}",
                flush=True,
            )

            image_emb = self._prepare_train_image_emb_scoped(
                batch=batch,
                latents=latents,
                allow_missing_precomputed=allow_missing_precomputed,
            )

        # --- 3. 扩散模型训练逻辑 ---
        self._maybe_trim_cuda_cache(stage='pre_diffusion')
        self._train_runtime_log('before_diffusion_prep', latents=latents, use_memory=use_memory, has_memory_features=_bank_has_real_tokens(memory_feature_bank_tokens_selected_raw))
        image_condition_available_cached = bool(
            isinstance(image_emb, dict) and (
                (image_emb.get('clip_feature', None) is not None) or
                (image_emb.get('y', None) is not None)
            )
        )
        self.pipe.device = self.device
        
        # 准备噪声和时间步（若提取端提供对齐 timestep，则优先复用）
        noise = torch.randn_like(latents)
        extracted_timestep = batch.get("extracted_timestep", None)
        if isinstance(extracted_timestep, torch.Tensor):
            if extracted_timestep.numel() > 0:
                extracted_timestep = float(extracted_timestep.detach().cpu().view(-1)[0].item())
            else:
                extracted_timestep = None
        elif isinstance(extracted_timestep, (list, tuple)):
            extracted_timestep = extracted_timestep[0] if len(extracted_timestep) > 0 else None
            if isinstance(extracted_timestep, torch.Tensor):
                extracted_timestep = float(extracted_timestep.detach().cpu().view(-1)[0].item()) if extracted_timestep.numel() > 0 else None

        if extracted_timestep is not None:
            timestep = torch.tensor([float(extracted_timestep)], device=self.device, dtype=latents.dtype)
        else:
            sampled_t = _sample_timestep_for_domain(
                sched_timesteps=self.pipe.scheduler.timesteps,
                train_noise_domain=self.train_noise_domain,
                num_train_timesteps=self.pipe.scheduler.num_train_timesteps,
                boundary_ratio=self.noise_domain_boundary,
            )
            if sampled_t is None:
                t_idx = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
                timestep = self.pipe.scheduler.timesteps[t_idx].to(device=self.device)
            else:
                timestep = torch.tensor([float(sampled_t)], device=self.device, dtype=latents.dtype)

        current_noise_domain = self._resolve_noise_domain_from_timestep(timestep)
        self.pipe.set_active_noise_domain(current_noise_domain)
        self._set_active_lora_for_noise_domain(current_noise_domain)
        character_wise_cross_attention_this_step = self._get_character_wise_cross_attention_for_domain(current_noise_domain)
        char_attn_enabled_this_step = bool(character_wise_cross_attention_this_step is not None)
        memory_enabled_this_step = bool(char_attn_enabled_this_step)
        dit_model = self.pipe.denoising_model()
        model_dtype = dit_model.patch_embedding.weight.dtype
        prompt_emb = prompt_emb.to(device=self.device, dtype=model_dtype)

        if self.global_step % 10 == 0:
            self.log("train/noise_domain_high", 1.0 if current_noise_domain == 'high_noise' else 0.0, on_step=True, logger=True)
            self.log("train/char_attn_enabled", 1.0 if char_attn_enabled_this_step else 0.0, on_step=True, logger=True)
        
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        target = self.pipe.scheduler.training_target(latents, noise, timestep)
        probe_timestep = timestep
        probe_noisy_latents = None
        if str(getattr(self, 'train_query_probe_timestep_mode', 'current')).strip().lower() == 'previous_scheduler':
            previous_t = _previous_scheduler_timestep_value(getattr(self.pipe.scheduler, 'timesteps', None), timestep)
            if previous_t is not None:
                probe_timestep = torch.tensor([float(previous_t)], device=self.device, dtype=latents.dtype)
                probe_noisy_latents = self.pipe.scheduler.add_noise(latents, noise, probe_timestep)
        noisy_latents = noisy_latents.to(device=self.device, dtype=model_dtype)
        if probe_noisy_latents is None:
            probe_noisy_latents = noisy_latents
        else:
            probe_noisy_latents = probe_noisy_latents.to(device=self.device, dtype=model_dtype)
        latent_dtype = latents.dtype
        # latents is no longer needed after noisy_latents/target are prepared.
        del latents
        # noise is no longer needed after target is prepared.
        del noise
        self._train_runtime_log('after_noise_prep', timestep=timestep, noisy_latents=noisy_latents, target=target)
        
        # 记忆注入与前向传播
        memory_condition_available = bool(
            use_memory and _bank_has_real_tokens(memory_feature_bank_tokens_selected_raw)
            and memory_enabled_this_step
        )
        stage2_writer_module_this_step = self._get_memory_writer_for_domain(current_noise_domain)
        stage2_writer_active_this_step = bool(
            getattr(self, 'memory_writer_enabled', False)
            and stage2_writer_module_this_step is not None
        )
        stage2_stage1_branch_this_step = False
        if (
            str(getattr(self, 'train_stage', 'stage1')).strip().lower() == 'stage2'
            and stage2_writer_active_this_step
            and float(getattr(self, 'stage2_stage1_branch_prob', 0.0)) > 0.0
        ):
            branch_prob = min(max(float(getattr(self, 'stage2_stage1_branch_prob', 0.0)), 0.0), 1.0)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                branch_tensor = torch.zeros(1, device=self.device, dtype=torch.int32)
                if int(getattr(self, 'global_rank', 0)) == 0:
                    branch_tensor.fill_(1 if random.random() < branch_prob else 0)
                torch.distributed.broadcast(branch_tensor, src=0)
                stage2_stage1_branch_this_step = bool(branch_tensor.item())
            else:
                stage2_stage1_branch_this_step = bool(random.random() < branch_prob)
        stage2_writer_forward_this_step = bool(stage2_writer_active_this_step and not stage2_stage1_branch_this_step)
        if stage2_writer_forward_this_step and memory_condition_available and not _bank_has_real_tokens(stage2_update_memory_feature_bank_tokens_selected_raw):
            raise RuntimeError(
                "Stage2 writer is enabled but stage2 update memory bank is missing or empty; "
                "expected dataloader to provide stage2_update_extra_video and inline extraction to return "
                "stage2_update_memory_feature_bank_tokens_selected."
            )
        self.log("train/stage2_stage1_branch", 1.0 if stage2_stage1_branch_this_step else 0.0, on_step=True, logger=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            mem_tensor = torch.zeros(1, device=self.device, dtype=torch.int32)
            mem_tensor.fill_(1 if memory_condition_available else 0)
            torch.distributed.all_reduce(mem_tensor, op=torch.distributed.ReduceOp.MAX)
            memory_path_required = bool(mem_tensor.item() > 0)
        else:
            memory_path_required = bool(memory_condition_available)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            drop_tensor = torch.zeros(1, device=self.device, dtype=torch.int32)
            if self.global_rank == 0:
                drop_tensor.fill_(1 if random.random() < self.memory_drop_prob else 0)
            torch.distributed.broadcast(drop_tensor, src=0)
            shared_drop = bool(drop_tensor.item())
        else:
            shared_drop = bool(random.random() < self.memory_drop_prob)
        query_role_boxes = None
        query_feature_payload = None
        if memory_path_required:
            local_has_real_memory = bool(memory_condition_available)
            if local_has_real_memory:
                # Bank is the source of truth; selected tensor key is kept as empty compatibility payload.
                memory_feature_tokens_selected = torch.zeros((0, int(self.patch_dim)), device=self.device, dtype=latent_dtype)
                raw_lengths = batch.get('memory_token_lengths_per_character')
                memory_feature_bank_tokens_selected = memory_feature_bank_tokens_selected_raw
                memory_feature_bank_token_meta_selected = memory_feature_bank_token_meta_selected_raw
                stage2_update_memory_feature_bank_tokens_selected = stage2_update_memory_feature_bank_tokens_selected_raw
                stage2_update_memory_feature_bank_token_meta_selected = stage2_update_memory_feature_bank_token_meta_selected_raw
                memory_bank_percents = batch.get('memory_bank_percents')
            else:
                memory_feature_tokens_selected = torch.zeros((1, int(self.patch_dim)), device=self.device, dtype=latent_dtype)
                # Keep ZeRO-3 submodule order aligned while disabling real memory effect on this rank.
                shared_drop = True
                raw_lengths = None
                memory_feature_bank_tokens_selected = None
                memory_feature_bank_token_meta_selected = None
                stage2_update_memory_feature_bank_tokens_selected = None
                stage2_update_memory_feature_bank_token_meta_selected = None
                memory_bank_percents = None
            memory_dropped = bool(shared_drop)
            
            N = _first_bank_token_count(memory_feature_bank_tokens_selected)
            memory_token_lengths_per_character = None
            if raw_lengths is not None:
                if isinstance(raw_lengths, torch.Tensor):
                    raw_lengths = raw_lengths.squeeze(0).tolist() if raw_lengths.dim() > 1 else raw_lengths.tolist()
                elif isinstance(raw_lengths, (list, tuple)) and len(raw_lengths) == 1 and isinstance(raw_lengths[0], torch.Tensor):
                    t0 = raw_lengths[0]
                    raw_lengths = t0.squeeze(0).tolist() if t0.dim() > 1 else t0.tolist()
                if isinstance(raw_lengths, (list, tuple)):
                    memory_token_lengths_per_character = [int(x) for x in raw_lengths if int(x) > 0]

            if not memory_token_lengths_per_character:
                if N % 2048 == 0 and N >= 2048:
                    memory_token_lengths_per_character = [2048] * (N // 2048)
                else:
                    memory_token_lengths_per_character = [N]
            
            clip_feat = image_emb.pop('clip_feature', None)
            y_tensor = image_emb.pop('y', None)
            self._train_runtime_log(
                'train_step_image_condition_pop',
                y=y_tensor,
                clip_feature=clip_feat,
                image_emb_keys=sorted(list(image_emb.keys())) if isinstance(image_emb, dict) else None,
            )
            if isinstance(clip_feat, torch.Tensor):
                clip_feat = clip_feat.to(device=self.device, dtype=model_dtype)
            if isinstance(y_tensor, torch.Tensor):
                y_tensor = y_tensor.to(device=self.device, dtype=model_dtype)
            num_condition_frames = image_emb.get('num_condition_frames', None)

            # Wan2.2 official I2V uses y only; there is no CLIP image encoder / clip_feature.
            probe_clip_feat = None
            probe_y_tensor = y_tensor
            probe_source = 'train_image_emb'
            # Keep probe on non-pixelated condition whenever original precomputed y exists.
            has_original_precomputed_probe = False
            if bool(pixelate_applied) and isinstance(original_precomputed_image_emb, dict):
                pre_y = original_precomputed_image_emb.get('y', None)
                if isinstance(pre_y, torch.Tensor):
                    probe_clip_feat = None
                    probe_y_tensor = pre_y.to(device=self.device, dtype=model_dtype)
                    probe_source = 'original_precomputed_image_emb_y'
                    has_original_precomputed_probe = True

            print(
                f"[PixelateProbe][Rank {int(getattr(self, 'global_rank', -1))}] "
                f"step={int(getattr(self, 'global_step', -1))} pixelate_applied={bool(pixelate_applied)} "
                f"probe_source={probe_source}",
                flush=True,
            )

            if isinstance(memory_bank_percents, torch.Tensor):
                memory_bank_percents = memory_bank_percents.detach().cpu().tolist()
            elif isinstance(memory_bank_percents, np.ndarray):
                memory_bank_percents = memory_bank_percents.tolist()

            need_probe_for_memory = bool((not memory_dropped) and char_attn_enabled_this_step)
            need_probe_for_loss = bool(getattr(self, 'probe_bbox_loss_weight_x2', False))
            if bool(pixelate_applied) and (need_probe_for_memory or need_probe_for_loss) and (not has_original_precomputed_probe):
                raise RuntimeError(
                    "Pixelate is applied on Wan2.2 training image condition y, but probe has no original precomputed y. "
                    "To keep probe non-pixelated, enable extraction-side precompute_image_emb and ensure precomputed_image_emb['y'] is present in batch."
                )
            if need_probe_for_memory or need_probe_for_loss:
                query_role_boxes, query_feature_payload = self._run_train_character_semantic_probe_scoped(
                    noisy_latents=probe_noisy_latents,
                    timestep=probe_timestep,
                    context=prompt_emb,
                    text=text,
                    clip_feature=probe_clip_feat,
                    y=probe_y_tensor,
                    memory_bank_token_meta=memory_feature_bank_token_meta_selected,
                )
            if probe_noisy_latents is not noisy_latents:
                del probe_noisy_latents

            self._train_runtime_log('before_memory_aware_dit_forward', memory_dropped=memory_dropped, memory_condition_available=memory_condition_available)
            noise_pred = self._memory_aware_dit_forward(
                noisy_latents=noisy_latents,
                memory_feature_tokens_selected=memory_feature_tokens_selected,
                memory_feature_bank_tokens_selected=memory_feature_bank_tokens_selected,
                memory_bank_percents=memory_bank_percents,
                memory_feature_bank_token_meta_selected=memory_feature_bank_token_meta_selected,
                stage2_update_memory_feature_bank_tokens_selected=stage2_update_memory_feature_bank_tokens_selected,
                stage2_update_memory_feature_bank_token_meta_selected=stage2_update_memory_feature_bank_token_meta_selected,
                query_role_boxes=query_role_boxes,
                query_feature_payload=query_feature_payload,
                force_disable_injection=memory_dropped,
                timestep=timestep,
                context=prompt_emb,
                clip_feature=clip_feat,
                y=y_tensor,
                memory_token_lengths_per_character=memory_token_lengths_per_character,
                allow_sparse_role_memory=char_attn_enabled_this_step,
                character_wise_cross_attention=character_wise_cross_attention_this_step,
                allow_jigsaw_stage2_writer=stage2_writer_forward_this_step,
                num_condition_frames=num_condition_frames,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
                **extra_input
            )
            # Free memory-path temporaries as soon as forward output is materialized.
            del memory_feature_tokens_selected
            if query_feature_payload is not None:
                del query_feature_payload
            if probe_clip_feat is not None and probe_clip_feat is not clip_feat:
                del probe_clip_feat
            if probe_y_tensor is not None and probe_y_tensor is not y_tensor:
                del probe_y_tensor
            if clip_feat is not None:
                del clip_feat
            if y_tensor is not None:
                del y_tensor
            if memory_feature_bank_tokens_selected is not None:
                del memory_feature_bank_tokens_selected
            if memory_feature_bank_token_meta_selected is not None:
                del memory_feature_bank_token_meta_selected
            if stage2_update_memory_feature_bank_tokens_selected is not None:
                del stage2_update_memory_feature_bank_tokens_selected
            if stage2_update_memory_feature_bank_token_meta_selected is not None:
                del stage2_update_memory_feature_bank_token_meta_selected
            if memory_bank_percents is not None:
                del memory_bank_percents
            if memory_token_lengths_per_character is not None:
                del memory_token_lengths_per_character
            if raw_lengths is not None:
                del raw_lengths
        else:
            if probe_noisy_latents is not noisy_latents:
                del probe_noisy_latents
            noise_pred = self.pipe.denoising_model()(
                noisy_latents, timestep=timestep,
                context=prompt_emb, **extra_input, **image_emb,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            )

        # Forward finished; condition/context tensors are no longer needed.
        if isinstance(image_emb, dict):
            image_emb.pop('clip_feature', None)
            image_emb.pop('y', None)
        
        # --- 4. 计算 Loss 与日志 ---
        loss_pre_weighted, loss_pre_no_bbox_x2, bbox_cover_ratio, bbox_weight_applied = self._compute_loss_with_optional_character_probe_weight(
            noise_pred=noise_pred,
            target=target,
            query_role_boxes=query_role_boxes,
        )
        self._train_runtime_log('after_loss', loss=loss_pre_weighted, noise_pred=noise_pred)
        train_weight = self.pipe.scheduler.training_weight(timestep)
        loss = loss_pre_weighted * train_weight
        loss_no_bbox_x2 = loss_pre_no_bbox_x2 * train_weight
        jigsaw_aux_loss = getattr(self, '_last_jigsaw_extra_encoder_aux_loss', None)
        if isinstance(jigsaw_aux_loss, torch.Tensor) and float(getattr(self, 'jigsaw_extra_encoder_aux_weight', 0.0)) > 0.0:
            loss = loss + jigsaw_aux_loss.to(device=loss.device, dtype=loss.dtype) * float(self.jigsaw_extra_encoder_aux_weight)
        if bool(getattr(self, 'jigsaw_extra_encoder_enabled', False)):
            jigsaw_module = self._get_jigsaw_extra_encoder_for_domain(current_noise_domain)
            if jigsaw_module is not None:
                zero_touch = None
                for p in jigsaw_module.parameters():
                    term = self._safe_zero_touch_term(p)
                    if term is None:
                        continue
                    zero_touch = term if zero_touch is None else zero_touch + term
                if isinstance(zero_touch, torch.Tensor):
                    loss = loss + zero_touch.to(device=loss.device, dtype=loss.dtype)
        if bool(getattr(self, 'enable_sparse_role_memory_attn', False)):
            sparse_module = self._get_character_wise_cross_attention_for_domain(current_noise_domain)
            if sparse_module is not None:
                zero_touch = None
                for p in sparse_module.parameters():
                    term = self._safe_zero_touch_term(p)
                    if term is None:
                        continue
                    zero_touch = term if zero_touch is None else zero_touch + term
                if isinstance(zero_touch, torch.Tensor):
                    loss = loss + zero_touch.to(device=loss.device, dtype=loss.dtype)
        if bool(getattr(self, 'memory_writer_enabled', False)):
            writer_module = self._get_memory_writer_for_domain(current_noise_domain)
            if writer_module is not None:
                zero_touch = None
                for p in writer_module.parameters():
                    term = self._safe_zero_touch_term(p)
                    if term is None:
                        continue
                    zero_touch = term if zero_touch is None else zero_touch + term
                if isinstance(zero_touch, torch.Tensor):
                    loss = loss + zero_touch.to(device=loss.device, dtype=loss.dtype)

        # Release large per-step tensor references early to lower transient peak memory.
        del extra_input, noisy_latents, target, noise_pred
        if query_role_boxes is not None:
            del query_role_boxes

        # Optional branches now simply remain unused when not taken.
        # No zero-touch / nominal parameter participation here.
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_loss_no_bbox_x2", loss_no_bbox_x2, prog_bar=False)
        if isinstance(jigsaw_aux_loss, torch.Tensor):
            self.log("train/memory_encoder_aux_loss", jigsaw_aux_loss.detach(), on_step=True, logger=True)
        if isinstance(getattr(self, '_last_jigsaw_extra_encoder_stats', None), dict):
            _jes = self._last_jigsaw_extra_encoder_stats
            self.log("monitor/memory_encoder_enabled", float(_jes.get('enabled', 0.0)), on_step=True, logger=True)
            self.log("monitor/memory_encoder_group", float(_jes.get('group', 0.0)), on_step=True, logger=True)
            self.log("monitor/memory_encoder_slots", float(_jes.get('output_slots', 0.0)), on_step=True, logger=True)
            self.log("monitor/memory_encoder_memory_side_rope", 0.0 if bool(getattr(self, 'jigsaw_disable_memory_side_rope', True)) else 1.0, on_step=True, logger=True)
        if isinstance(getattr(self, '_last_jigsaw_stage2_writer_stats', None), dict):
            _jws = self._last_jigsaw_stage2_writer_stats
            self.log("monitor/memory_writer_enabled", float(_jws.get('enabled', 0.0)), on_step=True, logger=True)
            self.log("monitor/memory_writer_updated_slots", float(_jws.get('updated_slots', 0.0)), on_step=True, logger=True)
            self.log("monitor/memory_writer_mean_gate", float(_jws.get('mean_gate', 0.0)), on_step=True, logger=True)
            self.log("monitor/memory_writer_mean_cos", float(_jws.get('mean_cos', 0.0)), on_step=True, logger=True)
            self.log("monitor/memory_writer_clipped_ratio", float(_jws.get('clipped_ratio', 0.0)), on_step=True, logger=True)
        self.log("probe_bbox_loss_weight_enabled", 1.0 if self.probe_bbox_loss_weight_x2 else 0.0)
        self.log("probe_bbox_loss_weight_applied", float(bbox_weight_applied))
        self.log("probe_bbox_cover_ratio", float(bbox_cover_ratio))
        self.log("memory_dropped", 1.0 if memory_dropped else 0.0)
        loss_value = float(loss.detach().item())
        if memory_dropped:
            self._recent_memory_drop_losses.append(loss_value)
        else:
            self._recent_non_memory_drop_losses.append(loss_value)
        if self.global_step % 10 == 0:
            if isinstance(getattr(self, '_last_sparse_role_memory_stats', None), dict):
                _s = self._last_sparse_role_memory_stats
                self.log("monitor/sparse_mem_enabled", float(_s.get('enabled', 0.0)), on_step=True, logger=True)
                self.log("monitor/sparse_mem_query_tokens", float(_s.get('selected_query_tokens', 0.0)), on_step=True, logger=True)
                self.log("monitor/sparse_mem_memory_tokens", float(_s.get('selected_memory_tokens', 0.0)), on_step=True, logger=True)
                self.log("monitor/sparse_mem_role_head_norm", float(_s.get('role_head_out_norm', 0.0)), on_step=True, logger=True)
                self.log("monitor/sparse_mem_plain_head_norm", float(_s.get('plain_head_out_norm', 0.0)), on_step=True, logger=True)
                self.log("monitor/sparse_mem_attn_entropy", float(_s.get('attn_entropy', 0.0)), on_step=True, logger=True)
        if self.timing_tracker:
            self.timing_tracker.end('training')
        def _to_scalar(v):
            if isinstance(v, torch.Tensor):
                if v.numel() == 0:
                    return None
                return v.detach().cpu().view(-1)[0].item()
            if isinstance(v, (list, tuple)):
                if len(v) == 0:
                    return None
                return _to_scalar(v[0])
            return v

        memory_effective_used = bool(
            use_memory and
            memory_condition_available and
            (not memory_dropped)
        )
        memory_condition_used = bool(memory_effective_used)

        image_condition_available = bool(image_condition_available_cached)
        image_condition_used = bool(image_condition_available)
        image_condition_dropped = bool(image_condition_available and (not image_condition_used))

        display_step = int(self.global_step) + int(getattr(self, '_display_step_offset_runtime', 0))
        fusion_meta = getattr(self, '_last_v9_fusion_meta', {}) if memory_condition_available else {}
            
        if self.loss_logger is not None:
            self.loss_logger.log(
                step=display_step,
                loss=loss.item(),
                loss_no_bbox_x2=loss_no_bbox_x2.item(),
                used_memory=memory_effective_used,
                memory_dropped=bool(memory_dropped),
                image_condition_available=image_condition_available,
                image_condition_used=image_condition_used,
                image_condition_dropped=image_condition_dropped,
                memory_condition_available=memory_condition_available,
                memory_condition_used=memory_condition_used,
                folder_name=_to_scalar(batch.get('folder_name')),
                sample_idx=_to_scalar(batch.get('sample_idx')),
                video_id=_to_scalar(batch.get('video_id')),
                group_index=_to_scalar(batch.get('group_index')),
                core_clip_index=_to_scalar(batch.get('core_clip_index')),
                memory_clip_index=_to_scalar(batch.get('memory_clip_index')),
                diffusion_timestep=_to_scalar(timestep),
                selected_bank_idx=fusion_meta.get('selected_bank_idx', None),
                selected_bank_percent=fusion_meta.get('selected_bank_percent', None),
                p_cur=fusion_meta.get('p_cur', None),
                p_fusion=fusion_meta.get('p_fusion', None),
                fusion_alpha=fusion_meta.get('fusion_alpha', None),
                fusion_quantile=fusion_meta.get('fusion_quantile', None),
                fusion_max_inject_ratio=fusion_meta.get('fusion_max_inject_ratio', None),
                fusion_inject_ratio=fusion_meta.get('fusion_inject_ratio', None),
                fusion_sim1_mean=fusion_meta.get('fusion_sim1_mean', None),
                fusion_tau_sim=fusion_meta.get('fusion_tau_sim', None),
                memory_bank_count=fusion_meta.get('memory_bank_count', None),
                memory_bank_percents=fusion_meta.get('memory_bank_percents', None),
            )
            if self.global_step % 50 == 0:
                self.loss_logger.save()

        perf_log_interval = int(getattr(self.args, 'perf_log_interval', 0) or 0) if hasattr(self, 'args') else 0
        if perf_log_interval > 0 and self.global_rank == 0 and ((self.global_step + 1) % perf_log_interval == 0):
            now_t = time.time()
            if not hasattr(self, '_perf_last_log_time'):
                self._perf_last_log_time = now_t
            window_sec = max(now_t - self._perf_last_log_time, 1e-6)
            steps_per_sec = perf_log_interval / window_sec
            self._perf_last_log_time = now_t
            print(f"[Train] throughput step={self.global_step+1} {steps_per_sec:.3f} steps/s", flush=True)

        # 仅保存 LoRA 权重（不触发 DeepSpeed ckpt）

        save_every = getattr(self.args, "checkpoint_save_every_n_steps", 0) if hasattr(self, "args") else 0
        should_ckpt_sync = bool(save_every) and (self.global_step > 0) and (display_step > 0) and (display_step % save_every == 0)
        dist_ready = dist.is_available() and dist.is_initialized()

        # 1) 所有 rank：保存前对齐（避免 rank0 保存卡住时其它 rank 继续跑导致 collective mismatch/timeout）
        if should_ckpt_sync and dist_ready:
            if dist.get_backend() == "gloo":
                dist.monitored_barrier(timeout=timedelta(minutes=30))
            else:
                dist.barrier()

        # 2) 只有 rank0 真正执行保存（其它 rank 只等 barrier）
        if should_ckpt_sync and self.global_rank == 0:
            try:
                # ⚠️ 关键：不要在这里 return（否则其它 rank 会卡在下面的 barrier）
                if self._lora_save_thread is not None and self._lora_save_thread.is_alive():
                    print(f"[Checkpoint] Previous LoRA save still running, skip step {display_step}", flush=True)
                else:
                    if self.use_projector:
                        lora_state_dict = {n: p.detach().cpu().clone()
                                        for n, p in self.named_parameters() if p.requires_grad}
                    else:
                        lora_state_dict = {n: p.detach().cpu().clone()
                                        for n, p in self.named_parameters()
                                        if p.requires_grad and not n.startswith("memory_projector.")}

                    denoising_lora_state_dict = self._collect_active_denoising_trainable_state_dict()
                    lora_state_dict.update(denoising_lora_state_dict)

                    if hasattr(self, "memory_embeddings") and self.memory_embeddings is not None:
                        if getattr(self.memory_embeddings, "pos_embed", None) is not None:
                            lora_state_dict["memory_pos_embed"] = self.memory_embeddings.pos_embed.data.detach().cpu().clone()
                        if getattr(self.memory_embeddings, "segment_embed", None) is not None:
                            lora_state_dict["memory_segment_embed"] = self.memory_embeddings.segment_embed.data.detach().cpu().clone()

                    lora_save_path = os.path.join(self.trainer.default_root_dir, f"lora_weights_step_{display_step}.pt")

                    def _save_lora_payload(payload, save_path):
                        try:
                            torch.save(payload, save_path)
                        except Exception as save_err:
                            print(f"[Checkpoint] Warning: Failed to save standalone LoRA weights: {save_err}", flush=True)

                    self._lora_save_thread = threading.Thread(
                        target=_save_lora_payload,
                        args=(lora_state_dict, lora_save_path),
                        daemon=True,
                    )
                    self._lora_save_thread.start()
            except Exception as e:
                # rank0 保存异常也要显式打印（否则其它 rank 只看到 timeout/abort）
                print(f"[Checkpoint] ERROR at step {display_step}: {e}", flush=True)
                raise

        # 3) 所有 rank：保存后对齐
        if should_ckpt_sync and dist_ready:
            if dist.get_backend() == "gloo":
                dist.monitored_barrier(timeout=timedelta(minutes=30))
            else:
                dist.barrier()
                
        return loss
    def configure_optimizers(self):
        trainable_modules = []
        seen = set()
        covered_named_params = set()

        def _register_params(params):
            unique_params = []
            unique_ids = set()
            for p in params:
                if not p.requires_grad:
                    continue
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                unique_ids.add(pid)
                unique_params.append(p)
            if unique_ids:
                for name, p in self.named_parameters():
                    if id(p) in unique_ids:
                        covered_named_params.add(name)
            return unique_params

        denoising_params = []
        for model in (getattr(self.pipe, 'low_noise_model', None), getattr(self.pipe, 'high_noise_model', None)):
            if model is None:
                continue
            denoising_params.extend(_register_params(list(model.parameters())))
        if denoising_params:
            trainable_modules.append({'params': denoising_params, 'lr': self.learning_rate})
        if self.use_projector and hasattr(self, 'memory_projector'):
            projector_params = _register_params(list(self.memory_projector.parameters()))
            if projector_params:
                trainable_modules.append({'params': projector_params, 'lr': self.learning_rate * 2.0})
        if hasattr(self, 'memory_embeddings') and self.memory_embeddings is not None:
            mem_embed_params = _register_params(list(self.memory_embeddings.parameters()))
            if len(mem_embed_params) > 0:
                trainable_modules.append({'params': mem_embed_params, 'lr': self.learning_rate * 2.0})
        sparse_modules = []
        for attr in ('sparse_role_memory_attn_low_noise', 'sparse_role_memory_attn_high_noise', 'sparse_role_memory_attn'):
            m = getattr(self, attr, None)
            if m is None:
                continue
            if any(m is x for x in sparse_modules):
                continue
            sparse_modules.append(m)
        for m in sparse_modules:
            sparse_params = _register_params(list(m.parameters()))
            if len(sparse_params) > 0:
                sparse_lr = (
                    float(getattr(self, 'stage2_char_attn_lr', 1e-5))
                    if str(getattr(self, 'train_stage', 'stage1')).strip().lower() == 'stage2'
                    else self.learning_rate * 2.0
                )
                trainable_modules.append({'params': sparse_params, 'lr': sparse_lr})
        extra_encoder_modules = []
        for attr in ('jigsaw_extra_encoder_low_noise', 'jigsaw_extra_encoder_high_noise', 'jigsaw_extra_encoder'):
            m = getattr(self, attr, None)
            if m is None:
                continue
            if any(m is x for x in extra_encoder_modules):
                continue
            extra_encoder_modules.append(m)
        for m in extra_encoder_modules:
            enc_params = _register_params(list(m.parameters()))
            if len(enc_params) > 0:
                enc_lr = (
                    float(getattr(self, 'stage2_memory_encoder_lr', 1e-5))
                    if str(getattr(self, 'train_stage', 'stage1')).strip().lower() == 'stage2'
                    else self.learning_rate * 2.0
                )
                trainable_modules.append({'params': enc_params, 'lr': enc_lr})
        writer_modules = []
        for attr in ('jigsaw_stage2_writer_low_noise', 'jigsaw_stage2_writer_high_noise', 'jigsaw_stage2_writer'):
            m = getattr(self, attr, None)
            if m is None:
                continue
            if any(m is x for x in writer_modules):
                continue
            writer_modules.append(m)
        for m in writer_modules:
            writer_params = _register_params(list(m.parameters()))
            if len(writer_params) > 0:
                trainable_modules.append({'params': writer_params, 'lr': self.learning_rate * 2.0})
        missing_trainable = sorted(
            name for name, p in self.named_parameters()
            if p.requires_grad and name not in covered_named_params
        )
        if missing_trainable:
            raise RuntimeError(
                "Optimizer is missing trainable parameters: " + ", ".join(missing_trainable)
            )
        optimizer = torch.optim.AdamW(trainable_modules)
        return optimizer
    def _save_projector_viz(self):
        # Disabled to avoid checkpoint-adjacent I/O stalls in distributed training.
        return

    def _save_attn_viz(self):
        # Disabled to avoid checkpoint-adjacent I/O stalls in distributed training.
        return

    def on_save_checkpoint(self, checkpoint):
        # Disabled intentionally to avoid any trainer checkpoint hook side-effects under DDP/DeepSpeed.
        # LoRA-only periodic saving is handled in training_step.
        return


# =============================================================================
# Data Processing Model (same as v3)
# =============================================================================

class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, ckpt_dir,
                 tiled=False, tile_size=(34, 34), tile_stride=(18, 16), args=None):
        super().__init__()
        runtime_device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        self.pipe = build_wan22_training_pipe(
            ckpt_dir=ckpt_dir,
            device=runtime_device,
            torch_dtype=torch.bfloat16,
            task="i2v-A14B",
            train_noise_domain=(getattr(args, 'train_noise_domain', 'low_noise') if args else 'low_noise'),
        )
        _install_lightweight_pipeline_lifecycle(self.pipe)
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}


# =============================================================================
# Argument Parser
# =============================================================================

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
        ("slotmem_memory_writer_context_mode", "jigsaw_stage2_writer_context_mode"),
        ("slotmem_memory_writer_attention_scale", "jigsaw_stage2_writer_attention_scale"),
        ("slotmem_disable_memory_side_rope", "jigsaw_disable_memory_side_rope"),
    ):
        if hasattr(args, public_name):
            setattr(args, internal_name, getattr(args, public_name))
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="SlotMem training with inline memory extraction")
    
    # Paths
    parser.add_argument("--exp_prefix", default='', type=str)
    parser.add_argument("--dataset_path", type=str, default=None, help="Optional; used only by training-side fallback. Data comes from candidate_groups_csv + character_lists + video_root.")
    parser.add_argument("--sample_list_path", type=str, default=None)
    # Candidate dataset (唯一数据源)
    parser.add_argument("--candidate_groups_csv", type=str, default=None, help="Optional candidate_groups dataset source.")
    parser.add_argument("--character_lists_dir", type=str, default=None, help="Optional character_lists source.")
    parser.add_argument("--video_root", type=str, default=None, help="Optional video root source.")
    parser.add_argument("--story_root", type=str, required=True, help="SlotMem sample root.")
    parser.add_argument("--dataloader_skip_log", type=str, default=None, help="Log file for skipped samples (frame count / empty overlapping)")
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--ckpt_dir", type=str, default="/models/Wan2.2-I2V-A14B")
    parser.add_argument("--text_encoder_path", type=str, default=None)
    parser.add_argument("--image_encoder_path", type=str, default=None)
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--dit_path", type=str, default=None)
    # 在 parse_args 函数中找到 Memory extraction 部分，添加以下代码：
    parser.add_argument("--suffix_attention_scale", type=float, default=1.0,
                    help="Scale factor for attention scores of words after the first underscore. "
                         "Default 1.0 (equal weight). Set to 0.5 to halve their contribution.")
    
    # VAE tiling
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size_height", type=int, default=34)
    parser.add_argument("--tile_size_width", type=int, default=34)
    parser.add_argument("--tile_stride_height", type=int, default=18)
    parser.add_argument("--tile_stride_width", type=int, default=16)
    
    # Training
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum optimizer steps to run. Set <=0 to disable; when enabled, training stops when either max_steps or max_epochs is reached.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed. Use -1 for auto-random seed each run (default).")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    
    # LoRA
    parser.add_argument("--train_architecture", type=str, default="lora", choices=["lora", "full"])
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=float, default=128.0)
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2")
    parser.add_argument("--init_lora_weights", type=str, default="kaiming")
    parser.add_argument("--pretrained_lora_path", type=str, default=None)
    parser.add_argument("--train_noise_domain", type=str, default="low_noise", choices=["low_noise", "high_noise"],
                        help="Training domain selection. low_noise/high_noise loads and trains only that Wan noise expert.")
    parser.add_argument("--low_noise_lora_adapter_name", type=str, default="low_noise",
                        help="Adapter name used for low-noise timestep domain when dual-noise LoRA is enabled.")
    parser.add_argument("--high_noise_lora_adapter_name", type=str, default="high_noise",
                        help="Adapter name used for high-noise timestep domain when dual-noise LoRA is enabled.")
    parser.add_argument("--pretrained_lora_path_low_noise", type=str, default=None,
                        help="Optional pretrained LoRA checkpoint path for the low-noise adapter.")
    parser.add_argument("--pretrained_lora_path_high_noise", type=str, default=None,
                        help="Optional pretrained LoRA checkpoint path for the high-noise adapter.")
    parser.add_argument("--noise_domain_boundary_ratio", type=float, default=0.9,
                        help="Timestep percent boundary in [0,1] for domain split. p>=boundary -> high_noise, else low_noise.")
    
    # Resume from broken DeepSpeed checkpoint (load weights only, not optimizer state)
    parser.add_argument("--resume_weights_from_ds_ckpt", type=str, default=None,
                        help="Path to a (broken) DeepSpeed checkpoint directory to load trained weights from. "
                            "This loads LoRA, memory_pos_embed, and memory_segment_embed weights but NOT optimizer state.")
    
    # Gradient checkpointing
    parser.add_argument("--use_gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_gradient_checkpointing_offload", action=argparse.BooleanOptionalAction, default=True)
    
    # Checkpoint saving
    parser.add_argument("--checkpoint_save_every_n_epochs", type=int, default=0)
    parser.add_argument("--checkpoint_save_every_n_steps", type=int, default=200)
    
    # Error recycling (compatible with v2/v3)
    parser.add_argument("--use_first_aug", action="store_true")
    parser.add_argument("--use_error_recycling", action="store_true")
    parser.add_argument("--error_buffer_k", type=int, default=500)
    parser.add_argument("--timestep_grid_size", type=int, default=25)
    parser.add_argument("--num_grids", type=int, default=50)
    parser.add_argument("--buffer_replacement_strategy", type=str, default="random")
    parser.add_argument("--buffer_warmup_iter", type=int, default=50)
    parser.add_argument("--error_modulate_factor", type=float, default=0.0)
    parser.add_argument("--ref_pad_num", type=int, default=0)
    parser.add_argument("--num_motion_frames", type=int, default=1)
    parser.add_argument("--num_overlap_frame", type=int, default=5)
    parser.add_argument("--num_motion_latent", type=int, default=None)
    parser.add_argument("--p_motion_threshold", type=float, default=0.9)
    parser.add_argument("--y_error_num", type=int, default=1)
    parser.add_argument("--y_error_sample_from_all_grids", action="store_true")
    parser.add_argument("--y_error_sample_range", type=str, default=None)
    parser.add_argument("--noise_prob", type=float, default=0.9)
    parser.add_argument("--y_prob", type=float, default=0.9)
    parser.add_argument("--latent_prob", type=float, default=0.9)
    parser.add_argument("--clean_prob", type=float, default=0.1)
    parser.add_argument("--ref_pad_cfg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeat_first_frame", action="store_true")
    parser.add_argument("--clean_buffer_update_prob", type=float, default=0.1)
    parser.add_argument("--use_last_y_error", action="store_true")
    
    # Memory-only mode
    parser.add_argument("--train_memory_only", action="store_true")
    
    # Memory dropping for CFG training
    parser.add_argument("--memory_drop_prob", type=float, default=0.0,
                        help="Probability of dropping memory tokens (set to zero) for training. Default: 0.0")
    parser.add_argument("--negative_prompt", type=str, default="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                        help="Negative prompt used for unconditional CFG branch during training.")
    
    # Memory extraction
    parser.add_argument("--extract_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--role_token_selection_mode", type=str, default="two_role_diff",
                        choices=["baseline", "two_role_diff", "layer7_single"],
                        help="How to choose role tokens for training-time memory extraction and query probe. two_role_diff uses A-B / B-A token purification when exactly two valid roles are available. layer7_single ignores extract_layers and runs one parallel probe at sparse_role_memory_layer_idx with baseline plus per-role token-drop branches, then uses normal-drop(role) responses as that role's token map.")
    parser.add_argument("--top_visual_tokens", type=float, default=0.1)
    parser.add_argument("--top_visual_tokens_per_head", type=int, default=0)
    parser.add_argument("--otsu_scope", type=str, default="frame", choices=["clip", "frame"])
    parser.add_argument("--token_weight", type=float, default=1)
    parser.add_argument("--max_memory_tokens_per_character", type=int, default=512,
                        help="Max memory tokens per character. Default: 512.")
    parser.add_argument("--max_memory_characters", type=int, default=2, 
                        help="Max number of characters to extract memory for. -1 means unlimited. Default: 2")
    parser.add_argument("--cfg_scale_extraction", type=float, default=5.0,
                        help="CFG scale for attention extraction (default: 5.0)")
    parser.add_argument("--train_stage", type=str, default="stage1", choices=["stage1", "stage2"],
                        help="Manual training stage switch for SlotMem.")
    parser.add_argument("--slotmem_memory_bank_mode", dest="slotmem_memory_bank_mode", type=str, default="single", choices=["single", "legacy_multi"],
                        help="Memory bank mode for SlotMem. single uses exact timestep alignment and bank key '0'; legacy_multi preserves multi-bank nearest-percent behavior.")
    parser.add_argument("--train_query_probe_timestep_mode", type=str, default="current", choices=["current", "previous_scheduler"],
                        help="Timestep source for training character-semantic probe. Main DiT forward/loss always use the current training timestep.")
    parser.add_argument("--memory_bank_percents", type=str, default="0.85,0.60,0.35,0.12",
                        help="Legacy_multi-only comma-separated bank percentages in [0,1], used for multi-bank extraction and nearest-bank selection.")
    parser.add_argument("--neighbor_filter_kernel", type=int, default=5,
                        help="Kernel size for neighborhood block filter (0 to disable). Default: 5.")
    parser.add_argument("--neighbor_filter_any_window", action=argparse.BooleanOptionalAction, default=True,
                        help="If true, token is kept when any kernel×kernel block containing it has enough selected patches.")
    parser.add_argument("--enable_viz_extraction", action="store_true",
                        help="Enable periodic viz_extraction visualization (saves by_timestep heatmaps). Disabled by default to avoid slowdown.")
    parser.add_argument("--precompute_image_emb", action=argparse.BooleanOptionalAction, default=True,
                        help="Precompute image conditioning embeddings in extraction process and pass to training.")
    parser.add_argument("--precompute_image_emb_strict", action=argparse.BooleanOptionalAction, default=True,
                        help="Fail fast if precomputed image embedding is missing or generation fails.")
    parser.add_argument("--image_condition_people_pixelate_prob", type=float, default=0.0,
                        help="Probability (per training step) to pixelate people in all image conditions on CPU. 0 disables.")
    parser.add_argument("--image_condition_people_pixelate_model_path", type=str, default=DEFAULT_YOLO_SEG_CKPT,
                        help="YOLO segmentation checkpoint path used by CPU people pixelation.")
    parser.add_argument("--image_condition_people_pixelate_conf", type=float, default=0.25,
                        help="Confidence threshold for CPU people segmentation during condition pixelation.")
    parser.add_argument("--image_condition_people_pixelate_block_size", type=int, default=12,
                        help="Pixel block size for masked people region pixelation.")
    parser.add_argument("--image_condition_people_pixelate_mask_dilate_kernel", type=int, default=9,
                        help="Mask dilation kernel size for people mask before pixelation.")
    parser.add_argument("--keep_image_encoder_on_gpu", action=argparse.BooleanOptionalAction, default=False,
                        help="Keep image encoder resident on GPU during training. Default false to minimize peak memory.")
    parser.add_argument("--offload_image_encoder_after_extraction", action=argparse.BooleanOptionalAction, default=True,
                        help="After extraction-side precompute, move image encoder back to CPU to reduce training-time memory.")
    parser.add_argument("--aggressive_vram_optimization", action=argparse.BooleanOptionalAction, default=True,
                        help="Unified aggressive VRAM mode: on-demand VAE GPU residency during training and forced cache release between extraction phases.")
    parser.add_argument("--inline_progress_print_every", type=int, default=1,
                        help="Print inline extract-train progress every N samples after initial warmup.")
    parser.add_argument("--inline_progress_print_max_initial", type=int, default=20,
                        help="Always print progress for first K inline samples per rank.")
    parser.add_argument("--sync_before_yield", action=argparse.BooleanOptionalAction, default=True,
                        help="Synchronize all distributed ranks before yielding each inline batch.")
    parser.add_argument("--yield_sync_timeout_minutes", type=int, default=20,
                        help="Timeout minutes for pre-yield monitored barrier in inline mode.")
    parser.add_argument("--ddp_skip_sync_timeout_minutes", type=int, default=10,
                        help="Timeout minutes for training-step ddp skip synchronization all-reduce.")
    parser.add_argument("--enable_rank_heartbeat", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable per-rank heartbeat logs in inline dataset mode.")
    parser.add_argument("--rank_heartbeat_interval_sec", type=int, default=30,
                        help="Heartbeat print interval in seconds for inline dataset rank status.")
    
    # distributed / ddp
    parser.add_argument("--num_train_gpus", type=int, default=-1,
                        help="Number of GPUs for training (-1 for auto)")
    
    # Projector
    parser.add_argument("--use_learnable_memory_pos", action="store_true", help="Use learnable embeddings for memory instead of RoPE")
    parser.add_argument("--use_attn_score_selection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--projector_bottleneck", type=int, default=256)
    parser.add_argument("--use_projector", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable projector in memory forward/save/load path. Use --no-use_projector to force raw-memory path.")
    parser.add_argument("--use_segment_embed", action="store_true",
                        help="Enable shared memory segment embedding (shape [1,1,dim]) for all memory tokens")
    parser.add_argument("--latent_dim", type=int, default=16,
                        help="Latent dimension for StyleAwareMemoryProjector. Default 16 for WanVideoVAE.")
    parser.add_argument("--use_grouped_cross_attn", action="store_true",
                        help="Enable grouped cross-attention (text/memory separate attention then gated fusion). Default off for backward compatibility.")
    parser.add_argument("--memory_similarity_mode", type=str, default="projected", choices=["projected", "token", "hybrid", "hybrid_feature"],
                        help="Similarity mode in top1 fusion: projected=q/k projection, token=raw token cosine, hybrid=projected coarse retrieval + token rerank, hybrid_feature=use probe layer features as query and memory layer features as key.")
    parser.add_argument("--feature_match_layer_idx", type=int, default=7,
                        help="Layer index used to capture token feature vectors for hybrid_feature matching.")
    parser.add_argument("--feature_vector_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"],
                        help="Storage dtype for captured feature vectors in hybrid_feature mode.")
    parser.add_argument("--hybrid_feature_use_legacy_memory_key", action="store_true",
                        help="If set, hybrid_feature falls back to legacy key path (memory token as key) instead of extraction-layer feature key.")
    parser.add_argument("--enable_sparse_role_memory_attn", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable standalone sparse role-aware memory cross-attention.")
    parser.add_argument("--sparse_role_memory_layer_idx", type=int, default=3,
                        help="Target DiT layer index for sparse role-aware memory branch insertion.")
    parser.add_argument("--sparse_role_memory_injection_layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
                        help="Comma-separated DiT layer indices for sparse role-aware memory branch insertion. "
                             "Default: 0-15; if explicitly unset, falls back to --sparse_role_memory_layer_idx.")
    parser.add_argument("--char_attn_noise_scope", type=str, default="low_noise", choices=["high_noise", "low_noise"],
                        help="Enable char-attn + memory path on which noise domain(s). If low_noise/high_noise only, the other domain trains as base LoRA without memory/char-attn.")
    parser.add_argument("--sparse_role_memory_num_heads", type=int, default=8,
                        help="Number of heads for sparse role-aware memory branch.")
    parser.add_argument("--sparse_role_memory_head_dim", type=int, default=128,
                        help="Head dimension for sparse role-aware memory branch.")
    parser.add_argument("--sparse_role_memory_rope_dim", type=int, default=256,
                        help="RoPE dimension used in sparse role-aware memory branch q/k domain replacement.")
    parser.add_argument("--sparse_role_memory_use_half_role_heads", action=argparse.BooleanOptionalAction, default=True,
                        help="Use half heads with role-relative rope and half plain heads in sparse role-memory branch.")
    parser.add_argument("--sparse_role_memory_feature_source", type=str, default="attn_out", choices=["attn_out", "self_attn_out", "block_out"],
                        help="Feature tap source for probe/extraction feature payload. Default attn_out.")
    parser.add_argument("--sparse_role_memory_init_scale", type=float, default=0.1,
                        help="Initial residual scale for sparse role-memory branch output.")
    parser.add_argument("--sparse_role_memory_time_gate", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable timestep-dependent gate in sparse role-memory branch.")
    parser.add_argument("--sparse_role_memory_query_chunk_size", type=int, default=128,
                        help="Query chunk size for sparse role-memory attention. 0 keeps full-query behavior; >0 chunks only query axis while keeping full memory axis.")
    parser.add_argument("--sparse_role_memory_layer_scales", type=str, default="",
                        help="Optional comma map layer:scale applied after sparse role-memory injection.")
    parser.add_argument("--debug_sparse_role_memory_attn", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable debug stats for sparse role-aware memory branch.")
    parser.add_argument("--slotmem_memory_encoder_mode", dest="slotmem_memory_encoder_mode", type=str, default="on",
                        choices=["off", "on", "true", "1", "extra", "extra_encoder", "slotmem_memory_encoder", "contrastive_encoder"],
                        help="Enable SlotMem memory encoder over raw layerwise memory tokens before memory_projector.")
    parser.add_argument("--slotmem_memory_encoder_layers", dest="slotmem_memory_encoder_layers", type=str, default="0-15",
                        help="Layers captured/injected by SlotMem memory encoder mode. Default: 0-15.")
    parser.add_argument("--slotmem_memory_encoder_layer_groups", dest="slotmem_memory_encoder_layer_groups", type=str, default="0-4,5-10,11-15",
                        help="Comma-separated layer groups for the three SlotMem memory encoders.")
    parser.add_argument("--slotmem_memory_encoder_slots", dest="slotmem_memory_encoder_slots", type=int, default=64,
                        help="Query slots per role per layer after SlotMem memory encoder compression.")
    parser.add_argument("--slotmem_memory_encoder_dim", dest="slotmem_memory_encoder_dim", type=int, default=512,
                        help="Internal dimension of the SlotMem memory encoder.")
    parser.add_argument("--slotmem_memory_encoder_hidden_dim", dest="slotmem_memory_encoder_hidden_dim", type=int, default=1024,
                        help="Hidden dimension of the SlotMem memory encoder FFN.")
    parser.add_argument("--slotmem_memory_encoder_use_t_embed", dest="slotmem_memory_encoder_use_t_embed", action=argparse.BooleanOptionalAction, default=True,
                        help="Condition SlotMem memory encoder compression on the current DiT timestep embedding.")
    parser.add_argument("--slotmem_memory_encoder_use_slot_index_embed", dest="slotmem_memory_encoder_use_slot_index_embed", action=argparse.BooleanOptionalAction, default=True,
                        help="Add learned slot-index identity embeddings after SlotMem memory encoder compression.")
    parser.add_argument("--slotmem_memory_encoder_aux_weight", dest="slotmem_memory_encoder_aux_weight", type=float, default=0.05,
                        help="Weight for SlotMem memory encoder role/background auxiliary contrastive CE loss.")
    parser.add_argument("--slotmem_memory_encoder_bg_tokens", dest="slotmem_memory_encoder_bg_tokens", type=int, default=64,
                        help="Detached non-query current-layer tokens sampled as background negatives for aux loss.")
    parser.add_argument("--slotmem_memory_writer_mode", dest="slotmem_memory_writer_mode", type=str, default="auto",
                        choices=["auto", "off", "on", "true", "1", "false", "0", "none", "residual"],
                        help="Stage2 stateless memory writer mode. auto enables residual writer only for --train_stage stage2.")
    parser.add_argument("--slotmem_memory_writer_hidden_dim", dest="slotmem_memory_writer_hidden_dim", type=int, default=1024,
                        help="Hidden dimension for the stage2 memory writer MLPs.")
    parser.add_argument("--slotmem_memory_writer_init_scale", dest="slotmem_memory_writer_init_scale", type=float, default=0.1,
                        help="Initial residual scale for the stage2 writer.")
    parser.add_argument("--slotmem_memory_writer_precision_tau", dest="slotmem_memory_writer_precision_tau", type=float, default=0.3,
                        help="Cosine precision-gate threshold for the stage2 writer.")
    parser.add_argument("--slotmem_memory_writer_precision_scale", dest="slotmem_memory_writer_precision_scale", type=float, default=10.0,
                        help="Cosine precision-gate sigmoid scale for the stage2 writer.")
    parser.add_argument("--slotmem_memory_writer_max_delta_ratio", dest="slotmem_memory_writer_max_delta_ratio", type=float, default=0.0,
                        help="Optional writer residual norm cap relative to slot norm. <=0 disables.")
    parser.add_argument("--slotmem_memory_writer_max_delta_norm", dest="slotmem_memory_writer_max_delta_norm", type=float, default=0.0,
                        help="Optional absolute writer residual norm cap. <=0 disables.")
    parser.add_argument("--slotmem_memory_writer_detach_c_short", dest="slotmem_memory_writer_detach_c_short", action=argparse.BooleanOptionalAction, default=True,
                        help="Detach current-layer short context before stage2 writer pooling.")
    parser.add_argument("--slotmem_memory_writer_context_mode", dest="slotmem_memory_writer_context_mode", type=str, default="mean",
                        choices=["mean", "avg", "average", "attention", "attn", "slot_attn", "slot_attention", "cross_attn", "cross_attention"],
                        help="How stage2 writer builds per-slot short context from same-role update slots.")
    parser.add_argument("--slotmem_memory_writer_attention_scale", dest="slotmem_memory_writer_attention_scale", type=float, default=10.0,
                        help="Cosine-attention logit scale when --slotmem_memory_writer_context_mode uses slot attention.")
    parser.add_argument("--stage2_memory_encoder_lr", type=float, default=1e-5,
                        help="Stage2 learning rate for unfrozen SlotMem memory encoder parameters.")
    parser.add_argument("--stage2_char_attn_lr", type=float, default=1e-5,
                        help="Stage2 learning rate for unfrozen sparse role-memory char attention parameters.")
    parser.add_argument("--stage2_unfreeze_memory_encoder", action=argparse.BooleanOptionalAction, default=True,
                        help="In stage2, keep the stage1 SlotMem memory encoder trainable instead of freezing it.")
    parser.add_argument("--stage2_unfreeze_char_attn", action=argparse.BooleanOptionalAction, default=True,
                        help="In stage2, keep the stage1 sparse role-memory char attention trainable instead of freezing it.")
    parser.add_argument("--stage2_stage1_branch_prob", type=float, default=0.2,
                        help="Stage2 replay probability: skip the memory writer and train the stage1-style memory injection branch on the same stage2 triplet batch.")
    parser.add_argument("--slotmem_disable_memory_side_rope", dest="slotmem_disable_memory_side_rope", action=argparse.BooleanOptionalAction, default=True,
                        help="Disable memory-side key RoPE and merge value RoPE-center addition in SlotMem memory encoder scripts.")
    parser.add_argument("--train_query_bbox_probe_use_eval_mode", action=argparse.BooleanOptionalAction, default=True,
                        help="Temporarily set denoising model to eval() during character-semantic probe.")
    parser.add_argument("--train_query_bbox_probe_use_no_grad", action=argparse.BooleanOptionalAction, default=True,
                        help="Use torch.no_grad() for character-semantic probe path.")
    parser.add_argument("--train_query_bbox_probe_debug", action=argparse.BooleanOptionalAction, default=False,
                        help="Print debug info for character-semantic probe path.")
    parser.add_argument("--probe_bbox_loss_weight_x2", action="store_true",
                        help="If enabled, multiply loss weight by 2 inside character probe regions for each latent t.")
    parser.add_argument("--use_detached_local_probe_for_zero3", action=argparse.BooleanOptionalAction, default=False,
                        help="Run character-semantic probe on a detached local full model under ZeRO-3.")
    parser.add_argument("--use_train_weights_for_extract_and_probe", action=argparse.BooleanOptionalAction, default=True,
                        help="When enabled, extraction/probe use current training-model weights (including LoRA) in eval/no-grad style paths and do not use detached local probe/extractor routes.")
    parser.add_argument("--probe_on_tp_leader_only", action=argparse.BooleanOptionalAction, default=True,
                        help="When detached probe is enabled, run probe only on TP leader and broadcast query boxes.")
    # Trainer
    parser.add_argument("--training_strategy", type=str, default="ddp")
    parser.add_argument("--model_slice_mode", type=str, default="none", choices=["none", "zero3"],
                        help="Model slicing/sharding mode. zero3 enables DeepSpeed ZeRO-3 parameter partitioning.")
    parser.add_argument("--use_detached_local_extractor_for_zero3", action=argparse.BooleanOptionalAction, default=False,
                        help="When true, extraction dataset builds a detached local full extractor pipe (DiT+VAE+encoders) under ZeRO-3.")
    parser.add_argument("--use_detached_local_vae_for_zero3", action=argparse.BooleanOptionalAction, default=False,
                        help="When true, extraction dataset builds a detached local VAE/text/image pipe under ZeRO-3. Default false to avoid duplicate model loading cost.")
    parser.add_argument("--offload_detached_extractor_after_extraction", action=argparse.BooleanOptionalAction, default=None,
                        help="Offload detached extractor DiT/text encoder to CPU between samples to reduce peak GPU memory. Default follows --aggressive_vram_optimization.")
    parser.add_argument("--tp_size", type=int, default=1,
                        help="Target tensor-parallel size. Current implementation uses it to configure ZeRO-3 partition groups.")
    parser.add_argument("--zero3_allow_unsafe_partition_size", action=argparse.BooleanOptionalAction, default=False,
                        help="Allow zero3_partition_size>1 even though this runtime may be unstable. Use only for TP2+DP2 experiments.")
    parser.add_argument("--zero3_offload_optimizer", action=argparse.BooleanOptionalAction, default=False,
                        help="When model_slice_mode=zero3, offload optimizer states to CPU.")
    parser.add_argument("--zero3_offload_parameters", action=argparse.BooleanOptionalAction, default=False,
                        help="When model_slice_mode=zero3, offload parameters to CPU (slower, lower GPU memory).")
    parser.add_argument("--zero3_reduce_bucket_size_mb", type=int, default=200,
                        help="ZeRO-3 reduce bucket size in MB.")
    parser.add_argument("--zero3_allgather_bucket_size_mb", type=int, default=200,
                        help="ZeRO-3 all-gather bucket size in MB.")
    parser.add_argument("--zero3_stage3_param_persistence_threshold", type=int, default=1000000,
                        help="ZeRO-3 stage3_param_persistence_threshold. Larger values keep more small params replicated.")
    parser.add_argument("--zero3_stage3_module_granularity_threshold", type=int, default=0,
                        help="ZeRO-3 stage3_module_granularity_threshold.")
    parser.add_argument("--zero3_sub_group_size", type=float, default=1e9,
                        help="ZeRO-3 sub_group_size.")
    parser.add_argument("--zero3_patch_secondary_partition_guard", action=argparse.BooleanOptionalAction, default=True,
                        help="Install runtime guard for ZeRO-3 secondary partition narrow overflow (IndexError: start out of range).")
    parser.add_argument("--zero3_partition_size", type=int, default=1,
                        help="ZeRO-3 partition group size. Set 2 for 4 GPUs -> two 2-GPU shard groups (experimental).")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--use_swanlab", action="store_true")
    parser.add_argument("--swanlab_mode", default=None)
    
    # Debug
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--perf_log_interval", type=int, default=0,
                        help="Print perf logs every N steps/batches (0 to disable).")
    parser.add_argument("--debug_runtime_print", action="store_true",
                        help="Print runtime tensor stats/shapes for train-test consistency debugging.")
    parser.add_argument("--debug_runtime_interval", type=int, default=20,
                        help="Runtime debug print interval in global steps (when --debug_runtime_print is enabled).")
    parser.add_argument("--enable_attn_monitor", action="store_true",
                        help="Enable per-layer attention response monitoring hooks (disabled by default for speed).")
    parser.add_argument("--cuda_trim_interval", type=int, default=10,
                        help="Check CUDA cache fragmentation every N train steps; trim only when large fragmentation is detected.")
    parser.add_argument("--cuda_trim_min_fragment_mb", type=int, default=1536,
                        help="Minimum reserved-allocated gap (MB) required before triggering cache trim.")
    
    return _populate_slotmem_internal_aliases(parser.parse_args())


# =============================================================================
# Main Training Function
# =============================================================================

def _normalize_role_wise_slot_memory_bank_args(args):
    if args is None:
        return args
    setattr(args, 'jigsaw_memory_extract_timestep_mode', 'aligned')
    setattr(args, 'jigsaw_memory_encoder_t_embed_source', 'current')
    mode = str(getattr(args, 'jigsaw_memory_bank_mode', 'single') or 'single').strip().lower()
    if mode not in ('single', 'legacy_multi'):
        mode = 'single'
    setattr(args, 'jigsaw_memory_bank_mode', mode)
    if mode == 'legacy_multi':
        bank_percents = str(getattr(args, 'memory_bank_percents', '0.85,0.60,0.35,0.12') or '0.85,0.60,0.35,0.12')
        setattr(args, 'memory_bank_percents', bank_percents)
        setattr(args, 'jigsaw_memory_bank_effective_percents', bank_percents)
    else:
        setattr(args, 'memory_bank_percents', '')
        setattr(args, 'jigsaw_memory_bank_effective_percents', 'single uses per-sample p_extract')
    return args


def train_slotmem(args):
    args = _normalize_role_wise_slot_memory_bank_args(args)
    if _memory_encoder_enabled(getattr(args, 'jigsaw_extra_encoder_mode', 'off')):
        wide_layer_list = _jigsaw_parse_layer_list(getattr(args, 'jigsaw_extra_encoder_layers', '0-15'))
        wide_layers_csv = ",".join(str(x) for x in wide_layer_list)
        args.extract_layers = wide_layer_list
        args.sparse_role_memory_injection_layers = wide_layers_csv
        print(
            f"[MemoryEncoder][Args] forced raw layerwise capture/injection layers={wide_layer_list}; raw slot memory",
            flush=True,
        )
    assert str(args.char_attn_noise_scope).strip().lower() == str(args.train_noise_domain).strip().lower(), (
        f"char_attn_noise_scope ({args.char_attn_noise_scope}) must equal "
        f"train_noise_domain ({args.train_noise_domain})"
    )
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        if 0 <= local_rank < torch.cuda.device_count():
            torch.cuda.set_device(local_rank)
    if getattr(args, 'seed', -1) is None or int(getattr(args, 'seed', -1)) < 0:
        args.seed = int(secrets.randbits(32))
    args.seed = int(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[Seed] Using run seed: {args.seed}")

    def _infer_resume_step_from_lora_path(lora_path):
        if not lora_path:
            return 0
        try:
            name = Path(str(lora_path)).name
            matched = re.search(r"step_(\d+)", name)
            if matched:
                return int(matched.group(1))
            return 0
        except Exception:
            return 0

    """Main training function with per-rank inline extract-then-train mechanism."""

    total_gpus = torch.cuda.device_count()
    if total_gpus <= 0:
        raise RuntimeError("No CUDA GPU available for training")

    requested_train_gpus = int(args.num_train_gpus) if int(args.num_train_gpus) > 0 else total_gpus
    trainer_devices = min(requested_train_gpus, total_gpus)
    if trainer_devices <= 0:
        raise RuntimeError(f"Invalid trainer devices: {trainer_devices}")

    # New execution mode: one process per GPU does extraction then training locally.
    timing_tracker = TimingTracker(window_size=4)

    env_rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
    env_world_size = int(os.environ.get('WORLD_SIZE', str(trainer_devices)))
    tp_size = max(1, int(getattr(args, 'tp_size', 1)))
    if trainer_devices != env_world_size:
        print(
            f"[Mode][Warning] trainer_devices ({trainer_devices}) != env_world_size ({env_world_size}). "
            "Prefer launching with torchrun/deepspeed so they match.",
            flush=True,
        )
    if env_world_size % tp_size != 0:
        raise ValueError(f"Invalid topology: world_size={env_world_size} is not divisible by tp_size={tp_size}")
    dp_size = env_world_size // tp_size

    print(
        f"[Mode] Inline extract-then-train enabled. trainer_devices={trainer_devices}, "
        f"rank={env_rank}, world_size={env_world_size}, tp_size={tp_size}, dp_size={dp_size}",
        flush=True,
    )

    dataset_config = {
        'dataset_path': args.dataset_path or args.output_path or ".",
        'sample_list_path': args.sample_list_path,
        'num_workers': 0,
        'seed': args.seed,
        'dataset_args': {
            'num_frames': args.num_frames,
            'height': args.height,
            'width': args.width,
            'sample_list_path': args.sample_list_path,
            'train_stage': args.train_stage,
            'require_stage2_triplets': bool(str(args.train_stage).strip().lower() == 'stage2' and _memory_writer_effective_mode(args.train_stage, args.jigsaw_stage2_writer_mode) != 'off'),
        },
        'candidate_groups_csv': args.candidate_groups_csv,
        'character_lists_dir': args.character_lists_dir,
        'video_root': args.video_root,
        'story_root': args.story_root,
        'dataloader_skip_log': getattr(args, 'dataloader_skip_log', None),
    }
    model_paths = {
        'ckpt_dir': args.ckpt_dir,
        'text_encoder_path': args.text_encoder_path,
        'vae_path': args.vae_path,
        'image_encoder_path': args.image_encoder_path,
        'dit_path': args.dit_path,
        'train_architecture': args.train_architecture,
    }
    tiler_kwargs = {
        'tiled': getattr(args, 'tiled', False),
        'tile_size': (getattr(args, 'tile_size_height', 32), getattr(args, 'tile_size_width', 32)),
        'tile_stride': (getattr(args, 'tile_stride_height', 18), getattr(args, 'tile_stride_width', 16))
    }
    extract_config = {
        'extract_layers': args.extract_layers,
        'role_token_selection_mode': args.role_token_selection_mode,
        'top_visual_tokens': args.top_visual_tokens,
        'top_visual_tokens_per_head': args.top_visual_tokens_per_head,
        'otsu_scope': args.otsu_scope,
        'token_weight': args.token_weight,
        'max_tokens': args.max_memory_tokens_per_character,
        'max_memory_characters': args.max_memory_characters,
        'cfg_scale': args.cfg_scale_extraction,
        'memory_similarity_mode': args.memory_similarity_mode,
        'enable_sparse_role_memory_attn': args.enable_sparse_role_memory_attn,
        'sparse_role_memory_injection_layers': args.sparse_role_memory_injection_layers,
        'sparse_role_memory_layer_scales': args.sparse_role_memory_layer_scales,
        'feature_match_layer_idx': args.feature_match_layer_idx,
        'feature_vector_dtype': args.feature_vector_dtype,
        'hybrid_feature_use_legacy_memory_key': args.hybrid_feature_use_legacy_memory_key,
        'sparse_role_memory_feature_source': args.sparse_role_memory_feature_source,
        'jigsaw_extra_encoder_mode': args.jigsaw_extra_encoder_mode,
        'jigsaw_extra_encoder_layers': args.jigsaw_extra_encoder_layers,
        'jigsaw_extra_encoder_layer_groups': args.jigsaw_extra_encoder_layer_groups,
        'jigsaw_extra_encoder_slots': args.jigsaw_extra_encoder_slots,
        'jigsaw_extra_encoder_dim': args.jigsaw_extra_encoder_dim,
        'jigsaw_extra_encoder_hidden_dim': args.jigsaw_extra_encoder_hidden_dim,
        'jigsaw_extra_encoder_use_t_embed': args.jigsaw_extra_encoder_use_t_embed,
        'jigsaw_memory_encoder_t_embed_source': args.jigsaw_memory_encoder_t_embed_source,
        'jigsaw_extra_encoder_use_slot_index_embed': args.jigsaw_extra_encoder_use_slot_index_embed,
        'jigsaw_extra_encoder_aux_weight': args.jigsaw_extra_encoder_aux_weight,
        'jigsaw_extra_encoder_bg_tokens': args.jigsaw_extra_encoder_bg_tokens,
        'jigsaw_stage2_writer_mode': args.jigsaw_stage2_writer_mode,
        'memory_writer_effective_mode': _memory_writer_effective_mode(
            args.train_stage,
            args.jigsaw_stage2_writer_mode,
        ),
        'jigsaw_stage2_writer_hidden_dim': args.jigsaw_stage2_writer_hidden_dim,
        'jigsaw_stage2_writer_init_scale': args.jigsaw_stage2_writer_init_scale,
        'jigsaw_stage2_writer_precision_tau': args.jigsaw_stage2_writer_precision_tau,
        'jigsaw_stage2_writer_precision_scale': args.jigsaw_stage2_writer_precision_scale,
        'jigsaw_stage2_writer_max_delta_ratio': args.jigsaw_stage2_writer_max_delta_ratio,
        'jigsaw_stage2_writer_max_delta_norm': args.jigsaw_stage2_writer_max_delta_norm,
        'jigsaw_stage2_writer_detach_c_short': args.jigsaw_stage2_writer_detach_c_short,
        'jigsaw_stage2_writer_context_mode': args.jigsaw_stage2_writer_context_mode,
        'jigsaw_stage2_writer_attention_scale': args.jigsaw_stage2_writer_attention_scale,
        'stage2_memory_encoder_lr': args.stage2_memory_encoder_lr,
        'stage2_char_attn_lr': args.stage2_char_attn_lr,
        'stage2_unfreeze_memory_encoder': args.stage2_unfreeze_memory_encoder,
        'stage2_unfreeze_char_attn': args.stage2_unfreeze_char_attn,
        'stage2_stage1_branch_prob': args.stage2_stage1_branch_prob,
        'tiler_kwargs': tiler_kwargs,
        'suffix_attention_scale': args.suffix_attention_scale,
        'train_stage': args.train_stage,
        'jigsaw_memory_bank_mode': args.jigsaw_memory_bank_mode,
        'jigsaw_memory_extract_timestep_mode': getattr(args, 'jigsaw_memory_extract_timestep_mode', 'aligned'),
        'train_query_probe_timestep_mode': args.train_query_probe_timestep_mode,
        'jigsaw_memory_bank_effective_percents': args.jigsaw_memory_bank_effective_percents,
        'memory_bank_percents': args.memory_bank_percents,
        'neighbor_filter_kernel': args.neighbor_filter_kernel,
        'neighbor_filter_any_window': args.neighbor_filter_any_window,
        'enable_viz_extraction': False,
        'perf_log_interval': args.perf_log_interval,
        'max_ref_frames_for_train': max(1, int(args.num_motion_frames), int(args.num_overlap_frame)),
        'precompute_image_emb': args.precompute_image_emb,
        'precompute_image_emb_strict': args.precompute_image_emb_strict,
        'offload_image_encoder_after_extraction': args.offload_image_encoder_after_extraction,
        'aggressive_vram_optimization': args.aggressive_vram_optimization,
        'offload_detached_extractor_after_extraction': (
            args.offload_detached_extractor_after_extraction
            if args.offload_detached_extractor_after_extraction is not None
            else args.aggressive_vram_optimization
        ),
        'inline_progress_print_every': args.inline_progress_print_every,
        'inline_progress_print_max_initial': args.inline_progress_print_max_initial,
        'sync_before_yield': args.sync_before_yield,
        'yield_sync_timeout_minutes': args.yield_sync_timeout_minutes,
        'enable_rank_heartbeat': args.enable_rank_heartbeat,
        'rank_heartbeat_interval_sec': args.rank_heartbeat_interval_sec,
        'num_motion_frames': args.num_motion_frames,
        'num_overlap_frame': args.num_overlap_frame,
        'num_motion_latent': args.num_motion_latent,
        'p_motion_threshold': args.p_motion_threshold,
        'repeat_first_frame': args.repeat_first_frame,
        'use_first_aug': args.use_first_aug,
        'ref_pad_cfg': args.ref_pad_cfg,
        'ref_pad_num': args.ref_pad_num,
        'resume_step_from_lora': _infer_resume_step_from_lora_path(getattr(args, 'pretrained_lora_path', None)),
        'training_strategy': args.training_strategy,
        'model_slice_mode': args.model_slice_mode,
        'use_detached_local_extractor_for_zero3': args.use_detached_local_extractor_for_zero3,
        'use_detached_local_vae_for_zero3': args.use_detached_local_vae_for_zero3,
        'use_train_weights_for_extract_and_probe': args.use_train_weights_for_extract_and_probe,
        'disable_lora_during_extraction': (not args.use_train_weights_for_extract_and_probe),
        'train_noise_domain': args.train_noise_domain,
        'noise_domain_boundary_ratio': args.noise_domain_boundary_ratio,
        'probe_bbox_loss_weight_x2': args.probe_bbox_loss_weight_x2,
        'require_memory_sample': bool(args.enable_sparse_role_memory_attn or args.probe_bbox_loss_weight_x2),
    }

    # Model initialization
    model = LightningModelForTrainWithMemoryV4(
        ckpt_dir=args.ckpt_dir,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
        model_VAE=None,
        args=args,
        projector_bottleneck=args.projector_bottleneck,
        latent_dim=args.latent_dim,
        timing_tracker=timing_tracker,
    )

    file_dataset = InlineExtractThenTrainDataset(
        dataset_config=dataset_config,
        model_paths=model_paths,
        extract_config=extract_config,
        rank=env_rank,
        world_size=env_world_size,
        tp_size=tp_size,
        dp_rank=(env_rank // tp_size),
        tp_rank=(env_rank % tp_size),
        debug=args.debug,
        shared_train_pipe=model.pipe,
        shared_vae_pipe=model.pipe_VAE,
    )
    from torch.utils.data import DataLoader
    dataloader = DataLoader(file_dataset, batch_size=1, num_workers=0, collate_fn=skip_none_collate)
    
    # -------------------------------------------------------------------------
    # Trainer initialization
    # -------------------------------------------------------------------------
    print(f"[Trainer] Initializing with {trainer_devices} devices")
    
    # 使用 TensorBoardLogger 确保 lightning_logs/version_N
    from lightning.pytorch.loggers import TensorBoardLogger
    tb_logger = TensorBoardLogger(save_dir=args.output_path, name="lightning_logs")
    
    # barrier 后第一次 collective（如梯度 all-reduce）易触发 NCCL 超时，将进程组超时设为 60 分钟
    # model_slice_mode=zero3：参数切片（ZeRO-3）
    strategy_name = str(getattr(args, "training_strategy", "auto") or "auto").strip().lower()
    slice_mode = str(getattr(args, "model_slice_mode", "none") or "none").strip().lower()
    if slice_mode == "zero3" and strategy_name != "deepspeed_stage_3":
        strategy_name = "deepspeed_stage_3"

    if strategy_name in ("deepspeed_stage_2", "deepspeed_stage_3"):
        from lightning.pytorch.strategies import DeepSpeedStrategy
        if strategy_name == "deepspeed_stage_3":
            install_deepspeed_zero_secondary_partition_guard(
                enabled=bool(getattr(args, 'zero3_patch_secondary_partition_guard', True))
            )
            reduce_bucket_bytes = int(max(1, int(getattr(args, 'zero3_reduce_bucket_size_mb', 200))) * 1024 * 1024)
            allgather_bucket_bytes = int(max(1, int(getattr(args, 'zero3_allgather_bucket_size_mb', 200))) * 1024 * 1024)
            requested_partition_size = int(max(1, int(getattr(args, 'zero3_partition_size', 1))))
            allow_unsafe_partition = bool(getattr(args, 'zero3_allow_unsafe_partition_size', False))
            if requested_partition_size > 1 and tp_size > 1 and requested_partition_size != tp_size:
                print(
                    f"[Trainer][Warning] zero3_partition_size ({requested_partition_size}) != tp_size ({tp_size}). "
                    "This may break TP-local assumptions for inline sampling/reproducibility.",
                    flush=True,
                )
            if requested_partition_size == 1 and tp_size > 1:
                print(
                    f"[Trainer][Hint] tp_size={tp_size} but zero3_partition_size=1. "
                    "If you need 2-GPU model shards, try --zero3_partition_size {tp_size} "
                    "with --zero3_allow_unsafe_partition_size (experimental).",
                    flush=True,
                )
            # NOTE:
            # zero_hpz_partition_size>1 currently triggers IndexError in this runtime
            # (DeepSpeed ZeRO-3 parameter partition init). Keep arg for compatibility,
            # but force safe fallback to avoid startup crash.
            effective_partition_size = 1
            if (
                allow_unsafe_partition and
                requested_partition_size > 1 and
                trainer_devices % requested_partition_size == 0
            ):
                effective_partition_size = requested_partition_size
                print(
                    "[Trainer][Warning] Using unsafe zero3_partition_size>1 experimental mode. "
                    "If startup fails, rerun with --no-zero3_allow_unsafe_partition_size.",
                    flush=True,
                )
            elif requested_partition_size > 1 or (trainer_devices % requested_partition_size != 0):
                print(
                    "[Trainer][Warning] Requested zero3_partition_size is unsupported in current runtime "
                    f"(requested={requested_partition_size}, trainer_devices={trainer_devices}). "
                    "Falling back to partition_size=1.",
                    flush=True,
                )

            ds_config = {
                "bf16": {"enabled": True},
                "zero_force_ds_cpu_optimizer": False,
                "zero_optimization": {
                    "stage": 3,
                    "offload_optimizer": {
                        "device": "cpu" if bool(getattr(args, 'zero3_offload_optimizer', False)) else "none"
                    },
                    "offload_param": {
                        "device": "cpu" if bool(getattr(args, 'zero3_offload_parameters', False)) else "none"
                    },
                    "reduce_bucket_size": reduce_bucket_bytes,
                    "allgather_bucket_size": allgather_bucket_bytes,
                    "sub_group_size": float(getattr(args, 'zero3_sub_group_size', 1e9)),
                    "stage3_param_persistence_threshold": int(getattr(args, 'zero3_stage3_param_persistence_threshold', 1000000)),
                    "stage3_module_granularity_threshold": int(getattr(args, 'zero3_stage3_module_granularity_threshold', 0)),
                    # Keep safe value due runtime bug with >1.
                    "zero_hpz_partition_size": effective_partition_size,
                },
            }

            _strategy = DeepSpeedStrategy(
                config=ds_config,
                timeout=timedelta(minutes=60),
            )
            print(
                "[Trainer] Model slicing enabled: DeepSpeed ZeRO-3 "
                f"(offload_optimizer={bool(getattr(args, 'zero3_offload_optimizer', False))}, "
                f"offload_parameters={bool(getattr(args, 'zero3_offload_parameters', False))}, "
                f"reduce_bucket_mb={int(getattr(args, 'zero3_reduce_bucket_size_mb', 200))}, "
                f"allgather_bucket_mb={int(getattr(args, 'zero3_allgather_bucket_size_mb', 200))}, "
                f"partition_size_requested={requested_partition_size}, "
                f"partition_size_effective={effective_partition_size}, "
                f"data_parallel_groups={trainer_devices // effective_partition_size})",
                flush=True,
            )
        else:
            _strategy = DeepSpeedStrategy(timeout=timedelta(minutes=60))
    else:
        if (
            strategy_name in ("ddp", "auto")
            and _memory_encoder_enabled(getattr(args, "jigsaw_extra_encoder_mode", "off"))
            and int(trainer_devices) > 1
        ):
            from lightning.pytorch.strategies import DDPStrategy
            _strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(minutes=60))
            print(
                "[Trainer] SlotMem memory encoder uses conditional memory paths; "
                "enabled DDP find_unused_parameters=True.",
                flush=True,
            )
        else:
            _strategy = getattr(args, "training_strategy", "auto")
    
    trainer_max_steps = int(getattr(args, 'max_steps', -1))
    if trainer_max_steps <= 0:
        trainer_max_steps = -1

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=trainer_max_steps,
        accelerator="gpu",
        devices=trainer_devices,
        num_nodes=args.num_nodes,
        enable_checkpointing=False,
        enable_progress_bar=False,
        precision="bf16",
        strategy=_strategy,
        default_root_dir=args.output_path,
        logger=tb_logger,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        limit_train_batches=args.steps_per_epoch,
        callbacks=[],
    )
    
    def _signal_shutdown_now():
        return

    try:
        trainer.fit(model, dataloader, ckpt_path=args.ckpt_path)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        _signal_shutdown_now()
    except RuntimeError as e:
        if "Shutdown requested" in str(e):
            print("\n[Main] Shutdown requested (another process exited).")
        else:
            _signal_shutdown_now()
            raise
    except Exception:
        _signal_shutdown_now()
        raise
    finally:
        pass


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    args = parse_args()

    def _resolve_ckpt_layout(_args):
        ckpt_dir = str(getattr(_args, "ckpt_dir", "") or "").strip()
        if not ckpt_dir:
            raise ValueError("--ckpt_dir is required")
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Wan2.2 checkpoint dir not found: {ckpt_dir}")

        _args.text_encoder_path = os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        _args.vae_path = os.path.join(ckpt_dir, "Wan2.1_VAE.pth")
        _args.image_encoder_path = None
        _args.dit_path = os.path.join(ckpt_dir, "low_noise_model")

        if not os.path.exists(_args.text_encoder_path):
            raise FileNotFoundError(f"Missing text encoder checkpoint: {_args.text_encoder_path}")
        if not os.path.exists(_args.vae_path):
            raise FileNotFoundError(f"Missing VAE checkpoint: {_args.vae_path}")
        if not os.path.isdir(os.path.join(ckpt_dir, "low_noise_model")):
            raise FileNotFoundError(f"Missing low_noise_model under checkpoint dir: {ckpt_dir}")
        if not os.path.isdir(os.path.join(ckpt_dir, "high_noise_model")):
            raise FileNotFoundError(f"Missing high_noise_model under checkpoint dir: {ckpt_dir}")

        print(f"[PathCheck] ckpt_dir={ckpt_dir}", flush=True)
        print(f"[PathCheck] text_encoder_path={_args.text_encoder_path}", flush=True)
        print(f"[PathCheck] vae_path={_args.vae_path}", flush=True)
        print(f"[PathCheck] story_root={getattr(_args, 'story_root', None)}", flush=True)

    _resolve_ckpt_layout(args)
    
    # Parse extract_layers from string to list of integers
    if isinstance(args.extract_layers, str):
        if args.extract_layers.strip() == '-1':
            args.extract_layers = [-1]
        else:
            args.extract_layers = [int(x.strip()) for x in args.extract_layers.split(',')]
    selection_mode = str(getattr(args, 'role_token_selection_mode', 'baseline')).strip().lower()
    if selection_mode == 'layer7_single':
        print("[Args] role_token_selection_mode=layer7_single: extract_layers is ignored for extraction/query probe token selection.", flush=True)
    elif selection_mode == 'two_role_diff':
        print("[Args] role_token_selection_mode=two_role_diff: two-role A-B / B-A purification is enabled when exactly two valid roles are available.", flush=True)
    args = _normalize_role_wise_slot_memory_bank_args(args)
    
    args = update_experiment_path(args, short=True)
    print_args(args)
    save_args_to_yaml(args, args.output_path)
    
    train_slotmem(args)
