"""Reference inference runtime helpers.

This module is intentionally import-only. It keeps the shared runtime helpers
used by the Wan2.2 inference entry scripts, plus the
``ReferenceInferenceRuntime.generate_chunk`` implementation they inherit.
Standalone CLI, old projector modules, and unused base inference methods were
removed.
"""

import glob
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict, deque

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    from attention_probe_utils import MultiCharacterAttentionMapExtractor
except ImportError:
    MultiCharacterAttentionMapExtractor = None

__all__ = [
    "ReferenceInferenceRuntime",
    "AttentionMapExtractorV8",
    "MemoryManager",
    "merge_chunk_videos",
    "pick_nearest_bank_by_percent",
    "resolve_reference_image_path",
    "save_chunk_memory_visualization",
    "save_denoise_step_edge_frames_visualization",
    "save_denoise_step_visualization",
    "save_feature_mapping_visualization",
]


def pick_nearest_bank_by_percent(current_percent, bank_percents):
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


def resolve_reference_image_path(cli_ref_image_path, json_path, json_data=None):
    """Resolve reference image path in a v8-compatible order with safe fallbacks."""
    if isinstance(cli_ref_image_path, str) and cli_ref_image_path.strip() != "":
        p = cli_ref_image_path.strip()
        if os.path.isfile(p):
            return p
        raise FileNotFoundError(f"--ref_image_path does not exist: {p}")

    json_dir = os.path.dirname(os.path.abspath(json_path))

    def _resolve_candidate(value):
        if not isinstance(value, str):
            return None
        v = value.strip()
        if v == "":
            return None
        if os.path.isabs(v) and os.path.isfile(v):
            return v
        local = os.path.join(json_dir, v)
        if os.path.isfile(local):
            return local
        return None

    # Try common field names in JSON root first.
    if isinstance(json_data, dict):
        for k in [
            "ref_image_path", "reference_image_path", "reference_image",
            "ref_image", "first_frame_path", "first_frame", "image_path", "cover_image"
        ]:
            found = _resolve_candidate(json_data.get(k))
            if found is not None:
                return found

        # Then inspect optional metadata section.
        meta = json_data.get("meta")
        if isinstance(meta, dict):
            for k in [
                "ref_image_path", "reference_image_path", "reference_image",
                "ref_image", "first_frame_path", "first_frame", "image_path", "cover_image"
            ]:
                found = _resolve_candidate(meta.get(k))
                if found is not None:
                    return found

    # Final fallback: search common file names under JSON directory.
    for name in [
        "frame.jpg", "frame.jpeg", "frame.png",
        "ref.jpg", "ref.jpeg", "ref.png",
        "reference.jpg", "reference.jpeg", "reference.png",
        "first_frame.jpg", "first_frame.jpeg", "first_frame.png",
        "cover.jpg", "cover.jpeg", "cover.png",
        "000000.jpg", "000000.png", "0.jpg", "0.png",
    ]:
        candidate = os.path.join(json_dir, name)
        if os.path.isfile(candidate):
            return candidate

    return None


class AttentionOutputFeatureTap:
    """Capture attention output features from one DiT block for one forward pass."""

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


class AttentionMapExtractorV8:
    """
    Attention map extractor with CFG difference + suffix scaling.
    Aligned with V8 training's AttentionMapExtractor.
    """
    def __init__(self, pipe, target_layers, target_token_indices,
                 suffix_token_indices=None, suffix_scale=1.0, cfg_scale=1.0, token_weight=0.2):
        self.pipe = pipe
        self.target_layers = target_layers
        self.target_token_indices = target_token_indices if isinstance(target_token_indices, list) else [target_token_indices]
        self.suffix_token_indices = suffix_token_indices if suffix_token_indices is not None else []
        self.suffix_scale = suffix_scale
        self.cfg_scale = cfg_scale
        self.token_weight = token_weight
        self.attention_maps = {}
        self.hooks = []
        self.modified_modules = []

    def register_hooks(self):
        dit_model = self.pipe.dit
        num_blocks = len(dit_model.blocks)
        processed_layers = list(range(num_blocks)) if -1 in self.target_layers else self.target_layers
        all_target_indices = self.target_token_indices + self.suffix_token_indices
        num_prefix = len(self.target_token_indices)

        for layer_idx in processed_layers:
            if layer_idx >= num_blocks or layer_idx < 0:
                continue
            self.attention_maps[layer_idx] = deque(maxlen=2)
            block = dit_model.blocks[layer_idx]

            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    if hasattr(module, 'attn_weights') and module.attn_weights is not None:
                        attn = module.attn_weights
                        if num_prefix > 0:
                            prefix_chunk = attn[..., :num_prefix]
                            if num_prefix > 1:
                                weights = torch.tensor(
                                    [1.0] + [self.token_weight] * (num_prefix - 1),
                                    device=attn.device, dtype=attn.dtype
                                ).view(1, 1, 1, -1)
                                prefix_score = (prefix_chunk * weights).sum(dim=-1)
                            else:
                                prefix_score = prefix_chunk.squeeze(-1)
                        else:
                            prefix_score = 0.0
                        if len(self.suffix_token_indices) > 0:
                            suffix_chunk = attn[..., num_prefix:]
                            suffix_score = suffix_chunk.sum(dim=-1)
                            if isinstance(prefix_score, float) and prefix_score == 0.0:
                                target_attn = suffix_score * self.suffix_scale
                            else:
                                target_attn = prefix_score + (suffix_score * self.suffix_scale)
                        else:
                            target_attn = prefix_score
                        self.attention_maps[layer_idx].append(target_attn.detach())
                        module.attn_weights = None
                return hook_fn

            target_module = getattr(block, 'cross_attn', getattr(block, 'attn2', None))
            if target_module is not None:
                target_module.save_attn_weights = True
                target_module.target_token_idx = all_target_indices
                self.modified_modules.append(target_module)
                self.hooks.append(target_module.register_forward_hook(make_hook(layer_idx)))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        for module in self.modified_modules:
            module.save_attn_weights = False
            if hasattr(module, 'target_token_idx'):
                del module.target_token_idx
        self.modified_modules = []
        self.attention_maps = {}

    def get_attention_maps(self):
        results = {}
        for layer_idx, dq in self.attention_maps.items():
            if not dq:
                continue
            idx = 0 if (self.cfg_scale > 1.0 and len(dq) >= 2) else -1
            results[layer_idx] = dq[idx]
        return results


class MemoryManager:
    def __init__(self):
        self.memory_bank = {}
        self.memory_meta_bank = {}
        self.chunk_frames = {}
        self.first_appearance = {}

    def register_appearance(self, char_id, chunk_idx):
        if char_id not in self.first_appearance:
            self.first_appearance[char_id] = chunk_idx
            return True
        return False

    def add_memory(self, char_id, tokens, bank_idx=0, token_meta=None, source_chunk_idx=None, source_video_frames=None,
                   first_appearance_only=False):
        if tokens is None or tokens.shape[0] == 0:
            return
        char_id = str(char_id)
        if bool(first_appearance_only):
            first_chunk_idx = self.first_appearance.get(char_id, None)
            if first_chunk_idx is not None and source_chunk_idx is not None and int(source_chunk_idx) != int(first_chunk_idx):
                return
        if char_id not in self.memory_bank or not isinstance(self.memory_bank[char_id], dict):
            self.memory_bank[char_id] = {}
        if char_id not in self.memory_meta_bank or not isinstance(self.memory_meta_bank[char_id], dict):
            self.memory_meta_bank[char_id] = {}

        token_count = int(tokens.shape[0])
        normalized_meta = []
        if isinstance(token_meta, list):
            for i in range(min(len(token_meta), token_count)):
                item = token_meta[i] if isinstance(token_meta[i], dict) else {}
                normalized_meta.append({
                    'char_id': str(item.get('char_id', char_id)),
                    'source_chunk_idx': int(item.get('source_chunk_idx', source_chunk_idx if source_chunk_idx is not None else -1)),
                    'bank_idx': int(item.get('bank_idx', bank_idx)),
                    'latent_t': int(item.get('latent_t', 0)),
                    'latent_h': int(item.get('latent_h', 0)),
                    'latent_w': int(item.get('latent_w', 0)),
                    'h_patch': int(item.get('h_patch', 1)),
                    'w_patch': int(item.get('w_patch', 1)),
                    'source_num_latent_frames': int(item.get('source_num_latent_frames', 1)),
                    'bbox_latent_xyxy': item.get('bbox_latent_xyxy', None),
                    'rel_l': float(item.get('rel_l', -1.0)),
                    'rel_r': float(item.get('rel_r', -1.0)),
                    'rel_t': float(item.get('rel_t', -1.0)),
                    'rel_b': float(item.get('rel_b', -1.0)),
                    'inside_box': bool(item.get('inside_box', False)),
                    'u': float(item.get('u', 0.0)),
                    'v': float(item.get('v', 0.0)),
                    'tau_local': float(item.get('tau_local', 0.0)),
                })
        if len(normalized_meta) < token_count:
            default_chunk = int(source_chunk_idx) if source_chunk_idx is not None else -1
            for _ in range(token_count - len(normalized_meta)):
                normalized_meta.append({
                    'char_id': str(char_id),
                    'source_chunk_idx': default_chunk,
                    'bank_idx': int(bank_idx),
                    'latent_t': 0,
                    'latent_h': 0,
                    'latent_w': 0,
                    'h_patch': 1,
                    'w_patch': 1,
                    'source_num_latent_frames': 1,
                    'bbox_latent_xyxy': None,
                    'rel_l': -1.0,
                    'rel_r': -1.0,
                    'rel_t': -1.0,
                    'rel_b': -1.0,
                    'inside_box': False,
                    'u': 0.0,
                    'v': 0.0,
                    'tau_local': 0.0,
                })

        if source_chunk_idx is not None and isinstance(source_video_frames, list):
            self.chunk_frames[int(source_chunk_idx)] = list(source_video_frames)

        self.memory_bank[char_id][str(bank_idx)] = tokens.cpu()
        self.memory_meta_bank[char_id][str(bank_idx)] = normalized_meta

    def get_memory(self, char_id, bank_idx=0):
        bank = self.memory_bank.get(char_id)
        if bank is None:
            return None
        if isinstance(bank, dict):
            return bank.get(str(bank_idx))
        return bank

    def get_memory_payload(self, char_id, bank_idx=0):
        tokens = self.get_memory(char_id, bank_idx)
        if tokens is None:
            return None
        meta_bank = self.memory_meta_bank.get(char_id, {})
        token_meta = meta_bank.get(str(bank_idx), []) if isinstance(meta_bank, dict) else []
        return {
            'tokens': tokens,
            'token_meta': token_meta,
        }

    def get_chunk_frames(self, chunk_idx):
        return self.chunk_frames.get(int(chunk_idx))


def merge_chunk_videos(output_dir, merged_filename="merged_chunks.mp4", pattern="chunk_*.mp4"):
    """
    Merge chunk videos after all chunks are generated.
    Prefer ffmpeg concat demuxer with stream copy for speed, then fallback to re-encode.
    Returns merged file path on success, or None if skipped/failed.
    """
    chunk_paths = sorted(glob.glob(os.path.join(output_dir, pattern)))
    if len(chunk_paths) == 0:
        print(f"[Merge] No files matched pattern '{pattern}' under {output_dir}, skip merge")
        return None
    if len(chunk_paths) == 1:
        print("[Merge] Only one chunk video found, merge skipped")
        return chunk_paths[0]

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        print("[Merge] ffmpeg not found in PATH, skip merge")
        return None

    merged_path = os.path.join(output_dir, merged_filename)
    list_file_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            list_file_path = f.name
            for p in chunk_paths:
                safe_p = os.path.abspath(p).replace("'", "'\\''")
                f.write(f"file '{safe_p}'\n")

        copy_cmd = [
            ffmpeg_bin, "-y", "-f", "concat", "-safe", "0",
            "-i", list_file_path, "-c", "copy", merged_path,
        ]
        proc = subprocess.run(copy_cmd, capture_output=True, text=True)
        if proc.returncode == 0 and os.path.exists(merged_path):
            print(f"[Merge] Success (stream copy): {merged_path}")
            return merged_path

        print("[Merge] Stream copy failed, fallback to re-encode")
        reencode_cmd = [
            ffmpeg_bin, "-y", "-f", "concat", "-safe", "0",
            "-i", list_file_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
            "-an", merged_path,
        ]
        proc2 = subprocess.run(reencode_cmd, capture_output=True, text=True)
        if proc2.returncode == 0 and os.path.exists(merged_path):
            print(f"[Merge] Success (re-encode): {merged_path}")
            return merged_path

        print("[Merge] Failed to merge chunks")
        tail1 = (proc.stderr or "")[-500:]
        tail2 = (proc2.stderr or "")[-500:]
        if tail1:
            print(f"[Merge][ffmpeg copy stderr tail]\n{tail1}")
        if tail2:
            print(f"[Merge][ffmpeg reencode stderr tail]\n{tail2}")
        return None
    except Exception as e:
        print(f"[Merge] Exception: {e}")
        return None
    finally:
        if list_file_path and os.path.exists(list_file_path):
            try:
                os.remove(list_file_path)
            except Exception:
                pass


def save_chunk_memory_visualization(viz_root_dir, chunk_idx, video_frames, char_positions_dict,
                                    h_patch, w_patch, vae_stride_t=4, char_boxes_dict=None):
    """Save per-character memory-token overlays for one chunk.

    Each latent-time overlay uses the first corresponding pixel frame:
    ``pixel_t = latent_t * vae_stride_t``.
    """
    if not video_frames or not char_positions_dict:
        return

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    chunk_dir = os.path.join(viz_root_dir, f"chunk_{chunk_idx:03d}")
    os.makedirs(chunk_dir, exist_ok=True)

    frames = []
    for frame in video_frames:
        if isinstance(frame, Image.Image):
            frames.append(np.array(frame.convert("RGB")))
        else:
            frames.append(np.array(frame))
    frames = np.stack(frames, axis=0)

    t_vid, h_vid, w_vid, _ = frames.shape
    stride_h = max(1, h_vid // max(1, h_patch))
    stride_w = max(1, w_vid // max(1, w_patch))

    colors = ['red', 'cyan', 'lime', 'yellow', 'magenta']

    for char_idx, (char_id, pos_payload) in enumerate(char_positions_dict.items()):
        if isinstance(pos_payload, dict):
            final_positions = pos_payload.get('selected_positions', pos_payload.get('final_positions'))
            top_positions = pos_payload.get('top_positions')
        else:
            final_positions = pos_payload
            top_positions = None

        if final_positions is None and top_positions is None:
            continue

        if final_positions is not None and final_positions.numel() == 0:
            final_positions = None
        if top_positions is not None and top_positions.numel() == 0:
            top_positions = None
        if final_positions is None and top_positions is None:
            continue

        safe_char_id = "".join([c if c.isalnum() else "_" for c in char_id])
        char_dir = os.path.join(chunk_dir, f"role_{safe_char_id}")
        os.makedirs(char_dir, exist_ok=True)

        final_np = final_positions.cpu().numpy() if final_positions is not None else np.zeros((0, 3), dtype=np.int64)
        top_np = top_positions.cpu().numpy() if top_positions is not None else np.zeros((0, 3), dtype=np.int64)
        active_lat_frames = sorted(list(set(final_np[:, 0].astype(int)).union(set(top_np[:, 0].astype(int)))))
        final_color = colors[char_idx % len(colors)]
        top_color = 'gold'

        for lat_t in active_lat_frames:
            frame_final_positions = final_np[final_np[:, 0] == lat_t]
            frame_top_positions = top_np[top_np[:, 0] == lat_t]
            pixel_t = int(lat_t * vae_stride_t)
            if pixel_t >= t_vid:
                continue

            img = frames[pixel_t].copy()

            frame_boxes = []
            if char_boxes_dict is not None:
                char_frame_boxes = char_boxes_dict.get(char_id, {})
                frame_boxes = char_frame_boxes.get(pixel_t, [])

            def _render_and_save(include_boxes, include_top, include_final, suffix):
                plt.figure(figsize=(10, 6))
                plt.imshow(img)
                ax = plt.gca()

                if include_boxes:
                    for box_item in frame_boxes:
                        if len(box_item) >= 4:
                            x1, y1, x2, y2 = box_item[:4]
                            rect = patches.Rectangle((x1, y1), max(1, x2 - x1), max(1, y2 - y1),
                                                     linewidth=1.2, edgecolor='deepskyblue', facecolor='none')
                            ax.add_patch(rect)

                if include_top:
                    for _, lat_h, lat_w in frame_top_positions:
                        y = int(lat_h) * stride_h
                        x = int(lat_w) * stride_w
                        rect = patches.Rectangle((x, y), stride_w, stride_h, linewidth=0.8,
                                                 edgecolor=top_color, facecolor=top_color, alpha=0.18)
                        ax.add_patch(rect)

                if include_final:
                    for _, lat_h, lat_w in frame_final_positions:
                        y = int(lat_h) * stride_h
                        x = int(lat_w) * stride_w
                        rect = patches.Rectangle((x, y), stride_w, stride_h, linewidth=1,
                                                 edgecolor=final_color, facecolor=final_color, alpha=0.45)
                        ax.add_patch(rect)

                plt.axis('off')
                plt.title(f"{char_id} | latent_t={lat_t} | pixel_t={pixel_t} | final={len(frame_final_positions)} top={len(frame_top_positions)} boxes={len(frame_boxes)}")
                out_name = f"frame_{pixel_t:04d}_{suffix}.jpg"
                plt.savefig(os.path.join(char_dir, out_name), bbox_inches='tight', pad_inches=0)
                plt.close()

            _render_and_save(include_boxes=True, include_top=True, include_final=True, suffix="overlay")

            if len(frame_top_positions) > 0:
                _render_and_save(include_boxes=False, include_top=True, include_final=False, suffix="top")
            if len(frame_final_positions) > 0:
                _render_and_save(include_boxes=False, include_top=False, include_final=True, suffix="final")
            if len(frame_boxes) > 0:
                _render_and_save(include_boxes=True, include_top=False, include_final=False, suffix="boxes")


def save_feature_mapping_visualization(
    viz_root_dir,
    generation_chunk_idx,
    generation_video_frames,
    generation_latents,
    feature_mapping_steps,
    memory_manager,
    draw_empty=True,
):
    """
    Save fine-grained feature mapping visualization.
    Left: memory source frame, Right: generation frame.
    """
    if not feature_mapping_steps:
        return

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    gen_frames = []
    for frame in generation_video_frames:
        if isinstance(frame, Image.Image):
            gen_frames.append(np.array(frame.convert("RGB")))
        else:
            gen_frames.append(np.array(frame))
    if len(gen_frames) == 0:
        return
    gen_frames = np.stack(gen_frames, axis=0)
    gen_t_vid, gen_h_vid, gen_w_vid, _ = gen_frames.shape
    gen_f_lat = int(generation_latents.shape[2]) if isinstance(generation_latents, torch.Tensor) else 1
    gen_vae_stride_t = max(1, (gen_t_vid - 1) // max(1, gen_f_lat - 1))

    root_dir = os.path.join(viz_root_dir, "feature_mapping", f"chunk_{generation_chunk_idx:03d}")
    os.makedirs(root_dir, exist_ok=True)

    colors = ['red', 'cyan', 'lime', 'yellow', 'magenta', 'orange', 'deepskyblue']

    for step_payload in feature_mapping_steps:
        step_idx = int(step_payload.get('step_idx', -1))
        variants = [
            ("after_otsu", step_payload.get('role_matches_after_otsu', None)),
            ("after_reverse", step_payload.get('role_matches_after_reverse', None)),
            ("after_topn", step_payload.get('role_matches_after_topn', None)),
            ("filtered_by_reverse", step_payload.get('role_matches_filtered_by_reverse', None)),
            ("filtered_by_topn", step_payload.get('role_matches_filtered_by_topn', None)),
        ]
        if (variants[0][1] is None and variants[1][1] is None) and step_payload.get('role_matches', None) is not None:
            variants = [("final", step_payload.get('role_matches', {}))]
        gen_h_patch = max(1, int(step_payload.get('gen_h_patch', 1)))
        gen_w_patch = max(1, int(step_payload.get('gen_w_patch', 1)))
        gen_f = max(1, int(step_payload.get('gen_f', gen_f_lat)))
        gen_stride_h = max(1, gen_h_vid // gen_h_patch)
        gen_stride_w = max(1, gen_w_vid // gen_w_patch)

        gen_lat_candidates = sorted(list(set([0, max(0, gen_f // 2), max(0, gen_f - 1)])))

        for variant_name, role_matches in variants:
            role_matches = role_matches if isinstance(role_matches, dict) else {}
            for role, role_payload in role_matches.items():
                safe_role = "".join([c if c.isalnum() else "_" for c in str(role)])
                source_chunk_idx = int(role_payload.get('source_chunk_idx', -1))
                source_frames = memory_manager.get_chunk_frames(source_chunk_idx) if source_chunk_idx >= 0 else None
                if not source_frames:
                    continue

                mem_frames = []
                for frame in source_frames:
                    if isinstance(frame, Image.Image):
                        mem_frames.append(np.array(frame.convert("RGB")))
                    else:
                        mem_frames.append(np.array(frame))
                if len(mem_frames) == 0:
                    continue
                mem_frames = np.stack(mem_frames, axis=0)
                mem_t_vid, mem_h_vid, mem_w_vid, _ = mem_frames.shape

                source_num_latent_frames = max(1, int(role_payload.get('source_num_latent_frames', 1)))
                mem_h_patch = max(1, int(role_payload.get('h_patch', gen_h_patch)))
                mem_w_patch = max(1, int(role_payload.get('w_patch', gen_w_patch)))
                mem_stride_h = max(1, mem_h_vid // mem_h_patch)
                mem_stride_w = max(1, mem_w_vid // mem_w_patch)
                mem_vae_stride_t = max(1, (mem_t_vid - 1) // max(1, source_num_latent_frames - 1))
                matches = role_payload.get('matches', []) if isinstance(role_payload.get('matches', []), list) else []

                step_dir = os.path.join(root_dir, f"role_{safe_role}", f"step_{step_idx:03d}", variant_name)
                os.makedirs(step_dir, exist_ok=True)

                for gen_lat_t in gen_lat_candidates:
                    gen_pixel_t = int(np.clip(gen_lat_t * gen_vae_stride_t, 0, gen_t_vid - 1))
                    gen_img = gen_frames[gen_pixel_t]

                    for mem_lat_t in range(source_num_latent_frames):
                        mem_pixel_t = int(np.clip(mem_lat_t * mem_vae_stride_t, 0, mem_t_vid - 1))
                        mem_img = mem_frames[mem_pixel_t]

                        frame_matches = [
                            m for m in matches
                            if int(m.get('gen_latent_t', -1)) == int(gen_lat_t)
                            and int(m.get('mem_latent_t', -1)) == int(mem_lat_t)
                        ]

                        if (not draw_empty) and len(frame_matches) == 0:
                            continue

                        gap = 24
                        canvas_h = max(mem_img.shape[0], gen_img.shape[0])
                        canvas_w = mem_img.shape[1] + gap + gen_img.shape[1]

                        fig_w = max(8.0, canvas_w / 200.0)
                        fig_h = max(4.0, canvas_h / 200.0)
                        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

                        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                        canvas[:mem_img.shape[0], :mem_img.shape[1], :] = mem_img
                        x_offset = mem_img.shape[1] + gap
                        canvas[:gen_img.shape[0], x_offset:x_offset + gen_img.shape[1], :] = gen_img
                        ax.imshow(canvas)

                        for mi, match in enumerate(frame_matches):
                            color = colors[mi % len(colors)]
                            mem_h = int(match.get('mem_latent_h', 0))
                            mem_w = int(match.get('mem_latent_w', 0))
                            gen_h = int(match.get('gen_latent_h', 0))
                            gen_w = int(match.get('gen_latent_w', 0))

                            lx = int(mem_w * mem_stride_w)
                            ly = int(mem_h * mem_stride_h)
                            rx = int(gen_w * gen_stride_w) + x_offset
                            ry = int(gen_h * gen_stride_h)

                            lrect = patches.Rectangle((lx, ly), mem_stride_w, mem_stride_h, linewidth=1.0,
                                                      edgecolor=color, facecolor=color, alpha=0.35)
                            rrect = patches.Rectangle((rx, ry), gen_stride_w, gen_stride_h, linewidth=1.0,
                                                      edgecolor=color, facecolor=color, alpha=0.35)
                            ax.add_patch(lrect)
                            ax.add_patch(rrect)

                            lcx = lx + mem_stride_w * 0.5
                            lcy = ly + mem_stride_h * 0.5
                            rcx = rx + gen_stride_w * 0.5
                            rcy = ry + gen_stride_h * 0.5
                            ax.plot([lcx, rcx], [lcy, rcy], color=color, linewidth=1.2, alpha=0.9)

                        ax.axis('off')
                        ax.set_title(
                            f"gen_chunk={generation_chunk_idx} mem_chunk={source_chunk_idx} role={role} "
                            f"step={step_idx} view={variant_name} gen_lat_t={gen_lat_t} mem_lat_t={mem_lat_t} matches={len(frame_matches)}"
                        )

                        out_name = f"genLat_{gen_lat_t:02d}_memLat_{mem_lat_t:02d}.jpg"
                        fig.savefig(os.path.join(step_dir, out_name), bbox_inches='tight', pad_inches=0)
                        plt.close(fig)


def parse_denoise_step_list(step_list_value, total_steps):
    """Parse 1-based denoising step list like '10,20,30,40,50'."""
    if total_steps <= 0:
        return []
    values = []
    if isinstance(step_list_value, str):
        raw_parts = [x.strip() for x in step_list_value.split(',')]
    elif isinstance(step_list_value, (list, tuple)):
        raw_parts = list(step_list_value)
    else:
        raw_parts = []
    for item in raw_parts:
        try:
            step = int(item)
        except Exception:
            continue
        if 1 <= step <= int(total_steps):
            values.append(step)
    return sorted(list(dict.fromkeys(values)))


def _to_rgb_numpy_frame(frame):
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    arr = np.array(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def make_frame_contact_sheet(video_frames, max_columns=4, bg_value=255):
    if not video_frames:
        return None
    frames = [_to_rgb_numpy_frame(f) for f in video_frames]
    h = max(int(f.shape[0]) for f in frames)
    w = max(int(f.shape[1]) for f in frames)
    n = len(frames)
    cols = max(1, min(int(max_columns), n))
    rows = int(math.ceil(float(n) / float(cols)))
    canvas = np.full((rows * h, cols * w, 3), int(bg_value), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        r = idx // cols
        c = idx % cols
        fh, fw = frame.shape[:2]
        y0 = r * h
        x0 = c * w
        canvas[y0:y0 + fh, x0:x0 + fw] = frame
    return Image.fromarray(canvas)


def compute_residual_video_frames(prev_video_frames, curr_video_frames, eps=1e-6):
    if not prev_video_frames or not curr_video_frames:
        return []
    frame_count = min(len(prev_video_frames), len(curr_video_frames))
    residual_frames = []
    for fi in range(frame_count):
        prev_arr = _to_rgb_numpy_frame(prev_video_frames[fi]).astype(np.float32)
        curr_arr = _to_rgb_numpy_frame(curr_video_frames[fi]).astype(np.float32)
        diff = np.abs(curr_arr - prev_arr).mean(axis=-1)
        if not np.isfinite(diff).any():
            diff_norm = np.zeros_like(diff, dtype=np.float32)
        else:
            vmax = float(np.percentile(diff, 99.0))
            if vmax <= eps:
                vmax = float(diff.max()) if diff.size > 0 else 0.0
            if vmax <= eps:
                diff_norm = np.zeros_like(diff, dtype=np.float32)
            else:
                diff_norm = np.clip(diff / vmax, 0.0, 1.0)
        diff_u8 = np.clip(diff_norm * 255.0, 0.0, 255.0).astype(np.uint8)
        heat = np.stack([diff_u8, np.zeros_like(diff_u8), 255 - diff_u8], axis=-1)
        residual_frames.append(Image.fromarray(heat))
    return residual_frames


def save_denoise_step_visualization(
    viz_root_dir,
    chunk_idx,
    step_records,
):
    """
    Save per-step decoded frames, residual heatmaps, and optional token overlays.
    """
    if not step_records:
        return
    chunk_dir = os.path.join(viz_root_dir, f"chunk_{chunk_idx:03d}")
    os.makedirs(chunk_dir, exist_ok=True)

    prev_decoded_frames = None
    for record in step_records:
        step_number = int(record.get('step_number', -1))
        decoded_frames = record.get('decoded_frames', [])
        if not decoded_frames:
            continue
        step_dir = os.path.join(chunk_dir, f"step_{step_number:03d}")
        os.makedirs(step_dir, exist_ok=True)

        meta = {
            'step_number': int(step_number),
            'loop_idx': int(record.get('loop_idx', step_number - 1)),
            'scheduler_timestep': int(record.get('timestep', -1)),
            'num_frames': int(len(decoded_frames)),
            'latent_frames': int(record.get('latent_frames', 1)),
        }
        with open(os.path.join(step_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        for fi, frame in enumerate(decoded_frames):
            frame_img = frame if isinstance(frame, Image.Image) else Image.fromarray(_to_rgb_numpy_frame(frame))
            frame_img.save(os.path.join(step_dir, f"frame_{fi:04d}_decoded.jpg"))

        decoded_contact = make_frame_contact_sheet(decoded_frames)
        if decoded_contact is not None:
            decoded_contact.save(os.path.join(step_dir, "decoded_contact_sheet.jpg"))

        if prev_decoded_frames is not None:
            residual_frames = compute_residual_video_frames(prev_decoded_frames, decoded_frames)
            if residual_frames:
                for fi, frame in enumerate(residual_frames):
                    frame.save(os.path.join(step_dir, f"frame_{fi:04d}_residual.jpg"))
                residual_contact = make_frame_contact_sheet(residual_frames)
                if residual_contact is not None:
                    residual_contact.save(os.path.join(step_dir, "residual_contact_sheet.jpg"))

        step_viz = record.get('viz', None)
        spatial_shape = record.get('spatial_shape', None)
        if isinstance(step_viz, dict) and len(step_viz) > 0 and spatial_shape is not None:
            h_patch, w_patch = spatial_shape
            latent_frames = max(1, int(record.get('latent_frames', 1)))
            t_vid = len(decoded_frames)
            vae_stride_t = max(1, (t_vid - 1) // max(1, latent_frames - 1))
            save_chunk_memory_visualization(
                viz_root_dir=os.path.join(step_dir, "token_overlay"),
                chunk_idx=chunk_idx,
                video_frames=decoded_frames,
                char_positions_dict=step_viz,
                h_patch=int(h_patch),
                w_patch=int(w_patch),
                vae_stride_t=int(vae_stride_t),
                char_boxes_dict=None,
            )

        prev_decoded_frames = decoded_frames


def save_denoise_step_edge_frames_visualization(
    viz_root_dir,
    chunk_idx,
    step_records,
):
    """Save first/last decoded frame pairs for each selected denoising step."""
    if not step_records:
        return
    chunk_dir = os.path.join(viz_root_dir, f"chunk_{chunk_idx:03d}")
    os.makedirs(chunk_dir, exist_ok=True)

    for record in step_records:
        decoded_frames = record.get('decoded_frames', [])
        if not isinstance(decoded_frames, list) or len(decoded_frames) == 0:
            continue
        step_number = int(record.get('step_number', -1))
        first_arr = _to_rgb_numpy_frame(decoded_frames[0])
        last_arr = _to_rgb_numpy_frame(decoded_frames[-1])

        h = max(int(first_arr.shape[0]), int(last_arr.shape[0]))
        w1 = int(first_arr.shape[1])
        w2 = int(last_arr.shape[1])
        gap = 12
        canvas = np.full((h, w1 + gap + w2, 3), 255, dtype=np.uint8)
        canvas[:first_arr.shape[0], :w1] = first_arr
        canvas[:last_arr.shape[0], w1 + gap:w1 + gap + w2] = last_arr

        out = Image.fromarray(canvas)
        out.save(os.path.join(chunk_dir, f"step_{step_number:03d}_first_last.jpg"))

class ReferenceInferenceRuntime:
    """Thin base class: current entry scripts override setup/probe/forward helpers."""

    def _build_query_boxes_from_selected_indices(self, selected_indices, h_patch, w_patch):
            if selected_indices is None:
                return {}
            if isinstance(selected_indices, set):
                selected_indices = list(selected_indices)
            if not isinstance(selected_indices, (list, tuple)):
                return {}
            if len(selected_indices) == 0:
                return {}
            h_patch = int(h_patch)
            w_patch = int(w_patch)
            if h_patch <= 0 or w_patch <= 0:
                return {}
            spatial = h_patch * w_patch
            per_t = {}
            for idx in selected_indices:
                try:
                    flat = int(idx)
                except Exception:
                    continue
                if flat < 0:
                    continue
                lt = flat // spatial
                sp = flat % spatial
                y = sp // w_patch
                x = sp % w_patch
                if lt not in per_t:
                    per_t[lt] = [x, y, x + 1, y + 1]
                else:
                    box = per_t[lt]
                    box[0] = min(box[0], x)
                    box[1] = min(box[1], y)
                    box[2] = max(box[2], x + 1)
                    box[3] = max(box[3], y + 1)
            out = {}
            for lt, box in per_t.items():
                out[int(lt)] = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            return out

    @torch.no_grad()
    def generate_chunk(self, prompt, memory_tokens=None, memory_bank_tokens=None, memory_bank_percents=None, memory_bank_token_meta=None,
                           memory_token_lengths_per_character=None,
                           ref_images=None, random_ref_frame=None, seed=42,
                           online_memory_chars=None, online_memory_bank_percents=None,
                           teacher_forced_probe=None):
            """
            Generate a chunk of video.
            memory_token_lengths_per_character: list of int, token count per character (for segment_embed).
            Variant B: use ref_images during generation to handle boundary continuity.
            """
            self._offload()
            print(f"  [Gen] Generating with seed {seed}...")

            # 1. Text encode
            self.pipe.text_encoder.to(device=self.device, dtype=torch.float32)
            if hasattr(self.pipe, 'prompter'):
                self.pipe.prompter.text_encoder.to(device=self.device, dtype=torch.float32)

            prompt_emb = self.pipe.encode_prompt(prompt, positive=True)
            neg_emb = self.pipe.encode_prompt(self.args.negative_prompt, positive=False)
            prompt_emb['context'] = prompt_emb['context'].to(dtype=self.dtype)
            neg_emb['context'] = neg_emb['context'].to(dtype=self.dtype)

            # 2. Image encode
            image_emb = {}
            num_condition_frames = None  # Actual number of conditioning frames.
            if ref_images is None and getattr(self.pipe.dit, 'has_image_input', False):
                raise ValueError(
                    "Reference image is required for this DiT (has_image_input=True). "
                    "Please provide --ref_image_path, or include a resolvable reference image path in JSON."
                )
            if ref_images is not None:
                if not isinstance(ref_images, list):
                    ref_images = [ref_images]
                if random_ref_frame is None:
                    random_ref_frame = ref_images[0]
            
                # Preserve the actual conditioning-frame count for the wrapper.
                num_condition_frames = len(ref_images)

                # Native WAN path guard:
                # In base pipeline encode_images_adaptive, ref_pad_cfg=False marks only the first frame as condition.
                # For multi-reference inputs this can mismatch condition masks and cause chunk drift/distortion.
                effective_ref_pad_cfg = bool(self.args.ref_pad_cfg)
                effective_ref_pad_num = int(self.args.ref_pad_num)
                if self.native_wan_inference and num_condition_frames > 1 and (not effective_ref_pad_cfg):
                    print(
                        f"  [NativeGuard] Detected multi-ref input (num_condition_frames={num_condition_frames}) with ref_pad_cfg=False. "
                        f"Force ref_pad_cfg=True to keep condition mask aligned.",
                        flush=True,
                    )
                    effective_ref_pad_cfg = True
                if self.native_wan_inference and num_condition_frames > 1 and effective_ref_pad_num > 0:
                    print(
                        f"  [NativeGuard] ref_pad_num={effective_ref_pad_num} may increase drift in autoregressive chunks. "
                        f"Consider ref_pad_num=0 for native multi-ref continuity.",
                        flush=True,
                    )

                self.pipe.image_encoder.to(self.device)
                image_emb = self.pipe.encode_images_adaptive(
                    first_frames=ref_images, random_ref_frame=random_ref_frame,
                    num_frames=self.args.context_frames,
                    height=self.args.height, width=self.args.width,
                    use_first_aug=self.args.use_first_aug,
                    ref_pad_cfg=effective_ref_pad_cfg,
                    ref_pad_num=effective_ref_pad_num
                )
                # Pass the actual count to the downstream forward wrapper.
                image_emb['num_condition_frames'] = num_condition_frames
                self.pipe.image_encoder.to("cpu")

            self.pipe.text_encoder.to("cpu")
            if hasattr(self.pipe, 'prompter'):
                self.pipe.prompter.text_encoder.to("cpu")
            torch.cuda.empty_cache()

            # 3. Setup DiT denoising
            self.pipe.dit.to(self.device)
            self.pipe.scheduler.set_timesteps(self.args.num_inference_steps)

            shape = (1, 16, (self.args.context_frames - 1) // 4 + 1,
                     self.args.height // 8, self.args.width // 8)
            latent_dtype = torch.float32 if bool(getattr(self.args, "inference_latents_fp32", True)) else self.dtype
            if teacher_forced_probe is None:
                gen = torch.Generator(device=self.device).manual_seed(seed)
                latents = torch.randn(shape, generator=gen, device=self.device, dtype=latent_dtype)
                denoise_schedule = list(enumerate(self.pipe.scheduler.timesteps))
            else:
                if not isinstance(teacher_forced_probe, dict):
                    raise TypeError("teacher_forced_probe must be a dict")
                timestep_index = int(teacher_forced_probe["timestep_index"])
                if timestep_index < 0 or timestep_index >= len(self.pipe.scheduler.timesteps):
                    raise IndexError(
                        f"teacher-forced timestep index {timestep_index} outside "
                        f"[0, {len(self.pipe.scheduler.timesteps)})"
                    )
                noisy_latents = teacher_forced_probe.get("noisy_latents")
                if not isinstance(noisy_latents, torch.Tensor):
                    raise TypeError("teacher_forced_probe.noisy_latents must be a tensor")
                if tuple(noisy_latents.shape) != tuple(shape):
                    raise ValueError(
                        f"teacher-forced noisy latent shape {tuple(noisy_latents.shape)} "
                        f"does not match generation shape {tuple(shape)}"
                    )
                latents = noisy_latents.to(device=self.device, dtype=latent_dtype).clone()
                denoise_schedule = [(timestep_index, self.pipe.scheduler.timesteps[timestep_index])]

            # 4. Denoising loop
            denoising_latents = latents.clone()
            # Remove wrapper-only metadata for the memory-free denoising path.
            # num_condition_frames is only for MemoryAwareDiTForward; denoising_model() does not accept it.
            image_emb_for_denoising = {k: v for k, v in image_emb.items() if k != 'num_condition_frames'}
        
            online_results = {
                'tokens': defaultdict(dict),
                'token_meta': defaultdict(dict),
                'viz': {},
                'spatial_shape': None,
                'feature_mapping_steps': [],
            }
            denoise_step_records = []
            target_denoise_steps = set()
            need_any_step_decode_viz = bool(
                bool(getattr(self.args, 'save_denoise_step_viz', False))
                or bool(getattr(self.args, 'save_denoise_step_edge_viz', True))
            )
            if need_any_step_decode_viz:
                parsed_steps = parse_denoise_step_list(
                    getattr(self.args, 'denoise_step_list', '10,20,30,40,50'),
                    total_steps=len(self.pipe.scheduler.timesteps),
                )
                target_denoise_steps = set([int(s) - 1 for s in parsed_steps])
            collect_chars = online_memory_chars if isinstance(online_memory_chars, list) else []
            collect_percents = online_memory_bank_percents if isinstance(online_memory_bank_percents, (list, tuple)) and len(online_memory_bank_percents) > 0 else self.memory_bank_percents
            # Always keep all configured banks for online extraction.
            collect_candidates = list(enumerate(collect_percents))

            step_percent_values = [float(x.item()) / float(max(int(getattr(self.pipe.scheduler, 'num_train_timesteps', 1000)), 1)) for x in self.pipe.scheduler.timesteps]
            step_to_collect_banks = defaultdict(list)
            for bank_idx, percent in collect_candidates:
                nearest_step = min(range(len(step_percent_values)), key=lambda sid: abs(step_percent_values[sid] - float(percent)))
                step_to_collect_banks[nearest_step].append(int(bank_idx))

            viz_collect_steps = set()
            if bool(getattr(self.args, 'save_memory_viz', False)) and len(step_percent_values) > 0:
                allow_outside_bank = bool(getattr(self.args, 'allow_injection_outside_bank_range', True))
                if (not allow_outside_bank) and isinstance(collect_percents, (list, tuple)) and len(collect_percents) > 0:
                    p_min = float(min(collect_percents))
                    p_max = float(max(collect_percents))
                    allowed_steps = [sid for sid, p in enumerate(step_percent_values) if p_min <= float(p) <= p_max]
                    viz_collect_steps = set(allowed_steps[-3:]) if len(allowed_steps) > 0 else set()
                else:
                    viz_collect_steps = set(sorted(step_to_collect_banks.keys())[-3:])

            selected_feature_steps = set()
            # Query-side cache for bbox-relative matching on next denoising step.
            # {char_id: {latent_t: [x1, y1, x2, y2]}} in latent patch coordinates.
            prev_step_role_boxes = {}
            if bool(getattr(self.args, 'save_feature_mapping_viz', False)):
                total_steps = len(self.pipe.scheduler.timesteps)
                override_steps = int(getattr(self.args, 'feature_mapping_num_steps', -1))
                if override_steps > 0:
                    num_steps_to_visualize = min(total_steps, override_steps)
                else:
                    ratio = float(getattr(self.args, 'feature_mapping_last_ratio', 0.05))
                    num_steps_to_visualize = max(1, int(math.ceil(total_steps * max(0.0, ratio))))
                    num_steps_to_visualize = min(total_steps, num_steps_to_visualize)
                selected_feature_steps = set(range(total_steps - num_steps_to_visualize, total_steps))

            cached_query_role_boxes = None
            cached_query_feature_payload = None

            for i, t in tqdm(denoise_schedule, desc="  Denoising"):
                t = t.to(self.device).unsqueeze(0)
                if hasattr(self, "_set_inference_noise_domain_from_timestep"):
                    self._set_inference_noise_domain_from_timestep(t)
                elif hasattr(self.pipe, "set_active_noise_domain_from_timestep"):
                    self.pipe.set_active_noise_domain_from_timestep(t)
                extra_input = self.pipe.prepare_extra_input(denoising_latents)
                selection_mode = self._get_role_token_selection_mode()

                has_valid_memory_tokens = bool(
                    isinstance(memory_tokens, torch.Tensor) and memory_tokens.dim() == 2 and memory_tokens.shape[0] > 0
                )
                has_valid_bank_tokens = False
                valid_bank_keys = []
                if memory_bank_tokens is not None and isinstance(memory_bank_tokens, dict) and len(memory_bank_tokens) > 0:
                    for k, v in memory_bank_tokens.items():
                        if isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape[0] > 0:
                            has_valid_bank_tokens = True
                            valid_bank_keys.append(str(k))

                has_memory = (
                    has_valid_memory_tokens or has_valid_bank_tokens
                )
                use_memory_path = has_memory
                should_collect_bank_tokens = bool(collect_chars and i in step_to_collect_banks)
                should_collect_viz = bool(collect_chars and bool(getattr(self.args, 'save_memory_viz', False)) and i in viz_collect_steps)
                should_collect_step_viz = bool(collect_chars and need_any_step_decode_viz and i in target_denoise_steps)
                need_probe_for_collection = bool(collect_chars and (should_collect_bank_tokens or should_collect_viz or should_collect_step_viz))

                if i == 0:
                    print(
                        f"  [MemoryDebug] has_memory={has_memory}, use_memory_path={use_memory_path}, "
                        f"has_valid_memory_tokens={has_valid_memory_tokens}, "
                        f"has_valid_bank_tokens={has_valid_bank_tokens}, "
                        f"valid_bank_keys={valid_bank_keys}",
                        flush=True,
                    )

                active_query_role_boxes = cached_query_role_boxes
                active_query_feature_payload = cached_query_feature_payload
                probe_extractor = None
                probe_feature_tap = None
                probe_ordered_roles = []
                probe_role_to_maps = None
                probe_layer_tokens = None
                current_query_role_boxes = None
                current_query_feature_payload = None
                if (use_memory_path and self.enable_sparse_role_memory_attn) or need_probe_for_collection:
                    probe_role_ids = self._collect_character_semantic_probe_role_ids(
                        collect_chars=collect_chars,
                        memory_bank_token_meta=memory_bank_token_meta,
                        memory_bank_percents=memory_bank_percents,
                        timestep=t,
                    )
                    if selection_mode == 'layer7_single':
                        per_char_step_maps, probe_ordered_roles, probe_layer_tokens = self._run_character_semantic_probe(
                            noisy_latents=denoising_latents,
                            timestep=t,
                            prompt=prompt,
                            role_ids=probe_role_ids,
                            cond_context=prompt_emb['context'],
                            uncond_context=neg_emb['context'],
                            image_emb_for_denoising=image_emb_for_denoising,
                            extra_input=extra_input,
                        )
                        if isinstance(per_char_step_maps, list) and len(per_char_step_maps) > 0 and isinstance(probe_ordered_roles, list):
                            probe_role_to_maps = {}
                            for rid, maps in zip(probe_ordered_roles, per_char_step_maps):
                                rid = str(rid)
                                if isinstance(maps, dict) and len(maps) > 0:
                                    probe_role_to_maps[rid] = maps
                        _, _, _, h_lat, w_lat = denoising_latents.shape
                        patch_size = self.pipe.dit.patch_size
                        h_patch = h_lat // patch_size[1]
                        w_patch = w_lat // patch_size[2]
                        current_query_role_boxes, current_query_feature_payload = self._build_character_mask_payload_from_probe(
                            per_char_step_maps=per_char_step_maps,
                            ordered_roles=probe_ordered_roles,
                            h_patch=h_patch,
                            w_patch=w_patch,
                            layer_tokens=probe_layer_tokens,
                        )
                        if isinstance(current_query_role_boxes, dict) and len(current_query_role_boxes) > 0:
                            active_query_role_boxes = current_query_role_boxes
                            cached_query_role_boxes = current_query_role_boxes
                            prev_step_role_boxes = dict(current_query_role_boxes)
                        if isinstance(current_query_feature_payload, dict) and len(current_query_feature_payload) > 0:
                            active_query_feature_payload = current_query_feature_payload
                            cached_query_feature_payload = current_query_feature_payload
                    else:
                        probe_char_configs, probe_ordered_roles = self._prepare_character_semantic_probe_configs(prompt=prompt, role_ids=probe_role_ids)
                        if len(probe_char_configs) > 0:
                            probe_extractor = MultiCharacterAttentionMapExtractor(
                                self.pipe,
                                self.args.extract_layers,
                                probe_char_configs,
                                cfg_scale=1.0,
                            )
                            probe_extractor.register_hooks()
                            if str(getattr(self, 'sparse_role_memory_feature_source', 'attn_out')).strip().lower() in ('attn_out', 'self_attn_out'):
                                probe_feature_tap = AttentionOutputFeatureTap(
                                    dit_model=self.pipe.denoising_model(),
                                    layer_idx=int(getattr(self, 'sparse_role_memory_layer_idx', 7)),
                                    keep_device='cpu',
                                    keep_dtype=torch.bfloat16,
                                    source=str(getattr(self, 'sparse_role_memory_feature_source', 'attn_out')),
                                )
                                probe_feature_tap.register()

                if use_memory_path:
                    mapping_recorder = {} if i in selected_feature_steps else None
                    emb_kwargs = dict(image_emb)
                    if memory_token_lengths_per_character is not None:
                        emb_kwargs['memory_token_lengths_per_character'] = memory_token_lengths_per_character
                    if isinstance(active_query_role_boxes, dict) and len(active_query_role_boxes) > 0:
                        emb_kwargs['query_role_boxes'] = active_query_role_boxes
                    if isinstance(active_query_feature_payload, dict) and len(active_query_feature_payload) > 0:
                        emb_kwargs['query_feature_payload'] = active_query_feature_payload
                    if mapping_recorder is not None:
                        emb_kwargs['feature_mapping_recorder'] = mapping_recorder
                    noise_pred_cond = self._memory_aware_dit_forward(
                        x=denoising_latents, t=t, context=prompt_emb['context'],
                        memory_tokens=memory_tokens,
                        memory_bank_tokens=memory_bank_tokens,
                        memory_bank_percents=memory_bank_percents,
                        memory_bank_token_meta=memory_bank_token_meta,
                        **emb_kwargs, **extra_input
                    )
                else:
                    mapping_recorder = None
                    noise_pred_cond = self.pipe.denoising_model()(
                        denoising_latents, t, context=prompt_emb['context'],
                        **image_emb_for_denoising, **extra_input
                    )

                next_query_role_boxes = None
                next_query_feature_payload = None
                if probe_extractor is not None:
                    per_char_step_maps = None
                    try:
                        per_char_step_maps = probe_extractor.get_attention_maps_per_character()
                    finally:
                        probe_extractor.remove_hooks()
                    captured_layer_tokens = None
                    if probe_feature_tap is not None:
                        captured_layer_tokens = probe_feature_tap.pop_tokens()
                        probe_feature_tap.remove()
                    probe_layer_tokens = captured_layer_tokens
                    if isinstance(per_char_step_maps, list) and len(per_char_step_maps) > 0 and isinstance(probe_ordered_roles, list):
                        probe_role_to_maps = {}
                        for rid, maps in zip(probe_ordered_roles, per_char_step_maps):
                            rid = str(rid)
                            if isinstance(maps, dict) and len(maps) > 0:
                                probe_role_to_maps[rid] = maps
                    _, _, _, h_lat, w_lat = denoising_latents.shape
                    patch_size = self.pipe.dit.patch_size
                    h_patch = h_lat // patch_size[1]
                    w_patch = w_lat // patch_size[2]
                    next_query_role_boxes, next_query_feature_payload = self._build_character_mask_payload_from_probe(
                        per_char_step_maps=per_char_step_maps,
                        ordered_roles=probe_ordered_roles,
                        h_patch=h_patch,
                        w_patch=w_patch,
                        layer_tokens=captured_layer_tokens,
                    )
                    if isinstance(next_query_role_boxes, dict) and len(next_query_role_boxes) > 0:
                        prev_step_role_boxes = dict(next_query_role_boxes)
                if isinstance(next_query_role_boxes, dict) and len(next_query_role_boxes) > 0:
                    cached_query_role_boxes = next_query_role_boxes
                if isinstance(next_query_feature_payload, dict) and len(next_query_feature_payload) > 0:
                    cached_query_feature_payload = next_query_feature_payload

                if use_memory_path and getattr(self.args, 'cfg_uncond_with_memory', True):
                    emb_kwargs_uncond = dict(image_emb)
                    if memory_token_lengths_per_character is not None:
                        emb_kwargs_uncond['memory_token_lengths_per_character'] = memory_token_lengths_per_character
                    if isinstance(active_query_role_boxes, dict) and len(active_query_role_boxes) > 0:
                        emb_kwargs_uncond['query_role_boxes'] = active_query_role_boxes
                    if isinstance(active_query_feature_payload, dict) and len(active_query_feature_payload) > 0:
                        emb_kwargs_uncond['query_feature_payload'] = active_query_feature_payload
                    noise_pred_uncond = self._memory_aware_dit_forward(
                        x=denoising_latents, t=t, context=neg_emb['context'],
                        memory_tokens=memory_tokens,
                        memory_bank_tokens=memory_bank_tokens,
                        memory_bank_percents=memory_bank_percents,
                        memory_bank_token_meta=memory_bank_token_meta,
                        **emb_kwargs_uncond, **extra_input
                    )
                else:
                    noise_pred_uncond = self.pipe.denoising_model()(
                        denoising_latents, t, context=neg_emb['context'],
                        **image_emb_for_denoising, **self.pipe.prepare_extra_input(denoising_latents)
                    )

                noise_pred = noise_pred_uncond + self.args.cfg_scale * (noise_pred_cond - noise_pred_uncond)
                if teacher_forced_probe is not None:
                    return {
                        "prediction": noise_pred,
                        "timestep_index": int(i),
                        "timestep": float(t.detach().float().reshape(-1)[0].item()),
                        "memory_read_hit": bool(use_memory_path),
                        "sparse_role_memory_stats": getattr(self, "_last_sparse_role_memory_stats", {}),
                        "sparse_role_memory_stats_by_layer": getattr(self, "_last_sparse_role_memory_stats_by_layer", {}),
                        "writer_stats": getattr(self, "_last_jigsaw_stage2_writer_stats", {}),
                    }
                denoising_latents = self.pipe.scheduler.step(noise_pred, t[0], denoising_latents)

                if mapping_recorder is not None and len(mapping_recorder) > 0:
                    role_matches = {}
                    selected_meta = mapping_recorder.get('selected_meta', [])
                    if isinstance(selected_meta, list):
                        # Pre-create role buckets so empty matches are still visualizable.
                        for m in selected_meta:
                            if not isinstance(m, dict):
                                continue
                            role = str(m.get('char_id', 'unknown'))
                            info = role_matches.setdefault(role, {
                                'source_chunk_idx': int(m.get('source_chunk_idx', -1)),
                                'source_num_latent_frames': int(m.get('source_num_latent_frames', 1)),
                                'h_patch': int(m.get('h_patch', 1)),
                                'w_patch': int(m.get('w_patch', 1)),
                                'matches': [],
                            })
                            if info['source_chunk_idx'] < 0 and int(m.get('source_chunk_idx', -1)) >= 0:
                                info['source_chunk_idx'] = int(m.get('source_chunk_idx', -1))

                    idx1_cpu = mapping_recorder.get('idx1', None)
                    sim1_cpu = mapping_recorder.get('sim1', None)
                    inject_mask_cpu = mapping_recorder.get('inject_mask', None)
                    inject_mask_after_otsu_cpu = mapping_recorder.get('inject_mask_after_otsu', None)
                    inject_mask_after_reverse_cpu = mapping_recorder.get('inject_mask_after_reverse', None)
                    inject_mask_after_topn_cpu = mapping_recorder.get('inject_mask_after_topn', None)
                    inject_mask_pre_reverse_cpu = mapping_recorder.get('inject_mask_pre_reverse', None)
                    inject_mask_post_reverse_cpu = mapping_recorder.get('inject_mask_post_reverse', None)
                    gen_pos_cpu = mapping_recorder.get('gen_pos', None)
                    selected_role_index_cpu = mapping_recorder.get('selected_role_index', None)
                    selected_role_names = mapping_recorder.get('selected_role_names', None)

                    def _build_role_matches_from_mask(mask_cpu):
                        role_matches_local = {}
                        if isinstance(selected_meta, list):
                            for m in selected_meta:
                                if not isinstance(m, dict):
                                    continue
                                role = str(m.get('char_id', 'unknown'))
                                info = role_matches_local.setdefault(role, {
                                    'source_chunk_idx': int(m.get('source_chunk_idx', -1)),
                                    'source_num_latent_frames': int(m.get('source_num_latent_frames', 1)),
                                    'h_patch': int(m.get('h_patch', 1)),
                                    'w_patch': int(m.get('w_patch', 1)),
                                    'matches': [],
                                })
                                if info['source_chunk_idx'] < 0 and int(m.get('source_chunk_idx', -1)) >= 0:
                                    info['source_chunk_idx'] = int(m.get('source_chunk_idx', -1))

                        if idx1_cpu is None or sim1_cpu is None or mask_cpu is None or gen_pos_cpu is None:
                            return role_matches_local

                        inject_indices = torch.nonzero(mask_cpu.to(torch.bool), as_tuple=False).view(-1)
                        for tid in inject_indices.tolist():
                            if tid < 0 or tid >= int(idx1_cpu.numel()):
                                continue
                            mem_idx = int(idx1_cpu[tid].item())
                            if mem_idx < 0 or not isinstance(selected_meta, list) or mem_idx >= len(selected_meta):
                                continue
                            meta = selected_meta[mem_idx]
                            if not isinstance(meta, dict):
                                continue
                            role = str(meta.get('char_id', 'unknown'))
                            bucket = role_matches_local.setdefault(role, {
                                'source_chunk_idx': int(meta.get('source_chunk_idx', -1)),
                                'source_num_latent_frames': int(meta.get('source_num_latent_frames', 1)),
                                'h_patch': int(meta.get('h_patch', 1)),
                                'w_patch': int(meta.get('w_patch', 1)),
                                'matches': [],
                            })
                            gen_t, gen_h, gen_w = [int(x) for x in gen_pos_cpu[tid].tolist()]
                            bucket['matches'].append({
                                'gen_latent_t': gen_t,
                                'gen_latent_h': gen_h,
                                'gen_latent_w': gen_w,
                                'mem_latent_t': int(meta.get('latent_t', 0)),
                                'mem_latent_h': int(meta.get('latent_h', 0)),
                                'mem_latent_w': int(meta.get('latent_w', 0)),
                                'sim': float(sim1_cpu[tid].item()),
                                'winner_similarity': float(sim1_cpu[tid].item()),
                                'mem_role_id': str(meta.get('char_id', 'unknown')),
                                'assigned_char_id': (
                                    str(selected_role_names[int(selected_role_index_cpu[tid].item())])
                                    if (selected_role_index_cpu is not None and isinstance(selected_role_names, list)
                                        and 0 <= int(selected_role_index_cpu[tid].item()) < len(selected_role_names))
                                    else None
                                ),
                                'mem_index': int(mem_idx),
                            })
                        return role_matches_local

                    mask_after_otsu_cpu = inject_mask_after_otsu_cpu
                    if mask_after_otsu_cpu is None:
                        mask_after_otsu_cpu = inject_mask_pre_reverse_cpu if inject_mask_pre_reverse_cpu is not None else inject_mask_cpu

                    mask_after_reverse_cpu = inject_mask_after_reverse_cpu
                    if mask_after_reverse_cpu is None:
                        mask_after_reverse_cpu = inject_mask_post_reverse_cpu if inject_mask_post_reverse_cpu is not None else inject_mask_cpu

                    mask_after_topn_cpu = inject_mask_after_topn_cpu if inject_mask_after_topn_cpu is not None else inject_mask_cpu

                    if mask_after_otsu_cpu is not None and mask_after_reverse_cpu is not None:
                        filtered_by_reverse_mask_cpu = mask_after_otsu_cpu.to(torch.bool) & (~mask_after_reverse_cpu.to(torch.bool))
                    else:
                        filtered_by_reverse_mask_cpu = None

                    if mask_after_reverse_cpu is not None and mask_after_topn_cpu is not None:
                        filtered_by_topn_mask_cpu = mask_after_reverse_cpu.to(torch.bool) & (~mask_after_topn_cpu.to(torch.bool))
                    else:
                        filtered_by_topn_mask_cpu = None

                    role_matches_after_otsu = _build_role_matches_from_mask(mask_after_otsu_cpu)
                    role_matches_after_reverse = _build_role_matches_from_mask(mask_after_reverse_cpu)
                    role_matches_after_topn = _build_role_matches_from_mask(mask_after_topn_cpu)
                    role_matches_filtered_by_reverse = _build_role_matches_from_mask(filtered_by_reverse_mask_cpu)
                    role_matches_filtered_by_topn = _build_role_matches_from_mask(filtered_by_topn_mask_cpu)
                    role_matches = role_matches_after_topn

                    updated_role_boxes = dict(prev_step_role_boxes) if isinstance(prev_step_role_boxes, dict) else {}
                    for role, info in role_matches_after_topn.items():
                        if not isinstance(info, dict):
                            continue
                        matches = info.get('matches', [])
                        if not isinstance(matches, list) or len(matches) == 0:
                            continue
                        per_t_acc = {}
                        for m in matches:
                            if not isinstance(m, dict):
                                continue
                            lt = int(m.get('gen_latent_t', 0))
                            gh = int(m.get('gen_latent_h', 0))
                            gw = int(m.get('gen_latent_w', 0))
                            if lt not in per_t_acc:
                                per_t_acc[lt] = [gw, gh, gw, gh]
                            else:
                                acc = per_t_acc[lt]
                                acc[0] = min(acc[0], gw)
                                acc[1] = min(acc[1], gh)
                                acc[2] = max(acc[2], gw)
                                acc[3] = max(acc[3], gh)
                        if len(per_t_acc) == 0:
                            continue
                        per_t_boxes = {}
                        for lt, acc in per_t_acc.items():
                            x1 = float(acc[0])
                            y1 = float(acc[1])
                            x2 = float(acc[2] + 1)
                            y2 = float(acc[3] + 1)
                            per_t_boxes[int(lt)] = [x1, y1, x2, y2]
                        updated_role_boxes[str(role)] = per_t_boxes
                    prev_step_role_boxes = updated_role_boxes

                    online_results['feature_mapping_steps'].append({
                        'step_idx': int(i),
                        'role_matches_after_otsu': role_matches_after_otsu,
                        'role_matches_after_reverse': role_matches_after_reverse,
                        'role_matches_after_topn': role_matches_after_topn,
                        'role_matches_filtered_by_reverse': role_matches_filtered_by_reverse,
                        'role_matches_filtered_by_topn': role_matches_filtered_by_topn,
                        'role_matches_before_reverse': role_matches_after_otsu,
                        'role_matches_filtered_out': role_matches_filtered_by_reverse,
                        'role_matches': role_matches,
                        'gen_f': int(mapping_recorder.get('f', 1)),
                        'gen_h_patch': int(mapping_recorder.get('h', 1)),
                        'gen_w_patch': int(mapping_recorder.get('w', 1)),
                        'selected_role_names': mapping_recorder.get('selected_role_names', None),
                    })

                step_viz_payload = {}
                step_viz_spatial_shape = None
                if collect_chars and (should_collect_bank_tokens or should_collect_viz or should_collect_step_viz):
                    for char in collect_chars:
                        role_step_maps = None
                        if isinstance(probe_role_to_maps, dict):
                            role_step_maps = probe_role_to_maps.get(str(char), None)
                        if not (isinstance(role_step_maps, dict) and len(role_step_maps) > 0):
                            continue
                        negative_role_step_maps = self._get_negative_character_semantic_responses(probe_role_to_maps, str(char))
                        extracted = self._extract_memory_from_step_maps(
                            step_maps=role_step_maps,
                            noisy_latents=denoising_latents,
                            char_id=char,
                            negative_step_maps=negative_role_step_maps,
                            char_latent_boxes=prev_step_role_boxes.get(char, {}) if isinstance(prev_step_role_boxes, dict) else None,
                            return_positions=getattr(self.args, 'save_memory_viz', False),
                            return_token_meta=True,
                            token_source_override=probe_layer_tokens,
                        )
                        if extracted is None:
                            continue

                        if getattr(self.args, 'save_memory_viz', False):
                            if isinstance(extracted, tuple) and len(extracted) == 5:
                                mem, positions, top_positions, spatial_shape, token_meta = extracted
                            elif isinstance(extracted, tuple) and len(extracted) == 4:
                                mem, positions, top_positions, spatial_shape = extracted
                                token_meta = []
                            else:
                                mem, positions, spatial_shape = extracted
                                top_positions = None
                                token_meta = []
                            if positions is not None and positions.numel() > 0:
                                prev_payload = online_results['viz'].get(char)
                                if prev_payload is None:
                                    merged_selected = positions
                                    merged_top = top_positions
                                else:
                                    prev_selected = prev_payload.get('selected_positions')
                                    prev_top = prev_payload.get('top_positions')
                                    merged_selected = torch.cat([prev_selected, positions], dim=0) if prev_selected is not None else positions
                                    if top_positions is not None and prev_top is not None:
                                        merged_top = torch.cat([prev_top, top_positions], dim=0)
                                    elif top_positions is not None:
                                        merged_top = top_positions
                                    else:
                                        merged_top = prev_top
                                online_results['viz'][char] = {
                                    'selected_positions': merged_selected,
                                    'top_positions': merged_top,
                                }
                                online_results['spatial_shape'] = spatial_shape
                            if should_collect_step_viz:
                                step_viz_payload[char] = {
                                    'selected_positions': positions,
                                    'top_positions': top_positions,
                                }
                                step_viz_spatial_shape = spatial_shape
                        else:
                            if isinstance(extracted, tuple) and len(extracted) == 2:
                                mem, token_meta = extracted
                            else:
                                mem = extracted
                                token_meta = []

                        if mem is None or (isinstance(mem, torch.Tensor) and mem.numel() == 0):
                            continue
                        if should_collect_bank_tokens:
                            for bank_idx in step_to_collect_banks[i]:
                                online_results['tokens'][char][int(bank_idx)] = mem
                                online_results['token_meta'][char][int(bank_idx)] = token_meta

                if need_any_step_decode_viz and i in target_denoise_steps:
                    denoise_step_records.append({
                        'step_number': int(i + 1),
                        'loop_idx': int(i),
                        'timestep': int(t[0].item()) if isinstance(t, torch.Tensor) else int(t),
                        'latents': denoising_latents.detach().to('cpu', dtype=torch.float32),
                        'viz': step_viz_payload,
                        'spatial_shape': step_viz_spatial_shape,
                        'latent_frames': int(denoising_latents.shape[2]),
                    })

                log_every = int(getattr(self.args, 'memory_runtime_log_every', 5) or 0)
                if log_every > 0 and (i % log_every == 0):
                    meta = getattr(self, '_last_v9_infer_fusion_meta', {}) if use_memory_path else {}
                    print(
                        f"  [V9][StepLog] step={i}/{len(self.pipe.scheduler.timesteps)-1} "
                        f"p_cur={meta.get('p_cur')} p_fusion={meta.get('p_fusion')} "
                        f"bank_idx={meta.get('selected_bank_idx')} bank_p={meta.get('selected_bank_percent')} "
                        f"outside_blocked={meta.get('outside_range_blocked')} memory_disabled={meta.get('memory_disabled')} "
                        f"inject_ratio={meta.get('inject_ratio')} sim1_mean={meta.get('sim1_mean')} tau_sim={meta.get('tau_sim')} sim_mode={meta.get('sim_mode')} "
                        f"relrope_en={meta.get('relrope_enabled')} relrope_q={meta.get('relrope_query_valid_ratio')} relrope_m={meta.get('relrope_memory_valid_ratio')} "
                        f"viz_hit={bool(i in viz_collect_steps)} viz_steps={sorted(viz_collect_steps)}",
                        flush=True,
                    )
                    sparse_stats = getattr(self, '_last_sparse_role_memory_stats', {}) if use_memory_path else {}
                    if isinstance(sparse_stats, dict) and len(sparse_stats) > 0:
                        print(
                            f"  [V9][SparseStats] step={i}/{len(self.pipe.scheduler.timesteps)-1} "
                            f"layer={sparse_stats.get('layer_idx')} "
                            f"enabled={sparse_stats.get('enabled')} "
                            f"q_tokens={sparse_stats.get('selected_query_tokens')} "
                            f"mem_tokens={sparse_stats.get('selected_memory_tokens')} "
                            f"winner_counts={sparse_stats.get('winner_counts')} "
                            f"role_norm={sparse_stats.get('role_head_out_norm')} "
                            f"plain_norm={sparse_stats.get('plain_head_out_norm')} "
                            f"attn_entropy={sparse_stats.get('attn_entropy')}",
                            flush=True,
                        )
                    sparse_stats_by_layer = getattr(self, '_last_sparse_role_memory_stats_by_layer', {}) if use_memory_path else {}
                    if isinstance(sparse_stats_by_layer, dict) and len(sparse_stats_by_layer) > 0:
                        for sparse_layer_key in sorted(sparse_stats_by_layer.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
                            layer_stats = sparse_stats_by_layer.get(sparse_layer_key, {})
                            if not isinstance(layer_stats, dict):
                                continue
                            print(
                                f"  [V9][SparseStatsByLayer] step={i}/{len(self.pipe.scheduler.timesteps)-1} "
                                f"layer={layer_stats.get('layer_idx', sparse_layer_key)} "
                                f"enabled={layer_stats.get('enabled')} "
                                f"q_tokens={layer_stats.get('selected_query_tokens')} "
                                f"mem_tokens={layer_stats.get('selected_memory_tokens')} "
                                f"winner_counts={layer_stats.get('winner_counts')} "
                                f"role_norm={layer_stats.get('role_head_out_norm')} "
                                f"plain_norm={layer_stats.get('plain_head_out_norm')} "
                                f"attn_entropy={layer_stats.get('attn_entropy')} "
                                f"layer_scale={layer_stats.get('applied_layer_scale')} "
                                f"raw_delta_norm={layer_stats.get('raw_delta_norm')} "
                                f"effective_delta_norm={layer_stats.get('effective_delta_norm')} "
                                f"host_token_norm={layer_stats.get('host_token_norm')}",
                                flush=True,
                            )

            def _online_result_has_tokens(payload):
                if not isinstance(payload, dict):
                    return False
                tokens_by_char = payload.get('tokens', None)
                if not isinstance(tokens_by_char, dict):
                    return False
                for bank_dict in tokens_by_char.values():
                    if not isinstance(bank_dict, dict):
                        continue
                    for value in bank_dict.values():
                        if isinstance(value, torch.Tensor) and value.numel() > 0:
                            return True
                        if isinstance(value, dict):
                            for layer_value in value.values():
                                if isinstance(layer_value, torch.Tensor) and layer_value.numel() > 0:
                                    return True
                return False

            online_extract_mode = str(getattr(self.args, 'online_memory_extract_mode', 'denoise_step')).strip().lower()
            if online_extract_mode == 'clean_recache' and len(collect_chars) > 0:
                fallback_to_denoise = bool(getattr(self.args, 'online_memory_clean_fallback_to_denoise', True))
                clean_results = None
                if hasattr(self, '_collect_online_memory_clean_recache') and callable(getattr(self, '_collect_online_memory_clean_recache')):
                    try:
                        clean_results = self._collect_online_memory_clean_recache(
                            prompt=prompt,
                            clean_latents=denoising_latents,
                            cond_context=prompt_emb['context'],
                            uncond_context=neg_emb['context'],
                            image_emb_for_denoising=image_emb_for_denoising,
                            collect_chars=collect_chars,
                        )
                    except Exception as exc:
                        if not fallback_to_denoise:
                            raise
                        print(f"  [OnlineMemoryCleanRecache] failed; fallback_to_denoise=1 error={exc}", flush=True)
                elif not fallback_to_denoise:
                    raise RuntimeError("online_memory_extract_mode=clean_recache requires _collect_online_memory_clean_recache")

                if _online_result_has_tokens(clean_results):
                    clean_results['denoise_step_records'] = denoise_step_records
                    online_results = clean_results
                    print("  [OnlineMemoryCleanRecache] collected clean t=0 online memory", flush=True)
                elif clean_results is not None and not fallback_to_denoise:
                    online_results = clean_results
                    print("  [OnlineMemoryCleanRecache] clean recache produced no tokens; fallback disabled", flush=True)
                elif clean_results is not None:
                    print("  [OnlineMemoryCleanRecache] clean recache produced no tokens; keeping denoise-step memory", flush=True)

            # 5. Decode
            self.pipe.dit.to("cpu")
            self.pipe.vae.to(self.device)

            if need_any_step_decode_viz and len(denoise_step_records) > 0:
                for record in denoise_step_records:
                    latent_snapshot = record.get('latents', None)
                    if latent_snapshot is None:
                        continue
                    latent_snapshot = latent_snapshot.to(self.device, dtype=self.dtype)
                    step_video_tensor = self.pipe.decode_video(
                        latent_snapshot, tiled=self.args.tiled,
                        tile_size=tuple(self.args.tile_size) if self.args.tiled else None,
                        tile_stride=tuple(self.args.tile_stride) if self.args.tiled else None
                    )
                    record['decoded_frames'] = self.pipe.tensor2video(step_video_tensor[0])
                    del step_video_tensor
                    torch.cuda.empty_cache()

            video_tensor = self.pipe.decode_video(
                denoising_latents, tiled=self.args.tiled,
                tile_size=tuple(self.args.tile_size) if self.args.tiled else None,
                tile_stride=tuple(self.args.tile_stride) if self.args.tiled else None
            )
            self.pipe.vae.to("cpu")

            result = self.pipe.tensor2video(video_tensor[0])
            online_results['denoise_step_records'] = denoise_step_records
            return result, denoising_latents, online_results
