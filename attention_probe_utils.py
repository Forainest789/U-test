import torch
import os
import pickle
import numpy as np
import random
import pickle
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn
import cv2
import sys  
import subprocess
from tqdm import tqdm
import torch, os
import time
from datetime import datetime
from utils.image_process import *
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm

height = 480
width = 832
seed = None
max_frames = 81
use_teacache = False

def load_prompts_from_file(prompt_file_path):
    """Load prompts from a prompt.txt file"""
    if not os.path.exists(prompt_file_path):
        print(f"Warning: prompt file not found at {prompt_file_path}")
        return ["Default prompt: the subject is moving naturally"]
    
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract prompts from the file content
        # Support both "prompt = [" and "prompts = [" formats
        local_vars = {}
        
        # Try to find either pattern
        if 'prompt = [' in content or 'prompts = [' in content:
            # Find the assignment statement
            for pattern in ['prompt = [', 'prompts = [']:
                start_idx = content.find(pattern)
                if start_idx != -1:
                    print(f"Found pattern '{pattern}' at index {start_idx}")
                    try:
                        # Execute the assignment to extract prompts
                        exec(content[start_idx:], {}, local_vars)
                        # Check for both 'prompt' and 'prompts' keys
                        result = local_vars.get('prompt') or local_vars.get('prompts')
                        if result:
                            print(f"Successfully loaded {len(result)} prompts from file")
                            # Debug: print first prompt preview
                            if len(result) > 0:
                                print(f"First prompt preview: '{result[0][:80]}...'")
                            return result
                    except Exception as exec_error:
                        print(f"Error executing prompt extraction: {exec_error}")
                        continue
        
        # If not in expected format, treat each line as a prompt
        print("Could not find 'prompt = [' or 'prompts = [', trying line-by-line parsing")
        lines = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
        if lines:
            print(f"Loaded {len(lines)} prompts (line-by-line mode)")
            return lines
        else:
            print("No valid prompts found, using default")
            return ["Default prompt: the subject is moving naturally"]
            
    except Exception as e:
        print(f"Error reading prompts from {prompt_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return ["Default prompt: the subject is moving naturally"]


def verify_target_text_is_single_token(pipe, target_text):
    """
    Verify and extract valid semantic tokens from target_text.
    Filters out meaningless tokens like articles, spaces, and punctuation.
    Returns: (valid_token_ids, valid_token_texts, total_token_count)
    """
    # Use the T5 tokenizer from the pipeline
    tokenizer = pipe.prompter.tokenizer
    
    # Tokenize the target text using the underlying HuggingFace tokenizer
    tokens = tokenizer.tokenizer.encode(target_text, add_special_tokens=False)
    
    # Define meaningless tokens to filter out (common articles, prepositions, punctuation, spaces)
    # T5 uses '▁' for space prefix
    meaningless_patterns = {
        '▁a', '▁an', '▁the', '▁of', '▁in', '▁on', '▁at', '▁to', '▁for', '▁with',
        '▁', ' ', '', ',', '.', '!', '?', ';', ':', '-', '_'
    }
    
    # Filter tokens
    valid_token_ids = []
    valid_token_texts = []
    
    for token_id in tokens:
        decoded = tokenizer.tokenizer.decode([token_id])
        # Check if token is meaningful (not in meaningless patterns and not just whitespace)
        if decoded.strip() and decoded not in meaningless_patterns:
            valid_token_ids.append(token_id)
            valid_token_texts.append(decoded)
    
    token_count = len(tokens)
    valid_count = len(valid_token_ids)
    

    
    return valid_token_ids, valid_token_texts, token_count


def find_token_index_in_prompt(pipe, prompt, target_text, token_ids, token_texts, use_all_occurrences=False):
    """
    Find the indices of all target tokens in the encoded prompt.
    直接匹配时忽略大小写：若精确匹配未找到，则在 prompt 中按忽略大小写查找 target_text，
    用 prompt 中的实际片段（如 "Tank"）再查 token 位置。
    
    Args:
        use_all_occurrences: If True, return ALL occurrences of each token (for multi-occurrence handling)
                            If False, return only the first occurrence of each token (default behavior)
    
    Returns: list of token indices
    """
    import re
    tokenizer = pipe.prompter.tokenizer
    
    # Encode the full prompt using the underlying HuggingFace tokenizer
    encoded = tokenizer.tokenizer.encode(prompt, add_special_tokens=True)
    
    # Common special token IDs to filter out (safety check)
    # These should not be in token_ids from verify_target_text_is_single_token, but double-check
    special_token_ids = {0, 1}  # <pad>=0, </s>=1
    
    def find_indices_for_tokens(tids, ttexts):
        found = []
        for token_id, token_text in zip(tids, ttexts):
            if token_id in special_token_ids:
                continue
            indices = [i for i, t in enumerate(encoded) if t == token_id]
            if indices:
                if use_all_occurrences:
                    found.extend(indices)
                else:
                    found.append(indices[0])
        return found if found else None
    
    # 1. 精确匹配
    found_indices = find_indices_for_tokens(token_ids, token_texts)
    
    # 2. 未找到且 target_text 非空时：忽略大小写在 prompt 中查找，用 prompt 中的实际片段再查 token
    if not found_indices and target_text and isinstance(prompt, str):
        m = re.search(re.escape(target_text), prompt, re.IGNORECASE)
        if m:
            actual_substring = m.group(0)
            if actual_substring != target_text:
                alt_token_ids, alt_token_texts, _ = verify_target_text_is_single_token(pipe, actual_substring)
                if alt_token_ids:
                    found_indices = find_indices_for_tokens(alt_token_ids, alt_token_texts)
    
    if not found_indices:
        return None
    
    return found_indices


from collections import deque
class AttentionMapExtractor:
    """
    Extracts cross-attention maps from specified DiT layers for target text tokens.
    Supports suffix weighting (e.g., for "Harry_Potter", weight "Harry" normally and "Potter" by a scale).
    """
    def __init__(self, pipe, target_layers, target_token_indices, suffix_token_indices=None, suffix_scale=1.0, cfg_scale=1.0, token_weight=0.2):
        self.pipe = pipe
        self.target_layers = target_layers
        
        # Primary tokens (Prefix)
        if isinstance(target_token_indices, int):
            self.target_token_indices = [target_token_indices]
        else:
            self.target_token_indices = target_token_indices
            
        # Suffix tokens (Secondary)
        self.suffix_token_indices = suffix_token_indices if suffix_token_indices is not None else []
        self.suffix_scale = suffix_scale
        
        self.cfg_scale = cfg_scale
        self.token_weight = token_weight  # Weight for non-first tokens in the prefix group
        self.attention_maps = {} 
        self.hooks = []
        self.modified_modules = []
        
    def register_hooks(self):
        """Register forward hooks to capture attention maps."""
        dit_model = self.pipe.dit
        num_blocks = len(dit_model.blocks)
        
        # Handle -1 as all layers
        if isinstance(self.target_layers, list) and -1 in self.target_layers:
            processed_layers = list(range(num_blocks))
        else:
            processed_layers = self.target_layers
        
        # Combine all indices for optimization (pass all to the kernel at once)
        # Sequence: [Prefix_Tokens..., Suffix_Tokens...]
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
                        # attn shape: [B, H, N_visual, N_selected_tokens]
                        # Columns 0 to num_prefix-1 are Prefix
                        # Columns num_prefix to end are Suffix
                        attn = module.attn_weights 
                        
                        target_attn = None
                        
                        # 1. Process Prefix Tokens (Apply token_weight logic)
                        if num_prefix > 0:
                            # Extract prefix columns: [B, H, N_vis, num_prefix]
                            prefix_attn_chunk = attn[..., :num_prefix]
                            
                            if num_prefix > 1:
                                # First prefix token gets 1.0, others get self.token_weight
                                weights = torch.tensor([1.0] + [self.token_weight] * (num_prefix - 1), 
                                                      device=attn.device, dtype=attn.dtype)
                                # Reshape for broadcasting: [1, 1, 1, num_prefix]
                                weights = weights.view(1, 1, 1, -1)
                                prefix_score = (prefix_attn_chunk * weights).sum(dim=-1)
                            else:
                                prefix_score = prefix_attn_chunk.squeeze(-1)
                        else:
                            prefix_score = 0.0
                            
                        # 2. Process Suffix Tokens (Apply suffix_scale)
                        if len(self.suffix_token_indices) > 0:
                            # Extract suffix columns: [B, H, N_vis, num_suffix]
                            suffix_attn_chunk = attn[..., num_prefix:]
                            
                            # Simple sum for suffix, then scale
                            suffix_score = suffix_attn_chunk.sum(dim=-1)
                            
                            if isinstance(prefix_score, float) and prefix_score == 0.0:
                                target_attn = suffix_score * self.suffix_scale
                            else:
                                target_attn = prefix_score + (suffix_score * self.suffix_scale)
                        else:
                            target_attn = prefix_score

                        self.attention_maps[layer_idx].append(target_attn.detach().float().cpu())
                        module.attn_weights = None # Clear memory
                return hook_fn
            
            # Identify Attention Module
            target_module = None
            if hasattr(block, 'cross_attn'):
                target_module = block.cross_attn
            elif hasattr(block, 'attn2'):
                target_module = block.attn2
            
            if target_module is not None:
                target_module.save_attn_weights = True
                # Pass COMBINED indices so the model computes attention for both groups efficiently
                target_module.target_token_idx = all_target_indices
                
                self.modified_modules.append(target_module)
                hook = target_module.register_forward_hook(make_hook(layer_idx))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        for module in self.modified_modules:
            module.save_attn_weights = False
            module.attn_weights = None
            if hasattr(module, 'target_token_idx'):
                del module.target_token_idx
        self.modified_modules = []
        self.attention_maps = {}
    
    def get_attention_maps(self):
        results = {}
        for layer_idx, dq in self.attention_maps.items():
            if not dq: continue
            if self.cfg_scale > 1.0:
                if len(dq) >= 2:
                    attn = dq[0]
                else:
                    attn = dq[-1]
                # CFG差值: cond - uncond, 突出角色独有的注意力区域
                if attn.dim() >= 2 and attn.shape[0] == 2:
                    attn = (attn[1] - attn[0]).clamp(min=0)
                results[layer_idx] = attn
            else:
                results[layer_idx] = dq[-1] # Cond (no uncond pass)
        return results


class MultiCharacterAttentionMapExtractor:
    """
    多角色共用一次 DiT forward：对 union 的 token 取 attention [B,H,Sq,K]，
    在 hook 内按每个角色的 prefix/suffix 列做加权求和，得到每个角色一张 map。
    char_configs: list of dict, 每项 {target_token_indices, suffix_token_indices, suffix_scale, token_weight}。
    """
    def __init__(self, pipe, target_layers, char_configs, cfg_scale=1.0):
        self.pipe = pipe
        self.target_layers = target_layers
        self.char_configs = char_configs
        self.cfg_scale = cfg_scale
        # 建 union 与每角色在 union 中的列下标
        all_indices = []
        for c in char_configs:
            pre = c.get('target_token_indices', [])
            suf = c.get('suffix_token_indices', [])
            if isinstance(pre, int):
                pre = [pre]
            if isinstance(suf, int):
                suf = [suf]
            all_indices.extend(pre)
            all_indices.extend(suf)
        self.union_indices = sorted(set(all_indices))
        self.union_to_col = {idx: i for i, idx in enumerate(self.union_indices)}
        self.char_slices = []
        for c in char_configs:
            pre = c.get('target_token_indices', [])
            suf = c.get('suffix_token_indices', [])
            if isinstance(pre, int):
                pre = [pre]
            if isinstance(suf, int):
                suf = [suf]
            pre_cols = [self.union_to_col[i] for i in pre if i in self.union_to_col]
            suf_cols = [self.union_to_col[i] for i in suf if i in self.union_to_col]
            self.char_slices.append({
                'prefix_cols': pre_cols,
                'suffix_cols': suf_cols,
                'suffix_scale': c.get('suffix_scale', 1.0),
                'token_weight': c.get('token_weight', 0.2),
            })
        self.attention_maps = {}  # (layer_idx, char_idx) -> deque
        self.hooks = []
        self.modified_modules = []

    def register_hooks(self):
        dit_model = self.pipe.dit
        num_blocks = len(dit_model.blocks)
        if isinstance(self.target_layers, list) and -1 in self.target_layers:
            processed_layers = list(range(num_blocks))
        else:
            processed_layers = self.target_layers

        for layer_idx in processed_layers:
            if layer_idx >= num_blocks or layer_idx < 0:
                continue
            for char_idx in range(len(self.char_slices)):
                self.attention_maps[(layer_idx, char_idx)] = deque(maxlen=2)
            block = dit_model.blocks[layer_idx]

            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    if not hasattr(module, 'attn_weights') or module.attn_weights is None:
                        return
                    attn = module.attn_weights  # [B, H, Sq, K]
                    for char_idx, sl in enumerate(self.char_slices):
                        pre_cols = sl['prefix_cols']
                        suf_cols = sl['suffix_cols']
                        suffix_scale = sl['suffix_scale']
                        token_weight = sl['token_weight']
                        prefix_score = 0.0
                        if pre_cols:
                            prefix_attn = attn[..., pre_cols]
                            n = len(pre_cols)
                            if n > 1:
                                w = torch.tensor(
                                    [1.0] + [token_weight] * (n - 1),
                                    device=attn.device, dtype=attn.dtype
                                ).view(1, 1, 1, -1)
                                prefix_score = (prefix_attn * w).sum(dim=-1)
                            else:
                                prefix_score = prefix_attn.squeeze(-1)
                        suffix_score = 0.0
                        if suf_cols:
                            suffix_attn = attn[..., suf_cols].sum(dim=-1)
                            suffix_score = suffix_attn * suffix_scale
                        if isinstance(prefix_score, float) and prefix_score == 0.0:
                            target_attn = suffix_score
                        else:
                            target_attn = prefix_score + suffix_score
                        self.attention_maps[(layer_idx, char_idx)].append(target_attn.detach().float().cpu())
                    module.attn_weights = None
                return hook_fn

            target_module = None
            if hasattr(block, 'cross_attn'):
                target_module = block.cross_attn
            elif hasattr(block, 'attn2'):
                target_module = block.attn2
            if target_module is not None:
                target_module.save_attn_weights = True
                target_module.target_token_idx = self.union_indices
                self.modified_modules.append(target_module)
                hook = target_module.register_forward_hook(make_hook(layer_idx))
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        for module in self.modified_modules:
            module.save_attn_weights = False
            module.attn_weights = None
            if hasattr(module, 'target_token_idx'):
                del module.target_token_idx
        self.modified_modules = []
        self.attention_maps = {}

    def get_attention_maps_per_character(self):
        """Return list of dict: [ {layer_idx: tensor}, ... ]，与 char_configs 一一对应。"""
        nchars = len(self.char_slices)
        out = [{} for _ in range(nchars)]
        for (layer_idx, char_idx), dq in self.attention_maps.items():
            if not dq:
                continue
            if self.cfg_scale > 1.0:
                attn = dq[0] if len(dq) >= 2 else dq[-1]
                if attn.dim() >= 2 and attn.shape[0] == 2:
                    attn = (attn[1] - attn[0]).clamp(min=0)
            else:
                attn = dq[-1]
            out[char_idx][layer_idx] = attn
        return out


def process_attention_map_to_mask(
    attention_map,
    threshold=0.6,
    top_k_per_head=0,
    spatial_shape=(60, 104),
    num_frames=None,
    otsu_scope="clip",
    neighbor_filter_kernel=0,
    neighbor_filter_any_window=False,
):
    """
    Process attention map to binary mask using hybrid selection strategy.
    
    Strategy:
    1. Per-head top-k: Each head selects its top-k tokens across ALL frames (if top_k_per_head > 0)
    2. Global top-p: Select tokens based on averaged attention across all heads and frames
    3. Final selection: Union of tokens from both strategies
    4. Neighbor filter: Keep tokens only if local support in kernel×kernel window is sufficient
    5. Visualization: Show the frame with highest total attention
    
    Args:
        attention_map: [B, num_heads, num_spatial_tokens] or [B, num_spatial_tokens]
        threshold: top-p threshold for global averaged attention (0-1). If -1, use Otsu's method.
        top_k_per_head: number of top tokens to select per head (int, 0 to disable)
        spatial_shape: (H, W) shape to reshape spatial tokens (Latent H, Latent W)
        num_frames: Number of frames (F_lat). If provided, will use this directly for shape resolution.
                   If None, will attempt to infer from num_tokens and spatial_shape.
        otsu_scope: "clip" or "frame". Determines if Otsu is calculated globally or per-frame.
        neighbor_filter_kernel: Kernel size for neighborhood block filter (0 to disable).
        neighbor_filter_any_window: If True, a token is kept when ANY kernel×kernel block
                        that contains this token reaches the auto threshold
                        ceil(kernel*kernel/2).
                        If False, uses centered block count.
    
    Returns:
        binary_mask: [H, W] binary mask for the frame with highest attention
        continuous_map: [H, W] continuous attention map for that frame
        selected_indices: set of selected token indices (1D flattened, across all frames)
    """
    # Handle different input shapes
    # Expected: [B, num_heads, num_tokens], [num_heads, num_tokens], [B, num_tokens], or [num_tokens]
    original_dim = attention_map.dim()
    
    # Remove batch dimension if present (assume batch size 1)
    if attention_map.dim() == 3:
        # [B, num_heads, num_tokens] or [num_heads, num_tokens, something]
        if attention_map.shape[0] == 1:
            attention_map = attention_map.squeeze(0)  # [num_heads, num_tokens]
        else:
            # Assume first dim is batch, take first batch
            attention_map = attention_map[0]  # [num_heads, num_tokens]
    elif attention_map.dim() == 2:
        # Could be [num_heads, num_tokens] or [batch, num_tokens]
        if attention_map.shape[0] == 1:
            # [1, num_tokens] - treat as single head, add head dimension
            attention_map = attention_map.squeeze(0).unsqueeze(0)  # [1, num_tokens]
        # Otherwise assume [num_heads, num_tokens] which is correct
    elif attention_map.dim() == 1:
        # [num_tokens] - add head dimension
        attention_map = attention_map.unsqueeze(0)  # [1, num_tokens]
    
    # Now attention_map should be [num_heads, num_tokens]
    if attention_map.dim() == 2:
        num_heads, num_tokens = attention_map.shape
        has_head_dim = True
    elif attention_map.dim() == 1:
        num_heads, num_tokens = 1, attention_map.shape[0]
        has_head_dim = False
        # Convert to 2D for consistency
        attention_map = attention_map.unsqueeze(0)  # [1, num_tokens]
        has_head_dim = True
    else:
        raise ValueError(f"Unexpected attention_map shape after processing: {attention_map.shape}, dim={attention_map.dim()}")
    
    # After processing, attention_map should always be [num_heads, num_tokens] with has_head_dim = True
    has_head_dim = True
    
    H_latent, W_latent = spatial_shape
    target_aspect = H_latent / W_latent if W_latent > 0 else 1.0
    
    # Calculate spatial tokens per frame
    spatial_tokens_per_frame = H_latent * W_latent
    
    # If num_frames is provided, use it directly (FAST PATH - no guessing needed!)
    found_shape = None
    min_aspect_diff = float('inf')
    
    if num_frames is not None:
        # FAST PATH: Use provided num_frames directly
        F_actual = num_frames
        expected_tokens = F_actual * spatial_tokens_per_frame
        
        if num_tokens == expected_tokens:
            found_shape = (F_actual, H_latent, W_latent)
        else:
            found_shape = (F_actual, H_latent, W_latent)
        
        # Initialize for FAST PATH case
        candidate_shapes = []
        possible_fs = []
    else:
        # SLOW PATH: Try to infer F from num_tokens (original logic)
        possible_fs = [1, 4, 5, 9, 13, 17, 21, 81]  # Common frame counts for Wan
        candidate_shapes = []
        for f in possible_fs:
            if num_tokens % f == 0:
                spatial_tokens = num_tokens // f
                h_est = np.sqrt(spatial_tokens * target_aspect)
                h = int(round(h_est))
                if h == 0:
                    continue
                w = spatial_tokens // h
                
                checked = False
                if h * w == spatial_tokens:
                    aspect = h / w
                    diff = abs(aspect - target_aspect)
                    checked = True
                    candidate_shapes.append((f, h, w, aspect, diff, 'exact'))
                    if diff < min_aspect_diff:
                        min_aspect_diff = diff
                        found_shape = (f, h, w)
                elif (h+1) * w == spatial_tokens:
                    aspect = (h+1) / w
                    diff = abs(aspect - target_aspect)
                    checked = True
                    candidate_shapes.append((f, h+1, w, aspect, diff, 'h+1'))
                    if diff < min_aspect_diff:
                        min_aspect_diff = diff
                        found_shape = (f, h+1, w)
                elif h * (w+1) == spatial_tokens:
                    aspect = h / (w+1)
                    diff = abs(aspect - target_aspect)
                    checked = True
                    candidate_shapes.append((f, h, w+1, aspect, diff, 'w+1'))
                    if diff < min_aspect_diff:
                        min_aspect_diff = diff
                        found_shape = (f, h, w+1)
            # else: num_tokens % f != 0, skip
        
        # Initialize for fallback case (when num_frames is provided, candidate_shapes won't be needed)
        if not candidate_shapes:
            candidate_shapes = []
    
    # Strategy 1: Per-head top-k selection (across ALL frames)
    per_head_indices_set = set()
    if top_k_per_head > 0 and has_head_dim:
        for head_idx in range(num_heads):
            head_scores = attention_map[head_idx]  # [num_spatial_tokens]
            k = min(top_k_per_head, num_tokens)
            _, top_k_indices = torch.topk(head_scores, k, dim=-1)  # [k]
            per_head_indices_set.update(top_k_indices.cpu().tolist())
    
    # Strategy 2: Global top-p selection based on averaged attention
    if has_head_dim:
        avg_scores = attention_map.mean(dim=0)  # [num_spatial_tokens]
    else:
        avg_scores = attention_map  # Already averaged or single head
    
    if threshold == -1:
        if otsu_scope == "frame" and found_shape is not None:
            f, h, w = found_shape
            spatial_dim = h * w
            expected_size = f * spatial_dim
            actual_size = avg_scores.shape[0]
            
            if expected_size != actual_size:
                found_shape = None  # Force fallback to global Otsu
            else:
                frames_scores = avg_scores.reshape(f, spatial_dim)
            
            global_indices_list = []
            
            for t in range(f):
                frame_scores = frames_scores[t]
                scores_np = frame_scores.cpu().float().numpy()
                min_val = scores_np.min()
                max_val = scores_np.max()
                
                if max_val > min_val:
                    # Normalize per frame
                    scores_uint8 = ((scores_np - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    _, binary_mask = cv2.threshold(scores_uint8.reshape(1, -1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    selected_mask = binary_mask.reshape(-1) > 0
                    local_indices = np.where(selected_mask)[0]
                    # Convert to global indices
                    global_offset = t * spatial_dim
                    global_indices_list.extend((local_indices + global_offset).tolist())
                else:
                    # Uniform frame, maybe keep all or none? Usually Otsu fails on uniform.
                    # Keep nothing or all? Let's keep none if perfectly uniform low/high, or all?
                    # If max == min, standard dev is 0.
                    pass 
            
            global_indices_set = set(global_indices_list)
            
        else:
            scores_np = avg_scores.cpu().float().numpy()
            min_val = scores_np.min()
            max_val = scores_np.max()
            if max_val > min_val:
                scores_uint8 = ((scores_np - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                scores_uint8 = np.zeros_like(scores_np, dtype=np.uint8)
            
            otsu_thresh, binary_mask = cv2.threshold(scores_uint8.reshape(1, -1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            selected_mask = binary_mask.reshape(-1) > 0
            global_indices_set = set(np.where(selected_mask)[0].tolist())
    else:
        # Top-p: cumulative probability threshold
        sorted_scores, sorted_indices = torch.sort(avg_scores, dim=-1, descending=True)
        cumsum_scores = torch.cumsum(torch.softmax(sorted_scores, dim=-1), dim=-1)
        num_select = torch.sum(cumsum_scores < threshold).item() + 1
        num_select = min(max(1, num_select), num_tokens)
        
        global_indices_set = set(sorted_indices[:num_select].cpu().tolist())
    
    # Merge: union of per-head and global indices
    all_indices = sorted(list(per_head_indices_set | global_indices_set))
    
    # Neighborhood support filter on per-frame masks.
    if neighbor_filter_kernel > 0 and found_shape is not None:
        f, h, w = found_shape
        spatial_dim = h * w
        if f * spatial_dim == num_tokens:
            filtered_indices = []
            kernel = np.ones((neighbor_filter_kernel, neighbor_filter_kernel), dtype=np.uint8)
            k = int(neighbor_filter_kernel)
            min_count = max(1, int((k * k + 1) // 2))
            for t in range(f):
                # Build per-frame binary mask
                frame_offset = t * spatial_dim
                frame_mask = np.zeros((h, w), dtype=np.uint8)
                for idx in all_indices:
                    if frame_offset <= idx < frame_offset + spatial_dim:
                        local_idx = idx - frame_offset
                        fh, fw = local_idx // w, local_idx % w
                        frame_mask[fh, fw] = 1
                # Count selected tokens in centered kernel window.
                block_count = cv2.filter2D(frame_mask.astype(np.float32), -1, kernel.astype(np.float32))

                # Precompute all valid kernel window sums for the "any containing window" rule.
                window_sum = None
                if neighbor_filter_any_window:
                    if h >= k and w >= k:
                        window_h = h - k + 1
                        window_w = w - k + 1
                        window_sum = np.zeros((window_h, window_w), dtype=np.float32)
                        for r0 in range(window_h):
                            for c0 in range(window_w):
                                window_sum[r0, c0] = float(frame_mask[r0:r0 + k, c0:c0 + k].sum())

                for idx in all_indices:
                    if frame_offset <= idx < frame_offset + spatial_dim:
                        local_idx = idx - frame_offset
                        fh, fw = local_idx // w, local_idx % w
                        keep = False
                        if neighbor_filter_any_window and window_sum is not None:
                            r0_min = max(0, int(fh) - k + 1)
                            r0_max = min(int(fh), h - k)
                            c0_min = max(0, int(fw) - k + 1)
                            c0_max = min(int(fw), w - k)
                            if r0_min <= r0_max and c0_min <= c0_max:
                                for r0 in range(r0_min, r0_max + 1):
                                    for c0 in range(c0_min, c0_max + 1):
                                        if window_sum[r0, c0] >= float(min_count):
                                            keep = True
                                            break
                                    if keep:
                                        break
                        else:
                            keep = bool(block_count[fh, fw] >= float(min_count))

                        if keep:
                            filtered_indices.append(idx)
            all_indices = sorted(filtered_indices)
    
    if found_shape:
        f, h, w = found_shape
        expected_size = f * h * w
        actual_size = avg_scores.shape[0]
        if expected_size != actual_size:
            found_shape = None
            attention_3d = None
        else:
            try:
                attention_3d = avg_scores.reshape(f, h, w)
            except RuntimeError:
                found_shape = None
                attention_3d = None
        
        if attention_3d is not None:
            frame_attentions = attention_3d.reshape(f, -1).sum(dim=1)  # [F]
            best_frame_idx = frame_attentions.argmax().item()
            
            # Select that frame for visualization
            attention_2d = attention_3d[best_frame_idx]  # [H, W]
            
            # Resize to original spatial_shape (Latent H, Latent W)
            attention_2d_np = attention_2d.cpu().numpy()
            attention_2d_resized = cv2.resize(attention_2d_np, (spatial_shape[1], spatial_shape[0]), interpolation=cv2.INTER_LINEAR)
            attention_2d = torch.from_numpy(attention_2d_resized)
            
            # Create binary mask from selected indices
            # Filter indices that belong to the best frame
            frame_start = best_frame_idx * h * w
            frame_end = frame_start + h * w
            frame_indices = [idx - frame_start for idx in all_indices if frame_start <= idx < frame_end]
            
            # Create mask for best frame
            binary_mask_2d = torch.zeros(h, w, dtype=torch.float32)
            for idx in frame_indices:
                binary_mask_2d.view(-1)[idx] = 1.0
            
            # Resize to spatial_shape
            binary_mask_np = binary_mask_2d.cpu().numpy()
            binary_mask_resized = cv2.resize(binary_mask_np, (spatial_shape[1], spatial_shape[0]), interpolation=cv2.INTER_NEAREST)
            binary_mask = binary_mask_resized
        else:
            # If reshape failed, set found_shape to None to trigger fallback
            found_shape = None
        
    else:
        # Fallback to 2D reshape
        h_tokens = int(np.sqrt(num_tokens * target_aspect))
        w_tokens = num_tokens // h_tokens
        fallback_area = h_tokens * w_tokens
        h, w = h_tokens, w_tokens
        
        attention_2d = avg_scores[:fallback_area].reshape(h_tokens, w_tokens)
        
        attention_2d_np = attention_2d.cpu().numpy()
        attention_2d_resized = cv2.resize(attention_2d_np, (spatial_shape[1], spatial_shape[0]), interpolation=cv2.INTER_LINEAR)
        attention_2d = torch.from_numpy(attention_2d_resized)
        
        valid_indices = [idx for idx in all_indices if idx < fallback_area]
        
        binary_mask_2d = torch.zeros(h_tokens, w_tokens, dtype=torch.float32)
        for idx in valid_indices:
            binary_mask_2d.view(-1)[idx] = 1.0
        
        binary_mask_np = binary_mask_2d.cpu().numpy()
        binary_mask_resized = cv2.resize(binary_mask_np, (spatial_shape[1], spatial_shape[0]), interpolation=cv2.INTER_NEAREST)
        binary_mask = binary_mask_resized
    
    # Normalize continuous map to [0, 1]
    attention_2d = (attention_2d - attention_2d.min()) / (attention_2d.max() - attention_2d.min() + 1e-8)
    
    # Prepare frame selection info
    frame_info = {
        'best_frame_idx': best_frame_idx if found_shape else 0,
        'frame_attention_score': frame_attentions[best_frame_idx].item() if found_shape else avg_scores.sum().item(),
        'num_frames': f if found_shape else 1,
        'frame_shape': (h, w),
        'num_selected_in_frame': len(frame_indices) if found_shape else len(valid_indices),
        'total_selected': len(all_indices)
    }
    
    return binary_mask, attention_2d.cpu().numpy(), set(all_indices), frame_info


def save_attention_visualizations(binary_mask, continuous_map, reference_image, 
                                   save_dir, clip_idx, layer_idx, target_text, selected_indices=None, frame_info=None):
    """
    Save attention visualizations: binary mask, heatmap overlay, and reference image.
    
    Args:
        binary_mask: [H, W] numpy array, binary mask
        continuous_map: [H, W] numpy array, continuous attention map
        reference_image: PIL Image or numpy array
        save_dir: directory to save visualizations
        clip_idx: clip index
        layer_idx: layer index
        target_text: target text for filename
        selected_indices: set of selected token indices
        frame_info: dict with frame selection information
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert reference image to numpy if needed
    if isinstance(reference_image, Image.Image):
        ref_img = np.array(reference_image)
    else:
        ref_img = reference_image
    
    h, w = ref_img.shape[:2]
    
    # Resize masks to original image size
    binary_mask_resized = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    continuous_map_resized = cv2.resize(continuous_map, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Create filename base
    if isinstance(layer_idx, int):
        layer_str = f"{layer_idx:02d}"
    else:
        layer_str = str(layer_idx)
    filename_base = f"clip_{clip_idx:02d}_layer_{layer_str}_{target_text.replace(' ', '_')}"
    
    # Add token count to filename if available
    if selected_indices is not None:
        filename_base += f"_tokens{len(selected_indices)}"
        print(f"  Visualizing {len(selected_indices)} selected tokens")
    
    # 1. Save reference image
    ref_path = os.path.join(save_dir, f"{filename_base}_reference.png")
    if isinstance(reference_image, Image.Image):
        reference_image.save(ref_path)
    else:
        cv2.imwrite(ref_path, cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR))
    print(f"  Saved reference image: {ref_path}")
    
    # 2. Save binary mask (black and white)
    mask_bw = (binary_mask_resized * 255).astype(np.uint8)
    mask_path = os.path.join(save_dir, f"{filename_base}_mask_bw.png")
    cv2.imwrite(mask_path, mask_bw)
    print(f"  Saved binary mask: {mask_path}")
    
    # 3. Save heatmap overlay (only for selected tokens)
    # Create colormap from original continuous map
    cmap = cm.get_cmap('jet')
    heatmap_full = cmap(continuous_map_resized)[:, :, :3]  # RGB
    heatmap_full = (heatmap_full * 255).astype(np.uint8)
    
    # Create alpha mask (only show heatmap where mask=1)
    alpha_mask = binary_mask_resized[:, :, np.newaxis]  # [H, W, 1]
    
    # Blend: show heatmap only in selected regions, original image elsewhere
    alpha = 0.5
    overlay = ref_img.copy()
    # Only blend where mask=1
    overlay = np.where(alpha_mask > 0, 
                       cv2.addWeighted(ref_img, 1-alpha, heatmap_full, alpha, 0),
                       ref_img)
    overlay = overlay.astype(np.uint8)
    
    overlay_path = os.path.join(save_dir, f"{filename_base}_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"  Saved overlay: {overlay_path}")
    
    # 4. Save continuous heatmap alone (masked to selected regions only)
    # Show heatmap only in selected regions, black elsewhere
    heatmap_masked = np.where(alpha_mask > 0, heatmap_full, 0).astype(np.uint8)
    heatmap_path = os.path.join(save_dir, f"{filename_base}_heatmap.png")
    cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_masked, cv2.COLOR_RGB2BGR))
    print(f"  Saved heatmap: {heatmap_path}")
    
    # 5. Save full continuous heatmap (all tokens, for comparison)
    full_heatmap_path = os.path.join(save_dir, f"{filename_base}_heatmap_full.png")
    cv2.imwrite(full_heatmap_path, cv2.cvtColor(heatmap_full, cv2.COLOR_RGB2BGR))
    print(f"  Saved full heatmap: {full_heatmap_path}")
    
    # 6. Save frame selection info to JSON
    if frame_info is not None:
        import json
        info_dict = {
            'clip_idx': clip_idx,
            'layer_idx': layer_idx,
            'target_text': target_text,
            'num_selected_tokens': len(selected_indices) if selected_indices else 0,
            'frame_selection': frame_info
        }
        json_path = os.path.join(save_dir, f"{filename_base}_info.json")
        with open(json_path, 'w') as f:
            json.dump(info_dict, f, indent=2)
        print(f"  Saved frame info: {json_path}")
def process_autoregressive_visualization(dino_model, has_dino, all_indices, frame_info, video_frames, dino_threshold, save_dir, clip_idx, target_text, attention_map_full=None, use_bound="None", grounding_model=None):
    """
    Apply DINOv2 autoregressive filtering with Anchor Frame strategy.
    
    Args:
        attention_map_full: [N_total_tokens] or [F, H, W] tensor of attention scores. 
                            Essential for the secondary filtering condition.
        use_bound: 'None', 'bound', or 'mask'.
        grounding_model: Tuple of (grounding_dino_model, sam_predictor) for bounding box extraction.
    """
    if not has_dino or dino_model is None:
        print("DINO model not loaded, skipping autoregressive visualization")
        return

    print(f"\nRunning Autoregressive DINO Visualization for Clip {clip_idx}...")
    
    F_vid = len(video_frames)
    F_attn = frame_info['num_frames']
    H_lat, W_lat = frame_info['frame_shape']
    
    # Ensure alignment
    num_frames = min(F_vid, F_attn)
    
    # -------------------------------------------------------------------------
    # 1. Feature Extraction (Extract DINO features for ALL frames first)
    # -------------------------------------------------------------------------
    # ... (Pre-processing image code remains same as before) ...
    dino_h = H_lat * 14
    dino_w = W_lat * 14
    
    # Prepare DINO inputs
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    dino_batch_list = []
    
    for i in range(num_frames):
        img = video_frames[i]
        if isinstance(img, np.ndarray): img = Image.fromarray(img)
        img_tensor = T.ToTensor()(img)
        img_resized = F.interpolate(img_tensor.unsqueeze(0), size=(dino_h, dino_w), mode='bilinear', align_corners=False).squeeze(0)
        img_norm = (img_resized - mean) / std
        dino_batch_list.append(img_norm)
        
    dino_batch = torch.stack(dino_batch_list).to("cuda")

    print("  Extracting DINO features...")
    with torch.no_grad():
        features_list = []
        chunk_size = 4
        for i in range(0, num_frames, chunk_size):
            batch_chunk = dino_batch[i:i+chunk_size]
            outputs = dino_model.forward_features(batch_chunk)
            feats = outputs['x_norm_patchtokens'] 
            feats = feats.reshape(len(batch_chunk), H_lat, W_lat, -1)
            features_list.append(feats.cpu())
    
    # [num_frames, H_lat*W_lat, Dim]
    dino_features_all = torch.cat(features_list, dim=0) 
    dino_features_flat = dino_features_all.reshape(num_frames, -1, dino_features_all.shape[-1])

    # -------------------------------------------------------------------------
    # 2. Logic: Find Anchor Frame (Best Frame)
    # -------------------------------------------------------------------------
    # Prepare Attention Map for easy access
    if attention_map_full is not None:
        total_spatial = num_frames * H_lat * W_lat
        if attention_map_full.numel() >= total_spatial:
             attn_scores_flat = attention_map_full.view(-1)[:total_spatial].reshape(num_frames, -1) # [F, H*W]
        else:
             print("Warning: Attention map size mismatch. Disabling attention score gating.")
             attn_scores_flat = None
    else:
        attn_scores_flat = None

    # Identify Best Frame (Anchor) based on Otsu-selected tokens
    all_indices_set = set(all_indices)
    best_frame_idx = 0
    max_frame_score = -1.0
    
    print("  Searching for Anchor Frame (Strongest Attention)...")
    for t in range(num_frames):
        frame_start = t * (H_lat * W_lat)
        frame_end = (t + 1) * (H_lat * W_lat)
        frame_indices_local = [idx - frame_start for idx in all_indices if frame_start <= idx < frame_end]
        
        if len(frame_indices_local) > 0 and attn_scores_flat is not None:
            scores = attn_scores_flat[t, frame_indices_local]
            total_score = scores.sum().item()
            if total_score > max_frame_score:
                max_frame_score = total_score
                best_frame_idx = t
    
    print(f"  ✓ Anchor Frame found at index {best_frame_idx} (Score: {max_frame_score:.2f})")

    # -------------------------------------------------------------------------
    # 2.5. Extract Bounding Box / Masks using GroundingSAM (if enabled)
    # -------------------------------------------------------------------------
    bounding_boxes_per_frame = {}
    binary_masks_per_frame = {}
    
    run_spatial = (use_bound != "None") and (grounding_model is not None)
    
    if run_spatial:
        print(f"\n  Extracting spatial constraints (mode='{use_bound}') for '{target_text}'...")
        grounding_dino_model, sam_predictor = grounding_model
        
        for t in range(num_frames):
            frame_img = video_frames[t]
            # Convert to PIL Image with robust handling (Copied from original)
            frame_img_pil = None
            try:
                if isinstance(frame_img, np.ndarray):
                    if len(frame_img.shape) == 3 and frame_img.shape[-1] == 3:
                        if frame_img.dtype == np.float32 or frame_img.dtype == np.float64:
                            frame_img = (frame_img * 255).clip(0, 255).astype(np.uint8)
                        elif frame_img.dtype != np.uint8:
                            frame_img = frame_img.astype(np.uint8)
                        frame_img_pil = Image.fromarray(frame_img, mode='RGB')
                elif isinstance(frame_img, torch.Tensor):
                    if frame_img.is_cuda: frame_img = frame_img.cpu()
                    if frame_img.dim() == 3:
                        if frame_img.shape[0] == 3: frame_img = frame_img.permute(1, 2, 0)
                        if frame_img.max() <= 1.0: frame_img = (frame_img * 255).clamp(0, 255)
                        frame_img_np = frame_img.numpy().astype(np.uint8)
                        frame_img_pil = Image.fromarray(frame_img_np, mode='RGB')
                elif isinstance(frame_img, Image.Image):
                    frame_img_pil = frame_img.copy()
                
                if frame_img_pil is None: continue 
                if frame_img_pil.mode != 'RGB': frame_img_pil = frame_img_pil.convert('RGB')
                frame_img_np_rgb = np.array(frame_img_pil)
                frame_img_bgr = frame_img_np_rgb[:, :, ::-1].copy()
                
                detections, phrases = grounding_dino_model.predict_with_caption(
                    image=frame_img_bgr,
                    caption=target_text,
                    box_threshold=0.25,
                    text_threshold=0.2
                )
            except Exception as e:
                print(f"    Frame {t}: Error: {e}")
                continue
            
            try:
                if detections is not None and len(detections.xyxy) > 0:
                    boxes_pixel = []
                    for box in detections.xyxy:
                        x1, y1, x2, y2 = box.astype(int)
                        boxes_pixel.append([x1, y1, x2, y2])
                    bounding_boxes_per_frame[t] = boxes_pixel
                    
                    if use_bound == "mask" and sam_predictor is not None:
                        try:
                            sam_predictor.set_image(frame_img_np_rgb)
                            H_img, W_img = frame_img_np_rgb.shape[:2]
                            boxes_torch = torch.from_numpy(detections.xyxy).to("cuda")
                            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_torch, (H_img, W_img))
                            masks, _, _ = sam_predictor.predict_torch(
                                point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False,
                            )
                            if masks.numel() > 0:
                                final_mask = torch.any(masks, dim=0).squeeze(0).cpu().numpy()
                                binary_masks_per_frame[t] = final_mask
                        except Exception as e_sam:
                            print(f"    Frame {t}: SAM error: {e_sam}")
            except Exception as e:
                print(f"    Frame {t}: Error post-processing: {e}")
        print(f"  ✓ Spatial extraction complete.")

    # -------------------------------------------------------------------------
    # 3. Initialize Memory Bank with Anchor Frame (MODIFIED: Applied Spatial Filter)
    # -------------------------------------------------------------------------
    memory_bank = None  # [N_stored, Dim]
    attn_score_threshold = 0.0
    
    frame_start = best_frame_idx * (H_lat * W_lat)
    frame_end = (best_frame_idx + 1) * (H_lat * W_lat)
    anchor_indices_local = [idx - frame_start for idx in all_indices if frame_start <= idx < frame_end]
    
    # === NEW: Spatial Filter for Anchor Frame Initial Tokens ===
    # We must ensure the initial memory does not contain background noise
    if use_bound != "None" and len(anchor_indices_local) > 0:
        print(f"  Filtering Anchor Frame (Frame {best_frame_idx}) with spatial constraints...")
        
        # Determine strictness variables
        check_spatial = False
        spatial_mode = 'none' 
        current_boxes = []
        current_mask = None

        if use_bound == 'mask' and best_frame_idx in binary_masks_per_frame:
             check_spatial = True
             current_mask = binary_masks_per_frame[best_frame_idx]
             spatial_mode = 'mask'
        elif use_bound == 'bound' and best_frame_idx in bounding_boxes_per_frame:
             check_spatial = True
             current_boxes = bounding_boxes_per_frame[best_frame_idx]
             spatial_mode = 'box'
        
        if check_spatial:
            # Get dimensions for coordinate mapping (Reusing helper logic)
            ref_img_pil_temp = video_frames[best_frame_idx]
            h_img, w_img = 0, 0
            # Handle different formats
            if isinstance(ref_img_pil_temp, np.ndarray):
                h_img, w_img = ref_img_pil_temp.shape[:2]
            elif isinstance(ref_img_pil_temp, Image.Image):
                w_img, h_img = ref_img_pil_temp.size
            else: # Tensor
                 h_img, w_img = ref_img_pil_temp.shape[1:3]

            filtered_anchor_indices = []
            for local_token_idx in anchor_indices_local:
                token_y = local_token_idx // W_lat
                token_x = local_token_idx % W_lat
                
                # Center of token in pixel coords
                pixel_x = int((token_x + 0.5) * (w_img / W_lat))
                pixel_y = int((token_y + 0.5) * (h_img / H_lat))
                
                # Clamp
                pixel_x_c = min(max(0, pixel_x), w_img - 1)
                pixel_y_c = min(max(0, pixel_y), h_img - 1)
                
                keep_token = False
                if spatial_mode == 'mask':
                    if current_mask[pixel_y_c, pixel_x_c]:
                        keep_token = True
                else: # box
                    for box in current_boxes:
                        x1, y1, x2, y2 = box
                        if x1 <= pixel_x <= x2 and y1 <= pixel_y <= y2:
                            keep_token = True
                            break
                
                if keep_token:
                    filtered_anchor_indices.append(local_token_idx)
            
            print(f"  Anchor Spatial Filter: Kept {len(filtered_anchor_indices)}/{len(anchor_indices_local)} tokens")
            anchor_indices_local = filtered_anchor_indices
        else:
             print("  Warning: Spatial constraint requested but no Box/Mask found for Anchor Frame. Keeping all Otsu tokens.")

    # Initialize Bank if we still have tokens
    if len(anchor_indices_local) > 0:
        # 1. Init Memory Bank (DINO Features)
        anchor_features = dino_features_flat[best_frame_idx, anchor_indices_local] # [N, Dim]
        memory_bank = F.normalize(anchor_features, p=2, dim=-1).to("cuda")
        
        # 2. Init Attention Threshold (Median of Anchor Tokens)
        if attn_scores_flat is not None:
            anchor_scores = attn_scores_flat[best_frame_idx, anchor_indices_local]
            attn_score_threshold = torch.median(anchor_scores).item()
            print(f"  ✓ Attention Score Threshold set to: {attn_score_threshold:.4f} (Median of Anchor Frame)")
    else:
        print("Warning: Anchor frame has no selected tokens after filtering. Fallback to Frame 0 logic (empty bank).")
        # Fallback empty

    # -------------------------------------------------------------------------
    # 4. Autoregressive Loop (Start from Frame 0)
    # -------------------------------------------------------------------------
    frames_dir = os.path.join(save_dir, f"clip_{clip_idx:02d}_{target_text}_autoregressive")
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"  Processing frames starting from 0 (Dual Gating: DINO Sim < {1-dino_threshold} AND Attn > {attn_score_threshold:.4f})...")

    # Sort indices for sequential access
    all_indices_sorted = sorted(list(all_indices))
    
    for t in range(num_frames):
        frame_start_idx = t * (H_lat * W_lat)
        frame_end_idx = (t + 1) * (H_lat * W_lat)
        
        # Get Otsu candidates for this frame
        frame_indices_global = [idx for idx in all_indices_sorted if frame_start_idx <= idx < frame_end_idx]
        frame_indices_local = [idx - frame_start_idx for idx in frame_indices_global]
        
        novel_indices_local = [] 
        
        if len(frame_indices_local) > 0:
            candidate_features = dino_features_flat[t, frame_indices_local].to("cuda") # [N_cand, Dim]
            candidate_features = F.normalize(candidate_features, p=2, dim=-1)
            
            if attn_scores_flat is not None:
                candidate_scores = attn_scores_flat[t, frame_indices_local].to("cuda")
            else:
                candidate_scores = None

            # Logic A: Empty Memory
            if memory_bank is None:
                # If memory is empty (e.g. anchor filtering removed everything), accept candidates subject to spatial/attn?
                # Or just accept everything as new anchor? Let's assume accept if spatial passes (checked below)
                # But here we initialize boolean mask as True, then filter down.
                is_novel_mask = torch.ones(len(frame_indices_local), dtype=torch.bool, device="cuda")
            
            # Logic B: Standard Check
            else:
                sim_matrix = torch.mm(candidate_features, memory_bank.t())
                max_sim_values, _ = sim_matrix.max(dim=1) # [N_cand]
                
                sim_threshold = 1.0 - dino_threshold
                cond_dino = max_sim_values < sim_threshold
                
                if candidate_scores is not None:
                    cond_attn = candidate_scores >= attn_score_threshold
                else:
                    cond_attn = torch.ones_like(cond_dino)
                
                # Spatial Constraint (Mask > Box)
                cond_spatial = torch.zeros(len(frame_indices_local), dtype=torch.bool, device="cuda")
                
                check_spatial = False
                spatial_mode = 'none' 
                
                if use_bound == 'mask' and t in binary_masks_per_frame:
                     check_spatial = True
                     current_mask = binary_masks_per_frame[t]
                     spatial_mode = 'mask'
                elif use_bound == 'bound' and t in bounding_boxes_per_frame:
                     check_spatial = True
                     current_boxes = bounding_boxes_per_frame[t]
                     spatial_mode = 'box'
                
                if check_spatial:
                    # Get image dimensions to map tokens to pixels
                    ref_img_pil_temp = video_frames[t]
                    if isinstance(ref_img_pil_temp, np.ndarray):
                        h_img, w_img = ref_img_pil_temp.shape[:2]
                    elif isinstance(ref_img_pil_temp, Image.Image):
                        w_img, h_img = ref_img_pil_temp.size
                    else:
                        h_img, w_img = ref_img_pil_temp.shape[1:3]
                    
                    for idx, local_token_idx in enumerate(frame_indices_local):
                        token_y = local_token_idx // W_lat
                        token_x = local_token_idx % W_lat
                        
                        pixel_x = int((token_x + 0.5) * (w_img / W_lat))
                        pixel_y = int((token_y + 0.5) * (h_img / H_lat))
                        
                        pixel_x_c = min(max(0, pixel_x), w_img - 1)
                        pixel_y_c = min(max(0, pixel_y), h_img - 1)
                        
                        if spatial_mode == 'mask':
                            if current_mask[pixel_y_c, pixel_x_c]:
                                cond_spatial[idx] = True
                        else:
                            for box in current_boxes:
                                x1, y1, x2, y2 = box
                                if x1 <= pixel_x <= x2 and y1 <= pixel_y <= y2:
                                    cond_spatial[idx] = True
                                    break
                else:
                    if use_bound != "None":
                         cond_spatial = torch.zeros(len(frame_indices_local), dtype=torch.bool, device="cuda")
                    else:
                         cond_spatial = torch.ones(len(frame_indices_local), dtype=torch.bool, device="cuda")
                
                # Condition 4: Anchor Frame Exception (Visualisation Only)
                # Note: We do NOT add anchor frame tokens to memory again here to avoid duplicates
                # But we do want to visualize the ones we kept.
                if t == best_frame_idx:
                    # For visualization, we want to show what is IN the memory bank from this frame.
                    # Which is exactly 'filtered_anchor_indices' calculated earlier.
                    # So we construct a mask that selects those specific local indices.
                    
                    # This is tricky because frame_indices_local comes from 'all_indices' (Otsu).
                    # filtered_anchor_indices is a subset of that.
                    is_novel_mask = torch.zeros(len(frame_indices_local), dtype=torch.bool, device="cuda")
                    
                    # Create a lookup for speed or just loop
                    # Since both are sorted subsets of all_indices, simple check works
                    if 'anchor_indices_local' in locals(): # Use the filtered list from Step 3
                        filtered_set = set(anchor_indices_local)
                        for i, val in enumerate(frame_indices_local):
                            if val in filtered_set:
                                is_novel_mask[i] = True
                    else:
                         is_novel_mask = cond_dino & cond_attn & cond_spatial # Fallback
                else:
                    # Combined Gating: DINO + Attention + Spatial
                    is_novel_mask = cond_dino & cond_attn & cond_spatial

            # Extract Indices
            frame_indices_tensor = torch.tensor(frame_indices_local, device="cuda")
            valid_indices = frame_indices_tensor[is_novel_mask].cpu().tolist()
            novel_indices_local.extend(valid_indices)
            
            # Update Memory Bank with NEW novel tokens
            # Only update if NOT anchor frame (Anchor is already in bank)
            if len(valid_indices) > 0 and t != best_frame_idx:
                new_features = candidate_features[is_novel_mask]
                if memory_bank is None:
                     memory_bank = new_features
                else:
                     memory_bank = torch.cat([memory_bank, new_features], dim=0)
                
                if memory_bank.shape[0] > 5000:
                    memory_bank = memory_bank[-5000:]

        # -------------------------------------------------------------------------
        # 5. Visualization (Draw on Current Frame t)
        # -------------------------------------------------------------------------
        # ... (Visualization code remains identical) ...
        ref_img_pil = video_frames[t]
        if isinstance(ref_img_pil, torch.Tensor): ref_img_pil = T.ToPILImage()(ref_img_pil)
        if isinstance(ref_img_pil, np.ndarray): ref_img_pil = Image.fromarray(ref_img_pil)
        ref_img_np = np.array(ref_img_pil)
        h_img, w_img = ref_img_np.shape[:2]
        
        mask = np.zeros((H_lat, W_lat), dtype=np.float32)
        if novel_indices_local:
            flat_mask = np.zeros(H_lat * W_lat, dtype=np.float32)
            flat_mask[novel_indices_local] = 1.0
            mask = flat_mask.reshape(H_lat, W_lat)
            
        mask_resized = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        
        # Overlay Logic
        overlay = ref_img_np.copy()
        color = np.array([255, 0, 255], dtype=np.uint8) # Magenta
        alpha = 0.55 
        
        bool_mask = mask_resized > 0
        if bool_mask.any():
             roi = overlay[bool_mask]
             blended = (roi * (1 - alpha) + color * alpha).astype(np.uint8)
             overlay[bool_mask] = blended
             
        if t == best_frame_idx:
            cv2.putText(overlay, "ANCHOR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if use_bound != "None":
            if t in binary_masks_per_frame:
                 sam_mask = binary_masks_per_frame[t].astype(np.uint8) * 255
                 if sam_mask.shape[:2] != (h_img, w_img):
                      sam_mask = cv2.resize(sam_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
                 contours, _ = cv2.findContours(sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                 cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                 cv2.putText(overlay, "SAM Mask Constraint", (10, h_img - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif t in bounding_boxes_per_frame:
                for box_idx, box in enumerate(bounding_boxes_per_frame[t]):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{target_text}_{box_idx}"
                    cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
             
        save_path = os.path.join(frames_dir, f"frame_{t:03d}_novel_tokens.png")
        Image.fromarray(overlay).save(save_path)
    
    print(f"  Finished. Memory Bank Final Size: {memory_bank.shape[0] if memory_bank is not None else 0}")
