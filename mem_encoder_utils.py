from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

LAYERWISE_MARKER = "__layerwise__"
LAYERWISE_LAYERS_KEY = "layers"


def _layer_key(layer_idx: Any) -> str:
    try:
        return str(int(layer_idx))
    except Exception:
        return str(layer_idx)


def is_layerwise_container(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value.get(LAYERWISE_MARKER, False))
        and isinstance(value.get(LAYERWISE_LAYERS_KEY, None), dict)
    )


def make_layerwise_container(layers: Dict[Any, Any]) -> Dict[str, Any]:
    return {
        LAYERWISE_MARKER: True,
        LAYERWISE_LAYERS_KEY: {_layer_key(k): v for k, v in dict(layers).items()},
    }


def iter_layerwise_items(value: Any) -> Iterable[Tuple[str, Any]]:
    if is_layerwise_container(value):
        for layer, payload in value.get(LAYERWISE_LAYERS_KEY, {}).items():
            yield _layer_key(layer), payload
        return
    if isinstance(value, dict):
        for layer, payload in value.items():
            if isinstance(payload, torch.Tensor) or isinstance(payload, (list, dict)):
                yield _layer_key(layer), payload


def select_layerwise_value(value: Any, layer_idx: Any, default: Any = None) -> Any:
    if is_layerwise_container(value):
        layers = value.get(LAYERWISE_LAYERS_KEY, {})
        return layers.get(_layer_key(layer_idx), default)
    return value if value is not None else default


def memory_encoder_enabled(mode: Any) -> bool:
    return str(mode or "off").strip().lower() in {
        "on",
        "true",
        "1",
        "extra",
        "extra_encoder",
        "slotmem_memory_encoder",
        "jigsaw_extra_encoder",
        "contrastive_encoder",
    }


def memory_writer_effective_mode(train_stage: Any, mode: Any = "auto") -> str:
    stage = str(train_stage or "stage1").strip().lower()
    mode_l = str(mode or "auto").strip().lower()
    if stage != "stage2":
        return "off"
    if mode_l in {"off", "none", "false", "0"}:
        return "off"
    if mode_l == "auto":
        return "residual"
    if mode_l in {"on", "true", "1", "residual"}:
        return "residual"
    return mode_l


def memory_writer_enabled(train_stage: Any, mode: Any = "auto") -> bool:
    return memory_writer_effective_mode(train_stage, mode) == "residual"


def parse_layer_list(text: Any, default: str = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15") -> List[int]:
    raw = str(text if text is not None else default).strip()
    if not raw:
        raw = default
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            step = 1 if hi >= lo else -1
            out.extend(range(lo, hi + step, step))
        else:
            out.append(int(part))
    seen = set()
    dedup = []
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(int(x))
    return dedup


def parse_layer_groups(text: Any, default: str = "0-4,5-10,11-15") -> List[List[int]]:
    groups = []
    for group_text in str(text if text is not None else default).split(","):
        layers = parse_layer_list(group_text, default="")
        if layers:
            groups.append(layers)
    return groups or [list(range(0, 5)), list(range(5, 11)), list(range(11, 16))]


def build_layer_to_group(groups: Sequence[Sequence[int]]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for gid, layers in enumerate(groups):
        for layer in layers:
            out[int(layer)] = int(gid)
    return out


def _role_id_from_meta(item: Any, fallback: str = "0") -> str:
    if isinstance(item, dict):
        rid = str(item.get("char_id", "")).strip()
        if rid:
            return rid
    return str(fallback)


def _make_slot_meta(role_meta: List[Any], role_id: str, slot_count: int, group_idx: int, layer_idx: int) -> List[Dict[str, Any]]:
    template = next((m for m in role_meta if isinstance(m, dict)), {})
    out = []
    for i in range(int(slot_count)):
        item = {
            "char_id": str(role_id),
            "is_jigsaw_extra_encoder_slot": True,
            "jigsaw_extra_encoder_group": int(group_idx),
            "jigsaw_extra_encoder_layer": int(layer_idx),
            "jigsaw_extra_encoder_slot": int(i),
            "jigsaw_slot_id": int(i),
            "jigsaw_slot_count": int(slot_count),
            "jigsaw_slot_coord": float(i) / float(max(int(slot_count) - 1, 1)),
            "inside_box": bool(template.get("inside_box", False)) if isinstance(template, dict) else False,
        }
        # Keep role-relative query matching available, but avoid carrying a
        # specific source composition into memory-side RoPE by default.
        for key in ("u", "v", "rel_l", "rel_r", "rel_t", "rel_b", "tau_local"):
            item[key] = 0.0
        for key in ("latent_t", "latent_h", "latent_w", "h_patch", "w_patch"):
            item[key] = 0
        out.append(item)
    return out


class GroupedMemoryEncoder(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        slots: int = 32,
        encoder_dim: int = 512,
        hidden_dim: int = 1024,
        use_t_embed: bool = False,
        use_slot_index_embed: bool = False,
        time_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = int(dim)
        self.slots = int(slots)
        self.encoder_dim = int(encoder_dim)
        self.use_t_embed = bool(use_t_embed)
        self.use_slot_index_embed = bool(use_slot_index_embed)
        self.time_embed_dim = int(time_embed_dim) if time_embed_dim is not None else self.dim
        self.input_norm = torch.nn.LayerNorm(self.dim)
        self.in_proj = torch.nn.Linear(self.dim, self.encoder_dim)
        self.query = torch.nn.Parameter(torch.randn(self.slots, self.encoder_dim) * 0.02)
        if self.use_t_embed:
            self.time_norm = torch.nn.LayerNorm(self.time_embed_dim)
            self.time_mod = torch.nn.Linear(self.time_embed_dim, self.encoder_dim * 2)
            torch.nn.init.zeros_(self.time_mod.weight)
            torch.nn.init.zeros_(self.time_mod.bias)
        else:
            self.time_norm = None
            self.time_mod = None
        if self.use_slot_index_embed:
            self.slot_index_embed = torch.nn.Parameter(torch.randn(self.slots, self.dim) * 0.02)
            self.slot_index_embed_scale = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        else:
            self.slot_index_embed = None
            self.slot_index_embed_scale = None
        self.ff = torch.nn.Sequential(
            torch.nn.LayerNorm(self.encoder_dim),
            torch.nn.Linear(self.encoder_dim, int(hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Linear(int(hidden_dim), self.dim),
        )

    def forward(self, tokens: torch.Tensor, t_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or int(tokens.shape[0]) <= 0:
            raise ValueError("GroupedMemoryEncoder expects [N,D] non-empty tokens")
        orig_dtype = tokens.dtype
        h = self.in_proj(self.input_norm(tokens).to(dtype=self.in_proj.weight.dtype))
        if self.use_t_embed:
            if not isinstance(t_embed, torch.Tensor):
                raise ValueError("GroupedMemoryEncoder use_t_embed=True requires t_embed")
            t = t_embed
            if t.ndim > 2:
                t = t.reshape(-1, int(t.shape[-1]))
            if t.ndim == 2:
                t = t.mean(dim=0)
            if t.ndim != 1 or int(t.shape[0]) != int(self.time_embed_dim):
                raise ValueError(
                    f"GroupedMemoryEncoder expected t_embed last dim {self.time_embed_dim}, "
                    f"got shape {tuple(t_embed.shape)}"
                )
            t = t.to(device=h.device, dtype=self.time_mod.weight.dtype)
            gamma_beta = self.time_mod(self.time_norm(t)).to(dtype=h.dtype)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            h = h * (1.0 + gamma.unsqueeze(0)) + beta.unsqueeze(0)
        q = self.query.to(device=h.device, dtype=h.dtype)
        scores = torch.matmul(q, h.transpose(0, 1)) / max(float(self.encoder_dim) ** 0.5, 1.0)
        attn = torch.softmax(scores.float(), dim=-1).to(dtype=h.dtype)
        pooled = torch.matmul(attn, h)
        out = self.ff(pooled)
        if self.use_slot_index_embed and isinstance(self.slot_index_embed, torch.nn.Parameter):
            slot_pos = self.slot_index_embed.to(device=out.device, dtype=out.dtype)
            slot_scale = self.slot_index_embed_scale.to(device=out.device, dtype=out.dtype)
            out = out + slot_pos * slot_scale
        return out.to(dtype=orig_dtype)


class MemoryEncoderBank(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        layer_groups: Sequence[Sequence[int]],
        slots: int = 32,
        encoder_dim: int = 512,
        hidden_dim: int = 1024,
        use_t_embed: bool = False,
        use_slot_index_embed: bool = False,
        time_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = int(dim)
        self.layer_groups = [list(map(int, g)) for g in layer_groups]
        self.layer_to_group = build_layer_to_group(self.layer_groups)
        self.slots = int(slots)
        self.use_t_embed = bool(use_t_embed)
        self.use_slot_index_embed = bool(use_slot_index_embed)
        self.time_embed_dim = int(time_embed_dim) if time_embed_dim is not None else self.dim
        self.group_encoders = torch.nn.ModuleList([
            GroupedMemoryEncoder(
                dim=dim,
                slots=slots,
                encoder_dim=encoder_dim,
                hidden_dim=hidden_dim,
                use_t_embed=self.use_t_embed,
                use_slot_index_embed=self.use_slot_index_embed,
                time_embed_dim=self.time_embed_dim,
            )
            for _ in self.layer_groups
        ])

    def group_for_layer(self, layer_idx: Any) -> int:
        try:
            layer_int = int(layer_idx)
        except Exception:
            layer_int = 0
        return int(self.layer_to_group.get(layer_int, 0))

    def encode_role_tokens(self, tokens: torch.Tensor, layer_idx: Any, t_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        gid = self.group_for_layer(layer_idx)
        return self.group_encoders[gid](tokens, t_embed=t_embed)

    def classify_slots(self, slots: torch.Tensor, layer_idx: Any) -> torch.Tensor:
        del layer_idx
        return slots


class MemoryWriter(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 1024,
        init_scale: float = 0.1,
        precision_tau: float = 0.3,
        precision_scale: float = 10.0,
        max_delta_ratio: float = 0.0,
        max_delta_norm: float = 0.0,
        detach_c_short: bool = True,
        context_mode: str = "mean",
        attention_scale: float = 10.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.init_scale = float(init_scale)
        self.precision_tau = float(precision_tau)
        self.precision_scale = float(precision_scale)
        self.max_delta_ratio = float(max_delta_ratio)
        self.max_delta_norm = float(max_delta_norm)
        self.detach_c_short = bool(detach_c_short)
        context_mode_l = str(context_mode or "mean").strip().lower()
        if context_mode_l in {"avg", "average"}:
            context_mode_l = "mean"
        if context_mode_l in {"attention", "attn", "slot_attn", "slot_attention", "cross_attn", "cross_attention"}:
            context_mode_l = "slot_attention"
        if context_mode_l not in {"mean", "slot_attention"}:
            context_mode_l = "mean"
        self.context_mode = context_mode_l
        self.attention_scale = float(attention_scale)
        hidden = max(1, int(hidden_dim))
        self.m_norm = torch.nn.LayerNorm(self.dim)
        self.c_norm = torch.nn.LayerNorm(self.dim)
        self.gate_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.dim * 2, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 1),
        )
        self.delta_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.dim * 2, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, self.dim),
        )
        torch.nn.init.zeros_(self.delta_mlp[-1].weight)
        torch.nn.init.zeros_(self.delta_mlp[-1].bias)

    @staticmethod
    def _stats(enabled: float = 0.0, input_slots: int = 0, updated_slots: int = 0, mean_gate: float = 0.0, mean_cos: float = 0.0, clipped_ratio: float = 0.0) -> Dict[str, float]:
        return {
            "enabled": float(enabled),
            "input_slots": int(input_slots),
            "updated_slots": int(updated_slots),
            "mean_gate": float(mean_gate),
            "mean_cos": float(mean_cos),
            "clipped_ratio": float(clipped_ratio),
        }

    @staticmethod
    def _payload_for_role(query_feature_payload: Any, role_id: str) -> Optional[Dict[str, Any]]:
        if not isinstance(query_feature_payload, dict):
            return None
        if role_id in query_feature_payload and isinstance(query_feature_payload.get(role_id), dict):
            return query_feature_payload.get(role_id)
        role_s = str(role_id)
        for key, payload in query_feature_payload.items():
            if str(key) == role_s and isinstance(payload, dict):
                return payload
        return None

    @staticmethod
    def _flat_idx_from_payload(payload: Any, max_tokens: int, device: torch.device) -> Optional[torch.Tensor]:
        if not isinstance(payload, dict):
            return None
        flat_idx = payload.get("flat_idx", None)
        if not isinstance(flat_idx, torch.Tensor) or flat_idx.numel() <= 0:
            return None
        idx = flat_idx.detach().reshape(-1).to(device=device, dtype=torch.long)
        valid = (idx >= 0) & (idx < int(max_tokens))
        if not bool(valid.any().item()):
            return None
        return idx[valid]

    def _context_for_role(self, m_norm: torch.Tensor, x_role: torch.Tensor) -> torch.Tensor:
        if (
            self.context_mode == "slot_attention"
            and isinstance(x_role, torch.Tensor)
            and x_role.ndim == 2
            and int(x_role.shape[0]) > 1
        ):
            x_norm = self.c_norm(x_role.to(dtype=m_norm.dtype))
            q = F.normalize(m_norm.float(), dim=-1, eps=1e-6)
            k = F.normalize(x_norm.float(), dim=-1, eps=1e-6)
            logits = torch.matmul(q, k.transpose(0, 1)) * float(self.attention_scale)
            attn = torch.softmax(logits, dim=-1)
            return torch.matmul(attn, x_role.float()).to(dtype=x_role.dtype)
        return x_role.mean(dim=0, keepdim=True).expand(int(m_norm.shape[0]), -1)

    def forward(
        self,
        m_long_slots: torch.Tensor,
        token_meta: Optional[List[Any]],
        query_feature_payload: Any,
        current_x_output: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not isinstance(m_long_slots, torch.Tensor) or m_long_slots.ndim != 2 or int(m_long_slots.shape[0]) <= 0:
            return m_long_slots, self._stats(enabled=0.0)
        input_slots = int(m_long_slots.shape[0])
        if int(m_long_slots.shape[-1]) != self.dim:
            return m_long_slots, self._stats(enabled=0.0, input_slots=input_slots)
        if not isinstance(current_x_output, torch.Tensor) or current_x_output.ndim != 3 or int(current_x_output.shape[0]) <= 0:
            return m_long_slots, self._stats(enabled=0.0, input_slots=input_slots)
        if int(current_x_output.shape[-1]) != self.dim:
            return m_long_slots, self._stats(enabled=0.0, input_slots=input_slots)

        meta = list(token_meta or [])
        if len(meta) < input_slots:
            meta.extend({"char_id": "0"} for _ in range(input_slots - len(meta)))
        role_to_idx: Dict[str, List[int]] = defaultdict(list)
        for i in range(input_slots):
            role_to_idx[_role_id_from_meta(meta[i] if i < len(meta) else None)].append(i)

        x_tokens = current_x_output.detach() if self.detach_c_short else current_x_output
        x_tokens = x_tokens[0].to(device=m_long_slots.device, dtype=m_long_slots.dtype)
        updated = m_long_slots
        updated_parts = []
        gate_values = []
        cos_values = []
        clipped_values = []
        any_update = False

        for role_id in sorted(role_to_idx.keys()):
            slot_indices = role_to_idx[role_id]
            payload = self._payload_for_role(query_feature_payload, role_id)
            flat_idx = self._flat_idx_from_payload(payload, int(x_tokens.shape[0]), m_long_slots.device)
            if flat_idx is None or len(slot_indices) <= 0:
                continue
            slot_idx = torch.tensor(slot_indices, device=m_long_slots.device, dtype=torch.long)
            m_role = m_long_slots.index_select(0, slot_idx)
            x_role = x_tokens.index_select(0, flat_idx)
            mlp_dtype = self.gate_mlp[0].weight.dtype
            m_norm = self.m_norm(m_role.to(dtype=mlp_dtype))
            c_short = self._context_for_role(m_norm, x_role.to(dtype=mlp_dtype))
            c_norm = self.c_norm(c_short.to(dtype=mlp_dtype))
            m_unit = F.normalize(m_norm.float(), dim=-1, eps=1e-6)
            c_unit = F.normalize(c_norm.float(), dim=-1, eps=1e-6)
            cos = (m_unit * c_unit).sum(dim=-1, keepdim=True)
            precision_gate = torch.sigmoid((cos - float(self.precision_tau)) * float(self.precision_scale))
            fused = torch.cat([m_norm, c_norm], dim=-1)
            learned_gate = torch.sigmoid(self.gate_mlp(fused)).float()
            gate = precision_gate * learned_gate
            delta = self.delta_mlp(fused)
            residual = delta.float() * gate * float(self.init_scale)

            clipped = torch.zeros((int(residual.shape[0]), 1), device=residual.device, dtype=torch.float32)
            max_ratio = float(self.max_delta_ratio)
            max_norm = float(self.max_delta_norm)
            if max_ratio > 0.0 or max_norm > 0.0:
                res_norm = residual.detach().float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
                if max_ratio > 0.0:
                    cap = m_role.detach().float().norm(dim=-1, keepdim=True).clamp(min=1e-6) * max_ratio
                    if max_norm > 0.0:
                        cap = torch.minimum(cap, torch.full_like(cap, max_norm))
                else:
                    cap = torch.full_like(res_norm, max_norm)
                factor = torch.minimum(torch.ones_like(res_norm), cap / res_norm)
                clipped = (factor.detach().float() < 0.999).float()
                residual = residual * factor

            updated_parts.append((slot_idx, m_role + residual.to(dtype=m_role.dtype)))
            gate_values.append(gate.detach().float())
            cos_values.append(cos.detach().float())
            clipped_values.append(clipped)
            any_update = True

        if any_update:
            updated = m_long_slots.clone()
            for slot_idx, role_updated in updated_parts:
                updated.index_copy_(0, slot_idx, role_updated)
            gate_cat = torch.cat(gate_values, dim=0) if gate_values else m_long_slots.new_zeros((0, 1), dtype=torch.float32)
            cos_cat = torch.cat(cos_values, dim=0) if cos_values else m_long_slots.new_zeros((0, 1), dtype=torch.float32)
            clipped_cat = torch.cat(clipped_values, dim=0) if clipped_values else m_long_slots.new_zeros((0, 1), dtype=torch.float32)
            stats = self._stats(
                enabled=1.0,
                input_slots=input_slots,
                updated_slots=int(gate_cat.shape[0]),
                mean_gate=float(gate_cat.mean().item()) if gate_cat.numel() > 0 else 0.0,
                mean_cos=float(cos_cat.mean().item()) if cos_cat.numel() > 0 else 0.0,
                clipped_ratio=float(clipped_cat.mean().item()) if clipped_cat.numel() > 0 else 0.0,
            )
            return updated, stats
        return m_long_slots, self._stats(enabled=1.0, input_slots=input_slots)


def encode_role_tokens_to_slots(
    encoder_bank: MemoryEncoderBank,
    tokens: torch.Tensor,
    token_meta: Optional[List[Any]],
    layer_idx: Any,
    t_embed: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], List[int], Dict[str, Any]]:
    if not isinstance(tokens, torch.Tensor) or tokens.ndim < 2 or int(tokens.shape[0]) <= 0:
        return tokens, [], [], {"enabled": 0.0, "input_tokens": 0, "output_slots": 0}
    meta = list(token_meta or [])
    if len(meta) < int(tokens.shape[0]):
        meta.extend({"char_id": "0"} for _ in range(int(tokens.shape[0]) - len(meta)))
    role_to_idx: Dict[str, List[int]] = defaultdict(list)
    use_n = min(int(tokens.shape[0]), len(meta))
    for i in range(use_n):
        role_to_idx[_role_id_from_meta(meta[i])].append(i)
    if not role_to_idx:
        role_to_idx["0"] = list(range(use_n))

    out_tokens = []
    out_meta: List[Dict[str, Any]] = []
    lengths = []
    gid = encoder_bank.group_for_layer(layer_idx)
    for role_id in sorted(role_to_idx.keys()):
        idx = torch.tensor(role_to_idx[role_id], device=tokens.device, dtype=torch.long)
        role_tokens = tokens.index_select(0, idx)
        slots = encoder_bank.encode_role_tokens(role_tokens, layer_idx, t_embed=t_embed)
        out_tokens.append(slots)
        lengths.append(int(slots.shape[0]))
        role_meta = [meta[i] for i in role_to_idx[role_id]]
        out_meta.extend(_make_slot_meta(role_meta, role_id, int(slots.shape[0]), gid, int(layer_idx)))
    encoded = torch.cat(out_tokens, dim=0) if out_tokens else tokens[:0]
    return encoded, out_meta, lengths, {
        "enabled": 1.0,
        "layer": int(layer_idx),
        "group": int(gid),
        "roles": int(len(role_to_idx)),
        "input_tokens": int(tokens.shape[0]),
        "output_slots": int(encoded.shape[0]),
        "slots_per_role": int(encoder_bank.slots),
    }


def encode_layerwise_role_memory_bank(
    encoder_bank: MemoryEncoderBank,
    memory_bank_tokens: Any,
    memory_bank_meta: Any,
    t_embed: Optional[torch.Tensor] = None,
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    if not is_layerwise_container(memory_bank_tokens):
        out, meta, lengths, stats = encode_role_tokens_to_slots(
            encoder_bank, memory_bank_tokens, memory_bank_meta, layer_idx=0, t_embed=t_embed
        )
        return out, meta, lengths, stats
    out_layers = {}
    meta_layers = {}
    length_layers = {}
    stats_layers = {}
    for layer, bank_map in iter_layerwise_items(memory_bank_tokens):
        meta_map = select_layerwise_value(memory_bank_meta, layer, default={})
        if not isinstance(bank_map, dict):
            continue
        out_bank = {}
        meta_bank = {}
        len_bank = {}
        stat_bank = {}
        for bank_key, bank_tokens in bank_map.items():
            bank_meta = meta_map.get(str(bank_key), []) if isinstance(meta_map, dict) else []
            encoded, encoded_meta, lengths, stats = encode_role_tokens_to_slots(
                encoder_bank, bank_tokens, bank_meta, layer_idx=layer, t_embed=t_embed
            )
            if isinstance(encoded, torch.Tensor) and encoded.ndim >= 2 and int(encoded.shape[0]) > 0:
                out_bank[str(bank_key)] = encoded
                meta_bank[str(bank_key)] = encoded_meta
                len_bank[str(bank_key)] = lengths
                stat_bank[str(bank_key)] = stats
        if out_bank:
            out_layers[layer] = out_bank
            meta_layers[layer] = meta_bank
            length_layers[layer] = len_bank
            stats_layers[layer] = stat_bank
    return make_layerwise_container(out_layers), make_layerwise_container(meta_layers), make_layerwise_container(length_layers), {"layers": stats_layers}


def memory_encoder_contrastive_loss(
    encoder_bank: MemoryEncoderBank,
    encoded_slots: torch.Tensor,
    encoded_meta: List[Any],
    layer_idx: Any,
    x_output: Optional[torch.Tensor] = None,
    query_feature_payload: Any = None,
    bg_tokens: int = 64,
    t_embed: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not isinstance(encoded_slots, torch.Tensor) or encoded_slots.ndim != 2 or int(encoded_slots.shape[0]) <= 1:
        zero = encoded_slots.new_zeros(()) if isinstance(encoded_slots, torch.Tensor) else torch.tensor(0.0)
        return zero, {"enabled": 0.0}
    device = encoded_slots.device
    role_ids = [_role_id_from_meta(m) for m in list(encoded_meta or [])[: int(encoded_slots.shape[0])]]
    uniq = sorted({r for r in role_ids if r})
    labels = torch.tensor([uniq.index(r) for r in role_ids], device=device, dtype=torch.long)
    feats = encoder_bank.classify_slots(encoded_slots, layer_idx).float()

    bg_count = 0
    if isinstance(x_output, torch.Tensor) and x_output.ndim == 3 and int(bg_tokens) > 0:
        total = int(x_output.shape[1])
        used = set()
        if isinstance(query_feature_payload, dict):
            for payload in query_feature_payload.values():
                if isinstance(payload, dict) and isinstance(payload.get("flat_idx", None), torch.Tensor):
                    used.update(int(i) for i in payload["flat_idx"].detach().cpu().reshape(-1).tolist() if 0 <= int(i) < total)
        candidates = [i for i in range(total) if i not in used]
        if candidates:
            take = min(int(bg_tokens), len(candidates))
            perm = torch.randperm(len(candidates), device=device)[:take].detach().cpu().tolist()
            idx = torch.tensor([candidates[i] for i in perm], device=device, dtype=torch.long)
            bg_raw = x_output.detach()[0].index_select(0, idx)
            bg_slots = encoder_bank.encode_role_tokens(bg_raw, layer_idx, t_embed=t_embed)
            bg_feat = encoder_bank.classify_slots(bg_slots, layer_idx).float()
            feats = torch.cat([feats, bg_feat], dim=0)
            bg_label = len(uniq)
            labels = torch.cat([labels, torch.full((int(bg_feat.shape[0]),), bg_label, device=device, dtype=torch.long)], dim=0)
            bg_count = int(bg_feat.shape[0])

    centroids = []
    for cid in range(int(labels.max().item()) + 1):
        mask = labels == cid
        centroids.append(feats[mask].mean(dim=0))
    class_w = F.normalize(torch.stack(centroids, dim=0), dim=-1, eps=1e-6)
    logits = torch.matmul(F.normalize(feats, dim=-1, eps=1e-6), class_w.t()) / 0.07
    loss = F.cross_entropy(logits, labels)
    return loss, {
        "enabled": 1.0,
        "classes": float(len(centroids)),
        "slots": float(int(encoded_slots.shape[0])),
        "bg_slots": float(bg_count),
        "group": float(encoder_bank.group_for_layer(layer_idx)),
    }


def extract_prefixed_state_dict(full_sd: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    out = {}
    p = str(prefix)
    for key, value in dict(full_sd or {}).items():
        key_s = str(key)
        if key_s.startswith(p):
            out[key_s[len(p):]] = value
    return out
