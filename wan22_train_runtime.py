import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diffsynth.core.loader import ModelConfig
from diffsynth.models.wan_video_dit import modulate, sinusoidal_embedding_1d
from diffsynth.pipelines.wan_video_svi_pro import WanVideoSviProPipeline


def _sorted_glob(pattern, required=True):
    paths = sorted(glob.glob(pattern))
    if required and len(paths) <= 0:
        raise FileNotFoundError(f"Missing required checkpoint files: {pattern}")
    return paths


def _get_module_dtype_device(module, default_device="cpu", default_dtype=torch.bfloat16):
    if module is None:
        return default_dtype, torch.device(default_device)
    if hasattr(module, "dtype") and hasattr(module, "device"):
        dtype = getattr(module, "dtype", default_dtype)
        device = getattr(module, "device", default_device)
        return dtype, torch.device(device)
    inner_model = getattr(module, "model", None)
    if inner_model is not None:
        try:
            param = next(inner_model.parameters())
            return param.dtype, param.device
        except Exception:
            pass
    try:
        param = next(module.parameters())
        return param.dtype, param.device
    except Exception:
        return default_dtype, torch.device(default_device)


def _module_is_on(module, device):
    """True if the module's real parameters already live on `device`.

    Reads a parameter rather than a cached ``.device`` attribute: inference moves the
    active expert to cpu before VAE decode without updating any such attribute.
    """
    if module is None:
        return True
    try:
        current = next(module.parameters()).device
    except StopIteration:
        return True
    target = torch.device(device)
    if current.type != target.type:
        return False
    return target.index is None or current.index == target.index


def _is_dtype_mismatch_error(exc):
    msg = str(exc)
    return (
        ("weight type" in msg and "Input type" in msg and "should be the same" in msg)
        or ("expected scalar type" in msg and "but found" in msg)
    )


def _pick_alternate_fp_dtype(dtype):
    if dtype == torch.float32:
        return torch.bfloat16
    return torch.float32


def _normalize_dual_expert_load_mode(mode):
    mode = str(mode or "standard").strip().lower()
    if mode in ("", "standard", "eager", "default", "off", "false", "0", "none"):
        return "standard"
    if mode in ("vram", "vram_management", "managed", "lazy", "low_vram", "low-vram"):
        return "vram_management"
    if mode in ("active", "sequential", "sequential_onload", "active_expert"):
        return "active"
    raise ValueError(
        "dual_expert_load_mode must be one of: standard, vram_management, active "
        f"(got {mode!r})"
    )


def _resolve_torch_dtype(dtype_value, default=torch.bfloat16):
    if isinstance(dtype_value, torch.dtype):
        return dtype_value
    text = str(dtype_value or "").strip().lower()
    if text in ("", "auto", "default"):
        return default
    if text in ("bf16", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    if text in ("fp16", "float16", "half", "torch.float16"):
        return torch.float16
    if text in ("fp32", "float32", "float", "torch.float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_value!r}")


class WanTrainingSchedulerAdapter:
    def __init__(self, scheduler):
        self._scheduler = scheduler

    def __getattr__(self, name):
        return getattr(self._scheduler, name)

    def set_timesteps(self, num_inference_steps, shift=1.0):
        self._scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            training=True,
            shift=shift,
        )
        return self._scheduler.timesteps

    def get_timesteps(self, num_inference_steps, denoising_strength=1.0, shift=5.0):
        self._scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            shift=shift,
        )
        return self._scheduler.timesteps

    def training_target(self, latents, noise, timestep):
        return self._scheduler.training_target(latents, noise, timestep)

    def training_weight(self, timestep):
        return self._scheduler.training_weight(timestep)


class WanPrompterAdapter:
    def __init__(self, pipe):
        self.pipe = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder

    def encode_prompt(self, prompt, positive=True, device="cuda"):
        del positive
        target_device = torch.device(device)
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(target_device)
        mask = mask.to(target_device)
        text_encoder_vram_managed = bool(getattr(self.text_encoder, "vram_management_enabled", False))
        use_vram_management = bool(text_encoder_vram_managed and hasattr(self.pipe, "load_models_to_device"))
        if use_vram_management:
            self.pipe.load_models_to_device(["text_encoder"])
        else:
            self.text_encoder.to(target_device)
        try:
            prompt_emb = self.text_encoder(ids, mask)
            seq_lens = mask.gt(0).sum(dim=1).long()
            for i, v in enumerate(seq_lens):
                prompt_emb[i, v:] = 0
            return prompt_emb
        finally:
            if use_vram_management:
                self.pipe.load_models_to_device([])


class _DiffSynthSelfAttentionAdapter(nn.Module):
    def __init__(self, base_attn, freq_builder):
        super().__init__()
        self.base_attn = base_attn
        self._freq_builder = freq_builder

    def forward(self, x, seq_lens=None, grid_sizes=None, freqs=None):
        del seq_lens
        expanded_freqs = freqs
        if expanded_freqs is None or (
            torch.is_tensor(expanded_freqs)
            and expanded_freqs.ndim >= 2
            and int(expanded_freqs.shape[0]) != int(x.shape[1])
        ):
            expanded_freqs = self._freq_builder(grid_sizes, x.device)
        elif torch.is_tensor(expanded_freqs) and expanded_freqs.device != x.device:
            expanded_freqs = expanded_freqs.to(device=x.device)
        return self.base_attn(x, expanded_freqs)


class _DiffSynthCrossAttentionAdapter(nn.Module):
    def __init__(self, base_attn):
        super().__init__()
        self.base_attn = base_attn
        self.save_attn_weights = False
        self.target_token_idx = None
        self.attn_weights = None
        self.attn_capture_q_chunk_size = 256

    def _capture_selected_text_attn(self, x, context):
        target_idx = self.target_token_idx
        if isinstance(target_idx, int):
            target_idx = [int(target_idx)]
        elif isinstance(target_idx, torch.Tensor):
            target_idx = [int(t) for t in target_idx.detach().reshape(-1).tolist()]
        elif isinstance(target_idx, (list, tuple)):
            target_idx = [int(t) for t in target_idx]
        else:
            target_idx = []
        if len(target_idx) <= 0:
            self.attn_weights = None
            return

        base = self.base_attn
        if bool(getattr(base, "has_image_input", False)):
            ctx = context[:, 257:]
        else:
            ctx = context
        lk = int(ctx.shape[1])
        target_idx = [idx for idx in target_idx if 0 <= idx < lk]
        if len(target_idx) <= 0:
            self.attn_weights = None
            return

        with torch.no_grad():
            b = int(x.shape[0])
            n = int(base.num_heads)
            d = int(base.head_dim)
            selected_idx = torch.tensor(target_idx, device=x.device, dtype=torch.long)
            q = base.norm_q(base.q(x)).view(b, -1, n, d).permute(0, 2, 1, 3).float()
            k = base.norm_k(base.k(ctx)).view(b, -1, n, d).permute(0, 2, 1, 3).float()
            scale = float(d) ** -0.5
            q_chunk_size = max(1, int(getattr(self, "attn_capture_q_chunk_size", 256)))
            chunks = []
            for q_start in range(0, int(q.shape[2]), q_chunk_size):
                q_end = min(q_start + q_chunk_size, int(q.shape[2]))
                logits = torch.einsum("bhqd,bhkd->bhqk", q[:, :, q_start:q_end, :], k) * scale
                attn = torch.softmax(logits, dim=-1).index_select(-1, selected_idx)
                chunks.append(attn.detach().to(device="cpu", dtype=x.dtype))
                del logits, attn
            self.attn_weights = torch.cat(chunks, dim=2) if chunks else None

    def forward(self, x, context, context_lens=None):
        del context_lens
        out = self.base_attn(x, context)
        if bool(getattr(self, "save_attn_weights", False)):
            self._capture_selected_text_attn(x, context)
        return out


class _DiffSynthBlockAdapter(nn.Module):
    def __init__(self, base_block, freq_builder):
        super().__init__()
        self.base_block = base_block
        self.self_attn = _DiffSynthSelfAttentionAdapter(base_block.self_attn, freq_builder)
        self.cross_attn = _DiffSynthCrossAttentionAdapter(base_block.cross_attn)
        self.norm1 = base_block.norm1
        self.norm2 = base_block.norm2
        self.norm3 = base_block.norm3
        self.ffn = base_block.ffn
        self.modulation = base_block.modulation

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        del seq_lens, context_lens
        expanded_freqs = freqs
        if expanded_freqs is None or (
            torch.is_tensor(expanded_freqs)
            and expanded_freqs.ndim >= 2
            and int(expanded_freqs.shape[0]) != int(x.shape[1])
        ):
            expanded_freqs = self.self_attn._freq_builder(grid_sizes, x.device)
        elif torch.is_tensor(expanded_freqs) and expanded_freqs.device != x.device:
            expanded_freqs = expanded_freqs.to(device=x.device)

        has_seq = len(e.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=e.dtype, device=e.device) + e
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )

        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.self_attn(input_x, grid_sizes=grid_sizes, freqs=expanded_freqs)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x


class WanDiffSynthVideoAdapter(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.dim = base_model.dim
        self.freq_dim = base_model.freq_dim
        self.in_dim = int(getattr(base_model, "in_dim", 0))
        self.out_dim = int(getattr(base_model, "out_dim", 0))
        self.patch_size = tuple(base_model.patch_size)
        self.patch_embedding = base_model.patch_embedding
        self.text_embedding = base_model.text_embedding
        self.time_embedding = base_model.time_embedding
        self.time_projection = base_model.time_projection
        self.head = base_model.head
        # DiffSynth's Wan2.2-I2V config can expose `has_image_input=False`
        # while still requiring concatenated VAE condition channels
        # (`in_dim=36`, `out_dim=16`). Trust the channel contract first.
        self.has_image_input = bool(
            getattr(base_model, "has_image_input", False)
            or (self.in_dim > self.out_dim > 0)
            or bool(getattr(base_model, "require_vae_embedding", False))
        )
        self.require_vae_embedding = bool(getattr(base_model, "require_vae_embedding", True))
        self.require_clip_embedding = bool(getattr(base_model, "require_clip_embedding", True))
        self.has_image_pos_emb = bool(getattr(base_model, "has_image_pos_emb", False))
        self.img_emb = getattr(base_model, "img_emb", None)
        self._freqs_tuple = base_model.freqs
        # Keep a tensor-like attr for existing training code paths that expect
        # `dit_model.freqs.device` to exist.
        self.freqs = base_model.freqs[0]
        self.blocks = nn.ModuleList(
            [_DiffSynthBlockAdapter(block, self._expand_freqs) for block in base_model.blocks]
        )

    def to(self, *args, **kwargs):
        if bool(getattr(self, "_active_offload_defer_cuda_to_runtime", False)):
            device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
            del dtype, non_blocking, convert_to_format
            if device is not None and torch.device(device).type == "cuda":
                # Legacy inference calls pipe.dit.to(cuda) before it knows the
                # timestep domain. Defer the actual onload to the pipeline
                # adapter so it can offload the inactive expert first.
                return self
        if bool(getattr(self.base_model, "vram_management_enabled", False)):
            device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
            del dtype, non_blocking, convert_to_format
            if device is not None and torch.device(device).type == "cpu":
                for module in self.modules():
                    if module is not self and hasattr(module, "offload"):
                        module.offload()
            # Managed modules move weights during their own forward path. Avoid
            # recursively moving a whole 14B expert because legacy inference code
            # still calls pipe.dit.to(cuda) before denoising.
            return self
        return super().to(*args, **kwargs)

    def force_to(self, *args, **kwargs):
        return super().to(*args, **kwargs)

    def _expand_freqs(self, grid_sizes, device):
        if isinstance(grid_sizes, torch.Tensor):
            if grid_sizes.ndim == 2:
                f, h, w = [int(v) for v in grid_sizes[0].tolist()]
            else:
                f, h, w = [int(v) for v in grid_sizes.tolist()]
        else:
            f, h, w = [int(v) for v in grid_sizes]
        return torch.cat(
            [
                self._freqs_tuple[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self._freqs_tuple[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self._freqs_tuple[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1).to(device=device)

    def patchify(self, x):
        x = self.base_model.patchify(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size

    def unpatchify(self, x, grid_size):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x,
        timestep,
        context,
        clip_feature=None,
        y=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        **kwargs,
    ):
        del kwargs
        model_dtype = self.patch_embedding.weight.dtype
        x = x.to(dtype=model_dtype)
        context = context.to(device=x.device, dtype=model_dtype)
        if y is None and self.has_image_input:
            expected_in = int(getattr(self.base_model, "in_dim", x.shape[1]))
            raise RuntimeError(
                f"WanDiffSynthVideoAdapter.forward missing y for image-input model: "
                f"x_shape={tuple(x.shape)}, x_channels={int(x.shape[1])}, expected_in_dim={expected_in}, "
                f"clip_feature_is_none={clip_feature is None}"
            )
        if y is not None:
            y = y.to(device=x.device, dtype=model_dtype)
            x = torch.cat([x, y], dim=1)

        x, (f, h, w) = self.patchify(x)
        b, seq_len, _ = x.shape
        device = x.device

        grid_sizes = torch.tensor([[f, h, w]] * b, device=device, dtype=torch.long)
        seq_lens = torch.full((b,), seq_len, device=device, dtype=torch.long)

        t_input = timestep.to(device=device)
        if t_input.dim() > 1:
            t_input = t_input.reshape(t_input.shape[0], -1)[:, 0]

        with torch.amp.autocast("cuda", dtype=torch.float32):
            t_embed = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t_input).float().to(device=device)
            )
            t_mod = self.time_projection(t_embed).unflatten(1, (6, self.dim))
        if t_mod.dtype != model_dtype:
            t_mod = t_mod.to(dtype=model_dtype)

        with torch.amp.autocast("cuda", dtype=model_dtype):
            context_emb = self.text_embedding(context)
            freqs = self._expand_freqs(grid_sizes, device)
            if (
                self.has_image_input
                and clip_feature is not None
                and self.img_emb is not None
                and self.require_clip_embedding
            ):
                clip_feature = clip_feature.to(device=device, dtype=model_dtype)
                clip_emb = self.img_emb(clip_feature)
                context_emb = torch.cat([clip_emb, context_emb], dim=1)

            for block in self.blocks:
                if self.training and use_gradient_checkpointing:
                    def _custom_forward(x_in, e_in):
                        return block(
                            x_in,
                            e=e_in,
                            seq_lens=seq_lens,
                            grid_sizes=grid_sizes,
                            freqs=freqs,
                            context=context_emb,
                            context_lens=None,
                        )

                    if use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            x = torch.utils.checkpoint.checkpoint(
                                _custom_forward,
                                x,
                                t_mod,
                                use_reentrant=False,
                            )
                    else:
                        x = torch.utils.checkpoint.checkpoint(
                            _custom_forward,
                            x,
                            t_mod,
                            use_reentrant=False,
                        )
                else:
                    x = block(
                        x,
                        e=t_mod,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        freqs=freqs,
                        context=context_emb,
                        context_lens=None,
                    )

            x = self.head(x, t_embed.float())
        return self.unpatchify(x, (f, h, w))


class WanDualExpertRouter(nn.Module):
    def __init__(self, pipe_adapter, primary_domain="low_noise"):
        super().__init__()
        self.pipe_adapter = pipe_adapter
        self.high_noise_model = pipe_adapter.high_noise_model
        self.low_noise_model = pipe_adapter.low_noise_model
        primary_model = pipe_adapter.get_noise_model(primary_domain)
        if primary_model is None:
            primary_model = pipe_adapter.high_noise_model if pipe_adapter.high_noise_model is not None else pipe_adapter.low_noise_model
        if primary_model is None:
            raise RuntimeError("WanDualExpertRouter requires at least one denoising model")
        self.dim = primary_model.dim
        self.freq_dim = primary_model.freq_dim
        self.in_dim = primary_model.in_dim
        self.out_dim = primary_model.out_dim
        self.patch_size = primary_model.patch_size
        self.has_image_input = bool(getattr(primary_model, "has_image_input", False))
        self.require_vae_embedding = bool(getattr(primary_model, "require_vae_embedding", True))
        self.require_clip_embedding = bool(getattr(primary_model, "require_clip_embedding", True))
        self.has_image_pos_emb = bool(getattr(primary_model, "has_image_pos_emb", False))
        self._sync_public_view(primary_model)

    def _sync_public_view(self, model):
        if model is None:
            return
        for attr in (
            "dim",
            "freq_dim",
            "in_dim",
            "out_dim",
            "patch_size",
            "has_image_input",
            "require_vae_embedding",
            "require_clip_embedding",
            "has_image_pos_emb",
            "patch_embedding",
            "text_embedding",
            "time_embedding",
            "time_projection",
            "head",
            "img_emb",
            "freqs",
            "blocks",
        ):
            if hasattr(model, attr):
                setattr(self, attr, getattr(model, attr))

    def _select_model(self, timestep):
        domain = self.pipe_adapter.resolve_noise_domain_from_timestep(timestep)
        return self.pipe_adapter.get_noise_model(domain)

    def forward(self, x, timestep, context, **kwargs):
        model = self._select_model(timestep)
        if model is None:
            raise RuntimeError("WanDualExpertRouter could not resolve an active denoising model")
        self._sync_public_view(model)
        return model(x, timestep=timestep, context=context, **kwargs)


class WanTrainPipelineAdapter:
    def __init__(
        self,
        ckpt_dir,
        device="cuda:0",
        torch_dtype=torch.bfloat16,
        task="i2v-A14B",
        train_noise_domain="low_noise",
        load_both_noise_models=False,
        dual_expert_load_mode="standard",
        dual_expert_offload_dtype=None,
        dual_expert_vram_limit=None,
        dual_expert_manage_aux_models=False,
    ):
        del task
        self.ckpt_dir = ckpt_dir
        self.device = str(device)
        self.torch_dtype = torch_dtype
        self.train_noise_domain = str(train_noise_domain or "low_noise").strip().lower()
        self.load_both_noise_models = bool(load_both_noise_models)
        self.dual_expert_load_mode = _normalize_dual_expert_load_mode(dual_expert_load_mode)
        self.dual_expert_vram_management_enabled = bool(
            self.load_both_noise_models and self.dual_expert_load_mode == "vram_management"
        )
        self.dual_expert_active_offload_enabled = bool(
            self.load_both_noise_models and self.dual_expert_load_mode == "active"
        )
        self.dual_expert_offload_dtype = _resolve_torch_dtype(
            dual_expert_offload_dtype,
            default=torch_dtype,
        )
        try:
            self.dual_expert_vram_limit = float(dual_expert_vram_limit) if dual_expert_vram_limit is not None else None
        except Exception:
            self.dual_expert_vram_limit = None
        if self.dual_expert_vram_limit is not None and self.dual_expert_vram_limit < 0:
            self.dual_expert_vram_limit = None
        self.dual_expert_manage_aux_models = bool(
            dual_expert_manage_aux_models and self.dual_expert_vram_management_enabled
        )
        if self.train_noise_domain not in ("low_noise", "high_noise"):
            raise ValueError(
                f"train_noise_domain must be 'low_noise' or 'high_noise', got {self.train_noise_domain!r}"
            )

        high_noise_paths = _sorted_glob(
            os.path.join(ckpt_dir, "high_noise_model", "diffusion_pytorch_model*.safetensors")
        )
        low_noise_paths = _sorted_glob(
            os.path.join(ckpt_dir, "low_noise_model", "diffusion_pytorch_model*.safetensors")
        )
        text_encoder_path = os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        vae_path = os.path.join(ckpt_dir, "Wan2.1_VAE.pth")
        tokenizer_path = os.path.join(ckpt_dir, "google", "umt5-xxl")

        def _model_config(path, enable_vram_management=False):
            kwargs = {"path": path, "offload_device": "cpu"}
            if enable_vram_management:
                kwargs.update(
                    {
                        "offload_dtype": self.dual_expert_offload_dtype,
                        "onload_dtype": torch_dtype,
                        "onload_device": "cpu",
                        "preparing_dtype": torch_dtype,
                        "preparing_device": self.device,
                        "computation_dtype": torch_dtype,
                        "computation_device": self.device,
                    }
                )
            return ModelConfig(**kwargs)

        dit_vram_management = bool(self.dual_expert_vram_management_enabled)
        aux_vram_management = bool(self.dual_expert_manage_aux_models)
        load_device = "cpu" if self.dual_expert_active_offload_enabled else self.device
        if self.load_both_noise_models:
            model_configs = [
                _model_config(high_noise_paths, enable_vram_management=dit_vram_management),
                _model_config(low_noise_paths, enable_vram_management=dit_vram_management),
                _model_config(text_encoder_path, enable_vram_management=aux_vram_management),
                _model_config(vae_path, enable_vram_management=aux_vram_management),
            ]
        else:
            active_noise_paths = high_noise_paths if self.train_noise_domain == "high_noise" else low_noise_paths
            model_configs = [
                _model_config(active_noise_paths, enable_vram_management=False),
                _model_config(text_encoder_path, enable_vram_management=False),
                _model_config(vae_path, enable_vram_management=False),
            ]

        raw_pipe = WanVideoSviProPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=load_device,
            model_configs=model_configs,
            tokenizer_config=ModelConfig(path=tokenizer_path, skip_download=True),
            redirect_common_files=False,
            vram_limit=self.dual_expert_vram_limit if self.dual_expert_vram_management_enabled else None,
        )
        raw_pipe.device = self.device
        raw_pipe.torch_dtype = torch_dtype
        if self.dual_expert_vram_management_enabled:
            print(
                "[WanTrainPipelineAdapter] dual expert VRAM management enabled: "
                f"mode={self.dual_expert_load_mode} offload_dtype={self.dual_expert_offload_dtype} "
                f"manage_aux_models={self.dual_expert_manage_aux_models} "
                f"vram_limit={self.dual_expert_vram_limit}",
                flush=True,
            )
        if self.dual_expert_active_offload_enabled:
            print(
                "[WanTrainPipelineAdapter] dual expert active offload enabled: "
                f"base_load_device={load_device} active_device={self.device}",
                flush=True,
            )

        self._base_pipe = raw_pipe
        self.scheduler = WanTrainingSchedulerAdapter(raw_pipe.scheduler)
        self.prompter = WanPrompterAdapter(raw_pipe)
        self.vae = raw_pipe.vae
        self.image_encoder = raw_pipe.image_encoder
        if self.load_both_noise_models:
            self.high_noise_model = WanDiffSynthVideoAdapter(raw_pipe.dit) if raw_pipe.dit is not None else None
            self.low_noise_model = WanDiffSynthVideoAdapter(raw_pipe.dit2) if getattr(raw_pipe, "dit2", None) is not None else None
            if self.dual_expert_active_offload_enabled:
                for model in (self.high_noise_model, self.low_noise_model):
                    if model is not None:
                        model._active_offload_defer_cuda_to_runtime = True
        else:
            active_model = WanDiffSynthVideoAdapter(raw_pipe.dit) if raw_pipe.dit is not None else None
            self.high_noise_model = active_model if self.train_noise_domain == "high_noise" else None
            self.low_noise_model = active_model if self.train_noise_domain == "low_noise" else None
            inactive_model = self.low_noise_model if self.train_noise_domain == "high_noise" else self.high_noise_model
            if inactive_model is not None:
                inactive_dtype, inactive_device = _get_module_dtype_device(inactive_model)
                if inactive_device.type == "cuda":
                    raise RuntimeError(
                        f"Inactive Wan2.2 expert is unexpectedly on GPU ({inactive_device}, {inactive_dtype}); "
                        f"training runtime should only load the active {self.train_noise_domain} expert."
                    )
        self.active_dit = None
        self.router_dit = WanDualExpertRouter(self, primary_domain=self.train_noise_domain) if self.load_both_noise_models else None
        self.dit = None
        self.current_noise_domain = self.train_noise_domain
        self.set_active_noise_domain(self.train_noise_domain)
        self.training = True

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._base_pipe, name)

    def offload_vram_managed_models(self):
        if self.dual_expert_vram_management_enabled and hasattr(self._base_pipe, "load_models_to_device"):
            self._base_pipe.load_models_to_device([])

    def _force_model_to(self, model, *args, **kwargs):
        if model is None:
            return None
        if hasattr(model, "force_to"):
            return model.force_to(*args, **kwargs)
        return model.to(*args, **kwargs)

    def _model_summary(self, model):
        if model is None:
            return "none"
        try:
            param = next(model.parameters())
            return f"{param.device}:{param.dtype}"
        except Exception:
            pass
        try:
            dtype, device = _get_module_dtype_device(model)
            return f"{device}:{dtype}"
        except Exception:
            return "unknown"

    def active_offload_device_summary(self):
        return {
            "current": str(getattr(self, "current_noise_domain", "")),
            "high": self._model_summary(self.high_noise_model),
            "low": self._model_summary(self.low_noise_model),
            "text": self._model_summary(getattr(self, "text_encoder", None)),
            "vae": self._model_summary(getattr(self, "vae", None)),
        }

    def set_active_noise_domain(self, noise_domain):
        domain = str(noise_domain or self.train_noise_domain).strip().lower()
        if domain not in ("low_noise", "high_noise"):
            domain = self.train_noise_domain
        self.current_noise_domain = domain
        self.active_dit = self.high_noise_model if domain == "high_noise" else self.low_noise_model
        if self.active_dit is None:
            self.active_dit = self.high_noise_model if self.high_noise_model is not None else self.low_noise_model
        # ponytail: denoising_model() re-enters here 2-4x per denoise step and the domain
        # never changes at inference, so this used to empty_cache() a ~38GB pool and re-walk
        # a 14B expert on every call. Skip it when the wanted expert is already resident.
        if self.dual_expert_active_offload_enabled and not _module_is_on(self.active_dit, self.device):
            for candidate in (self.high_noise_model, self.low_noise_model):
                if candidate is not None and candidate is not self.active_dit:
                    self._force_model_to(candidate, "cpu")
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
            if self.active_dit is not None:
                self._force_model_to(self.active_dit, self.device)
        self.dit = self.active_dit
        if self.router_dit is not None:
            self.router_dit._sync_public_view(self.active_dit)
        if self.dual_expert_vram_management_enabled and self.dual_expert_load_mode == "active":
            raw_model_name = "dit" if domain == "high_noise" else "dit2"
            if getattr(self._base_pipe, raw_model_name, None) is None:
                raw_model_name = "dit"
            self._base_pipe.load_models_to_device([raw_model_name])
        return self.active_dit

    def set_active_noise_domain_from_timestep(self, timestep, boundary_ratio=0.9):
        del timestep, boundary_ratio
        return self.set_active_noise_domain(self.train_noise_domain)

    def resolve_noise_domain_from_timestep(self, timestep, boundary_ratio=0.9):
        del timestep, boundary_ratio
        return self.train_noise_domain

    def denoising_model(self):
        if self.dual_expert_active_offload_enabled and self.active_dit is not None:
            self.set_active_noise_domain(self.current_noise_domain)
        return self.active_dit

    def active_denoising_model(self):
        if self.dual_expert_active_offload_enabled and self.active_dit is not None:
            self.set_active_noise_domain(self.current_noise_domain)
        return self.active_dit

    def get_noise_model(self, noise_domain):
        domain = str(noise_domain or "").strip().lower()
        if domain == "high_noise":
            return self.high_noise_model
        if domain == "low_noise":
            return self.low_noise_model
        return None

    def preprocess_image(self, image, **kwargs):
        return self._base_pipe.preprocess_image(
            image,
            torch_dtype=self.torch_dtype,
            device=self.device,
            **kwargs,
        )

    def prepare_extra_input(self, latents=None):
        self._base_pipe.device = self.device
        self._base_pipe.torch_dtype = self.torch_dtype
        if hasattr(self._base_pipe, "prepare_extra_input"):
            return self._base_pipe.prepare_extra_input(latents)
        return {}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        vae_dtype, _ = _get_module_dtype_device(self.vae, default_device=self.device, default_dtype=self.torch_dtype)
        if vae_dtype is not None:
            latents = latents.to(dtype=vae_dtype)
        return self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

    def encode_prompt(self, prompt, positive=True):
        return self.prompter.encode_prompt(prompt, positive=positive, device=self.device)

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
        if not first_frames:
            raise ValueError("encode_images_adaptive requires at least one frame")
        if not isinstance(first_frames, list):
            first_frames = [first_frames]
        first_frames = first_frames[: int(num_frames)]
        if num_motion_latent is None:
            num_motion_latent = getattr(self, "default_num_motion_latent", None)
        original_dtype = getattr(self, "torch_dtype", torch.bfloat16)
        pipe_device = torch.device(getattr(self, "device", "cuda"))
        if random_ref_frame is None:
            random_ref_frame = first_frames[0]

        vae = getattr(self, "vae", None)
        image_encoder = getattr(self, "image_encoder", None)
        # Only inspect the active DiT config here. Calling denoising_model() in
        # active-offload mode would move the DiT onto GPU before VAE/image
        # encoding has released its peak memory.
        dit = self.get_noise_model(getattr(self, "current_noise_domain", self.train_noise_domain))
        if dit is None:
            dit = self.active_dit
        require_clip_embedding = bool(getattr(dit, "require_clip_embedding", False))
        use_clip_feature = bool(image_encoder is not None and require_clip_embedding)
        if vae is None:
            raise RuntimeError("encode_images_adaptive requires vae")
        aux_vram_loaded = False
        if self.dual_expert_manage_aux_models and hasattr(self._base_pipe, "load_models_to_device"):
            aux_names = ["vae"]
            if use_clip_feature:
                aux_names.append("image_encoder")
            self._base_pipe.load_models_to_device(aux_names)
            aux_vram_loaded = True

        vae_dtype, vae_device = _get_module_dtype_device(vae, default_device=pipe_device, default_dtype=original_dtype)
        image_dtype, image_device = _get_module_dtype_device(
            image_encoder,
            default_device=pipe_device,
            default_dtype=original_dtype,
        ) if use_clip_feature else (None, pipe_device)
        if vae_dtype is None:
            vae_dtype = original_dtype
        if use_clip_feature and image_dtype is None:
            image_dtype = original_dtype
        vae_device = torch.device(vae_device) if vae_device is not None else pipe_device
        image_device = torch.device(image_device) if image_device is not None else pipe_device
        if self.dual_expert_active_offload_enabled or self.dual_expert_vram_management_enabled:
            if vae is not None and hasattr(vae, "to"):
                vae.to(pipe_device)
            if use_clip_feature and image_encoder is not None and hasattr(image_encoder, "to"):
                image_encoder.to(pipe_device)
            vae_device = pipe_device
            image_device = pipe_device
        if aux_vram_loaded:
            # VRAM-managed aux models keep their owning module metadata on CPU but
            # materialize layer weights on the pipeline device during forward.
            vae_device = pipe_device
            image_device = pipe_device

        num_condition_frames = len(first_frames)
        remaining_frames = max(0, int(num_frames) - num_condition_frames)

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

        msk = torch.ones(1, int(num_frames), height // 8, width // 8, device=vae_device, dtype=vae_dtype)
        if bool(ref_pad_cfg):
            msk[:, num_condition_frames:] = 0
        else:
            msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        if num_condition_frames > 1:
            condition_tensors = [
                self.preprocess_image(frame.resize((width, height))).to(device=vae_device, dtype=vae_dtype)
                for frame in first_frames
            ]
            vae_input_condition = torch.cat(condition_tensors, dim=0).permute(1, 0, 2, 3)
        else:
            vae_input_condition = (
                self.preprocess_image(first_frames[0].resize((width, height)))
                .to(device=vae_device, dtype=vae_dtype)
                .transpose(0, 1)
            )

        if int(ref_pad_num or 0) == 0:
            vae_input_pad = torch.zeros(3, remaining_frames, height, width, device=vae_device, dtype=vae_dtype)
        elif ref_pad_num is not None and int(ref_pad_num) > 0:
            pad_count = min(int(ref_pad_num), remaining_frames)
            pad_imgs = [random_ref_tensor.transpose(0, 1)] * pad_count
            if remaining_frames > pad_count:
                pad_imgs.append(torch.zeros(3, remaining_frames - pad_count, height, width, device=vae_device, dtype=vae_dtype))
            vae_input_pad = torch.cat(pad_imgs, dim=1) if pad_imgs else torch.empty(3, 0, height, width, device=vae_device, dtype=vae_dtype)
        elif int(ref_pad_num) == -1:
            vae_input_pad = random_ref_tensor.transpose(0, 1).repeat(1, remaining_frames, 1, 1)
        else:
            vae_input_pad = torch.zeros(3, remaining_frames, height, width, device=vae_device, dtype=vae_dtype)

        vae_input = torch.concat([vae_input_condition, vae_input_pad], dim=1)
        try:
            y_latent = self.vae.encode(
                [vae_input],
                device=vae_device,
                tiled=False,
                tile_size=(34, 34),
                tile_stride=(18, 16),
            )[0]
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
            if aux_vram_loaded:
                retry_vae_device = pipe_device
            y_latent = self.vae.encode(
                [vae_input.to(device=retry_vae_device, dtype=retry_vae_dtype)],
                device=retry_vae_device,
                tiled=False,
                tile_size=(34, 34),
                tile_stride=(18, 16),
            )[0]

        if num_motion_latent is not None:
            keep_motion = max(0, int(num_motion_latent))
            keep_latents = min(int(y_latent.shape[1]), 1 + keep_motion)
            if keep_latents < int(y_latent.shape[1]):
                y_latent = torch.cat([y_latent[:, :keep_latents], torch.zeros_like(y_latent[:, keep_latents:])], dim=1)
        y = torch.concat([msk.to(device=y_latent.device, dtype=y_latent.dtype), y_latent], dim=0).unsqueeze(0)
        out = {"y": y.to(dtype=original_dtype, device=pipe_device)}
        if clip_context is not None:
            out["clip_feature"] = clip_context.to(dtype=original_dtype, device=pipe_device)
        if aux_vram_loaded:
            self._base_pipe.load_models_to_device([])
        elif self.dual_expert_vram_management_enabled or self.dual_expert_active_offload_enabled:
            if vae is not None and hasattr(vae, "to"):
                vae.to("cpu")
            if use_clip_feature and image_encoder is not None and hasattr(image_encoder, "to"):
                image_encoder.to("cpu")
            if torch.cuda.is_available() and pipe_device.type == "cuda":
                torch.cuda.empty_cache()
        return out


def build_wan22_training_pipe(
    ckpt_dir,
    device="cuda:0",
    torch_dtype=torch.bfloat16,
    task="i2v-A14B",
    train_noise_domain="low_noise",
    load_both_noise_models=False,
    dual_expert_load_mode="standard",
    dual_expert_offload_dtype=None,
    dual_expert_vram_limit=None,
    dual_expert_manage_aux_models=False,
):
    return WanTrainPipelineAdapter(
        ckpt_dir=ckpt_dir,
        device=device,
        torch_dtype=torch_dtype,
        task=task,
        train_noise_domain=train_noise_domain,
        load_both_noise_models=load_both_noise_models,
        dual_expert_load_mode=dual_expert_load_mode,
        dual_expert_offload_dtype=dual_expert_offload_dtype,
        dual_expert_vram_limit=dual_expert_vram_limit,
        dual_expert_manage_aux_models=dual_expert_manage_aux_models,
    )
