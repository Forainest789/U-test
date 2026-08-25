"""Invariants for the two inference hot-path fixes: expert residency guard, RoPE dtype.

Both modules pull in a working torch (and diffsynth), so these skip off-server.
"""
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest


def _load(module_name, attr):
    """Import `attr`, skipping when this box lacks a working torch / diffsynth."""
    pytest.importorskip("torch")
    try:
        module = __import__(module_name)
    except Exception as exc:
        pytest.skip(f"{module_name} unavailable here: {exc}")
    return getattr(module, attr)


def test_residency_guard_reads_parameters_not_cached_device_attr() -> None:
    torch = pytest.importorskip("torch")
    is_on = _load("wan22_train_runtime", "_module_is_on")
    model = torch.nn.Linear(2, 2)
    model.device = "cuda"  # a stale attr must not convince the guard the expert moved
    assert is_on(model, "cpu") is True
    assert is_on(model, "cuda") is False
    assert is_on(model, torch.device("cpu")) is True


def test_residency_guard_is_permissive_when_there_is_nothing_to_move() -> None:
    torch = pytest.importorskip("torch")
    is_on = _load("wan22_train_runtime", "_module_is_on")
    assert is_on(None, "cuda") is True
    assert is_on(torch.nn.Module(), "cuda") is True


def test_rope_rotates_without_float64_and_keeps_norm_and_dtype() -> None:
    torch = pytest.importorskip("torch")
    cls = _load("train_slotmem", "CharacterWiseCrossAttention")
    attn = cls(dim=32, num_heads=2, head_dim=8, rope_dim=8, query_chunk_size=0)

    x = torch.randn(6, 8, dtype=torch.float32)
    coord = torch.rand(6, dtype=torch.float64)
    out = attn._apply_rope_1d(x, coord)

    assert out.dtype == torch.float32
    # RoPE is a rotation: per-row L2 norm is invariant. Catches a broken pair split.
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), rtol=1e-5, atol=1e-5)
    # coord=0 is the identity rotation.
    assert torch.allclose(attn._apply_rope_1d(x, torch.zeros(6, dtype=torch.float64)), x, atol=1e-6)
    # bf16 in, bf16 out -- the fp32 rotation must not leak into the returned dtype.
    assert attn._apply_rope_1d(x.bfloat16(), coord).dtype == torch.bfloat16


def test_teacher_forced_query_prepass_builds_current_step_payload() -> None:
    torch = pytest.importorskip("torch")
    engine_cls = _load("infer_slotmem", "SlotMemInferenceEngine")
    engine = object.__new__(engine_cls)
    engine.pipe = SimpleNamespace(dit=SimpleNamespace(patch_size=(1, 2, 2)))
    calls = []

    def run_probe(self, **kwargs):
        calls.append(("probe", kwargs))
        return [{"layer": "maps"}], ["Evan"], {"layer": "tokens"}

    def build_payload(self, **kwargs):
        calls.append(("build", kwargs))
        return {"Evan": {0: [0, 0, 1, 1]}}, {"Evan": {"feature": "current"}}

    engine._run_character_semantic_probe = MethodType(run_probe, engine)
    engine._build_character_mask_payload_from_probe = MethodType(build_payload, engine)
    noisy_latents = torch.zeros((1, 16, 5, 6, 8))

    boxes, payload = engine._prepare_teacher_forced_query_payload(
        noisy_latents=noisy_latents,
        timestep=torch.tensor([833.0]),
        prompt="Evan returns",
        role_ids=["Evan"],
        cond_context="cond",
        uncond_context="uncond",
        image_emb_for_denoising={"y": "image"},
        extra_input={"seq_len": 1},
    )

    assert boxes == {"Evan": {0: [0, 0, 1, 1]}}
    assert payload == {"Evan": {"feature": "current"}}
    assert [name for name, _ in calls] == ["probe", "build"]
    assert calls[1][1]["h_patch"] == 3
    assert calls[1][1]["w_patch"] == 4


def test_teacher_forced_runtime_wires_current_step_query_before_forward() -> None:
    root = Path(__file__).resolve().parents[2]
    engine_source = (root / "infer_slotmem.py").read_text(encoding="utf-8")
    runtime_source = (root / "reference_inference_runtime.py").read_text(encoding="utf-8")
    setup_start = runtime_source.index("active_query_feature_payload = cached_query_feature_payload")
    measured_forward = runtime_source.index("                if use_memory_path:", setup_start)
    query_setup = runtime_source[setup_start:measured_forward]

    assert "def _prepare_teacher_forced_query_payload(" in engine_source
    assert "_prepare_teacher_forced_query_payload(" in query_setup


def test_shared_memory_survives_layerwise_query_payload() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "infer_slotmem.py").read_text(encoding="utf-8")
    assert "if layerwise_memory_banks:\n            selected_mem = None" in source
    assert "if layerwise_sparse_payload:\n            selected_mem = None" not in source


def test_probe_returns_conditional_velocity_beside_the_cfg_composite() -> None:
    root = Path(__file__).resolve().parents[2]
    runtime = (root / "reference_inference_runtime.py").read_text(encoding="utf-8")
    probe = (root / "utest" / "qstar_probe.py").read_text(encoding="utf-8")

    # the flow target is defined against the unguided conditional output, so Q* must not
    # be scored on uncond + cfg_scale * (cond - uncond).
    assert '"prediction_cond": noise_pred_cond,' in runtime
    assert '"prediction": result["prediction_cond"].detach().cpu(),' in probe
    assert 'native_result["prediction_cond"]' in probe


def test_memory_off_arm_stays_on_the_memory_aware_forward() -> None:
    root = Path(__file__).resolve().parents[2]
    runtime = (root / "reference_inference_runtime.py").read_text(encoding="utf-8")
    probe = (root / "utest" / "qstar_probe.py").read_text(encoding="utf-8")

    # no_memory has no payload, so without the force flag it would take the stock DiT
    # forward while every other confirmatory arm takes _memory_aware_dit_forward.
    assert "use_memory_path = has_memory or force_memory_path" in runtime
    assert '"force_memory_path": True,' in probe
    # the probe prepass is keyed on real memory, not on the forced path: a forced arm has
    # no role ids and the probe would just burn a forward pass.
    assert "if (has_memory and self.enable_sparse_role_memory_attn) or need_probe_for_collection:" in runtime
    # native is base Wan and must not be dragged onto the SlotMem forward.
    native_call = probe[probe.index("native_result = native_engine.generate_chunk("):]
    assert "force_memory_path" not in native_call[: native_call.index("native_predictions")]


def test_teacher_forced_prepass_captures_single_layer_query() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "infer_slotmem.py").read_text(encoding="utf-8")
    start = source.index("    def _run_character_semantic_probe(")
    end = source.index("    def _prepare_teacher_forced_query_payload(", start)
    body = source[start:end]
    assert "for feature_layer_idx in (map_layer_idx,):" in body
    assert "if len(captured_by_layer) > 1 or bool(getattr(self, \"jigsaw_extra_encoder_enabled\", False)):" not in body
    assert (
        "if len(captured_by_layer) > 1:\n"
        "                captured_layer_tokens = _make_layerwise_container(captured_by_layer)"
    ) in body


def test_sparse_token_diagnostics_are_opt_in_at_the_runtime_boundary() -> None:
    root = Path(__file__).resolve().parents[2]
    train_source = (root / "train_slotmem.py").read_text(encoding="utf-8")
    infer_source = (root / "infer_slotmem.py").read_text(encoding="utf-8")

    assert "capture_token_diagnostics=False" in train_source
    assert "debug['token_diagnostics']" in train_source
    assert "capture_sparse_token_diagnostics" in infer_source
    assert 'token_diagnostics["effective_delta_features"]' in infer_source


def test_sparse_attention_token_diagnostics_are_aligned() -> None:
    torch = pytest.importorskip("torch")
    cls = _load("train_slotmem", "CharacterWiseCrossAttention")
    module = cls(dim=8, num_heads=2, head_dim=4, rope_dim=0, time_gate=False)
    tokens = torch.randn(1, 6, 8)
    memory = torch.randn(1, 3, 8)
    payload = {"Mara": {"flat_idx": torch.tensor([1, 4])}}
    meta = [{"char_id": "Mara"} for _ in range(3)]

    _, quiet = module.forward_sparse(
        tokens,
        memory,
        query_feature_payload=payload,
        memory_bank_token_meta=meta,
        latent_h=2,
        latent_w=3,
    )
    _, captured = module.forward_sparse(
        tokens,
        memory,
        query_feature_payload=payload,
        memory_bank_token_meta=meta,
        latent_h=2,
        latent_w=3,
        capture_token_diagnostics=True,
    )

    assert "token_diagnostics" not in quiet
    diagnostics = captured["token_diagnostics"]
    assert diagnostics["flat_idx"].tolist() == [1, 4]
    assert diagnostics["host_features"].shape == (2, 8)
    assert diagnostics["raw_delta_features"].shape == (2, 8)
    assert all(
        diagnostics[name].shape == (2,)
        for name in (
            "host_norm",
            "raw_delta_norm",
            "raw_cosine_max",
            "read_logsumexp",
        )
    )
