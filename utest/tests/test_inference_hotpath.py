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
