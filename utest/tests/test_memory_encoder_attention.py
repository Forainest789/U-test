from pathlib import Path

import pytest
import torch

from mem_encoder_utils import MemoryEncoderBank, encode_role_tokens_to_slots


def test_attention_capture_preserves_slot_output() -> None:
    torch.manual_seed(0)
    bank = MemoryEncoderBank(
        dim=8,
        layer_groups=[[0]],
        slots=4,
        encoder_dim=6,
        hidden_dim=10,
    )
    tokens = torch.randn(5, 8)
    meta = [{"char_id": "Ana"}] * 5

    baseline = encode_role_tokens_to_slots(bank, tokens, meta, 0)
    sink = {}
    captured = encode_role_tokens_to_slots(
        bank,
        tokens,
        meta,
        0,
        attention_sink=sink,
    )

    assert torch.equal(baseline[0], captured[0])
    assert sink["Ana"].shape == (4, 5)
    assert torch.allclose(sink["Ana"].sum(dim=-1), torch.ones(4), atol=1e-6)


def test_attention_capture_separates_roles() -> None:
    bank = MemoryEncoderBank(
        dim=4,
        layer_groups=[[0]],
        slots=2,
        encoder_dim=4,
        hidden_dim=4,
    )
    sink = {}

    encode_role_tokens_to_slots(
        bank,
        torch.randn(4, 4),
        [
            {"char_id": "Ana"},
            {"char_id": "Ana"},
            {"char_id": "Bo"},
            {"char_id": "Bo"},
        ],
        0,
        attention_sink=sink,
    )

    assert set(sink) == {"Ana", "Bo"}
    assert sink["Ana"].shape == sink["Bo"].shape == (2, 2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_capture_returns_precast_float32_attention(dtype) -> None:
    torch.manual_seed(0)
    bank = MemoryEncoderBank(
        dim=8,
        layer_groups=[[0]],
        slots=4,
        encoder_dim=8,
        hidden_dim=8,
    ).to(dtype=dtype)
    tokens = torch.randn(5, 8, dtype=dtype)
    meta = [{"char_id": "Ana"}] * 5
    try:
        baseline = encode_role_tokens_to_slots(bank, tokens, meta, 0)
        sink = {}
        captured = encode_role_tokens_to_slots(
            bank,
            tokens,
            meta,
            0,
            attention_sink=sink,
        )
    except RuntimeError as exc:
        pytest.skip(f"CPU {dtype} encoder operations unavailable: {exc}")

    assert torch.equal(baseline[0], captured[0])
    assert sink["Ana"].dtype == torch.float32
    assert torch.allclose(sink["Ana"].sum(dim=-1), torch.ones(4), atol=1e-6)


def test_default_path_does_not_retain_precast_attention_reference() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "mem_encoder_utils.py"
    ).read_text(encoding="utf-8")
    start = source.index("        attn = torch.softmax(scores.float(), dim=-1)")
    end = source.index("        pooled = torch.matmul(attn, h)", start)
    attention_setup = source[start:end]

    assert "if return_attention:\n            capture_attn = attn" in attention_setup
    assert "attn = attn.to(dtype=h.dtype)" in attention_setup
    assert attention_setup.index("capture_attn = attn") < attention_setup.index(
        "attn = attn.to(dtype=h.dtype)"
    )
