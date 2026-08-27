"""Invariants for the two inference hot-path fixes: expert residency guard, RoPE dtype.

Both modules pull in a working torch (and diffsynth), so these skip off-server.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest


def _load_capture_writer():
    torch = pytest.importorskip("torch")
    source_path = Path(__file__).resolve().parents[2] / "infer_slotmem.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {
        "_json_safe",
        "_hash_nested_tensors",
        "_subject_subspace_tensor_sha256",
        "_subject_subspace_model_identity",
        "_validate_subject_subspace_capture_preflight",
        "_validate_subject_subspace_capture",
        "_validate_subject_subspace_provenance",
        "_prepare_subject_subspace_capture_output",
        "_write_subject_subspace_capture",
    }
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "np": None,
        "os": os,
        "torch": torch,
        "sha256_file": lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["_write_subject_subspace_capture"]


def _load_model_identity():
    return _load_capture_writer().__globals__["_subject_subspace_model_identity"]


def _load_capture_preflight():
    return _load_capture_writer().__globals__["_validate_subject_subspace_capture_preflight"]


def _load_capture_output_guard():
    source_path = Path(__file__).resolve().parents[2] / "infer_slotmem.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_prepare_subject_subspace_capture_output"
    )
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["_prepare_subject_subspace_capture_output"]


def _tensor_sha256(value) -> str:
    torch = pytest.importorskip("torch")
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _attention_sha256(value) -> str:
    torch = pytest.importorskip("torch")
    digest = hashlib.sha256()
    for role in sorted(value):
        digest.update(str(role).encode("utf-8"))
        tensor = value[role].detach().contiguous().cpu()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _capture_provenance() -> dict:
    return {
        "source_json_path": "C:/frozen/story.json",
        "source_json_sha256": "1" * 64,
        "reference_file_sha256": "2" * 64,
        "fixed_reference_scope": "source_only",
        "source_seed": 42,
        "code_identity": {
            "infer_slotmem_sha256": "3" * 64,
            "mem_encoder_utils_sha256": "5" * 64,
        },
        "model_identity": {
            "high_noise": [{
                "path": "C:/frozen/high.pt",
                "sha256": "6" * 64,
            }],
            "low_noise": [],
        },
        "runtime_identity": {
            "python_version": "3.11",
            "torch_version": "2.x",
            "inference_args_sha256": "4" * 64,
        },
    }


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


def test_conditional_only_skips_unconditional_but_default_keeps_cfg() -> None:
    root = Path(__file__).resolve().parents[2]
    runtime = (root / "reference_inference_runtime.py").read_text(encoding="utf-8")
    infer = (root / "infer_slotmem.py").read_text(encoding="utf-8")

    assert 'conditional_only = bool(teacher_forced_probe.get("conditional_only", False))' in runtime
    assert "if conditional_only:" in runtime
    assert '"prediction_semantics": prediction_semantics' in runtime
    assert '"cfg_composite_available": not conditional_only' in runtime
    assert '"dit_forward_counts": {' in runtime
    assert "self._last_teacher_forced_semantic_prepass_count = 1" in infer

    conditional_branch = runtime[runtime.index("if conditional_only:"):]
    default_unconditional = conditional_branch.index("noise_pred_uncond =")
    return_block = conditional_branch.index('"prediction_cond": noise_pred_cond')
    assert default_unconditional < return_block


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
    native_call = probe[probe.index("native_result = context_engine.generate_chunk("):]
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


def test_teacher_forced_token_controls_are_wired_before_conditional_forward() -> None:
    root = Path(__file__).resolve().parents[2]
    engine_source = (root / "infer_slotmem.py").read_text(encoding="utf-8")
    runtime_source = (root / "reference_inference_runtime.py").read_text(encoding="utf-8")

    assert "def _zero_context_positions(" in engine_source
    assert "def _override_query_indices(" in engine_source
    assert "context_zero_indices" in runtime_source
    assert "query_indices_by_role" in runtime_source
    assert "conditional_sparse_stats_by_layer" in runtime_source
    assert runtime_source.index("conditional_sparse_stats_by_layer") < runtime_source.index(
        "noise_pred_uncond ="
    )


def test_query_override_preserves_features_and_context_zeroing_preserves_layout() -> None:
    torch = pytest.importorskip("torch")
    engine_cls = _load("infer_slotmem", "SlotMemInferenceEngine")
    engine = object.__new__(engine_cls)
    payload = {"Mara": {"flat_idx": torch.tensor([1, 2]), "feature": "keep"}}

    overridden = engine._override_query_indices(
        payload, {"Mara": [3, 5]}, num_tokens=8
    )
    assert overridden["Mara"]["flat_idx"].tolist() == [3, 5]
    assert overridden["Mara"]["feature"] == "keep"
    with pytest.raises(ValueError, match="outside"):
        engine._override_query_indices(payload, {"Mara": [8]}, num_tokens=8)

    context = torch.ones(1, 6, 4)
    output = engine._zero_context_positions(context, [1, 4])
    assert output.shape == context.shape
    assert torch.all(output[:, [1, 4]] == 0)
    assert torch.all(context == 1)


def test_semantic_capture_only_returns_before_measured_forward() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "reference_inference_runtime.py").read_text(encoding="utf-8")
    capture_return = source.index("if teacher_forced_probe is not None and semantic_capture_only:")
    measured_forward = source.index("                if use_memory_path:", capture_return)
    scheduler_step = source.index("self.pipe.scheduler.step", measured_forward)

    assert capture_return < measured_forward < scheduler_step
    assert '"semantic_attention_maps"' in source[capture_return:measured_forward]


def test_subject_subspace_capture_is_opt_in_and_serializes_source_only(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    write_capture = _load_capture_writer()
    record = {
        "character": "Ana",
        "bank": 0,
        "layer": 0,
        "raw_tokens": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "raw_token_meta": [{"char_id": "Ana", "inside_box": True}] * 3,
        "encoded_slots": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "attention": {"Ana": torch.ones(2, 3) / 3},
    }

    disabled_path = tmp_path / "disabled.pt"
    provenance = _capture_provenance()
    assert write_capture(
        None,
        source_chunk_idx=0,
        captures=[record],
        provenance=provenance,
    ) is False
    assert not disabled_path.exists()

    artifact_path = tmp_path / "source_capture.pt"
    assert write_capture(
        artifact_path,
        source_chunk_idx=0,
        captures=[record],
        provenance=provenance,
    ) is True
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)

    assert artifact["source_chunk_idx"] == 0
    assert artifact["target_evidence_read"] is False
    assert artifact["provenance"] == provenance
    saved = artifact["captures"][0]
    assert saved["tensor_shapes"] == {
        "raw_tokens": [3, 4],
        "encoded_slots": [2, 4],
        "attention": {"Ana": [2, 3]},
    }
    assert set(saved["sha256"]) == {
        "raw_tokens",
        "raw_token_meta",
        "encoded_slots",
        "attention",
    }
    assert saved["sha256"] == {
        "raw_tokens": _tensor_sha256(saved["raw_tokens"]),
        "raw_token_meta": hashlib.sha256(json.dumps(
            saved["raw_token_meta"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        "encoded_slots": _tensor_sha256(saved["encoded_slots"]),
        "attention": _attention_sha256(saved["attention"]),
    }
    canonical = {
        "schema_version": artifact["schema_version"],
        "source_chunk_idx": artifact["source_chunk_idx"],
        "target_evidence_read": artifact["target_evidence_read"],
        "provenance": artifact["provenance"],
        "captures": [{
            key: saved[key]
            for key in ("character", "bank", "layer", "tensor_shapes", "sha256")
        }],
    }
    assert artifact["canonical_artifact_sha256"] == hashlib.sha256(json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    assert saved["raw_tokens"].device.type == "cpu"
    assert saved["encoded_slots"].device.type == "cpu"
    assert saved["attention"]["Ana"].device.type == "cpu"
    with pytest.raises(ValueError, match="source chunk 0"):
        write_capture(
            tmp_path / "target_capture.pt",
            source_chunk_idx=1,
            captures=[record],
            provenance=provenance,
        )


def test_subject_subspace_capture_rejects_malformed_payloads(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    write_capture = _load_capture_writer()

    def valid_record() -> dict:
        return {
            "character": "Ana",
            "bank": 0,
            "layer": 0,
            "raw_tokens": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "raw_token_meta": [{"char_id": "Ana"}] * 3,
            "encoded_slots": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "attention": {"Ana": torch.ones(2, 3) / 3},
        }

    malformed = []
    record = valid_record()
    record["attention"] = {}
    malformed.append(record)
    record = valid_record()
    record["raw_tokens"][0, 0] = torch.nan
    malformed.append(record)
    record = valid_record()
    record["encoded_slots"][0, 0] = torch.inf
    malformed.append(record)
    record = valid_record()
    record["raw_token_meta"] = record["raw_token_meta"][:-1]
    malformed.append(record)
    record = valid_record()
    record["encoded_slots"] = record["encoded_slots"][:1]
    malformed.append(record)
    record = valid_record()
    record["attention"]["Ana"] = torch.ones(2, 2) / 2
    malformed.append(record)
    record = valid_record()
    record["attention"] = {"Bo": torch.ones(2, 3) / 3}
    malformed.append(record)
    record = valid_record()
    record["attention"]["Ana"] = torch.full((2, 3), 0.2)
    malformed.append(record)
    record = valid_record()
    record["attention"]["Ana"][0, 0] = torch.nan
    malformed.append(record)
    record = valid_record()
    record["attention"]["Ana"][0, 0] = -0.1
    record["attention"]["Ana"][0, 1] += 0.1
    malformed.append(record)
    record = valid_record()
    record["encoded_slots"] = torch.ones(2, 5)
    malformed.append(record)

    for index, capture in enumerate(malformed):
        with pytest.raises(ValueError, match="subject subspace capture"):
            write_capture(
                tmp_path / f"malformed_{index}.pt",
                source_chunk_idx=0,
                captures=[capture],
                provenance=_capture_provenance(),
            )


def test_subject_subspace_capture_rejects_incomplete_provenance(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    write_capture = _load_capture_writer()
    capture = {
        "character": "Ana",
        "bank": 0,
        "layer": 0,
        "raw_tokens": torch.ones(3, 4),
        "raw_token_meta": [{"char_id": "Ana"}] * 3,
        "encoded_slots": torch.ones(2, 4),
        "attention": {"Ana": torch.ones(2, 3) / 3},
    }
    provenance = _capture_provenance()

    for key in tuple(provenance):
        incomplete = dict(provenance)
        incomplete.pop(key)
        with pytest.raises(ValueError, match="provenance"):
            write_capture(
                tmp_path / f"missing_{key}.pt",
                source_chunk_idx=0,
                captures=[capture],
                provenance=incomplete,
            )

    invalid = dict(provenance)
    invalid["source_json_sha256"] = "not-a-sha"
    with pytest.raises(ValueError, match="provenance"):
        write_capture(
            tmp_path / "invalid_sha.pt",
            source_chunk_idx=0,
            captures=[capture],
            provenance=invalid,
        )
    invalid = dict(provenance)
    invalid["code_identity"] = {"infer_slotmem_sha256": "3" * 64}
    with pytest.raises(ValueError, match="provenance"):
        write_capture(
            tmp_path / "missing_mem_encoder_hash.pt",
            source_chunk_idx=0,
            captures=[capture],
            provenance=invalid,
        )
    invalid = dict(provenance)
    invalid["runtime_identity"] = {
        "python_version": "3.11",
        "torch_version": "2.x",
    }
    with pytest.raises(ValueError, match="provenance"):
        write_capture(
            tmp_path / "missing_invocation_hash.pt",
            source_chunk_idx=0,
            captures=[capture],
            provenance=invalid,
        )


def test_subject_subspace_capture_request_rejects_existing_frozen_output_and_requires_chunk_zero(
    tmp_path: Path,
) -> None:
    prepare_output = _load_capture_output_guard()
    stale = tmp_path / "source_capture.pt"
    stale.write_bytes(b"stale")

    with pytest.raises(ValueError, match="chunk 0"):
        prepare_output(stale, processed_chunk_indices=[1, 2])
    assert stale.read_bytes() == b"stale"

    with pytest.raises(ValueError, match="already exists"):
        prepare_output(stale, processed_chunk_indices=[0, 1])
    assert stale.read_bytes() == b"stale"
    fresh = tmp_path / "fresh_capture.pt"
    assert prepare_output(fresh, processed_chunk_indices=[0, 1]) == fresh
    assert prepare_output(None, processed_chunk_indices=[]) is None


def test_subject_subspace_capture_publish_is_atomic(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    write_capture = _load_capture_writer()
    capture = {
        "character": "Ana",
        "bank": 0,
        "layer": 0,
        "raw_tokens": torch.ones(3, 4),
        "raw_token_meta": [{"char_id": "Ana"}] * 3,
        "encoded_slots": torch.ones(2, 4),
        "attention": {"Ana": torch.ones(2, 3) / 3},
    }
    output = tmp_path / "source_capture.pt"

    def failing_save(_payload, path) -> None:
        Path(path).write_bytes(b"partial")
        raise OSError("simulated save failure")

    monkeypatch.setattr(torch, "save", failing_save)
    with pytest.raises(OSError, match="simulated"):
        write_capture(
            output,
            source_chunk_idx=0,
            captures=[capture],
            provenance=_capture_provenance(),
        )
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_subject_subspace_model_identity_tracks_exact_checkpoint_bytes(tmp_path: Path) -> None:
    model_identity = _load_model_identity()
    checkpoint = tmp_path / "expert.pt"
    checkpoint.write_bytes(b"first")

    first = model_identity(high_noise=str(checkpoint), low_noise=None)
    checkpoint.write_bytes(b"second")
    second = model_identity(high_noise=str(checkpoint), low_noise=None)

    assert first["high_noise"][0]["path"] == str(checkpoint.resolve())
    assert first["high_noise"][0]["sha256"] == hashlib.sha256(b"first").hexdigest()
    assert second["high_noise"][0]["sha256"] == hashlib.sha256(b"second").hexdigest()
    assert first != second
    with pytest.raises(ValueError, match="regular file"):
        model_identity(high_noise=str(tmp_path), low_noise=None)


def test_subject_subspace_capture_rejects_deferred_lora_before_engine_init() -> None:
    preflight = _load_capture_preflight()
    preflight(None, defer_lora_until_after_first_chunk=True)
    preflight("source_capture.pt", defer_lora_until_after_first_chunk=False)
    with pytest.raises(ValueError, match="defer_lora_until_after_first_chunk"):
        preflight("source_capture.pt", defer_lora_until_after_first_chunk=True)

    source = (Path(__file__).resolve().parents[2] / "infer_slotmem.py").read_text(
        encoding="utf-8"
    )
    main_start = source.index("def main():")
    preflight_call = source.index("_validate_subject_subspace_capture_preflight(", main_start)
    engine_init = source.index("engine = SlotMemInferenceEngine(args)", main_start)
    assert preflight_call < engine_init


def test_subject_subspace_capture_is_wired_at_the_source_writer_boundary() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "infer_slotmem.py").read_text(encoding="utf-8")

    assert 'parser.add_argument("--subject_subspace_capture_path", type=str, default=None)' in source
    encode_start = source.index("    def _stage2_prepare_payload_for_bank(")
    writer_start = source.index("            online_tokens =", encode_start)
    record_capture = source.index("_record_subject_subspace_capture(", encode_start)
    memory_write = source.index("mem_manager.add_memory(", writer_start)
    persist_capture = source.index("_write_subject_subspace_capture(", writer_start)
    prepare_output = source.index(
        "subject_subspace_capture_output = _prepare_subject_subspace_capture_output("
    )
    freeze_provenance = source.index("        subject_subspace_provenance = {")
    chunk_loop = source.index("    for chunk_idx, chunk in enumerate(chunks_to_iterate")

    assert "attention_sink=attention_sink" in source[encode_start:writer_start]
    assert "int(chunk_idx) == 0" in source[encode_start:writer_start]
    assert "cannot use already-encoded slot metadata" in source
    assert prepare_output < freeze_provenance < chunk_loop
    assert record_capture < memory_write < persist_capture
    frozen_body = source[freeze_provenance:chunk_loop]
    assert all(
        field in frozen_body
        for field in (
            '"source_json_path"',
            '"source_json_sha256"',
            '"reference_file_sha256"',
            '"fixed_reference_scope"',
            '"source_seed"',
            '"code_identity"',
            '"runtime_identity"',
            '"inference_args_sha256"',
            '"model_identity"',
        )
    )
    assert "_subject_subspace_model_identity(" in frozen_body
    assert source.rindex("if subject_subspace_capture_output is not None:", 0, freeze_provenance) < freeze_provenance
    persist_body = source[persist_capture:source.index("        memory_bank_stats", persist_capture)]
    assert "sha256_file(" not in persist_body
    assert "provenance=subject_subspace_provenance" in persist_body
