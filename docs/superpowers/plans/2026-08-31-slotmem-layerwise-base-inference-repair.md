# SlotMem Base-Inference Layerwise Memory Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the shared SlotMem base-inference path so one DiT forward captures layers `0-15`, preserves the existing character-token/query semantics, encodes 64 slots per layer, and rejects flat or incomplete donor artifacts before completion.

**Architecture:** Add one strict layerwise payload validator at the existing input-contract boundary and reuse it from donor completion and donor bundle publication. In the normal inference loop, register one feature tap for each configured MemoryEncoder layer plus the shared query layer, then split the same forward's captures into a single query tensor and a complete layerwise memory container; the existing token selector, MemoryEncoder writer, RoleWiseSlotMemoryBank, and layer-matched injection paths remain authoritative.

**Tech Stack:** Python 3.10+, PyTorch, pytest, existing ViStoryBench/SlotMem harnesses; no new dependencies or CLI options.

## Global Constraints

- Implement approved方案 A in the existing SlotMem base-inference path.
- Capture raw features for configured MemoryEncoder layers `0-15` in one normal DiT forward.
- Preserve `two_role_diff`, the current selected token indices/order/limits, and the shared query payload at `sparse_role_memory_layer_idx`.
- Encode each layer independently into exactly 64 slots and inject layer `i` only into layer `i`.
- Do not copy, tile, synthesize, or reinterpret another layer's features when a capture is missing.
- Do not add an extra model forward, Song-specific runtime branch, dependency, or CLI argument.
- Fail closed on missing/extra layers, non-string layer keys, non-2D tensors, non-floating/non-finite tensors, wrong slot count, or inconsistent hidden dimensions.
- Preserve event identity, donor identity, selection scope, audit, command, repository, checkpoint, runtime hashes, and all no-clobber behavior.
- Keep `vistorybench_song_yuchen_exploratory_v1` immutable as failed evidence; operational recovery uses a fresh `vistorybench_song_yuchen_exploratory_v2` root.

---

## File Map

- `utest/input_contract.py`: own the reusable structural validator for a layerwise slot payload.
- `utest/tests/test_input_contract.py`: cover the validator's complete valid/invalid tensor geometry on CPU.
- `reference_inference_runtime.py`: capture a union of query and MemoryEncoder layers during the existing conditional forward and split the two consumers.
- `utest/tests/test_inference_hotpath.py`: prove the split preserves the query tensor, returns all memory layers, fails closed, and is wired into the normal path without changing the teacher-forced path.
- `utest/vistory_donor_harness.py`: enforce exact target-character bank-0 identity and complete geometry before writing or accepting completion.
- `utest/vistory_donor_bundle.py`: replace its duplicate tensor-geometry checks with the shared validator while retaining bundle identity/provenance ownership.
- `utest/tests/test_vistory_donor_harness.py`: make donor fixtures protocol-valid and prove invalid geometry cannot receive completion.
- `utest/tests/test_vistory_donor_bundle.py`: retain publication regression coverage against the shared validator.
- `utest/README.md`: document immutable-v1 to fresh-v2 Song recovery and the 16-layer/64-slot gate.

---

### Task 1: Add the shared strict layerwise payload validator

**Files:**
- Modify: `utest/input_contract.py:62-76`
- Modify: `utest/tests/test_input_contract.py:1-80`

**Interfaces:**
- Consumes: a payload object, `expected_layers: Sequence[int]`, and `expected_slots: int`.
- Produces: `validate_layerwise_slot_payload(payload: object, *, expected_layers: Sequence[int], expected_slots: int) -> dict[str, list[int]]`.
- Preserves: `payload_slot_shapes(tokens, payload_key)` for existing generic donor-bundle compatibility.

- [ ] **Step 1: Write the valid-geometry test**

Add `torch` and the new import to `utest/tests/test_input_contract.py`, then add:

```python
import torch

from utest.input_contract import (
    validate_donor_bundle,
    validate_layerwise_slot_payload,
    validate_teacher_bundle,
)


def _layerwise_payload() -> dict:
    return {
        "__layerwise__": True,
        "layers": {
            str(layer): torch.zeros((64, 3), dtype=torch.float16)
            for layer in range(16)
        },
    }


def test_layerwise_slot_payload_accepts_exact_frozen_geometry() -> None:
    shapes = validate_layerwise_slot_payload(
        _layerwise_payload(), expected_layers=range(16), expected_slots=64
    )

    assert shapes == {str(layer): [64, 3] for layer in range(16)}
```

- [ ] **Step 2: Run the valid-geometry test and verify RED**

Run:

```bash
pytest -q utest/tests/test_input_contract.py::test_layerwise_slot_payload_accepts_exact_frozen_geometry
```

Expected: collection fails because `validate_layerwise_slot_payload` does not exist.

- [ ] **Step 3: Add the invalid-geometry matrix**

Add this test beside the valid case:

```python
@pytest.mark.parametrize(
    ("malformation", "message"),
    [
        ("flat", "layerwise"),
        ("marker_int", "layerwise"),
        ("empty", "layers"),
        ("integer_key", "layers"),
        ("missing", "layers"),
        ("extra", "layers"),
        ("wrong_slots", "64-slot"),
        ("rank", "2D"),
        ("integer_tensor", "floating"),
        ("nonfinite", "finite"),
        ("hidden_dim", "hidden dimension"),
    ],
)
def test_layerwise_slot_payload_rejects_malformed_geometry(
    malformation: str, message: str
) -> None:
    payload = _layerwise_payload()
    layers = payload["layers"]
    if malformation == "flat":
        payload = torch.zeros((64, 3), dtype=torch.float16)
    elif malformation == "marker_int":
        payload["__layerwise__"] = 1
    elif malformation == "empty":
        layers.clear()
    elif malformation == "integer_key":
        layers[0] = layers.pop("0")
    elif malformation == "missing":
        layers.pop("15")
    elif malformation == "extra":
        layers["16"] = torch.zeros((64, 3), dtype=torch.float16)
    elif malformation == "wrong_slots":
        layers["0"] = torch.zeros((63, 3), dtype=torch.float16)
    elif malformation == "rank":
        layers["0"] = torch.zeros((64, 1, 3), dtype=torch.float16)
    elif malformation == "integer_tensor":
        layers["0"] = torch.zeros((64, 3), dtype=torch.int64)
    elif malformation == "nonfinite":
        layers["0"][0, 0] = float("nan")
    else:
        layers["15"] = torch.zeros((64, 4), dtype=torch.float16)

    with pytest.raises(ValueError, match=message):
        validate_layerwise_slot_payload(
            payload, expected_layers=range(16), expected_slots=64
        )
```

- [ ] **Step 4: Implement the minimal shared validator**

Add `Sequence` to the imports in `utest/input_contract.py`, then add the validator immediately after `payload_slot_shapes`:

```python
def validate_layerwise_slot_payload(
    payload: object,
    *,
    expected_layers: Sequence[int],
    expected_slots: int,
) -> dict[str, list[int]]:
    import torch

    expected_keys = tuple(str(int(layer)) for layer in expected_layers)
    if (
        not isinstance(payload, Mapping)
        or payload.get("__layerwise__") is not True
        or not isinstance(payload.get("layers"), Mapping)
    ):
        raise ValueError("selected donor payload must be a layerwise tensor payload")
    layers = payload["layers"]
    if any(type(layer) is not str or not layer for layer in layers) or set(layers) != set(expected_keys):
        raise ValueError(f"selected donor payload layers must be exactly {list(expected_keys)}")

    shapes: dict[str, list[int]] = {}
    hidden_dims: set[int] = set()
    for layer in expected_keys:
        tensor = layers[layer]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"selected donor payload layer {layer} must be a tensor")
        if tensor.ndim != 2:
            raise ValueError(f"selected donor payload layer {layer} must be a 2D tensor")
        if int(tensor.shape[0]) != int(expected_slots):
            raise ValueError(f"selected donor payload layer {layer} must be a {expected_slots}-slot tensor")
        if not tensor.is_floating_point():
            raise ValueError(f"selected donor payload layer {layer} must be floating point")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"selected donor payload layer {layer} must be finite")
        shapes[layer] = [int(value) for value in tensor.shape]
        hidden_dims.add(int(tensor.shape[1]))
    if len(hidden_dims) != 1:
        raise ValueError("selected donor payload layers must share one hidden dimension")
    return shapes
```

- [ ] **Step 5: Run the validator tests and the existing input-contract tests**

Run:

```bash
pytest -q utest/tests/test_input_contract.py
```

Expected: all tests pass, including the existing flat generic `validate_donor_bundle` case.

- [ ] **Step 6: Commit Task 1**

```bash
git add utest/input_contract.py utest/tests/test_input_contract.py
git commit -m "test: define strict layerwise slot payload contract"
```

---

### Task 2: Split one base-inference forward into shared query and layerwise memory consumers

**Files:**
- Modify: `reference_inference_runtime.py:1175-1267,1344-1375`
- Modify: `utest/tests/test_inference_hotpath.py:110-290`

**Interfaces:**
- Consumes: captures from `AttentionOutputFeatureTap`, configured `jigsaw_extra_encoder_layers`, `sparse_role_memory_layer_idx`, and `jigsaw_extra_encoder_enabled`.
- Produces: `_partition_layerwise_feature_capture(captured_by_layer: Mapping[object, torch.Tensor], *, memory_layers: Sequence[int], query_layer: int, layerwise_memory: bool) -> tuple[torch.Tensor, torch.Tensor | dict]`.
- The first returned value is the unchanged single-layer query tensor; the second is either that tensor in legacy non-MemoryEncoder mode or `{"__layerwise__": True, "layers": ...}` in MemoryEncoder mode.
- Preserves: `_build_character_mask_payload_from_probe` receives only the query-layer tensor, while `_extract_memory_from_step_maps(..., token_source_override=...)` receives the memory value.

- [ ] **Step 1: Write CPU tests for the capture partition**

Add `Mapping` and `Sequence` imports used by the helper implementation, then add these tests in `utest/tests/test_inference_hotpath.py`:

```python
def test_base_capture_keeps_shared_query_and_all_layerwise_memory() -> None:
    torch = pytest.importorskip("torch")
    partition = _load(
        "reference_inference_runtime", "_partition_layerwise_feature_capture"
    )
    captured = {
        str(layer): torch.full((5, 3), float(layer), dtype=torch.float32)
        for layer in range(16)
    }

    query, memory = partition(
        captured,
        memory_layers=range(16),
        query_layer=3,
        layerwise_memory=True,
    )

    assert query is captured["3"]
    assert memory["__layerwise__"] is True
    assert list(memory["layers"]) == [str(layer) for layer in range(16)]
    assert all(memory["layers"][str(layer)] is captured[str(layer)] for layer in range(16))


def test_base_capture_allows_query_layer_outside_memory_layers_without_publishing_it() -> None:
    torch = pytest.importorskip("torch")
    partition = _load(
        "reference_inference_runtime", "_partition_layerwise_feature_capture"
    )
    captured = {
        str(layer): torch.zeros((5, 3))
        for layer in (3, 11, 12, 13, 14, 15)
    }

    query, memory = partition(
        captured,
        memory_layers=range(11, 16),
        query_layer=3,
        layerwise_memory=True,
    )

    assert query is captured["3"]
    assert set(memory["layers"]) == {"11", "12", "13", "14", "15"}


@pytest.mark.parametrize(
    ("missing_layer", "message"),
    [("15", "capture layers"), ("3", "query layer")],
)
def test_base_capture_fails_closed_when_a_required_layer_is_missing(
    missing_layer: str, message: str
) -> None:
    torch = pytest.importorskip("torch")
    partition = _load(
        "reference_inference_runtime", "_partition_layerwise_feature_capture"
    )
    captured = {
        str(layer): torch.zeros((5, 3))
        for layer in range(16)
        if str(layer) != missing_layer
    }

    with pytest.raises(ValueError, match=message):
        partition(
            captured,
            memory_layers=range(16),
            query_layer=3,
            layerwise_memory=True,
        )
```

- [ ] **Step 2: Run the capture tests and verify RED**

Run:

```bash
pytest -q \
  utest/tests/test_inference_hotpath.py::test_base_capture_keeps_shared_query_and_all_layerwise_memory \
  utest/tests/test_inference_hotpath.py::test_base_capture_allows_query_layer_outside_memory_layers_without_publishing_it \
  utest/tests/test_inference_hotpath.py::test_base_capture_fails_closed_when_a_required_layer_is_missing
```

Expected: tests fail because `_partition_layerwise_feature_capture` is absent.

- [ ] **Step 3: Implement the capture partition helper**

Add `Mapping` and `Sequence` from `collections.abc` near the top of `reference_inference_runtime.py`, then place this helper after `AttentionOutputFeatureTap`:

```python
def _partition_layerwise_feature_capture(
    captured_by_layer: Mapping[object, torch.Tensor],
    *,
    memory_layers: Sequence[int],
    query_layer: int,
    layerwise_memory: bool,
) -> tuple[torch.Tensor, torch.Tensor | dict]:
    query_key = str(int(query_layer))
    memory_keys = tuple(str(int(layer)) for layer in memory_layers)
    expected_keys = set(memory_keys) | {query_key}
    captured = {str(layer): tensor for layer, tensor in captured_by_layer.items()}
    if query_key not in captured:
        raise ValueError(f"missing shared query layer capture: {query_key}")
    if set(captured) != expected_keys:
        raise ValueError(
            f"feature capture layers mismatch: expected={sorted(expected_keys)} "
            f"actual={sorted(captured)}"
        )
    for layer, tensor in captured.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError(f"feature capture layer {layer} must be a 2D tensor")
    query_tokens = captured[query_key]
    if not layerwise_memory:
        return query_tokens, query_tokens
    return query_tokens, {
        "__layerwise__": True,
        "layers": {layer: captured[layer] for layer in memory_keys},
    }
```

The explicit query check must remain before the set-equality check so a missing query layer has a distinct fail-closed diagnostic.

- [ ] **Step 4: Replace the one-tap normal-path setup with the union of memory and query layers**

In `ReferenceInferenceRuntime.generate_chunk`, replace `probe_feature_tap = None` with `probe_feature_taps = []`. Inside the existing `attn_out/self_attn_out` branch, register the exact union:

```python
query_layer = int(getattr(self, "sparse_role_memory_layer_idx", 7))
layerwise_memory = bool(getattr(self, "jigsaw_extra_encoder_enabled", False))
memory_capture_layers = (
    tuple(int(layer) for layer in getattr(self, "jigsaw_extra_encoder_layers", ()))
    if layerwise_memory
    else (query_layer,)
)
capture_layers = tuple(sorted(set(memory_capture_layers) | {query_layer}))
for tap_layer in capture_layers:
    feature_tap = AttentionOutputFeatureTap(
        dit_model=self.pipe.denoising_model(),
        layer_idx=tap_layer,
        keep_device="cpu",
        keep_dtype=torch.bfloat16,
        source=str(getattr(self, "sparse_role_memory_feature_source", "attn_out")),
    )
    feature_tap.register()
    probe_feature_taps.append((tap_layer, feature_tap))
```

Do not alter `MultiCharacterAttentionMapExtractor`, `probe_ordered_roles`, attention-map construction, `two_role_diff`, or query overrides.

- [ ] **Step 5: Partition the captures after the same forward and wire each consumer**

Replace the single `pop_tokens()` block with:

```python
captured_by_layer = {}
for tap_layer, feature_tap in probe_feature_taps:
    captured = feature_tap.pop_tokens()
    feature_tap.remove()
    if isinstance(captured, torch.Tensor) and captured.ndim == 2:
        captured_by_layer[str(tap_layer)] = captured

query_layer_tokens = None
probe_layer_tokens = None
if probe_feature_taps:
    query_layer_tokens, probe_layer_tokens = _partition_layerwise_feature_capture(
        captured_by_layer,
        memory_layers=memory_capture_layers,
        query_layer=query_layer,
        layerwise_memory=layerwise_memory,
    )
```

Then pass `layer_tokens=query_layer_tokens` to `_build_character_mask_payload_from_probe`. Leave the later call exactly as `token_source_override=probe_layer_tokens`; `_extract_memory_from_step_maps` will apply the existing selected indices to each layer's own raw tensor.

- [ ] **Step 6: Add a wiring regression around the normal path**

Add this source-boundary test to `utest/tests/test_inference_hotpath.py`:

```python
def test_normal_base_path_partitions_query_and_memory_from_one_forward() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "reference_inference_runtime.py"
    ).read_text(encoding="utf-8")
    start = source.index("                probe_feature_taps = []")
    end = source.index("                if conditional_only:", start)
    body = source[start:end]

    assert "capture_layers = tuple(sorted(set(memory_capture_layers) | {query_layer}))" in body
    assert "for tap_layer in capture_layers:" in body
    assert body.count("noise_pred_cond =") == 2  # existing memory-aware/base alternatives
    assert "_partition_layerwise_feature_capture(" in body
    assert "layer_tokens=query_layer_tokens" in body
    assert "token_source_override=probe_layer_tokens" not in body

    writer = source[source.index("                if collect_chars and", end):]
    assert "token_source_override=probe_layer_tokens" in writer
```

The two `noise_pred_cond` assignments are mutually exclusive branches of one conditional pass; do not introduce a separate probe forward in this block.

- [ ] **Step 7: Run focused inference tests, including the unchanged teacher-forced contract**

Run:

```bash
pytest -q \
  utest/tests/test_inference_hotpath.py::test_base_capture_keeps_shared_query_and_all_layerwise_memory \
  utest/tests/test_inference_hotpath.py::test_base_capture_allows_query_layer_outside_memory_layers_without_publishing_it \
  utest/tests/test_inference_hotpath.py::test_base_capture_fails_closed_when_a_required_layer_is_missing \
  utest/tests/test_inference_hotpath.py::test_normal_base_path_partitions_query_and_memory_from_one_forward \
  utest/tests/test_inference_hotpath.py::test_teacher_forced_prepass_captures_single_layer_query \
  utest/tests/test_inference_hotpath.py::test_teacher_forced_runtime_wires_current_step_query_before_forward
```

Expected: all six tests pass.

- [ ] **Step 8: Run the complete inference hot-path file**

```bash
pytest -q utest/tests/test_inference_hotpath.py
```

Expected: all runnable tests pass; environment-dependent imports may skip with their existing reason.

- [ ] **Step 9: Commit Task 2**

```bash
git add reference_inference_runtime.py utest/tests/test_inference_hotpath.py
git commit -m "fix: capture layerwise memory in base inference"
```

---

### Task 3: Move complete geometry enforcement to donor completion

**Files:**
- Modify: `utest/vistory_donor_harness.py:441-488`
- Modify: `utest/vistory_donor_bundle.py:1-16,137-203`
- Modify: `utest/tests/test_vistory_donor_harness.py:198-233,939-969,1469-1500,1544-1570`
- Verify: `utest/tests/test_vistory_donor_bundle.py:480-585`

**Interfaces:**
- Consumes: `validate_layerwise_slot_payload(...)` from Task 1 and `validate_slotmem_memory_encoder_geometry(frozen_args) -> tuple[tuple[int, ...], int]`.
- Produces: donor completion that is writable/reusable only when the artifact contains exactly `<event.character_name>|0` and the declared layerwise geometry.
- Preserves: bundle-level selection binding, event/donor identity, payload hashes, audit, runtime contract, dtype publication, slot counts, and atomic no-clobber publication.

- [ ] **Step 1: Convert the harness payload fixture to the formal 0-15/64 contract**

Replace `_materialize_payload` in `utest/tests/test_vistory_donor_harness.py` with this fixture body:

```python
def _materialize_payload(job: dict) -> None:
    payload = Path(job["donor_payload"])
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload_key = f'{job["event"]["character_name"]}|0'
    layer_shapes = {str(layer): [64, 3] for layer in range(16)}
    torch.save(
        {
            "format": "slotmem_donor_payload_v2",
            "event": job["event"],
            "payloads": {
                payload_key: {
                    "__layerwise__": True,
                    "layers": {
                        str(layer): torch.zeros((64, 3), dtype=torch.float16)
                        for layer in range(16)
                    },
                }
            },
        },
        payload,
    )
    _write_json(
        Path(job["donor_payload_info"]),
        {
            "format": "slotmem_donor_payload_v2",
            "payload_path": str(payload.resolve()),
            "payload_sha256": _sha256(payload),
            "payload_keys": [payload_key],
            "payload_slot_shapes": {payload_key: layer_shapes},
            "event": job["event"],
        },
    )
    _write_json(
        Path(job["dump_dir"]) / "correct" / "audit.json",
        {
            "arm": "correct",
            "seed": 0,
            "target_character": job["event"]["character_name"],
            "target_chunk_idx": job["event"]["target_chunk_idx"],
            "target_read_hits": 1,
            "intervention_effective": True,
            "donor_dumped": str(payload.resolve()),
            "donor_sha256": _sha256(payload),
            "runtime_contract": job["dump_runtime_contract"],
        },
    )
    _materialize_execution(job, "dump")
```

Update the two sidecar-tamper tests to derive `payload_key = info["payload_keys"][0]` and mutate layer `"0"`; use `[99, 99]` for shape mismatch and `[64.0, 3.0]` for type-strict mismatch.

- [ ] **Step 2: Add a completion-boundary RED test for the observed flat payload**

Add this test to `utest/tests/test_vistory_donor_harness.py`:

```python
def test_completion_rejects_flat_payload_before_record_is_written(tmp_path: Path) -> None:
    selection_path = _exploratory_selection(tmp_path)
    base, platform = _inputs(tmp_path)
    run = build_donor_run_manifest(
        selection_path=selection_path,
        output_root=tmp_path / "run",
        base_inference_args_path=base,
        platform_manifest_path=platform,
        python_executable=sys.executable,
    )
    job = run["jobs"][0]
    _materialize_prefix(job, platform)
    _materialize_payload(job)
    payload_path = Path(job["donor_payload"])
    artifact = torch.load(payload_path, map_location="cpu", weights_only=True)
    payload_key = next(iter(artifact["payloads"]))
    artifact["payloads"][payload_key] = torch.zeros((64, 3), dtype=torch.float16)
    torch.save(artifact, payload_path)
    info_path = Path(job["donor_payload_info"])
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["payload_sha256"] = _sha256(payload_path)
    info["payload_slot_shapes"][payload_key] = {"0": [64, 3]}
    _write_json(info_path, info)

    with pytest.raises(ValueError, match="layerwise"):
        donor_harness._write_completion(job, run)

    assert not Path(job["completion"]).exists()
```

- [ ] **Step 3: Run the completion test and verify RED**

```bash
pytest -q utest/tests/test_vistory_donor_harness.py::test_completion_rejects_flat_payload_before_record_is_written
```

Expected before the production change: the test fails because the harness writes a completion for the flat tensor.

- [ ] **Step 4: Enforce exact identity and shared geometry inside `_validate_payload`**

Import `validate_layerwise_slot_payload` from `.input_contract` at module scope in `utest/vistory_donor_harness.py`. Replace the current generic multi-key shape block with:

```python
event = job.get("event")
if not isinstance(event, Mapping):
    raise ValueError("donor job event is missing")
expected_key = f'{event.get("character_name", "")}|0'
keys = info.get("payload_keys")
payloads = artifact["payloads"]
if keys != [expected_key] or set(payloads) != {expected_key}:
    raise ValueError("donor payload must contain exactly the target-character bank 0 key")
runtime = job.get("dump_runtime_contract")
frozen_args = runtime.get("frozen_args") if isinstance(runtime, Mapping) else None
if not isinstance(frozen_args, Mapping):
    raise ValueError("donor dump runtime is missing frozen args")
expected_layers, expected_slots = validate_slotmem_memory_encoder_geometry(frozen_args)
actual_shapes = validate_layerwise_slot_payload(
    payloads[expected_key],
    expected_layers=expected_layers,
    expected_slots=expected_slots,
)
shapes = info.get("payload_slot_shapes")
if not isinstance(shapes, Mapping) or set(shapes) != {expected_key}:
    raise ValueError("donor payload keys/shapes are invalid")
if not _json_equal_strict(shapes[expected_key], actual_shapes):
    raise ValueError("donor payload slot shapes do not match payload info")
```

Keep all existing event, payload path/SHA, audit, runtime-contract, target-hit, effectiveness, and donor hash checks around this block.

- [ ] **Step 5: Prove missing geometry also fails before completion**

Add a sibling test that starts from the valid fixture, removes layer `"15"`, updates `payload_sha256` and `payload_slot_shapes` to the remaining 15 layers, calls `donor_harness._write_completion(job, run)`, expects `ValueError` matching `"layers"`, and asserts the completion path does not exist. Use this exact mutation:

```python
del artifact["payloads"][payload_key]["layers"]["15"]
torch.save(artifact, payload_path)
info["payload_sha256"] = _sha256(payload_path)
del info["payload_slot_shapes"][payload_key]["15"]
_write_json(info_path, info)
```

- [ ] **Step 6: Reuse the shared validator in bundle publication**

In `utest/vistory_donor_bundle.py`:

- import `validate_layerwise_slot_payload` beside `payload_slot_shapes`;
- remove `LAYERS_KEY`, `_is_layerwise`, and `FROZEN_MEMORY_ENCODER_SLOTS` imports if no longer referenced;
- after exact one-key/target-character-bank-0 selection, call:

```python
shapes = validate_layerwise_slot_payload(
    payload,
    expected_layers=expected_layers,
    expected_slots=expected_slots,
)
if not _json_equal_strict(info.get("payload_slot_shapes", {}).get(key), shapes):
    raise ValueError("donor payload shape differs from payload info")
tensors = payload["layers"]
dtypes = {
    layer: str(tensor.dtype).removeprefix("torch.")
    for layer, tensor in tensors.items()
}
slot_counts = {layer: shape[0] for layer, shape in shapes.items()}
```

Delete only the duplicate layer marker/key/rank/slot/dtype/finiteness/hidden-dimension conditional. Keep `_payload_metadata`'s runtime mode, expected geometry, exact key, target-character, sidecar, dtype, and slot-count responsibilities.

- [ ] **Step 7: Run harness completion and reuse tests**

```bash
pytest -q \
  utest/tests/test_vistory_donor_harness.py::test_completion_rejects_flat_payload_before_record_is_written \
  utest/tests/test_vistory_donor_harness.py::test_completion_rejects_missing_layer_before_record_is_written \
  utest/tests/test_vistory_donor_harness.py::test_completed_run_accepts_one_fully_valid_exploratory_job \
  utest/tests/test_vistory_donor_harness.py::test_completed_run_gate_returns_only_three_fully_valid_jobs \
  utest/tests/test_vistory_donor_harness.py::test_resume_skips_three_fully_valid_jobs_without_subprocess
```

Expected: all five tests pass; the valid fixture completes and resumes, while both invalid artifacts leave no completion.

- [ ] **Step 8: Run the existing bundle geometry matrix**

```bash
pytest -q \
  utest/tests/test_vistory_donor_bundle.py::test_flat_tensor_payload_is_rejected_before_freeze \
  utest/tests/test_vistory_donor_bundle.py::test_empty_layerwise_payload_is_rejected_before_freeze \
  utest/tests/test_vistory_donor_bundle.py::test_layerwise_payload_requires_complete_floating_tensor_layers \
  utest/tests/test_vistory_donor_bundle.py::test_freeze_emits_exactly_three_valid_event_level_donor_pairs
```

Expected: all cases pass against the shared validator.

- [ ] **Step 9: Run complete donor harness and bundle files**

```bash
pytest -q utest/tests/test_vistory_donor_harness.py utest/tests/test_vistory_donor_bundle.py
```

Expected: all tests pass.

- [ ] **Step 10: Commit Task 3**

```bash
git add \
  utest/vistory_donor_harness.py \
  utest/vistory_donor_bundle.py \
  utest/tests/test_vistory_donor_harness.py \
  utest/tests/test_vistory_donor_bundle.py
git commit -m "fix: reject incomplete donor memory at completion"
```

---

### Task 4: Document Song v2 recovery and verify the complete repair

**Files:**
- Modify: `utest/README.md:1066-1268`
- Verify: `utest/tests/test_input_contract.py`
- Verify: `utest/tests/test_inference_hotpath.py`
- Verify: `utest/tests/test_vistory_donor_harness.py`
- Verify: `utest/tests/test_vistory_donor_bundle.py`
- Verify: `utest/tests/test_subject_reappearance_harness.py`

**Interfaces:**
- Consumes: repaired runtime, strengthened completion, existing immutable Song v1 selection, and frozen target manifest hash `66c3f38c54924f06d375345637eaf8c5d36a081440ea7c55014bd5797010e89d`.
- Produces: a documented fresh `vistorybench_song_yuchen_exploratory_v2` donor/bundle/target workflow with an explicit 16-layer/64-slot artifact gate.

- [ ] **Step 1: Add the immutable-v1 recovery subsection to the Song runbook**

Directly after the Song pilot introduction in `utest/README.md`, add:

```markdown
#### Recovery after the layerwise base-inference repair

If `vistorybench_song_yuchen_exploratory_v1` already contains the flat one-layer donor,
retain it as failed evidence. Reuse its frozen selection by copying the whole portable
selection directory into a fresh no-clobber v2 root; do not copy `donor_run`,
`donor_bundle`, or `target_run`:

```bash
export SONG_V1="$PWD/runs/vistorybench_song_yuchen_exploratory_v1"
export SONG_ROOT="$PWD/runs/vistorybench_song_yuchen_exploratory_v2"
test -f "$SONG_V1/selection/selection.json"
test ! -e "$SONG_ROOT"
mkdir -p "$SONG_ROOT"
cp -a "$SONG_V1/selection" "$SONG_ROOT/selection"
test "$(sha256sum "$SONG_V1/selection/selection.json" | cut -d' ' -f1)" = \
     "$(sha256sum "$SONG_ROOT/selection/selection.json" | cut -d' ' -f1)"

python -m utest.vistory_donor_harness dry-run \
  --selection "$SONG_ROOT/selection/selection.json" \
  --output "$SONG_ROOT/donor_run" \
  --base-inference-args "$BASE_ARGS" \
  --platform-manifest "$PLATFORM_MANIFEST"

CUDA_VISIBLE_DEVICES=0 \
DUAL_EXPERT_LOAD_MODE=active \
DUAL_EXPERT_MANAGE_AUX_MODELS=1 \
python -m utest.vistory_donor_harness resume \
  --manifest "$SONG_ROOT/donor_run/run_manifest.json"
```

Before freezing the donor map, require the exact target-character bank-0 payload,
layers `0..15`, and 64 finite floating slots per layer:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

import torch

root = Path(os.environ["SONG_ROOT"])
run = json.loads((root / "donor_run/run_manifest.json").read_text())
assert len(run["jobs"]) == 1
job = run["jobs"][0]
artifact = torch.load(job["donor_payload"], map_location="cpu", weights_only=True)
key = f'{job["event"]["character_name"]}|0'
assert list(artifact["payloads"]) == [key]
payload = artifact["payloads"][key]
assert payload.get("__layerwise__") is True
layers = payload.get("layers")
assert isinstance(layers, dict)
assert set(layers) == {str(layer) for layer in range(16)}
hidden_dims = set()
for layer in range(16):
    tensor = layers[str(layer)]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 2 and tensor.shape[0] == 64
    assert tensor.is_floating_point() and torch.isfinite(tensor).all().item()
    hidden_dims.add(int(tensor.shape[1]))
assert len(hidden_dims) == 1
assert Path(job["completion"]).is_file()
print("READY: one bank, layers 0-15, 64 finite slots per layer")
PY
```

Continue with the existing donor-map freeze, target dry-run, prefix, qualification,
four-arm preflight, and generation commands using the v2 `SONG_ROOT`.
```

Ensure the outer Markdown fence is represented correctly in the file: use a four-backtick fence around the full excerpt or write the subsection directly so its inner Bash/Python fences render normally.

- [ ] **Step 2: Run syntax compilation for every changed Python module**

```bash
python -m py_compile \
  reference_inference_runtime.py \
  utest/input_contract.py \
  utest/vistory_donor_harness.py \
  utest/vistory_donor_bundle.py
```

Expected: exit code 0 with no output.

- [ ] **Step 3: Run the focused repair suite**

```bash
pytest -q \
  utest/tests/test_input_contract.py \
  utest/tests/test_inference_hotpath.py \
  utest/tests/test_vistory_donor_harness.py \
  utest/tests/test_vistory_donor_bundle.py \
  utest/tests/test_subject_reappearance_harness.py
```

Expected: all runnable tests pass and only pre-existing environment skips remain.

- [ ] **Step 4: Run the full CPU test suite**

```bash
pytest -q utest/tests
```

Expected: all runnable tests pass; record the exact pass/skip count in the implementation handoff.

- [ ] **Step 5: Review the diff for scope and unchanged semantics**

Run:

```bash
git diff --check
git diff --stat HEAD~3
git diff HEAD~3 -- \
  reference_inference_runtime.py \
  utest/input_contract.py \
  utest/vistory_donor_harness.py \
  utest/vistory_donor_bundle.py \
  utest/tests/test_input_contract.py \
  utest/tests/test_inference_hotpath.py \
  utest/tests/test_vistory_donor_harness.py \
  utest/tests/test_vistory_donor_bundle.py \
  utest/README.md
```

Confirm from the diff that there is no change to token-scoring/selection functions, `two_role_diff`, query overrides, MemoryEncoder internals, RoleWiseSlotMemoryBank, per-layer injection selection, arm definitions, seeds, or CLI parsing.

- [ ] **Step 6: Request independent review before the final commit**

Give the reviewer the approved design spec and the current diff. Require explicit answers to these checks:

1. Does the normal base path use only its existing conditional forward for all taps?
2. Is the query consumer still a single tensor from `sparse_role_memory_layer_idx`?
3. Does the memory consumer contain each configured layer's own tensor exactly once?
4. Can any flat/incomplete payload still receive or reuse completion?
5. Are all unrelated working-tree files untouched?

Address every finding with a focused regression test before proceeding.

- [ ] **Step 7: Commit documentation and any review-only correction**

```bash
git add utest/README.md
git commit -m "docs: add Song layerwise donor recovery gate"
```

If review required a code correction, commit that correction separately with its regression test before this documentation commit.

- [ ] **Step 8: Run the A100 Song v2 acceptance gate after deployment**

On `/data/long_term_data/shixiao/videomem/U-test-vistory-8f0b728`, pull the reviewed commits, execute the new recovery subsection, and stop immediately unless the payload gate prints:

```text
READY: one bank, layers 0-15, 64 finite slots per layer
```

Then run `freeze_vistory_donor_map.py`, build `target_run/run_manifest.json`, and run Song seed-0 preflight. Acceptance requires donor freeze to succeed and the four arms to be exactly `full_correct`, `no_memory`, `zero_path`, and `wrong_subject`.

---

## Final Acceptance Checklist

- [ ] Base inference registers the exact union of query and configured memory layers in one conditional forward.
- [ ] Query construction receives only `sparse_role_memory_layer_idx` and existing teacher-forced single-layer tests remain green.
- [ ] The existing selector applies unchanged indices to every layer's own raw feature tensor.
- [ ] Stored donor memory contains exactly string keys `"0"` through `"15"`, each shaped `[64, D]`, floating, finite, and sharing `D`.
- [ ] Existing layer-specific encoder calls and layer-matched injection remain unchanged.
- [ ] Flat and incomplete artifacts fail before completion is written and cannot reuse a stale completion.
- [ ] A valid artifact passes donor completion, bundle freeze, and target-harness compatibility.
- [ ] Song v1 is untouched and Song v2 passes the A100 geometry gate.
- [ ] No new dependency, CLI option, extra forward, or Song-specific runtime behavior exists.

