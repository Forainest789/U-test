# Fast Identity-Token Causal Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, single-load A100 probe that screens SlotMem content causality over layer/timestep cells and causally validates a small identity-bearing video-token set.

**Architecture:** Reuse the frozen Q* context and arm payload builders. Keep score, grouping, intervention, and gate arithmetic in one GPU-free module; add narrowly scoped teacher-forced controls and per-token diagnostics to the existing inference path; schedule S0-S2 from one orchestrator and expose it through one strict server runner. Normal rollout behavior is unchanged unless a `teacher_forced_probe` option is present.

**Tech Stack:** Python 3.10+, PyTorch, pytest, Bash, existing DiffSynth/Wan2.2 runtime, BF16, FlashAttention 2, matplotlib already present in the repository environment.

## Global Constraints

- Target server is one NVIDIA A100 80GB; timed forwards use BF16 and `torch.inference_mode()`.
- Set `DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_2` and `SLOTMEM_OFFLOAD_MODELS=0`; fail closed when FA2 is not selected unless `--allow-attention-fallback` is explicit.
- Do not change SlotMem weights, sampler math, training configuration, or normal rollout behavior.
- S0-S2 must never VAE-decode generated frames or call `scheduler.step`.
- Freeze and hash prefix, target, prompt, embeddings, latent, noise, timestep, payloads, model, dtype, and attention backend.
- S0+S1 uses about 25 measured forwards; S0-S2 stops at about 50 measured forwards plus diagnostic captures.
- Correct memory must beat no memory and matched-wrong memory before identity labels are allowed.
- Attention alone never emits `identity-core candidate`; group and equal-budget causal controls are mandatory.
- Add no dependency and preserve all unrelated dirty-worktree files.

## File Map

- Create `utest/identity_token_scoring.py`: pure score equations, deterministic grouping, matched-budget masks, estimands, labels, and report gates.
- Create `utest/identity_token_probe.py`: CLI, frozen run schedule, single-load GPU orchestration, semantic captures, interventions, records, figures, and summary report.
- Modify `train_slotmem.py`: opt-in per-query sparse-memory diagnostics only.
- Modify `infer_slotmem.py`: pass opt-in capture/query controls through the memory-aware forward and expose effective attention backend.
- Modify `reference_inference_runtime.py`: teacher-forced context-position neutralization, query override, semantic capture return, and no-step enforcement.
- Modify `utest/qstar_probe.py`: allow the identity orchestrator to load the frozen Q* context without paying for native Wan reference forwards.
- Create `scripts/run_slotmem_identity_probe.sh`: strict A100 entry, preflight, dry-run, command/environment manifest, and optional S3.
- Create `utest/tests/test_identity_token_scoring.py`: GPU-free score/group/mask/gate checks.
- Create `utest/tests/test_identity_token_probe.py`: schedule, selection, cache, output, and fail-closed orchestration checks with a fake predictor.
- Modify `utest/tests/test_inference_hotpath.py`: opt-in runtime controls and normal-path non-regression.
- Create `utest/tests/test_identity_token_runner.py`: shell dry-run and manifest checks.
- Modify `utest/README.md`: exact A100 smoke/full commands, outputs, gates, and interpretation boundary.

---

### Task 1: Pure token scores and causal gates

**Files:**
- Create: `utest/identity_token_scoring.py`
- Create: `utest/tests/test_identity_token_scoring.py`

**Interfaces:**
- Produces: `percentile_rank(values: Sequence[float]) -> list[float]`
- Produces: `score_token_channels(rows: Sequence[Mapping]) -> list[dict]`
- Produces: `causal_metrics(losses: Mapping[str, float], epsilon: float = 1e-12) -> dict`
- Produces: `classify_token(row: Mapping, *, repeat_margin: float, benefit_margin: float, validation_direction: bool) -> list[str]`
- Consumes: finite scalar diagnostic channels and loss dictionaries; no torch/model objects.

- [ ] **Step 1: Write the failing score tests**

```python
from __future__ import annotations

import math
import pytest

from utest.identity_token_scoring import (
    causal_metrics,
    classify_token,
    percentile_rank,
    score_token_channels,
)


def test_percentile_rank_is_deterministic_for_ties_and_singletons() -> None:
    assert percentile_rank([3.0]) == [0.5]
    assert percentile_rank([1.0, 1.0, 3.0]) == pytest.approx([0.25, 0.25, 1.0])
    with pytest.raises(ValueError, match="finite"):
        percentile_rank([0.0, math.inf])


def test_identity_and_action_scores_follow_frozen_equations() -> None:
    rows = score_token_channels([
        {
            "flat_idx": 4,
            "name_raw": 9.0,
            "attribute_raw": 8.0,
            "persistence_raw_margin": 7.0,
            "persistence_read_margin": 6.0,
            "action_attention_raw": 6.0,
            "action_hidden_raw": 5.0,
            "scene_hidden_raw": 1.0,
            "random_hidden_raw": 2.0,
            "scene_raw": 3.0,
        },
        {
            "flat_idx": 7,
            "name_raw": 1.0,
            "attribute_raw": 2.0,
            "persistence_raw_margin": 3.0,
            "persistence_read_margin": 2.0,
            "action_attention_raw": 1.0,
            "action_hidden_raw": 1.0,
            "scene_hidden_raw": 1.0,
            "random_hidden_raw": 1.0,
            "scene_raw": 8.0,
        },
    ])
    assert rows[0]["s_pre"] == pytest.approx(1.0)
    assert rows[0]["action_hidden_net_raw"] == pytest.approx(3.0)
    assert rows[0]["s_action"] > rows[1]["s_action"]


def test_causal_metrics_and_identity_label_require_correct_content() -> None:
    metrics = causal_metrics({
        "no_memory": 1.0,
        "full_correct": 0.5,
        "identity_only": 0.6,
        "drop_identity": 0.9,
        "drop_random": 0.55,
        "drop_low": 0.52,
        "wrong_identity": 0.85,
    })
    assert metrics["r_keep"] == pytest.approx(0.8)
    assert metrics["r_drop"] == pytest.approx(0.8)
    row = {
        "s_name": 0.9,
        "s_attr": 0.8,
        "s_persist": 0.85,
        "s_action": 0.2,
        "s_scene": 0.1,
        "group_causal_score": 0.7,
        "group_control_floor": 0.1,
        "content_delta": 0.25,
    }
    assert "identity-core candidate" in classify_token(
        row, repeat_margin=0.01, benefit_margin=0.01, validation_direction=True
    )
    assert "identity-core candidate" not in classify_token(
        row, repeat_margin=0.01, benefit_margin=0.3, validation_direction=True
    )
```

- [ ] **Step 2: Run the focused test and verify red**

Run: `python -m pytest utest/tests/test_identity_token_scoring.py -q`

Expected: collection fails because `utest.identity_token_scoring` does not exist.

- [ ] **Step 3: Implement the exact score and gate functions**

```python
def percentile_rank(values: Sequence[float]) -> list[float]:
    finite = [_finite(value, "rank value") for value in values]
    if len(finite) == 1:
        return [0.5]
    order = sorted(range(len(finite)), key=lambda index: (finite[index], index))
    output = [0.0] * len(finite)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and finite[order[end]] == finite[order[start]]:
            end += 1
        average_zero_based_rank = 0.5 * (start + end - 1)
        value = average_zero_based_rank / float(len(order) - 1)
        for position in range(start, end):
            output[order[position]] = value
        start = end
    return output
```

`score_token_channels` computes channel ranks within the supplied cell, `s_persist = 0.5 * (rank(persistence_raw_margin) + rank(persistence_read_margin))`, `action_hidden_net_raw = max(0, action_hidden_raw - max(scene_hidden_raw, random_hidden_raw))`, `s_action = sqrt(rank(action_attention_raw) * rank(action_hidden_net_raw))`, and `s_pre = cbrt(s_name * s_attr * s_persist)`. `causal_metrics` computes `B_full`, `R_keep`, and `R_drop` and rejects non-positive denominators. `classify_token` implements every conjunctive threshold in design section 10.3 and returns multi-label strings.

- [ ] **Step 4: Run score tests green**

Run: `python -m pytest utest/tests/test_identity_token_scoring.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add utest/identity_token_scoring.py utest/tests/test_identity_token_scoring.py
git commit -m "Add identity token score equations"
```

---

### Task 2: Deterministic groups and equal-budget interventions

**Files:**
- Modify: `utest/identity_token_scoring.py`
- Modify: `utest/tests/test_identity_token_scoring.py`

**Interfaces:**
- Produces: `flat_to_coord(flat_idx: int, height: int, width: int) -> tuple[int, int, int]`
- Produces: `build_candidate_groups(indices: Sequence[int], *, height: int, width: int, max_groups: int = 8, min_group_size: int = 4) -> list[dict]`
- Produces: `build_intervention_masks(original: Sequence[int], universe: Sequence[int], scores: Mapping[int, float], *, budget_fraction: float, seed: int, height: int, width: int) -> dict[str, list[int]]`
- Produces: `group_membership_sha256(groups: Sequence[Mapping]) -> str`
- Consumes: integer flattened token positions and pre-causal scores from Task 1.

- [ ] **Step 1: Add failing grouping and mask tests**

```python
from utest.identity_token_scoring import (
    build_candidate_groups,
    build_intervention_masks,
    group_membership_sha256,
)


def test_grouping_is_coordinate_only_bounded_and_deterministic() -> None:
    indices = [0, 1, 4, 5, 16, 17, 20, 21, 32, 33, 36, 37]
    first = build_candidate_groups(indices, height=4, width=4, max_groups=3, min_group_size=4)
    second = build_candidate_groups(list(reversed(indices)), height=4, width=4, max_groups=3, min_group_size=4)
    assert first == second
    assert 1 <= len(first) <= 3
    assert sorted(index for group in first for index in group["indices"]) == sorted(indices)
    assert all(len(group["indices"]) >= 4 for group in first)
    assert group_membership_sha256(first) == group_membership_sha256(second)


def test_interventions_match_count_and_per_frame_histogram() -> None:
    original = [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51]
    universe = original + [4, 20, 36, 52]
    scores = {index: float(index % 16) for index in universe}
    masks = build_intervention_masks(
        original, universe, scores, budget_fraction=0.25, seed=7, height=4, width=4
    )
    keep_names = ["identity_top", "random", "low_score", "wrong_identity"]
    assert {len(masks[name]) for name in keep_names} == {4}
    histogram = lambda values: [sum(index // 16 == frame for index in values) for frame in range(4)]
    assert {tuple(histogram(masks[name])) for name in keep_names} == {
        tuple(histogram(masks["identity_top"]))
    }
    assert all(len(universe) - len(masks[name]) == 4 for name in ("drop_identity", "drop_random", "drop_low"))
```

- [ ] **Step 2: Run the two new tests and verify red**

Run: `python -m pytest utest/tests/test_identity_token_scoring.py -q`

Expected: import errors for the four new functions.

- [ ] **Step 3: Implement coordinate-only grouping and strict matching**

Use eight-neighborhood edges in one frame and a one-cell neighborhood in adjacent frames. Merge undersized components by nearest normalized centroid, merge the smallest component while more than `max_groups` remain, then median-split the widest normalized coordinate while the child sizes remain at least `min_group_size`. Resolve ties lexicographically by `(t,h,w)`.

```python
def flat_to_coord(flat_idx: int, height: int, width: int) -> tuple[int, int, int]:
    spatial = int(height) * int(width)
    if flat_idx < 0 or spatial <= 0:
        raise ValueError("invalid flattened coordinate")
    return flat_idx // spatial, (flat_idx % spatial) // width, flat_idx % width


def _frame_histogram(indices: Sequence[int], spatial: int) -> dict[int, int]:
    output: dict[int, int] = {}
    for index in indices:
        frame = int(index) // spatial
        output[frame] = output.get(frame, 0) + 1
    return output
```

For interventions set `K = max(4, ceil(0.25 * N))`, reject `N < 4`, select identity top-K deterministically by `(-score, flat_idx)`, and sample random/low controls separately inside each identity frame bucket. `drop_*` is `U` minus the corresponding K-set. `wrong_identity` reuses exactly the identity positions.

- [ ] **Step 4: Run all pure tests green**

Run: `python -m pytest utest/tests/test_identity_token_scoring.py -q`

Expected: all tests pass without torch.

- [ ] **Step 5: Commit Task 2**

```bash
git add utest/identity_token_scoring.py utest/tests/test_identity_token_scoring.py
git commit -m "Add deterministic identity token interventions"
```

---

### Task 3: Opt-in per-token sparse-memory diagnostics

**Files:**
- Modify: `train_slotmem.py:1498-2060`
- Modify: `infer_slotmem.py:2680-3025`
- Modify: `utest/tests/test_inference_hotpath.py`

**Interfaces:**
- Produces: `CharacterWiseCrossAttention.forward_sparse(..., capture_token_diagnostics: bool = False)` with `debug["token_diagnostics"]` only when enabled.
- Produces per enabled layer: `flat_idx`, `host_norm`, `raw_delta_norm`, `effective_delta_norm`, `raw_cosine_max`, `read_logsumexp`, plus BF16 `host_features`, `raw_delta_features`, and `effective_delta_features` tensors on CPU.
- Consumes: existing selected query indices, host tokens, memory logits, scaled deltas, and applied layer scale.

- [ ] **Step 1: Add a failing CPU unit test for diagnostic shape and opt-in behavior**

```python
def test_sparse_attention_token_diagnostics_are_opt_in_and_aligned() -> None:
    torch = pytest.importorskip("torch")
    cls = _load("train_slotmem", "CharacterWiseCrossAttention")
    module = cls(dim=8, num_heads=2, head_dim=4, rope_dim=0, time_gate=False)
    tokens = torch.randn(1, 6, 8)
    memory = torch.randn(1, 3, 8)
    payload = {"Mara": {"flat_idx": torch.tensor([1, 4])}}
    meta = [{"char_id": "Mara"} for _ in range(3)]

    _, quiet = module.forward_sparse(
        tokens, memory, query_feature_payload=payload,
        memory_bank_token_meta=meta, latent_h=2, latent_w=3,
    )
    _, captured = module.forward_sparse(
        tokens, memory, query_feature_payload=payload,
        memory_bank_token_meta=meta, latent_h=2, latent_w=3,
        capture_token_diagnostics=True,
    )
    assert "token_diagnostics" not in quiet
    diag = captured["token_diagnostics"]
    assert diag["flat_idx"].tolist() == [1, 4]
    assert diag["host_features"].shape == (2, 8)
    assert diag["raw_delta_features"].shape == (2, 8)
    assert diag["effective_delta_features"].shape == (2, 8)
    assert all(diag[name].shape == (2,) for name in (
        "host_norm", "raw_delta_norm", "raw_cosine_max", "read_logsumexp"
    ))
```

- [ ] **Step 2: Run the focused test and verify red**

Run: `python -m pytest utest/tests/test_inference_hotpath.py::test_sparse_attention_token_diagnostics_are_opt_in_and_aligned -q`

Expected: `token_diagnostics` is absent in the enabled call.

- [ ] **Step 3: Capture aligned tensors without changing the default path**

Inside `forward_sparse`, allocate capture tensors only when the flag is true. For every role/query chunk compute `raw_cosine_max` against that role's pre-q/k memory tokens and `read_lse = logsumexp(attn_logits, dim=-1).mean(dim=1)`. Aggregate role overlap with the same winner mask already used for `out_selected`, then emit CPU tensors after the final selected-union update:

```python
if capture_token_diagnostics:
    raw_delta = out_selected.detach().float()
    host = tokens.index_select(1, selected_union).detach().float()
    debug["token_diagnostics"] = {
        "flat_idx": selected_union.detach().cpu(),
        "host_features": host[0].to(device="cpu", dtype=torch.bfloat16),
        "raw_delta_features": raw_delta[0].to(device="cpu", dtype=torch.bfloat16),
        "host_norm": host.norm(dim=-1)[0].cpu(),
        "raw_delta_norm": raw_delta.norm(dim=-1)[0].cpu(),
        "raw_cosine_max": raw_cosine_selected.detach().float()[0].cpu(),
        "read_logsumexp": read_lse_selected.detach().float()[0].cpu(),
    }
```

In `_memory_aware_dit_forward`, read `capture_sparse_token_diagnostics=False` from kwargs, pass it into the sparse module, and after layer scaling add `effective_delta_norm_by_token` plus `effective_delta_features = raw_delta_features * total_layer_scale`. The default result schema remains scalar-only.

- [ ] **Step 4: Run hot-path tests**

Run: `python -m pytest utest/tests/test_inference_hotpath.py -q`

Expected: the new diagnostic test and every existing hot-path invariant pass or skip for a documented missing local torch runtime.

- [ ] **Step 5: Commit Task 3**

```bash
git add train_slotmem.py infer_slotmem.py utest/tests/test_inference_hotpath.py
git commit -m "Expose opt-in sparse token diagnostics"
```

---

### Task 4: Teacher-forced semantic capture and fixed query/text interventions

**Files:**
- Modify: `infer_slotmem.py:1600-1910`
- Modify: `reference_inference_runtime.py:1080-1380`
- Modify: `utest/tests/test_inference_hotpath.py`

**Interfaces:**
- Consumes `teacher_forced_probe` keys:
  - `query_indices_by_role: Mapping[str, Sequence[int]] | None`
  - `context_zero_indices: Sequence[int] | None`
  - `semantic_role_ids: Sequence[str] | None`
  - `capture_sparse_token_diagnostics: bool`
- Consumes opt-in `generate_chunk(..., query_indices_by_role: Mapping[str, Sequence[int]] | None = None)` for decoded S3; when absent, normal rollout behavior is byte-for-byte on the existing branch.
- Produces teacher-forced result keys: `query_feature_payload`, `semantic_attention_maps`, `sparse_role_memory_stats_by_layer`, and `attention_implementation`.
- Preserves sequence length and position IDs by replacing selected conditional context rows with zeros; unconditional context and every frozen latent/input remain unchanged.

- [ ] **Step 1: Add failing tests for query override and context neutralization**

```python
def test_teacher_forced_controls_preserve_layout_and_override_only_flat_indices() -> None:
    torch = pytest.importorskip("torch")
    engine_cls = _load("infer_slotmem", "SlotMemInferenceEngine")
    engine = object.__new__(engine_cls)
    payload = {"Mara": {"flat_idx": torch.tensor([1, 2]), "feature": "keep"}}
    overridden = engine._override_query_indices(payload, {"Mara": [3, 5]}, num_tokens=8)
    assert overridden["Mara"]["flat_idx"].tolist() == [3, 5]
    assert overridden["Mara"]["feature"] == "keep"
    with pytest.raises(ValueError, match="outside"):
        engine._override_query_indices(payload, {"Mara": [8]}, num_tokens=8)


def test_context_zeroing_clones_and_keeps_shape() -> None:
    torch = pytest.importorskip("torch")
    zero = _load("infer_slotmem", "_zero_context_positions")
    context = torch.ones(1, 6, 4)
    output = zero(context, [1, 4])
    assert output.shape == context.shape
    assert torch.all(output[:, [1, 4]] == 0)
    assert torch.all(context == 1)
```

Add a source invariant asserting `scheduler.step` occurs only after the teacher-forced return and that these keys are read only inside the teacher-forced branch.

- [ ] **Step 2: Run focused controls and verify red**

Run: `python -m pytest utest/tests/test_inference_hotpath.py -q`

Expected: `_override_query_indices` and `_zero_context_positions` do not exist.

- [ ] **Step 3: Implement strict controls in the existing prepass**

```python
def _zero_context_positions(context, indices):
    if not indices:
        return context
    selected = sorted({int(index) for index in indices})
    if selected[0] < 0 or selected[-1] >= int(context.shape[1]):
        raise ValueError("context_zero_indices outside encoded prompt")
    output = context.clone()
    output[:, selected] = 0
    return output
```

After the current semantic prepass builds `current_query_feature_payload`, apply `_override_query_indices` before `_memory_aware_dit_forward`. The same override is applied per denoising step only when the opt-in `generate_chunk` keyword is present, enabling S3 without changing the default rollout. Pass the capture flag through. For semantic groups, reuse `_prepare_character_semantic_probe_configs` and `MultiCharacterAttentionMapExtractor`; return CPU float maps keyed by exact role/group string. Clear hooks in `finally`. Read the actual backend from `diffsynth.core.attention.attention.ATTENTION_IMPLEMENTATION`, record the requested environment value separately, and fail if they differ outside the explicit fallback mode.

- [ ] **Step 4: Verify normal rollout has no new branch cost**

Run: `python -m pytest utest/tests/test_inference_hotpath.py -q`

Expected: tests prove controls are absent when `teacher_forced_probe is None`, the measured branch returns before `scheduler.step`, hooks are removed, and all existing tests pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add infer_slotmem.py reference_inference_runtime.py utest/tests/test_inference_hotpath.py
git commit -m "Add teacher-forced token intervention controls"
```

---

### Task 5: Frozen screening schedule and single-load orchestrator

**Files:**
- Create: `utest/identity_token_probe.py`
- Create: `utest/tests/test_identity_token_probe.py`
- Modify: `utest/qstar_probe.py:323-509`

**Interfaces:**
- Produces: `ScreeningRun(stage: str, timestep_index: int, layer_group: tuple[int, ...], arm: str)`
- Produces: `build_screening_schedule(timesteps=(0,25,49), layer_groups=((0,1,2,3,4),(5,6,7,8,9,10),(11,12,13,14,15))) -> list[ScreeningRun]`
- Produces: `select_cells(records: Sequence[Mapping], *, repeat_margin: float, influence_floor: float) -> dict`
- Produces: `run_probe(args, *, context_loader=_load_probe_context) -> dict`
- Changes: `_load_probe_context(args, *, include_native: bool = True)`; existing Q* caller retains the default, identity probe passes `False`.

- [ ] **Step 1: Write failing schedule and cell-selection tests**

```python
from utest.identity_token_probe import build_screening_schedule, select_cells


def test_s0_s1_schedule_has_25_unique_measured_forwards() -> None:
    schedule = build_screening_schedule()
    keys = {(row.timestep_index, row.layer_group, row.arm) for row in schedule}
    assert len(schedule) == len(keys) == 25
    assert sum(row.arm == "no_memory" for row in schedule) == 3
    assert any(row.arm == "correct_repeat" and row.timestep_index == 25 for row in schedule)
    assert any(row.arm == "wrong" and len(row.layer_group) == 16 for row in schedule)


def test_cell_selection_prioritizes_positive_content_delta() -> None:
    selected = select_cells([
        {"timestep_index": 0, "layer_group": [0,1,2,3,4], "q_content": -0.1, "delta_host_ratio": 9.0},
        {"timestep_index": 25, "layer_group": [5,6,7,8,9,10], "q_content": 0.2, "delta_host_ratio": 0.1},
        {"timestep_index": 49, "layer_group": [11,12,13,14,15], "q_content": 0.1, "delta_host_ratio": 0.5},
    ], repeat_margin=0.01, influence_floor=0.0)
    assert selected["primary"]["timestep_index"] == 25
    assert selected["validation"]["timestep_index"] == 49
```

- [ ] **Step 2: Run tests and verify red**

Run: `python -m pytest utest/tests/test_identity_token_probe.py -q`

Expected: module import fails.

- [ ] **Step 3: Implement the exact reusable schedule**

The 25 unique runs are:

- S0 middle/all: `correct`, `correct_repeat`, `zero`, `no_memory` = 4.
- S1 `no_memory` at indices 0 and 49; middle reuses S0 = 2.
- S1 `correct/wrong` for 3 timesteps x 3 layer groups = 18.
- S1 middle/all `wrong`; middle/all `correct` reuses S0 = 1.

```python
@dataclass(frozen=True)
class ScreeningRun:
    stage: str
    timestep_index: int
    layer_group: tuple[int, ...]
    arm: str


ALL_LAYERS = tuple(range(16))
DEFAULT_GROUPS = (tuple(range(0, 5)), tuple(range(5, 11)), tuple(range(11, 16)))
```

The orchestrator calls `_load_probe_context(args, include_native=False)` once, wraps every model call in `torch.inference_mode()`, mutates only `engine.sparse_role_memory_injection_layers` between frozen calls, and restores it in `finally`. Compute loss with `qstar_probe._mse`, hashes with `tensor_sha256`, `Q_presence = L_no - L_correct`, and `Q_content = L_wrong - L_correct`. Measure CUDA time with events after one warm-up and record peak allocated/reserved VRAM.

- [ ] **Step 4: Add a fake-context test proving one load, no decode, and fail-closed injection**

```python
def test_orchestrator_loads_once_and_stops_before_s2_without_content_signal(tmp_path) -> None:
    calls = {"loads": 0, "forwards": 0}

    class FakeEngine:
        device = "cpu"
        sparse_role_memory_injection_layers = list(range(16))

        def generate_chunk(self, **kwargs):
            calls["forwards"] += 1
            assert kwargs["teacher_forced_probe"]["force_memory_path"] is True
            return {
                "prediction_cond": [0.0],
                "prediction": [0.0],
                "sparse_role_memory_stats_by_layer": {
                    "0": {"enabled": 1, "selected_memory_tokens": 4, "effective_delta_norm": 0.1}
                },
                "attention_implementation": "flash_attention_2",
            }

    def load(args, include_native):
        calls["loads"] += 1
        assert include_native is False
        return make_fake_identity_context(FakeEngine())

    report = run_probe(make_args(tmp_path), context_loader=load)
    assert calls == {"loads": 1, "forwards": 25}
    assert report["gates"]["content_causality"]["status"] == "BLOCK"
    assert report["forward_count"] == 25
```

Define `make_fake_identity_context` and `make_args` in the test with one scalar flow target, frozen payload dictionaries for correct/zero/no/wrong, and no VAE/scheduler methods. The fake deliberately makes every arm equal so S2 must not run.

- [ ] **Step 5: Run orchestrator and Q* regressions**

Run: `python -m pytest utest/tests/test_identity_token_probe.py utest/tests/test_qstar.py -q`

Expected: all tests pass and existing production Q* still computes native predictions by default.

- [ ] **Step 6: Commit Task 5**

```bash
git add utest/identity_token_probe.py utest/qstar_probe.py utest/tests/test_identity_token_probe.py
git commit -m "Add identity causal screening orchestrator"
```

---

### Task 6: S2 semantic proposal, group knockouts, and equal-budget confirmation

**Files:**
- Modify: `utest/identity_token_probe.py`
- Modify: `utest/tests/test_identity_token_probe.py`

**Interfaces:**
- Produces: `semantic_group_manifest(story: Mapping, event: Mapping) -> dict[str, list[str]]`
- Produces: `build_token_rows(full, wrong, name_drop, diagnostic, action_drop, scene_drop, random_drop) -> list[dict]`
- Produces: `run_s2(context, selected_cells, screening_records, args) -> dict`
- Consumes: Task 3 per-layer token diagnostics, Task 4 semantic maps, Task 1 scores, and Task 2 groups/masks.

- [ ] **Step 1: Add failing semantic-manifest and S2 budget tests**

```python
from utest.identity_token_probe import semantic_group_manifest


def test_delta8_semantic_manifest_separates_identity_action_and_scene() -> None:
    story = json.loads((ROOT / "utest/events/person_reappearance_delta8_story.json").read_text())
    manifest = semantic_group_manifest(story, {"character_name": "Mara", "target_chunk_idx": 8})
    assert manifest["identity_name"] == ["Mara"]
    assert len(manifest["stable_attributes"]) == 4
    assert manifest["action_core"] == ["runs", "two steps", "catches", "closing", "looks up", "toward camera"]
    assert set(manifest["scene"]) == {"platform", "tram", "rain", "commuters", "dusk"}


def test_s2_never_exceeds_declared_measured_budget() -> None:
    assert s2_forward_budget(max_groups=8, has_validation=True) <= 25
    assert 25 + s2_forward_budget(max_groups=8, has_validation=True) <= 50
```

- [ ] **Step 2: Run focused S2 tests and verify red**

Run: `python -m pytest utest/tests/test_identity_token_probe.py -q`

Expected: semantic manifest and S2 functions are missing.

- [ ] **Step 3: Implement the fixed diagnostic capture sequence**

At the primary cell run:

1. Recompute expanded-universe `full_correct`, `wrong`, and `no_memory` once.
2. Run normal name capture and a branch with only Mara token positions in `context_zero_indices`.
3. Run one diagnostic prompt containing all four stable attributes, action groups, and scene groups with `semantic_role_ids` equal to those exact phrases.
4. Run action-core, scene, and seeded equal-count random text-position neutralizations.
5. Run at most eight `drop_group` arms; when a validation cell exists, cap primary-cell group knockouts at seven so all measured S2 arms remain within 25.
6. Build frozen masks and run `identity_only`, `random`, `low_score`, `drop_identity`, `drop_random`, `drop_low`, and `wrong_identity`.
7. In the validation cell run the expanded-universe full baseline plus final identity keep/drop/wrong confirmation. Reuse its S1 no-memory result because no query memory is injected in that arm.

The code asserts the cumulative measured count is at most 50 including S0/S1. Semantic-only captures are labeled diagnostic and excluded from Q* estimands.

- [ ] **Step 4: Implement raw channel alignment and labels**

Join records on `(timestep_index, layer_idx, flat_idx)`. Compute:

```python
name_raw = max(0.0, normal_name_attention - dropped_name_attention)
attribute_raw = median(attribute_attention[group] for group in valid_attributes)
persistence_raw_margin = raw_cosine_correct - raw_cosine_wrong
persistence_read_margin = read_logsumexp_correct - read_logsumexp_wrong
content_delta = norm(effective_delta_features_correct - effective_delta_features_wrong) / max(host_norm, 1e-12)
action_hidden_raw = norm(host_full - host_drop_action) / max(norm(host_full), 1e-12)
```

Reject any missing/non-finite join, fewer than three valid stable attributes, candidate/original count below four, count/frame mismatch, non-positive full benefit, or payload arm without measured injection. Apply `score_token_channels`, `build_candidate_groups`, group causal score, `build_intervention_masks`, `causal_metrics`, and `classify_token`.

- [ ] **Step 5: Add a deterministic fake S2 pass test**

Use a fake predictor whose losses satisfy `full_correct=0.5`, `no_memory=1.0`, `wrong=0.9`, `identity_only=0.6`, `drop_identity=0.9`, `drop_random=0.55`, `drop_low=0.52`, and `wrong_identity=0.85`. Return aligned four-token semantic and persistence diagnostics. Assert:

```python
assert report["gates"]["identity_set"]["status"] == "PASS"
assert report["metrics"]["r_keep"] >= 0.8
assert any("identity-core candidate" in row["labels"] for row in report["token_rows"])
assert report["measured_forward_count"] <= 50
```

- [ ] **Step 6: Run all identity tests**

Run: `python -m pytest utest/tests/test_identity_token_scoring.py utest/tests/test_identity_token_probe.py -q`

Expected: pass with deterministic records and no GPU.

- [ ] **Step 7: Commit Task 6**

```bash
git add utest/identity_token_probe.py utest/tests/test_identity_token_probe.py
git commit -m "Add causal identity token confirmation"
```

---

### Task 7: Versioned records, cache keys, figures, and self-check

**Files:**
- Modify: `utest/identity_token_probe.py`
- Modify: `utest/tests/test_identity_token_probe.py`

**Interfaces:**
- Produces: `cache_key(kind: str, inputs: Mapping) -> str`
- Produces: `write_outputs(output: Path, result: Mapping) -> None`
- Produces CLI: `python -m utest.identity_token_probe --self-check` and production options from the design spec.
- Writes: `runtime_manifest.json`, `input_contract.json`, `screening_cells.jsonl`, `selected_cells.json`, `diagnostic_prompt_manifest.json`, `token_scores.jsonl`, `token_groups.json`, `interventions.jsonl`, `identity_probe_report.json`, and three PNG figures.

- [ ] **Step 1: Add failing cache/output tests**

```python
def test_cache_key_changes_for_every_frozen_boundary() -> None:
    base = {"prefix": "a", "prompt": "b", "timestep": 25, "layers": [5,6], "backend": "fa2"}
    original = cache_key("cell", base)
    for key, value in (("prompt", "c"), ("timestep", 49), ("layers", [7]), ("backend", "sdpa")):
        changed = dict(base)
        changed[key] = value
        assert cache_key("cell", changed) != original


def test_output_report_is_complete_without_reading_figures(tmp_path) -> None:
    write_outputs(tmp_path, complete_fake_result())
    expected = {
        "runtime_manifest.json", "input_contract.json", "screening_cells.jsonl",
        "selected_cells.json", "diagnostic_prompt_manifest.json", "token_scores.jsonl",
        "token_groups.json", "interventions.jsonl", "identity_probe_report.json",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    report = json.loads((tmp_path / "identity_probe_report.json").read_text())
    assert report["gates"] and report["timing"] and report["forward_count"] <= 50
```

- [ ] **Step 2: Run tests and verify red**

Run: `python -m pytest utest/tests/test_identity_token_probe.py -q`

Expected: cache/output functions are missing.

- [ ] **Step 3: Implement atomic JSON/JSONL outputs and figures**

Use canonical JSON SHA256 cache keys. Write to `path.with_suffix(path.suffix + ".tmp")`, then `replace`. Convert tensors to CPU scalar/list form before serialization. Make the report contain PASS/BLOCK/PENDING plus reasons, thresholds, exact forward counts, cache hits, per-stage timing, peak VRAM, attention backend, git commit, argv, package versions, and input hashes.

Use matplotlib only for:

- `layer_timestep_qcontent.png`: 3x3 Q-content heatmap plus all-layer middle reference;
- `token_type_maps.png`: identity/action/scene ranked maps by latent frame;
- `group_causal_map.png`: group ID and `C_g` maps.

If figure rendering fails, set `figures.status=BLOCK` and retain the complete numeric report; do not mark the scientific gate failed solely because plotting failed.

- [ ] **Step 4: Implement parser and self-check**

Parser options exactly include:

```text
--prefix --future-target-video --arms-root --donor --donor-manifest --output
--timestep-indices 0,25,49 --layer-groups 0-4,5-10,11-15
--max-groups 8 --identity-budget 0.25 --noise-seed 0
--repeat-loss-tolerance 0 --repeat-influence-tolerance 0
--benefit-margin 0 --influence-floor 0
--allow-attention-fallback --run-decoded-validation --self-check
```

`--self-check` runs score, grouping, schedule, budget, and report assertions without importing torch.

- [ ] **Step 5: Run self-check and identity tests**

Run: `python -m utest.identity_token_probe --self-check`

Run: `python -m pytest utest/tests/test_identity_token_scoring.py utest/tests/test_identity_token_probe.py -q`

Expected: both commands pass.

- [ ] **Step 6: Commit Task 7**

```bash
git add utest/identity_token_probe.py utest/tests/test_identity_token_probe.py
git commit -m "Write versioned identity probe reports"
```

---

### Task 8: Strict A100 server runner and optional decoded validation

**Files:**
- Create: `scripts/run_slotmem_identity_probe.sh`
- Create: `utest/tests/test_identity_token_runner.py`
- Modify: `utest/README.md`

**Interfaces:**
- Consumes required environment variables from `run_slotmem_qstar_event.sh`: `EVENT_JSON`, `FUTURE_TARGET_VIDEO`, `FUTURE_TARGET_MANIFEST`, `BASE_INFERENCE_ARGS`, `PLATFORM_MANIFEST`, `DONOR_PAYLOAD`, `DONOR_MANIFEST`, `EVENT_RUN_ROOT`.
- Consumes optional: `IDENTITY_SMOKE=0`, `RUN_DECODED_VALIDATION=0`, `DRY_RUN=0`, `ALLOW_DIRTY_SOURCE=0`, `ALLOW_ATTENTION_FALLBACK=0`, `PYTHON_BIN=python3`, `UTEST_ENV`.
- Produces: fresh `${EVENT_RUN_ROOT}/identity_probe`, `.commands.jsonl` during execution, and final `${EVENT_RUN_ROOT}/run_manifest.json`.

- [ ] **Step 1: Write a failing dry-run test**

```python
def test_identity_runner_dry_run_records_fast_a100_chain(tmp_path: Path) -> None:
    bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    if not bash.is_file():
        pytest.skip("Git Bash is not installed")
    env = make_runner_environment(tmp_path)
    env.update({"DRY_RUN": "1", "ALLOW_DIRTY_SOURCE": "1", "PYTHON_BIN": "python"})
    completed = subprocess.run(
        [str(bash), "scripts/run_slotmem_identity_probe.sh"],
        cwd=ROOT, env=env, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    manifest = json.loads((Path(env["EVENT_RUN_ROOT"]) / "run_manifest.json").read_text())
    assert [row["name"] for row in manifest["commands"]] == [
        "identity-self-check", "input-contract-preflight", "prepare-prefix", "identity-probe"
    ]
    probe = manifest["commands"][-1]["argv"]
    assert probe[probe.index("--timestep-indices") + 1] == "0,25,49"
    assert manifest["environment"]["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] == "flash_attention_2"
```

- [ ] **Step 2: Run runner test and verify red**

Run: `python -m pytest utest/tests/test_identity_token_runner.py -q`

Expected: shell script does not exist.

- [ ] **Step 3: Implement the strict runner by reusing Q* shell patterns**

The script uses `set -euo pipefail`, `normalize_path`, `record_command`, `run_step`, and `finalize_manifest` behavior from the existing strict runner. It requires fresh output, rejects a dirty source unless explicitly allowed, activates the named conda environment, and exports:

```bash
export DIFFSYNTH_ATTENTION_IMPLEMENTATION="flash_attention_2"
export SLOTMEM_OFFLOAD_MODELS="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

For a real run, preflight with Python asserts CUDA is available, device name contains `A100`, total memory is at least 75 GiB, BF16 is supported, `flash_attn` imports, and the runtime selector reports FA2. `DRY_RUN=1` records but skips hardware checks and commands.

Run stages in this order: identity self-check, existing input contract, existing prefix preparation with timesteps `0,25,49`, then `python -m utest.identity_token_probe`. If `RUN_DECODED_VALIDATION=1`, pass `--run-decoded-validation`; after S2 PASS the orchestrator reuses its loaded engine, frozen prefix references, seed, and payloads for `full_correct,no_memory,identity_only,drop_identity`. It passes the frozen `U`, identity, or `U - identity` positions through the opt-in `query_indices_by_role` keyword and saves returned frames with the existing `diffsynth.utils.data.save_video`. S3 remains PENDING when S2 blocks.

- [ ] **Step 4: Document exact smoke and full commands**

Add to `utest/README.md`:

```bash
EVENT_JSON=/data/VideoMemory/utest/events/person_reappearance_delta8.json \
FUTURE_TARGET_VIDEO=/data/assets/mara_chunk8_target.mp4 \
FUTURE_TARGET_MANIFEST=/data/assets/mara_chunk8_target.manifest.json \
BASE_INFERENCE_ARGS=/data/config/inference_args.yaml \
PLATFORM_MANIFEST=/data/config/platform.manifest.json \
DONOR_PAYLOAD=/data/assets/matched_wrong_v2.pt \
DONOR_MANIFEST=/data/assets/matched_wrong.manifest.json \
EVENT_RUN_ROOT=/data/runs/identity_probe_smoke \
IDENTITY_SMOKE=1 RUN_DECODED_VALIDATION=0 \
bash scripts/run_slotmem_identity_probe.sh
```

and the full command with a new output root and `IDENTITY_SMOKE=0`. Explain that smoke fixes timestep 25, layers 5-10, max groups 4; full uses the 25-forward screen and conditional S2. Include output filenames and the identity-core claim boundary.

- [ ] **Step 5: Run runner and full regression suite**

Run: `python -m pytest utest/tests/test_identity_token_runner.py utest/tests/test_qstar_runner.py -q`

Run: `python -m pytest utest/tests -q`

Run: `python -m utest.identity_token_probe --self-check`

Expected: all available tests pass; torch/GPU-specific local tests may skip explicitly, not fail.

- [ ] **Step 6: Shell syntax and dirty-tree verification**

Run: `bash -n scripts/run_slotmem_identity_probe.sh`

Run: `git diff --check`

Expected: both exit zero.

- [ ] **Step 7: Commit Task 8**

```bash
git add scripts/run_slotmem_identity_probe.sh utest/tests/test_identity_token_runner.py utest/README.md
git commit -m "Add strict A100 identity probe runner"
```

---

### Task 9: Server handoff manifest and first A100 execution gates

**Files:**
- Verify only: committed source tree and server run directory
- Do not create: generated outputs in git

**Interfaces:**
- Consumes: a clean checkout containing Tasks 1-8 and the seven frozen input paths.
- Produces: one smoke run directory and, only after smoke PASS, one full S0-S2 run directory.

- [ ] **Step 1: Freeze server provenance**

Run on A100:

```bash
git rev-parse HEAD
git status --porcelain
python -V
python -c 'import torch,flash_attn; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory, flash_attn.__version__)'
```

Expected: clean source, A100 device, at least 75 GiB, and importable FlashAttention 2.

- [ ] **Step 2: Run the smoke command**

Use the README smoke command with a fresh output root. Require:

- runtime backend exactly `flash_attention_2`;
- identical frozen hashes across arms;
- exact repeat within configured tolerances;
- positive finite effective injection in a payload arm;
- no decode/scheduler-step evidence;
- measured forward count within the smoke budget;
- finite timing and VRAM below 80 GiB.

- [ ] **Step 3: Inspect smoke report before full run**

```bash
python - <<'PY'
import json, pathlib
p = pathlib.Path('/data/runs/identity_probe_smoke/identity_probe/identity_probe_report.json')
r = json.loads(p.read_text())
assert r['runtime']['attention_implementation'] == 'flash_attention_2'
assert r['runtime']['peak_allocated_bytes'] < 80 * 1024**3
assert r['gates']['runtime_contract']['status'] == 'PASS'
assert r['forward_count'] <= r['forward_budget']
print(json.dumps(r['gates'], indent=2))
PY
```

Expected: assertions pass. Replace only the absolute smoke output path with the path used in Step 2.

- [ ] **Step 4: Run full S0-S2 only after smoke PASS**

Use the README full command with a different fresh output root. Do not enable decoded validation on the first full run.

- [ ] **Step 5: Apply the scientific stop rules**

Read `identity_probe_report.json`:

- If `content_causality=BLOCK`, report that correct historical content was not distinguishable and do not type tokens.
- If `identity_set=BLOCK`, retain semantic maps as diagnostics but do not emit an identity-core claim.
- If S2 passes, rerun with another fresh output root and `RUN_DECODED_VALIDATION=1` for the four S3 arms.
- Archive the report, raw JSONL, commands, package manifest, figures, and terminal log together.

- [ ] **Step 6: Record server-only result commit separately**

Do not commit generated videos or run directories. If a small markdown result note is requested after inspection, commit only that note and exact report hashes in a separate commit from implementation.
