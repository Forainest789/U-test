from pathlib import Path
from types import SimpleNamespace


import torch

from utest.subject_reappearance_target_runner import (
    TARGET_ARMS,
    build_target_arm_bundles,
    run_target_arm_loop,
)


def _layerwise(layers):
    return {"__layerwise__": True, "layers": dict(layers)}


class _FakeMemoryBank:
    def __init__(self) -> None:
        self.memory_bank = {}
        self.memory_meta_bank = {}
        self.first_appearance = {}

    def get_memory_payload_for_read(self, char_id, bank_idx=0):
        return self.memory_bank.get(char_id, {}).get(str(bank_idx))


FAKE_RUNTIME = SimpleNamespace(
    RoleWiseSlotMemoryBank=_FakeMemoryBank,
    _is_layerwise_token_payload=lambda value: isinstance(value, dict)
    and value.get("__layerwise__") is True,
    _iter_layerwise_items=lambda value: value["layers"].items(),
    _select_layerwise_value=lambda value, layer, default=None: (
        value["layers"].get(str(layer), default)
        if isinstance(value, dict) and value.get("__layerwise__") is True
        else default
    ),
    _make_layerwise_container=_layerwise,
)


class _FakeEngine:
    def __init__(self) -> None:
        self.calls = []
        self.runtime_chunk_warnings = ["stale"]
        self.runtime_role_states = [{"stale": True}]
        self._last_sparse_role_memory_stats = {"stale": 1}
        self._last_sparse_role_memory_stats_by_layer = {"0": {"stale": 1}}
        self._last_jigsaw_stage2_writer_stats = {"stale": 1}

    def generate_chunk(self, **kwargs):
        self.calls.append(
            {
                **kwargs,
                "runtime_chunk_warnings": list(self.runtime_chunk_warnings),
                "runtime_role_states": list(self.runtime_role_states),
            }
        )
        arm = kwargs["memory_bank_tokens"]
        self._last_sparse_role_memory_stats_by_layer = {
            "0": {
                "enabled": 0.0 if arm is None else 1.0,
                "selected_memory_tokens": 0 if arm is None else 8,
                "effective_delta_norm": 0.0 if arm is None else 0.1,
            }
        }
        return [f"frame-{len(self.calls)}"], None, {}


def test_target_arm_loop_reuses_engine_and_generates_only_one_target_per_arm(
    tmp_path: Path,
) -> None:
    engine = _FakeEngine()
    bundles = {
        "full_correct": {"memory_bank_tokens": "correct"},
        "no_memory": {"memory_bank_tokens": None},
        "zero_path": {"memory_bank_tokens": "zero"},
        "wrong_subject": {"memory_bank_tokens": "wrong"},
    }
    saved = []

    report = run_target_arm_loop(
        engine,
        bundles,
        prompt="target prompt",
        seed=0,
        reference_frames=["previous"],
        fixed_reference=None,
        output_root=tmp_path,
        target_chunk_idx=6,
        save_video_fn=lambda frames, path, fps: saved.append((frames, Path(path), fps)),
    )

    assert tuple(report) == TARGET_ARMS
    assert len(engine.calls) == 4
    assert [call["memory_bank_tokens"] for call in engine.calls] == [
        "correct", None, "zero", "wrong"
    ]
    assert all(call["prompt"] == "target prompt" for call in engine.calls)
    assert all(call["seed"] == 0 for call in engine.calls)
    assert all(call["online_memory_chars"] == [] for call in engine.calls)
    assert all(call["online_memory_bank_percents"] == [] for call in engine.calls)
    assert all(call["runtime_chunk_warnings"] == [] for call in engine.calls)
    assert all(call["runtime_role_states"] == [] for call in engine.calls)
    assert [path.name for _, path, _ in saved] == ["chunk_006.mp4"] * 4
    assert [path.parent.name for _, path, _ in saved] == list(TARGET_ARMS)
    assert all(fps == 16 for _, _, fps in saved)


def test_build_target_arm_bundles_reloads_prefix_and_changes_only_target(
    tmp_path: Path,
) -> None:
    target = torch.tensor([[1.0, 2.0]])
    other = torch.tensor([[7.0, 8.0]])
    state = {
        "memory_bank": {
            name: {
                "0": {
                    "tokens": _layerwise({"0": tokens}),
                    "token_meta": _layerwise({"0": [{"char_id": name}]}),
                }
            }
            for name, tokens in (("Target", target), ("Other", other))
        },
        "memory_meta_bank": {},
        "first_appearance": {"Target": 0, "Other": 0},
    }
    original = _FakeMemoryBank.get_memory_payload_for_read
    installed = []

    def install_fake(*, arm, report_path, **_kwargs):
        installed.append(arm)

        def patched(self, char_id, bank_idx=0):
            payload = original(self, char_id, bank_idx)
            if char_id != "Target":
                return payload
            if arm == "no_memory":
                return None
            if arm == "zero_path":
                payload["tokens"]["layers"]["0"] = torch.zeros_like(
                    payload["tokens"]["layers"]["0"]
                )
            if arm == "wrong_subject":
                payload["tokens"]["layers"]["0"] = torch.full_like(
                    payload["tokens"]["layers"]["0"], 9
                )
            return payload

        _FakeMemoryBank.get_memory_payload_for_read = patched

        def flush():
            _FakeMemoryBank.get_memory_payload_for_read = original
            Path(report_path).write_text("{}", encoding="utf-8")

        return flush

    class _Engine:
        args = type("Args", (), {"max_memory_characters": 4})()

        @staticmethod
        def _use_legacy_multi_memory_banks():
            return False

        @staticmethod
        def _single_online_memory_bank_percents():
            return [0.5]

    bundles = build_target_arm_bundles(
        FAKE_RUNTIME,
        _Engine(),
        state=state,
        target_chunk={"character_list": ["Target", "Other"]},
        event={"event_id": "event", "character_name": "Target", "target_chunk_idx": 6},
        seed=0,
        mask_manifest={},
        runtime_contract={},
        event_file_sha256="a" * 64,
        manifest_file_sha256="b" * 64,
        report_root=tmp_path,
        audit_installer=install_fake,
    )

    def layer(arm):
        return bundles[arm]["memory_bank_tokens"]["layers"]["0"]["0"]

    assert installed == list(TARGET_ARMS)
    assert torch.equal(layer("full_correct"), torch.cat((target, other)))
    assert torch.equal(layer("no_memory"), other)
    assert torch.equal(layer("zero_path"), torch.cat((torch.zeros_like(target), other)))
    assert torch.equal(layer("wrong_subject"), torch.cat((torch.full_like(target, 9), other)))
    assert _FakeMemoryBank.get_memory_payload_for_read is original
