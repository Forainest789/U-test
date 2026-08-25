from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _runner_environment(tmp_path: Path) -> dict[str, str]:
    inputs = {}
    for name in (
        "event",
        "target",
        "teacher_manifest",
        "args",
        "platform",
        "donor",
        "donor_manifest",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text("{}", encoding="utf-8")
        inputs[name] = path
    return {
        **os.environ,
        "EVENT_JSON": str(inputs["event"]),
        "FUTURE_TARGET_VIDEO": str(inputs["target"]),
        "FUTURE_TARGET_MANIFEST": str(inputs["teacher_manifest"]),
        "BASE_INFERENCE_ARGS": str(inputs["args"]),
        "PLATFORM_MANIFEST": str(inputs["platform"]),
        "DONOR_PAYLOAD": str(inputs["donor"]),
        "DONOR_MANIFEST": str(inputs["donor_manifest"]),
        "EVENT_RUN_ROOT": str(tmp_path / "run"),
        "DRY_RUN": "1",
        "ALLOW_DIRTY_SOURCE": "1",
        "PYTHON_BIN": "python",
    }


def test_identity_runner_dry_run_records_fast_a100_chain(tmp_path: Path) -> None:
    bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    if not bash.is_file():
        pytest.skip("Git Bash is not installed")
    environment = _runner_environment(tmp_path)

    completed = subprocess.run(
        [str(bash), "scripts/run_slotmem_identity_probe.sh"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    manifest = json.loads(
        (Path(environment["EVENT_RUN_ROOT"]) / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert [row["name"] for row in manifest["commands"]] == [
        "identity-self-check",
        "input-contract-preflight",
        "prepare-prefix",
        "identity-probe",
    ]
    probe = manifest["commands"][-1]["argv"]
    assert probe[probe.index("--timestep-indices") + 1] == "0,25,49"
    assert (
        manifest["environment"]["DIFFSYNTH_ATTENTION_IMPLEMENTATION"]
        == "flash_attention_2"
    )
