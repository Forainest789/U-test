from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


def test_strict_runner_dry_run_writes_complete_command_chain(tmp_path: Path) -> None:
    bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    if not bash.is_file():
        pytest.skip("Git Bash is not installed")
    repo = Path(__file__).resolve().parents[2]
    inputs = {}
    for name in ("event", "target", "args", "platform", "donor", "donor_manifest"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}", encoding="utf-8")
        inputs[name] = path
    output = tmp_path / "run"
    environment = {
        **os.environ,
        "EVENT_JSON": str(inputs["event"]),
        "FUTURE_TARGET_VIDEO": str(inputs["target"]),
        "BASE_INFERENCE_ARGS": str(inputs["args"]),
        "PLATFORM_MANIFEST": str(inputs["platform"]),
        "DONOR_PAYLOAD": str(inputs["donor"]),
        "DONOR_MANIFEST": str(inputs["donor_manifest"]),
        "EVENT_RUN_ROOT": str(output),
        "DRY_RUN": "1",
        "RUN_ROLLOUT": "1",
        "ALLOW_DIRTY_SOURCE": "1",
        "PYTHON_BIN": "python",
    }

    completed = subprocess.run(
        [str(bash), "scripts/run_slotmem_qstar_event.sh"],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert [row["name"] for row in manifest["commands"]] == [
        "content-self-check",
        "qstar-self-check",
        "prepare-prefix",
        "qstar-probe",
        "seven-rollouts",
        "validate-rollouts",
    ]
    assert manifest["dry_run"] is True
    assert manifest["seven_runs"] == [
        "correct", "correct_repeat", "no_memory", "zero", "random", "wrong", "native"
    ]
