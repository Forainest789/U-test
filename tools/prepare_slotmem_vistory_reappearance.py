#!/usr/bin/env python3
"""Prepare the frozen ViStoryBench-derived SlotMem reappearance inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utest.vistory_reappearance import prepare_dataset


DEFAULT_SELECTION = (
    Path(__file__).resolve().parents[1]
    / "utest"
    / "events"
    / "vistorybench_reappearance_v1.json"
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate and convert the frozen ViStoryBench reappearance events."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    args = parser.parse_args(argv)

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    report = prepare_dataset(args.data_root, args.output_root, selection)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
