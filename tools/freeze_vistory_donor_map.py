#!/usr/bin/env python3
"""Freeze completed donor payloads into the subject-harness event map."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utest.vistory_donor_bundle import freeze_vistory_donor_map


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Freeze validated ViStoryBench donor payloads")
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--donor-run-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    result = freeze_vistory_donor_map(
        target_inputs_path=args.targets,
        selection_path=args.selection,
        donor_run_manifest_path=args.donor_run_manifest,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
