#!/usr/bin/env python3
"""Survey and freeze matched ViStoryBench donor events."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utest.vistory_donors import (
    build_donor_candidate_survey,
    freeze_donor_selection,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Survey and freeze pre-generation ViStoryBench donor events."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    survey = commands.add_parser(
        "survey", help="Enumerate structurally matched official recurrence candidates."
    )
    survey.add_argument("--data-root", required=True, type=Path)
    survey.add_argument("--targets", required=True, type=Path)
    survey.add_argument("--output", required=True, type=Path)

    freeze = commands.add_parser(
        "freeze",
        help="Freeze formal three-event or explicit exploratory single-event donor bundles.",
        description=(
            "Freeze formal three-event or explicit exploratory single-event donor "
            "bundles."
        ),
    )
    freeze.add_argument("--data-root", required=True, type=Path)
    freeze.add_argument("--targets", required=True, type=Path)
    freeze.add_argument("--survey", required=True, type=Path)
    freeze.add_argument("--review", required=True, type=Path)
    freeze.add_argument("--output-root", required=True, type=Path)
    freeze.add_argument("--exploratory-target-event-id")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "survey":
        result = build_donor_candidate_survey(
            data_root=args.data_root,
            target_inputs_path=args.targets,
            output_path=args.output,
        )
    else:
        result = freeze_donor_selection(
            data_root=args.data_root,
            target_inputs_path=args.targets,
            survey_path=args.survey,
            review_path=args.review,
            output_root=args.output_root,
            exploratory_target_event_id=args.exploratory_target_event_id,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
