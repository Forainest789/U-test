#!/usr/bin/env python3
"""Survey, review, and freeze a replacement ViStoryBench target event."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utest.vistory_target_selection import (
    build_replacement_target_survey,
    freeze_replacement_selection,
    write_replacement_review_template,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Survey and freeze a reviewed ViStoryBench replacement target."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    survey = commands.add_parser(
        "survey", help="Enumerate donor-supported replacement target candidates."
    )
    survey.add_argument("--data-root", required=True, type=Path)
    survey.add_argument("--selection", required=True, type=Path)
    survey.add_argument("--output", required=True, type=Path)

    review = commands.add_parser(
        "review-template", help="Write the strict female-character review template."
    )
    review.add_argument("--survey", required=True, type=Path)
    review.add_argument("--output", required=True, type=Path)

    freeze = commands.add_parser(
        "freeze", help="Freeze the deterministically ranked reviewed replacement."
    )
    freeze.add_argument("--data-root", required=True, type=Path)
    freeze.add_argument("--selection", required=True, type=Path)
    freeze.add_argument("--survey", required=True, type=Path)
    freeze.add_argument("--review", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "survey":
        result = build_replacement_target_survey(
            data_root=args.data_root,
            selection_path=args.selection,
            output_path=args.output,
        )
    elif args.command == "review-template":
        result = write_replacement_review_template(
            survey_path=args.survey,
            output_path=args.output,
        )
    else:
        result = freeze_replacement_selection(
            data_root=args.data_root,
            selection_path=args.selection,
            survey_path=args.survey,
            review_path=args.review,
            output_path=args.output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
