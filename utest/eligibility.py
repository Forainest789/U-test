#!/usr/bin/env python3
"""E0: count the NarraStream stories that actually contain a memory-requiring event.

The whole method line is gated on this number. A recurrence event is a character that is
present, then absent for at least one FULL chunk, then present again -- only then does the
target chunk need long-term memory rather than the local window. N_e is the count of
distinct stories holding at least one such event that survives every eligibility filter.

Zero GPU, zero network. It reads the same ``rewrite_caption.json`` that
``tools/prepare_slotmem_narrastream_inputs.py`` reads, so it audits the inputs the
inference path will actually see::

    {"global_prompt": ..., "chunks": [{"character_list": [...], "content": ...}, ...]}

Run it BEFORE spending anything on M0. If N_e < 128 the single-source controller line does
not exist and no amount of later GPU changes that; the 2026-08-09 audit moved this from
week 5 to week 1 for exactly that reason.

    python scripts/narrastream_eligibility_audit.py --data-root <dir> --out e0.json
    python scripts/narrastream_eligibility_audit.py --self-check

Every exclusion is reported with its reason and an example, because the ambiguity rule
below is a heuristic and the protocol is to trust the audit report, not the rule.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# A name is only an identity anchor if it picks out ONE instance. These do not, so a
# "correct vs wrong" contrast built on them is undefined. Matched as whole normalized
# names or as a leading token, never as a substring: "the long-haired man" must survive
# while "another man" must not.
GENERIC_NAMES = {
    "man", "woman", "person", "people", "group", "crowd", "boy", "girl", "child",
    "someone", "somebody", "figure", "character", "unknown", "unnamed", "narrator",
    "the man", "the woman", "the person", "a man", "a woman", "a person",
    "men", "women", "children", "others", "background", "background people",
}
AMBIGUOUS_PREFIX = re.compile(r"^(another|other|the other|second|third|some)\b")

# §4.4 bands: which formal split the count can support.
BANDS = ((192, "test=64 / val=32", "full"), (128, "test=48 / val=24", "reduced"))
MIN_VIABLE = 128


def normalize(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def name_problem(name: str) -> str | None:
    """Why this character name cannot serve as an identity anchor, or None."""
    norm = normalize(name)
    if not norm:
        return "empty_name"
    if norm in GENERIC_NAMES:
        return "generic_name"
    if AMBIGUOUS_PREFIX.match(norm):
        return "ambiguous_reference"
    return None


def load_story(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks") if isinstance(payload, dict) else payload
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"no chunks in {path}")
    return {
        "chunks": chunks,
        "global_prompt": payload.get("global_prompt", "") if isinstance(payload, dict) else "",
        "characters": payload.get("characters") if isinstance(payload, dict) else None,
    }


def presence_disagreements(story: dict) -> list[str]:
    """Cross-check ``character_list`` against the ``characters`` per-chunk descriptions.

    The schema carries presence twice: as a per-chunk name list, and as a per-character
    list of chunk descriptions that is empty where the character is absent. They should
    agree. If they ever do not, N_e -- the number gating the whole method line -- is
    computed from a signal the data does not actually support, so the disagreement is
    reported rather than silently resolved in favour of either source.
    """
    characters = story.get("characters")
    if not isinstance(characters, dict):
        return []
    n = len(story["chunks"])
    from_list: dict[str, set[int]] = defaultdict(set)
    for idx, chunk in enumerate(story["chunks"]):
        for raw in chunk.get("character_list") or []:
            from_list[normalize(raw)].add(idx)
    out: list[str] = []
    for raw, descriptions in characters.items():
        if not isinstance(descriptions, list) or len(descriptions) != n:
            continue
        name = normalize(raw)
        from_desc = {i for i, text in enumerate(descriptions) if str(text).strip()}
        if from_desc != from_list.get(name, set()):
            out.append(f"{name}: list={sorted(from_list.get(name, set()))} "
                       f"desc={sorted(from_desc)}")
    return out


def recurrence_events(chunks: list[dict]) -> tuple[list[dict], Counter]:
    """Present -> absent for >=1 full chunk -> present again, per character.

    Only the FIRST return per absence is an event: the second return is a different
    regime (the memory was just refreshed) and would not be an independent observation.
    """
    presence: dict[str, list[int]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        for raw in chunk.get("character_list") or []:
            presence[normalize(raw)].append(idx)

    events: list[dict] = []
    rejected: Counter = Counter()
    for name, seen in presence.items():
        problem = name_problem(name)
        if problem:
            rejected[problem] += 1
            continue
        seen = sorted(set(seen))
        for prev, nxt in zip(seen, seen[1:]):
            gap = nxt - prev - 1
            if gap < 1:
                continue  # adjacent chunks: the local window still covers it
            if nxt < 2:
                rejected["no_freezable_prefix"] += 1
                continue  # nothing to snapshot before the target
            if not str(chunks[nxt].get("content") or "").strip():
                rejected["empty_target_prompt"] += 1
                continue
            if not str(chunks[prev].get("content") or "").strip():
                rejected["empty_source_prompt"] += 1
                continue
            events.append({
                "character_name": name,
                "memory_chunk_idx": prev,
                "target_chunk_idx": nxt,
                "gap_chunks": gap,
            })
            break  # first return only
    return events, rejected


def audit(data_root: Path, *, require_reference: bool = True) -> dict:
    stories: list[dict] = []
    rejected: Counter = Counter()
    unreadable: list[str] = []
    examples: dict[str, str] = {}
    disagreements: dict[str, list[str]] = {}

    for caption in sorted(data_root.glob("*/rewrite_caption.json")):
        story_id = caption.parent.name
        try:
            story = load_story(caption)
        except (ValueError, json.JSONDecodeError) as exc:
            unreadable.append(f"{story_id}: {exc}")
            continue
        # The reference image is the I2V identity anchor; without it C_id has nothing to
        # score against and the story cannot enter the outcome vector.
        has_reference = any(
            (caption.parent / candidate).exists()
            for candidate in ("frame.jpg", "frame.png", "subject_references")
        )
        if require_reference and not has_reference:
            rejected["no_reference_image"] += 1
            examples.setdefault("no_reference_image", story_id)
            continue
        conflict = presence_disagreements(story)
        if conflict:
            disagreements[story_id] = conflict
        events, story_rejected = recurrence_events(story["chunks"])
        rejected.update(story_rejected)
        for reason in story_rejected:
            examples.setdefault(reason, story_id)
        if not events:
            rejected["no_recurrence_event"] += 1
            examples.setdefault("no_recurrence_event", story_id)
            continue
        for event in events:
            event["entity_uid"] = f"{story_id}::{event['character_name']}"
        stories.append({
            "story_id": story_id,
            "n_chunks": len(story["chunks"]),
            "events": events,
        })

    # A name shared across stories is a different instance in each. Reported so §4.3's
    # "text-identical names must not make cross-story instances correct" is auditable
    # rather than assumed.
    by_name: dict[str, set[str]] = defaultdict(set)
    for story in stories:
        for event in story["events"]:
            by_name[event["character_name"]].add(story["story_id"])
    collisions = {name: sorted(ids) for name, ids in by_name.items() if len(ids) > 1}

    n_e = len(stories)
    band, split, strength = "insufficient", None, "blocked"
    for threshold, split_plan, label in BANDS:
        if n_e >= threshold:
            band, split, strength = f">={threshold}", split_plan, label
            break

    return {
        "n_eligible_stories": n_e,
        "n_events": sum(len(s["events"]) for s in stories),
        "band": band,
        "formal_split": split,
        "headline_strength": strength,
        "passes_min_viable": n_e >= MIN_VIABLE,
        "gap_histogram": dict(sorted(Counter(
            event["gap_chunks"] for story in stories for event in story["events"]
        ).items())),
        "events_per_story": dict(sorted(Counter(
            len(story["events"]) for story in stories
        ).items())),
        "rejected": dict(rejected.most_common()),
        "rejection_examples": examples,
        "cross_story_name_collisions": collisions,
        "presence_signal_disagreements": disagreements,
        "unreadable": unreadable,
        "stories": stories,
    }


def _self_check() -> None:
    import tempfile

    fixtures = {
        # present(0) -> absent(1) -> present(2): a real event, prefix exists
        "ok": [["ana"], ["bo"], ["ana"]],
        # adjacent chunks only: the local window covers it, not a memory event
        "adjacent": [["ana"], ["ana"], ["ana"]],
        # returns at chunk 1, so chunks 0:0 is not a freezable prefix
        "no_prefix": [["ana"], ["ana"]],
        # generic name cannot anchor identity
        "generic": [["the man"], ["bo"], ["the man"]],
        # ambiguous reference
        "ambiguous": [["another man"], ["bo"], ["another man"]],
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, chunk_names in fixtures.items():
            story = root / name
            story.mkdir()
            (story / "frame.jpg").write_bytes(b"")
            (story / "rewrite_caption.json").write_text(json.dumps({
                "global_prompt": name,
                "chunks": [
                    {"character_list": names, "content": f"chunk {i}"}
                    for i, names in enumerate(chunk_names)
                ],
            }), encoding="utf-8")
        # a story with no reference image is excluded before parsing
        bare = root / "no_ref"
        bare.mkdir()
        (bare / "rewrite_caption.json").write_text(json.dumps({
            "chunks": [{"character_list": ["ana"], "content": "x"},
                       {"character_list": ["bo"], "content": "y"},
                       {"character_list": ["ana"], "content": "z"}],
        }), encoding="utf-8")

        report = audit(root)
        assert report["n_eligible_stories"] == 1, report["n_eligible_stories"]
        assert report["stories"][0]["story_id"] == "ok"
        event = report["stories"][0]["events"][0]
        assert event == {
            "character_name": "ana", "memory_chunk_idx": 0, "target_chunk_idx": 2,
            "gap_chunks": 1, "entity_uid": "ok::ana",
        }, event
        assert report["rejected"]["generic_name"] == 1
        assert report["rejected"]["ambiguous_reference"] == 1
        assert report["rejected"]["no_reference_image"] == 1
        assert report["rejected"]["no_recurrence_event"] >= 2  # adjacent + no_prefix
        assert report["passes_min_viable"] is False
        assert report["headline_strength"] == "blocked"

        # gap must be a FULL absent chunk, and a longer gap still counts once
        long_gap = audit(root)["gap_histogram"]
        assert long_gap == {1: 1}, long_gap
    print("[e0] self-check OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="dir of <story>/rewrite_caption.json")
    parser.add_argument("--out", type=Path, help="write the full report as JSON")
    parser.add_argument("--allow-missing-reference", action="store_true",
                        help="count stories with no reference image (they cannot score C_id)")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        _self_check()
        return
    if args.data_root is None:
        parser.error("--data-root is required unless --self-check")

    report = audit(args.data_root, require_reference=not args.allow_missing_reference)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[e0] N_e = {report['n_eligible_stories']} eligible stories, "
          f"{report['n_events']} events")
    print(f"[e0] band {report['band']} -> formal split {report['formal_split']} "
          f"({report['headline_strength']})")
    print(f"[e0] gap histogram (chunks absent): {report['gap_histogram']}")
    print(f"[e0] rejected: {report['rejected']}")
    if report["cross_story_name_collisions"]:
        print(f"[e0] WARNING {len(report['cross_story_name_collisions'])} character names "
              "appear in more than one story -- entity_uid must gate correct/wrong")
    if report["presence_signal_disagreements"]:
        print(f"[e0] WARNING character_list and characters[] disagree on presence in "
              f"{len(report['presence_signal_disagreements'])} stories -- N_e is not "
              "trustworthy until this is resolved")
    if report["unreadable"]:
        print(f"[e0] WARNING {len(report['unreadable'])} unreadable stories")
    if not report["passes_min_viable"]:
        print(f"[e0] BLOCKED: N_e < {MIN_VIABLE}. Single-source controller line does not "
              "exist; add a second training source before spending GPU.")


if __name__ == "__main__":
    main()
