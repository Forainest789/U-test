#!/usr/bin/env python3
"""Run the SlotMem U-test arm queue, with a gate between every arm.

The 2026-08-19 sample_5 queue produced two arms that were bit-identical to each other and
to baseline: the intervention address was never read, and nothing checked before the
next arm started. Twelve GPU-hours bought zero bits. Everything here exists to make that
outcome impossible to reach again -- the queue stops at the first arm that cannot prove
it did something.

Order, and what each step is allowed to assume:

  0  self-check     arm semantics, on the CPU, no weights.            (seconds)
  1  preflight      the target is inside character_list[:cap].        (seconds)
  2  donor dump     a correct arm on ANOTHER story, --dump-donor.     (~1 chunk)
  3  arms           correct, no_memory, zero, random, wrong, native.  (~1 chunk each)
  4  distinctness   no two arms may share an output hash.             (seconds)

Arms are paired by construction: SlotMem derives the sampler seed from seed_base +
chunk_idx, so every arm denoises the same chunk from the same noise. Any difference in
the output is the intervention, not the sampler -- and, since the 2026-08-19 pair proved
this pipeline is bit-reproducible across separate processes on one GPU, two arms with the
same output hash means the intervention did not fire. That is an assertion, not a metric.

  python scripts/run_slotmem_utest_arms.py \
      --contract runs/utest/sample_5/prefix/arms/prefix_contract.json \
      --event    runs/utest/sample_5/event.json \
      --donor-contract runs/utest/sample_77/prefix/arms/prefix_contract.json \
      --donor-event    runs/utest/sample_77/event.json \
      --out-root runs/utest/sample_5/arms

Add --dry-run to print every command without booking a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # --dry-run must compose on a laptop; the real check needs the env
    torch = None

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "scripts" / "slotmem_content_audit.py"

# native is not an audit arm: it is SlotMem's own --native_wan_inference, no patch at all.
PATCHED_ARMS = ("correct", "no_memory", "zero", "random", "scramble", "wrong")
DEFAULT_ARMS = ("correct", "no_memory", "zero", "random", "wrong", "native")

# Paths the queue owns. Anything else comes from the frozen contract untouched, so an arm
# cannot quietly differ from the platform in a way nobody reads.
OWNED = ("--output_path", "--efficiency_metrics_path", "--save_state_path",
         "--resume_state_path", "--max_chunks", "--json_path", "--ref_image_path")


def set_flag(argv: list[str], name: str, value: str) -> list[str]:
    """Replace or append a value-taking flag. Boolean flags are never passed here -- OWNED
    is a closed list and every entry takes a value -- so a next-token that looks like a
    flag means the contract is malformed and we say so rather than corrupting argv."""
    out = list(argv)
    if out.count(name) > 1:
        raise SystemExit(
            f"[queue] {name} appears {out.count(name)}x in the contract. argparse honours the "
            "last one, so overriding the first would be silently defeated; de-duplicate it."
        )
    if name in out:
        i = out.index(name)
        if i + 1 >= len(out) or out[i + 1].startswith("--"):
            raise SystemExit(f"[queue] {name} in the contract carries no value; contract is malformed")
        out[i + 1] = str(value)
        return out
    return out + [name, str(value)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str], log_path: Path, cwd: Path | None, dry: bool) -> int:
    print(f"\n[queue] $ {' '.join(cmd)}", flush=True)
    if dry:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        for line in proc.stdout:
            log.write(line)
            if any(k in line for k in ("[audit]", "[preflight]", "GATE", "Error", "Traceback")):
                print(line.rstrip(), flush=True)
        return proc.wait()


# Server paths stay strings: Path() rewrites a POSIX "/data/..." into "\data\..." when the
# queue is composed on Windows, and the contract's paths are the server's, not ours.
def arm_argv(base: list[str], event: dict, arm_dir: Path, resume_state: str,
             json_path: str, ref_image: str) -> list[str]:
    argv = list(base)
    for flag, value in (
        ("--output_path", arm_dir),
        ("--efficiency_metrics_path", arm_dir / "efficiency.json"),
        ("--save_state_path", arm_dir / "resume_state.pt"),
        ("--resume_state_path", resume_state),
        # Absolute bound, applied before the resume slice (infer_slotmem.py ~3548), so the
        # target chunk is the only chunk any arm generates: chunk scoping for free, and
        # half the bill of the 2026-08-19 two-chunk arms.
        ("--max_chunks", event["max_chunks"]),
        ("--json_path", json_path),
        ("--ref_image_path", ref_image),
    ):
        argv = set_flag(argv, flag, str(value))
    return argv


def check_prefix_stop(state_path: str, contract: dict, target_chunk: int, label: str) -> None:
    """The prefix must stop exactly where the target chunk starts, or the run resumes at the
    wrong chunk and reads a target that is not under test. Milliseconds -- the alternative is
    finding out seventy minutes into the arm."""
    stop = (contract.get("event") or {}).get("target_chunk_idx")
    if torch is not None and Path(state_path).exists():
        # The state file carries the truth; the contract only carries an intention.
        # ponytail: loads the whole bank to read one int -- it is 7 MB and runs once.
        stop = torch.load(state_path, map_location="cpu", weights_only=False).get("next_chunk_idx")
    if stop is None or int(stop) == int(target_chunk):
        return
    raise SystemExit(
        f"[queue] the {label} prefix stops at chunk {stop}, but its event targets chunk "
        f"{target_chunk}. Extend the prefix first (one chunk, not a rebuild):\n"
        f"        python infer_slotmem.py <base args> --resume_state_path {state_path} \\\n"
        f"            --max_chunks {target_chunk} --save_state_path <new prefix_state.pt>\n"
        "        then point the contract at the new snapshot."
    )


def verify_snapshot(state_path: str, expected, label: str):
    """Every arm resumes from this one file. If it is not the frozen one, or it moves between
    arms, the arms are not paired and no difference between them means anything."""
    if not Path(state_path).exists():
        return None
    actual = sha256_file(Path(state_path))
    if expected and actual != expected:
        raise SystemExit(
            f"[queue] the {label} prefix state {state_path} hashes {actual[:16]}... but its "
            f"contract was frozen against {str(expected)[:16]}.... This is not the prefix that "
            "contract describes; the arms would not be paired."
        )
    return actual


def gate(report: Path, arm: str, dry: bool) -> None:
    if dry:
        return
    if not report.exists():
        raise SystemExit(f"[queue] arm '{arm}' wrote no audit report at {report}; queue stops")
    stats = json.loads(report.read_text(encoding="utf-8"))
    if not stats.get("intervention_effective"):
        raise SystemExit(
            f"[queue] arm '{arm}' FAILED its gate; queue stops before the next arm.\n"
            f"        {stats.get('gate_failure', 'no reason recorded')}\n"
            f"        target_reads={stats.get('target_reads')} "
            f"target_transforms={stats.get('target_transforms')} "
            f"writer_path_reads={stats.get('writer_path_reads')} "
            f"off_target_chunk_skips={stats.get('off_target_chunk_skips')}"
        )
    print(f"[queue] arm '{arm}' gate OK "
          f"(target_reads={stats['target_reads']}, target_transforms={stats['target_transforms']})",
          flush=True)


def distinctness(out_root: Path, arms: list[str], target_chunk: int, dry: bool) -> dict:
    """Same bytes => the intervention did not fire. This pipeline is deterministic across
    processes (measured 2026-08-19: no_memory and zero, launched nine hours apart on one
    GPU, byte-identical), so the technical-replicate floor is exactly zero and any
    collision here is a harness fault, never sampler noise."""
    hashes, report = {}, {}
    for arm in arms:
        # The estimand is the target chunk. merged_chunks.mp4 is a fallback only: a
        # single-chunk arm need not produce one, and comparing a merge against a chunk
        # would compare two different things.
        video = out_root / arm / f"chunk_{target_chunk:03d}.mp4"
        if not video.exists():
            video = out_root / arm / "merged_chunks.mp4"
        if not video.exists():
            candidates = sorted((out_root / arm).glob("chunk_*.mp4"))
            video = candidates[-1] if candidates else None
        if video is None:
            if dry:
                continue
            raise SystemExit(f"[queue] arm '{arm}' produced no video under {out_root / arm}")
        digest = sha256_file(video)
        report[arm] = {"video": str(video), "sha256": digest}
        if digest in hashes:
            raise SystemExit(
                f"[queue] arms '{hashes[digest]}' and '{arm}' produced BYTE-IDENTICAL video "
                f"({digest[:16]}...). Their interventions did not reach the model. "
                "This is the 2026-08-19 failure; do not read any metric from this queue."
            )
        hashes[digest] = arm
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contract", required=True, help="prefix_contract.json for the TARGET story")
    ap.add_argument("--event", required=True, help="event.json from slotmem_content_audit --preflight")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--donor-contract", help="prefix_contract.json for the DONOR story (wrong arm)")
    ap.add_argument("--donor-event", help="event.json for the DONOR story")
    ap.add_argument("--donor", help="reuse an existing slotmem_donor_v3 dump instead of making one")
    ap.add_argument("--prefix-state", help="override the contract's snapshot: an extended prefix_state.pt")
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    ap.add_argument("--seed", type=int, default=0, help="arm-transform seed; NOT the sampler seed")
    # The repo this script ships in IS the fork the contract froze; upstream is not.
    ap.add_argument("--slotmem-dir", default=os.environ.get("UTEST") or str(REPO),
                    help="the SlotMem checkout that froze the platform (this fork, not upstream)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    event = json.loads(Path(args.event).read_text(encoding="utf-8"))
    if "target_character" not in event:
        raise SystemExit(f"[queue] {args.event} has no target: {event.get('gate_failure', 'preflight failed')}")
    base = list(contract["base_inference_args"])
    # Two flags the arms must never inherit. --start_chunk_idx overrides the resume state's
    # next_chunk_idx (infer_slotmem.py ~3617) and would start every arm at a chunk the prefix
    # did not stop at. --target_character sorts the target to the front of character_list
    # before the cap (~3631), reordering the frozen platform's read window for every chunk of
    # every arm. Refuse rather than override: a contract carrying either was frozen under
    # different rules, so its other args are suspect too.
    for banned, why in (
        ("--start_chunk_idx", "arms must inherit their start from the resume state"),
        ("--target_character", "it reorders character_list before the memory-read cap"),
    ):
        if banned in base:
            raise SystemExit(f"[queue] the contract carries {banned}; {why}. Remove it from "
                             "base_inference_args and re-freeze the contract.")
    prefix_state = args.prefix_state or contract["snapshot"]["path"]
    # An overridden prefix is a deliberately different file, so only the contract's own
    # snapshot is checked against the contract's hash. Either way keep the hash and re-check
    # it after the arms: nothing may move underneath them.
    prefix_sha = verify_snapshot(
        prefix_state,
        None if args.prefix_state else (contract.get("snapshot") or {}).get("sha256"),
        "target",
    )
    check_prefix_stop(prefix_state, contract, event["target_chunk_idx"], "target")
    story_json = contract["inputs"]["source_json_path"]
    story_ref = contract["inputs"]["reference_path"]

    # 0. Arm semantics, before anything expensive.
    if run([args.python, str(AUDIT), "--self-check"], out_root / "self_check.log", None, args.dry_run):
        raise SystemExit("[queue] self-check failed; the arms do not do what they claim")

    # 1. The target address, re-derived from the story the run will actually use.
    pf_event = out_root / "preflight_event.json"
    cap = str(contract["runtime_contract"]["frozen_args"].get("max_memory_characters", "2"))
    rc = run([args.python, str(AUDIT), "--preflight",
              "--json-path", story_json,
              "--story-id", str(event.get("story_id", "")),
              "--max-memory-characters", cap,
              "--target-character", str(event["target_character"]),
              "--target-chunk-idx", str(event["target_chunk_idx"]),
              "--slotmem-dir", args.slotmem_dir,
              "--out", str(pf_event)],
             out_root / "preflight.log", None, args.dry_run)
    if rc:
        raise SystemExit(
            f"[queue] preflight rejects {event['target_character']}@chunk{event['target_chunk_idx']}: "
            f"SlotMem reads only character_list[:{cap}] and this target is below the cap. "
            f"See {pf_event} for the events it WILL read."
        )
    # From here the re-derived event is the only source of truth. Taking max_chunks from
    # one file and the target address from another is how two files quietly disagree.
    if pf_event.exists():
        event = json.loads(pf_event.read_text(encoding="utf-8"))
    if "max_chunks" not in event:
        raise SystemExit(f"[queue] {pf_event} has no max_chunks; it did not come from --preflight")

    # 2. The donor, from a different story. Never keyed by character name: NarraStream
    #    reuses one roster across stories, so a name-keyed donor exact-matches the target
    #    story's own same-named character and swaps the wrong person.
    donor_path = Path(args.donor) if args.donor else None
    if "wrong" in args.arms and donor_path is None:
        if not (args.donor_contract and args.donor_event):
            raise SystemExit("[queue] the wrong arm needs --donor, or --donor-contract + --donor-event")
        d_contract = json.loads(Path(args.donor_contract).read_text(encoding="utf-8"))
        d_event = json.loads(Path(args.donor_event).read_text(encoding="utf-8"))
        if str(d_event.get("story_id")) == str(event.get("story_id")):
            raise SystemExit("[queue] donor story == target story; that is a correct arm wearing a hat")
        if "max_chunks" not in d_event:
            raise SystemExit(f"[queue] {args.donor_event} did not come from --preflight "
                             f"(no max_chunks): {d_event.get('gate_failure', '')}")
        d_state = d_contract["snapshot"]["path"]
        verify_snapshot(d_state, (d_contract.get("snapshot") or {}).get("sha256"), "donor")
        check_prefix_stop(d_state, d_contract, d_event["target_chunk_idx"], "donor")
        d_dir = out_root / "donor_dump"
        donor_path = out_root / "donor_payload.pt"
        d_argv = arm_argv(list(d_contract["base_inference_args"]), d_event, d_dir, d_state,
                          d_contract["inputs"]["source_json_path"],
                          d_contract["inputs"]["reference_path"])
        rc = run([args.python, str(AUDIT), "--arm", "correct",
                  "--event", args.donor_event, "--seed", str(args.seed),
                  "--report", str(d_dir / "audit.json"),
                  "--dump-donor", str(donor_path),
                  "--slotmem-dir", args.slotmem_dir, "--"] + d_argv,
                 d_dir / "run.log", None, args.dry_run)
        if rc and not args.dry_run:
            raise SystemExit(f"[queue] donor dump exited {rc}; see {d_dir}/run.log")
        gate(d_dir / "audit.json", "donor_dump", args.dry_run)

    # 3. The arms.
    for arm in args.arms:
        arm_dir = out_root / arm
        argv = arm_argv(base, event, arm_dir, prefix_state, story_json, story_ref)
        if arm == "native":
            # Base Wan2.2: no LoRA, no memory path, but the SAME prefix frames, so the only
            # difference from correct is the thing under test. Their own flag, no patch.
            rc = run([args.python, "infer_slotmem.py"] + argv + ["--native_wan_inference"],
                     arm_dir / "run.log", Path(args.slotmem_dir), args.dry_run)
            if rc and not args.dry_run:
                raise SystemExit("[queue] native arm failed")
            # native has no audit report, so this line is its only proof that the memory
            # path was skipped rather than silently left on -- the exact 2026-08-19 fault.
            log = (arm_dir / "run.log")
            if not args.dry_run and "native_wan_inference=True" not in log.read_text(
                    encoding="utf-8", errors="replace"):
                raise SystemExit(
                    "[queue] the native arm never printed native_wan_inference=True; it ran "
                    "WITH the memory path. That is a duplicate of correct, not a baseline."
                )
            continue
        if arm not in PATCHED_ARMS:
            raise SystemExit(f"[queue] unknown arm '{arm}'; choose from {PATCHED_ARMS + ('native',)}")
        cmd = [args.python, str(AUDIT), "--arm", arm, "--event", str(pf_event),
               "--seed", str(args.seed), "--report", str(arm_dir / "audit.json"),
               "--slotmem-dir", args.slotmem_dir]
        if arm == "wrong":
            cmd += ["--donor", str(donor_path)]
        rc = run(cmd + ["--"] + argv, arm_dir / "run.log", None, args.dry_run)
        gate(arm_dir / "audit.json", arm, args.dry_run)
        if rc and not args.dry_run:
            raise SystemExit(f"[queue] arm '{arm}' exited {rc}")

    # 4. The prefix every arm resumed from must still be the file it was at the start.
    if prefix_sha and not args.dry_run:
        verify_snapshot(prefix_state, prefix_sha, "target")

    # 5. No two arms may agree byte-for-byte.
    report = distinctness(out_root, list(args.arms), int(event["target_chunk_idx"]), args.dry_run)
    summary = out_root / "queue_summary.json"
    if not args.dry_run:
        summary.write_text(json.dumps({
            "event": event, "arms": list(args.arms), "outputs": report,
            "donor": str(donor_path) if donor_path else None,
        }, indent=2), encoding="utf-8")
    print(f"\n[queue] all arms passed their gates and are mutually distinct -> {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
