#!/usr/bin/env bash
# SlotMem content-causality U-test: six arms on one frozen prefix, end to end.
#
# The 2026-08-19 queue spent twelve GPU-hours and bought zero bits: the intervention
# address was never read, and nothing checked before the next arm started. Every step here
# that costs seconds runs before every step that costs an hour, and the queue stops at the
# first arm that cannot prove it did something.
#
#   0  self-check    arm semantics AND patch wiring, on the CPU.        seconds
#   1  preflight     the target is inside character_list[:cap].         seconds
#   2  prefix        extend by one chunk, then freeze a new contract.   ~1 chunk
#   3  arms          correct no_memory zero random wrong native.        ~1 chunk each
#   4  report        gates, hashes, injection norms, wall clock.        seconds
#   5  utility       decoded outcome per arm, signed against no_memory.   minutes
#
# The frozen platform is not touched: max_memory_characters stays at whatever the contract
# froze, character_list is never reordered, and the target is chosen to be one the reader
# already opens. See --preflight's readability table for why chunk 3 was not usable.
#
#   bash scripts/run_slotmem_utest.sh                 # dry-run first, then ask
#   RUN_FOR_REAL=1 bash scripts/run_slotmem_utest.sh  # book the GPU
set -euo pipefail

# ---------------------------------------------------------------- config
# This script ships inside the fork it drives, so it locates everything from its own path.
# Override UTEST only if the data and runs live somewhere other than this checkout. The
# frozen platform is this fork, never upstream YilaiLiu-HKU/SlotMem: that clone has neither
# the reader seam the patch needs nor the chunk stamp, and the audit refuses it by name.
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
UTEST=${UTEST:-$REPO}
SLOTMEM=${SLOTMEM:-$UTEST}
PYTHON=${PYTHON:-python}
RUN=${RUN:-$UTEST/runs/utest_content_$(date +%Y%m%d_%H%M)}

TARGET_STORY=${TARGET_STORY:-sample_5}
TARGET_CHAR=${TARGET_CHAR:-evan}
DONOR_STORY=${DONOR_STORY:-sample_77}
ARMS=${ARMS:-"correct no_memory zero random wrong native"}

# The contracts that froze each story's prefix. Override if they live elsewhere.
PREFIX_CONTRACT=${PREFIX_CONTRACT:-$UTEST/runs/narrastream_build/${TARGET_STORY}_prefix_20260814_152943/arms/prefix_contract.json}
DONOR_CONTRACT=${DONOR_CONTRACT:-}

# Where the decoded-outcome scorer lives. It imports fumd.eval (identity, quality,
# labelling), which is not in this repo, so step 5 is skipped with the command to run by
# hand when FUMD is unset. Videos and queue_summary.json are the whole interface.
FUMD=${FUMD:-}

# Whole-module CPU offload between chunks. Off is the 80GB fast profile; set to 1 for the
# low-VRAM one. It moves weights, not numbers, so it does not change any arm's output.
export SLOTMEM_OFFLOAD_MODELS=${SLOTMEM_OFFLOAD_MODELS:-0}

AUDIT="$REPO/scripts/slotmem_content_audit.py"
QUEUE="$REPO/scripts/run_slotmem_utest_arms.py"
STORY_JSON="$UTEST/data_curation/narrastream_slotmem/$TARGET_STORY/rewrite_caption.json"
DONOR_JSON="$UTEST/data_curation/narrastream_slotmem/$DONOR_STORY/rewrite_caption.json"

need() { [ -e "$1" ] || { echo "missing: $1${2:+  ($2)}" >&2; exit 1; }; }
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

need "$AUDIT"; need "$QUEUE"; need "$SLOTMEM/infer_slotmem.py" "SLOTMEM must be the U-test fork that froze the platform"
need "$STORY_JSON"; need "$PREFIX_CONTRACT"
if [[ "$ARMS" == *wrong* ]]; then
  [ -n "$DONOR_CONTRACT" ] || { echo "the wrong arm needs DONOR_CONTRACT=<$DONOR_STORY's prefix_contract.json>" >&2; exit 1; }
  need "$DONOR_CONTRACT"; need "$DONOR_JSON"
fi
mkdir -p "$RUN"
echo "run dir: $RUN"

# ---------------------------------------------------------------- 0. self-check
say "0. self-check (arm semantics + patch wiring, no GPU)"
"$PYTHON" "$AUDIT" --self-check

# ---------------------------------------------------------------- 1. preflight
say "1. preflight: an address the reader will actually open"
CAP=$("$PYTHON" -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))['runtime_contract']['frozen_args'].get('max_memory_characters','2'))" "$PREFIX_CONTRACT")
echo "max_memory_characters = $CAP (from the frozen contract; NOT raised)"
"$PYTHON" "$AUDIT" --preflight \
  --json-path "$STORY_JSON" --story-id "$TARGET_STORY" \
  --max-memory-characters "$CAP" --target-character "$TARGET_CHAR" \
  --slotmem-dir "$SLOTMEM" --out "$RUN/event.json"

if [[ "$ARMS" == *wrong* ]]; then
  # Pin the donor to the chunk its own prefix already stops at, so no donor prefix has to
  # be extended. Any entity from another story serves as a wrong identity, so the character
  # is left to preflight: widest gap wins.
  DONOR_STOP=$("$PYTHON" -c "
import json,sys,torch
c=json.load(open(sys.argv[1],encoding='utf-8'))
print(int(torch.load(c['snapshot']['path'],map_location='cpu',weights_only=False)['next_chunk_idx']))" "$DONOR_CONTRACT")
  echo "donor prefix stops at chunk $DONOR_STOP; picking a readable event there"
  "$PYTHON" "$AUDIT" --preflight \
    --json-path "$DONOR_JSON" --story-id "$DONOR_STORY" \
    --max-memory-characters "$CAP" --target-chunk-idx "$DONOR_STOP" \
    --slotmem-dir "$SLOTMEM" --out "$RUN/donor_event.json"
fi

# ---------------------------------------------------------------- 2. prefix
say "2. prefix: extend to the target chunk, then freeze a contract for it"
"$PYTHON" - "$PREFIX_CONTRACT" "$RUN/event.json" "$RUN/extend.argv" <<'PLAN'
import json, sys
from pathlib import Path
import torch

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
event = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if "target_character" not in event:
    raise SystemExit(f"[prefix] preflight found no readable event: {event.get('gate_failure','')}")

state = contract["snapshot"]["path"]
stop = int(torch.load(state, map_location="cpu", weights_only=False)["next_chunk_idx"])
want = int(event["prefix_max_chunks"])           # the prefix must stop where the target starts
print(f"[prefix] current stop={stop} target chunk={event['target_chunk_idx']} needs stop={want}")
if stop > want:
    raise SystemExit(f"[prefix] the prefix already ran past the target ({stop} > {want}); "
                     "pick a later event or rebuild, do not truncate a state file")
if stop == want:
    Path(sys.argv[3]).write_text("", encoding="utf-8")
    print("[prefix] already stops in the right place; nothing to generate")
    raise SystemExit(0)

# The queue owns these; passing the contract's stale copies would just be overridden.
OWNED = {"--output_path", "--efficiency_metrics_path", "--save_state_path",
         "--resume_state_path", "--max_chunks"}
argv, skip = [], False
for tok in contract["base_inference_args"]:
    if skip:
        skip = False
        continue
    if tok in OWNED:
        skip = True
        continue
    argv.append(tok)
out = Path(sys.argv[2]).parent / "prefix"
argv += ["--resume_state_path", state,
         "--max_chunks", str(want),
         "--save_state_path", str(out / "prefix_state.pt"),
         "--output_path", str(out),
         "--efficiency_metrics_path", str(out / "efficiency.json")]
Path(sys.argv[3]).write_bytes(b"\0".join(a.encode() for a in argv))
print(f"[prefix] continuing {want - stop} chunk(s) from {state}")
PLAN

if [ -s "$RUN/extend.argv" ]; then
  mapfile -d '' -t EXTEND < "$RUN/extend.argv"
  if [ -n "${RUN_FOR_REAL:-}" ]; then
    ( cd "$SLOTMEM" && "$PYTHON" infer_slotmem.py "${EXTEND[@]}" ) 2>&1 | tee "$RUN/prefix_extend.log"
  else
    echo "[dry-run] cd $SLOTMEM && $PYTHON infer_slotmem.py ${EXTEND[*]}"
  fi
fi

"$PYTHON" - "$PREFIX_CONTRACT" "$RUN/event.json" "$RUN/prefix/prefix_contract.json" <<'FREEZE'
import hashlib, json, sys
from pathlib import Path

old = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
event = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
out = Path(sys.argv[3])
extended = out.parent / "prefix_state.pt"
state = extended if extended.exists() else Path(old["snapshot"]["path"])
if not state.exists():
    print(f"[freeze] no state at {state} yet (dry run?); contract not written")
    raise SystemExit(0)

# Two flags the arms must never inherit: --start_chunk_idx would override the resume
# state's next_chunk_idx, and --target_character reorders character_list before the
# memory-read cap. Neither may ride along inside a "frozen" contract.
argv, skip = [], False
for tok in old["base_inference_args"]:
    if skip:
        skip = False
        continue
    if tok in ("--start_chunk_idx", "--target_character"):
        skip = True
        continue
    argv.append(tok)
dupes = sorted({a for a in argv if a.startswith("--") and argv.count(a) > 1})
if dupes:
    raise SystemExit(f"[freeze] {dupes} appear twice in base_inference_args; argparse honours "
                     "the last, so an override would be silently defeated. Fix by hand.")

digest = hashlib.sha256(state.read_bytes()).hexdigest()
new = dict(old)
new["snapshot"] = {"path": str(state), "bytes": state.stat().st_size, "sha256": digest}
new["event"] = dict(old.get("event") or {},
                    character_name=event["target_character"],
                    target_chunk_idx=int(event["target_chunk_idx"]),
                    memory_chunk_idx=int(event["first_appearance_chunk"]),
                    gap_chunks=int(event["gap_chunks"]),
                    entity_uid=event["entity_uid"],
                    story_id=event["story_id"])
new["base_inference_args"] = argv
new["preregistration"] = {
    "target_selected_by": "slotmem_content_audit.py --preflight",
    "note": ("the originally planned chunk was not readable: SlotMem reads "
             "character_list[:max_memory_characters] with no ranking, so a target below the "
             "cap is never addressed. max_memory_characters was NOT raised and "
             "character_list was NOT reordered; a readable event was chosen instead."),
    "max_memory_characters": event["max_memory_characters"],
    "rank_in_character_list": event["rank_in_character_list"],
    "co_readable": event["co_readable"],
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(new, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[freeze] {out}  state={state.name} sha={digest[:16]}... stop-at-chunk={event['target_chunk_idx']}")
FREEZE

# ---------------------------------------------------------------- 3. arms
say "3. arms"
CONTRACT="$RUN/prefix/prefix_contract.json"
[ -f "$CONTRACT" ] || CONTRACT="$PREFIX_CONTRACT"
QARGS=(--contract "$CONTRACT" --event "$RUN/event.json" --out-root "$RUN/arms"
       --slotmem-dir "$SLOTMEM" --arms $ARMS)
if [[ "$ARMS" == *wrong* ]]; then
  QARGS+=(--donor-contract "$DONOR_CONTRACT" --donor-event "$RUN/donor_event.json")
fi

"$PYTHON" "$QUEUE" "${QARGS[@]}" --dry-run
if [ -z "${RUN_FOR_REAL:-}" ]; then
  echo
  echo "dry run only. Read the commands above, then:"
  echo "  RUN_FOR_REAL=1 RUN=$RUN bash ${BASH_SOURCE[0]}"
  exit 0
fi
"$PYTHON" "$QUEUE" "${QARGS[@]}"

# ---------------------------------------------------------------- 4. report
say "4. report"
"$PYTHON" - "$RUN" <<'REPORT'
import json, sys
from datetime import datetime
from pathlib import Path

run = Path(sys.argv[1])
summary = json.loads((run / "arms" / "queue_summary.json").read_text(encoding="utf-8"))
event = summary["event"]
print(f"target {event['target_character']} @ chunk {event['target_chunk_idx']} "
      f"(first seen chunk {event['first_appearance_chunk']}, gap {event['gap_chunks']}, "
      f"rank {event['rank_in_character_list']}/{event['max_memory_characters']})\n")

head = f"{'arm':<10} {'reads':>6} {'transf':>7} {'writer':>7} {'skips':>6} {'gate':>6}"
print(head + "\n" + "-" * len(head))
for arm in summary["arms"]:
    a = run / "arms" / arm / "audit.json"
    if not a.exists():
        print(f"{arm:<10} {'-':>6} {'-':>7} {'-':>7} {'-':>6} {'native':>6}")
        continue
    s = json.loads(a.read_text(encoding="utf-8"))
    print(f"{arm:<10} {s['target_reads']:>6} {s['target_transforms']:>7} "
          f"{s['writer_path_reads']:>7} {s['off_target_chunk_skips']:>6} "
          f"{'OK' if s['intervention_effective'] else 'FAIL':>6}")

# authority = how far memory moves a video token, relative to that token's own norm.
# role/plain is a head-balance diagnostic and says nothing about that: an arm can differ
# in head balance while the injection is too small to change any decoded frame.
print(f"\n{'arm':<10} {'sha256(video)':<20} {'role_norm':>10} {'plain_norm':>11} {'ratio':>8} {'authority':>10}  finished")
print("-" * 100)
for arm in summary["arms"]:
    sha = summary["outputs"].get(arm, {}).get("sha256", "?")[:16]
    eff = run / "arms" / arm / "efficiency.json"
    role = plain = ratio = authority = "-"
    when = ""
    if eff.exists():
        d = json.loads(eff.read_text(encoding="utf-8"))
        st = (d.get("full_buffer_target_chunk") or {}).get("last_sparse_role_memory_stats") or {}
        r, p = st.get("role_head_out_norm"), st.get("plain_head_out_norm")
        if isinstance(r, (int, float)) and isinstance(p, (int, float)):
            role, plain = f"{r:.6f}", f"{p:.6f}"
            ratio = f"{r / p:.6f}" if p else "inf"
        dn, hn = st.get("raw_delta_norm"), st.get("host_token_norm")
        if isinstance(dn, (int, float)) and isinstance(hn, (int, float)) and hn:
            authority = f"{dn / hn:.3e}"
        # File mtime, not total_elapsed_s: that field is polluted across a resume.
        when = datetime.fromtimestamp(eff.stat().st_mtime).strftime("%m-%d %H:%M")
    print(f"{arm:<10} {sha:<20} {role:>10} {plain:>11} {ratio:>8} {authority:>10}  {when}")

print("\nvideos:")
for arm, v in summary["outputs"].items():
    print(f"  {arm:<10} {v['video']}")
REPORT

# ---------------------------------------------------------------- 5. utility
say "5. decoded outcome per arm"
# The identity anchor is the target's first-appearance chunk, which lives in the ORIGINAL
# prefix generation, not the extension: every arm inherits it through the shared prefix, so
# it is identical across arms by construction and cannot itself carry an arm difference.
REFERENCE=$("$PYTHON" - "$CONTRACT" <<'REF'
import json, sys
from pathlib import Path

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
argv = contract["base_inference_args"]
out = Path(argv[argv.index("--output_path") + 1]) if "--output_path" in argv else None
first = int((contract.get("event") or {}).get("memory_chunk_idx", 0))
print(out / f"chunk_{first:03d}.mp4" if out else "")
REF
)
if [ -z "$FUMD" ]; then
  echo "FUMD unset, so no scoring here. When ready, from the videomem checkout:"
  echo "  python scripts/slotmem_arm_utility.py --run-dir $RUN/arms --reference $REFERENCE --out $RUN/arms/utility.json"
elif [ ! -f "$REFERENCE" ]; then
  echo "identity anchor $REFERENCE is missing; scoring skipped." >&2
  echo "It is the target's first-appearance chunk from the original prefix generation." >&2
else
  "$PYTHON" "$FUMD/scripts/slotmem_arm_utility.py" --self-check
  "$PYTHON" "$FUMD/scripts/slotmem_arm_utility.py" \
    --run-dir "$RUN/arms" --reference "$REFERENCE" --out "$RUN/arms/utility.json" \
    ${DELTA_ID:+--delta-id "$DELTA_ID"} \
    ${DYNAMIC_DEGREE_FLOOR:+--dynamic-degree-floor "$DYNAMIC_DEGREE_FLOOR"}
fi
