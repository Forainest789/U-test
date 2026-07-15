"""Streamlit app for blind pairwise video preference evaluation."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import streamlit as st


ROOT = Path(__file__).resolve().parent
ASSET_ROOT = Path(os.getenv("PAIRWISE_ASSET_ROOT", ROOT / "pairwise_slotmem_assets")).expanduser()
MANIFEST_PATH = os.getenv("PAIRWISE_MANIFEST", "").strip()
RESULTS_DIR = Path(os.getenv("PAIRWISE_RESULTS_DIR", ROOT / "pairwise_eval_results")).expanduser()
ALL_BASELINES = "--all-baselines" in sys.argv or os.getenv("PAIRWISE_ALL_BASELINES", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SCHEMA_VERSION = "pairwise_multidim_v2"
OPTION_LABELS = ("A", "B")
TIE_LABEL = "Tie"
ANSWER_LABELS = ("A", TIE_LABEL, "B")
CRITERIA = (
    {
        "key": "subject_consistency",
        "label": "Subject Consistency",
    },
    {
        "key": "prompt_alignment",
        "label": "Prompt Alignment",
    },
    {
        "key": "aesthetic_quality",
        "label": "Aesthetic Quality",
    },
    {
        "key": "motion_naturalness",
        "label": "Physical Naturalness",
    },
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_id(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._-") or "participant"


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@st.cache_data(show_spinner=False)
def load_manifest(path: str) -> dict:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["_manifest_path"] = str(manifest_path)
    data["_manifest_hash"] = file_digest(manifest_path)
    validate_manifest(data, manifest_path)
    return data


def combine_manifests(manifests: list[dict]) -> dict:
    if not manifests:
        raise ValueError("No manifests to combine.")

    source_meta = []
    tasks = []
    primary = manifests[0].get("primary_method", {"key": "slotmem", "display_name": "SlotMem"})
    for manifest_index, manifest in enumerate(manifests):
        source_meta.append(
            {
                "dataset_id": manifest.get("dataset_id"),
                "manifest_path": manifest.get("_manifest_path"),
                "manifest_hash": manifest.get("_manifest_hash"),
                "baseline": manifest.get("baseline"),
                "task_count": len(manifest.get("tasks", [])),
                "sample_count": manifest.get("sample_count"),
            }
        )
        for task_index, task in enumerate(manifest.get("tasks", [])):
            merged_task = json.loads(json.dumps(task, ensure_ascii=False))
            merged_task["source_manifest_index"] = manifest_index
            merged_task["source_task_index"] = task_index
            merged_task["source_dataset_id"] = manifest.get("dataset_id")
            merged_task["source_baseline"] = manifest.get("baseline")
            tasks.append(merged_task)

    hash_payload = json.dumps(source_meta, ensure_ascii=False, sort_keys=True)
    source_sample_counts = {
        item.get("sample_count")
        for item in source_meta
        if isinstance(item.get("sample_count"), int) and item.get("sample_count") > 0
    }
    sample_count = source_sample_counts.pop() if len(source_sample_counts) == 1 else None
    return {
        "dataset_id": "slotmem_pairwise_all_baselines",
        "created_from": source_meta,
        "primary_method": primary,
        "baseline": {
            "key": "all_baselines",
            "display_name": "All Baselines",
        },
        "sample_count": sample_count,
        "task_count": len(tasks),
        "tasks": tasks,
        "_manifest_path": str(ASSET_ROOT / "manifests" / "all_baselines.json"),
        "_manifest_hash": text_digest(hash_payload),
        "_blind_set_label": "Evaluation Set 1",
    }


def validate_manifest(data: dict, path: Path) -> None:
    primary_key = data.get("primary_method", {}).get("key", "slotmem")
    baseline_key = data.get("baseline", {}).get("key")
    tasks = data.get("tasks")
    errors: list[str] = []

    if not baseline_key:
        errors.append("missing baseline.key")
    if not isinstance(tasks, list) or not tasks:
        errors.append("missing tasks")

    expected_methods = {primary_key, baseline_key}
    for idx, task in enumerate(tasks if isinstance(tasks, list) else []):
        options = task.get("options") if isinstance(task, dict) else None
        if not isinstance(options, list) or len(options) != 2:
            errors.append(f"task {idx}: expected exactly two options")
            continue
        labels = [opt.get("label") for opt in options if isinstance(opt, dict)]
        methods = {opt.get("method_key") for opt in options if isinstance(opt, dict)}
        if sorted(labels) != ["A", "B"]:
            errors.append(f"task {idx}: labels must be A/B")
        if methods != expected_methods:
            errors.append(f"task {idx}: methods must be primary vs baseline")

    if errors:
        preview = "; ".join(errors[:5])
        raise ValueError(f"Invalid pairwise manifest {path}: {preview}")


@st.cache_data(show_spinner=False, max_entries=32)
def video_data_uri(path: str) -> str:
    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{payload}"


def discover_manifests() -> list[Path]:
    if MANIFEST_PATH and not ALL_BASELINES:
        return [Path(MANIFEST_PATH).expanduser()]
    manifest_dir = ASSET_ROOT / "manifests"
    if not manifest_dir.is_dir():
        return []
    return sorted(manifest_dir.glob("*.json"))


def load_evaluation_sets(paths: list[Path]) -> list[dict]:
    loaded = []
    for set_index, path in enumerate(paths, start=1):
        try:
            manifest = load_manifest(str(path))
        except Exception:
            st.error(f"Failed to load evaluation set {set_index}. Please ask the study administrator to check it.")
            continue
        label = f"Evaluation Set {set_index}"
        manifest["_blind_set_label"] = label
        loaded.append(manifest)

    if ALL_BASELINES and loaded:
        return [combine_manifests(loaded)]
    return loaded


def result_path_for(manifest: dict, participant_id: str) -> Path:
    manifest_stem = Path(manifest["_manifest_path"]).stem
    return RESULTS_DIR / f"{safe_id(participant_id)}__{safe_id(manifest_stem)}.json"


def write_result(data: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(data["result_path"]).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_responses(data: dict) -> None:
    responses = data.get("responses")
    if isinstance(responses, list):
        data["responses"] = {
            str(entry.get("task_index")): entry
            for entry in responses
            if isinstance(entry, dict) and entry.get("task_index") is not None
        }
    elif not isinstance(responses, dict):
        data["responses"] = {}


def load_or_create_run(manifest: dict, participant_id: str) -> dict:
    result_path = result_path_for(manifest, participant_id)
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        normalize_responses(existing)
        if (
            existing.get("manifest_hash") == manifest["_manifest_hash"]
            and existing.get("participant_id") == participant_id
        ):
            existing["result_path"] = str(result_path)
            existing.setdefault("schema_version", SCHEMA_VERSION)
            return existing

    data = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": manifest.get("dataset_id"),
        "manifest_path": manifest["_manifest_path"],
        "manifest_hash": manifest["_manifest_hash"],
        "created_from": manifest.get("created_from"),
        "participant_id": participant_id,
        "result_path": str(result_path),
        "started_at_utc": now_iso(),
        "completed_at_utc": None,
        "responses": {},
    }
    write_result(data)
    return data


def response_is_complete(response: dict | None) -> bool:
    if not isinstance(response, dict):
        return False
    answers = response.get("answers")
    if not isinstance(answers, dict):
        return False
    return all(answers.get(criterion["key"]) in ANSWER_LABELS for criterion in CRITERIA)


def is_answered(data: dict, idx: int) -> bool:
    return response_is_complete(get_response(data, idx))


def answered_count(data: dict, manifest: dict) -> int:
    return sum(1 for idx in range(len(manifest["tasks"])) if is_answered(data, idx))


def get_response(data: dict, idx: int) -> dict | None:
    value = data.get("responses", {}).get(str(idx))
    return value if isinstance(value, dict) else None


def first_unanswered_index(data: dict, manifest: dict) -> int:
    for idx in range(len(manifest["tasks"])):
        if not is_answered(data, idx):
            return idx
    return 0


def next_index_after(data: dict, manifest: dict, idx: int) -> int:
    total = len(manifest["tasks"])
    for offset in range(1, total + 1):
        candidate = (idx + offset) % total
        if not is_answered(data, candidate):
            return candidate
    return min(idx + 1, total - 1)


def set_run(manifest: dict, data: dict) -> None:
    st.session_state.manifest = manifest
    st.session_state.run_data = data
    st.session_state.current_index = first_unanswered_index(data, manifest)
    st.session_state.task_started_at = time.time()


def current_task_index() -> int | None:
    manifest = st.session_state.get("manifest")
    if not manifest:
        return None
    idx = int(st.session_state.get("current_index", 0))
    return idx if 0 <= idx < len(manifest["tasks"]) else None


def go_to_task(idx: int) -> None:
    manifest = st.session_state.manifest
    if not 0 <= idx < len(manifest["tasks"]):
        return
    st.session_state.current_index = idx
    st.session_state.task_started_at = time.time()


def save_current_answers(answers: dict[str, str]) -> None:
    manifest = st.session_state.manifest
    data = st.session_state.run_data
    idx = current_task_index()
    if idx is None:
        return

    task = manifest["tasks"][idx]
    option_map = {opt["label"]: opt for opt in task["options"]}
    previous = get_response(data, idx)

    def selected_method(answer: str) -> dict[str, str] | None:
        if answer == TIE_LABEL:
            return {"method_key": "tie", "method_display_name": "Tie"}
        option = option_map.get(answer)
        if option:
            return {
                "method_key": option["method_key"],
                "method_display_name": option["method_display_name"],
            }
        return None

    selected_methods = {
        criterion["key"]: selected_method(answers.get(criterion["key"], ""))
        for criterion in CRITERIA
    }
    missing = [key for key, value in selected_methods.items() if value is None]
    if missing:
        return

    chosen_method_keys = {
        criterion["key"]: selected_methods[criterion["key"]]["method_key"] for criterion in CRITERIA
    }
    chosen_method_display_names = {
        criterion["key"]: selected_methods[criterion["key"]]["method_display_name"] for criterion in CRITERIA
    }
    data["responses"][str(idx)] = {
        "schema_version": SCHEMA_VERSION,
        "task_index": idx,
        "task_id": task["task_id"],
        "sample_id": task["sample_id"],
        "source_dataset_id": task.get("source_dataset_id"),
        "source_baseline": task.get("source_baseline"),
        "answers": {criterion["key"]: answers[criterion["key"]] for criterion in CRITERIA},
        "chosen_method_keys": chosen_method_keys,
        "chosen_method_display_names": chosen_method_display_names,
        "dimensions": {
            criterion["key"]: {
                "label": criterion["label"],
                "answer_label": answers[criterion["key"]],
                "chosen_method_key": chosen_method_keys[criterion["key"]],
                "chosen_method_display_name": chosen_method_display_names[criterion["key"]],
            }
            for criterion in CRITERIA
        },
        "criteria": [
            {
                "key": criterion["key"],
                "label": criterion["label"],
                "answer": answers[criterion["key"]],
                "chosen_method_key": chosen_method_keys[criterion["key"]],
                "chosen_method_display_name": chosen_method_display_names[criterion["key"]],
            }
            for criterion in CRITERIA
        ],
        "option_method_map": {
            opt["label"]: {
                "method_key": opt["method_key"],
                "method_display_name": opt["method_display_name"],
                "video_path": opt["video_path"],
            }
            for opt in task["options"]
        },
        "elapsed_sec": round(time.time() - st.session_state.task_started_at, 3),
        "answered_at_utc": now_iso(),
        "redo_count": int(previous.get("redo_count", 0)) + 1 if previous else 0,
    }

    if answered_count(data, manifest) >= len(manifest["tasks"]):
        data["completed_at_utc"] = now_iso()
    else:
        data["completed_at_utc"] = None

    write_result(data)
    st.session_state.run_data = data
    st.session_state.current_index = next_index_after(data, manifest, idx)
    st.session_state.task_started_at = time.time()
    st.rerun()


def render_synced_player(task: dict) -> None:
    opt_a, opt_b = sorted(task["options"], key=lambda item: item["label"])
    src_a = video_data_uri(opt_a["video_path"])
    src_b = video_data_uri(opt_b["video_path"])
    segments = json.dumps(task.get("prompt_segments", []), ensure_ascii=False)
    duration = float(task.get("duration_s", 0.0) or 0.0)
    element_id = f"pairwise_{safe_id(task['task_id'])}"

    doc = f"""
    <div id="{element_id}" class="pairwise-root">
      <div class="prompt-header" id="{element_id}_chunk">Chunk 1/1</div>
      <div class="prompt-text" id="{element_id}_prompt"></div>
      <div class="video-grid">
        <div>
          <div class="candidate-label">A</div>
          <video id="{element_id}_a" preload="metadata" playsinline muted src="{src_a}"></video>
        </div>
        <div>
          <div class="candidate-label">B</div>
          <video id="{element_id}_b" preload="metadata" playsinline muted src="{src_b}"></video>
        </div>
      </div>
      <div class="controls">
        <button id="{element_id}_play">Play</button>
        <button id="{element_id}_pause">Pause</button>
        <button id="{element_id}_back">-2s</button>
        <button id="{element_id}_fwd">+2s</button>
        <button id="{element_id}_replay">Replay</button>
      </div>
      <div class="timeline">
        <input class="scrubber" id="{element_id}_scrub" type="range" min="0" max="{duration:.4f}" step="0.05" value="0" />
        <div class="chunk-markers" id="{element_id}_markers"></div>
      </div>
      <div class="time-row"><span id="{element_id}_time">0.00</span>s</div>
    </div>
    <style>
      .pairwise-root {{
        color: #222;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      .prompt-header {{
        color: #3f444a;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0;
        margin-bottom: 0.35rem;
      }}
      .prompt-text {{
        border: 1px solid #d6d8dc;
        background: #f7f8fa;
        min-height: 4.2rem;
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.8rem;
        font-size: 1rem;
        line-height: 1.45;
      }}
      .video-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.8rem;
        align-items: start;
      }}
      video {{
        width: 100%;
        aspect-ratio: 832 / 480;
        background: #000;
        display: block;
      }}
      .candidate-label {{
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
      }}
      .controls {{
        display: flex;
        gap: 0.45rem;
        margin-top: 0.75rem;
        align-items: center;
      }}
      button {{
        border: 1px solid #b7bcc5;
        background: #fff;
        border-radius: 0.35rem;
        cursor: pointer;
        font-size: 0.92rem;
        padding: 0.45rem 0.75rem;
      }}
      .scrubber {{
        display: block;
        width: 100%;
        margin: 0;
      }}
      .timeline {{
        margin-top: 0.55rem;
        position: relative;
      }}
      .chunk-markers {{
        inset: 0;
        pointer-events: none;
        position: absolute;
      }}
      .chunk-marker {{
        background: rgba(31, 41, 51, 0.72);
        border-radius: 999px;
        height: 1.35rem;
        position: absolute;
        top: 50%;
        transform: translate(-50%, -50%);
        width: 2px;
      }}
      .time-row {{
        color: #555;
        font-size: 0.86rem;
        margin-top: 0.05rem;
      }}
    </style>
    <script>
      (() => {{
        const root = document.getElementById("{element_id}");
        const videos = [
          document.getElementById("{element_id}_a"),
          document.getElementById("{element_id}_b"),
        ];
        const promptEl = document.getElementById("{element_id}_prompt");
        const chunkEl = document.getElementById("{element_id}_chunk");
        const scrub = document.getElementById("{element_id}_scrub");
        const markersEl = document.getElementById("{element_id}_markers");
        const timeEl = document.getElementById("{element_id}_time");
        const segments = {segments};
        const targetDuration = {duration:.4f};

        function clampTime(t) {{
          return Math.max(0, Math.min(t, targetDuration || t));
        }}
        function segmentIndexFor(t) {{
          if (!segments.length) return -1;
          for (const seg of segments) {{
            const idx = segments.indexOf(seg);
            if (t >= seg.start_s && t < seg.end_s) return idx;
          }}
          return segments.length - 1;
        }}
        function renderMarkers() {{
          markersEl.innerHTML = "";
          if (!segments.length || !targetDuration) return;
          for (let idx = 1; idx < segments.length; idx += 1) {{
            const boundary = Number(segments[idx].start_s || 0);
            const pct = Math.max(0, Math.min(100, boundary / targetDuration * 100));
            const marker = document.createElement("div");
            marker.className = "chunk-marker";
            marker.style.left = `${{pct}}%`;
            marker.title = `Chunk ${{idx + 1}} starts`;
            markersEl.appendChild(marker);
          }}
        }}
        function updatePrompt(t) {{
          const segmentIdx = segmentIndexFor(t);
          const segment = segmentIdx >= 0 ? segments[segmentIdx] : null;
          promptEl.textContent = segment ? (segment.text || "") : "";
          chunkEl.textContent = segments.length ? `Chunk ${{segmentIdx + 1}}/${{segments.length}}` : "Chunk 0/0";
          scrub.value = String(clampTime(t));
          timeEl.textContent = clampTime(t).toFixed(2);
        }}
        function seekBoth(t) {{
          const nextT = clampTime(t);
          for (const v of videos) {{
            try {{ v.currentTime = nextT; }} catch (e) {{}}
          }}
          updatePrompt(nextT);
        }}
        function currentT() {{
          return videos[0].currentTime || videos[1].currentTime || 0;
        }}
        document.getElementById("{element_id}_play").onclick = () => {{
          const t = currentT();
          seekBoth(t);
          for (const v of videos) v.play();
        }};
        document.getElementById("{element_id}_pause").onclick = () => {{
          for (const v of videos) v.pause();
        }};
        document.getElementById("{element_id}_back").onclick = () => seekBoth(currentT() - 2);
        document.getElementById("{element_id}_fwd").onclick = () => seekBoth(currentT() + 2);
        document.getElementById("{element_id}_replay").onclick = () => {{
          seekBoth(0);
          for (const v of videos) v.play();
        }};
        scrub.oninput = () => seekBoth(Number(scrub.value));
        videos[0].ontimeupdate = () => {{
          const t = currentT();
          updatePrompt(t);
          for (const v of videos.slice(1)) {{
            if (!v.paused && Math.abs(v.currentTime - t) > 0.12) {{
              v.currentTime = t;
            }}
          }}
        }};
        for (const v of videos) {{
          v.onended = () => videos.forEach(x => x.pause());
        }}
        renderMarkers();
        updatePrompt(0);
      }})();
    </script>
    """
    st.iframe(doc, height=700)


def render_task(manifest: dict, data: dict) -> None:
    idx = current_task_index()
    total = len(manifest["tasks"])
    if total == 0 or idx is None:
        st.error("No tasks found in the selected manifest.")
        return

    task = manifest["tasks"][idx]
    existing = get_response(data, idx)
    completed = answered_count(data, manifest)
    st.progress(completed / total, text=f"Answered {completed} of {total}")
    st.markdown(f"### Question {idx + 1} of {total}")
    render_synced_player(task)

    existing_answers = existing.get("answers", {}) if isinstance(existing, dict) else {}
    if response_is_complete(existing):
        saved = ", ".join(
            f"{criterion['label']}: {existing_answers[criterion['key']]}" for criterion in CRITERIA
        )
        st.info(f"Saved ratings: {saved}")

    with st.form(f"ratings-{task['task_id']}-{idx}", clear_on_submit=False):
        answers: dict[str, str | None] = {}
        for criterion in CRITERIA:
            current = existing_answers.get(criterion["key"])
            index = ANSWER_LABELS.index(current) if current in ANSWER_LABELS else None
            answers[criterion["key"]] = st.radio(
                criterion["label"],
                ANSWER_LABELS,
                index=index,
                horizontal=True,
                key=f"{task['task_id']}-{criterion['key']}",
            )
        submitted = st.form_submit_button("Save ratings and continue", type="primary", use_container_width=True)

    if submitted:
        missing_labels = [
            criterion["label"] for criterion in CRITERIA if answers.get(criterion["key"]) not in ANSWER_LABELS
        ]
        if missing_labels:
            st.warning("Please choose A, Tie, or B for: " + ", ".join(missing_labels))
        else:
            save_current_answers({key: value for key, value in answers.items() if value in ANSWER_LABELS})


def render_question_nav(manifest: dict, data: dict) -> None:
    st.markdown("Question List")
    current = current_task_index()
    for start in range(0, len(manifest["tasks"]), 4):
        cols = st.columns(4)
        for offset, col in enumerate(cols):
            idx = start + offset
            if idx >= len(manifest["tasks"]):
                continue
            answered = is_answered(data, idx)
            label = str(idx + 1)
            if idx == current:
                col.markdown(f"<div class='current-task-cell'>{label}</div>", unsafe_allow_html=True)
                continue
            if answered:
                label = f"✓ {label}"
            col.button(label, key=f"nav-{idx}", use_container_width=True, on_click=go_to_task, args=(idx,))


def render_start() -> None:
    st.title("Pairwise Video Preference Evaluation")
    manifests = discover_manifests()
    if not manifests:
        st.error(f"No manifests found under {ASSET_ROOT / 'manifests'}")
        return

    loaded = load_evaluation_sets(manifests)
    if not loaded:
        st.error("No valid pairwise evaluation manifests were found.")
        return

    labels = [manifest.get("_blind_set_label", f"Evaluation Set {idx}") for idx, manifest in enumerate(loaded, start=1)]
    selected = st.selectbox("Evaluation set", labels)
    participant_id = st.text_input("Participant ID", value="participant")
    manifest = loaded[labels.index(selected)]
    sample_count = manifest.get("sample_count")
    if isinstance(sample_count, int) and sample_count > 0:
        st.write(f"Samples: {sample_count}")
        st.write(f"Pairwise questions: {len(manifest.get('tasks', []))}")
    else:
        st.write(f"Pairwise questions: {len(manifest.get('tasks', []))}")
    if st.button("Start / Resume", use_container_width=True):
        data = load_or_create_run(manifest, participant_id)
        set_run(manifest, data)
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Pairwise Video Preference", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
            max-width: 1500px;
        }
        .current-task-cell {
            align-items: center;
            background-color: #3f444a;
            border-radius: 0.35rem;
            color: white;
            display: flex;
            font-size: 0.9rem;
            font-weight: 600;
            justify-content: center;
            min-height: 2.45rem;
            width: 100%;
        }
        div.stButton > button[kind="primary"] {
            background-color: #198754;
            border-color: #198754;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "run_data" not in st.session_state or "manifest" not in st.session_state:
        render_start()
        return

    manifest = st.session_state.manifest
    data = st.session_state.run_data
    with st.sidebar:
        set_label = manifest.get("_blind_set_label", "Evaluation Set")
        st.write(f"Set: `{set_label}`")
        st.write(f"Participant: `{data.get('participant_id')}`")
        st.write(f"Progress: {answered_count(data, manifest)}/{len(manifest['tasks'])}")
        st.divider()
        render_question_nav(manifest, data)

    render_task(manifest, data)


if __name__ == "__main__":
    main()
