"""Plot pairwise preference results as one multi-panel figure.

Each input JSON is treated as one model row. The figure contains the evaluation
evaluation dimensions as side-by-side panels. Each row is a thick horizontal
stacked bar:

  win | tie | lose

Here "win" means the primary method, SlotMem by default, wins. Ties are
silently treated as zero when the result file has no tie option.

Run:
  python plot_pairwise_results.py
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ROOT / "pairwise_eval_results"
DEFAULT_OUTPUT_DIR = ROOT / "pairwise_eval_plots"
DEFAULT_PRIMARY_METHOD = "slotmem"
BASELINE_ORDER = {
    "wan22_native": 0,
    "storydiffusion": 1,
    "storymem": 2,
    "iamflow_i2v_kvselfattn": 3,
}
ROW_LABELS = {
    "wan22_native": "Wan22-I2V",
    "storydiffusion": "+StoryDiffusion",
    "storymem": "+StoryMem",
    "iamflow_i2v_kvselfattn": "+IAMFlow",
}

CRITERIA = (
    ("subject_consistency", "Subject Consistency"),
    ("prompt_alignment", "Prompt Alignment"),
    ("aesthetic_quality", "Aesthetic Quality"),
    ("motion_naturalness", "Physical Naturalness"),
)
SEGMENTS = (
    ("win", "win", "#F7B267"),
    ("tie", "tie", "#D5D8DE"),
    ("lose", "lose", "#9EC5E8"),
)
BAR_TEXT_COLOR = "#1F2933"


def normalize_responses(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [item for _, item in sorted(value.items(), key=lambda kv: str(kv[0])) if isinstance(item, dict)]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def method_display_lookup(responses: list[dict[str, Any]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for response in responses:
        option_map = response.get("option_method_map")
        if not isinstance(option_map, dict):
            continue
        for option in option_map.values():
            if not isinstance(option, dict):
                continue
            key = option.get("method_key")
            name = option.get("method_display_name")
            if isinstance(key, str) and key:
                lookup[key] = name if isinstance(name, str) and name else key
    return lookup


def infer_methods(responses: list[dict[str, Any]], primary_method: str) -> tuple[str, str, dict[str, str]]:
    displays = method_display_lookup(responses)
    method_keys = set(displays)
    for response in responses:
        chosen = response.get("chosen_method_keys")
        if isinstance(chosen, dict):
            method_keys.update(key for key in chosen.values() if isinstance(key, str) and key)

    if primary_method in method_keys and len(method_keys) >= 2:
        primary_key = primary_method
        opponent_key = sorted(method_keys - {primary_method})[0]
    else:
        sorted_keys = sorted(method_keys)
        if len(sorted_keys) < 2:
            raise ValueError("Could not infer a two-method comparison from the result file.")
        opponent_key, primary_key = sorted_keys[:2]

    displays.setdefault(opponent_key, opponent_key)
    displays.setdefault(primary_key, primary_key)
    return opponent_key, primary_key, displays


def count_responses(
    *,
    path: Path,
    data: dict[str, Any],
    responses: list[dict[str, Any]],
    primary_method: str,
    opponent_key: str | None = None,
    opponent_name: str | None = None,
) -> dict[str, Any]:
    if opponent_key:
        primary_key = primary_method
        displays = method_display_lookup(responses)
        displays.setdefault(primary_key, primary_key)
        displays.setdefault(opponent_key, opponent_name or opponent_key)
    else:
        opponent_key, primary_key, displays = infer_methods(responses, primary_method)
    criterion_counts: dict[str, Counter[str]] = {key: Counter({"win": 0, "tie": 0, "lose": 0}) for key, _ in CRITERIA}
    criterion_totals: Counter[str] = Counter()

    for response in responses:
        chosen = response.get("chosen_method_keys")
        if not isinstance(chosen, dict):
            continue
        for criterion_key, _ in CRITERIA:
            method_key = chosen.get(criterion_key)
            if method_key == primary_key:
                criterion_counts[criterion_key]["win"] += 1
            elif method_key == opponent_key:
                criterion_counts[criterion_key]["lose"] += 1
            elif method_key in {"tie", "draw", "equal"}:
                criterion_counts[criterion_key]["tie"] += 1
            else:
                continue
            criterion_totals[criterion_key] += 1

    return {
        "path": str(path),
        "stem": path.stem,
        "dataset_id": data.get("dataset_id"),
        "participant_id": data.get("participant_id"),
        "opponent_key": opponent_key,
        "primary_key": primary_key,
        "opponent_name": displays[opponent_key],
        "primary_name": displays[primary_key],
        "counts": {key: dict(value) for key, value in criterion_counts.items()},
        "totals": dict(criterion_totals),
    }


def response_source_baseline(response: dict[str, Any], primary_method: str) -> tuple[str, str] | None:
    source = response.get("source_baseline")
    if isinstance(source, dict):
        key = source.get("key")
        name = source.get("display_name")
        if isinstance(key, str) and key:
            return key, name if isinstance(name, str) and name else key

    displays = method_display_lookup([response])
    opponent_keys = sorted(key for key in displays if key != primary_method)
    if opponent_keys:
        key = opponent_keys[0]
        return key, displays[key]
    return None


def count_result_file(path: Path, primary_method: str) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    responses = normalize_responses(data.get("responses"))
    dataset_id = data.get("dataset_id")
    if dataset_id == "slotmem_pairwise_all_baselines":
        grouped: dict[str, dict[str, Any]] = {}
        for response in responses:
            source = response_source_baseline(response, primary_method)
            if not source:
                continue
            key, name = source
            group = grouped.setdefault(key, {"name": name, "responses": []})
            group["responses"].append(response)
        rows = [
            count_responses(
                path=path,
                data=data,
                responses=group["responses"],
                primary_method=primary_method,
                opponent_key=key,
                opponent_name=group["name"],
            )
            for key, group in sorted(grouped.items(), key=lambda item: (BASELINE_ORDER.get(item[0], 99), item[0]))
        ]
        for row in rows:
            row["stem"] = f"{path.stem}__{row['opponent_key']}"
        return rows

    return [count_responses(path=path, data=data, responses=responses, primary_method=primary_method)]


def copied_label(stem: str) -> str:
    label = stem.split("__", 1)[-1]
    return label.replace("storymem", "storyMem")


def display_row_label(row: dict[str, Any], fallback_label: str) -> str:
    key = row.get("opponent_key")
    if isinstance(key, str) and key in ROW_LABELS:
        return ROW_LABELS[key]
    return fallback_label


def disambiguate_labels(rows: list[dict[str, Any]]) -> None:
    seen: Counter[str] = Counter()
    for row in rows:
        base_label = row["opponent_name"]
        seen[base_label] += 1
        fallback = base_label if seen[base_label] == 1 else copied_label(row["stem"])
        row["row_label"] = display_row_label(row, fallback)


def collect_rows(results_dir: Path, primary_method: str, result_pattern: str) -> list[dict[str, Any]]:
    paths = sorted(results_dir.glob(result_pattern))
    rows = [row for path in paths for row in count_result_file(path, primary_method)]
    rows = [row for row in rows if any(row["totals"].get(key, 0) > 0 for key, _ in CRITERIA)]
    disambiguate_labels(rows)
    return rows


def plot_combined(rows: list[dict[str, Any]], *, output_dir: Path, formats: list[str]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("No plottable result rows found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_height = max(2.8, 0.50 * len(rows) + 0.85)
    fig, axes = plt.subplots(1, len(CRITERIA), figsize=(13.8, fig_height), sharey=True)
    if len(CRITERIA) == 1:
        axes = [axes]

    y_positions = list(range(len(rows)))
    summary: list[dict[str, Any]] = []

    for panel_idx, (ax, (criterion_key, criterion_label)) in enumerate(zip(axes, CRITERIA)):
        offsets = [0.0 for _ in rows]
        for segment_key, segment_label, color in SEGMENTS:
            values = []
            for row in rows:
                total = row["totals"].get(criterion_key, 0)
                count = row["counts"].get(criterion_key, {}).get(segment_key, 0)
                values.append(count / total if total else 0.0)

            ax.barh(
                y_positions,
                values,
                left=offsets,
                height=0.66,
                color=color,
                edgecolor="none",
                linewidth=0,
                label=segment_label,
            )
            for row_idx, value in enumerate(values):
                if value >= 0.15:
                    ax.text(
                        offsets[row_idx] + value / 2,
                        row_idx,
                        f"{value * 100:.1f}",
                        ha="center",
                        va="center",
                        color=BAR_TEXT_COLOR,
                        fontsize=10,
                        fontweight="bold",
                    )
                offsets[row_idx] += value

        criterion_rows = []
        for row in rows:
            counts = row["counts"].get(criterion_key, {})
            criterion_rows.append(
                {
                    "row_label": row["row_label"],
                    "primary_model": row["primary_name"],
                    "opponent_model": row["opponent_name"],
                    "n": row["totals"].get(criterion_key, 0),
                    "wins": counts.get("win", 0),
                    "ties": counts.get("tie", 0),
                    "losses": counts.get("lose", 0),
                }
            )
        summary.append({"criterion": criterion_key, "label": criterion_label, "rows": criterion_rows})

        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_xlabel(criterion_label, fontsize=12, fontweight="bold", labelpad=2)
        ax.grid(False)
        ax.tick_params(axis="x", length=0, labelbottom=False)
        ax.tick_params(axis="y", length=0)
        for spine in ("top", "right", "bottom", "left"):
            ax.spines[spine].set_visible(False)
        if panel_idx == 0:
            ax.set_yticks(y_positions)
            ax.set_yticklabels([row["row_label"] for row in rows], fontsize=12, fontweight="bold")
        else:
            ax.tick_params(labelleft=False)

    axes[0].invert_yaxis()
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        frameon=False,
        handlelength=1.9,
        handletextpad=1.0,
        labelspacing=1.2,
        prop={"size": 11, "weight": "bold"},
    )
    fig.subplots_adjust(left=0.13, right=0.87, top=0.88, bottom=0.16, wspace=0.07)

    written = []
    for fmt in formats:
        out_path = output_dir / f"pairwise_preference_summary.{fmt}"
        fig.savefig(out_path, dpi=240, bbox_inches="tight")
        written.append(str(out_path))
    plt.close(fig)
    summary.append({"files": written})
    return summary


def write_summary(summary: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--primary-method", default=DEFAULT_PRIMARY_METHOD)
    parser.add_argument("--result-pattern", default="*.json", help="Glob pattern under --results-dir.")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated output formats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = [item.strip().lstrip(".") for item in args.formats.split(",") if item.strip()]
    rows = collect_rows(args.results_dir.expanduser(), args.primary_method, args.result_pattern)
    summary = plot_combined(rows, output_dir=args.output_dir.expanduser(), formats=formats)
    summary_path = write_summary(summary, args.output_dir.expanduser())
    print(f"[OK] plotted {len(rows)} result rows")
    for path in summary[-1]["files"]:
        print(path)
    print(summary_path)


if __name__ == "__main__":
    main()
