"""Output-video timing derived from encoded frame counts."""
from __future__ import annotations

from collections.abc import Iterable, Mapping


def generated_timing(
    previous_chunks: Iterable[Mapping],
    *,
    frames: int,
    fps: float,
) -> dict[str, float]:
    """Locate a generated chunk on the concatenated media timeline."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    start = sum(
        int(row.get("frames", 0) or 0) / float(row.get("fps", fps) or fps)
        for row in previous_chunks
        if row.get("video_saved", True)
    )
    duration = int(frames) / float(fps)
    return {
        "generated_timeline_start_s": start,
        "generated_timeline_end_s": start + duration,
        "generated_duration_s": duration,
    }
