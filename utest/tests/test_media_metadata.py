from __future__ import annotations

from utest.media_metadata import generated_timing


def test_generated_timing_uses_saved_frames_not_source_caption_times() -> None:
    previous = [
        {"frames": 76, "fps": 16, "video_saved": True, "start": 0.0, "end": 3.375},
        {"frames": 76, "fps": 16, "video_saved": True, "start": 3.375, "end": 6.75},
    ]
    assert generated_timing(previous, frames=81, fps=16) == {
        "generated_timeline_start_s": 9.5,
        "generated_timeline_end_s": 14.5625,
        "generated_duration_s": 5.0625,
    }
