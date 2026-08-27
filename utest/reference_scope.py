"""Reference selection policy for chunked SlotMem inference."""
from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from typing import Any


def image_png_bytes(image: Any) -> bytes:
    """Serialize an image canonically as RGB PNG bytes."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def image_png_sha256(image: Any) -> str:
    """Hash the RGB pixels consumed by image conditioning."""
    return hashlib.sha256(image_png_bytes(image)).hexdigest()


def build_reference_conditioning_audit(
    *,
    scope: str,
    chunk_idx: int,
    fixed_reference: Any,
    previous_frames: Sequence[Any] | None,
    random_reference: Any,
) -> dict[str, object]:
    """Describe the effective image used as the runtime random reference."""
    fixed_used = bool(
        fixed_reference is not None and random_reference is fixed_reference
    )
    fallback_used = random_reference is None and bool(previous_frames)
    effective_reference = previous_frames[0] if fallback_used else random_reference
    if fixed_used:
        source = "initial_fixed"
        source_chunk_idx = None
    elif fallback_used:
        source = "prior_chunk_tail"
        source_chunk_idx = int(chunk_idx) - 1
    else:
        source = "none"
        source_chunk_idx = None
    return {
        "fixed_reference_scope": str(scope),
        "fixed_reference_used": fixed_used,
        "random_reference_source": source,
        "random_reference_source_chunk_idx": source_chunk_idx,
        "effective_random_reference_sha256": (
            image_png_sha256(effective_reference)
            if effective_reference is not None
            else None
        ),
    }


def choose_random_reference(
    scope: str,
    chunk_idx: int,
    fixed_reference: Any,
    previous_frames: Sequence[Any] | None,
) -> Any | None:
    """Choose whether to pass the initial fixed image as random reference."""
    if scope == "all_chunks":
        return fixed_reference
    if scope == "source_only":
        return fixed_reference if int(chunk_idx) == 0 else None
    raise ValueError(f"unknown fixed reference scope: {scope}")


def validate_reference_resume(
    scope: str,
    *,
    start_chunk_idx: int,
    has_fixed_reference: bool,
    restored_previous_frames: bool,
    resume_next_chunk_idx: int | None,
) -> None:
    """Reject a resumed source-only run that would fall back to the initial image."""
    if (
        scope == "source_only"
        and int(start_chunk_idx) > 0
        and has_fixed_reference
        and not restored_previous_frames
    ):
        raise ValueError(
            "--fixed_reference_scope=source_only with --start_chunk_idx>0 requires "
            "a resume state containing previous frames"
        )
    if (
        scope == "source_only"
        and int(start_chunk_idx) > 0
        and has_fixed_reference
        and restored_previous_frames
        and resume_next_chunk_idx != int(start_chunk_idx)
    ):
        raise ValueError(
            "--fixed_reference_scope=source_only requires resume state "
            f"next_chunk_idx={int(start_chunk_idx)}; got {resume_next_chunk_idx}"
        )
