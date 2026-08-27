import pytest
from PIL import Image

from utest.reference_scope import (
    build_reference_conditioning_audit,
    choose_random_reference,
    image_png_sha256,
    validate_reference_resume,
)


def test_native_scope_keeps_initial_reference() -> None:
    assert choose_random_reference("all_chunks", 6, "initial", ["absence"]) == "initial"


def test_source_only_scope_blocks_initial_reference_after_chunk_zero() -> None:
    assert choose_random_reference("source_only", 0, "initial", ["initial"]) == "initial"
    assert choose_random_reference("source_only", 6, "initial", ["absence"]) is None


def test_reference_scope_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="fixed reference scope"):
        choose_random_reference("leaky", 1, "initial", ["absence"])


def test_source_only_resume_requires_restored_continuity_frames() -> None:
    with pytest.raises(ValueError, match="resume state containing previous frames"):
        validate_reference_resume(
            "source_only",
            start_chunk_idx=6,
            has_fixed_reference=True,
            restored_previous_frames=False,
            resume_next_chunk_idx=6,
        )


def test_source_only_resume_requires_aligned_continuity_frames() -> None:
    validate_reference_resume(
        "source_only",
        start_chunk_idx=6,
        has_fixed_reference=True,
        restored_previous_frames=True,
        resume_next_chunk_idx=6,
    )

    with pytest.raises(ValueError, match="next_chunk_idx=6"):
        validate_reference_resume(
            "source_only",
            start_chunk_idx=6,
            has_fixed_reference=True,
            restored_previous_frames=True,
            resume_next_chunk_idx=3,
        )


def test_image_hash_is_canonical_for_equivalent_rgb_pixels() -> None:
    rgb = Image.new("RGB", (2, 1), (10, 20, 30))
    rgba = Image.new("RGBA", (2, 1), (10, 20, 30, 255))

    assert image_png_sha256(rgb) == image_png_sha256(rgba)
    assert len(image_png_sha256(rgb)) == 64


def test_reference_audit_identifies_initial_fixed_image() -> None:
    initial = Image.new("RGB", (1, 1), "red")
    prior_tail = Image.new("RGB", (1, 1), "blue")

    assert build_reference_conditioning_audit(
        scope="all_chunks",
        chunk_idx=6,
        fixed_reference=initial,
        previous_frames=[prior_tail],
        random_reference=initial,
    ) == {
        "fixed_reference_scope": "all_chunks",
        "fixed_reference_used": True,
        "random_reference_source": "initial_fixed",
        "random_reference_source_chunk_idx": None,
        "effective_random_reference_sha256": image_png_sha256(initial),
    }


def test_reference_audit_hashes_source_only_prior_tail_fallback() -> None:
    initial = Image.new("RGB", (1, 1), "red")
    prior_tail = Image.new("RGB", (1, 1), "blue")

    audit = build_reference_conditioning_audit(
        scope="source_only",
        chunk_idx=6,
        fixed_reference=initial,
        previous_frames=[prior_tail],
        random_reference=None,
    )

    assert audit["fixed_reference_used"] is False
    assert audit["random_reference_source"] == "prior_chunk_tail"
    assert audit["random_reference_source_chunk_idx"] == 5
    assert audit["effective_random_reference_sha256"] == image_png_sha256(prior_tail)
