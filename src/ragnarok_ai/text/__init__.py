"""Text utilities shared across the library: sentence segmentation."""

from __future__ import annotations

from ragnarok_ai.text.segmentation import (
    CITATION_MARKER,
    MIN_WORDS,
    RULES_VERSION,
    SEGMENTER_ID,
    SKIP_CITATION,
    SKIP_FRAGMENT,
    SKIP_NO_ALNUM,
    Segment,
    segment_sentences,
    segmenter_id,
    sentence_offsets,
    skip_reason,
)

__all__ = [
    "CITATION_MARKER",
    "MIN_WORDS",
    "RULES_VERSION",
    "SEGMENTER_ID",
    "SKIP_CITATION",
    "SKIP_FRAGMENT",
    "SKIP_NO_ALNUM",
    "Segment",
    "segment_sentences",
    "segmenter_id",
    "sentence_offsets",
    "skip_reason",
]
