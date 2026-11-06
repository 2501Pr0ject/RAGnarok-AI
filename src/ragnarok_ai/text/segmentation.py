"""Sentence segmentation shared by the judges and the judge-training pipeline.

The rules are the ones of the RAGTruth conversion used to build the training
data, so that a judge trained on those sentences is evaluated on sentences
cut the same way:

- the text is split on line breaks first, then each line into sentences with
  pysbd (``clean=False``: the text is never altered);
- every segment keeps its character offsets in the source text, and
  ``source[start:end] == text``;
- a citation marker that follows a sentence on the same line ("(Passage 3)",
  "[Source 1]", "Passage 1 and 2:") is merged into that sentence;
- segments with no alphanumeric content, standalone citation markers, and
  fragments of fewer than ``MIN_WORDS`` words with no digit ("Sure!",
  "Ingredients:") are returned with a reason in ``skipped``. They are never
  dropped, so a caller can account for every character of the answer.

``SEGMENTER_ID`` names the segmenter, the rules version and the pysbd
version; it is recorded in ``JudgeConfig.segmenter`` so that two runs
segmented differently are never compared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pysbd  # type: ignore[import-untyped]  # pure-Python package without type information

RULES_VERSION = "1"
MIN_WORDS = 3

# A letter or digit in any script (underscore excluded)
_ALNUM = re.compile(r"[^\W_]", re.UNICODE)
# Citation markers emitted by retrieval-augmented generators: "(Passage 3)", "Source 1 and 2:"
CITATION_MARKER = re.compile(
    r"^[\(\[]?\s*(?:source|passage|reference)s?\s*\d+(?:\s*(?:,|&|and|-)\s*\d+)*\s*[\)\]]?\s*[:.]?$",
    re.IGNORECASE,
)
_WORD = re.compile(r"\w+")
_DIGIT = re.compile(r"\d")
_LINE = re.compile(r"[^\n]+")

SKIP_NO_ALNUM = "no_alphanumeric_content"
SKIP_CITATION = "citation_marker_only"
SKIP_FRAGMENT = "short_fragment"


@dataclass(frozen=True)
class Segment:
    """One sentence of a text, with its offsets and, when excluded, the reason.

    Attributes:
        text: The sentence, exactly ``source[start:end]``.
        start: Offset of the first character in the source text.
        end: Offset after the last character.
        skipped: ``None`` for a sentence to judge; otherwise the exclusion
            reason: ``no_alphanumeric_content``, ``citation_marker_only`` or
            ``short_fragment``.
    """

    text: str
    start: int
    end: int
    skipped: str | None = None

    @property
    def span(self) -> tuple[int, int]:
        """``(start, end)``."""
        return (self.start, self.end)


def segmenter_id(lang: str = "en") -> str:
    """Identifier recorded in ``JudgeConfig.segmenter``: name, rules version, pysbd version, language."""
    return f"ragnarok-sentences:{RULES_VERSION}:pysbd-{pysbd.__version__}:{lang}"


SEGMENTER_ID = segmenter_id()


def skip_reason(sentence: str) -> str | None:
    """Why *sentence* is not judged, or ``None`` when it is.

    Checked in this order: no alphanumeric character; a citation marker on
    its own; fewer than ``MIN_WORDS`` words (``\\w+`` runs) and no digit.
    """
    if not _ALNUM.search(sentence):
        return SKIP_NO_ALNUM
    if CITATION_MARKER.match(sentence):
        return SKIP_CITATION
    if len(_WORD.findall(sentence)) < MIN_WORDS and not _DIGIT.search(sentence):
        return SKIP_FRAGMENT
    return None


_SEGMENTERS: dict[str, pysbd.Segmenter] = {}


def _segmenter(lang: str) -> pysbd.Segmenter:
    if lang not in _SEGMENTERS:
        _SEGMENTERS[lang] = pysbd.Segmenter(language=lang, clean=False)
    return _SEGMENTERS[lang]


def sentence_offsets(text: str, lang: str = "en") -> list[tuple[int, int]]:
    """``(start, end)`` of every non-empty sentence of *text*, citation markers merged.

    Raises:
        RuntimeError: If pysbd returns a segment that cannot be located in
            its line (it never alters the text with ``clean=False``).
    """
    segmenter = _segmenter(lang)
    offsets: list[tuple[int, int]] = []
    for line_match in _LINE.finditer(text):
        line = line_match.group()
        base = line_match.start()
        cursor = 0
        for segment in segmenter.segment(line):
            stripped = segment.strip()
            if not stripped:
                continue
            pos = line.find(stripped, cursor)
            if pos < 0:
                msg = f"Cannot locate segment {stripped!r} in its line"
                raise RuntimeError(msg)
            start, end = base + pos, base + pos + len(stripped)
            # A trailing citation marker belongs to the sentence it cites (same line only)
            if CITATION_MARKER.match(stripped) and offsets and offsets[-1][1] >= base:
                offsets[-1] = (offsets[-1][0], end)
            else:
                offsets.append((start, end))
            cursor = pos + len(stripped)
    return offsets


def segment_sentences(text: str, lang: str = "en") -> list[Segment]:
    """Split *text* into sentences with offsets; excluded segments carry a ``skipped`` reason.

    Args:
        text: The answer to segment.
        lang: pysbd language code (default English).

    Returns:
        Segments in reading order. ``text[s.start:s.end] == s.text`` for each.
    """
    segments: list[Segment] = []
    for start, end in sentence_offsets(text, lang):
        sentence = text[start:end]
        segments.append(Segment(text=sentence, start=start, end=end, skipped=skip_reason(sentence)))
    return segments


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
