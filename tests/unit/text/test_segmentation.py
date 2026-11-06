"""Sentence segmentation: rules, invariants, and regression against the external reference."""

from __future__ import annotations

import json
from pathlib import Path

import pysbd
import pytest

from ragnarok_ai.text import (
    CITATION_MARKER,
    MIN_WORDS,
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

FIXTURE = Path(__file__).parent / "fixtures" / "segmentation_reference.json"
REFERENCE = json.loads(FIXTURE.read_text(encoding="utf-8"))


def case_ids() -> list[str]:
    return [c["name"] for c in REFERENCE["cases"]]


class TestReferenceRegression:
    def test_fixture_comes_from_the_same_pysbd_and_rules(self):
        ref = REFERENCE["reference"]
        assert ref["pysbd_version"] == pysbd.__version__
        assert ref["min_words"] == MIN_WORDS
        assert "path" not in ref and all("/" not in str(v) or v.startswith("tests/") for v in ref.values())
        assert len(REFERENCE["cases"]) >= 10

    @pytest.mark.parametrize("case", REFERENCE["cases"], ids=case_ids())
    def test_offsets_text_and_exclusions_match_the_reference(self, case):
        """Same segments, same offsets, same exclusion reasons as the conversion script."""
        ours = [
            {"start": s.start, "end": s.end, "text": s.text, "skipped": s.skipped}
            for s in segment_sentences(case["text"])
        ]
        assert ours == case["segments"]


class TestInvariants:
    @pytest.mark.parametrize("case", REFERENCE["cases"], ids=case_ids())
    def test_segments_are_exact_slices_ordered_and_disjoint(self, case):
        text = case["text"]
        segments = segment_sentences(text)
        previous_end = -1
        for s in segments:
            assert text[s.start : s.end] == s.text
            assert s.text == s.text.strip()
            assert s.start >= previous_end  # disjoint; adjacent segments are allowed
            previous_end = s.end
        assert [s.span for s in segments] == sentence_offsets(text)

    def test_nothing_is_dropped_only_marked(self):
        segments = segment_sentences("Sure!\n---\nPassage 2:\nA complete sentence with words.")
        assert [s.skipped for s in segments] == [SKIP_FRAGMENT, SKIP_NO_ALNUM, SKIP_CITATION, None]

    def test_empty_or_blank_text_gives_no_segment(self):
        assert segment_sentences("") == []
        assert segment_sentences("\n\n  \n") == []
        assert Segment(text="x", start=0, end=1).span == (0, 1)


class TestRules:
    @pytest.mark.parametrize(
        ("sentence", "expected"),
        [
            ("---", SKIP_NO_ALNUM),
            ("...", SKIP_NO_ALNUM),
            ("(Passage 3)", SKIP_CITATION),
            ("[Source 1]", SKIP_CITATION),
            ("Passage 1 and 2:", SKIP_CITATION),
            ("References 4, 5.", SKIP_CITATION),
            ("Sure!", SKIP_FRAGMENT),
            ("Ingredients:", SKIP_FRAGMENT),
            ("Thanks a lot", None),  # three words
            ("Step 2.", None),  # a digit keeps a short fragment
            ("Version 2", None),
            ("The daemon reads its configuration.", None),
        ],
    )
    def test_skip_reason(self, sentence, expected):
        assert skip_reason(sentence) == expected

    def test_citation_marker_pattern(self):
        assert CITATION_MARKER.match("(passage 12)")
        assert CITATION_MARKER.match("Sources 1 & 2")
        assert not CITATION_MARKER.match("Passage 3 explains the policy.")
        assert not CITATION_MARKER.match("See passage")

    def test_citation_marker_is_merged_only_on_the_same_line(self):
        same_line = segment_sentences("The policy is no (Passage 3). Next sentence here.")
        assert [s.text for s in same_line] == ["The policy is no (Passage 3).", "Next sentence here."]
        assert all(s.skipped is None for s in same_line)

        next_line = segment_sentences("The policy is no.\n(Passage 3)")
        assert [s.text for s in next_line] == ["The policy is no.", "(Passage 3)"]
        assert [s.skipped for s in next_line] == [None, SKIP_CITATION]

    def test_lines_are_segmented_separately(self):
        segments = segment_sentences("First line without period\nSecond line. Third sentence.")
        assert [s.text for s in segments] == ["First line without period", "Second line.", "Third sentence."]

    def test_segmenter_id_names_rules_and_pysbd(self):
        assert f"ragnarok-sentences:1:pysbd-{pysbd.__version__}:en" == SEGMENTER_ID
        assert segmenter_id("fr").endswith(":fr")
