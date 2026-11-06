"""Apply an answer-level judge sentence by sentence.

``SentenceLevel`` wraps any ``JudgeProtocol`` judge: the answer is segmented
with the shared segmenter, each sentence is judged as if it were the whole
answer (question and context unchanged), and the unit judgments are
aggregated into an answer-level score and verdict. This is what lets a
prompted judge be compared with sentence-level classifiers on the same
units.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ragnarok_ai.judges.aggregation import aggregate_units
from ragnarok_ai.judges.contract import AggregationPolicy, JudgeConfig, UnitJudgment
from ragnarok_ai.text.segmentation import segment_sentences, segmenter_id

if TYPE_CHECKING:
    from ragnarok_ai.core.protocols import JudgeProtocol
    from ragnarok_ai.evaluators.judge import JudgeResult

DEFAULT_AGGREGATION = AggregationPolicy(score_rule="fraction_pass", verdict_rule="worst")


class SentenceLevel:
    """A ``JudgeProtocol`` judge that scores each sentence of the answer with an inner judge.

    Attributes:
        inner: The answer-level judge applied to each sentence.
        aggregation: How unit scores and verdicts become the answer's;
            default ``fraction_pass`` score (share of PASS sentences) and
            ``worst`` verdict, as for the claim-level evaluators and the
            grounding judge; ``min`` and ``mean`` remain available.
        lang: pysbd language code passed to the segmenter.

    Example:
        >>> judge = SentenceLevel(LLMJudge())
        >>> result = await judge.judge("faithfulness", question=q, context=c, answer=a)
        >>> result.units  # one UnitJudgment per sentence, skipped ones included
    """

    def __init__(
        self,
        inner: JudgeProtocol,
        *,
        aggregation: AggregationPolicy = DEFAULT_AGGREGATION,
        lang: str = "en",
    ) -> None:
        """Initialize SentenceLevel.

        Args:
            inner: The judge applied to each sentence.
            aggregation: Aggregation policy recorded in the configuration.
            lang: Segmentation language.
        """
        self.inner = inner
        self.aggregation = aggregation
        self.lang = lang

    @property
    def criteria(self) -> frozenset[str]:
        """The inner judge's criteria."""
        return self.inner.criteria

    def config(self, criterion: str) -> JudgeConfig:
        """The inner configuration at ``unit="sentence"``, with the aggregation and the segmenter."""
        inner = self.inner.config(criterion)
        return replace(
            inner,
            judge_class=f"{type(self).__module__}.{type(self).__qualname__}[{inner.judge_class}]",
            unit="sentence",
            aggregation=self.aggregation,
            segmenter=segmenter_id(self.lang),
        )

    async def judge(self, criterion: str, *, question: str, context: str, answer: str) -> JudgeResult:
        """Judge each sentence of *answer*, then aggregate.

        Skipped segments (no alphanumeric content, citation marker alone,
        fragment too short) appear in ``units`` with their reason and are
        not sent to the inner judge.

        Raises:
            ValueError: If *criterion* is unknown to the inner judge.
            NoUnitsError: If no sentence could be judged.
            JudgeParseError: If the inner judge's output cannot be parsed
                for one of the sentences.
        """
        if criterion not in self.criteria:
            msg = f"Unknown criterion {criterion!r}; expected one of {sorted(self.criteria)}"
            raise ValueError(msg)
        config = self.config(criterion)
        units: list[UnitJudgment] = []
        last: JudgeResult | None = None
        explanations: list[str] = []
        for segment in segment_sentences(answer, self.lang):
            if segment.skipped is not None:
                units.append(UnitJudgment(text=segment.text, span=segment.span, skipped=segment.skipped))
                continue
            result = await self.inner.judge(criterion, question=question, context=context, answer=segment.text)
            last = result
            explanations.append(f"[{segment.start}-{segment.end}] {result.verdict}: {result.explanation}".rstrip(": "))
            units.append(
                UnitJudgment(
                    text=segment.text,
                    span=segment.span,
                    output=result.output,
                    score=result.score,
                    verdict=result.verdict,
                )
            )
        score, verdict = aggregate_units(units, self.aggregation, config.verdict_policy)
        assert last is not None  # aggregate_units raised NoUnitsError otherwise
        # Answer level: the aggregated score and verdict. There is no native
        # output nor single raw score for an aggregate: both are None.
        return replace(
            last,
            verdict=verdict,
            score=score,
            explanation="\n".join(explanations),
            raw_score=None,
            output=None,
            config=config,
            trace=None,
            units=tuple(units),
        )


__all__ = ["DEFAULT_AGGREGATION", "SentenceLevel"]
