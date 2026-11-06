"""Run a judge over many samples and keep one ``JudgeAttempt`` per sample.

``judge_many`` is the single entry point the calibrator, the CLI and the
benchmark harness use to run a judge. It turns per-sample failures into
``JudgeFailure`` values and never hides configuration errors.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol

from ragnarok_ai.core.exceptions import (
    ConfigurationError,
    EvaluationError,
    JudgeParseError,
    LLMConnectionError,
    NoUnitsError,
)
from ragnarok_ai.judges.contract import CallTrace, JudgeAttempt, JudgeFailure

if TYPE_CHECKING:
    from collections.abc import Sequence
    from dataclasses import replace as _replace

    from ragnarok_ai.core.protocols import JudgeProtocol
    from ragnarok_ai.evaluators.judge import JudgeResult
else:
    from dataclasses import replace as _replace


class JudgeSample(Protocol):
    """What ``judge_many`` needs from a sample: the three texts.

    ``CalibrationSample`` satisfies it. Optional attributes: ``sample_id``
    (defaults to the sample's position) and ``group_id``.
    """

    question: str
    context: str
    answer: str


def sample_identifier(sample: Any, index: int) -> str:
    """``sample_id`` when the sample carries one, else its zero-based position."""
    explicit = getattr(sample, "sample_id", None)
    return str(explicit) if explicit not in (None, "") else str(index)


def _failure_for(exc: Exception, keep_raw_output: bool) -> JudgeFailure:
    """Map a per-sample exception to a ``JudgeFailure``."""
    if isinstance(exc, JudgeParseError):
        return JudgeFailure("parse", str(exc), raw_output=exc.raw_output if keep_raw_output else None)
    if isinstance(exc, NoUnitsError):
        return JudgeFailure("no_units", str(exc))
    if isinstance(exc, asyncio.TimeoutError):
        return JudgeFailure("timeout", str(exc) or "timed out")
    if isinstance(exc, LLMConnectionError) and "timed out" in str(exc).lower():
        return JudgeFailure("timeout", str(exc))
    return JudgeFailure("backend", str(exc) or type(exc).__name__)


def _strip_raw_output(result: JudgeResult) -> JudgeResult:
    if result.trace is None or result.trace.raw_output is None:
        return result
    return _replace(result, trace=_replace(result.trace, raw_output=None))


async def judge_many(
    judge: JudgeProtocol,
    criterion: str,
    samples: Sequence[JudgeSample],
    *,
    concurrency: int = 1,
    keep_raw_output: bool = False,
) -> list[JudgeAttempt]:
    """Judge every sample on *criterion*; one ``JudgeAttempt`` per sample, in order.

    Per-sample failures become ``JudgeFailure`` values: an unreadable output
    (``JudgeParseError``, kind ``parse``), an answer with nothing to judge
    (``NoUnitsError``, kind ``no_units``), a timeout (kind ``timeout``) or a
    backend error during the call (``LLMConnectionError`` or another
    ``EvaluationError``, kind ``backend``). Configuration errors are never
    caught: ``judge.config(criterion)`` runs first, so a misconfigured judge
    stops before the first call, and a ``ConfigurationError`` raised during
    a call propagates.

    Args:
        judge: Any ``JudgeProtocol`` implementation.
        criterion: The criterion to judge; must be in ``judge.criteria``.
        samples: Objects exposing ``question``, ``context`` and ``answer``;
            ``sample_id`` and ``group_id`` are used when present.
        concurrency: Calls in flight at once. The default, 1, is the only
            setting that keeps Ollama runs reproducible: with several
            concurrent requests the server batches them and token sampling
            is no longer bit-for-bit repeatable, even at temperature 0 with a
            fixed seed. Raise it only for backends documented as deterministic
            under concurrency, or when reproducibility does not matter.
        keep_raw_output: Keep the model's raw text in traces and parse
            failures. Off by default: the text can copy sensitive context.

    Returns:
        The attempts in the order of *samples*, each carrying the
        ``JudgeConfig`` of the run and the sample's identifier.

    Raises:
        ValueError: If *criterion* is unknown to the judge or *concurrency* < 1.
        ConfigurationError, JudgeConnectionError, JudgeDependencyError,
        JudgeModelNotFoundError: Configuration problems, never turned into failures.
    """
    if concurrency < 1:
        msg = f"concurrency must be at least 1, got {concurrency}"
        raise ValueError(msg)
    if criterion not in judge.criteria:
        msg = f"Unknown criterion {criterion!r}; the judge supports {sorted(judge.criteria)}"
        raise ValueError(msg)
    config = judge.config(criterion)  # configuration errors surface here, before any call

    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int, sample: JudgeSample) -> JudgeAttempt:
        sample_id = sample_identifier(sample, index)
        async with semaphore:
            try:
                result = await judge.judge(
                    criterion, question=sample.question, context=sample.context, answer=sample.answer
                )
            except ConfigurationError:
                raise
            except (EvaluationError, LLMConnectionError, asyncio.TimeoutError) as exc:
                return JudgeAttempt(sample_id, criterion, config, result=None, error=_failure_for(exc, keep_raw_output))
        if not keep_raw_output:
            result = _strip_raw_output(result)
        trace: CallTrace | None = result.trace
        return JudgeAttempt(sample_id, criterion, config, result=result, error=None, trace=trace)

    if concurrency == 1:
        return [await one(i, s) for i, s in enumerate(samples)]
    return list(await asyncio.gather(*(one(i, s) for i, s in enumerate(samples))))


__all__ = ["JudgeSample", "judge_many", "sample_identifier"]
