"""Context-aware disambiguation strategies for medical abbreviations.

The keyword scorer in ``MedicalAbbreviationNormalizer`` resolves most
ambiguous abbreviations for free, but when no context keyword matches it
falls back to the highest-priority meaning — silently. The strategies in
this module give the normalizer an escalation path for those uncertain
cases.

``SLMDisambiguator`` delegates the choice to a small, locally-run language
model (any ``LLMProtocol`` implementation, e.g. Ollama running qwen2.5:0.5b
or phi3:mini). The task is framed as closed-set classification — the model
picks a candidate *number* among the dictionary's own expansions — so even
a sub-1B model is reliable at it and can never introduce an expansion that
is not already in the dictionary. Results are cached per (abbreviation,
context window), so repeated contexts in batch evaluations cost one call.

Example:
    >>> from ragnarok_ai.adapters.llm.ollama import OllamaLLM
    >>> from ragnarok_ai.evaluators.medical import (
    ...     MedicalAbbreviationNormalizer,
    ...     SLMDisambiguator,
    ... )
    >>> llm = OllamaLLM(model="qwen2.5:0.5b")
    >>> normalizer = MedicalAbbreviationNormalizer(
    ...     disambiguator=SLMDisambiguator(llm)
    ... )
    >>> text, expansions = await normalizer.normalize_text_async(
    ...     "Echo shows severe MS with elevated gradients"
    ... )
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import OrderedDict
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ragnarok_ai.core.protocols import LLMProtocol
    from ragnarok_ai.evaluators.medical.abbreviations import AmbiguousEntry

logger = logging.getLogger(__name__)


@runtime_checkable
class DisambiguationStrategy(Protocol):
    """Strategy for resolving an ambiguous abbreviation from its context.

    Any class implementing ``resolve`` can be plugged into
    ``MedicalAbbreviationNormalizer``. Returning ``None`` means the
    strategy abstains and the normalizer falls back to its default
    priority-based resolution.
    """

    async def resolve(
        self,
        abbrev: str,
        candidates: list[AmbiguousEntry],
        context: str,
    ) -> str | None:
        """Pick the best expansion for *abbrev*, or ``None`` to abstain.

        Args:
            abbrev: The ambiguous abbreviation (e.g. ``"MS"``).
            candidates: The possible meanings, from the abbreviation
                dictionary.
            context: Window of text surrounding the abbreviation.

        Returns:
            The chosen ``full_form``, or ``None`` if undecidable.
        """
        ...


DISAMBIGUATION_PROMPT = """You are reading clinical text. An abbreviation in the excerpt below is ambiguous.

Excerpt: "{context}"

In this excerpt, "{abbrev}" most likely stands for:
{options}
0. cannot tell from this excerpt

Answer with the number only."""


class SLMDisambiguator:
    """Resolve ambiguous abbreviations with a small local language model.

    The model is asked to pick among the dictionary's own candidate
    expansions (closed-set classification), never to generate one. An
    unparseable answer, an out-of-range number, ``0`` ("cannot tell"), or
    any LLM error all resolve to ``None`` — the normalizer then falls back
    to its priority default, so this layer can only refine behavior, never
    break it.

    Decisions are memoized in a bounded LRU cache keyed by
    (abbreviation, context window), which makes repeated contexts in
    batch evaluations effectively free.

    Attributes:
        llm: The LLM provider used for classification.

    Example:
        >>> disambiguator = SLMDisambiguator(OllamaLLM(model="qwen2.5:0.5b"))
        >>> full_form = await disambiguator.resolve("MS", candidates, context)
    """

    def __init__(self, llm: LLMProtocol, *, cache_size: int = 4096) -> None:
        """Initialize the disambiguator.

        Args:
            llm: Any ``LLMProtocol`` implementation. Intended for small,
                fast local models (e.g. qwen2.5:0.5b, phi3:mini).
            cache_size: Maximum number of (abbreviation, context) decisions
                to memoize.
        """
        self.llm = llm
        self._cache_size = cache_size
        self._cache: OrderedDict[str, str | None] = OrderedDict()

    async def resolve(
        self,
        abbrev: str,
        candidates: list[AmbiguousEntry],
        context: str,
    ) -> str | None:
        """Pick the best expansion for *abbrev*, or ``None`` to abstain."""
        if not candidates:
            return None

        key = self._cache_key(abbrev, context)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        choice = await self._classify(abbrev, candidates, context)
        self._cache[key] = choice
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return choice

    async def _classify(
        self,
        abbrev: str,
        candidates: list[AmbiguousEntry],
        context: str,
    ) -> str | None:
        """Run the closed-set classification prompt against the SLM."""
        options = "\n".join(f"{i}. {entry.full_form}" for i, entry in enumerate(candidates, start=1))
        prompt = DISAMBIGUATION_PROMPT.format(context=context, abbrev=abbrev, options=options)

        try:
            response = await self.llm.generate(prompt)
        except Exception:
            logger.warning("SLM disambiguation call failed for %r; falling back to priority.", abbrev, exc_info=True)
            return None

        match = re.search(r"\d+", response)
        if match is None:
            return None
        index = int(match.group())
        if not 1 <= index <= len(candidates):
            return None
        return candidates[index - 1].full_form

    @staticmethod
    def _cache_key(abbrev: str, context: str) -> str:
        digest = hashlib.sha256(context.strip().lower().encode()).hexdigest()[:16]
        return f"{abbrev}:{digest}"
