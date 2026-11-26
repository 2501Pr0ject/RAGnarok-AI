"""Scripted LLMs and ready-made claim-level judges for the tests (and for ``--judge python:...``)."""

from __future__ import annotations

import json
import re

from ragnarok_ai.evaluators import FaithfulnessEvaluator, HallucinationDetector
from ragnarok_ai.judges import ClaimLevelJudge


class ScriptedLLM:
    """Stands for a model: one claim per sentence of the response; a claim holding '(unsupported)' fails.

    A response holding '(no claims)' gives an empty extraction.

    Counts its calls by kind, so a test can tell an extraction from a check.
    """

    is_local = True

    def __init__(self, model: str | None = "scripted-1", options: dict[str, object] | None = None) -> None:
        if model is not None:
            self.model = model
        if options is not None:
            self.options = options
        self.extractions = 0
        self.checks = 0

    async def generate(self, prompt: str) -> str:
        if prompt.startswith("Extract all factual claims"):
            self.extractions += 1
            response = prompt.split("Response: ", 1)[1].split("\n\nReturn a JSON array", 1)[0]
            if "(no claims)" in response:
                return "[]"
            return json.dumps([s.strip() for s in re.split(r"(?<=[.!?])\s+", response.strip()) if s.strip()])
        self.checks += 1
        claim = prompt.split("Claim: ", 1)[1].split("\n\nContext: ", 1)[0]
        fails = "(unsupported)" in claim
        if prompt.startswith("Analyze if the following claim is a hallucination"):
            return json.dumps(
                {"is_hallucination": fails, "reason": "not in the context" if fails else "in the context"}
            )
        return json.dumps({"supported": not fails, "reasoning": "not in the context" if fails else "in the context"})

    async def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.0]


class ScriptedFaithfulnessJudge(ClaimLevelJudge):
    """``FaithfulnessEvaluator`` on a scripted model, buildable without arguments."""

    def __init__(self) -> None:
        super().__init__(FaithfulnessEvaluator(ScriptedLLM()))


class ScriptedHallucinationFreeJudge(ClaimLevelJudge):
    """``HallucinationDetector`` on a scripted model, buildable without arguments."""

    def __init__(self) -> None:
        super().__init__(HallucinationDetector(ScriptedLLM()))
