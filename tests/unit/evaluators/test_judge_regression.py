"""Regression against the 1.12.3 reference fixture.

``fixtures/judge_reference_v1_12_3.json`` was produced by running the code of
tag v1.12.3 with a scripted LLM (see ``fixtures/make_judge_reference.py``).
The current code must send the very same prompts and return the very same
results, for both prompt formats, the four criteria, with and without
medical mode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ragnarok_ai.core.exceptions import EvaluationError, JudgeParseError
from ragnarok_ai.evaluators.judge import LLMJudge
from ragnarok_ai.judges import OrdinalScore

FIXTURE = Path(__file__).parent / "fixtures" / "judge_reference_v1_12_3.json"
REFERENCE = json.loads(FIXTURE.read_text(encoding="utf-8"))
CRITERIA = ("faithfulness", "relevance", "hallucination", "completeness")


class ReplayLLM:
    """Same scripted client as the generator: returns the recorded responses, records the prompts."""

    def __init__(self, model: str, responses: list[str]) -> None:
        self.model = model
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0)

    async def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.0]


def named_call(judge: LLMJudge, criterion: str, inputs: dict[str, str]) -> Any:
    """The four public methods, called exactly as the generator called them."""
    if criterion == "faithfulness":
        return judge.evaluate_faithfulness(
            context=inputs["context"], question=inputs["question"], answer=inputs["answer"]
        )
    if criterion == "relevance":
        return judge.evaluate_relevance(question=inputs["question"], answer=inputs["answer"])
    if criterion == "hallucination":
        return judge.detect_hallucination(context=inputs["context"], answer=inputs["answer"])
    return judge.evaluate_completeness(question=inputs["question"], answer=inputs["answer"], context=inputs["context"])


def five_fields(result: Any) -> dict[str, Any]:
    return {
        "verdict": result.verdict,
        "score": result.score,
        "explanation": result.explanation,
        "criteria": result.criteria,
        "raw_score": result.raw_score,
    }


def case_ids() -> list[str]:
    return [f"{c['format']}-{'medical' if c['medical_mode'] else 'plain'}" for c in REFERENCE["cases"]]


def test_fixture_was_produced_by_the_reference_version() -> None:
    """The reference values come from v1.12.3, never from the code under test."""
    ref = REFERENCE["reference"]
    assert ref["ragnarok_ai_version"] == "1.12.3"
    assert (
        ref["commit"] == "fa08ee6535010b969a1c85c8646a4202f783e81c"
    )  # merge commit of the v1.12.3 release PR, tag v1.12.3
    assert "module" not in ref, "no local path in the fixture"
    assert len(REFERENCE["cases"]) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("case", REFERENCE["cases"], ids=case_ids())
@pytest.mark.parametrize("criterion", CRITERIA)
@pytest.mark.parametrize("entry_point", ["named_method", "judge"])
async def test_prompt_and_result_match_1_12_3(case: dict[str, Any], criterion: str, entry_point: str) -> None:
    """Exact prompt, exact five result fields, through the named method and through judge()."""
    expected = case["criteria"][criterion]
    llm = ReplayLLM(case["model"], [expected["response"]])
    judge = LLMJudge(llm=llm, medical_mode=case["medical_mode"])
    inputs = case["inputs"]

    if entry_point == "named_method":
        result = await named_call(judge, criterion, inputs)
    else:
        context = inputs["context"] if criterion != "relevance" else ""
        question = inputs["question"] if criterion != "hallucination" else ""
        result = await judge.judge(criterion, question=question, context=context, answer=inputs["answer"])

    assert llm.prompts == [expected["prompt"]]
    assert five_fields(result) == expected["result"]
    # the contract fields are filled on top, without changing the historical ones
    assert result.output == OrdinalScore(expected["result"]["raw_score"], 1, 5)
    assert result.config is not None and result.config.criterion == criterion
    assert result.trace is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("case", REFERENCE["cases"], ids=case_ids())
async def test_evaluate_all_matches_1_12_3(case: dict[str, Any]) -> None:
    """Four prompts in the same order, same combined serialization."""
    expected = case["evaluate_all"]
    responses = [case["criteria"][c]["response"] for c in CRITERIA]
    llm = ReplayLLM(case["model"], responses)
    judge = LLMJudge(llm=llm, medical_mode=case["medical_mode"])
    inputs = case["inputs"]

    results = await judge.evaluate_all(context=inputs["context"], question=inputs["question"], answer=inputs["answer"])

    assert llm.prompts == expected["prompts"]
    assert results.to_dict() == expected["to_dict"]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", REFERENCE["cases"], ids=case_ids())
async def test_unparseable_output_still_raises_an_evaluation_error(case: dict[str, Any]) -> None:
    """1.12.3 raised EvaluationError; the new JudgeParseError is one, with the same message."""
    expected = case["unparseable"]
    assert expected["exception"] == "EvaluationError"
    llm = ReplayLLM(case["model"], [expected["response"]])
    judge = LLMJudge(llm=llm, medical_mode=case["medical_mode"])
    inputs = case["inputs"]

    with pytest.raises(EvaluationError) as info:
        await named_call(judge, "faithfulness", inputs)

    assert isinstance(info.value, JudgeParseError)
    assert str(info.value) == expected["message"]
    assert info.value.criterion == "faithfulness"
    assert info.value.raw_output == expected["response"]


@pytest.mark.parametrize("fmt", ["prometheus", "score-n"])
def test_medical_mode_normalizes_three_criteria_not_relevance(fmt: str) -> None:
    """From the fixture itself: the normalized form appears in three prompts, never in relevance."""
    case = next(c for c in REFERENCE["cases"] if c["format"] == fmt and c["medical_mode"])
    for criterion in ("faithfulness", "hallucination", "completeness"):
        assert "congestive heart failure" in case["criteria"][criterion]["prompt"]
    relevance_prompt = case["criteria"]["relevance"]["prompt"]
    assert "congestive heart failure" not in relevance_prompt
    assert "pt have" in relevance_prompt  # the question is sent as written
