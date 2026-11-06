"""Execute every Python code block of the README against the real API.

Prevents the README from drifting away from the library: a renamed
function, a removed argument or a missing attribute makes this test fail.

Each block runs in a fresh namespace (top-level ``await`` allowed) seeded
with the placeholder objects the README refers to (``my_rag``, ``testset``,
``create_rag``, ``results``...). No network: the Ollama client used by
``LLMJudge`` is replaced by a scripted fake, and blocks run in a temporary
working directory so written files (reports, metrics) stay isolated.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from typing import Any

import pytest

from ragnarok_ai import evaluate
from ragnarok_ai.core.types import Document, Query, RAGResponse, TestSet
from ragnarok_ai.evaluators.judge import PROMETHEUS_ABS_SYSTEM_PROMPT, LLMJudge
from ragnarok_ai.judges.hhem import HHEMArtifacts
from ragnarok_ai.judges.minicheck import CHECKPOINTS, MiniCheckArtifacts

README = Path(__file__).resolve().parents[2] / "README.md"
PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)


def _python_blocks() -> list[tuple[int, str]]:
    text = README.read_text(encoding="utf-8")
    return [(text.count("\n", 0, m.start()) + 1, m.group(1)) for m in PYTHON_BLOCK.finditer(text)]


BLOCKS = _python_blocks()


DOCUMENTS = [
    Document(id="doc_python", content="Python was created by Guido van Rossum and first released in 1991."),
    Document(id="doc_rust", content="Rust is a systems programming language focused on memory safety."),
]


class StubRAG:
    """Deterministic in-memory pipeline standing in for the user's RAG."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    async def query(self, question: str) -> RAGResponse:
        return RAGResponse(answer=f"Answer to {question}", retrieved_docs=DOCUMENTS)


class FakeOllamaLLM:
    """Scripted stand-in for OllamaLLM answering in the format the prompt asks for."""

    def __init__(self, model: str = "fake", **kwargs: Any) -> None:  # noqa: ARG002
        self.model = model

    async def generate(self, prompt: str) -> str:
        if prompt.startswith(PROMETHEUS_ABS_SYSTEM_PROMPT):
            return "The answer is grounded in the context. [RESULT] 5"
        return "Score: 5\nVerdict: PASS\nExplanation: The answer is grounded in the context."

    async def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.0]


class FakeHHEMBackend:
    """Stand-in for the loaded HHEM model: whitespace tokens, P(consistent) 0.9."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])

    def consistency(self, prompts: list[str]) -> list[float]:
        return [0.9 for _ in prompts]


class FakeMiniCheckBackend:
    """Stand-in for a loaded MiniCheck checkpoint: naive sentence split, P(supported) 0.9 per chunk."""

    def split_sentences(self, text: str) -> list[str]:
        return [s for s in text.replace("? ", "?\n").replace(". ", ".\n").split("\n") if s]

    def splitter_id(self) -> str:
        return "fake-splitter"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def support(self, chunks: list[str], claim: str) -> tuple[list[float], int, bool]:  # noqa: ARG002
        return [0.9 for _ in chunks], sum(len(c.split()) for c in chunks), False


def test_readme_has_python_examples() -> None:
    assert BLOCKS, "no ```python block found in README.md"


@pytest.mark.parametrize(("line", "source"), BLOCKS, ids=[f"README.md:{line}" for line, _ in BLOCKS])
async def test_readme_example_runs(line: int, source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ragnarok_ai.adapters.llm.ollama.OllamaLLM", FakeOllamaLLM)
    monkeypatch.setattr(LLMJudge, "_get_available_models", lambda self, base_url: [LLMJudge.DEFAULT_MODEL])  # noqa: ARG005
    # No weights download: HHEMJudge gets a scripted backend and fake artifacts
    artifacts = HHEMArtifacts(tmp_path, tmp_path, "pinned", "pinned", "local", "fake-digest")
    monkeypatch.setattr("ragnarok_ai.judges.hhem.resolve_artifacts", lambda *a, **k: artifacts)  # noqa: ARG005
    monkeypatch.setattr("ragnarok_ai.judges.hhem.TransformersBackend", lambda artifacts, device: FakeHHEMBackend())  # noqa: ARG005
    minicheck_artifacts = MiniCheckArtifacts(CHECKPOINTS["flan-t5-large"], tmp_path, "pinned", None, "local", "fake")
    monkeypatch.setattr("ragnarok_ai.judges.minicheck.resolve_artifacts", lambda *a, **k: minicheck_artifacts)  # noqa: ARG005
    monkeypatch.setattr(
        "ragnarok_ai.judges.minicheck.TransformersBackend",
        lambda artifacts, device, offline=False: FakeMiniCheckBackend(),  # noqa: ARG005
    )

    testset = TestSet(queries=[Query(text="Who created Python?", ground_truth_docs=["doc_python"])])
    results = await evaluate(StubRAG(), testset)
    namespace: dict[str, Any] = {
        "__name__": "readme_example",
        "my_rag": StubRAG(),
        "rag": StubRAG(),
        "create_rag": StubRAG,
        "documents": DOCUMENTS,
        "testset": testset,
        "results": results,
        "baseline_results": results,
        "improved_results": results,
    }

    code = compile(source, f"README.md:{line}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    outcome = eval(code, namespace)  # trusted source: the repository README
    if inspect.iscoroutine(outcome):
        await outcome
