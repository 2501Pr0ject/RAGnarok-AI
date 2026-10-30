"""Public import paths, each checked in a fresh interpreter.

The adapter packages resolve some names lazily. In-process tests cannot
check that reliably: whatever earlier tests imported is already cached in
``sys.modules`` and hides bugs (a ``RecursionError`` on
``from ragnarok_ai.adapters import OllamaLLM`` shipped in 1.12.0 that way).
Every case below therefore runs in its own interpreter, with httpx (the
optional dependency of the LLM adapters) installed or blocked.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

LLM_NAMES = "AnthropicLLM, GroqLLM, MistralLLM, OllamaLLM, OpenAILLM, TogetherLLM, VLLMAdapter"

# Imports that must work whether or not httpx is installed
ALWAYS_IMPORTABLE = {
    "adapters package": "import ragnarok_ai.adapters",
    "llm package": "import ragnarok_ai.adapters.llm",
    "vector stores from adapters": "from ragnarok_ai.adapters import ChromaVectorStore, QdrantVectorStore",
    "vectorstore classification": "from ragnarok_ai.adapters import LOCAL_VECTORSTORE_ADAPTERS, CLOUD_VECTORSTORE_ADAPTERS",
    "frameworks and agents": "from ragnarok_ai.adapters import LangChainAdapter, ReActAdapter",
    "vectorstore package": "from ragnarok_ai.adapters.vectorstore import ChromaVectorStore",
    "llm subpackage as attribute": "import ragnarok_ai.adapters as a; assert a.llm.__name__ == 'ragnarok_ai.adapters.llm'",
    "unknown attribute on adapters": "import ragnarok_ai.adapters as a; assert not hasattr(a, 'NoSuchAdapter')",
    "unknown attribute on llm": "import ragnarok_ai.adapters.llm as m; assert not hasattr(m, 'NoSuchAdapter')",
    "core API": "from ragnarok_ai import LLMJudge, evaluate, evaluate_stream",
    "plugins API": "from ragnarok_ai.plugins import PluginRegistry, PluginUnavailableError",
}

# Imports that need httpx
NEEDS_HTTPX = {
    "llm names from adapters": f"from ragnarok_ai.adapters import {LLM_NAMES}",
    "llm classification": "from ragnarok_ai.adapters import LOCAL_LLM_ADAPTERS, CLOUD_LLM_ADAPTERS",
    "llm names from llm package": f"from ragnarok_ai.adapters.llm import {LLM_NAMES}",
    "llm module": "from ragnarok_ai.adapters.llm.ollama import OllamaLLM",
    "llm module as attribute": "import ragnarok_ai.adapters.llm as m; assert m.ollama.OllamaLLM.__name__ == 'OllamaLLM'",
    "star import": "from ragnarok_ai.adapters import *",
}


def _run(code: str, *, block_httpx: bool) -> subprocess.CompletedProcess[str]:
    prelude = "import sys\nsys.modules['httpx'] = None\n" if block_httpx else ""
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("block_httpx", [False, True], ids=["with-httpx", "without-httpx"])
@pytest.mark.parametrize("code", ALWAYS_IMPORTABLE.values(), ids=ALWAYS_IMPORTABLE.keys())
def test_import_works(code: str, block_httpx: bool) -> None:
    proc = _run(code, block_httpx=block_httpx)

    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("code", NEEDS_HTTPX.values(), ids=NEEDS_HTTPX.keys())
def test_llm_import_works_with_httpx(code: str) -> None:
    proc = _run(code, block_httpx=False)

    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("code", NEEDS_HTTPX.values(), ids=NEEDS_HTTPX.keys())
def test_llm_import_without_httpx_raises_import_error(code: str) -> None:
    """Without httpx, LLM adapters fail with a plain ImportError, nothing else."""
    proc = _run(
        f"""
        try:
            {code}
        except ImportError:
            pass
        else:
            raise SystemExit("expected ImportError")
        """,
        block_httpx=True,
    )

    assert proc.returncode == 0, proc.stderr
