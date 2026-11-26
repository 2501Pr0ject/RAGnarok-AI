"""QdrantVectorStore against the real qdrant-client, in its in-memory local mode: no server, no mock.

``test_qdrant.py`` replaces ``qdrant_client`` by a mock in ``sys.modules`` for the whole session, and a
mock accepts any method name: that is how a call to a method the library had removed went unseen
(``AsyncQdrantClient.search``, gone in qdrant-client 1.16.0). The scenario below therefore runs in a
subprocess, where the real package is imported whatever the session has done to ``sys.modules``.

Skipped when qdrant-client is not installed. The CI job "Vector stores (real clients)" installs the
``qdrant`` extra and runs this file, with the locked version and with the lower bound of the extra.
"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
import textwrap

import pytest

try:
    QDRANT_CLIENT_VERSION = importlib.metadata.version("qdrant-client")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover - depends on the environment
    QDRANT_CLIENT_VERSION = None

pytestmark = pytest.mark.skipif(QDRANT_CLIENT_VERSION is None, reason="qdrant-client not installed")

SCENARIO = textwrap.dedent(
    """
    import asyncio, json, sys
    from unittest.mock import Mock

    import qdrant_client
    from qdrant_client import AsyncQdrantClient

    from ragnarok_ai.adapters.vectorstore.qdrant import QdrantVectorStore
    from ragnarok_ai.core.types import Document

    assert not isinstance(qdrant_client, Mock), "the real package is expected here"

    IDS = [f"{i}{i}{i}{i}{i}{i}{i}{i}-0000-0000-0000-000000000000" for i in (1, 2, 3)]


    async def main():
        store = QdrantVectorStore(vector_size=3, collection_name="real_client_test")
        # The client's local mode: the same class and the same methods as against a server
        store._client = AsyncQdrantClient(location=":memory:")
        await store._ensure_collection()
        await store.add(
            [
                Document(id=IDS[0], content="about x", metadata={"embedding": [1.0, 0.0, 0.0], "source": "a"}),
                Document(id=IDS[1], content="about y", metadata={"embedding": [0.0, 1.0, 0.0], "source": "b"}),
                Document(id=IDS[2], content="mostly x", metadata={"embedding": [0.9, 0.1, 0.0], "source": "c"}),
            ]
        )
        out = {"count": await store.count()}
        hits = await store.search([1.0, 0.0, 0.0], k=2)
        out["hits"] = [[doc.id, doc.content, doc.metadata.get("source"), round(score, 4)] for doc, score in hits]
        out["all"] = len(await store.search([1.0, 0.0, 0.0], k=10))
        await store.delete([IDS[0]])
        out["count_after_delete"] = await store.count()
        out["hits_after_delete"] = [doc.id for doc, _ in await store.search([1.0, 0.0, 0.0], k=2)]
        out["ids"] = IDS
        print(json.dumps(out))


    asyncio.run(main())
    """
)


def run_scenario() -> dict[str, object]:
    done = subprocess.run([sys.executable, "-c", SCENARIO], capture_output=True, text=True, check=False, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    result: dict[str, object] = json.loads(done.stdout.strip().splitlines()[-1])
    return result


def test_add_search_delete_and_count_with_the_real_client():
    out = run_scenario()
    first, second, third = out["ids"]  # type: ignore[misc]
    assert out["count"] == 3
    # nearest first, with content, metadata and a cosine score
    assert out["hits"] == [[first, "about x", "a", 1.0], [third, "mostly x", "c", 0.9939]]
    assert out["all"] == 3  # k above the number of points returns them all
    assert out["count_after_delete"] == 2
    assert out["hits_after_delete"] == [third, second]


def test_the_client_method_the_adapter_calls_exists():
    """The check a mock cannot make: the method is on the real class, for the installed version."""
    done = subprocess.run(
        [sys.executable, "-c", "from qdrant_client import AsyncQdrantClient as C; print(hasattr(C, 'query_points'))"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert done.stdout.strip() == "True", f"qdrant-client {QDRANT_CLIENT_VERSION} has no query_points"
