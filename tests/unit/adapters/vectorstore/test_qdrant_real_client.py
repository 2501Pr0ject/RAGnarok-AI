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

        # A collection info without a count, as the API allows: the real response, its count removed
        real_get_collection = store._client.get_collection

        async def without_count(*args, **kwargs):
            info = await real_get_collection(*args, **kwargs)
            return info.model_copy(update={"points_count": None})

        store._client.get_collection = without_count
        out["count_without_points_count"] = await store.count()
        store._client.get_collection = real_get_collection
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
    # no count in the collection info: asked to the real count method, not int(None)
    assert out["count_without_points_count"] == 2
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


ERRORS = textwrap.dedent(
    """
    import asyncio, json

    from qdrant_client import AsyncQdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse

    from ragnarok_ai.adapters.vectorstore.qdrant import QdrantVectorStore
    from ragnarok_ai.core.exceptions import VectorStoreConnectionError

    out = {}


    async def outcome(name, call):
        try:
            out[name] = ["returned", repr(await call())]
        except Exception as exc:
            cause = exc.__cause__
            out[name] = [type(exc).__name__, type(cause).__name__ if cause else None,
                         isinstance(exc, VectorStoreConnectionError)]


    async def main():
        # nothing listens on port 1 of the loopback interface: the server cannot be reached
        closed = QdrantVectorStore(url="http://127.0.0.1:1", vector_size=3, timeout=5)
        await outcome("closed port, search", lambda: closed.search([1.0, 0.0, 0.0], k=1))
        await outcome("closed port, is_available", lambda: QdrantVectorStore(url="http://127.0.0.1:1", timeout=5).is_available())

        # a response of the server: the real exception class of the client
        store = QdrantVectorStore(vector_size=3, collection_name="t")
        store._client = AsyncQdrantClient(location=":memory:")
        await store._ensure_collection()

        async def refused(*args, **kwargs):
            raise UnexpectedResponse(status_code=401, reason_phrase="Unauthorized", content=b"", headers=None)

        store._client.query_points = refused
        await outcome("server response, search", lambda: store.search([1.0, 0.0, 0.0], k=1))
        store._client.get_collections = refused
        await outcome("server response, is_available", lambda: store.is_available())

        # the client's own refusal in local mode: a collection that does not exist
        absent = QdrantVectorStore(vector_size=3, collection_name="absent")
        absent._client = AsyncQdrantClient(location=":memory:")
        await outcome("missing collection, count", lambda: absent.count())

        # a mistake in the adapter's call
        broken = QdrantVectorStore(vector_size=3, collection_name="t2")
        broken._client = AsyncQdrantClient(location=":memory:")
        await broken._ensure_collection()

        async def wrong(*args, **kwargs):
            raise AttributeError("'AsyncQdrantClient' object has no attribute 'search'")

        broken._client.query_points = wrong
        await outcome("bug, search", lambda: broken.search([1.0, 0.0, 0.0], k=1))
        print(json.dumps(out))


    asyncio.run(main())
    """
)


def test_only_a_server_that_cannot_be_reached_is_a_connection_failure():
    done = subprocess.run([sys.executable, "-c", ERRORS], capture_output=True, text=True, check=False, timeout=180)
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    # unreachable: the client's transport error, reported as a connection failure; is_available says False
    assert out["closed port, search"] == ["VectorStoreConnectionError", "ResponseHandlingException", True]
    assert out["closed port, is_available"] == ["returned", "False"]
    # the server answered: its response propagates as it is, and is_available does not hide it
    assert out["server response, search"] == ["UnexpectedResponse", None, False]
    assert out["server response, is_available"] == ["UnexpectedResponse", None, False]
    # neither is the client's own refusal, nor a bug of the adapter
    assert out["missing collection, count"][2] is False and out["missing collection, count"][0] != "returned"
    assert out["bug, search"] == ["AttributeError", None, False]


def test_a_missing_package_is_an_import_error_with_the_install_message():
    script = textwrap.dedent(
        """
        import asyncio, json, sys
        sys.modules["qdrant_client"] = None   # as if the package were not installed

        from ragnarok_ai.adapters.vectorstore.qdrant import QdrantVectorStore
        from ragnarok_ai.core.exceptions import ConfigurationError, VectorStoreConnectionError, VectorStoreDependencyError

        async def main():
            results = {}
            for name, call in (("search", lambda s: s.search([0.0] * 768, k=1)), ("is_available", lambda s: s.is_available())):
                try:
                    await call(QdrantVectorStore())
                    results[name] = "returned"
                except Exception as exc:
                    results[name] = [type(exc).__name__, isinstance(exc, ImportError), isinstance(exc, ConfigurationError),
                                     isinstance(exc, VectorStoreConnectionError), isinstance(exc, VectorStoreDependencyError), str(exc)]
            print(json.dumps(results))

        asyncio.run(main())
        """
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    expected = [
        "VectorStoreDependencyError",
        True,  # an ImportError
        True,
        False,  # not a connection error
        True,
        "qdrant-client is not installed. Install it with: pip install qdrant-client",
    ]
    assert out == {"search": expected, "is_available": expected}
