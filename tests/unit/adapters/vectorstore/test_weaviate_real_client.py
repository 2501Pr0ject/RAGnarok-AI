"""WeaviateVectorStore against the real weaviate-client: how it connects, without a server and without a mock.

The unit tests of ``test_weaviate.py`` replace ``_ensure_client``, so the calls that open the connection
were never executed: ``connect_to_custom`` was called without its three required gRPC arguments, and a
self-hosted instance with an API key could not be reached. Here the real functions are involved:

- their real signatures are bound with the arguments the adapter passes (a missing or unknown argument
  fails, as it would at run time);
- a real connection attempt to a port nothing listens on must fail as a connection failure, not as a
  ``TypeError``.

The scenarios run in a subprocess, so that the real package is imported whatever the session has put
in ``sys.modules``. Skipped when weaviate-client is not installed; the CI job "Vector stores (real
clients)" installs the ``weaviate`` extra and runs this file.
"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
import textwrap

import pytest

try:
    WEAVIATE_CLIENT_VERSION = importlib.metadata.version("weaviate-client")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover - depends on the environment
    WEAVIATE_CLIENT_VERSION = None

pytestmark = pytest.mark.skipif(WEAVIATE_CLIENT_VERSION is None, reason="weaviate-client not installed")


def run(script: str) -> dict[str, object]:
    done = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    result: dict[str, object] = json.loads(done.stdout.strip().splitlines()[-1])
    return result


# Each connect function is replaced by a recorder that first binds the call to the real signature
BINDING = """
    import asyncio, inspect, json
    import weaviate
    from ragnarok_ai.adapters.vectorstore.weaviate import WeaviateVectorStore

    calls = []
    # What each real function requires, read before it is replaced
    required = {
        name: [n for n, p in inspect.signature(getattr(weaviate, name)).parameters.items()
               if p.default is inspect.Parameter.empty]
        for name in ("connect_to_custom", "connect_to_local", "connect_to_weaviate_cloud")
    }

    def recorder(name):
        real = getattr(weaviate, name)
        signature = inspect.signature(real)

        def record(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)  # TypeError here is the TypeError of the real call
            calls.append([name, {k: (v if isinstance(v, (str, int, bool)) else type(v).__name__)
                                 for k, v in bound.arguments.items()}])
            raise ConnectionError("stop before any network access")

        return record

    for name in ("connect_to_custom", "connect_to_local", "connect_to_weaviate_cloud"):
        setattr(weaviate, name, recorder(name))


    async def attempt(**kwargs):
        store = WeaviateVectorStore(**kwargs)
        try:
            await store._ensure_client()
        except Exception as exc:
            return [type(exc).__name__, type(exc.__cause__).__name__, str(exc.__cause__)]
        return ["no error", "", ""]
"""


def test_self_hosted_with_an_api_key_passes_every_required_argument_of_connect_to_custom():
    out = run(
        BINDING
        + """
    async def main():
        results = {
            "default": await attempt(url="http://weaviate.internal:8081", api_key="k"),
            "https": await attempt(url="https://weaviate.example.org:8443", api_key="k"),
            "explicit": await attempt(
                url="https://weaviate.example.org:8443", api_key="k",
                grpc_host="grpc.example.org", grpc_port=50443, grpc_secure=False,
            ),
        }
        print(json.dumps({"results": results, "calls": calls, "required": required["connect_to_custom"]}))

    asyncio.run(main())
    """
    )
    # the call reached the function: no TypeError, the recorder's own error is what stopped it
    for name in ("default", "https", "explicit"):
        assert out["results"][name] == [
            "VectorStoreConnectionError",
            "ConnectionError",
            "stop before any network access",
        ]  # type: ignore[index]
    calls = out["calls"]
    assert [call[0] for call in calls] == ["connect_to_custom"] * 3  # type: ignore[union-attr]
    default, https, explicit = (call[1] for call in calls)  # type: ignore[union-attr]
    assert default == {
        "http_host": "weaviate.internal",
        "http_port": 8081,
        "http_secure": False,
        "grpc_host": "weaviate.internal",
        "grpc_port": 50051,
        "grpc_secure": False,
        "auth_credentials": "_APIKey",
    }
    assert (https["http_secure"], https["grpc_host"], https["grpc_port"], https["grpc_secure"]) == (
        True,
        "weaviate.example.org",
        50051,
        True,
    )
    assert (explicit["grpc_host"], explicit["grpc_port"], explicit["grpc_secure"]) == ("grpc.example.org", 50443, False)
    # every parameter the real function requires is passed
    assert set(out["required"]) <= set(default)  # type: ignore[arg-type]
    assert {"grpc_host", "grpc_port", "grpc_secure"} <= set(out["required"])  # type: ignore[arg-type]


def test_what_the_real_connect_functions_receive_for_each_form_of_url():
    """http or https, with or without a port: the values bound to the real signatures (#201)."""
    out = run(
        BINDING
        + """
    async def main():
        results = {}
        for url in ("https://weaviate.example.org", "https://weaviate.example.org:8443",
                    "http://weaviate.internal", "http://weaviate.internal:8081"):
            calls.clear()
            await attempt(url=url, api_key="k")
            results["key " + url] = calls[0]
            calls.clear()
            await attempt(url=url)
            results["no key " + url] = calls[0]
        print(json.dumps({"results": results}))

    import os
    os.environ.pop("WEAVIATE_API_KEY", None)
    asyncio.run(main())
    """
    )
    r = out["results"]

    def custom(url: str) -> tuple[object, ...]:
        name, kw = r["key " + url]  # type: ignore[index]
        return name, kw["http_host"], kw["http_port"], kw["http_secure"], kw["grpc_host"], kw["grpc_secure"]

    assert custom("https://weaviate.example.org") == (
        "connect_to_custom",
        "weaviate.example.org",
        443,  # was 8080: TLS on Weaviate's plain HTTP port
        True,
        "weaviate.example.org",
        True,
    )
    assert custom("https://weaviate.example.org:8443")[2] == 8443
    assert custom("http://weaviate.internal") == (
        "connect_to_custom",
        "weaviate.internal",
        8080,
        False,
        "weaviate.internal",
        False,
    )
    assert custom("http://weaviate.internal:8081")[2] == 8081
    # without a key: connect_to_local with the same host and port
    assert r["no key http://weaviate.internal"] == ["connect_to_local", {"host": "weaviate.internal", "port": 8080}]  # type: ignore[index]
    assert r["no key http://weaviate.internal:8081"][1]["port"] == 8081  # type: ignore[index]
    assert r["no key https://weaviate.example.org:8443"][1] == {"host": "weaviate.example.org", "port": 8443}  # type: ignore[index]


def test_the_two_other_ways_to_connect_still_bind_to_their_real_signatures():
    out = run(
        BINDING
        + """
    async def main():
        results = {
            "local": await attempt(url="http://localhost:8080"),
            "cloud": await attempt(url="https://my-cluster.weaviate.cloud", api_key="k"),
        }
        print(json.dumps({"results": results, "calls": calls}))

    import os
    os.environ.pop("WEAVIATE_API_KEY", None)
    asyncio.run(main())
    """
    )
    assert [call[0] for call in out["calls"]] == ["connect_to_local", "connect_to_weaviate_cloud"]  # type: ignore[union-attr]
    assert out["calls"][0][1] == {"host": "localhost", "port": 8080}  # type: ignore[index]
    assert out["calls"][1][1]["cluster_url"] == "https://my-cluster.weaviate.cloud"  # type: ignore[index]
    for name in ("local", "cloud"):
        assert out["results"][name][1] == "ConnectionError"  # type: ignore[index]


def test_a_real_connection_attempt_to_a_closed_port_is_a_connection_failure_not_a_type_error():
    out = run(
        """
    import asyncio, json
    from ragnarok_ai.adapters.vectorstore.weaviate import WeaviateVectorStore
    from ragnarok_ai.core.exceptions import VectorStoreConnectionError


    async def main():
        # nothing listens on ports 1 and 2 of the loopback interface
        store = WeaviateVectorStore(url="http://127.0.0.1:1", api_key="not-a-real-key", grpc_port=2)
        try:
            await store.search([0.1, 0.2], k=1)
        except VectorStoreConnectionError as exc:
            cause = exc.__cause__
            print(json.dumps({"raised": "VectorStoreConnectionError", "cause": type(cause).__name__,
                              "cause_module": type(cause).__module__.split(".")[0], "message": str(exc)[:200]}))
            return
        print(json.dumps({"raised": "nothing"}))

    asyncio.run(main())
    """
    )
    assert out["raised"] == "VectorStoreConnectionError"
    # the client tried to connect and failed to: its own error, where 1.13.1 had a TypeError on the call
    assert out["cause_module"] == "weaviate" and out["cause"] != "TypeError", out
    assert "missing" not in str(out["message"])
