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
        "additional_config": "NoneType",  # the environment is not read unless trust_env is asked
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
    # https without a key: connect_to_custom with TLS and no credentials, not connect_to_local, which
    # has no TLS option and sent the URL in plain HTTP (#209). Same gRPC defaults as with a key
    for url, port in (("https://weaviate.example.org", 443), ("https://weaviate.example.org:8443", 8443)):
        name, kw = r["no key " + url]  # type: ignore[index]
        assert name == "connect_to_custom"
        assert (kw["http_host"], kw["http_port"], kw["http_secure"]) == ("weaviate.example.org", port, True)
        assert (kw["grpc_host"], kw["grpc_port"], kw["grpc_secure"]) == ("weaviate.example.org", 50051, True)
        assert kw.get("auth_credentials") in (None, "NoneType")
        assert kw.get("additional_config") in (None, "NoneType")  # the environment is not read unless asked
    assert r["key https://weaviate.example.org"][1]["auth_credentials"] != "NoneType"  # type: ignore[index]


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


def test_only_an_instance_that_cannot_be_reached_is_a_connection_failure():
    out = run(
        """
    import asyncio, json
    import weaviate
    from weaviate.exceptions import UnexpectedStatusCodeError
    from ragnarok_ai.adapters.vectorstore.weaviate import WeaviateVectorStore
    from ragnarok_ai.core.exceptions import VectorStoreConnectionError

    out = {}


    async def outcome(name, call):
        try:
            out[name] = ["returned", repr(await call())]
        except Exception as exc:
            cause = exc.__cause__
            out[name] = [type(exc).__name__, type(cause).__module__.split(".")[0] if cause else None,
                         isinstance(exc, VectorStoreConnectionError)]


    def store():
        return WeaviateVectorStore(url="http://127.0.0.1:1", api_key="not-a-real-key", grpc_port=2)


    async def main():
        # nothing listens on ports 1 and 2 of the loopback interface
        await outcome("closed port, search", lambda: store().search([0.1, 0.2], k=1))
        await outcome("closed port, is_available", lambda: store().is_available())

        # a mistake in the adapter's call: what #191 was
        real = weaviate.connect_to_custom

        def wrong(*args, **kwargs):
            raise TypeError("connect_to_custom() missing 3 required positional arguments")

        weaviate.connect_to_custom = wrong
        await outcome("bug, search", lambda: store().search([0.1, 0.2], k=1))
        await outcome("bug, is_available", lambda: store().is_available())

        # a response of the server: the client's own class for an unexpected status code
        import httpx

        def refused(*args, **kwargs):
            raise UnexpectedStatusCodeError("Connect", httpx.Response(401, json={"error": "unauthorized"}))

        weaviate.connect_to_custom = refused
        await outcome("server response, search", lambda: store().search([0.1, 0.2], k=1))
        await outcome("server response, is_available", lambda: store().is_available())
        weaviate.connect_to_custom = real
        print(json.dumps(out))

    asyncio.run(main())
    """
    )
    assert out["closed port, search"] == ["VectorStoreConnectionError", "weaviate", True]
    assert out["closed port, is_available"] == ["returned", "False"]
    assert out["bug, search"] == ["TypeError", None, False]
    assert out["bug, is_available"] == ["TypeError", None, False]  # no longer "False": the server is not the cause
    assert out["server response, search"] == ["UnexpectedStatusCodeError", None, False]
    assert out["server response, is_available"] == ["UnexpectedStatusCodeError", None, False]


def test_a_missing_package_is_an_import_error_with_the_install_message():
    out = run(
        """
    import asyncio, json, sys
    sys.modules["weaviate"] = None   # as if the package were not installed

    from ragnarok_ai.adapters.vectorstore.weaviate import WeaviateVectorStore
    from ragnarok_ai.core.exceptions import VectorStoreConnectionError, VectorStoreDependencyError

    async def main():
        results = {}
        calls = {
            "search": lambda s: s.search([0.1], k=1),
            "delete": lambda s: s.delete(["a"]),
            "is_available": lambda s: s.is_available(),
        }
        for name, call in calls.items():
            try:
                await call(WeaviateVectorStore())
                results[name] = "returned"
            except Exception as exc:
                results[name] = [type(exc).__name__, isinstance(exc, ImportError),
                                 isinstance(exc, VectorStoreConnectionError), str(exc)]
        print(json.dumps(results))

    asyncio.run(main())
    """
    )
    expected = [
        "VectorStoreDependencyError",
        True,
        False,
        "weaviate-client is not installed. Install it with: pip install weaviate-client",
    ]
    assert out == {"search": expected, "delete": expected, "is_available": expected}


# A listener that keeps the first bytes the client sends: a TLS handshake starts with the record type
# 0x16 (handshake) and the major version 0x03, plain HTTP with the request line
FIRST_BYTES = """
    import asyncio, json, os, socket, threading
    from ragnarok_ai.adapters.vectorstore.weaviate import WeaviateVectorStore
    from ragnarok_ai.core.exceptions import VectorStoreConnectionError

    os.environ.pop("WEAVIATE_API_KEY", None)


    def listener():
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(5)
        server.settimeout(20)
        received = []

        def accept():
            try:
                connection, _ = server.accept()
                connection.settimeout(5)
                received.append(connection.recv(16))
                connection.close()
            except OSError as exc:
                received.append(repr(exc).encode())
            server.close()

        thread = threading.Thread(target=accept)
        thread.start()
        return server.getsockname()[1], received, thread


    async def first_bytes(scheme, **kwargs):
        port, received, thread = listener()
        store = WeaviateVectorStore(url=f"{scheme}://127.0.0.1:{port}", grpc_port=1, **kwargs)
        try:
            await store.search([0.1, 0.2], k=1)
            error = "no error"
        except VectorStoreConnectionError:
            error = "VectorStoreConnectionError"
        except Exception as exc:
            error = type(exc).__name__
        thread.join()
        data = received[0]
        kind = "tls handshake" if data[:2] == bytes([0x16, 0x03]) else "plain http" if data[:4] == b"GET " else repr(data)
        return [kind, error]


    async def main():
        print(json.dumps({
            "https, no key": await first_bytes("https"),
            "https, key": await first_bytes("https", api_key="not-a-real-key"),
            "http, no key": await first_bytes("http"),
            "http, key": await first_bytes("http", api_key="not-a-real-key"),
        }))

    asyncio.run(main())
"""


def test_the_scheme_of_the_url_decides_what_the_real_client_sends_with_or_without_a_key():
    """What the client writes on the socket first: an https URL without a key was sent in plain HTTP (#209)."""
    out = run(FIRST_BYTES)
    assert out["https, no key"] == ["tls handshake", "VectorStoreConnectionError"]  # was "plain http"
    assert out["https, key"] == ["tls handshake", "VectorStoreConnectionError"]
    assert out["http, no key"] == ["plain http", "VectorStoreConnectionError"]
    assert out["http, key"] == ["plain http", "VectorStoreConnectionError"]


# An instance with a self-signed certificate: an HTTPS server that answers Weaviate's start-up requests and
# a gRPC server, with TLS, that answers the health check. The certificate is made by the openssl command
SELF_SIGNED = """
    import asyncio, http.server, json, os, ssl, subprocess, sys, tempfile, threading
    from concurrent import futures

    import grpc

    from ragnarok_ai.adapters.vectorstore.weaviate import WeaviateVectorStore
    from ragnarok_ai.core.exceptions import VectorStoreConnectionError

    for name in ("WEAVIATE_API_KEY", "SSL_CERT_FILE", "SSL_CERT_DIR", "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"):
        os.environ.pop(name, None)

    folder = tempfile.mkdtemp()
    cert, key = os.path.join(folder, "cert.pem"), os.path.join(folder, "key.pem")
    made = subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", cert, "-days", "2",
         "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1"],
        capture_output=True, check=False,
    )
    if made.returncode != 0:
        print(json.dumps({"skipped": "openssl could not make a certificate"}))
        sys.exit(0)


    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if "openid" in self.path:  # no OIDC: the instance has no authentication
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"version": "1.30.0", "modules": {}, "grpcMaxMessageSize": 10000000}).encode())

        def log_message(self, *args):
            pass


    web = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    web.socket = context.wrap_socket(web.socket, server_side=True)
    threading.Thread(target=web.serve_forever, daemon=True).start()

    # grpc.health.v1.Health/Check answering SERVING: field 1, varint 1
    health = grpc.method_handlers_generic_handler(
        "grpc.health.v1.Health",
        {"Check": grpc.unary_unary_rpc_method_handler(lambda request, context: bytes([0x08, 0x01]))},
    )
    rpc = grpc.server(futures.ThreadPoolExecutor(2), handlers=[health])
    with open(key, "rb") as k, open(cert, "rb") as c:
        rpc_port = rpc.add_secure_port("127.0.0.1:0", grpc.ssl_server_credentials([(k.read(), c.read())]))
    rpc.start()


    async def connect(**kwargs):
        store = WeaviateVectorStore(url=f"https://localhost:{web.server_address[1]}", grpc_port=rpc_port, **kwargs)

        async def no_collection():
            return None

        store._ensure_collection = no_collection  # the stand-in server has no schema
        try:
            client = await store._ensure_client()
            ready = bool(client.is_ready())
            client.close()
            return ["connected", ready]
        except VectorStoreConnectionError as exc:
            return ["VectorStoreConnectionError", "CERTIFICATE_VERIFY_FAILED" in str(exc)]
        except Exception as exc:
            return [type(exc).__name__, str(exc)[:200]]


    async def main():
        out = {"refused": await connect(), "trust_env alone": await connect(trust_env=True)}
        os.environ["SSL_CERT_FILE"] = cert
        os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = cert
        out["variables without trust_env"] = (await connect())[0]
        out["documented procedure"] = await connect(trust_env=True)
        print(json.dumps(out))

    asyncio.run(main())
    rpc.stop(0)
    web.shutdown()
"""


def test_a_self_signed_certificate_is_refused_and_the_documented_procedure_connects():
    """https now verifies the certificate; trust_env with the two variables is the way to trust a private one."""
    out = run(SELF_SIGNED)
    if "skipped" in out:
        pytest.skip(str(out["skipped"]))
    # refused on the verification of the certificate, as a connection failure that says so
    assert out["refused"] == ["VectorStoreConnectionError", True]
    assert out["trust_env alone"] == ["VectorStoreConnectionError", True]
    # the procedure of the documentation: trust_env=True, SSL_CERT_FILE and GRPC_DEFAULT_SSL_ROOTS_FILE_PATH
    assert out["documented procedure"] == ["connected", True]
    # without trust_env the variables are not enough for every supported version of the client (4.6.2 reads
    # SSL_CERT_FILE by itself, later versions do not): either outcome, never a crash
    assert out["variables without trust_env"] in ("connected", "VectorStoreConnectionError")
