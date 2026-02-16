"""Tests for monitor daemon."""

import asyncio
import json
from collections.abc import AsyncGenerator

import pytest

from ragnarok_ai.monitor.daemon import MonitorDaemon


@pytest.fixture
async def daemon() -> AsyncGenerator[MonitorDaemon, None]:
    """Create a test daemon with in-memory store."""
    d = MonitorDaemon(
        host="127.0.0.1",
        port=0,  # Let OS assign port
        db_path=":memory:",
    )
    await d.start()
    yield d
    await d.stop()


def get_daemon_port(daemon: MonitorDaemon) -> int:
    """Get the actual port the daemon is listening on."""
    if daemon._server is None:
        raise RuntimeError("Daemon not started")
    sockets = daemon._server.sockets
    if not sockets:
        raise RuntimeError("No sockets available")
    return sockets[0].getsockname()[1]


async def http_request(host: str, port: int, method: str, path: str, body: str | None = None) -> tuple[int, str, str]:
    """Make a simple HTTP request.

    Returns:
        Tuple of (status_code, content_type, body).
    """
    reader, writer = await asyncio.open_connection(host, port)

    try:
        # Build request
        request_lines = [f"{method} {path} HTTP/1.1", f"Host: {host}:{port}"]

        if body:
            request_lines.append(f"Content-Length: {len(body)}")
            request_lines.append("Content-Type: application/json")

        request_lines.append("")
        request_lines.append(body or "")

        request = "\r\n".join(request_lines)
        writer.write(request.encode())
        await writer.drain()

        # Read response
        response = await asyncio.wait_for(reader.read(65536), timeout=5.0)
        response_str = response.decode("utf-8")

        # Parse response
        header_end = response_str.find("\r\n\r\n")
        if header_end == -1:
            header_end = response_str.find("\n\n")
            body_start = header_end + 2
        else:
            body_start = header_end + 4

        headers = response_str[:header_end]
        resp_body = response_str[body_start:]

        # Parse status
        status_line = headers.split("\r\n")[0].split("\n")[0]
        status_code = int(status_line.split()[1])

        # Parse content-type
        content_type = "text/plain"
        for line in headers.split("\n"):
            if line.lower().startswith("content-type:"):
                content_type = line.split(":", 1)[1].strip()
                break

        return status_code, content_type, resp_body

    finally:
        writer.close()
        await writer.wait_closed()


class TestMonitorDaemonEndpoints:
    """Test HTTP endpoints."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, daemon: MonitorDaemon) -> None:
        """Test GET /health."""
        port = get_daemon_port(daemon)
        status, content_type, body = await http_request("127.0.0.1", port, "GET", "/health")

        assert status == 200
        assert "application/json" in content_type

        data = json.loads(body)
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data
        assert "traces_collected" in data

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, daemon: MonitorDaemon) -> None:
        """Test GET /metrics."""
        port = get_daemon_port(daemon)
        status, content_type, body = await http_request("127.0.0.1", port, "GET", "/metrics")

        assert status == 200
        assert "text/plain" in content_type
        assert "ragnarok_requests_total" in body
        assert "ragnarok_success_rate" in body

    @pytest.mark.asyncio
    async def test_stats_endpoint(self, daemon: MonitorDaemon) -> None:
        """Test GET /stats."""
        port = get_daemon_port(daemon)
        status, content_type, body = await http_request("127.0.0.1", port, "GET", "/stats")

        assert status == 200
        assert "application/json" in content_type

        data = json.loads(body)
        assert "uptime_seconds" in data
        assert "traces_total" in data
        assert "success_rate" in data
        assert "latency" in data
        assert "p50" in data["latency"]
        assert "p95" in data["latency"]
        assert "p99" in data["latency"]

    @pytest.mark.asyncio
    async def test_ingest_endpoint(self, daemon: MonitorDaemon) -> None:
        """Test POST /ingest."""
        port = get_daemon_port(daemon)

        traces = [
            {
                "query_hash": "abc123",
                "query_length": 42,
                "total_latency_ms": 100.5,
            }
        ]
        body = json.dumps({"traces": traces})

        status, content_type, resp_body = await http_request("127.0.0.1", port, "POST", "/ingest", body)

        assert status == 200
        assert "application/json" in content_type

        data = json.loads(resp_body)
        assert data["accepted"] == 1
        assert data["dropped"] == 0

        # Verify trace was stored
        assert daemon.store.count() == 1

    @pytest.mark.asyncio
    async def test_ingest_batch(self, daemon: MonitorDaemon) -> None:
        """Test ingesting multiple traces."""
        port = get_daemon_port(daemon)

        traces = [
            {
                "query_hash": f"hash{i}",
                "query_length": i * 10,
                "total_latency_ms": float(i * 100),
            }
            for i in range(5)
        ]
        body = json.dumps({"traces": traces})

        status, _, resp_body = await http_request("127.0.0.1", port, "POST", "/ingest", body)

        assert status == 200
        data = json.loads(resp_body)
        assert data["accepted"] == 5

        assert daemon.store.count() == 5

    @pytest.mark.asyncio
    async def test_ingest_invalid_json(self, daemon: MonitorDaemon) -> None:
        """Test POST /ingest with invalid JSON."""
        port = get_daemon_port(daemon)

        status, _, _ = await http_request("127.0.0.1", port, "POST", "/ingest", "not json")

        assert status == 400

    @pytest.mark.asyncio
    async def test_not_found(self, daemon: MonitorDaemon) -> None:
        """Test 404 for unknown path."""
        port = get_daemon_port(daemon)

        status, _, _ = await http_request("127.0.0.1", port, "GET", "/unknown")

        assert status == 404


class TestMonitorDaemonLifecycle:
    """Test daemon lifecycle."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        """Test starting and stopping daemon."""
        daemon = MonitorDaemon(
            host="127.0.0.1",
            port=0,
            db_path=":memory:",
        )

        await daemon.start()
        assert daemon._server is not None
        assert daemon._running is True

        await daemon.stop()
        assert daemon._running is False

    @pytest.mark.asyncio
    async def test_uptime_tracking(self, daemon: MonitorDaemon) -> None:
        """Test that uptime is tracked."""
        # Wait a bit
        await asyncio.sleep(0.1)

        port = get_daemon_port(daemon)
        _, _, body = await http_request("127.0.0.1", port, "GET", "/health")

        data = json.loads(body)
        assert data["uptime_seconds"] >= 0.1
