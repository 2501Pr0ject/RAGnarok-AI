"""HTTP daemon for production monitoring.

Provides endpoints for:
- POST /ingest - Receive traces from MonitorClient
- GET /metrics - Prometheus scrape endpoint
- GET /health - Health check
- GET /stats - JSON stats for CLI
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragnarok_ai.monitor.metrics import format_prometheus_metrics
from ragnarok_ai.monitor.models import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    LatencyStats,
    StatsResponse,
    TraceEvent,
)
from ragnarok_ai.monitor.store import DEFAULT_DB_PATH, MonitorStore

if TYPE_CHECKING:
    from asyncio import AbstractServer

# Default configuration
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9090
PID_FILE = Path.home() / ".ragnarok" / "monitor.pid"


class MonitorDaemon:
    """HTTP server daemon for production monitoring."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        db_path: str | Path = DEFAULT_DB_PATH,
        retention_days: int = 7,
    ) -> None:
        """Initialize the daemon.

        Args:
            host: Host to bind to.
            port: Port to listen on.
            db_path: Path to SQLite database.
            retention_days: Days to keep raw traces.
        """
        self.host = host
        self.port = port
        self.store = MonitorStore(db_path=db_path, retention_days=retention_days)
        self.start_time = datetime.now(timezone.utc)
        self._server: AbstractServer | None = None
        self._running = False

    async def handle_request(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle an incoming HTTP request."""
        try:
            # Read request line
            request_line = await asyncio.wait_for(
                reader.readline(), timeout=30.0
            )
            if not request_line:
                return

            request_line = request_line.decode("utf-8").strip()
            parts = request_line.split(" ")
            if len(parts) < 2:
                await self._send_response(writer, HTTPStatus.BAD_REQUEST, "Bad Request")
                return

            method, path = parts[0], parts[1]

            # Read headers
            headers: dict[str, str] = {}
            content_length = 0
            while True:
                line = await reader.readline()
                if line == b"\r\n" or line == b"\n" or not line:
                    break
                line = line.decode("utf-8").strip()
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.strip().lower()] = value.strip()
                    if key.strip().lower() == "content-length":
                        content_length = int(value.strip())

            # Read body if present
            body = b""
            if content_length > 0:
                body = await reader.read(content_length)

            # Route request
            if method == "GET" and path == "/health":
                await self._handle_health(writer)
            elif method == "GET" and path == "/metrics":
                await self._handle_metrics(writer)
            elif method == "GET" and path == "/stats":
                await self._handle_stats(writer)
            elif method == "POST" and path == "/ingest":
                await self._handle_ingest(writer, body)
            else:
                await self._send_response(writer, HTTPStatus.NOT_FOUND, "Not Found")

        except asyncio.TimeoutError:
            await self._send_response(writer, HTTPStatus.REQUEST_TIMEOUT, "Timeout")
        except Exception as e:
            await self._send_response(
                writer, HTTPStatus.INTERNAL_SERVER_ERROR, str(e)
            )
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _handle_health(self, writer: asyncio.StreamWriter) -> None:
        """Handle GET /health."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        response = HealthResponse(
            status="healthy",
            uptime_seconds=uptime,
            traces_collected=self.store.count(),
        )
        await self._send_json(writer, response.model_dump())

    async def _handle_metrics(self, writer: asyncio.StreamWriter) -> None:
        """Handle GET /metrics (Prometheus format)."""
        metrics = format_prometheus_metrics(self.store)
        await self._send_response(
            writer,
            HTTPStatus.OK,
            metrics,
            content_type="text/plain; charset=utf-8",
        )

    async def _handle_stats(self, writer: asyncio.StreamWriter) -> None:
        """Handle GET /stats (JSON stats for CLI)."""
        now = datetime.now(timezone.utc)
        last_1h = now - timedelta(hours=1)
        last_24h = now - timedelta(hours=24)

        uptime = (now - self.start_time).total_seconds()
        total = self.store.count()
        last_hour = self.store.count_since(last_1h)
        success_rate = self.store.get_success_rate(since=last_24h)
        p50, p95, p99 = self.store.get_latency_percentiles(since=last_24h)

        response = StatsResponse(
            uptime_seconds=uptime,
            traces_total=total,
            traces_last_hour=last_hour,
            success_rate=success_rate,
            latency=LatencyStats(
                p50=p50 / 1000.0,  # Convert to seconds
                p95=p95 / 1000.0,
                p99=p99 / 1000.0,
            ),
        )
        await self._send_json(writer, response.model_dump())

    async def _handle_ingest(
        self, writer: asyncio.StreamWriter, body: bytes
    ) -> None:
        """Handle POST /ingest."""
        try:
            data = json.loads(body.decode("utf-8"))
            request = IngestRequest(
                traces=[TraceEvent(**t) for t in data.get("traces", [])]
            )

            # Insert traces
            accepted = self.store.insert_batch(request.traces)

            response = IngestResponse(accepted=accepted, dropped=0)
            await self._send_json(writer, response.model_dump())

        except json.JSONDecodeError:
            await self._send_response(
                writer, HTTPStatus.BAD_REQUEST, "Invalid JSON"
            )
        except Exception as e:
            await self._send_response(
                writer, HTTPStatus.BAD_REQUEST, f"Invalid request: {e}"
            )

    async def _send_response(
        self,
        writer: asyncio.StreamWriter,
        status: HTTPStatus,
        body: str,
        content_type: str = "text/plain",
    ) -> None:
        """Send an HTTP response."""
        response = (
            f"HTTP/1.1 {status.value} {status.phrase}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body.encode('utf-8'))}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
            f"{body}"
        )
        writer.write(response.encode("utf-8"))
        await writer.drain()

    async def _send_json(
        self, writer: asyncio.StreamWriter, data: dict[str, Any]
    ) -> None:
        """Send a JSON response."""
        body = json.dumps(data)
        await self._send_response(
            writer,
            HTTPStatus.OK,
            body,
            content_type="application/json",
        )

    async def start(self) -> None:
        """Start the HTTP server."""
        self._server = await asyncio.start_server(
            self.handle_request, self.host, self.port
        )
        self._running = True
        self.start_time = datetime.now(timezone.utc)

    async def serve_forever(self) -> None:
        """Serve requests until stopped."""
        if self._server is None:
            await self.start()

        if self._server:
            async with self._server:
                await self._server.serve_forever()

    async def stop(self) -> None:
        """Stop the server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        self.store.close()


def write_pid_file() -> None:
    """Write current PID to file."""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def remove_pid_file() -> None:
    """Remove PID file."""
    if PID_FILE.exists():
        PID_FILE.unlink()


def read_pid() -> int | None:
    """Read PID from file."""
    if PID_FILE.exists():
        try:
            return int(PID_FILE.read_text().strip())
        except (ValueError, OSError):
            return None
    return None


def is_daemon_running() -> bool:
    """Check if daemon is running."""
    pid = read_pid()
    if pid is None:
        return False

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        # Process not running, clean up stale PID file
        remove_pid_file()
        return False


async def run_daemon(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    db_path: str | Path = DEFAULT_DB_PATH,
    retention_days: int = 7,
) -> None:
    """Run the monitor daemon.

    Args:
        host: Host to bind to.
        port: Port to listen on.
        db_path: Path to SQLite database.
        retention_days: Days to keep raw traces.
    """
    daemon = MonitorDaemon(
        host=host,
        port=port,
        db_path=db_path,
        retention_days=retention_days,
    )

    # Set up signal handlers
    loop = asyncio.get_running_loop()
    shutdown_task: asyncio.Task[None] | None = None

    def handle_signal() -> None:
        nonlocal shutdown_task
        shutdown_task = loop.create_task(daemon.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_signal)

    write_pid_file()
    try:
        await daemon.serve_forever()
    finally:
        remove_pid_file()


def stop_daemon() -> bool:
    """Stop a running daemon.

    Returns:
        True if daemon was stopped, False if not running.
    """
    pid = read_pid()
    if pid is None:
        return False

    try:
        os.kill(pid, signal.SIGTERM)
        # Wait for process to exit
        for _ in range(50):  # 5 seconds max
            try:
                os.kill(pid, 0)
                import time
                time.sleep(0.1)
            except OSError:
                break
        remove_pid_file()
        return True
    except OSError:
        remove_pid_file()
        return False


def daemonize() -> None:
    """Fork into background daemon process (Unix only)."""
    if sys.platform == "win32":
        raise RuntimeError("Daemonization not supported on Windows")

    # First fork
    pid = os.fork()
    if pid > 0:
        sys.exit(0)

    # Decouple from parent
    os.setsid()

    # Second fork
    pid = os.fork()
    if pid > 0:
        sys.exit(0)

    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = Path("/dev/null")
    with devnull.open("rb") as f:
        os.dup2(f.fileno(), sys.stdin.fileno())
    with devnull.open("ab") as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
    with devnull.open("ab") as f:
        os.dup2(f.fileno(), sys.stderr.fileno())
