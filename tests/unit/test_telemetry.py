"""Unit tests for the telemetry module."""

from __future__ import annotations

import pytest

from ragnarok_ai.core.types import Document, RAGResponse
from ragnarok_ai.telemetry.exporters import OTLPExporter, create_otlp_exporter
from ragnarok_ai.telemetry.tracer import (
    NoOpTracer,
    RAGTracer,
    SpanContext,
    _normalize_attribute,
    get_tracer,
    set_tracer,
)

# ============================================================================
# SpanContext Tests
# ============================================================================


class TestSpanContext:
    """Tests for SpanContext."""

    def test_set_attribute(self) -> None:
        """Test setting attributes on span context."""
        tracer = NoOpTracer()
        ctx = SpanContext("test", tracer)
        ctx.set_attribute("key", "value")
        assert ctx._attributes["key"] == "value"

    def test_set_multiple_attributes(self) -> None:
        """Test setting multiple attributes."""
        tracer = NoOpTracer()
        ctx = SpanContext("test", tracer, {"initial": "attr"})
        ctx.set_attribute("key1", "value1")
        ctx.set_attribute("key2", 42)
        assert ctx._attributes["initial"] == "attr"
        assert ctx._attributes["key1"] == "value1"
        assert ctx._attributes["key2"] == 42

    def test_set_status_ok_no_span(self) -> None:
        """Test set_status_ok with no span (no-op)."""
        tracer = NoOpTracer()
        ctx = SpanContext("test", tracer)
        # Should not raise
        ctx.set_status_ok()

    def test_set_status_error_no_span(self) -> None:
        """Test set_status_error with no span (no-op)."""
        tracer = NoOpTracer()
        ctx = SpanContext("test", tracer)
        # Should not raise
        ctx.set_status_error("error message")

    def test_record_exception_no_span(self) -> None:
        """Test record_exception with no span (no-op)."""
        tracer = NoOpTracer()
        ctx = SpanContext("test", tracer)
        # Should not raise
        ctx.record_exception(ValueError("test error"))


# ============================================================================
# NoOpTracer Tests
# ============================================================================


class TestNoOpTracer:
    """Tests for NoOpTracer."""

    def test_init(self) -> None:
        """Test NoOpTracer initialization."""
        tracer = NoOpTracer()
        assert tracer._service_name == "noop"
        assert not tracer.enabled

    def test_start_span(self) -> None:
        """Test starting a span with NoOpTracer."""
        tracer = NoOpTracer()
        with tracer.start_span("test_span") as span:
            assert isinstance(span, SpanContext)
            span.set_attribute("key", "value")

    def test_start_span_with_attributes(self) -> None:
        """Test starting a span with initial attributes."""
        tracer = NoOpTracer()
        with tracer.start_span("test_span", attributes={"key": "value"}) as span:
            assert span._attributes["key"] == "value"

    def test_shutdown(self) -> None:
        """Test NoOpTracer shutdown (no-op)."""
        tracer = NoOpTracer()
        tracer.shutdown()  # Should not raise

    def test_nested_spans(self) -> None:
        """Test nested spans with NoOpTracer."""
        tracer = NoOpTracer()
        with tracer.start_span("parent") as parent:
            parent.set_attribute("level", "parent")
            with tracer.start_span("child") as child:
                child.set_attribute("level", "child")
                assert child._attributes["level"] == "child"
            assert parent._attributes["level"] == "parent"


# ============================================================================
# RAGTracer Tests
# ============================================================================


class TestRAGTracer:
    """Tests for RAGTracer."""

    def test_init_without_otel(self) -> None:
        """Test RAGTracer initialization without OpenTelemetry."""
        # This test checks that RAGTracer gracefully handles missing OTel
        tracer = RAGTracer(service_name="test-service")
        # enabled depends on whether OTel is installed
        assert tracer._service_name == "test-service"

    def test_start_span_without_otel(self) -> None:
        """Test starting a span without OpenTelemetry."""
        tracer = RAGTracer(service_name="test-service")
        with tracer.start_span("test_span") as span:
            assert isinstance(span, SpanContext)
            span.set_attribute("key", "value")

    def test_shutdown(self) -> None:
        """Test RAGTracer shutdown."""
        tracer = RAGTracer(service_name="test-service")
        tracer.shutdown()  # Should not raise


# ============================================================================
# Global Tracer Tests
# ============================================================================


class TestGlobalTracer:
    """Tests for global tracer functions."""

    def test_get_tracer_returns_noop_when_not_set(self) -> None:
        """Test get_tracer returns NoOpTracer when not set."""
        set_tracer(None)
        tracer = get_tracer()
        assert isinstance(tracer, NoOpTracer)

    def test_set_and_get_tracer(self) -> None:
        """Test setting and getting global tracer."""
        custom_tracer = RAGTracer(service_name="custom")
        set_tracer(custom_tracer)
        try:
            assert get_tracer() is custom_tracer
        finally:
            set_tracer(None)

    def test_set_tracer_to_none(self) -> None:
        """Test setting tracer to None."""
        custom_tracer = RAGTracer(service_name="custom")
        set_tracer(custom_tracer)
        set_tracer(None)
        tracer = get_tracer()
        assert isinstance(tracer, NoOpTracer)


# ============================================================================
# Attribute Normalization Tests
# ============================================================================


class TestNormalizeAttribute:
    """Tests for _normalize_attribute."""

    def test_normalize_none(self) -> None:
        """Test normalizing None."""
        assert _normalize_attribute(None) == ""

    def test_normalize_string(self) -> None:
        """Test normalizing string."""
        assert _normalize_attribute("test") == "test"

    def test_normalize_int(self) -> None:
        """Test normalizing int."""
        assert _normalize_attribute(42) == 42

    def test_normalize_float(self) -> None:
        """Test normalizing float."""
        assert _normalize_attribute(3.14) == 3.14

    def test_normalize_bool(self) -> None:
        """Test normalizing bool."""
        assert _normalize_attribute(True) is True
        assert _normalize_attribute(False) is False

    def test_normalize_list(self) -> None:
        """Test normalizing list."""
        result = _normalize_attribute([1, "two", 3.0])
        assert result == [1, "two", 3.0]

    def test_normalize_object(self) -> None:
        """Test normalizing complex object."""
        obj = {"key": "value"}
        assert _normalize_attribute(obj) == "{'key': 'value'}"

    def test_normalize_nested_list(self) -> None:
        """Test normalizing nested list."""
        result = _normalize_attribute([1, [2, 3], "four"])
        assert result == [1, [2, 3], "four"]


# ============================================================================
# OTLPExporter Tests
# ============================================================================


class TestOTLPExporter:
    """Tests for OTLPExporter."""

    def test_init_defaults(self) -> None:
        """Test OTLPExporter with default values."""
        exporter = OTLPExporter()
        assert exporter.endpoint == "http://localhost:4317"
        assert exporter.protocol == "grpc"

    def test_init_custom_endpoint(self) -> None:
        """Test OTLPExporter with custom endpoint."""
        exporter = OTLPExporter(endpoint="http://custom:4317")
        assert exporter.endpoint == "http://custom:4317"

    def test_init_http_protocol(self) -> None:
        """Test OTLPExporter with HTTP protocol."""
        exporter = OTLPExporter(
            endpoint="http://localhost:4318/v1/traces",
            protocol="http",
        )
        assert exporter.protocol == "http"

    def test_init_with_headers(self) -> None:
        """Test OTLPExporter with custom headers."""
        exporter = OTLPExporter(headers={"Authorization": "Bearer token"})
        assert exporter._headers == {"Authorization": "Bearer token"}

    def test_get_span_exporter_grpc_without_package(self) -> None:
        """Test getting gRPC exporter without package installed."""
        exporter = OTLPExporter(protocol="grpc")
        # This may or may not raise depending on whether OTel is installed
        try:
            exporter.get_span_exporter()
        except ImportError as e:
            assert "opentelemetry-exporter-otlp-proto-grpc" in str(e)

    def test_get_span_exporter_http_without_package(self) -> None:
        """Test getting HTTP exporter without package installed."""
        exporter = OTLPExporter(protocol="http")
        try:
            exporter.get_span_exporter()
        except ImportError as e:
            assert "opentelemetry-exporter-otlp-proto-http" in str(e)


class TestCreateOTLPExporter:
    """Tests for create_otlp_exporter factory function."""

    def test_create_with_defaults(self) -> None:
        """Test creating exporter with defaults."""
        exporter = create_otlp_exporter()
        assert isinstance(exporter, OTLPExporter)
        assert exporter.endpoint == "http://localhost:4317"

    def test_create_with_custom_options(self) -> None:
        """Test creating exporter with custom options."""
        exporter = create_otlp_exporter(
            endpoint="http://custom:4317",
            protocol="http",
            headers={"X-Custom": "header"},
            timeout=60,
            insecure=False,
        )
        assert exporter.endpoint == "http://custom:4317"
        assert exporter.protocol == "http"
        assert exporter._headers == {"X-Custom": "header"}
        assert exporter._timeout == 60
        assert exporter._insecure is False


# ============================================================================
# Integration with Evaluate Tests
# ============================================================================


class MockRAGPipeline:
    """Mock RAG pipeline for testing."""

    def __init__(self, response: RAGResponse) -> None:
        self._response = response

    async def query(self, question: str) -> RAGResponse:  # noqa: ARG002
        return self._response


class TestEvaluateWithTelemetry:
    """Tests for evaluate function with telemetry."""

    @pytest.mark.asyncio
    async def test_evaluate_with_noop_tracer(self) -> None:
        """Test evaluate with NoOpTracer."""
        from ragnarok_ai.core.evaluate import evaluate
        from ragnarok_ai.core.types import Query, TestSet

        pipeline = MockRAGPipeline(
            RAGResponse(
                answer="Paris",
                retrieved_docs=[Document(id="doc1", content="France info")],
            )
        )
        testset = TestSet(queries=[Query(text="What is the capital?", ground_truth_docs=["doc1"])])

        tracer = NoOpTracer()
        result = await evaluate(pipeline, testset, tracer=tracer)

        assert result is not None
        assert len(result.responses) == 1

    @pytest.mark.asyncio
    async def test_evaluate_with_rag_tracer(self) -> None:
        """Test evaluate with RAGTracer (OTel may or may not be installed)."""
        from ragnarok_ai.core.evaluate import evaluate
        from ragnarok_ai.core.types import Query, TestSet

        pipeline = MockRAGPipeline(
            RAGResponse(
                answer="Paris",
                retrieved_docs=[Document(id="doc1", content="France info")],
            )
        )
        testset = TestSet(queries=[Query(text="What is the capital?", ground_truth_docs=["doc1"])])

        tracer = RAGTracer(service_name="test-eval")
        result = await evaluate(pipeline, testset, tracer=tracer)

        assert result is not None
        assert len(result.responses) == 1
        tracer.shutdown()

    @pytest.mark.asyncio
    async def test_evaluate_without_tracer(self) -> None:
        """Test evaluate without tracer (uses NoOpTracer internally)."""
        from ragnarok_ai.core.evaluate import evaluate
        from ragnarok_ai.core.types import Query, TestSet

        pipeline = MockRAGPipeline(
            RAGResponse(
                answer="Paris",
                retrieved_docs=[Document(id="doc1", content="France info")],
            )
        )
        testset = TestSet(queries=[Query(text="What is the capital?", ground_truth_docs=["doc1"])])

        result = await evaluate(pipeline, testset)

        assert result is not None
        assert len(result.responses) == 1

    @pytest.mark.asyncio
    async def test_evaluate_stream_with_tracer(self) -> None:
        """Test evaluate_stream with tracer."""
        from ragnarok_ai.core.evaluate import evaluate_stream
        from ragnarok_ai.core.types import Query, TestSet

        pipeline = MockRAGPipeline(
            RAGResponse(
                answer="Paris",
                retrieved_docs=[Document(id="doc1", content="France info")],
            )
        )
        testset = TestSet(
            queries=[
                Query(text="Q1?", ground_truth_docs=["doc1"]),
                Query(text="Q2?", ground_truth_docs=["doc1"]),
            ]
        )

        tracer = NoOpTracer()
        results = []
        async for query, metric, answer in evaluate_stream(pipeline, testset, tracer=tracer):
            results.append((query, metric, answer))

        assert len(results) == 2


# ============================================================================
# Module Import Tests
# ============================================================================


class TestModuleImports:
    """Tests for module imports."""

    def test_import_from_telemetry(self) -> None:
        """Test importing from telemetry module."""
        from ragnarok_ai.telemetry import (
            OTLPExporter,
            RAGTracer,
            create_otlp_exporter,
            get_tracer,
            set_tracer,
        )

        assert OTLPExporter is not None
        assert RAGTracer is not None
        assert create_otlp_exporter is not None
        assert get_tracer is not None
        assert set_tracer is not None
