"""Tests for trajectory analysis and visualization."""

from __future__ import annotations

import pytest

from ragnarok_ai.agents.analysis import (
    FailurePoint,
    TrajectoryAnalyzer,
    TrajectorySummary,
    export_trajectory_html,
    format_trajectory_ascii,
    format_trajectory_mermaid,
    generate_trajectory_html,
    print_trajectory,
)
from ragnarok_ai.agents.types import AgentResponse, AgentStep, ToolCall

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_response() -> AgentResponse:
    """Simple response with a few steps."""
    return AgentResponse(
        answer="Paris is the capital of France",
        steps=[
            AgentStep(step_type="thought", content="I need to search for information"),
            AgentStep(
                step_type="action",
                content="search",
                tool_call=ToolCall(
                    name="search",
                    input={"query": "capital of France"},
                    output="Paris is the capital of France",
                ),
            ),
            AgentStep(step_type="observation", content="Paris is the capital of France"),
            AgentStep(step_type="final_answer", content="Paris is the capital of France"),
        ],
        total_latency_ms=1500.0,
    )


@pytest.fixture
def response_with_failures() -> AgentResponse:
    """Response with failed tool calls."""
    return AgentResponse(
        answer="Could not find information",
        steps=[
            AgentStep(step_type="thought", content="I need to search"),
            AgentStep(
                step_type="action",
                content="search",
                tool_call=ToolCall(
                    name="search",
                    input={"query": "test"},
                    output="",
                    error="API timeout",
                ),
            ),
            AgentStep(step_type="thought", content="Search failed, trying different approach"),
            AgentStep(
                step_type="action",
                content="lookup",
                tool_call=ToolCall(
                    name="lookup",
                    input={"term": "test"},
                    output="No results",
                ),
            ),
            AgentStep(step_type="final_answer", content="Could not find information"),
        ],
        total_latency_ms=3000.0,
    )


@pytest.fixture
def response_with_loop() -> AgentResponse:
    """Response with a reasoning loop."""
    return AgentResponse(
        answer="42",
        steps=[
            AgentStep(step_type="thought", content="I need to calculate the answer"),
            AgentStep(
                step_type="action",
                content="calculate",
                tool_call=ToolCall(
                    name="calculate",
                    input={"expression": "6 * 7"},
                    output="42",
                ),
            ),
            AgentStep(
                step_type="action",
                content="calculate",
                tool_call=ToolCall(
                    name="calculate",
                    input={"expression": "6 * 7"},
                    output="42",
                ),
            ),
            AgentStep(step_type="final_answer", content="42"),
        ],
        total_latency_ms=500.0,
    )


@pytest.fixture
def empty_response() -> AgentResponse:
    """Response with no steps."""
    return AgentResponse(answer="", steps=[], total_latency_ms=0.0)


# =============================================================================
# TrajectorySummary Tests
# =============================================================================


class TestTrajectorySummary:
    """Tests for TrajectorySummary dataclass."""

    def test_str_basic(self) -> None:
        """Test basic string representation."""
        summary = TrajectorySummary(
            total_steps=5,
            tool_calls={"search": 2, "calculate": 1},
            total_latency_ms=2300.0,
        )
        result = str(summary)
        assert "Steps: 5" in result
        assert "search (2x)" in result
        assert "calculate (1x)" in result
        assert "2.3s" in result

    def test_str_with_issues(self) -> None:
        """Test string representation with issues."""
        summary = TrajectorySummary(
            total_steps=10,
            loops_detected=2,
            dead_ends=1,
            failed_tool_calls=3,
        )
        result = str(summary)
        assert "Loops: 2" in result
        assert "Dead ends: 1" in result
        assert "Failed: 3" in result

    def test_str_milliseconds(self) -> None:
        """Test latency shown in ms for short durations."""
        summary = TrajectorySummary(total_steps=1, total_latency_ms=500.0)
        result = str(summary)
        assert "500ms" in result

    def test_str_no_tools(self) -> None:
        """Test summary without tool calls."""
        summary = TrajectorySummary(total_steps=2)
        result = str(summary)
        assert "Steps: 2" in result
        assert "Tools" not in result


# =============================================================================
# FailurePoint Tests
# =============================================================================


class TestFailurePoint:
    """Tests for FailurePoint dataclass."""

    def test_creation(self) -> None:
        """Test basic creation."""
        failure = FailurePoint(
            step_index=3,
            failure_type="tool_error",
            description="API call failed",
            severity="error",
        )
        assert failure.step_index == 3
        assert failure.failure_type == "tool_error"
        assert failure.description == "API call failed"
        assert failure.severity == "error"

    def test_default_severity(self) -> None:
        """Test default severity is warning."""
        failure = FailurePoint(
            step_index=0,
            failure_type="loop",
            description="Loop detected",
        )
        assert failure.severity == "warning"

    def test_frozen(self) -> None:
        """Test that FailurePoint is immutable."""
        failure = FailurePoint(
            step_index=0,
            failure_type="dead_end",
            description="Dead end",
        )
        with pytest.raises(AttributeError):
            failure.step_index = 1  # type: ignore[misc]


# =============================================================================
# TrajectoryAnalyzer Tests
# =============================================================================


class TestTrajectoryAnalyzer:
    """Tests for TrajectoryAnalyzer."""

    def test_analyze_basic(self, simple_response: AgentResponse) -> None:
        """Test basic analysis."""
        analyzer = TrajectoryAnalyzer()
        summary = analyzer.analyze(simple_response)

        assert summary.total_steps == 4
        assert summary.tool_calls == {"search": 1}
        assert summary.total_latency_ms == 1500.0
        assert summary.step_types == {
            "thought": 1,
            "action": 1,
            "observation": 1,
            "final_answer": 1,
        }
        assert summary.failed_tool_calls == 0

    def test_analyze_with_failures(self, response_with_failures: AgentResponse) -> None:
        """Test analysis with failed tool calls."""
        analyzer = TrajectoryAnalyzer()
        summary = analyzer.analyze(response_with_failures)

        assert summary.total_steps == 5
        assert summary.failed_tool_calls == 1
        assert summary.tool_calls == {"search": 1, "lookup": 1}

    def test_analyze_empty(self, empty_response: AgentResponse) -> None:
        """Test analysis of empty response."""
        analyzer = TrajectoryAnalyzer()
        summary = analyzer.analyze(empty_response)

        assert summary.total_steps == 0
        assert summary.tool_calls == {}
        assert summary.loops_detected == 0

    def test_find_failures_tool_error(self, response_with_failures: AgentResponse) -> None:
        """Test finding tool errors."""
        analyzer = TrajectoryAnalyzer()
        failures = analyzer.find_failures(response_with_failures)

        tool_errors = [f for f in failures if f.failure_type == "tool_error"]
        assert len(tool_errors) == 1
        assert tool_errors[0].step_index == 1
        assert "search" in tool_errors[0].description
        assert "API timeout" in tool_errors[0].description

    def test_find_failures_dead_end(self, response_with_failures: AgentResponse) -> None:
        """Test finding dead ends."""
        analyzer = TrajectoryAnalyzer()
        failures = analyzer.find_failures(response_with_failures)

        dead_ends = [f for f in failures if f.failure_type == "dead_end"]
        assert len(dead_ends) == 1

    def test_detect_loops(self, response_with_loop: AgentResponse) -> None:
        """Test loop detection."""
        analyzer = TrajectoryAnalyzer()
        loops = analyzer.detect_loops(response_with_loop)

        assert len(loops) == 1
        assert loops[0] == (1, 2)  # Consecutive identical tool calls

    def test_detect_loops_similar_thoughts(self) -> None:
        """Test loop detection for similar thoughts."""
        response = AgentResponse(
            answer="done",
            steps=[
                AgentStep(step_type="thought", content="I need to search for the answer"),
                AgentStep(
                    step_type="action",
                    content="search",
                    tool_call=ToolCall(name="search", input={"q": "test"}, output="result"),
                ),
                AgentStep(step_type="observation", content="result"),
                AgentStep(step_type="thought", content="I need to search for the answer"),  # Similar
                AgentStep(step_type="final_answer", content="done"),
            ],
        )
        analyzer = TrajectoryAnalyzer(similarity_threshold=0.85)
        loops = analyzer.detect_loops(response)

        # Should detect similar thoughts at indices 0 and 3
        assert any(loop == (0, 3) for loop in loops)

    def test_detect_dead_ends_no_retry(self) -> None:
        """Test dead end detection when failed tool is not retried."""
        response = AgentResponse(
            answer="gave up",
            steps=[
                AgentStep(
                    step_type="action",
                    content="api_call",
                    tool_call=ToolCall(name="api_call", input={}, output="", error="Failed"),
                ),
                AgentStep(step_type="thought", content="I'll try something else"),
                AgentStep(
                    step_type="action",
                    content="different_tool",
                    tool_call=ToolCall(name="different_tool", input={}, output="ok"),
                ),
            ],
        )
        analyzer = TrajectoryAnalyzer()
        dead_ends = analyzer.detect_dead_ends(response)

        assert 0 in dead_ends

    def test_detect_dead_ends_with_retry(self) -> None:
        """Test that retried tools are not marked as dead ends."""
        response = AgentResponse(
            answer="success",
            steps=[
                AgentStep(
                    step_type="action",
                    content="api_call",
                    tool_call=ToolCall(name="api_call", input={}, output="", error="Failed"),
                ),
                AgentStep(step_type="thought", content="Retry"),
                AgentStep(
                    step_type="action",
                    content="api_call",
                    tool_call=ToolCall(name="api_call", input={}, output="ok"),
                ),
            ],
        )
        analyzer = TrajectoryAnalyzer()
        dead_ends = analyzer.detect_dead_ends(response)

        assert 0 not in dead_ends

    def test_failures_sorted_by_index(self, response_with_failures: AgentResponse) -> None:
        """Test that failures are sorted by step index."""
        analyzer = TrajectoryAnalyzer()
        failures = analyzer.find_failures(response_with_failures)

        indices = [f.step_index for f in failures]
        assert indices == sorted(indices)


# =============================================================================
# ASCII Visualization Tests
# =============================================================================


class TestFormatTrajectoryAscii:
    """Tests for ASCII visualization."""

    def test_basic_formatting(self, simple_response: AgentResponse) -> None:
        """Test basic ASCII output."""
        result = format_trajectory_ascii(simple_response)

        assert "┌" in result
        assert "└" in result
        assert "Step 1:" in result
        assert "thought" in result
        assert "search" in result

    def test_empty_trajectory(self, empty_response: AgentResponse) -> None:
        """Test empty trajectory output."""
        result = format_trajectory_ascii(empty_response)
        assert "Empty trajectory" in result

    def test_tool_call_details(self, simple_response: AgentResponse) -> None:
        """Test that tool call details are shown."""
        result = format_trajectory_ascii(simple_response)

        assert "Input:" in result
        assert "Output:" in result

    def test_max_width(self, simple_response: AgentResponse) -> None:
        """Test max width constraint."""
        result = format_trajectory_ascii(simple_response, max_width=40)

        for line in result.split("\n"):
            # Lines should not exceed max_width
            assert len(line) <= 40

    def test_error_display(self, response_with_failures: AgentResponse) -> None:
        """Test that errors are displayed."""
        result = format_trajectory_ascii(response_with_failures)
        assert "Error:" in result


class TestPrintTrajectory:
    """Tests for print_trajectory function."""

    def test_prints_output(self, simple_response: AgentResponse, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that print_trajectory outputs to stdout."""
        print_trajectory(simple_response)
        captured = capsys.readouterr()
        assert "Step 1:" in captured.out


# =============================================================================
# Mermaid Visualization Tests
# =============================================================================


class TestFormatTrajectoryMermaid:
    """Tests for Mermaid visualization."""

    def test_basic_mermaid(self, simple_response: AgentResponse) -> None:
        """Test basic Mermaid output."""
        result = format_trajectory_mermaid(simple_response)

        assert "```mermaid" in result
        assert "graph TD" in result
        assert "```" in result.split("```mermaid")[1]

    def test_empty_trajectory(self, empty_response: AgentResponse) -> None:
        """Test empty trajectory Mermaid output."""
        result = format_trajectory_mermaid(empty_response)
        assert "No steps" in result

    def test_node_connections(self, simple_response: AgentResponse) -> None:
        """Test that nodes are connected."""
        result = format_trajectory_mermaid(simple_response)
        assert "-->" in result

    def test_step_type_prefixes(self, simple_response: AgentResponse) -> None:
        """Test that step types get correct prefixes."""
        result = format_trajectory_mermaid(simple_response)

        # Should have T for thought, A for action, O for observation, F for final_answer
        assert "T1[" in result
        assert "A2[" in result
        assert "O3[" in result
        assert "F4[" in result

    def test_failed_step_styling(self, response_with_failures: AgentResponse) -> None:
        """Test that failed steps get error styling."""
        result = format_trajectory_mermaid(response_with_failures)
        assert "fill:#ffcccc" in result

    def test_emoji_in_labels(self, simple_response: AgentResponse) -> None:
        """Test that labels contain emojis."""
        result = format_trajectory_mermaid(simple_response)
        # Check for thought emoji
        assert "💭" in result or "🔧" in result or "👁" in result or "✅" in result


# =============================================================================
# HTML Visualization Tests
# =============================================================================


class TestGenerateTrajectoryHtml:
    """Tests for HTML generation."""

    def test_basic_html(self, simple_response: AgentResponse) -> None:
        """Test basic HTML output."""
        result = generate_trajectory_html(simple_response)

        assert "<!DOCTYPE html>" in result
        assert "<html" in result
        assert "</html>" in result

    def test_title_in_html(self, simple_response: AgentResponse) -> None:
        """Test that title appears in HTML."""
        result = generate_trajectory_html(simple_response, title="Test Trajectory")
        assert "Test Trajectory" in result

    def test_steps_in_html(self, simple_response: AgentResponse) -> None:
        """Test that steps appear in HTML."""
        result = generate_trajectory_html(simple_response)

        assert "Step 1" in result
        assert "thought" in result.lower()

    def test_summary_stats(self, simple_response: AgentResponse) -> None:
        """Test that summary stats appear."""
        result = generate_trajectory_html(simple_response)

        assert "Steps" in result
        assert "4" in result  # 4 steps
        assert "1500" in result  # latency

    def test_tool_usage_summary(self, simple_response: AgentResponse) -> None:
        """Test that tool usage is summarized."""
        result = generate_trajectory_html(simple_response)
        assert "search" in result

    def test_html_escaping(self) -> None:
        """Test that content is properly HTML escaped."""
        response = AgentResponse(
            answer="<script>alert('xss')</script>",
            steps=[
                AgentStep(
                    step_type="thought",
                    content="Testing <b>HTML</b> & entities",
                ),
            ],
        )
        result = generate_trajectory_html(response)

        # Should be escaped
        assert "<script>" not in result
        assert "&lt;script&gt;" in result or "script" not in result.split("<style>")[0]
        assert "&lt;b&gt;" in result or "&amp;" in result

    def test_error_styling(self, response_with_failures: AgentResponse) -> None:
        """Test that errors get special styling."""
        result = generate_trajectory_html(response_with_failures)
        assert "error" in result.lower()


class TestExportTrajectoryHtml:
    """Tests for HTML export function."""

    def test_export_creates_file(self, simple_response: AgentResponse, tmp_path: pytest.TempPathFactory) -> None:
        """Test that export creates a file."""
        output_path = tmp_path / "trajectory.html"  # type: ignore[operator]
        export_trajectory_html(simple_response, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content

    def test_export_with_custom_title(self, simple_response: AgentResponse, tmp_path: pytest.TempPathFactory) -> None:
        """Test export with custom title."""
        output_path = tmp_path / "custom.html"  # type: ignore[operator]
        export_trajectory_html(simple_response, output_path, title="Custom Title")

        content = output_path.read_text()
        assert "Custom Title" in content
