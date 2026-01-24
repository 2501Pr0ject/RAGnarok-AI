"""Main CLI entry point for ragnarok-ai.

This module defines the Typer application and all CLI commands.
"""

from __future__ import annotations

from typing import Annotated

import typer

from ragnarok_ai import __version__

# Create the main Typer app
app = typer.Typer(
    name="ragnarok",
    help="ragnarok-ai: Local-first RAG evaluation framework for LLM applications.",
    add_completion=False,
    no_args_is_help=True,
)

# Global state for options
state: dict[str, bool] = {
    "json": False,
    "no_color": False,
}


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"ragnarok-ai v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[  # noqa: ARG001
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output results in JSON format.",
        ),
    ] = False,
    no_color: Annotated[
        bool,
        typer.Option(
            "--no-color",
            help="Disable colored output.",
        ),
    ] = False,
) -> None:
    """ragnarok-ai: Local-first RAG evaluation framework.

    Evaluate, benchmark, and monitor your RAG pipelines — 100% locally.
    """
    state["json"] = json_output
    state["no_color"] = no_color


@app.command()
def version() -> None:
    """Show the current version."""
    typer.echo(f"ragnarok-ai v{__version__}")


@app.command()
def evaluate(
    config: Annotated[  # noqa: ARG001
        str | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration file.",
        ),
    ] = None,
    output: Annotated[  # noqa: ARG001
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path for results.",
        ),
    ] = None,
    fail_under: Annotated[  # noqa: ARG001
        float | None,
        typer.Option(
            "--fail-under",
            help="Fail if average score is below this threshold.",
        ),
    ] = None,
) -> None:
    """Evaluate a RAG pipeline against a test set.

    Coming soon: This command will evaluate your RAG pipeline using
    configurable metrics and test sets.

    Example:
        ragnarok evaluate --config ragnarok.yaml --output results.json
    """
    if state["json"]:
        typer.echo('{"status": "coming_soon", "message": "Evaluate command not yet implemented"}')
    else:
        typer.echo()
        typer.echo("  [Coming Soon]")
        typer.echo()
        typer.echo("  The evaluate command will support:")
        typer.echo("    - RAG pipeline evaluation with custom metrics")
        typer.echo("    - Configurable test sets from YAML/JSON")
        typer.echo("    - Multiple output formats (console, JSON, HTML)")
        typer.echo("    - CI/CD integration with --fail-under threshold")
        typer.echo()
        typer.echo("  Stay tuned for the next release!")
        typer.echo()


@app.command()
def generate(
    docs: Annotated[  # noqa: ARG001
        str | None,
        typer.Option(
            "--docs",
            "-d",
            help="Path to knowledge base documents.",
        ),
    ] = None,
    num_questions: Annotated[  # noqa: ARG001
        int,
        typer.Option(
            "--num",
            "-n",
            help="Number of questions to generate.",
        ),
    ] = 50,
    output: Annotated[  # noqa: ARG001
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path for generated test set.",
        ),
    ] = None,
) -> None:
    """Generate a synthetic test set from a knowledge base.

    Coming soon: This command will generate diverse test questions
    from your knowledge base documents.

    Example:
        ragnarok generate --docs ./knowledge_base/ --num 100 --output testset.json
    """
    if state["json"]:
        typer.echo('{"status": "coming_soon", "message": "Generate command not yet implemented"}')
    else:
        typer.echo()
        typer.echo("  [Coming Soon]")
        typer.echo()
        typer.echo("  The generate command will support:")
        typer.echo("    - Synthetic question generation from documents")
        typer.echo("    - Multiple question types (simple, multi-hop, adversarial)")
        typer.echo("    - Checkpointing for long-running generation")
        typer.echo("    - Local LLM support via Ollama")
        typer.echo()
        typer.echo("  Stay tuned for the next release!")
        typer.echo()


@app.command()
def benchmark(
    configs: Annotated[  # noqa: ARG001
        str | None,
        typer.Option(
            "--configs",
            "-c",
            help="Path to benchmark configurations.",
        ),
    ] = None,
    output: Annotated[  # noqa: ARG001
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path for benchmark results.",
        ),
    ] = None,
) -> None:
    """Benchmark multiple RAG configurations.

    Coming soon: This command will compare different RAG configurations
    side-by-side.

    Example:
        ragnarok benchmark --configs benchmark.yaml --output comparison.html
    """
    if state["json"]:
        typer.echo('{"status": "coming_soon", "message": "Benchmark command not yet implemented"}')
    else:
        typer.echo()
        typer.echo("  [Coming Soon]")
        typer.echo()
        typer.echo("  The benchmark command will support:")
        typer.echo("    - Side-by-side configuration comparison")
        typer.echo("    - Multiple embedding models and chunk sizes")
        typer.echo("    - Visual comparison reports")
        typer.echo("    - Statistical significance testing")
        typer.echo()
        typer.echo("  Stay tuned for the next release!")
        typer.echo()


if __name__ == "__main__":
    app()
