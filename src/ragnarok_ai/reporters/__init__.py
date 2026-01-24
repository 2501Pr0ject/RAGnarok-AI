"""Reporters module for ragnarok-ai.

This module provides output formatters for evaluation results:
- Console: Terminal output with tables and colors
- JSON: Machine-readable format (future)
- HTML: Visual reports (future)
- Markdown: Documentation-friendly (future)
"""

from __future__ import annotations

from ragnarok_ai.reporters.console import ConsoleReporter

__all__ = [
    "ConsoleReporter",
]
