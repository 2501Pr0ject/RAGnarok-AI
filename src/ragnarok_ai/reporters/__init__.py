"""Reporters module for ragnarok-ai.

This module provides output formatters for evaluation results:
- Console: Terminal output with tables and colors
- JSON: Machine-readable format
- HTML: Interactive visual reports with drill-down
- Markdown: Documentation-friendly (future)
"""

from __future__ import annotations

from ragnarok_ai.reporters.console import ConsoleReporter
from ragnarok_ai.reporters.html import HTMLReporter
from ragnarok_ai.reporters.json import JSONReporter

__all__ = [
    "ConsoleReporter",
    "HTMLReporter",
    "JSONReporter",
]
