"""JSON parsing utilities for test set generation.

This module provides utilities for parsing JSON from LLM responses,
which may contain extra text around the JSON content.
"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_array(text: str) -> list[str]:
    """Parse a JSON array from LLM response.

    Handles cases where the JSON array is embedded in surrounding text.

    Args:
        text: The LLM response text.

    Returns:
        Parsed list of strings, or empty list if parsing fails.

    Example:
        >>> parse_json_array('["Q1", "Q2"]')
        ['Q1', 'Q2']
        >>> parse_json_array('Here are questions:\\n["Q1"]\\nDone.')
        ['Q1']
    """
    text = text.strip()

    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from LLM response.

    Handles cases where the JSON object is embedded in surrounding text.

    Args:
        text: The LLM response text.

    Returns:
        Parsed dictionary, or empty dict if parsing fails.

    Example:
        >>> parse_json_object('{"valid": true}')
        {'valid': True}
        >>> parse_json_object('Result: {"valid": false}')
        {'valid': False}
    """
    text = text.strip()

    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return {}
