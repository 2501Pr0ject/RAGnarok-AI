"""Load documents from RAGnarok-Forge bundles.

This module provides functions to load documents from bundles produced
by RAGnarok-Forge without importing from the forge package (no cross-imports).

Bundle structure:
    bundle/
    ├── manifest.json       # Metadata with _schema_version or schema_version
    ├── documents.jsonl     # Documents with doc_id or id
    ├── chunks.jsonl        # Optional chunked content
    └── errors.jsonl        # Optional processing errors

Tolerant read policy:
    - Accepts both `doc_id` (Forge convention) and `id` (fallback)
    - Accepts both `content` (Forge) and `text` (fallback) for text
    - Accepts both `_schema_version` and `schema_version` in manifest
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragnarok_ai.core.types import Document


class ForgeLoadError(Exception):
    """Error loading a Forge bundle."""


def load_forge_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Load the manifest from a Forge bundle.

    Args:
        bundle_path: Path to the bundle directory.

    Returns:
        The manifest dictionary with normalized schema version.

    Raises:
        ForgeLoadError: If the bundle or manifest is invalid.

    Example:
        >>> manifest = load_forge_bundle("./my-bundle/")
        >>> print(manifest["document_count"])
        42
    """
    path = Path(bundle_path)

    if not path.exists():
        raise ForgeLoadError(f"Bundle not found: {path}")

    if not path.is_dir():
        raise ForgeLoadError(f"Bundle path is not a directory: {path}")

    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise ForgeLoadError(f"manifest.json not found in bundle: {path}")

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ForgeLoadError(f"Invalid manifest.json: {e}") from e

    if not isinstance(raw, dict):
        raise ForgeLoadError("manifest.json must be a JSON object")

    manifest: dict[str, Any] = raw

    # Tolerant read: accept both _schema_version and schema_version
    schema_version = manifest.get("_schema_version") or manifest.get("schema_version")
    if schema_version:
        # Normalize to _schema_version (Forge convention)
        manifest["_schema_version"] = schema_version

    return manifest


def load_forge_documents(bundle_path: str | Path) -> list[Document]:
    """Load documents from a Forge bundle.

    Reads the documents.jsonl file and converts each entry to a Document.
    Uses tolerant reading to accept both Forge and alternative field names.

    Args:
        bundle_path: Path to the bundle directory.

    Returns:
        List of Document objects loaded from the bundle.

    Raises:
        ForgeLoadError: If the bundle or documents file is invalid.

    Example:
        >>> docs = load_forge_documents("./my-bundle/")
        >>> for doc in docs:
        ...     print(f"{doc.id}: {doc.content[:50]}...")
    """
    path = Path(bundle_path)

    if not path.exists():
        raise ForgeLoadError(f"Bundle not found: {path}")

    if not path.is_dir():
        raise ForgeLoadError(f"Bundle path is not a directory: {path}")

    documents_path = path / "documents.jsonl"
    if not documents_path.exists():
        raise ForgeLoadError(f"documents.jsonl not found in bundle: {path}")

    documents: list[Document] = []

    try:
        with documents_path.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ForgeLoadError(f"Invalid JSON at line {line_num}: {e}") from e

                doc = _parse_document(data, line_num)
                documents.append(doc)
    except OSError as e:
        raise ForgeLoadError(f"Error reading documents.jsonl: {e}") from e

    return documents


def _parse_document(data: dict[str, Any], line_num: int) -> Document:
    """Parse a document from Forge bundle JSON.

    Tolerant read policy:
        - ID: doc_id (Forge) > id (fallback)
        - Content: content (Forge) > text (fallback)
        - Metadata: merge title, format, source_uri into metadata

    Args:
        data: Raw JSON dictionary from documents.jsonl.
        line_num: Line number for error reporting.

    Returns:
        Document object.

    Raises:
        ForgeLoadError: If required fields are missing.
    """
    # Tolerant read for ID: doc_id (Forge convention) or id (fallback)
    doc_id = data.get("doc_id") or data.get("id")
    if not doc_id:
        raise ForgeLoadError(f"Document at line {line_num} has no doc_id or id")

    # Tolerant read for content: content (Forge) or text (fallback)
    content = data.get("content") or data.get("text")
    if content is None:
        raise ForgeLoadError(f"Document at line {line_num} has no content or text")

    # Build metadata from Forge fields
    metadata: dict[str, Any] = {}

    # Copy original metadata if present
    if "metadata" in data and isinstance(data["metadata"], dict):
        metadata.update(data["metadata"])

    # Add Forge-specific fields to metadata
    for field in ("title", "format", "source_uri", "source_hash"):
        if field in data and data[field] is not None:
            metadata[field] = data[field]

    return Document(
        id=str(doc_id),
        content=str(content),
        metadata=metadata,
    )
