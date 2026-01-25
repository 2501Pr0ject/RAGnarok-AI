"""Golden set support for ragnarok-ai.

This module provides support for human-validated, versioned question sets
(golden sets) for production RAG evaluation.
"""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ragnarok_ai.core.types import Query, TestSet


class GoldenQuestion(BaseModel):
    """A human-validated question with verified answer.

    Attributes:
        id: Unique identifier for the question.
        question: The question text.
        answer: The verified correct answer.
        tags: List of tags for categorization.
        source_docs: List of source document IDs.
        validated_by: Who validated this question.
        validated_at: When the question was validated.
        difficulty: Difficulty level (easy, medium, hard).
        priority: Priority level (low, medium, high, critical).
        metadata: Additional metadata.
    """

    id: str = Field(..., description="Unique question identifier")
    question: str = Field(..., description="The question text")
    answer: str = Field(..., description="The verified correct answer")
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    source_docs: list[str] = Field(default_factory=list, description="Source document IDs")
    validated_by: str | None = Field(default=None, description="Validator identity")
    validated_at: str | None = Field(default=None, description="Validation timestamp")
    difficulty: str | None = Field(default=None, description="Difficulty level")
    priority: str | None = Field(default=None, description="Priority level")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class GoldenSetDiff(BaseModel):
    """Represents differences between two golden set versions.

    Attributes:
        old_version: Version of the old set.
        new_version: Version of the new set.
        added: Questions added in new version.
        removed: Questions removed from old version.
        modified: Questions that changed.
        unchanged: Questions that stayed the same.
    """

    old_version: str = Field(..., description="Old version")
    new_version: str = Field(..., description="New version")
    added: list[str] = Field(default_factory=list, description="Added question IDs")
    removed: list[str] = Field(default_factory=list, description="Removed question IDs")
    modified: list[str] = Field(default_factory=list, description="Modified question IDs")
    unchanged: list[str] = Field(default_factory=list, description="Unchanged question IDs")

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return bool(self.added or self.removed or self.modified)

    @property
    def summary(self) -> str:
        """Get a summary of changes."""
        parts = []
        if self.added:
            parts.append(f"+{len(self.added)} added")
        if self.removed:
            parts.append(f"-{len(self.removed)} removed")
        if self.modified:
            parts.append(f"~{len(self.modified)} modified")
        if self.unchanged:
            parts.append(f"={len(self.unchanged)} unchanged")
        return ", ".join(parts) if parts else "no changes"


class GoldenSet(BaseModel):
    """A versioned collection of human-validated questions.

    Attributes:
        version: Version string for this golden set.
        name: Name of the golden set.
        description: Description of the golden set.
        questions: List of golden questions.
        metadata: Additional metadata.

    Example:
        >>> golden = GoldenSet.load("golden_set.yaml")
        >>> critical = golden.filter(tags=["critical"])
        >>> results = await evaluate(rag, critical.to_testset())
    """

    version: str = Field(default="1.0.0", description="Version string")
    name: str | None = Field(default=None, description="Golden set name")
    description: str | None = Field(default=None, description="Description")
    questions: list[GoldenQuestion] = Field(default_factory=list, description="Golden questions")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def __len__(self) -> int:
        """Return number of questions."""
        return len(self.questions)

    def __iter__(self) -> Any:
        """Iterate over questions."""
        return iter(self.questions)

    def get_by_id(self, question_id: str) -> GoldenQuestion | None:
        """Get a question by its ID.

        Args:
            question_id: The question ID to find.

        Returns:
            The question if found, None otherwise.
        """
        for q in self.questions:
            if q.id == question_id:
                return q
        return None

    def filter(
        self,
        tags: list[str] | None = None,
        difficulty: str | None = None,
        priority: str | None = None,
        ids: list[str] | None = None,
    ) -> GoldenSet:
        """Filter questions by criteria.

        Args:
            tags: Only include questions with any of these tags.
            difficulty: Only include questions with this difficulty.
            priority: Only include questions with this priority.
            ids: Only include questions with these IDs.

        Returns:
            New GoldenSet with filtered questions.
        """
        filtered = self.questions

        if tags:
            filtered = [q for q in filtered if any(t in q.tags for t in tags)]

        if difficulty:
            filtered = [q for q in filtered if q.difficulty == difficulty]

        if priority:
            filtered = [q for q in filtered if q.priority == priority]

        if ids:
            filtered = [q for q in filtered if q.id in ids]

        return GoldenSet(
            version=self.version,
            name=self.name,
            description=self.description,
            questions=filtered,
            metadata=self.metadata,
        )

    def get_tags(self) -> set[str]:
        """Get all unique tags in the golden set.

        Returns:
            Set of all tags.
        """
        tags: set[str] = set()
        for q in self.questions:
            tags.update(q.tags)
        return tags

    def get_difficulties(self) -> set[str]:
        """Get all unique difficulty levels.

        Returns:
            Set of difficulty levels.
        """
        return {q.difficulty for q in self.questions if q.difficulty}

    def get_priorities(self) -> set[str]:
        """Get all unique priority levels.

        Returns:
            Set of priority levels.
        """
        return {q.priority for q in self.questions if q.priority}

    def to_testset(self) -> TestSet:
        """Convert to a TestSet for evaluation.

        Returns:
            TestSet with queries from golden questions.
        """
        queries = [
            Query(
                text=q.question,
                ground_truth_docs=q.source_docs,
                expected_answer=q.answer,
                metadata={
                    "golden_id": q.id,
                    "tags": q.tags,
                    "difficulty": q.difficulty,
                    "priority": q.priority,
                    "validated_by": q.validated_by,
                    "validated_at": q.validated_at,
                },
            )
            for q in self.questions
        ]

        return TestSet(
            queries=queries,
            name=self.name or f"golden_set_v{self.version}",
            description=self.description or f"Golden set version {self.version}",
            metadata={
                "golden_version": self.version,
                **self.metadata,
            },
        )

    def diff(self, other: GoldenSet) -> GoldenSetDiff:
        """Compare this golden set with another.

        Args:
            other: The other golden set to compare with.

        Returns:
            GoldenSetDiff with the differences.
        """
        self_ids = {q.id for q in self.questions}
        other_ids = {q.id for q in other.questions}

        added = list(other_ids - self_ids)
        removed = list(self_ids - other_ids)

        # Check for modifications in common questions
        common_ids = self_ids & other_ids
        modified = []
        unchanged = []

        for qid in common_ids:
            self_q = self.get_by_id(qid)
            other_q = other.get_by_id(qid)

            if self_q and other_q:
                # Compare question and answer (main content)
                if self_q.question != other_q.question or self_q.answer != other_q.answer:
                    modified.append(qid)
                else:
                    unchanged.append(qid)

        return GoldenSetDiff(
            old_version=self.version,
            new_version=other.version,
            added=sorted(added),
            removed=sorted(removed),
            modified=sorted(modified),
            unchanged=sorted(unchanged),
        )

    def save(self, path: Path | str, format: str | None = None) -> None:
        """Save golden set to file.

        Args:
            path: Output file path.
            format: Format to use (json, yaml, csv). Auto-detected from extension if not provided.
        """
        path = Path(path)
        fmt = format or self._detect_format(path)

        if fmt == "yaml":
            self._save_yaml(path)
        elif fmt == "csv":
            self._save_csv(path)
        else:
            self._save_json(path)

    @classmethod
    def load(cls, path: Path | str, format: str | None = None) -> GoldenSet:
        """Load golden set from file.

        Args:
            path: Input file path.
            format: Format to use (json, yaml, csv). Auto-detected from extension if not provided.

        Returns:
            Loaded GoldenSet.
        """
        path = Path(path)
        fmt = format or cls._detect_format(path)

        if fmt == "yaml":
            return cls._load_yaml(path)
        elif fmt == "csv":
            return cls._load_csv(path)
        else:
            return cls._load_json(path)

    @staticmethod
    def _detect_format(path: Path) -> str:
        """Detect format from file extension."""
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            return "yaml"
        elif suffix == ".csv":
            return "csv"
        return "json"

    def _save_json(self, path: Path) -> None:
        """Save as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "questions": [q.model_dump() for q in self.questions],
        }
        path.write_text(json.dumps(data, indent=2))

    def _save_yaml(self, path: Path) -> None:
        """Save as YAML."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "questions": [q.model_dump() for q in self.questions],
        }
        path.write_text(yaml.safe_dump(data, default_flow_style=False, allow_unicode=True))

    def _save_csv(self, path: Path) -> None:
        """Save as CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)
        output = StringIO()
        fieldnames = [
            "id",
            "question",
            "answer",
            "tags",
            "source_docs",
            "validated_by",
            "validated_at",
            "difficulty",
            "priority",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for q in self.questions:
            writer.writerow(
                {
                    "id": q.id,
                    "question": q.question,
                    "answer": q.answer,
                    "tags": ";".join(q.tags),
                    "source_docs": ";".join(q.source_docs),
                    "validated_by": q.validated_by or "",
                    "validated_at": q.validated_at or "",
                    "difficulty": q.difficulty or "",
                    "priority": q.priority or "",
                }
            )

        path.write_text(output.getvalue())

    @classmethod
    def _load_json(cls, path: Path) -> GoldenSet:
        """Load from JSON."""
        data = json.loads(path.read_text())
        return cls._from_dict(data)

    @classmethod
    def _load_yaml(cls, path: Path) -> GoldenSet:
        """Load from YAML."""
        data = yaml.safe_load(path.read_text())
        return cls._from_dict(data)

    @classmethod
    def _load_csv(cls, path: Path) -> GoldenSet:
        """Load from CSV."""
        questions = []
        reader = csv.DictReader(StringIO(path.read_text()))

        for row in reader:
            questions.append(
                GoldenQuestion(
                    id=row["id"],
                    question=row["question"],
                    answer=row["answer"],
                    tags=row.get("tags", "").split(";") if row.get("tags") else [],
                    source_docs=row.get("source_docs", "").split(";") if row.get("source_docs") else [],
                    validated_by=row.get("validated_by") or None,
                    validated_at=row.get("validated_at") or None,
                    difficulty=row.get("difficulty") or None,
                    priority=row.get("priority") or None,
                )
            )

        return cls(version="1.0.0", questions=questions)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> GoldenSet:
        """Create from dictionary."""
        questions = [GoldenQuestion(**q) for q in data.get("questions", [])]
        return cls(
            version=data.get("version", "1.0.0"),
            name=data.get("name"),
            description=data.get("description"),
            questions=questions,
            metadata=data.get("metadata", {}),
        )
