"""Medical Abbreviation Normalizer for RAG Evaluation.

Normalizes medical abbreviations in both context and generated responses
before faithfulness/hallucination evaluation. This reduces false positives
where an LLM judge might consider "CHF" and "congestive heart failure"
as contradictory or unsupported claims.

Integration:
    This module is used by evaluators when ``medical_mode=True``.
    No changes to evaluator code are required — the normalizer's
    ``normalize_text`` method returns ``tuple[str, list[str]]``
    matching the expected call signature.

    >>> from ragnarok_ai.evaluators.medical.medical_normalizer import MedicalAbbreviationNormalizer
    >>> normalizer = MedicalAbbreviationNormalizer()
    >>> normalized, expansions = normalizer.normalize_text("Patient has CHF and MI")
    >>> normalized
    'Patient has congestive heart failure and myocardial infarction'
    >>> expansions
    ['CHF → congestive heart failure', 'MI → myocardial infarction']
"""

from __future__ import annotations

import re

from .abbreviations import (
    ABBREVIATIONS,
    AMPERSAND_ABBREVIATIONS,
    BAR_ABBREVIATIONS,
    DEGREE_ABBREVIATIONS,
    DOTTED_ABBREVIATIONS,
    FALSE_POSITIVES,
    MIXEDCASE_ABBREVIATIONS,
    SLASH_ABBREVIATIONS,
    AmbiguousEntry,
)

# ========================== Normalizer ======================================


class MedicalAbbreviationNormalizer:
    """Normalize medical abbreviations for RAG evaluation.

    Designed as a **drop-in preprocessing step** for evaluators.  When both
    the retrieved context and the generated response are normalized to the
    same surface forms, the LLM judge is far less likely to flag a claim as
    unsupported simply because one side used an abbreviation and the other
    used the full term.

    Features:
        - Context-aware disambiguation for ambiguous abbreviations
          (e.g. ``MS`` → multiple sclerosis vs. mitral stenosis).
        - False-positive filtering (``OR``, ``US``, ``IT``, etc.).
        - Explicit-definition detection: ``CHF (Congestive Heart Failure)``
          is left untouched.
        - Extensible via ``custom_abbreviations``.

    Returns:
        ``normalize_text`` returns ``tuple[str, list[str]]`` — matching the
        call signature expected by ``FaithfulnessEvaluator`` and
        ``HallucinationEvaluator``.

    Example:
        >>> normalizer = MedicalAbbreviationNormalizer()
        >>> text, expansions = normalizer.normalize_text("CHF with EF 30%")
        >>> text
        'congestive heart failure with ejection fraction 30%'
        >>> expansions
        ['CHF → congestive heart failure', 'EF → ejection fraction']
    """

    # 2+ chars starting with uppercase letter, can include digits (T2DM, A1C, HBA1C)
    _ABBREV_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,})\b")

    # "ABBREV (Full Form)" — already defined inline, skip
    _EXPLICIT_DEF_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,})\s*\(([^)]+)\)")

    # Dotted abbreviations: matches patterns like q.d, q.d., Q.D., b.i.d, B.I.D.
    _DOTTED_RE = re.compile(r"\b([A-Za-z](?:\.[A-Za-z0-9])+)\.?\b")

    # Slash abbreviations: matches s/p, c/o, w/o, n/v/d, w/ (trailing slash)
    _SLASH_RE = re.compile(r"\b([A-Za-z]/(?:[A-Za-z]/)*[A-Za-z]?\b)|(\b[A-Za-z]/(?=\s|$))")

    # Ampersand abbreviations: matches I&D, D&C, A&Ox3, etc.
    _AMPERSAND_RE = re.compile(r"\b([A-Za-z]+&[A-Za-z0-9]+)\b")

    # Mixed-case shorthand: matches Dx, Tx, SpO2, FiO2, HbA1c, pCO2, etc.
    # Built dynamically from dictionary keys for exact matching.
    _MIXEDCASE_RE: re.Pattern[str] | None = None

    # Degree abbreviations: matches 1°, 2°, 3°, 4°
    _DEGREE_RE = re.compile(r"\b([1-4])°")

    def __init__(
        self,
        *,
        custom_abbreviations: dict[str, str] | None = None,
        context_window: int = 10,
    ) -> None:
        """Initialize the normalizer.

        Args:
            custom_abbreviations: Extra abbreviation → full-form pairs to
                merge into the built-in dictionary (unambiguous only).
            context_window: Number of surrounding words to consider when
                disambiguating ambiguous abbreviations.
        """
        self._context_window = context_window

        self._abbreviations: dict[str, str | list[AmbiguousEntry]] = {
            **ABBREVIATIONS,
        }
        if custom_abbreviations:
            self._abbreviations.update(custom_abbreviations)

        # Build mixed-case regex from dictionary keys (longest first to avoid
        # partial matches, e.g. "HbA1c" before "Hb")
        sorted_keys = sorted(MIXEDCASE_ABBREVIATIONS.keys(), key=len, reverse=True)
        escaped = [re.escape(k) for k in sorted_keys]
        self._mixedcase_re = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(escaped) + r")(?![A-Za-z0-9])")

    # ── Public API (tuple return for evaluator compatibility) ───────────

    def normalize_text(self, text: str) -> tuple[str, list[str]]:
        """Expand medical abbreviations in *text*.

        Processing order (each pass runs on the output of the previous):
          1. Bar-notation  (c̄ → with)
          2. Degree symbols (2° → secondary)
          3. Slash forms    (s/p → status post)
          4. Ampersand      (I&D → incision and drainage)
          5. Dotted forms   (q.d. → once a day)
          6. Mixed-case     (SpO2 → oxygen saturation, Dx → diagnosis)
          7. Standard UPPER (CHF → congestive heart failure)

        Args:
            text: Clinical / medical text that may contain abbreviations.

        Returns:
            A ``(normalized_text, expansions)`` tuple where *expansions*
            is a list of strings like ``"CHF → congestive heart failure"``.
        """
        explicit_defs = self._extract_explicit_defs(text)
        normalized = text
        expansions: list[str] = []
        seen: set[str] = set()
        full_form: str | None

        # ── Pass 1: Bar-notation (c̄, s̄, p̄, ā) ──────────────────────────
        for bar_char, full_form in BAR_ABBREVIATIONS.items():
            if bar_char in normalized:
                normalized = normalized.replace(bar_char, full_form)
                expansions.append(f"{bar_char} → {full_form}")

        # ── Pass 2: Degree abbreviations (1°, 2°, 3°) ───────────────────
        def _replace_degree(m: re.Match[str]) -> str:
            key = f"{m.group(1)}°"
            result = DEGREE_ABBREVIATIONS.get(key, m.group(0))
            return result

        degree_matches = list(self._DEGREE_RE.finditer(normalized))
        if degree_matches:
            for m in degree_matches:
                key = f"{m.group(1)}°"
                full = DEGREE_ABBREVIATIONS.get(key)
                if full and key not in seen:
                    seen.add(key)
                    expansions.append(f"{key} → {full}")
            normalized = self._DEGREE_RE.sub(_replace_degree, normalized)

        # ── Pass 3: Slash abbreviations (s/p, c/o, w/, w/o, n/v/d) ──────
        # Sort longest-first to match "n/v/d" before "n/v"
        for slash_abbrev in sorted(SLASH_ABBREVIATIONS, key=len, reverse=True):
            # Case-insensitive search
            pattern = re.compile(
                rf"(?<![A-Za-z]){re.escape(slash_abbrev)}(?![A-Za-z])",
                re.IGNORECASE,
            )
            if pattern.search(normalized) and slash_abbrev.lower() not in seen:
                full_form = SLASH_ABBREVIATIONS[slash_abbrev]
                normalized = pattern.sub(full_form, normalized)
                seen.add(slash_abbrev.lower())
                expansions.append(f"{slash_abbrev} → {full_form}")

        # ── Pass 4: Ampersand abbreviations (I&D, D&C, T&S) ─────────────
        for m in self._AMPERSAND_RE.finditer(normalized):
            raw = m.group(1)
            key = raw.upper()
            if key in seen:
                continue
            full_form = AMPERSAND_ABBREVIATIONS.get(raw) or AMPERSAND_ABBREVIATIONS.get(key)
            if full_form is None:
                continue
            seen.add(key)
            pattern = re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(raw)}(?![A-Za-z0-9])",
                re.IGNORECASE,
            )
            normalized = pattern.sub(full_form, normalized)
            expansions.append(f"{raw} → {full_form}")

        # ── Pass 5: Dotted abbreviations (q.d., b.i.d., p.r.n.) ─────────
        for match in self._DOTTED_RE.finditer(normalized):
            raw = match.group(0)
            canonical = match.group(1).lower().rstrip(".")

            if canonical in seen:
                continue

            full_form = DOTTED_ABBREVIATIONS.get(canonical)
            if full_form is None:
                continue

            seen.add(canonical)
            escaped = re.escape(canonical)
            pattern_str = rf"(?<!\w){escaped}\.?(?!\w)"
            normalized = re.sub(pattern_str, full_form, normalized, flags=re.IGNORECASE)
            expansions.append(f"{raw.rstrip('.')} → {full_form}")

        # ── Pass 6: Mixed-case shorthand (Dx, Tx, SpO2, FiO2, HbA1c) ───
        for m in self._mixedcase_re.finditer(normalized):
            raw = m.group(1)
            if raw in seen:
                continue
            full_form = MIXEDCASE_ABBREVIATIONS.get(raw)
            if full_form is None or full_form == raw:
                continue  # Skip identity mappings like pH → pH
            seen.add(raw)
            normalized = normalized.replace(raw, full_form)
            expansions.append(f"{raw} → {full_form}")

        # ── Pass 7: Standard uppercase abbreviations (CHF, MI, etc.) ─────
        for abbrev in self._ABBREV_RE.findall(normalized):
            if abbrev in seen or abbrev in explicit_defs:
                continue
            seen.add(abbrev)

            # Skip common English false positives
            if abbrev in FALSE_POSITIVES and abbrev not in self._abbreviations:
                continue
            if abbrev in FALSE_POSITIVES:
                continue

            full_form = self._resolve(abbrev, self._get_context(normalized, abbrev))
            if full_form is None:
                continue

            normalized = re.sub(rf"\b{re.escape(abbrev)}\b", full_form, normalized)
            expansions.append(f"{abbrev} → {full_form}")

        return normalized, expansions

    # ── Resolution ─────────────────────────────────────────────────────

    def _resolve(self, abbrev: str, context: str) -> str | None:
        """Return the best expansion for *abbrev* given surrounding *context*."""
        entry = self._abbreviations.get(abbrev)
        if entry is None:
            return None

        # Unambiguous
        if isinstance(entry, str):
            return entry

        # Ambiguous — score each candidate by context-keyword overlap
        context_lower = context.lower()
        best: AmbiguousEntry | None = None
        best_score = -1

        for candidate in entry:
            hits = sum(1 for kw in candidate.context_keywords if kw in context_lower)
            score = hits * 100 + candidate.priority
            if score > best_score:
                best_score = score
                best = candidate

        return best.full_form if best else None

    # ── Helpers ─────────────────────────────────────────────────────────

    def _get_context(self, text: str, abbrev: str) -> str:
        """Return a window of words around the first occurrence of *abbrev*."""
        match = re.search(rf"\b{re.escape(abbrev)}\b", text)
        if not match:
            return ""
        words = text.split()
        word_idx = len(text[: match.start()].split()) - 1
        start = max(0, word_idx - self._context_window)
        end = min(len(words), word_idx + self._context_window + 1)
        return " ".join(words[start:end])

    @staticmethod
    def _extract_explicit_defs(text: str) -> dict[str, str]:
        """Find abbreviations already defined inline (e.g. ``CHF (Congestive Heart Failure)``)."""
        return {m.group(1): m.group(2).strip() for m in MedicalAbbreviationNormalizer._EXPLICIT_DEF_RE.finditer(text)}
