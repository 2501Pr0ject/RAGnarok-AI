"""HHEM-2.1-Open as a judge: the probability that an answer is consistent with its context.

HHEM-2.1-Open (Vectara, ``vectara/hallucination_evaluation_model``,
Apache-2.0) is a ``T5ForTokenClassification`` head fine-tuned from
``google/flan-t5-base`` (Apache-2.0). Given a premise and a hypothesis it
returns P(consistent): 1 when the hypothesis is fully supported by the
premise, 0 when it is not evidenced at all. The context is the premise, the
answer (or each of its sentences) is the hypothesis; the question is not
part of the model's input.

What the published model does, reproduced here without running the
repository's remote code (``trust_remote_code`` is never set):

- the pair is formatted with the training prompt ``HHEM_PROMPT``;
- the foundation tokenizer encodes it (padding, end-of-sequence token);
- the logits of the first token (the leading ``<pad>``) go through a
  softmax; class 1 is ``consistent``.

Both artifacts are pinned to a commit: the HHEM weights (``HHEM_REVISION``)
and the flan-t5-base tokenizer and configuration (``FOUNDATION_REVISION``),
which the remote code would otherwise fetch unpinned at construction time.
Both revisions are recorded in ``JudgeConfig.model_digest``.

Local-first: the weights are looked up in the Hugging Face cache before any
download, and ``HHEMJudge.download(directory)`` writes a self-contained
folder that ``model_path=`` loads with no network access.

Context length: the foundation uses relative positions and has no hard
input limit; HHEM-2.1-Open is documented with an unlimited context length
(its training length is 512 tokens). By default nothing is truncated and
``trace.context_truncated`` is ``False``. With ``max_input_tokens`` the
premise is cut to fit, the hypothesis is kept whole, and
``context_truncated`` is ``True``. ``trace.prompt_tokens`` is always the
number of tokens actually sent.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ragnarok_ai.core.exceptions import ConfigurationError, JudgeDependencyError
from ragnarok_ai.judges.contract import BinaryProbability, CallTrace, JudgeConfig, ScoreThreshold
from ragnarok_ai.judges.sentence_level import DEFAULT_AGGREGATION, SentenceLevel
from ragnarok_ai.text.segmentation import segmenter_id

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ragnarok_ai.evaluators.judge import JudgeResult
    from ragnarok_ai.judges.contract import AggregationPolicy, Unit

HHEM_REPO = "vectara/hallucination_evaluation_model"
HHEM_REVISION = "de9a5e6ba9ba49531a022dba17e35ffeb2a6fad8"
HHEM_FILES = ("config.json", "model.safetensors")
FOUNDATION_REPO = "google/flan-t5-base"
FOUNDATION_REVISION = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
# Tokenizer and configuration only: the HHEM checkpoint holds every weight of the encoder
FOUNDATION_FILES = ("config.json", "special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json")
# The training prompt, as in configuration_hhem_v2.py at HHEM_REVISION
HHEM_PROMPT = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
HHEM_PROMPT_SHA256 = hashlib.sha256(HHEM_PROMPT.encode("utf-8")).hexdigest()
CONSISTENT_CLASS = 1  # id2label: 0 hallucinated, 1 consistent
# The model card reads the probability at 0.5. Not calibrated on any ragnarok-ai label set.
PASS_THRESHOLD = 0.5
TRAINING_LENGTH = 512
DEVICES = ("cpu", "mps", "cuda")
WEIGHTS_FILE = "model.safetensors"
FOUNDATION_SUBDIR = "foundation"
REVISIONS_FILE = "REVISIONS.json"
_TIED_EMBEDDING = "transformer.encoder.embed_tokens.weight"  # the same tensor as transformer.shared.weight


# =============================================================================
# Artifacts: where the weights and the tokenizer come from
# =============================================================================


@dataclass(frozen=True)
class HHEMArtifacts:
    """The two local folders a judge loads from, and what is known of their origin.

    Attributes:
        hhem_dir: Folder holding ``model.safetensors``.
        foundation_dir: Folder holding the flan-t5-base tokenizer and ``config.json``.
        hhem_revision: Pinned commit of the HHEM repository, when known.
        foundation_revision: Pinned commit of the foundation repository, when known.
        source: ``hub`` (Hugging Face cache or download) or ``local`` (``model_path``).
        digest: What goes in ``JudgeConfig.model_digest``: both revisions, or
            the SHA-256 of the weights file when a local folder carries no
            ``REVISIONS.json``.
    """

    hhem_dir: Path
    foundation_dir: Path
    hhem_revision: str | None
    foundation_revision: str | None
    source: str
    digest: str


def _digest(hhem_revision: str | None, foundation_revision: str | None, weights: Path | None) -> str:
    if hhem_revision is None and weights is not None:
        sha = hashlib.sha256()
        with weights.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                sha.update(chunk)
        hhem = f"sha256:{sha.hexdigest()}"
    else:
        hhem = hhem_revision or "unknown"
    return f"{HHEM_REPO}@{hhem};{FOUNDATION_REPO}@{foundation_revision or 'unknown'}"


def _snapshot(repo: str, revision: str, files: Sequence[str], *, offline: bool) -> str:
    """The local snapshot folder of *repo* at *revision*: the cache first, a download only when allowed."""
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise JudgeDependencyError(extra="hhem", module="huggingface_hub", judge="HHEMJudge", alternative=None) from exc
    try:
        return str(snapshot_download(repo, revision=revision, allow_patterns=list(files), local_files_only=True))
    except Exception as exc:  # not in the cache (LocalEntryNotFoundError and friends)
        if offline:
            msg = (
                f"{repo}@{revision[:12]} is not in the local Hugging Face cache and offline=True. "
                "Run HHEMJudge.download(directory) on a connected machine and pass model_path=, "
                "or allow the download"
            )
            raise ConfigurationError(msg) from exc
    return str(snapshot_download(repo, revision=revision, allow_patterns=list(files)))


def resolve_artifacts(model_path: str | Path | None = None, *, offline: bool = False) -> HHEMArtifacts:
    """Locate the weights and the tokenizer.

    Args:
        model_path: A folder written by ``HHEMJudge.download``: ``model.safetensors``,
            a ``foundation/`` subfolder with the tokenizer, and ``REVISIONS.json``.
            Nothing is fetched. Without it, the pinned revisions are looked up
            in the Hugging Face cache, then downloaded unless *offline*.
        offline: Never download; fail when the cache does not hold the pinned revisions.

    Raises:
        ConfigurationError: Incomplete *model_path*, or weights absent while offline.
        JudgeDependencyError: ``huggingface_hub`` is not installed (``[hhem]`` extra).
    """
    if model_path is not None:
        root = Path(model_path)
        weights = root / WEIGHTS_FILE
        foundation = root / FOUNDATION_SUBDIR
        has_tokenizer = (foundation / "tokenizer.json").is_file() or (foundation / "spiece.model").is_file()
        if not weights.is_file() or not has_tokenizer:
            msg = (
                f"HHEM model_path {root} must hold {WEIGHTS_FILE} and a {FOUNDATION_SUBDIR}/ folder with the "
                f"flan-t5-base tokenizer; create it with HHEMJudge.download({str(root)!r})"
            )
            raise ConfigurationError(msg)
        revisions: dict[str, Any] = {}
        if (root / REVISIONS_FILE).is_file():
            revisions = json.loads((root / REVISIONS_FILE).read_text(encoding="utf-8"))
        hhem_revision = revisions.get("hhem_revision")
        foundation_revision = revisions.get("foundation_revision")
        return HHEMArtifacts(
            hhem_dir=root,
            foundation_dir=foundation,
            hhem_revision=hhem_revision,
            foundation_revision=foundation_revision,
            source="local",
            digest=_digest(hhem_revision, foundation_revision, weights),
        )
    hhem_dir = _snapshot(HHEM_REPO, HHEM_REVISION, HHEM_FILES, offline=offline)
    foundation_dir = _snapshot(FOUNDATION_REPO, FOUNDATION_REVISION, FOUNDATION_FILES, offline=offline)
    return HHEMArtifacts(
        hhem_dir=Path(hhem_dir),
        foundation_dir=Path(foundation_dir),
        hhem_revision=HHEM_REVISION,
        foundation_revision=FOUNDATION_REVISION,
        source="hub",
        digest=_digest(HHEM_REVISION, FOUNDATION_REVISION, None),
    )


def download_artifacts(directory: str | Path) -> HHEMArtifacts:
    """Write a self-contained folder for ``model_path=``: weights, tokenizer and ``REVISIONS.json``."""
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise JudgeDependencyError(extra="hhem", module="huggingface_hub", judge="HHEMJudge", alternative=None) from exc
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(HHEM_REPO, revision=HHEM_REVISION, allow_patterns=list(HHEM_FILES), local_dir=str(root))
    snapshot_download(
        FOUNDATION_REPO,
        revision=FOUNDATION_REVISION,
        allow_patterns=list(FOUNDATION_FILES),
        local_dir=str(root / FOUNDATION_SUBDIR),
    )
    revisions = {
        "hhem_repo": HHEM_REPO,
        "hhem_revision": HHEM_REVISION,
        "foundation_repo": FOUNDATION_REPO,
        "foundation_revision": FOUNDATION_REVISION,
    }
    (root / REVISIONS_FILE).write_text(json.dumps(revisions, indent=2) + "\n", encoding="utf-8")
    return resolve_artifacts(root)


# =============================================================================
# Backend, the loaded tokenizer and model
# =============================================================================


class HHEMBackend(Protocol):
    """What the judge needs from a loaded model; ``TransformersBackend`` is the real one."""

    def count_tokens(self, text: str) -> int:
        """Number of tokens of *text* as the model receives it."""
        ...  # pragma: no cover

    def truncate(self, text: str, max_tokens: int) -> str:
        """*text* cut to its first *max_tokens* tokens."""
        ...  # pragma: no cover

    def consistency(self, prompts: Sequence[str]) -> list[float]:
        """P(consistent) for each formatted prompt."""
        ...  # pragma: no cover


def _validate_device(device: str) -> None:
    if device.split(":", 1)[0] not in DEVICES:
        msg = f"HHEMJudge device must be one of {DEVICES} (optionally 'cuda:N'), got {device!r}"
        raise ValueError(msg)


class TransformersBackend:
    """HHEM-2.1-Open loaded with ``transformers`` and ``torch`` on an explicit device, in eval mode."""

    def __init__(self, artifacts: HHEMArtifacts, device: str = "cpu") -> None:
        """Load the tokenizer and the weights from *artifacts* onto *device*.

        Raises:
            JudgeDependencyError: ``torch``, ``transformers`` or ``safetensors`` missing.
            ConfigurationError: Device not available, or weights not matching the architecture.
            ValueError: Unknown device name.
        """
        _validate_device(device)
        try:
            import torch
            from safetensors.torch import load_file
            from transformers import AutoTokenizer, T5Config, T5ForTokenClassification
        except ModuleNotFoundError as exc:
            raise JudgeDependencyError(
                extra="hhem", module=exc.name or "torch", judge="HHEMJudge", alternative=None
            ) from exc
        kind = device.split(":", 1)[0]
        if kind == "mps" and not torch.backends.mps.is_available():
            msg = "HHEMJudge device 'mps' requested but Metal is not available on this machine"
            raise ConfigurationError(msg)
        if kind == "cuda" and not torch.cuda.is_available():
            msg = "HHEMJudge device 'cuda' requested but CUDA is not available on this machine"
            raise ConfigurationError(msg)

        config = T5Config.from_pretrained(str(artifacts.foundation_dir), num_labels=2)
        model = T5ForTokenClassification(config)
        state = load_file(str(artifacts.hhem_dir / WEIGHTS_FILE))
        state = {key.removeprefix("t5."): tensor for key, tensor in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        unexplained = sorted(set(missing) - {_TIED_EMBEDDING})
        if unexpected or unexplained:
            msg = (
                f"HHEM weights do not match T5ForTokenClassification: missing {unexplained}, "
                f"unexpected {sorted(unexpected)}"
            )
            raise ConfigurationError(msg)
        self.model = model.to(device).eval()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(str(artifacts.foundation_dir))
        # 512 is the foundation's training length, not a limit: silence the tokenizer's warning
        warnings = getattr(self.tokenizer, "deprecation_warnings", None)
        if isinstance(warnings, dict):
            warnings["sequence-length-is-longer-than-the-specified-maximum"] = True
        self._torch = torch

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text)["input_ids"])

    def truncate(self, text: str, max_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:max_tokens]
        return str(self.tokenizer.decode(ids, skip_special_tokens=True))

    def consistency(self, prompts: Sequence[str]) -> list[float]:
        inputs = self.tokenizer(list(prompts), return_tensors="pt", padding=True).to(self.device)
        with self._torch.inference_mode():
            logits = self.model(**inputs).logits[:, 0, :]  # the leading <pad> token
            probabilities = self._torch.softmax(logits.float(), dim=-1)[:, CONSISTENT_CLASS]
        return [float(p) for p in probabilities.cpu().tolist()]


# =============================================================================
# Judge
# =============================================================================


class _AnswerLevel:
    """The answer-level judge SentenceLevel applies to each sentence."""

    def __init__(self, owner: HHEMJudge) -> None:
        self._owner = owner

    @property
    def criteria(self) -> frozenset[str]:
        return self._owner.criteria

    def config(self, criterion: str) -> JudgeConfig:
        return self._owner._answer_config(criterion)

    async def judge(self, criterion: str, *, question: str, context: str, answer: str) -> JudgeResult:  # noqa: ARG002
        return self._owner._judge_answer(criterion, context=context, answer=answer)


class HHEMJudge:
    """HHEM-2.1-Open as a ``JudgeProtocol`` judge.

    Criteria ``faithfulness`` and ``hallucination`` give the same judgment:
    the score is P(consistent) of the answer given the context, PASS at
    0.5 or above. The threshold is the model card's reading of the
    probability and is not calibrated on any ragnarok-ai label set
    (``judge_specific`` says so); ``JudgeCalibrator`` can recommend another.

    With the default ``unit="sentence"`` the answer is segmented with the
    shared segmenter, each sentence is the hypothesis of one call, and the
    answer's score is the fraction of PASS sentences with the worst verdict
    (``SentenceLevel``). With ``unit="answer"`` the whole answer is the
    hypothesis of a single call; this is what ``judge-bench`` uses on a
    sentence-level set, where the segmentation was done by the conversion.

    Example:
        >>> judge = HHEMJudge(device="cpu")
        >>> result = await judge.judge("faithfulness", question=q, context=c, answer=a)
        >>> result.score  # P(consistent) aggregated over sentences
    """

    CRITERIA = frozenset({"faithfulness", "hallucination"})
    VERDICT_POLICY = ScoreThreshold(pass_threshold=PASS_THRESHOLD)

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        device: str = "cpu",
        unit: Unit = "sentence",
        max_input_tokens: int | None = None,
        offline: bool = False,
        aggregation: AggregationPolicy = DEFAULT_AGGREGATION,
        lang: str = "en",
        backend: HHEMBackend | None = None,
        model_digest: str | None = None,
    ) -> None:
        """Load HHEM-2.1-Open.

        Args:
            model_path: Folder written by ``HHEMJudge.download``; nothing is
                fetched. Without it, the pinned revisions come from the
                Hugging Face cache, then from a download unless *offline*.
            device: ``cpu``, ``mps`` or ``cuda`` (``cuda:N`` accepted). Recorded in the configuration.
            unit: ``sentence`` (default) or ``answer``.
            max_input_tokens: Cut the context so the whole input fits; ``None`` sends everything.
            offline: Never download.
            aggregation: Sentence aggregation policy (``unit="sentence"``).
            lang: Segmentation language (``unit="sentence"``).
            backend: A loaded ``HHEMBackend``; for tests and custom loaders.
                Then *model_path*, *device* availability and *offline* are not consulted.
            model_digest: Recorded with *backend*; the resolved artifacts' digest otherwise.

        Raises:
            ValueError: Unknown device name, unit or non-positive ``max_input_tokens``.
            JudgeDependencyError: The ``[hhem]`` extra is not installed.
            ConfigurationError: Weights missing offline, incomplete ``model_path``, device unavailable.
        """
        _validate_device(device)
        if unit not in ("answer", "sentence"):
            msg = f"HHEMJudge unit must be 'sentence' or 'answer', got {unit!r}"
            raise ValueError(msg)
        if max_input_tokens is not None and max_input_tokens < 1:
            msg = f"HHEMJudge max_input_tokens must be at least 1, got {max_input_tokens}"
            raise ValueError(msg)
        self.device = device
        self.unit: Unit = unit
        self.max_input_tokens = max_input_tokens
        self.aggregation = aggregation
        self.lang = lang
        self.artifacts: HHEMArtifacts | None = None
        if backend is None:
            self.artifacts = resolve_artifacts(model_path, offline=offline)
            self.model_digest = self.artifacts.digest
            self.backend: HHEMBackend = TransformersBackend(self.artifacts, device)
        else:
            self.backend = backend
            self.model_digest = model_digest or "injected-backend"
        self._sentence_level = SentenceLevel(_AnswerLevel(self), aggregation=aggregation, lang=lang)

    @staticmethod
    def download(directory: str | Path) -> HHEMArtifacts:
        """Download both pinned artifacts into *directory* for later ``model_path=`` use."""
        return download_artifacts(directory)

    @property
    def criteria(self) -> frozenset[str]:
        """``faithfulness`` and ``hallucination``."""
        return self.CRITERIA

    def _check_criterion(self, criterion: str) -> None:
        if criterion not in self.CRITERIA:
            msg = f"Unknown criterion {criterion!r}; expected one of {sorted(self.CRITERIA)}"
            raise ValueError(msg)

    def _answer_config(self, criterion: str) -> JudgeConfig:
        from ragnarok_ai import __version__

        return JudgeConfig(
            judge_class=f"{type(self).__module__}.{type(self).__qualname__}",
            judge_kind="classifier",
            model=HHEM_REPO,
            criterion=criterion,
            verdict_policy=self.VERDICT_POLICY,
            ragnarok_ai_version=str(__version__),
            backend="transformers",
            model_digest=self.model_digest,
            options=(("device", self.device), ("dtype", "float32"), ("max_input_tokens", self.max_input_tokens)),
            prompt_format=None,
            prompt_template_sha256=HHEM_PROMPT_SHA256,
            judge_specific=(
                ("foundation", FOUNDATION_REPO),
                ("question_used", False),
                ("threshold_calibrated", False),
            ),
        )

    def config(self, criterion: str) -> JudgeConfig:
        """The run-level configuration for *criterion*.

        Raises:
            ValueError: If *criterion* is unknown.
        """
        self._check_criterion(criterion)
        base = self._answer_config(criterion)
        if self.unit == "answer":
            return base
        return replace(base, unit="sentence", aggregation=self.aggregation, segmenter=segmenter_id(self.lang))

    def _judge_answer(self, criterion: str, *, context: str, answer: str) -> JudgeResult:
        """One model call: *context* is the premise, *answer* the hypothesis."""
        from ragnarok_ai.evaluators.judge import JudgeResult

        started = time.perf_counter()
        premise = context
        prompt = HHEM_PROMPT.format(text1=premise, text2=answer)
        tokens = self.backend.count_tokens(prompt)
        truncated = False
        if self.max_input_tokens is not None and tokens > self.max_input_tokens:
            fixed = self.backend.count_tokens(HHEM_PROMPT.format(text1="", text2=answer))
            budget = max(0, self.max_input_tokens - fixed)
            premise = self.backend.truncate(premise, budget) if budget else ""
            prompt = HHEM_PROMPT.format(text1=premise, text2=answer)
            tokens = self.backend.count_tokens(prompt)
            truncated = True
        probability = min(1.0, max(0.0, float(self.backend.consistency([prompt])[0])))
        output = BinaryProbability(probability)
        verdict = self.VERDICT_POLICY.verdict(output)
        explanation = (
            f"HHEM-2.1-Open P(consistent) = {probability:.3f}; PASS at >= {PASS_THRESHOLD} "
            "(model card default, not calibrated)"
        )
        if truncated:
            explanation += f"; context truncated to fit {self.max_input_tokens} tokens"
        return JudgeResult(
            verdict=verdict,
            score=probability,
            explanation=explanation,
            criteria=criterion,
            raw_score=None,
            output=output,
            config=self._answer_config(criterion),
            trace=CallTrace(
                prompt_tokens=tokens,
                completion_tokens=None,
                context_truncated=truncated,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            ),
        )

    async def judge(self, criterion: str, *, question: str, context: str, answer: str) -> JudgeResult:
        """Judge *answer* against *context*; *question* is not part of HHEM's input.

        Raises:
            ValueError: If *criterion* is unknown.
            NoUnitsError: With ``unit="sentence"``, if no sentence could be judged.
        """
        self._check_criterion(criterion)
        if self.unit == "answer":
            return self._judge_answer(criterion, context=context, answer=answer)
        result = await self._sentence_level.judge(criterion, question=question, context=context, answer=answer)
        return replace(result, config=self.config(criterion))


__all__ = [
    "DEVICES",
    "FOUNDATION_FILES",
    "FOUNDATION_REPO",
    "FOUNDATION_REVISION",
    "HHEM_FILES",
    "HHEM_PROMPT",
    "HHEM_PROMPT_SHA256",
    "HHEM_REPO",
    "HHEM_REVISION",
    "PASS_THRESHOLD",
    "TRAINING_LENGTH",
    "HHEMArtifacts",
    "HHEMBackend",
    "HHEMJudge",
    "TransformersBackend",
    "download_artifacts",
    "resolve_artifacts",
]
