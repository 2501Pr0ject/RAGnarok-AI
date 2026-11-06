"""MiniCheck as a judge: the probability that a claim is supported by a document.

MiniCheck (Tang, Laban and Durrett, 2024) checks one claim against one
document. The checkpoints are small fine-tuned encoders published on
Hugging Face under the MIT licence: ``MiniCheck-Flan-T5-Large`` (default),
``MiniCheck-RoBERTa-Large`` and ``MiniCheck-DeBERTa-v3-Large``.
``Bespoke-MiniCheck-7B`` is not offered: its licence is CC BY-NC 4.0 and
the official code serves it through vLLM only.

The official code is https://github.com/Liyan06/MiniCheck (Apache-2.0),
``minicheck/inference.py`` at commit ``OFFICIAL_COMMIT``. This module
reproduces its path without depending on that package:

- the document is split into sentences with nltk's ``sent_tokenize``
  block by block (``sent_tokenize_with_newlines``), then into chunks of at
  most ``chunk_size`` words (Flan-T5) or tokens (RoBERTa, DeBERTa), a
  sentence never being cut (``official_chunks``);
- each chunk and the claim are joined by the tokenizer's end-of-sequence
  token, prefixed with ``predict: `` for Flan-T5, tokenized with truncation
  at ``max_model_len`` and padding, in batches of 16;
- Flan-T5: one decoder step, logits of the tokens 3 (unsupported) and 209
  (supported), softmax; the classifiers: softmax of the two logits;
- the document's probability is the maximum over its chunks.

The official label is ``1 if prob > 0.5``; the verdict policy here is
``ScoreThreshold(0.5)``, PASS at ``>= 0.5``, which differs only at exactly
0.5. The question is not part of the model's input: the context is the
document, the answer (or each of its sentences) the claim.

Pinned revisions, Hugging Face cache before any download,
``MiniCheckJudge.download(folder)`` and ``model_path=`` for a machine with
no network, explicit device and eval mode, as for ``HHEMJudge``. nltk's
``punkt_tab`` sentence model is the one piece that cannot be pinned by
revision; ``download`` stores it in the same folder, and the configuration
records nltk's version and the SHA-256 of the ``punkt_tab`` files actually
loaded (``judge_specific["sentence_splitter"]``), so two runs that split
sentences differently are never comparable.

``training_length`` in ``judge_specific`` is, for MiniCheck, the official
code's truncation length (``max_model_len``: 2048 tokens for Flan-T5 and
DeBERTa, 512 for RoBERTa), not a training length; ``training_length_kind``
says so and the benchmark report shows it as such. ``trace.prompt_tokens``
is the number of tokens sent over all chunks; ``trace.context_truncated``
says whether the tokenizer cut at least one chunk at ``max_model_len``, as
the official code does silently.

Weights: at the pinned revisions the checkpoints ship ``pytorch_model.bin``
(a pickle) only. The loader prefers ``model.safetensors`` when the folder
holds one and otherwise passes ``weights_only=True`` explicitly, so a
pickle carrying anything but tensors is refused; the extra requires
``torch>=2.6``. The Hub's automatic safetensors conversions live on pull
request refs, not on the pinned revision, and are not used.

Fidelity to the official code is measured on CPU in float32. On ``mps`` or
``cuda`` small numeric differences are possible; the device is part of the
configuration, so a CPU run and an MPS run are never comparable.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from ragnarok_ai.core.exceptions import ConfigurationError, JudgeDependencyError
from ragnarok_ai.judges.contract import BinaryProbability, CallTrace, JudgeConfig, ScoreThreshold
from ragnarok_ai.judges.hhem import DEVICES, _validate_device
from ragnarok_ai.judges.sentence_level import DEFAULT_AGGREGATION, SentenceLevel
from ragnarok_ai.text.segmentation import segmenter_id

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ragnarok_ai.evaluators.judge import JudgeResult
    from ragnarok_ai.judges.contract import AggregationPolicy, Unit

OFFICIAL_CODE = "https://github.com/Liyan06/MiniCheck"
OFFICIAL_COMMIT = "b58b9fa69acbd1015ec970fa65dd752413a053d2"  # minicheck 0.1.0, 2025-08-27
OFFICIAL_BATCH_SIZE = 16
FLAN_PREFIX = "predict: "
# Official comment: "3 for no support and 209 for support" (Flan-T5 vocabulary)
FLAN_LABEL_TOKEN_IDS = (3, 209)
PASS_THRESHOLD = 0.5
NLTK_SUBDIR = "nltk_data"
PUNKT_LANGUAGE = "english"  # what nltk.sent_tokenize loads by default
TRAINING_LENGTH_KIND = "official truncation length (max_model_len)"
REVISIONS_FILE = "REVISIONS.json"
TOKENIZER_FILES = ("added_tokens.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json")
# Preferred when a revision ships it; none of the pinned revisions does
SAFETENSORS_FILE = "model.safetensors"
PICKLE_FILE = "pytorch_model.bin"


@dataclass(frozen=True)
class MiniCheckCheckpoint:
    """One published checkpoint and the official defaults that go with it.

    Attributes:
        name: Short name, as in the official code (``flan-t5-large``, ...).
        repo: Hugging Face repository.
        revision: Pinned commit.
        licence: Licence of the weights, as declared on the model card.
        architecture: ``seq2seq`` (Flan-T5) or ``classifier`` (RoBERTa, DeBERTa).
        max_model_len: Official truncation length of one chunk plus claim, also
            declared as the judge's ``training_length``.
        chunk_size: Official chunk size.
        chunk_unit: ``words`` (Flan-T5) or ``tokens`` (classifiers).
        files: Files needed from the repository.
    """

    name: str
    repo: str
    revision: str
    licence: str
    architecture: Literal["seq2seq", "classifier"]
    max_model_len: int
    chunk_size: int
    chunk_unit: Literal["words", "tokens"]
    files: tuple[str, ...]


CHECKPOINTS: dict[str, MiniCheckCheckpoint] = {
    "flan-t5-large": MiniCheckCheckpoint(
        name="flan-t5-large",
        repo="lytang/MiniCheck-Flan-T5-Large",
        revision="96eafd01cee2d16cf81aaa2fb226b14f422a37b3",
        licence="MIT",
        architecture="seq2seq",
        max_model_len=2048,
        chunk_size=500,
        chunk_unit="words",
        files=(
            "config.json",
            "generation_config.json",
            PICKLE_FILE,
            SAFETENSORS_FILE,
            "spiece.model",
            *TOKENIZER_FILES,
        ),
    ),
    "roberta-large": MiniCheckCheckpoint(
        name="roberta-large",
        repo="lytang/MiniCheck-RoBERTa-Large",
        revision="74c8919647e61ed0f71bc177d94f10930f090068",
        licence="MIT",
        architecture="classifier",
        max_model_len=512,
        chunk_size=400,
        chunk_unit="tokens",
        files=("config.json", PICKLE_FILE, SAFETENSORS_FILE, "merges.txt", "vocab.json", *TOKENIZER_FILES),
    ),
    "deberta-v3-large": MiniCheckCheckpoint(
        name="deberta-v3-large",
        repo="lytang/MiniCheck-DeBERTa-v3-Large",
        revision="2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb",
        licence="MIT",
        architecture="classifier",
        max_model_len=2048,
        chunk_size=400,
        chunk_unit="tokens",
        files=("config.json", PICKLE_FILE, SAFETENSORS_FILE, "spm.model", *TOKENIZER_FILES),
    ),
}
DEFAULT_CHECKPOINT = "flan-t5-large"
EXCLUDED_CHECKPOINTS = {
    "Bespoke-MiniCheck-7B": "licence CC BY-NC 4.0 (non-commercial) and served by the official code through vLLM only",
}


# =============================================================================
# The official algorithm, as pure functions
# =============================================================================


def sent_tokenize_with_newlines(text: str, split_sentences: Callable[[str], list[str]]) -> list[str]:
    """Official ``sent_tokenize_with_newlines``: sentences block by block, a ``"\\n"`` item between blocks."""
    tokenized: list[str] = []
    for block in text.split("\n"):
        tokenized.extend(split_sentences(block))
        tokenized.append("\n")
    return tokenized[:-1]


def official_chunks(sentences: Sequence[str], chunk_size: int, measure: Callable[[str], int]) -> list[str]:
    """Official ``chunks``: greedy groups of sentences whose measure stays within *chunk_size*.

    A sentence longer than *chunk_size* makes a chunk of its own; the
    chunk open before it is yielded even when empty (filtered by the caller).
    """
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        size = measure(sentence)
        if count + size > chunk_size:
            chunks.append(" ".join(current))
            current = [sentence]
            count = size
        else:
            current.append(sentence)
            count += size
    if current:
        chunks.append(" ".join(current))
    return chunks


def document_chunks(
    document: str,
    checkpoint: MiniCheckCheckpoint,
    split_sentences: Callable[[str], list[str]],
    count_tokens: Callable[[str], int],
) -> list[str]:
    """Official ``inference_per_example`` preparation: the chunks of *document*, empty ones removed."""
    sentences = sent_tokenize_with_newlines(document, split_sentences) or [""]
    if checkpoint.chunk_unit == "words":

        def measure(sentence: str) -> int:
            return len(sentence.split())

    else:

        def measure(sentence: str) -> int:
            # The official measure tokenizes the sentence without special tokens, truncated at max_model_len
            return min(count_tokens(sentence), checkpoint.max_model_len)

    chunks = [
        chunk.replace(" \n ", "\n").strip() for chunk in official_chunks(sentences, checkpoint.chunk_size, measure)
    ]
    return [chunk for chunk in chunks if chunk != ""]


# =============================================================================
# Artifacts
# =============================================================================


@dataclass(frozen=True)
class MiniCheckArtifacts:
    """Where the checkpoint and the sentence model come from.

    Attributes:
        checkpoint: The checkpoint description.
        model_dir: Folder holding the weights and the tokenizer.
        revision: Pinned commit, when known.
        nltk_dir: Folder holding nltk's ``punkt_tab`` when bundled (``model_path``), else ``None``.
        source: ``hub`` or ``local``.
        digest: ``<repo>@<revision>``, or ``<repo>@sha256:<weights>`` for a folder without ``REVISIONS.json``.
    """

    checkpoint: MiniCheckCheckpoint
    model_dir: Path
    revision: str | None
    nltk_dir: Path | None
    source: str
    digest: str


def _file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def punkt_digest(language_dir: str | Path) -> str:
    """SHA-256 of the ``punkt_tab`` files of one language (names and contents, sorted)."""
    sha = hashlib.sha256()
    for path in sorted(Path(language_dir).iterdir()):
        if path.is_file():
            sha.update(path.name.encode("utf-8") + b"\0")
            sha.update(_file_sha256(path).encode("ascii") + b"\n")
    return sha.hexdigest()


def _dependency(module: str) -> JudgeDependencyError:
    return JudgeDependencyError(extra="minicheck", module=module, judge="MiniCheckJudge", alternative=None)


def resolve_artifacts(
    checkpoint: MiniCheckCheckpoint, model_path: str | Path | None = None, *, offline: bool = False
) -> MiniCheckArtifacts:
    """Locate the checkpoint: ``model_path`` as written by ``download``, else the Hugging Face cache, else a download.

    Raises:
        ConfigurationError: Incomplete *model_path*, or weights absent while offline.
        JudgeDependencyError: ``huggingface_hub`` is not installed (``[minicheck]`` extra).
    """
    if model_path is not None:
        root = Path(model_path)
        weights = root / (SAFETENSORS_FILE if (root / SAFETENSORS_FILE).is_file() else PICKLE_FILE)
        if not weights.is_file() or not (root / "config.json").is_file():
            msg = (
                f"MiniCheck model_path {root} must hold {PICKLE_FILE} (or {SAFETENSORS_FILE}) and config.json; "
                f"create it with MiniCheckJudge.download({str(root)!r}, checkpoint={checkpoint.name!r})"
            )
            raise ConfigurationError(msg)
        revisions: dict[str, Any] = {}
        if (root / REVISIONS_FILE).is_file():
            revisions = json.loads((root / REVISIONS_FILE).read_text(encoding="utf-8"))
        if revisions.get("repo") not in (None, checkpoint.repo):
            msg = f"MiniCheck model_path {root} holds {revisions.get('repo')}, not {checkpoint.repo}"
            raise ConfigurationError(msg)
        revision = revisions.get("revision")
        nltk_dir = root / NLTK_SUBDIR
        digest = f"{checkpoint.repo}@{revision}" if revision else f"{checkpoint.repo}@sha256:{_file_sha256(weights)}"
        return MiniCheckArtifacts(checkpoint, root, revision, nltk_dir if nltk_dir.is_dir() else None, "local", digest)
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise _dependency("huggingface_hub") from exc
    kwargs: dict[str, Any] = {"revision": checkpoint.revision, "allow_patterns": list(checkpoint.files)}
    try:
        model_dir = str(snapshot_download(checkpoint.repo, local_files_only=True, **kwargs))
    except Exception as exc:  # not in the cache
        if offline:
            msg = (
                f"{checkpoint.repo}@{checkpoint.revision[:12]} is not in the local Hugging Face cache and "
                "offline=True. Run MiniCheckJudge.download(directory) on a connected machine and pass "
                "model_path=, or allow the download"
            )
            raise ConfigurationError(msg) from exc
        model_dir = str(snapshot_download(checkpoint.repo, **kwargs))
    return MiniCheckArtifacts(
        checkpoint, Path(model_dir), checkpoint.revision, None, "hub", f"{checkpoint.repo}@{checkpoint.revision}"
    )


def download_artifacts(directory: str | Path, checkpoint: str = DEFAULT_CHECKPOINT) -> MiniCheckArtifacts:
    """Write a self-contained folder: weights, tokenizer, nltk ``punkt_tab`` and ``REVISIONS.json``."""
    spec = checkpoint_named(checkpoint)
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise _dependency("huggingface_hub") from exc
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(spec.repo, revision=spec.revision, allow_patterns=list(spec.files), local_dir=str(root))
    try:
        import nltk
    except ModuleNotFoundError as exc:
        raise _dependency("nltk") from exc
    nltk.download("punkt_tab", download_dir=str(root / NLTK_SUBDIR), quiet=True)
    payload = {"checkpoint": spec.name, "repo": spec.repo, "revision": spec.revision, "licence": spec.licence}
    (root / REVISIONS_FILE).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return resolve_artifacts(spec, root)


def checkpoint_named(name: str) -> MiniCheckCheckpoint:
    """The checkpoint description for *name*; excluded names say why."""
    if name in EXCLUDED_CHECKPOINTS:
        msg = f"MiniCheck checkpoint {name!r} is not offered: {EXCLUDED_CHECKPOINTS[name]}"
        raise ValueError(msg)
    if name not in CHECKPOINTS:
        msg = f"Unknown MiniCheck checkpoint {name!r}; expected one of {sorted(CHECKPOINTS)}"
        raise ValueError(msg)
    return CHECKPOINTS[name]


# =============================================================================
# Backend, the loaded tokenizer and model
# =============================================================================


class MiniCheckBackend(Protocol):
    """What the judge needs from a loaded checkpoint; ``TransformersBackend`` is the real one."""

    def split_sentences(self, text: str) -> list[str]:
        """Sentences of one block of text (nltk ``sent_tokenize`` officially)."""
        ...  # pragma: no cover

    def splitter_id(self) -> str:
        """What splits the sentences, precisely enough to tell two splittings apart (nltk version, punkt hash)."""
        ...  # pragma: no cover

    def count_tokens(self, text: str) -> int:
        """Tokens of *text* without special tokens (the classifiers' chunk measure)."""
        ...  # pragma: no cover

    def support(self, chunks: Sequence[str], claim: str) -> tuple[list[float], int, bool]:
        """P(supported) per chunk, tokens sent in all, whether any input was truncated."""
        ...  # pragma: no cover


def weights_format(model_dir: str | Path) -> str:
    """``safetensors`` when the folder holds ``model.safetensors``, else ``pytorch`` (the pinned revisions)."""
    return "safetensors" if (Path(model_dir) / SAFETENSORS_FILE).is_file() else "pytorch"


def _nltk_sentence_splitter(nltk_dir: Path | None, *, offline: bool) -> tuple[Callable[[str], list[str]], str]:
    """nltk's ``sent_tokenize`` and its identifier: version plus the hash of the ``punkt_tab`` files it loads."""
    try:
        import nltk
    except ModuleNotFoundError as exc:
        raise _dependency("nltk") from exc
    if nltk_dir is not None and str(nltk_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_dir))
    resource = f"tokenizers/punkt_tab/{PUNKT_LANGUAGE}/"
    try:
        found = nltk.data.find(resource)
    except LookupError as exc:
        if offline:
            msg = (
                "nltk's punkt_tab sentence model is not installed and offline=True. Run "
                "MiniCheckJudge.download(directory) on a connected machine, or nltk.download('punkt_tab')"
            )
            raise ConfigurationError(msg) from exc
        nltk.download("punkt_tab", quiet=True)
        found = nltk.data.find(resource)
    from nltk.tokenize import sent_tokenize

    def split(text: str) -> list[str]:
        return [str(s) for s in sent_tokenize(text, language=PUNKT_LANGUAGE)]

    identifier = f"nltk-{nltk.__version__}:punkt_tab-{PUNKT_LANGUAGE}:{punkt_digest(str(found))}"
    return split, identifier


class TransformersBackend:
    """A MiniCheck checkpoint loaded with ``transformers`` and ``torch`` on an explicit device, in eval mode."""

    def __init__(self, artifacts: MiniCheckArtifacts, device: str = "cpu", *, offline: bool = False) -> None:
        """Load the tokenizer, the weights and the sentence model.

        Raises:
            JudgeDependencyError: ``torch``, ``transformers`` or ``nltk`` missing.
            ConfigurationError: Device not available, sentence model absent offline.
            ValueError: Unknown device name.
        """
        _validate_device(device)
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise _dependency(exc.name or "torch") from exc
        kind = device.split(":", 1)[0]
        if kind == "mps" and not torch.backends.mps.is_available():
            msg = "MiniCheckJudge device 'mps' requested but Metal is not available on this machine"
            raise ConfigurationError(msg)
        if kind == "cuda" and not torch.cuda.is_available():
            msg = "MiniCheckJudge device 'cuda' requested but CUDA is not available on this machine"
            raise ConfigurationError(msg)
        self.checkpoint = artifacts.checkpoint
        folder = str(artifacts.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(folder, use_fast=True)
        # The chunk measure counts tokens beyond the tokenizer's nominal length on purpose: no warning
        warnings = getattr(self.tokenizer, "deprecation_warnings", None)
        if isinstance(warnings, dict):
            warnings["sequence-length-is-longer-than-the-specified-maximum"] = True
        self.weights_format = weights_format(folder)
        loader = (
            AutoModelForSeq2SeqLM if self.checkpoint.architecture == "seq2seq" else AutoModelForSequenceClassification
        )
        extra: dict[str, Any] = {} if self.checkpoint.architecture == "seq2seq" else {"num_labels": 2}
        try:
            # weights_only=True whatever the torch version: a pickle carrying anything but tensors is refused
            model = loader.from_pretrained(
                folder, use_safetensors=self.weights_format == "safetensors", weights_only=True, **extra
            )
        except Exception as exc:  # unpickling refused, corrupt file, architecture mismatch
            msg = (
                f"MiniCheck checkpoint in {folder} could not be loaded as plain tensors "
                f"({type(exc).__name__}: {str(exc)[:200]})"
            )
            raise ConfigurationError(msg) from exc
        model = model.to(device).eval()
        if device.split(":", 1)[0] == "cpu":
            # Parameters loaded from a pickle stay lazily backed on CPU (transformers 5) and every forward
            # pays seconds to rematerialize them; one contiguous copy brings a short input to ~150 ms
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.data = parameter.data.contiguous().clone()
        self.model = model
        self.device = device
        self._split, self._splitter_id = _nltk_sentence_splitter(artifacts.nltk_dir, offline=offline)
        self._torch = torch

    def split_sentences(self, text: str) -> list[str]:
        return self._split(text)

    def splitter_id(self) -> str:
        return self._splitter_id

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def support(self, chunks: Sequence[str], claim: str) -> tuple[list[float], int, bool]:
        torch = self._torch
        spec = self.checkpoint
        eos = self.tokenizer.eos_token
        texts = [eos.join([chunk, claim]) for chunk in chunks]
        if spec.architecture == "seq2seq":
            texts = [FLAN_PREFIX + text for text in texts]
        probabilities: list[float] = []
        tokens = 0
        truncated = False
        for start in range(0, len(texts), OFFICIAL_BATCH_SIZE):
            batch = texts[start : start + OFFICIAL_BATCH_SIZE]
            full = self.tokenizer(batch, padding=False, truncation=False)["input_ids"]
            truncated = truncated or any(len(ids) > spec.max_model_len for ids in full)
            tokens += sum(min(len(ids), spec.max_model_len) for ids in full)
            inputs = self.tokenizer(
                batch, max_length=spec.max_model_len, truncation=True, padding=True, return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                if spec.architecture == "seq2seq":
                    decoder_input_ids = torch.zeros((inputs["input_ids"].size(0), 1), dtype=torch.long).to(self.device)
                    logits = self.model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        decoder_input_ids=decoder_input_ids,
                    ).logits.squeeze(1)
                    label_logits = logits[:, torch.tensor(FLAN_LABEL_TOKEN_IDS)].cpu()
                    probs = torch.nn.functional.softmax(label_logits, dim=-1)
                else:
                    probs = torch.nn.functional.softmax(self.model(**inputs).logits, dim=1).cpu()
            probabilities.extend(float(p) for p in probs[:, 1].tolist())
        return probabilities, tokens, truncated


# =============================================================================
# Judge
# =============================================================================


class _AnswerLevel:
    def __init__(self, owner: MiniCheckJudge) -> None:
        self._owner = owner

    @property
    def criteria(self) -> frozenset[str]:
        return self._owner.criteria

    def config(self, criterion: str) -> JudgeConfig:
        return self._owner._answer_config(criterion)

    async def judge(self, criterion: str, *, question: str, context: str, answer: str) -> JudgeResult:  # noqa: ARG002
        return self._owner._judge_answer(criterion, context=context, answer=answer)


class MiniCheckJudge:
    """MiniCheck as a ``JudgeProtocol`` judge.

    Criteria ``faithfulness`` and ``hallucination`` give the same judgment:
    the score is the maximum over the document's chunks of P(supported) for
    the claim, PASS at 0.5 or above. The threshold is the official code's
    reading and is not calibrated on any ragnarok-ai label set.

    With the default ``unit="sentence"`` each sentence of the answer is a
    claim checked against the whole context, as the model card recommends,
    and the answer's score is the fraction of PASS sentences with the worst
    verdict (``SentenceLevel``). With ``unit="answer"`` the whole answer is
    one claim.

    Example:
        >>> judge = MiniCheckJudge(device="cpu")  # MiniCheck-Flan-T5-Large
        >>> result = await judge.judge("faithfulness", question=q, context=c, answer=a)
    """

    CRITERIA = frozenset({"faithfulness", "hallucination"})
    VERDICT_POLICY = ScoreThreshold(pass_threshold=PASS_THRESHOLD)

    def __init__(
        self,
        *,
        checkpoint: str = DEFAULT_CHECKPOINT,
        model_path: str | Path | None = None,
        device: str = "cpu",
        unit: Unit = "sentence",
        chunk_size: int | None = None,
        max_model_len: int | None = None,
        offline: bool = False,
        aggregation: AggregationPolicy = DEFAULT_AGGREGATION,
        lang: str = "en",
        backend: MiniCheckBackend | None = None,
        model_digest: str | None = None,
    ) -> None:
        """Load a MiniCheck checkpoint.

        Args:
            checkpoint: ``flan-t5-large`` (default), ``roberta-large`` or ``deberta-v3-large``.
            model_path: Folder written by ``MiniCheckJudge.download``; nothing is fetched.
            device: ``cpu``, ``mps`` or ``cuda`` (``cuda:N`` accepted).
            unit: ``sentence`` (default) or ``answer``.
            chunk_size: Official default per checkpoint unless given (500 words for Flan-T5).
            max_model_len: Official truncation length unless given (2048 for Flan-T5).
            offline: Never download (weights, nor nltk's sentence model).
            aggregation: Sentence aggregation policy (``unit="sentence"``).
            lang: Segmentation language of the answer (``unit="sentence"``).
            backend: A loaded ``MiniCheckBackend``; for tests and custom loaders.
            model_digest: Recorded with *backend*.

        Raises:
            ValueError: Unknown checkpoint, excluded checkpoint, device name, unit, or non-positive sizes.
            JudgeDependencyError: The ``[minicheck]`` extra is not installed.
            ConfigurationError: Weights or sentence model missing offline, incomplete ``model_path``, device unavailable.
        """
        spec = checkpoint_named(checkpoint)
        _validate_device(device)
        if unit not in ("answer", "sentence"):
            msg = f"MiniCheckJudge unit must be 'sentence' or 'answer', got {unit!r}"
            raise ValueError(msg)
        if (chunk_size is not None and chunk_size < 1) or (max_model_len is not None and max_model_len < 1):
            msg = "MiniCheckJudge chunk_size and max_model_len must be at least 1"
            raise ValueError(msg)
        if chunk_size is not None or max_model_len is not None:
            spec = replace(
                spec,
                chunk_size=chunk_size if chunk_size is not None else spec.chunk_size,
                max_model_len=max_model_len if max_model_len is not None else spec.max_model_len,
            )
        self.checkpoint = spec
        self.device = device
        self.unit: Unit = unit
        self.aggregation = aggregation
        self.lang = lang
        self.artifacts: MiniCheckArtifacts | None = None
        if backend is None:
            self.artifacts = resolve_artifacts(spec, model_path, offline=offline)
            self.model_digest = self.artifacts.digest
            self.backend: MiniCheckBackend = TransformersBackend(self.artifacts, device, offline=offline)
        else:
            self.backend = backend
            self.model_digest = model_digest or "injected-backend"
        self._sentence_level = SentenceLevel(_AnswerLevel(self), aggregation=aggregation, lang=lang)

    @staticmethod
    def download(directory: str | Path, checkpoint: str = DEFAULT_CHECKPOINT) -> MiniCheckArtifacts:
        """Download the checkpoint and nltk's sentence model into *directory* for later ``model_path=`` use."""
        return download_artifacts(directory, checkpoint)

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

        spec = self.checkpoint
        return JudgeConfig(
            judge_class=f"{type(self).__module__}.{type(self).__qualname__}",
            judge_kind="classifier",
            model=spec.repo,
            criterion=criterion,
            verdict_policy=self.VERDICT_POLICY,
            ragnarok_ai_version=str(__version__),
            backend="transformers",
            model_digest=self.model_digest,
            options=(("device", self.device), ("dtype", "float32"), ("max_model_len", spec.max_model_len)),
            prompt_format=None,
            prompt_template_sha256=None,
            judge_specific=(
                ("checkpoint", spec.name),
                ("chunk_size", spec.chunk_size),
                ("chunk_unit", spec.chunk_unit),
                ("official_code", f"{OFFICIAL_CODE}@{OFFICIAL_COMMIT}"),
                ("question_used", False),
                ("sentence_splitter", self.backend.splitter_id()),
                ("threshold_calibrated", False),
                ("training_length", spec.max_model_len),
                ("training_length_kind", TRAINING_LENGTH_KIND),
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

    def chunks(self, document: str) -> list[str]:
        """The chunks the official code would check the claim against."""
        return document_chunks(document, self.checkpoint, self.backend.split_sentences, self.backend.count_tokens)

    def _judge_answer(self, criterion: str, *, context: str, answer: str) -> JudgeResult:
        """One official ``score`` call: *context* is the document, *answer* the claim."""
        from ragnarok_ai.evaluators.judge import JudgeResult

        started = time.perf_counter()
        chunks = self.chunks(context)
        if chunks:
            per_chunk, tokens, truncated = self.backend.support(chunks, answer)
            probability = min(1.0, max(0.0, max(per_chunk)))
        else:
            # The official code fails on a document without any chunk; nothing supports the claim here
            per_chunk, tokens, truncated = [], 0, False
            probability = 0.0
        output = BinaryProbability(probability)
        verdict = self.VERDICT_POLICY.verdict(output)
        explanation = (
            f"MiniCheck {self.checkpoint.name} max P(supported) = {probability:.3f} over {len(chunks)} chunk(s)"
            + (f" [{', '.join(f'{p:.3f}' for p in per_chunk)}]" if len(per_chunk) > 1 else "")
            + f"; PASS at >= {PASS_THRESHOLD} (official default, not calibrated)"
        )
        if not chunks:
            explanation += "; empty document"
        if truncated:
            explanation += f"; at least one chunk truncated at {self.checkpoint.max_model_len} tokens"
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
        """Judge *answer* against *context*; *question* is not part of MiniCheck's input.

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
    "CHECKPOINTS",
    "DEFAULT_CHECKPOINT",
    "DEVICES",
    "EXCLUDED_CHECKPOINTS",
    "FLAN_LABEL_TOKEN_IDS",
    "FLAN_PREFIX",
    "OFFICIAL_BATCH_SIZE",
    "OFFICIAL_CODE",
    "OFFICIAL_COMMIT",
    "PASS_THRESHOLD",
    "PUNKT_LANGUAGE",
    "TRAINING_LENGTH_KIND",
    "MiniCheckArtifacts",
    "MiniCheckBackend",
    "MiniCheckCheckpoint",
    "MiniCheckJudge",
    "TransformersBackend",
    "checkpoint_named",
    "document_chunks",
    "download_artifacts",
    "official_chunks",
    "punkt_digest",
    "resolve_artifacts",
    "sent_tokenize_with_newlines",
    "weights_format",
]
