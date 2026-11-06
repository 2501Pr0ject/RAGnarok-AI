"""MiniCheckJudge: official chunking, contract, truncation flag, sentence unit, artifacts, CLI; integration by hand."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ragnarok_ai.core import JudgeProtocol
from ragnarok_ai.core.exceptions import ConfigurationError, JudgeDependencyError, NoUnitsError
from ragnarok_ai.judges import AggregationPolicy, BinaryProbability, MiniCheckJudge, ScoreThreshold
from ragnarok_ai.judges import minicheck as minicheck_module
from ragnarok_ai.judges.minicheck import (
    CHECKPOINTS,
    DEFAULT_CHECKPOINT,
    EXCLUDED_CHECKPOINTS,
    FLAN_LABEL_TOKEN_IDS,
    FLAN_PREFIX,
    OFFICIAL_BATCH_SIZE,
    OFFICIAL_CODE,
    OFFICIAL_COMMIT,
    TRAINING_LENGTH_KIND,
    MiniCheckArtifacts,
    TransformersBackend,
    checkpoint_named,
    document_chunks,
    official_chunks,
    punkt_digest,
    resolve_artifacts,
    sent_tokenize_with_newlines,
    weights_format,
)
from ragnarok_ai.text import SEGMENTER_ID

REFERENCE_FIXTURE = Path(__file__).parent / "fixtures" / "minicheck_reference.json"
REFERENCE = json.loads(REFERENCE_FIXTURE.read_text(encoding="utf-8"))
FLAN = CHECKPOINTS["flan-t5-large"]


def naive_sentences(text: str) -> list[str]:
    """Split on sentence-final punctuation followed by a space; good enough to stand in for nltk."""
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


class FakeBackend:
    """Whitespace tokens; P(supported) of a chunk is read from a '(p=0.x)' marker in the chunk or 0.1 if absent."""

    def __init__(self, splitter: str = "naive-regex:v1") -> None:
        self.calls: list[tuple[list[str], str]] = []
        self.splitter = splitter

    def split_sentences(self, text: str) -> list[str]:
        return naive_sentences(text)

    def splitter_id(self) -> str:
        return self.splitter

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def support(self, chunks, claim):
        self.calls.append((list(chunks), claim))
        probabilities = []
        for chunk in chunks:
            found = re.search(r"\(p=([0-9.]+)\)", chunk)
            probabilities.append(float(found.group(1)) if found else 0.1)
        tokens = sum(len(c.split()) + len(claim.split()) for c in chunks)
        return probabilities, tokens, any(len(c.split()) > 60 for c in chunks)


def make_judge(splitter: str = "naive-regex:v1", **kwargs) -> tuple[MiniCheckJudge, FakeBackend]:
    backend = FakeBackend(splitter)
    return MiniCheckJudge(backend=backend, **kwargs), backend


CONTEXT = "The policy allows refunds within 30 days of purchase (p=0.95). Gifts are excluded."


class TestOfficialChunking:
    def test_sent_tokenize_with_newlines_keeps_a_newline_item_between_blocks(self):
        out = sent_tokenize_with_newlines("One. Two.\nThree.\n\nFour.", naive_sentences)
        assert out == ["One.", "Two.", "\n", "Three.", "\n", "\n", "Four."]
        assert sent_tokenize_with_newlines("", naive_sentences) == []

    def test_official_chunks_never_cut_a_sentence_and_yield_the_open_chunk_before_a_long_one(self):
        sentences = ["a b c", "d e", "f g h i j k", "l"]
        assert official_chunks(sentences, 5, lambda s: len(s.split())) == ["a b c d e", "f g h i j k", "l"]
        # a first sentence longer than the size: the empty open chunk is yielded first (official behavior)
        assert official_chunks(["x y z", "w"], 2, lambda s: len(s.split())) == ["", "x y z", "w"]
        assert official_chunks([], 5, lambda s: len(s.split())) == []

    def test_document_chunks_matches_the_official_preparation(self):
        from dataclasses import replace

        doc = "One two three. Four five.\nSix seven eight nine. Ten."
        assert document_chunks(doc, FLAN, naive_sentences, lambda _s: 0) == [doc]  # 500 words: a single chunk
        small = replace(FLAN, chunk_size=5)
        chunks = document_chunks(doc, small, naive_sentences, lambda _s: 0)
        # the "\n" item counts for zero words and is normalized to a line break inside a chunk
        assert chunks == ["One two three. Four five.", "Six seven eight nine. Ten."]
        assert document_chunks("", FLAN, naive_sentences, lambda _s: 0) == []
        assert document_chunks("\n\n", FLAN, naive_sentences, lambda _s: 0) == []

    def test_token_measure_is_capped_at_max_model_len_for_classifiers(self):
        from dataclasses import replace

        roberta = replace(CHECKPOINTS["roberta-large"], chunk_size=3, max_model_len=4)
        chunks = document_chunks("aaaa. bb. c.", roberta, naive_sentences, lambda s: 100 if s.startswith("a") else 1)
        # "aaaa." measures min(100, 4) = 4 > 3: alone; "bb." and "c." fit together
        assert chunks == ["aaaa.", "bb. c."]


class TestContract:
    def test_is_a_judge_with_two_criteria_and_a_binary_policy(self):
        judge, _ = make_judge()
        assert isinstance(judge, JudgeProtocol)
        assert judge.criteria == frozenset({"faithfulness", "hallucination"})
        assert ScoreThreshold(pass_threshold=0.5) == judge.VERDICT_POLICY

    def test_config_records_checkpoint_revision_device_official_code_and_uncalibrated_threshold(self):
        judge, _ = make_judge(unit="answer", device="mps", model_digest=f"{FLAN.repo}@{FLAN.revision}")
        config = judge.config("faithfulness")
        assert config.judge_class == "ragnarok_ai.judges.minicheck.MiniCheckJudge"
        assert (config.judge_kind, config.backend, config.model) == ("classifier", "transformers", FLAN.repo)
        assert config.model_digest == f"{FLAN.repo}@{FLAN.revision}"
        assert config.options == (("device", "mps"), ("dtype", "float32"), ("max_model_len", 2048))
        assert config.prompt_format is None and config.prompt_template_sha256 is None
        assert dict(config.judge_specific) == {
            "checkpoint": "flan-t5-large",
            "chunk_size": 500,
            "chunk_unit": "words",
            "official_code": f"{OFFICIAL_CODE}@{OFFICIAL_COMMIT}",
            "question_used": False,
            "sentence_splitter": "naive-regex:v1",
            "threshold_calibrated": False,
            "training_length": 2048,
            "training_length_kind": TRAINING_LENGTH_KIND,
        }
        assert TRAINING_LENGTH_KIND == "official truncation length (max_model_len)"
        other, _ = make_judge(unit="answer", checkpoint="roberta-large")
        specific = dict(other.config("faithfulness").judge_specific)
        assert (specific["chunk_size"], specific["chunk_unit"], specific["training_length"]) == (400, "tokens", 512)
        assert not config.comparable_with(other.config("faithfulness"))

    def test_two_runs_with_different_sentence_splitters_are_never_comparable(self):
        """The splitter's identity (nltk version and punkt_tab hash for the real backend) is in the configuration."""
        a, _ = make_judge("nltk-3.9.2:punkt_tab-english:" + "a" * 64, unit="answer")
        b, _ = make_judge("nltk-3.9.2:punkt_tab-english:" + "b" * 64, unit="answer")
        same, _ = make_judge("nltk-3.9.2:punkt_tab-english:" + "a" * 64, unit="answer")
        assert not a.config("faithfulness").comparable_with(b.config("faithfulness"))
        assert a.config("faithfulness").comparable_with(same.config("faithfulness"))
        assert a.config("faithfulness").config_hash != b.config("faithfulness").config_hash

    def test_punkt_digest_depends_on_file_names_and_contents(self, tmp_path: Path):
        english = tmp_path / "english"
        english.mkdir()
        (english / "abbrev_types.txt").write_text("a\nb\n")
        (english / "collocations.tab").write_text("x\ty\n")
        first = punkt_digest(english)
        assert first == punkt_digest(english) and len(first) == 64
        (english / "abbrev_types.txt").write_text("a\nb\nc\n")
        assert punkt_digest(english) != first
        (english / "abbrev_types.txt").write_text("a\nb\n")
        (english / "sent_starters.txt").write_text("")
        assert punkt_digest(english) != first  # a new file changes it too

    def test_custom_chunk_size_and_max_model_len_are_part_of_the_configuration(self):
        judge, _ = make_judge(unit="answer", chunk_size=100, max_model_len=1024)
        config = judge.config("faithfulness")
        assert dict(config.judge_specific)["chunk_size"] == 100 and dict(config.options)["max_model_len"] == 1024
        assert dict(config.judge_specific)["training_length"] == 1024
        default, _ = make_judge(unit="answer")
        assert not config.comparable_with(default.config("faithfulness"))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"checkpoint": "Bespoke-MiniCheck-7B"}, "not offered: licence CC BY-NC 4.0"),
            ({"checkpoint": "gpt"}, "Unknown MiniCheck checkpoint"),
            ({"device": "tpu"}, "device must be one of"),
            ({"unit": "claim"}, "unit must be 'sentence' or 'answer'"),
            ({"chunk_size": 0}, "must be at least 1"),
            ({"max_model_len": 0}, "must be at least 1"),
        ],
    )
    def test_invalid_parameters(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            make_judge(**kwargs)

    def test_checkpoints_are_mit_and_pinned_and_bespoke_is_excluded(self):
        assert set(CHECKPOINTS) == {"flan-t5-large", "roberta-large", "deberta-v3-large"}
        assert DEFAULT_CHECKPOINT == "flan-t5-large"
        for spec in CHECKPOINTS.values():
            assert spec.licence == "MIT" and re.fullmatch(r"[0-9a-f]{40}", spec.revision)
            assert "pytorch_model.bin" in spec.files and "config.json" in spec.files
        assert "Bespoke-MiniCheck-7B" in EXCLUDED_CHECKPOINTS
        assert checkpoint_named("deberta-v3-large").max_model_len == 2048
        assert (FLAN_PREFIX, FLAN_LABEL_TOKEN_IDS, OFFICIAL_BATCH_SIZE) == ("predict: ", (3, 209), 16)

    @pytest.mark.asyncio
    async def test_unknown_criterion(self):
        judge, _ = make_judge()
        with pytest.raises(ValueError, match="Unknown criterion 'relevance'"):
            await judge.judge("relevance", question="q", context=CONTEXT, answer="Refunds take 30 days.")


class TestAnswerLevel:
    @pytest.mark.asyncio
    async def test_score_is_the_max_over_chunks_and_the_verdict_follows_the_threshold(self):
        judge, backend = make_judge(unit="answer", chunk_size=8)
        context = "First block of words here now (p=0.2). Second block with the answer (p=0.7). Third one (p=0.4)."
        result = await judge.judge("faithfulness", question="q", context=context, answer="The answer is here.")

        chunks, claim = backend.calls[0]
        assert len(chunks) == 3 and claim == "The answer is here."  # the claim is never split
        assert (result.score, result.verdict, result.raw_score) == (0.7, "PASS", None)
        assert result.output == BinaryProbability(0.7)
        assert "over 3 chunk(s) [0.200, 0.700, 0.400]" in result.explanation and "not calibrated" in result.explanation
        assert result.units is None and result.config == judge.config("faithfulness")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("marker", "verdict"), [("(p=0.5)", "PASS"), ("(p=0.49)", "FAIL")])
    async def test_threshold_boundary(self, marker, verdict):
        judge, _ = make_judge(unit="answer")
        result = await judge.judge("hallucination", question="q", context=f"Some context {marker}.", answer="Claim.")
        assert result.verdict == verdict

    @pytest.mark.asyncio
    async def test_trace_counts_tokens_sent_and_reports_truncation_from_the_backend(self):
        judge, _ = make_judge(unit="answer")
        short = await judge.judge("faithfulness", question="q", context=CONTEXT, answer="Refunds within a month.")
        assert short.trace is not None
        assert short.trace.prompt_tokens == len(CONTEXT.split()) + 4
        assert short.trace.context_truncated is False and short.trace.completion_tokens is None

        long_context = " ".join(["word"] * 80) + " (p=0.9)."  # one 81-word sentence: a chunk of its own, > 60 tokens
        cut = await judge.judge("faithfulness", question="q", context=long_context, answer="Claim.")
        assert cut.trace is not None and cut.trace.context_truncated is True
        assert "truncated at 2048 tokens" in cut.explanation

    @pytest.mark.asyncio
    async def test_empty_document_gives_probability_zero_instead_of_the_official_crash(self):
        judge, backend = make_judge(unit="answer")
        result = await judge.judge("faithfulness", question="q", context="\n", answer="Anything.")
        assert backend.calls == []  # nothing is sent
        assert (result.score, result.verdict) == (0.0, "FAIL")
        assert "empty document" in result.explanation
        assert result.trace is not None and result.trace.prompt_tokens == 0


class TestSentenceLevelDefault:
    ANSWER = "Refunds are possible within a month. Gifts can be refunded too. Sure!"

    @pytest.mark.asyncio
    async def test_each_sentence_is_a_claim_checked_against_the_whole_context(self):
        judge, backend = make_judge()
        result = await judge.judge("faithfulness", question="q", context=CONTEXT, answer=self.ANSWER)

        assert [claim for _, claim in backend.calls] == [
            "Refunds are possible within a month.",
            "Gifts can be refunded too.",
        ]
        assert all(chunks == [CONTEXT] for chunks, _ in backend.calls)
        assert result.units is not None and [u.skipped for u in result.units] == [None, None, "short_fragment"]
        assert [u.verdict for u in result.units[:2]] == ["PASS", "PASS"]  # the fake scores the chunk, not the claim
        assert (result.score, result.verdict) == (1.0, "PASS")
        assert result.raw_score is None and result.output is None and result.trace is None

    @pytest.mark.asyncio
    async def test_config_is_at_sentence_unit_under_the_minicheck_class_name(self):
        judge, _ = make_judge()
        config = judge.config("faithfulness")
        assert config.judge_class == "ragnarok_ai.judges.minicheck.MiniCheckJudge"
        assert config.unit == "sentence" and config.segmenter == SEGMENTER_ID
        assert config.aggregation == AggregationPolicy("fraction_pass", "worst")
        result = await judge.judge("faithfulness", question="q", context=CONTEXT, answer=self.ANSWER)
        assert result.config == config

    @pytest.mark.asyncio
    async def test_no_units(self):
        judge, _ = make_judge()
        with pytest.raises(NoUnitsError):
            await judge.judge("faithfulness", question="q", context=CONTEXT, answer="Sure!")


class TestArtifacts:
    @staticmethod
    def local_folder(tmp_path: Path, *, revisions: bool = True, nltk: bool = True) -> Path:
        root = tmp_path / "minicheck"
        root.mkdir()
        (root / "pytorch_model.bin").write_bytes(b"weights")
        (root / "config.json").write_text("{}")
        if nltk:
            (root / "nltk_data").mkdir()
        if revisions:
            (root / "REVISIONS.json").write_text(json.dumps({"repo": FLAN.repo, "revision": FLAN.revision}))
        return root

    def test_local_folder_with_revisions(self, tmp_path: Path):
        root = self.local_folder(tmp_path)
        artifacts = resolve_artifacts(FLAN, root)
        assert artifacts == MiniCheckArtifacts(
            FLAN, root, FLAN.revision, root / "nltk_data", "local", f"{FLAN.repo}@{FLAN.revision}"
        )

    def test_local_folder_without_revisions_hashes_the_weights(self, tmp_path: Path):
        root = self.local_folder(tmp_path, revisions=False, nltk=False)
        artifacts = resolve_artifacts(FLAN, root)
        assert artifacts.revision is None and artifacts.nltk_dir is None
        assert artifacts.digest == f"{FLAN.repo}@sha256:{hashlib.sha256(b'weights').hexdigest()}"

    def test_local_folder_of_another_checkpoint_is_refused(self, tmp_path: Path):
        root = self.local_folder(tmp_path)
        with pytest.raises(
            ConfigurationError, match="holds lytang/MiniCheck-Flan-T5-Large, not lytang/MiniCheck-RoBERTa-Large"
        ):
            resolve_artifacts(CHECKPOINTS["roberta-large"], root)
        with pytest.raises(ConfigurationError, match=r"must hold pytorch_model\.bin"):
            resolve_artifacts(FLAN, tmp_path)

    @pytest.fixture
    def fake_hub(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        calls: list[dict[str, object]] = []
        cached: set[str] = set()

        def snapshot_download(repo_id, *, revision=None, allow_patterns=None, local_files_only=False, local_dir=None):
            calls.append(
                {
                    "repo": repo_id,
                    "revision": revision,
                    "patterns": tuple(allow_patterns or ()),
                    "local_only": local_files_only,
                    "local_dir": local_dir,
                }
            )
            if local_files_only and repo_id not in cached:
                msg = "not in cache"
                raise OSError(msg)
            target = Path(local_dir) if local_dir else tmp_path / repo_id.replace("/", "--") / str(revision)
            target.mkdir(parents=True, exist_ok=True)
            return str(target)

        module = ModuleType("huggingface_hub")
        module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)
        return calls, cached

    def test_hub_cache_first_then_pinned_download_limited_to_the_needed_files(self, fake_hub):
        calls, cached = fake_hub
        cached.add(FLAN.repo)
        artifacts = resolve_artifacts(FLAN)
        assert [c["local_only"] for c in calls] == [True] and artifacts.source == "hub"
        assert artifacts.digest == f"{FLAN.repo}@{FLAN.revision}" and artifacts.nltk_dir is None
        calls.clear()
        resolve_artifacts(CHECKPOINTS["deberta-v3-large"])
        download = [c for c in calls if not c["local_only"]]
        assert len(download) == 1 and download[0]["revision"] == CHECKPOINTS["deberta-v3-large"].revision
        assert download[0]["patterns"] == CHECKPOINTS["deberta-v3-large"].files
        assert not any(name.endswith((".png", ".py", ".txt")) for name in download[0]["patterns"])

    def test_offline_refuses_to_download(self, fake_hub):
        calls, _ = fake_hub
        with pytest.raises(ConfigurationError, match="offline=True"):
            resolve_artifacts(FLAN, offline=True)
        assert all(c["local_only"] for c in calls)

    def test_download_writes_weights_nltk_data_and_revisions(
        self, fake_hub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        calls, _ = fake_hub
        downloads: list[tuple[str, str | None]] = []
        nltk = ModuleType("nltk")
        nltk.download = lambda name, download_dir=None, **_kw: downloads.append((name, download_dir)) or True  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "nltk", nltk)
        root = tmp_path / "bundle"
        root.mkdir()
        (root / "pytorch_model.bin").write_bytes(b"w")  # what the fake hub would have written
        (root / "config.json").write_text("{}")
        (root / "nltk_data").mkdir()
        artifacts = MiniCheckJudge.download(root)
        assert calls[0]["local_dir"] == str(root) and calls[0]["revision"] == FLAN.revision
        assert downloads == [("punkt_tab", str(root / "nltk_data"))]
        assert json.loads((root / "REVISIONS.json").read_text())["revision"] == FLAN.revision
        assert artifacts.source == "local" and artifacts.nltk_dir == root / "nltk_data"

    def test_missing_huggingface_hub_names_the_extra(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(JudgeDependencyError, match=r"MiniCheckJudge needs 'huggingface_hub'.*\[minicheck\]"):
            resolve_artifacts(FLAN)


class Payload:
    """Pickled into a fake checkpoint: unpickling it would run setattr, which weights_only refuses."""

    executed = False

    def __reduce__(self):
        return (setattr, (Payload, "executed", True))


class TestWeightsLoading:
    def test_safetensors_is_preferred_when_present(self, tmp_path: Path):
        assert weights_format(tmp_path) == "pytorch"
        (tmp_path / "model.safetensors").write_bytes(b"")
        assert weights_format(tmp_path) == "safetensors"
        for spec in CHECKPOINTS.values():
            assert "model.safetensors" in spec.files and "pytorch_model.bin" in spec.files

    def test_a_pickle_carrying_anything_but_tensors_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """weights_only=True: a checkpoint holding an arbitrary object fails cleanly and its payload never runs."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from transformers import AutoTokenizer, T5Config, T5TokenizerFast

        folder = tmp_path / "evil"
        folder.mkdir()
        config = T5Config(d_model=8, d_ff=16, num_layers=1, num_decoder_layers=1, num_heads=2, d_kv=4, vocab_size=32)
        config.save_pretrained(folder)
        torch.save({"shared.weight": torch.zeros(32, 8), "payload": Payload()}, folder / "pytorch_model.bin")
        monkeypatch.setattr(
            AutoTokenizer, "from_pretrained", lambda *_a, **_k: T5TokenizerFast.__new__(T5TokenizerFast)
        )
        monkeypatch.setattr(minicheck_module, "_nltk_sentence_splitter", lambda *_a, **_k: (naive_sentences, "fake"))
        artifacts = MiniCheckArtifacts(FLAN, folder, None, None, "local", "x")
        with pytest.raises(ConfigurationError, match="could not be loaded as plain tensors"):
            TransformersBackend(artifacts, "cpu")
        assert Payload.executed is False


class TestDependencies:
    def test_missing_torch_names_the_extra(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setitem(sys.modules, "torch", None)
        artifacts = MiniCheckArtifacts(FLAN, tmp_path, None, None, "local", "x")
        with pytest.raises(JudgeDependencyError, match=r"needs 'torch'.*ragnarok-ai\[minicheck\]"):
            TransformersBackend(artifacts, "cpu")

    def test_backend_validates_the_device_name_first(self, tmp_path: Path):
        with pytest.raises(ValueError, match="device must be one of"):
            TransformersBackend(MiniCheckArtifacts(FLAN, tmp_path, None, None, "local", "x"), "gpu")


class TestReferenceFixture:
    """The official package's values, recorded once at the pinned commit and checkpoint (no model needed here)."""

    def test_fixture_is_pinned_to_the_same_code_and_checkpoint(self):
        ref = REFERENCE["reference"]
        assert ref["official_commit"] == OFFICIAL_COMMIT and ref["checkpoint_revision"] == FLAN.revision
        assert (ref["chunk_size"], ref["max_model_len"], ref["batch_size"]) == (500, 2048, 16)
        assert ref["label_tokens_ids"] == [3, 209] and ref["tolerance"] == 1e-5
        assert "/Users" not in REFERENCE_FIXTURE.read_text(encoding="utf-8")

    def test_fixture_covers_multi_chunk_documents_and_truncation(self):
        groups = {p["group"] for p in REFERENCE["pairs"]}
        assert {"short", "newlines", "three_chunks", "five_chunks", "run_on_truncated"} <= groups
        assert max(len(p["official_chunks"]) for p in REFERENCE["pairs"]) >= 5
        assert any(p["prompt_tokens"] > 2048 for p in REFERENCE["pairs"] if p["group"] == "run_on_truncated") or any(
            p["context_truncated"] for p in REFERENCE["pairs"]
        )

    def test_our_chunking_reproduces_the_official_chunks_with_nltk(self):
        """Same chunks as recorded, given nltk's sentence splitter."""
        nltk = pytest.importorskip("nltk")
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            pytest.skip("nltk punkt_tab not installed")
        from nltk.tokenize import sent_tokenize

        for pair in REFERENCE["pairs"]:
            assert document_chunks(pair["document"], FLAN, sent_tokenize, lambda _s: 0) == pair["official_chunks"], (
                pair["group"]
            )


class TestCli:
    def test_judge_bench_builds_an_answer_level_minicheck_on_the_given_device(self, monkeypatch: pytest.MonkeyPatch):
        from ragnarok_ai.cli import main as cli

        built: list[dict[str, object]] = []

        class Recorded:
            def __init__(self, **kwargs):
                built.append(kwargs)

        monkeypatch.setattr(cli, "_require_optional", lambda *_a, **_k: None)
        monkeypatch.setattr(minicheck_module, "MiniCheckJudge", Recorded)
        cli._load_bench_judge("minicheck", ollama_url="http://x", prompt_format="auto", device="mps")
        cli._load_bench_judge("minicheck:roberta-large", ollama_url="http://x", prompt_format="auto")
        assert built == [
            {"checkpoint": "flan-t5-large", "unit": "answer", "device": "mps"},
            {"checkpoint": "roberta-large", "unit": "answer", "device": "cpu"},
        ]


# =============================================================================
# Integration: the real checkpoint. Run by hand:
#   uv run --extra minicheck pytest tests/unit/judges/test_minicheck.py -m integration -v
# Downloads about 3 GB (MiniCheck-Flan-T5-Large) plus nltk's punkt_tab on first run.
# =============================================================================


@pytest.fixture(scope="module")
def real_judge() -> MiniCheckJudge:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("nltk")
    return MiniCheckJudge(unit="answer", device="cpu")


@pytest.mark.integration
class TestRealModel:
    def test_digest_label_tokens_and_splitter_identity(self, real_judge: MiniCheckJudge):
        import nltk

        assert real_judge.model_digest == f"{FLAN.repo}@{FLAN.revision}"
        backend = real_judge.backend
        assert isinstance(backend, TransformersBackend)
        assert (
            backend.tokenizer.convert_ids_to_tokens(list(FLAN_LABEL_TOKEN_IDS))
            == REFERENCE["reference"]["label_tokens"]
        )
        splitter = dict(real_judge.config("faithfulness").judge_specific)["sentence_splitter"]
        assert re.fullmatch(rf"nltk-{re.escape(nltk.__version__)}:punkt_tab-english:[0-9a-f]{{64}}", splitter)
        assert splitter.endswith(punkt_digest(str(nltk.data.find("tokenizers/punkt_tab/english/"))))

    @pytest.mark.asyncio
    async def test_matches_the_official_package_within_tolerance(self, real_judge: MiniCheckJudge):
        """Same chunks, same probability per chunk and same maximum as the official code, 1e-5."""
        tolerance = REFERENCE["reference"]["tolerance"]
        worst = 0.0
        for pair in REFERENCE["pairs"]:
            assert real_judge.chunks(pair["document"]) == pair["official_chunks"], pair["group"]
            result = await real_judge.judge("faithfulness", question="", context=pair["document"], answer=pair["claim"])
            gap = abs(result.score - pair["official_max_support_prob"])
            worst = max(worst, gap)
            assert gap <= tolerance, (pair["group"], pair["claim"], gap)
            assert result.trace is not None
            assert result.trace.prompt_tokens == pair["prompt_tokens"]
            assert result.trace.context_truncated == pair["context_truncated"]
            assert result.verdict == ("PASS" if pair["official_max_support_prob"] >= 0.5 else "FAIL")
        assert worst <= tolerance

    @pytest.mark.asyncio
    async def test_download_then_offline_load_gives_the_same_score(self, real_judge: MiniCheckJudge, tmp_path: Path):
        artifacts = MiniCheckJudge.download(tmp_path / "bundle")
        assert artifacts.digest == real_judge.model_digest and artifacts.nltk_dir is not None
        local = MiniCheckJudge(model_path=tmp_path / "bundle", unit="answer", device="cpu", offline=True)
        pair = next(p for p in REFERENCE["pairs"] if p["group"] == "three_chunks")
        result = await local.judge("faithfulness", question="", context=pair["document"], answer=pair["claim"])
        assert result.score == pytest.approx(pair["official_max_support_prob"], abs=1e-5)
