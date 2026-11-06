"""HHEMJudge: contract, prompt, truncation, sentence unit, artifacts, dependencies, CLI; integration by hand."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ragnarok_ai.core import JudgeProtocol
from ragnarok_ai.core.exceptions import ConfigurationError, JudgeDependencyError, NoUnitsError
from ragnarok_ai.judges import AggregationPolicy, BinaryProbability, HHEMJudge, ScoreThreshold
from ragnarok_ai.judges import hhem as hhem_module
from ragnarok_ai.judges.hhem import (
    FOUNDATION_FILES,
    FOUNDATION_REPO,
    FOUNDATION_REVISION,
    HHEM_FILES,
    HHEM_PROMPT,
    HHEM_PROMPT_SHA256,
    HHEM_REPO,
    HHEM_REVISION,
    PASS_THRESHOLD,
    HHEMArtifacts,
    TransformersBackend,
    resolve_artifacts,
)
from ragnarok_ai.text import SEGMENTER_ID


class FakeBackend:
    """Whitespace tokens; P(consistent) read from a '(p=0.x)' marker in the hypothesis, 0.9 by default."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])

    def consistency(self, prompts):
        self.prompts.extend(prompts)
        scores = []
        for prompt in prompts:
            hypothesis = prompt.split("Hypothesis: ", 1)[1]
            found = re.search(r"\(p=([0-9.]+)\)", hypothesis)
            scores.append(float(found.group(1)) if found else 0.9)
        return scores


def make_judge(**kwargs) -> tuple[HHEMJudge, FakeBackend]:
    backend = FakeBackend()
    return HHEMJudge(backend=backend, **kwargs), backend


CONTEXT = "The policy allows refunds within 30 days of purchase."


class TestContract:
    def test_is_a_judge_with_two_criteria_and_a_binary_policy(self):
        judge, _ = make_judge()
        assert isinstance(judge, JudgeProtocol)
        assert judge.criteria == frozenset({"faithfulness", "hallucination"})
        assert ScoreThreshold(pass_threshold=0.5) == judge.VERDICT_POLICY
        assert PASS_THRESHOLD == 0.5

    def test_config_records_both_revisions_device_prompt_and_uncalibrated_threshold(self):
        judge, _ = make_judge(unit="answer", device="mps", model_digest=f"{HHEM_REPO}@abc;{FOUNDATION_REPO}@def")
        config = judge.config("faithfulness")
        assert config.judge_class == "ragnarok_ai.judges.hhem.HHEMJudge"
        assert (config.judge_kind, config.backend, config.model) == ("classifier", "transformers", HHEM_REPO)
        assert config.model_digest == f"{HHEM_REPO}@abc;{FOUNDATION_REPO}@def"
        assert config.options == (("device", "mps"), ("dtype", "float32"), ("max_input_tokens", None))
        assert config.prompt_format is None and config.prompt_template_sha256 == HHEM_PROMPT_SHA256
        assert config.unit == "answer" and config.aggregation is None and config.segmenter is None
        assert dict(config.judge_specific) == {
            "foundation": FOUNDATION_REPO,
            "question_used": False,
            "threshold_calibrated": False,
        }
        assert config.verdict_policy.pass_threshold_on_score() == 0.5
        other, _ = make_judge(unit="answer", device="cpu")
        assert not config.comparable_with(other.config("faithfulness"))  # the device is part of the run
        assert config.to_dict()["options"] == [["device", "mps"], ["dtype", "float32"], ["max_input_tokens", None]]

    def test_default_digest_without_artifacts(self):
        judge, _ = make_judge()
        assert judge.model_digest == "injected-backend" and judge.artifacts is None

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"device": "tpu"}, "device must be one of"),
            ({"unit": "claim"}, "unit must be 'sentence' or 'answer'"),
            ({"max_input_tokens": 0}, "max_input_tokens must be at least 1"),
        ],
    )
    def test_invalid_parameters(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            make_judge(**kwargs)

    def test_cuda_index_is_a_valid_device_name(self):
        judge, _ = make_judge(device="cuda:1")
        assert judge.config("faithfulness").options[0] == ("device", "cuda:1")

    @pytest.mark.asyncio
    async def test_unknown_criterion(self):
        judge, _ = make_judge()
        with pytest.raises(ValueError, match="Unknown criterion 'relevance'"):
            judge.config("relevance")
        with pytest.raises(ValueError, match="Unknown criterion 'relevance'"):
            await judge.judge("relevance", question="q", context=CONTEXT, answer="Refunds take 30 days.")


class TestAnswerLevel:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("marker", "probability", "verdict"),
        [("(p=0.91)", 0.91, "PASS"), ("(p=0.5)", 0.5, "PASS"), ("(p=0.49)", 0.49, "FAIL"), ("(p=0.02)", 0.02, "FAIL")],
    )
    async def test_probability_is_the_score_and_the_verdict_follows_the_threshold(self, marker, probability, verdict):
        judge, _ = make_judge(unit="answer")
        answer = f"Refunds are possible within a month {marker}."
        result = await judge.judge("faithfulness", question="Can I get a refund?", context=CONTEXT, answer=answer)

        assert (result.verdict, result.score, result.raw_score) == (verdict, probability, None)
        assert result.output == BinaryProbability(probability)
        assert result.criteria == "faithfulness"
        assert "not calibrated" in result.explanation and f"{probability:.3f}" in result.explanation
        assert result.config == judge.config("faithfulness")
        assert result.units is None

    @pytest.mark.asyncio
    async def test_prompt_is_the_training_template_without_the_question(self):
        judge, backend = make_judge(unit="answer")
        answer = "Refunds are possible within a month."
        await judge.judge("hallucination", question="Can I get a refund?", context=CONTEXT, answer=answer)

        assert backend.prompts == [HHEM_PROMPT.format(text1=CONTEXT, text2=answer)]
        assert backend.prompts[0].startswith("<pad> Determine if the hypothesis is true given the premise?")
        assert "Can I get a refund?" not in backend.prompts[0]

    @pytest.mark.asyncio
    async def test_trace_counts_the_tokens_sent_and_reports_no_truncation_by_default(self):
        judge, backend = make_judge(unit="answer")
        answer = "Refunds are possible within a month."
        result = await judge.judge("faithfulness", question="q", context=CONTEXT * 50, answer=answer)

        assert result.trace is not None
        assert result.trace.prompt_tokens == backend.count_tokens(backend.prompts[0])
        assert result.trace.prompt_tokens > 400  # well beyond the 512-token training length is sent as is
        assert result.trace.context_truncated is False
        assert result.trace.completion_tokens is None and result.trace.latency_ms is not None

    @pytest.mark.asyncio
    async def test_max_input_tokens_cuts_the_context_and_keeps_the_hypothesis(self):
        judge, backend = make_judge(unit="answer", max_input_tokens=20)
        context = " ".join(f"fact{i}" for i in range(100))
        answer = "Refunds are possible within a month (p=0.3)."
        result = await judge.judge("faithfulness", question="q", context=context, answer=answer)

        prompt = backend.prompts[0]
        assert result.trace is not None and result.trace.context_truncated is True
        assert result.trace.prompt_tokens <= 20 and result.trace.prompt_tokens == backend.count_tokens(prompt)
        assert prompt.endswith(f"Hypothesis: {answer}")  # the hypothesis is whole
        assert "fact0" in prompt and "fact99" not in prompt  # the premise lost its tail
        assert "truncated to fit 20 tokens" in result.explanation
        assert result.config is not None and dict(result.config.options)["max_input_tokens"] == 20
        assert (result.verdict, result.score) == ("FAIL", 0.3)

    @pytest.mark.asyncio
    async def test_short_input_is_not_truncated_even_with_a_limit(self):
        judge, _ = make_judge(unit="answer", max_input_tokens=200)
        result = await judge.judge("faithfulness", question="q", context=CONTEXT, answer="Refunds within a month.")
        assert result.trace is not None and result.trace.context_truncated is False

    @pytest.mark.asyncio
    async def test_hypothesis_longer_than_the_limit_leaves_an_empty_premise(self):
        judge, backend = make_judge(unit="answer", max_input_tokens=5)
        answer = " ".join(["word"] * 10)
        result = await judge.judge("faithfulness", question="q", context=CONTEXT, answer=answer)
        assert backend.prompts[0] == HHEM_PROMPT.format(text1="", text2=answer)
        assert result.trace is not None and result.trace.context_truncated is True


class TestSentenceLevelDefault:
    ANSWER = "Refunds are possible within a month. The policy covers gifts too (p=0.2). Sure!"

    @pytest.mark.asyncio
    async def test_each_sentence_is_a_hypothesis_aggregated_as_fraction_pass_and_worst(self):
        judge, backend = make_judge()
        result = await judge.judge("faithfulness", question="q", context=CONTEXT, answer=self.ANSWER)

        assert [p.split("Hypothesis: ", 1)[1] for p in backend.prompts] == [
            "Refunds are possible within a month.",
            "The policy covers gifts too (p=0.2).",
        ]
        assert all(p.startswith(HHEM_PROMPT.format(text1=CONTEXT, text2="")) for p in backend.prompts)
        assert result.units is not None and [u.skipped for u in result.units] == [None, None, "short_fragment"]
        assert [u.verdict for u in result.units[:2]] == ["PASS", "FAIL"]
        assert result.units[1].output == BinaryProbability(0.2)
        assert (result.score, result.verdict) == (0.5, "FAIL")
        assert result.raw_score is None and result.output is None and result.trace is None

    @pytest.mark.asyncio
    async def test_config_is_at_sentence_unit_under_the_hhem_class_name(self):
        judge, _ = make_judge()
        config = judge.config("faithfulness")
        assert config.judge_class == "ragnarok_ai.judges.hhem.HHEMJudge"  # not SentenceLevel[...]
        assert config.unit == "sentence"
        assert config.aggregation == AggregationPolicy("fraction_pass", "worst")
        assert config.segmenter == SEGMENTER_ID
        result = await judge.judge("faithfulness", question="q", context=CONTEXT, answer=self.ANSWER)
        assert result.config == config
        answer_level, _ = make_judge(unit="answer")
        assert not config.comparable_with(answer_level.config("faithfulness"))

    @pytest.mark.asyncio
    async def test_other_aggregation_and_no_units(self):
        judge, _ = make_judge(aggregation=AggregationPolicy("min", "worst"))
        result = await judge.judge("faithfulness", question="q", context=CONTEXT, answer=self.ANSWER)
        assert (result.score, result.verdict) == (0.2, "FAIL")
        with pytest.raises(NoUnitsError):
            await judge.judge("faithfulness", question="q", context=CONTEXT, answer="Sure!")


class TestArtifacts:
    @staticmethod
    def local_folder(tmp_path: Path, *, revisions: bool = True) -> Path:
        root = tmp_path / "hhem"
        (root / "foundation").mkdir(parents=True)
        (root / "model.safetensors").write_bytes(b"weights")
        (root / "foundation" / "tokenizer.json").write_text("{}")
        if revisions:
            (root / "REVISIONS.json").write_text(
                json.dumps({"hhem_revision": HHEM_REVISION, "foundation_revision": FOUNDATION_REVISION})
            )
        return root

    def test_local_folder_with_revisions(self, tmp_path: Path):
        root = self.local_folder(tmp_path)
        artifacts = resolve_artifacts(root)
        assert artifacts == HHEMArtifacts(
            hhem_dir=root,
            foundation_dir=root / "foundation",
            hhem_revision=HHEM_REVISION,
            foundation_revision=FOUNDATION_REVISION,
            source="local",
            digest=f"{HHEM_REPO}@{HHEM_REVISION};{FOUNDATION_REPO}@{FOUNDATION_REVISION}",
        )

    def test_local_folder_without_revisions_hashes_the_weights(self, tmp_path: Path):
        root = self.local_folder(tmp_path, revisions=False)
        artifacts = resolve_artifacts(root)
        weights_sha = hashlib.sha256(b"weights").hexdigest()
        assert artifacts.hhem_revision is None and artifacts.foundation_revision is None
        assert artifacts.digest == f"{HHEM_REPO}@sha256:{weights_sha};{FOUNDATION_REPO}@unknown"

    def test_incomplete_local_folder(self, tmp_path: Path):
        with pytest.raises(ConfigurationError, match=r"must hold model\.safetensors"):
            resolve_artifacts(tmp_path)

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

    def test_hub_cache_is_used_before_any_download(self, fake_hub):
        calls, cached = fake_hub
        cached.update({HHEM_REPO, FOUNDATION_REPO})
        artifacts = resolve_artifacts()
        assert [c["local_only"] for c in calls] == [True, True]  # no download attempted
        assert artifacts.source == "hub" and artifacts.hhem_revision == HHEM_REVISION
        assert artifacts.digest == f"{HHEM_REPO}@{HHEM_REVISION};{FOUNDATION_REPO}@{FOUNDATION_REVISION}"

    def test_download_is_pinned_and_limited_to_the_needed_files(self, fake_hub):
        calls, _ = fake_hub
        resolve_artifacts()
        downloads = [c for c in calls if not c["local_only"]]
        assert [(c["repo"], c["revision"], c["patterns"]) for c in downloads] == [
            (HHEM_REPO, HHEM_REVISION, HHEM_FILES),
            (FOUNDATION_REPO, FOUNDATION_REVISION, FOUNDATION_FILES),
        ]
        assert "model.safetensors" in HHEM_FILES and "tokenizer.json" in FOUNDATION_FILES
        assert not any(name.endswith((".bin", ".h5", ".msgpack")) for name in HHEM_FILES + FOUNDATION_FILES)

    def test_offline_refuses_to_download(self, fake_hub):
        calls, _ = fake_hub
        with pytest.raises(ConfigurationError, match="offline=True"):
            resolve_artifacts(offline=True)
        assert all(c["local_only"] for c in calls)

    def test_download_writes_a_self_contained_folder(self, fake_hub, tmp_path: Path):
        calls, _ = fake_hub
        root = tmp_path / "bundle"
        (root / "foundation").mkdir(parents=True)  # what the fake hub would have written
        (root / "model.safetensors").write_bytes(b"w")
        (root / "foundation" / "tokenizer.json").write_text("{}")
        artifacts = HHEMJudge.download(root)
        assert [c["local_dir"] for c in calls] == [str(root), str(root / "foundation")]
        assert json.loads((root / "REVISIONS.json").read_text())["hhem_revision"] == HHEM_REVISION
        assert artifacts.source == "local" and artifacts.digest.endswith(FOUNDATION_REVISION)

    def test_missing_huggingface_hub_names_the_extra(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(JudgeDependencyError, match=r"HHEMJudge needs 'huggingface_hub'.*\[hhem\]") as excinfo:
            resolve_artifacts()
        assert "llm=" not in str(excinfo.value)


class TestDependencies:
    def test_missing_torch_names_the_extra(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setitem(sys.modules, "torch", None)
        artifacts = HHEMArtifacts(tmp_path, tmp_path, None, None, "local", "x")
        with pytest.raises(JudgeDependencyError, match=r"needs 'torch'.*ragnarok-ai\[hhem\]"):
            TransformersBackend(artifacts, "cpu")

    def test_backend_validates_the_device_name_before_importing(self, tmp_path: Path):
        artifacts = HHEMArtifacts(tmp_path, tmp_path, None, None, "local", "x")
        with pytest.raises(ValueError, match="device must be one of"):
            TransformersBackend(artifacts, "gpu")


class TestPrompt:
    def test_template_is_the_published_one(self):
        assert HHEM_PROMPT == (
            "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
        )
        assert hashlib.sha256(HHEM_PROMPT.encode()).hexdigest() == HHEM_PROMPT_SHA256

    def test_pinned_revisions_are_full_commit_shas(self):
        assert re.fullmatch(r"[0-9a-f]{40}", HHEM_REVISION) and re.fullmatch(r"[0-9a-f]{40}", FOUNDATION_REVISION)
        assert hhem_module.TRAINING_LENGTH == 512


class TestCli:
    def test_judge_bench_builds_an_answer_level_hhem_on_the_given_device(self, monkeypatch: pytest.MonkeyPatch):
        from ragnarok_ai.cli import main as cli

        built: list[dict[str, object]] = []

        class Recorded:
            def __init__(self, **kwargs):
                built.append(kwargs)

        monkeypatch.setattr(cli, "_require_optional", lambda *_a, **_k: None)
        monkeypatch.setattr(hhem_module, "HHEMJudge", Recorded)
        cli._load_bench_judge("hhem", ollama_url="http://x", prompt_format="auto", device="mps")
        assert built == [{"unit": "answer", "device": "mps"}]
        with pytest.raises(ValueError, match="takes no model name"):
            cli._load_bench_judge("hhem:large", ollama_url="http://x", prompt_format="auto")

    def test_judge_bench_without_the_extra(self, tmp_path: Path):
        from typer.testing import CliRunner

        from ragnarok_ai.cli.main import app

        if importlib.util.find_spec("transformers") is not None:
            pytest.skip("transformers installed: the install hint cannot be observed")
        data = tmp_path / "d.jsonl"
        data.write_text(json.dumps({"id": "a", "question": "q", "context": "c", "answer": "x", "label": "PASS"}) + "\n")
        result = CliRunner().invoke(
            app, ["judge-bench", "--judge", "hhem", "--data", str(data), "--out", str(tmp_path / "o")]
        )
        assert result.exit_code == 1 and "[hhem]" in result.output


# =============================================================================
# Integration: the real model. Run by hand:
#   uv run --extra hhem pytest tests/unit/judges/test_hhem.py -m integration -v
# Downloads about 1 GB into the Hugging Face cache on first run.
# =============================================================================

# The seven example pairs and expected scores of the HHEM-2.1-Open model card
# (vectara/hallucination_evaluation_model, README.md at HHEM_REVISION), copyright
# Vectara, Apache License 2.0; reproduced to check this adapter against the
# published behavior.
README_PAIRS = [
    ("The capital of France is Berlin.", "The capital of France is Paris.", 0.0111),
    ("I am in California", "I am in United States.", 0.6474),
    ("I am in United States", "I am in California.", 0.1290),
    ("A person on a horse jumps over a broken down airplane.", "A person is outdoors, on a horse.", 0.8969),
    (
        "A boy is jumping on skateboard in the middle of a red bridge.",
        "The boy skates down the sidewalk on a red bridge",
        0.1846,
    ),
    (
        "A man with blond-hair, and a brown shirt drinking out of a public water fountain.",
        "A blond man wearing a brown shirt is reading a book.",
        0.0050,
    ),
    ("Mark Wahlberg was a fan of Manny.", "Manny was a fan of Mark Wahlberg.", 0.0543),
]


@pytest.fixture(scope="module")
def real_judge() -> HHEMJudge:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return HHEMJudge(unit="answer", device="cpu")


@pytest.mark.integration
class TestRealModel:
    @pytest.mark.asyncio
    async def test_matches_the_model_card_examples(self, real_judge: HHEMJudge):
        """Same probabilities as the published remote code, within 0.01."""
        for premise, hypothesis, expected in README_PAIRS:
            result = await real_judge.judge("faithfulness", question="", context=premise, answer=hypothesis)
            assert result.score == pytest.approx(expected, abs=0.01), (premise, hypothesis)
            assert result.verdict == ("PASS" if expected >= 0.5 else "FAIL")

    def test_digest_carries_both_pinned_revisions(self, real_judge: HHEMJudge):
        assert real_judge.model_digest == f"{HHEM_REPO}@{HHEM_REVISION};{FOUNDATION_REPO}@{FOUNDATION_REVISION}"
        assert real_judge.artifacts is not None and real_judge.artifacts.source == "hub"

    def test_prompt_matches_the_pinned_remote_configuration(self):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(HHEM_REPO, "configuration_hhem_v2.py", revision=HHEM_REVISION)
        source = Path(path).read_text(encoding="utf-8")
        found = re.search(r'prompt = "(.*)"', source)
        assert found is not None and found.group(1).encode().decode("unicode_escape") == HHEM_PROMPT

    @pytest.mark.asyncio
    async def test_inputs_beyond_the_training_length_run_untruncated(self, real_judge: HHEMJudge):
        """No hard window: a 1500-token premise is sent whole, the flag says so, the output stays a probability."""
        premise = " ".join(f"Sentence number {i} states a fact about item {i}." for i in range(250))
        result = await real_judge.judge("faithfulness", question="", context=premise, answer="Item 3 is mentioned.")
        assert result.trace is not None
        assert result.trace.prompt_tokens > 1500 and result.trace.context_truncated is False
        assert 0.0 <= result.score <= 1.0

        limited = HHEMJudge(unit="answer", device="cpu", max_input_tokens=512, backend=real_judge.backend)
        cut = await limited.judge("faithfulness", question="", context=premise, answer="Item 3 is mentioned.")
        assert cut.trace is not None
        assert cut.trace.context_truncated is True and cut.trace.prompt_tokens <= 512

    @pytest.mark.asyncio
    async def test_download_then_offline_load_gives_the_same_score(self, real_judge: HHEMJudge, tmp_path: Path):
        artifacts = HHEMJudge.download(tmp_path / "bundle")
        assert artifacts.digest == real_judge.model_digest
        local = HHEMJudge(model_path=tmp_path / "bundle", unit="answer", device="cpu", offline=True)
        premise, hypothesis, expected = README_PAIRS[3]
        result = await local.judge("faithfulness", question="", context=premise, answer=hypothesis)
        assert result.score == pytest.approx(expected, abs=0.01)
        assert local.model_digest == real_judge.model_digest
