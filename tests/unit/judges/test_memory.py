"""Device out-of-memory errors of the local judges: a per-sample failure, never the end of the run."""

from __future__ import annotations

import json
import re
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from ragnarok_ai.core.exceptions import EvaluationError, JudgeMemoryError
from ragnarok_ai.judges import HHEMJudge, MiniCheckJudge
from ragnarok_ai.judges._memory import guard_device_memory, is_out_of_memory, release_device_cache
from ragnarok_ai.judges.runner import judge_many

if TYPE_CHECKING:
    from pathlib import Path

# The message of a real run (torch 2.14.1, HHEM on MPS, 13129 tokens): one line, a plain RuntimeError
MPS_MESSAGE = (
    "MPS backend out of memory (MPS allocated: 1.01 GiB, other allocations: 640.00 KiB, max allowed: 2.01 GiB). "
    "Tried to allocate 1.31 GiB on shared pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit "
    "for memory allocations (may cause system failure)."
)
CUDA_MESSAGE = "CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 7.79 GiB"
SECRET = "Zanzibar"  # stands for the input's text: must never reach a failure message


class OutOfMemoryError(RuntimeError):
    """Stands for ``torch.cuda.OutOfMemoryError``, a ``RuntimeError`` subclass."""


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A loaded ``torch`` whose CUDA error class is ``OutOfMemoryError``, with or without the real package."""
    torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OutOfMemoryError))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


class HHEMBackend:
    """Whitespace tokens, P(consistent) 0.9; '(oom)' in the prompt raises the MPS error, '(crash)' another one."""

    def __init__(self) -> None:
        self.released = 0

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])

    def consistency(self, prompts):
        if any("(oom)" in p for p in prompts):
            raise RuntimeError(MPS_MESSAGE)
        if any("(crash)" in p for p in prompts):
            raise RuntimeError("shape mismatch in the forward pass")
        return [0.9 for _ in prompts]

    def _release_cache(self) -> None:
        self.released += 1


class MiniCheckBackend:
    """Whitespace tokens, P(supported) 0.8; '(oom)' in the claim raises the CUDA error."""

    def __init__(self) -> None:
        self.released = 0

    def split_sentences(self, text: str) -> list[str]:
        return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]

    def splitter_id(self) -> str:
        return "naive-regex:v1"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def support(self, chunks, claim):
        if "(oom)" in claim:
            raise OutOfMemoryError(CUDA_MESSAGE)
        return [0.8 for _ in chunks], sum(len(c.split()) + len(claim.split()) for c in chunks), False

    def _release_cache(self) -> None:
        self.released += 1


def samples(marker: str = "(oom)") -> list[SimpleNamespace]:
    context = f"The policy allows refunds within 30 days. {SECRET} is mentioned here."
    answers = [
        "Refunds are allowed.",
        f"{SECRET} refunds last 30 days {marker}.",
        "Refunds exist for every customer of the shop.",
    ]
    return [SimpleNamespace(question="Q?", context=context, answer=a, sample_id=f"s{i}") for i, a in enumerate(answers)]


class TestDetection:
    def test_mps_is_recognized_by_its_message(self):
        assert is_out_of_memory(RuntimeError(MPS_MESSAGE)) is True
        assert is_out_of_memory(RuntimeError("shape mismatch")) is False
        assert is_out_of_memory(ValueError(MPS_MESSAGE)) is False

    @pytest.mark.usefixtures("fake_torch")
    def test_cuda_is_recognized_by_its_class_not_by_its_message(self):
        assert is_out_of_memory(OutOfMemoryError(CUDA_MESSAGE)) is True
        assert is_out_of_memory(OutOfMemoryError("")) is True
        assert is_out_of_memory(RuntimeError(CUDA_MESSAGE)) is False  # the text alone is not the device's error

    def test_without_torch_only_the_mps_message_counts(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        assert is_out_of_memory(OutOfMemoryError(CUDA_MESSAGE)) is False
        assert is_out_of_memory(RuntimeError(MPS_MESSAGE)) is True

    def test_release_calls_the_cache_of_the_device_kind(self):
        calls: list[str] = []
        torch = SimpleNamespace(
            mps=SimpleNamespace(empty_cache=lambda: calls.append("mps")),
            cuda=SimpleNamespace(empty_cache=lambda: calls.append("cuda")),
        )
        for device in ("cpu", "mps", "cuda", "cuda:1"):
            release_device_cache(torch, device)
        assert calls == ["mps", "cuda", "cuda"]

    def test_guard_works_with_a_backend_that_has_no_cache_to_release(self):
        def call():
            raise RuntimeError(MPS_MESSAGE)

        with pytest.raises(JudgeMemoryError, match=r"J: out of memory on device 'mps' for 3 tokens\. Device message"):
            guard_device_memory(call, backend=object(), judge="J", device="mps", input_size=lambda: "3 tokens")
        # still an EvaluationError: code written for 1.13.x, which catches that, keeps working
        assert issubclass(JudgeMemoryError, EvaluationError)
        assert guard_device_memory(lambda: 7, backend=object(), judge="J", device="mps", input_size=lambda: "") == 7


class TestHHEM:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("unit", ["answer", "sentence"])
    async def test_one_sample_out_of_three_fails_and_the_run_completes(self, unit):
        backend = HHEMBackend()
        judge = HHEMJudge(backend=backend, device="mps", unit=unit)
        attempts = await judge_many(judge, "faithfulness", samples())
        assert [a.result is not None for a in attempts] == [True, False, True]
        failure = attempts[1].error
        assert failure is not None and failure.kind == "memory"
        # the prompt of the failing call: the whole answer (29 whitespace tokens) or its one sentence, the same here
        assert "HHEMJudge: out of memory on device 'mps' for an input of 29 tokens." in failure.message
        assert "MPS allocated: 1.01 GiB" in failure.message and "Tried to allocate 1.31 GiB" in failure.message
        assert "PYTORCH_MPS" not in failure.message  # the advice to lift the limit is not kept
        assert SECRET not in failure.message
        assert backend.released == 1

    @pytest.mark.asyncio
    async def test_injected_backend_without_a_cache_to_release(self):
        class Bare:
            """The three methods of ``HHEMBackend`` and nothing else."""

            count_tokens = staticmethod(lambda text: len(text.split()))
            truncate = staticmethod(lambda text, max_tokens: " ".join(text.split()[:max_tokens]))

            def consistency(self, prompts):
                if any("(oom)" in p for p in prompts):
                    raise RuntimeError(MPS_MESSAGE)
                return [0.9 for _ in prompts]

        assert not hasattr(Bare(), "_release_cache")
        attempts = await judge_many(HHEMJudge(backend=Bare(), device="mps", unit="answer"), "faithfulness", samples())
        assert [a.result is not None for a in attempts] == [True, False, True]
        assert attempts[1].error is not None and attempts[1].error.kind == "memory"

    @pytest.mark.asyncio
    async def test_another_runtime_error_still_ends_the_run(self):
        backend = HHEMBackend()
        judge = HHEMJudge(backend=backend, device="mps", unit="answer")
        with pytest.raises(RuntimeError, match="shape mismatch"):
            await judge_many(judge, "faithfulness", samples("(crash)"))
        assert backend.released == 0


class TestMiniCheck:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("fake_torch")
    async def test_one_sample_out_of_three_fails_and_the_run_completes(self):
        backend = MiniCheckBackend()
        judge = MiniCheckJudge(backend=backend, device="cuda", unit="answer")
        attempts = await judge_many(judge, "faithfulness", samples())
        assert [a.result is not None for a in attempts] == [True, False, True]
        failure = attempts[1].error
        assert failure is not None and failure.kind == "memory"
        assert "MiniCheckJudge: out of memory on device 'cuda' for 1 chunk(s)" in failure.message
        assert "the longest input about 17 tokens" in failure.message  # 11 of the chunk, 6 of the claim
        assert "CUDA out of memory" in failure.message and SECRET not in failure.message
        assert backend.released == 1


# =============================================================================
# Integration: a real out-of-memory error on MPS. Run by hand on Apple silicon:
#   uv run --extra hhem pytest tests/unit/judges/test_memory.py -m integration -v
# The allocator's limit is lowered to the loaded model plus 1 GiB for the time
# of the test, so a 13000-token input cannot fit while a short one does.
# =============================================================================


@pytest.mark.integration
class TestRealMPS:
    @pytest.mark.asyncio
    async def test_out_of_memory_is_recorded_and_the_next_sample_passes(self, tmp_path: Path):
        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers")
        if not torch.backends.mps.is_available():
            pytest.skip("needs the MPS device")
        from ragnarok_ai.benchmarks.judges import parse_record, run_benchmark

        judge = HHEMJudge(unit="answer", device="mps")
        long_context = " ".join(f"Sentence number {i} states a fact about item {i}." for i in range(900))
        rows = [
            {
                "id": "short-1",
                "answer": "Item 3 is mentioned.",
                "context": "Item 3 is mentioned here.",
                "label": "PASS",
            },
            {"id": "long", "answer": "Item 3 is mentioned.", "context": long_context, "label": "PASS"},
            {"id": "short-2", "answer": "Item 9 is red.", "context": "Item 9 is blue.", "label": "FAIL"},
        ]
        records = [parse_record(row, i + 1) for i, row in enumerate(rows)]
        limit = (torch.mps.driver_allocated_memory() + 2**30) / torch.mps.recommended_max_memory()
        torch.mps.set_per_process_memory_fraction(limit)
        try:
            # What the device raises, unguarded: a plain RuntimeError that the detection recognizes
            prompt = f"Premise: {long_context}\n\nHypothesis: Item 3 is mentioned."
            with pytest.raises(RuntimeError, match="MPS backend out of memory") as raised:
                judge.backend.consistency([prompt])
            assert type(raised.value) is RuntimeError and is_out_of_memory(raised.value)
            print(f"\nDevice error: {raised.value}")
            del raised
            judge.backend._release_cache()  # type: ignore[attr-defined]

            run = await run_benchmark(judge, records, out_dir=tmp_path / "out", bootstrap_resamples=0)
        finally:
            torch.mps.set_per_process_memory_fraction(0.0)  # no limit for the rest of the process
        assert run.report["records"] == {"total": 3, "evaluated": 3, "scored": 2, "failed": 1}
        assert run.report["failures"] == {"memory": 1}
        written = {row["sample_id"]: row for row in map(json.loads, run.attempts_path.read_text().splitlines()[1:])}
        assert written["short-1"]["error"] is None and written["short-2"]["error"] is None  # the one after passes
        message = written["long"]["error"]["message"]
        print(f"Recorded failure: {message}")
        assert message.startswith("HHEMJudge: out of memory on device 'mps' for an input of ")
        assert "MPS backend out of memory" in message and "Sentence number" not in message
