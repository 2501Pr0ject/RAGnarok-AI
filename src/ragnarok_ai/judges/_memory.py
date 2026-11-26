"""Out-of-memory errors of a local judge's device, turned into per-sample failures.

A forward pass that does not fit on the device concerns one input, not the
run: the local judges raise a ``JudgeMemoryError`` for it (an
``EvaluationError``), which ``judge_many`` records as a failure of kind
``memory``. Any other ``RuntimeError`` keeps propagating.

The two devices report it differently. CUDA raises a dedicated class,
``torch.cuda.OutOfMemoryError``. MPS raises a plain ``RuntimeError`` whose
message starts with "MPS backend out of memory" (observed with torch
2.14.1); it has no class of its own, so the message is the only test.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, TypeVar

from ragnarok_ai.core.exceptions import JudgeMemoryError

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")

# The device's own message, cut to this length in the failure (it holds sizes, never input text)
_DEVICE_MESSAGE_LIMIT = 300
_MPS_OUT_OF_MEMORY = "mps backend out of memory"
# MPS ends its message with the advice to lift the allocator's limit ("may cause system failure"): not kept
_MPS_ADVICE = " Use PYTORCH_MPS_HIGH_WATERMARK_RATIO"


def is_out_of_memory(exc: BaseException) -> bool:
    """Whether *exc* is the device running out of memory.

    CUDA is recognized by its class, ``torch.cuda.OutOfMemoryError``, never
    by the message. ``torch`` is read from ``sys.modules``, not imported:
    an error raised by the device implies it is loaded. MPS is recognized
    by the message of its ``RuntimeError``, the only thing that tells it
    apart.
    """
    torch = sys.modules.get("torch")
    cuda_error = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if isinstance(cuda_error, type) and isinstance(exc, cuda_error):
        return True
    return isinstance(exc, RuntimeError) and _MPS_OUT_OF_MEMORY in str(exc).lower()


def release_device_cache(torch: Any, device: str) -> None:
    """Give the allocator's cached blocks back to the device (``mps`` or ``cuda``); nothing to do on ``cpu``."""
    kind = device.split(":", 1)[0]
    if kind == "mps":
        torch.mps.empty_cache()
    elif kind == "cuda":
        torch.cuda.empty_cache()


def guard_device_memory(
    call: Callable[[], T], *, backend: object, judge: str, device: str, input_size: Callable[[], str]
) -> T:
    """Run *call* (the forward pass); an out-of-memory error becomes a ``JudgeMemoryError``.

    The backend's cache is released through its ``_release_cache()`` method
    when it has one (an injected backend need not), so the next input
    starts from a clean allocator. The
    message names the device and the input's size in tokens (*input_size*,
    called only after a failure), never the input's text.

    Raises:
        JudgeMemoryError: The device ran out of memory during *call*.
    """
    try:
        return call()
    except RuntimeError as exc:
        if not is_out_of_memory(exc):
            raise
        lines = str(exc).strip().splitlines()
        device_message = (lines[0] if lines else type(exc).__name__).split(_MPS_ADVICE, 1)[0]
        device_message = device_message[:_DEVICE_MESSAGE_LIMIT]
    # Outside the handler: the traceback, which kept the forward pass's tensors alive, is gone
    release = getattr(backend, "_release_cache", None)
    if callable(release):
        release()
    msg = f"{judge}: out of memory on device {device!r} for {input_size()}. Device message: {device_message}"
    raise JudgeMemoryError(msg)
