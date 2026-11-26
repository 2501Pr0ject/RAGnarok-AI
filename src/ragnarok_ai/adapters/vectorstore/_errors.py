"""Which errors of a vector store client are connection failures.

A ``VectorStoreConnectionError`` says that the store could not be reached.
An adapter must not use it for anything else: a response of the server
(collection not found, authentication refused), a missing package or a
mistake in the adapter's own call would then send the user to check a
server that is not the cause. That is how two broken calls stayed hidden
(#190, #191): an ``AttributeError`` and a ``TypeError`` were reported as
"Failed to search" and "Failed to connect", and ``is_available()``
returned ``False``.

Used by ``QdrantVectorStore`` and ``WeaviateVectorStore``. The other vector
store adapters still wrap every exception.
"""

from __future__ import annotations

import asyncio
import sys


def is_connection_failure(exc: BaseException, exceptions_module: str, names: tuple[str, ...]) -> bool:
    """Whether *exc* means that the store could not be reached.

    True for an ``OSError`` (connection refused or reset, name not
    resolved, socket timeout), an asyncio timeout, and the client
    library's own connection errors: the classes named *names* in the
    module *exceptions_module*. The module is read from ``sys.modules``,
    not imported: an error raised by the library implies it is loaded. A
    name the installed version does not have is skipped, so the list can
    name classes added by later versions.
    """
    if isinstance(exc, (OSError, asyncio.TimeoutError)):
        return True
    module = sys.modules.get(exceptions_module)
    for name in names:
        candidate = getattr(module, name, None)
        if isinstance(candidate, type) and isinstance(exc, candidate):
            return True
    return False
