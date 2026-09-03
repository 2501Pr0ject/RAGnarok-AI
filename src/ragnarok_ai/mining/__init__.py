"""Mine production traffic into evaluation test sets.

Build test sets from the questions users actually ask — the frequent
ones, the failing ones, the slow ones — instead of only synthetic
questions. Requires opt-in query capture on the monitor client.
"""

from ragnarok_ai.mining.miner import TestsetMiner

__all__ = [
    "TestsetMiner",
]
