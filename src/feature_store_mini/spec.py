"""FeatureSpec + FeatureValue — the contract every store works against."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FeatureSpec:
    """Describes one feature.

    ``dtype`` is a Python type (``int``, ``float``, ``str``, ``bool``). Anything
    else is stored as-is but won't pass the parity-checker casting rule.

    ``freshness_window_secs`` is how stale the *online* value is allowed to be
    relative to ``now()``. The parity checker flags features whose online value
    is older than this window.

    ``ttl_secs`` (optional) instructs in-memory and Redis stores to evict values
    past their TTL on read. ``None`` means no auto-eviction.
    """

    name: str
    dtype: type
    freshness_window_secs: float = 3600.0
    ttl_secs: float | None = None
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class FeatureValue:
    """One value for one (feature, entity) pair, with the event timestamp."""

    value: Any
    event_time: float  # unix seconds when the value was *valid* (not when stored)
    stored_at: float = field(default_factory=lambda: time.time())

    def is_stale(self, spec: FeatureSpec, now: float | None = None) -> bool:
        """True iff this value is older than the freshness window."""
        ref = now if now is not None else time.time()
        return (ref - self.event_time) > spec.freshness_window_secs

    def is_expired(self, spec: FeatureSpec, now: float | None = None) -> bool:
        """True iff TTL is set and the value has been stored longer than TTL."""
        if spec.ttl_secs is None:
            return False
        ref = now if now is not None else time.time()
        return (ref - self.stored_at) > spec.ttl_secs
