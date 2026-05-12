"""Storage backends + the OnlineOfflineStore that wraps a pair of them."""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Iterable, Protocol, runtime_checkable

from feature_store_mini.spec import FeatureSpec, FeatureValue


@runtime_checkable
class StoreBackend(Protocol):
    """Tiny key/value backend with TTL awareness."""

    def get(self, key: str) -> FeatureValue | None: ...
    def set(self, key: str, value: FeatureValue) -> None: ...
    def multi_get(self, keys: Iterable[str]) -> dict[str, FeatureValue]: ...
    def delete(self, key: str) -> None: ...
    def keys(self) -> list[str]: ...
    def clear(self) -> None: ...


def _make_key(feature: str, entity_id: str) -> str:
    return f"{feature}:{entity_id}"


# ---------------------------------------------------------------------------
# InMemoryStore
# ---------------------------------------------------------------------------


class InMemoryStore:
    """Thread-safe dict-backed store with explicit TTL eviction on read.

    Suited for tests, the parity-check benchmark, and as the offline/training
    side of an :class:`OnlineOfflineStore`. Reads are O(1).
    """

    def __init__(self, spec: FeatureSpec | None = None) -> None:
        self._data: dict[str, FeatureValue] = {}
        self._lock = threading.Lock()
        self.spec = spec  # used for TTL eviction; can be None

    def get(self, key: str) -> FeatureValue | None:
        with self._lock:
            v = self._data.get(key)
            if v is None:
                return None
            if self.spec is not None and v.is_expired(self.spec):
                del self._data[key]
                return None
            return v

    def set(self, key: str, value: FeatureValue) -> None:
        with self._lock:
            self._data[key] = value

    def multi_get(self, keys: Iterable[str]) -> dict[str, FeatureValue]:
        out: dict[str, FeatureValue] = {}
        with self._lock:
            for k in keys:
                v = self._data.get(k)
                if v is None:
                    continue
                if self.spec is not None and v.is_expired(self.spec):
                    del self._data[k]
                    continue
                out[k] = v
        return out

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


# ---------------------------------------------------------------------------
# RedisStore
# ---------------------------------------------------------------------------


class RedisStore:
    """Redis-backed store. Each value is stored as a JSON blob plus the
    event/stored timestamps. TTL is enforced by Redis.

    Construct with an existing ``redis.Redis`` client so connection pooling and
    auth live outside this class.
    """

    def __init__(self, client: Any, namespace: str = "fsm", spec: FeatureSpec | None = None) -> None:
        try:
            import redis  # noqa: F401 — used implicitly via the client type
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "redis is not installed. Install with `pip install feature-store-mini[redis]`."
            ) from exc
        self.client = client
        self.namespace = namespace
        self.spec = spec

    def _ns_key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def get(self, key: str) -> FeatureValue | None:
        raw = self.client.get(self._ns_key(key))
        if raw is None:
            return None
        return self._decode(raw)

    def set(self, key: str, value: FeatureValue) -> None:
        body = json.dumps({"value": value.value, "event_time": value.event_time, "stored_at": value.stored_at})
        ex = int(self.spec.ttl_secs) if (self.spec and self.spec.ttl_secs) else None
        self.client.set(self._ns_key(key), body, ex=ex)

    def multi_get(self, keys: Iterable[str]) -> dict[str, FeatureValue]:
        keys_list = list(keys)
        if not keys_list:
            return {}
        ns_keys = [self._ns_key(k) for k in keys_list]
        raws = self.client.mget(ns_keys)
        out: dict[str, FeatureValue] = {}
        for k, raw in zip(keys_list, raws):
            if raw is None:
                continue
            v = self._decode(raw)
            if v is not None:
                out[k] = v
        return out

    def delete(self, key: str) -> None:
        self.client.delete(self._ns_key(key))

    def keys(self) -> list[str]:
        pattern = f"{self.namespace}:*"
        return [
            (k.decode() if isinstance(k, bytes) else k)[len(self.namespace) + 1 :]
            for k in self.client.scan_iter(match=pattern)
        ]

    def clear(self) -> None:
        for k in self.keys():
            self.client.delete(self._ns_key(k))

    @staticmethod
    def _decode(raw: Any) -> FeatureValue | None:
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return FeatureValue(
            value=data.get("value"),
            event_time=float(data.get("event_time", time.time())),
            stored_at=float(data.get("stored_at", time.time())),
        )


# ---------------------------------------------------------------------------
# OnlineOfflineStore — the wrapper everything else uses
# ---------------------------------------------------------------------------


class OnlineOfflineStore:
    """Holds a *pair* of stores: one for the online (serving) tier and one for
    the offline (training) tier. The :class:`ParityChecker` compares them.

    A real deployment would map ``online`` → Redis and ``offline`` → a parquet/
    Iceberg dataset. Here both are :class:`StoreBackend`s so the same API works.
    """

    def __init__(self, online: StoreBackend, offline: StoreBackend) -> None:
        self.online = online
        self.offline = offline

    def write(
        self,
        feature: FeatureSpec,
        entity_id: str,
        value: Any,
        event_time: float | None = None,
        *,
        online: bool = True,
        offline: bool = True,
    ) -> None:
        ts = event_time if event_time is not None else time.time()
        fv = FeatureValue(value=value, event_time=ts)
        key = _make_key(feature.name, entity_id)
        if online:
            self.online.set(key, fv)
        if offline:
            self.offline.set(key, fv)

    def read_online(self, feature: FeatureSpec, entity_id: str) -> FeatureValue | None:
        return self.online.get(_make_key(feature.name, entity_id))

    def read_offline(self, feature: FeatureSpec, entity_id: str) -> FeatureValue | None:
        return self.offline.get(_make_key(feature.name, entity_id))

    def multi_read_online(self, feature: FeatureSpec, entity_ids: Iterable[str]) -> dict[str, FeatureValue]:
        keys = [_make_key(feature.name, eid) for eid in entity_ids]
        raw = self.online.multi_get(keys)
        prefix = f"{feature.name}:"
        return {k.removeprefix(prefix): v for k, v in raw.items() if k.startswith(prefix)}

    def keys_for(self, feature: FeatureSpec, *, side: str = "online") -> list[str]:
        store = self.online if side == "online" else self.offline
        prefix = f"{feature.name}:"
        return [k.removeprefix(prefix) for k in store.keys() if k.startswith(prefix)]
