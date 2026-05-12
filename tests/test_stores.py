"""Store tests: in-memory get/set/multi_get + TTL eviction + hypothesis round-trip."""

from __future__ import annotations

import time

import pytest
from hypothesis import given, strategies as st

from feature_store_mini import FeatureSpec, FeatureValue, InMemoryStore, OnlineOfflineStore


def test_in_memory_set_get_roundtrip() -> None:
    store = InMemoryStore()
    fv = FeatureValue(value=42, event_time=time.time())
    store.set("user:abc", fv)
    out = store.get("user:abc")
    assert out is not None
    assert out.value == 42


def test_in_memory_missing_key_returns_none() -> None:
    assert InMemoryStore().get("does_not_exist") is None


def test_in_memory_multi_get() -> None:
    store = InMemoryStore()
    for i in range(5):
        store.set(f"k:{i}", FeatureValue(value=i, event_time=time.time()))
    out = store.multi_get([f"k:{i}" for i in (0, 2, 4, 9)])
    assert sorted(out.keys()) == ["k:0", "k:2", "k:4"]
    assert out["k:0"].value == 0


def test_in_memory_ttl_evicts_on_read() -> None:
    spec = FeatureSpec(name="ttl", dtype=int, ttl_secs=0.05)
    store = InMemoryStore(spec=spec)
    store.set("x", FeatureValue(value=1, event_time=time.time()))
    assert store.get("x") is not None
    time.sleep(0.1)
    assert store.get("x") is None
    assert "x" not in store.keys()  # eviction is persisted


def test_in_memory_clear() -> None:
    store = InMemoryStore()
    store.set("a", FeatureValue(value=1, event_time=time.time()))
    store.clear()
    assert store.keys() == []


def test_online_offline_write_to_both_sides_by_default() -> None:
    spec = FeatureSpec(name="balance", dtype=float)
    store = OnlineOfflineStore(online=InMemoryStore(), offline=InMemoryStore())
    store.write(spec, "user-1", 100.0)
    assert store.read_online(spec, "user-1").value == 100.0
    assert store.read_offline(spec, "user-1").value == 100.0


def test_online_offline_one_side_only() -> None:
    spec = FeatureSpec(name="balance", dtype=float)
    store = OnlineOfflineStore(online=InMemoryStore(), offline=InMemoryStore())
    store.write(spec, "u", 100.0, online=True, offline=False)
    assert store.read_online(spec, "u") is not None
    assert store.read_offline(spec, "u") is None


def test_keys_for_filters_by_feature() -> None:
    spec_a = FeatureSpec(name="a", dtype=int)
    spec_b = FeatureSpec(name="b", dtype=int)
    store = OnlineOfflineStore(online=InMemoryStore(), offline=InMemoryStore())
    store.write(spec_a, "1", 1)
    store.write(spec_b, "1", 2)
    assert set(store.keys_for(spec_a, side="online")) == {"1"}
    assert set(store.keys_for(spec_b, side="online")) == {"1"}


@given(value=st.integers(min_value=-10**6, max_value=10**6))
def test_int_roundtrip_property(value: int) -> None:
    store = InMemoryStore()
    fv = FeatureValue(value=value, event_time=time.time())
    store.set("k", fv)
    assert store.get("k").value == value


@given(text=st.text(min_size=0, max_size=100))
def test_str_roundtrip_property(text: str) -> None:
    store = InMemoryStore()
    store.set("k", FeatureValue(value=text, event_time=time.time()))
    assert store.get("k").value == text


def test_feature_value_is_stale() -> None:
    spec = FeatureSpec(name="x", dtype=int, freshness_window_secs=60.0)
    now = time.time()
    fresh = FeatureValue(value=1, event_time=now)
    stale = FeatureValue(value=1, event_time=now - 3600)
    assert fresh.is_stale(spec, now=now) is False
    assert stale.is_stale(spec, now=now) is True
