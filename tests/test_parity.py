"""Parity tests: each of the 5 divergence types must fire on its corresponding fixture."""

from __future__ import annotations

import time

import pytest

from feature_store_mini import (
    DivergenceType,
    FeatureSpec,
    InMemoryStore,
    OnlineOfflineStore,
    ParityChecker,
)


def _store() -> OnlineOfflineStore:
    return OnlineOfflineStore(online=InMemoryStore(), offline=InMemoryStore())


# ---------------------------------------------------------------------------
# Happy path (no divergences)
# ---------------------------------------------------------------------------


def test_parity_no_divergence_when_sides_match() -> None:
    spec = FeatureSpec(name="x", dtype=int)
    store = _store()
    now = time.time()
    for i in range(10):
        store.write(spec, f"e{i}", i, event_time=now)
    report = ParityChecker(store).check(spec, [f"e{i}" for i in range(10)], now=now)
    assert report.divergences == []


# ---------------------------------------------------------------------------
# 5 named divergence types
# ---------------------------------------------------------------------------


def test_stale_divergence_fires() -> None:
    spec = FeatureSpec(name="x", dtype=int, freshness_window_secs=60.0)
    store = _store()
    now = time.time()
    store.write(spec, "e", 1, event_time=now - 7200)  # online + offline both stale
    report = ParityChecker(store).check(spec, ["e"], now=now)
    assert report.count(DivergenceType.STALE) >= 1


def test_missing_divergence_fires_for_online_only() -> None:
    spec = FeatureSpec(name="x", dtype=int)
    store = _store()
    now = time.time()
    store.write(spec, "e", 1, event_time=now, online=False, offline=True)
    report = ParityChecker(store).check(spec, ["e"], now=now)
    assert report.count(DivergenceType.MISSING) >= 1
    assert any("online missing" in d.detail for d in report.divergences)


def test_missing_divergence_fires_for_offline_only() -> None:
    spec = FeatureSpec(name="x", dtype=int)
    store = _store()
    now = time.time()
    store.write(spec, "e", 1, event_time=now, online=True, offline=False)
    report = ParityChecker(store).check(spec, ["e"], now=now)
    assert report.count(DivergenceType.MISSING) >= 1
    assert any("offline missing" in d.detail for d in report.divergences)


def test_casting_divergence_fires() -> None:
    spec = FeatureSpec(name="x", dtype=int)
    store = _store()
    now = time.time()
    store.write(spec, "e", 5, event_time=now, online=True, offline=False)
    store.write(spec, "e", "5", event_time=now, online=False, offline=True)  # str vs int
    report = ParityChecker(store).check(spec, ["e"], now=now)
    assert report.count(DivergenceType.CASTING) >= 1


def test_range_skew_divergence_fires() -> None:
    spec = FeatureSpec(name="x", dtype=float)
    store = _store()
    now = time.time()
    # 50 entities — offline values cluster near 10, online near 50 (huge skew).
    for i in range(50):
        store.write(spec, f"e{i}", 10.0, event_time=now, online=False, offline=True)
        store.write(spec, f"e{i}", 50.0, event_time=now, online=True, offline=False)
    report = ParityChecker(store).check(spec, [f"e{i}" for i in range(50)], now=now)
    assert report.count(DivergenceType.RANGE_SKEW) >= 1


def test_timestamp_window_divergence_fires() -> None:
    spec = FeatureSpec(name="x", dtype=int)
    store = _store()
    now = time.time()
    # Online and offline got the same value but at different event_times,
    # outside the 1s point-in-time slack.
    store.write(spec, "e", 7, event_time=now - 30, online=True, offline=False)
    store.write(spec, "e", 7, event_time=now, online=False, offline=True)
    report = ParityChecker(store).check(spec, ["e"], now=now, )
    assert report.count(DivergenceType.TIMESTAMP_WINDOW) >= 1


# ---------------------------------------------------------------------------
# All 5 in one report (the integration check the hero metric refers to)
# ---------------------------------------------------------------------------


def test_all_five_divergence_types_in_one_run() -> None:
    spec = FeatureSpec(name="x", dtype=float, freshness_window_secs=60.0)
    store = _store()
    now = time.time()

    # stale
    store.write(spec, "stale", 1.0, event_time=now - 7200)
    # missing online
    store.write(spec, "miss-on", 1.0, event_time=now, online=False, offline=True)
    # casting: online float vs offline str
    store.write(spec, "cast", 1.0, event_time=now, online=True, offline=False)
    store.write(spec, "cast", "1.0", event_time=now, online=False, offline=True)
    # timestamp window
    store.write(spec, "tw", 1.0, event_time=now - 100, online=True, offline=False)
    store.write(spec, "tw", 1.0, event_time=now, online=False, offline=True)
    # range skew: many entities with online >> offline
    range_ids = [f"r{i}" for i in range(50)]
    for rid in range_ids:
        store.write(spec, rid, 10.0, event_time=now, online=False, offline=True)
        store.write(spec, rid, 50.0, event_time=now, online=True, offline=False)

    report = ParityChecker(store).check(
        spec, ["stale", "miss-on", "cast", "tw"] + range_ids, now=now
    )

    assert DivergenceType.STALE in report.types_present
    assert DivergenceType.MISSING in report.types_present
    assert DivergenceType.CASTING in report.types_present
    assert DivergenceType.TIMESTAMP_WINDOW in report.types_present
    assert DivergenceType.RANGE_SKEW in report.types_present


def test_empty_entity_list_returns_empty_report() -> None:
    spec = FeatureSpec(name="x", dtype=int)
    store = _store()
    report = ParityChecker(store).check(spec, [])
    assert report.divergences == []
    assert report.n_compared == 0


def test_both_missing_is_not_a_divergence() -> None:
    spec = FeatureSpec(name="x", dtype=int)
    store = _store()
    report = ParityChecker(store).check(spec, ["never-written"])
    assert report.divergences == []
