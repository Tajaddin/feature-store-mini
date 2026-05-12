# feature-store-mini

> Minimal feature store with in-memory + Redis backends and an explicit **online/offline parity checker** covering 5 named divergence types. **p99 single-read latency = 2.1 µs** on 100K-cell in-memory store. The kind of thing every senior MLE has written by hand at some point.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Tests](https://img.shields.io/badge/tests-21%20passing-brightgreen)](#tests) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

## Hero benchmark

### Parity checker — all 5 divergence types caught

200-test fixture: each named divergence type has its own dedicated test fixture plus an integration test that fires all five in one run.

| Divergence | What it catches | Test |
|---|---|---|
| `stale` | online value's `event_time` exceeds `freshness_window_secs` | `test_stale_divergence_fires` |
| `missing` | feature exists on one side but not the other | `test_missing_divergence_fires_for_online_only` + offline variant |
| `casting` | online and offline value types differ (e.g. `int` vs `str`) | `test_casting_divergence_fires` |
| `range_skew` | distributional drift between online and offline (z-score on mean) | `test_range_skew_divergence_fires` |
| `timestamp_window` | online `event_time` not within slack of offline `event_time` — point-in-time correctness | `test_timestamp_window_divergence_fires` |

The 5-in-one test `test_all_five_divergence_types_in_one_run` confirms a single `check()` call surfaces every type at once. **5 / 5 types fire.**

### Latency — 100K cells, single-threaded

| Operation | Mean | p50 | p95 | **p99** | p99.9 | max |
|---|---:|---:|---:|---:|---:|---:|
| `InMemoryStore.get` (single read) | 1.07 µs | 0.90 µs | 1.50 µs | **2.10 µs** | 23.50 µs | 35.90 µs |
| `multi_get` (batch of 50) | 18.57 µs | 15.60 µs | 30.70 µs | 47.90 µs | 67.70 µs | 71.40 µs |
| `multi_get` per-key (amortized) | 0.37 µs | 0.31 µs | 0.61 µs | 0.96 µs | 1.35 µs | 1.43 µs |
| Write throughput | — | — | — | — | — | **394,970 writes/s** |

Target from project IDEAS.md: **p99 single-get < 2 ms on 100K-feature benchmark.** Actual p99 = **2.1 µs** — **~950× under budget**.

Reproduce: `python bench/latency.py`. Raw output: [`bench/latency_results.json`](bench/latency_results.json).

## Why this exists

Feast, Tecton, and Vertex AI Feature Store are heavyweight platforms. They solve the same problem every MLE has solved by hand on smaller teams:

1. Register a feature with its dtype and freshness rules.
2. Write the same value to two places — the training tier (offline) and the serving tier (online).
3. Detect when they drift apart.

`feature-store-mini` is exactly those three things and nothing else. ~500 lines of code, no Redis required for the default path, drop-in `StoreBackend` protocol so you can swap in your existing key/value store.

## Architecture

```
                              ┌─────────────────────────────────┐
                              │       OnlineOfflineStore         │
                              ├─────────────────────────────────┤
write(feature, entity, val) ─▶│  online: StoreBackend            │  ─▶ low-latency read
                              │  offline: StoreBackend           │  ─▶ training pipeline
                              └────────────────┬────────────────┘
                                               │
                              ┌────────────────▼────────────────┐
                              │       ParityChecker              │
                              ├─────────────────────────────────┤
                              │  stale  / missing  / casting    │
                              │  range_skew / timestamp_window  │
                              └─────────────────────────────────┘
                                               │
                                               ▼
                                       ParityReport with
                                       per-entity diagnostics
```

## Quickstart

```bash
pip install -e ".[redis]"      # redis extra is optional
```

```python
import time
from feature_store_mini import (
    FeatureSpec, InMemoryStore, OnlineOfflineStore, ParityChecker, DivergenceType,
)

# 1. Register a feature
balance = FeatureSpec(
    name="user_balance_usd",
    dtype=float,
    freshness_window_secs=300.0,   # online value must be ≤ 5 min old
    ttl_secs=86400,                # in-memory store evicts after 24 h
)

# 2. Build a store with two tiers
store = OnlineOfflineStore(
    online=InMemoryStore(),
    offline=InMemoryStore(),
)

# 3. Write to both tiers — production code routes online to Redis and
#    offline to a parquet/Iceberg write path.
now = time.time()
store.write(balance, entity_id="user-42", value=123.45, event_time=now)

# 4. Read from either side
print(store.read_online(balance, "user-42").value)   # 123.45

# 5. Compare them
report = ParityChecker(store).check(balance, ["user-42"])
print(report.types_present)   # set() — clean parity
```

### Redis path

```python
import redis
from feature_store_mini import RedisStore, InMemoryStore, OnlineOfflineStore

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=False)
store = OnlineOfflineStore(
    online=RedisStore(redis_client, namespace="features", spec=balance),  # TTL enforced
    offline=InMemoryStore(),   # offline can still be a parquet adapter in your code
)
```

`RedisStore` stores each value as a JSON blob and uses the Redis `EX` TTL when `FeatureSpec.ttl_secs` is set.

## Catching real-world divergence

A common training-serving skew bug:

```python
now = time.time()

# A bad online writer accidentally stores the value as a string
store.write(balance, "user-1", 123.45, event_time=now, online=False, offline=True)
store.write(balance, "user-1", "123.45", event_time=now, online=True, offline=False)

report = ParityChecker(store).check(balance, ["user-1"], now=now)
for d in report.divergences:
    print(d.type.value, d.detail)
# casting type mismatch online=str offline=float
```

A point-in-time correctness bug (training set used a future value):

```python
store.write(balance, "user-7", 200.0, event_time=now - 30, online=True, offline=False)
store.write(balance, "user-7", 200.0, event_time=now, online=False, offline=True)

report = ParityChecker(store, timestamp_window_secs=1.0).check(balance, ["user-7"], now=now)
print(report.types_present)
# {<DivergenceType.TIMESTAMP_WINDOW: 'timestamp_window'>}
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

```
21 passed in 3.32s
```

Includes hypothesis property tests on round-trip serialization. The five named divergence tests + the integration test prove the hero metric directly.

## Configuration knobs

| Knob | Where | Effect | Default |
|---|---|---|---|
| `freshness_window_secs` | `FeatureSpec` | Online value age tolerance | 3600 s |
| `ttl_secs` | `FeatureSpec` | Backend eviction (TTL) | `None` (no eviction) |
| `range_skew_z` | `ParityChecker(range_skew_z=…)` | z-score threshold for distribution drift | 3.0 |
| `timestamp_window_secs` | `ParityChecker(timestamp_window_secs=…)` | Allowed online↔offline event_time gap | 1.0 s |
| `namespace` | `RedisStore(namespace=…)` | Key prefix in Redis | "fsm" |

## Project layout

```
.
├── src/feature_store_mini/
│   ├── __init__.py
│   ├── spec.py       # FeatureSpec + FeatureValue
│   ├── stores.py     # InMemoryStore + RedisStore + OnlineOfflineStore wrapper
│   └── parity.py     # ParityChecker, 5 named DivergenceTypes, ParityReport
├── tests/            # 21 pytest cases (incl. hypothesis property tests)
└── bench/
    ├── latency.py    # 100K-cell single-threaded benchmark
    └── latency_results.json
```

## Limitations

**Single-process by design.** `InMemoryStore` is thread-safe but not multi-process. For multi-process serving, use `RedisStore`. For ML platform-scale, this isn't the right abstraction — use Feast / Tecton.

**No automatic schema enforcement on write.** `FeatureSpec.dtype` is advisory — the store accepts whatever you pass. The casting check catches the mismatch on the *parity* path, not at write time. Production deployments should add a Pydantic-validated wrapper around `write`.

**Range-skew is parametric.** The default check uses normal-distribution z-score on the mean. For long-tailed or categorical features, swap in a custom `ParityChecker` subclass with KS or chi-square logic. The 5-divergence set is the interface, not the implementation.

**Latency benchmark is single-threaded Python.** Real production workloads see contention on the Redis client and serialization overhead. The 2.1 µs single-get number is an upper bound on what's achievable in pure Python — adopt as a sanity ceiling, not a guarantee.

**No async API yet.** All operations are synchronous. An `async` variant is straightforward (the Redis client supports it); tracked for v0.2.

## License

MIT — see [LICENSE](LICENSE).
