"""Minimal feature store with online/offline parity check.

Three building blocks:

* :class:`FeatureSpec`  — describes one feature (name, dtype, freshness window)
* ``StoreBackend``      — in-memory or Redis storage of feature values + timestamps
* :class:`ParityChecker` — compares offline (training) and online (serving) views
  of the same feature, surfaces 5 named divergence types

The package is intentionally tiny — every API is exercised by the tests in
``tests/`` and the latency benchmark in ``bench/latency.py``.
"""

from feature_store_mini.parity import (
    DivergenceType,
    Divergence,
    ParityChecker,
    ParityReport,
)
from feature_store_mini.spec import FeatureSpec, FeatureValue
from feature_store_mini.stores import (
    InMemoryStore,
    OnlineOfflineStore,
    RedisStore,
    StoreBackend,
)

__version__ = "0.1.0"

__all__ = [
    "FeatureSpec",
    "FeatureValue",
    "StoreBackend",
    "InMemoryStore",
    "RedisStore",
    "OnlineOfflineStore",
    "ParityChecker",
    "ParityReport",
    "Divergence",
    "DivergenceType",
]
