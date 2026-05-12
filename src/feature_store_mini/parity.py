"""Online/offline parity checker.

Compares the online (serving) and offline (training) views of the *same*
feature for a fixed list of entity IDs and surfaces five named divergence
types:

* ``stale``               — online value is older than its freshness window
* ``missing``             — feature exists on one side but not the other
* ``casting``             — value types differ (e.g., int offline vs string online)
* ``range_skew``          — for numeric features, online distribution drifts
  from offline by more than ``range_skew_z`` standard deviations on mean
* ``timestamp_window``    — online event_time isn't within the expected window
  of the offline event_time (point-in-time-correctness violation)

Every divergence is reported with the entity id, severity, and an explanation
so a reviewer can pinpoint the bad row.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from feature_store_mini.spec import FeatureSpec, FeatureValue
from feature_store_mini.stores import OnlineOfflineStore


class DivergenceType(str, Enum):
    STALE = "stale"
    MISSING = "missing"
    CASTING = "casting"
    RANGE_SKEW = "range_skew"
    TIMESTAMP_WINDOW = "timestamp_window"


@dataclass
class Divergence:
    """One mismatch between online and offline."""

    type: DivergenceType
    entity_id: str | None
    detail: str
    online_value: object = None
    offline_value: object = None


@dataclass
class ParityReport:
    feature: str
    n_compared: int
    divergences: list[Divergence] = field(default_factory=list)

    @property
    def types_present(self) -> set[DivergenceType]:
        return {d.type for d in self.divergences}

    def count(self, dtype: DivergenceType) -> int:
        return sum(1 for d in self.divergences if d.type == dtype)


class ParityChecker:
    """Runs the five-divergence sweep over a feature.

    ``range_skew_z``: threshold on absolute z-score of (online_mean - offline_mean)
    normalized by offline_std. Higher = more permissive.

    ``timestamp_window_secs``: allowed |online.event_time - offline.event_time|.
    A point-in-time-correct online value should match the offline event_time
    within this slack (default: 1 second — assumes both came from the same
    write pipeline).
    """

    def __init__(
        self,
        store: OnlineOfflineStore,
        *,
        range_skew_z: float = 3.0,
        timestamp_window_secs: float = 1.0,
    ) -> None:
        self.store = store
        self.range_skew_z = range_skew_z
        self.timestamp_window_secs = timestamp_window_secs

    def check(
        self,
        feature: FeatureSpec,
        entity_ids: Iterable[str],
        *,
        now: float | None = None,
    ) -> ParityReport:
        entity_list = list(entity_ids)
        report = ParityReport(feature=feature.name, n_compared=len(entity_list))
        if not entity_list:
            return report

        online_vals: list[tuple[str, FeatureValue]] = []
        offline_vals: list[tuple[str, FeatureValue]] = []
        for eid in entity_list:
            on = self.store.read_online(feature, eid)
            off = self.store.read_offline(feature, eid)
            if on is None and off is None:
                continue  # both missing — counted as 0 coverage on both sides, not a divergence
            if on is None and off is not None:
                report.divergences.append(
                    Divergence(DivergenceType.MISSING, eid, "online missing", None, off.value)
                )
                offline_vals.append((eid, off))
                continue
            if off is None and on is not None:
                report.divergences.append(
                    Divergence(DivergenceType.MISSING, eid, "offline missing", on.value, None)
                )
                online_vals.append((eid, on))
                continue
            assert on is not None and off is not None  # for type checker
            online_vals.append((eid, on))
            offline_vals.append((eid, off))

            # Stale check (online only — offline isn't "stale", it's historical).
            if on.is_stale(feature, now=now):
                report.divergences.append(
                    Divergence(
                        DivergenceType.STALE,
                        eid,
                        f"online event_time is older than freshness window ({feature.freshness_window_secs}s)",
                        on.value,
                        off.value,
                    )
                )

            # Casting check.
            if type(on.value) is not type(off.value):
                report.divergences.append(
                    Divergence(
                        DivergenceType.CASTING,
                        eid,
                        f"type mismatch online={type(on.value).__name__} offline={type(off.value).__name__}",
                        on.value,
                        off.value,
                    )
                )

            # Timestamp-window check (point-in-time correctness).
            if abs(on.event_time - off.event_time) > self.timestamp_window_secs:
                report.divergences.append(
                    Divergence(
                        DivergenceType.TIMESTAMP_WINDOW,
                        eid,
                        (
                            f"event_time differs by "
                            f"{abs(on.event_time - off.event_time):.2f}s "
                            f"(>{self.timestamp_window_secs}s slack)"
                        ),
                        on.event_time,
                        off.event_time,
                    )
                )

        # Range-skew (numeric only). Compared distribution-level so it's one entry per feature.
        if feature.dtype in (int, float) and offline_vals and online_vals:
            try:
                on_numeric = [float(v.value) for _, v in online_vals if isinstance(v.value, (int, float))]
                off_numeric = [float(v.value) for _, v in offline_vals if isinstance(v.value, (int, float))]
            except (TypeError, ValueError):
                on_numeric, off_numeric = [], []
            if len(on_numeric) >= 5 and len(off_numeric) >= 5:
                off_mean = statistics.fmean(off_numeric)
                off_std = statistics.pstdev(off_numeric) or 1e-9
                on_mean = statistics.fmean(on_numeric)
                z = (on_mean - off_mean) / off_std
                if abs(z) >= self.range_skew_z:
                    report.divergences.append(
                        Divergence(
                            DivergenceType.RANGE_SKEW,
                            None,
                            f"online_mean={on_mean:.4g} offline_mean={off_mean:.4g} "
                            f"z={z:+.2f} (threshold {self.range_skew_z})",
                            on_mean,
                            off_mean,
                        )
                    )

        return report
