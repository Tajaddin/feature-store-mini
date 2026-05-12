"""Latency benchmark.

Writes ``n_features`` × ``n_entities`` cells into an :class:`InMemoryStore`,
then measures get + multi_get latencies in microseconds.

Target (from project IDEAS.md): in-memory p99 < 2,000 µs (2 ms) for single-get
on 100K cells.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feature_store_mini import FeatureSpec, FeatureValue, InMemoryStore


def quantile(samples: list[int], q: float) -> int:
    if not samples:
        return 0
    s = sorted(samples)
    k = int(q * (len(s) - 1))
    return s[k]


def summarize(label: str, samples_ns: list[int]) -> dict:
    if not samples_ns:
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": len(samples_ns),
        "mean_us": round(sum(samples_ns) / len(samples_ns) / 1000, 3),
        "p50_us": round(quantile(samples_ns, 0.50) / 1000, 3),
        "p95_us": round(quantile(samples_ns, 0.95) / 1000, 3),
        "p99_us": round(quantile(samples_ns, 0.99) / 1000, 3),
        "p999_us": round(quantile(samples_ns, 0.999) / 1000, 3),
        "max_us": round(max(samples_ns) / 1000, 3),
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    p = argparse.ArgumentParser()
    p.add_argument("--n-cells", type=int, default=100_000, help="Total feature-value cells")
    p.add_argument("--n-reads", type=int, default=20_000, help="Reads to measure")
    p.add_argument("--multi-get-batch", type=int, default=50)
    p.add_argument("--out", type=str, default="bench/latency_results.json")
    args = p.parse_args()

    spec = FeatureSpec(name="bench", dtype=int)
    store = InMemoryStore()

    # Warm + populate.
    write_start = time.perf_counter()
    for i in range(args.n_cells):
        store.set(f"k:{i}", FeatureValue(value=i, event_time=write_start))
    write_elapsed = time.perf_counter() - write_start
    print(f"Populated {args.n_cells:,} cells in {write_elapsed:.2f}s "
          f"({args.n_cells / max(write_elapsed, 1e-9):,.0f} writes/s)")

    # Single-get measurement.
    keys = [f"k:{i % args.n_cells}" for i in range(args.n_reads)]
    # warm
    for k in keys[: min(2000, len(keys))]:
        store.get(k)

    get_samples: list[int] = []
    for k in keys:
        t0 = time.perf_counter_ns()
        store.get(k)
        get_samples.append(time.perf_counter_ns() - t0)

    # Multi-get measurement.
    batches: list[list[str]] = []
    for i in range(0, args.n_reads, args.multi_get_batch):
        batches.append(keys[i : i + args.multi_get_batch])
    mget_samples: list[int] = []
    mget_per_key_samples: list[int] = []
    for batch in batches:
        t0 = time.perf_counter_ns()
        out = store.multi_get(batch)
        elapsed = time.perf_counter_ns() - t0
        mget_samples.append(elapsed)
        if batch:
            mget_per_key_samples.append(elapsed // len(batch))

    get_summary = summarize("InMemoryStore.get", get_samples)
    mget_summary = summarize(f"InMemoryStore.multi_get (batch={args.multi_get_batch})", mget_samples)
    mget_per_key_summary = summarize("multi_get per-key (amortized)", mget_per_key_samples)

    result = {
        "n_cells": args.n_cells,
        "n_reads": args.n_reads,
        "write_throughput_per_sec": int(args.n_cells / max(write_elapsed, 1e-9)),
        "single_get": get_summary,
        "multi_get": mget_summary,
        "multi_get_per_key": mget_per_key_summary,
        "in_memory_p99_under_2ms": get_summary["p99_us"] < 2000,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print()
    print(f"{'Label':<48} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'p999':>8} {'max':>10}  (us)")
    for s in (get_summary, mget_summary, mget_per_key_summary):
        print(
            f"{s['label']:<48} {s['mean_us']:>8.2f} {s['p50_us']:>8.2f} "
            f"{s['p95_us']:>8.2f} {s['p99_us']:>8.2f} {s['p999_us']:>8.2f} {s['max_us']:>10.2f}"
        )
    print(f"\nTarget: single-get p99 < 2000 µs on 100K cells")
    print(f"Result: {'PASS' if result['in_memory_p99_under_2ms'] else 'FAIL'} "
          f"(actual p99 = {get_summary['p99_us']} µs)")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
