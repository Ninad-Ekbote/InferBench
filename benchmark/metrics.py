"""Per-request metrics, CSV persistence, and summary aggregation.

Nothing in this module knows about HTTP or vLLM specifically — it only
deals with `RequestMetrics` records, so it stays reusable regardless of
what future client changes (new endpoint, new server flags under test)
look like.
"""
from __future__ import annotations

import csv
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("benchmark.metrics")


@dataclass
class RequestMetrics:
    """Timing and outcome data for a single request."""

    request_id: int
    user_id: int
    prompt_index: int
    prompt_words: int

    success: bool = False
    error: str = ""

    start_time: float = 0.0
    end_time: float = 0.0

    ttft: Optional[float] = None  # seconds from request start to first streamed token
    inter_token_gaps: list[float] = field(default_factory=list)  # seconds between consecutive tokens
    tokens_generated: int = 0

    @property
    def latency(self) -> float:
        """End-to-end wall-clock latency in seconds."""
        return self.end_time - self.start_time

    @property
    def avg_itl(self) -> Optional[float]:
        """Average inter-token latency in seconds, or None if fewer than 2 tokens arrived."""
        if not self.inter_token_gaps:
            return None
        return statistics.mean(self.inter_token_gaps)

    @property
    def tokens_per_sec(self) -> float:
        if self.latency <= 0 or self.tokens_generated == 0:
            return 0.0
        return self.tokens_generated / self.latency

    def to_row(self) -> dict:
        """Flatten this record into a CSV-writable dict."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "prompt_index": self.prompt_index,
            "prompt_words": self.prompt_words,
            "success": self.success,
            "error": self.error,
            "start_time": round(self.start_time, 6),
            "end_time": round(self.end_time, 6),
            "latency_s": round(self.latency, 6),
            "ttft_s": round(self.ttft, 6) if self.ttft is not None else "",
            "avg_itl_s": round(self.avg_itl, 6) if self.avg_itl is not None else "",
            "tokens_generated": self.tokens_generated,
            "tokens_per_sec": round(self.tokens_per_sec, 4),
        }


CSV_FIELDS = [
    "request_id", "user_id", "prompt_index", "prompt_words", "success", "error",
    "start_time", "end_time", "latency_s", "ttft_s", "avg_itl_s",
    "tokens_generated", "tokens_per_sec",
]


def write_csv(results: list[RequestMetrics], path: Path) -> None:
    """Write one row per request to `path`, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_row())
    logger.info("Wrote %d rows to %s", len(results), path)


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile, matching numpy's default method."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


@dataclass
class Summary:
    """Aggregate statistics for a completed benchmark run."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    error_rate: float

    total_tokens: int
    wall_time_s: float
    requests_per_sec: float
    tokens_per_sec: float

    avg_ttft: float
    p50_ttft: float
    p95_ttft: float
    p99_ttft: float

    avg_itl: float

    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float


def summarize(results: list[RequestMetrics], wall_time_s: float) -> Summary:
    """Compute aggregate statistics across all requests in a run."""
    total = len(results)
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    ttfts = [r.ttft for r in successes if r.ttft is not None]
    itls = [gap for r in successes for gap in r.inter_token_gaps]
    latencies = [r.latency for r in successes]
    total_tokens = sum(r.tokens_generated for r in successes)

    return Summary(
        total_requests=total,
        successful_requests=len(successes),
        failed_requests=len(failures),
        success_rate=(len(successes) / total * 100) if total else 0.0,
        error_rate=(len(failures) / total * 100) if total else 0.0,
        total_tokens=total_tokens,
        wall_time_s=wall_time_s,
        requests_per_sec=(total / wall_time_s) if wall_time_s > 0 else 0.0,
        tokens_per_sec=(total_tokens / wall_time_s) if wall_time_s > 0 else 0.0,
        avg_ttft=statistics.mean(ttfts) if ttfts else 0.0,
        p50_ttft=_percentile(ttfts, 50),
        p95_ttft=_percentile(ttfts, 95),
        p99_ttft=_percentile(ttfts, 99),
        avg_itl=statistics.mean(itls) if itls else 0.0,
        avg_latency=statistics.mean(latencies) if latencies else 0.0,
        p50_latency=_percentile(latencies, 50),
        p95_latency=_percentile(latencies, 95),
        p99_latency=_percentile(latencies, 99),
    )


def print_summary(summary: Summary) -> None:
    """Print a human-readable summary block to stdout."""
    rule = "=" * 60
    print(f"\n{rule}")
    print("BENCHMARK SUMMARY")
    print(rule)
    print(f"Total requests:          {summary.total_requests}")
    print(f"Successful:              {summary.successful_requests}")
    print(f"Failed:                  {summary.failed_requests}")
    print(f"Success rate:            {summary.success_rate:.2f}%")
    print(f"Error rate:              {summary.error_rate:.2f}%")
    print(f"Wall clock time:         {summary.wall_time_s:.2f} s")
    print(f"Requests/sec:            {summary.requests_per_sec:.3f}")
    print(f"Total tokens generated:  {summary.total_tokens}")
    print(f"Tokens/sec:              {summary.tokens_per_sec:.2f}")
    print("-" * 60)
    print(f"Avg TTFT:                {summary.avg_ttft * 1000:.2f} ms")
    print(f"P50 TTFT:                {summary.p50_ttft * 1000:.2f} ms")
    print(f"P95 TTFT:                {summary.p95_ttft * 1000:.2f} ms")
    print(f"P99 TTFT:                {summary.p99_ttft * 1000:.2f} ms")
    print(f"Avg ITL:                 {summary.avg_itl * 1000:.2f} ms")
    print("-" * 60)
    print(f"Avg latency:             {summary.avg_latency:.3f} s")
    print(f"P50 latency:             {summary.p50_latency:.3f} s")
    print(f"P95 latency:             {summary.p95_latency:.3f} s")
    print(f"P99 latency:             {summary.p99_latency:.3f} s")
    print(rule)
