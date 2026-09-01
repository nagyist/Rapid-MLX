# SPDX-License-Identifier: Apache-2.0
"""Per-model request performance counters for the text scheduler.

The scheduler already records process-lifetime token and cancellation counters,
but those aggregates are not enough for a per-model inspector: they cannot say
which requests failed, how long the first token took, or how fast a request
decoded after that first token.  This ledger keeps the small amount of additional
state needed by :mod:`vllm_mlx.routes.metrics` without changing request
semantics.

The object is per ``Scheduler`` rather than process-global.  A sidecar serves one
primary engine at a time; the Prometheus series carry the model label, and the
later Desktop inspector is responsible for retaining observations across sidecar
restarts if it wants longer-term history.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

# Prometheus histograms need fixed buckets.  These cover the local-model
# operating range without making every bucket width a UI policy: TTFT spans
# warm-cache requests through long cold prefills, and decode speed spans tiny
# dense models through large MoE deployments.
TTFT_SECONDS_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
    math.inf,
)

DECODE_TOKENS_PER_SECOND_BUCKETS = (
    1.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    500.0,
    math.inf,
)


def _empty_bucket_counts(buckets: tuple[float, ...]) -> dict[str, int]:
    """Return zeroed cumulative histogram buckets."""
    counts = dict.fromkeys(
        (
            "+Inf" if math.isinf(bucket) else _format_bucket(bucket)
            for bucket in buckets
        ),
        0,
    )

    return counts


def _bucket_count(
    counts: dict[str, int],
    value: float,
    buckets: tuple[float, ...],
) -> None:
    """Add one finite, non-negative value to cumulative histogram buckets."""
    for bucket in buckets:
        label = "+Inf" if math.isinf(bucket) else _format_bucket(bucket)
        if value <= bucket:
            counts[label] += 1


def _format_bucket(value: float) -> str:
    return f"{value:g}"


@dataclass(frozen=True)
class ModelPerformanceSnapshot:
    """An immutable Prometheus-ready view of one model's request outcomes."""

    model_name: str
    requests_succeeded: int
    requests_cancelled: int
    requests_failed: int
    prompt_tokens: int
    completion_tokens: int
    ttft_bucket_counts: dict[str, int]
    ttft_seconds_count: int
    ttft_seconds_sum: float
    ttft_seconds_max: float | None
    decode_bucket_counts: dict[str, int]
    decode_observations: int
    decode_tokens_per_second_sum: float
    decode_tokens_per_second_max: float | None
    last_decode_tokens_per_second: float | None

    @property
    def total_requests(self) -> int:
        return self.requests_succeeded + self.requests_cancelled + self.requests_failed


class ModelPerformanceLedger:
    """Thread-safe, process-lifetime performance observations for one model."""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or ""
        self._lock = threading.Lock()
        self._seen_request_ids: set[str] = set()
        self._requests_succeeded = 0
        self._requests_cancelled = 0
        self._requests_failed = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._ttft_bucket_counts = _empty_bucket_counts(TTFT_SECONDS_BUCKETS)
        self._ttft_observations = 0
        self._ttft_seconds_sum = 0.0
        self._ttft_seconds_max: float | None = None
        self._decode_bucket_counts = _empty_bucket_counts(
            DECODE_TOKENS_PER_SECOND_BUCKETS
        )
        self._decode_observations = 0
        self._decode_tokens_per_second_sum = 0.0
        self._decode_tokens_per_second_max: float | None = None
        self._last_decode_tokens_per_second: float | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def record_success(
        self,
        request_id: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        ttft_seconds: float | None,
        decode_tokens_per_second: float | None,
    ) -> bool:
        """Record a completed request; return False when already accounted."""
        with self._lock:
            if request_id in self._seen_request_ids:
                return False
            self._seen_request_ids.add(request_id)
            self._requests_succeeded += 1
            self._prompt_tokens += max(0, int(prompt_tokens))
            self._completion_tokens += max(0, int(completion_tokens))
            self._observe_timings(
                ttft_seconds=ttft_seconds,
                decode_tokens_per_second=decode_tokens_per_second,
            )
            return True

    def record_cancelled(
        self,
        request_id: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        ttft_seconds: float | None,
        decode_tokens_per_second: float | None,
    ) -> bool:
        """Record an explicitly cancelled request exactly once."""
        with self._lock:
            if request_id in self._seen_request_ids:
                return False
            self._seen_request_ids.add(request_id)
            self._requests_cancelled += 1
            self._prompt_tokens += max(0, int(prompt_tokens))
            self._completion_tokens += max(0, int(completion_tokens))
            self._observe_timings(
                ttft_seconds=ttft_seconds,
                decode_tokens_per_second=decode_tokens_per_second,
            )
            return True

    def record_failure(self, request_id: str) -> bool:
        """Record an engine/runtime failure exactly once."""
        with self._lock:
            if request_id in self._seen_request_ids:
                return False
            self._seen_request_ids.add(request_id)
            self._requests_failed += 1
            return True

    def snapshot(self) -> ModelPerformanceSnapshot:
        """Return a coherent copy of the counters."""
        with self._lock:
            return ModelPerformanceSnapshot(
                model_name=self._model_name,
                requests_succeeded=self._requests_succeeded,
                requests_cancelled=self._requests_cancelled,
                requests_failed=self._requests_failed,
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
                ttft_bucket_counts=dict(self._ttft_bucket_counts),
                ttft_seconds_count=self._ttft_observations,
                ttft_seconds_sum=self._ttft_seconds_sum,
                ttft_seconds_max=self._ttft_seconds_max,
                decode_bucket_counts=dict(self._decode_bucket_counts),
                decode_observations=self._decode_observations,
                decode_tokens_per_second_sum=self._decode_tokens_per_second_sum,
                decode_tokens_per_second_max=self._decode_tokens_per_second_max,
                last_decode_tokens_per_second=self._last_decode_tokens_per_second,
            )

    def _observe_timings(
        self,
        *,
        ttft_seconds: float | None,
        decode_tokens_per_second: float | None,
    ) -> None:
        if (
            ttft_seconds is not None
            and math.isfinite(ttft_seconds)
            and ttft_seconds >= 0
        ):
            _bucket_count(
                self._ttft_bucket_counts,
                ttft_seconds,
                TTFT_SECONDS_BUCKETS,
            )
            self._ttft_observations += 1
            self._ttft_seconds_sum += ttft_seconds
            if self._ttft_seconds_max is None or ttft_seconds > self._ttft_seconds_max:
                self._ttft_seconds_max = ttft_seconds

        if (
            decode_tokens_per_second is not None
            and math.isfinite(decode_tokens_per_second)
            and decode_tokens_per_second >= 0
        ):
            _bucket_count(
                self._decode_bucket_counts,
                decode_tokens_per_second,
                DECODE_TOKENS_PER_SECOND_BUCKETS,
            )
            self._decode_observations += 1
            self._decode_tokens_per_second_sum += decode_tokens_per_second
            self._last_decode_tokens_per_second = decode_tokens_per_second
            if (
                self._decode_tokens_per_second_max is None
                or decode_tokens_per_second > self._decode_tokens_per_second_max
            ):
                self._decode_tokens_per_second_max = decode_tokens_per_second
