# SPDX-License-Identifier: Apache-2.0
"""Regression tests for per-model engine performance metrics."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from vllm_mlx.output_collector import RequestOutputCollector
from vllm_mlx.runtime.model_performance import ModelPerformanceLedger

if TYPE_CHECKING:
    from vllm_mlx.request import Request
    from vllm_mlx.scheduler import Scheduler


def _scheduler() -> Scheduler:
    pytest.importorskip("mlx")

    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    tokenizer = MagicMock()
    tokenizer.encode = lambda text: list(range(len(text.split())))
    tokenizer.decode = lambda tokens, **_kwargs: " ".join(map(str, tokens))
    scheduler = Scheduler(
        MagicMock(),
        tokenizer,
        SchedulerConfig(max_num_seqs=1, model_name="model-under-test"),
    )
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.remove.return_value = {}
    return scheduler


def _running_request(scheduler: Scheduler, request_id: str) -> Request:
    from vllm_mlx.request import Request, RequestStatus, SamplingParams

    request = Request(
        request_id,
        "ignored prompt",
        SamplingParams(max_tokens=16),
    )
    request.status = RequestStatus.RUNNING
    request.num_prompt_tokens = 5
    request.arrival_time = time.time() - 0.25
    request.first_token_time = time.time() - 0.2
    for token in (11, 12):
        request.append_output_token(token)
    scheduler.running[request_id] = request
    scheduler.uid_to_request_id[1] = request_id
    scheduler.requests[request_id] = request
    scheduler.request_id_to_uid[request_id] = 1
    return request


def _terminal_response() -> MagicMock:
    response = MagicMock(
        uid=1,
        token=13,
        finish_reason="stop",
        logprobs=None,
    )
    del response.prompt_cache
    return response


def test_ledger_records_outcomes_once_and_ignores_bad_values():
    ledger = ModelPerformanceLedger("model-a")
    assert ledger.record_success(
        "success",
        prompt_tokens=8,
        completion_tokens=12,
        ttft_seconds=0.07,
        decode_tokens_per_second=42.0,
    )
    assert ledger.record_cancelled(
        "cancelled",
        prompt_tokens=4,
        completion_tokens=2,
        ttft_seconds=0.2,
        decode_tokens_per_second=18.0,
    )
    assert ledger.record_failure("failure")

    snapshot = ledger.snapshot()
    assert snapshot.total_requests == 3
    assert snapshot.requests_succeeded == 1
    assert snapshot.requests_cancelled == 1
    assert snapshot.requests_failed == 1

    assert not ledger.record_success(
        "success",
        prompt_tokens=99,
        completion_tokens=99,
        ttft_seconds=1,
        decode_tokens_per_second=1,
    )
    assert not ledger.record_failure("success")
    assert not ledger.record_cancelled(
        "cancelled",
        prompt_tokens=99,
        completion_tokens=99,
        ttft_seconds=99,
        decode_tokens_per_second=99,
    )
    assert ledger.model_name == "model-a"
    assert ledger.snapshot().prompt_tokens == 12


def test_ledger_ignores_unusable_timing_observations():
    ledger = ModelPerformanceLedger("model-b")
    ledger.record_success(
        "invalid-timings",
        prompt_tokens=1,
        completion_tokens=2,
        ttft_seconds=float("nan"),
        decode_tokens_per_second=-5,
    )

    snapshot = ledger.snapshot()
    assert snapshot.requests_succeeded == 1
    assert snapshot.prompt_tokens == 1
    assert snapshot.completion_tokens == 2
    assert snapshot.ttft_seconds_count == 0
    assert snapshot.decode_observations == 0
    assert snapshot.ttft_seconds_max is None
    assert snapshot.decode_tokens_per_second_max is None


def test_ledger_best_effort_helpers_ignore_unusable_timings_and_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    from types import SimpleNamespace

    request = SimpleNamespace(
        arrival_time=time.time() - 0.2,
        first_token_time=time.time() + 5.0,
        num_output_tokens=2,
        num_prompt_tokens=3,
        request_id="future-first-token",
    )
    ledger = ModelPerformanceLedger("model-b")

    assert ledger.decode_rate_for_request(request) is None

    monkeypatch.setattr(
        ledger,
        "record_success",
        MagicMock(side_effect=RuntimeError("accounting failure")),
    )
    monkeypatch.setattr(
        ledger,
        "record_cancelled",
        MagicMock(side_effect=RuntimeError("accounting failure")),
    )

    ledger.record_finished_performance(request)
    ledger.record_cancelled_performance(request)

    snapshot = ledger.snapshot()
    assert snapshot.total_requests == 0


def test_ledger_histograms_are_cumulative_and_memory_bounded():
    ledger = ModelPerformanceLedger("model-a")
    for ttft, decode in ((0.05, 4.0), (0.3, 25.0), (1.2, 150.0)):
        ledger.record_success(
            str(ttft),
            prompt_tokens=1,
            completion_tokens=4,
            ttft_seconds=ttft,
            decode_tokens_per_second=decode,
        )

    snapshot = ledger.snapshot()
    assert snapshot.ttft_bucket_counts == {
        "0.05": 1,
        "0.1": 1,
        "0.25": 1,
        "0.5": 2,
        "1": 2,
        "2": 3,
        "5": 3,
        "10": 3,
        "30": 3,
        "+Inf": 3,
    }
    assert snapshot.decode_bucket_counts["1"] == 0
    assert snapshot.decode_bucket_counts["5"] == 1
    assert snapshot.decode_bucket_counts["20"] == 1
    assert snapshot.decode_bucket_counts["50"] == 2
    assert snapshot.decode_bucket_counts["+Inf"] == 3
    assert snapshot.ttft_seconds_count == 3
    assert snapshot.decode_observations == 3


def test_scheduler_records_terminal_success_once():
    pytest.importorskip("mlx")

    scheduler = _scheduler()
    request = _running_request(scheduler, "success")
    response = _terminal_response()

    outputs, finished = scheduler._process_batch_responses([response])
    assert finished == {"success"}
    assert outputs[0].finished is True

    performance = scheduler.performance.snapshot()
    assert performance.model_name == "model-under-test"
    assert performance.requests_succeeded == 1
    assert performance.prompt_tokens == 5
    assert performance.completion_tokens == 3
    assert performance.ttft_seconds_count == 1
    assert performance.ttft_seconds_sum >= 0.04
    assert performance.decode_observations == 1

    # Re-delivery of the same terminal response must not double-count.
    scheduler.performance.record_finished_performance(request)
    assert scheduler.performance.snapshot().requests_succeeded == 1


def test_scheduler_records_explicit_cancellation_once():
    pytest.importorskip("mlx")

    scheduler = _scheduler()
    _running_request(scheduler, "cancelled")

    assert scheduler._do_abort_request("cancelled") is True
    performance = scheduler.performance.snapshot()
    assert performance.requests_cancelled == 1
    assert performance.prompt_tokens == 5
    assert performance.completion_tokens == 2

    scheduler.performance.record_cancelled_performance(scheduler.requests["cancelled"])
    assert scheduler.performance.snapshot().requests_cancelled == 1


@pytest.mark.asyncio
async def test_engine_loop_records_pending_failures():
    pytest.importorskip("mlx")

    from vllm_mlx.engine_core import EngineConfig, EngineCore

    engine = EngineCore(
        MagicMock(), MagicMock(), EngineConfig(model_name="model-under-test")
    )

    class _BoomScheduler:
        performance = ModelPerformanceLedger("model-under-test")

        def has_requests(self):
            return True

        def step(self):
            raise RuntimeError("Metal command buffer failure")

        def add_request(self, *_args, **_kwargs):
            pass

        def abort_request(self, *_args, **_kwargs):
            return True

        def remove_finished_request(self, *_args, **_kwargs):
            pass

    engine.scheduler = _BoomScheduler()
    engine._output_collectors["failure"] = RequestOutputCollector(aggregate=True)
    engine._finished_events["failure"] = asyncio.Event()
    engine._running = True
    loop_task = asyncio.create_task(engine._engine_loop())
    try:
        await asyncio.wait_for(engine._finished_events["failure"].wait(), timeout=1)
    finally:
        engine._running = False
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

    performance = engine.scheduler.performance.snapshot()
    assert performance.requests_failed == 1
    final = engine._output_collectors["failure"].get_nowait()
    assert final is not None and final.finished and final.error


def test_metrics_renders_model_performance_series():
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.metrics import _reset_accumulator_for_tests, router

    ledger = ModelPerformanceLedger("gemma-4-12b")
    ledger.record_success(
        "1",
        prompt_tokens=7,
        completion_tokens=4,
        ttft_seconds=0.07,
        decode_tokens_per_second=120,
    )
    ledger.record_success(
        "2",
        prompt_tokens=5,
        completion_tokens=3,
        ttft_seconds=0.4,
        decode_tokens_per_second=80,
    )
    ledger.record_success(
        "3",
        prompt_tokens=3,
        completion_tokens=2,
        ttft_seconds=0.9,
        decode_tokens_per_second=20,
    )
    ledger.record_failure("4")

    cfg = reset_config()
    cfg.model_name = "gemma-4-12b"
    _reset_accumulator_for_tests()
    app = FastAPI()
    app.include_router(router)
    cfg.engine = SimpleNamespace(
        get_stats=lambda: {"model_performance": ledger.snapshot().__dict__}
    )
    body = TestClient(app).get("/metrics").text

    assert (
        'rapid_mlx_model_requests_total{model="gemma-4-12b",outcome="succeeded"} 3'
        in body
    )
    assert (
        'rapid_mlx_model_requests_total{model="gemma-4-12b",outcome="failed"} 1' in body
    )
    assert 'rapid_mlx_model_prompt_tokens_total{model="gemma-4-12b"} 15' in body
    assert 'rapid_mlx_model_completion_tokens_total{model="gemma-4-12b"} 9' in body
    assert 'rapid_mlx_model_ttft_seconds_bucket{model="gemma-4-12b",le="0.1"} 1' in body
    assert (
        'rapid_mlx_model_ttft_seconds_bucket{model="gemma-4-12b",le="+Inf"} 3' in body
    )
    assert (
        'rapid_mlx_model_decode_tokens_per_second_bucket{model="gemma-4-12b",le="50"} 1'
        in body
    )
    assert 'rapid_mlx_model_ttft_seconds_max{model="gemma-4-12b"} 0.9' in body
    assert (
        'rapid_mlx_model_decode_tokens_per_second_last{model="gemma-4-12b"} 20' in body
    )

    reset_config()
    _reset_accumulator_for_tests()
