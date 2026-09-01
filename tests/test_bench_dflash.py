# SPDX-License-Identifier: Apache-2.0
"""Fail-closed contracts for the DFlash qualification harness."""

from unittest.mock import MagicMock

import pytest

from scripts import bench_dflash
from scripts.bench_dflash import WORKLOADS, _qualify


def _passing_speedups() -> dict[str, float]:
    return {name: 1.4 for name in WORKLOADS}


def test_rejected_chat_cannot_qualify_from_code_results_alone() -> None:
    speedups = _passing_speedups()
    speedups.pop("chat")

    result = _qualify(speedups, gate=1.3, non_code_floor=1.0)

    assert result.ship is False
    assert "missing valid workloads: chat" in result.decision


def test_partially_rejected_code_cannot_qualify_from_one_code_result() -> None:
    speedups = {"fibonacci": 1.8, "chat": 1.1}

    result = _qualify(speedups, gate=1.3, non_code_floor=1.0)

    assert result.ship is False
    assert "quicksort" in result.decision
    assert "hashtable" in result.decision
    assert "sortedlist" in result.decision


def test_complete_mixed_workload_can_qualify() -> None:
    result = _qualify(_passing_speedups(), gate=1.3, non_code_floor=1.0)

    assert result.ship is True
    assert result.decision == "SHIP (supports_dflash=true)"


def test_start_server_requires_health_algorithm_receipt(monkeypatch) -> None:
    proc = MagicMock()
    proc.poll.return_value = None
    proc.wait.return_value = 0
    monkeypatch.setattr(bench_dflash.subprocess, "Popen", lambda *args, **kwargs: proc)

    def _get(url: str, timeout: float):
        del timeout
        response = MagicMock(status_code=200)
        response.json.return_value = (
            {"algorithm": "dflash2"} if url.endswith("/healthz") else {}
        )
        return response

    monkeypatch.setattr(bench_dflash.httpx, "get", _get)

    handle = bench_dflash.start_server(
        "target",
        8765,
        True,
        draft_model="draft",
        expected_algorithm="dflash2",
    )
    try:
        assert handle.algorithm == "dflash2"
    finally:
        handle.stop()


def test_start_server_requires_expected_algorithm_before_spawn(monkeypatch) -> None:
    popen = MagicMock()
    monkeypatch.setattr(bench_dflash.subprocess, "Popen", popen)

    with pytest.raises(ValueError, match="requires expected_algorithm"):
        bench_dflash.start_server("target", 8765, True, draft_model="draft")

    popen.assert_not_called()


def test_start_server_stops_process_on_algorithm_mismatch(monkeypatch) -> None:
    proc = MagicMock()
    proc.poll.return_value = None
    proc.wait.return_value = 0
    monkeypatch.setattr(bench_dflash.subprocess, "Popen", lambda *args, **kwargs: proc)

    def _get(url: str, timeout: float):
        del timeout
        response = MagicMock(status_code=200)
        response.json.return_value = (
            {"algorithm": "dflash"} if url.endswith("/healthz") else {}
        )
        return response

    monkeypatch.setattr(bench_dflash.httpx, "get", _get)

    try:
        bench_dflash.start_server(
            "target",
            8765,
            True,
            draft_model="draft",
            expected_algorithm="dflash2",
        )
    except RuntimeError as exc:
        assert "algorithm mismatch" in str(exc)
    else:
        raise AssertionError("mismatched runtime algorithm must fail")
    proc.send_signal.assert_called()
    proc.wait.assert_called()
