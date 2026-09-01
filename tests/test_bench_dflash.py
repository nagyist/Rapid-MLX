# SPDX-License-Identifier: Apache-2.0
"""Fail-closed contracts for the DFlash qualification harness."""

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
