"""Per-model Metal memory budgeting tests (#2858).

Covers the three pieces the issue's acceptance criteria name:

* ``plan_metal_limit`` — auto per-model budget selection, including the two
  required fixtures: a model that fits at the default (0.90 floor) budget
  and one that requires a higher per-model budget, plus the explicit
  operator override.
* ``Scheduler.preflight_metal_admission`` — a model whose resident
  footprint can never admit a request fails startup with the actionable
  required-vs-available message instead of reporting healthy and 503ing
  every request.
* The rewritten D-METAL-CAP admission 503 — required vs available plus a
  concrete remediation, while keeping the tokens existing regression tests
  grep for.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

from vllm_mlx.memory_budget import (  # noqa: E402
    AUTO_UTILIZATION_CEILING,
    AUTO_UTILIZATION_FLOOR,
    MetalPreflightError,
    format_preflight_error,
    plan_metal_limit,
)
from vllm_mlx.request import Request, SamplingParams  # noqa: E402
from vllm_mlx.scheduler import (  # noqa: E402
    BackpressureError,
    Scheduler,
    SchedulerConfig,
)

GB = 10**9


def _make_scheduler(*, gpu_memory_utilization: float = 0.9) -> Scheduler:
    config = SchedulerConfig(
        max_num_seqs=8,
        max_concurrent_requests=64,
        enable_prefix_cache=False,
        use_memory_aware_cache=False,
        use_paged_cache=False,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    tokenizer = MagicMock()
    tokenizer.encode = lambda s: list(range(len(s)))
    return Scheduler(model=MagicMock(), tokenizer=tokenizer, config=config)


class TestPlanMetalLimit:
    def test_small_model_fits_at_default_floor(self):
        """#2858 acceptance fixture 1: a model with plenty of headroom
        resolves to exactly the historical 0.90 default — auto mode is
        byte-identical to the old behavior for every model that already
        fit comfortably."""
        plan = plan_metal_limit(
            weights_bytes=2 * GB,
            device_budget_bytes=20 * GB,
        )
        assert plan.mode == "auto"
        assert plan.resolved_utilization == AUTO_UTILIZATION_FLOOR
        assert plan.limit_bytes == int(20 * GB * AUTO_UTILIZATION_FLOOR)

    def test_large_model_gets_higher_per_model_budget(self):
        """#2858 acceptance fixture 2: the issue's 20B-MoE-on-16GB shape —
        weights close to the device budget push the resolved utilization
        ABOVE the 0.90 floor without the operator touching anything."""
        weights = 11 * GB
        device = 12_800_000_000  # ~the working-set budget of a 16 GB Mac
        plan = plan_metal_limit(weights_bytes=weights, device_budget_bytes=device)
        assert plan.mode == "auto"
        assert plan.resolved_utilization > AUTO_UTILIZATION_FLOOR
        assert plan.resolved_utilization <= AUTO_UTILIZATION_CEILING
        # The resolved limit must actually cover the weights.
        assert plan.limit_bytes > weights

    def test_oversized_model_clamps_at_ceiling(self):
        """Auto mode never plans past the ceiling — a model that cannot
        fit resolves to the ceiling and the scheduler preflight (not the
        planner) is what refuses startup."""
        plan = plan_metal_limit(
            weights_bytes=15 * GB,
            device_budget_bytes=12 * GB,
        )
        assert plan.resolved_utilization == AUTO_UTILIZATION_CEILING

    def test_explicit_override_is_honored_verbatim(self):
        """The advanced manual override must bypass auto sizing entirely,
        including values BELOW what auto would pick."""
        plan = plan_metal_limit(
            weights_bytes=11 * GB,
            device_budget_bytes=12 * GB,
            requested_utilization=0.5,
        )
        assert plan.mode == "manual"
        assert plan.resolved_utilization == 0.5
        assert plan.limit_bytes == 6 * GB

    def test_missing_measurement_falls_back_to_floor(self):
        plan = plan_metal_limit(weights_bytes=0, device_budget_bytes=20 * GB)
        assert plan.resolved_utilization == AUTO_UTILIZATION_FLOOR

    def test_invalid_device_budget_raises(self):
        with pytest.raises(ValueError):
            plan_metal_limit(weights_bytes=GB, device_budget_bytes=0)


class TestPreflightErrorMessage:
    def test_message_reports_required_available_and_remediation(self):
        """#2858 acceptance: a rejection must report required vs available
        memory and at least one concrete remediation."""
        message = format_preflight_error(
            required_bytes=11_200_000_000,
            active_bytes=10_700_000_000,
            min_kv_bytes=500_000_000,
            cap_bytes=9_100_000_000,
            utilization=0.75,
            device_budget_bytes=12_100_000_000,
        )
        assert "11.2 GB" in message  # required
        assert "9.1 GB" in message  # current limit
        assert "--gpu-memory-utilization" in message  # remediation 1
        assert "smaller model" in message  # remediation 2
        assert "close memory-heavy apps" in message.lower()  # remediation 3

    def test_no_impossible_advice_at_the_ceiling(self):
        """Codex round 1 NIT: at or above the auto ceiling, 'increase
        --gpu-memory-utilization' is impossible advice — the message must
        say the Mac lacks the memory instead."""
        message = format_preflight_error(
            required_bytes=15_000_000_000,
            active_bytes=14_500_000_000,
            min_kv_bytes=500_000_000,
            cap_bytes=11_700_000_000,
            utilization=0.97,
            device_budget_bytes=12_100_000_000,
        )
        assert "Increase --gpu-memory-utilization" not in message
        assert "does not have enough unified memory" in message
        assert "smaller model" in message


class TestSchedulerPreflight:
    def test_noop_when_cap_disabled(self):
        sched = _make_scheduler(gpu_memory_utilization=0.0)
        with patch.object(sched, "_current_metal_active_bytes", return_value=10**15):
            sched.preflight_metal_admission()  # must not raise

    def test_noop_when_active_unreadable(self):
        """Non-Metal CI hosts and unit-test stubs read 0 active bytes —
        the preflight must stay silent there exactly like the gate."""
        sched = _make_scheduler()
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=10 * GB),
            patch.object(sched, "_current_metal_active_bytes", return_value=0),
        ):
            sched.preflight_metal_admission()

    def test_passes_when_model_fits(self):
        sched = _make_scheduler()
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=12 * GB),
            patch.object(sched, "_current_metal_active_bytes", return_value=8 * GB),
        ):
            sched.preflight_metal_admission()

    def test_fails_startup_when_weights_exceed_cap(self):
        """The healthy-but-every-request-503s configuration must be caught
        at startup with the actionable message (#2858 acceptance)."""
        sched = _make_scheduler()
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=9_100_000_000),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=11_200_000_000
            ),
            pytest.raises(MetalPreflightError) as exc_info,
        ):
            sched.preflight_metal_admission()
        message = str(exc_info.value)
        assert "9.1 GB" in message
        assert "--gpu-memory-utilization" in message

    def test_fails_when_smallest_request_cannot_fit(self):
        """Codex round 2 BLOCKING #1: positive but insufficient headroom
        for even a one-token exchange is still a deterministic-503 config
        and must be refused."""
        sched = _make_scheduler()
        per_tok = 100_000
        sched.config.metal_cap_kv_bytes_per_token = per_tok
        cap = 10 * GB
        # Room for one token of KV, but the smallest request needs two.
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=cap),
            patch.object(
                sched,
                "_current_metal_active_bytes",
                return_value=cap - per_tok,
            ),
            pytest.raises(MetalPreflightError),
        ):
            sched.preflight_metal_admission()

    def test_passes_when_weights_just_under_cap(self):
        """Codex round 1 BLOCKING #1: a memory-tight config whose weights
        leave room for the smallest valid request can still serve short
        requests, so preflight must NOT refuse it even though a nominal
        1024-token request's projected KV would overflow."""
        sched = _make_scheduler()
        per_tok = 100_000  # bytes per token, via the operator override path
        sched.config.metal_cap_kv_bytes_per_token = per_tok
        cap = 10 * GB
        min_kv = per_tok * Scheduler.PREFLIGHT_NOMINAL_TOKENS
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=cap),
            patch.object(
                sched,
                "_current_metal_active_bytes",
                return_value=cap - min_kv // 2,
            ),
        ):
            sched.preflight_metal_admission()

    def test_recovers_when_cache_clear_frees_enough(self):
        """Codex round 1 BLOCKING #4: reclaimable allocator cache must not
        fail a load. When the post-clear re-measure drops below the cap,
        preflight passes."""
        sched = _make_scheduler()
        cap = 10 * GB
        readings = iter([cap + GB, cap - 2 * GB])  # over, then under
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=cap),
            patch.object(
                sched,
                "_current_metal_active_bytes",
                side_effect=lambda: next(readings),
            ),
        ):
            sched.preflight_metal_admission()


class TestProcessUtilizationRatchet:
    """Codex round 2 BLOCKING #2: a resident scheduler's cap must follow
    the process-wide utilization ratchet upward."""

    @pytest.fixture(autouse=True)
    def _isolated_floor(self):
        import vllm_mlx.memory_budget as mb

        with mb._process_floor_lock:
            saved = (mb._process_utilization_floor, mb._process_floor_generation)
            mb._process_utilization_floor = 0.0
            mb._process_floor_generation += 1
        yield
        with mb._process_floor_lock:
            mb._process_utilization_floor = saved[0]
            mb._process_floor_generation += 1

    def _fake_device(self):
        import vllm_mlx.scheduler as sched_mod

        metal = MagicMock()
        metal.is_available.return_value = True
        device_info = MagicMock(return_value={"memory_size": 100 * GB})
        return patch.multiple(
            sched_mod.mx, metal=metal, device_info=device_info, create=True
        )

    def test_cap_follows_ratchet_upward(self):
        from vllm_mlx.memory_budget import note_resolved_utilization

        sched = _make_scheduler(gpu_memory_utilization=0.5)
        with self._fake_device():
            assert sched._resolve_metal_cap_bytes() == 50 * GB
            note_resolved_utilization(0.9)
            assert sched._resolve_metal_cap_bytes() == 90 * GB
            # A LOWER later resolution must not lower the enforced cap.
            note_resolved_utilization(0.6)
            assert sched._resolve_metal_cap_bytes() == 90 * GB

    def test_disabled_cap_stays_disabled(self):
        from vllm_mlx.memory_budget import note_resolved_utilization

        sched = _make_scheduler(gpu_memory_utilization=0.0)
        with self._fake_device():
            note_resolved_utilization(0.97)
            assert sched._resolve_metal_cap_bytes() == 0


class TestActionableAdmission503:
    def test_backpressure_message_carries_remediation(self):
        """The runtime D-METAL-CAP 503 must state required vs available and
        a remediation, while keeping the historical grep tokens."""
        sched = _make_scheduler(gpu_memory_utilization=0.5)
        req = Request(
            request_id="req-503",
            prompt="x" * 16,
            prompt_token_ids=list(range(16)),
            sampling_params=SamplingParams(max_tokens=1),
        )
        req.num_prompt_tokens = 16
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * GB),
            patch.object(sched, "_current_metal_active_bytes", return_value=100 * GB),
            pytest.raises(BackpressureError) as exc_info,
        ):
            sched._enforce_metal_cap_at_admission(req)
        message = str(exc_info.value)
        assert "D-METAL-CAP" in message
        assert "reserved KV" in message
        assert "current limit is" in message
        assert "--gpu-memory-utilization" in message

    def test_no_utilization_advice_when_cap_maxed(self):
        """Codex round 2 NIT: at an enforced utilization with no headroom
        left, the 503 must not suggest raising --gpu-memory-utilization."""
        sched = _make_scheduler(gpu_memory_utilization=1.0)
        sched._metal_cap_effective_utilization = 1.0
        req = Request(
            request_id="req-503-max",
            prompt="x" * 16,
            prompt_token_ids=list(range(16)),
            sampling_params=SamplingParams(max_tokens=1),
        )
        req.num_prompt_tokens = 16
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * GB),
            patch.object(sched, "_current_metal_active_bytes", return_value=100 * GB),
            pytest.raises(BackpressureError) as exc_info,
        ):
            sched._enforce_metal_cap_at_admission(req)
        message = str(exc_info.value)
        assert "D-METAL-CAP" in message
        assert "--gpu-memory-utilization" not in message
