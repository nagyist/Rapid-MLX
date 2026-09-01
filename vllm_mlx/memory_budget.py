"""Per-model Metal memory budgeting (#2858).

The engine historically exposed one global knob — ``--gpu-memory-utilization``,
default ``0.90`` — that had to be tuned per model by hand: a 12B checkpoint
runs fine at ``0.75`` while a 20B MoE with an ~11.2 GB Metal footprint needs
``0.95`` on the same 16 GB Mac.  Worse, when the cap was too low the model
still loaded and reported healthy while the D-METAL-CAP admission gate
deterministically rejected every request with HTTP 503.

This module closes that loop with two pieces, both pure and unit-testable:

* :func:`plan_metal_limit` — resolve the effective utilization for one loaded
  model.  When the operator passed an explicit ``--gpu-memory-utilization``
  the value is honored verbatim (advanced override).  In auto mode (the new
  default) the limit is sized to the model actually loaded: the MEASURED
  weight footprint (``mx.get_active_memory()`` right after load — no disk
  heuristics) plus a runtime headroom, clamped to
  ``[AUTO_UTILIZATION_FLOOR, AUTO_UTILIZATION_CEILING]`` of the device's
  recommended working-set budget.  The floor keeps small models byte-identical
  to the historical ``0.90`` default; the ceiling leaves the OS a margin the
  same way vLLM's ``gpu_memory_utilization`` never goes to 1.0.

* :func:`format_preflight_error` — the actionable admission-impossible
  message required by #2858: required vs available memory plus concrete
  remediations.  Raised (wrapped in :class:`MetalPreflightError`) by
  ``Scheduler.preflight_metal_admission`` when even a modest request could
  never be admitted under the resolved cap, so startup fails BEFORE the
  server reports the model ready instead of serving deterministic 503s.
"""

from __future__ import annotations

from dataclasses import dataclass

# Auto-mode bounds. The floor matches the historical global default so any
# model that fit comfortably before resolves to a byte-identical limit; the
# ceiling reserves ~3% of the device budget for allocator slack so auto mode
# never plans right up to the working-set edge.
AUTO_UTILIZATION_FLOOR = 0.90
AUTO_UTILIZATION_CEILING = 0.97

# Runtime headroom charged on top of the measured weight footprint in auto
# mode: KV cache for in-flight requests, activation workspace, Metal heap
# fragmentation. Fractional so big models reserve proportionally more, with an
# absolute floor so tiny models still get a workable slice.
_AUTO_HEADROOM_FRACTION = 0.08
_AUTO_HEADROOM_MIN_BYTES = 512 * 1024**2


class MetalPreflightError(RuntimeError):
    """The resolved Metal cap can never admit a request for this model.

    Raised during engine startup — before the server reports the model
    ready — so the operator sees one actionable message instead of a
    healthy-looking model whose every request returns HTTP 503.
    """


@dataclass(frozen=True)
class MetalBudgetPlan:
    """The resolved Metal allocation budget for one loaded model."""

    weights_bytes: int
    device_budget_bytes: int
    requested_utilization: float | None
    resolved_utilization: float
    limit_bytes: int
    mode: str  # "manual" (operator override) or "auto"


def plan_metal_limit(
    *,
    weights_bytes: int,
    device_budget_bytes: int,
    requested_utilization: float | None = None,
) -> MetalBudgetPlan:
    """Resolve the Metal allocation limit for one loaded model.

    Args:
        weights_bytes: Measured Metal footprint of the loaded weights
            (``mx.get_active_memory()`` right after load). ``<= 0`` means
            "no measurement available" and auto mode falls back to the
            historical floor.
        device_budget_bytes: The device's recommended working-set size
            (``max_recommended_working_set_size``). Must be positive.
        requested_utilization: The operator's explicit
            ``--gpu-memory-utilization``, or ``None`` for auto.

    Returns:
        A :class:`MetalBudgetPlan`. ``resolved_utilization`` is the value the
        engine must feed to BOTH ``mx.set_memory_limit`` and the scheduler's
        D-METAL-CAP admission gate — the two enforcement points must always
        agree on the same cap.
    """
    if device_budget_bytes <= 0:
        raise ValueError("device_budget_bytes must be positive")

    if requested_utilization is not None:
        resolved = float(requested_utilization)
        mode = "manual"
    elif weights_bytes <= 0:
        resolved = AUTO_UTILIZATION_FLOOR
        mode = "auto"
    else:
        headroom = max(
            int(weights_bytes * _AUTO_HEADROOM_FRACTION), _AUTO_HEADROOM_MIN_BYTES
        )
        needed = (weights_bytes + headroom) / device_budget_bytes
        resolved = min(max(needed, AUTO_UTILIZATION_FLOOR), AUTO_UTILIZATION_CEILING)
        mode = "auto"

    return MetalBudgetPlan(
        weights_bytes=max(0, int(weights_bytes)),
        device_budget_bytes=int(device_budget_bytes),
        requested_utilization=requested_utilization,
        resolved_utilization=resolved,
        limit_bytes=int(device_budget_bytes * resolved),
        mode=mode,
    )


def format_preflight_error(
    *,
    required_bytes: int,
    active_bytes: int,
    min_kv_bytes: int,
    cap_bytes: int,
    utilization: float,
    device_budget_bytes: int,
) -> str:
    """Build the actionable admission-impossible startup message (#2858)."""
    return (
        f"This model needs approximately {required_bytes / 1e9:.1f} GB of "
        f"Metal memory for the current configuration (weights and runtime "
        f"{active_bytes / 1e9:.1f} GB + minimum KV cache "
        f"{min_kv_bytes / 1e9:.1f} GB), but the current limit is "
        f"{cap_bytes / 1e9:.1f} GB "
        f"(gpu_memory_utilization={utilization:.2f} of the "
        f"{device_budget_bytes / 1e9:.1f} GB Metal working-set budget). "
        f"Increase --gpu-memory-utilization, reduce context length or "
        f"concurrency, close memory-heavy apps, or choose a smaller model."
    )
