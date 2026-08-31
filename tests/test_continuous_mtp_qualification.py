# SPDX-License-Identifier: Apache-2.0
"""Per-artifact qualification for continuous self-MTP."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _args(model: str, payload: str, *, force: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        speculative_config=payload,
        enable_ddtree=False,
        enable_dflash=False,
        enable_mtp=False,
        no_spec_decode=False,
        spec_decode="none",
        dflash_drafter_path="",
        mtp_num_draft_tokens=1,
        mtp_optimistic=False,
        mtp_sidecar=None,
        mtp_max_k=None,
        mtp_disable_auto_k=False,
        force_spec_decode=force,
        suffix_decoding=False,
        suffix_max_draft=None,
        suffix_max_suffix_len=None,
        suffix_min_confidence=None,
        suffix_min_draft_len=None,
    )


@pytest.mark.parametrize(
    ("alias", "tier"),
    [
        ("qwen3.5-4b-4bit", "blocked"),
        ("qwen3.5-9b-4bit", "verified"),
        ("qwen3.6-27b-4bit", "verified"),
        ("qwen3.8-27b-4bit", "verified"),
        ("qwen3.5-9b-8bit", "unknown"),
    ],
)
def test_catalog_records_only_exact_measured_artifacts(alias: str, tier: str) -> None:
    from vllm_mlx.model_aliases import resolve_profile

    profile = resolve_profile(alias)
    assert profile is not None
    assert profile.mtp_continuous_batching_tier == tier


def test_verified_alias_can_request_continuous_mtp_without_force() -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _args(
        "qwen3.5-9b-4bit",
        '{"method":"mtp","continuous_batching":true}',
    )
    _normalize_speculative_config_or_exit(args)

    assert args.mtp_continuous_batching is True
    assert args.mtp_continuous_batching_tier == "verified"
    assert args.mtp_sidecar == "mlx-community/Qwen3.5-9B-MTP-4bit"


@pytest.mark.parametrize(
    ("alias", "tier", "message"),
    [
        ("qwen3.5-4b-4bit", "blocked", "failed paired output qualification"),
        (
            "qwen3.5-9b-8bit",
            "unknown",
            "has not completed paired output qualification",
        ),
    ],
)
def test_unqualified_alias_fails_closed(
    alias: str, tier: str, message: str, capsys
) -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _args(alias, '{"method":"mtp","continuous_batching":true}')
    with pytest.raises(SystemExit) as excinfo:
        _normalize_speculative_config_or_exit(args)

    assert excinfo.value.code == 2
    assert args.mtp_continuous_batching_tier == tier
    assert message in capsys.readouterr().err


def test_force_override_keeps_unqualified_artifact_experimental() -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _args(
        "qwen3.5-4b-4bit",
        '{"method":"mtp","continuous_batching":true}',
        force=True,
    )
    _normalize_speculative_config_or_exit(args)

    assert args.mtp_continuous_batching is True
    assert args.mtp_continuous_batching_tier == "blocked"


def test_ordinary_mtp_is_not_rejected_by_continuous_qualification() -> None:
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    args = _args("qwen3.5-4b-4bit", '{"method":"mtp"}')
    _normalize_speculative_config_or_exit(args)

    assert args.mtp_continuous_batching is False
    assert args.mtp_continuous_batching_tier == "blocked"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"kv_cache_turboquant": "k8v4"},
            "--kv-cache-turboquant k8v4 is incompatible",
        ),
        (
            {"kv_cache_quantization": True},
            "--kv-cache-quantization is incompatible",
        ),
        (
            {"kv_cache_dtype": "int8"},
            "--kv-cache-dtype int8 is incompatible",
        ),
    ],
)
def test_continuous_mtp_rejects_explicit_quantized_cache(
    overrides: dict[str, object], message: str
) -> None:
    from vllm_mlx.cli import continuous_mtp_cache_conflict

    args = SimpleNamespace(
        mtp_continuous_batching=True,
        kv_cache_turboquant=None,
        kv_cache_quantization=False,
        kv_cache_dtype="bf16",
    )
    for name, value in overrides.items():
        setattr(args, name, value)

    assert message in (continuous_mtp_cache_conflict(args) or "")


def test_continuous_mtp_accepts_bf16_and_explicit_turboquant_off() -> None:
    from vllm_mlx.cli import continuous_mtp_cache_conflict

    args = SimpleNamespace(
        mtp_continuous_batching=True,
        kv_cache_turboquant="none",
        kv_cache_quantization=False,
        kv_cache_dtype="bf16",
    )

    assert continuous_mtp_cache_conflict(args) is None


def test_ordinary_mtp_preserves_existing_cache_defaults() -> None:
    from vllm_mlx.cli import continuous_mtp_cache_conflict

    args = SimpleNamespace(
        mtp_continuous_batching=False,
        kv_cache_turboquant="k8v4",
        kv_cache_quantization=False,
        kv_cache_dtype="int4",
    )

    assert continuous_mtp_cache_conflict(args) is None


def test_continuous_mtp_suppresses_alias_turboquant_auto_default() -> None:
    from vllm_mlx.cli import _resolve_turboquant_with_mtp_policy

    args = SimpleNamespace(
        mtp_continuous_batching=True,
        kv_cache_turboquant=None,
        kv_cache_quantization=False,
    )
    detected = SimpleNamespace(turboquant_tier="k8v4_verified")

    assert (
        _resolve_turboquant_with_mtp_policy(
            args,
            model_name="qwen3.5-9b-4bit",
            _detected_config=detected,
        )
        is None
    )


def test_ordinary_mtp_keeps_alias_turboquant_auto_default() -> None:
    from vllm_mlx.cli import _resolve_turboquant_with_mtp_policy

    args = SimpleNamespace(
        mtp_continuous_batching=False,
        kv_cache_turboquant=None,
        kv_cache_quantization=False,
    )
    detected = SimpleNamespace(turboquant_tier="k8v4_verified")

    assert (
        _resolve_turboquant_with_mtp_policy(
            args,
            model_name="qwen3.5-9b-4bit",
            _detected_config=detected,
        )
        == "k8v4"
    )


@pytest.mark.parametrize("tier", ["verified", "blocked"])
def test_non_unknown_tier_requires_an_mtp_target(tier: str) -> None:
    from vllm_mlx.model_aliases import _coerce

    with pytest.raises(ValueError, match="requires supports_native_mtp"):
        _coerce(
            "bad",
            {
                "hf_path": "example/model",
                "mtp_continuous_batching_tier": tier,
            },
        )


def test_invalid_tier_fails_alias_registry_validation() -> None:
    from vllm_mlx.model_aliases import _coerce

    with pytest.raises(ValueError, match="must be one of"):
        _coerce(
            "bad",
            {
                "hf_path": "example/model",
                "supports_native_mtp": True,
                "mtp_speculative_tokens": 1,
                "mtp_continuous_batching_tier": "probably",
            },
        )


def test_qualification_benchmark_dry_run_is_network_free() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "bench" / "bench_continuous_mtp_server.py"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--label",
            "candidate",
            "--model",
            "example/model",
            "--runs",
            "2",
            "--concurrency",
            "3",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["planned_requests"] == 6
