# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for multimodal runtime fail-fast handling (#2860)."""

from __future__ import annotations

import pytest

from vllm_mlx.models import mllm


def test_vision_runtime_reports_incompatible_mlx_vlm_version(monkeypatch):
    monkeypatch.setattr(mllm, "version", lambda _distribution: "0.7.0")

    status, detail = mllm.vision_runtime_status()

    assert status is mllm.VisionRuntimeStatus.INCOMPATIBLE
    assert detail == "0.7.0"


def test_cli_incompatible_runtime_is_actionable_and_not_reported_as_oom(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        mllm,
        "vision_runtime_status",
        lambda: (mllm.VisionRuntimeStatus.INCOMPATIBLE, "installed 0.7.0"),
    )
    monkeypatch.setattr(mllm, "_managed_desktop_runtime", lambda: False)
    monkeypatch.setattr(mllm.sys, "executable", "/active/runtime/bin/python")

    with pytest.raises(SystemExit) as exc_info:
        mllm.require_mlx_vlm_or_exit("publisher/vision-model")

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "installed 0.7.0" in stderr
    assert "not a Metal out-of-memory error" in stderr
    assert "/active/runtime/bin/python -m pip" in stderr
    assert f"mlx-vlm=={mllm.VALIDATED_MLX_VLM_VERSION}" in stderr


def test_engine_guard_reports_missing_runtime_with_model_context(monkeypatch):
    monkeypatch.setattr(
        mllm,
        "vision_runtime_status",
        lambda: (mllm.VisionRuntimeStatus.ABSENT, "mlx_vlm"),
    )

    with pytest.raises(ImportError) as exc_info:
        mllm._require_mlx_vlm("publisher/vision-model")

    message = str(exc_info.value)
    assert "publisher/vision-model" in message
    assert "optional `mlx-vlm` dependency" in message


def test_validated_runtime_version_is_accepted(monkeypatch):
    monkeypatch.setattr(
        mllm, "version", lambda _distribution: mllm.VALIDATED_MLX_VLM_VERSION
    )

    status, detail = mllm.vision_runtime_status()

    assert status is mllm.VisionRuntimeStatus.OK
    assert detail is None
