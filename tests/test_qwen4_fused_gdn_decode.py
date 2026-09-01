# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.gated_delta import gated_delta_update

from vllm_mlx.kernels import qwen4_fused_gdn_decode as fused_gdn
from vllm_mlx.models import qwen4_exp


class FakeArray:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


class FakeCache:
    def __init__(self, conv_state=None, recurrent_state=None):
        self.cache = [conv_state, recurrent_state]
        self.lengths = None
        self.advanced = 0

    def __getitem__(self, index):
        return self.cache[index]

    def __setitem__(self, index, value):
        self.cache[index] = value

    def advance(self, amount):
        self.advanced += amount


def production_values(dtype=mx.bfloat16):
    return {
        "qkv": FakeArray((1, 1, 10240), dtype),
        "z": FakeArray((1, 1, 6144), dtype),
        "beta": FakeArray((1, 1, 48), dtype),
        "alpha": FakeArray((1, 1, 48), dtype),
        "conv_state": FakeArray((1, 3, 10240), dtype),
        "recurrent_state": FakeArray((1, 48, 128, 128), mx.float32),
        "conv_weight": FakeArray((10240, 4, 1), dtype),
        "a_log": FakeArray((48,), mx.float32),
        "dt_bias": FakeArray((48,), dtype),
        "norm_weight": FakeArray((128,), dtype),
    }


def admission(**overrides):
    values = production_values()
    values.update(overrides)
    return fused_gdn.admit_qwen4_fused_gdn_decode(
        **values,
        mask=None,
        cache_lengths=None,
        record_rollback=False,
        training=False,
        sharded=False,
        num_key_heads=16,
        num_value_heads=48,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel=4,
        gate_activation="sigmoid",
    )


def tiny_args():
    return SimpleNamespace(
        hidden_size=16,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1.0e-6,
        output_gate_type="sigmoid",
        hidden_act="silu",
    )


class Identity:
    def __call__(self, value):
        return value


def test_production_single_token_decode_is_admitted():
    result = admission()
    assert result.accepted, result.reason


def test_batch_prefill_mask_ragged_and_speculation_fall_back():
    result = admission(qkv=FakeArray((2, 1, 10240), mx.bfloat16))
    assert not result.accepted
    assert "qkv shape" in result.reason

    values = production_values()
    base = {
        **values,
        "training": False,
        "sharded": False,
        "num_key_heads": 16,
        "num_value_heads": 48,
        "key_head_dim": 128,
        "value_head_dim": 128,
        "conv_kernel": 4,
        "gate_activation": "sigmoid",
    }
    result = fused_gdn.admit_qwen4_fused_gdn_decode(
        **base,
        mask=object(),
        cache_lengths=None,
        record_rollback=False,
    )
    assert result.reason == "masked decode"
    result = fused_gdn.admit_qwen4_fused_gdn_decode(
        **base,
        mask=None,
        cache_lengths=object(),
        record_rollback=False,
    )
    assert result.reason == "ragged cache lengths"
    result = fused_gdn.admit_qwen4_fused_gdn_decode(
        **base,
        mask=None,
        cache_lengths=None,
        record_rollback=True,
    )
    assert result.reason == "speculative rollback"


def test_dtype_and_geometry_are_strict():
    result = admission(a_log=FakeArray((48,), mx.float16))
    assert not result.accepted
    assert "A_log" in result.reason
    assert admission(a_log=FakeArray((48,), mx.bfloat16)).accepted

    values = production_values()
    result = fused_gdn.admit_qwen4_fused_gdn_decode(
        **values,
        mask=None,
        cache_lengths=None,
        record_rollback=False,
        training=False,
        sharded=False,
        num_key_heads=24,
        num_value_heads=48,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel=4,
        gate_activation="sigmoid",
    )
    assert not result.accepted
    assert "unsupported geometry" in result.reason


def test_kernel_dispatch_is_one_threadgroup_per_value_head():
    calls = []

    def fake_kernel(**kwargs):
        calls.append(kwargs)
        return [
            FakeArray(shape, dtype)
            for shape, dtype in zip(
                kwargs["output_shapes"], kwargs["output_dtypes"], strict=True
            )
        ]

    values = production_values()
    with patch.object(fused_gdn, "_kernel", return_value=fake_kernel):
        outputs = fused_gdn.qwen4_fused_gdn_decode(
            values["qkv"],
            values["z"],
            values["beta"],
            values["alpha"],
            values["conv_state"],
            values["conv_weight"],
            values["a_log"],
            values["dt_bias"],
            values["recurrent_state"],
            values["norm_weight"],
            1.0e-6,
            threadgroup_y=16,
        )
    assert calls[0]["grid"] == (32, 16, 48)
    assert calls[0]["threadgroup"] == (32, 16, 1)
    assert outputs[0].shape == (1, 1, 6144)
    assert outputs[1].shape == (1, 3, 10240)
    assert outputs[2].shape == (1, 48, 128, 128)
    assert ("RATIO", 3) in calls[0]["template"]


def test_concurrent_probe_publishes_only_after_initialization():
    entered = Event()
    release = Event()
    calls = 0

    def blocking_kernel(*args, **kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=5)
        return (object(), object(), object())

    with (
        patch.object(fused_gdn, "_PROBE_COMPLETE", False),
        patch.object(fused_gdn, "_PROBED_THREADGROUP_Y", None),
        patch.object(fused_gdn, "_PROBE_LOCK", Lock()),
        patch.object(fused_gdn, "fused_gdn_runtime_supported", return_value=True),
        patch.object(fused_gdn, "qwen4_fused_gdn_decode", side_effect=blocking_kernel),
        patch.object(fused_gdn.mx, "eval"),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        first = pool.submit(fused_gdn.probe_qwen4_fused_gdn_decode, mx.bfloat16)
        assert entered.wait(timeout=5)
        second = pool.submit(fused_gdn.probe_qwen4_fused_gdn_decode, mx.bfloat16)
        release.set()
        assert first.result(timeout=5) == 32
        assert second.result(timeout=5) == 32

    assert calls == 1


def test_resident_switch_preserves_weights_and_defaults_stock():
    with patch.object(qwen4_exp, "_FUSED_GDN_DEFAULT", False):
        layer = qwen4_exp.GatedDeltaNet(tiny_args())
    weight = layer.conv1d.weight
    assert qwen4_exp.qwen4_fused_gdn_mode_counts(layer) == {
        "stock": 1,
        "fused": 0,
    }
    assert qwen4_exp.set_qwen4_fused_gdn_mode(layer, "fused") == 1
    assert layer.conv1d.weight is weight
    assert layer.fused_gdn_decode_mode == "fused"
    assert qwen4_exp.set_qwen4_fused_gdn_mode(layer, "stock") == 1
    assert layer.conv1d.weight is weight


def test_uninitialized_and_speculative_cache_do_not_probe_metal():
    layer = qwen4_exp.GatedDeltaNet(tiny_args())
    layer.eval()
    layer.set_fused_gdn_decode_mode("fused")
    values = production_values()
    with patch.object(qwen4_exp, "fused_gdn_runtime_supported") as runtime:
        result = layer._try_fused_decode(
            values["qkv"],
            values["z"],
            values["beta"],
            values["alpha"],
            None,
            FakeCache(),
            record_rollback=False,
        )
    assert result is None
    runtime.assert_not_called()
    assert layer.fused_gdn_decode_last_fallback == "uninitialized cache"

    cache = FakeCache(values["conv_state"], values["recurrent_state"])
    result = layer._try_fused_decode(
        values["qkv"],
        values["z"],
        values["beta"],
        values["alpha"],
        None,
        cache,
        record_rollback=True,
    )
    assert result is None
    assert layer.fused_gdn_decode_last_fallback == "speculative rollback"


def test_admitted_path_updates_cache_and_counter_without_real_kernel():
    layer = qwen4_exp.GatedDeltaNet(tiny_args())
    layer.eval()
    layer.set_fused_gdn_decode_mode("fused")
    layer.out_proj = Identity()
    values = production_values()
    cache = FakeCache(values["conv_state"], values["recurrent_state"])
    fused_output = FakeArray((1, 1, 6144), mx.bfloat16)
    next_conv = object()
    next_state = object()
    accepted = fused_gdn.FusedGdnAdmission(True, "eligible")
    with (
        patch.object(qwen4_exp, "admit_qwen4_fused_gdn_decode", return_value=accepted),
        patch.object(qwen4_exp, "fused_gdn_runtime_supported", return_value=True),
        patch.object(qwen4_exp, "probe_qwen4_fused_gdn_decode", return_value=8),
        patch.object(
            qwen4_exp,
            "qwen4_fused_gdn_decode",
            return_value=(fused_output, next_conv, next_state),
        ) as execute,
    ):
        result = layer._try_fused_decode(
            values["qkv"],
            values["z"],
            values["beta"],
            values["alpha"],
            None,
            cache,
            record_rollback=False,
        )
    assert result is fused_output
    assert cache[0] is next_conv
    assert cache[1] is next_state
    assert cache.advanced == 1
    assert layer.fused_gdn_decode_calls == 1
    assert execute.call_args.kwargs["threadgroup_y"] == 8


@pytest.mark.requires_mlx
def test_real_metal_kernel_matches_stock_for_32_sequential_steps():
    """Guard every BF16 boundary that the fused dispatch replaces."""
    if not mx.metal.is_available():
        pytest.skip("requires a Metal GPU")
    previous_device = mx.default_device()
    mx.set_default_device(mx.gpu)
    dtype = mx.bfloat16
    conv_weight = (
        mx.random.normal(
            (fused_gdn.CONV_DIM, fused_gdn.CONV_KERNEL, 1),
            key=mx.random.key(1),
        )
        * 0.02
    ).astype(dtype)
    a_log = (
        mx.random.normal((fused_gdn.NUM_VALUE_HEADS,), key=mx.random.key(2)) * 0.2
    ).astype(mx.float32)
    dt_bias = (
        mx.random.normal((fused_gdn.NUM_VALUE_HEADS,), key=mx.random.key(3)) * 0.2
    ).astype(dtype)
    norm_weight = (
        mx.random.normal((fused_gdn.VALUE_HEAD_DIM,), key=mx.random.key(4)) * 0.05 + 1
    ).astype(dtype)
    stock_conv = mx.zeros(
        (1, fused_gdn.CONV_KERNEL - 1, fused_gdn.CONV_DIM), dtype=dtype
    )
    fused_conv = mx.array(stock_conv)
    stock_state = mx.zeros(
        (
            1,
            fused_gdn.NUM_VALUE_HEADS,
            fused_gdn.VALUE_HEAD_DIM,
            fused_gdn.KEY_HEAD_DIM,
        ),
        dtype=mx.float32,
    )
    fused_state = mx.array(stock_state)

    try:
        for step in range(32):
            qkv = (
                mx.random.normal(
                    (1, 1, fused_gdn.CONV_DIM), key=mx.random.key(100 + step)
                )
                * 0.2
            ).astype(dtype)
            z = (
                mx.random.normal(
                    (1, 1, fused_gdn.VALUE_DIM), key=mx.random.key(200 + step)
                )
                * 0.2
            ).astype(dtype)
            beta = (
                mx.random.normal(
                    (1, 1, fused_gdn.NUM_VALUE_HEADS),
                    key=mx.random.key(300 + step),
                )
                * 0.2
            ).astype(dtype)
            alpha = (
                mx.random.normal(
                    (1, 1, fused_gdn.NUM_VALUE_HEADS),
                    key=mx.random.key(400 + step),
                )
                * 0.2
            ).astype(dtype)

            conv_input = mx.concatenate([stock_conv, qkv], axis=1)
            next_stock_conv = mx.contiguous(
                conv_input[:, -(fused_gdn.CONV_KERNEL - 1) :, :]
            )
            convolved = nn.silu(
                mx.conv1d(
                    conv_input,
                    conv_weight,
                    groups=fused_gdn.CONV_DIM,
                )
            )
            query, key, value = [
                item.reshape(1, 1, heads, dim)
                for item, heads, dim in zip(
                    mx.split(
                        convolved,
                        [fused_gdn.KEY_DIM, 2 * fused_gdn.KEY_DIM],
                        axis=-1,
                    ),
                    [
                        fused_gdn.NUM_KEY_HEADS,
                        fused_gdn.NUM_KEY_HEADS,
                        fused_gdn.NUM_VALUE_HEADS,
                    ],
                    [
                        fused_gdn.KEY_HEAD_DIM,
                        fused_gdn.KEY_HEAD_DIM,
                        fused_gdn.VALUE_HEAD_DIM,
                    ],
                )
            ]
            query = query * mx.rsqrt(
                mx.sum(mx.square(query), axis=-1, keepdims=True) + 1e-6
            )
            key = key * mx.rsqrt(mx.sum(mx.square(key), axis=-1, keepdims=True) + 1e-6)
            query = query * (fused_gdn.KEY_HEAD_DIM**-0.5)
            stock_output, next_stock_state = gated_delta_update(
                query,
                key,
                value,
                alpha,
                beta,
                a_log,
                dt_bias,
                stock_state,
                use_kernel=True,
            )
            stock_output = (
                mx.fast.rms_norm(stock_output, norm_weight, 1e-6).astype(mx.float32)
                * mx.sigmoid(
                    z.reshape(
                        1,
                        1,
                        fused_gdn.NUM_VALUE_HEADS,
                        fused_gdn.VALUE_HEAD_DIM,
                    ).astype(mx.float32)
                )
            ).astype(dtype)
            stock_output = stock_output.reshape(1, 1, fused_gdn.VALUE_DIM)

            fused_output, next_fused_conv, next_fused_state = (
                fused_gdn.qwen4_fused_gdn_decode(
                    qkv,
                    z,
                    beta,
                    alpha,
                    fused_conv,
                    conv_weight,
                    a_log,
                    dt_bias,
                    fused_state,
                    norm_weight,
                    1e-6,
                    threadgroup_y=32,
                )
            )
            mx.eval(
                stock_output,
                fused_output,
                next_stock_conv,
                next_fused_conv,
                next_stock_state,
                next_fused_state,
            )
            assert mx.array_equal(stock_output, fused_output).item(), step
            assert mx.array_equal(next_stock_conv, next_fused_conv).item(), step
            assert mx.array_equal(next_stock_state, next_fused_state).item(), step
            stock_conv, fused_conv = next_stock_conv, next_fused_conv
            stock_state, fused_state = next_stock_state, next_fused_state
    finally:
        mx.set_default_device(previous_device)
