# Qwen4 fused GDN single-token decode

## Scope

This experiment fuses the Qwen4-Exp single-token Gated DeltaNet recurrence into
one Metal dispatch. It includes the causal-convolution state update, SiLU,
query/key normalization, decay and beta gates, recurrent update, and gated
RMSNorm. The affine-q4 output projection remains on Rapid's stock path.

The implementation is opt-in through
`RAPID_MLX_QWEN4_FUSED_GDN_DECODE=1` or the resident `stock|fused` selector.
Prefill, batching, masks, ragged caches, training, and speculative rollback use
the stock implementation.

## Prior qualification

The same recurrence-only kernel was qualified in the mlx-uag source tree on an
M5 Max before this Rapid port:

- twelve 32-token trajectories across 1K, 4K, 8K, 16K, 32K, and 64K contexts;
- exact full logits, convolution cache, and fp32 recurrent state;
- 13,824 of 13,824 eligible fused layer calls with zero fallback;
- median decode improvement between 6.39% and 6.93% at every context rung.

The separately tested GDN plus affine-q4 output-projection epilogue was slower
and is intentionally not included.

## Rapid port verification

The focused admission, dispatch, resident-selector, cache-update, and fallback
contracts pass: 7 tests. Python compilation and `git diff --check` also pass.

Rapid-specific Metal and real-model validation is pending because the shared
GPU was occupied after the port was prepared. Run the following only on an
idle GPU:

```bash
MLX_ENABLE_TF32=0 PYTHONPATH=. python scripts/bench_qwen4_fused_gdn_decode.py \
  --execute-metal \
  --model /path/to/Qwen3.8-Flash-Next-MLX-4bit-MTP \
  --output /tmp/qwen4-fused-gdn-decode.json
```

Promotion requires 32 exact sequential steps, zero fallback, and a positive
interleaved stock/fused median on the target host. Until that gate passes, keep
the environment flag disabled.
