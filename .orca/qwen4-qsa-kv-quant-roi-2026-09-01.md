# Qwen3.8 Flash-Next QSA attention-KV quantization ROI (2026-09-01)

## Decision

No-go. Keep the current Qwen4/QSA `CacheList` exclusion. The experimental
int8 live-KV adapter saved only 1.14 GB of MLX active memory at 32K while
reducing decode throughput by 10.6%. No product PR should be opened from this
branch.

## Design and precedent

The experiment followed the established selective-cache pattern:

- vLLM quantizes only eligible KV cache layers and has an explicit skip-layer
  surface for sensitive or unsupported attention types.
- SGLang resolves cache dtype against the concrete attention backend and keeps
  incompatible speculative/draft cache owners in the model dtype.
- MLX-LM owns `KVCache -> QuantizedKVCache` conversion at the cache owner and
  keeps `CacheList` merge/extract semantics compositional.

The Rapid experiment therefore converted only the audited Qwen4 shape
`CacheList(KVCache, QSAIndexCache)`: the attention KV child became
`_QuantizableKVCache`; the QSA index ledger and every recurrent state stayed in
native precision. Arbitrary `CacheList` owners remained fail-closed.

## Environment

- Host: Studio, M3 Ultra, 256 GB unified memory
- Base: `origin/main` `82b703adc`
- Candidate: `28756decb`
- Model revision: `dcf657e4acda2aae72da99cde65b6c491cd96998`
- Text lane (`--no-mllm`), thinking off, spec decode off
- bf16 versus explicit `--kv-cache-dtype int8`
- 128/2K/8K/32K prompts, 256 requested decode tokens, three cold-cache runs
- Median TTFT/decode; MLX active memory from `/v1/status`

## Results

| Prompt | bf16 TTFT | int8 TTFT | TTFT delta | bf16 decode | int8 decode | Decode delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 420.82 ms | 421.45 ms | +0.15% | 25.499 tok/s | 24.842 tok/s | -2.58% |
| 2K | 2368.05 ms | 2377.14 ms | +0.38% | 24.043 tok/s | 23.494 tok/s | -2.28% |
| 8K | 9458.85 ms | 9517.92 ms | +0.62% | 23.345 tok/s | 22.181 tok/s | -4.99% |
| 32K | 45572.84 ms | 46324.72 ms | +1.65% | 21.473 tok/s | 19.197 tok/s | -10.60% |

After the final 32K run:

| Mode | MLX active | Retained-cache accounting | Peak RSS |
| --- | ---: | ---: | ---: |
| bf16 | 105.63 GB | 2709.45 MB | 54.51 GiB |
| int8 | 104.49 GB | 2710.10 MB | 54.51 GiB |

The explicit dtype metric reported int8 and the server logged that the live
continuous-batching KV hook was installed, so this was not a silent bf16
fallback. The approximately 1.1 GB saving is too small to change the practical
machine class for a roughly 103 GB weight footprint, while the long-context
decode penalty is user-visible.

The performance gate failed before a promotion-grade correctness battery was
warranted. Unit coverage did prove that only the Qwen4 attention-KV child was
converted and that prefix-cache hit/miss types stayed coherent (91 focused
tests passed). The experimental implementation remains attributable to Pierre
Lamy and is preserved only for reproducibility.

## Load-stability observation

Each post-shutdown cold reload could fail once inside `mlx_lm.utils.load_model`
at `mx.eval(model.parameters())` with Metal
`kIOGPUCommandBufferCallbackErrorTimeout`; one bounded retry then succeeded.
The failure occurred before cache creation in both bf16 and int8 modes, with no
other model resident and 96% memory free. This is independent of the
quantization result and deserves separate diagnosis.
