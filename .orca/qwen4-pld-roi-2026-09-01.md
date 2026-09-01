# Qwen3.8 Flash-Next prompt-lookup experiment (2026-09-01)

## Environment

- Host: Studio, M3 Ultra, 256 GB unified memory
- Base: `origin/main` at `82b703adc`
- Model: `rapid-mlx/Qwen3.8-Flash-Next-4bit`
- Revision: `dcf657e4acda2aae72da99cde65b6c491cd96998`
- Serve mode: text lane, thinking off, MTP fixed K=1, temperature 0
- Cache: cleared before every measured request
- Runs: median of 3, same process per variant
- Candidate: request-scoped prompt history plus prompt-lookup proposals, experimental

## Performance

| Workload | MTP-only TTFT | MTP+PLD TTFT | MTP-only decode | MTP+PLD decode | Decode delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact source copy | 2727.8 ms | 2652.5 ms | 38.29 tok/s | 55.01 tok/s | +43.7% |
| One-line code edit, full-file return | 2740.7 ms | 2692.8 ms | 38.22 tok/s | 64.61 tok/s | +69.1% |
| Original creative prose | 314.1 ms | 309.8 ms | 34.46 tok/s | 39.02 tok/s | +13.2% |
| Engineering chat | 287.4 ms | 283.7 ms | 36.48 tok/s | 41.97 tok/s | +15.0% |

The candidate is a **correctness no-go despite the apparent speedup**. The
exact-copy output first diverged after `transform_007`: the candidate emitted
`transform_009`, then 010, 012, 014, and continued skipping source rows. The
code-edit output likewise skipped 022 and then emitted 023 and 025. Each
variant was deterministic across its three runs, so this is not measurement
noise.

The likely failure class is a hybrid-state verification oracle mismatch. The
batched QSA/GatedDeltaNet verify path can produce a locally accepted proposal
without proving that every recurrent and sparse-attention state matches the
sequential target path. The full upstream contribution includes exact-state
attestation, request-scoped installation, checkpoint identity binding, and a
latched route specifically to prevent this shortcut. A production restack must
include those contracts; simply enabling the existing lookup index is unsafe.

## Load stability observation

Across clean shutdown/reload cycles, the first subsequent load sometimes
aborted inside `mlx_lm.utils.load_model` with Metal
`kIOGPUCommandBufferCallbackErrorTimeout`. It occurred before PLD code or any
request ran, with no second model resident and 96% system memory reported free.
One bounded retry loaded successfully. Treat this as an independent 100 GB
cold-load stability finding, not as PLD correctness evidence.

## Reproduction

Baseline:

```shell
RAPID_MLX_MTP_PROMPT_LOOKUP=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3.12 -m vllm_mlx.cli serve MODEL_SNAPSHOT \
  --served-model-name qwen3.8-flash-next-4bit --host 127.0.0.1 \
  --port 8465 --no-thinking \
  --speculative-config '{"method":"mtp","disable_auto_k":true}'

python3.12 .orca/qwen4-pld-benchmark.py \
  --url http://127.0.0.1:8465/v1 --label mtp-only \
  --output /private/tmp/qwen4-pld-mtp-only.json
```

Candidate uses the same commands with `RAPID_MLX_MTP_PROMPT_LOOKUP=1`.

## Decision

Do not ship or default-enable this experimental route. Restack the full exact
hybrid-state attestation and request-latching design, then repeat both the
correctness battery and this performance matrix before opening a product PR.
