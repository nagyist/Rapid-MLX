# Qwen3.5-family continuous MTP qualification

Date: 2026-08-31  
Host: Mac Studio, M3 Ultra, 256 GB unified memory  
Code base: stacked on continuous-MTP foundation PR #2842 and Qwen dense
adapter PR #2854

## Decision

Continuous MTP is qualified per concrete target artifact. Sharing the
`qwen3_5` model type, MTP tensor ABI, and cache layout is necessary but not
sufficient evidence that batched target verification preserves the existing
greedy output. The catalog therefore records `verified`, `blocked`, or
`unknown`; ordinary MTP remains available for every existing alias regardless
of this continuous-batching tier.

An unverified artifact fails closed when a user requests
`continuous_batching=true`. `--force-spec-decode` remains an explicit operator
override for controlled experiments.

Continuous MTP also requires an unquantized BF16 KV cache for transactional
trim/restore. A verified alias's automatic cache-compression default yields to
that method requirement. Explicit `--kv-cache-turboquant`, legacy cache
quantization, or `--kv-cache-dtype int4|int8` requests fail before model load
with an actionable error instead of silently falling back to ordinary MTP.

## Design precedent

Production speculative schedulers register MTP support through an explicit
model implementation and self-describing checkpoint metadata, then validate
draft depth, quantization, cache ownership, and target verification at startup.
Rapid-MLX retains those runtime gates. The additional catalog tier records the
artifact-level evidence that runtime structure cannot prove: paired output
identity and a measured concurrent throughput win.

## Methodology

- one model resident; the task-owned server was stopped between conditions
- prefix cache disabled
- thinking disabled; temperature 0
- 567 prompt tokens, 128 completion tokens
- four simultaneous requests, three cohorts per condition
- aggregate decode rate = 512 completed tokens / cohort wall time
- every condition had 12/12 complete responses and one stable SHA-256 within
  the condition
- qualification requires the continuous and ordinary-MTP SHA-256 values to
  match as well as a positive throughput result

The checked-in client is `bench/bench_continuous_mtp_server.py`. Example:

```bash
python3.12 bench/bench_continuous_mtp_server.py \
  --label continuous \
  --model "$TARGET_MODEL" \
  --base-url http://127.0.0.1:8475/v1 \
  --runs 3 --concurrency 4 --max-tokens 128
```

The two server conditions differed only in the final two speculative-config
fields:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3.12 -m vllm_mlx.cli serve \
  "$TARGET_MODEL" --host 127.0.0.1 --port 8475 \
  --max-num-seqs 4 --max-concurrent-requests 4 \
  --disable-prefix-cache --no-thinking --force-spec-decode \
  --speculative-config \
  "{\"method\":\"mtp\",\"model\":\"$MTP_MODEL\",\"num_speculative_tokens\":2,\"disable_auto_k\":true,\"continuous_batching\":false,\"allow_dynamic_membership\":false}"
```

For the continuous condition, set `continuous_batching` and
`allow_dynamic_membership` to `true`.

## Results

| Target artifact | Target revision | MTP revision | Ordinary MTP aggregate | Continuous MTP aggregate | Change | Output gate | Tier |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| Qwen3.5-4B MLX 4-bit | `32f3e8ec` | `ab6f59bc` | 106.61 tok/s | 137.90 tok/s | +29.4% | SHA-256 mismatch | blocked |
| Qwen3.5-9B MLX 4-bit | `8b2b98c0` | `222dfd2c` | 77.99 tok/s | 92.35 tok/s | +18.4% | byte-identical | verified |
| Qwen3.6-27B MLX 4-bit | `c000ac2c` | `83795d54` | 28.26 tok/s | 32.90 tok/s | +16.4% | byte-identical | verified |
| Qwen3.8-27B MLX 4-bit MTP | `aa985c29` | self-contained | 24.33 tok/s | 28.48 tok/s | +17.1% | byte-identical | verified |

The Qwen3.5-4B result is a deliberate no-go despite its throughput gain. Its
continuous output was deterministic, target-verified, and semantically
plausible, but it did not preserve the existing greedy byte sequence. Other
quantizations and model sizes remain `unknown` until the same paired gate is
run on their exact artifacts.

Peak active/peak MLX memory observed for Qwen3.6-27B was approximately
17.7/19.6 GB for four continuous lanes and 16.3/17.2 GB for ordinary MTP.

The final alias-path check served `qwen3.5-9b-4bit` offline without a force
override. Startup selected BF16, installed the continuous coordinator, and a
four-request cohort completed through the `continuous_planned` route. The
task-owned server was then stopped.
