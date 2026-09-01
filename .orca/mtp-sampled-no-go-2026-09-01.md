# Continuous self-MTP transformed-sampling ROI — no-go

Date: 2026-09-01 (America/Los_Angeles)

Owner: Vector. Source implementation and credit: Pierre Lamy, extracted from
community PR #2809. Experiment branch: `experiment/mtp-sampled-roi`, source
commit `8aa72bd4e16d4e506edc72fbc4c32a1fc0df7f1b`.

## Question

Does the transformed-distribution verifier make continuous self-MTP beneficial
for a common non-greedy profile, before investing in production per-request RNG
and mixed-sampling-profile support?

## Reference contract

Primary serving precedents require speculative sampling to verify target and
draft distributions with residual sampling. They also isolate random state per
request; a process-global RNG is not batch-invariant. The source hook implements
the exact residual-distribution algorithm, but this experiment intentionally
used one fixed sampling profile and the process MLX RNG. That is sufficient for
an ROI measurement only and is not a production-safe routing contract.

## Environment

- Host: Mac Studio, Apple silicon, model resident alone.
- Model: `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX` from the existing offline cache.
- Base: `origin/main` `186972242a5c5c17035a80c74ab735997a9dd41b`.
- Candidate: source commit `8aa72bd4e16d4e506edc72fbc4c32a1fc0df7f1b` plus an in-process experiment-only
  runtime capability wrapper; no scheduler/product wiring was committed.
- Prefix cache and APC autoload disabled.
- Fixed cohort B4, dynamic membership disabled, three runs.
- Each lane: 603 prompt tokens, 128 requested output tokens, thinking disabled,
  `temperature=0.7`, `top_p=0.9`, `top_k=20`.

## Result

| Variant | Median B4 wall | Median aggregate output tok/s | Relative |
| --- | ---: | ---: | ---: |
| Main ordinary sampled batching | 13.323 s | 38.43 | baseline |
| Continuous self-MTP transformed verifier | 17.037 s | 30.05 | -21.8% |

Candidate cohorts were 29.92, 30.05, and 30.58 aggregate output tok/s.
All 12 responses emitted 128 tokens and obeyed their lane-specific required
first line. Later requested class declarations did not appear within the common
128-token truncation window, so that observation is inconclusive rather than a
candidate-only quality failure.

The candidate was slower by 3.71 seconds per B4 cohort and 21.8% in aggregate
throughput. The implementation therefore does not justify the additional
per-request sampling metadata/RNG integration work and must not be enabled or
opened as a product PR in its present form.

## Local verification

- `99 passed` across the residual-sampling, continuous backend, continuous
  engine, and router contracts.
- Ruff and `git diff --check` passed on the extracted implementation.
- Server health confirmed the exact model loaded; it was stopped after the
  experiment and port 8483 was released.

## Verdict

No-go. Preserve the source branch and evidence, but do not merge or default-on
this transformed-sampling continuous path. Any future revisit must first show a
positive B2/B4 ROI and then add per-request RNG, mixed-profile routing, seeded
reproducibility, cancellation, and distribution-level correctness evidence.
