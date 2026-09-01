# Qwen4 fused-down extraction microbenchmark

Date: 2026-09-01 (America/Los_Angeles)

Owner: Vector. Kernel implementation and full source credit: Pierre Lamy,
extracted from community PR #2809 commit
`eef68235bd1e2a12c74ce8f59cff4e2a5e6da8b8` without product wiring.

## Question

Does the proposed fixed-shape Metal kernel have enough isolated latency benefit
and numerical fidelity to justify restacking it onto current main for a real
Flash-Next qualification?

## Method

- Host: Mac Studio, Apple silicon, no model resident.
- Base: `origin/main` `186972242a5c5c17035a80c74ab735997a9dd41b`.
- Production geometry: 512 experts, top-k 10, hidden 640, output 2560,
  affine q4 group size 64, bf16 activations/scales/biases.
- Full 472,216,812-byte synthetic expert table was resident on Metal.
- Reference: stock `mx.gather_qmm`, bf16 router weighting, then top-k reduction.
- Candidate: Pierre's `scalar` and `tile4` fused down+weight+reduce kernels.
- Deterministic non-zero packed weights, random bf16 activations and normalized
  router scores, fixed seed 20260901.
- 25 warmups and 200 timed/evaluated iterations per path.

## Result

| Width | Candidate | Stock median | Fused median | Isolated speedup | Tensor fidelity |
| ---: | --- | ---: | ---: | ---: | --- |
| M=1 | scalar | 0.304 ms | 0.283 ms | 1.07x | not exact; max abs 0.0009766 |
| M=1 | tile4 | 0.304 ms | 0.279 ms | 1.09x | not exact; max abs 0.0009766 |
| M=3 | scalar | 0.380 ms | 0.326 ms | 1.17x | exact on this vector |

The M=1 relative L2 delta was 0.00441 on this deliberately low-magnitude
synthetic vector. The difference is consistent with a different bf16 dot/reduce
order, but that does not prove model-level losslessness. Short 15-run trials
were much noisier and overstated the candidate at 1.36-1.67x; the 200-run table
is the decision input.

## Current verdict

Promising only as an isolated kernel experiment, not a product PR. The stable
microbenchmark gain is 7-17%, not an end-to-end claim, and M=1 is not tensor
bit-exact. Production work remains blocked on a real-model ladder that proves:

1. every intended layer and M=1/M=3 route actually dispatches;
2. five deterministic user outputs are byte-identical or any divergence passes
   the agreed quality battery;
3. end-to-end decode improves materially against the exact current-main stock
   path, not only against a component microbenchmark;
4. unsupported widths, layouts, quantization, training, and CPU routes fall
   back structurally; and
5. the explicit hard-off escape hatch restores the stock graph.

At the time of the isolated microbenchmark, the local 4-bit snapshot was
missing and only a 335 GB unquantized source cache remained. That source cache
was removed as a recoverable, unloadable artifact before the real-model stage;
the exact 4-bit revision was then restored on RTL-2T.

## Real-model qualification

The immutable 4-bit artifact was restored to the standard Hub cache on
RTL-2T and every file passed the artifact's `SHA256SUMS.txt`. The exact
revision was `dcf657e4acda2aae72da99cde65b6c491cd96998` (28 safetensor shards,
104,682,036,373 bytes). No other model process was resident.

The baseline was exact current main `82b703adcb712beb80ee1b2d894444bb18eeec95`.
The candidate was `1279709ebb23a8deb4c46691ae6cd80330151129` with
`MLX_QWEN4_FUSED_EXPERT_KERNEL=auto`; it selects Pierre's `tile4` kernel for
ordinary M=1 decode. Both variants used the same Python environment, checkpoint,
server command, prefix-cache clearing, 256-token decode, and three-run median
methodology from `.orca/flash-next-eval/benchmark.py`.

| Target prompt | Stock TTFT | Fused TTFT | Stock decode | Fused decode | Decode delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.419 s | 0.425 s | 24.827 tok/s | 24.665 tok/s | -0.65% |
| 2,048 | 2.426 s | 2.432 s | 23.458 tok/s | 22.430 tok/s | -4.38% |
| 8,192 | 9.713 s | 9.702 s | 22.810 tok/s | 22.533 tok/s | -1.21% |
| 32,768 | 46.510 s | 46.637 s | 20.984 tok/s | 21.015 tok/s | +0.15% |

Peak RSS was unchanged at about 54.4 GiB. The two deterministic capture prompts
remained coherent and task-equivalent, but their content hashes differed from
stock (first token divergence at generated positions 5 and 64). Because the
candidate does not provide a material end-to-end speedup, the full correctness
battery is unnecessary: performance already fails the product gate.

One first candidate reload hit a Metal command-buffer timeout inside
`mlx_lm.utils.load_model`, before the candidate kernel compiled or executed.
After the terminated process released all memory (96% system memory free), one
bounded retry loaded and completed the full benchmark. This is retained as
environment evidence, not attributed to fused-down execution.

## Reference check and final verdict

Private precedent inspection covered the primary serving engines' fused-MoE
dispatch, configuration, native fallback, and per-shape tuning paths, plus the
MLX-native switch-layer `gather_qmm` implementation. The pattern adopted for
the experiment was narrow structural admission with a stock fallback and a
shape-specific variant. The real model demonstrates the missing requirement:
an isolated-kernel win is insufficient when the replacement introduces a graph
boundary that prevents surrounding work from scheduling efficiently.

**Final verdict: no-go.** The 7-17% isolated down-projection win becomes
-4.38% to +0.15% end to end. Do not open a product PR and do not enable the
kernel by default. Revisit only if a future fused-MoE primitive can compose the
gate/up, activation, down projection, router weighting, and reduction without
the extra scheduling boundary.
