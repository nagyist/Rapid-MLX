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

The only local Flash-Next 4-bit snapshot used by earlier qualification is no
longer present in the HF cache. RTL-2T contains the 335 GB unquantized source
artifact, which cannot be loaded on the 192 GB host. No product PR should be
opened from this branch until the 4-bit artifact is restored and the real-model
gate above passes.
