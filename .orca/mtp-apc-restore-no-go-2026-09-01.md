# Continuous MTP APC prepared-state restore: real-model NO-GO

Date: 2026-09-01  
Host: M3 Ultra Studio, 256 GB unified memory  
Candidate: `experiment/mtp-apc-restore@cefa1a08972c98f017a511d9c161d76ac4c194d3`  
Baseline: landed `main@186972242a5c5c17035a80c74ab735997a9dd41b`  
Model: `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX@aa985c29ff5b334cbfdcbbc787d47e66e9d9e456`

Pierre Lamy authored the extracted implementation. The experiment preserves
his commit authorship and is pushed only as an experimental branch; no Rapid
PR is opened because the production hot path did not meet the ROI gate.

## Contract and precedent

The candidate attaches speculative prepared state only when checkpoint,
runtime, prefix digest, confirmed token boundary, draft depth, and cache shape
all match. A mismatch fails open to ordinary target-prefix reuse. Proposal
scratch stays transaction-local and only state paired with a committed target
prefix becomes reusable. This follows the established scheduler pattern of an
exact prefix identity plus atomic sidecar commit; it does not add a second
cache authority.

## Reproduction

Both baseline and candidate used one resident model, prefix-cache autoload
disabled, thinking off, greedy sampling, BF16 cache, and fixed continuous MTP:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RAPID_MLX_PREFIX_CACHE_AUTOLOAD=0 \
python3.12 -m vllm_mlx.cli serve qwen3.8-27b-4bit \
  --host 127.0.0.1 --port 8481 \
  --max-num-seqs 4 --max-concurrent-requests 4 \
  --hybrid-cache-entries 4 --no-thinking --force-spec-decode \
  --speculative-config \
  '{"method":"mtp","continuous_batching":true,"allow_dynamic_membership":false}'
```

Each second turn reused the committed first-turn target prefix. The B=2 and
B=4 long second turns requested up to 256 output tokens.

## Results

| Cohort | Candidate APC evidence | Baseline second-turn wall | Candidate second-turn wall | Verdict |
| --- | --- | ---: | ---: | --- |
| B=1 | No speculative state capture; request stayed on mature single-lane route | 0.817 s short cached turn | not eligible | No product effect |
| B=2 | 2/2 eligible; captures and commits accepted | 8.714-8.858 s | 10.072-10.321 s | 15-18% slower |
| B=4 | 4/4 eligible; captures and commits accepted | 15.093-15.234 s | 16.916-17.929 s | 12-18% slower |

The cumulative candidate receipt reported 12/12 capture attempts accepted,
12/12 commits accepted, six eligible restores, eight retained entries, and no
attach failure or identity mismatch. The bridge therefore executed; the
negative result is not a routing miss.

First-turn outputs were byte-identical lane by lane. Restored second-turn
outputs remained coherent and task-correct, but were not byte-identical to the
ordinary path at close greedy boundaries; several 256-token runs ended at the
budget. No ordinary-pass/candidate-fail semantic outcome was observed.

## Decision

NO-GO for production and default routing. Reusing the prepared draft sidecar
does not offset restore/attachment and continuous transaction overhead on the
available dense MTP artifact. Keep the branch as evidence and do not open a PR
unless a later scheduler/kernel change reverses the B=2 and B=4 result under
the same-method comparison.

An independent B=1 continuous-route experiment follows on a clean branch; it
must not inherit this APC implementation.
