# Community benchmark protocol

This directory is the cross-product source of truth for Rapid-MLX community
benchmark payloads. Server, CLI, Desktop, and website code consume the same
versioned JSON Schema contracts rather than maintaining product-specific field
lists.

The folder is named `proto` in the sense of a shared wire protocol. The wire
format is JSON, not Protocol Buffers, because submissions are public JSON and
the website must be able to validate and render them without a binary codec.

## Versions

- [`v1/model-identity.schema.json`](v1/model-identity.schema.json) identifies
  the concrete weights, revision, format, and quantization.
- [`v1/machine-observation.schema.json`](v1/machine-observation.schema.json)
  separates a reusable hardware profile from volatile before/after conditions.
- [`v1/execution-config.schema.json`](v1/execution-config.schema.json) records
  the effective runtime, load, generation, MTP/speculative, KV-cache, prefix
  cache, and prefill configuration.
- [`v1/benchmark-run.schema.json`](v1/benchmark-run.schema.json) composes the
  three reusable contracts with workload and raw measurements.

Once a version has shipped, its accepted meaning is immutable. Additive fields
still require a new version directory: `additionalProperties: false` is a
deliberate privacy allowlist, so silently widening an old schema would silently
widen user consent. Readers may support several version directories during a
migration.

The existing `community-benchmarks/schema.json` v1-v3 contract remains the
production wire format until a separate adapter/rollout PR switches producers
and ingestion. These protocol files define that migration target; they do not
change today's CLI upload behavior.

## Two validation layers

JSON Schema enforces types, bounds, required fields, feature conditionals, and
the public-data allowlist. Ingestion must also run semantic validation that JSON
Schema cannot express cleanly:

1. Recompute every digest instead of trusting the client value.
2. Require `completed_at >= started_at` and a bounded run duration.
3. Require every measurement `case_id` to exist in `workload.cases`.
4. Require unique `(case_id, round_index)` pairs and the declared number of
   measured rounds for every case.
5. Require observed token counts to satisfy the protocol tolerance and reject
   incomplete samples from speed aggregates.
6. Check `performance_cores + efficiency_cores == cpu_cores` when both optional
   core counts are present.
7. Apply deduplication, anomaly, correctness, identity, and trust checks on the
   server.
8. Require `total_duration_ms >= ttft_ms + decode_duration_ms` within the
   protocol's documented timer tolerance.
9. For an experiment group, require exactly one baseline; identical model,
   machine profile, runtime stack, workload, and collector version; unique arms
   and sequence indexes; close timestamps; and effective configs that differ
   only at `varied_fields`. Reject a causal speedup when thermal, power, or
   memory-pressure drift crosses the protocol threshold.

The client cannot upload `verified`, `comparable`, `outlier`, rank, or a derived
TPS summary. Those are server conclusions. The client uploads raw observations;
the server derives display and recommendation evidence.

A failed run uploads only a closed `outcome.failure_code` and any earlier
completed samples; it never uploads exception text. Failures are censored
model-fit evidence (especially OOM), not zero-speed samples and never members of
a speed aggregate. Recording them prevents successful-run survivorship bias in
model recommendations.

`identity_strength: unresolved` deliberately remains valid so an arbitrary
compatible model can participate as exploratory data. It has no
`identity_digest`, cannot enter a formal comparison cohort, and cannot support a
promoted recommendation until ingestion or the registry resolves it. This is a
campaign participation lane, not a fail-open trust verdict.

## Canonical digests

Canonicalize each projection with JSON Canonicalization Scheme (RFC 8785), hash
the UTF-8 bytes with SHA-256, encode lowercase hexadecimal, and prefix the result
with `sha256:`.

The three digest values in
[`v1/examples/benchmark-run.example.json`](v1/examples/benchmark-run.example.json)
are normative cross-language golden vectors. A consumer must reproduce them
byte-for-byte before it can emit v1 payloads.

| Digest | Exact v1 projection | Excluded on purpose |
| --- | --- | --- |
| `model.identity_digest` | `{schema_version, source, artifact, quantization}` | `display`, `family`, `identity_strength` |
| `machine.profile_digest` | `machine.profile` | OS, run conditions, install ID |
| `execution.config_digest` | `{load, generation, features}` | runtime versions |
| `workload.protocol_digest` | Published protocol document selected by `protocol_id` + `protocol_version` | Result measurements |

Artifact manifests have two closed v1 bases:

- `huggingface_tree`: RFC 8785 array sorted by repository-relative POSIX path;
  each entry is `{path, size_bytes, blob_oid}` from the immutable resolved
  revision. Paths are NFC-normalized and must not contain `..`.
- `content_sha256`: RFC 8785 array sorted by model-root-relative POSIX path;
  each required config, tokenizer, and weight entry is
  `{path, size_bytes, sha256}`. It hashes file content, never the absolute model
  root. Local models must use this basis.

The uploaded artifact contains only the resulting manifest digest, not manifest
paths. The full manifest stays local unless a future, separately consented
contract adds it.

Display aliases and family metadata remain searchable, but never decide whether
two artifacts are identical. Machine profile digests are deliberately shared by
all Macs with the same declared configuration and must never contain serials,
hardware UUIDs, MAC addresses, hostnames, or usernames.

## Comparison and recommendation keys

The server derives keys; clients do not upload them:

```text
artifact key      = model.identity_digest
machine class key = machine.profile_digest
environment key   = machine.os
execution key     = execution.config_digest
runtime key       = execution.runtime
workload key      = workload.protocol_digest + case_id

comparison cohort = artifact + machine + environment + execution + runtime + workload
```

Recommendation uses the same evidence without conflating model selection and
runtime tuning:

```text
model-fit evidence:
  artifact × machine profile × workload

runtime-profile evidence:
  artifact × machine profile × workload -> ranked execution configs
```

Available memory, power, thermal state, and memory pressure are safety and
quality filters. They do not mutate the stable machine class. A promoted
recommendation also remains gated by runtime compatibility and correctness; raw
community results never write production defaults directly.

Only `profile_completeness: complete` enters a formal machine cohort. Partial
profiles remain visible exploratory evidence and cannot support a promoted
machine-specific recommendation.

## Privacy boundary

The schemas do not admit prompt text, output text, free-form notes, file paths,
environment variables, usernames, hostnames, IP addresses, hardware serials, or
hardware UUIDs. `install_id`, when present, is random and resettable and is not
part of model, machine, execution, or comparison identity. A client must obtain
explicit submission consent before including it; ingestion must publish a
retention/rotation policy and must not expose the raw value on the public site.
