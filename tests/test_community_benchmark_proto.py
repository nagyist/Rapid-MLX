# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the shared community-benchmark wire protocol."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")
referencing = pytest.importorskip("referencing")

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTO_ROOT = REPO_ROOT / "proto" / "community-benchmark" / "v1"
EXAMPLES_ROOT = PROTO_ROOT / "examples"

SCHEMA_FILES = (
    "model-identity.schema.json",
    "machine-observation.schema.json",
    "execution-config.schema.json",
    "benchmark-run.schema.json",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schemas() -> dict[str, dict]:
    return {name: _load(PROTO_ROOT / name) for name in SCHEMA_FILES}


@pytest.fixture(scope="module")
def registry(schemas):
    resources = (
        (schema["$id"], referencing.Resource.from_contents(schema))
        for schema in schemas.values()
    )
    return referencing.Registry().with_resources(resources)


def _validator(schema: dict, registry):
    return jsonschema.Draft202012Validator(
        schema,
        registry=registry,
        format_checker=jsonschema.FormatChecker(),
    )


def test_all_protocol_schemas_are_valid_draft_2020_12(schemas) -> None:
    for schema in schemas.values():
        jsonschema.Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_model_identity_example_validates(schemas, registry) -> None:
    example = _load(EXAMPLES_ROOT / "model-identity.example.json")
    _validator(schemas["model-identity.schema.json"], registry).validate(example)


def test_composed_benchmark_run_example_validates_offline(schemas, registry) -> None:
    example = _load(EXAMPLES_ROOT / "benchmark-run.example.json")
    _validator(schemas["benchmark-run.schema.json"], registry).validate(example)


@pytest.mark.parametrize(
    ("section", "forbidden_key", "forbidden_value"),
    (
        ("model", "local_path", "/Users/alice/models/qwen"),
        ("machine", "hostname", "alice-macbook"),
        ("execution", "environment", {"TOKEN": "secret"}),
    ),
)
def test_upload_rejects_fields_outside_privacy_allowlist(
    schemas, registry, section, forbidden_key, forbidden_value
) -> None:
    example = _load(EXAMPLES_ROOT / "benchmark-run.example.json")
    example[section][forbidden_key] = forbidden_value

    errors = list(
        _validator(schemas["benchmark-run.schema.json"], registry).iter_errors(example)
    )
    assert errors
    assert any("Additional properties are not allowed" in error.message for error in errors)


def test_client_cannot_upload_server_verdicts(schemas, registry) -> None:
    example = _load(EXAMPLES_ROOT / "benchmark-run.example.json")
    example["validation"] = {
        "verified": True,
        "comparable": True,
        "rank": 1,
    }

    errors = list(
        _validator(schemas["benchmark-run.schema.json"], registry).iter_errors(example)
    )
    assert errors
    assert any("validation" in error.message for error in errors)


def test_repository_identity_requires_immutable_revision(schemas, registry) -> None:
    example = _load(EXAMPLES_ROOT / "model-identity.example.json")
    del example["source"]["resolved_revision"]

    errors = list(
        _validator(schemas["model-identity.schema.json"], registry).iter_errors(
            example
        )
    )
    assert errors
    assert any("resolved_revision" in error.message for error in errors)


def test_local_identity_rejects_repository_coordinates(schemas, registry) -> None:
    example = _load(EXAMPLES_ROOT / "model-identity.example.json")
    example["identity_strength"] = "local_manifest"
    example["source"]["kind"] = "local"

    errors = list(
        _validator(schemas["model-identity.schema.json"], registry).iter_errors(
            example
        )
    )
    assert errors


def test_quantized_model_requires_method_and_weight_bits(schemas, registry) -> None:
    example = _load(EXAMPLES_ROOT / "model-identity.example.json")
    del example["quantization"]["method"]
    del example["quantization"]["weight_bits"]

    errors = list(
        _validator(schemas["model-identity.schema.json"], registry).iter_errors(
            example
        )
    )
    assert {"method", "weight_bits"}.issubset(
        {field for error in errors for field in ("method", "weight_bits") if field in error.message}
    )


def test_mtp_requires_draft_depth(schemas, registry) -> None:
    example = _load(EXAMPLES_ROOT / "benchmark-run.example.json")
    del example["execution"]["features"]["speculative_decoding"][
        "max_draft_tokens"
    ]

    errors = list(
        _validator(schemas["benchmark-run.schema.json"], registry).iter_errors(example)
    )
    assert errors
    assert any("max_draft_tokens" in error.message for error in errors)


def test_external_draft_method_requires_draft_artifact(schemas, registry) -> None:
    example = _load(EXAMPLES_ROOT / "benchmark-run.example.json")
    spec = example["execution"]["features"]["speculative_decoding"]
    spec["method"] = "dflash"

    errors = list(
        _validator(schemas["benchmark-run.schema.json"], registry).iter_errors(example)
    )
    assert errors
    assert any("draft_model_identity_digest" in error.message for error in errors)


def test_quantized_kv_cache_requires_bit_width(schemas, registry) -> None:
    example = _load(EXAMPLES_ROOT / "benchmark-run.example.json")
    del example["execution"]["features"]["kv_cache"]["bits"]

    errors = list(
        _validator(schemas["benchmark-run.schema.json"], registry).iter_errors(example)
    )
    assert errors
    assert any("bits" in error.message for error in errors)


def test_complete_machine_profile_requires_all_cohort_axes(schemas, registry) -> None:
    example = _load(EXAMPLES_ROOT / "benchmark-run.example.json")
    del example["machine"]["profile"]["gpu_cores"]

    errors = list(
        _validator(schemas["benchmark-run.schema.json"], registry).iter_errors(example)
    )
    assert errors
    assert any("gpu_cores" in error.message for error in errors)


def test_partial_machine_profile_remains_valid_exploratory_data(
    schemas, registry
) -> None:
    example = _load(EXAMPLES_ROOT / "benchmark-run.example.json")
    example["machine"]["profile_completeness"] = "partial"
    del example["machine"]["profile"]["gpu_cores"]
    del example["machine"]["profile"]["performance_cores"]
    del example["machine"]["profile"]["efficiency_cores"]
    _validator(schemas["benchmark-run.schema.json"], registry).validate(example)


def test_source_runtime_requires_immutable_revision(schemas, registry) -> None:
    example = _load(EXAMPLES_ROOT / "benchmark-run.example.json")
    runtime = example["execution"]["runtime"]
    runtime["distribution"] = "source"

    errors = list(
        _validator(schemas["benchmark-run.schema.json"], registry).iter_errors(example)
    )
    assert errors
    assert any("rapid_mlx_revision" in error.message for error in errors)


def test_failed_outcome_is_structured_and_does_not_require_measurements(
    schemas, registry
) -> None:
    example = _load(EXAMPLES_ROOT / "benchmark-run.example.json")
    example["outcome"] = {
        "status": "failed",
        "failure_code": "model_load_oom",
    }
    del example["measurements"]
    _validator(schemas["benchmark-run.schema.json"], registry).validate(example)

    example["outcome"]["error_message"] = "/Users/alice/private-model failed"
    errors = list(
        _validator(schemas["benchmark-run.schema.json"], registry).iter_errors(example)
    )
    assert errors
    assert any("error_message" in error.message for error in errors)


def test_completed_outcome_requires_measurements(schemas, registry) -> None:
    example = _load(EXAMPLES_ROOT / "benchmark-run.example.json")
    del example["measurements"]
    errors = list(
        _validator(schemas["benchmark-run.schema.json"], registry).iter_errors(example)
    )
    assert errors
    assert any("measurements" in error.message for error in errors)


def test_randomized_experiment_requires_assignment_seed(schemas, registry) -> None:
    example = _load(EXAMPLES_ROOT / "benchmark-run.example.json")
    del example["experiment"]["assignment_seed"]
    errors = list(
        _validator(schemas["benchmark-run.schema.json"], registry).iter_errors(example)
    )
    assert errors
    assert any("assignment_seed" in error.message for error in errors)


def test_experiment_can_vary_only_execution_fields(schemas, registry) -> None:
    example = _load(EXAMPLES_ROOT / "benchmark-run.example.json")
    example["experiment"]["varied_fields"] = ["/machine/profile/memory_gib"]
    errors = list(
        _validator(schemas["benchmark-run.schema.json"], registry).iter_errors(example)
    )
    assert errors
    assert any("does not match" in error.message for error in errors)


def test_machine_profile_digest_does_not_change_with_run_conditions() -> None:
    """The normative digest projection is profile-only, not a device ID."""
    example = _load(EXAMPLES_ROOT / "benchmark-run.example.json")
    before = copy.deepcopy(example["machine"]["profile"])
    example["machine"]["conditions_after"]["thermal_state"] = "serious"
    example["machine"]["conditions_after"]["available_memory_mib"] = 1024
    assert example["machine"]["profile"] == before


def _digest(value: object) -> str:
    """RFC 8785-equivalent for the integer/string/bool/null v1 golden vectors."""
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def test_cross_language_digest_golden_vectors() -> None:
    example = _load(EXAMPLES_ROOT / "benchmark-run.example.json")
    model = example["model"]
    model_projection = {
        key: model[key]
        for key in ("schema_version", "source", "artifact", "quantization")
    }
    execution = example["execution"]
    execution_projection = {
        key: execution[key] for key in ("load", "generation", "features")
    }

    assert _digest(model_projection) == model["identity_digest"]
    assert _digest(example["machine"]["profile"]) == example["machine"][
        "profile_digest"
    ]
    assert _digest(execution_projection) == execution["config_digest"]


def test_example_measurements_match_declared_cases() -> None:
    """Pin the cross-array semantic invariant used by future ingestion."""
    example = _load(EXAMPLES_ROOT / "benchmark-run.example.json")
    cases = {case["case_id"]: case for case in example["workload"]["cases"]}
    seen: set[tuple[str, int]] = set()

    for measurement in example["measurements"]:
        case_id = measurement["case_id"]
        assert case_id in cases
        pair = (case_id, measurement["round_index"])
        assert pair not in seen
        seen.add(pair)

    for case_id, case in cases.items():
        assert sum(measurement["case_id"] == case_id for measurement in example["measurements"]) == case[
            "measured_rounds"
        ]
