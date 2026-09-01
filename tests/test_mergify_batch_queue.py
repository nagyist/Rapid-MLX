# SPDX-License-Identifier: Apache-2.0
"""Fail-closed contracts for the managed batch merge queue."""

import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / ".mergify.yml"
REQUIRED_CHECKS = {
    "check-success = @github-actions/tests",
    "check-success = @github-actions/desktop-tests",
    "check-success = @github-actions/version-bump-guard",
}
HEAD_AUTHORIZATION = "check-success = merge-ready-head"
REQUEUE_TRIGGER = "label = merge-requeue-trigger"
REQUEUE_REQUIRED = "label = merge-requeue-required"
LANE_CHECKS = {
    "no-mac-batch": "check-success = @github-actions/merge-lane-no-mac",
    "mac-batch": "check-success = @github-actions/merge-lane-mac",
}


def _config() -> dict[str, object]:
    return yaml.safe_load(CONFIG.read_text())


def _rules_by_name(kind: str) -> dict[str, dict[str, object]]:
    return {rule["name"]: rule for rule in _config()[kind]}


def test_queue_batches_four_ready_prs_after_a_bounded_wait():
    config = _config()
    queue = config["merge_queue"]
    rules = _rules_by_name("queue_rules")

    assert queue["mode"] == "serial"
    assert queue["max_parallel_checks"] == 1
    assert queue["skip_intermediate_results"] is False
    assert set(rules) == {"no-mac-batch", "mac-batch"}
    assert rules["no-mac-batch"]["batch_size"] == 4
    assert rules["no-mac-batch"]["batch_max_wait_time"] == "5 min"
    assert rules["mac-batch"]["batch_size"] == 4
    assert rules["mac-batch"]["batch_max_wait_time"] == "15 min"
    assert {rule["checks_timeout"] for rule in rules.values()} == {"90 min"}


def test_queue_revalidates_every_required_check_on_the_combined_batch():
    rules = _rules_by_name("queue_rules")

    for name, rule in rules.items():
        assert set(rule["queue_conditions"]) >= REQUIRED_CHECKS
        assert HEAD_AUTHORIZATION in rule["queue_conditions"]
        assert LANE_CHECKS[name] in rule["queue_conditions"]
        assert not ({*LANE_CHECKS.values()} - {LANE_CHECKS[name]}) & set(
            rule["queue_conditions"]
        )
        assert set(rule["merge_conditions"]) == REQUIRED_CHECKS
        assert HEAD_AUTHORIZATION not in rule["merge_conditions"]
        assert not set(LANE_CHECKS.values()) & set(rule["merge_conditions"])
        assert rule["branch_protection_injection_mode"] == "queue"
        assert rule["queue_branch_prefix"] == "mergify/merge-queue/"


def test_ready_labels_autoqueue_without_enabling_blind_retries():
    config = _config()
    queues = _rules_by_name("queue_rules")
    auto_merge = config["merge_protections_settings"]["auto_merge_conditions"]
    enqueue_rules = _rules_by_name("pull_request_rules")

    assert auto_merge == [{"or": ["label = merge-ready", "label = merge-ready-mac"]}]
    assert set(enqueue_rules) == {
        "enqueue an explicitly authorized no-Mac head",
        "enqueue an explicitly authorized Mac head",
    }

    expected_enqueue = {
        "enqueue an explicitly authorized no-Mac head": (
            {"label = merge-ready", "-label = merge-ready-mac"},
            "no-mac-batch",
        ),
        "enqueue an explicitly authorized Mac head": (
            {"label = merge-ready-mac", "-label = merge-ready"},
            "mac-batch",
        ),
    }
    for name, (labels, queue_name) in expected_enqueue.items():
        rule = enqueue_rules[name]
        conditions = set(rule["conditions"])
        assert labels <= conditions
        assert {
            "base = main",
            "-draft",
            "-from-fork",
            REQUEUE_REQUIRED,
            REQUEUE_TRIGGER,
            "-label = dequeued",
            HEAD_AUTHORIZATION,
        } <= conditions
        assert rule["actions"] == {
            "queue": {"name": queue_name},
            "label": {"remove": ["merge-requeue-trigger"]},
        }

    expected_labels = {
        "no-mac-batch": {"label = merge-ready", "-label = merge-ready-mac"},
        "mac-batch": {"label = merge-ready-mac", "-label = merge-ready"},
    }
    for name, queue_rule in queues.items():
        assert expected_labels[name] <= set(queue_rule["queue_conditions"])
        assert "-from-fork" in queue_rule["queue_conditions"]
        assert queue_rule["max_checks_retries"] == 0
        assert queue_rule["batch_max_failure_resolution_attempts"] == 2


def test_ready_labels_are_mutually_exclusive_in_every_rule():
    for rule in _config()["queue_rules"]:
        conditions = set(rule["queue_conditions"])
        assert {"label = merge-ready", "-label = merge-ready-mac"} <= conditions or {
            "label = merge-ready-mac",
            "-label = merge-ready",
        } <= conditions


def test_release_bumps_cannot_enter_the_general_batch_queue():
    config = _config()
    exclusions = {
        "-label = version-bump",
        "-label = skip-version-bump",
        "-title ~= ^chore: bump version to ",
    }

    for queue_rule in config["queue_rules"]:
        assert exclusions <= set(queue_rule["queue_conditions"])
        assert queue_rule["merge_method"] == "squash"


def test_head_updates_revoke_both_merge_ready_authorizations():
    workflow = yaml.load(
        (ROOT / ".github/workflows/revoke-merge-ready.yml").read_text(),
        Loader=yaml.BaseLoader,
    )

    assert workflow["on"] == {"pull_request_target": {"types": ["synchronize"]}}
    assert workflow["permissions"] == {}

    job = workflow["jobs"]["revoke-merge-ready"]
    assert "merge-ready" in job["if"]
    assert "merge-ready-mac" in job["if"]
    assert "merge-requeue-required" in job["if"]
    assert "merge-requeue-trigger" in job["if"]
    assert job["permissions"] == {"issues": "write", "pull-requests": "read"}

    (step,) = job["steps"]
    assert step["uses"].startswith("actions/github-script@")
    script = step["with"]["script"]
    assert '"merge-requeue-required"' in script
    assert '"merge-requeue-trigger"' in script
    assert "github.rest.pulls.get" in script
    assert "github.paginate" in script
    assert "github.rest.issues.listEvents" in script
    assert "context.payload.pull_request.updated_at" in script
    assert "github.rest.issues.removeLabel" in script
    assert "checkout" not in script.lower()


def _run_revocation_script(
    *,
    labels: list[str],
    remove_errors: dict[str, int] | None = None,
    sync_updated_at: str = "2026-09-01T04:00:00Z",
    label_event_at: str = "2026-09-01T03:59:00Z",
    label_event_times: dict[str, str] | None = None,
    event_head: str = "head-sha",
    live_head: str = "head-sha",
) -> list[list[str]]:
    """Execute the exact revocation github-script against deterministic mocks."""

    workflow = yaml.load(
        (ROOT / ".github/workflows/revoke-merge-ready.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    script = workflow["jobs"]["revoke-merge-ready"]["steps"][0]["with"]["script"]
    scenario = json.dumps(
        {
            "labels": labels,
            "removeErrors": remove_errors or {},
            "syncUpdatedAt": sync_updated_at,
            "labelEventAt": label_event_at,
            "labelEventTimes": label_event_times or {},
            "eventHead": event_head,
            "liveHead": live_head,
        }
    )
    harness = f"""
const scenario = {scenario};
const calls = [];
const context = {{
  repo: {{ owner: "owner", repo: "repo" }},
  issue: {{ number: 42 }},
  payload: {{
    pull_request: {{
      labels: scenario.labels.map((name) => ({{ name }})),
      head: {{ sha: scenario.eventHead }},
      updated_at: scenario.syncUpdatedAt,
    }},
  }},
}};
const github = {{
  paginate: async () => {{
    calls.push(["events"]);
    return scenario.labels.map((name) => ({{
      event: "labeled",
      label: {{ name }},
      created_at: scenario.labelEventTimes[name] || scenario.labelEventAt,
    }}));
  }},
  rest: {{
    pulls: {{ get: async () => {{
      calls.push(["get"]);
      return {{ data: {{
        head: {{ sha: scenario.liveHead }},
        labels: scenario.labels.map((name) => ({{ name }})),
      }} }};
    }} }},
    issues: {{ removeLabel: async (args) => {{
      calls.push(["remove", args.name]);
      const status = scenario.removeErrors[args.name];
      if (status) throw Object.assign(new Error(`remove ${{args.name}} failure`), {{ status }});
    }} }},
  }},
}};
const core = {{ notice: (message) => calls.push(["notice", message]) }};
(async () => {{
  try {{
    await (async () => {{
{script}
    }})();
  }} catch (error) {{
    calls.push(["threw", error.message]);
  }}
  process.stdout.write(JSON.stringify(calls));
}})();
"""
    completed = subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_head_update_revocation_continues_after_concurrent_404():
    labels = [
        "merge-ready-mac",
        "merge-requeue-required",
        "merge-requeue-trigger",
    ]
    calls = _run_revocation_script(
        labels=labels, remove_errors={"merge-ready-mac": 404}
    )

    assert [call for call in calls if call[0] == "remove"] == [
        ["remove", "merge-ready-mac"],
        ["remove", "merge-requeue-required"],
        ["remove", "merge-requeue-trigger"],
    ]
    assert all(call[0] != "threw" for call in calls)


def test_head_update_revocation_propagates_non_404_failure():
    calls = _run_revocation_script(
        labels=["merge-ready-mac", "merge-requeue-required"],
        remove_errors={"merge-ready-mac": 500},
    )

    assert calls == [
        ["get"],
        ["events"],
        ["remove", "merge-ready-mac"],
        ["threw", "remove merge-ready-mac failure"],
    ]


def test_revocation_refreshes_generation_before_each_label_delete():
    calls = _run_revocation_script(
        labels=["merge-ready-mac", "merge-requeue-required"],
        label_event_times={
            "merge-ready-mac": "2026-09-01T03:59:00Z",
            "merge-requeue-required": "2026-09-01T04:01:00Z",
        },
    )

    assert [call for call in calls if call[0] == "remove"] == [
        ["remove", "merge-ready-mac"]
    ]
    assert [call[0] for call in calls].count("events") == 2
    assert any(
        call
        == [
            "notice",
            "Preserved merge-requeue-required; it was applied after this head update.",
        ]
        for call in calls
    )


def test_stale_synchronize_event_cannot_touch_a_newer_head():
    calls = _run_revocation_script(
        labels=["merge-ready-mac", "merge-requeue-required"],
        live_head="newer-head",
    )

    assert calls == [
        ["get"],
        ["notice", "A newer head exists; this synchronize event is stale."],
    ]


def test_ready_authorization_is_bound_to_the_exact_head_commit():
    workflow = yaml.load(
        (ROOT / ".github/workflows/authorize-merge-ready.yml").read_text(),
        Loader=yaml.BaseLoader,
    )

    assert workflow["on"] == {"pull_request_target": {"types": ["labeled"]}}
    assert workflow["permissions"] == {}

    job = workflow["jobs"]["authorize-ready-head"]
    assert "head.repo.full_name == github.repository" in job["if"]
    assert "merge-ready" in job["if"]
    assert "merge-ready-mac" in job["if"]
    assert job["permissions"] == {
        "issues": "write",
        "pull-requests": "read",
        "statuses": "write",
    }

    (step,) = job["steps"]
    assert step["uses"].startswith("actions/github-script@")
    script = step["with"]["script"]
    assert "github.rest.repos.createCommitStatus" in script
    assert "sha: context.payload.pull_request.head.sha" in script
    assert 'context: "merge-ready-head"' in script
    assert "present.length === 1" in script
    assert "GITHUB_RUN_ATTEMPT" in script
    assert "github.rest.pulls.get" in script
    assert "livePull.head.sha === context.payload.pull_request.head.sha" in script
    assert 'currentLabels.has("dequeued")' in script
    assert 'name: "dequeued"' in script
    assert 'labels: ["merge-requeue-required", "merge-requeue-trigger"]' in script
    assert "github.paginate" in script
    assert "github.rest.issues.listEvents" in script
    assert "readyLabeledAt > dequeuedLabeledAt" in script
    assert "authorized &&" in script
    assert "error.status !== 404" in script
    assert "checkout" not in script.lower()


def _run_authorization_script(
    *,
    labels: list[str],
    run_attempt: int = 1,
    event_label: str = "merge-ready-mac",
    live_head: str = "head-sha",
    fail_status_call: int | None = None,
    fail_get: bool = False,
    fail_add: bool = False,
    remove_error_status: int | None = None,
    ready_event_at: str = "2026-09-01T04:01:00Z",
    dequeue_event_at: str = "2026-09-01T04:00:00Z",
    late_dequeue_event_at: str | None = None,
    fail_events: bool = False,
) -> dict[str, object]:
    """Execute the exact github-script body against deterministic API mocks."""

    workflow = yaml.load(
        (ROOT / ".github/workflows/authorize-merge-ready.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    script = workflow["jobs"]["authorize-ready-head"]["steps"][0]["with"]["script"]
    scenario = json.dumps(
        {
            "labels": labels,
            "runAttempt": run_attempt,
            "eventLabel": event_label,
            "liveHead": live_head,
            "failStatusCall": fail_status_call,
            "failGet": fail_get,
            "failAdd": fail_add,
            "removeErrorStatus": remove_error_status,
            "readyEventAt": ready_event_at,
            "dequeueEventAt": dequeue_event_at,
            "lateDequeueEventAt": late_dequeue_event_at,
            "failEvents": fail_events,
        }
    )
    harness = f"""
const scenario = {scenario};
const calls = [];
let statusCalls = 0;
let eventCalls = 0;
process.env.GITHUB_RUN_ATTEMPT = String(scenario.runAttempt);
const context = {{
  repo: {{ owner: "owner", repo: "repo" }},
  issue: {{ number: 42 }},
  serverUrl: "https://github.example",
  payload: {{
    label: {{ name: scenario.eventLabel }},
    pull_request: {{ head: {{ sha: "head-sha" }} }},
  }},
}};
const github = {{
  paginate: async () => {{
    calls.push(["events"]);
    eventCalls += 1;
    if (scenario.failEvents) throw Object.assign(new Error("events failure"), {{ status: 500 }});
    const dequeueAt = eventCalls > 1 && scenario.lateDequeueEventAt
      ? scenario.lateDequeueEventAt
      : scenario.dequeueEventAt;
    return [
      {{ event: "labeled", label: {{ name: scenario.eventLabel }}, created_at: scenario.readyEventAt }},
      {{ event: "labeled", label: {{ name: "dequeued" }}, created_at: dequeueAt }},
    ];
  }},
  rest: {{
  pulls: {{ get: async () => {{
    calls.push(["get"]);
    if (scenario.failGet) throw Object.assign(new Error("get failure"), {{ status: 500 }});
    return {{ data: {{
      head: {{ sha: scenario.liveHead }},
      labels: scenario.labels.map((name) => ({{ name }})),
    }} }};
  }} }},
  repos: {{ createCommitStatus: async (args) => {{
    statusCalls += 1;
    calls.push(["status", args.state]);
    if (scenario.failStatusCall === statusCalls) throw Object.assign(new Error("status failure"), {{ status: 500 }});
  }} }},
  issues: {{
    removeLabel: async () => {{
      calls.push(["remove"]);
      if (scenario.removeErrorStatus) throw Object.assign(new Error("remove failure"), {{ status: scenario.removeErrorStatus }});
    }},
    addLabels: async (args) => {{
      calls.push(["add", args.labels]);
      if (scenario.failAdd) throw Object.assign(new Error("add failure"), {{ status: 500 }});
    }},
  }},
}} }};
const core = {{
  setFailed: (message) => calls.push(["failed", message]),
  notice: (message) => calls.push(["notice", message]),
}};
(async () => {{
  try {{
    await (async () => {{
{script}
    }})();
  }} catch (error) {{
    calls.push(["threw", error.message]);
  }}
  process.stdout.write(JSON.stringify(calls));
}})();
"""
    completed = subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"calls": json.loads(completed.stdout)}


def test_dequeued_head_is_blocked_before_marker_removal_then_reauthorized():
    result = _run_authorization_script(labels=["merge-ready-mac", "dequeued"])
    operations = [call[0] for call in result["calls"]]

    assert operations == [
        "status",
        "get",
        "events",
        "add",
        "remove",
        "notice",
        "events",
        "status",
    ]
    assert [call[1] for call in result["calls"] if call[0] == "status"] == [
        "pending",
        "success",
    ]
    assert result["calls"][3] == [
        "add",
        ["merge-requeue-required", "merge-requeue-trigger"],
    ]


def test_initial_authorization_does_not_issue_a_requeue_trigger():
    result = _run_authorization_script(labels=["merge-ready-mac"])

    assert result["calls"] == [
        ["status", "pending"],
        ["get"],
        ["status", "success"],
    ]


def test_consumed_trigger_can_be_reissued_from_persistent_recovery_marker():
    result = _run_authorization_script(
        labels=["merge-ready-mac", "merge-requeue-required"]
    )

    assert result["calls"] == [
        ["status", "pending"],
        ["get"],
        ["events"],
        ["add", ["merge-requeue-required", "merge-requeue-trigger"]],
        ["status", "success"],
    ]


def test_status_failure_cannot_remove_the_dequeue_circuit_breaker():
    result = _run_authorization_script(
        labels=["merge-ready-mac", "dequeued"], fail_status_call=1
    )

    assert result["calls"] == [
        ["status", "pending"],
        ["threw", "status failure"],
    ]


def test_delayed_ready_event_cannot_authorize_a_later_dequeue():
    result = _run_authorization_script(
        labels=["merge-ready-mac", "dequeued"],
        ready_event_at="2026-09-01T04:00:00Z",
        dequeue_event_at="2026-09-01T04:01:00Z",
    )

    assert result["calls"] == [
        ["status", "pending"],
        ["get"],
        ["events"],
        ["status", "failure"],
        ["failed", "Re-apply merge-ready after the latest dequeue"],
    ]


def test_recovery_event_lookup_failure_remains_pending():
    result = _run_authorization_script(
        labels=["merge-ready-mac", "dequeued"], fail_events=True
    )

    assert result["calls"] == [
        ["status", "pending"],
        ["get"],
        ["events"],
        ["threw", "events failure"],
    ]


def test_live_pull_failure_remains_blocked_by_pending_status():
    result = _run_authorization_script(labels=["merge-ready-mac"], fail_get=True)

    assert result["calls"] == [
        ["status", "pending"],
        ["get"],
        ["threw", "get failure"],
    ]


def test_trigger_failure_keeps_both_queue_circuit_breakers():
    result = _run_authorization_script(
        labels=["merge-ready-mac", "dequeued"], fail_add=True
    )

    assert result["calls"] == [
        ["status", "pending"],
        ["get"],
        ["events"],
        ["add", ["merge-requeue-required", "merge-requeue-trigger"]],
        ["threw", "add failure"],
    ]


def test_marker_failure_leaves_trigger_blocked_and_retryable():
    result = _run_authorization_script(
        labels=["merge-ready-mac", "dequeued"], remove_error_status=500
    )

    assert result["calls"] == [
        ["status", "pending"],
        ["get"],
        ["events"],
        ["add", ["merge-requeue-required", "merge-requeue-trigger"]],
        ["remove"],
        ["threw", "remove failure"],
    ]


def test_concurrent_marker_removal_proceeds_to_final_authorization():
    result = _run_authorization_script(
        labels=["merge-ready-mac", "dequeued"], remove_error_status=404
    )

    assert result["calls"] == [
        ["status", "pending"],
        ["get"],
        ["events"],
        ["add", ["merge-requeue-required", "merge-requeue-trigger"]],
        ["remove"],
        ["events"],
        ["status", "success"],
    ]


def test_new_dequeue_between_check_and_removal_restores_circuit_breaker():
    result = _run_authorization_script(
        labels=["merge-ready-mac", "dequeued"],
        late_dequeue_event_at="2026-09-01T04:02:00Z",
    )

    assert result["calls"] == [
        ["status", "pending"],
        ["get"],
        ["events"],
        ["add", ["merge-requeue-required", "merge-requeue-trigger"]],
        ["remove"],
        [
            "notice",
            "Cleared stale dequeued state before re-authorizing merge-ready-mac.",
        ],
        ["events"],
        ["add", ["dequeued"]],
        ["status", "failure"],
        ["failed", "Re-apply merge-ready after the latest dequeue"],
    ]


def test_activation_docs_provision_the_internal_requeue_label():
    docs = (ROOT / "docs/engineering/operations/path-aware-merge-queue.md").read_text()

    assert "Create the `merge-ready`, `merge-ready-mac`," in docs
    assert "`merge-requeue-required`, and `merge-requeue-trigger` labels" in docs
    assert "internal, bot-owned" in docs


def test_final_status_failure_leaves_trigger_blocked_and_retryable():
    result = _run_authorization_script(
        labels=["merge-ready-mac", "dequeued"], fail_status_call=2
    )

    assert result["calls"] == [
        ["status", "pending"],
        ["get"],
        ["events"],
        ["add", ["merge-requeue-required", "merge-requeue-trigger"]],
        ["remove"],
        [
            "notice",
            "Cleared stale dequeued state before re-authorizing merge-ready-mac.",
        ],
        ["events"],
        ["status", "success"],
        ["threw", "status failure"],
    ]


def test_historical_authorization_rerun_cannot_clear_a_newer_dequeue():
    result = _run_authorization_script(
        labels=["merge-ready-mac", "dequeued"], run_attempt=2
    )

    assert result["calls"][0][0] == "failed"
    assert all(
        call[0] not in {"get", "status", "remove", "add"} for call in result["calls"]
    )


def test_stale_head_or_double_ready_labels_cannot_clear_dequeue():
    for labels, live_head in (
        (["merge-ready-mac", "dequeued"], "new-head"),
        (["merge-ready", "merge-ready-mac", "dequeued"], "head-sha"),
    ):
        result = _run_authorization_script(labels=labels, live_head=live_head)
        assert [call[0] for call in result["calls"]] == [
            "status",
            "get",
            "status",
            "failed",
        ]
        assert result["calls"][2] == ["status", "failure"]
