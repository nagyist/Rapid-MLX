#!/usr/bin/env python3
"""Compare MTP-only and prompt-lookup-assisted Flash-Next workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


def scenarios() -> dict[str, tuple[list[dict[str, str]], int]]:
    source = "\n".join(
        [
            "FEATURE_FLAG = False",
            "",
            *[
                f"def transform_{index:03d}(value: int) -> int:\n"
                f"    return value + {index}"
                for index in range(96)
            ],
        ]
    )
    return {
        "copy_exact": (
            [
                {
                    "role": "system",
                    "content": "Follow the requested output format exactly.",
                },
                {
                    "role": "user",
                    "content": (
                        "Return the complete file between BEGIN_FILE and END_FILE "
                        "verbatim. Output only file contents, with no fence.\n"
                        f"BEGIN_FILE\n{source}\nEND_FILE"
                    ),
                },
            ],
            512,
        ),
        "code_edit": (
            [
                {
                    "role": "system",
                    "content": "Act as a precise software-engineering assistant.",
                },
                {
                    "role": "user",
                    "content": (
                        "Change FEATURE_FLAG to True and return the complete updated "
                        "file. Preserve every other line exactly. Output only file "
                        f"contents, with no fence.\nBEGIN_FILE\n{source}\nEND_FILE"
                    ),
                },
            ],
            512,
        ),
        "creative": (
            [
                {
                    "role": "system",
                    "content": "Write original prose and do not quote the prompt.",
                },
                {
                    "role": "user",
                    "content": (
                        "Write an original 350-word science-fiction scene about an "
                        "engineer repairing a weather satellite during a solar storm. "
                        "Use vivid sensory detail, natural dialogue, and a decisive "
                        "ending. Do not repeat these instructions."
                    ),
                },
            ],
            512,
        ),
        "chat": (
            [
                {
                    "role": "system",
                    "content": "Answer as a concise senior engineering advisor.",
                },
                {
                    "role": "user",
                    "content": (
                        "Explain the tradeoffs between optimistic concurrency control "
                        "and row locking for a multi-tenant job scheduler. Include "
                        "failure modes and a practical recommendation."
                    ),
                },
            ],
            384,
        ),
    }


def stream_request(
    client: httpx.Client,
    api_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_visible: float | None = None
    content: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason = None
    with client.stream(
        "POST",
        f"{api_url}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "enable_thinking": False,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            visible = delta.get("content") or delta.get("reasoning_content") or ""
            if visible:
                if first_visible is None:
                    first_visible = time.perf_counter()
                content.append(visible)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    finished = time.perf_counter()
    if first_visible is None:
        raise RuntimeError("request completed without visible output")
    completion_tokens = int(usage["completion_tokens"])
    decode_seconds = max(finished - first_visible, 1e-9)
    text = "".join(content)
    return {
        "ttft_ms": round((first_visible - started) * 1000, 3),
        "total_ms": round((finished - started) * 1000, 3),
        "decode_tokens_per_second": round(completion_tokens / decode_seconds, 3),
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "content": text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8465/v1")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    api_url = args.url.rstrip("/")
    root_url = api_url.removesuffix("/v1")
    with httpx.Client(timeout=3600) as client:
        model = client.get(f"{api_url}/models").json()["data"][0]["id"]
        before = client.get(f"{root_url}/v1/status").json()
        results: dict[str, Any] = {}
        for name, (messages, max_tokens) in scenarios().items():
            rows = []
            for run in range(1, args.runs + 1):
                client.post(f"{root_url}/v1/cache/clear").raise_for_status()
                row = stream_request(client, api_url, model, messages, max_tokens)
                row["run"] = run
                rows.append(row)
                print(
                    f"{name} run={run} ttft={row['ttft_ms']:.1f}ms "
                    f"decode={row['decode_tokens_per_second']:.2f} tok/s "
                    f"tokens={row['completion_tokens']}"
                )
            results[name] = {
                "median_ttft_ms": round(
                    statistics.median(r["ttft_ms"] for r in rows), 3
                ),
                "median_decode_tokens_per_second": round(
                    statistics.median(r["decode_tokens_per_second"] for r in rows), 3
                ),
                "runs": rows,
            }
        after = client.get(f"{root_url}/v1/status").json()

    payload = {
        "label": args.label,
        "model": model,
        "method": {
            "temperature": 0.0,
            "thinking": False,
            "runs": args.runs,
            "cache": "cleared before each request",
        },
        "status_before": before,
        "status_after": after,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"RESULTS {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
