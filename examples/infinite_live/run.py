#!/usr/bin/env python3
"""Start a Rapid-MLX video server and the local infinite channel together."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="wan2.2-ti2v-5b-q8")
    parser.add_argument("--model-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=8787)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--size", default="512x320")
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=Path(".rapid-live"))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--reuse-model-server",
        action="store_true",
        help="connect to an already-running Rapid-MLX server on --model-port",
    )
    return parser.parse_args()


def wait_for_server(url: str, process: subprocess.Popen | None, timeout: int = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"child process exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise TimeoutError(f"Rapid-MLX did not become ready within {timeout}s")


def terminate(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    args = parse_args()
    model_process = None
    app_process = None
    try:
        if not args.reuse_model_server:
            environment = os.environ.copy()
            environment["RAPID_MLX_WAN_STEPS"] = str(args.steps)
            environment["RAPID_MLX_DISABLE_VERSION_CHECK"] = "1"
            environment.setdefault("RAPID_MLX_WAN_SCHEDULER", "unipc")
            model_process = subprocess.Popen(
                [
                    "rapid-mlx",
                    "serve",
                    args.model,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.model_port),
                    "--log-level",
                    "INFO",
                ],
                env=environment,
                start_new_session=True,
            )
        print("Waiting for the Rapid-MLX video server…", flush=True)
        wait_for_server(
            f"http://127.0.0.1:{args.model_port}/v1/videos/capabilities",
            model_process,
        )
        app_process = subprocess.Popen(
            [
                sys.executable,
                str(HERE / "app.py"),
                "--port",
                str(args.ui_port),
                "--video-api",
                f"http://127.0.0.1:{args.model_port}",
                "--model",
                args.model,
                "--size",
                args.size,
                "--frames",
                str(args.frames),
                "--output-dir",
                str(args.output_dir),
            ]
        )
        wait_for_server(f"http://127.0.0.1:{args.ui_port}/api/state", app_process, 30)
        url = f"http://127.0.0.1:{args.ui_port}"
        print(f"Rapid Infinite Live is running at {url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        return app_process.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        terminate(app_process)
        terminate(model_process)


if __name__ == "__main__":
    raise SystemExit(main())
