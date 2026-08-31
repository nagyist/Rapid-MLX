from __future__ import annotations

import io
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest

from vllm_mlx.video.encoding import (
    VideoEncodingError,
    _ffmpeg_command,
    encode_rgb_video,
)


def test_ffmpeg_command_uses_bundled_videotoolbox_contract(tmp_path: Path) -> None:
    output = tmp_path / "video.mp4"
    command = _ffmpeg_command(
        "/bundle/bin/ffmpeg", width=64, height=32, fps=12, output_path=output
    )

    assert command[0] == "/bundle/bin/ffmpeg"
    assert command[command.index("-video_size") + 1] == "64x32"
    assert command[command.index("-c:v") + 1] == "h264_videotoolbox"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[-1] == str(output)


def test_encode_streams_frames_and_atomically_replaces_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = {}
    stdin_closed = threading.Event()

    class CaptureBytesIO(io.BytesIO):
        def close(self) -> None:
            stdin_closed.set()

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            created["command"] = command
            self._stdin = CaptureBytesIO()
            self.stdin = self._stdin
            self.returncode = None

        def wait(self, timeout=None):
            created["timeout"] = timeout
            assert stdin_closed.wait(timeout=1), "encoder input was not closed"
            created["bytes"] = self._stdin.getvalue()
            Path(created["command"][-1]).write_bytes(b"mp4")
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(
        "vllm_mlx.runtime.video_lane._resolve_ffmpeg", lambda: "/bundle/bin/ffmpeg"
    )
    monkeypatch.setattr("vllm_mlx.video.encoding.subprocess.Popen", FakeProcess)
    output = tmp_path / "result.mp4"
    output.write_bytes(b"old")
    frames = np.zeros((2, 4, 8, 3), dtype=np.uint8)

    encode_rgb_video(frames, output, 8)

    assert output.read_bytes() == b"mp4"
    assert len(created["bytes"]) == frames.nbytes
    assert created["timeout"] == 120


def test_encode_timeout_kills_process_and_unblocks_a_stalled_pipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    killed = threading.Event()

    class BlockingInput:
        def write(self, _data):
            killed.wait()
            raise BrokenPipeError

        def close(self):
            pass

    class StalledProcess:
        def __init__(self, *_args, **_kwargs):
            self.stdin = BlockingInput()
            self.returncode = None

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            return self.returncode

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9
            killed.set()

    monkeypatch.setattr(
        "vllm_mlx.runtime.video_lane._resolve_ffmpeg", lambda: "/bundle/bin/ffmpeg"
    )
    monkeypatch.setattr("vllm_mlx.video.encoding.subprocess.Popen", StalledProcess)

    with pytest.raises(VideoEncodingError, match="timed out"):
        encode_rgb_video(
            np.zeros((2, 4, 4, 3), dtype=np.uint8), tmp_path / "result.mp4", 8
        )

    assert killed.is_set()
    assert not list(tmp_path.glob("*.encoding.mp4"))


@pytest.mark.parametrize(
    "frames",
    [
        np.zeros((1, 4, 4, 4), dtype=np.uint8),
        np.zeros((1, 4, 4, 3), dtype=np.float32),
        np.zeros((0, 4, 4, 3), dtype=np.uint8),
    ],
)
def test_encode_rejects_invalid_frame_contract(frames: np.ndarray) -> None:
    with pytest.raises(VideoEncodingError):
        encode_rgb_video(frames, "/tmp/unused.mp4", 8)
