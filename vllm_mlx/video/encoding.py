"""Small, auditable RGB-to-MP4 bridge for Desktop video runtimes."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


class VideoEncodingError(RuntimeError):
    """Raised when the local encoder cannot produce a usable MP4."""


def _ffmpeg_command(
    ffmpeg: str, *, width: int, height: int, fps: int, output_path: Path
) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "h264_videotoolbox",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def encode_rgb_video(frames, output_path: str | Path, fps: int) -> None:
    """Atomically encode ``[frames, height, width, rgb]`` uint8 pixels.

    Frames are streamed one at a time so a long generation does not allocate
    a second full raw-video buffer. The destination is replaced only after a
    successful, non-empty MP4 has been produced.
    """
    import numpy as np

    pixels = np.asarray(frames)
    if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise VideoEncodingError("video frames must be uint8 [T, H, W, 3] RGB")
    frame_count, height, width, _ = pixels.shape
    if frame_count < 1 or height < 1 or width < 1 or fps < 1:
        raise VideoEncodingError(
            "video dimensions, frame count and fps must be positive"
        )

    from ..runtime.video_lane import _resolve_ffmpeg

    ffmpeg = _resolve_ffmpeg()
    if ffmpeg is None:
        raise VideoEncodingError("ffmpeg is unavailable")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    process: subprocess.Popen | None = None
    with tempfile.TemporaryFile() as stderr_file:
        try:
            handle = tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.stem}.",
                suffix=".encoding.mp4",
                delete=False,
            )
            temporary = Path(handle.name)
            handle.close()
            temporary.unlink(missing_ok=True)
            process = subprocess.Popen(
                _ffmpeg_command(
                    ffmpeg,
                    width=width,
                    height=height,
                    fps=fps,
                    output_path=temporary,
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
            )
            if process.stdin is None:  # pragma: no cover - guaranteed by PIPE
                raise VideoEncodingError("ffmpeg input pipe is unavailable")
            for frame in pixels:
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
            process.stdin.close()
            process.stdin = None
            return_code = process.wait(timeout=120)
            if return_code != 0:
                stderr_file.seek(0, os.SEEK_END)
                size = stderr_file.tell()
                stderr_file.seek(max(0, size - 65_536))
                detail = stderr_file.read().decode("utf-8", errors="replace").strip()
                raise VideoEncodingError(
                    f"ffmpeg exited with status {return_code}"
                    + (f": {detail}" if detail else "")
                )
            if (
                temporary is None
                or not temporary.is_file()
                or temporary.stat().st_size == 0
            ):
                raise VideoEncodingError("ffmpeg completed without an MP4 output")
            os.replace(temporary, destination)
            temporary = None
        except (OSError, subprocess.SubprocessError) as exc:
            raise VideoEncodingError(
                f"video encoding failed: {type(exc).__name__}"
            ) from exc
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            if temporary is not None:
                temporary.unlink(missing_ok=True)
