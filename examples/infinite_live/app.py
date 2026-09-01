#!/usr/bin/env python3
"""Audience-directed local video channel backed by Rapid-MLX.

This is deliberately an example-level MVP. Rapid-MLX owns model serving and
video jobs; this process owns suggestions, clip scheduling, continuity, and
the browser experience.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

LOGGER = logging.getLogger("rapid_infinite_live")
HERE = Path(__file__).resolve().parent

DEFAULT_FILLERS = (
    "A tiny late-night noodle shop floating through a neon cloud city, the chef waves at passing airships, cinematic miniature world",
    "A curious red panda explores a cozy retro-futurist television studio and discovers a mysterious glowing button",
    "An improvised nature documentary about miniature robots building a village in a mossy garden, warm macro photography",
    "A whimsical local news broadcast from a town populated entirely by capybaras, expressive presenters and practical studio lighting",
)


@dataclass
class Settings:
    video_api: str = "http://127.0.0.1:8000"
    model: str = "wan2.2-ti2v-5b-q8"
    size: str = "512x320"
    frames: int = 25
    output_dir: Path = Path(".rapid-live")
    poll_seconds: float = 2.0
    retry_seconds: float = 8.0
    max_clips: int = 16
    style: str = (
        "A continuous playful late-night television channel, cinematic lighting, "
        "clear subject motion, coherent composition."
    )


@dataclass
class Suggestion:
    id: str
    prompt: str
    author: str
    votes: int
    created_at: float
    attempts: int = 0

    def public(self) -> dict:
        value = asdict(self)
        value.pop("attempts")
        return value


@dataclass
class Clip:
    id: str
    sequence: int
    prompt: str
    author: str
    source: str
    created_at: float
    generation_seconds: float
    media_path: Path = field(repr=False)

    def public(self) -> dict:
        value = asdict(self)
        value.pop("media_path")
        value["url"] = f"/api/clips/{self.id}/content"
        return value


class SuggestionInput(BaseModel):
    prompt: str = Field(min_length=3, max_length=600)
    author: str = Field(default="local viewer", min_length=1, max_length=40)


class Channel:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.clips_dir = self.settings.output_dir / "clips"
        self.clips_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.reference_path = self.settings.output_dir / "continuity.png"
        self.suggestions: list[Suggestion] = []
        self.clips: list[Clip] = []
        self.generating: dict | None = None
        self.last_error: str | None = None
        self.started_at = time.time()
        self.sequence = 0
        self.filler_index = random.randrange(len(DEFAULT_FILLERS))
        self.lock = asyncio.Lock()
        self.wake = asyncio.Event()
        self.stop = asyncio.Event()
        self.worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        self.worker_task = asyncio.create_task(self._worker(), name="live-generator")

    async def close(self) -> None:
        self.stop.set()
        self.wake.set()
        if self.worker_task is not None:
            self.worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.worker_task

    async def submit(self, payload: SuggestionInput) -> Suggestion:
        suggestion = Suggestion(
            id=uuid.uuid4().hex,
            prompt=" ".join(payload.prompt.split()),
            author=" ".join(payload.author.split()),
            votes=1,
            created_at=time.time(),
        )
        async with self.lock:
            self.suggestions.append(suggestion)
        self.wake.set()
        return suggestion

    async def vote(self, suggestion_id: str) -> Suggestion:
        async with self.lock:
            suggestion = next(
                (item for item in self.suggestions if item.id == suggestion_id), None
            )
            if suggestion is None:
                raise KeyError(suggestion_id)
            suggestion.votes += 1
            return suggestion

    async def snapshot(self) -> dict:
        async with self.lock:
            suggestions = sorted(
                self.suggestions, key=lambda item: (-item.votes, item.created_at)
            )
            clips = list(self.clips)
            return {
                "channel": "LOCAL 01 · DREAM LOOP",
                "model": self.settings.model,
                "size": self.settings.size,
                "frames": self.settings.frames,
                "online": True,
                "uptime_seconds": int(time.time() - self.started_at),
                "generating": self.generating,
                "last_error": self.last_error,
                "suggestions": [item.public() for item in suggestions],
                "clips": [clip.public() for clip in clips],
            }

    async def clip_path(self, clip_id: str) -> Path:
        async with self.lock:
            clip = next((item for item in self.clips if item.id == clip_id), None)
            if clip is None:
                raise KeyError(clip_id)
            return clip.media_path

    async def _next_prompt(self) -> tuple[str, str, str, Suggestion | None]:
        async with self.lock:
            if self.suggestions:
                selected = min(
                    self.suggestions, key=lambda item: (-item.votes, item.created_at)
                )
                self.suggestions.remove(selected)
                return selected.prompt, selected.author, "viewer", selected
        prompt = DEFAULT_FILLERS[self.filler_index % len(DEFAULT_FILLERS)]
        self.filler_index += 1
        return prompt, "auto director", "filler", None

    def _direct(self, idea: str, has_reference: bool) -> str:
        continuity = (
            " Begin exactly from the supplied first frame, preserve the main "
            "subjects and visual identity, then naturally continue into this next scene:"
            if has_reference
            else " Open the channel with this scene:"
        )
        return f"{self.settings.style}{continuity} {idea}"

    async def _worker(self) -> None:
        while not self.stop.is_set():
            prompt, author, source, suggestion = await self._next_prompt()
            started = time.monotonic()
            async with self.lock:
                self.generating = {
                    "prompt": prompt,
                    "author": author,
                    "source": source,
                    "started_at": time.time(),
                }
                self.last_error = None
            try:
                clip_path = await self._generate_clip(prompt)
                await self._extract_reference(clip_path)
                elapsed = time.monotonic() - started
                clip = Clip(
                    id=uuid.uuid4().hex,
                    sequence=self.sequence,
                    prompt=prompt,
                    author=author,
                    source=source,
                    created_at=time.time(),
                    generation_seconds=round(elapsed, 2),
                    media_path=clip_path,
                )
                self.sequence += 1
                async with self.lock:
                    self.clips.append(clip)
                    stale = self.clips[:-self.settings.max_clips]
                    self.clips = self.clips[-self.settings.max_clips :]
                    self.generating = None
                for old_clip in stale:
                    with suppress(OSError):
                        old_clip.media_path.unlink()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - MVP keeps the channel alive
                LOGGER.exception("Clip generation failed")
                if suggestion is not None and suggestion.attempts < 2:
                    suggestion.attempts += 1
                    async with self.lock:
                        self.suggestions.append(suggestion)
                async with self.lock:
                    self.generating = None
                    self.last_error = str(exc)[:400]
                try:
                    await asyncio.wait_for(self.stop.wait(), self.settings.retry_seconds)
                except TimeoutError:
                    pass

    async def _generate_clip(self, idea: str) -> Path:
        has_reference = self.reference_path.is_file()
        data = {
            "model": self.settings.model,
            "prompt": self._direct(idea, has_reference),
            "seconds": "1",
            "frames": str(self.settings.frames),
            "size": self.settings.size,
            "seed": str(random.randrange(1, 2**31)),
        }
        files = None
        if has_reference:
            files = {
                "input_reference": (
                    "continuity.png",
                    self.reference_path.read_bytes(),
                    "image/png",
                )
            }
        timeout = httpx.Timeout(connect=10, read=60, write=60, pool=10)
        async with httpx.AsyncClient(base_url=self.settings.video_api, timeout=timeout) as client:
            response = await client.post("/v1/videos", data=data, files=files)
            response.raise_for_status()
            job = response.json()
            job_id = job["id"]
            while True:
                if self.stop.is_set():
                    raise asyncio.CancelledError
                await asyncio.sleep(self.settings.poll_seconds)
                status_response = await client.get(f"/v1/videos/{job_id}")
                status_response.raise_for_status()
                status = status_response.json()
                if status["status"] == "failed":
                    detail = status.get("error") or {}
                    raise RuntimeError(detail.get("message", "video generation failed"))
                if status["status"] == "completed":
                    break
            content = await client.get(f"/v1/videos/{job_id}/content")
            content.raise_for_status()
        destination = self.clips_dir / f"{self.sequence:06d}-{job_id}.mp4"
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(content.content)
        temporary.replace(destination)
        return destination

    async def _extract_reference(self, clip_path: Path) -> None:
        temporary = self.reference_path.with_suffix(".tmp.png")
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-sseof",
            "-0.08",
            "-i",
            str(clip_path),
            "-frames:v",
            "1",
            str(temporary),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                "could not extract continuity frame: "
                + stderr.decode("utf-8", errors="replace")[-300:]
            )
        temporary.replace(self.reference_path)


def create_app(settings: Settings) -> FastAPI:
    channel = Channel(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await channel.start()
        try:
            yield
        finally:
            await channel.close()

    app = FastAPI(title="Rapid Infinite Live MVP", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse((HERE / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/state")
    async def state():
        return await channel.snapshot()

    @app.post("/api/suggestions")
    async def suggest(payload: SuggestionInput):
        return (await channel.submit(payload)).public()

    @app.post("/api/suggestions/{suggestion_id}/vote")
    async def vote(suggestion_id: str):
        try:
            return (await channel.vote(suggestion_id)).public()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="suggestion not found") from exc

    @app.get("/api/clips/{clip_id}/content")
    async def clip_content(clip_id: str):
        try:
            path = await channel.clip_path(clip_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="clip not found") from exc
        return FileResponse(path, media_type="video/mp4")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--video-api", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="wan2.2-ti2v-5b-q8")
    parser.add_argument("--size", default="512x320")
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=Path(".rapid-live"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings(
        video_api=args.video_api.rstrip("/"),
        model=args.model,
        size=args.size,
        frames=args.frames,
        output_dir=args.output_dir.resolve(),
    )
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
