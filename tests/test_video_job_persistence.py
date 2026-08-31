# SPDX-License-Identifier: Apache-2.0
"""Durable completed-job contract for the asynchronous Videos API."""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_mlx import cli, server
from vllm_mlx.routes import video


@pytest.fixture(autouse=True)
def _isolated_video_store():
    configure = video.configure_video_jobs
    configure(None)
    video.start_video_jobs()
    yield
    configure(None)
    video.start_video_jobs()


async def _wait_for_completion(video_id: str) -> dict:
    for _ in range(200):
        current = await video.retrieve_video(video_id)
        if current["status"] == "completed":
            return current
        await asyncio.sleep(0.01)
    raise AssertionError(f"video job {video_id} did not complete")


def _completed_job(job_id: str, *, created_at: int = 1) -> video._VideoJob:
    return video._VideoJob(
        id=job_id,
        model="ltx-2.3-mlx-q4",
        prompt=f"prompt {created_at}",
        seconds="1",
        size="512x512",
        status="completed",
        progress=100,
        created_at=created_at,
        completed_at=created_at + 1,
        output_path=str(video._jobs_root / job_id / "output.mp4"),
        generation_finished=True,
    )


def _write_completed_job(job: video._VideoJob) -> None:
    job_dir = video._jobs_root / job.id
    job_dir.mkdir(mode=0o700)
    (job_dir / "output.mp4").write_bytes(b"generated-mp4")
    video._persist_completed_job(job)


@pytest.mark.asyncio
async def test_completed_job_survives_store_reconfiguration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "videos"
    assert video.configure_video_jobs(store) == store.resolve()
    video.start_video_jobs()

    class FakeEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            output_path.write_bytes(b"generated-mp4")

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    created = await video.create_video(
        prompt="Ocean waves at sunset",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=42,
        input_reference=None,
    )
    completed = await _wait_for_completion(created["id"])
    assert completed["progress"] == 100
    job_dir = store / created["id"]
    assert (job_dir / "job.json").is_file()
    assert not list(job_dir.glob(".job-*.json.tmp"))

    # Reconfiguration clears process memory and models a fresh server process
    # selecting the same operator-owned store.
    await asyncio.sleep(0)
    video.configure_video_jobs(store)
    assert created["id"] not in video._jobs
    video.start_video_jobs()

    restored = await video.retrieve_video(created["id"])
    assert restored == completed
    listing = await video.list_videos(limit=20)
    assert [item["id"] for item in listing["data"]] == [created["id"]]
    response = await video.retrieve_video_content(created["id"])
    assert (
        b"".join([chunk async for chunk in response.body_iterator]) == b"generated-mp4"
    )

    deleted = await video.delete_video(created["id"])
    assert deleted["deleted"] is True
    assert not job_dir.exists()


@pytest.mark.asyncio
async def test_default_store_remains_process_temporary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert video._jobs_are_persistent is False

    class FakeEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            output_path.write_bytes(b"generated-mp4")

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    created = await video.create_video(
        prompt="Temporary result",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=1,
        input_reference=None,
    )
    await _wait_for_completion(created["id"])
    assert not (video._jobs_root / created["id"] / "job.json").exists()
    await video.delete_video(created["id"])


def test_restore_ignores_partial_malformed_and_noncompleted_records(
    tmp_path: Path,
) -> None:
    store = tmp_path / "videos"
    video.configure_video_jobs(store)

    malformed_id = "video_" + "a" * 32
    malformed = store / malformed_id
    malformed.mkdir()
    (malformed / "output.mp4").write_bytes(b"mp4")
    (malformed / "job.json").write_text("{not-json", encoding="utf-8")

    partial_id = "video_" + "b" * 32
    partial = store / partial_id
    partial.mkdir()
    partial_job = _completed_job(partial_id)
    (partial / "job.json").write_text(
        json.dumps(video._completed_job_record(partial_job)), encoding="utf-8"
    )

    queued_id = "video_" + "c" * 32
    queued = store / queued_id
    queued.mkdir()
    (queued / "output.mp4").write_bytes(b"mp4")
    queued_record = video._completed_job_record(_completed_job(queued_id))
    queued_record.update(status="queued", progress=0, completed_at=None)
    (queued / "job.json").write_text(json.dumps(queued_record), encoding="utf-8")

    video.start_video_jobs()

    assert video._jobs == {}
    # Invalid records are ignored, not destructively removed. A future server
    # version or an operator can still inspect and recover them.
    assert malformed.exists()
    assert partial.exists()
    assert queued.exists()


@pytest.mark.parametrize(
    "record",
    [
        [],
        {"schema_version": 2},
        {"schema_version": 1, "object": "not-video"},
        {
            "schema_version": 1,
            "object": "video",
            "status": "completed",
            "progress": 100,
            "error": {"code": "unexpected"},
        },
        {
            "schema_version": 1,
            "object": "video",
            "status": "completed",
            "progress": 100,
            "error": None,
            "model": "",
        },
    ],
)
def test_restore_rejects_invalid_completed_metadata(
    tmp_path: Path, record: object
) -> None:
    video.configure_video_jobs(tmp_path / "videos")
    job_id = "video_" + "e" * 32
    job_dir = video._jobs_root / job_id
    job_dir.mkdir()
    (job_dir / "output.mp4").write_bytes(b"mp4")
    if isinstance(record, dict):
        record = {"id": job_id, **record}
    (job_dir / "job.json").write_text(json.dumps(record), encoding="utf-8")

    assert video._load_completed_job(job_dir) is None


def test_restore_rejects_unowned_shapes_and_symlinks(tmp_path: Path) -> None:
    video.configure_video_jobs(tmp_path / "videos")
    assert video._load_completed_job(video._jobs_root / "not-a-video-id") is None

    empty_id = "video_" + "f" * 32
    empty_dir = video._jobs_root / empty_id
    empty_dir.mkdir()
    (empty_dir / "output.mp4").write_bytes(b"")
    (empty_dir / "job.json").write_text("{}", encoding="utf-8")
    assert video._load_completed_job(empty_dir) is None

    linked_id = "video_" + "1" * 32
    linked_dir = video._jobs_root / linked_id
    linked_dir.mkdir()
    (linked_dir / "output.mp4").write_bytes(b"mp4")
    external_metadata = tmp_path / "external.json"
    external_metadata.write_text("{}", encoding="utf-8")
    (linked_dir / "job.json").symlink_to(external_metadata)
    assert video._load_completed_job(linked_dir) is None


def test_restore_scan_failure_is_nonfatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video.configure_video_jobs(tmp_path / "videos")

    def fail_scan(self):
        raise OSError("offline volume")

    monkeypatch.setattr(Path, "iterdir", fail_scan)
    assert video._restore_completed_jobs() == []


def test_restore_enforces_existing_hundred_job_retention(tmp_path: Path) -> None:
    store = tmp_path / "videos"
    video.configure_video_jobs(store)
    oldest_id = "video_" + f"{0:032x}"
    for index in range(video._MAX_JOBS + 1):
        job_id = "video_" + f"{index:032x}"
        _write_completed_job(_completed_job(job_id, created_at=index))

    video.start_video_jobs()

    assert len(video._jobs) == video._MAX_JOBS
    assert oldest_id not in video._jobs
    assert not (store / oldest_id).exists()


def test_failed_metadata_replace_leaves_no_partial_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video.configure_video_jobs(tmp_path / "videos")
    job = _completed_job("video_" + "d" * 32)
    job_dir = video._jobs_root / job.id
    job_dir.mkdir()
    (job_dir / "output.mp4").write_bytes(b"mp4")

    def fail_replace(source, destination) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(video.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        video._persist_completed_job(job)

    assert not (job_dir / "job.json").exists()
    assert not list(job_dir.glob(".job-*.json.tmp"))


@pytest.mark.asyncio
async def test_metadata_failure_keeps_completed_video_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    video.configure_video_jobs(tmp_path / "videos")
    video.start_video_jobs()

    class FakeEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            output_path.write_bytes(b"generated-mp4")

    def fail_persist(job) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    monkeypatch.setattr(video, "_persist_completed_job", fail_persist)
    created = await video.create_video(
        prompt="Keep the completed output",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=2,
        input_reference=None,
    )

    assert (await _wait_for_completion(created["id"]))["status"] == "completed"
    assert "Unable to persist completed video job metadata" in caplog.text
    response = await video.retrieve_video_content(created["id"])
    assert (
        b"".join([chunk async for chunk in response.body_iterator]) == b"generated-mp4"
    )
    await video.delete_video(created["id"])


def test_video_output_directory_rejects_a_file(tmp_path: Path) -> None:
    destination = tmp_path / "not-a-directory"
    destination.write_text("occupied", encoding="utf-8")

    with pytest.raises((FileExistsError, ValueError)):
        video.configure_video_jobs(destination)


def test_video_store_rejects_blank_path_and_live_reconfiguration(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        video.configure_video_jobs("   ")

    marker = threading.current_thread()
    video._generation_threads.add(marker)
    try:
        with pytest.raises(RuntimeError, match="while jobs run"):
            video.configure_video_jobs(tmp_path / "videos")
    finally:
        video._generation_threads.discard(marker)


def test_ephemeral_cleanup_visits_every_owned_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed: list[Path] = []
    monkeypatch.setattr(
        video.shutil,
        "rmtree",
        lambda root, *, ignore_errors: removed.append(Path(root)),
    )

    video._cleanup_jobs()

    assert set(removed) == video._ephemeral_jobs_roots


def test_unified_serve_parser_exposes_video_output_directory(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(
        ["serve", "ltx-2.3-mlx-q4", "--video-output-dir", str(tmp_path)]
    )

    assert args.video_output_dir == str(tmp_path)


def test_both_server_entrypoints_configure_the_shared_video_store() -> None:
    unified_source = inspect.getsource(cli.serve_command)
    standalone_source = inspect.getsource(server.main)

    assert (
        'configure_video_jobs(getattr(args, "video_output_dir", None))'
        in unified_source
    )
    assert "_add_video_job_args_to_server_parser(parser)" in standalone_source
    assert "configure_video_jobs(args.video_output_dir)" in standalone_source
    assert unified_source.index("configure_video_jobs(") < unified_source.index(
        "_ensure_model_downloaded("
    )


def test_unified_serve_reports_video_store_configuration_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        video,
        "configure_video_jobs",
        lambda output_dir: (_ for _ in ()).throw(OSError("read-only")),
    )

    with pytest.raises(SystemExit) as exc:
        cli.serve_command(SimpleNamespace(video_output_dir="/unwritable"))

    assert exc.value.code == 2
    assert (
        "cannot configure video output directory: read-only" in capsys.readouterr().err
    )


def test_standalone_server_reports_video_store_configuration_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        video,
        "configure_video_jobs",
        lambda output_dir: (_ for _ in ()).throw(OSError("read-only")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["vllm_mlx.server", "--video-output-dir", "/unwritable"],
    )

    with pytest.raises(SystemExit) as exc:
        server.main()

    assert exc.value.code == 2
    assert (
        "cannot configure video output directory: read-only" in capsys.readouterr().err
    )
