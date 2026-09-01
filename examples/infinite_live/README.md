# Rapid Infinite Live MVP

This local prototype turns Rapid-MLX video generation into an audience-directed
channel. Viewers submit and vote on the next scene in the browser. The Director
prioritizes viewer ideas, generates filler while chat is idle, and feeds the
last frame of each completed clip into the next Wan TI2V request for continuity.

The browser loops the latest completed clip if generation falls behind. This is
intentional: the channel remains watchable while the MVP measures the Mac's real
generation/playback ratio. It is not yet a constant-rate RTMP broadcast.

## Run

Requirements: Apple silicon, `ffmpeg`, Rapid-MLX with the video extra, and the
`wan2.2-ti2v-5b-q8` checkpoint. The one-command launcher starts both servers:

```bash
python examples/infinite_live/run.py
```

Then open <http://127.0.0.1:8787>. The launcher defaults to a 512x320, 25-frame,
20-step generation profile. On an M3 Ultra 256 GB, the first measured T2V run
took about 33 seconds for a 1.04-second clip. Eight steps took about 18 seconds
but produced visibly weak prompt adherence; use the faster proof profile only
when iterating on orchestration rather than image quality:

```bash
python examples/infinite_live/run.py --steps 4 --frames 13
```

If a compatible Rapid-MLX video server is already running on port 8000:

```bash
python examples/infinite_live/run.py --reuse-model-server
```

Generated clips and the current continuity frame are kept under `.rapid-live/`,
which is ignored by Git. Stop the launcher with Control-C; it terminates the
child channel and model-server processes.

## MVP boundaries

- Local browser playback, not Twitch/YouTube RTMP yet.
- Viewer prompts are trusted local input; production needs moderation and rate
  limits before any LAN or public exposure.
- Wan TI2V has no native cross-clip memory. Continuity is last-frame
  conditioning plus a standing style prompt.
- The current process keeps recent clip metadata in memory. Durable queue state,
  admin controls, and restart recovery belong in the next iteration.
