"""Finishing a captured take: trim to the shorter input, mux the audio, fade the edges.

A pass off the capture card is video only and runs edge to edge, because the rig
has no notion of a beginning or an end. Mastering is one ffmpeg invocation, built
here so it can be asserted on without one installed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from syncsummoner.progress import stage

#: Seconds of fade at each edge unless a caller says otherwise.
DEFAULT_FADE_S = 1.0


def probe_duration(path: str | Path, *, ffprobe: str = "ffprobe") -> float:
    """Container duration in seconds, from the format header."""
    done = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=False,
        capture_output=True,
    )
    if done.returncode:
        raise RuntimeError(f"probing {path} failed: {done.stderr.decode('utf-8', 'replace')[:200]}")
    return float(json.loads(done.stdout or "{}").get("format", {}).get("duration", 0.0) or 0.0)


def common_duration(*paths: str | Path | None, ffprobe: str = "ffprobe") -> float:
    """Length every input covers; a render is the shorter of picture and sound."""
    spans = [d for d in (probe_duration(p, ffprobe=ffprobe) for p in paths if p is not None) if d > 0]
    if not spans:
        raise ValueError("no probeable inputs")
    return min(spans)


def fade_filters(duration: float, *, fade_in: float, fade_out: float) -> tuple[str, str]:
    """Video and audio fade chains for a clip of ``duration``.

    Each edge is clamped to half the clip, so on a short one they meet rather
    than overlapping into a fade-out over material still fading in.
    """
    room = max(duration, 0.0) / 2
    fade_in, fade_out = min(max(fade_in, 0.0), room), min(max(fade_out, 0.0), room)
    video = [f"fade=t=in:st=0:d={fade_in:.3f}"] if fade_in > 0 else []
    audio = [f"afade=t=in:st=0:d={fade_in:.3f}"] if fade_in > 0 else []
    if fade_out > 0:
        start = max(duration - fade_out, 0.0)
        video.append(f"fade=t=out:st={start:.3f}:d={fade_out:.3f}")
        audio.append(f"afade=t=out:st={start:.3f}:d={fade_out:.3f}")
    return ",".join(video), ",".join(audio)


def master_command(
    take: str | Path,
    audio: str | Path | None,
    out: str | Path,
    *,
    duration: float,
    fade_in: float = DEFAULT_FADE_S,
    fade_out: float = DEFAULT_FADE_S,
    audio_start: float = 0.0,
    video_start: float = 0.0,
    ffmpeg: str = "ffmpeg",
    crf: int = 16,
) -> list[str]:
    """Argv that trims, fades and muxes one take into the finished clip.

    ``audio_start`` seeks the track, so an excerpt taken from the middle of a
    piece keeps the sound it was composed against. ``video_start`` drops the take's
    lead-in as a filter rather than a seek: a capture's timestamps are the card's,
    and a seek against them lands near the frame asked for rather than on it.
    """
    video_chain, audio_chain = fade_filters(duration, fade_in=fade_in, fade_out=fade_out)
    if video_start > 0:
        trim = f"trim=start={video_start:.3f},setpts=PTS-STARTPTS"
        video_chain = f"{trim},{video_chain}" if video_chain else trim
    argv = [ffmpeg, "-loglevel", "error", "-y", "-i", str(take)]
    if audio is not None:
        argv += (["-ss", f"{audio_start:.3f}"] if audio_start > 0 else []) + ["-i", str(audio)]
    argv += ["-t", f"{duration:.3f}"]
    if video_chain:
        argv += ["-vf", video_chain]
    argv += ["-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p"]
    if audio is None:
        argv += ["-an"]
    else:
        if audio_chain:
            argv += ["-af", audio_chain]
        argv += ["-c:a", "aac", "-b:a", "192k", "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
    return argv + [str(out)]


def master(
    take: str | Path,
    audio: str | Path | None,
    out: str | Path,
    *,
    seconds: float | None = None,
    fade_in: float = DEFAULT_FADE_S,
    fade_out: float = DEFAULT_FADE_S,
    audio_start: float = 0.0,
    video_start: float = 0.0,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> float:
    """Write the finished clip and return its duration.

    ``seconds`` overrides the measured length; without it the clip runs as long
    as both the take and the track do.
    """
    duration = seconds if seconds else common_duration(take, audio, ffprobe=ffprobe) - video_start
    argv = master_command(  # pylint: disable=duplicate-code
        take,
        audio,
        out,
        duration=duration,
        fade_in=fade_in,
        fade_out=fade_out,
        audio_start=audio_start,
        video_start=video_start,
        ffmpeg=ffmpeg,
    )
    with stage("master", seconds=f"{duration:.1f}", fades=f"{fade_in:g}/{fade_out:g}", out=str(out)):
        done = subprocess.run(argv, check=False, capture_output=True)
    if done.returncode:
        raise RuntimeError(f"mastering failed: {done.stderr.decode('utf-8', 'replace')[:200]}")
    return duration
