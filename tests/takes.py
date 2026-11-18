"""Small MJPEG takes written as the recorder writes them now and as it wrote them before deduplication."""

import hashlib
import subprocess
from fractions import Fraction
from types import SimpleNamespace

import cv2
import numpy as np

from syncsummoner.device import v4l2rec as vr

WIDTH, HEIGHT = 32, 16


def digest(data):
    """The recorder's digest of a capture."""
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def jpegs(shades):
    """One JPEG per shade, equal shades encoding to equal bytes."""
    found = {}
    for shade in set(shades):
        image = np.full((HEIGHT, WIDTH, 3), (shade * 37) % 256, np.uint8)
        image[:, : shade % WIDTH] = 255
        found[shade] = cv2.imencode(".jpg", image)[1].tobytes()  # pylint: disable=no-member
    return [found[s] for s in shades]


def capture(k, data):
    """A dequeued frame as linuxpy hands it over, ``k`` sixtieths of a second in."""
    t_us = k * 16_667
    buff = SimpleNamespace(
        sequence=k,
        timestamp=SimpleNamespace(secs=t_us // 10**6, usecs=t_us % 10**6),
        bytesused=len(data),
        flags=0,
    )
    return SimpleNamespace(buff=buff, data=data)


def recorded(path, datas, rate=Fraction(60)):
    """A take the recorder writes: each run stored once, its sidecar deduplicated."""
    argv = vr.encoder_argv(["-c:v", "copy"], str(path), ffmpeg="ffmpeg")
    with subprocess.Popen(argv, stdin=subprocess.PIPE) as enc, open(vr.sidecar(path), "wb") as log:
        vr.record(
            [capture(k, d) for k, d in enumerate(datas)],
            enc.stdin,
            log,
            stop=lambda: False,
            layout=vr.Layout(vr.FOURCC["mjpeg"], WIDTH, HEIGHT),
            rate=rate,
        )
        enc.stdin.close()
        assert enc.wait() == 0
    return path


def old(path, datas, rate=60):
    """A take as the recorder wrote it before: every capture piped as MJPEG, one sidecar row each."""
    argv = ["ffmpeg", "-loglevel", "error", "-y", "-f", "mjpeg", "-framerate", f"{rate:g}", "-i", "pipe:0"]
    subprocess.run([*argv, "-c:v", "copy", str(path)], input=b"".join(datas), check=True)
    rows = [(k, k * 16_667_000, len(d), 0, digest(d)) for k, d in enumerate(datas)]
    vr.sidecar(path).write_bytes(np.array(rows, dtype=vr.FRAME).tobytes())
    return path


def framemd5(source, *extra, out=()):
    """Each decoded frame's line, from ffmpeg reading ``source`` input arguments."""
    argv = ["ffmpeg", "-nostdin", "-loglevel", "error", *extra, *source, "-map", "0:v:0", *out]
    argv += ["-fps_mode", "passthrough", "-f", "framemd5", "pipe:1"]
    out = subprocess.run(argv, capture_output=True, check=True, text=True).stdout
    return [line for line in out.splitlines() if not line.startswith("#")]
