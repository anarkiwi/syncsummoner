"""Audio and video feature extraction driving the composer.

Beats, downbeats, tempo curve, onset strength, four band envelopes for the CV
inputs, self-similarity segmentation, and information content over the onset
sequence. librosa is optional; scipy fallbacks cover every function.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from scipy import ndimage, signal
from scipy.io import wavfile

try:
    import librosa
except ImportError:
    librosa = None

EPS = 1e-12
CV_BANDS = ((20.0, 120.0), (120.0, 600.0), (600.0, 3000.0), (3000.0, 12000.0))


@dataclass(frozen=True)
class Section:
    """One structural span of the track, from the self-similarity segmentation."""

    start: float
    end: float
    label: str
    destroy: bool = False

    @property
    def duration(self) -> float:
        """Section length in seconds."""
        return self.end - self.start


@dataclass(frozen=True, eq=False)
class AudioFeatures:
    """Musical structure of the track at the two control timescales."""

    sr: int
    duration: float
    rate: float
    tempo: float
    times: np.ndarray
    beats: np.ndarray
    downbeats: np.ndarray
    tempo_curve: np.ndarray
    onset_strength: np.ndarray
    onset_ic: np.ndarray
    bands: np.ndarray
    sections: tuple[Section, ...]

    @property
    def bar_duration(self) -> float:
        """Median downbeat spacing, falling back to four beats at the estimated tempo."""
        return float(np.median(np.diff(self.downbeats))) if self.downbeats.size > 1 else 4 * 60.0 / self.tempo


@dataclass(frozen=True, eq=False)
class VideoFeatures:
    """Source footage statistics: the arrangement responds to the footage, not only the music."""

    fps: float
    n_frames: int
    shot_boundaries: np.ndarray
    motion_energy: np.ndarray
    luma: np.ndarray
    chroma: np.ndarray


@dataclass(frozen=True, eq=False)
class Features:
    """Everything the planner reads about one track and its source footage."""

    audio: AudioFeatures | None
    video: VideoFeatures | None


def load_audio(path: str | Path, *, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Read an audio file to mono float32 at ``sr``, via librosa when available."""
    if librosa is not None:
        y, out_sr = librosa.load(str(path), sr=sr, mono=True)
        return np.asarray(y, dtype=np.float32), int(out_sr)
    native_sr, data = wavfile.read(str(path))
    y = np.asarray(data, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if np.issubdtype(np.asarray(data).dtype, np.integer):
        y = y / float(np.iinfo(np.asarray(data).dtype).max)
    if native_sr != sr:
        g = np.gcd(int(native_sr), int(sr))
        y = signal.resample_poly(y, sr // g, native_sr // g)
    return y.astype(np.float32), int(sr)


def _spectrogram(y: np.ndarray, *, n_fft: int, hop: int) -> np.ndarray:
    """Magnitude spectrogram as ``(bins, frames)``, zero-padded to at least one frame."""
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop]
    return np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=-1)).T


def band_spectrogram(y: np.ndarray, *, n_fft: int = 2048, hop: int = 512, n_bands: int = 64) -> np.ndarray:
    """Log-compressed spectrogram pooled into geometrically spaced bands, ``(n_bands, frames)``."""
    mag = _spectrogram(y, n_fft=n_fft, hop=hop)
    edges = np.unique(np.round(np.geomspace(1, mag.shape[0] - 1, n_bands + 1)).astype(int))
    return np.log1p(np.add.reduceat(mag, edges[:-1], axis=0))


def onset_strength(y: np.ndarray, sr: int, *, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Half-wave rectified spectral flux over log-compressed bands, normalized to unit peak."""
    if librosa is not None:
        env = librosa.onset.onset_strength(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    else:
        bands = band_spectrogram(y, n_fft=n_fft, hop=hop)
        env = np.concatenate(([0.0], np.maximum(np.diff(bands, axis=1), 0.0).sum(axis=0)))
    return (env / (env.max() + EPS)).astype(np.float64)


def tempo_period(
    env: np.ndarray,
    rate: float,
    *,
    bpm_range: tuple[float, float] = (50.0, 200.0),
    prior_bpm: float = 120.0,
    prior_octaves: float = 1.0,
) -> float:
    """Autocorrelation tempo estimate in frames per beat, under a log-normal tempo prior."""
    x = env - env.mean()
    n = int(2 ** np.ceil(np.log2(max(4, 2 * x.size))))
    ac = np.fft.irfft(np.abs(np.fft.rfft(x, n)) ** 2, n)[: x.size]
    lags = np.arange(1, max(2, x.size))
    bpm = 60.0 * rate / lags
    ok = (bpm >= bpm_range[0]) & (bpm <= bpm_range[1])
    if not ok.any():
        return max(1.0, 60.0 * rate / prior_bpm)
    prior = np.exp(-0.5 * (np.log2(bpm / prior_bpm) / prior_octaves) ** 2)
    return float(lags[ok][np.argmax(ac[lags[ok]] * prior[ok])])


def beat_track(env: np.ndarray, rate: float, *, period: float | None = None, tightness: float = 100.0):
    """Ellis dynamic-programming beat tracker over the onset envelope; returns beat times and tempo."""
    period = tempo_period(env, rate) if period is None else period
    p = max(2, int(round(period)))
    local = (env - env.mean()) / (env.std() + EPS)
    cum = local.astype(np.float64).copy()
    back = np.full(local.size, -1, dtype=np.int64)
    offsets = np.arange(-2 * p, -(p // 2) + 1)
    txwt = -tightness * np.log(-offsets / float(p)) ** 2
    for i in range(local.size):
        idx = offsets + i
        ok = idx >= 0
        if not ok.any():
            continue
        cand = txwt[ok] + cum[idx[ok]]
        j = int(np.argmax(cand))
        cum[i] = local[i] + cand[j]
        back[i] = idx[ok][j]
    beats: list[int] = []
    i = int(np.argmax(cum))
    while i >= 0:
        beats.append(i)
        i = int(back[i])
    times = np.asarray(beats[::-1], dtype=np.float64) / rate
    tempo = 60.0 / np.median(np.diff(times)) if times.size > 1 else 60.0 * rate / p
    return times, float(tempo)


def downbeat_times(beats: np.ndarray, env: np.ndarray, rate: float, *, meter: int = 4) -> np.ndarray:
    """Beats on the metrical phase whose mean onset strength is greatest."""
    if beats.size < meter:
        return beats[:1]
    idx = np.clip(np.round(beats * rate).astype(int), 0, env.size - 1)
    strength = env[idx]
    phase = int(np.argmax([strength[p::meter].mean() for p in range(meter)]))
    return beats[phase::meter]


def tempo_curve(beats: np.ndarray, times: np.ndarray, tempo: float) -> np.ndarray:
    """Instantaneous tempo interpolated from inter-beat intervals onto a time grid."""
    if beats.size < 2:
        return np.full(times.size, tempo)
    d = np.diff(beats)
    return np.interp(times, beats[:-1] + d / 2, 60.0 / d)


def band_envelopes(
    y: np.ndarray, sr: int, *, bands: Sequence[tuple[float, float]] = CV_BANDS, smooth_hz: float = 20.0
) -> np.ndarray:
    """Rectified, low-passed envelopes of four bands, one row per device CV input."""
    out = np.zeros((len(bands), y.size), dtype=np.float32)
    smooth = signal.butter(2, min(smooth_hz, 0.4 * sr), btype="low", fs=sr, output="sos")
    for i, (lo, hi) in enumerate(bands):
        sos = signal.butter(4, (lo, min(hi, 0.45 * sr)), btype="band", fs=sr, output="sos")
        env = signal.sosfilt(smooth, np.abs(signal.sosfilt(sos, y)))
        out[i] = env / (env.max() + EPS)
    return out


def write_cv_wav(path: str | Path, bands: np.ndarray, sr: int) -> None:
    """Write band envelopes as an interleaved WAV for the device's audio/CV inputs."""
    wavfile.write(str(path), int(sr), (np.clip(bands.T, -1.0, 1.0) * 32767).astype(np.int16))


def self_similarity(feat: np.ndarray) -> np.ndarray:
    """Cosine self-similarity matrix of a ``(D, T)`` feature sequence."""
    f = feat / (np.linalg.norm(feat, axis=0, keepdims=True) + EPS)
    return f.T @ f


def novelty(ssm: np.ndarray, *, kernel_frames: int) -> np.ndarray:
    """Foote checkerboard novelty, computed separably because the kernel is rank one."""
    half = max(1, min(kernel_frames // 2, ssm.shape[0] // 2))
    x = np.linspace(-2.0, 2.0, 2 * half + 1)
    w = np.sign(x) * np.exp(-0.5 * x * x)
    corr = ndimage.correlate1d(ssm, w, axis=0, mode="nearest")
    nov = np.diagonal(ndimage.correlate1d(corr, w, axis=1, mode="nearest")).copy()
    nov -= nov.min()
    return nov / (nov.max() + EPS)


def segment(
    feat: np.ndarray,
    rate: float,
    *,
    kernel_s: float = 8.0,
    min_section_s: float = 6.0,
    prominence_sigma: float = 1.0,
) -> tuple[Section, ...]:
    """Sections from checkerboard novelty peaks, labelled by mutual feature similarity."""
    n = feat.shape[1]
    total = n / rate
    if n < 4:
        return (Section(0.0, total, "A"),)
    nov = novelty(self_similarity(feat), kernel_frames=max(3, int(kernel_s * rate)))
    peaks, _ = signal.find_peaks(
        nov, distance=max(1, int(min_section_s * rate)), prominence=prominence_sigma * nov.std()
    )
    bounds = np.unique(np.concatenate(([0], peaks, [n])))
    means = np.stack([feat[:, a:b].mean(axis=1) for a, b in zip(bounds[:-1], bounds[1:])], axis=1)
    sim = self_similarity(means)
    off = sim[~np.eye(sim.shape[0], dtype=bool)]
    thresh = float(np.median(off)) if off.size else 1.0
    labels: list[str] = []
    for k in range(sim.shape[0]):
        prev = sim[k, :k]
        j = int(np.argmax(prev)) if prev.size else -1
        labels.append(labels[j] if j >= 0 and prev[j] > thresh else chr(ord("A") + len(set(labels))))
    return tuple(
        Section(float(a / rate), float(b / rate), lab) for a, b, lab in zip(bounds[:-1], bounds[1:], labels)
    )


def information_content(
    series: np.ndarray, *, rng: np.random.Generator, order: int = 3, n_bins: int = 16
) -> np.ndarray:
    """Per-event surprisal, deferring to the aesthetics analyzer when that package is installed."""
    try:
        from syncsummoner import aesthetics

        out = np.asarray(aesthetics.information_content(series, rng=rng, order=order, n_bins=n_bins))
        if out.shape == np.shape(series):
            return out
    except Exception:
        pass
    return markov_information_content(series, rng=rng, order=order, n_bins=n_bins)


def markov_information_content(
    series: np.ndarray, *, rng: np.random.Generator, order: int = 3, n_bins: int = 16
) -> np.ndarray:
    """Surprisal ``-log2 P(x_t | context)`` under a quantile-binned fixed-order Markov model."""
    x = np.asarray(series, dtype=np.float64).ravel()
    if x.size <= order:
        return np.zeros(x.size, dtype=np.float32)
    x = x + rng.normal(0.0, (x.std() + EPS) * 1e-6, x.size)
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
    sym = np.digitize(x, edges)
    ctx = np.zeros(x.size - order, dtype=np.int64)
    for k in range(order):
        ctx = ctx * n_bins + sym[k : x.size - order + k]
    ctx = np.unique(ctx, return_inverse=True)[1]
    joint = ctx * n_bins + sym[order:]
    jc = np.bincount(joint, minlength=(ctx.max() + 1) * n_bins).astype(np.float64)
    cc = jc.reshape(-1, n_bins).sum(axis=1)
    prob = (jc[joint] + 1.0) / (cc[ctx] + n_bins)
    return np.concatenate((np.zeros(order), -np.log2(prob))).astype(np.float32)


def analyze_audio(
    y: np.ndarray,
    sr: int,
    *,
    rng: np.random.Generator,
    hop: int = 512,
    n_fft: int = 2048,
    meter: int = 4,
    order: int = 3,
    **segment_kw: Any,
) -> AudioFeatures:
    """Full audio analysis: beats, downbeats, tempo curve, sections, CV envelopes, onset surprisal."""
    rate = sr / hop
    env = onset_strength(y, sr, n_fft=n_fft, hop=hop)
    times = np.arange(env.size) / rate
    beats, tempo = beat_track(env, rate)
    downs = downbeat_times(beats, env, rate, meter=meter)
    beat_idx = np.clip(np.round(beats * rate).astype(int), 0, env.size - 1)
    return AudioFeatures(
        sr=sr,
        duration=float(y.size / sr),
        rate=rate,
        tempo=tempo,
        times=times,
        beats=beats,
        downbeats=downs,
        tempo_curve=tempo_curve(beats, times, tempo),
        onset_strength=env,
        onset_ic=information_content(env[beat_idx], rng=rng, order=order),
        bands=band_envelopes(y, sr),
        sections=segment(band_spectrogram(y, n_fft=n_fft, hop=hop), rate, **segment_kw),
    )


def _video_features(
    diff: np.ndarray, *, fps: float, n_frames: int, luma: np.ndarray, chroma: np.ndarray, shot_sigma: float
) -> VideoFeatures:
    """Assemble features from per-frame series, so a stack and a stream agree exactly."""
    peaks, _ = signal.find_peaks(diff, prominence=shot_sigma * (diff.std() + EPS)) if diff.size else ([], {})
    return VideoFeatures(
        fps=float(fps),
        n_frames=int(n_frames),
        shot_boundaries=np.asarray(peaks, dtype=np.int64) + 1,
        motion_energy=np.concatenate((diff[:1], diff)) if diff.size else np.zeros(n_frames),
        luma=luma,
        chroma=chroma,
    )


def analyze_frames(frames: np.ndarray, *, fps: float, shot_sigma: float = 3.0) -> VideoFeatures:
    """Shot boundaries, motion energy and level statistics from an RGB float32 frame stack."""
    x = np.asarray(frames, dtype=np.float32)
    plane = x.mean(axis=3)
    diff = np.abs(np.diff(plane, axis=0)).mean(axis=(1, 2)) if x.shape[0] > 1 else np.zeros(0)
    return _video_features(
        diff,
        fps=fps,
        n_frames=x.shape[0],
        luma=plane.mean(axis=(1, 2)),
        chroma=x.std(axis=3).mean(axis=(1, 2)),
        shot_sigma=shot_sigma,
    )


def read_frames(path: str | Path, *, max_frames: int | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(fps, rgb)`` per frame as float32 in ``[0, 1]``; OpenCV's BGR stops at this boundary."""
    # pylint: disable=no-member
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    sent = 0
    try:
        while max_frames is None or sent < max_frames:
            ok, bgr = cap.read()
            if not ok:
                return
            sent += 1
            yield fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    finally:
        cap.release()


def read_video(path: str | Path, *, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    """Decode a whole video to an RGB float32 frame stack, holding every frame at once."""
    fps, frames = 30.0, []
    for fps, frame in read_frames(path, max_frames=max_frames):
        frames.append(frame)
    stack = np.stack(frames) if frames else np.zeros((0, 1, 1, 3), np.float32)
    return stack, float(fps)


def analyze_video(
    path: str | Path, *, max_frames: int | None = None, shot_sigma: float = 3.0
) -> VideoFeatures:
    """Stream a clip into :class:`VideoFeatures`, holding one frame rather than the clip.

    Every feature is a per-frame scalar or a difference against the frame before
    it, so a stack is never needed: 180s of 576p costs 21GB as one and 1.5MB here.
    """
    fps, count, prev = 30.0, 0, None
    luma, chroma, diff = [], [], []
    for fps, frame in read_frames(path, max_frames=max_frames):
        count += 1
        plane = frame.mean(axis=2)
        luma.append(plane.mean())
        chroma.append(frame.std(axis=2).mean())
        if prev is not None:
            diff.append(np.abs(plane - prev).mean())
        prev = plane
    return _video_features(
        np.asarray(diff, np.float32),
        fps=fps,
        n_frames=count,
        luma=np.asarray(luma, np.float32),
        chroma=np.asarray(chroma, np.float32),
        shot_sigma=shot_sigma,
    )


def analyze(
    video_path: str | Path | None,
    audio_path: str | Path | None,
    *,
    rng: np.random.Generator,
    sr: int = 22050,
    max_frames: int | None = None,
    **audio_kw: Any,
) -> Features:
    """Analyze a source clip and its audio track into the planner's input structure."""
    audio = None
    if audio_path is not None:
        y, out_sr = load_audio(audio_path, sr=sr)
        audio = analyze_audio(y, out_sr, rng=rng, **audio_kw)
    video = None
    if video_path is not None:
        video = analyze_video(video_path, max_frames=max_frames)
    return Features(audio=audio, video=video)
