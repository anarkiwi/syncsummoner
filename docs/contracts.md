# Interface contracts

Normative signatures shared across packages. Changing anything here is a
coordinated change; changing `aesthetics` is a version bump.

## Conventions

| Kind | Type |
| --- | --- |
| Frame | `np.ndarray (H, W, 3) float32` in `[0, 1]`, **RGB** |
| Frame stack | `np.ndarray (T, H, W, 3) float32` in `[0, 1]`, RGB |
| Audio | `np.ndarray (samples,) float32` plus `sr: int` |
| Scalar series | `np.ndarray (T,) float32` |
| Randomness | `np.random.Generator` passed by the caller |

OpenCV decodes BGR. Convert at the boundary. `aesthetics` never sees BGR.

The capture card advertises `YUYV` and delivers `YVYU`, so `device.capture` is
the only place that decodes 4:2:2 and it does so as YVYU; see `docs/hardware.md`.

## `syncsummoner.aesthetics`

```python
gabor_bank(*, n_orientations=4, n_scales=5, ksize=31) -> np.ndarray      # (S, O, ksize, ksize) float32
gabor_energy(frame, *, bank=None, n_orientations=4, n_scales=5) -> ChannelEnergy

@dataclass(frozen=True)
class ChannelEnergy:
    energy: np.ndarray        # (n_scales, n_orientations) float32, sums to 1
    concentration: float      # normalized Herfindahl of energy; 1.0 == all in one cell ("mud")
    peak: tuple[int, int]     # (scale, orientation) argmax

spectral_stats(frame) -> SpectralStats
@dataclass(frozen=True)
class SpectralStats:
    slope: float              # log-log radial power slope; natural scenes ~ -1.0..-1.4
    r2: float                 # goodness of the linear fit
    fractal_dimension: float  # (7 + 2*slope)/2 clamped to [1, 2]; preference peaks ~1.3-1.5

level_stats(frame) -> LevelStats
@dataclass(frozen=True)
class LevelStats:
    luma_mean: float
    luma_std: float
    chroma_mean: float        # mean sqrt(u^2+v^2) in YUV
    chroma_std: float
    clip_frac: float          # fraction of samples at 0.0 or 1.0
    illegal_frac: float       # fraction outside broadcast-legal 16/255..235/255
    colourfulness: float      # Hasler-Susstrunk

passthrough_distance(source, output) -> float   # 0.0 == identical; does this program do anything

motion_stats(prev, curr) -> MotionStats
@dataclass(frozen=True)
class MotionStats:
    framediff_energy: float
    flow_magnitude: float     # mean |flow|, px/frame
    flow_coherence: float     # |mean(flow)| / mean(|flow|); 1.0 == rigid translation

analyze_dynamics(series, *, fps) -> DynamicsResult
@dataclass(frozen=True)
class DynamicsResult:
    period_frames: float | None
    periodicity_strength: float   # peak autocorrelation outside lag 0, in [0, 1]
    winding_number: float | None  # cycles per frame, relative to fps
    stability: StabilityClass

class StabilityClass(enum.Enum):
    STATIC = "static"
    PERIODIC = "periodic"
    QUASIPERIODIC = "quasiperiodic"
    CHAOTIC = "chaotic"

winding_number(series, *, fps) -> float | None

information_content(series, *, rng, order=3, n_bins=16) -> np.ndarray   # (T,) float32, -log2 P
entropy_rate(series, *, rng, order=3, n_bins=16) -> float
class SurprisalModel:
    def __init__(self, *, order=3, n_bins=16, rng): ...
    def fit(self, series) -> "SurprisalModel": ...
    def information_content(self, series) -> np.ndarray: ...

av_correlation(visual, audio_strength, *, visual_fps, audio_fps, max_lag_s=0.5) -> SyncResult
@dataclass(frozen=True)
class SyncResult:
    lag_s: float              # >0 means visual lags audio
    correlation: float

describe_clip(frames, *, fps, rng, source=None) -> ClipDescriptor
score_clip(descriptor, weights=None) -> float
@dataclass(frozen=True)
class ClipDescriptor:
    analyzer_version: str
    n_frames: int
    fps: float
    channel_energy: np.ndarray    # (n_scales, n_orientations) mean over frames
    concentration: float
    spectral_slope: float
    fractal_dimension: float
    levels: LevelStats            # means over frames
    motion: MotionStats           # means over frames
    dynamics: DynamicsResult
    information_content: np.ndarray  # (T,)
    passthrough_distance: float | None
```

`ScoreWeights` is a frozen dataclass of float weights with a documented default
instance; `score_clip` is a pure weighted aggregate over `ClipDescriptor`.

## `syncsummoner.device`

```python
class Transport:                     # wraps pyvmancer; the only module that imports it
    @classmethod
    def open(cls, *, serial=None, report=LOG.info, stall=None) -> "Transport": ...   # PortBusyError if the tty is held
    def programs(self) -> list[str]: ...
    def program_info(self, name=None) -> ProgramInfo: ...
    def load_program(self, name, hold=None) -> None: ...   # hold() brackets only the load command
    def set_param(self, index: int, value: float | bool) -> None: ...   # index 1..12
    def program_state(self) -> np.ndarray: ...   # (12,) int, 0..1023 combined
    def video_status(self) -> VideoStatus: ...
    def set_video_timing(self, timing: str) -> None: ...
    def transport_play(self) / transport_stop(self) / set_bpm(bpm) -> None: ...
    def close(self) -> None: ...

class Session:                       # settle timing, write rate limiting, state cache
    def __init__(self, transport, *, cc_budget_hz=CC_BUDGET_HZ): ...  # parameter writes per second
    def set_params(self, values: Mapping[int, float | bool]) -> None: ...  # one write per 1/cc_budget_hz, cached
    def park(self) -> None: ...      # drive every parameter to a known reference
    def load_program(self, name, *, park=True, link=None) -> None: ...  # returns once the device runs it
    def working_point(self, info, *, exclude=()) -> dict[int, float | bool]: ...
    def arm_modulation(self, info, operators, rng, *, exclude=(), count=5) -> list[str]: ...
    def disarm_modulation(self) -> None: ...
    def ensure_live(self, capture, *, require_motion=True) -> None: ...  # resyncs, then raises

class Capture:                       # long-lived; never reopened per sample
    def __init__(self, device="/dev/video0", *, width=720, height=576, fps=50): ...
    def __enter__ / __exit__
    def read(self) -> np.ndarray | None      # RGB float32 (H, W, 3)
    def wait_for_lock(self, timeout_s=10.0) -> bool   # a structured() frame arrives
    def wait_for_content(self, timeout_s=15.0) -> bool   # frames arrive AND move

class Recorder:                      # ffmpeg owns the card; the host touches no frame
    def __init__(self, device="/dev/video0", *, width, height, fps, mode=MJPEG, copyts=False): ...
    def command(self, path, *, seconds=None) -> list[str]     # -t is unusable under copyts
    def recording(self, path, *, seconds=None) -> ContextManager   # records for the block
    def timestamps(self, path) -> np.ndarray   # per-frame CLOCK_MONOTONIC capture times
inspect_take(path) -> TakeReport   # picture/uniform counts, luma range, motion; .mostly_picture (load), .usable (render), .ever_structured (archive), .frozen
held(frames) -> (motion, frozen)   # consecutive-difference correlation; -1/2 for one picture plus noise
settle(recorder, transport, *, program, timing, timeout_s, probe_path) -> TakeReport  # transmitting() first, then the first probe with frames; SignalLostError / BlankTakeError
require_signal(transport, *, program, timing, when) -> VideoStatus   # brackets every pass; raises SignalLostError
transmitting(status, timing) -> bool | None   # hdmi_tx driving at the session timing; None when unreported

class Link:                          # the HDMI link out of the stimulus source host
    def down(self) / up(self) -> str: ...
    def quiet(self) -> ContextManager["Link"]: ...   # held down for a program load command
```

`ParamSpec` / `ProgramProfile` and the measurement record schema live in
`syncsummoner/device/profile.py` and are the serialization contract for
`profiles/`. `ParamSpec.measured` carries the measured value of each metric in
`fit.PROXY_METRICS` at every visited setpoint, parallel to `values`; the compose
proxy reads its objective back out of it rather than inferring it. A profile
written before those curves existed loads with `measured == {}` and composes off
the older inferences.

`ParamSpec.role` (`profile.Role`: `effect`, `bypass`, `wet`, `momentary`) is what a
gesture may do with the parameter, from `classify_role(index, name)` over the
label the device reports. Profile schema 2 stores it; a schema 1 profile loads with
it derived from its stored name. `ProgramInfo.from_json` sets it and keeps a `wet`
slot continuous whatever its native range. `Transport.program_payload()` is the
unparsed `program info` reply, which `FrameArchive.commit(program_info=...)` and
`FrameArchive.annotate(...)` store under `archive.PROGRAM_INFO` in the archive
metadata, and `names.archived_specs(meta)` turns back into specs for a refit.

## `syncsummoner.probe`

```python
patterns.generate(name, *, width, height, frame_index=0, rng) -> frame
patterns.with_state_index(frame, index, *, bits=8, strip_px=8) -> frame   # gray-code edge strip
patterns.read_state_index(frame, *, bits=8, strip_px=8) -> int | None
patterns.crop_strip(frame, *, strip_px=8) -> frame

plans.oat(spec) -> Iterator[dict[int, float | bool]]
plans.sobol(spec, *, n, rng) -> Iterator[dict[int, float | bool]]
plans.tongue_raster(spec, pair, *, n) -> Iterator[dict[int, float | bool]]
plans.hysteresis(spec, index, *, n) -> Iterator[dict[int, float | bool]]

runner.run_plan(session, capture, plan, *, program, analyzer) -> list[MeasurementRecord]
fit.fit_profile(records, *, specs=None) -> ProgramProfile
fit.PROXY_METRICS: tuple[str, ...]        # kept per setpoint on ParamSpec.measured

behaviour.dwells(rows) -> list[Dwell]           # the native stream positions each fit reads
behaviour.behaviour(dwells, frames, *, loop, seed) -> Behaviour   # frames: position -> native
# Behaviour is (registered, pointwise): what a refit can only read off the frames.
# style is not per-program - it is a band over the whole library, from
# style.measured_styles(pointwise_by_program, labelled).

archive.FrameArchive(directory).scratch(program) -> Path        # record here, commit is a rename
FrameArchive.commit(program, key, video, rows, *, width, height, fps) -> dict
FrameReader.stream(*, start=0, count=None, width=None) -> Iterator[rgb24]  # ffmpeg scales
FrameReader.passes(*, width=None, native=()) -> Iterator[(scaled, full | None)]  # one decode

replay.replay(archive, *, analyzer, programs=None, jobs=None) -> dict[str, Replayed]   # a worker per program
replay.replay_program(archive, program, *, analyzer, loop) -> Replayed   # records, profile, behaviour, notes
replay.setpoints(reader, *, width) -> Iterator[(setpoint, params, frames)]   # GAP rows dropped

harvest.sweep(session, base, vectors, *, config) -> list[Window]   # when each setpoint was held
harvest.attribute(times, windows, program, *, base) -> list[FrameRow]   # frame -> setpoint
harvest.harvest(archive, *, open_transport, recorder, player=None, link=None,
                programs=None, config=HarvestConfig(), session_factory=Session,
                retry_dark=False) -> HarvestReport   # dark = no frame ever structured
harvest.control(session, recorder, transport, *, config) -> str   # Passthru control: "" only on a picture
harvest.rig_fault(session, recorder, transport, *, config) -> str   # "" unless gone, wrong route or a black Passthru
recorder.require_route(transport, *, program, route) -> int | None   # RouteError unless Vid Route Mode reads route
HarvestReport.summary() -> list[str]   # totals, each failed program with its error, the fault
```

## `syncsummoner.compose`

```python
features.analyze(video_path, audio_path, *, rng) -> Features
vocabulary.GESTURES: dict[str, Gesture]
planner.search(profiles, features, *, style, rng, budget) -> Score
planner.STYLE_WEIGHTS: dict[str, ObjectiveWeights]   # naturalness is not a cost for every style
render.device_programs() -> Offered                 # names the attached device offers, .authors enumerated
render.unknown_programs(programs, available) -> list[str]   # named but not offered; None checks nothing
# render_cuts lists again once when a name is missing, then raises UnknownProgramError
# before the first pass with both listings (count, authors, sample, near names), so
# firmware that moved under the archive costs no rig time.
# render_played: a load probe or take with no picture runs the Passthru control first;
# a failed control raises RigFaultError, a passing one reloads and retries the pass
# PASS_RETRIES times before BlankTakeError "the program outputs no picture".
score.Score  # timeline IR, YAML round-trip
render.render(score, source, out, *, passes=1) -> None
render.Rig(session, capture, playout, link=None)   # link is dropped across every program change
locktake.frame_plan(auto, *, fps, early_bias_s) -> PLAN array   # (target frame, param, value), first frame at or after t - bias
locktake.locked_take(rig, recorder, out, *, program, seconds, config, plan, working, load) -> Record  # every write sent for its frame
locktake.take_evidence(take, *, program, clip, config, names=None) -> evidence.Evidence   # the three links; .ok is the hard check
evidence.send_time(frames, ticks, times, delta) -> ns   # just after the latest instant the tick before each frame's latching tick can fall
```
