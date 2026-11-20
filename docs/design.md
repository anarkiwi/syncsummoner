# syncsummoner — Design

Automated characterization of the LZX Videomancer, and generative offline video
processing driven by measured device behaviour plus musical structure.

- **Status:** design draft
- **Depends on:** `pyvmancer` (hardware transport)
- **Target hardware:** LZX Videomancer, firmware 1.0.0-rc.13 or later

---

## 0. Name mapping

| Old | New | Role |
|---|---|---|
| `pyvmancer` | `pyvmancer` (unchanged) | hardware transport — MIDI CC, USB CDC serial, program load |
| `vmlab` | `syncsummoner` | the project: probe, profile, compose, render, CLI |
| `vjasthetic` | `syncsummoner.aesthetics` | perceptual metrics library, extraction-ready |

Two things change shape beyond the rename:

1. `syncsummoner` absorbing the lab role means the probe subsystem is no longer a
   sibling tool that hands off artifacts across a process boundary — it is a
   first-class subcommand sharing the device layer with the renderer.
2. `vjasthetic` becoming an internal package means the extraction seam has to be
   built deliberately *now*, or pulling it out later becomes a rewrite.

---

## 1. Layout

```
syncsummoner/
  cli.py
  device/
    transport.py     # wraps pyvmancer: CC, serial verbs, program load
    session.py       # settle timing, CC rate limiting, state cache
    capture.py       # HDMI capture card ingest
    playout.py       # source video -> device input
    profile.py       # ProgramProfile / ParamSpec dataclasses
  probe/
    patterns.py      # stimulus generation
    plans.py         # OAT, Sobol, tongue-raster, hysteresis
    runner.py        # execute, capture, align, log
    sim.py           # GHDL vhdl-image-tester backend
    fit.py           # measurements -> ProgramProfile
  compose/
    features.py      # audio/video feature extraction
    vocabulary.py    # gesture primitives
    planner.py       # search over profile -> score
    score.py         # timeline IR + serialization
    render.py        # offline capture passes
  aesthetics/        # <- former vjasthetic
    __init__.py      # THE public API; nothing else is imported elsewhere
    channels.py      # Gabor / steerable pyramid energy
    spectrum.py      # 1/f slope, fractal dimension
    motion.py        # optical flow, temporal statistics
    dynamics.py      # periodicity, winding number, stability class
    surprisal.py     # predictive model -> IC, entropy
    sync.py          # AV alignment measures
    score.py         # weighted aggregate facade
    py.typed
```

---

## 2. The extraction seam

The point of naming `aesthetics` a library-within-a-project is that it leaves
cleanly later. That requires enforcement, not intent.

1. **`aesthetics` imports nothing from `syncsummoner.*`.** Enforced in CI with an
   `import-linter` forbidden contract, not by discipline.
2. **Plain data at the boundary.** Frames in as `np.ndarray (H, W, 3) float32 in
   [0, 1]`; audio as `(samples,) float32` plus `sr: int`. No `Frame` wrapper, no
   device handles, no profile objects. Every dataclass it returns is defined
   inside `aesthetics`.
3. **Pure and seeded.** No I/O except an explicit `aesthetics.io` shim that the
   rest of the project does not use. RNG passed in, never module-global.
4. **Own dependency group** in `pyproject.toml` as
   `[project.optional-dependencies] aesthetics = [...]`. Extraction becomes a
   `git filter-repo` plus a `pyproject` rename, not a dependency untangling
   exercise.
5. **Facade discipline.** `syncsummoner` only ever calls
   `from syncsummoner import aesthetics` and touches names re-exported in
   `__init__.py`. That file is literally the future public API — treat breaking
   it as a version bump today.
6. **Independent version string.** `aesthetics.__version__`, recorded into every
   measurement record. Metrics evolve; without this, profiles measured across
   versions silently become incomparable and the planner optimizes against noise.
7. **Own tests** under `tests/aesthetics/` with zero hardware fixtures, plus
   `python -m syncsummoner.aesthetics score clip.mp4` as a standalone entry
   point. If that command works with the device layer uninstalled, the seam is
   real.

---

## 3. Probe — discovery and measurement

The device exposes 24 programs, 12 parameters each (11 knobs plus the fader as
parameter 12), three macros (Time / Space / Slope), full MIDI CC addressing,
and a USB CDC serial command interface for program and preset management. That is
enough to drive an unattended sweep. No slot means the same in every program: each
parameter's kind and initial value come from the program's own `.vmprog`
declaration (`syncsummoner.program`), its label from the device's `program info`,
and which parameters can reach the picture in a given state from the program's HDL
([program-model.md](program-model.md)).

### 3.1 Stimulus battery

Programs are strongly content-dependent, so a single test pattern will lie to
you. Minimum set:

| Pattern | Reads out |
|---|---|
| `zoneplate` | sine zone plate — all spatial frequencies x orientations in one frame; single capture gives the spatial-frequency transfer directly |
| `grating_sweep` | drifting gratings at chosen (cpd, Hz) — samples the spatiotemporal surface |
| `smpte_bars` | colorimetry, chroma phase handling |
| `luma_ramp`, `chroma_ramp` | transfer curve, quantization step count |
| `dot_lattice` | displacement fields, by correspondence |
| `siemens_star` | orientation-dependent aliasing |
| `noise_1f` | behaviour on natural-statistics input |
| `flat_grey`, `black` | noise floor; self-oscillation in feedback programs |
| `motion_ball` | trajectory with a hard impact point — temporal response |
| `stimulus` | the archive's moving picture, made from a seed (`compose.stimulus`): shots of band-limited colour noise under known drift, spin and zoom, each frame's motion scaled to a TI drawn afresh for it (so the picture's TI is white and every frame is an independent sample for the follow test), their texture scale, contrast, level and colour spanning set ranges, so what a program keeps of its input is measured against motion known on every frame, lead-in and lead-out included (§9) |

### 3.2 Frame identification

**This is the crux of unattended sweeping.** The device is effectively
zero-latency but the capture chain is not, and settle times vary per program.

Burn a binary state-index blob into a thin edge strip of every emitted pattern
frame (8-bit gray code, high contrast, cropped before analysis). Each captured
frame then self-identifies which parameter vector produced it. This removes all
timing guesswork from the sweep and permits fast stepping without conservative
dwell times.

Footage cannot carry the state index in its picture, and needs a clip frame
index instead: its strip carries the clip frame, the setpoint comes from the
capture time, and which clip frame each capture shows comes from a fitted player
clock (§9.1).

### 3.3 Plans, in order of cost

- **Census** — enumerate programs over serial, pull TOML parameter metadata,
  snapshot defaults.
- **Settle characterization** — step one parameter, measure
  frames-until-framediff-plateaus. Programs that never plateau are flagged
  `non_settling` and route to the dynamics analyzer rather than the static one.
- **OAT sweep** — each parameter across its range at ~32 steps, others at
  default. Cheap; yields monotonicity, dead zones, thresholds, discontinuities.
- **Sobol sample** — quasi-random over the 12-D cube, a few hundred points per
  program, to catch interactions that OAT structurally cannot see.
- **Tongue raster** — for any parameter pair where OAT showed periodic response,
  dense 2-D raster of Time-macro x parameter, computing the winding number of the
  output's temporal periodicity against frame rate. Produces the Arnold-tongue
  map: locked regions (stable pattern), narrow-ratio regions (slow drift), and
  chaotic regions. This is the consonance/dissonance control surface, measured
  rather than guessed.
- **Hysteresis probe** — approach identical setpoints from below and above, and
  with slow vs. fast ramps. Feedback programs on an FPGA are stateful;
  path-dependence is a feature to catalogue, not an artifact to average away.

### 3.4 Simulation backend

The SDK's `vhdl-image-tester` runs programs as GHDL simulations against still
images with no hardware attached. Use it for coarse pre-screening of static
parameter response and for programs without hardware time available. It cannot
give temporal or feedback behaviour, so every measurement record carries
`source: hw | sim`, and the planner discounts sim-derived entries.

### 3.5 Per-sample metrics

All computed via `aesthetics`:

- channel energy vector over a 4-orientation x 5-scale Gabor bank
- spectral slope and fractal dimension
- luma / chroma histogram stats, illegal-level and clipping fractions
- frame-diff energy, optical flow magnitude and coherence
- autocorrelation periodicity, estimated winding number
- distance-from-passthrough (does this program do anything on this input at all)
- stability class in `{static, periodic, quasiperiodic, chaotic}`

### 3.6 Fit output — `ProgramProfile`

```yaml
program: bitcrush_displace
firmware: 1.0.0-rc.13
analyzer: aesthetics 0.4.1
source: hw
params:
  - id: 3
    name: threshold                # the device's own label, from program info
    kind: continuous               # boolean | quantized | continuous | unused, as declared
    axis: color_destruction        # canonical axis assignment
    response: [...]                # metric vs value curve
    sensitivity: 0.82
    monotonic: false
    dead_zone: [0, 19]
    cliffs:
      - at: 64
        jump: 0.61
        metrics: [clip_frac, ic]
    hysteresis: true
interactions: [[3, 7, 0.44], ...]  # pairwise non-additivity
lock_map:
  pairs:
    - a: time_macro
      b: 5
      tongues: [...]
stability_by_region: [...]
```

Two derived artifacts matter downstream:

- **Cliff atlas** — parameter positions where a small delta produces a large
  metric jump. These are the glitch seeds; glitch becomes targeted rather than
  random.
- **Axis assignment** — maps each program's idiosyncratic parameter onto a
  canonical axis in `{texture_scale, motion_rate, displacement,
  color_destruction, key_threshold, feedback_gain, noise}`, so the composer can
  write program-agnostic gestures.

Stored as versioned YAML/TOML plus a thumbnail sprite sheet plus a Parquet file
of raw measurements.

---

## 4. Compose — video + beats to score

### 4.1 Two control timescales, deliberately split

The device has four audio/CV inputs. Fast reactive work — per-onset, audio-rate —
should be patched into those directly from band-split audio, so the FPGA handles
it with no round-trip. `syncsummoner` handles the slow structural layer over MIDI
CC: bar-level, section-level, arrangement.

Do not attempt transient response over MIDI; it loses to jitter and CC bandwidth.

### 4.2 Audio features

Beats, downbeats, tempo curve; onset strength; four band envelopes rendered to a
WAV for the CV inputs; structural segmentation via self-similarity matrix and
checkerboard kernel, giving sections.

Also compute audio information content over the onset sequence. The interesting
objective is not visual surprise in isolation — it is *correlation between visual
surprisal and musical surprisal*.

### 4.3 Gesture vocabulary

Each primitive carries a musical anchor and consumes a specific profile field.

| Gesture | Anchor | Profile field used |
|---|---|---|
| `hold` | bar | safe range |
| `ramp` | bar / phrase | monotonic segments |
| `cliff_cross` | downbeat | cliff atlas |
| `detune_drift` | across N bars | tongue map, near-lock offset |
| `lock_snap` | section boundary | tongue centers |
| `punch` | onset | boolean parameters, pulsed and released |
| `hysteresis_loop` | 2-bar | path-dependent parameters |
| `setting` | bar | `effects`: one measured setting, every parameter |

A gesture picks only parameters of the declared kinds it drives: `punch` the
booleans, every other drawn gesture the continuous and quantized ones; an `unused`
slot is never written. `setting` is the composer's (§12): it writes a measured
setting's slots one a frame, as archived, and is neither held to its cut nor gated.

### 4.4 Timing rules

From the perceptual constraints:

- Schedule so the visual **arrival** lands on the beat, not the cut. Motion with
  an impact point entrains far better than a flash.
- Compensate end-to-end latency measured during probe.
- Bias visual events 10–20 ms early. Audio-leading-video is detected at roughly
  30–50 ms, while the opposite direction tolerates over 100 ms — so the error
  budget is asymmetric and visuals should never run late.

### 4.5 Planner

The planner maps measured effects onto the music and the source (§12). The cut plan
is composed, not left to the render: the score stores it (`cuts:`) and the render
splices by it (`compose.cut.cut_plan` is the one function both call).

---

## 5. Render

Offline, multi-pass, all real-time captures.

1. Burn a Gray-code frame index into a croppable edge strip of the source, after
   a strip-coded lead-in; the Pi's locked player plays it over HDMI (§10).
2. Per take: load the program and hold it dry, start recording, start the clip;
   toggle the dry control in the lead-in to time the device's frame tick, attribute
   the growing recording, then send every CC half a frame before the instant it
   must precede to land on its frame ([evidence.md](evidence.md)). A program with no dry control loads
   inside the lead-in under Passthru instead. The player's flip log and the wire
   record are kept beside the take (`<take>.lock.npz`, `<take>.wire.npz`), and the
   take must prove every write landed on its frame and read back.
3. Attribute every clip frame to one capture exactly (§10), crop the strip, write
   the pass.
4. Check every take against the clip it was fed, bar by bar, and cover or fail
   before mastering (§9.6); screen the mastered picture for flashing, and fail
   rather than ship it.
5. Optionally re-feed pass output through a second program with its own score
   layer, then composite offline. Layering is where depth beyond a single program
   comes from.

`audition` renders 30 seconds at low resolution for planner iteration; `render`
does the full pass.

---

## 6. CLI

```
syncsummoner device list | ping | programs

syncsummoner probe run --program all --plan oat,sobol,tongue --out profiles/
syncsummoner probe sim --program blur --image zoneplate.png

syncsummoner profile show | diff | atlas --program bitcrush_displace

syncsummoner analyze clip.mp4 --audio track.wav -o features.json

syncsummoner compose clip.mp4 --audio track.wav --profiles profiles/ \
                     [--follow 0.5] [--intensity 0,1] [--change-bars N] [--quicken 0] \
                     [--passes N] [--kind auto|bw|colour] --format 1080p30 -o score.yaml

syncsummoner audition score.yaml --seconds 30
syncsummoner render score.yaml --source clip.mp4 --passes 2 -o out.mov
syncsummoner check score.yaml --source timecoded.mkv --takes takes/ --cut-programs A,B --splice cut.mp4
syncsummoner check score.yaml --source timecoded.mkv --master cut.mp4
syncsummoner screen final.mp4
syncsummoner lockcheck source.mkv --take lockcheck.mkv [--program NAME]
```

Standalone library entry point, which doubles as a seam test:

```
python -m syncsummoner.aesthetics score clip.mp4
```

---

## 7. Migration checklist

1. `git mv vmlab syncsummoner`; `git mv syncsummoner/vjasthetic syncsummoner/aesthetics`.
2. Update `pyproject.toml`: project name, script entry point, add the
   `aesthetics` optional-dependency group.
3. Add `aesthetics/__init__.py` as an explicit re-export list; route every call
   site in `syncsummoner` through it and fix what breaks.
4. Add the `import-linter` contract forbidding `syncsummoner.aesthetics` ->
   `syncsummoner.*`; wire into CI.
5. Add `aesthetics.__version__`; stamp it into `ProgramProfile` and every
   measurement record. Add a `--rescore` path so existing captures can be
   re-analyzed under a new analyzer version rather than discarded.
6. Add `tests/aesthetics/` and a CI job that installs only the aesthetics extra
   and runs them. That job failing is the early warning that the seam has leaked.

---

## 8. Load-bearing decision

Axis assignment and the cliff atlas live in the **profile**, not the planner.

Gestures are therefore written against canonical axes, which means a new firmware
release with new programs gets absorbed by re-probing rather than by editing the
composer.

---

## 9. Footage and survival

Synthetic patterns say what a program does to a test signal; whether a take is
usable depends on what it keeps of the footage it was fed. An archive run can
therefore play real footage, and every measure below asks one question: does the
output still show the source's picture and motion, frame against exact frame.

### 9.1 Footage stimulus and frame identity

`prepare_footage` writes the footage as a timecoded clip at the session raster:
a 16-bit Gray-code clip frame index in the bottom strip, the picture above it.
The clip and its record (`stimulus.mkv`, `stimulus.json`, with the clip's
SHA-256, frame count, rate, bits and strip height) live in the archive
directory, and every archive committed against it names it (schema 4). The Pi
plays it with `ClipPlayer(loop=True)`, the locked player (§10) decoding the clip
from tmpfs, uploaded once per run. An encoded clip is used because a few minutes
of it fit tmpfs; looping keeps one uninterrupted picture under a whole sweep
([hardware.md](hardware.md#locked-playout)).

Setpoints attribute by capture time, as for the codeframe loop. The clip frame a
capture shows comes from a `ClockModel` of the player: frame `n` is on show
around `origin + n * interval`, wrapped over the loop.

| Step | What | Why |
| --- | --- | --- |
| agreeing runs | keep captures whose strip agrees with the modal lag over more than `1 + log2(n) / bits` distinct frames | a misread lands on a given lag with odds `2^-bits`, so such runs are expected less than once by chance; a held strip spans one frame |
| L1 fit | LP for start and period minimising total time outside each capture's frame interval | a torn or misread strip costs only its own slack; least squares would let it pull the line |
| max-margin fit | LP over the captures the L1 line fits exactly, maximising the least margin | centres the line in what the captures allow, instead of tilting with the slow beat between card and player phase |
| refit | repeat over every legible capture the line places in its own frame | uses all the evidence the strip gave |
| motion fallback | where no strip reads: output TI summed on grids at the clip rate, rank cross-correlated with the source TI at every whole-frame lag (circular over a loop), grid phase stepped by half the capture spacing | ranks so no single cut decides; the phase step is as fine as the captures resolve |

The card captures faster than the player shows, so `attribute` keeps, per clip
frame, the capture nearest the centre of its display interval, furthest from
either switch; the others are unattributed. A program that destroys the strip
loses no frames, since attribution is by time.

### 9.2 Survival measures and their tests

`aesthetics.survival` reads ITU-T P.910 spatial information (SI, standard
deviation of the Sobel magnitude of luma) and temporal information (TI, standard
deviation of the luma difference between consecutive frames), both on 160x90
area-scaled luma, for the source clip and for the attributed captures.

| Measure | Definition | Threshold and its origin |
| --- | --- | --- |
| `si_ratio`, `ti_ratio` | output mean SI, TI over source mean, same frames | live floor: geometric mean of the source's value and quantisation noise alone (Weber–Fechner: perceived magnitude is logarithmic, so the midpoint between source and noise is their geometric mean) |
| quantisation SI | `q * sqrt((4 - pi) / 2)`, `q = 1/255` | the Sobel magnitude of i.i.d. quantisation noise is Rayleigh with scale `q` |
| quantisation TI | `q / sqrt(6)` | difference of two i.i.d. `q^2/12` fields |
| `motion_follow` | Pearson correlation of output TI with source TI, at lead 0 or 1 | Fisher z of `r` at the effective sample count must exceed the three-sigma one-sided level, Bonferroni-corrected over the two leads |
| effective samples | `n (1 - ab) / (1 + ab)`, `a`, `b` the lag-one autocorrelations | TI series are autocorrelated; AR(1) correction (Bretherton et al. 1999) keeps the z honest |

Lead 1 remains only for recordings made before the locked player, whose unsynced
framebuffer could scan out the strip one frame ahead of the picture above it.

### 9.3 Profiles carry survival

A refit of a footage archive scores each setpoint group on the clip frames its
captures show, adding `si_ratio`, `ti_ratio` and `motion_follow` to its record,
and the fit keeps them among each parameter's measured curves, first-order
marginals `E[metric | parameter value]` by Nadaraya–Watson regression with a
leave-one-out bandwidth. The composer no longer predicts survival from them: a
sum of one-parameter averages cannot carry one parameter gating another, and the
archive measures each setting's intensity and safety outright (§11), so the
composer plays measured settings only (§12). Survival is judged on the takes
themselves (§9.6).

### 9.5 Phrase-length sections

`phrase_bars` is the longest whole-bar grid, counted from the first downbeat, on
which more than half the inner section boundaries (each snapped to its nearest
downbeat) lie; one bar always qualifies. Sections are split on that grid,
keeping label and destroy mark, so gestures and cuts change at phrase length
rather than only at the segmentation's boundaries. The phrase is the composer's
default spacing of effect changes (§12); the score records `bar` and `phrase_bars`
in its metadata.

### 9.6 Take check before mastering

`render_cuts` runs `checked_plan` on every take before splicing, and `check`
runs it standalone. Each recording is decoded once (body at 160x90 luma, strip at
full width, times from the decoder's own `showinfo`) and the decode kept beside it
([judging.md](judging.md)). A locked take is attributed
exactly (§10) and `checked_plan` passes that attribution to `check_takes`;
`check` reads it from `<take>.picked.npy`. Only a recording without one, an
unlocked recording or a master, is traced by the strip clock and the motion
clock, keeping the clock under which output motion follows the source best, and
`check` warns that its clock is fitted. Each bar is then judged, the bars being
the planner's: the score's `downbeats` (else every `bar` seconds from zero),
split at every cut, so each cut is judged on the frames it shows alone:

| Fault | Condition |
| --- | --- |
| `flat` | at most half the frames span more than one 8-bit code, or SI under the live floor |
| `flashing` | a transition between two of the bar's frames counts toward a second that fails the general or red flash test (§11.2), on the pictures the take shows, clip frame by clip frame; judged first |
| `flat` | at most half the frames span more than one 8-bit code, or SI under the live floor |
| `frozen` | TI under the live floor, or `recorder.held` cannot tell the bar from one picture under capture noise and its motion does not follow the source (a held picture's TI is capture noise, independent of the source's) |

A bar is dead when blank (`flat`) or static (`frozen`), as §11 judges a
setting; motion that does not follow the source's is alive. A dead run inside a
cut is replaced by the take of another program alive across the whole run that
follows best (highest mean z), a run reaching the cut's edge taking that edge
itself so no sliver of the cut is left beside it; a lone dead bar inside a
`destroy` section stays. A `flashing` bar is covered the same way, and never stays.
A run nothing covers raises `DeadTakeError` naming the program and time range.
On a fitted trace, where a strip reads, its clock stands unless the motion clock makes the
output's moving frames follow the source better by more than the critical z
(chance among the lags searched), which a strip a program scrambled loses.

The flash test reads the calibration-size planar YUV (96x54) the decode already
keeps, so nothing is decoded again for it. That size is an area scaling of the
picture, and the test counts only screen means and the share of the screen that
changed, which an area scaling keeps; it is also under the 128-pixel side
`flash_risk` scales to itself. The planes are converted to RGB as a decoder
shows them: in the range and matrix the decode's own `showinfo` reports for
that branch (ffmpeg keeps a YUV stream's range and scales RGB to studio swing;
BT.601 unless tagged BT.709), with the chroma pair exchanged back for a take
that came back exchanged, as the splice reads it. A clip frame no capture shows
holds the one before. Each take's peak general and red flash rates are logged
on its alignment line. A transition across a bar's edge belongs to neither bar:
covering either changes it, so the splice's own joins are left to the last gate.

`master` writes the finished clip beside its output and screens it whole
(`takecheck.gate`), every frame decoded to the same size and read the same way.
One that fails is kept as `<output>.unsafe.<ext>` and the render stops with
`UnsafeOutputError` naming the seconds that fail and the peak rates; nothing is
left at the output's name. `screen` runs the gate on any finished picture.

A trace's `picked` is the attribution: per clip frame, the capture nearest the
centre of its display interval, or -1. It is the one interface between alignment
and everything after it: `check_takes(attributions=...)` takes one outright
instead of fitting a clock, and `splice` cuts by it. The splice writes clip frame
`n` from `takes[program(n)]` at capture `picked[n]`, decoding every take once,
forward, in parallel, each handing over one frame at a time, so memory is bounded
and no seek rounds a cut. A spliced master is therefore checked on the identity
clock (`ClockModel.identity`).

---

## 10. Locked frame clock and exact attribution

Every stimulus is played by the native player (`native/kmsplay`) on the Pi: atomic KMS
page flips, each clip frame held `refresh / fps` vblanks, every flip logged with
its vblank sequence and time, and any missed vblank a fault
([hardware.md](hardware.md#locked-playout)). `ClipPlayer` plays an uploaded clip,
`LoopPlayer` cycles uploaded raw frames through the same player at a rate that
is a whole number of vblanks (10 fps by default). The card's captures are kept,
with their driver sequence, timestamp and digest, in a `<take>.v4l2` sidecar by
`syncsummoner.device.v4l2rec`
([hardware.md](hardware.md#capture-is-stored-never-decoded-while-it-runs)).

A take's clip opens on a lead-in (`RenderConfig.lead_in_s`, 40 s) whose strips
number its frames `-L..-1`, wrapped to the strip's 16 bits, and closes on a lead-out
as long, its last frame held with the numbering carried on (`render.tail_frames`), so
captures under the dry program, or before a program load and after Passthru is reloaded,
carry legible strips at both ends whatever the program does between.

`compose.lock.attribute_take(take, config=)` reads the take's strips from its decode, its sidecar
and its lock record, and returns an `Attribution`:

| Field | Meaning |
| --- | --- |
| `picked[j]` | per clip frame `j`, the capture index (into the take's stored frames, in order) nearest the frame's centre on its frame line, the later on a tie; -1 where none shows it |
| `shown[k]`, `starts[k]` | per capture, the clip frame it shows, negative in the lead-in, and whether it is its frame's first |
| `per_frame`, `lead` | captures per clip frame, rounded; lead-in frames played |
| `line` | the frame line of the run the clip plays in (`compose.frameline.Line`): `origin + n * period` on the capture clock, the polygon of lines the take's frame starts admit, its slack and residual |
| `drift_ppm`, `drift_range_ppm`, `slips` | the period against `per_frame` capture periods, the range the polygon admits, and the frames captured other than `per_frame` times |
| `jitter_ns`, `uncertain`, `tied` | the timestamps' largest residual off the capture line; captures a boundary the polygon admits may cross, and those whose bytes match both neighbours or neither, which keep the line's frame |
| `frame0_ns` | when clip frame 0 started, on the capture clock |
| `read`, `agreed`, `ahead`, `faults`, `strays`, `rejected` | legible strips, those reading the frame `shown` says and those a later one; strips a frame off at an uncertain capture; byte changes in the clip no line meets; lead strips whose group no line through the largest meets |

`summary()` is one line of all of it, `record()` the same as JSON. `picked` is saved as
`<take>.picked.npy`. The call raises `LockError` unless every clip frame is attributed and no
fault or stray contradicts the line, and `LostTimecodeError` for a take with no lock record, which
is then unusable for a render.

Each gap-free run's frame starts lie on a straight line on the capture clock
([hardware.md](hardware.md#the-cards-clock)). A capture starts a frame where its strip reads the
next frame or its bytes differ from the capture before's; each puts the start between the capture
before and itself. The strips of the dry lead-in, and of the dry lead-out after the clip, name their
frames; every other start is named, outward from them, by the one frame the polygon of lines through
the named starts admits, and clips it in turn. A miss within the capture jitter widens its bracket;
a lead strip group missing by more is refused, the largest one's a `LockError`. Each capture then
shows the frame its time falls in on the line, and one a boundary may cross goes by its bytes.
A program load breaks the capture and the card relocks at a new phase
([hardware.md](hardware.md#a-program-load-breaks-the-capture)); each run has its own line, and one
with no legible strip, as under a program that destroys it, is numbered by the lag from the Pi's
flips, which the load leaves alone.

`syncsummoner lockcheck` is the acceptance test of all of this on the rig: it
plays a timecoded clip through Passthru, or a program loaded in the lead-in, and
reports missed vblanks, driver drops, gaps, picture changes off a frame start, strip
agreement and, under Passthru, tears (`compose.lockcheck`).

## 11. Effect intensity and safety

What the composer needs from an archive is how far each setting takes the picture, from subtle to
exaggerated, and whether it leaves a picture that can be watched. `probe effects` measures both for every
archived setting, its captures against the exact clip frames they show (§9.1), and writes them into the
program's profile under `effects`, keyed `bw` or `colour` by the saturation of the synthetic clip played,
so one profile carries both clips.

```sh
syncsummoner probe timecode --synthetic 4 --shot-frames 45 --colour bw   --seconds 351.1667 --out bw.mkv
syncsummoner probe timecode --synthetic 4 --shot-frames 45 --colour full --seconds 351.1667 --out colour.mkv
syncsummoner probe effects --archive archive/ --profiles profiles/ --jobs 8 [--sheets sheets/]
syncsummoner probe doctor ARCHIVE_DIR   # each program's safe intensity range and flag counts
```

### 11.1 Clips

Both clips are the one-texture stimulus (`Coverage.one_texture`): every shot the same texture statistics,
only motion drawn. `--colour bw` sets saturation to zero, so the luma is the colour clip's and the pair
isolates what a program does with chroma. `--colour full` removes each shot's tint and sets the
saturation so three standard deviations of chroma, the luma's own span, meet the nearest BT.709 gamut
edge at the mean level (`gamut_saturation`). Untinted chroma noise spans every hue in every frame, so no
setting sees a hue another does not, and hue is held by construction rather than drawn per shot.

### 11.2 Measures

Captures and clip frames are compared with the strip cropped, area-scaled by Wang et al.'s SSIM rule:
the short side over 256, rounded (480x268 at 1080p).

| Measure | Definition | Source |
| --- | --- | --- |
| `structure` | mean DSSIM, `(1 - SSIM) / 2`, of luma: 0 identical, 0.5 unrelated, 1 inverted | Wang, Bovik, Sheikh and Simoncelli 2004, their window and constants |
| `colour` | mean CIEDE2000 in CIELAB, D65, from sRGB | CIE 142-2001; scikit-image |
| `intensity` | larger of `structure` and `colour / 100`, both fractions of the most an output can depart | 100 is CIEDE2000 between black and white |
| `blank` | output SI below the live floor between the source's SI and quantisation SI | P.910 SI, §9.2 |
| `static` | output TI below the live floor between the source's TI and quantisation TI | P.910 TI, §9.2 |
| `flashing` | more than 3 general flashes in any second | ITU-R BT.1702, WCAG 2.3.1, as EA IRIS implements them |
| `red_flashing` | the same for saturated red | WCAG 2.2 red flash definition |

A flash is a pair of opposing transitions. Each frame's change is the change in the screen's mean
relative luminance, counted only where at least a quarter of the screen changed; same-signed changes
within a second accumulate, and a run becomes one transition when it first reaches 0.1 with the darker
frame below 0.8. The red test runs the same on `320 * (R - G - B)` of linear RGB where red is at least
0.8 of the sum, with a threshold of 20. A setting is safe when it raises none of the four flags.
SSIM, not MS-SSIM: a contrast-reversed scale has no real fractional power, and clipping it at zero
saturates a large share of settings at the top of the scale, where ranking matters.

### 11.3 Between measured settings

`probe.effects.lookup` takes the nearest measured setting's intensity, over the parameters the archive
varied, each over its span, and calls a setting safe only when every measured setting within the median
nearest-neighbour spacing, the design's own resolution, is safe. At archive density an unmeasured
setting's intensity is barely informed by its neighbours across a dozen parameters, so the composer
chooses among measured settings and uses `lookup` only for a setting a score already holds.

---

## 12. Effect mapping

The composer needs only what §11 measures: how far each setting takes a picture and whether it stays
watchable. At render time it reads the beat, whether the source is black and white or colour, and the
source's own activity, and maps a configurable curve from those onto the measured settings
(`compose.planner`).

```sh
syncsummoner compose source.mkv --audio track.flac --profiles profiles/ --format 1080p30 \
    --follow 0.5 --intensity 0,1 [--change-bars 8] [--quicken 0] [--passes N] [--kind auto] -o score.yaml
```

| Step | What | Why |
| --- | --- | --- |
| clip | the measured clip whose RMS chroma per axis is nearer the source's; a clip's is `saturation / 6` (`stimulus.chroma_sigma`), the colour clip's set by the gamut (§11.1) | the only threshold is the midpoint between the two stimuli the settings were measured on; it assumes untinted clips, which `--colour bw` and `--colour full` are |
| source chroma | per frame, `sqrt(mean(Cb^2 + Cr^2) / 2)` of Rec.709 full-range `Cb`, `Cr` (`aesthetics.chroma_rms`), RMS over the clip | the same quantity the clip's saturation sets |
| bars | downbeat to downbeat, a pickup joined to the first bar | settings change on the bar, effects on downbeats |
| drive | per bar, the rank of `follow * rank(source TI) + (1 - follow) * rank(onset strength)`, each a bar mean | ranks are scale-free, so P.910 TI and onset strength mix without a conversion; the most and least driven bars always reach the bounds |
| target | `low + (high - low) * drive`, a quantile of every program's safe intensities on the clip | quantiles of what was measured, so `0` is the subtlest safe setting and `1` the most exaggerated whatever the archive holds |
| changes | each bar adds `(1 + quicken (2 drive - 1)) / bars` to a count; an effect starts where the count passes a whole number | `quicken 0` changes every `bars` bars on the phrase grid; positive crowds changes where the drive is high, negative where it is low, keeping their number |
| program | Viterbi over runs of bars: cost the summed distance from each bar's target to the program's nearest safe setting, no program twice running; `--passes` keeps programs greedily, each lowering the least total most | a change is a change of effect; each program kept is one capture |
| setting | each bar plays its program's safe setting nearest its target, its changed slots written one a frame so the last lands as the bar starts; a run's first, every slot, from when its program last left the screen | measured settings only, as measured: at archive density a neighbour's measurement does not predict a setting's (§11.3); one slot a frame is how the archive wrote them (`locked.staggered`) and always fits the link; off screen, the first settles by the cut |

Settings change at most once a bar, so a change cannot itself flash at the three a second §11.2
counts. The score's `meta` records the clip, both chromas, the curve, and each bar's edge, drive, target
and chosen intensity. Defaults: `follow 0.5` weighs picture and music alike; `intensity 0,1` spans
every safe setting; `change-bars` the track's phrase (§9.5) and `quicken 0` keep changes on its grid;
`passes` uses every program with a safe setting; `kind auto` lets the source choose.

---

## Appendix — theoretical grounding

Why these particular metrics, briefly:

- **Gabor channel energy.** The visual system performs frequency analysis on
  spatial and temporal structure (not wavelength), with channels of roughly
  one-octave bandwidth and ~30 degrees orientation bandwidth. Content within a
  single channel mutually masks and reads as mud; separation across channels
  segregates into legible layers. This is the closest real analogue to critical
  bands in hearing.
- **Spectral slope and fractal dimension.** Natural scenes and most artwork sit
  near 1/f (log-log slope about -1 to -1.4); preference tends to peak around
  fractal dimension 1.3–1.5. Cheap real-time proxies for whether a frame is
  statistically natural and therefore fluently processed.
- **Winding number and tongue maps.** Analog and scan-locked video is a coupled
  oscillator system. Locking regions in parameter space are widest at simple
  rational ratios and narrow rapidly for complex ones — structurally identical to
  why simple frequency ratios are the stable ones in music. Detuning against
  scan-derived rates gives a continuous, measurable tension parameter.
- **Information content / surprisal.** Consonance in practice is largely learned
  statistical expectation, its violation, and its resolution. Modality-neutral,
  and computable as `-log2 P(event | context)` over a learned model. Pleasure
  appears to track the *rate of prediction-error reduction*, which is why
  sustained detuning is dull and resolved detuning is satisfying.
- **AV asynchrony asymmetry.** Sound arriving after light is ecologically normal;
  tolerance for audio lag exceeds 100 ms, while audio lead is detected at roughly
  30–50 ms. Hence the early-bias rule in section 4.4.

Note that pitch-to-hue mapping is deliberately absent throughout. The cochlea
performs a running Fourier analysis so partials remain separable; the retina
integrates to three dimensions with no recoverable components. Consonance
mechanisms depending on partial interaction have no chromatic analogue. Wilfred
reached the same conclusion in the 1920s and moved to form, motion, and rate.

---

## Appendix B — review corrections

Recorded against the draft above after hardware bring-up. Items 1-4 are
corrections to what the draft specifies; items 5-9 are regressions against
`predecessor-design.md`, which the rename dropped and which are load-bearing.

### 1. Stale facts

Firmware is `1.0.0-rc.37`, not `1.0.0-rc.13`. The device reports 21 programs,
not 24. Parameter names and native ranges come from the serial `program info`
verb, not from `.vmprog` TOML — no file parsing is needed. The CC budget of
~200 msg/s in section 4.5 is asserted, never measured; it should be measured
during bring-up and stored per firmware.

### 2. Session format is not a free choice

Section 5 inherits "720p59.94 or 1080p30" from the predecessor. The device
genlocks to its source and cannot convert, so the format is whatever both ends
already run: the session is the Pi over HDMI at 1920x1080@30. What the card will
accept depends on what it is genlocked to — under an NTSC genlock only `720p60`
was usable. See `hardware.md`.

### 3. Parameters are additive, and heterogeneous

The draft's section 3 sweeps assume `set_param` is absolute. It is not:
`Parameter = Manual + Modulation + MIDI`, so MIDI CC is an offset onto the
physical knob position. Every sweep silently measures the wrong thing unless the
panel is parked at a known reference and addressing is verified empirically.

Section 3.3's "each parameter across its range at ~32 steps" also ignores that
P7-P11 are boolean and that some continuous parameters are quantised to a few
native steps. Sampling is now type-aware.

### 4. `feedback_gain` is not measurable as specified

There is no frame store and no internal frame feedback, so section 3.1's
"self-oscillation in feedback programs" and the `feedback_gain` canonical axis
require an external Dual In loop that does not currently exist.

### 5. No safety veto

The draft has none. The predecessor specified a photosensitive-seizure veto as a
hard constraint overriding any fitness value: flag more than three luminance
transitions per second over more than a quarter of the frame, plus the
saturated-red transition case, implemented as a relative-luminance proxy with
its limits stated. The output of this project is glitch content intended for
projection at audiences, which makes this the most consequential omission in the
rename. Mitigate by temporally low-passing the offending window rather than
discarding the take. The render instead covers a flashing bar from another
take and refuses a finished picture that still flashes (§9.6).

### 6. No noise floor

Repeat a handful of states twice per program and record the feature-space
distance between repeats. That number is the resolution limit of every
downstream conclusion. Section 3.6 reports `sensitivity` per parameter with no
reference to it, and a sensitivity smaller than the noise floor is not a
sensitivity.

### 7. Screening should be Morris, not OAT plus Sobol

Section 3.3 spends its budget on a 32-step OAT sweep per parameter and then a
few hundred Sobol points to recover the interactions OAT structurally cannot
see. Morris elementary effects answers both questions in `r * (k + 1)`
evaluations and is the standard tool for expensive black boxes. It yields `mu*`,
overall influence, which identifies inert parameters that can be frozen to
shrink the search space, and `sigma`, the spread of elementary effects, which
identifies parameters whose effect depends on where the others sit. High-sigma
parameters are both the interesting ones for glitch and the ones OAT
mischaracterises. Keep OAT only where a response curve itself is wanted.

### 8. The surprisal model must not be fitted in-sample

Fit on a corpus of prior work, not on the clip being scored. A model fitted
in-sample reports every clip as unsurprising, which silently inverts the
section 4.5 objective that rewards correlation between visual and musical
surprisal.

### 9. Beware the naturalness objective

Section 4.5 keeps spectral slope "in the natural band by default". Published
aesthetic priors are fitted to photographs, and glitch aesthetics systematically
violate them, so a naturalness term will confidently penalise exactly the output
this project exists to produce. Report the axis as a signed distance and let the
weighting decide the target, rather than treating conformance as good.

### 10. Frame identification supersedes PTS

The predecessor argued for PyAV over `cv2.VideoCapture` to get presentation
timestamps for drop and duplicate detection. The Gray-code state-index strip of
section 3.2 makes that unnecessary and is strictly stronger: it ties identity to
frame content rather than to container metadata, so a repeated index is a
duplicate and a skipped index is a drop, directly. Verified bit-exact through
the full analog chain.
