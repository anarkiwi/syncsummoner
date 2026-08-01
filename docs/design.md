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

The device exposes 24 programs, 12 parameters each (11 knobs plus the crossfader
as parameter 12), three macros (Time / Space / Slope), full MIDI CC addressing,
and a USB CDC serial command interface for program and preset management. That is
enough to drive an unattended sweep. Program parameter names and ranges come from
`.vmprog` TOML metadata rather than being hardcoded.

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
| `natural_clip` | 4-second real footage loop; programs behave differently on camera sources |

### 3.2 Frame identification

**This is the crux of unattended sweeping.** The device is effectively
zero-latency but the capture chain is not, and settle times vary per program.

Burn a binary state-index blob into a thin edge strip of every emitted pattern
frame (8-bit gray code, high contrast, cropped before analysis). Each captured
frame then self-identifies which parameter vector produced it. This removes all
timing guesswork from the sweep and permits fast stepping without conservative
dwell times.

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
    name: threshold
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
| `punch` | onset | effect buttons, momentary |
| `morph` | phrase | crossfader between two states |
| `hysteresis_loop` | 2-bar | path-dependent parameters |

### 4.4 Timing rules

From the perceptual constraints:

- Schedule so the visual **arrival** lands on the beat, not the cut. Motion with
  an impact point entrains far better than a flash.
- Compensate end-to-end latency measured during probe.
- Bias visual events 10–20 ms early. Audio-leading-video is detected at roughly
  30–50 ms, while the opposite direction tolerates over 100 ms — so the error
  budget is asymmetric and visuals should never run late.

### 4.5 Planner

Sample candidate scores, render a fast proxy (30-second excerpt, downscaled),
score it, keep and mutate the best. A simple evolutionary loop is sufficient and
debuggable.

Objective terms:

- reward correlation between visual IC and audio onset strength / IC
- penalize channel-energy concentration (mud — energy piled into one Gabor cell)
- keep spectral slope in the natural band by default, with deliberate excursions
  permitted on hits
- penalize illegal levels and clipping, unless the section is marked `destroy`
- reward motif return at section boundaries — visual rhyme, states revisited with
  variation
- penalize low variance across any bar (anti-boredom floor)
- hard constraint: reachability, given measured settle times and a CC budget of
  roughly 200 msg/s aggregate

---

## 5. Render

Offline, multi-pass, all real-time captures.

1. Burn timecode into a croppable edge strip of the source; play out over HDMI.
2. Run the score's CC automation against the transport; capture the return.
3. Align by timecode, crop the strip, write the pass.
4. Optionally re-feed pass output through a second program with its own score
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

syncsummoner compose --profiles profiles/ --features features.json \
                     --style glitchy --seed 7 -o score.yaml

syncsummoner audition score.yaml --seconds 30
syncsummoner render score.yaml --source clip.mp4 --passes 2 -o out.mov
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

Section 5 inherits "720p59.94 or 1080p30" from the predecessor. Measured on this
rig, `1080p30`, `1080i60` and `480p` produce no signal at the capture card;
only `NTSC` and `720p60` do. Playout is a Pi on composite at 720x480i, so the
whole chain is NTSC. See `hardware.md`.

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
discarding the take.

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
