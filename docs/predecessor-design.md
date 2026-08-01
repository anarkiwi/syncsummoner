# vmlab + vjaesthetic

**A hardware-in-the-loop characterisation harness and generative renderer for the LZX Videomancer, driven by `pyvmancer`.**

Two deliverables, deliberately split:

- **`vjaesthetic`** — a standalone, hardware-agnostic library that turns video into interpretable perceptual features and fitted preference scores. Depends on numpy/cv2/scipy only. Publishable on its own.
- **`vmlab`** — the Videomancer harness: device control, capture, probe batteries, effect characterisation, archive search, and beat-driven rendering. Depends on `vjaesthetic` and `pyvmancer`. The dependency never runs the other way.

---

## 0. What the hardware forces on the design

These come from the Videomancer manual and the `pyvmancer` README, and several of them are load-bearing. Getting them wrong produces a harness that silently measures the wrong thing.

**Parameters are additive: `Parameter = Manual + Modulation + MIDI`.**
MIDI CC is an *offset* onto the physical knob position, not an absolute set. If P1's knob sits at centre, `set_param(1, 0.75)` may land at clip. Before any measurement session, park every physical control at a known reference (full CCW for knobs and the fader, all toggles off) and verify empirically — see `calibrate_param_addressing()` in §3. Do not assume `set_param` is absolute; measure it.

**Twelve parameters, heterogeneous types.** P1–P6 continuous knobs, P7–P11 boolean toggles, P12 continuous fader. Boolean parameters resolve as `on` when the combined value ≥ 512 of 1024. So the per-program search space is **7 continuous dimensions × 32 binary corners**, at 10-bit resolution. That is small enough to characterise exhaustively-ish and large enough to need a real search strategy.

**Modulation is deterministic from timecode.** Oscillators phase-reset to zero at `00:00:00:00`, and `STOP` resets timecode to zero. If Time properties are unchanged, *the same modulation pattern is generated every time playback starts from zero*. This is the single most useful fact in the whole design: renders are reproducible, so multi-take selection, A/B comparison, and offline scoring all become sound. Prefer internal transport or MIDI Clock (which resets on stop) for repeatability; use MTC only when you need pause/resume without phase reset.

**Modulation updates at frame/field rate.** No mid-frame tearing, and no sub-frame modulation resolution — except `Audio In`, which can run at scanline rate. Your control-signal Nyquist is half the frame rate. Don't design envelopes finer than that.

**Program loading blackouts.** Outputs drop for several seconds on program change. Program change is the expensive operation in any search loop: **batch all evaluations for a program together**, and treat program as the outermost loop variable, never as a mutable gene inside an inner loop.

**No frame store.** A few lines of memory only. There is no internal frame feedback. Feedback effects must come from an external loop via **Dual In** mode (HDMI in → analog out → external chain → analog in → processor). If you want the classic feedback-chaos behaviour, that loop is where it lives, and it becomes a characterisable subsystem in its own right.

**Genlock, no conversion.** Videomancer adopts the source's resolution and frame rate and cannot change either. Pick one format for the entire pipeline (recommend **720p59.94** or **1080p30** — high enough for detail, low enough that capture is reliable) and never vary it. Format is a session constant, not a parameter.

**All outputs are always active.** You can capture HDMI *and* composite simultaneously from the same run. Use HDMI as the measurement tap (clean, repeatable) and composite as the aesthetic tap (analog encode artefacts, chroma crawl). Two capture devices, one pass. This is worth doing from day one.

**Effectively zero device latency** (sub-microsecond). All the latency in your loop is player + capture card, not the instrument. That simplifies compensation to a single measured constant.

**Serial exposes named parameters with native units.** `set_named("Posterize", 5)` implies the program descriptor carries names, ranges, and types. Harvest these — they give you free semantic labels for the archive, and they tell you which continuous parameters are actually quantised (a "Posterize 0..7" knob has 8 meaningful positions, not 1024, and sampling it densely is wasted hardware time).

---

## 1. Package layout

```
vjaesthetic/                  # standalone, no hardware, no pyvmancer
├── features/
│   ├── spatial.py            # Gabor bank, radial FFT slope, fractal dim, edge stats
│   ├── colour.py             # colourfulness, hue stats, chroma gain, LAB moments
│   ├── temporal.py           # flow, flicker spectra, autocorrelation, beat energy
│   ├── structure.py          # MS-SSIM & departure, saliency dispersion
│   └── surprisal.py          # predictive model, information content, entropy
├── aggregate.py              # frame features -> clip descriptor
├── scoring/
│   ├── base.py               # Scorer protocol
│   ├── axes.py               # interpretable named axes
│   ├── veto.py               # hard constraints (flash safety, dead frames)
│   └── learned.py            # Bradley-Terry fit over pairwise human prefs
├── calibrate/
│   └── pairwise.py           # CLI/web tool to collect A/B preferences
└── io.py                     # decode -> frame iterator, downscale, colour convert

vmlab/
├── device.py                 # pyvmancer wrapper + state discipline
├── capture.py                # PyAV capture, PTS handling, alignment
├── chain.py                  # session config: format, routing, taps, latency
├── probes/                   # test signal generation (numpy/cv2 -> video files)
├── measure/
│   ├── runner.py             # settle/capture/retry loop, drift canaries
│   ├── transfer.py           # LUT, chroma matrix, spatial & temporal response
│   └── screen.py             # Morris elementary effects, sensitivity ranking
├── archive/
│   ├── genome.py             # parameter + modulation encoding
│   ├── mapelites.py          # QD search over the archive
│   └── store.py              # parquet + video thumbnails, queryable
└── render/
    ├── audio.py              # librosa analysis
    ├── source.py             # cv2 shot/motion analysis of input video
    ├── score.py              # arrangement generation
    ├── realize.py            # timed execution against MIDI clock
    └── select.py             # multi-take scoring and splicing
```

---

## 2. Device layer

Thin wrapper over `pyvmancer` whose entire job is **state discipline**. Every measurement must be attributable to a fully-specified device state, and the device has more hidden state than its API surface suggests.

```python
@dataclass(frozen=True)
class DeviceState:
    program: str
    params: tuple[float, ...]          # 12 values; bools coerced to 0.0/1.0
    sources: tuple[str, ...]           # modulation operator per parameter
    mod: tuple[tuple[float, float, float], ...]   # (time, space, slope) per param
    bpm: float
    transport: Literal["stopped", "playing"]
    route: str                          # HDMI In / Dual In / Standalone / Analog In

    def digest(self) -> str: ...        # stable hash -> cache key, filename
```

```python
class Device:
    def apply(self, state: DeviceState, *, force_program: bool = False) -> None:
        """Minimal-diff application. Reloads program only when it changed."""

    def quiesce(self) -> None:
        """Modulation disabled on all 12, transport stopped, soft pickup off,
        MIDI channel pinned. The reference state for all static measurement."""

    def settle_ms(self, prev: DeviceState, new: DeviceState) -> int:
        """Program change -> ~4000ms. Source change -> ~200ms.
        Param-only -> ~2 frame periods. Measured once, cached per firmware."""
```

Three disciplines that matter:

1. **`quiesce()` before every static measurement.** With operators disabled and transport stopped, the device is a pure function of parameters. That is the only regime where repeat measurements are comparable.
2. **Cache keyed on `DeviceState.digest()`.** Hardware evaluations cost seconds; never pay twice.
3. **Firmware version is part of every record.** `pyvmancer` is verified against `1.0.0-rc.37`; programs and operator lists are actively changing between release candidates. An archive built on rc.37 may not be valid on rc.44. Store it, check it on load, warn loudly on mismatch.

Enumerate at runtime rather than hardcoding — `vmancer programs`, `vmancer operators`. The README mentions 25 operators while the manual's Motion Overview symbol table lists around thirty; treat both as unreliable and ask the device.

---

## 3. Signal chain, capture, and alignment

### Physical chain

```
                                ┌─ HDMI out ──> capture A (measurement tap, clean)
computer HDMI ──> Videomancer ──┤
                                └─ composite ──> capture B (aesthetic tap, analog)

              (optional Dual In feedback loop: analog out ──> chain ──> analog in)
```

### Capture

Use **PyAV**, not `cv2.VideoCapture`. You need presentation timestamps and control over pixel format; cv2's UVC path discards both. Decode to `numpy` and hand off to cv2 for processing.

```python
class Capture:
    def frames(self, n: int, *, drop_first: int = 0) -> Iterator[tuple[int, np.ndarray]]:
        """Yields (pts_ns, BGR frame). Detects duplicate/dropped frames by
        comparing PTS deltas against the nominal frame period."""
```

Flag any run where PTS deltas deviate by more than half a frame period. Dropped frames corrupt every temporal feature, and they happen constantly on cheap capture hardware. A run with drops gets retried, not silently averaged.

### Alignment

The hard engineering problem. Three mechanisms, used together:

**Session latency calibration.** Once per session, load `Passthru`, play out a leader (2 s black, 3-frame white flash, 2 s black), cross-correlate captured luma against the known impulse. Yields end-to-end player→capture latency to ±1 frame. Re-run at session end; if it has drifted, the session's timing metadata is suspect.

**Program-independent optical marker.** The `SYSTEM → Test Pattern` setting toggles the ADV7393's built-in bars/hatch on the analog output, and it is settable over serial. Toggling it produces a hard visual event at a known command timestamp, independent of whatever program is loaded and whatever it does to the image. This is your fallback when the loaded program destroys the source enough that source-embedded markers are unrecoverable — which will be often.

**Guard frames.** After applying a state, discard `settle_ms` worth of frames, then capture `N` frames. Never capture across a transition. For a static measurement `N = 8` is enough; for temporal features you need `N ≥ 4 × frame_rate` to resolve slow oscillation.

### The addressing calibration

Run this before trusting a single measurement:

```python
def calibrate_param_addressing(dev, cap) -> ParamCalibration:
    """For each continuous parameter, sweep set_param 0..1 in 17 steps
    against a flat mid-grey field, measure output response.

    Detects:
      - whether set_param is absolute or additive onto knob position
      - actual usable range (early clipping => a knob isn't parked)
      - effective quantisation (a 0..7 native param shows 8 plateaux)
      - dead parameters for this program
      - hysteresis: repeat the sweep descending, compare
    """
```

Hysteresis matters more than it looks. If ascending and descending sweeps disagree, every subsequent measurement depends on approach direction, and the search must always approach setpoints from a fixed direction to stay reproducible.

---

## 4. Probe battery

Generated with numpy/cv2, encoded once to lossless files at the session format, played out. Each probe targets a different part of the transfer characteristic. Don't use a single test pattern — different probes are needed to identify different effect families, and the whole battery is only a few minutes of hardware time.

| Probe | What it identifies |
|---|---|
| **Luma staircase** (33 steps, ascending + descending) | Transfer LUT, gamma, clip points, posterisation level count, hysteresis |
| **EBU 75% colour bars** | Hue rotation angle, chroma gain, luma/chroma crosstalk, patch confusion |
| **2D zone plate** | Full spatial-frequency response at all orientations simultaneously, aliasing, ringing. The single most informative static probe. |
| **Multiburst** | 1D spatial frequency response, cheaper to interpret than the zone plate |
| **Moving dot** (known trajectory) | Point spread, displacement field, motion trails, temporal smearing |
| **Frame impulse** (1 white frame in black) | Temporal impulse response, decay constant, any line-memory behaviour |
| **Slow luma ramp over time** | Key/threshold position and hysteresis, comparator behaviour |
| **1/f noise field** (known spectral slope) | How the program transforms natural image statistics |
| **Reference footage** (10 s, fixed) | Ecological check — the manual is explicit that the same knob behaves differently on camera footage vs synthetic patterns |

For **Standalone** programs that generate without input, the probe set collapses to the temporal ones plus a null input; the harness must branch on route mode.

The zone plate is worth generating carefully:

```python
def zone_plate(h, w, k=0.35):
    y, x = np.mgrid[-h//2:h//2, -w//2:w//2].astype(np.float32)
    return (0.5 + 0.5 * np.cos(k * (x**2 + y**2) / max(h, w))).astype(np.float32)
```

Its instantaneous spatial frequency increases linearly with radius, so a single frame probes every (frequency, orientation) cell at once. Compare input and output 2D FFT magnitudes to get a full transfer surface, and look for output energy at frequencies absent from the input — that is aliasing, and it is a signature of the bit-reduction and displacement programs.

---

## 5. Measurement and screening

### Feature extraction

Everything here calls into `vjaesthetic.features` — the harness computes no perceptual features of its own. What's specific to `vmlab` is the *paired* comparison against a known probe input:

```python
@dataclass
class ProbeResponse:
    state_digest: str
    probe: str
    luma_lut: np.ndarray | None          # 256 -> 256, from staircase
    hue_rotation_deg: float | None
    chroma_gain: float | None
    posterize_levels: int | None
    spatial_transfer: np.ndarray | None  # 2D, from zone plate
    displacement: np.ndarray | None      # dense flow input->output
    temporal_decay_tau: float | None
    aliasing_energy: float
    noise_floor: float                   # from repeat measurement
```

The **noise floor** field is not optional. Re-measure a handful of states twice per program and record the feature-space distance between repeats. That number is the resolution limit of every downstream conclusion; a sensitivity smaller than the noise floor is not a sensitivity.

### Screening: which parameters actually do anything

One-at-a-time sweeps miss interactions, and full factorial is unaffordable. **Morris elementary effects** is the right tool for expensive black boxes: `r` random trajectories through the parameter space, each perturbing one parameter per step, `r × (k+1)` evaluations total.

With `k = 12` and `r = 10` that's 130 evaluations per program. At roughly 1.5 s each with no program reload, about 3 minutes per program — under two hours for a 25-program library including load overhead. Comfortably an afternoon.

Morris yields two statistics per parameter:

- **μ\*** — overall influence. Low μ\* means the parameter is inert in this program and can be frozen, shrinking the search space.
- **σ** — spread of elementary effects. High σ means the parameter's effect depends on where the others are, i.e. genuine interaction or strong nonlinearity. High-σ parameters are exactly the interesting ones for glitch, and the ones a naive OAT sweep would mischaracterise.

Output is a per-program **sensitivity report** that drives everything downstream: which dimensions the archive search bothers with, and which named parameters map to which measured effects.

### Empirical effect taxonomy

Rather than trusting program names, cluster the measured feature vectors: standardise → UMAP → HDBSCAN. This produces an empirical map of what the library actually does, cutting across program boundaries — you'll likely find that a corner of `Scramble` and a corner of `Bitcullis` are neighbours, which is exactly the kind of thing you want to know when building a set. Label clusters by hand once; the labels persist as archive metadata.

---

## 6. `vjaesthetic` — the separable library

This is the piece with a life beyond the Videomancer, so its boundaries need to be clean: **frames in, named numbers out, no hardware, no I/O beyond decoding.**

### Design stance

Ship **interpretable axes plus a fitted personal aggregator**, never a built-in universal "pleasingness" scalar. Two reasons. First, taste is genuinely personal and genre-specific. Second — and this is the sharp one for glitch work — most published aesthetic priors are fitted to photographs, and glitch aesthetics *systematically violate* them. A naive naturalness scorer will confidently penalise precisely the output you want. The axes are useful; a canned scalar built from them is not.

### Axes

Drawing on the perceptual grounding discussed earlier:

```python
class Axis(Protocol):
    name: str
    def __call__(self, clip: ClipFeatures) -> float: ...
```

| Axis | Computation | Note |
|---|---|---|
| `spectral_naturalness` | Radially-averaged FFT amplitude slope; distance from ≈ −1.2 band | Natural scenes and most artworks cluster near 1/f. **Signed** distance — glitch often wants controlled departure, not conformance. |
| `complexity` | Box-counting fractal dimension | Preference tends to peak around D ≈ 1.3–1.5, but with modest effect sizes. Report the value; let the fit decide the target. |
| `channel_balance` | Entropy of energy across a Gabor bank (≈1-octave, 30° bins) | Low entropy = energy piled into one (frequency, orientation) cell = visual mud, the direct analogue of stacking synths in one octave. |
| `mud` | Explicit detector: single-cell dominance ∧ low global contrast | A veto-adjacent axis; cheap and catches a common failure. |
| `departure` | `1 − MS-SSIM(source, output)` | The glitch axis. Wants a **band**, not a maximum: unmodified is boring, pure noise is unwatchable. |
| `motion_coherence` | Coherence of dense optical flow (DIS or Farnebäck) | Separates "moving" from "noisy". Both are useful; you want to control the mix, not maximise either. |
| `colourfulness` | Hasler & Süsstrunk metric | Cheap, well-validated, robust to glitch content. |
| `surprisal_mean` / `surprisal_burstiness` | See below | The deepest axis, and the one that carries musical structure. |
| `resolution_ratio` | Fraction of surprisal peaks followed by decay within N frames | Operationalises tension-and-release: pleasure tracks prediction-*error reduction*, not low error. |

### The surprisal model

The modality-neutral part of the harmony/dissonance parallel, and the axis that actually encodes musical structure:

```python
class SurprisalModel:
    """Quantise the per-frame feature vector into a discrete alphabet
    (k-means, k≈64), fit a variable-order Markov model over the resulting
    symbol stream, emit per-frame information content -log2 P(s_t | context)
    and predictive entropy H(s_t | context).

    Fit on a corpus of the operator's own prior work, not on the clip being
    scored — a model fitted in-sample reports every clip as unsurprising.
    """
    def fit(self, corpus: Iterable[ClipFeatures]) -> None: ...
    def information_content(self, clip: ClipFeatures) -> np.ndarray: ...
```

This is a direct visual transposition of information-dynamic models of musical expectation. High surprisal following a low-entropy context is the dissonance analogue; return to low surprisal is the cadence. Burstiness matters more than mean — surprisal that arrives in structured bursts aligned to the beat reads as intent, and uniformly high surprisal reads as noise.

### Vetoes

Hard constraints, not scores. These override any fitness value.

```python
class Veto(Protocol):
    def check(self, clip: ClipFeatures) -> VetoResult: ...
```

- **Photosensitive-seizure risk.** Flag sequences with more than three luminance transitions per second over more than a quarter of the frame area, plus the saturated-red-transition case. Implement as a **proxy** on relative luminance — a true Harding assessment needs absolute cd/m² and display characterisation, which you don't have. Treat it as a screening filter that catches obvious hazards, and say so in the docstring. Anything intended for a public room deserves a real check on the real rig. Given that the whole point is generating glitch content that will be projected at audiences, do not treat this as optional garnish.
- **Dead output** — sustained full black or full white.
- **Capture corruption** — tearing, frozen frames, PTS anomalies.

Auto-mitigation for flash violations: temporal low-pass the offending window rather than discarding the take.

### Learned aggregator

```python
class PreferenceModel:
    """Bradley-Terry over pairwise comparisons, fitted on axis deltas.
    ~200-500 pairs suffices for a linear model over ~20 axes.
    """
    def fit(self, pairs: list[tuple[ClipFeatures, ClipFeatures, int]]) -> None: ...
    def score(self, clip: ClipFeatures) -> float: ...
    def report(self) -> FitReport:
        """Held-out pairwise ranking accuracy, per-axis weights with CIs.
        If accuracy is near chance, say so loudly — an unfit preference
        model driving an overnight search wastes the whole night."""
```

The calibration tool presents A/B clip pairs sampled from the archive, preferring high-uncertainty pairs (active learning). This is what makes the library genuinely reusable: **bring your own taste**, fitted in an evening.

---

## 7. Archive search

### Why quality-diversity rather than optimisation

You do not want *the* best preset. You want a **palette**: many distinct looks, each good, indexed by how they behave, so that during arrangement you can ask for "something with high departure and low motion" and get a real answer. **MAP-Elites** produces exactly that structure, tolerates expensive noisy evaluation, and degrades gracefully if you stop it early.

### Two-stage genome

The full space (12 parameters × 30 operators × 3 modulation properties each) is too large to search directly, and stage separation exploits the determinism property:

**Stage A — static.** Operators disabled, transport stopped. Genome is `(p1..p6, p12) ∈ [0,1]⁷` and `(p7..p11) ∈ {0,1}⁵`, restricted to the parameters Morris flagged as live. Deterministic, fast, cacheable. This finds the *looks*.

**Stage B — animated.** Seed from Stage A elites. Genome adds per-parameter `(operator, time, space, slope)`. Evaluation needs several seconds of capture rather than a few frames, so it's roughly 10× costlier — hence seeding rather than searching from scratch. This finds the *motion*.

### Behaviour descriptors

Two to four dimensions, chosen per run. Sensible defaults:

- `departure` (how far from source)
- `channel_balance` (busy vs clean)
- `motion_energy` (Stage B only)

Fitness is the fitted `PreferenceModel` score, with vetoes as hard rejections.

### Budget

At roughly 2 s per Stage A evaluation, 10k evaluations is about six hours — one overnight run per program family. Stage B at ~8 s and 2k evaluations is about four hours. This is an overnight-job-shaped problem, which is fine, and it argues for making the runner resumable and crash-tolerant from the start.

### Drift canaries

Re-evaluate a fixed set of canary states every few hundred evaluations. If their features drift beyond the measured noise floor, something has changed — thermal, capture gain, a knocked knob, a firmware reboot. Flag the run, checkpoint, and either recalibrate or halt. Silent drift over an eight-hour run poisons the archive in a way that is very hard to detect afterwards.

### Storage

Parquet for feature vectors and genomes; a short video thumbnail per elite; the whole thing queryable:

```python
archive.query(departure=(0.4, 0.7), motion_energy=(0.0, 0.2), program="Isotherm")
archive.neighbours(state_digest, k=8)     # "more like this"
archive.path(from_digest, to_digest, steps=16)   # interpolation for transitions
```

`archive.path` is what makes performance transitions possible — a route through behaviour space between two elites, which the renderer can traverse over a musical phrase.

---

## 8. Generative rendering

Input: one video file, optionally one audio track. Output: a rendered video file. Not live, which buys multi-take selection.

### Audio analysis (librosa)

Beat times, downbeats, tempo, onset strength envelope, RMS, spectral flux and centroid, harmonic/percussive separation, and — most importantly — **structural segmentation** (Laplacian/recurrence-based) to recover section boundaries. Sections are what drive large-scale visual arrangement; beats only drive local modulation.

### Source analysis (cv2)

Shot boundaries, per-shot motion energy, luma and colour statistics. The arrangement should respond to the footage, not just the music — a program that looks good on a static wide shot may collapse on fast handheld.

### Arrangement

The generated score is a timeline of `(time, DeviceState-delta)` events plus modulation assignments. Four principles, each with a concrete mechanism on this instrument:

**Sections map to archive regions.** Each musical section gets a neighbourhood in behaviour space; within a section, moves stay local; at boundaries, jump. This gives visual form that tracks musical form.

**Tension via detuning against the beat grid.** The instrument exposes this directly. `Sync LFO`'s Time property is a beat-division *ratio* — simple ratios (1/1, 1/2, 4/1) lock; a `Free LFO` set near but not at a beat multiple produces a slow visible drift at the difference frequency. Ramp the detuning up across a build, snap to a locked Sync LFO at the drop. That is a continuous, measurable tension parameter under a single control, and it is the closest thing the instrument has to a consonance axis.

**Chaos via the Logistic Map operator.** Its parameter range traverses the period-doubling route to chaos: stable → period-2 → period-4 → chaotic. A second, orthogonal tension axis, and one whose transitions are sharp and legible. Characterise its bifurcation structure during Stage B and store the parameter values of the transitions — those are the musically useful points.

**Arrive on the beat, don't cut on it.** Visual beat entrainment from abrupt flashes is measurably weaker than from continuous motion with a clear impact point. Prefer `Trigger Envelope` operators shaped so their peak lands on the beat, scheduled early by the attack time, over instantaneous parameter jumps at the beat.

**Never let visuals lag.** Tolerance for audio-lagging-video is large; tolerance for visuals arriving late is roughly 30–50 ms. Since the instrument contributes essentially no latency, schedule every parameter event earlier by the measured player+capture constant from §3.

### Realisation

```python
def realize(score: Score, dev: Device, cap: Capture, out: Path) -> Take:
    """Drive MIDI clock at the track BPM from Python, start transport and
    source playback together, execute scheduled events against the clock,
    capture both taps to disk. Records actual vs intended event times."""
```

Log intended-vs-actual event timing for every take. Timing jitter above a few milliseconds means the scheduler is losing to Python's GIL or USB latency, and you want to know that from the log rather than by squinting at output.

### Multi-take selection

Because modulation is deterministic from timecode, a given score renders identically every time. So:

1. Generate `M` candidate scores (varying archive route, detuning schedule, operator assignment).
2. Render each. `M = 6` on a four-minute track is roughly half an hour of hardware time.
3. Score every take with `vjaesthetic`, per section.
4. Either pick the best whole take, or **splice per section**, cutting at downbeats where the visual state is closest across takes.

Then the veto pass, mitigation of any flagged windows, audio mux, encode. Capture to FFV1 or ProRes throughout; encode to delivery format once at the end.

---

## 9. Build order

Roughly in dependency order, with each step producing something usable on its own:

1. **`chain.py` + `capture.py` + latency calibration.** Nothing downstream is trustworthy until you can prove which captured frame corresponds to which command. Ship this first and validate it hard.
2. **`device.py` + `calibrate_param_addressing`.** Resolve the additive-MIDI question empirically. Measure hysteresis and settling times.
3. **`vjaesthetic.features`.** Pure functions, unit-testable against synthetic inputs with known answers, no hardware needed. Parallel-developable.
4. **Probes + `measure/`.** Now you can characterise. First real output: a per-program sensitivity report.
5. **Morris screening across the library.** An afternoon of hardware time. Produces the map of what actually does what.
6. **Preference calibration.** Needs an archive to sample pairs from, so bootstrap with random states from step 5.
7. **MAP-Elites.** Overnight runs, resumable.
8. **Renderer.** Everything above is prerequisite.

### Risks worth naming up front

- **Alignment is the real engineering problem.** Budget more time for step 1 than seems reasonable.
- **Firmware churn.** Programs and operators change between release candidates. Version-stamp everything; expect to re-run characterisation after updates.
- **Aesthetic scorers are weak priors.** The learned aggregator is what makes them useful. Report held-out accuracy honestly and don't run overnight searches against a model that hasn't beaten chance.
- **Naturalness priors fight glitch aesthetics by construction.** This is a feature if you treat the axes as signed descriptors and let the fit choose targets; it is a trap if anyone bolts a fixed "quality" scalar on top.
- **Thermal and mechanical drift** over long unattended runs. Canaries, checkpointing, and a willingness to throw away a poisoned run.
