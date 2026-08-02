# Measured rig behaviour

Facts established on hardware, not inferred. Several are load-bearing: getting
them wrong produces a harness that silently measures the wrong thing.

## Rig

| Part | Value |
| --- | --- |
| Videomancer serial | `E464B0605F113625` |
| Firmware | `1.0.0-rc.37` |
| Programs installed | 21 |
| MIDI node | `/dev/snd/midiC4D0` |
| Serial node | `/dev/ttyACM0` (needs `dialout` or an ACL) |
| Capture card | AVerMedia Live Gamer Ultra 2.1, `/dev/video0`, uvcvideo |
| Session format | **720x576i PAL @ 50**, `YUYV` advertised, **YVYU delivered** |

The serial ACL is dropped whenever the device re-enumerates; `/dev/ttyACM0`
returns as `c---------`. Re-run `setfacl` after any replug, or install a udev
rule.

## The capture card lies about its byte order

`v4l2-ctl --list-formats-ext` reports exactly one format, `YUYV 4:2:2`, but the
bytes are `YVYU`: Cb and Cr arrive exchanged, which swaps red and blue. A full
red source reads back `Y=40.9 Cb=240.8 Cr=109.0`, which BT.601 calls blue;
decoding the same buffer as YVYU recovers `R=208.6`, the limited-range red the
playout chain actually sends. The card's own `Colorbars` program shows the same
exchange with no source in the path, so the fault is the capture card's.

Consequences, all handled in `syncsummoner/device/capture.py`:

* Decode native buffers with `COLOR_YUV2BGR_YVYU`, never `..._YUYV`.
* Never let the card convert (`CAP_PROP_CONVERT_RGB=1`). It uses the advertised
  order **and clips**: full red becomes a clipped `B=254.5`, so the true `R=208.6`
  is gone and no channel swap or matrix correction recovers it.
* Raw archives must tag the pipe `yvyu422`. Under `yuyv422` the bytes still
  round-trip bit-exactly while the stored `yuv422p` holds the chroma planes
  exchanged, which bakes the swap in permanently.

## Parameters are additive

`Parameter = Manual + Modulation + MIDI`. MIDI CC is an **offset** onto the
physical knob position, not an absolute set. If a knob sits at centre,
`set_param(1, 0.75)` may land at clip.

Park every physical control at a known reference before any measurement session
and verify addressing empirically. `Session.park()` exists for this. Do not
assume `set_param` is absolute.

## Parameters are heterogeneous

P1-P6 continuous, P7-P11 boolean, P12 fader, at 10-bit resolution. Booleans
resolve as `on` at combined value >= 512 of 1024. Observed on Colorbars:

```
program state -> m:[512,512,512,512,512,512, 1,0,0,0,0, 512]
```

`program info` reports per-parameter names and native ranges directly over
serial, and names unused parameters `-`. No `.vmprog` TOML parsing is needed.

```json
{"name":"Colorbars","parameters":[
  {"name":"-","min":0,"max":100}, ...
  {"name":"Level","min":0,"max":1},{"name":"Pattern","min":0,"max":1},
  {"name":"Blue Only","min":0,"max":1},{"name":"Mono","min":0,"max":1},
  {"name":"Bypass","min":0,"max":1},{"name":"-","min":0,"max":100}]}
```

A parameter with native range `0..7` has 8 meaningful positions, not 1024.
Sampling it at 32 steps wastes hardware time; see `ParamSpec.sample_values`.

## Output timing is constrained by the capture card

Measured against Colorbars, holding the stream open past lock:

| `video timing` | Result |
| --- | --- |
| `NTSC` | signal |
| `480p` | **no signal** |
| `720p60` | signal |
| `1080p30` | **no signal** |
| `1080i60` | **no signal** |

Measured while genlocked to an NTSC source. 720p60 was the only usable
progressive format. Since the source is now PAL and the device auto-genlocks,
no override is used and the session runs at 720x576i throughout. Format is a
session constant, never a parameter.

## The capture card synthesizes a "No Signal" splash

With no valid input the card emits a high-contrast slide reading *No Signal*
with a QR code, at whatever resolution was requested. It is not black and it has
high variance, so any liveness test based on `std > 0` scores it as **content**.
This is a silent data-corruption hazard.

`Capture.is_no_signal` detects it structurally: the splash is achromatic
(fraction of pixels with HSV saturation > 60 is ~0.000, versus ~0.69 for real
Colorbars) while being far from uniformly black.

## Capture lock is slow

| Event | Cost |
| --- | --- |
| Output timing change, card resync | ~3.2 s |
| Stream reopen at unchanged timing | ~0.5 s |

Both are orders of magnitude above any per-sample dwell. The capture session
must be opened once and held for the whole sweep. This is what makes the
gray-code state-index strip load-bearing rather than a convenience: frames must
self-identify because you cannot resynchronize by reopening.

## Modulation is deterministic from timecode

Oscillators phase-reset to zero at `00:00:00:00`, and `STOP` resets timecode to
zero. With Time properties unchanged, the same modulation pattern is generated
every time playback starts from zero. Renders are therefore reproducible, which
is what makes multi-take selection and offline scoring sound.

Prefer internal transport or MIDI Clock, which reset on stop. Use MTC only when
pause/resume without phase reset is needed.

Modulation updates at frame/field rate, except `Audio In` which runs at scanline
rate. Control-signal Nyquist is half the frame rate; do not design envelopes
finer than that.

## Program loading blacks out

Outputs drop for several seconds on program change. Program change is the
expensive operation in any search loop: batch all evaluations for a program
together and treat program as the outermost loop variable, never as a mutable
gene in an inner loop.

## No frame store

A few lines of memory only, and no internal frame feedback. Feedback effects
require an external loop via Dual In (HDMI in, analog out, external chain,
analog in). The `feedback_gain` canonical axis is unmeasurable until that loop
exists.

## Playout is a Raspberry Pi on composite

This host has no usable HDMI output, and the Videomancer's HDMI input reads
`connected: false`. Stimulus playout is instead a Raspberry Pi 4B (`pi@videopi`)
driving composite video into the Videomancer's analog input, which genlocks to
it. `video input analog` selects it; `video input` accepts only `analog|hdmi`.

| Pi property | Value |
| --- | --- |
| Framebuffer | `/dev/fb0`, 720x576, 16bpp RGB565, stride 1440, no padding |
| Driver | `vc4drmfb`, `enable_tvout=1`, `dtoverlay=vc4-kms-v3d,composite` |
| TV norm | `vc4.tv_norm=PAL` on the kernel cmdline |
| Composite modes | `720x576i`, `720x480i`, `720x288`, `720x240` |
| Compositor | none running, so DRM is free |
| Frame size | 829440 bytes |

A frame is displayed by writing raw RGB565 little-endian bytes to `/dev/fb0`.
The Pi's HDMI outputs are unused.

## Measured chain transfer

Playing a bars-plus-ramp-plus-strip pattern from the Pi through `Passthru`,
captured at 720x480. `Passthru` reports all twelve parameters as `Null`, so it
is genuinely transparent and cannot account for any of this.

**Pi composite direct into the analog input** — luma is good, chroma is absent.
Every colour bar collapses to its correct Rec.601 luma as a neutral grey; the
input path recovers luma and sync only.

| Property | NTSC | PAL |
| --- | --- | --- |
| Luma gain | 0.958 | 1.028 |
| Ramp linearity | linear to x=712 of 720 | linear to x=707 of 720 |
| Black level | 18/255, the 7.5 IRE pedestal | 0, exact |
| Mean chroma radius | 0.53/255, max 2.24 | zero |
| Raster | 720x480i at 59.94 | 720x576i at 50 |

PAL is the better luma path and is the session default: it has no 7.5 IRE setup
so black lands on zero, gain sits nearer unity, and the raster carries 20% more
lines. The device auto-genlocks with no timing override needed.

Set on the Pi by appending `vc4.tv_norm=PAL` to `/boot/firmware/cmdline.txt`
and rebooting. `fb0` then reports 720x576, stride 1440, 829440 bytes per frame.

**Pi composite through the RGB encoder** — chroma is present but wrong:

| Bar | Expected | Measured |
| --- | --- | --- |
| white | 255,255,255 | 255,255,255 |
| yellow | 255,255,0 | 201,255,183 |
| cyan | 0,255,255 | 250,166,255 |
| red | 255,0,0 | 78,153,69 |
| blue | 0,0,255 | 255,255,255 |
| black | 0,0,0 | 81,38,88 |

Luma gain was 1.21 and the ramp clipped to white at x=491 of 720. Blue clipped
to white, black lifted to purple, and hues rotated. Colour requires the encoder,
but the encoder needs calibrating before any colorimetry measurement means
anything. Suspect excess input level and a chroma standard or burst-phase
mismatch.

## The analog input decodes luma and sync only

Composite fed straight in yields zero chroma under both NTSC and PAL, while luma
decodes correctly and standard-aware (7.5 IRE pedestal honoured under NTSC,
absent under PAL). A subcarrier standard mismatch would give wrong colour, not
no colour, and colour does appear when an external RGB encoder is inserted.

The input therefore recovers luma and sync but does not demodulate the colour
subcarrier, which needs a comb filter and quadrature demodulation. The external
encoder is a required part of the chain for colour work, not an accessory.
Whether a device setting also selects the analog input format is unresolved.

## The HDMI input drops into a bad state, and only a power cycle clears it

Observed repeatedly on `1.0.0-rc.37`: the input stops passing video while the
device still reports `hdmi.locked: true`, usually with `hdmi.connected: false`,
and sometimes with the top-level `locked` false while the sub-status disagrees.
The front-panel HDMI input light stays lit, and the source keeps driving a valid
mode, so it is a firmware state rather than a cable fault.

| Recovery attempt | Result |
| --- | --- |
| `program load <name>` | no effect |
| `video input analog` then `hdmi` | recovered once, not reliably |
| Power cycle | works every time |

There is no reboot verb short of the bootloader: `reboot` is rejected bare and
with any argument other than `bootloader`, which enters the firmware-flashing
state rather than restarting.

The strongest reset the serial interface does offer is a **timing bounce**, and
it is what `Transport.resync()` performs. Measured against a live signal:

| Verb | Disturbs the pipeline | Recovers |
| --- | --- | --- |
| `modulation reset` | no | - |
| `video input <same>` | no | - |
| `video timing <native>` | no | - |
| `fpga preferred-variant <v>` | rejected; query-only | - |
| `program load <different>` | yes | yes |
| `video timing <other>` | yes | **no**, output dies |
| `video timing <other>` then `<native>` | yes | yes |

Forcing a timing the genlocked source cannot satisfy kills the output; restoring
the native timing brings it back, re-initialising the output raster on the way.
A timing change also drops the input selection, so `resync()` reasserts it.

Two consequences of the bounce. It leaves `overridden: true`, so the device no
longer follows a source format change; that is harmless while format is a
session constant. And the top-level `locked` flag tracks **genlock**, so it
reads false whenever timing is overridden even though the input is fine. The
selected input's own sub-status is the authoritative field, which is what
`VideoStatus.source_locked` reads.

Every one of these flags is advisory. The firmware reports `hdmi.connected:
false` while passing video perfectly, and reports a locked input while passing
nothing at all. Only a capture proves frames are arriving, so anything that
records must check the frames themselves.

`VideoStatus.source_locked` exists because of this: it cross-checks the selected
input's own sub-status against the top-level flag, so an unattended sweep fails
loudly instead of recording hours of black frames as measurements.

## Program change costs more than the load blackout

Loading a program blacks the output out, and the capture card then falls back to
its placard and needs seconds to re-lock. A fixed dwell after `program load`
therefore samples the placard, not the program: a 48-program survey silently
skipped nine consecutive entries that way, including two that measure fine once
the wait polls for real content instead of guessing a duration.

Poll until frames are both non-placard and moving. The drop this rig is prone to
presents as **frozen output, not the placard**, so a placard test alone passes
dead frames straight through into the data; only the motion check catches both.

`Transport.resync()` clears it: measured against a live failure, output motion
went from 0.00000 to 0.02802 with no power cycle.

## The device stops holding any program after ~15 loads

Measured 2026-08-02 over a whole-library archive run. After roughly fifteen
program loads the device still answers normally, firmware reads succeed and the
HDMI input stays locked, but `program info` returns `[3] no program loaded` and
no program will load again.

There is no software recovery. Reloading, `Transport.resync()` (which returns
False) and `set_video_timing()` all fail; only a power cycle clears it. A long
run must therefore checkpoint per program, detect the wedge and stop rather than
burning through the rest of the library, and resume by itself once power is
cycled. `probe.harvest` does exactly that.

Because the device wedges under serial load, an archive run keys its results on
name plus firmware (`KeyKind.NAME_FIRMWARE`) rather than on the program binary:
`program_key` hashes each binary over the serial shell, which its own docstring
notes runs at wire speed and can time out.

## Programs freeze while passthrough still carries the source

Measured 2026-08-03, a third fault and the only one the canary calls healthy. The
device answers, loads any program, and `Passthru` passes a moving source frame for
frame; every other program emits a still picture. With the source playing at 30fps
and a program loaded, the capture saw:

| program | distinct pictures in 155 grabs | timecodes decoded |
| --- | --- | --- |
| `Passthru` | 148 | 148 |
| `Jammer`, `Sabattier`, `Scramble`, `Derez`, `Stochasm` | 2 | 0 |

`Capture.wait_for_content` returns False throughout, and waiting 25s past the load
changes nothing. A power cycle clears it: the same programs then returned 152 and
153 distinct pictures with their stamps intact.

`carries_stimulus` loads `Passthru` to decide whether the rig or the program is
dark, so this fault reads as a healthy rig and a dark program, once per program,
for the whole library. A canary that only proves the passthrough cannot see it.

## The device emits pure black while reporting a live source

Measured 2026-08-02, a second fault distinct from the wedge above. The device
answers normally, loads programs on request and reports `source_locked=True`,
but emits only black. An archive run stored **18 consecutive programs of black
frames**, every one at mean luma 11.2, and labelled them DARK without acting.

A dark frame on its own does not prove a fault: one program in the same run
(`Corollas`) was genuinely black while the rig was healthy. Only a **passthrough
canary** separates the two. `probe.harvest` therefore deletes a dark program's
archive, loads `Passthru` and re-measures: content back means only that program
was dark and the run continues; black back means the device is faulted and the
run stops. Both faults need a power cycle, and the report keeps them apart
(`wedged` against `blacked`) because the diagnosis differs.

The fault recurs, and how soon is not predictable. Two runs on `1.0.0-rc.40`, each
started from a freshly power-cycled device, went black after **14 programs in 17.8
minutes** and after **3 programs in 6.3 minutes**. Neither load count nor elapsed
time predicts the next one, so an archive run cannot be planned around a cycle
count: it commits per program and resumes from what is stored, and the operator is
told which fault stopped it. A 49-program library costs an unknown number of power
cycles, several at least.

The same fault frame also opens an otherwise healthy program. Scanned across a
complete 49-program archive (2026-08-02, firmware `1.0.0-rc.40`), that frame is
byte-identical everywhere it appears and held **33 of 1506 setpoints across 12
programs**, always as a leading run of up to 5 setpoints. Timestamps put the
picture back **12.2-19.1s after the load**, past the 10s `content_timeout_s`
whose result the caller discarded, so the blanked output was archived against
real sweep vectors. The wait is now honoured and the budget covers the measured
relock; a per-program mean cannot catch this, because the program is not dark.

## Reads outrun the card

A 30-frame burst spanned **0.20-0.50s** against a 30fps capture, with 20-92% of
its reads under 10ms apart: `VideoCapture.read` hands back the previous buffer
rather than blocking. Bursts held 19-30 distinct frames of 30 where the picture
moved and **as little as one** where it did not, so an unpaced burst samples the
few milliseconds the reads take instead of the setpoint. `native_burst` drops a
frame equal to the one stored before it and paces at the capture period.

`DARK_LUMA` is 20.0. The codeframe stimulus reaches the card near mid grey
(Y~128), healthy programs measured 62-180 and the two faults measured 0.0 and
11.2, so the threshold sits in a wide empirical gap rather than on a tuned edge.

## `settings export` and `settings get` hang the serial shell

On `1.0.0-rc.37` both verbs return no response and leave the command processor
unresponsive to every subsequent command. MIDI and video output continue
working; a USB-level `USBDEVFS_RESET` does not recover it, and a power cycle is
required. The harness must not issue these verbs.

## The state-index strip survives the analog chain

An 8-bit Gray-code strip played from the Pi decoded back **bit-exact** after
composite encode, analog transport, `Passthru`, HDMI output and USB capture.
Frame identification per design 3.2 is therefore sound on this rig. The decoder
must threshold against the strip's own midpoint rather than a fixed constant,
because of the 0.958 gain and the 18/255 pedestal.
