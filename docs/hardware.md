# Measured rig behaviour

Facts established on hardware, not inferred. Several are load-bearing: getting
them wrong produces a harness that silently measures the wrong thing.

## Rig

| Part | Value |
| --- | --- |
| Videomancer serial | `E464B0605F113625` |
| Firmware | `1.0.0-rc.40` |
| Programs installed | 52 |
| MIDI node | `/dev/snd/midiC4D0` |
| Serial node | `/dev/ttyACM0` (needs `dialout` or an ACL) |
| Capture card | AVerMedia Live Gamer Ultra 2.1, `/dev/video0`, uvcvideo |
| Session format | **1920x1080 @ 30**, `YUYV` advertised, **chroma pair exchanged** |

The serial ACL is dropped whenever the device re-enumerates; `/dev/ttyACM0`
returns as `c---------`. Re-run `setfacl` after any replug, or install a udev
rule.

## The chroma pair arrives exchanged

`v4l2-ctl --list-formats-ext` reports exactly one format, `YUYV 4:2:2`, but Cb and
Cr arrive exchanged, which swaps red and blue. The proof needs no source: the
device's own `Colorbars` decodes as white, **cyan, yellow**, green, magenta,
**blue, red**, black under the advertised order, and as textbook SMPTE order once
the pair is put back. The card's MJPEG output carries the same exchange, so
choosing a format does not avoid it.

Consequences:

* `syncsummoner/device/capture.py` decodes native buffers with
  `COLOR_YUV2BGR_YVYU`, never `..._YUYV`.
* The archive recorder captures `yuyv422` and filters `-vf swapuv` before
  encoding, so what is stored is already right and nothing downstream corrects it.
* A recording stored with `-c copy` cannot be filtered, so an MJPEG take keeps the
  exchange and has to be swapped wherever it is re-encoded.
* Never let the card convert (`CAP_PROP_CONVERT_RGB=1`). It uses the advertised
  order **and clips**: full red becomes a clipped `B=254.5`, so the true `R=208.6`
  is gone and no channel swap or matrix correction recovers it.

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

Measured while genlocked to an NTSC source, where 720p60 was the only usable
progressive format. The session now genlocks to the Pi over HDMI at 1920x1080@30
with no override. Format is a session constant, never a parameter.

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

## Playout is a Raspberry Pi

This host has no usable HDMI output. Stimulus playout is a Raspberry Pi 4B
(`pi@videopi`) writing its framebuffer, which the Videomancer genlocks to.
`video input` accepts only `analog|hdmi`. The composite path below was the
original chain and is kept for its transfer measurements; the session now runs
1920x1080 over HDMI, which carries chroma where composite did not.

| Pi property | Value |
| --- | --- |
| Framebuffer | `/dev/fb0`, 1920x1080, 16bpp, stride 3840 |
| Word layout | **BGR565**: `rgb565le` bytes come back with red and blue exchanged |
| Frame size | 4147200 bytes |
| Composite framebuffer | 720x576, stride 1440, 829440 bytes, `vc4.tv_norm=PAL` |
| Composite modes | `720x576i`, `720x480i`, `720x288`, `720x240` |
| Compositor | none running, so DRM is free |

A frame is displayed by writing raw 16-bit little-endian words to `/dev/fb0`,
blue in the high bits. `ffmpeg -pix_fmt bgr565le` produces them directly.

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
mode. Check the cable before the firmware: on 2026-08-03 a failing output cable
produced the same picture, and `hdmi.connected` went false to true when it was
replaced.

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

Every one of these flags is advisory in isolation. The firmware has reported
`hdmi.connected: false` while passing video perfectly, and a locked input while
passing nothing at all; it has also tracked a real cable fault exactly. Only a
capture proves frames are arriving, so anything that records must check the
frames themselves.

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

## Retracted: "programs freeze while passthrough carries the source"

Recorded here on 2026-08-03 as a third fault and **withdrawn the same day**. Two
faults of the harness produced it together: capture was a per-frame Python loop
that could not hold the session rate and handed back repeated buffers, and the
HDMI cable from the Videomancer to the capture card was failing. Replacing the
cable restored the picture with no power cycle, and `hdmi.connected` went false to
true across the swap. Under ffmpeg-owned capture a 60s take of `Teletext` on the
same device held **894 distinct frames of 900**.

The lesson is kept because it was expensive: liveness must be judged the same way
a take is judged, from a recording, and a rig that a person can watch working is
working.

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

How much of this survives the failing HDMI cable found on 2026-08-03 is not
settled: some of the black attributed to the device was that cable, and the
49-program archive it is measured from was captured through both faults. The
recurrence and the passthrough canary are kept because the canary is cheap and
the diagnosis it gives is the one an operator needs; the numbers are not to be
relied on until a run on the repaired rig either reproduces them or does not.

The load blackout itself is real and slow: timestamps put the picture back
**12.2-19.1s after the load**, which is why the settle budget is 40s. It is now
covered by construction rather than by a heuristic — the sweep only opens a
setpoint's window once the picture is back and the parameters have landed, so a
frame captured during the blackout is outside every window and is never
attributed to a vector.

## A per-frame host loop cannot hold the session rate

`VideoCapture.read` does not block: a 30-frame burst spanned **0.20-0.50s** with
20-92% of its reads under 10ms apart, handing back the previous buffer rather
than waiting, and held **as little as one distinct frame** where the picture was
still. Pacing and de-duplicating the reads hid the shortfall rather than fixing
it, and made a working rig look faulted.

Measured throughput at 1920x1080, same card and host:

| Path | Rate |
| --- | --- |
| Python loop, frames into an FFV1 encoder in-process | 7.7 fps |
| Python loop, frames discarded | 25 fps |
| `ffmpeg -f v4l2 -input_format yuyv422 -c:v ffv1` | 29.8 fps |
| `ffmpeg -f v4l2 -input_format mjpeg -c copy` | 59.8 fps |

Capture is therefore owned by ffmpeg and the host touches no frame while the rig
runs. Under `-copyts` the stored presentation times are the card's own
`CLOCK_MONOTONIC` capture times, which is the clock `time.monotonic` reads, so
frames can be attributed to the setpoint that was being held when each arrived
with no alignment step. `-t` must not be used with `-copyts`: it is read against
the card's clock and ends the recording immediately.

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
