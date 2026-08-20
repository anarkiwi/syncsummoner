# Measured rig behaviour

Facts established on hardware. Several are load-bearing: getting them wrong
produces a harness that silently measures the wrong thing. Wiring and bring-up
are in [rig.md](rig.md); the render workflow is in [workflow.md](workflow.md).

## Rig

| Part | Value |
| --- | --- |
| Videomancer serial | `E464B0605F113625` |
| Firmware | `1.0.0-rc.46` |
| Programs installed | 56 |
| MIDI node | ALSA rawmidi, discovered per enumeration (`midiC4D0`, `C5D0` and `C6D0` all observed) |
| Serial node | `/dev/ttyACM0`, group `dialout`; needs `pyvmancer[serial]` |
| Capture card | AVerMedia Live Gamer Ultra 2.1, `/dev/video0`, uvcvideo |
| Session format | 1920x1080 @ 30 over HDMI, genlocked to the Pi, no override |
| Playout | Raspberry Pi 4B `pi@videopi`; this host has no usable HDMI output |

Nothing about the device's own nodes is fixed: the rawmidi index moves across
replugs, which is why discovery reads them rather than hardcoding them.

## The chroma pair arrives exchanged

`v4l2-ctl --list-formats-ext` reports one format, `YUYV 4:2:2`, but Cb and Cr
arrive exchanged, swapping red and blue. The device's own `Colorbars` decodes as
white, cyan, yellow, green, magenta, blue, red, black under the advertised order,
and as SMPTE order once the pair is put back. MJPEG carries the same exchange.

- `device/capture.py` decodes with `COLOR_YUV2BGR_YVYU`, never `..._YUYV`.
- The archive recorder filters `-vf swapuv` before encoding, so what is stored is
  already right.
- `-c copy` cannot filter, so an MJPEG take keeps the exchange until re-encoded.
- Never let the card convert (`CAP_PROP_CONVERT_RGB=1`): it uses the advertised
  order and clips, so full red returns `B=254.5` and the true `R=208.6` is
  unrecoverable.

## Parameters are additive

`Parameter = Manual + Modulation + MIDI`. CC is an offset onto the physical knob,
not an absolute set: with a knob at centre, `set_param(1, 0.75)` may land at clip.
Park the panel before measuring; `Session.park()` exists for it.

## Parameters are heterogeneous

P1-P6 continuous, P7-P11 boolean, P12 fader, 10-bit. Booleans resolve `on` at a
combined value >= 512 of 1024. `program info` reports per-parameter names and
native ranges over serial and names unused parameters `-`, so no `.vmprog` TOML
parsing is needed. A parameter with native range `0..7` has 8 meaningful
positions, not 1024; sampling it at 32 steps wastes rig time. See
`ParamSpec.sample_values`.

## Format is a session constant

The device genlocks to its source and cannot convert. The card constrains what
the far end can be: genlocked to NTSC, `720p60` was the only usable progressive
format, with `480p`, `1080p30` and `1080i60` all returning no signal. Genlocked
to the Pi at 1920x1080@30 the same chain is fine. Pick one format for the whole
pipeline and never vary it.

The card returns frames at its own rate, not the session's: a 10 s pass at
`1080p30` comes back as ~600 frames, about 60 fps. Nothing downstream counts
frames — alignment is by the timecode strip and cuts are by time.

## The capture card synthesizes a "No Signal" splash

With no valid input the card emits a high-contrast slide reading *No Signal* with
a QR code, at whatever resolution was requested. It is not black and it has high
variance, so any liveness test based on `std > 0` scores it as content.
`Capture.is_no_signal` detects it structurally: the splash is achromatic (pixels
with HSV saturation > 60 at ~0.000, versus ~0.69 for real Colorbars) while being
far from uniformly black.

## Status flags are advisory; only a capture proves frames

The firmware has reported `hdmi.connected: false` while passing video perfectly,
and a locked input while passing nothing. The top-level `locked` flag tracks
genlock, so it reads false whenever timing is overridden even though the input is
fine; the selected input's own sub-status is authoritative, which is what
`VideoStatus.source_locked` reads. Anything that records checks the frames
themselves: `settle()` judges liveness from a recording, exactly as a take is
judged, and `inspect_take` judges the take.

## Capture lock is slow

| Event | Cost |
| --- | --- |
| Output timing change, card resync | ~3.2 s |
| Stream reopen at unchanged timing | ~0.5 s |
| Program load to picture back | 12.2-19.1 s |

All are orders of magnitude above any per-sample dwell, so the capture session is
opened once and held for a whole sweep, and the settle budget is 40 s. This is
what makes the gray-code strip load-bearing rather than a convenience: frames
must self-identify, because resynchronising by reopening is not affordable.

Program change is the expensive operation in any search loop. Batch every
evaluation for a program together and treat program as the outermost loop
variable, never as a mutable gene in an inner loop.

## Modulation is deterministic from timecode

Oscillators phase-reset at `00:00:00:00` and `STOP` resets timecode to zero, so
with Time properties unchanged the same modulation is generated every time
playback starts from zero. Renders are reproducible, which is what makes
multi-take selection and offline scoring sound. Prefer internal transport or MIDI
Clock, which reset on stop; use MTC only when pause/resume without phase reset is
needed.

Modulation updates at frame or field rate, except `Audio In` which runs at
scanline rate. Control-signal Nyquist is half the frame rate.

## No frame store

A few lines of memory, and no internal frame feedback. Feedback effects need an
external loop through Dual In, so the `feedback_gain` canonical axis is
unmeasurable until that loop exists.

## Playout framebuffer

| Property | Value |
| --- | --- |
| Framebuffer | `/dev/fb0`, 1920x1080, 16bpp, stride 3840 |
| Word layout | BGR565: `rgb565le` bytes come back with red and blue exchanged |
| Frame size | 4147200 bytes |
| Compositor | none running, so DRM is free |

A frame is displayed by writing raw 16-bit little-endian words to `/dev/fb0`,
blue in the high bits; `ffmpeg -pix_fmt bgr565le` produces them directly.

## The analog input decodes luma and sync only

Composite fed straight in yields zero chroma under both NTSC and PAL while luma
decodes correctly and standard-aware — the 7.5 IRE pedestal is honoured under
NTSC and absent under PAL, and PAL is the better luma path (gain 1.028 against
0.958, black at 0 rather than 18/255, 20% more lines). Colour appears only with
an external RGB encoder, which measured badly out of calibration: luma gain 1.21,
ramp clipped at x=491 of 720, blue clipped to white, black lifted to purple.
Colour work over the analog path needs that encoder calibrated first. The session
runs HDMI, which carries chroma.

## Capture is owned by ffmpeg

`VideoCapture.read` does not block: a 30-frame burst spanned 0.20-0.50 s, 20-92%
of reads under 10 ms apart, handing back the previous buffer rather than waiting,
and held as little as one distinct frame where the picture was still. Pacing and
de-duplicating hid the shortfall rather than fixing it, and made a working rig
look faulted.

| Path | Rate at 1920x1080 |
| --- | --- |
| Python loop, frames into an FFV1 encoder in-process | 7.7 fps |
| Python loop, frames discarded | 25 fps |
| `ffmpeg -f v4l2 -input_format yuyv422 -c:v ffv1` | 29.8 fps |
| `ffmpeg -f v4l2 -input_format mjpeg -c copy` | 59.8 fps |

The host touches no frame while the rig runs. Under `-copyts` the stored
presentation times are the card's own `CLOCK_MONOTONIC` capture times, the clock
`time.monotonic` reads, so frames attribute to the setpoint being held when each
arrived with no alignment step. `-t` must not be used with `-copyts`: it is read
against the card's clock and ends the recording immediately.

## The state-index strip survives the chain

An 8-bit Gray-code strip played from the Pi decoded back bit-exact through
composite encode, analog transport, `Passthru`, HDMI output and USB capture, and
a 16-bit strip decodes the same way through the HDMI session. The decoder
thresholds against the strip's own midpoint rather than a fixed constant, because
of the analog path's 0.958 gain and 18/255 pedestal.

A destructive program mangles individual strips: inside a `Derez` take, frames
decode correctly either side of ones that decode to nonsense. Alignment therefore
takes the lag a take's frames agree on rather than trusting any single frame.

## `settings export` and `settings get` hang the serial shell

Both verbs return no response and leave the command processor unresponsive to
every subsequent command; MIDI and video output continue, `USBDEVFS_RESET` does
not recover it, and a power cycle is required. The harness must not issue them.
