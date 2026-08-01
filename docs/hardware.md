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
| Session format | **720x576i PAL @ 50**, YUYV 4:2:2 |

The serial ACL is dropped whenever the device re-enumerates; `/dev/ttyACM0`
returns as `c---------`. Re-run `setfacl` after any replug, or install a udev
rule.

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
