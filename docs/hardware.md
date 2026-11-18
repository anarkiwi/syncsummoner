# Measured rig behaviour

Facts established on hardware. Several are load-bearing: getting them wrong
produces a harness that silently measures the wrong thing. Wiring and bring-up
are in [rig.md](rig.md); the render workflow is in [workflow.md](workflow.md).

## Rig

| Part | Value |
| --- | --- |
| Videomancer serial | `E464B0605F113625` |
| Firmware | `1.0.1` |
| Programs installed | 56 |
| MIDI node | ALSA rawmidi, discovered per enumeration (`midiC4D0`, `C5D0` and `C6D0` all observed) |
| Serial node | `/dev/ttyACM0`, group `dialout`; needs `pyvmancer[serial]` |
| Capture card | AVerMedia Live Gamer Ultra 2.1, `/dev/video0`, uvcvideo; needs USB 3 |
| Session format | 1920x1080 @ 30 over HDMI, genlocked to the Pi, no override |
| Playout | Raspberry Pi 4B `pi@videopi`; this host has no usable HDMI output |

Nothing about the device's own nodes is fixed: the rawmidi index moves across
replugs, which is why discovery reads them rather than hardcoding them.

## The capture card needs USB 3

On a USB 2 link (480 Mbit/s) the card offers only SD uncompressed modes: no
MJPEG and no 1080p, so the recorder cannot open the session format. On USB 3
(5000 Mbit/s) it offers MJPEG up to 3840x2160. The card also carries a USB 2 hub
of its own for its RGB lighting controller, which is normal; only the video
function's link speed matters. `syncsummoner device doctor` checks both.

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
native ranges over serial and names unused parameters `-`; each parameter's kind
comes from the program's `.vmprog` declaration. A parameter with native range `0..7` has 8 meaningful
positions, not 1024; sampling it at 32 steps wastes rig time. See
`ParamSpec.sample_values`.

## MIDI write timing

Measured with pyvmancer 0.6.0 (`vmancer latency`, `syncsummoner device doctor
--latency`) on firmware 1.0.1 at 1080p30:

| Measure | Value |
| --- | --- |
| CC write to readback | about 33 ms, one frame period |
| Backlog, one parameter | none at 80 sets/s (160 messages/s) |
| Backlog, six parameters | none at 40 sets/s each, 240 sets/s (about 480 messages/s) |

A set is one parameter write: two CC messages at high resolution (14-bit), one
at low. Only `Session`, which runs without `midi monitor` (probe sweeps, parking),
still paces open loop at `CC_BUDGET_HZ`, inside that clean range: an unpaced park
burst drops the LSB of 14-bit pairs.

A locked take has no rate or capacity. How fast the device drains MIDI depends on
what the firmware and the loaded program are doing, so a fitted rate is a guess
that breaks with the program. Unbounded bursts do lose messages; on firmware
1.0.2-rc.4 at 1080p30, with `midi monitor on`:

| Burst | Received |
| --- | --- |
| 62 writes, 124 messages, within 2 ms, all for one frame | the first 62 messages; the other 62 never, so the slot held the ramp's midpoint |
| 12 writes, 24 messages, within 1.1 ms | all, but the last 20 applied a tick after the first 4 |

So the take is flow controlled by the device's own echoes (`wire.Echoes`): one
write in flight, the next sent once the last one's messages come back from
`midi monitor`, and none before its frame's send instant. A write is overdue once
the longest echo the take has acknowledged has passed and a reply asked for after
that has come back without it; it is written once more, and the copy is recorded
(`resent` in the wire record). Lateness is then measured per write (docs/evidence.md).

Over the 11 frame-locked takes with receipts (rc4/takes, full/takes), the echo of
a write came 4.2 to 5.4 ms (median per take) after the write returned, at worst
9.7 to 37 ms when a poll reply stalled. The receipts cannot say when within that
bracket the device took a message; polling back to back while a write waits
narrows it to one poll round trip.

The serial link stays far from the backlog that reboots the device (about 11 KB
of commands queued while the shell is busy, captures/HANDOFF.md): the poller has
one command in flight, `modulation status` at 18 bytes, and with one write in
flight a reply carries at most two monitor lines of at most 36 bytes each beside
its 279-byte status, about 350 bytes, where the 124-message burst put 62 lines
(about 2.2 KB) on one reply.

Every write keeps its MSB/LSB pair. The serial guide documents the CC map (MSB,
LSB and auto-pair per slot) but not how the firmware combines CC n with n+32,
nor whether an MSB or LSB alone updates a slot; MIDI 1.0 resets the LSB on an
MSB, so an MSB alone would be safe only for a new LSB of 0. Untested on the
device, so no message is dropped.

## CCs latch on the device's frame tick

Measured on firmware 1.0.1 at 1080p30 with the locked player, a still strip-coded
lead-in, and a bypass or crossfader toggled at random phases on eight programs:

| Measure | Value |
| --- | --- |
| Picture change | always on a frame start, never part way down a frame (0 mixed captures in 120 toggles) |
| Readback (`modulation status` combined value) | changes once per output frame; its serial timing is too coarse to time the latch, so landings rest on the picture |
| CC to readback value | exact for all 1024 values, high-resolution CC with manual parked at 0 |
| Serial `modulation status` round trip | about 1.4 to 2.2 ms, 279-byte reply |
| Send to the start of the frame a write lands on | 1.0 to 1.5 frames, from 16 calibration writes; the frame start is the take's frame line, known to the width of the lines its captures admit ([the card's clock](#the-cards-clock)) |

Render therefore sends each write half a frame before the instant it must precede
to land on its target frame, and proves where it landed ([evidence.md](evidence.md)). The early bias
of design 4.4 picks the target: the first frame starting at or after the write's
time less 15 ms.

## P12 is not a crossfader

pyvmancer calls P12 a crossfader that gates every program's output. That is false. On the
still lead-in, against Passthru's capture of the same picture (noise 0.12 mean
absolute difference in 8-bit codes):

| Program | P12 role | P12 = 0 | Bypass on | Strip legible held dry |
| --- | --- | --- | --- | --- |
| boneoh/YUV Bit Rotator | Global Blend, wet | dry (0.24) | dry | yes |
| teletect/Teletect RamBend | Wet | dry (0.23) | no bypass | yes |
| Pinwheel | Chroma Crush | black (54.4) | dry | yes |
| YUV Amplifier | Fade Amount, wet | flat field (15.0) | dry | yes |
| homegrownvhs/Pigment | Mix, wet | black (54.4) | dry | yes |
| czartroniks/Quilt | Density | 48 to 53 | no bypass | no |
| czartroniks/Fray | Length | 14.6 | no bypass | no |
| czartroniks/Porthole | Drift | 0.55, 1.03 at the working point | no bypass | yes |

So P12 gives dry passthrough for two of the eight, and does something different in
each: parameters are patch-specific, and no slot means the same in every program.
`locktake.dry_state` therefore uses only a parameter the program itself labels
Bypass; the four with one pass the strip with it on. The rest load inside a
Passthru lead-in.

## Toggle slots switch at 512

The firmware reads a toggle slot (P7 to P11) as on from device value 512, and off below
it. On firmware 1.0.2-rc.4, raw 14-bit CCs sent past pyvmancer to Pinwheel's Luma Invert
(slot 8) across the range read back `o=0` with the off picture for 0..511 and `o=1023`
with the on picture for 512..1023, the edge bisected to 511/512; YUV Bit Rotator's
Direction (slot 7) read back the same way, off at 496 and on at 512 (its picture was not
live in that state). Evidence: `captures/archive-rc4/toggle/*.json`, tool
`captures/diag/togglethreshold.py`.

pyvmancer's `set_param` coerces any nonzero toggle value to on, so a value in 1..511 went
out on while the firmware, and `BOOL_THRESHOLD`, read it as off. Every toggle value is now
snapped as the firmware reads it before it is planned, sent, echoed or recorded
(`profile.snap_toggles`, in `Transport.set_param`, `locked_take`'s schedule and
`locked.carried`).

## The MIDI CC map must be checked

On firmware 1.0.1 a program load with the source HDMI link dropped, which is how
`Session.load_program(..., link=...)` loads, can arm MIDI Assign (MIDI learn) on
one modulator slot with no button pressed. The next CC received is learned onto
that slot. Park's first write is CC0, so the slot then follows P1 and every take
looks like plain passthrough. The arming happens after the load returns: a sink
CC sent during the load, with the link still down, is too early. The live map
can also differ from settings keys 200..211, which a capture does not change.

syncsummoner, through pyvmancer, therefore:

- rewrites the whole map on every load (each key cleared, then set) and reads
  `modulation cc-map` back, logging and journalling the assignment;
- sends a sink CC that no slot listens on immediately before park's first CC;
- reasserts the map after park: a slot that captured the sink is reassigned and
  recorded in `Session.assign_events` with the program, slots and whether the
  load dropped the link; any other change raises `CcMapError`;
- reads the map back at the end of every render, audition and probe pass, and
  fails the pass on any change.

Check it with `syncsummoner device doctor` (`midi.cc_map`) or `vmancer cc-map`;
repair it by hand with `vmancer cc-map --restore --yes`.

## Format is a session constant

The device genlocks to its source and cannot convert. The card constrains what
the far end can be: genlocked to NTSC, `720p60` was the only usable progressive
format, with `480p`, `1080p30` and `1080i60` all returning no signal. Genlocked
to the Pi at 1920x1080@30 the same chain is fine. Pick one format for the whole
pipeline and never vary it.

The card returns frames at its own rate, not the session's: a 10 s pass at
`1080p30` comes back as ~600 frames, two per session frame
([the card's clock](#the-cards-clock)).

## Signal comes from the Videomancer's transmitter; the card exposes none

The capture card (AVerMedia Live Gamer Ultra 2.1, `uvcvideo`) has no readable
signal state. V4L2 `ENUMINPUT` always reports ok, there is no DV-timings ioctl,
the format list does not change with the source, and its vendor UVC extension
unit is an undocumented request/response mailbox that stays the same across
signal loss. With no input it synthesizes a high-contrast achromatic *No Signal*
slide, which no pixel test can tell from black-and-white program output
(`Lumarian`, `Colorbars` and `Pinwheel`'s effect frames all read as it).

The card's only input is the Videomancer's HDMI output, which keeps driving even
while the Pi's link is down; the card loses signal only when the transmitter
stops, as it does for a few seconds of every program load. The transmitter
state is in `video status`:

```text
"output":{"timing":"1080p30","hdmi_connected":true,...},"hdmi_tx":{"vic":34,"de":"2FE90F",...}   driving
"hdmi_tx":{"vic":0,"de":"000000",...}                                                         stopped
```

`transmitting(status, timing)` in `device/transport.py` is the rig's one signal
predicate: pyvmancer's `video.output_active` (a sink attached, a nonzero VIC and
data enable) and an output timing equal to the session format. A sink reported
absent is false; a firmware that omits the transmitter fields gives `None`,
unknown, which every caller treats as no signal.

- `settle()` polls it over serial after a load, and records nothing until it is
  true; a transmitter that never returns fails with the last `video status`.
- Render and archive passes check it again at the start and end of every
  recording, alongside the CC-map read-back, and fail the pass if it dropped.
- While it holds, every captured frame is the program's own output, so takes are
  judged by structure and motion alone.

## Input status flags are advisory

The firmware has reported `hdmi.connected: false` while passing video perfectly,
and a locked input while passing nothing. The top-level `locked` flag tracks
genlock, so it reads false whenever timing is overridden even though the input is
fine; the selected input's own sub-status is authoritative, which is what
`VideoStatus.source_locked` reads. None of these says the output is being driven;
only the transmitter state does.

Picture is judged by structure, never brightness or colour. A dead program is a
spatially uniform field: `Roll Shutter` outputs video black at exactly 16 on
every pixel of every frame, even under Bypass. A uniform field is one DC level
per JPEG block, which MJPEG and resampling reproduce exactly, so a frame is a
uniform field when its luma ranges over no more than one 8-bit step, and is
picture otherwise. Takes are judged at a point-sampled 160x90, so the samples
are pixels the card delivered rather than blends. Dark footage carries
full-range structure: dark 1960s black-and-white material through `Fray`
measured `YMIN=0 YMAX=255 YAVG≈31`, which a mean-level threshold called blank.

| Property | Means | Used by |
| --- | --- | --- |
| `mostly_picture` | most frames are picture, moving or not | render's load probe; the archive's passthrough canary |
| `frozen` | the take holds one picture plus noise | |
| `usable` | `mostly_picture` and not `frozen` | render, audition |
| `ever_structured` | any frame is picture | the archive's `dark` verdict |

MJPEG noise makes every frame of a still picture differ, so counting distinct
frames proves nothing. `frozen` uses the take as its own noise floor: for one
picture plus noise independent between frames, consecutive differences share one
noise term with opposite signs and correlate at exactly -1/2, whatever the
noise's level or texture. Motion pulls that correlation up and flicker towards
-1. The take is frozen when the mean correlation sits within three standard
errors of -1/2 and a one-way analysis of variance over pixels shows the frames
share a picture, which is what separates a held picture from snow, whose frames
also correlate at -1/2. Exact repeats are dropped first, and fewer than four
distinct frames cannot show motion.

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

## Locked playout

The Pi plays every stimulus with `syncsummoner.device.kmsplay`, a DRM/KMS player built on
pykms. ffmpeg decodes the clip to raw `bgr565le` with no `-re`; the player prerolls two
seconds of frames in RAM, then commits one atomic page flip per vblank to three RG16 dumb
buffers. Each clip frame is held for exactly `refresh / fps` vblanks, so the frame clock is
the display's pixel clock and flips cannot tear.

| Property | Value |
| --- | --- |
| Device | `/dev/dri/card1`, connector `HDMI-A-1`, primary plane |
| Mode | the EDID's preferred `1920x1080@30.00`, 74.25 MHz; kept, never re-set, when it divides the clip rate |
| Format | RG16, the bytes `ffmpeg -pix_fmt bgr565le` writes (red and blue come back exchanged otherwise) |
| DRM master | free: nothing holds `card1`, fbcon is an in-kernel client; `device doctor` fails `source.drm_free` otherwise |
| Software decode | 1080p30 H.264 at about 1.85x real time on the Pi 4's CPU; `h264_v4l2m2m` stalled |
| Scheduling | the pacing thread runs `SCHED_FIFO`, its ceiling raised with `sudo prlimit` for the player's shell only, with the garbage collector frozen |

Every flip-complete event's vblank sequence is checked against the vblank it was committed
for and logged with its time (`flips.bin`). A flip that lands late is logged on the vblank it
landed on, flagged `MISSED` (2, beside `SEGMENT`'s 1), and counted in `status.json`'s `missed`;
the next commit is for the vblank after it, on the same grid, so the frames due meanwhile are
skipped and every later frame lands on its own vblank. The host faults only the takes resting on a
frame it may have mis-shown ([evidence.md](evidence.md#frame-locked-sends)). An undecoded frame,
a flip that never completes or a mode that does not divide the clip rate ends the player with a
fault in `status.json`, which the host raises as `PlayoutError`. Without real-time scheduling a garbage-collector pause or the
decoder's teardown delayed one commit past its vblank at the end of every clip; with it none
has been missed.

When the player exits, fbdev takes the display back, so it leaves its last frame in
`/dev/fb0` and the picture stands. While it runs, fbdev blanking does nothing, so `Link`
signals the player instead: `SIGUSR1` turns the CRTC off (the link drops and `dpms` reads
`Off`), `SIGUSR2` turns it back on. The clip's timeline keeps running on the event clock
meanwhile, and the frame due when the link returns is shown. The returning CRTC's vblanks
sit at an arbitrary phase to the old ones, which the flip log records.

## The card's clock

At 1920x1080 the card delivers about 60 fps whatever rate is asked for, on its own crystal: it
is not locked to the signal. From the 30 Hz signal it delivers about two captures a frame, and
two captures of one frame are byte-identical MJPEG packets, so a change in a capture's bytes
marks a new frame. The frame period on the host's clock is not two capture periods: it runs some
tens of parts per million off them (measured per take below), so the capture a frame starts on
moves by one capture, a slip, once or twice in a full take. At a slip a frame is captured once
(frames shorter than two captures) or three times (longer).

uvcvideo's buffer sequence numbers have no gaps there; the recorder keeps each capture's
sequence, timestamp and a digest of its bytes in a `.v4l2` sidecar, so a frame the driver loses
shows as a sequence gap. The timestamps carry 1 µs resolution and arrive late by a varying amount:
against the straight line through them, the card's capture clock, the earliest sit a few hundred
microseconds early and the latest up to the take's jitter, its largest residual (0.8 to 2.6 ms
over the takes below). The capture line removes that jitter, so a capture's time is its place on
the line.

The device's frames start on a straight line of their own, `origin + n * period`: two crystals at a
constant ratio. Each capture that starts a frame puts that frame's start after the capture before
it and at or before itself; `compose.frameline` keeps the polygon of lines meeting every such
bracket and fits the least-squares line inside it. Between slips the brackets leave the line one
capture period (16.7 ms) wide; a slip pins it to the capture where it happens, and two slips pin
its period. A take with no slip bounds its drift to about one capture period over its length
(about 43 ppm over 400 s) and no closer.

| Take | Frame line drift, ppm (range the captures admit) | Slips | Frame 0's start known to |
| --- | --- | --- | --- |
| param-evidence full Pinwheel | -0.0 (-42.5 to +42.5) | 0 | 16.67 ms |
| param-evidence full boneoh_YUV_Bit_Rotator-1473b2c4 | +0.0 (-42.0 to +42.0) | 0 | 16.67 ms |
| param-evidence full czartroniks_Quilt-803bc8ea | -42.7 (-54.2 to -30.0) | 1 | 1.11 ms |
| param-evidence rc4 Pinwheel | -78.5 (-78.5 to -30.0) | 2 | 8.40 ms |
| param-evidence rc4 YUV_Amplifier-6b88e9be | +0.0 (-43.0 to +43.0) | 0 | 16.67 ms |
| param-evidence rc4 boneoh_YUV_Bit_Rotator-1473b2c4 | -0.0 (-41.9 to +41.9) | 0 | 16.67 ms |
| param-evidence rc4 homegrownvhs_Pigment-fde77c40 | +0.0 (-42.5 to +42.5) | 0 | 16.67 ms |
| param-evidence rc4 czartroniks_Fray-fad121db | +0.0 (-44.1 to +44.1) | 0 | 16.67 ms |
| param-evidence rc4 czartroniks_Porthole-9444a020 | +0.0 (-44.1 to +44.1) | 0 | 16.67 ms |
| param-evidence rc4 czartroniks_Quilt-803bc8ea | -0.0 (-44.6 to +44.6) | 0 | 16.67 ms |
| param-evidence rc4 teletect_Teletect_RamBend-3c9af621 | +0.0 (-43.2 to +43.2) | 0 | 16.67 ms |
| param-evidence short YUV_Amplifier-6b88e9be | +0.0 (-316.8 to +316.8) | 0 | 16.67 ms |
| param-evidence short boneoh_YUV_Bit_Rotator-1473b2c4 | +0.0 (-239.0 to +239.0) | 0 | 16.67 ms |
| param-evidence short homegrownvhs_Pigment-fde77c40 | -0.0 (-257.5 to +257.5) | 0 | 16.67 ms |
| param-evidence short teletect_Teletect_RamBend-3c9af621 | -0.0 (-235.7 to +235.7) | 0 | 16.67 ms |
| param-evidence short czartroniks_Fray-fad121db | not attributed: the driver lost frames in the clip | | |
| param-evidence short czartroniks_Porthole-9444a020 | +0.0 (-283.6 to +283.6) | 0 | 16.67 ms |
| param-evidence short czartroniks_Quilt-803bc8ea | -0.0 (-285.7 to +285.7) | 0 | 16.67 ms |
| param-evidence Pinwheel | -0.0 (-320.3 to +320.3) | 0 | 16.67 ms |
| archive-rc4 failed YUV_Amplifier-6b88e9be-00 | -59.9 (-71.8 to -30.0) | 1 | 5.06 ms |
| fw-upgrade lock-passthru (lockcheck) | +0.0 (-153.5 to +153.5) | 0 | 16.67 ms |
| fw-upgrade lock-passthru-2 (lockcheck) | +0.0 (-150.2 to +150.2) | 0 | 16.67 ms |

## A program load breaks the capture

A load stops the Videomancer's transmitter. The card then delivers nothing for a few
hundred milliseconds at a time, with no gap in its sequence numbers, and relocks with its
60 Hz grid at a new phase to the frames. Counting captures cannot carry a frame index across
a load. The device's own frames keep their lag behind the Pi's flips across it: in the two
rc4 takes with legible strips on both sides of their load (czartroniks/Porthole and
teletect/RamBend), the lag every line admits on one side meets the other's.

| Load | Load call | Captures missing |
| --- | --- | --- |
| Passthru after Passthru | 3.0 to 4.1 s | two gaps, about 0.3 and 0.75 s |
| Pinwheel after Passthru | 3.1 s | about 0.35 and 0.7 s |
| YUV Amplifier after czartroniks/Fray | 9.2 s | |
| czartroniks/Fray after Pinwheel | 13.7 s | |

A load that changes author is the slow one: one that enumerates another author takes about 20 s. A session's lead-in holds a load only for a program with no dry control, and is sized by the longest load timed on the rig (`device.loadtimes`), so a slow load timed once is held for after.

Across eleven link-dropped Passthru loads in one recording, the capture time of each frame's
first capture, against the Pi's flip time for that frame, moved by between -4.5 and +10.0 ms
from one relock to the next, a spread of 14.4 ms: the card's capture phase, which moves within
one capture period. `compose.lock.attribute` therefore fits each run's line apart, the period
bounded by the run with most strips; a run with no legible strip is numbered by the one whole-frame
shift that puts its Pi lag where the strip-numbered runs' lies.

Only strips on unprocessed frames are evidence about the source frame: Passthru before a
program's load and after the lead-out's reload, and a program's model-proven bypass. While a
program processes its input, its strips are its own picture (czartroniks/Quilt tiles past frames:
on region takes most of its in-clip strips read a frame ahead or behind), so they neither anchor
the frame line nor count against it (`lock.processed`): from the first write off the dry state, or
the program's load command, to the close of the landing window of the write or reload returning
it, its echo or return plus the latency band and the frame it lands on. A locked take records its
loads' times for this (`loads` in the wire record); a take without them is processed from the end
of its first run to the start of its last, the loads' outages. A program delaying its picture so
shows as measured latency, not inside the line. A strip-less run the flips cannot number, such as a
second of the card relocking during a load, is left unattributed; the take stays exact only if
every clip frame has a capture elsewhere.

A processed run between two Passthru runs, a no-dry take's clip, is numbered by the capture
clock alone (`compose.relock`): frame `n` of run `r` starts at `t0 + n * T + latency_r`, one `T` a
take and one latency a run, changing only where the card relocks. `T` is the range every run's
frame line admits; each Passthru run's latency is the range its strip-named brackets admit at it,
and the processed run's, numbered from its first change event, is the range its own change events
admit. A numbering is a candidate when some period in range puts the processed run's latency within
half a frame of both Passthru runs' (a linear feasibility problem, `scipy.optimize.linprog`). No
candidate fails the take; the run is numbered by the highest, and its captures carry `ambiguity`,
how many frames earlier the others put them. The lock record's `clock` keeps, per such run, the
candidates, the period range, the three latencies and the frame a gap's rounding would name, a
cross-check only. The Pi's flip log disagreeing with the strips is then reported, never acted on.

A run has one numbering, so where the clock leaves several, `probe sweep collect` chooses jointly over
the run's settings (`locked.number_runs`): every shift back from the highest candidate scores each
setting's follow at that shift's own leads, the best lead's z Bonferroni over them as a one-sided p,
and pools the settings by Fisher's combination (`replay.pooled_follow`, `matched.fisher_z`). A shift
whose pooled z beats every other's by more than the bound for a difference of two unit z scores,
`matched.critical` over the shifts after dividing by the square root of two, renumbers the run, its
ambiguity gone and its leads points (`replay.select_shift`); otherwise the run keeps its candidates'
union. Nothing assumes a program's delay is not negative. The selection, pooled z per shift and margin
join the run's `clock` record, which `collect` writes into the take's lock record (`lock.save_clock`,
`lock.read_clock`) beside the keys the capture wrote.

## The analog input decodes luma and sync only

Composite fed straight in yields zero chroma under both NTSC and PAL while luma
decodes correctly and standard-aware — the 7.5 IRE pedestal is honoured under
NTSC and absent under PAL, and PAL is the better luma path (gain 1.028 against
0.958, black at 0 rather than 18/255, 20% more lines). Colour appears only with
an external RGB encoder, which measured badly out of calibration: luma gain 1.21,
ramp clipped at x=491 of 720, blue clipped to white, black lifted to purple.
Colour work over the analog path needs that encoder calibrated first. The session
runs HDMI, which carries chroma.

## Capture is stored, never decoded, while it runs

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

The host touches no frame while the rig runs: `syncsummoner.device.v4l2rec` dequeues
the card's buffers with linuxpy and pipes them untouched into ffmpeg, which stores them
(`-c:v copy` for MJPEG, FFV1 for an archive). The sidecar's timestamps are the driver's
`CLOCK_MONOTONIC` buffer times, the clock `time.monotonic` reads, so frames attribute to
the setpoint being held when each arrived with no alignment step. A recording's length is
counted in frames at the rate the card actually runs.

### Each run of identical captures is stored once

Two captures of one frame are byte-identical, so the recorder stores a buffer only when its
bytes differ from the last one stored (a matching digest is confirmed byte for byte) and frames
the stored ones as IVF, each with its capture's timestamp, so the file keeps real time. The
sidecar (version 2: a header naming the stored fourcc, size and time base, then one row per
capture) adds to each capture's `FRAME` row the stored frame showing it and its timestamp.
`read_log` returns the `FRAME` rows of either version. Every reader opens a take through
`syncsummoner.device.stored.captures`, which hands ffmpeg every capture as it arrived: an old
take directly, a deduplicated one through a FIFO fed from the stored frames, each checked
against its logged size and digest.

`syncsummoner probe dedupe TAKE...` rewrites old takes: beside the take, verified capture for
capture by bytes and by decoded frame (framemd5), then hard-linked over the take and every other
name of it found under `--links DIR` (sidecars first). A take with a name it cannot account
for, or in a read-only folder, is left alone and its saving reported; `--dry-run` writes nothing.
A rewrite interrupted between its renames leaves `.TAKE.dedupe.mkv` beside the take, which a
rerun reports rather than overwrites.

## The state-index strip survives the chain

An 8-bit Gray-code strip played from the Pi decoded back bit-exact through
composite encode, analog transport, `Passthru`, HDMI output and USB capture, and
a 16-bit strip decodes the same way through the HDMI session. The decoder
thresholds against the strip's own midpoint rather than a fixed constant, because
of the analog path's 0.958 gain and 18/255 pedestal.

A destructive program mangles individual strips: inside a `Derez` take, frames
decode correctly either side of ones that decode to nonsense. Alignment therefore
takes the lag a take's frames agree on rather than trusting any single frame.

Over the 8 programs of `takes6`, a 16-bit strip stayed legible on 4 and in
patches on the others, which is why footage frames are identified by a clock
fitted to whatever strip reads, with motion as the fallback, rather than by
reading every frame.

## `settings export` and `settings get` hung the serial shell before 1.0.1

On earlier firmware both verbs returned no response and left the command
processor unresponsive to every subsequent command; MIDI and video output
continued, `USBDEVFS_RESET` did not recover it, and a power cycle was required.
On 1.0.1 `settings get <key>` answers `{"key":k,"type":"uint32","value":v}`, and
the CC-map assignment on every load relies on it for keys 200..211.

## Loads block the shell

On firmware 1.0.1, traced with `fpga status` polled every 100 ms, the shell answers
nothing while it works on the card: `programs enumerate lzx` took about 6 s,
`programs enumerate czartroniks` about 10.7 s, and `program load` about 1 s once its
author was active (0.02 s between two lzx programs). One raw `program load Fray`
with lzx active answered after 11.4 s. Queued polls went unanswered and no log
line appeared, while the device stayed enumerated as 16d0:14db. The FPGA can then
sit at `state idle, configured true, program ''`, or keep running the previous
program, before running the new one about 1 s later, and one `fpga status` poll can
itself go unanswered meanwhile. A 20 s reply deadline once failed a Passthru load
after a route change that completed on its own.

None of that is a fault, so syncsummoner waits on liveness through pyvmancer: a
load ends on the program running and `program info` answering, an error reply,
the device leaving USB, or the FPGA settling on a different program. The
transmitter's return after it is waited for the same way. The capture probe that
follows keeps a budget, `probe_timeout_s`, because that is the card and recorder,
not the device.

## One shell client, one command at a time

The shell answers one connection and one command at a time, and a reply that a
client stopped waiting for would be read as the next command's. pyvmancer awaits
every reply on USB liveness with no deadline, opens the tty exclusively, and
sends a sync barrier before a client's first command. syncsummoner opens one
`Transport` at a time and closes it before the next opens: a second open, here
or by any other process, raises `PortBusyError`.

## A link-dropped load holds the link down for the command only

`Session.load_program(..., link=...)` drops the source link while the device
takes `program load`, and restores it once `@program:ok` commits the load. The
FPGA runs on the clock the receiver recovers from that link, so a program cannot
reach running while it is down: waiting for `fpga status` to run the program, or
for the transmitter or a picture, inside the drop is a circular wait that only an
unblank ends. Those waits therefore start only after the link is back, and
`Transport.watch` raises `HeldLinkError` if one starts or polls while the link is
held down. Progress lines written during the drop end in `link down`.

## Vid Route Mode decides what the processor is fed

SETTINGS > Vid Route Mode is settings key 1, valued as the Serial Port Guide's
table that pyvmancer carries as `ROUTE_MODES`: 0 Analog In, 1 HDMI In, 2 Dual In,
3 Standalone, 4 Auto. With the route found on Dual
In, `video status` still read source `hdmi`, the HDMI input locked and the
transmitter active, Passthru loaded and ran, and every program output exact
zero, so the input check below passed while nothing was fed to the processor.
Programs looked dead until the route was set back to HDMI In.

`device doctor` fails `video.route` unless key 1 reads the expected value (HDMI
In, 1, unless `--route` says otherwise), and render, audition and archive check
it before the run and after every load and pass, next to the CC map, stopping
with `RouteError`. A program is judged dark only after a Passthru control on
the same rig shows a picture; a dark control is a rig fault reported with the
route, input lock and transmitter state.

## Vid Route Mode Auto can strand the input

Under SETTINGS > Vid Route Mode Auto, dropping the source HDMI link made the
device fall back to the analog input and stay there, so its output was black with
nothing else amiss. `video status` shows it as `source: analog`. `device doctor`
fails `device.input` when the selected input is not the session's `hdmi`, and
archive and render passes check it after every load, failing that pass with
`InputRouteError` rather than reading the black as a dark program or a dark rig.

## Archive decoding on a GPU is slice-bound

FFV1's range coder is serial within a slice, and ffmpeg's Vulkan decoder
dispatches one workgroup per slice, so a frame decodes no faster than its largest
slice does on one GPU thread. The archives are recorded with 4 slices. On
defroster (RTX 5080, driver 620.70.06, ffmpeg 8.1.3, 24 cores), one Vulkan stream
of an archive decoded 2.8 frames/s against 37 for one CPU core, and four
concurrent Vulkan decoders, each its own process, totalled 2.3: the GPU runs one
process's work at a time. Re-encoded with more slices, the first 120 frames of
Bitcullis decoded on the GPU at 3.0, 11.3 and 36.8 frames/s for 4, 16 and 64
slices, against about 53 on one core each time, and bit-identically to the CPU
throughout. A library refit decodes one stream per core, so the GPU does not
speed it up on this host at any slice count measured, and `refit --decode auto`
stays on the CPU.
