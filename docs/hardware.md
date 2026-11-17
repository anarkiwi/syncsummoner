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
native ranges over serial and names unused parameters `-`, so no `.vmprog` TOML
parsing is needed. A parameter with native range `0..7` has 8 meaningful
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
at low. The render budget, `--cc-budget`, is in sets per second at whatever
resolution the transport runs; its default of 200 sits inside the measured
clean range. To re-derive it, run `device doctor --latency` with
`--latency-params` naming as many knobs as a score drives and take the
`midi.backlog` finding's fastest clean rate times the parameter count.

## CCs latch on the device's frame tick

Measured on firmware 1.0.1 at 1080p30 with the locked player, a still strip-coded
lead-in, and a bypass or crossfader toggled at random phases on eight programs:

| Measure | Value |
| --- | --- |
| Picture change | always on a frame start, never part way down a frame (0 mixed captures in 120 toggles) |
| Readback (`modulation status` combined value) | changes once per output frame; its serial timing is too coarse to time the latch, so landings rest on the picture |
| CC to readback value | exact for all 1024 values, high-resolution CC with manual parked at 0 |
| Serial `modulation status` round trip | about 1.4 to 2.2 ms, 279-byte reply |
| Send to the card's first capture of the frame a write lands on | 1.0 to 1.5 frames, known per take to about 2 ms from 16 calibration writes; it moves between takes with the card's capture phase, over less than half a frame |

Render therefore sends each write half a frame before the instant it must precede
to land on its target frame, and proves where it landed ([evidence.md](evidence.md)). The early bias
of design 4.4 picks the target: the first frame starting at or after the write's
time less 15 ms.

## The crossfader is not a dry control

pyvmancer calls P12 the crossfader that gates every program's output. On the
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

So P12 gives dry passthrough for two of the eight. A program's own bypass, or a
P12 whose role is wet where there is no bypass, does for all five that have one;
`locktake.dry_state` picks it.

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
([the card's cadence](#the-cards-cadence)).

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
for and logged with its time (`flips.bin`). A late flip, an undecoded frame or a mode that
does not divide the clip rate ends the player with a fault in `status.json`, which the host
raises as `PlayoutError`. Without real-time scheduling a garbage-collector pause or the
decoder's teardown delayed one commit past its vblank at the end of every clip; with it none
has been missed.

When the player exits, fbdev takes the display back, so it leaves its last frame in
`/dev/fb0` and the picture stands. While it runs, fbdev blanking does nothing, so `Link`
signals the player instead: `SIGUSR1` turns the CRTC off (the link drops and `dpms` reads
`Off`), `SIGUSR2` turns it back on. The clip's timeline keeps running on the event clock
meanwhile, and the frame due when the link returns is shown. The returning CRTC's vblanks
sit at an arbitrary phase to the old ones, which the flip log records.

## The card's cadence

At 1920x1080 the card delivers about 60 fps whatever rate is asked for. From the locked
30 Hz signal it delivers exactly two captures per frame, and the two are byte-identical MJPEG
packets: over a minute, 898 pairs and the two partial ones at the ends, every strip on one
cadence. uvcvideo's buffer sequence numbers have no gaps there; the recorder keeps each
capture's sequence, timestamp and a digest of its bytes in a `.v4l2` sidecar, so a frame the
driver loses shows as a sequence gap and the card's own repeats show which captures start a
frame.

## A program load breaks the capture

A load stops the Videomancer's transmitter. The card then delivers nothing for a few
hundred milliseconds at a time, with no gap in its sequence numbers, and relocks with its
60 Hz grid at a new phase to the frames. Counting captures cannot carry a frame index across
a load.

| Load | Load call | Captures missing |
| --- | --- | --- |
| Passthru after Passthru | 3.0 to 4.1 s | two gaps, about 0.3 and 0.75 s |
| Pinwheel after Passthru | 3.1 s | about 0.35 and 0.7 s |
| YUV Amplifier after czartroniks/Fray | 9.2 s | |
| czartroniks/Fray after Pinwheel | 13.7 s | |

A load that changes author is the slow one. A program loaded inside a locked take's lead-in, which only a program with no dry control still is, needs most of its 40 s for that reason: a load that enumerates another author takes about 20 s inside the take.

Across eleven link-dropped Passthru loads in one recording, the capture time of each frame's
first capture, against the Pi's flip time for that frame, moved by between -4.5 and +10.0 ms
from one relock to the next, a spread of 14.4 ms. Two captures a frame put a frame's first
capture within one capture period (16.7 ms) of the frame's flip plus a fixed latency, so that
offset cannot move by half a frame. `compose.lock.attribute` therefore takes the captures
that start a frame from the card's repeats, and names the frame each shows by the nearest
flip time; every run of the eleven matched its strips. A run whose frame starts disagree is a
`LockError`.

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
