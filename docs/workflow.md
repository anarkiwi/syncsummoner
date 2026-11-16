# Example render workflow

One source clip and one audio track in, one finished piece out: the effects are
chosen from what the track does, checked on a 30 second audition, then committed
to a full render that fades up from black and silence and back down again.

Everything below runs through `docker run`, on the rig described in
[rig.md](rig.md) and measured in [hardware.md](hardware.md).

## The rig

```
   Raspberry Pi 4B "videopi"                     LZX Videomancer
   +---------------------------+                +----------------------------+
   | /dev/fb0, 1080p30         |     HDMI       | HDMI IN                    |
   | no compositor, DRM free   |===============>|                            |
   | plays the timecoded clip  |  source clip   | 6 knobs, 5 toggles,        |
   +-------------+-------------+                | crossfader (P12)           |
                 |                              |                            |
                 | ethernet                     | HDMI OUT ------------------+---+
                 | ssh, BatchMode:              | CVBS / S-video (alternate) |   |
                 | playout and link blanking    +------+-------------+-------+   | HDMI
                 |                                     |             |           | processed
                 |                          USB (CDC)  |             | USB MIDI  v  picture
                 |                       /dev/ttyACM0  |             |     +-----------------+
                 |                       16d0:14db     |             |     | AVerMedia Live  |
                 |                                     |             |     | Gamer Ultra 2.1 |
                 v                                     v             v     +--------+--------+
   +-------------------------------------------------------------------+            |
   |  workstation:  docker run anarkiwi/syncsummoner                   |<-----------+
   |  /dev/video0   /dev/ttyACM0   /dev/snd/midiC*D0   ssh             |  USB 3 (UVC)
   +-------------------------------------------------------------------+
```

| Cable | From | To | Carries |
| --- | --- | --- | --- |
| HDMI | videopi HDMI0 | Videomancer HDMI IN | the source clip, played out at rate |
| HDMI | Videomancer HDMI OUT | AVerMedia HDMI IN | the processed picture |
| USB 3 | AVerMedia | workstation | the capture, as `/dev/video0` |
| USB | Videomancer | workstation | CDC shell (`/dev/ttyACM0`) and MIDI (`/dev/snd/midiC*D0`) |
| ethernet | videopi | workstation LAN | ssh: playout, and blanking the framebuffer to drop the link |

The Videomancer genlocks to its input and cannot convert, so the whole chain runs
at one format. `1080p30` is this rig. The composite path into the analog input is
the alternative, at `ntsc` or PAL, and is not mixed with the HDMI one.

Audio never reaches the rig. The track drives the composition and is muxed back in
at the end; nothing is recorded through the capture card but picture.

## The image

The released image is what the rig is driven from; building the working tree
over the same name is how a change gets tried before it is tagged.

```sh
docker pull anarkiwi/syncsummoner
docker build --target runtime -t anarkiwi/syncsummoner .
```

```sh
ss() {
  docker run --rm -it --network host \
    --user "$(id -u):$(id -g)" -e HOME="$HOME" \
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    --group-add "$(getent group dialout | cut -d: -f3)" \
    --group-add "$(getent group audio | cut -d: -f3)" \
    --group-add "$(getent group video | cut -d: -f3)" \
    --device /dev/video0 --device /dev/ttyACM0 --device /dev/snd \
    -v "$PWD:/work" -v "$HOME/.ssh:$HOME/.ssh:ro" \
    anarkiwi/syncsummoner "$@"
}
```

| Flag | Why |
| --- | --- |
| `--device /dev/ttyACM0` | the CDC shell: programs, video status, settings |
| `--device /dev/snd` | ALSA rawmidi, which is how parameters are driven |
| `--device /dev/video0` | the capture card; ffmpeg inside the container records it |
| `--network host` | reaches `videopi` by the name the host resolves it with |
| `-v $HOME/.ssh:$HOME/.ssh:ro` | ssh runs `BatchMode=yes`, so the key must already work unattended |
| `-v $PWD:/work` | media, profiles and outputs; `/work` is the image's working directory |
| `--user`, `-e HOME`, `/etc/passwd` | ssh refuses a config or key it does not own, and root in the container owns none of yours. Running as yourself also leaves the renders owned by you rather than by root |
| `--group-add` | the device nodes are group `dialout`, `audio` and `video`; as a non-root user those groups have to be carried in |

Every command logs each stage it enters and how long that stage took, to stderr;
`-v` adds per-program detail and `-q` leaves only warnings and errors. Where a
step has a known number of items — frames to timecode, programs to evolve or to
render — it draws a progress bar with a rate and an ETA, which is what a
half-hour render needs to be legible. A pass is a real-time capture with nothing
to count, so it meters elapsed seconds against the score's length instead. The
bars want a terminal, hence `-it` in the wrapper; piped into a file they become a
line every 30 seconds, so a long pass is never silent.

Run as yourself, not as root: ssh refuses a config or private key owned by
someone else, so a container running as root against a mounted `~/.ssh` fails
with `Bad owner or permissions`, and everything it does write into the work
directory comes out owned by root. The wrapper above takes care of both.

The Pi side has two requirements the workstation cannot supply for it: the ssh
key must already work unattended (`BatchMode=yes` never prompts, so the host key
has to be in the mounted `known_hosts` before the first run), and dropping the
HDMI link writes `/sys/class/graphics/fb0/blank`, which needs passwordless sudo:

```sh
ssh pi@videopi 'echo ok'      # must succeed with no prompt
ssh pi@videopi 'sudo -n true' # must not ask for a password; blanking is a sudo write
```

Check the rig answers before anything long starts:

```sh
ss device list          # serial, rawmidi node, tty, usbfs
ss device ping          # firmware, and the video status the device reports
ss device programs      # what can actually be composed for
ssh pi@videopi 'echo 0 | sudo tee /sys/class/graphics/fbcon/cursor_blink'
ss device doctor --latency  # rig checks, plus MIDI CC latency and backlog at the session's frame rate
```

That last line matters: a blinking console cursor inverts its cell on top of the
framebuffer and lands in every take. The other pre-flight gotchas — parking the
panel, one owner of `/dev/video0`, a capture card with no signal state of its own — are in
[rig.md](rig.md#gotchas).

## Inputs

```
work/
  source.mkv      the clip to process   (any format; it is conformed and padded, never stretched)
  track.flac      the audio to drive it  (any format ffmpeg reads: WAV, FLAC, MP3, AAC in .m4a/.mp4, ...)
  profiles/       fitted profiles, one YAML per program
```

To rehearse the whole workflow with no footage of your own, synthesize both. The
clip is three 20 s segments so the analysis has shot boundaries to find, and the
track is deliberately longer than the clip, so the length rule has something to
do:

```sh
docker run --rm -v "$PWD:/work" --entrypoint /bin/sh anarkiwi/syncsummoner -c '
for i in 0 1 2; do
  case $i in
    0) SRC="testsrc2=size=640x480:rate=25:duration=20";;
    1) SRC="smptehdbars=size=640x480:rate=25:duration=20";;
    2) SRC="rgbtestsrc=size=640x480:rate=25:duration=20";;
  esac
  ffmpeg -loglevel error -y -f lavfi -i "$SRC" -c:v libx264 -crf 20 -pix_fmt yuv420p p$i.mkv
done
printf "file p0.mkv\nfile p1.mkv\nfile p2.mkv\n" > l.txt
ffmpeg -loglevel error -y -f concat -safe 0 -i l.txt -c copy source.mkv
rm -f p0.mkv p1.mkv p2.mkv l.txt
ffmpeg -loglevel error -y -f lavfi -i "aevalsrc=\
0.7*sin(2*PI*60*t)*exp(-9*mod(t\,0.5))\
+0.35*sin(2*PI*3200*t)*exp(-45*mod(t\,0.25))\
+between(t\,30\,75)*0.3*sin(2*PI*220*t)\
+between(t\,55\,75)*0.25*sin(2*PI*880*t)*exp(-20*mod(t\,1.0)):d=75:s=44100" -c:a flac track.flac'
```

That is 60 s of picture against 75 s of sound: every step below reports 60 s,
because a render is the length both inputs cover.

The composer can only reach for a program it has a profile for, and only for
parameters that were measured to move something. With no profiles there is
nothing to compose. Build them once per firmware:

```sh
ss probe archive --out archive/ --source-host pi@videopi   # native frames, resumable, hours
ss probe refit --archive archive/ --out profiles/          # profiles from the archive, no device
ss profile show --profiles profiles/
```

A program is dark, and its frames are discarded, only when every inspected frame
of its sweep is a uniform field: a sweep legitimately visits flat setpoints and a
generator may hold one picture. The sweep is recorded only while the device's
HDMI transmitter drives the session format, so every frame is the program's own
and black-and-white output is picture like any other. The verdict is left as `<program>.dark.json` in the archive directory
and a resume skips it; each run lists the programs it skips. To re-judge them
under the current criterion (an archive committed for one supersedes its verdict):

```sh
ss probe archive --out archive/ --source-host pi@videopi --retry-dark
ss probe archive --out archive/ --source-host pi@videopi --retry-dark --program Lumarian
```

A program that fails (no picture after its load, a CC map that changed, a shell
error, the device reading an input other than `hdmi`) is logged as `FAILED` with
its error, leaves no archive and no dark verdict, and is retried on the next
resume; the run goes on. Only a rig fault ends a run, and only on positive
evidence: the device gone from USB, or a fresh Passthru that loads, reads `hdmi`
and drives the session timing but shows no picture (which a lost capture card also
produces). A load that merely takes long is never a fault. The closing summary
lists every failed program with its error and names the fault, if any.

Loads, and the transmitter's return after them, wait for as long as the device
stays on USB, logging `still loading ...` lines as they go; author switches take
several seconds of shell silence. `--stall SECONDS` (before the subcommand) is an
opt-in guard for unattended runs: it fails a wait once nothing the device reports
has changed for that long. It is off by default.

An archive keeps the device's own `program info` reply beside its frames, and a
refit carries it into every parameter's name, native range and role. An archive
recorded before that is back-filled from the device without re-archiving: each
program is loaded once and asked, a few seconds apiece, resumably. The SD card's
manifest and `.vmprog` files cannot stand in for this, since firmware built-ins are
absent from them and the card is only readable over the same serial link.

```sh
ss probe names --archive archive/      # program info into each archive, resumable
ss probe refit --archive archive/ --out profiles/
```

Each parameter's `role` decides what the composer may do with it, and comes from
the label its program gives it:

| Role | Labelled | Composer |
| --- | --- | --- |
| `bypass` | `Bypass` | never driven, held off for the whole pass |
| `wet` | P12 as `Mix`, `Blend`, `Fade`, `Wet` or left unnamed | driven only by `morph`, within the score's wet range |
| `momentary` | `Reset`, `Trigger`, `Retrig`, `Serve`, `Fire`, `Impulse` | only pulsed by `punch`, and released |
| `effect` | anything else | any gesture |

A profile with unnamed parameters (`p1` to `p12`) has every slot an `effect`
except P12, so it will drive a bypass blind: refit it from a named archive.

A profile fitted before 0.2.0 still composes, off the inferences the planner used
to make for itself. It will not carry the measured curves the objective now scores
against, nor `registered`, `pointwise` and `style`. Refitting is what picks those
up, and it needs the archive rather than the rig, so it costs CPU and no rig time.

## 1. Hear what the composer hears

```sh
ss analyze source.mkv --audio track.flac -o features.json
```

```
09:17:22 analyzing audio (path=track.flac)
09:17:24 analyzing audio done in 2s: seconds=75.0 bpm=117.5 sections=2
09:17:24 analyzing video (path=source.mkv)
09:17:42 analyzing video done in 18s: seconds=60.0 fps=25.00 shots=2
09:17:42 render length is 60.0s, the shorter of what was given
```

Tempo, beats, downbeats, the self-similarity segmentation into sections, onset
surprisal, and per-frame video statistics. This step is inspection only — compose
runs the same analysis itself — but it is worth a look when a composition comes
out wrong: if the sections are nonsense the score will be too, and that is an
analysis problem, not a rig problem.

## 2. Compose — this is where the effects are chosen

```sh
ss compose source.mkv --audio track.flac --profiles profiles/ --format 1080p30 \
     --style glitchy --seed 7 --passes 8 --density 0.5 -o score.yaml
```

For every program with a usable profile, the planner evolves a layer of gestures
anchored to the track's beats, downbeats and section edges, scores it against the
measured response of that program, and keeps the best. `--passes 8` keeps the
eight strongest programs, one capture pass each; `--style` biases which gestures
are drawn on; `--seed` makes the whole search reproducible.

`--budget` is the total number of candidates evaluated, spread over every program
with a profile, so the generations each program gets is
`budget / (6 * programs)`. Against twenty profiles the default 48 buys a single
generation — raise it to `20 * 6 * 10` for ten.

```
09:18:22 composing 60.0s over 20 programs: 1 generation of 6 candidates, 2 sections at 117.5 bpm
09:18:23 kept Passthru: score -0.483, 15 gestures
09:18:23 kept Delirium: score -0.531, 15 gestures
09:18:23 kept Loadstar: score -0.555, 15 gestures
09:18:23 kept Bitcullis: score -0.560, 15 gestures
score.yaml
```

The path of whatever was written is the only thing on stdout, so a command can be
used in a pipeline; every stage line above is stderr.

The score's duration is the length both inputs cover — the shorter of clip and
track — and the sections are clipped to it. Everything downstream inherits that,
so the audition, the render and the finished file are all the same length by
construction. What came out:

```sh
grep -E '^(duration|bpm)' score.yaml
```

The wet range is how far into the effect the wet control is driven, as fractions
of its travel. Neither end blacks the output on any archived program, so it is a
look rather than a limit: `default` and `smooth` stay within 0.10..0.50, `glitchy`
reaches fully wet at 0.10..1.00, and `--wet 0.3,1` sets it outright. A score
records its range; one written before it did renders within 0.10..0.50.

Before spending rig time on it, see what the score will actually drive:

```sh
ss score show score.yaml --profiles profiles/
```

```
wet range 0.10..0.50 (102..512)
layer 2: Breakout
   P1  Ball Spd         effect    continuous   23 writes     0..1023 ends 898   hold x1, hysteresis_loop x3, morph x1
   ...
   P7  Serve            momentary boolean       6 writes     0..1023 ends 0     punch x3
   P11 Bypass           bypass    boolean    undriven
   P12 Mix              wet       continuous   39 writes   102..512  ends 512   morph x3
layer 4: Pinwheel
   ...
   P12 Chroma Crush     effect    continuous   84 writes   416..748  ends 748   cliff_cross x2
   gestures that write nothing: morph x6
```

Every layer lists all twelve parameters by name with their role, how many writes
the render will send, the range they cover, where they end, and which gestures
asked for them; that is the same plan the render plays. A line marked `!` is a
bypass being driven or a toggle left on at the end of the pass; a layer marked
`!` has no profile or drives nothing, and gestures that render nothing are
listed under their layer. It exits 1 when anything is flagged.

## 3. Audition before committing the rig

```sh
ss audition score.yaml --source source.mkv --audio track.flac \
     --format 1080p30 --source-host pi@videopi \
     --seconds 30 --start 60 -o audition.mp4
```

The score is windowed to those 30 seconds, the footage and the track are both
seeked to where the excerpt starts, and it renders as one real-time pass through
the score's strongest program, mastered exactly as the full render will be. Cost
is one program load plus 30 seconds. `--start` has to sit inside the score, so
against the 60 s synthesized clip use something like `--start 20 --seconds 10`.

To hear the cut as well as the effect, audition across programs — one pass each,
so the cost scales with how many are named. Programs are assigned to the score's
sections in rotation, so a 30 second window with three sections renders three of
them however many are named; the surplus is reported rather than silently
dropped:

```sh
ss audition score.yaml --source source.mkv --audio track.flac \
     --format 1080p30 --source-host pi@videopi --seconds 30 --start 60 \
     --cut-programs Derez,Lorenz,Scramble,Sfumato --takes takes/ -o audition.mp4
```

A real 30 second excerpt costs about a minute, most of it the load:

```
09:56:57 audition: 10.0s from 20.0s
09:57:03 timecoded 300 frames into timecoded.mkv at 1920x1080
09:57:03 upload (clip=timecoded.mkv)
09:57:04 upload done in 1s: bytes=948134
09:57:05 load (program=Passthru)
09:57:17 load done in 12s
09:57:18 pass (program=Passthru seconds=10.0)
09:57:32 pass done in 14s: writes=9
09:57:34 audition.take.mp4: 185 frames: 184 picture, 1 uniform; luma range 235, motion +0.71, mean luma 0.383
09:57:34 picture starts 1.20s into the take
09:57:34 master (seconds=10.0 fades=1/1 out=audition.mp4)
09:57:37 master done in 3s
09:57:37 audition.mp4: 10.0s mastered from audition.take.mp4
audition.mp4
```

The take is longer than the pass because it opens on lead-in, and holds more
frames than 10 s at 30 implies because the card returns its own rate. What makes
the finished clip right is neither of those but the `picture starts` line.

Iterate here, not on the full render. A different `--seed`, `--style` or
`--density` at step 2 is seconds of CPU; a full render is most of an hour.

## 4. Render

```sh
ss render score.yaml --source source.mkv --audio track.flac \
     --format 1080p30 --source-host pi@videopi \
     --cut-programs Derez,Lorenz,Scramble,Sfumato,Kaledos,Folio,Isotherm,Howler \
     --takes takes/ --scratch timecoded.mkv \
     --fade 1.5 -o final.mp4
```

What happens, in order:

1. The source is conformed to the session raster, a gray-code frame index is burnt
   into a croppable edge strip, and the result is written to `--scratch` **once**;
   every pass replays that same file.
2. Per program: the scratch is pushed over ssh into the Pi's `/dev/shm` and played
   from there. Nothing is copied to the Pi by hand — the source stays on the
   workstation, and the timecoded clip is the only thing that crosses, once per
   pass. It has to fit in the Pi's tmpfs, which is half its RAM.
3. Then the HDMI link is dropped, the program loaded, and the pass waits
   for the picture to come back before recording — a load blacks the output for
   12 to 19 seconds on this rig, and the take must not start inside that. The wait
   polls the device's HDMI transmitter over serial until it drives the session
   format, then records a short probe that must be mostly picture, all within a
   40 s budget; exceeding it fails the pass rather than recording black. The
   transmitter is checked again as the recording starts and once it ends.
4. Recording starts, and a few seconds later the clip is played a second time,
   from its first frame, for the take itself. The first play is only what gives
   the liveness probe something moving to judge; it is over before the load
   finishes, and a take recorded against it would be its last frame held still.
5. The Pi decodes and blits the clip at rate while ffmpeg records the card and the
   host writes only the parameters that are due. Each program is driven by its own
   evolved layer; the take covers the whole timeline.
6. Each pass lands in `--takes` as `take-<program>.mkv` (an `author/Name` program
   becomes `take-author_Name-<digest>.mkv`, the archive's name for it), and they are cut together
   on the score's sections, in rotation, into `final.take.mp4`.
7. The take opens on lead-in the picture has not reached yet, so its start is
   read back out of the timecode strip rather than assumed — that is what the
   strip is for — and reported as `picture starts N.NNs into the take`. The Pi's
   player takes a few frames to reach rate; the lag held by a run of consecutive
   frames counting up one per frame is what the take is cut against, and the
   fade covers the ramp. The held lead-in frame and scattered misreads form no
   such run, so they cannot set it. A program that
   moves the picture geometrically can destroy the strip; that take is cut at the
   median start of the run's other takes (the lag is the Pi's and the capture's,
   which every pass shares) and the log says so:
   `Pinwheel: timecode strip did not survive; using the run's median picture start 1.90s`.
   If no take in the run kept its strip, the render stops rather than cut from
   the take's first frame.
8. Mastering: cut from where the picture starts, black out the timecode strip
   (cropped off the bottom edge and padded back, so the frame keeps its size and
   nothing is resampled), trim to the length both inputs cover, mux the track, fade the picture up from black and the audio up from
   silence over `--fade` seconds, and the same back down at the end.

The capture card returns frames at its own rate — this one gives 60 for a
`1080p30` session — so a take holds more frames than the session's rate implies.
Nothing downstream counts frames: alignment is by the timecode strip, and the
finished clip is cut by time.

`--fade-in` and `--fade-out` override each edge separately; `--fade 0` renders
hard cuts at both ends. On a clip shorter than two fades, each is clamped to half
the clip so they meet rather than overlap. `--no-master` stops at the raw take,
and `--take` puts it somewhere other than beside the output.

`--reuse-takes` cuts from any `take-<program>.mkv` already in `--takes` instead
of rendering it again (`<program>: reusing take <path>`), so a render can be
re-cut, re-mastered or resumed after an interruption; with every take present the
rig is never opened. Without `--cut-programs` it reuses `--take`.

Writes are sent ahead of their cue by the device's write latency plus a 15 ms
bias, so changes land on the beat rather than behind it; the default latency is
one frame of `--format`, `--latency-ms` overrides it and `--measure-latency`
measures it per program. `--cc-budget` caps parameter writes per second for both
the planner and the session's pacing; writes due closer together than the budget
allows are brought forward into their own slots, never sent late. See
[hardware.md](hardware.md#midi-write-timing) for the measured figures and how to
re-derive them with `device doctor --latency`.

Without `--cut-programs` the render is a single pass through the score's first
layer: one program for the whole piece, and one pass of rig time instead of one
per program.

## 5. What runs without the rig

Everything up to the first pass does, which is what makes the loop above cheap to
rehearse: `analyze`, `compose`, and the timecoding at the head of a render all
run on the workstation alone. `device list` needs only the USB link; `device
ping` and `programs` need the CDC shell as well. From the upload onwards the Pi
and the capture card are load-bearing, and a missing one is reported as the stage
that failed:

```
09:19:48 timecoded 300 frames into timecoded.mkv at 1920x1080
09:19:48 upload (clip=timecoded.mkv)
09:19:48 upload failed after 0s: pi@videopi: `cat > /dev/shm/syncsummoner-clip.mkv` exited 255: pi@videopi: Permission denied (publickey).
```

## 6. Check what came out

```sh
ss render --help    # every flag above

docker run --rm -v "$PWD:/work" --entrypoint /venv/bin/python anarkiwi/syncsummoner \
  -m syncsummoner.aesthetics score final.mp4   # perceptual metrics, no device needed
```

The photosensitive-seizure veto is a hard constraint rather than a score: a take
with offending windows is temporally low-passed, not discarded.

## Time it takes

A cut render says what it is about to cost before it starts the first pass —
`N cuts over M programs: M passes of <length> plus a load and relock each, <total>
of capture` — and every stage after that reports its own elapsed time.

For a three minute piece at `1080p30`, per program: the load and relock, measured
at 12 to 19 s, then three minutes of real-time capture. Eight programs is
therefore around half an hour of rig time, on top of one timecoded-source build.
Nothing about a real-time capture makes that faster; what makes it cheap is not
doing it twice, which is what the audition is for.

| Step | Where | Cost |
| --- | --- | --- |
| `probe archive` + `refit` | rig, then CPU | hours, once per firmware |
| `analyze` / `compose` | CPU | seconds to a minute |
| `audition` | rig | one load plus the excerpt |
| `render` | rig | (load + relock + duration) per program |
| mastering | CPU | one encode |

## When it goes wrong

`syncsummoner device doctor` checks most of these at once.

| Symptom | Cause |
| --- | --- |
| `the transmitter is not driving 1080p30 within 40.0s of the load` | the device's HDMI output never came back at the session format after the load; the error carries the last `video status`: `hdmi_tx` `vic` 0 or `de` `000000` is a stopped transmitter, a different `output.timing` is a format mismatch (`ss device doctor`, `device.format`) |
| `the firmware reports no transmitter state` | `video status` has no `hdmi_tx` or `output.hdmi_connected`, so signal cannot be confirmed and nothing is recorded; update pyvmancer or the firmware |
| `the transmitter is not driving ... as the pass began` / `as the pass ended` (or `sweep`) | the output dropped around a recording; the pass is failed rather than kept, and an archive goes on to its passthrough check |
| `the program outputs no picture` | the transmitter is up but the load probe was mostly uniform fields: the program outputs a constant, e.g. video black at 16. Brightness and colour play no part, so dark or black-and-white footage is picture |
| `transmitting 1080p30 but nothing captured` | the device sends video but the card delivered no frames: check the HDMI cable to the card and that nothing else holds `/dev/video0` |
| take ends `UNUSABLE: frozen` | the take holds one picture: `motion` sits at -0.50, which is what codec noise alone gives; the output froze (a drop) or the program ignores its input |
| take ends `UNUSABLE: no picture` | as for the load, over the whole take: check the program and `ss device ping` |
| recorder exits immediately | something else holds `/dev/video0`; only one process may |
| small flickering square in every take | `fbcon` cursor blink on the Pi; disable it, `chvt` does not fix it |
| parameters have no effect | MIDI CC is an offset onto the physical knob, so park the panel first |
| take looks like plain passthrough | a corrupted MIDI CC map: a slot learned P1's CC and follows it. `syncsummoner device doctor` (`midi.cc_map`) or `vmancer cc-map` shows the map; every load rewrites it, and `vmancer cc-map --restore --yes` repairs it by hand. See [hardware.md](hardware.md#the-midi-cc-map-must-be-checked) |
| output starts mid-clip, or holds one frame | the picture's start was not found; check the `picture starts` line and that the strip survives the program; a `timecode strip did not survive` warning means that take was cut at the run's median start instead |
| `timecode strip did not survive, so the picture start is unknown` | no take kept its strip; check the program, or that the pass recorded the clip at all |
| pass dies uploading the clip | `/dev/shm` on the Pi is half its RAM; a long scratch at `crf 14` will not fit |
| `Host key verification failed` | the mounted `known_hosts` has no entry for the Pi; `BatchMode` will not prompt for one |
| `Permission denied (publickey)` | the mounted key is not the one the Pi authorizes |
| `sudo: a password is required` | link blanking needs passwordless sudo on the Pi |
| `Bad owner or permissions on ~/.ssh/config` | the container is running as root against your ssh config; use the `--user` form above |
| `Permission denied` writing the scratch | files left by an earlier run as root; `sudo chown -R "$USER" .` in the work directory |
| `cannot cache function ... no locator available` | numba has nowhere to write librosa's jit cache; the image sets `NUMBA_CACHE_DIR`, so rebuild it, or pass `-e NUMBA_CACHE_DIR=/tmp/numba-cache` |
| `more programs than sections, N never comes up` | cuts follow the score's sections; audition a longer span, or name fewer programs |
| black output; `device.input` fails, or a pass fails with `InputRouteError` | the device fell back to another input (Vid Route Mode Auto after the source link dropped); select `hdmi` on the front panel, or set Vid Route Mode to HDMI |
| `video status` holds `NTSC` (or another timing) with `overridden` while the source is fine | a front panel override: set SETTINGS > Vid Route Mode to Auto, SYSTEM > Pref. Timing to the session format, power-cycle |
| recorder cannot open the session format; the card lists only SD YUYV/NV12 | the card enumerated on USB 2; plug it straight into a USB 3 port with a USB 3 cable |
| device reports no input lock, Pi `fb0/blank` reads 4 | the Pi framebuffer was left powered down; `syncsummoner device doctor --fix` |
| `error: no serial link is open` | the CDC tty was not passed in, or `pyvmancer` is installed without its serial extra |

What the rig was measured to do, and why each of these follows from it, is in
[hardware.md](hardware.md).
