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
   | KMS player, 1080p30       |     HDMI       | HDMI IN                    |
   | no compositor, DRM free   |===============>|                            |
   | plays the timecoded clip  |  source clip   | 6 knobs, 5 toggles,        |
   +-------------+-------------+                | fader (P12)                |
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
| ethernet | videopi | workstation LAN | ssh: playout, and dropping the link |

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
has to be in the mounted `known_hosts` before the first run), and passwordless
sudo: the player raises its real-time priority with `sudo -n prlimit`, and
dropping the HDMI link with no player running writes
`/sys/class/graphics/fb0/blank`:

```sh
ssh pi@videopi 'echo ok'      # must succeed with no prompt
ssh pi@videopi 'sudo -n true' # must not ask for a password; prlimit and blanking are sudo
```

Check the rig answers before anything long starts:

```sh
ss device list          # serial, rawmidi node, tty, usbfs
ss device ping          # firmware, and the video status the device reports
ss device programs      # what can actually be composed for
ssh pi@videopi 'echo 0 | sudo tee /sys/class/graphics/fbcon/cursor_blink'
ss device doctor --latency  # rig checks, plus MIDI CC latency and backlog at the session's frame rate
```

The cursor line matters: between plays fbdev shows the framebuffer, and a
blinking console cursor inverts its cell on top of it. The other pre-flight
gotchas — parking the panel, one owner of `/dev/video0`, a capture card with no
signal state of its own — are in [rig.md](rig.md#gotchas).

### Check the frame clock

A render attributes every clip frame to one capture exactly, which holds only if
the Pi's player, the device and the card keep one clock
([design.md](design.md#10-locked-frame-clock-and-exact-attribution)). Check that
end to end before a long render:

```sh
ss lockcheck source.mkv --seconds 30 --source-host pi@videopi              # through Passthru
ss lockcheck source.mkv --seconds 30 --program Pinwheel --source-host pi@videopi
```

It timecodes the footage into `--scratch` (`--prepared` plays one already
timecoded), plays it with the render's lead-in (`--lead-in` overrides it),
records `--take`, and prints one line, `LOCKED` or `NOT LOCKED`, with the Pi's
flips and missed vblanks, the card's lost frames, gaps and off-cadence frame
changes, strip agreement and, through Passthru, torn captures. It exits 1 unless
locked; `--judge` re-judges a recorded `--take` without playing.

A full clip through `Pinwheel`, loaded in the lead-in, with `--prepared`:

```
23:12:14 upload (clip=clip.mkv)
23:13:46 upload done in 1.5 min: bytes=765352701
23:13:46 passthru (program=Passthru)
23:13:49 Passthru: cc-map: assigned P1..P12 -> CC0..11 (was: ok)
23:13:50 passthru done in 4s
23:13:50 take (program=Pinwheel seconds=369.8)
23:13:58 Pinwheel: cc-map: assigned P1..P12 -> CC0..11 (was: ok)
23:20:10 take done in 6.3 min: load_s=3.18
23:21:02 pinwheel.mkv: 10493/10493 clip frames attributed at 2 captures a frame, 3 runs, clip in captures 475..22640 (repeats on cadence 100.0%, timestamps agree 100.0%, worst 0.00 of 16.67 ms), strips agree 21991/22257, 0 frames lost by the driver
23:21:02 pinwheel.mkv: LOCKED: 10493/10493 clip frames attributed; Pi 11068 flips, 0 missed vblanks, 0 off cadence; card 0 lost frames, 0 gaps and 0 frame changes off cadence in the clip; strips 20986/20986 agree; 0/0 torn
```

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

A refit decodes each archive once, in order: one ffmpeg feeds the analysis-width
frames to the metrics and the few native dwell captures to the behaviour fits. Each
program runs whole in a worker process with single-threaded decode and OpenCV, one
worker per core by default (`--jobs` caps it), so its cost is the FFV1 decode plus
the per-pair optical flow, spread evenly over the cores.

The codeframe loop says what each parameter does to a test signal. To also
measure what each keeps of real footage, archive against a clip:

```sh
ss probe archive --footage source.mkv --out archive/ --source-host pi@videopi
ss probe archive --footage source.mkv --footage-start 30 --footage-seconds 120 --out archive/ ...
```

The footage is conformed to the session raster, timecoded, and written to
`archive/stimulus.mkv`, with `archive/stimulus.json` (source, excerpt, rate,
strip, SHA-256) written last as its commit marker; the Pi plays it on a loop for
the whole run. A resume reuses the clip when the record names the same excerpt
and the clip still hashes the same, and skips a program only when its archive
was recorded against that same clip; one recorded against the codeframe loop or
another clip is recorded again. The clip has to fit the Pi's tmpfs.

A refit of a footage archive adds three survival curves to each parameter:
`si_ratio` and `ti_ratio`, the share of the source's spatial and temporal
information kept, and `motion_follow`, how far the output's motion follows the
source's. Each program logs `footage clock from strip`, `motion` or `none`; with
`none` it has no survival. Codeframe archives still refit as before, without
survival, and the loop behaviour fits run only on them. See
[design.md](design.md#9-footage-and-survival).

`--decode` picks where the archive is decoded: `cpu`, `vulkan` (FFV1 in compute
shaders, ffmpeg 8 and later), or `auto`, the default. `auto` decodes the first
program's opening frames both ways and takes Vulkan only when the frames are
byte for byte the CPU's and the GPU's per-stream rate is at least the CPU's, since
every worker decodes one stream; otherwise it stays on the CPU and logs why, so
the same command runs on any host. `vulkan` forces the GPU, and fails rather than
fall back. Vulkan decode is bound by the archive's slice count (see
[hardware.md](hardware.md#archive-decoding-on-a-gpu-is-slice-bound)), so the
archives recorded so far decode faster on the CPU and `auto` keeps them there.

The GPU image is `Dockerfile.gpu`: NVIDIA's base, with a pinned ffmpeg built with
the Vulkan FFV1 hwaccel. The host needs the NVIDIA driver and
nvidia-container-toolkit, and the GPU is passed through by its CDI name:

```sh
docker build -f Dockerfile.gpu -t syncsummoner:gpu .
docker run --rm --device nvidia.com/gpu=all --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/work" syncsummoner:gpu probe refit --archive archive/ --out profiles/ --decode auto
docker run --rm --device nvidia.com/gpu=all -v "$PWD:/work" syncsummoner:gpu \
  probe decoders --archive archive/ --digests 3 --streams 1,4 --frames 120
```

The CDI spec (`nvidia-ctk cdi generate`, under `/etc/cdi` or `/var/run/cdi`)
mounts the driver's Vulkan ICD manifest, its implicit layers and
`/dev/nvidia-modeset`. `--gpus all` does not: through Docker's built-in GPU hook
the container gets the driver libraries and no ICD, and Vulkan fails with
`VK_ERROR_INCOMPATIBLE_DRIVER`. Under CDI the spec decides what is mounted; the
image still sets `NVIDIA_DRIVER_CAPABILITIES` to include `graphics`, which the
toolkit's own runtime reads in its other modes. `probe decoders` digests each program's
whole decoded stream on every backend (exit status 1 on any difference) and times
concurrent decoders.

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
refit carries it into every parameter's name and native range. An archive
recorded before that is back-filled from the device without re-archiving: each
program is loaded once and asked, a few seconds apiece, resumably.

```sh
ss probe names --archive archive/      # program info into each archive, resumable
ss probe refit --archive archive/ --out profiles/
```

What the composer may do with a parameter is its `kind`, and that comes from the
program's own declaration, never from its slot or its label: the `control_mode` and
`value_labels` in the config payload of the program's `.vmprog`. The declaration is
found by device name in a copy of the card's `programs/` directory
(`captures/card/programs`, or `$SYNCSUMMONER_CARD`) or, for a firmware built-in,
inside the firmware `.uf2` image (`captures/firmware`, or `$SYNCSUMMONER_FIRMWARE`).
A program declared in neither stops with `UndeclaredProgramError`, which names both
places; copy the card or the firmware image there and run again.

| Kind | Declared as | Composer |
| --- | --- | --- |
| `boolean` | boolean control mode, or two value labels | only pulsed by `punch`, and released |
| `quantized` | more than two value labels, or a stepped control mode | any other gesture, over its declared levels |
| `continuous` | any other control mode | any other gesture |
| `unused` | not declared | never driven |

Every pass starts from the program's declared initial values, and every parameter
a gesture leaves alone stays at its own. Which of the declared parameters can
reach the picture at a given setting is a different question, answered exactly
from the program's HDL by `ss program show` and `ss program step`
([program-model.md](program-model.md)):

```sh
ss program list                                  # every program the card copy and firmware declare
ss program show Pinwheel                         # register bits, levels, live bits and gates at the initial values
ss program step Pinwheel 3 0 512 --set 12=1023   # can this one step change the picture
```

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

With profiles refit from footage, the planner predicts, bar by bar, whether each
program keeps the clip's picture and motion, using the clip's own SI and TI, so
pass the clip being rendered. A bar predicted dead outside a `destroy` section,
or past the first bar of a dead run inside one, ranks a candidate below any that
has fewer, whatever its score. A program whose layer still breaks that budget
logs `N bars predicted dead outside the destruction budget`; one without survival
logs `no survival measured, so its picture is unverified; refit from footage`.
Sections are split at the track's phrase length, which the score records as
`phrase_bars`.

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

Before spending rig time on it, see what the score will actually drive:

```sh
ss score show score.yaml --profiles profiles/
```

```
layer 2: Breakout
   P1  Ball Spd         continuous   23 writes     0..1023 ends 898   hold x1, hysteresis_loop x3
   ...
   P7  Serve            boolean       6 writes     0..1023 ends 0     punch x3
   P11 Bypass           boolean    undriven
layer 4: Pinwheel
   ...
   P12 Chroma Crush     continuous   84 writes   416..748  ends 748   cliff_cross x2
```

Every layer lists all twelve parameters by name with their declared kind, how many
writes the render will send, the range they cover, where they end, and which
gestures asked for them; that is the same plan the render plays. A line marked `!`
is a toggle left latched on at the end of the pass; a layer marked `!` has no
profile or drives nothing, and gestures that render nothing are listed under their
layer. It exits 1 when anything is flagged.

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

An excerpt costs a program load, the lead-in and the excerpt. The program is
loaded before playout and held dry through the lead-in by its bypass switch (the
declared switch that, turned on, leaves every other control dead in its HDL),
toggled to time the device; one with none loads inside a Passthru
lead-in ([evidence.md](evidence.md)). Each take logs the measured
CC-to-picture latency, its attribution summary, the evidence report (every write's
landing frame, readback and receipt counts, and one line per parameter: acted upon
or not, and why), then the take's picture report. A take whose capture driver lost
frames is recaptured once (`TakeFaultError`).

The take is longer than the excerpt because it opens on the lead-in, and holds
more captures than frames because the card runs its own cadence
([hardware.md](hardware.md#the-cards-cadence)). What makes the finished clip right
is the attribution line: every clip frame has a capture, and the clip's run agrees
on its cadence. `ss evidence takes/take-Derez.mkv --source timecoded.mkv` prints a
take's evidence again from its sidecars; with `--predict` it sets each parameter's
verdict against the program model's prediction for the take's steps, and flags a
parameter predicted dead in every step that the picture shows moved as a
contradiction ([program-model.md](program-model.md)).

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
   from there by the locked player ([rig.md](rig.md#playout)). Nothing is copied
   to the Pi by hand — the source stays on the workstation, and the timecoded clip
   is the only thing that crosses, once per pass. It has to fit in the Pi's tmpfs,
   which is half its RAM.
3. Passthru is loaded, recording starts, and the player starts the clip, which
   opens on a strip-coded lead-in (`lead_in_s`, 20 s). The program is loaded
   inside the lead-in with the HDMI link dropped, and the pass waits for the
   transmitter to drive the session format within a 40 s budget; a load not done
   before clip frame 0 is a `LeadInError`. The transmitter is checked again once
   the take ends.
4. From clip frame 0 the host writes only the parameters that are due while the
   Pi flips the clip on its vblanks and the card's captures are stored untouched,
   each with its sequence, timestamp and digest in `<take>.v4l2`. Each program is
   driven by its own evolved layer; the take covers the whole timeline. When the
   player stops, its flip log is saved as `<take>.lock.npz`.
5. The take is attributed exactly
   ([design.md](design.md#10-locked-frame-clock-and-exact-attribution)): per clip
   frame, the capture nearest its centre, saved as `<take>.picked.npy` and logged
   as one line (`take-Pinwheel.mkv: 5400/5400 clip frames attributed at 2 captures
   a frame, ...`). A take that cannot be attributed frame for frame is a
   `LockError`, and one whose clip frame 0 reached the card before the load ended
   a `LeadInError`.
6. Each pass lands in `--takes` as `take-<program>.mkv` (an `author/Name` program
   becomes `take-author_Name-<digest>.mkv`, the archive's name for it), and they are spliced
   together on the score's sections, in rotation, into `final.take.mp4`.
7. Before the takes are spliced together, each is checked against the timecoded
   clip it was fed, frame for frame by its attribution, and every bar is judged `flat`, `frozen` or `unfollowed` on what it kept of the
   same bar of the clip. A dead run inside a cut is covered from another
   program's take alive across it (`Derez:   42.10-  46.20s frozen, covered by
   Lorenz`); a lone dead bar inside a `destroy` section stays. A run no take
   covers stops the render with `DeadTakeError: dead picture no take can cover:`
   and the program and time range, before anything is mastered.
8. The splice writes one frame per clip frame at the clip's rate: frame `n` is the
   capture its cut's take shows clip frame `n` in. Every take is decoded once,
   forward, at full size, all at once, and nothing is seeked, so no cut is a frame
   early or late. A clip frame the chosen take has no capture of stops the splice
   with `SpliceError: <program>: no capture shows clip frames <from>-<to>s`.
9. Mastering: black out the timecode strip
   (cropped off the bottom edge and padded back, so the frame keeps its size and
   nothing is resampled), trim to the length both inputs cover, mux the track, fade the picture up from black and the audio up from
   silence over `--fade` seconds, and the same back down at the end.

The capture card returns frames at its own rate
([hardware.md](hardware.md#the-cards-cadence)), so a take holds more frames than
the session's rate implies. Attribution picks one capture per clip frame, so the
splice is at the clip's rate.

`--fade-in` and `--fade-out` override each edge separately; `--fade 0` renders
hard cuts at both ends. On a clip shorter than two fades, each is clamped to half
the clip so they meet rather than overlap. `--no-master` stops at the raw take,
and `--take` puts it somewhere other than beside the output.

The same check runs on its own, against takes already rendered or a spliced
master, with no rig. It logs each take's alignment and every dead span, then the
covered cut plan, and exits 1 when any span is uncovered. `--splice OUT` then
splices the covered plan from the takes (`--as-planned` splices the plan as
rendered, dead spans and all, which is what to check a master check against):

```sh
ss check score.yaml --source timecoded.mkv --takes takes/ \
     --cut-programs Derez,Lorenz,Scramble,Sfumato,Kaledos,Folio,Isotherm,Howler
ss check score.yaml --source timecoded.mkv --takes takes/ \
     --cut-programs Derez,Lorenz,Scramble,Sfumato,Kaledos,Folio,Isotherm,Howler \
     --splice final.take.mp4
ss check score.yaml --source timecoded.mkv --master final.take.mp4
```

`--source` is the render's `--scratch`. Each take is checked on its
`<take>.picked.npy` (`Pinwheel: aligned by given`); a take without one, recorded
before the locked player, has its clock fitted from its strip and motion, with a
warning. A spliced master is frame for frame the clip's, so it is checked on the
identity clock: master frame `n` is clip frame `n`.

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

For a three minute piece at `1080p30`, per program: a Passthru load, the 20 s
lead-in the program loads inside, then three minutes of real-time capture. Eight programs is
therefore around half an hour of rig time, on top of one timecoded-source build.
Nothing about a real-time capture makes that faster; what makes it cheap is not
doing it twice, which is what the audition is for.

| Step | Where | Cost |
| --- | --- | --- |
| `probe archive` | rig | hours, once per firmware |
| `refit` | CPU | minutes, decode-bound |
| `check` | CPU | one decode per take |
| `analyze` / `compose` | CPU | seconds to a minute |
| `audition` | rig | one load plus the excerpt |
| `render` | rig | (Passthru load + lead-in + duration) per program |
| mastering | CPU | one encode |

## When it goes wrong

`syncsummoner device doctor` checks most of these at once.

| Symptom | Cause |
| --- | --- |
| `the transmitter is not driving 1080p30 within 40.0s of the load` | the device's HDMI output never came back at the session format after the load; the error carries the last `video status`: `hdmi_tx` `vic` 0 or `de` `000000` is a stopped transmitter, a different `output.timing` is a format mismatch (`ss device doctor`, `device.format`) |
| `the firmware reports no transmitter state` | `video status` has no `hdmi_tx` or `output.hdmi_connected`, so signal cannot be confirmed and nothing is recorded; update pyvmancer or the firmware |
| `the transmitter is not driving ... as the pass began` / `as the pass ended` (or `sweep`) | the output dropped around a recording; the pass is failed rather than kept, and an archive goes on to its passthrough check |
| output black everywhere, every program dark, Passthru included, while `video status` reads `hdmi` locked and transmitting | check front panel SETTINGS > Vid Route Mode: under any route but HDMI In (Dual In, for one) the processor is fed nothing. `device doctor` fails `video.route`; render, audition and archive check it before the run and after every load and stop with `RouteError` naming the value read. Set HDMI In, or pass `--route` for a rig meant to run on another route |
| `the program outputs no picture` | the transmitter is up and the take was mostly uniform fields, while a Passthru control on the same rig showed a picture: the program outputs a constant, e.g. video black at 16, and was reloaded and retried once before this. Brightness and colour play no part, so dark or black-and-white footage is picture |
| `RigFaultError: ... the Passthru control failed` | a program showed no picture and so did Passthru: the rig, not the program, is dark. The error carries the route, the input lock and the transmitter `vic`/`de`; check Vid Route Mode, the source, then power-cycle |
| `the device does not offer ...` | the program listing, taken twice, lacked a named program; the error lists both listings' counts, the authors enumerated, a sample, and where a missing name was listed under another author. A listing pyvmancer finds inconsistent with the device's own count raises `ProgramListError`, and a reply spliced with another's `InterleavedReplyError`: close any other client on the serial port |
| `transmitting 1080p30 but nothing captured` | the device sends video but the card delivered no frames: check the HDMI cable to the card and that nothing else holds `/dev/video0` |
| take ends `UNUSABLE: frozen` | the take holds one picture: `motion` sits at -0.50, which is what codec noise alone gives; the output froze (a drop) or the program ignores its input |
| take ends `UNUSABLE: no picture` | as for the load, over the whole take: check the program and `ss device ping` |
| recorder exits immediately | something else holds `/dev/video0`; only one process may |
| small flickering square in every take | `fbcon` cursor blink on the Pi; disable it, `chvt` does not fix it |
| parameters have no effect | MIDI CC is an offset onto the physical knob, so park the panel first |
| take looks like plain passthrough | a corrupted MIDI CC map: a slot learned P1's CC and follows it. `syncsummoner device doctor` (`midi.cc_map`) or `vmancer cc-map` shows the map; every load rewrites it, and `vmancer cc-map --restore --yes` repairs it by hand. See [hardware.md](hardware.md#the-midi-cc-map-must-be-checked) |
| `SpliceError: <program>: no capture shows clip frames ...` | the take ends early, or a fitted take was never aligned; check its `aligned by` line in the take check |
| `DeadTakeError: dead picture no take can cover: <program>: <start>-<end>s <fault>, ...` | that span of that program's take is flat, frozen or not following the clip, and no other take is alive across it: render another program over it, or recompose with profiles refit from footage |
| `no survival measured, so its picture is unverified` at compose | the profile was fitted from a codeframe archive; archive with `--footage` and refit |
| `LostTimecodeError: ... no lock record` | the take was recorded before the locked player, so nothing attributes it exactly; render it again |
| `LockError: ...` | the take's captures do not attribute frame for frame; its attribution summary in the message says what failed (frames covered, runs, cadence, strip agreement, driver drops). `ss lockcheck` checks the rig the same way |
| `LeadInError: ...` | the program's load ran past the lead-in, or clip frame 0 reached the card before it ended; raise `RenderConfig.lead_in_s` (`lockcheck --program NAME --lead-in S` tries a value) |
| `PlayoutError: the Pi player faulted ...` | the player missed a vblank, ran out of decoded frames or met a mode that does not divide the clip rate; read its `status.json` and log in the Pi's `/dev/shm` player directory |
| `source.drm_free` fails | another DRM master holds `/dev/dri/card1` (a compositor or a stray player); stop the pid it names |
| pass dies uploading the clip | `/dev/shm` on the Pi is half its RAM; a long scratch at `crf 14` will not fit |
| `Host key verification failed` | the mounted `known_hosts` has no entry for the Pi; `BatchMode` will not prompt for one |
| `Permission denied (publickey)` | the mounted key is not the one the Pi authorizes |
| `sudo: a password is required` | the player's `prlimit` and link blanking need passwordless sudo on the Pi |
| `Bad owner or permissions on ~/.ssh/config` | the container is running as root against your ssh config; use the `--user` form above |
| `Permission denied` writing the scratch | files left by an earlier run as root; `sudo chown -R "$USER" .` in the work directory |
| `cannot cache function ... no locator available` | numba has nowhere to write librosa's jit cache; the image sets `NUMBA_CACHE_DIR`, so rebuild it, or pass `-e NUMBA_CACHE_DIR=/tmp/numba-cache` |
| `more programs than sections, N never comes up` | cuts follow the score's sections; audition a longer span, or name fewer programs |
| black output; `device.input` fails, or a pass fails with `InputRouteError` | the device fell back to another input (Vid Route Mode Auto after the source link dropped); select `hdmi` on the front panel, or set Vid Route Mode to HDMI |
| `video status` holds `NTSC` (or another timing) with `overridden` while the source is fine | a front panel override: set SETTINGS > Vid Route Mode to Auto, SYSTEM > Pref. Timing to the session format, power-cycle |
| recorder cannot open the session format; the card lists only SD YUYV/NV12 | the card enumerated on USB 2; plug it straight into a USB 3 port with a USB 3 cable |
| device reports no input lock, Pi `fb0/blank` reads 4 | the Pi framebuffer was left powered down; `syncsummoner device doctor --fix` |
| `error: no serial link is open` | the CDC tty was not passed in, or `pyvmancer` is installed without its serial extra |
| `... is held by another process` (`PortBusyError`; `device doctor` fails `device.port_free`) | the shell serves one client at a time and pyvmancer holds the tty exclusively: close the other client (`fuser /dev/ttyACM0` names it), e.g. a `vmancer shell` or another syncsummoner run. Archive stops at once rather than waiting out its health deadline |
| `ReplyShapeError: reply @... lacks fields ...` | a reply of the wrong shape for its command: a broken or misattributed reply, never read as a missing feature. Rerun; if it persists, check nothing else is talking to the serial port |

What the rig was measured to do, and why each of these follows from it, is in
[hardware.md](hardware.md).
