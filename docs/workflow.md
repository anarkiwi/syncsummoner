# Example render workflow

One source clip and one audio track in, one finished piece out: the effects are
chosen from what the track does, checked on a 30 second audition, then committed
to a full render that fades up from black and silence and back down again.

Everything below runs through `docker run`, on the rig described in
[rig.md](rig.md) and measured in [hardware.md](hardware.md).

## The rig

```
                    +--------------------------------------+
                    |  workstation (runs syncsummoner)     |
                    |                                      |
                    |   /dev/ttyACM0   /dev/snd/midiC*D0   |
                    |   /dev/video0    ssh                 |
                    +--+--------+---------+-----------+----+
                       |        |         |           |
        USB-C (serial) |        | USB     | USB 3     | ethernet
        CDC 16d0:14db  |        | MIDI    | UVC       | ssh, BatchMode
                       |        |         |           |
                       v        v         |           v
             +---------+--------+---+     |    +------+-----------------+
             |     LZX Videomancer  |     |    | Raspberry Pi 4B        |
             |                      |     |    | "videopi"              |
             |  HDMI IN  <----------+-----+----+ HDMI0  (/dev/fb0)      |
             |                      |  HDMI    |                        |
             |  6 knobs, 5 toggles, |  1080p30 | no compositor, DRM free|
             |  crossfader (P12)    |          +------------------------+
             |                      |
             |  HDMI OUT  ----------+---> +--------------------------+
             |                      |     | AVerMedia Live Gamer     |
             |  CVBS/S-video (alt)  |     | Ultra 2.1  (UVC)         |
             +----------------------+     +-----------+--------------+
                                                      |
                                       USB 3 to workstation /dev/video0
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

```sh
docker build --target runtime -t syncsummoner .
```

```sh
ss() {
  docker run --rm -it --network host \
    --device /dev/video0 --device /dev/ttyACM0 --device /dev/snd \
    -v "$PWD:/work" -v "$HOME/.ssh:/root/.ssh:ro" \
    syncsummoner "$@"
}
```

| Flag | Why |
| --- | --- |
| `--device /dev/ttyACM0` | the CDC shell: programs, video status, settings |
| `--device /dev/snd` | ALSA rawmidi, which is how parameters are driven |
| `--device /dev/video0` | the capture card; ffmpeg inside the container records it |
| `--network host` | reaches `videopi` by the name the host resolves it with |
| `-v $HOME/.ssh:/root/.ssh:ro` | ssh runs `BatchMode=yes`, so the key must already work unattended |
| `-v $PWD:/work` | media, profiles and outputs; `/work` is the image's working directory |

Check the rig answers before anything long starts:

```sh
ss device list          # serial, rawmidi node, tty
ss device ping          # firmware, and the video status the device reports
ss device programs      # what can actually be composed for
ssh pi@videopi 'echo 0 | sudo tee /sys/class/graphics/fbcon/cursor_blink'
```

That last line matters: a blinking console cursor inverts its cell on top of the
framebuffer and lands in every take. The other pre-flight gotchas — parking the
panel, one owner of `/dev/video0`, the capture card's "No Signal" splash — are in
[rig.md](rig.md#gotchas).

## Inputs

```
work/
  source.mkv      the clip to process   (any format; it is conformed and padded, never stretched)
  track.flac      the audio to drive it
  profiles/       fitted profiles, one YAML per program
```

The composer can only reach for a program it has a profile for, and only for
parameters that were measured to move something. With no profiles there is
nothing to compose. Build them once per firmware:

```sh
ss probe archive --out archive/ --source-host pi@videopi   # native frames, resumable, hours
ss probe refit --archive archive/ --out profiles/          # profiles from the archive, no device
ss profile show --profiles profiles/
```

## 1. Hear what the composer hears

```sh
ss analyze source.mkv --audio track.flac -o features.json
```

Tempo, beats, downbeats, the self-similarity segmentation into sections, onset
surprisal, and per-frame video statistics. Worth a look when a composition comes
out wrong: if the sections are nonsense the score will be too, and that is an
analysis problem, not a rig problem.

## 2. Compose — this is where the effects are chosen

```sh
ss compose source.mkv --audio track.flac --profiles profiles/ \
     --style glitchy --seed 7 --passes 8 --density 0.5 -o score.yaml
```

For every program with a usable profile, the planner evolves a layer of gestures
anchored to the track's beats, downbeats and section edges, scores it against the
measured response of that program, and keeps the best. `--passes 8` keeps the
eight strongest programs, one capture pass each; `--style` biases which gestures
are drawn on; `--seed` makes the whole search reproducible.

The score's duration is the length both inputs cover — the shorter of clip and
track — and the sections are clipped to it. Everything downstream inherits that,
so the audition, the render and the finished file are all the same length by
construction. What came out:

```sh
grep -E '^(duration|bpm)' score.yaml
```

## 3. Audition before committing the rig

```sh
ss audition score.yaml --source source.mkv --audio track.flac \
     --format 1080p30 --source-host pi@videopi \
     --seconds 30 --start 60 -o audition.mp4
```

The score is windowed to those 30 seconds, the footage and the track are both
seeked to where the excerpt starts, and it renders as one real-time pass through
the score's strongest program, mastered exactly as the full render will be. Cost
is one program load plus 30 seconds.

To hear the cut as well as the effect, audition across programs — one pass each,
so the cost scales with how many are named:

```sh
ss audition score.yaml --source source.mkv --audio track.flac \
     --format 1080p30 --source-host pi@videopi --seconds 30 --start 60 \
     --cut-programs Derez,Lorenz,Scramble,Sfumato --takes takes/ -o audition.mp4
```

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
2. Per program: the HDMI link is dropped, the program loaded, and the pass waits
   for the picture to come back before recording — a load blacks the output for
   12 to 19 seconds on this rig, and the take must not start inside that. The wait
   is a liveness probe judged exactly as a take is, with a 40 s budget; exceeding
   it fails the pass rather than recording black.
3. The Pi plays the clip at rate while ffmpeg records the card and the host writes
   only the parameters that are due. Each program is driven by its own evolved
   layer; the take covers the whole timeline.
4. Each pass lands in `--takes` as `take-<program>.mkv`, and they are cut together
   on the score's sections, in rotation, into `final.take.mp4`.
5. Mastering: trim to the length both inputs cover, mux the track, fade the
   picture up from black and the audio up from silence over `--fade` seconds, and
   the same back down at the end.

`--fade-in` and `--fade-out` override each edge separately; `--fade 0` renders
hard cuts at both ends. On a clip shorter than two fades, each is clamped to half
the clip so they meet rather than overlap. `--no-master` stops at the raw take,
and `--take` puts it somewhere other than beside the output.

Without `--cut-programs` the render is a single pass through the score's first
layer: one program for the whole piece, and one pass of rig time instead of one
per program.

## 5. Check what came out

```sh
ss render --help    # every flag above

docker run --rm -v "$PWD:/work" --entrypoint /venv/bin/python syncsummoner \
  -m syncsummoner.aesthetics score final.mp4   # perceptual metrics, no device needed
```

The photosensitive-seizure veto is a hard constraint rather than a score: a take
with offending windows is temporally low-passed, not discarded.

## Time it takes

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

| Symptom | Cause |
| --- | --- |
| take is uniform grey and "usable" | the card's "No Signal" splash, not the rig; check `ss device ping` |
| recorder exits immediately | something else holds `/dev/video0`; only one process may |
| small flickering square in every take | `fbcon` cursor blink on the Pi; disable it, `chvt` does not fix it |
| parameters have no effect | MIDI CC is an offset onto the physical knob, so park the panel first |
| picture never returns after a load | relock; the settle budget is 40 s, and `video status` is advisory |

Deeper detail on the video path and the intermittent input fault is in
[signal-loss.md](signal-loss.md).
