# Rig, wiring and bring-up

Companion to `hardware.md`, which records what the rig was measured to do. This
one records how it is put together and how to get it running.

## Signal path

```
Raspberry Pi 4B (videopi)
  KMS player, card1 HDMI-A-1, 1920x1080 at 30 Hz, vsync-locked
        |
        +-- HDMI ---------> Videomancer HDMI in (source hdmi)
                                    |
                                    +-- HDMI out --> AVerMedia Live Gamer Ultra 2.1
                                                     /dev/video0 on the host
Host
  USB MIDI   -> /dev/snd/midiC*D0    parameters, presets, clock, transport
  USB serial -> /dev/ttyACM*         programs, modulation, video, settings, fs
```

The Videomancer genlocks to its source and cannot convert. Pick one format for
the whole pipeline and never vary it; format is a session constant, not a
parameter.

## Permissions

MIDI needs read/write on `/dev/snd/midiC*D*`, usually granted by a desktop ACL.
Serial needs the `dialout` group or an ACL:

```sh
sudo usermod -aG dialout "$USER"          # persists, takes effect next login
sudo setfacl -m "u:$USER:rw" /dev/ttyACM0 # immediate, lost on replug
```

Capture needs read on `/dev/video0`, granted by the `video` group or an ACL.

An ACL is lost when the device re-enumerates, which happens on every replug and
on some firmware operations. Make it permanent instead:

```sh
sudo tee /etc/udev/rules.d/70-videomancer.rules <<'RULE'
SUBSYSTEM=="tty", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="14db", MODE="0660", GROUP="dialout"
RULE
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Playout

Every stimulus is played on the Pi by the native player (`native/kmsplay`), a
vsync-locked DRM/KMS player on libdrm: ffmpeg decodes the clip to raw frames, and the
player commits one atomic page flip per vblank, holding each frame `refresh / fps`
vblanks, logging every flip (`flips.bin`) and faulting on any missed vblank
(`status.json`, raised on the host as `PlayoutError`). The image's `kmsplay-build`
stage cross-builds it as one static aarch64 binary (`$SYNCSUMMONER_KMSPLAY`), which the
host pushes to `/dev/shm/syncsummoner/kmsplay` on every start; the Pi needs no
packages for it. Its pacing thread runs `SCHED_FIFO` with its memory locked, the
ceilings raised with `sudo -n prlimit` for the player's shell, so the Pi user needs
passwordless sudo. The host starts and stops it by pidfile in
`/dev/shm`. `ClipPlayer` plays an uploaded clip, once or looped; `LoopPlayer`
cycles uploaded raw frames through the same player, at 10 fps by default (a
whole number of vblanks at 30 Hz; a rate that does not divide the refresh is
refused). Nothing else may hold the DRM device (`source.drm_free`). Measured
properties are in [hardware.md](hardware.md#locked-playout).

While the player runs, `Link` drops the HDMI link by signalling it (`SIGUSR1`
down, `SIGUSR2` up); with no player running it blanks fbdev. A stopped player
leaves its last frame in `/dev/fb0`.

`syncsummoner lockcheck` checks the whole frame clock on the rig: it plays a
timecoded clip through Passthru (or `--program NAME`, loaded in the lead-in),
records it, and prints `LOCKED` or `NOT LOCKED` with the flips, missed vblanks,
card drops, gaps, picture changes off a frame start, strip agreement and, under Passthru,
torn captures; it exits 1 unless locked. `--prepared` plays `--scratch` as an
already timecoded clip, and `--judge` re-judges `--take` without playing.

## Bring-up checks

```sh
syncsummoner device doctor                # read-only: Pi, capture card, device
syncsummoner device doctor --loop         # also push bars through Passthru and capture them
syncsummoner device doctor --latency      # also time MIDI CC writes against serial readback
syncsummoner device doctor --fix          # unblank the Pi framebuffer, stop the cursor blink, recheck
```

Options: `--source-host pi@videopi`, `--format 1080p30`, `--capture /dev/video0`,
and with `--latency`, `--latency-params 1,2` and `--latency-rates 5,10,20,40,80`.
The JSON report goes to stdout, one line per finding to stderr, and the exit
status is 1 when anything fails. Each finding names its check and, on failure,
the fix:

| Check | What it reads |
| --- | --- |
| `ssh.reachable` | BatchMode ssh to the source host; host checks are skipped without it |
| `source.fb_unblanked` | `/sys/class/graphics/fb0/blank` is 0; 4 means no HDMI source for the device |
| `source.cursor_blink` | `/sys/class/graphics/fbcon/cursor_blink` is 0 |
| `source.fb_geometry` | framebuffer `virtual_size` and 16 bpp match `--format` |
| `source.drm_free` | nothing but the player holds `/dev/dri/card1`; another DRM master keeps the player off the display |
| `source.mode` | `kmsprint` CRTC size and rate match `--format`; skipped without `kmsprint` |
| `capture.usb_speed` | the card's video function is on a 5000 Mbit/s link |
| `capture.mode` | the card offers MJPEG at the session size (`ffmpeg -f v4l2 -list_formats all`) |
| `firmware.version` ... `video.output_connected` | pyvmancer's own device checks |
| `device.format` | the device's output timing is `--format` |
| `midi.cc_map` | every slot listens on the CC the MIDI controller in use sends it, and no CC drives two slots; the fix is `vmancer cc-map --restore --yes` |
| `midi.latency` | pyvmancer's single CC write to serial readback median is within two output frames plus the readback round trip |
| `midi.backlog` | paced ramps at each `--latency-rates` leave a sentinel no more than a frame behind a single write; the fix names the fastest clean rate |
| `loop.signal`, `loop.index`, `loop.bars` | a still shown by `LoopPlayer.show`: the HDMI transmitter drives `--format` and the capture is not a uniform field; the strip decodes to the pushed index; SMPTE bars read in order, so a red/blue swap fails |

`--latency` writes CC offsets to the measured parameters, then restores them
(`midi.restored` fails if they did not read back). By default it measures the
loaded program's first continuous knob, else P1; pyvmancer refuses toggles,
modulated parameters and ones with no CC headroom, and the refusal is reported
as a failed `midi.latency`. `--loop` changes state: it loads `Passthru` and
leaves its still on the Pi's display.
`Passthru` reports all twelve parameters as `Null`, so anything wrong in that
capture is the chain, not the program. The device alone is `vmancer doctor`
(pyvmancer 0.6). Over the serial shell, `doctor` reads what these do by hand:

```sh
vmancer shell "version"             # firmware; must match the profile archive
vmancer shell "fpga status"         # state, configured, program
vmancer shell "video status"        # source, timing, lock, output
```

## Gotchas

Only one process may hold `/dev/video0`. `syncsummoner.device.v4l2rec` records
the card, so nothing else may open it during a pass; a `Capture` handle left open
makes the recorder exit immediately. Reopening costs a relock
([hardware.md](hardware.md#capture-lock-is-slow)), so a pass records once rather
than per sample.

The capture card reports no signal state and synthesizes a "No Signal" slide
rather than going black, so whether there is signal comes from the Videomancer's
HDMI transmitter in `video status`, never from the picture; see
[hardware.md](hardware.md#signal-comes-from-the-videomancers-transmitter-the-card-exposes-none).

Park the panel before measuring. MIDI CC is an offset onto the physical knob
position, not an absolute set.

Program change costs a multi-second blackout. Batch every evaluation for a
program together.

Any VT on the Pi bleeds into every recording made while no player runs. fbdev
shows `/dev/fb0` then, which holds the player's last frame, and `fbcon`'s cursor
blink inverts its cell on top of it as a small flickering square wherever the
console's cursor sits. `chvt` to another VT does not fix this: systemd spawns a
fresh getty on whatever VT becomes active, with its own cursor at a new position.
Disable the blink outright before a session instead:

```sh
echo 0 | sudo tee /sys/class/graphics/fbcon/cursor_blink
```
