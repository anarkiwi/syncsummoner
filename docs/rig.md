# Rig, wiring and bring-up

Companion to `hardware.md`, which records what the rig was measured to do. This
one records how it is put together and how to get it running.

## Signal path

```
Raspberry Pi 4B (videopi)
  /dev/fb0, 720x576i PAL composite
        |
        +-- [RGB encoder] --> Videomancer analog in   (colour, needs calibration)
        |
        +-- direct CVBS ----> Videomancer analog in   (luma only, clean)
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

The Pi runs no compositor, so DRM is free and the framebuffer can be written
directly. Convert a frame and push it:

```sh
ffmpeg -i frame.png -pix_fmt rgb565le -s 720x576 -f rawvideo -y /tmp/f.raw
ssh pi@videopi 'cat > /dev/fb0' < /tmp/f.raw
```

`syncsummoner.device.playout` does the conversion with numpy and pushes over a
runner callable, so it is testable without the Pi attached.

A per-frame push costs about 1.1 s at 1920x1080 (4.1 MB of BGR565 at the 3.8
MB/s the framebuffer write sustains), so a 24-frame loop played that way is 26 s
per pass. `LoopPlayer` instead uploads the whole loop once into `/dev/shm` and
starts a detached Pi-side process that cycles it, addressed by pidfile:

```sh
python3 -c 'from syncsummoner.device import LoopPlayer; p = LoopPlayer(); print(p.is_running())'
```

Measured on this rig, 24 frames is 99.5 MB raw, uploading in 10.3 s; gzipped it
is 10.0 MB and 2.2 s, so the upload is compressed. The loop is invariant across
programs and setpoints, so that upload is paid once per run rather than per pass.

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
| `source.mode` | `kmsprint` CRTC size and rate match `--format`; skipped without `kmsprint` |
| `capture.usb_speed` | the card's video function is on a 5000 Mbit/s link |
| `capture.mode` | the card offers MJPEG at the session size (`ffmpeg -f v4l2 -list_formats all`) |
| `firmware.version` ... `video.output_connected` | pyvmancer's own device checks |
| `device.format` | the device's output timing is `--format` |
| `midi.cc_map` | every slot listens on the CC the MIDI controller in use sends it, and no CC drives two slots; the fix is `vmancer cc-map --restore --yes` |
| `midi.latency` | pyvmancer's single CC write to serial readback median is within two output frames plus the readback round trip |
| `midi.backlog` | paced ramps at each `--latency-rates` leave a sentinel no more than a frame behind a single write; the fix names the fastest clean rate |
| `loop.signal`, `loop.index`, `loop.bars` | not the No Signal splash; the strip decodes to the pushed index; SMPTE bars read in order, so a red/blue swap fails |

`--latency` writes CC offsets to the measured parameters, then restores them
(`midi.restored` fails if they did not read back). By default it measures the
loaded program's first continuous knob, else P1; pyvmancer refuses toggles,
modulated parameters and ones with no CC headroom, and the refusal is reported
as a failed `midi.latency`. `--loop` changes state: it loads `Passthru` and
overwrites the Pi framebuffer.
`Passthru` reports all twelve parameters as `Null`, so anything wrong in that
capture is the chain, not the program. The device alone is `vmancer doctor`
(pyvmancer 0.6). Over the serial shell, `doctor` reads what these do by hand:

```sh
vmancer shell "version"             # firmware; must match the profile archive
vmancer shell "fpga status"         # state, configured, program
vmancer shell "video status"        # source, timing, lock, output
```

## Gotchas

Only one process may hold `/dev/video0`. ffmpeg records the card, so nothing
else may open it during a pass; a `Capture` handle left open makes the recorder
exit immediately. Reopening also costs about 0.5 s of lock, so a pass records
once rather than per sample.

The capture card synthesizes a "No Signal" splash rather than going black. It
has high variance and will pass any naive liveness test as content.

Park the panel before measuring. MIDI CC is an offset onto the physical knob
position, not an absolute set.

Program change costs a multi-second blackout. Batch every evaluation for a
program together.

Any VT on the Pi bleeds into every recording. `LoopPlayer` writes `/dev/fb0`
directly, but `fbcon`'s cursor blink inverts its cell on top of that
regardless, as a small flickering square wherever the console's cursor
sits. `chvt` to another VT does not fix this: systemd spawns a fresh getty
on whatever VT becomes active, with its own cursor at a new position.
Disable the blink outright before a session instead:

```sh
echo 0 | sudo tee /sys/class/graphics/fbcon/cursor_blink
```
