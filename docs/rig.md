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

## Bring-up checks

```sh
vmancer devices                     # serial, rawmidi node, tty, usbfs
vmancer shell "version"             # firmware; must match the profile archive
vmancer shell "fpga status"         # state, configured, program, program_count
vmancer shell "video status"        # source, timing, lock, output
vmancer programs                    # installed FPGA programs
```

Then confirm the loop end to end: load `Passthru`, play a known pattern from the
Pi, capture, and check the Gray-code strip decodes. `Passthru` reports all
twelve parameters as `Null`, so anything wrong in that capture is the chain, not
the program.

## Gotchas

Hold the capture stream open. Lock costs are seconds, far above any per-sample
dwell, and reopening per sample makes the sweep meaningless.

The capture card synthesizes a "No Signal" splash rather than going black. It
has high variance and will pass any naive liveness test as content.

Park the panel before measuring. MIDI CC is an offset onto the physical knob
position, not an absolute set.

Program change costs a multi-second blackout. Batch every evaluation for a
program together.
