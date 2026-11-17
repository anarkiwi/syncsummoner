# Parameter evidence

Every render take proves, frame by frame, that each parameter write was sent for a
known clip frame, that the device took it, and whether the picture moved because of
it. The report is written beside every take and logged; a take is not aborted for an
unproven parameter. A take whose capture driver lost frames is a `TakeFaultError`
and is recaptured once. `syncsummoner evidence` recomputes the report from a take's
sidecars.

| Link | Question | Record |
| --- | --- | --- |
| 1 | intended vs sent | plan and sends in `<take>.wire.npz` |
| 2 | sent vs applied | readbacks and `midi monitor` receipts in `<take>.wire.npz` |
| 3 | applied vs acted upon | the take's own pictures against the clip's |

## Where a write lands

The device latches MIDI CCs once per output frame: the picture changes on a frame
start, never part way down it (docs/hardware.md). So a write lands on the first frame
whose first capture reaches the host at least a fixed latency after the write left it.
That latency is measured from the picture in every dry take's lead-in, and is the
only thing the landing frame rests on; the serial readback's timing is not used.

The latency is the card's capture phase plus constants, and the phase changes with
every recording, so it is measured per take: each calibration write sent at `t` and
first shown in frame `F` puts it in `(T(F-1) - t, T(F) - t]`, `T` being the frame's
first capture time; the bounds of all writes intersect in one band
(`evidence.latency`). Writes that disagree by half a frame or more stop the take.
Every frame's capture time less the band gives the instants a write must precede to
land on it (`evidence.grid`).

## Frame-locked sends

1. The program is loaded before playout and held dry for the lead-in: its bypass on,
   or, for a program without one, a crossfader slot whose role is wet set to 0
   (`locktake.dry_state`). The strip-coded lead-in then passes through unprocessed,
   so the take is one gap-free run that the strips anchor; no load interrupts it.
2. The dry control is toggled at random intervals and each toggle's first frame in
   the lead-in's picture measures the latency band.
3. While the recording runs, its stored captures are read (`lock.read_live`) and
   attributed (`lock.live_attribution`), giving each clip frame's capture time.
4. Each planned write for frame `j` is sent half a frame before the instant it must
   precede (`evidence.send_time`): the live clock can err either way and a send can
   wake late. The take runs with a short interpreter switch interval so the sending
   thread wakes promptly; each send records when it was due, and the report gives the
   worst wake-up. The frame line is refitted over the whole run from the growing
   sidecar while the take runs.
5. Two seconds before clip frame 0 the dry control is set to its take value; the
   last readback before frame 0's latch must equal frame 0's planned parameters,
   and `fpga status` must name the program (`locktake.loaded_before`).

A program with neither a bypass nor a wet crossfader keeps the old lead-in: Passthru
under the strips, the program loaded inside it, the outage bridged by the Pi's flip
log. Its latency band is the dry takes' (`ccpicture.json`) widened by the part of the
capture phase's half-frame range they have not shown; `render --cut-programs` renders
such programs after a dry one.

The boot count (settings key 9) is read before and after every take; any rise is a
`RebootError`.

## Link 1: intended vs sent

The plan is the score's automation, each write targeting the first frame that starts
at or after its time less the early bias. The k-th send of a `(frame, parameter)`
pair answers its k-th planned write (`evidence.landed`):

| Count | Meaning |
| --- | --- |
| missing | planned, never sent |
| duplicated | sent with nothing planned for it |
| out of order | sent before a write planned ahead of it |
| late, early | landed on a frame other than its target |
| on a tick | its landing tick could not be told from the next: inside the tick bracket, or read back a tick later than predicted |

A write whose send falls inside the latency band of a frame's instant is counted as
on a tick: the picture cannot say which side of it the write fell.

## Link 2: sent vs applied

Each readback is bounded by when its request left and its reply came back. Widened by
a frame each side for the readback's own lag, that interval spans the frames it can
report, and the slot must read one of the values it held over them
(`evidence.applied`). Parameters are additive (docs/hardware.md), and the take parks
every manual value at zero with no modulation, so the combined value must equal a CC
exactly; the CC-to-readback mapping is exact for all 1024 values.

`midi monitor on` makes the firmware log every channel message it receives. The
receipts are matched in order against the messages the sends put on the wire
(`evidence.received`). The monitor's output travels device to host only and is read
by the polling thread, which never has more than one command queued at the device,
so it cannot build the serial backlog that reboots it.

## Link 3: applied vs acted upon

Per clip frame, the output change is the mean absolute difference of the frame's
capture from the previous frame's, over all three planes at 96x54 (chroma included,
so a hue change counts); the source change is the same measure on the clip
(`evidence.frame_change`).

- **Controls** are frames with no write landing on them or the frame before.
- **Matched score.** Each frame's output change is scored against the `sqrt(n)`
  controls nearest it in source change: its departure from their median in units of
  their MAD (Loftsgaarden and Quesenberry's neighbour count; the measure's resolution
  stands in for a MAD of zero).
- **Per-step p.** A step's score is placed among the scores of every control (an
  empirical mid-p), so no distribution is assumed.
- **Per parameter.** The steps' p values are combined by Fisher's method, its
  statistic and degrees of freedom shrunk by the effective sample fraction of the
  series (Brown), and tested at lags 0 and 1 frame after the modelled landing.
- **Error rate.** The critical z is the one-sided three-sigma tail divided across
  every parameter and lag tested in the take (Bonferroni), as survival.py does.

A parameter that only ever lands together with others is tested on those frames and
reported with its partners. Why a parameter is not acted upon comes from the first
link that broke: not sent on time; the firmware did not take it; the program names
the slot unused; the picture was a uniform field at most of its steps; or the
firmware took it and the picture did not move.

## Commands

```sh
syncsummoner render score.yaml --source clip.mp4 --cut-programs ... --takes takes/   # evidence per take, hard check
syncsummoner evidence takes/take-Pinwheel.mkv --source timecoded.mkv           # recompute and print it
```

Each take leaves `<take>.wire.npz` (plan, sends, readbacks, receipts, cc map, the
measured latency, program names), `<take>.lock.npz`, `<take>.picked.npy` and
`<take>.evidence.json`.
