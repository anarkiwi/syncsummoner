# Parameter evidence

Every render take proves, frame by frame, that each parameter write was sent for a
known clip frame, that the device took it, and whether the picture moved because of
it. Takes that fail either of the first two links fail the render
(`UnverifiedTakeError`); the third is reported per parameter.
`syncsummoner evidence` recomputes the report from a take's sidecars.

| Link | Question | Record | Hard check |
| --- | --- | --- | --- |
| 1 | intended vs sent | plan and sends in `<take>.wire.npz` | yes |
| 2 | sent vs applied | readbacks and `midi monitor` receipts in `<take>.wire.npz` | yes |
| 3 | applied vs acted upon | the take's own pictures against the clip's | reported |

## The device's frame tick

The device latches MIDI CCs once per output frame. Its own `modulation status`
readback changes on that tick and nowhere else, and the picture changes on the
frame that tick latches, never part way down it (docs/hardware.md). A write
therefore lands on the frame of the first tick after it reaches the device.

During a take one thread keeps exactly one `modulation status` in flight, every
few milliseconds, timestamping each request and reply on host `CLOCK_MONOTONIC`.
Each readback change brackets one tick; the brackets of all changes, folded onto
the frame period measured from the capture timestamps, intersect in one interval
that holds every tick (`evidence.tick_bracket`). Only readback that rules that period
out, which takes minutes of changes, sets the grid's period itself, as the Chebyshev
centre of the interval constraints; no feasible grid is an error, not a fit.

## Frame-locked sends

1. The program is loaded before playout and held dry for the lead-in: its bypass on,
   or, for a program without one, a crossfader slot whose role is wet set to 0
   (`locktake.dry_state`). The strip-coded lead-in then passes through unprocessed,
   so the take is one gap-free run that the strips anchor; no load interrupts it.
2. The dry control is toggled at random intervals. Each toggle reads back on a tick
   and shows in the lead-in's picture on a frame start, giving the delay from the
   latching tick to the card's first capture of that frame (`evidence.latency`).
   Every toggle must agree to within half a frame, else no single tick-to-frame map
   holds and the take stops.
3. While the recording runs, its stored captures are read (`lock.read_live`) and
   attributed (`lock.live_attribution`), giving each clip frame's capture time.
4. Each planned write for frame `j` is sent just after the latest instant the tick
   before the one that latches `j` can fall (`evidence.send_time`). A sending thread
   can only wake late, so this leaves almost a whole frame for it; each send records
   when it was due, and the report gives the worst wake-up. The frame clock is carried
   on from the growing sidecar while the take runs.
5. Two seconds before clip frame 0 the dry control is set to its take value; the
   last readback before frame 0's tick must equal frame 0's planned parameters,
   and `fpga status` must name the program (`locktake.loaded_before`).

A program with neither a bypass nor a wet crossfader keeps the old lead-in: Passthru
under the strips, the program loaded inside it, the outage bridged by the Pi's flip
log. It uses the latency the last dry take in the same directory measured
(`cclatency.json`); `render --cut-programs` renders such programs after a dry one.

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

A write's landing tick is the readback's own where the write changed its slot, and
the first tick after it otherwise. The report also brackets how long before a tick a
write must leave the host to be latched by it (`deadline_ms`), from calibration
writes that fell on either side.

## Link 2: sent vs applied

Every readback slot is checked against the value the writes landed so far say it
holds; a read whose request interval contains a tick may show either side of it
(`evidence.applied`). Parameters are additive (docs/hardware.md), and the take parks
every manual value at zero with no modulation, so the combined value must equal the
last CC exactly; the CC-to-readback mapping is exact for all 1024 values.

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
