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
that starts at least a fixed latency after the device took it. Frame starts are the
take's frame line on the capture clock (`evidence.frame_times`,
[hardware.md](hardware.md#the-cards-clock)), so they carry no capture phase. The latency
is measured from the picture in every dry take's lead-in, and is the only thing the
landing frame rests on; the serial readback's timing is not used.

Each calibration write taken at `t` and first shown in frame `F` puts the latency in
`(T(F-1) - t, T(F) - t]`, `T` being the frame's start on the line; the bounds of all
writes intersect in one band (`evidence.latency`). Writes that disagree by half a
frame or more stop the take. Every frame's start less the band gives the instants a
write must precede to land on it (`evidence.grid`). Where the take's captures admit
lines other than the one fitted, frame starts lie within `line_lo..line_hi` of it, so
the device's own latency lies in `Latency.bounds`, the band widened by them; each record
says its basis (`"basis": "start"`).

## Frame-locked sends

1. The program is loaded before playout and held dry for the lead-in: its bypass
   switch turned on (`locktake.dry_state`), the declared switch that, turned on,
   leaves every other control dead and the input reaching the video in the program's
   HDL (`ProgramModel.bypass`, [program-model.md](program-model.md)); no slot or
   label is assumed to do the same thing across programs. The strip-coded lead-in then passes through unprocessed,
   so the take is one gap-free run that the strips anchor; no load interrupts it.
2. The dry control is toggled at random intervals and each toggle's first frame in
   the lead-in's picture measures the latency band.
3. While the recording runs, its stored captures are read (`lock.read_live`) and
   attributed (`lock.live_attribution`), giving the frame line from the lead-in's strips.
4. Each planned write for frame `j` is sent half a frame before the instant it must
   precede (`evidence.send_time`): the live clock can err either way and a send can
   wake late. The take runs with a short interpreter switch interval so the sending
   thread wakes promptly; each send records when it was due, and the report gives the
   worst wake-up. The frame line is carried on while the take runs (`locktake.Clock.extend`):
   each capture the growing sidecar shows starting a frame, its bytes differing from the
   capture before's, is named by the line so far and narrows it, as the offline fit does.
5. Two seconds before clip frame 0 the dry control is set to its take value; the
   last readback before frame 0's latch must equal frame 0's planned parameters,
   and `fpga status` must name the program (`locktake.loaded_before`).
6. On the frame after the clip's last the dry control is set dry again, and the take
   runs on as long as the lead-in's wet stretch into the clip's lead-out, whose strips
   then pin the frame line's far end.

A program with no bypass switch keeps the old lead-in: Passthru
under the strips, the program loaded inside it, the outage bridged by the Pi's flip
log; Passthru is loaded again for the lead-out. Its latency band is where every dry
take's `Latency.bounds` meet (`ccpicture.json`, start-basis records only; a judged dry take
records its own); `render --cut-programs` renders such programs after a dry one.

The boot count (settings key 9) is read before and after every take; any rise is a
`RebootError`.

## Link 1: intended vs sent

The plan is the score's automation, each write targeting the first frame that starts
at or after its time less the early bias. The device applies a slot's value once a
frame, so the plan keeps one write per `(frame, parameter)`, the last asked for, and
drops any write that repeats the value its slot already holds, the take's start state
included (`locktake.schedule`). Automation before the clip clamps onto frame 0, which
the start state already sets.

Nothing bounds a frame's writes ahead of time: the device's drain rate depends on what
it and the program are doing (docs/hardware.md). Every write instead waits for the
previous one's `midi monitor` echo, and none goes before its frame's send instant
(`wire.Echoes`, `wire.Sender`); the start state goes out the same way, skipping slots
already parked at their value. A write with no echo once the take's longest echo has
passed is written once more, and the copy's row is kept in the wire record's `resent`.
`syncsummoner score plan SCORE --program NAME` prints a pass's writes and messages per
frame without a device.

**Realized, not planned.** A message's echo comes with the first poll or command
reply the device writes after taking it. So it was taken after the request of the
reply before that one left, and after its write began, and no later than its own
reply came back (`wire.echoed`); a write is taken once its last message is. Its
landing is the first tick after that bracket, and `landed` is the frame it is in effect
from; `clear_ns` is how far the nearer end of the bracket clears a tick, negative when a
tick's bracket overlaps it, so the write may be in effect a tick either side. A take
without receipts falls back to the send interval. `evidence.realized` is the state every
clip frame actually held, from those landings and the wire values.

The k-th send of a `(frame, parameter)` pair answers its k-th planned write, resent
copies aside (`evidence.landed`):

| Count | Meaning |
| --- | --- |
| missing | planned, never sent |
| never echoed | sent, but some message of it, and of its copy, never came back |
| duplicated | sent with nothing planned for it |
| out of order | sent before a write planned ahead of it |
| late, early | in effect from a frame other than its target, measured from its echo |
| on a tick | its landing tick could not be told from the next: a tick inside its bracket, or read back a tick from where its echo puts it |

The least slack, echo to the target's tick, is reported too. The CC-to-picture latency
of a dry take is calibrated on the same brackets, so the tick grid and the landings are
in one time base. A take without a dry control uses a latency on file; one measured
before echoes were used was timed from the send's return, earlier than the device took
it, which leans its landings late.

Re-judged this way, the 11 frame-locked takes with receipts (rc4/takes and full/takes)
have no write unambiguously late or early, and no readback disagreement. 94 writes are
on a tick and count late: their echo bracket overlaps the tick they targeted, almost all
because the echo came with a poll reply stalled 25 to 37 ms, the others because the send
itself woke late. The Bit Rotator rc.4 take's 31 writes lost in the 124-message burst were
never echoed, and its readback agrees they never took effect.

The cyclic garbage collector is off while a take plays. A full collection over the
poller's growing records holds the interpreter lock for longer than the half frame a send
has: on an rc.4 take one stalled a send and the read in flight together for 30 ms and
landed the write a frame late.

## Link 2: sent vs applied

Each readback is bounded by when its request left and its reply came back. Widened by
a frame each side for the readback's own lag, that interval spans the frames it can
report, and the slot must read one of the values it held over them
(`evidence.applied`). Parameters are additive (docs/hardware.md), and the take parks
every manual value at zero with no modulation, so the combined value must equal a CC
exactly; the CC-to-readback mapping is exact for all 1024 values.

`midi monitor on` makes the firmware log every channel message it receives. The
receipts are aligned against the messages the sends put on the wire
(`evidence.received`), as a common subsequence (difflib's matching blocks), so a run of
lost messages costs only those messages. Each run is reported with the sends that
carried it, their parameters, target frames and send times. The monitor's output travels
device to host only and is read by the polling thread, which never has more than one
command queued at the device, so it cannot build the serial backlog that reboots it.
Readbacks are checked against the realized landings: a write never echoed held nothing.

## Link 3: applied vs acted upon

Per clip frame, the output change is the mean absolute difference of the frame's
capture from the previous frame's, over all three planes at 96x54 (chroma included,
so a hue change counts); the source change is the same measure on the clip
(`matched.frame_change`).

- **Controls** are frames with no write landing on them or the frame before.
- **Matched score.** Each frame's output change is scored against the `sqrt(n)`
  controls nearest it in source change: its departure from their median in units of
  their MAD (Loftsgaarden and Quesenberry's neighbour count; the measure's resolution
  stands in for a MAD of zero).
- **Switch regimes.** With the program model, every clip frame's parameter state is
  replayed from the take's start state and the sent writes from their landing frames
  (`program.predict.frame_states`), and its regime is the register values of the
  program's switches, the one-bit controls a non-knob-gated parameter's gating class
  ties (`program.predict.regimes`). A step is tested only against controls whose
  switches equal its frame's, its own switch aside (`matched.Regime`): with Bypass on
  the output follows the source, so its change at equal source change is not the
  processed frames'. Each regime's controls form its own null, scored against
  themselves. A regime with fewer than two controls has no null, and its steps are
  left untested and counted (`steps_unmatched`, `dead_steps_unmatched`). Without the
  model, or with no switches, all controls form one regime.
- **Per-step p.** A step's score is placed among the scores of every control of its
  regime (an empirical mid-p), so no distribution is assumed.
- **Per parameter.** The steps' p values, across every regime, are combined by Fisher's method, its
  statistic and degrees of freedom shrunk by the effective sample fraction of the
  series (Brown), and tested at lags 0 and 1 frame after the modelled landing.
- **Error rate.** The critical z is the one-sided three-sigma tail divided across
  every parameter and lag tested in the take (Bonferroni), as survival.py does.

**Steps judged by the program model.** Each sent write is a step from the state the
take's parameter stream held when it landed: every write landing on the same frame
applied, its own parameter at its value before the frame
(`program.predict.pre_states`). The model's observability conditions judge each step
(`program.predict.classify`): **dead** (no bit of the parameter reaches the video
there), **no-op** (its two values differ only in bits that do not reach it) or
**live**. A step that writes the value already held is **unchanged** and not tested.
No step is simulated: that is `program step`'s job, on demand.

The model is queried once per gating class. A parameter the model finds dead at
its reference state and revived by a knob is queried with every other control
tied, at each state its steps start from. Any other parameter is queried with
every knob free, so only the
switches tie it and a take has a handful of classes. Dead with knobs free holds
for every knob value; live with knobs free means some knob value reaches the
video, so a knob that kills it elsewhere can leave a dead step in the live pool,
where it can only fail to move.

A parameter's steps are then tested in two pools: those the model leaves live
(live), and those it proves cannot change the picture (dead and
no-op). A live step is tested on frames where no other live step lands, so a proven
no-op beside it is not a partner. A dead step is tested only more than the longest
lag from every live step, its own parameter's included, and where nothing else lands. Steps
that never land alone are tested with their partners. Every pool tested counts
toward the family-wise bound.

**Verdicts** (`Evidence.verdict`), per parameter:

- **live, moved**: its live steps moved the picture beyond chance.
- **live, not seen**: they did not. The report gives the minimum detectable effect in
  the measure's own units, `critical * scatter / sqrt(effective steps)`, where
  `scatter` is the spread of the step-free controls' change about their matched
  median: the null the z score is taken against, its variance weighted by the steps
  each regime's null scored. It never compares against another
  parameter. With a positive observed mean change, it also gives the steps at which
  that change would clear the bound; these go to the take directory's
  `schedule.json`.
- **dead, moved**: its proven no-op or dead steps moved the picture. That is a hard
  finding: the model or the firmware is wrong (`Evidence.contradictions`).
- **never live**: every step was dead or a proven no-op in the state it was taken
  from. It is not tested, and says nothing about the parameter's effect elsewhere;
  `schedule.json` lists it with 0 steps, so `probe archive --schedule` steps it
  through every level in a base state where the model makes it live.

"No effect" is never a statistical verdict. It is only what the model proves, as
dead or no-op steps. With `--no-model`, or when the model cannot be built, every
step is tested in one pool and the verdict is **moved** or **not seen**.

Before a not-seen verdict, the reason comes from the first link that broke: not sent on
time; the firmware did not take it; the program names the slot unused; or the
picture was a uniform field at most of its steps.

## Commands

```sh
syncsummoner render score.yaml --source clip.mp4 --cut-programs ... --takes takes/   # evidence per take
syncsummoner evidence takes/take-Pinwheel.mkv --source timecoded.mkv           # recompute and print it
```

Each take leaves `<take>.wire.npz` (plan, sends, readbacks, receipts, cc map, the
measured latency, program names), `<take>.lock.npz`, `<take>.picked.npy` and
`<take>.evidence.json`.
