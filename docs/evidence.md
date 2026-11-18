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

Nothing is timed by synchronized or fitted clocks. Every host-side event (CC send, serial
request and reply, MIDI receipt, capture buffer, Pi report received) is stamped on the host's
one `CLOCK_MONOTONIC`; the Pi's and the device's own clocks are never compared with it. They
contribute only counters, ordered against host events by causality:

| Channel | Hook | What it proves |
| --- | --- | --- |
| Pi player | each flip's clip frame, streamed live and stamped on receipt (`playout.Shown`) | a frame reported by `t` was on screen by `t` |
| Pi player | request and reply: the last flip done when the request arrived | the next flip came after the request left |
| Videomancer serial | `transport status` `field_count`, polled between readbacks (`wire.Poller`) | replies keep command order, so each readback and each send lies between two counts |
| Videomancer MIDI | `midi monitor` receipts (`wire.echoed`) | the device took the write by its echo |
| Capture card | buffer timestamps | a capture done before a send cannot show it |

`field_count` is the transport's counter: it advances once per frame only while the transport
plays, so a session puts the transport in play and leaves it as it found it, and a take whose count
never advanced is refused (`evidence.StoppedError`). Each tick read also records the transport's beat
phase, tempo and play state. The device latches CCs on its frame tick, the frame `field_count` counts: its readback never
changes between two polls that read the same count (docs/hardware.md). For each write:

- **Floor** `F`: the clip frame the host last heard the Pi show before sending it. No output
  of an earlier clip frame can show it. A write sent before any frame was heard has no floor
  and faults its take.
- **Latch** `L`: the device frame it takes effect on. A count `c` read before the send means
  `L >= c + 1`; a count `c'` asked for after its echo means `L <= c' + 1`; the readback change,
  between the counts read either side of it, pins `L` itself.
- **Offset** `O`: device frame `n` processes clip frame `n - O`, one integer per session as the
  device genlocks to the Pi: it ticks for clip frame `g` after the Pi flips `g` and before it
  flips `g + 1`. A Pi reply of `f` to a request sent after a count `c` was read gives `O >= c - f`;
  a count `c'` asked for after the host heard `f` shown gives `O <= c' - f + 1`; a write with
  floor `F` and latest latch `L` gives `O <= L - F` (`evidence.offset`). Bounds that cross fault
  the session; bounds a frame apart widen every write's window by that frame.
- **Response**: the clip frames after `L - O` until the picture shows the write, from that
  write's own captures. It is the program's processing of that CC; it is never pooled across
  writes, parameters or programs.

Each of these faults its take: a picture change attributed to a write before its floor; a floor
after its target frame (sent late); an echo before its send; a gap in the Pi's flip reports.

## Frame-locked sends

Takes are recorded in sessions (`locktake.locked_session`, `compose.session`): a program is
loaded once and all its takes, a sweep's chunks or a render's pass, play back to back. The Pi
plays the end of the clip's strip-coded lead-in, each take's clip frames from frame 0 in turn and
a lead-out as one stream (`--spans`, `session.Layout.spans`), so the card never relocks and one
frame line spans the session. The recorder rolls from one take's file to the next once the Pi is
heard showing the frame two before the boundary, both files keeping the captures from then to two
frames past it (`v4l2rec`'s `roll T0 T1 PATH`); takes stay separate files with their own sidecars. A
failed take is recaptured alone, as a session of one, which loads again.

1. A program with a bypass setting (`locktake.dry_state`) is loaded before playout and held
   dry by it. This is the control setting that the program's HDL shows leaves every other control
   dead and the input reaching the video: a declared switch turned on, otherwise a fader such as a
   Mix at the end of its range that returns the input exactly (`ProgramModel.bypass`,
   [program-model.md](program-model.md)), found from the declaration before anything loads. No slot
   or label is assumed to do the same thing across programs. No Passthru plays at all.
2. Once the start state has settled, the device-to-clip offset is bound from the frame ticks, the
   Pi's replies and its reports so far (`evidence.offset`); one bound only after the first write was
   due is a `LeadInTimeError`.
3. Each planned write for session frame `j` is sent once the device has ticked for frame
   `j - 1` (`Poller.wait_count`), its floor the frame the Pi was last heard showing. The take runs
   with a short interpreter switch interval so the sending thread wakes promptly; each send records
   when it was due, and the report gives the worst wake-up.
4. Two seconds before clip frame 0 the dry control is set to its take value; the last readback
   before frame 0 was heard shown must equal frame 0's planned parameters, and `fpga status` must
   name the program (`locktake.loaded_before`). A later take's last readback before its frame 0
   must equal the state the take before left.
5. On the frame after the last take's last the dry control is set dry again, held as long as the
   lead-in's wet stretch and `ANCHOR_FRAMES` more, whose strips pin the line's far end.

The frame line is fitted after the session from every take's strips (`lock.attribute_take`); a
session whose strips name no two frames is an `AnchorError`, never relaxed.

A program with no bypass setting has Passthru only at the session's ends: the lead-in plays
under Passthru, the program is loaded once `ANCHOR_FRAMES` frames have started, the outage
bridged by the Pi's flip log, and Passthru is loaded again after the last take. Its writes land by
its own session's events, as a dry program's do; no other take is consulted.

The lead-in holds exactly what it must (`session.lead_in_frames`): with a bypass, the start
state's settle and the offset's bounding (`LeadIn.bound_s`), then the wet stretch; without one,
the anchor, the longest load timed on the rig (`device.loadtimes`), the settle and the bounding.
Nothing is decoded before the first write. The lead-out (`session.lead_out_frames`) holds the dry
state as long as the lead-in held the wet one, or Passthru's load and the frame it latches on, then
the anchor. A load past the lead-in fails `loaded_before`, and a clip whose lead-in or lead-out is
shorter than the session's is refused before anything plays. Device waits fail as stalled once
they see nothing for longer than the longest load timed (`render.open_rig`), unless `--stall` says
otherwise; with no load on record and no `--stall`, a rig is not opened at all.

Every load a session times goes into one record per rig, `loads.jsonl` beside the program cache
(`SYNCSUMMONER_LOADS` overrides it), keyed by program and firmware and appended under an exclusive
lock, so every takes directory sizes its lead-ins and stall guard by every load timed on the rig.
`python -m syncsummoner.device.loadtimes captures/` seeds it from the loads takes on disk recorded:
takes directories' `loads.json` and wire records' `loads`.

Each take's lock record carries the session's id, its whole capture log, the strips read on it
(its anchors) and when a program processed the picture (`lock.session_record`). Takes 2..n carry
no anchor of their own: each is attributed on the session's line through every capture of the
session, then cut to its own captures and clip frames (`lock.View`). Each take's wire record holds
every send of the session, those it owns marked (`own`): its landings and receipts are judged on
its own sends, its readbacks against every send while it recorded.

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
reply came back (`wire.echoed`); a write is taken once its last message is. Its device
frame is its latch window (`evidence.latches`), and its clip frames that window less the
session's offset, never before its floor: `first` to `last`, `landed` the first. A take
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
| late | cannot have landed by its target, or its floor is past it |
| early | must have landed before its target |
| over more than one frame | its window spans frames, counted without failing the take |
| sent before any frame was seen | no floor; fails the take |

Takes recorded before the frame ticks, floors and Pi reports were kept cannot be landed this way
and are refused by `take_evidence`; they are recorded again.

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

- **Landing windows.** A step is scored by the picture's change across its landing
  window, from the frame before `first` to `last` plus the lag (`matched.frame_change`
  over the window's span), so it is caught wherever in the window it landed. A window
  of one frame is the frame-to-frame change.
- **Controls** are spans as long as the step's window with no window of any write
  reaching into them, nor the lags after one; each span length has its own null.
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
no-op beside it is not a partner. Steps stand alone when no other parameter's landing
window meets theirs. A dead step is tested only more than the longest
lag from every live step's window, its own parameter's included, and where nothing else lands. Steps
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
