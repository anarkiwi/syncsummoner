# HDMI input signal loss

Diagnosis of the intermittent fault in which the Videomancer stops passing video
while still reporting the input locked. Written without touching the device.
Sources: `hardware.md`, `pyvmancer/docs/{protocol,firmware-notes}.md`,
`predecessor-design.md`, one instrumented journal of 49 program loads, and the
public LZX SDK, firmware release notes and manual.

Claims are tagged **F** documented or measured, **I** inferred from an **F** with
the inference stated, **G** guess held only because nothing rules it out.

## 1. Video path

### Block diagram

```
 Raspberry Pi 4B --HDMI 1920x1080p30--> ADV7611 --LLC 74.25 MHz + RGB444 + HS/VS/DE--+
 composite ---------------------------> ADV7181C --LLC + 12b DDR RGB----------------+|
                                                                                    vv
                       microSD                                          Lattice iCE40 HX4K TQ144
                    *.vmprog (VMPG)                                     3520 LC / 20 BRAM / 1 PLL
                            |                                        vid_clk <= i_hdmi_rx_clk
                            v                                     video_sync_generator(LOCK_TO_REF)
   RP2040 --slave-SPI bitstream config--> [fabric, fully reloaded per program]
      |  --16b write-only ABI, no readback--^   |            |
      |  --I2C--> ADV7611, ADV7513, ADV7181C,   |            |
      |           ADV7393 (FPGA has no I2C)     |            |
      |                                         v            v
      +-- USB 16d0:14db: CDC shell, MIDI, MSC  ADV7513     ADV7393
      +-- panel: 6 knobs, 5 toggles, fader     HDMI out    CVBS / S-video
                                                  |
                                                  v
                                          AVerMedia capture card
```

Three features of this diagram carry the whole diagnosis.

**The iCE40 has no clock of its own.** In every HDMI-input bitstream
`vid_clk <= i_hdmi_rx_clk` — the fabric runs directly on the ADV7611's LLC
output, with no PLL in the path at `C_HD_CLOCK_DIVISOR = 1` and no oscillator on
the FPGA's clock nets. Genlock is achieved by clock adoption, not by a recovery
loop. If the ADV7611 stops producing LLC, the fabric stops entirely and both
outputs die.

**The MCU-to-FPGA ABI is write-only.** 16-bit SPI frames, bit 15 always `0`
(write), no read path. The RP2040 cannot read back anything the fabric holds.
Every field `video status` and `fpga status` report is therefore the firmware's
*intent*, never the fabric's state — which is the architectural reason those
flags are advisory rather than merely unreliable.

**The FPGA has no I2C.** The pin map contains only the four ADV parallel buses,
the RP2040 SPI, `RP2040_GPOUT_CLK` and the sync output. The RP2040 alone
configures all four ADV parts. A loaded program has no channel by which to
disturb the receiver, the transmitter or either analog converter.

### Provenance

| Element | Tag | Source |
| --- | --- | --- |
| FPGA is a Lattice iCE40 HX4K, TQ144; MCU is an RP2040 | **F** | [`hardware.toml`](https://github.com/lzxindustries/videomancer-sdk/blob/main/fpga/hardware/rev_b/config/hardware.toml) `platform="ice40" device="hx4k" package="tq144"`; [SDK `CHANGELOG.md`](https://github.com/lzxindustries/videomancer-sdk/blob/main/CHANGELOG.md) "Platform: Videomancer (RP2040 + Lattice ICE40HX4K FPGA)" |
| 3520 LCs, 20 BRAMs, 1 PLL; library programs fail synthesis "over LC capacity" | **F** | `build_programs.sh` nextpnr parsing, SDK CHANGELOG |
| HDMI receiver ADV7611; transmitter ADV7513; analog decoder ADV7181C; analog encoder ADV7393 | **F** | [`pin_map.pcf`](https://github.com/lzxindustries/videomancer-sdk/blob/main/fpga/hardware/rev_b/constraints/pin_map.pcf), `hardware_top.vhd`. The ADV7393 corroborates `predecessor-design.md` §3 |
| HDMI is not implemented in fabric | **F** | receiver and transmitter present as parallel 24-bit buses; iCE40 HX4K has no TMDS SERDES |
| No I2C on the FPGA; no oscillator net; RP2040 configures all four ADV parts | **F** | `pin_map.pcf` is 110 `set_io` lines containing only the ADV buses, RP2040 SPI, `RP2040_GPOUT_CLK`, `VID_SYNC_OUT_P/N` |
| `vid_clk <= i_hdmi_rx_clk` in all HDMI-input configurations | **F** | [`core_top.vhd`](https://github.com/lzxindustries/videomancer-sdk/blob/main/fpga/core/gbr444_30b/rtl/core_top.vhd) `GEN_HD_HDMI_IN`, `GEN_SD_HDMI_IN`; same in `yuv422_20b`, `gbr422_20b` |
| Build clocks are exactly 74.25 MHz (HD) and 27 MHz (SD) | **F** | `build_programs.sh` |
| The on-chip PLL is used only for standalone mode and for `C_HD_CLOCK_DIVISOR > 1` program-clock decimation | **F** | `core_top.vhd` `GEN_DIRECT_PROG_CLK` / `GEN_DIV2_PLL` / `GEN_DIV4_PLL`; all shipped `core_config` packages set the divisor to 1 |
| Output raster comes from `video_sync_generator` with `G_LOCK_TO_REF` and `G_PHASE_ADVANCE`, driven by a firmware-supplied timing ID and a per-program `phase_advance_clks` | **F** | `core_top.vhd`; [`auto_resolution_detector.vhd`](https://github.com/lzxindustries/videomancer-sdk/blob/main/fpga/common/rtl/video_timing/auto_resolution_detector.vhd) header: "driven by a firmware-supplied timing ID" |
| Resolution is measured in fabric, defaulting to 1920x1080 until the first full frame stabilises | **F** | `auto_resolution_detector.vhd` |
| MCU-to-FPGA ABI is 16-bit, write-only, 100 kHz-10 MHz, registers 0x00-0x1F; 0x09 is `sync_phase_advance_clks` | **F** | [`abi-format.md`](https://github.com/lzxindustries/videomancer-sdk/blob/main/docs/abi-format.md) |
| A `.vmprog` carries up to eight full bitstreams, `{sd,hd}` x `{analog,hdmi,dual,standalone}`, Ed25519-signed, optionally DEFLATE-compressed | **F** | [`vmprog-format.md`](https://github.com/lzxindustries/videomancer-sdk/blob/main/docs/vmprog-format.md). `bitstream_hd_hdmi` is TOC type 6, matching this unit's `fpga preferred-variant = hd_hdmi` |
| Every program load is a full bitstream reload; iCE40 has no partial reconfiguration | **F** | manual: "the FPGA simply loads a different firmware program"; `program-development-guide.md`: "decompresses transparently during FPGA configuration"; fault codes `0x2C000007` SPI transfer to FPGA failed, `0x2C000008` FPGA configuration timeout |
| Program load disables **all** outputs, video and sync, for several seconds | **F** | [user manual](https://lzxindustries.net/instruments/videomancer/manual/user-manual) |
| No frame store; a few lines of video only | **F** | user manual; consistent with 20 BRAMs |
| Genlock only; adopts the source's timing, resolution and frame rate; cannot convert | **F** | user manual, [specs](https://lzxindustries.net/instruments/videomancer/specs) |
| 1080p30 is supported and is the receiver's default detailed timing descriptor; 1080p50 and 1080p60 were removed from the EDID at firmware 0.1.2 | **F** | specs; [firmware releases](https://github.com/lzxindustries/videomancer-firmware/releases) 0.1.2, 0.1.4 |
| A dedicated ADV7611 fault domain exists: `0x08300007` unsupported timing, `0x0830000A` I2C communication error, `0x08300034` self-test failed | **F** | [fault codes reference](https://lzxindustries.net/instruments/videomancer/manual/fault-codes-reference) |
| The sync path was rewritten immediately before this unit's firmware | **F** | releases: rc.32 sync generator in the HDMI RX clock domain with ref-based phase lock; rc.33 unified sync path locking to program input on `vid_clk` in all routing modes, SPI reg 0x09 written at program load; rc.39 HDMI RX HS/VS into ADV7181C EXT sync delayed ~50 LLC clocks, and FPGA reload on dual analog-in changes |
| Loss of the ADV7611 LLC necessarily kills every output | **I** | `vid_clk` is that LLC and there is no fallback in HDMI-input bitstreams |
| A frozen frame at the capture card is the card holding its last frame | **I** | the device has no frame store and cannot repeat one |
| **State that survives a program load is not in the FPGA fabric** | **I** | the fabric is erased and rewritten in full on every load; reloading the program is measured never to recover the hard fault |
| `video input` changes cost a full bitstream reload and move the fabric onto a different clock source | **I** | rc.39 "FPGA reload on dual analog_in changes"; analog configurations use `vid_clk <= i_vid_dec_clk` |
| 74.25 MHz through a video pipeline on an HX4K is close to the part's limit | **G** | the library already fails synthesis on LC capacity; no timing-closure margins are published |

### Not found

No teardown, FCC filing, PCB photograph, schematic or BOM exists publicly, so
nothing above is corroborated by inspection — it all comes from LZX's own RTL
and pin constraints. No customer-facing LZX page names the FPGA. The iCE40 speed
grade and the config-flash topology are unstated. Shipped EDID contents are
unpublished beyond the 0.1.2 and 0.1.4 change notes. No release note in any
`1.0.0-rc.x` describes an HDMI input lockup or a receiver reinit, but most rc
bodies are auto-published placeholders, so that absence is weak evidence.

## 2. What the journal shows

`device_journal.jsonl`: 49 `load_program` calls over 371 s, each followed by
`Session.ensure_live` and then `probe_health` capturing 6 frames against an
8.0 s budget. Cadence was uniform — every load was issued 0.44-0.46 s after the
previous probe returned.

It is **not** a clean run. Eight loads were followed by capture starvation.

| Metric | Value |
| --- | --- |
| Loads with a capture probe | 48 |
| Probes returning fewer than 6 frames within 8.0 s | 8 |
| Hazard per load | 0.167, 95% Wilson CI [0.087, 0.296] |
| `ensure_live` failures | 0 of 49 |
| `resync` events | 0 |
| Serial shell failures | 0 |

Starved loads by index: 5, 9, 13, 15, 27, 30, 39, 49 — `Combing`, `Derez`,
`Faultplane`, `Folio`, `Lumarian`, `Pinwheel`, `Sfumato`, `Zollner`. Seven
returned zero frames; `Folio` returned one, fully black. In all eight the device
reported `locked: true`, `source_locked: true`, `timing: 1080p30`.

The three other `ok: false` probes (`Plumber`, `Ramp Logic`, `Vectorscope`)
captured a full six frames and failed only the `mean > 0.002` brightness test in
`probe_health`. They are dark programs, not signal loss, and `Pong` passing the
same test at `mean = 0.0081` shows how thin that margin is.

### The starvation is memoryless in load count

| Test | Result |
| --- | --- |
| Split half | 4 events in loads 1-24, 4 in loads 25-49 |
| Failure against load index, point-biserial | r = -0.051, p = 0.73 |
| Failure positions against uniform, KS | p = 0.83 |
| Inter-event gaps | 5, 4, 4, 2, 12, 3, 9, 10; mean 6.1 against 6.0 expected for iid at p = 0.163 |

This is the journal's main contribution and it cuts both ways.

**It supports the program-load correlation** and strengthens it well past "weak
but consistent": a transient carrying the reported fault's exact signature —
device reports the input locked while the capture receives nothing — occurs
after one load in six, and occurs only after loads. Nothing else was done in
this run.

**It undermines any accumulating-state account.** A leak, a fragmenting heap or
a thermal ramp produces a hazard rising with load count. This one is flat to
within the resolution of 49 loads.

### But the transient and the hard fault are not the same failure

The transients cleared on the next load. The hard fault is measured never to
clear on a load. If the hard fault were the same memoryless per-load lottery at
p = 0.167, then reloading would clear it about five times in six; across roughly
six occurrences it cleared none. Two failures share one visible signature:

| | Soft | Hard |
| --- | --- | --- |
| Rate | ~1 load in 6 | ~6 in a day of sweeps |
| Clears on next `program load` | yes | no |
| Clears on timing bounce | untested | once |
| Clears on `video input` bounce | untested | once |
| Clears on power cycle | n/a | every time |
| Implied location | anywhere, including the fabric | **not in the fabric** |

The second row of the last line is a deduction, not a preference: the fabric is
erased and rewritten in full on every load, so anything a reload cannot clear is
held in the ADV7611, the ADV7513, the ADV7181C, or the RP2040.

### Two instrumentation caveats, both material

**`ensure_live: ok` is not evidence the link was live.** The sweep calls it with
`require_motion=False`, so `Capture.wait_for_lock` returns on the first frame
that is neither a failed grab nor the no-signal splash. In every starved case it
returned `ok` about 0.38 s before a probe that then saw nothing for 8 s. Either
the link died inside that 0.38 s, or — more likely — `wait_for_lock` popped a
V4L2 buffer captured before the load blackout, and `frames(settle=4)` drained
the rest before starving. Program load disables sync as well as video (**F**),
so the card genuinely loses signal at every load and stale buffers are expected.
`Capture` uses `cv2.VideoCapture`, which discards presentation timestamps, so
the two readings cannot be separated from this record. `design.md` §10 retired
PTS in favour of the Gray-code strip; that reasoning holds for frame *identity*
and does not extend to proving *when* a frame was captured.

**Only `load_program` is journalled.** `Session._note` records nothing else, so
the 10.4-12.4 s pauses following four of the eight starvations are unattributed
caller activity. No `resync` was journalled, so the timing bounce was not used
and the starvations cleared without it.

## 3. Hypotheses, ranked

Ranked by likelihood for the **hard** fault, the one that needs a power cycle.
H2 is the confound that has to be excluded first regardless of its rank.

### H1 — the ADV7611 stops delivering LLC, or delivers it wrongly, and stays that way

The fabric's only clock in `hd_hdmi` bitstreams is the ADV7611 LLC. The receiver
sits on the RP2040's I2C bus, outside everything a program load touches. If it
drops into a state where its HDMI status registers still read locked but its
pixel output is stopped or mistimed, the fabric is dead, both outputs are dead,
and no amount of reconfiguration helps.

| For | |
| --- | --- |
| Explains the single most discriminating datum | reloading the program cannot recover, because a reload does not touch the ADV7611 |
| The receiver is the fabric's sole clock source | **F**, `core_top.vhd` |
| The firmware's model of this receiver is demonstrably wrong already | `hdmi.connected` reads false immediately after a power cycle on a healthy link |
| The receiver has its own fault domain, including I2C error and self-test failure | `0x083000xx` |
| Power cycle always works | it is the only thing that re-initialises the ADV7611 |
| `video input analog` then `hdmi` recovered it once | that forces a bitstream reload onto the ADV7181C LLC and back, and re-runs routing configuration over I2C |
| The device reports the input locked throughout | it reads the ADV7611's own registers, which are honest about HDMI lock and say nothing about LLC integrity |

| Against | |
| --- | --- |
| Does not explain the `state: idle` occurrence | that belongs to H4 |
| No firmware release note reports a receiver lockup | but most rc bodies are placeholders |
| A timing bounce recovering it once is not obviously predicted | unless the bounce's reassertion of `video input` re-touches the receiver |

**Discriminator.** During a fault, select `video input analog` and leave it
there. Under H1 the composite path should carry video normally, because it runs
on the ADV7181C LLC in a different bitstream. If analog also fails, the fault is
not receiver-specific. Second: raise `log level` and capture the unprefixed log
lines; a `0x083000xx` code names the receiver directly.

### H2 — the fault is at the capture card, not the device

Every observation of "the output is black" is mediated by one AVerMedia card
read through `cv2`.

| For | |
| --- | --- |
| "Frozen frame" cannot originate at the device | no frame store (**F**); necessarily the card holding (**I**) |
| The journal measures the card, not the device | all 8 starvations are `Capture.frames` returning nothing |
| The `ensure_live`/probe contradiction has a clean host-side explanation | stale V4L2 buffers, §2 |
| The card fabricates a splash and needs seconds to re-lock | `hardware.md` |
| Every load removes sync as well as video | **F**; the card is forced through a full re-lock 49 times per sweep |

| Against | |
| --- | --- |
| The `state: idle` occurrence is unambiguously device-side | `program load` returned success and left the FPGA idle |
| `video input analog` then `hdmi` recovered it once | that verb does not touch the card except through the output |
| Power cycling the device fixes it | non-discriminating: it is also a hotplug to the card |

**Discriminator.** One measurement settles it: a second, independent sink on the
Videomancer HDMI output — a monitor or a second capture card — observed during a
fault. Until that exists, every rate in this document is a property of the pair,
not of the Videomancer.

### H3 — the fabric receives a wrong timing ID or phase advance and generates an invalid raster

The output raster is produced by `video_sync_generator(G_LOCK_TO_REF,
G_PHASE_ADVANCE)` parameterised by a firmware-supplied timing ID and by
`sync_phase_advance_clks`, written over the write-only SPI ABI at program load
(rc.33). The RP2040 cannot read back what the fabric holds. A write lost, or
applied while the fabric is unclocked, leaves the device generating a raster
nothing downstream can lock to, while `video status` reports the firmware's
intent — which in all eight journal starvations was `timing: 1080p30`, unchanged.

| For | |
| --- | --- |
| Explains the measured recovery table exactly | a reload rewrites the same auto-derived timing ID and fails; `video timing <native>` alone writes the value already believed and fails; `video timing <other>` then `<native>` writes an explicitly overridden ID and succeeds, leaving `overridden: true` |
| Explains why every status flag is advisory | the ABI has no read path, so firmware reports intent and cannot detect divergence |
| The mechanism is new in this firmware line | rc.32, rc.33 and rc.39 all rework this exact path, immediately before rc.37/rc.40 |
| Resolution is measured in fabric with a 1920x1080 default until the first full frame stabilises | a load that starts mid-frame has a defined-but-possibly-wrong starting state |
| Matches the memoryless per-load transient | a race resolves independently each load |

| Against | |
| --- | --- |
| Predicts reload should clear the hard fault about five times in six | it never did |

**Verdict.** This is the best available account of the **soft** 17%-per-load
transient and a poor one for the hard fault. Ranked third overall, first for the
transient.

### H4 — the RP2040-side load path fails and leaves the FPGA unconfigured

The `state: idle` occurrence, with `program load` returning success and leaving
the FPGA idle, is a configuration failure, not a video failure.

| For | |
| --- | --- |
| A documented fault domain exists for exactly this | `0x2C000007` SPI transfer to FPGA failed, `0x2C000008` FPGA configuration timeout, plus bitstream read and decompression failures |
| RP2040-side resets in the storage path were a live problem at this firmware | rc.40's sole change is watchdog feeding during SD writes so uploads do not reset mid-transfer |
| Explains why `program load` reported success while doing nothing | the shell's `ok` is not a configuration acknowledgement |

| Against | |
| --- | --- |
| Observed once | against many normal occurrences |
| The fault code was never captured | nobody has read `!code:message` or the log stream on a failure |

**Discriminator.** Capture the error code and the log stream. A `0x2C0000xx`
code confirms it outright, and this costs one `log level` verb.

### H5 — marginal timing closure at 74.25 MHz on an HX4K

| For | |
| --- | --- |
| The part is small and the library already pushes it | programs fail synthesis "over LC capacity" |
| HD runs at 74.25 MHz, SD at 27 MHz | a 2.75x margin difference between the two regimes |
| Would be program-specific and temperature-sensitive | matches "follows sustained sweeps" without matching any single verb |

| Against | |
| --- | --- |
| No warm-up trend across 6 minutes of continuous loading | §2 |
| Recovery in seconds via a timing bounce is too fast for a thermal effect | measured |

**Discriminator.** Repeat the sweep at 720x576i PAL over the analog input, which
uses the SD bitstreams at 27 MHz. The rig already has that chain. If the hazard
vanishes, the failure is clock-rate or receiver specific; if it persists, it is
in the load path.

### H6 — cumulative firmware resource exhaustion

Retained only to record that the journal argues against it: hazard flat against
load index (p = 0.73), gaps not shortening. Untested rather than excluded,
because `cpu` and `ram` were never sampled.

## 4. The owner's proposal, evaluated

*Load a known-simple passthrough program before loading each new program.*

**Not recommended as a preventive.** Three independent arguments, the first
decisive.

**The mechanism is architecturally excluded.** The proposal requires the
outgoing program's identity to affect the incoming load. Reconfiguration erases
the fabric in full, so the only surviving state is in the four ADV parts, the
clock tree and the RP2040. The FPGA has no I2C (**F**, `pin_map.pcf`) and the
MCU-to-FPGA ABI is write-only (**F**, `abi-format.md`), so a VHDL image program
has no channel whatsoever by which to alter receiver, transmitter or converter
state. There is no path along which "arbitrary bitstream" could differ from
"known-simple bitstream" as a *predecessor*.

**The deduction in §2 removes the target.** The hard fault's state is not in the
fabric, because a full reload cannot clear it. A passthrough bitstream is a
fabric intervention. It cannot reach the thing that is broken.

**The counting argument runs the wrong way.** The transient hazard is memoryless
per reconfiguration at 0.167. Interposing `Passthru` performs two
reconfigurations per useful program, raising expected hazards per useful program
from 0.167 to `2p - p²` = 0.31, and doubling exposure to the `0x2C0000xx` load
path. Under H3 it also doubles the number of timing-ID writes.

**One benefit survives, and it is diagnostic rather than preventive.**
`Passthru` is transparent — all twelve parameters report `Null` (`hardware.md`)
— so a liveness check on it is unambiguous: any failure is the chain. The
journal shows precisely the ambiguity this resolves, with `Ramp Logic`,
`Vectorscope` and `Plumber` scoring `ok: false` only for being dark. That
benefit is worth having once per sweep and on failure, not once per program.
`Passthru` is embedded in firmware rather than SD-backed (SDK CHANGELOG,
"Embedded `passthru_rgb`"), so using it as a canary avoids the SD read path.

**Could the fault sit in the receiver's initialisation after reconfiguration
rather than in the fabric?** Yes, and on this evidence it probably does — that
is H1, ranked first. Note that this makes the proposal worse rather than better:
if each load re-touches the receiver, doubling loads doubles the exposure.

**The one condition that would rescue it** is a hazard concentrated in
particular *outgoing* programs. The journal has one observation per program and
cannot test it. §6 tests it directly, which is the honest way to settle it.

## 5. Operating sequence

Ordered device verbs, all wrapped by `pyvmancer` today.

### Session bring-up, once

| # | Verb | Rationale |
| --- | --- | --- |
| 1 | `log level <debug>` | The firmware has a documented fault vocabulary — `0x083000xx` receiver, `0x081000xx` decoder, `0x2C0000xx` program load, `0x3400000x` routing — and nothing on this rig has ever read it. Unprefixed lines on the shell are log output (`protocol.md`). This is the cheapest instrument available and it separates H1, H3 and H4 by itself. |
| 2 | `version` | Fault behaviour is firmware-scoped; the sync path changed at rc.32, rc.33 and rc.39. |
| 3 | `fpga status` | Require `state: running`, `configured: true`, `bridge: true`. Starting from an idle FPGA yields a run of silent successes that only a power cycle clears. |
| 4 | `video status` | Record `timing` and the selected input's sub-status. Require `overridden: false` so the device is free to follow the source. Ignore `connected` — it reads false on a healthy link. |
| 5 | `video input hdmi` | Assert the input rather than inherit it; a previous session's timing bounce drops the selection. |
| 6 | `program load Passthru` | Transparent reference fabric, firmware-embedded, no SD read. Anything wrong from here is the chain. |
| 7 | *confirm frames externally, with motion* | The only proof. The ABI has no read path, so no device flag can substitute. |
| 8 | `cpu`, `ram` | Baseline, so a later reading is comparable. |

### Per program

| # | Verb | Rationale |
| --- | --- | --- |
| 9 | `program load <name>` | Capture the reply. `!code:message` and the log stream are the only place a fault code appears. |
| 10 | *confirm frames externally* | Wait for lock, then motion, against a deadline. Do not gate on `video status`: it reports a locked input while nothing passes. |
| 11 | on failure: `fpga status` | Branch before escalating. If `state` is not `running` or `program` is empty, **halt** — that is H4 and the ladder below cannot clear it. |
| 12 | `video input analog`; confirm | Diagnostic before recovery. Under H1 the analog path still works, because it runs a different bitstream on the ADV7181C LLC. This one verb tells you whether the receiver is the problem, and it costs a reconfiguration you were going to spend anyway. |
| 13 | `video input hdmi`; confirm | Completes the bounce. Recovered the fault once. |
| 14 | `video timing <alternate>`, `video timing <native>`, `video input hdmi`; confirm by polling | The strongest reset the shell offers, and under H3 the only rung that writes a timing ID differing from the one already believed. The timing change drops the input selection, hence the third verb. Leaves `overridden: true`, so the device stops following source format changes for the rest of the session — acceptable while format is a session constant. Lock flags lag the signal, so poll. |
| 15 | on further failure: **halt and require a power cycle** | Do not reload the program: measured to have no effect, and now explained — a reload cannot reach state outside the fabric. Recording black frames is worse than stopping. |
| 16 | `ram`, `cpu` | Once per load; the only way H6 gets tested at all. |
| 17 | *dwell on confirmed content before the next load* | The journal issued every load 0.45 s after the previous probe returned, so the input path was never given settled time. Under H3 this is the intervention with a mechanism, and it costs one reconfiguration rather than two. Its value is untested — §6 measures it. |

Rungs 12-14 are themselves reconfigurations and carry the same per-load hazard,
so each is confirmed independently and the ladder is retried a bounded number of
times before halting, rather than concluded after one pass.

### Standing prohibitions

| Rule | Basis |
| --- | --- |
| Never issue `settings export` or `settings get` | Both hang the command processor irrecoverably on rc.37; a USB reset does not clear it |
| Never treat `hdmi.connected` as diagnostic | False on a healthy link immediately after a power cycle |
| Never treat top-level `locked` as input health | It tracks genlock and reads false whenever timing is overridden |
| Never treat any device flag as proof of frames | The MCU-to-FPGA ABI is write-only; firmware reports intent, not state |
| Do not interpose `Passthru` before every load | §4 |
| Do not write `fpga register` blind | The register map is 0x00-0x1F write-only with no readback, so a wrong write is undetectable and unrecoverable short of a reload |
| Record the session's final `video status` | So the next session knows whether `overridden` was left true |

## 6. Experiment

Confirms or refutes H1 — that the hard fault is receiver state outside the
fabric, and that the per-load transient is a separate, program-independent,
memoryless failure — and settles the owner's proposal in the same run.
Unattended, scored offline.

### Preconditions

1. A second sink on the Videomancer HDMI output, recording signal presence
   independently of the AVerMedia card. Without it, H2 is not excluded and every
   number below describes the device-plus-card pair. If unavailable, record the
   card's own signal-detect separately from decoded frames and say so.
2. `log level` raised, with all unprefixed shell output captured to the journal
   alongside every `!code:message`. Half the hypotheses are separated by a fault
   code nobody has yet read.
3. `Session._note` extended to journal every verb, not only `load_program`.

### Arms

| Arm | Sequence per target program | Reconfigurations |
| --- | --- | --- |
| A, control | `program load <target>` | 1 |
| B, owner's proposal | `program load Passthru`, confirm, `program load <target>` | 2 |
| C, dwell | as A, with `D = 15 s` of continuously confirmed content before the next load | 1 |
| D, SD regime | as A, at 720x576i PAL over the analog input | 1 |

Targets drawn uniformly at random with replacement from `programs list`. Arms
interleaved in blocks of 10 rather than run in sequence, so time of day and
temperature do not align with arm. Arm D tests H5 and needs the existing Pi
composite chain, nothing new.

### Metrics

| Metric | Definition |
| --- | --- |
| **Primary — hazard per reconfiguration** | loads after which content is not confirmed within `T = 20 s`, over total loads |
| Hazard per useful program | same numerator over target programs completed. Reported alongside the primary: arm B spends two reconfigurations per program and a per-program metric alone would flatter it |
| Hard-fault rate | failures not cleared by rungs 12-14 of §5 |
| Time to confirmed content | per load, so a shifted distribution shows even when the hazard does not |

One JSONL record per load: arm, block, target, predecessor, whether the target
is SD-backed per `sd:/programs/manifest.json`, wall clock, elapsed session time,
load index, confirmation outcome and latency, escalation rung reached, every
fault code and log line emitted, `fpga status`, `ram`, `cpu`, second-sink signal
presence, and a `video status` trace sampled at 5 Hz from the load command until
confirmation or timeout.

### Size

Two-proportion test, alpha 0.05, power 0.8, against the measured p = 0.167:

| Effect | n per arm |
| --- | --- |
| Doubling to 0.31, arm B's prediction under H1 and H3 | 145 |
| Fall to 0.05, the owner's hoped effect | 106 |

150 loads per arm covers both. Arm B needs 75 targets for 150 reconfigurations;
A, C and D need 150 each. About 600 reconfigurations at ~5.5 s, plus 20 s
timeouts on roughly 100 expected failures, plus arm C's dwell: under 3 hours.

### Scoring

Wilson intervals on the hazard per arm, then logistic regression of failure on
arm, load index, elapsed session time, target, predecessor, and SD-backed.

| Coefficient or observation | Tests |
| --- | --- |
| arm B vs A, per reconfiguration | the owner's proposal in the direction that matters |
| arm C vs A | whether cadence is the controllable variable |
| arm D vs A | H5, clock rate and timing closure |
| load index | H6, accumulation |
| elapsed session time | H5, thermal |
| predecessor program | the only condition under which the owner's proposal has a mechanism |
| SD-backed | load path versus reconfiguration |
| Any `0x083000xx` code | H1 outright |
| Any `0x2C0000xx` code | H4 outright |
| Whether `hdmi.locked` ever drops during a load | receiver touched per load, or not |
| Whether the second sink loses signal at the same instants | H2 |
| On a hard fault: does `video input analog` alone carry video? | H1 — yes means the receiver is the broken part |

### Predictions

| Hypothesis | Prediction |
| --- | --- |
| H1 | Hard faults accompanied by a receiver fault code or by analog-only recovery; no arm effect on the transient hazard. |
| H2 | The second sink holds signal through the starvations, and the whole transient result is about the capture card. |
| H3 | Transient hazard equal across A and B per reconfiguration, so B's per-program hazard is ~2x A's; C below A; no predecessor effect. |
| H4 | `0x2C0000xx` codes on the idle occurrences, correlated with SD-backed targets. |
| H5 | Arm D materially below A. |
| H6 | Positive load-index coefficient. |
| Owner's | B's per-reconfiguration hazard materially below A's, plus a predecessor-program effect. |

H3 and the owner's hypothesis make opposite predictions on the primary metric,
so the experiment separates them rather than merely measuring both.

### Halt conditions

| Condition | Action |
| --- | --- |
| `fpga status.state != running` after a load | Stop. This is the event of real interest; record the full trace rather than continuing past it. |
| Escalation ladder exhausted | Stop. A power cycle is needed and the run cannot supply one. |
| Serial shell unresponsive | Stop. |
| Wall-clock cap exceeded | Stop and score what completed. |

## 7. What cannot be settled without the device

| Question | Single measurement |
| --- | --- |
| Device or capture card? | A second, independent sink on the HDMI output, observed during one fault |
| Which subsystem faults? | `log level` raised, and the fault code read on one failure |
| Is the receiver re-touched at program load? | `video status` sampled at 5 Hz across one load |
| Is the hard fault receiver-specific? | During a fault, `video input analog` alone — does composite carry video? |
| Does `video status` still answer when `fpga status` reads `idle`? | One `video status` in that state; it decides whether the receiver sits outside the reconfigured fabric |
| Do the failing programs share a core or a clock divisor? | Pull the `.vmprog` config blobs (TOC type 1) over `fs read` and parse them offline against `vmprog-format.md`; no video involved |
| Does firmware resource use grow across a sweep? | `ram` and `cpu` per load |
