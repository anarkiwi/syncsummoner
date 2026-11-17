# Program model

What each program's parameters are, which register bits they drive, and which of
those bits can still change the picture in a given state, all read from the
program itself: its `.vmprog` package for the declarations, and its VHDL or its
iCE40 bitstream for liveness. Nothing is inferred from a parameter's name, slot
or range.

```sh
syncsummoner program list
syncsummoner program show czartroniks/Quilt                       # at its declared initial values
syncsummoner program show Pinwheel --state "Luma Rainbow" --set 11=1023
syncsummoner program step czartroniks/Quilt 3 512 518 --state 512,512,512,262,512,512,0,0,0,0,0,512
syncsummoner evidence takes/take-czartroniks_Quilt.mkv --source timecoded.mkv --predict
```

```python
from syncsummoner.program import model

m = model("czartroniks/Quilt")
m.params                      # ParamDecl per declared control, exactly as packaged
m.field(12)                   # r7[9:0]
m.resolution()                # {parameter_id: bits the program reads at all}
m.live(state)                 # {parameter_id: Liveness(live_bits, gated_by)}
m.step_matters(state, 3, 512, 518)   # Step(matters=True | False | None, how)
m.schedule(state)             # {parameter_id: one manual value per setting the program can tell apart}
```

## Declarations

`syncsummoner.program.vmprog` reads the package layout of the SDK's
`src/lzx/videomancer/vmprog_format.hpp`: the header, the table of contents with
every section's BLAKE2b-256 checked, the 7936-byte config (program id, version,
core, twelve parameter declarations, eight presets) and the bitstreams, inflating
those flagged `compressed_deflate` (raw DEFLATE, 1 KiB window). The tests build
packages with the SDK's own `toml_to_config_binary.py` and `vmprog_pack.py` and
read them back.

A parameter's kind comes from its declaration alone: `boolean` for control mode
`boolean` or two value labels, `quantized` with N levels for N > 2 value labels
or a `steps_N` mode, `continuous` otherwise, and `unused` for a control the
program does not declare.

The firmware writes the manual value through the SDK's
`apply_parameter_control_curve` (`vmprog_parameter_utils.hpp`); min and max
scale only the display. Value-labelled knobs declare `0..N-1` and their programs
read the top bits of the raw word, which the raw value reaches and the scaled
one never would.

## Register map

Controls reach the program as `registers_in` words: pots 1-6 at SPI addresses
0-5, switches 7-11 at bits 0-4 of address 6, the fader at address 7, video timing
at address 8 (SDK `videomancer_abi.hpp`). `core_top.vhd` latches SPI address *i*
into `registers_in(i)` on each HSYNC falling edge for words 0-11 and leaves 12-31
at zero. The bitstream path below confirms this map on every program it
decompiles, from the hardware's own behaviour.

## Liveness

A netlist of the program is built once per package and IO configuration:

- **From source** (community and SDK programs whose `program_id` and version
  match a published TOML): GHDL analyses the SDK core's `program_top` entity,
  the common RTL, the IO configuration package and the program's VHDL; the
  Yosys GHDL plugin elaborates, flattens and keeps memories whole.
- **From the bitstream** (programs with no published source): `iceunpack` and
  `icebox_vlog` decompile the package's bitstream for the IO configuration into
  the whole chip's netlist, named by the rev B pin map. Icarus then simulates
  it while SPI frames are driven onto the RP2040 pins exactly as
  `spi_peripheral.vhd` receives them (address, write bit, data, MSB first) with
  random words, and HSYNC is pulsed. The FFs holding each `registers_in` bit are
  the ones that show the previous word before the HSYNC edge and the new one
  after it, across every trial. The SPI RAM FFs already show the new word, so
  they are excluded. Those FFs are cut out and replaced by a `registers_in`
  port, which leaves the same interface as the source path. A bit with no FF is
  one the synthesised design does not read.

A published source is used only when it reads exactly the bits the package's own
bitstream latches. Otherwise the bitstream is used, and the mismatch is reported
as the model's `note`. The same program id and version do not guarantee that the
published VHDL is what was built.

For a state, every register bit is tied to the constant the state writes,
except the bits under question, which stay free inputs. Undefined constants also
become free inputs, as do registers 9-11 (core-only) and register 8 unless a
timing is given. Yosys then runs `opt`, `techmap` and `abc` to a fixpoint. Each
round also replaces any FF whose next value is fixed, once its enables have
fired, with that constant: a plain or enabled FF with a constant D, or a reset FF
whose D equals its reset value. This is the steady state a held parameter
vector reaches. A bit is **live** when it is still in the structural fan-in cone
of a video output. Dead is exact: no output depends on the bit. Live is
structural: the bit is wired to the picture, which does not by itself prove a
step on it changes it.

- **Resolution** is the live set with every register free.
- **Gates** of a parameter dead in a state are the other controls (and timing)
  whose freeing brings it back.
- **Step check** `a -> b`:
  1. No change if the two values differ only in bits dead in that state.
  2. Otherwise both states are folded, and the result is no change if the two
     netlists are the same circuit. This is shown by an explicit cell mapping,
     found by colour refinement anchored on port names and then checked on every
     type, parameter, init value and connection.
  3. Otherwise no change if Yosys proves them equivalent (`equiv_struct`,
     `equiv_simple`, and `equiv_induct` where no memory is left).
  4. Otherwise changes, if a shared random stimulus from power-up drives the
     two to different outputs in co-simulation. That is a concrete witness.
  5. Otherwise undetermined, said as such. Typical cases are effects that need
     structured frames rather than random sync, and slow accumulators.

Results are cached under `SYNCSUMMONER_PROGRAM_CACHE` (default
`.program-cache/`, gitignored), keyed by program id, version, package digest,
netlist source and IO configuration, then by the exact ties of each query.

## Finding programs

`Library.default()` reads the card copy (`captures/card/programs`,
`SYNCSUMMONER_CARD`), the firmware images (`captures/firmware/*.uf2`,
`SYNCSUMMONER_FIRMWARE`; the packages built into the firmware are found inside
the flattened flash image), the SDK (`SYNCSUMMONER_SDK`) and the community
programs (`SYNCSUMMONER_COMMUNITY`). Programs are found by device name
(`author/Name`, or the bare name for a firmware built-in) or by program id.
`Transport.program_info()` takes each parameter's kind from the declaration
found this way and refuses a program it cannot find.

## Tools

The `hdl` Docker stage installs a pinned OSS CAD Suite (Yosys, GHDL with its
Yosys plugin, IceStorm, Icarus) and the SDK and community sources at pinned
commits. The lint, test and runtime targets build on it. Outside Docker, set
`SYNCSUMMONER_HDL_BIN` to the suite's `bin`.

## Integration

- `syncsummoner evidence --predict` adds to each parameter's row the planned
  steps the model calls live and dead, and a check: predicted live and moved,
  predicted live and did not move, predicted dead and did not move, or a
  contradiction (predicted dead in every step, yet moved).
  `Evidence.contradictions` lists the parameters with a contradiction.
- `ProgramModel.schedule(state)` gives a planner or harvest one manual value per
  setting the program can see in a state. `predict.steps` and
  `ProgramModel.visible` classify any planned step. The harvest sweep does not
  use them yet, so wiring them into `probe.harvest.sweep_vectors` is the
  follow-up.
