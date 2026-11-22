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
syncsummoner program cover homegrownvhs/Pigment                  # what makes each parameter live, and the base states that step all
syncsummoner program survey czartroniks/Ridgeline                # build and time the analyses judging its takes reads
syncsummoner evidence takes/take-czartroniks_Quilt.mkv --source timecoded.mkv
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
m.reach(state, [4, 9])        # live bits with P4 and P9 both free: dead here is dead for every value of both

from syncsummoner.program.gates import cover
c = cover(m)                  # gating structure and covering base states, from the initial state
c.gating()                    # {parameter_id: ["dead at the reference", "live when P9 in 0..511", ...]}

from syncsummoner.program.gates import structure
structure(m)                  # {parameter_id: ["P9 in 0..511", ...]}: every setting of the others that makes it live
c.chosen, c.stepped           # the base states to sweep, and the parameters stepped in each
c.vectors()                   # (from, to) full states in sweep order
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

Yosys runs once more per build (`hdl.lower`): the same `opt`, `techmap`, `abc`
flow, with every `registers_in` bit and every undefined bit made an input, to
Yosys's gate library. The result is cached as `gates.json`. Every liveness
question is then answered in-process from it (`program.observe`), with no tool
process per query.

**Symbolic values.** Each net's value is a partition of the control
assignments into the leaf the net takes there: 0, 1, a literal (a video or
other input bit, a register output or a black-box output, or its complement),
or *any* function. Each part is a BDD (`dd` over CUDD, sifting reorder) over the
`registers_in` bits of words 0-8, the only symbolic variables. Registers 9-11
are free inputs and words 12-31 are zero. Gates combine leaves exactly where
they can (`x & 0 = 0`, `x ^ 0 = x`, `s ? x : x = x`, `x & ~x = 0`), so constant
propagation and the identity of a passed-through bit are both exact over every
control setting at once. A function of two different free signals becomes
*any*. This is the abstraction: it never claims a dependence is absent when it
is not.

**Queries.** A query frees some controls and ties the rest to a state. One
analysis serves every query with the same freed set. The freed controls are
*varying*: they may change while the rest are held, so no register settles on
their value. Every control stays a BDD variable, so the analysis is exact in
the freed controls too. Analyses are cached per freed set and built on demand,
one process per CPU when a search needs several at once (`ProgramModel.prepare`).

**Registers.** Every storage cell starts free, and the fixpoint below never
lets a free output feed its own rule.
- **Loads.** Where the next state, with every enable taken to fire, is the
  same constant for every value of the free signals and varying controls,
  the output is that constant. This is the steady state a held setting
  reaches.
- **Keeps.** Otherwise, where the next state only ever keeps the output or
  gives its initial value, the output is the initial value.
- **Otherwise** the output is free. It is a fresh BDD variable when its next
  state reads only controls and itself, as a pipeline of a varying control
  does, so the logic it feeds keeps its correlations. It is a literal
  otherwise.

A register settled under some controls stays settled there. The fixpoint is the
least one, and rounds re-evaluate only the cones that changed. Taking a free
register to hold any value is the sequential over-approximation. It errs
toward live.

**Observability.** A source is a control bit, or the video input bits
together. A forward pass to a fixpoint marks, for each net, the controls under
which flipping the source can change it:
- where the net's two cofactors on the source take different leaves;
- where they share a literal whose net can change;
- where they are both *any* function and a pin the cell is sensitised to can
  change. A pin is sensitised where its two values give different outputs with
  the others held, each other *any* leaf standing for its own literal;
- where it depends on a register variable that can change.

A free register can change where its next state can, for some value of the
varying controls and register variables: the difference it takes on outlives the
cycle that made it, while those move and every register steps on, so only the
held controls condition it. A black box (memory, iCE40 RAM, a gate on a
combinational loop) can change wherever any input can. The source's
condition is the union over the outputs, with register variables quantified
out. A select whose arms agree is dead there even when the arms are separate
gates. **Live** bits of a parameter at a state are those whose condition holds
for some value of the freed controls with the rest tied. That is one BDD
restriction per bit. `ProgramModel.revivable` asks the same of a held setting
of other controls, and the cover search uses it.

**Fallback.** An analysis may take on the memory the process may still use
when it starts: `MemAvailable`, cut to the cgroup's `memory.max`, split evenly
among analyses run at once. Its bound is the resident memory the process has
gained since, read from `/proc/self/statm` as each net is evaluated. Conditions
share their nodes in one CUDD manager, so no one condition's size bounds it.
`observe.exhausted` in the cache records the budget a program outgrew; a later
run with a larger budget tries the BDDs again. `program survey NAME... | all
[--jobs N]` builds every analysis judging a take reads, by BDDs alone, and
prints one JSON line per program with the seconds and peak memory it took, or
the budget it outgrew. Past its budget, the SAT miter (`program.sat`) answers
instead, query by query, on one incremental CaDiCaL instance (PySAT):

- **Miter.** Two copies of the gate-level netlist share every free signal
  (control bit, input bit, storage output) as a pair of variables held equal
  unless the pair's enable is left free. A query is a set of assumptions: the
  tie set on copy *a*, the source bit 0 in *a* and 1 in *b*, the settled
  storage at its values, and the enables of the storage that can differ. A
  black box's outputs are equal in both copies wherever all its inputs are.
- **Registers.** The rules above, applied exactly: a storage bit loads a
  constant where its next state with every enable fired is that constant for
  every value of the free signals and untied controls, else keeps its initial
  value where its next state from that value is that value, to the least
  fixpoint. Random concrete lanes (64-bit words) refute most candidates; the
  nets feeding the rest that the lanes show constant and propagation does not
  are proven constant in topological order, then the candidates themselves.
  A tie set that differs from a settled one only in some controls decides
  again only the storage depending on them. A free storage bit can differ
  where its next states can, to the least fixpoint, grown from lanes that
  differ and then from models.
- **Observability.** A source is observable where some model makes an output
  differ. The ternary simulator first screens every tie set in lanes, which
  proves most dead sources, and bounds the rest to the nets their sensitised
  pins reach: nets outside cannot differ, so their copies are assumed equal.
  Random lanes then find most live sources. The nets that never differ in a
  lane are proven equal in topological order, each from those before, while
  the output query races them at a doubling share of the conflicts spent.
- **Held controls.** A tie set settles storage only on its tied bits. A
  control left untied but held is free there, except that storage whose next
  state reads only tied and held controls, through such storage, is linked to
  that next state: it settles to it at every held setting. This
  over-approximates holding the control. A held setting is decided exactly by
  tying it: `ProgramModel.revivable` and the cover's revivals refute a set of
  gates with its controls held untied, then complete it gate by gate over
  their resolution classes, the first class the rest can still complete kept,
  as the conditions do. The classes are screened in lanes first.

SAT and the BDDs agree wherever the BDDs' leaves are exact; where a function of
two free signals cancels (`x ^ x`), the miter calls dead what the BDDs call
live. Only where the miter cannot be built does the vectorised ternary simulator
(`program.ternary`) answer, one lane per tie set: the same register rules and
exact local sensitisation, every free signal X, so it can call live what the
others call dead. `ProgramModel.method` says which answered, and the log names
it. `--method bdd|sat|ternary` on the `program` commands forces one.

- **Resolution** is the live set with every register free.
- **Gates** of a parameter dead in a state are the other controls (and timing)
  whose freeing brings it back.
- **Step check** `a -> b`, on demand for `program step` only:
  1. No change if the two values differ only in bits dead in that state.
  2. Otherwise both states are folded by Yosys, and the result is no change if
     the two netlists are the same circuit. This is shown by an explicit cell
     mapping, found by colour refinement anchored on port names and then
     checked on every type, parameter, init value and connection.
  3. Otherwise no change if Yosys proves them equivalent (`equiv_struct`,
     `equiv_simple`, and `equiv_induct` where no memory is left).
  4. Otherwise changes, if a shared random stimulus from power-up drives the
     two to different outputs in co-simulation. That is a concrete witness.
  5. Otherwise undetermined, said as such. Typical cases are effects that need
     structured frames rather than random sync, and slow accumulators.

The netlist, `gates.json` and the conditions (`observe.json`) are cached under
`SYNCSUMMONER_PROGRAM_CACHE` (default `.program-cache/`, gitignored), keyed by
program id, version, package digest, netlist source and IO configuration. Step
verdicts are cached there by their exact ties. Conditions, resolutions and
bypasses also carry the revision of the liveness rules (`observe.RULES`), so a
change to the rules never reads what the old ones derived.

`program crosscheck NAME... | all [--jobs N]` answers each program's queries (the
initial state and every preset, each declared control freed alone and with every
knob) by the BDDs, the miter and the ternary simulator, and prints one JSON line
per program with the seconds each took and every query on which they differ. The
miter is exact under the rules, so a bit it finds live that another finds dead is
that method's fault, listed in `faults`; the command exits 1 if any program has one.

## Bypass

`ProgramModel.bypass()` finds the one control setting that returns the input with
every other declared control dead. It returns it as a `Bypass`: the control, the
manual value to write, and every manual value the proof covers.

1. A declared switch comes first. Turned on, it must leave every other control
   dead with the rest held at the state, and the video input must still reach
   the outputs (`carries_input`).
2. If there is no such switch, any control can qualify through `passes()` at a
   setting. The input must reach the outputs there, and no bit of any other
   declared control may be observable for any value of those controls taken
   together. The candidate's own bits stay BDD variables. The other controls'
   conditions are ORed and their variables quantified away
   (`Conditions.heard`), so each register value of the control is decided in
   one pass. The joint analysis, with every other control varying, is built
   only for a control that first passes the cheaper test of each other control
   being dead alone.

This finds a Mix or Blend fader at the end of its range that returns the input
exactly, such as the SDK's `interpolator_u` at `t = 0`. It does not find an end
that only approaches the input. A fader's bypass is written at the middle of the
longest run of proven values. Names and roles play no part. `syncsummoner
program bypass NAME [--json]` prints the bypass.

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

## Gating and the covering sweep

`program.gates.cover` answers, for every parameter the program reads, what makes
it live, and picks the fewest base states that step every one where it is.

- **Live** at a state: some bit of the parameter is observable there.
- **Revival.** For a parameter that is dead, finding it dead for every held
  setting of all the other controls rules out any revival
  (`ProgramModel.revivable`). Otherwise single controls are tried, each
  refuted exactly when no held setting of it revives the parameter. Each control left is evaluated over every setting it can tell apart
  (its resolution classes): the condition restricted to the state, then to each
  class. The settings that revive it are merged into maximal intervals of
  manual values. Only when no single control revives it are pairs tried, then
  triples, with a witness setting found by tying one control at a time.
  A parameter that is live only while another control moves, such as
  RamBend's Pot 6 Page with P6, has no held revival. `program show` names the
  gate, and the cover lists the parameter's bits as never covered.
- **Structure.** `gates.structure` states each parameter's condition as
  disjoint branches over the other controls in parameter order. In each branch,
  a control's manual values are grouped by the condition they leave, as
  maximal intervals. The table form of `program cover` prints it.
- **States.** Starting from the reference (the declared initial values), each
  revival's move gives a new state, and revivals are followed from every state
  reached, each control moved at most once along a path. Every query is cached.
- **Cover.** The cover must reach every bit each parameter reads, live in some
  stepped state, and, for every gate found, the parameter live with that gate in
  a reviving interval. Binary `y[s]` picks a state and `z[s, p]` steps `p` in it
  (`z <= y`). Each element needs one covering `z`. The objective weights a state
  above every step combined, so it minimises states first and then steps. It is
  solved exactly by `scipy.optimize.milp`. Elements no state reaches are reported
  as never covered.

`probe archive` and `probe sweep plan` take their sweep from the cover
(`Cover.vectors`). It holds each chosen base state, then steps each parameter
assigned to it through every setting it shows there, one at a time, and returns
to the base before the next. The settings are walked by `gates.tour`: evens
ascending then odds descending from the base value, a closed walk that moves
`2 * (max - min)` in all, the least any walk reaching both ends can, and never
more than two settings at a step, so no step is a jump across the range. Every setpoint is the full declared state, and the archive's `schedule`
records each setpoint's state together with the one it was stepped from. A
`--schedule` focus steps its parameter in a base state where the cover makes it
live. The HDL tool processes of a build and of `program step` share one bound
sized to the CPUs the process may use: the cgroup `cpu.max` quota, else its
affinity (`hdl.cpus`).

## Integration

- `syncsummoner evidence` judges every step of a take with `predict.classify`
  and tests only the steps the model leaves live ([evidence.md](evidence.md)).
- `compose.planner.plan_automation` passes each layer through its program's gate
  (`compose.gating`). A write to a parameter with no live bit in the state the
  plan holds then is written together with a gate move from the cover that makes
  it live there, or is refused.
- `ProgramModel.schedule(state)` gives one manual value per setting the program
  can see in a state.
