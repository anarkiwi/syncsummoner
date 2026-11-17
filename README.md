# syncsummoner

Automated characterization of the LZX Videomancer, and generative offline video
processing driven by measured device behaviour plus musical structure.

Probes the instrument to build a per-program profile of what every parameter
actually does, then composes and renders against that profile rather than
against guesswork.

## Install

```sh
pip install syncsummoner            # full harness
pip install "syncsummoner[audio]"   # adds librosa for audio features
```

The perceptual metrics library is usable on its own, with no device stack:

```sh
pip install "syncsummoner[aesthetics]"
python -m syncsummoner.aesthetics score clip.mp4
```

## Use

```sh
syncsummoner device list | ping | programs
syncsummoner device doctor [--format 1080p30] [--loop] [--fix]   # rig diagnostics, JSON report
syncsummoner manifest /media/card/programs [--write]   # rebuild manifest.json from the .vmprog files
syncsummoner lockcheck source.mkv [--program NAME]   # check the locked frame clock end to end, exit 1 unless exact

syncsummoner program list   # programs on the card copy and firmware images
syncsummoner program show NAME [--state initial|PRESET|v1,..,v12] [--set P=V] \
                     [--timing ID] [--source rtl|bitstream] [--json]   # declared parameters, live bits and gates
syncsummoner program step NAME P A B [--state ...]   # can one step of P from A to B change the picture

syncsummoner probe run --program all --plan oat,sobol,tongue --out profiles/
syncsummoner probe sim --program blur --image zoneplate.png
syncsummoner probe archive --out archive/    # native frames for every program, resumable
syncsummoner probe archive --footage source.mkv --out archive/   # the same, against looped real footage
syncsummoner probe names --archive archive/   # add each program's program info to its archive
syncsummoner probe refit --archive archive/ --out profiles/   # profiles from the archive, no device

syncsummoner profile show | diff | atlas --program bitcrush_displace

syncsummoner analyze clip.mp4 --audio track.wav -o features.json

syncsummoner compose clip.mp4 --audio track.wav --profiles profiles/ \
                     --style glitchy --seed 7 --passes 8 -o score.yaml
syncsummoner score show score.yaml --profiles profiles/   # what each layer drives, flagged

syncsummoner audition score.yaml --source clip.mp4 --audio track.wav --seconds 30 -o audition.mp4
syncsummoner render score.yaml --source clip.mp4 --audio track.wav \
                    --cut-programs Derez,Lorenz,Scramble --fade 1.5 -o final.mp4
syncsummoner check score.yaml --source timecoded.mkv --takes takes/ \
                   --cut-programs Derez,Lorenz,Scramble [--splice cut.mp4]   # dead picture per take
syncsummoner check score.yaml --source timecoded.mkv --master cut.mp4   # the same, on a splice
syncsummoner judge takes/take-*.mkv --source timecoded.mkv [--score score.yaml] [--jobs 2]   # offline: lock, evidence, picture, bars
syncsummoner evidence takes/take-Derez.mkv --source timecoded.mkv --predict   # the same; each verdict against the program model
```

Every command logs each stage and its elapsed time to stderr (`-v` for detail,
`-q` for silence) and draws a progress bar where the work is countable; stdout
carries only what was written. A render is the length both inputs cover, muxes
the track back in, and fades up from black and silence and back down.
[docs/workflow.md](docs/workflow.md) is the whole thing end to end, on the rig,
through `docker run`.

## Layout

| Package | Role |
| --- | --- |
| `syncsummoner.program` | each program's declared parameters, and their liveness from its VHDL or bitstream |
| `syncsummoner.device` | pyvmancer transport, session discipline, capture, playout |
| `syncsummoner.probe` | stimulus battery, sweep plans, runner, frame archive, offline replay, GHDL simulation, profile fitting |
| `syncsummoner.compose` | audio and video features, gesture vocabulary, planner, score IR, render |
| `syncsummoner.aesthetics` | perceptual metrics; imports nothing from the rest, extraction-ready |

## Docs

- [docs/workflow.md](docs/workflow.md) — worked example: clip and track in, finished render out
- [docs/design.md](docs/design.md) — the design being realized
- [docs/program-model.md](docs/program-model.md) — each program's parameters from its `.vmprog`, and which are live from its HDL
- [docs/hardware.md](docs/hardware.md) — measured rig behaviour and its consequences
- [docs/evidence.md](docs/evidence.md) — frame-locked parameter writes and the per-take proof they landed
- [docs/judging.md](docs/judging.md) — judging finished takes offline from one decode per recording
- [docs/contracts.md](docs/contracts.md) — normative cross-package signatures
- [docs/rig.md](docs/rig.md) — wiring, permissions, and bring-up

## Development

```sh
docker pull anarkiwi/syncsummoner                          # the released image
docker build --target runtime -t anarkiwi/syncsummoner .   # the working tree over it
docker build --target lint .
docker build --target test -t syncsummoner-test . && docker run --rm syncsummoner-test
docker build --target test-aesthetics -t ss-aes . && docker run --rm ss-aes
docker build -f Dockerfile.gpu -t syncsummoner:gpu .        # Vulkan archive decoding (docs/workflow.md)
```

The `test-aesthetics` target installs only the aesthetics extra. It failing is
the early warning that the extraction seam has leaked.

The `runtime`, `lint` and `test` images build on the `hdl` target, which carries
the HDL tools and upstream sources the program model reads.

Pushing a `v*` tag runs the test target and publishes `runtime` to Docker Hub as
`anarkiwi/syncsummoner`; it needs the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`
repository secrets.

## License

Apache-2.0
