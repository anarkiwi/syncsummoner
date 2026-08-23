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

syncsummoner probe run --program all --plan oat,sobol,tongue --out profiles/
syncsummoner probe sim --program blur --image zoneplate.png
syncsummoner probe archive --out archive/    # native frames for every program, resumable
syncsummoner probe refit --archive archive/ --out profiles/   # profiles from the archive, no device

syncsummoner profile show | diff | atlas --program bitcrush_displace

syncsummoner analyze clip.mp4 --audio track.wav -o features.json

syncsummoner compose clip.mp4 --audio track.wav --profiles profiles/ \
                     --style glitchy --seed 7 --passes 8 -o score.yaml

syncsummoner audition score.yaml --source clip.mp4 --audio track.wav --seconds 30 -o audition.mp4
syncsummoner render score.yaml --source clip.mp4 --audio track.wav \
                    --cut-programs Derez,Lorenz,Scramble --fade 1.5 -o final.mp4
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
| `syncsummoner.device` | pyvmancer transport, session discipline, capture, playout |
| `syncsummoner.probe` | stimulus battery, sweep plans, runner, frame archive, offline replay, GHDL simulation, profile fitting |
| `syncsummoner.compose` | audio and video features, gesture vocabulary, planner, score IR, render |
| `syncsummoner.aesthetics` | perceptual metrics; imports nothing from the rest, extraction-ready |

## Docs

- [docs/workflow.md](docs/workflow.md) — worked example: clip and track in, finished render out
- [docs/design.md](docs/design.md) — the design being realized
- [docs/hardware.md](docs/hardware.md) — measured rig behaviour and its consequences
- [docs/contracts.md](docs/contracts.md) — normative cross-package signatures
- [docs/rig.md](docs/rig.md) — wiring, permissions, and bring-up

## Development

```sh
docker pull anarkiwi/syncsummoner                          # the released image
docker build --target runtime -t anarkiwi/syncsummoner .   # the working tree over it
docker build --target lint .
docker build --target test -t syncsummoner-test . && docker run --rm syncsummoner-test
docker build --target test-aesthetics -t ss-aes . && docker run --rm ss-aes
```

The `test-aesthetics` target installs only the aesthetics extra. It failing is
the early warning that the extraction seam has leaked.

Pushing a `v*` tag runs the test target and publishes `runtime` to Docker Hub as
`anarkiwi/syncsummoner`; it needs the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`
repository secrets.

## License

Apache-2.0
