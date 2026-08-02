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

syncsummoner profile show | diff | atlas --program bitcrush_displace

syncsummoner analyze clip.mp4 --audio track.wav -o features.json

syncsummoner compose --profiles profiles/ --features features.json \
                     --style glitchy --seed 7 -o score.yaml

syncsummoner audition score.yaml --seconds 30
syncsummoner render score.yaml --source clip.mp4 --passes 2 -o out.mov
```

## Layout

| Package | Role |
| --- | --- |
| `syncsummoner.device` | pyvmancer transport, session discipline, capture, playout |
| `syncsummoner.probe` | stimulus battery, sweep plans, runner, frame archive, GHDL simulation, profile fitting |
| `syncsummoner.compose` | audio and video features, gesture vocabulary, planner, score IR, render |
| `syncsummoner.aesthetics` | perceptual metrics; imports nothing from the rest, extraction-ready |

## Docs

- [docs/design.md](docs/design.md) — the design being realized
- [docs/hardware.md](docs/hardware.md) — measured rig behaviour and its consequences
- [docs/signal-loss.md](docs/signal-loss.md) — video path, and diagnosis of the intermittent input fault
- [docs/contracts.md](docs/contracts.md) — normative cross-package signatures
- [docs/rig.md](docs/rig.md) — wiring, permissions, and bring-up

## Development

```sh
docker build --target lint .
docker build --target test -t syncsummoner-test . && docker run --rm syncsummoner-test
docker build --target test-aesthetics -t ss-aes . && docker run --rm ss-aes
```

The `test-aesthetics` target installs only the aesthetics extra. It failing is
the early warning that the extraction seam has leaked.

## License

Apache-2.0
