# Judging a take

A finished take is judged from its files alone: the recording, its `.v4l2` frame
log, its `.lock.npz` flip log and its `.wire.npz` wire record. No device is
opened, so one process can capture the next take while another judges the last.

```sh
syncsummoner judge takes/take-*.mkv --source timecoded.mkv \
    [--score score.yaml] [--jobs 2] [--workers N] [--predict]
```

Each take gets, beside it:

| File | What |
| --- | --- |
| `<take>.picked.npy` | per clip frame, the capture showing it |
| `<take>.evidence.json` | every write sent, landed, read back and acted upon ([evidence.md](evidence.md)) |
| `<take>.schedule.json` | the undetermined parameters and the steps a sweep needs |
| `<take>.check.json` | with `--score`, every bar judged against the clip, as a render's take check |
| `<take>.meta.json` | under `judged`: the lock, evidence and picture outcome |
| `<take>.features.npz` | the decode everything above was read from |

Exit 2 when a take does not lock, 1 when its evidence or check fails, else 0.
`syncsummoner evidence` is the same command.

## One decode per recording

`syncsummoner.probe.decoded.features(path, spec)` decodes a recording once to,
per capture: its presentation time, its strip code, survival-size (160x90) luma
and calibration-size (96x54) planar YUV, plus the frame log. Attribution,
evidence, the picture check, the take check and lockcheck's tear check all read
this; nothing decodes a recording twice. The played clip is decoded the same way,
once per clip.

The decode keeps the full-resolution strip rows: each branch of the one filter
graph crops before it converts, so the strip is read exactly as the capture
stored it and no branch converts the whole frame. The body is area-scaled, and
MJPEG `-lowres` is not used: its DCT-domain reduction gives different samples
from an area scale, and the strip needs every row.

The MJPEG decoder runs on one core, so the recording is cut at keyframes (every
frame of an MJPEG take, every GOP of an H.264 clip) into one segment per core.
Each segment is seeked exactly (`-copyts -ss`), decoded by its own single-threaded
ffmpeg, and keeps only the captures timed before the next segment starts; the
joined times are checked against the demuxer's own packet times. The strip codes
are read in numpy over blocks of captures as each segment streams. On a 16-core
host this out-runs NVDEC's MJPEG decode, which also scales differently from
swscale's area filter.

The result sits in `<recording>.features.npz`, keyed by the SHA-256 of the
recording and its frame log (taken while the decode runs), the feature schema
version and the spec. The size and modification time it was hashed at let an
untouched recording skip rehashing; a changed one is rehashed, and decoded afresh
only if its bytes changed. A cache that cannot be written is skipped with a warning.

## Parallel judging

`--jobs N` judges N takes at once, each in its own process, the cores shared
between their decoders (`--workers` overrides) and BLAS held to two threads per
process. The clip is decoded first, once, so no two jobs decode it side by side.
Judging next to a capture takes cores from it; run the judge under `nice` there.
