"""Decode backends: the GPU path's argv, the automatic choice and its fallback, and the checks."""

import hashlib
import shutil
import subprocess

import numpy as np
import pytest

from syncsummoner.probe import decoders
from syncsummoner.probe.archive import DECODERS, PROBE_FRAMES, ArchiveError
from tests.probe.test_archive import KEY, archive_at, irregular_recording, write_archive

DOWNLOAD = "hwdownload,format=yuv422p"


def vulkan_reader(tmp_path):
    """A committed archive's reader set to decode on the GPU."""
    archive, made, _ = write_archive(tmp_path, count=4)
    return archive.reader("bitcrush", decode="vulkan"), made


def test_a_cpu_reader_asks_for_no_hardware(tmp_path):
    """The CPU backend is the decoder as it always was: no device, nothing downloaded."""
    archive, _, _ = write_archive(tmp_path)
    argv = archive.reader("bitcrush").command(count=2)
    assert "-hwaccel" not in argv and "-vf" not in argv and archive.decode == "cpu"


def test_a_vulkan_reader_decodes_on_the_device_and_downloads_before_any_filter(tmp_path):
    """Frames stay on the GPU unless downloaded, so the download leads every filter chain."""
    reader, _ = vulkan_reader(tmp_path)
    argv = reader.command(start=1, count=2, width=16)
    flags = argv[: argv.index("-i")]
    assert all(flag in flags for flag in DECODERS["vulkan"]), "device and hwaccel are input options"
    assert flags[flags.index("-hwaccel_output_format") + 1] == "vulkan"
    assert argv[argv.index("-vf") + 1] == f"{DOWNLOAD},scale=16:-2"
    assert reader.command()[reader.command().index("-vf") + 1] == f"{DOWNLOAD},null"


def test_a_vulkan_one_pass_graph_downloads_once_ahead_of_the_split(tmp_path):
    """Both outputs of the split take the one downloaded frame, so neither redoes the transfer."""
    reader, _ = vulkan_reader(tmp_path)
    seen = []

    def popen(argv, **_kwargs):
        seen.append(argv)
        raise OSError("stop")

    reader._popen = popen  # pylint: disable=protected-access
    for native in ([], [1]):
        with pytest.raises(OSError):
            list(reader.passes(width=16, native=native))
    graphs = [argv[argv.index("-filter_complex") + 1] for argv in seen]
    assert graphs[0] == f"[0:v]{DOWNLOAD},scale=16:-2[s]"
    assert graphs[1].startswith(f"[0:v]{DOWNLOAD},split[a][b];") and graphs[1].count("hwdownload") == 1


def test_the_download_is_to_the_format_the_archive_stored(tmp_path):
    """The stored format comes from the archive's own metadata, not an assumption."""
    archive, _, meta = write_archive(tmp_path)
    meta["stored_pix_fmt"] = "yuv444p"
    reader = type(archive.reader("bitcrush"))(
        archive.paths("bitcrush")[0], meta, [], ffmpeg="ffmpeg", run=None, decode="vulkan"
    )
    assert "hwdownload,format=yuv444p," in reader.command()[reader.command().index("-vf") + 1]


class Decoders:
    """``subprocess.run`` stand-in answering per backend, on a clock it moves as it decodes.

    Each backend has a start-up cost and a cost per frame, so a rate taken from two
    decode lengths recovers the per-frame cost alone.
    """

    def __init__(self, cpu=(0, b"frames", b""), vulkan=(0, b"frames", b""), pace=None):
        self.answers = {"cpu": cpu, "vulkan": vulkan}
        self.pace = pace or {"cpu": (0.1, 0.02), "vulkan": (2.0, 0.01)}
        self.calls = []
        self.now = 0.0

    def __call__(self, argv, capture_output=False, check=False):
        del capture_output, check
        backend = "vulkan" if "-hwaccel" in argv else "cpu"
        self.calls.append((backend, argv))
        setup, per_frame = self.pace[backend]
        self.now += setup + per_frame * int(argv[argv.index("-frames:v") + 1])
        code, out, err = self.answers[backend]
        return subprocess.CompletedProcess(argv, code, stdout=out, stderr=err)

    def timer(self):
        """Seconds of decoding so far."""
        return self.now


def probed(tmp_path, choice="auto", **answers):
    """The archive's choice of backend under the given answers, and what it ran."""
    archive, _, _ = write_archive(tmp_path)
    archive.run = Decoders(**answers)
    return archive.choose_decode(choice, timer=archive.run.timer), archive.run.calls


def test_auto_takes_the_gpu_when_it_decodes_the_same_bytes(tmp_path):
    """Lossless means identical: the GPU is trusted only once it has matched the CPU."""
    (decode, why), calls = probed(tmp_path)
    assert decode == "vulkan" and "bit-identical" in why
    assert "100.0 frames/s per stream against the CPU's 50.0" in why, "start-up is cancelled out"
    assert [backend for backend, _ in calls] == ["cpu", "cpu", "vulkan", "vulkan"]
    assert [argv[argv.index("-frames:v") + 1] for _, argv in calls] == ["1", str(PROBE_FRAMES)] * 2


def test_auto_keeps_the_cpu_when_the_gpu_is_identical_but_slower(tmp_path):
    """One stream per worker on both, so a slower stream is a slower refit."""
    slow = {"cpu": (0.1, 0.02), "vulkan": (0.5, 0.2)}
    (decode, why), _ = probed(tmp_path, pace=slow)
    assert decode == "cpu" and "slower" in why and "5.0 frames/s" in why


def test_asking_for_vulkan_takes_it_even_when_slower(tmp_path):
    """An explicit request is honoured on pace; only a wrong or failed decode refuses it."""
    (decode, why), _ = probed(tmp_path, "vulkan", pace={"cpu": (0.0, 0.01), "vulkan": (0.0, 1.0)})
    assert decode == "vulkan" and "1.0 frames/s" in why


def test_auto_falls_back_with_the_decoders_own_reason(tmp_path):
    """A host with no GPU says why in ffmpeg's words, and the refit goes on on the CPU."""
    stderr = b"[Vulkan] Instance creation failure\nDevice creation failed: -542398533.\n"
    (decode, why), _ = probed(tmp_path, vulkan=(1, b"", stderr))
    assert decode == "cpu" and why.endswith("Device creation failed: -542398533.")


def test_auto_refuses_a_gpu_that_decodes_differently(tmp_path):
    """A GPU decode that differs by a byte is a wrong decode, whatever its speed."""
    (decode, why), _ = probed(tmp_path, vulkan=(0, b"framez", b""))
    assert decode == "cpu" and "differs" in why


@pytest.mark.parametrize("vulkan", [(1, b"", b""), (0, b"framez", b"")], ids=["fails", "differs"])
def test_asking_for_vulkan_outright_raises_where_auto_would_fall_back(tmp_path, vulkan):
    """An explicit request is not quietly downgraded."""
    with pytest.raises(ArchiveError):
        probed(tmp_path, "vulkan", vulkan=vulkan)


def test_cpu_is_taken_as_asked_with_no_probe(tmp_path):
    """Choosing the CPU runs nothing."""
    (decode, why), calls = probed(tmp_path, "cpu")
    assert (decode, why) == ("cpu", "asked for") and not calls


def test_an_archive_the_cpu_cannot_decode_is_an_error_not_a_fallback(tmp_path):
    """With nothing to compare against there is no decision to make."""
    with pytest.raises(ArchiveError, match="no message"):
        probed(tmp_path, cpu=(1, b"", b""))


def test_an_empty_archive_falls_back_or_refuses(tmp_path):
    """No archive, no comparison: auto stays on the CPU and an explicit request fails."""
    archive = archive_at(tmp_path)
    archive.run = Decoders()
    assert archive.choose_decode("auto")[0] == "cpu" and not archive.run.calls
    with pytest.raises(ArchiveError, match="no committed archive"):
        archive.choose_decode("vulkan")


def test_the_probe_uses_the_first_named_program_that_is_archived(tmp_path):
    """Programs not archived are passed over, not probed."""
    archive, _, _ = write_archive(tmp_path)
    archive.run = Decoders()
    assert archive.choose_decode("auto", ["absent", "bitcrush"], timer=archive.run.timer)[0] == "vulkan"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_a_real_decoder_either_matches_on_the_gpu_or_falls_back(tmp_path):
    """Against the installed ffmpeg the choice is one the bytes bear out, wherever it runs."""
    reader, _ = irregular_recording(tmp_path)
    archive = archive_at(tmp_path)
    decode, why = archive.choose_decode("auto")
    assert decode in DECODERS and why
    if decode == "vulkan":
        assert decoders.digest(archive.reader("bitcrush", KEY, decode="vulkan")) == decoders.digest(reader)


def test_digest_covers_every_frame_in_order(tmp_path):
    """Two decodes agree only if every frame does, in the same order."""
    reader, made = vulkan_reader(tmp_path)
    reader.stream = lambda width=None: iter(made)
    expected = hashlib.sha256(b"".join(f.tobytes() for f in made)).hexdigest()
    assert decoders.digest(reader) == (expected, len(made))
    reader.stream = lambda width=None: iter(made[::-1])
    assert decoders.digest(reader)[0] != expected


class Ticking:
    """A clock the fake decoder moves on, one second per decode."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_rate_is_frames_decoded_over_wall_time_and_cycles_the_programs(tmp_path):
    """Every stream gets work to the end, and the decoder's own output is discarded."""
    archive, _, _ = write_archive(tmp_path, count=4)
    clock, seen = Ticking(), []

    def run(argv, stdout=None, stderr=None, check=False):
        del stderr, check
        seen.append((argv, stdout))
        clock.now += 1.0
        return subprocess.CompletedProcess(argv, 0, stderr=b"")

    fps = decoders.rate(archive, ["bitcrush"], "vulkan", 1, clock=clock, run=run)
    assert len(seen) == 2 and fps == pytest.approx(8 / 2.0)
    assert all(out == subprocess.DEVNULL and "-hwaccel" in argv for argv, out in seen)


def test_rate_raises_on_a_failed_decode(tmp_path):
    """A failed decode is not a fast one."""
    archive, _, _ = write_archive(tmp_path)

    def run(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 1, stderr=b"boom")

    with pytest.raises(RuntimeError, match="boom"):
        decoders.rate(archive, ["bitcrush"], "cpu", 2, run=run)


def test_survey_rates_every_backend_at_every_stream_count(tmp_path):
    """The table a concurrency choice is read from."""
    archive, _, _ = write_archive(tmp_path)

    def ok(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, stderr=b"")

    table = decoders.survey(archive, ["bitcrush"], ["cpu", "vulkan"], [1, 2], run=ok)
    assert set(table) == {"cpu", "vulkan"} and all(set(row) == {1, 2} for row in table.values())
    assert all(np.isfinite(v) and v > 0 for row in table.values() for v in row.values())


def test_rate_can_time_only_the_opening_frames(tmp_path):
    """A slow backend is timed on a cut of each program, and only what was decoded counts."""
    archive, _, _ = write_archive(tmp_path, count=4)
    clock, seen = Ticking(), []

    def run(argv, **_kwargs):
        seen.append(argv[argv.index("-frames:v") + 1])
        clock.now += 1.0
        return subprocess.CompletedProcess(argv, 0, stderr=b"")

    assert decoders.rate(archive, ["bitcrush"], "cpu", 1, frames=3, runs=3, clock=clock, run=run) == 3.0
    assert seen == ["3"] * 3
