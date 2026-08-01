"""Runner: gray-code frame identification, drop/duplicate tolerance, metric stamping."""

from types import SimpleNamespace

import numpy as np
import pytest

from syncsummoner.device.profile import PARAM_MAX, Source
from syncsummoner.probe import patterns, runner

WIDTH, HEIGHT = 64, 48


class FakeAnalyzer:
    """Stand-in for ``syncsummoner.aesthetics`` with the contracted return shapes."""

    __version__ = "9.9.9"

    def gabor_energy(self, frame):
        """Gabor energy."""
        energy = np.full((5, 4), 0.05, dtype=np.float32)
        energy[0, 0] = float(frame.mean())
        return SimpleNamespace(energy=energy, concentration=float(frame.std()), peak=(0, 0))

    def spectral_stats(self, frame):
        """Spectral stats."""
        return SimpleNamespace(slope=-1.2 + float(frame.mean()), r2=0.9, fractal_dimension=1.4)

    def level_stats(self, frame):
        """Level stats."""
        return SimpleNamespace(
            luma_mean=float(frame.mean()),
            luma_std=float(frame.std()),
            chroma_mean=0.1,
            chroma_std=0.02,
            clip_frac=float((frame >= 1.0).mean()),
            illegal_frac=0.0,
            colourfulness=0.3,
        )

    def motion_stats(self, prev, curr):
        """Motion stats."""
        return SimpleNamespace(
            framediff_energy=float(np.abs(curr - prev).mean()), flow_magnitude=0.5, flow_coherence=0.8
        )

    def analyze_dynamics(self, series, *, fps):
        """Analyze dynamics."""
        return SimpleNamespace(
            period_frames=2.0,
            periodicity_strength=float(np.std(series)),
            winding_number=0.5 / fps * 60.0,
            stability=SimpleNamespace(value="periodic"),
        )

    def passthrough_distance(self, source, output):
        """Passthrough distance."""
        return float(np.abs(np.asarray(source) - np.asarray(output)).mean())


class FakeSession:
    """Records parameter vectors in the order the runner drives them."""

    firmware = "1.0.0-rc.37"

    def __init__(self):
        """init  ."""
        self.calls = []

    def set_params(self, values):
        """Set params."""
        self.calls.append(dict(values))


class FakeCapture:
    """Replays a scripted frame sequence; ``None`` models a dropped read."""

    def __init__(self, frames, *, no_signal=()):
        """init  ."""
        self.frames = list(frames)
        self.no_signal = {id(f) for f in no_signal}
        self.reads = 0

    def read(self):
        """Read."""
        if not self.frames:
            return None
        self.reads += 1
        return self.frames.pop(0)

    def is_no_signal(self, frame):
        """Is no signal."""
        return id(frame) in self.no_signal


def base_frame(seed=0):
    """Base frame."""
    return patterns.generate("noise_1f", width=WIDTH, height=HEIGHT, rng=np.random.default_rng(seed))


def tagged(state, *, seed=0, bits=8, strip_px=8):
    """Tagged."""
    return patterns.with_state_index(base_frame(seed), state, bits=bits, strip_px=strip_px)


def plan(n):
    """Plan."""
    return [{1: i / max(n - 1, 1), 4: bool(i % 2)} for i in range(n)]


def run(capture, vectors, **kwargs):
    """Run."""
    session = FakeSession()
    records = runner.run_plan(
        session,
        capture,
        vectors,
        program="bitcrush_displace",
        analyzer=FakeAnalyzer(),
        frames_per_point=kwargs.pop("frames_per_point", 2),
        **kwargs,
    )
    return session, records


def test_run_plan_identifies_frames_by_state_index():
    """Run plan identifies frames by state index."""
    frames = [tagged(state, seed=state) for state in range(3) for _ in range(2)]
    session, records = run(FakeCapture(frames), plan(3))
    assert len(session.calls) == 3
    assert [r.state_index for r in records] == [0, 1, 2]
    assert [r.params[0] for r in records] == [0, 512, PARAM_MAX]
    assert [r.params[3] for r in records] == [0, PARAM_MAX, 0]
    assert all(r.program == "bitcrush_displace" for r in records)
    assert records[0].analyzer == "aesthetics 9.9.9"
    assert records[0].firmware == "1.0.0-rc.37"
    assert records[0].source is Source.HW


def test_metrics_cover_the_design_battery():
    """Metrics cover the design battery."""
    frames = [tagged(0, seed=i) for i in range(8)]
    _, records = run(FakeCapture(frames), plan(1), frames_per_point=8)
    metrics = records[0].metrics
    assert {"ch_s0_o0", "ch_s4_o3", "concentration", "peak_scale"} <= set(metrics)
    assert {"spectral_slope", "spectral_r2", "fractal_dimension"} <= set(metrics)
    assert {"luma_mean", "chroma_mean", "clip_frac", "illegal_frac", "colourfulness"} <= set(metrics)
    assert {"framediff_energy", "flow_magnitude", "flow_coherence"} <= set(metrics)
    assert {"periodicity_strength", "period_frames", "winding_number", "stability"} <= set(metrics)
    assert metrics["stability"] == runner.STABILITY_ORDER.index("periodic")


def test_passthrough_distance_recorded_against_reference():
    """Passthrough distance recorded against reference."""
    frames = [tagged(0, seed=0), tagged(0, seed=1)]
    reference = patterns.crop_strip(frames[-1])
    _, records = run(FakeCapture(frames), plan(1), reference=reference)
    assert records[0].metrics["passthrough_distance"] == pytest.approx(0.0, abs=1e-6)


def test_no_signal_frames_are_skipped():
    """No signal frames are skipped."""
    splash = tagged(0, seed=99)
    frames = [splash, tagged(0, seed=0), tagged(0, seed=1)]
    capture = FakeCapture(frames, no_signal=[splash])
    _, records = run(capture, plan(1))
    assert len(records) == 1
    assert records[0].metrics["luma_mean"] != pytest.approx(float(splash.mean()))


def test_duplicated_frames_are_collapsed():
    """Duplicated frames are collapsed."""
    duplicate = tagged(0, seed=0)
    frames = [duplicate, duplicate, duplicate, tagged(0, seed=1)]
    _, records = run(FakeCapture(frames), plan(1), frames_per_point=2)
    assert records[0].metrics["framediff_energy"] > 0


def test_dropped_state_is_skipped_not_misattributed():
    """Dropped state is skipped not misattributed."""
    frames = [tagged(0, seed=0), tagged(0, seed=1), tagged(2, seed=2), tagged(2, seed=3)]
    _, records = run(FakeCapture(frames), plan(3))
    assert [r.state_index for r in records] == [0, 2]


def test_unreadable_frames_do_not_stall_the_sweep():
    """Unreadable frames do not stall the sweep."""
    blank = np.full((HEIGHT, WIDTH, 3), 0.5, dtype=np.float32)
    frames = [blank, blank, tagged(0, seed=0), tagged(0, seed=1)]
    _, records = run(FakeCapture(frames), plan(1))
    assert [r.state_index for r in records] == [0]


def test_allow_untagged_attributes_frames_to_the_current_vector():
    """Allow untagged attributes frames to the current vector."""
    blank = [np.full((HEIGHT, WIDTH, 3), v, dtype=np.float32) for v in (0.4, 0.6)]
    _, records = run(FakeCapture(blank), plan(1), allow_untagged=True)
    assert len(records) == 1
    assert records[0].metrics["luma_mean"] == pytest.approx(0.6, abs=1e-6)


def test_emit_publishes_the_state_index_before_waiting():
    """Emit publishes the state index before waiting."""
    published = []
    frames = [tagged(state, seed=state) for state in range(3) for _ in range(2)]
    run(FakeCapture(frames), plan(3), emit=published.append)
    assert published == [0, 1, 2]


def test_state_index_wraps_at_capacity():
    """State index wraps at capacity."""
    frames = [tagged(state % 4, seed=state, bits=2) for state in range(6) for _ in range(2)]
    _, records = run(FakeCapture(frames), plan(6), bits=2)
    assert [r.state_index for r in records] == [0, 1, 2, 3, 0, 1]


def test_unknown_firmware_when_session_is_silent():
    """Unknown firmware when session is silent."""

    class Bare(FakeSession):
        """Session that reports no firmware string."""

        firmware = None

    session = Bare()
    records = runner.run_plan(
        session,
        FakeCapture([tagged(0)]),
        plan(1),
        program="p",
        analyzer=FakeAnalyzer(),
        frames_per_point=1,
    )
    assert records[0].firmware == "unknown"


def test_firmware_from_a_transport_method():
    """Firmware from a transport method."""

    class Wired(FakeSession):
        """Session whose transport reports firmware over serial."""

        firmware = None
        transport = SimpleNamespace(firmware=lambda: "1.0.0-rc.99")

    records = runner.run_plan(
        Wired(),
        FakeCapture([tagged(0)]),
        plan(1),
        program="p",
        analyzer=FakeAnalyzer(),
        frames_per_point=1,
    )
    assert records[0].firmware == "1.0.0-rc.99"


def test_firmware_lookup_survives_a_dead_link():
    """Firmware lookup survives a dead link."""

    def explode():
        raise OSError("serial link down")

    class Broken(FakeSession):
        """Session whose firmware query raises."""

        firmware = staticmethod(explode)

    records = runner.run_plan(
        Broken(),
        FakeCapture([tagged(0)]),
        plan(1),
        program="p",
        analyzer=FakeAnalyzer(),
        frames_per_point=1,
    )
    assert records[0].firmware == "unknown"


def test_settle_frames_counts_until_plateau():
    """Settle frames counts until plateau."""
    moving = [np.full((8, 8, 3), v, dtype=np.float32) for v in (0.0, 0.4, 0.8, 0.81, 0.81, 0.81)]
    assert runner.settle_frames(moving) == 2
    assert runner.settle_frames(moving[:1]) == 0
    assert runner.settle_frames([moving[0], moving[0]]) == 0


def test_raw_params_encodes_booleans_at_the_rails():
    """Raw params encodes booleans at the rails."""
    assert runner.raw_params({1: 0.0, 2: 1.0, 7: True, 8: False})[:8] == (
        0,
        PARAM_MAX,
        0,
        0,
        0,
        0,
        PARAM_MAX,
        0,
    )


def test_allow_untagged_ignores_a_spuriously_decoded_index():
    """Untagged playout has no strip, so any decoded index is noise, not a mismatch."""
    rng = np.random.default_rng(4)
    shape = (24, 32, 3)
    frames = [rng.random(shape).astype(np.float32) for _ in range(40)]
    # Top rows carry ordinary content that happens to decode as some index.
    for frame in frames:
        frame[:8] = np.tile(np.array([1.0, 0.0], np.float32).repeat(16)[:32], (8, 1))[:, :, None]

    capture = FakeCapture(frames)
    session = FakeSession()
    setpoints = [{1: 0.0}, {1: 0.5}, {1: 1.0}]
    records = runner.run_plan(
        session, capture, setpoints, program="P", analyzer=FakeAnalyzer(), allow_untagged=True
    )
    assert len(records) == len(setpoints), "every setpoint must yield a record when untagged"
