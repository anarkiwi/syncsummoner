"""Each sent step judged by the program model from the state it was taken in."""

# pylint: disable=missing-function-docstring

import numpy as np

from syncsummoner.compose.evidence import LANDED
from syncsummoner.program import predict as P


def rows_of(*writes):
    """Landed rows ``(frame, param, value)``, every one sent unless its frame is None."""
    rows = np.zeros(len(writes), dtype=LANDED)
    for i, (frame, param, value) in enumerate(writes):
        rows[i]["param"], rows[i]["value"], rows[i]["wire"] = param, value, value
        rows[i]["sent"] = -1 if frame is None else i
        rows[i]["landed"] = -1 if frame is None else frame
        rows[i]["target"] = 0 if frame is None else frame
    return rows


def test_writes_on_one_frame_step_from_the_frames_new_state():
    rows = rows_of((5, 2, 9), (1, 1, 4), (5, 1, 7), (None, 3, 8))
    pre, a, b = P.pre_states(np.arange(12), rows)
    assert (a[1], b[1]) == (0, 4) and pre[1][0] == 0
    assert (a[0], b[0], pre[0][0]) == (1, 9, 7), "P2 steps with P1 already at its new value"
    assert (a[2], b[2], pre[2][1]) == (4, 7, 9)
    assert (a[3], b[3]) == (2, 8)


def test_the_tiny_design_judges_each_step(tiny):
    rows = rows_of(
        (1, 1, 256), (2, 1, 256), (3, 7, 1023), (4, 1, 900), (5, 2, 101), (6, 2, 700), (None, 1, 5)
    )
    kinds = P.classify(tiny, tiny.initial(), rows)
    assert kinds.tolist() == [P.DEAD, P.UNCHANGED, P.LIVE, P.LIVE, P.DEAD, P.DEAD, P.UNSENT]
    on = rows_of((1, 7, 1023), (2, 1, 544))
    assert P.classify(tiny, tiny.initial(), on).tolist() == [P.LIVE, P.NOOP], "bit 5 of Gain is never read"
    assert P.knob_gated(tiny) == frozenset({8}), "Bypass is revived by Bias's far bit"
    dry = rows_of((1, 8, 1023), (2, 8, 0), (3, 2, 700), (4, 8, 1023))
    assert P.classify(tiny, tiny.initial(), dry).tolist() == [P.DEAD, P.DEAD, P.LIVE, P.LIVE]


def test_a_take_of_an_unknown_program_gets_no_verdicts(monkeypatch, tmp_path):
    from syncsummoner.compose.locktake import (
        judged_steps,
        step_counts,
    )  # pylint: disable=import-outside-toplevel

    monkeypatch.setenv("SYNCSUMMONER_CARD", str(tmp_path))
    monkeypatch.setenv("SYNCSUMMONER_FIRMWARE", str(tmp_path))
    assert judged_steps("nobody/Nothing", np.zeros(12, dtype=int), rows_of()) is None
    rows = rows_of((1, 1, 4), (2, 1, 5), (None, 2, 3))
    assert step_counts(rows, np.array([P.DEAD, P.LIVE, P.UNSENT], dtype=object)) == {
        1: {P.DEAD: 1, P.LIVE: 1}
    }
    assert not step_counts(rows, None)
