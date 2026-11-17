"""Gating a plan: writes to a dead parameter are paired with the move that revives it, or refused."""

# pylint: disable=missing-function-docstring

import numpy as np

from syncsummoner.compose import gating
from syncsummoner.compose.vocabulary import Automation


def live(state, param):
    """P1 is live only with P7 on; P2 only with P7 off and P3 high; P7 always."""
    if param == 1:
        return state[6] >= 512
    if param == 2:
        return state[6] < 512 <= state[2]
    return True


def revive(_state, param):
    return {1: [((7, 1023),)], 2: [((7, 0),)]}.get(param, [])


def auto(*events):
    t, p, v = zip(*events)
    return Automation(np.asarray(t, float), np.asarray(p, np.int32), np.asarray(v, np.int32))


def test_a_write_to_a_dead_parameter_is_paired_with_its_gate_or_refused():
    start = np.zeros(12, dtype=np.int64)
    got = gating.gate_automation(
        auto((0.0, 1, 100), (1.0, 1, 200), (2.0, 2, 50), (3.0, 7, 0)), start, live, revive
    )
    assert got.paired == 1 and got.refused == 1
    assert got.auto.indices.tolist() == [7, 1, 1, 7] and got.auto.values.tolist() == [1023, 100, 200, 0]
    assert got.auto.times.tolist() == [0.0, 0.0, 1.0, 3.0]


def test_a_write_that_changes_nothing_passes_untouched():
    start = np.zeros(12, dtype=np.int64)
    got = gating.gate_automation(auto((0.0, 1, 0)), start, live, revive)
    assert (got.paired, got.refused, len(got.auto)) == (0, 0, 1)


def test_a_program_without_a_model_is_not_gated():
    assert gating.model_gate("nobody/Nothing") is None
