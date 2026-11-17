"""Gating structure and the covering base states, on the tiny design whose gating is known by construction."""

# pylint: disable=missing-function-docstring

import numpy as np

from syncsummoner.program.gates import DEAD, LIVE, Gating, Revival, cover, intervals, solve, Status


def test_intervals_are_maximal_runs():
    ok = np.zeros(10, dtype=bool)
    ok[[1, 2, 3, 7, 9]] = True
    assert intervals(ok) == [(1, 3), (7, 7), (9, 9)]
    assert not intervals(np.zeros(3, dtype=bool))


def test_the_tiny_design_is_covered_by_two_states(tiny):
    found = cover(tiny)
    gating = found.gating()
    assert gating[1][0] == "dead at the reference" and "live when P7 in 512..1023" in gating[1]
    assert any(g.startswith("live when P2 in 513..527, 529..543") for g in gating[8])
    assert "live when P7 in 512..1023" in gating[8]
    assert set(found.read) == {1, 2, 7, 8}
    assert len(found.chosen) == 2 and not found.uncovered
    stepped = {p for ps in found.stepped.values() for p in ps}
    assert stepped == {1, 2, 7, 8}
    for i in found.chosen:
        for p in found.stepped[i]:
            assert found.status[i][p].state == LIVE
    on = [i for i in found.chosen if found.states[i][6] >= 512]
    assert on and 1 in found.stepped[on[0]]


def test_the_sweep_steps_one_parameter_at_a_time_from_each_base(tiny):
    found = cover(tiny)
    pairs = found.vectors()
    bases = [found.states[i] for i in found.chosen]
    for before, after in pairs:
        moved = np.flatnonzero(before != after)
        assert moved.size == 1 or any(np.array_equal(after, b) for b in bases)
    singles = sum(1 for before, after in pairs if np.count_nonzero(before != after) == 1)
    assert singles >= found.steps() - sum(map(len, found.stepped.values()))
    for i in found.chosen:
        for p in found.stepped[i]:
            hit = {int(a[p - 1]) for b, a in pairs if np.count_nonzero(b != a) == 1 and b[p - 1] != a[p - 1]}
            assert set(found.schedule(i, p).tolist()) - {int(found.states[i][p - 1])} <= hit
    assert found.as_dict()["one_at_a_time"]["live"] == 2
    assert "P1" in found.table()


def test_a_revival_names_its_gate():
    assert str(Revival(1, DEAD, ((7, 767),), ((512, 1023),))) == "P7 in 512..1023"
    assert str(Revival(1, DEAD, ((7, 1023),), ((3, 3), (1023, 1023)))) == "P7 in 3, 1023"
    assert Revival(1, DEAD, ((7, 767), (2, 600))).gate == (7, 2)
    assert str(Revival(1, DEAD, ((7, 767), (2, 600)))) == "P7 = 767 and P2 = 600"


def test_the_cover_is_exact_on_a_hand_made_instance():
    live = Status(1, (0,), ())
    states = [np.zeros(12, dtype=np.int64)] * 3
    status = [{1: Status(1, ())}, {1: Status(1, (0, 1))}, {1: Status(1, (1,))}]
    for s in status:
        s[1] = Status(1, s[1].live_bits, () if not s[1].live_bits else (_Step(True),))
    chosen, stepped, missed = solve(
        states, status, [("bit", 1, 0), ("bit", 1, 1), ("bit", 1, 2)], lambda s, p: s
    )
    assert chosen == [1] and stepped == {1: [1]} and missed == [("bit", 1, 2)]
    assert live.state in (DEAD, "no-op")


class _Step:  # pylint: disable=too-few-public-methods
    def __init__(self, matters):
        self.matters = matters


def test_gating_remembers_states(tiny):
    g = Gating(tiny)
    first = g.statuses(tiny.initial())
    assert g.statuses(tiny.initial()) is first


def test_the_model_gate_pairs_a_gain_write_with_gain_on(tiny):
    from syncsummoner.compose.gating import ModelGate  # pylint: disable=import-outside-toplevel
    from syncsummoner.compose.vocabulary import Automation  # pylint: disable=import-outside-toplevel

    found = cover(tiny)
    revivals = {p: [r.moves for r in rs] for here in found.revivals for p, rs in here.items()}
    gate = ModelGate(tiny, revivals)
    plan = Automation(np.array([0.0, 1.0]), np.array([1, 3], np.int32), np.array([900, 700], np.int32))
    out = gate(plan)
    assert out.indices.tolist() == [7, 1] and out.values[0] >= 512
    assert gate.live(tiny.initial(), 7) and not gate.live(tiny.initial(), 1)


def test_a_structural_cover_takes_live_bits_as_live(tiny):
    found = cover(tiny, gating=Gating(tiny, steps=False))
    assert all(st.steps is None for st in found.status[0].values() if st.live_bits)
    assert {p for ps in found.stepped.values() for p in ps} == {1, 2, 7, 8} and not found.uncovered


def test_the_report_is_the_cover_as_table_or_json(tiny):
    import json  # pylint: disable=import-outside-toplevel

    from syncsummoner.program.gates import report  # pylint: disable=import-outside-toplevel

    assert "base states" in report(tiny, tiny.initial(), steps=False)
    assert json.loads(report(tiny, tiny.initial(), as_json=True))["one_at_a_time"]["states"] == 1
