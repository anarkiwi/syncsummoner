"""Gating structure and the covering base states, on the tiny design whose gating is known by construction."""

# pylint: disable=missing-function-docstring

import numpy as np

from syncsummoner.program.gates import DEAD, LIVE, Gating, Revival, cover, intervals, solve, Status, structure


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
    assert str(Revival(1, ((7, 767),), ((512, 1023),))) == "P7 in 512..1023"
    assert str(Revival(1, ((7, 1023),), ((3, 3), (1023, 1023)))) == "P7 in 3, 1023"
    assert Revival(1, ((7, 767), (2, 600))).gate == (7, 2)
    assert str(Revival(1, ((7, 767), (2, 600)))) == "P7 = 767 and P2 = 600"


def test_the_cover_is_exact_on_a_hand_made_instance():
    states = [np.zeros(12, dtype=np.int64)] * 3
    status = [{1: Status(1, ())}, {1: Status(1, (0, 1))}, {1: Status(1, (1,))}]
    chosen, stepped, missed = solve(
        states, status, [("bit", 1, 0), ("bit", 1, 1), ("bit", 1, 2)], lambda s, p: s
    )
    assert chosen == [1] and stepped == {1: [1]} and missed == [("bit", 1, 2)]
    assert status[0][1].state == DEAD and status[1][1].state == LIVE


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


def test_a_cover_at_a_timing_holds_register_8(tiny):
    found = cover(tiny, gating=Gating(tiny, timing=3))
    assert {p for ps in found.stepped.values() for p in ps} == {1, 2, 7, 8} and not found.uncovered


def test_the_report_is_the_cover_as_table_or_json(tiny):
    import json  # pylint: disable=import-outside-toplevel

    from syncsummoner.program.gates import report  # pylint: disable=import-outside-toplevel

    assert "base states" in report(tiny, tiny.initial())
    assert json.loads(report(tiny, tiny.initial(), as_json=True))["one_at_a_time"]["states"] == 1


def test_the_tour_moves_at_most_two_settings_and_no_more_than_any_closed_walk():
    from syncsummoner.program.gates import tour  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(3)
    for n in range(1, 12):
        values = np.sort(rng.choice(1024, n, replace=False))
        for base in (int(values[0]), int(values[n // 2]), int(values[-1]), 1023, 0):
            walk = tour(values, base)
            assert sorted(walk.tolist()) == sorted(set(values.tolist()) - {base})
            line = np.union1d(values, [base])
            steps = np.abs(np.diff(np.searchsorted(line, np.r_[base, walk, base])))
            assert steps.max(initial=0) <= 2
            moved = np.abs(np.diff(np.r_[base, walk, base])).sum()
            assert moved == 2 * (line[-1] - line[0]) or line.size == 1


def test_the_structure_reads_every_reviving_setting_off_the_condition(tiny):
    found = structure(tiny)
    assert found[1] == ["P7 in 512..1023 and P8 in 0..511"]
    assert any(line.startswith("P1 in 64..1023 and P2 in 513..527, 529..543") for line in found[8])
    assert found[3] == ["never"] and found[2] == ["P7 in 0..511 and P8 in 0..511"]
    assert "live where P7 in 512..1023 and P8 in 0..511" in cover(tiny).table()


def _synthetic():
    from dd import cudd  # pylint: disable=import-outside-toplevel

    bdd = cudd.BDD()
    bdd.declare("a0", "a1", "b0", "c0")
    held = {1: {"a0": False, "a1": False}, 2: {"b0": False}, 3: {"c0": False}}
    domains = {
        1: [{"a0": x, "a1": y} for y in (False, True) for x in (False, True)],
        2: [{"b0": False}, {"b0": True}],
        3: [{"c0": False}, {"c0": True}],
    }
    return bdd, held, domains


def test_the_fewest_gates_are_the_smallest_satisfiable_selector_count():
    from syncsummoner.program.gates import fewest  # pylint: disable=import-outside-toplevel

    bdd, held, domains = _synthetic()
    a0, a1, b0, c0 = (bdd.var(n) for n in ("a0", "a1", "b0", "c0"))
    assert fewest(bdd, b0 | c0, held, domains) == [(2,), (3,)]
    assert fewest(bdd, a1 & b0, held, domains) == [(1, 2)]
    assert fewest(bdd, (a1 & b0 & c0) | (a0 & ~a1 & c0), held, domains) == [(1, 3)]
    assert fewest(bdd, a1 & b0 & c0, held, domains) == [(1, 2, 3)]
    assert not fewest(bdd, bdd.false, held, domains)
    narrow = {**domains, 1: domains[1][:2]}
    assert not fewest(bdd, a0 & a1, held, narrow), "no setting in the domain reaches it"
    assert fewest(bdd, a0 & a1, held, domains) == [(1,)]
    moved = {**held, 2: {"b0": True}}
    assert fewest(bdd, a1 & b0, moved, domains) == [(1,)], "a gate already at its setting is not moved"


def test_a_witness_takes_the_first_completable_setting_and_spans_the_last_gate():
    from syncsummoner.program.gates import witness  # pylint: disable=import-outside-toplevel

    bdd, _, domains = _synthetic()
    a0, a1, b0 = (bdd.var(n) for n in ("a0", "a1", "b0"))
    picks, ok = witness(bdd, (a1 & b0) | (a0 & a1 & ~b0), (1, 2), domains)
    assert picks == [2] and ok.tolist() == [False, True]
    picks, ok = witness(bdd, a0 & a1, (1, 2), domains)
    assert picks == [3] and ok.tolist() == [True, True]
    picks, ok = witness(bdd, a1 & ~a1, (1, 2), domains)
    assert not picks and not ok.any()
    assert witness(bdd, ~a0, (1,), domains)[1].tolist() == [True, False, True, False]


def test_revivals_from_the_condition_match_the_simulated_search(tiny):
    found = cover(tiny)
    g = Gating(tiny)
    for i, state in enumerate(found.states):
        for p, st in found.status[i].items():
            if st.state != LIVE:
                others = [q for q in sorted(g.read) if q != p]
                expected = g._simulated(state, p, others)  # pylint: disable=protected-access
                assert found.revivals[i][p] == expected
    both = tiny.initial()
    both[[6, 7]] = 0, 1023
    pair = g.revivals(both, Status(1, ()))
    assert [r.gate for r in pair] == [(7, 8)] and not pair[0].spans
    assert pair == g._simulated(both, 1, [2, 7, 8])  # pylint: disable=protected-access
    assert g.statuses(np.r_[both[:6], pair[0].moves[0][1], pair[0].moves[1][1], both[8:]])[1].state == LIVE


def test_a_cover_without_conditions_revives_by_the_ternary_simulator(tiny, tmp_path):
    from syncsummoner.program.model import ProgramModel  # pylint: disable=import-outside-toplevel

    coarse = ProgramModel(tiny.program, tiny.library, cache=tmp_path)
    coarse.memory = 0
    found = cover(coarse)
    assert coarse.method == "ternary" and not found.uncovered
    assert found.gating()[1] == ["dead at the reference", "live when P7 in 512..1023"]
