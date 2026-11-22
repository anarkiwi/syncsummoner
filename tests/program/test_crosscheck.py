"""The three liveness methods answered side by side, and a method that loses a live bit named at fault."""

# pylint: disable=missing-function-docstring

import json

from syncsummoner.program import crosscheck
from syncsummoner.program.model import ProgramModel, model


def build(tiny):
    return lambda p, **k: model(p, tiny.library, source="rtl", cache=tiny.cache.parent.parent, **k)


def test_the_tiny_designs_methods_agree_on_every_query(tiny, monkeypatch):
    monkeypatch.setattr(crosscheck, "build", build(tiny))
    asked = crosscheck.queries(tiny, presets=True)
    pids = [d.parameter_id for d in tiny.params]
    assert {free for _, _, free in asked} >= {(p,) for p in pids}
    presets = tiny.program.package.config.presets
    assert {name for name, _, _ in asked} == {"initial", *(p.name for p in presets)}
    assert {name for name, _, _ in crosscheck.queries(tiny)} == {"initial"}
    found = crosscheck.verdicts(tiny.program, presets=True)
    assert found["queries"] == len(asked) and set(found["seconds"]) == set(crosscheck.METHODS)
    assert not found["faults"]
    assert all(set(d["sat"]) <= set(d["ternary"]) for d in found["disagreements"])


def test_a_method_that_loses_a_live_bit_is_at_fault(tiny, monkeypatch, capsys):
    from syncsummoner.program import Library  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(crosscheck, "build", build(tiny))
    monkeypatch.setattr(Library, "default", classmethod(lambda cls: tiny.library))
    monkeypatch.setattr(tiny.library, "_programs", [tiny.program])
    reach = ProgramModel.reach

    def lossy(self, state, free, timing=None):
        found = reach(self, state, free, timing)
        return {p: [] for p in found} if self.prefer == "ternary" else found

    monkeypatch.setattr(ProgramModel, "reach", lossy)
    from syncsummoner import cli  # pylint: disable=import-outside-toplevel

    assert cli.main(["program", "crosscheck", "all"]) == 1
    row = json.loads(capsys.readouterr().out)
    assert row["faults"] == ["ternary"] and row["disagreements"]
    assert all(d["ternary"] == [] and d["sat"] for d in row["disagreements"])


def test_with_faults_the_miter_answers_only_where_the_bdds_drop_a_bit_the_simulator_keeps(tiny, monkeypatch):
    monkeypatch.setattr(crosscheck, "build", build(tiny))
    whole = crosscheck.verdicts(tiny.program)
    dropped = {
        (d["state"], tuple(d["free"])) for d in whole["disagreements"] if set(d["ternary"]) - set(d["bdd"])
    }
    assert crosscheck.verdicts(tiny.program, faults=True)["asked"]["sat"] == len(dropped)
    reach = ProgramModel.reach

    def lossy(self, state, free, timing=None):
        found = reach(self, state, free, timing)
        return {p: [] for p in found} if self.prefer == "bdd" else found

    monkeypatch.setattr(ProgramModel, "reach", lossy)
    found = crosscheck.verdicts(tiny.program, faults=True)
    assert found["faults"] == ["bdd"] and 0 < found["asked"]["sat"] <= whole["asked"]["sat"]
