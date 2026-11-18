"""Every test keeps its loads in a record of its own, never the rig's."""

import pytest


@pytest.fixture(autouse=True)
def _own_load_record(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("SYNCSUMMONER_LOADS", str(tmp_path_factory.mktemp("rig") / "loads.jsonl"))
