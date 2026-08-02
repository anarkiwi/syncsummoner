"""Result store: binary-keyed artefacts, atomic commit, resume after interruption."""

import json

import pytest

from syncsummoner.device.profile import MeasurementRecord, Source
from syncsummoner.probe import fit, store

FIRMWARE = "1.0.0-rc.37"
CLOCK = 1_700_000_000.0


class FakeManifest:
    """Manifest fake: SD programs only, exactly as the device reports."""

    def __init__(self, files):
        self._files = files

    def get(self, name):
        """Entry for a program, or None for a firmware built-in."""
        return type("Entry", (), {"file": self._files[name]})() if name in self._files else None


class FakeTransport:
    """Transport fake with a scriptable, optionally failing, file hash."""

    def __init__(self, hashes=None, *, manifest=None, fail=False, firmware=FIRMWARE):
        self.hashes = hashes or {}
        self.manifest = manifest
        self.fail = fail
        self._firmware = firmware
        self.hashed = []

    def program_manifest(self):
        """Installed program library, or None when the device carries none."""
        return self.manifest

    def file_hash(self, path):
        """Device-computed digest; raises when the slow serial hash times out."""
        self.hashed.append(path)
        if self.fail:
            raise TimeoutError("hash timed out")
        return self.hashes[path]

    def firmware(self):
        """Firmware version string."""
        return self._firmware


def record(program, value, metrics=None):
    """Measurement record."""
    params = [512] * 12
    params[0] = value
    return MeasurementRecord(
        program=program,
        firmware=FIRMWARE,
        analyzer="aesthetics 0.1.0",
        source=Source.HW,
        params=tuple(params),
        state_index=value % 8,
        stimulus="zoneplate",
        metrics=metrics if metrics is not None else {"luma_mean": value / 1023, "clip_frac": 0.0},
        settle_frames=3,
    )


def records(program, n=4):
    """A program's worth of records."""
    return [record(program, i * 128) for i in range(n)]


@pytest.fixture(name="store_at")
def store_at_fixture(tmp_path):
    """Store rooted in a temp directory, with a frozen clock."""
    return store.ResultStore(tmp_path / "results", clock=lambda: CLOCK)


def binary_key(program="bitcrush", material="deadbeef"):
    """Binary-keyed program key."""
    return store.ProgramKey(program, material, store.KeyKind.BINARY)


def test_key_digest_separates_kind_program_and_material():
    """A digest distinguishes every field, so no two identities collide."""
    base = binary_key()
    assert base.digest == binary_key().digest
    assert base.digest != binary_key(material="cafe").digest
    assert base.digest != binary_key(program="warp").digest
    assert base.digest != store.ProgramKey("bitcrush", "deadbeef", store.KeyKind.NAME_FIRMWARE).digest


def test_program_key_prefers_the_binary_hash():
    """The binary hash is the key when the device can supply it."""
    manifest = FakeManifest({"bitcrush": "bitcrush.bin"})
    transport = FakeTransport({"sd:/programs/bitcrush.bin": "sha256:abc"}, manifest=manifest)
    key = store.program_key(transport, "bitcrush")
    assert key.kind is store.KeyKind.BINARY
    assert key.material == "sha256:abc"
    assert transport.hashed == ["sd:/programs/bitcrush.bin"]


def test_program_key_keeps_an_absolute_manifest_path():
    """A manifest entry that already names a volume is not re-rooted."""
    manifest = FakeManifest({"bitcrush": "sd:/other/bitcrush.bin"})
    transport = FakeTransport({"sd:/other/bitcrush.bin": "sha256:abc"}, manifest=manifest)
    assert store.program_key(transport, "bitcrush").material == "sha256:abc"


@pytest.mark.parametrize(
    "transport",
    [
        FakeTransport(manifest=FakeManifest({"bitcrush": "bitcrush.bin"}), fail=True),
        FakeTransport(manifest=FakeManifest({})),
        FakeTransport(manifest=None),
        FakeTransport({"sd:/programs/bitcrush.bin": ""}, manifest=FakeManifest({"bitcrush": "bitcrush.bin"})),
    ],
    ids=["hash-timeout", "builtin-not-in-manifest", "no-manifest", "empty-hash"],
)
def test_program_key_degrades_to_name_and_firmware(transport):
    """A slow or absent hash degrades to the documented weaker key, never crashes."""
    key = store.program_key(transport, "bitcrush")
    assert key.kind is store.KeyKind.NAME_FIRMWARE
    assert key.material == FIRMWARE
    assert key.program == "bitcrush"


def test_program_key_fallback_survives_an_unreachable_firmware():
    """Firmware that cannot be read is recorded as unknown rather than raising."""

    class Dead(FakeTransport):
        """Transport whose shell is gone."""

        def firmware(self):
            """Raise, as a dead serial link does."""
            raise OSError("serial gone")

    assert store.program_key(Dead(manifest=None), "bitcrush").material == "unknown"


def test_put_then_get_round_trips_records(store_at):
    """Stored records come back as records, not rows."""
    key = binary_key()
    store_at.put("bitcrush", key, records("bitcrush"))
    loaded = store_at.get("bitcrush", key)
    assert [r.params for r in loaded] == [r.params for r in records("bitcrush")]
    assert loaded[0].metrics == pytest.approx(records("bitcrush")[0].metrics)
    assert loaded[0].source is Source.HW
    assert loaded[0].settle_frames == 3
    assert loaded[0].firmware == FIRMWARE


def test_stored_metrics_stay_per_record_under_a_union(store_at):
    """Heterogeneous metric keys union in Parquet but do not leak between records."""
    key = binary_key()
    rows = [record("bitcrush", 0, {"luma_mean": 0.5}), record("bitcrush", 1, {"winding_number": 0.25})]
    store_at.put("bitcrush", key, rows)
    loaded = store_at.get("bitcrush", key)
    assert set(loaded[0].metrics) == {"luma_mean"}
    assert set(loaded[1].metrics) == {"winding_number"}


def test_metadata_is_self_describing(store_at):
    """Metadata names the key material, its kind, provenance and the injected clock."""
    store_at.put("bitcrush", binary_key(), records("bitcrush"))
    meta = json.loads(store_at.paths("bitcrush")[1].read_text(encoding="utf-8"))
    assert meta["program"] == "bitcrush"
    assert meta["key_kind"] == "binary"
    assert meta["key_material"] == "deadbeef"
    assert meta["key_digest"] == binary_key().digest
    assert meta["firmware"] == FIRMWARE
    assert meta["analyzer"] == "aesthetics 0.1.0"
    assert meta["records"] == 4
    assert meta["data"] == store_at.paths("bitcrush")[0].name
    assert meta["created"].startswith("2023-11-14T")


def test_a_different_binary_invalidates_the_result(store_at):
    """A reflashed program is not served from the old artefact."""
    store_at.put("bitcrush", binary_key(), records("bitcrush"))
    updated = binary_key(material="feedface")
    assert store_at.has("bitcrush", updated) is False
    assert store_at.get("bitcrush", updated) is None
    assert store_at.get("bitcrush") is not None


def test_a_weaker_key_does_not_match_a_binary_key(store_at):
    """Results keyed on the fallback never mix with binary-keyed ones."""
    weak = store.ProgramKey("bitcrush", FIRMWARE, store.KeyKind.NAME_FIRMWARE)
    store_at.put("bitcrush", weak, records("bitcrush"))
    assert store_at.has("bitcrush", weak) is True
    assert store_at.has("bitcrush", store.ProgramKey("bitcrush", FIRMWARE)) is False


def test_pending_reports_the_remaining_worklist(store_at):
    """One call yields the programs a resumed sweep still has to measure."""
    keys = {name: binary_key(name, f"h{name}") for name in ("a", "b", "c")}
    store_at.put("b", keys["b"], records("b"))
    assert store_at.pending(keys, keys) == ["a", "c"]
    assert store_at.pending(("a", "b", "c"), keys.get) == ["a", "c"]
    assert store_at.pending(("a", "b", "unknown"), keys) == ["a", "unknown"]


def test_committed_lists_only_complete_artefacts(store_at, tmp_path):
    """The listing skips uncommitted data and unparsable sidecars."""
    store_at.put("bitcrush", binary_key(), records("bitcrush"))
    (store_at.directory / "junk.json").write_text("{not json", encoding="utf-8")
    (store_at.directory / "orphan.json").write_text('{"program": "orphan"}', encoding="utf-8")
    assert sorted(store_at.committed()) == ["bitcrush"]
    assert not store.ResultStore(tmp_path / "empty").committed()


def test_resume_skips_finished_programs_and_keeps_their_records(store_at):
    """A resumed sweep measures only what is missing and recovers the rest from disk."""
    keys = {name: binary_key(name, f"h{name}") for name in ("a", "b", "c")}
    for name in ("a", "b"):
        store_at.put(name, keys[name], records(name))
    remaining = store_at.pending(keys, keys)
    assert remaining == ["c"]
    store_at.put("c", keys["c"], records("c"))
    recovered = [r for name in keys for r in store_at.get(name, keys[name])]
    assert len(recovered) == 12
    assert sorted({r.program for r in recovered}) == ["a", "b", "c"]
    assert store_at.pending(keys, keys) == []


def test_a_kill_during_the_data_write_leaves_nothing(store_at, monkeypatch):
    """A process killed writing Parquet commits nothing and leaves no temp file."""

    def die(_records, path):
        """Write a partial file, then die."""
        path.write_bytes(b"PAR1 truncated")
        raise KeyboardInterrupt

    monkeypatch.setattr(store, "save_measurements", die)
    store_at.directory.mkdir(parents=True, exist_ok=True)
    with pytest.raises(KeyboardInterrupt):
        store_at.put("bitcrush", binary_key(), records("bitcrush"))
    assert not list(store_at.directory.iterdir())
    assert store_at.has("bitcrush", binary_key()) is False


class DyingClock:
    """Clock that dies on its first read, where the metadata write begins."""

    def __init__(self):
        self.calls = 0

    def __call__(self):
        """Raise once, then keep time."""
        self.calls += 1
        if self.calls == 1:
            raise KeyboardInterrupt
        return CLOCK


def test_a_kill_between_data_and_metadata_is_not_a_commit(tmp_path):
    """Data on disk without its metadata marker is redone, not trusted."""
    killed = store.ResultStore(tmp_path / "results", clock=DyingClock())
    key = binary_key()
    with pytest.raises(KeyboardInterrupt):
        killed.put("bitcrush", key, records("bitcrush"))
    assert killed.paths("bitcrush")[0].exists()
    assert killed.has("bitcrush", key) is False
    assert killed.pending(["bitcrush"], {"bitcrush": key}) == ["bitcrush"]
    killed.put("bitcrush", key, records("bitcrush", 2))
    assert len(killed.get("bitcrush", key)) == 2


def test_a_missing_or_newer_artefact_is_not_served(store_at):
    """Absent, truncated and future-schema artefacts all read as work still to do."""
    key = binary_key()
    assert store_at.get("bitcrush", key) is None
    assert store_at.has("bitcrush", key) is False
    store_at.put("bitcrush", key, records("bitcrush"))
    meta_path = store_at.paths("bitcrush")[1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.write_text(json.dumps(meta | {"schema_version": 99}), encoding="utf-8")
    assert store_at.has("bitcrush", key) is False
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    store_at.paths("bitcrush")[0].unlink()
    assert store_at.has("bitcrush", key) is False


def test_put_rejects_an_empty_result(store_at):
    """An empty sweep is not a commit; the program stays on the worklist."""
    with pytest.raises(ValueError, match="no records"):
        store_at.put("bitcrush", binary_key(), [])


def test_awkward_program_names_get_distinct_artefacts(store_at):
    """Names needing escaping stay distinct on disk."""
    keys = {name: binary_key(name) for name in ("a/b", "a b", "a_b")}
    for name, key in keys.items():
        store_at.put(name, key, records(name))
    stems = {store_at.paths(name)[0] for name in keys}
    assert len(stems) == 3
    assert all(store_at.get(name, key)[0].program == name for name, key in keys.items())
    assert store_at.paths("a_b")[0].name == "a_b.parquet"


def test_store_reuses_the_parquet_helpers(store_at):
    """The artefact is exactly what ``fit.load_measurements`` reads."""
    store_at.put("bitcrush", binary_key(), records("bitcrush"))
    rows = fit.load_measurements(store_at.paths("bitcrush")[0])
    assert len(rows) == 4
    assert rows[0]["program"] == "bitcrush"
    assert fit.fit_profile(store_at.get("bitcrush")).program == "bitcrush"
