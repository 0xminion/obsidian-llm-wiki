"""Operator import binds the exact attested bytes and publishes without overwrite.

Closes two findings in ``podcast_cache_recovery.import_legacy``:

* Byte binding - the supplied file was hashed, reopened and parsed, and then the
  transcript was loaded again *by identity from the default cache*. A backup copy
  A, attested by its SHA-256, imported the different same-key default-cache
  entry B (or was refused when no default copy existed). Now one validated
  snapshot of the supplied file is hashed, decoded by the shared cache-admission
  decoder and imported; nothing else is read.
* Publication - ``target.exists()`` followed by ``save_transcript_cache``
  (``tmp.replace``) could overwrite a fresh entry written between check and
  replace, and replaced a dangling symlink. Import publication is now an atomic
  no-overwrite link of a fully written, fsynced temporary file; a failed import
  leaves no entry and no temporary file. Normal cache refresh still replaces.

Network, DNS and ASR are the doubles of ``test_podcast_binding_epoch``.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
from pathlib import Path

import pytest

from obsidian_llm_wiki.ingest import extractors
from obsidian_llm_wiki.ingest import podcast_cache_recovery as rec
from obsidian_llm_wiki.ingest import transcript_resolver as tr
from tests.new import test_podcast_binding_epoch as _epoch
from tests.new.test_podcast_binding_epoch import (
    A_TEXT,
    AUDIO,
    B_TEXT,
    FEED,
    VTT_URL,
    A,
    apple_routes,
    apple_url,
    bound_payloads,
    cache_dir,
    rss,
    snapshot,
    vtt_body,
    write_head_entry,
)

world = _epoch.world  # the shared network/DNS/ASR doubles (a pytest fixture)

@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    """Belt and braces: even a lost ``VAULT_PATH`` never reaches the real ~/.llmwiki."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


LEGACY_ID = f"G1|{AUDIO}"
A_AT = "2024-01-05T00:00:00Z"
B_AT = "2024-02-02T00:00:00Z"


def head_payload(identity: str, text: str, **fields) -> dict:
    return {
        "text": text, "provider": "assemblyai_remote_url", "artifact_url": AUDIO,
        "language": "", "timed": False, "speaker_labeled": False, "document_title": "",
        "identity": identity, "normalization_version": 2, "podcast_admission_version": 1,
        "acquired_at": A_AT, **fields,
    }


def write_backup(tmp_path: Path, raw: bytes | dict, identity: str = LEGACY_ID) -> Path:
    """An operator backup copy, outside the cache, under its cache-key file name."""
    folder = tmp_path / "operator-backup"
    folder.mkdir(exist_ok=True)
    path = folder / f"{tr._cache_key(identity)}.json"
    if isinstance(raw, dict):
        raw = json.dumps(raw, ensure_ascii=False, indent=2).encode("utf-8")
    path.write_bytes(raw)
    return path


def audio_binding(world) -> tuple[str, dict]:
    routes, requests, asr, vault = world
    apple_routes(routes, rss("Pub", {**A, "guid": "G1", "audio": AUDIO}))
    url = apple_url("1001")
    [audio] = [b for b in rec.describe(url)["bindings"] if b["artifact"] == ["audio", AUDIO]]
    return url, audio


def attest(path: Path, binding: dict) -> dict:
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"legacy_sha256": sha, "binding_identity": binding["identity"],
            "acknowledgement": rec.acknowledgement(sha, binding["digest"])}


def target_of(binding: dict) -> Path:
    return tr._cache_dir() / f"{tr._cache_key(binding['identity'])}.json"


def temporaries(vault: Path) -> list[Path]:
    folder = cache_dir(vault)
    return sorted(folder.glob(".transcript-*")) if folder.exists() else []


# ── Byte binding: the attested snapshot is what is imported ────────────────

def test_backup_a_is_imported_not_the_same_key_default_cache_entry_b(world, tmp_path):
    routes, requests, asr, vault = world
    url, audio = audio_binding(world)
    default_b = write_head_entry(LEGACY_ID, B_TEXT, provider="assemblyai_remote_url",
                                 artifact_url=AUDIO, acquired_at=B_AT)
    backup_a = write_backup(tmp_path, head_payload(
        LEGACY_ID, A_TEXT, language="en", speaker_labeled=True, document_title="A doc"))
    assert backup_a.name == default_b.name  # same identity, different bytes
    b_bytes, a_bytes = default_b.read_bytes(), backup_a.read_bytes()
    args = attest(backup_a, audio)

    record = rec.import_legacy(backup_a, operator="op", **args)

    payload = json.loads(Path(record["file"]).read_text(encoding="utf-8"))
    assert payload["text"] == A_TEXT.strip() and "EPISODE-B" not in payload["text"]
    assert (payload["language"], payload["speaker_labeled"], payload["document_title"]) == (
        "en", True, "A doc")
    assert payload["acquired_at"] == A_AT  # the snapshot's, not B's
    imp = payload["podcast_binding"]["import"]
    assert imp["legacy_sha256"] == hashlib.sha256(a_bytes).hexdigest()
    assert imp["original_acquired_at"] == A_AT
    assert default_b.read_bytes() == b_bytes and backup_a.read_bytes() == a_bytes
    source = extractors.extract(url)
    assert "EPISODE-A-SPOKEN" in source.content and "EPISODE-B" not in source.content
    assert asr == []


def test_valid_backup_with_no_default_cache_copy_is_imported(world, tmp_path):
    routes, requests, asr, vault = world
    url, audio = audio_binding(world)
    backup = write_backup(tmp_path, head_payload(LEGACY_ID, A_TEXT))
    assert not (tr._cache_dir() / backup.name).exists()

    record = rec.import_legacy(backup, **attest(backup, audio))

    [payload] = bound_payloads(vault)
    assert record["file"] == str(target_of(audio))
    assert payload["text"] == A_TEXT.strip() and payload["acquired_at"] == A_AT
    assert sorted(snapshot(vault)) == [target_of(audio).name]  # nothing else written
    assert "EPISODE-A-SPOKEN" in extractors.extract(url).content and asr == []


def test_source_rewritten_after_its_snapshot_never_reaches_the_import(world, tmp_path,
                                                                       monkeypatch):
    """Any reopen of the supplied file after hashing would read B and diverge."""
    routes, requests, asr, vault = world
    url, audio = audio_binding(world)
    backup = write_backup(tmp_path, head_payload(LEGACY_ID, A_TEXT))
    args = attest(backup, audio)
    b_bytes = json.dumps(head_payload(LEGACY_ID, B_TEXT, acquired_at=B_AT)).encode()
    opens = []
    real_io_open, real_os_open = io.open, os.open

    def swap_after_first_open(path):
        if os.fspath(path) == os.fspath(backup):
            opens.append(path)
            if len(opens) == 2:  # someone edits the backup between two reads
                with real_io_open(backup, "wb") as handle:
                    handle.write(b_bytes)

    def io_open(file, *a, **k):
        if isinstance(file, (str, os.PathLike)):
            swap_after_first_open(file)
        return real_io_open(file, *a, **k)

    def os_open(path, *a, **k):
        swap_after_first_open(path)
        return real_os_open(path, *a, **k)

    with monkeypatch.context() as m:  # scoped: the world fixture's patches stay
        m.setattr(io, "open", io_open)
        m.setattr("builtins.open", io_open)
        m.setattr(os, "open", os_open)
        rec.import_legacy(backup, **args)

    assert len(opens) == 1  # exactly one snapshot of the supplied file
    [payload] = bound_payloads(vault)
    assert payload["text"] == A_TEXT.strip()
    assert payload["podcast_binding"]["import"]["legacy_sha256"] == args["legacy_sha256"]


def test_source_changed_during_publication_leaves_only_the_attested_entry(
    world, tmp_path, monkeypatch,
):
    """The former post-write re-hash raised *after* publishing a trusted entry."""
    routes, requests, asr, vault = world
    url, audio = audio_binding(world)
    backup = default_legacy()
    args = attest(backup, audio)
    real_fsync = os.fsync

    def fsync(fd):
        if backup.read_bytes().find(b"EPISODE-B") < 0:
            backup.write_bytes(json.dumps(head_payload(LEGACY_ID, B_TEXT)).encode())
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    try:
        rec.import_legacy(backup, **args)
    except rec.ImportRefusedError:
        assert bound_payloads(vault) == []  # a refusal must leave nothing trusted
        raise
    [payload] = bound_payloads(vault)
    assert payload["text"] == A_TEXT.strip()
    assert payload["podcast_binding"]["import"]["legacy_sha256"] == args["legacy_sha256"]


# ── Admission: the shared decoder, never bypassed ──────────────────────────

def _public(**fields):
    return head_payload("G1", A_TEXT, provider="rss_podcast_transcript",
                        artifact_url=VTT_URL, **fields)


ADMISSION_CASES = {
    "pre_admission_public": (lambda p: p.pop("podcast_admission_version"), "G1", "tag"),
    "old_normalization_public": (lambda p: p.update(normalization_version=1), "G1", "tag"),
    "short_text_asr": (lambda p: p.update(text="too short"), LEGACY_ID, "audio"),
    "blank_text_asr": (lambda p: p.update(text=" " * 400), LEGACY_ID, "audio"),
}


@pytest.mark.parametrize("case", sorted(ADMISSION_CASES))
def test_backup_only_entry_rejected_by_current_admission_is_refused(world, tmp_path, case):
    routes, requests, asr, vault = world
    mutate, identity, kind = ADMISSION_CASES[case]
    routes[FEED] = rss("Pub", {**A, "guid": "G1", "vtt": VTT_URL, "audio": AUDIO})
    routes[VTT_URL] = vtt_body("EPISODE-A-SPOKEN")
    [binding] = [b for b in rec.describe(FEED)["bindings"] if b["artifact"][0] == kind]
    payload = _public() if kind == "tag" else head_payload(identity, A_TEXT)
    mutate(payload)
    backup = write_backup(tmp_path, payload, identity)
    assert tr.decode_transcript_cache_bytes(backup.read_bytes()) is None  # shared decoder
    with pytest.raises(rec.ImportRefusedError, match="admission"):
        rec.import_legacy(backup, **attest(backup, binding))
    assert bound_payloads(vault) == [] and temporaries(vault) == []


MALFORMED = {
    "not_json": b"{ this is not json",
    "truncated": json.dumps(head_payload(LEGACY_ID, A_TEXT)).encode()[:-7],
    "json_list": json.dumps([head_payload(LEGACY_ID, A_TEXT)]).encode(),
    "invalid_utf8": b'{"text": "\xff\xfe", "identity": "G1"}',
    "empty": b"",
}


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_malformed_backup_is_refused_and_writes_nothing(world, tmp_path, case):
    routes, requests, asr, vault = world
    url, audio = audio_binding(world)
    backup = write_backup(tmp_path, MALFORMED[case])
    with pytest.raises(rec.ImportRefusedError, match="not a transcript cache payload"):
        rec.import_legacy(backup, **attest(backup, audio))
    assert bound_payloads(vault) == [] and temporaries(vault) == []


def test_decoder_is_shared_with_the_normal_cache_loader(world, tmp_path):
    raw = json.dumps(head_payload(LEGACY_ID, A_TEXT, language="en")).encode()
    decoded = tr.decode_transcript_cache_bytes(raw)
    path = tr._cache_dir() / f"{tr._cache_key(LEGACY_ID)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    assert decoded is not None and tr.load_transcript_cache_record(LEGACY_ID) == decoded


def test_hash_mismatch_and_wrong_acknowledgement_on_a_backup_write_nothing(world, tmp_path):
    routes, requests, asr, vault = world
    url, audio = audio_binding(world)
    backup = write_backup(tmp_path, head_payload(LEGACY_ID, A_TEXT))
    args = attest(backup, audio)
    with pytest.raises(rec.ImportRefusedError, match="sha256 mismatch"):
        rec.import_legacy(backup, **{**args, "legacy_sha256": "0" * 64})
    with pytest.raises(rec.ImportRefusedError, match="acknowledgement"):
        rec.import_legacy(backup, **{**args, "acknowledgement": args["acknowledgement"] + "."})
    renamed = backup.with_name("0" * 64 + ".json")
    renamed.write_bytes(backup.read_bytes())
    with pytest.raises(rec.ImportRefusedError, match="file name"):
        rec.import_legacy(renamed, **args)
    with pytest.raises(rec.ImportRefusedError, match="unreadable|regular file"):
        rec.import_legacy(tmp_path / "operator-backup", **args)  # a directory
    assert bound_payloads(vault) == [] and temporaries(vault) == []


# ── Publication: atomic, never overwriting ─────────────────────────────────


def default_legacy() -> Path:
    """The ordinary case: the legacy entry still sits in the default cache."""
    return write_head_entry(LEGACY_ID, A_TEXT, provider="assemblyai_remote_url",
                            artifact_url=AUDIO, acquired_at=A_AT)

def test_concurrent_fresh_entry_is_never_overwritten_by_an_import(world, tmp_path,
                                                                  monkeypatch):
    routes, requests, asr, vault = world
    url, audio = audio_binding(world)
    backup = default_legacy()
    args = attest(backup, audio)
    target = target_of(audio)
    real_fsync = os.fsync
    raced = []

    def fresh_writer_wins(fd):
        # A fresh acquisition publishes after the import's existence check.
        if not raced:
            raced.append(True)
            tr.save_transcript_cache(audio["identity"], tr.TranscriptResult(
                text=B_TEXT, provider="assemblyai_remote_url", artifact_url=AUDIO,
            ), provenance={"origin": "acquired", "marker": "fresh"})
        return real_fsync(fd)

    with monkeypatch.context() as m, pytest.raises(rec.ImportRefusedError, match="already"):
        m.setattr(os, "fsync", fresh_writer_wins)
        rec.import_legacy(backup, **args)
    assert raced
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["podcast_binding"] == {"origin": "acquired", "marker": "fresh"}
    assert payload["text"] == B_TEXT
    assert temporaries(vault) == []


def test_dangling_symlink_at_the_target_is_not_replaced(world, tmp_path):
    routes, requests, asr, vault = world
    url, audio = audio_binding(world)
    backup = default_legacy()
    target = target_of(audio)
    target.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside" / "evil.json"
    target.symlink_to(outside)
    with pytest.raises(rec.ImportRefusedError, match="already"):
        rec.import_legacy(backup, **attest(backup, audio))
    assert target.is_symlink() and os.readlink(target) == str(outside)
    assert not outside.exists() and not outside.parent.exists()
    assert temporaries(vault) == []


def test_failed_write_leaves_no_entry_and_no_temporary(world, tmp_path, monkeypatch):
    routes, requests, asr, vault = world
    url, audio = audio_binding(world)
    backup = default_legacy()
    args = attest(backup, audio)

    def disk_full(fd):
        raise OSError(errno.ENOSPC, "No space left on device")

    with monkeypatch.context() as m, pytest.raises(rec.ImportRefusedError, match="not published"):
        m.setattr(os, "fsync", disk_full)
        rec.import_legacy(backup, **args)
    assert not os.path.lexists(target_of(audio)) and temporaries(vault) == []
    assert tr._cache_dir() == cache_dir(vault).resolve()
    rec.import_legacy(backup, **args)  # the operator may simply retry
    assert len(bound_payloads(vault)) == 1


def test_publication_failure_after_the_write_leaves_nothing(world, tmp_path, monkeypatch):
    routes, requests, asr, vault = world
    url, audio = audio_binding(world)
    backup = default_legacy()

    def broken(*a, **k):
        raise OSError(errno.EIO, "I/O error")

    monkeypatch.setattr(os, "link", broken)
    monkeypatch.setattr(tr, "_rename_noreplace", broken)
    with pytest.raises(rec.ImportRefusedError, match="not published"):
        rec.import_legacy(backup, **attest(backup, audio))
    assert not os.path.lexists(target_of(audio)) and temporaries(vault) == []


def test_filesystem_without_hard_links_still_publishes_without_overwrite(
    world, tmp_path, monkeypatch,
):
    routes, requests, asr, vault = world
    url, audio = audio_binding(world)
    backup = default_legacy()
    args = attest(backup, audio)

    def no_links(*a, **k):
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", no_links)
    if not tr._rename_noreplace_supported():
        pytest.skip("renameat2(RENAME_NOREPLACE) unavailable on this platform")
    rec.import_legacy(backup, **args)
    assert len(bound_payloads(vault)) == 1 and temporaries(vault) == []
    target = target_of(audio)
    before = target.read_bytes()
    target.unlink()
    target.symlink_to(tmp_path / "nowhere.json")
    with pytest.raises(rec.ImportRefusedError, match="already"):
        rec.import_legacy(backup, **args)
    assert target.is_symlink() and temporaries(vault) == []
    assert before  # the first, completed publication was a full payload


def test_normal_cache_refresh_still_replaces(world):
    first = tr.TranscriptResult(text=A_TEXT, provider="assemblyai_remote_url")
    second = tr.TranscriptResult(text=B_TEXT, provider="assemblyai_remote_url")
    tr.save_transcript_cache("refresh-identity", first)
    tr.save_transcript_cache("refresh-identity", second)
    assert tr.load_transcript_cache("refresh-identity").text == B_TEXT.strip()


def test_exclusive_save_respects_writes_disabled_policy(world, tmp_path):
    from obsidian_llm_wiki.ingest.runtime import extraction_policy

    routes, requests, asr, vault = world
    url, audio = audio_binding(world)
    backup = default_legacy()
    with extraction_policy(allow_writes=False), pytest.raises(
        rec.ImportRefusedError, match="disabled",
    ):
        rec.import_legacy(backup, **attest(backup, audio))
    assert bound_payloads(vault) == []


# ── CLI ─────────────────────────────────────────────────────────────────────

def _cli(backup: Path, binding: dict, tmp_path: Path, **override) -> list[str]:
    args = attest(backup, binding)
    args.update(override)
    binding_file = tmp_path / "binding.txt"
    binding_file.write_text(args["binding_identity"] + "\n", encoding="utf-8")
    return ["import", "--legacy-file", str(backup), "--legacy-sha256", args["legacy_sha256"],
            "--binding-file", str(binding_file), "--acknowledgement", args["acknowledgement"],
            "--operator", "cli-op"]


def test_cli_imports_backup_a_over_default_b(world, tmp_path, capsys):
    routes, requests, asr, vault = world
    url, audio = audio_binding(world)
    default_b = write_head_entry(LEGACY_ID, B_TEXT, provider="assemblyai_remote_url",
                                 artifact_url=AUDIO, acquired_at=B_AT)
    b_bytes = default_b.read_bytes()
    backup = write_backup(tmp_path, head_payload(LEGACY_ID, A_TEXT))
    assert rec.main(_cli(backup, audio, tmp_path)) == 0
    out = json.loads(capsys.readouterr().out)
    payload = json.loads(Path(out["file"]).read_text(encoding="utf-8"))
    assert payload["text"] == A_TEXT.strip() and payload["acquired_at"] == A_AT
    assert payload["podcast_binding"]["import"]["operator"] == "cli-op"
    assert default_b.read_bytes() == b_bytes


@pytest.mark.parametrize("case", ["malformed", "hash_mismatch", "admission", "exists"])
def test_cli_refusals_exit_2_and_write_nothing(world, tmp_path, capsys, case):
    routes, requests, asr, vault = world
    url, audio = audio_binding(world)
    override = {}
    if case == "malformed":
        backup = write_backup(tmp_path, b"[]")
    else:
        backup = write_backup(tmp_path, head_payload(
            LEGACY_ID, "short" if case == "admission" else A_TEXT))
    if case == "hash_mismatch":
        override = {"legacy_sha256": "f" * 64}
    if case == "exists":
        target = target_of(audio)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(tmp_path / "dangling.json")
    before = {p.name: (p.is_symlink(), p.is_symlink() or p.read_bytes())
              for p in cache_dir(vault).glob("*")} if cache_dir(vault).exists() else {}
    argv = _cli(backup, audio, tmp_path, **override)
    if case == "hash_mismatch":
        argv[argv.index("--acknowledgement") + 1] = rec.acknowledgement("f" * 64, audio["digest"])
    assert rec.main(argv) == 2
    assert "import refused" in capsys.readouterr().err
    after = {p.name: (p.is_symlink(), p.is_symlink() or p.read_bytes())
             for p in cache_dir(vault).glob("*")} if cache_dir(vault).exists() else {}
    assert after == before
