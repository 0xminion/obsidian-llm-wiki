"""Operator-attested recovery of legacy podcast transcript cache entries.

Podcast transcript caches moved to binding epoch 3 (see
``docs/podcast-authority-binding.md``). Version-1 (media-keyed) and version-2
entries are never read automatically: neither can prove which episode its text
belongs to. Their files are left byte-identical.

This module is the only way an old entry may be served again, and only after
an operator has checked it and attests to it exactly:

``describe(URL)``
    Resolves the episode identity (metadata + feed discovery only: no
    transcript fetch, no ASR, no cache writes) and reports, per publishable
    artifact, the exact epoch-3 binding identity, its digest, any existing
    epoch-3 entry, and legacy v1/v2 files naming that artifact.
``import_legacy(file, legacy_sha256=, binding_identity=, acknowledgement=)``
    Copies one legacy entry under that binding with ``origin=operator_import``.
    It requires the file's exact SHA-256, the exact epoch-3 binding identity and
    the exact acknowledgement sentence for that pair. The supplied file is read
    exactly once; that one snapshot is hashed, decoded by the shared cache
    decoder (same admission rules as the cache itself) and imported, so the
    attested SHA-256 always names the bytes actually imported. The file may be
    the default-cache entry or an operator backup copy kept under its cache-key
    name; no other file is read. The original ``acquired_at`` is kept, and the
    import record (legacy epoch, identity, file, SHA-256, import time, operator)
    is stored with it. Publication is atomic and never overwrites: if anything,
    even a fresh entry written concurrently or a dangling symlink, occupies the
    epoch-3 key, the import is refused and nothing is written. Extraction labels
    the transcript as an operator-attested import and never as freshly acquired.

Nothing here contacts a transcription provider or deletes or rewrites a file.

CLI::

    python -m obsidian_llm_wiki.ingest.podcast_cache_recovery describe URL
    python -m obsidian_llm_wiki.ingest.podcast_cache_recovery import \\
        --legacy-file FILE --legacy-sha256 SHA --binding-file BINDING.txt \\
        --acknowledgement "I attest ..." [--operator NAME]

``VAULT_PATH`` selects the vault cache directory, as for extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from obsidian_llm_wiki.ingest import transcript_resolver as tr
from obsidian_llm_wiki.ingest.extractors import podcast

__all__ = ["ImportRefusedError", "acknowledgement", "describe", "import_legacy"]


class ImportRefusedError(RuntimeError):
    """An import precondition failed; nothing was written."""


def acknowledgement(legacy_sha256: str, binding_digest: str) -> str:
    """The exact sentence an operator must supply for one (file, binding) pair."""
    return (
        f"I attest that legacy podcast transcript {legacy_sha256} is the artifact of "
        f"episode binding {binding_digest}; it was not freshly acquired"
    )


_acknowledgement = acknowledgement  # import_legacy's parameter shadows the name


def _digest(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _platform_for(url: str) -> tuple[str, str | None]:
    parsed = urlparse(url)
    if podcast._is_spotify(parsed, url):
        return "spotify", None
    if podcast._is_apple_podcast(parsed, url):
        return "apple", None
    if podcast._is_rss_feed(url):
        return "rss", podcast._fetch_rss_text(url)
    return "generic", None


# Far above any transcript (public artifacts are capped at 10 MB before
# normalization); bounds memory if an operator points at the wrong file.
_MAX_LEGACY_BYTES = 64 * 1024 * 1024


def _snapshot(path: Path) -> bytes:
    """Read ``path`` exactly once through one descriptor; refuse a moving target.

    ``O_NONBLOCK`` keeps a FIFO from hanging the open; only regular files are
    accepted. A size/mtime/ctime change across the read (a concurrent writer)
    refuses rather than importing a torn copy.
    """
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ImportRefusedError(f"legacy file unreadable: {exc}") from exc
    chunks: list[bytes] = []
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ImportRefusedError("legacy file is not a regular file")
        total = 0
        while chunk := os.read(fd, 1 << 20):
            total += len(chunk)
            if total > _MAX_LEGACY_BYTES:
                raise ImportRefusedError("legacy file is too large to be a transcript entry")
            chunks.append(chunk)
        after = os.fstat(fd)
    except OSError as exc:
        raise ImportRefusedError(f"legacy file unreadable: {exc}") from exc
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    moved = (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if moved or len(raw) != after.st_size:
        raise ImportRefusedError("legacy file changed while it was being read; retry")
    return raw


def _legacy_epoch(identity: str) -> int:
    if identity.startswith(podcast._CACHE_BINDING_HEAD + "\n"):
        return podcast._CACHE_BINDING_VERSION
    if identity.startswith(podcast._LEGACY_V2_HEAD + "\n"):
        return 2
    return 1


def _artifact_of(payload: dict) -> tuple[str, str]:
    provider = str(payload.get("provider", ""))
    return podcast._artifact_kind(provider), str(payload.get("artifact_url", ""))


def _artifact_matches(payload: dict, kind: str, url: str) -> bool:
    got_kind, got_url = _artifact_of(payload)
    return got_kind == kind and (got_url == url or (kind == "audio" and not got_url))


def _candidates(binding: dict, kind: str, url: str) -> list[dict]:
    identities = [*podcast._legacy_v1_identities(binding),
                  podcast._legacy_v2_identity(binding, kind, url)]
    out = []
    for identity in identities:
        path = tr._cache_dir() / f"{tr._cache_key(identity)}.json"
        try:  # one read: the reported sha256 and admissibility describe the same bytes
            raw = path.read_bytes() if path.is_file() else None
        except OSError:
            raw = None
        payload = tr._parse_cache_bytes(raw) if raw is not None else None
        if payload is None or not _artifact_matches(payload, kind, url):
            continue
        sha = hashlib.sha256(raw).hexdigest()
        epoch = _legacy_epoch(identity)
        out.append({
            "file": str(path),
            "sha256": sha,
            "epoch": epoch,
            "identity": identity,
            "provider": payload.get("provider", ""),
            "artifact_url": payload.get("artifact_url", ""),
            "acquired_at": payload.get("acquired_at", ""),
            "chars": len(str(payload.get("text", ""))),
            "admissible": tr._admit_cache_payload(payload) is not None,
            "warning": (
                "version-1 keys name media, not an episode; verify the text is this episode"
                if epoch == 1 else
                "version-2 entries may be unmarked copies of a version-1 entry made by the "
                "former automatic migration; verify the text is this episode"
            ),
        })
    return out


def describe(url: str) -> dict:
    """Read-only report of exact epoch-3 bindings and legacy candidates for ``url``."""
    platform, rss_text = _platform_for(url)
    ctx = podcast._episode_context(url, platform, rss_text)
    warnings = []
    if ctx.asset.media_shared:
        warnings.append(
            "another item in this feed shares this item's GUID or enclosure: a legacy "
            "media-keyed entry may belong to that other episode"
        )
    bindings = []
    for kind, artifact_url in ctx.plan:
        identity = podcast._bound_cache_identity(ctx.binding, kind, artifact_url)
        digest = _digest(identity)
        record = tr.load_transcript_cache_record(identity)
        current = None
        if record is not None:
            prov = record[1].get("podcast_binding") or {}
            current = {
                "file": str(tr._cache_dir() / f"{tr._cache_key(identity)}.json"),
                "origin": prov.get("origin", ""), "acquired_at": record[1].get("acquired_at"),
            }
        candidates = _candidates(ctx.binding, kind, artifact_url)
        for candidate in candidates:
            candidate["acknowledgement"] = acknowledgement(candidate["sha256"], digest)
        bindings.append({
            "artifact": [kind, artifact_url], "identity": identity, "digest": digest,
            "v3_entry": current, "legacy_candidates": candidates,
        })
    return {
        "url": url, "platform": platform, "episode": ctx.binding,
        "warnings": warnings, "bindings": bindings,
    }


_NOT_A_BINDING = "binding identity is not an exact epoch-3 podcast binding"


def _parse_binding(identity: str) -> tuple[dict, str, str]:
    head, sep, body = identity.partition("\n")
    if head != podcast._CACHE_BINDING_HEAD or not sep:
        raise ImportRefusedError(_NOT_A_BINDING)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ImportRefusedError(_NOT_A_BINDING) from exc
    artifact = data.pop("artifact", None) if isinstance(data, dict) else None
    if (
        not isinstance(artifact, list) or len(artifact) != 2
        or artifact[0] not in ("tag", "page", "link", "audio") or not artifact[1]
        or data.get("episode") not in ("feed-item", "requested-page")
        or podcast._bound_cache_identity(data, *artifact) != identity
    ):
        raise ImportRefusedError(_NOT_A_BINDING)
    return data, artifact[0], artifact[1]


def import_legacy(
    legacy_file: str | Path,
    *,
    legacy_sha256: str,
    binding_identity: str,
    acknowledgement: str,  # noqa: A002 - the exact operator sentence
    operator: str = "",
) -> dict:
    """Import one legacy entry under an exact epoch-3 binding; never overwrite.

    Everything imported comes from one snapshot of ``legacy_file``: the bytes
    whose SHA-256 the operator attested. Nothing is re-read from that file or
    loaded from the cache by identity.
    """
    path = Path(legacy_file)
    raw = _snapshot(path)
    sha = hashlib.sha256(raw).hexdigest()
    if sha != legacy_sha256.strip().lower():
        raise ImportRefusedError(f"legacy file sha256 mismatch (file is {sha})")
    binding, kind, url = _parse_binding(binding_identity)
    digest = _digest(binding_identity)
    expected = _acknowledgement(sha, digest)
    if acknowledgement != expected:
        raise ImportRefusedError("acknowledgement does not match the exact required sentence")

    payload = tr._parse_cache_bytes(raw)
    if payload is None:
        raise ImportRefusedError("legacy file is not a transcript cache payload")
    identity = str(payload.get("identity", ""))
    if not identity or path.name != f"{tr._cache_key(identity)}.json":
        raise ImportRefusedError("legacy file name does not match its stored identity")
    epoch = _legacy_epoch(identity)
    if epoch == podcast._CACHE_BINDING_VERSION:
        raise ImportRefusedError("file is already an epoch-3 entry; nothing to import")
    if epoch == 2 and identity != podcast._legacy_v2_identity(binding, kind, url):
        raise ImportRefusedError("legacy v2 identity names a different episode binding or artifact")
    if epoch == 1 and identity not in podcast._legacy_v1_identities(binding):
        raise ImportRefusedError("legacy v1 identity is not a version-1 key of this binding")
    if not _artifact_matches(payload, kind, url):
        got = _artifact_of(payload)
        raise ImportRefusedError(f"legacy artifact {list(got)} differs from binding artifact "
                            f"{[kind, url]}")
    # The shared cache decoder, applied to the attested snapshot itself.
    result = tr._admit_cache_payload(payload)
    if result is None:
        raise ImportRefusedError(
            "legacy entry is rejected by current cache admission (public artifacts are "
            "refetched for free instead)"
        )
    target = tr._cache_dir() / f"{tr._cache_key(binding_identity)}.json"
    if os.path.lexists(target):  # early, friendly; the publication itself is exclusive
        raise ImportRefusedError(f"binding already has an epoch-3 entry: {target}")

    original_acquired_at = payload.get("acquired_at") or None
    imported_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    provenance = podcast._binding_provenance(
        binding, kind, url, origin="operator_import",
        **{"import": {
            "legacy_epoch": epoch,
            "legacy_identity": identity,
            "legacy_file": path.name,
            "legacy_sha256": sha,
            "original_provider": payload.get("provider", ""),
            "original_acquired_at": original_acquired_at,
            "imported_at": imported_at,
            "operator": operator,
            "attestation_sha256": hashlib.sha256(acknowledgement.encode("utf-8")).hexdigest(),
            "binding_digest": digest,
        }},
    )
    try:
        written = tr.save_transcript_cache_exclusive(
            binding_identity, result, provenance=provenance,
            acquired_at=original_acquired_at or "unknown",
        )
    except FileExistsError as exc:
        raise ImportRefusedError(f"binding already has an epoch-3 entry: {target}") from exc
    except OSError as exc:
        raise ImportRefusedError(f"import not published (no entry written): {exc}") from exc
    if written is None:
        raise ImportRefusedError("cache writes are disabled or the transcript is too short")
    return {"file": str(written), "identity": binding_identity, "digest": digest,
            "legacy_epoch": epoch, "legacy_sha256": sha,
            "original_acquired_at": original_acquired_at, "imported_at": imported_at}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="podcast_cache_recovery",
        description="Operator-attested recovery of legacy podcast transcript cache entries.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("describe", help="read-only binding/legacy report (JSON)")
    d.add_argument("url")
    i = sub.add_parser("import", help="operator-attested import of one legacy entry")
    i.add_argument("--legacy-file", required=True)
    i.add_argument("--legacy-sha256", required=True)
    i.add_argument("--binding-file", required=True,
                   help="file containing the exact epoch-3 binding identity from describe")
    i.add_argument("--acknowledgement", required=True)
    i.add_argument("--operator", default="")
    args = parser.parse_args(argv)
    if args.command == "describe":
        print(json.dumps(describe(args.url), indent=2, ensure_ascii=False))
        return 0
    try:
        # An identity never ends in a newline; tolerate the one an editor adds.
        identity = Path(args.binding_file).read_text(encoding="utf-8").removesuffix("\n")
        print(json.dumps(import_legacy(
            args.legacy_file, legacy_sha256=args.legacy_sha256, binding_identity=identity,
            acknowledgement=args.acknowledgement, operator=args.operator,
        ), indent=2, ensure_ascii=False))
    except (ImportRefusedError, OSError) as exc:
        print(f"import refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
