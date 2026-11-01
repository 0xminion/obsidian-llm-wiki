"""Immutable prospective publication contract, shared by all durable consumers.

Version v1 stores canonical JSON, not mutable model objects. Producer IDs are
proposals; the payloads here are the effective assignments. The generation is a
content digest of the complete committed source/review snapshot, not wall time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from obsidian_llm_wiki.core.models import source_synthesis_to_dict

PLAN_VERSION = "identity-publication-plan-v1"


def canonical(value) -> str:
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (tuple, list)):
            return [clean(x) for x in v]
        return v

    return json.dumps(
        clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def registry_generation(registry) -> str:
    return digest(
        {
            "sources": [
                list(row)
                for row in registry.connection.execute(
                    "SELECT "
                    "s.filename,s.content_hash,sy.payload_json,"
                    "sy.prompt_version,sy.schema_version,sy.model_id "
                    "FROM source s LEFT JOIN synthesis sy ON sy.source_id=s.id ORDER BY s.filename"
                )
            ],
            "concepts": [
                list(row)
                for row in registry.connection.execute(
                    "SELECT id,slug,payload_json FROM concept ORDER BY id"
                )
            ],
            "mocs": [
                list(row)
                for row in registry.connection.execute(
                    "SELECT id,slug,payload_json FROM moc ORDER BY id"
                )
            ],
            "owners": [
                list(row)
                for row in registry.connection.execute(
                    "SELECT s.filename,cs.concept_id FROM concept_source cs JOIN "
                    "source s ON s.id=cs.source_id ORDER BY s.filename,cs.concept_id"
                )
            ],
            "contexts": [
                list(row)
                for row in registry.connection.execute(
                    "SELECT filename,revision,content FROM "
                    "identity_source_context ORDER BY filename,revision"
                )
            ],
        }
    )


def file_snapshot(paths) -> tuple[tuple[str, str | None], ...]:
    return tuple(
        (str(p), hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None)
        for p in sorted(set(map(Path, paths)))
    )


def output_snapshot(config):
    from obsidian_llm_wiki.render.obsidian import _load_page_hashes

    paths = []
    # Durable positive renderer inventory, not a scan of human source inputs.
    # Include missing owned paths so deletion invalidates generation agreement.
    for relative in _load_page_hashes(config.wiki_dir):
        path = Path(relative)
        if path.parts and path.parts[0] == "sources":
            paths.append(config.wiki_dir / path)
    for directory in (
        config.concepts_dir,
        config.wiki_dir / "entries",
        config.wiki_dir / "mocs",
        config.llmwiki_dir / "cache",
    ):
        if directory.exists():
            paths.extend(p for p in directory.rglob("*") if p.is_file())
    paths.extend(
        p
        for p in config.wiki_dir.iterdir()
        if p.is_file() and p.suffix in {".md", ".json", ".canvas"} and p.name != "log.md"
    )
    if config.state_file.is_file():
        paths.append(config.state_file)
    from obsidian_llm_wiki.core.generated_paths import checked_generated_path
    return file_snapshot(checked_generated_path(path, config.vault) for path in paths)


def checkpoint_files(registry, config):
    from obsidian_llm_wiki.core.generated_paths import checked_generated_path
    from obsidian_llm_wiki.core.source_files import source_file_path
    paths = [config.llmwiki_dir / "equivalences.json", config.llmwiki_dir / "schema.yaml",
             config.vault / ".env"]
    paths.extend(source_file_path(config.sources_dir, name) for name in registry.source_filenames()
                 if source_file_path(config.sources_dir, name).is_file())
    return file_snapshot(checked_generated_path(path, config.vault) for path in paths)


def publication_current(registry, config, policy):
    from obsidian_llm_wiki.core.generated_paths import checked_generated_path
    from obsidian_llm_wiki.core.publication_transaction import publication_pending

    if publication_pending(config.llmwiki_dir):
        return False
    try:
        path = checked_generated_path(
            config.llmwiki_dir / "identity-publication.json", config.vault,
        )
        marker = json.loads(path.read_text())
        return (
            marker["version"] == PLAN_VERSION
            and marker["status"] == "committed"
            and not marker.get("errors")
            and marker["policy"] == policy
            and marker["generation"] == registry_generation(registry)
            and marker["source_files"] == list(map(list, checkpoint_files(registry, config)))
            and marker["output_files"] == list(map(list, output_snapshot(config)))
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


@dataclass(frozen=True)
class ProspectivePublicationPlan:
    """Single immutable assignment + publication decision, before any apply."""

    version: str
    generation: str
    policy: str
    sources: tuple[tuple[str, str], ...]
    files: tuple[tuple[str, str | None], ...]
    bundle_json: str
    publication_json: str

    @classmethod
    def freeze(cls, registry, policy, bundle, publication, files):
        published = {concept.canonical_id: concept.slug for concept in bundle.concepts}
        if "" in published or len(set(published.values())) != len(published):
            raise RuntimeError("identity publication plan has incomplete or conflicting entity IDs")
        for synthesis in bundle.sources:
            identities = [concept.canonical_id for concept in synthesis.concepts]
            if len(set(identities)) != len(identities):
                raise RuntimeError("identity publication plan collapses distinct local subjects")
            for concept in synthesis.concepts:
                if not concept.canonical_id or published.get(concept.canonical_id) != concept.slug:
                    raise RuntimeError(
                        "identity publication plan omits or retargets a contribution"
                    )
        return cls(
            PLAN_VERSION,
            registry_generation(registry),
            policy,
            tuple(
                sorted(
                    (s.source_file, canonical(source_synthesis_to_dict(s))) for s in bundle.sources
                )
            ),
            tuple(files),
            canonical(asdict(bundle)),
            canonical(asdict(publication)),
        )

    @property
    def plan_id(self):
        return digest(asdict(self))

    def check_preconditions(self, registry, policy):
        if self.version != PLAN_VERSION or self.policy != policy:
            raise RuntimeError("stale identity publication plan: policy/version changed")
        if registry_generation(registry) != self.generation:
            raise RuntimeError(
                "stale identity publication plan: registry/review generation changed"
            )
        self.check_files()

    def check_files(self):
        if file_snapshot(name for name, _sha in self.files) != self.files:
            raise RuntimeError("stale identity publication plan: source or policy bytes changed")

    def check_source(self, synthesis):
        expected = dict(self.sources).get(synthesis.source_file)
        if expected != canonical(source_synthesis_to_dict(synthesis)):
            raise RuntimeError(f"identity publication plan source drift: {synthesis.source_file}")

    def check_render(self, bundle, publication):
        if canonical(asdict(bundle)) != self.bundle_json:
            raise RuntimeError("identity publication plan: rendered bundle drift")
        if canonical(asdict(publication)) != self.publication_json:
            raise RuntimeError("identity publication plan: publication decision drift")

    def verify_consumers(self, registry, config):
        from obsidian_llm_wiki.core.cache import load_synthesis

        for filename, payload in self.sources:
            row = registry.connection.execute(
                "SELECT sy.payload_json FROM synthesis sy JOIN source s ON s.id=sy.source_id "
                "WHERE s.filename=?",
                (filename,),
            ).fetchone()
            if row is None or canonical(json.loads(row[0])) != payload:
                raise RuntimeError(f"identity publication registry disagreement: {filename}")
            cached = load_synthesis(config.llmwiki_dir, filename)
            if cached is None or canonical(source_synthesis_to_dict(cached)) != payload:
                raise RuntimeError(f"identity publication cache disagreement: {filename}")
        bundle = json.loads(self.bundle_json)
        for concept in bundle["concepts"]:
            path = config.concepts_dir / (concept["slug"] + ".md")
            if not path.is_file():
                raise RuntimeError(f"identity publication missing page: {concept['slug']}")
            from obsidian_llm_wiki.render.frontmatter import parse_frontmatter

            if (
                parse_frontmatter(path.read_text())[0].get("canonical_id")
                != concept["canonical_id"]
            ):
                raise RuntimeError(
                    f"identity publication rendered ID disagreement: {concept['slug']}"
                )
            row = registry.connection.execute(
                "SELECT slug FROM concept WHERE id=?", (concept["canonical_id"],)
            ).fetchone()
            if row is None or row[0] != concept["slug"]:
                raise RuntimeError(
                    f"identity publication canonical row disagreement: {concept['slug']}"
                )

    def write_marker(self, registry, config, status, errors=()):
        from obsidian_llm_wiki.render.obsidian import atomic_write

        assignments = {
            name: [c["canonical_id"] for c in json.loads(payload)["concepts"]]
            for name, payload in self.sources
        }
        from obsidian_llm_wiki.core.generated_paths import checked_generated_path

        archive = checked_generated_path(
            config.llmwiki_dir / "publications" / (self.plan_id + ".json"), config.vault
        )
        if not archive.exists():
            atomic_write(archive, canonical(asdict(self)))
        marker_path = checked_generated_path(
            config.llmwiki_dir / "identity-publication.json", config.vault
        )
        if status == "pending":
            # Retain the last committed checkpoint while still publishing the
            # pending marker normally. This does not skip staging or recovery.
            try:
                previous = json.loads(marker_path.read_text())
            except (OSError, ValueError):
                previous = None
            registry._previous_publication_marker = previous
        marker = {
            "version": self.version,
            "plan_id": self.plan_id,
            "status": status,
            "errors": list(errors),
            "generation": registry_generation(registry),
            "policy": self.policy,
            "assignments": assignments,
            "source_files": checkpoint_files(registry, config),
            "output_files": output_snapshot(config),
            "source_payloads": {
                name: digest(json.loads(payload)) for name, payload in self.sources
            },
        }
        if status == "committed":
            previous = getattr(registry, "_previous_publication_marker", None)
            registry._previous_publication_marker = None
            # The prospective plan includes its PRE-apply generation and the
            # diagnostic reuse basis. Neither is a change to committed output.
            # Keep the old checkpoint only if every post-apply fact is exactly
            # equal, and its immutable archive still verifies. Source, ownership,
            # review, policy, errors or output changes all invalidate equality.
            if isinstance(previous, dict) and canonical({
                k: v for k, v in previous.items() if k != "plan_id"
            }) == canonical({k: v for k, v in marker.items() if k != "plan_id"}):
                old_id = previous.get("plan_id")
                if isinstance(old_id, str) and len(old_id) == 64 and all(
                    char in "0123456789abcdef" for char in old_id
                ):
                    old_archive = checked_generated_path(
                        config.llmwiki_dir / "publications" / (old_id + ".json"), config.vault
                    )
                    try:
                        archived = json.loads(old_archive.read_text())
                    except (OSError, ValueError):
                        archived = None
                    if archived is not None and digest(archived) == old_id:
                        marker = previous
        atomic_write(marker_path, canonical(marker))
