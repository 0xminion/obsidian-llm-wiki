"""Build a deterministic graph from rendered Markdown wiki pages.

The graph is intentionally file-format focused: callers provide a mapping of
vault-relative paths to raw Markdown, so indexing never needs configuration or
network access.  Links and relation frontmatter are resolved only to pages
present in that mapping; dangling references are retained nowhere.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

__all__ = [
    "GraphEdge",
    "WikiGraph",
    "WikiPage",
    "build_graph",
    "build_graph_from_vault",
    "normalize_reference",
]


@dataclass(frozen=True, slots=True)
class WikiPage:
    """One indexed Markdown page, identified by its vault-relative path."""

    path: str
    title: str
    aliases: tuple[str, ...]
    body: str
    orphaned: bool = False
    orphaned_reason: str = ""
    publication_status: str = "unverifiable"
    status_reasons: tuple[str, ...] = ()
    source_statuses: tuple[tuple[str, str], ...] = ()
    rendered_sha256: str = ""

    @property
    def current_eligible(self) -> bool:
        return not self.orphaned and self.publication_status == "eligible"


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """One resolved, directed relation between pages."""

    source: str
    target: str
    relation: str


@dataclass(frozen=True, slots=True)
class WikiGraph:
    """Immutable wiki pages and their resolved directed relations."""

    pages: Mapping[str, WikiPage]
    edges: tuple[GraphEdge, ...]
    outbound: Mapping[str, tuple[str, ...]]
    inbound: Mapping[str, tuple[str, ...]]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


def normalize_reference(value: object) -> str:
    """Return a stable lookup key for a wikilink, page path, or alias."""
    text = str(value or "").strip().removeprefix("[[").removesuffix("]]")
    text = text.split("|", 1)[0].split("#", 1)[0].strip().replace("\\", "/")
    return text.removesuffix(".md").casefold()


def build_graph(pages: Mapping[str, str]) -> WikiGraph:
    """Parse ``pages`` into a stable graph of wikilinks and relations.

    ``pages`` keys must be vault-relative Markdown paths.  The builder uses
    the render package's frontmatter/wikilink helpers lazily so importing this
    module has no YAML dependency or renderer side effects.
    """
    from obsidian_llm_wiki.render.frontmatter import extract_wikilinks, parse_frontmatter

    parsed: dict[str, WikiPage] = {}
    metadata: dict[str, dict] = {}
    aliases: dict[str, str] = {}
    for raw_path, raw_markdown in sorted(pages.items(), key=lambda item: str(item[0])):
        path = _normalize_path(raw_path)
        meta, body = parse_frontmatter(raw_markdown)
        title = str(meta.get("title") or PurePosixPath(path).stem)
        page_aliases = _string_values(meta.get("aliases"))
        orphaned = bool(meta.get("orphaned", False))
        reason = str(meta.get("orphaned_reason") or "")
        page = WikiPage(
            path=path, title=title, aliases=page_aliases, body=body,
            orphaned=orphaned, orphaned_reason=reason,
            publication_status="orphaned" if orphaned else "unverifiable",
            status_reasons=(reason or "Source owners deleted",) if orphaned else (),
            # File-backed callers decode exact UTF-8 bytes without newline
            # normalization. Bind the very text parsed here, not a later read.
            rendered_sha256=hashlib.sha256(raw_markdown.encode("utf-8")).hexdigest(),
        )
        parsed[path] = page
        metadata[path] = meta
        for reference in (path, PurePosixPath(path).stem, title, *page_aliases):
            key = normalize_reference(reference)
            if key and key not in aliases:
                aliases[key] = path

    edges: list[GraphEdge] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for path, page in parsed.items():
        targets = [(target, "wikilink") for target, _alias in extract_wikilinks(page.body)]
        targets.extend(_frontmatter_relations(metadata[path]))
        for target_ref, relation in targets:
            target = aliases.get(normalize_reference(target_ref))
            if not target or target == path:
                continue
            edge = GraphEdge(source=path, target=target, relation=relation)
            edge_key = (edge.source, edge.target, edge.relation)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append(edge)

    edges.sort(key=lambda edge: (edge.source, edge.target, edge.relation))
    outbound = {path: [] for path in parsed}
    inbound = {path: [] for path in parsed}
    for edge in edges:
        outbound[edge.source].append(edge.target)
        inbound[edge.target].append(edge.source)
    normalized_outbound = {
        key: tuple(sorted(set(value))) for key, value in outbound.items()
    }
    normalized_inbound = {
        key: tuple(sorted(set(value))) for key, value in inbound.items()
    }
    return WikiGraph(
        pages=MappingProxyType(dict(parsed)),
        edges=tuple(edges),
        outbound=MappingProxyType(normalized_outbound),
        inbound=MappingProxyType(normalized_inbound),
    )


def publication_read_token(wiki_dir: str | Path) -> str:
    """Read a durable-generation token without opening or modifying a registry."""
    from obsidian_llm_wiki.core.generated_paths import checked_generated_path
    from obsidian_llm_wiki.core.publication_transaction import require_publication_ready

    wiki = Path(wiki_dir)
    from obsidian_llm_wiki.core.immutable_publication import current_publication
    committed = current_publication(wiki)
    if committed:
        return committed[1]['generation']
    metadata = wiki / ".llmwiki"
    require_publication_ready(metadata)
    marker = checked_generated_path(metadata / "identity-publication.json", wiki)
    raw = marker.read_bytes() if marker.exists() else b""
    if marker.exists():
        try:
            data = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("invalid publication marker; recovery must be resolved") from exc
        if not isinstance(data, dict) or data.get("status") != "committed":
            raise RuntimeError("publication is not committed; recovery must be resolved")
    require_publication_ready(metadata)
    return hashlib.sha256(raw).hexdigest()


def build_graph_from_vault(
    vault_path: str | Path, *, committed=None, config=None,
) -> WikiGraph:
    """Inspect one pinned generation, retaining lifecycle data for safe retrieval.

    Omitted config resolves only the independent review policy with CLI vault
    precedence. Receipt integrity alone never establishes current eligibility.
    Unavailable policy leaves pages inspectable as unverified history.
    """
    vault = Path(vault_path)
    from obsidian_llm_wiki.core.immutable_publication import current_publication
    if committed is None:
        committed = current_publication(vault)
    if committed:
        directory, manifest = committed
        # Read the pinned directory only, never historical generations or legacy
        # human prose. Fully qualified keys match native vault-root wikilinks.
        pages = {
            '04-Wiki/' + path.relative_to(vault).as_posix(): path.read_bytes().decode('utf-8')
            for name in manifest['files']
            if name.endswith('.md') and '.llmwiki' not in Path(name).parts
            for path in [directory / name]
        }
        from obsidian_llm_wiki.core.query_freshness import classify_pages

        policy_problem = ""
        if config is None:
            from obsidian_llm_wiki.config import load_review_policy
            from obsidian_llm_wiki.core.generated_paths import checked_generated_path

            # Resolve from the caller's vault, never a receipt or historical root.
            # This is the same pure resolver used by migration inspection; it
            # constructs no Config, provider, directories or mutable registry.
            try:
                root = vault.resolve().parent
                env = checked_generated_path(root / ".env", root)
                config = load_review_policy(
                    str(env) if env.exists() else None, VAULT_PATH=str(root),
                )
                if config.grounding_review != "required":
                    policy_problem = "Current review policy does not require admission review"
            except (OSError, ValueError, TypeError) as exc:
                policy_problem = f"Current review policy unavailable: {exc}"
        return classify_pages(
            build_graph(pages), directory, manifest, config, policy_problem=policy_problem,
        )
    token = publication_read_token(vault)
    if not vault.is_dir():
        return build_graph({})
    pages: dict[str, str] = {}
    for path in sorted(vault.rglob("*.md")):
        relative = path.relative_to(vault)
        if any(part in {".llmwiki", ".obsidian", ".git", "OLW-Publications"}
               for part in relative.parts):
            continue
        if not path.is_file():
            continue
        try:
            pages[relative.as_posix()] = path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    if publication_read_token(vault) != token:
        raise RuntimeError("publication changed during retrieval; retry the query")
    return build_graph(pages)


def _normalize_path(path: object) -> str:
    normalized = str(path).replace("\\", "/").lstrip("/")
    return str(PurePosixPath(normalized))


def _string_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _frontmatter_relations(meta: Mapping[str, object]) -> list[tuple[str, str]]:
    """Accept the common list and mapping shapes used for relations metadata."""
    raw = meta.get("relations", meta.get("related", []))
    if isinstance(raw, str):
        return [(raw, "relation")]
    if isinstance(raw, Mapping):
        return [(str(target), str(relation or "relation")) for target, relation in raw.items()]
    if not isinstance(raw, (list, tuple)):
        return []

    relations: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, str):
            # Rendered relations are pipe-separated: "slug|relation_type|display".
            # Use the first segment as the target slug; the second (if present)
            # as the relation type.
            parts = item.split("|")
            target = parts[0].strip()
            if not target:
                continue
            relation_type = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "relation"
            relations.append((target, relation_type))
        elif isinstance(item, Mapping):
            target = item.get("target", item.get("slug", item.get("path", "")))
            relation = item.get("relation", item.get("type", "relation"))
            if target:
                relations.append((str(target), str(relation or "relation")))
    return relations
