"""Build a deterministic graph from rendered Markdown wiki pages.

The graph is intentionally file-format focused: callers provide a mapping of
vault-relative paths to raw Markdown, so indexing never needs configuration or
network access.  Links and relation frontmatter are resolved only to pages
present in that mapping; dangling references are retained nowhere.
"""

from __future__ import annotations

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
        page = WikiPage(path=path, title=title, aliases=page_aliases, body=body)
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


def build_graph_from_vault(vault_path: str | Path) -> WikiGraph:
    """Read Markdown below ``vault_path`` and build a graph using relative paths."""
    vault = Path(vault_path)
    if not vault.is_dir():
        return build_graph({})
    pages: dict[str, str] = {}
    for path in sorted(vault.rglob("*.md")):
        if not path.is_file():
            continue
        try:
            pages[path.relative_to(vault).as_posix()] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
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
            relations.append((item, "relation"))
        elif isinstance(item, Mapping):
            target = item.get("target", item.get("slug", item.get("path", "")))
            relation = item.get("relation", item.get("type", "relation"))
            if target:
                relations.append((str(target), str(relation or "relation")))
    return relations
