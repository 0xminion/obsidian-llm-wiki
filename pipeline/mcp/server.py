"""MCP (Model Context Protocol) server entry point for obsidian-llm-wiki.

Exposes pipeline tools (ingest, compile, query, search, lint, status) as MCP tools.
Read-only vault views as MCP resources.

Transport: stdio-compatible function dispatch (simplified for Python).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pipeline.config import Config, load_config
from pipeline.compile.core import run_compile, CompileResult
# Semantic compile imports — available when compile/semantic.py has v2.0 features
# from pipeline.compile.semantic import resolve_crosslinks, rebuild_mocs

log = logging.getLogger(__name__)


class WikiMCPServer:
    """MCP server exposing wiki tools and resources."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.tools = self._register_tools()
        self.resources = self._register_resources()

    def _register_tools(self) -> dict[str, Any]:
        return {
            "ingest_source": self._tool_ingest,
            "compile_wiki": self._tool_compile,
            "query_wiki": self._tool_query,
            "search_pages": self._tool_search,
            "read_page": self._tool_read,
            "lint_wiki": self._tool_lint,
            "wiki_status": self._tool_status,
        }

    def _register_resources(self) -> dict[str, Any]:
        return {
            "llmwiki://index": self._resource_index,
            "llmwiki://concept/{slug}": self._resource_concept,
            "llmwiki://sources": self._resource_sources,
            "llmwiki://state": self._resource_state,
        }

    # ─── Tools ─────────────────────────────────────────────────────────────────

    def _tool_ingest(self, source: str) -> dict:
        """Ingest a URL or local file into the vault."""
        from pipeline.cli.ingest import ingest_command
        from pipeline.config import load_config
        cfg = load_config()
        result = ingest_command([source], cfg=cfg)
        return {"status": "ok", "result": result}

    def _tool_compile(self) -> dict:
        """Run incremental compile pipeline."""
        result = run_compile(self.cfg)
        return {
            "compiled": result.compiled,
            "skipped": result.skipped,
            "deleted": result.deleted,
            "pages": result.pages,
            "errors": result.errors,
        }

    def _tool_query(self, question: str, save: bool = False) -> dict:
        """Query the wiki for answers."""
        from pipeline.qmd import run_qmd_concept_search
        return {"answer": f"Query received: {question}", "sources": []}

    def _tool_search(self, question: str) -> dict:
        """Search pages relevant to a question."""
        from pipeline.qmd import run_qmd_concept_search
        queries = {"search": question}
        matches = run_qmd_concept_search(queries, self.cfg)
        return {"pages": matches}

    def _tool_read(self, slug: str) -> dict:
        """Read a single concept page by slug."""
        concept_path = self.cfg.concepts_dir / f"{slug}.md"
        if not concept_path.exists():
            return {"error": f"Page not found: {slug}"}
        content = concept_path.read_text(encoding="utf-8")
        return {"slug": slug, "content": content}

    def _tool_lint(self) -> dict:
        """Run quality checks and return diagnostics."""
        from pipeline.lint.runner import run_lint
        issues = run_lint(self.cfg.vault_path)
        return {"issues": len(issues), "details": [str(i) for i in issues]}

    def _tool_status(self) -> dict:
        """Get wiki status: page count, source count, orphans, pending changes."""
        concepts = list(self.cfg.concepts_dir.glob("*.md")) if self.cfg.concepts_dir.exists() else []
        entries = list(self.cfg.entries_dir.glob("*.md")) if self.cfg.entries_dir.exists() else []
        sources = list(self.cfg.sources_dir.glob("*.md")) if self.cfg.sources_dir.exists() else []
        return {
            "concepts": len(concepts),
            "entries": len(entries),
            "sources": len(sources),
            "total_pages": len(concepts) + len(entries),
        }

    # ─── Resources ────────────────────────────────────────────────────────────

    def _resource_index(self) -> str:
        """Return wiki index.md content."""
        index_path = self.cfg.wiki_index
        if index_path.exists():
            return index_path.read_text(encoding="utf-8")
        return ""

    def _resource_concept(self, slug: str) -> dict:
        """Return a single concept page."""
        return self._tool_read(slug)

    def _resource_sources(self) -> list[dict]:
        """List ingested source files."""
        sources = []
        if self.cfg.sources_dir.exists():
            for md in self.cfg.sources_dir.glob("*.md"):
                try:
                    content = md.read_text(encoding="utf-8")
                    from pipeline.utils import parse_frontmatter
                    fm = parse_frontmatter(content)
                    sources.append({"filename": md.name, "frontmatter": fm})
                except OSError:
                    continue
        return sources

    def _resource_state(self) -> dict:
        """Return compilation state."""
        from pipeline.store import ContentStore
        with ContentStore.open_vault_cache(self.cfg.vault_path) as store:
            raw = store.cache_get("source_hashes_v2")
            return {"state_json": raw or "{}"}


def create_server(vault_path: str | None = None) -> WikiMCPServer:
    """Factory: create MCP server for a vault."""
    cfg = load_config(Path(vault_path) if vault_path else None)
    return WikiMCPServer(cfg)


def run_stdio_server(vault_path: str | None = None) -> None:
    """Run MCP server reading JSON-RPC-ish requests from stdin."""
    server = create_server(vault_path)
    import sys
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            method = req.get("method", "")
            params = req.get("params", {})

            if method in server.tools:
                result = server.tools[method](**params)
                print(json.dumps({"result": result}))
            elif method.startswith("llmwiki://"):
                slug = method.split("/")[-1]
                result = server._tool_read(slug)
                print(json.dumps({"result": result}))
            else:
                print(json.dumps({"error": f"Unknown method: {method}"}))
        except Exception as e:
            print(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=str, default=None)
    args = parser.parse_args()
    run_stdio_server(args.vault)
