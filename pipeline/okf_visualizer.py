"""OKF bundle visualizer — self-contained Cytoscape.js HTML graph.

Generates a single ``viz.html`` file inside the OKF bundle that renders an
interactive concept graph using Cytoscape.js (loaded from CDN). The HTML is
fully self-contained except for the CDN script tag, so it can be opened
directly in any modern browser.

The graph is built from the OKF concept markdown files (frontmatter +
standard markdown links). Internal links are resolved to concept IDs and
rendered as graph edges; backlinks are computed so the detail panel can
show "cited by" information.

Public entry point: :func:`generate_visualization`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pipeline.okf_markdown import (
    extract_links,
    parse_frontmatter,
    safe_read_file,
)

__all__ = ["generate_visualization"]


# Files to skip when scanning for concepts.
_SKIP_FILES = {"index.md", "log.md", "viz.html"}

# Cytoscape.js CDN URL (pinned version).
_CYTOSCAPE_CDN = (
    "https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.4/cytoscape.min.js"
)


# ── Public entry point ────────────────────────────────────────────────────


def generate_visualization(
    bundle_dir: str | Path,
    output_path: str | Path | None = None,
    name: str | None = None,
) -> Path:
    """Generate a self-contained HTML visualization for an OKF bundle.

    Args:
        bundle_dir: Path to the OKF bundle root (the directory containing
            concept ``.md`` files, ``index.md``, ``log.md``, etc.).
        output_path: Where to write the HTML file. Defaults to
            ``<bundle_dir>/viz.html``.
        name: Display name for the wiki (shown in the HTML title / header).
            Defaults to the bundle directory name.

    Returns:
        The :class:`~pathlib.Path` to the generated HTML file.
    """
    bd = Path(bundle_dir)
    if not bd.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {bd}")

    out = bd / "viz.html" if output_path is None else Path(output_path)

    if name is None:
        name = bd.name

    concepts = _collect_concepts(bd)
    nodes, edges, backlinks = _build_graph_data(concepts, bd)
    html = _render_html(name, nodes, edges, backlinks, concepts)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


# ── Concept collection ────────────────────────────────────────────────────


def _collect_concepts(bundle_dir: Path) -> list[dict]:
    """Scan all ``.md`` files in ``bundle_dir`` (recursively).

    Skips ``index.md``, ``log.md``, and ``viz.html``.

    Returns a list of dicts with keys: ``id``, ``file``, ``title``, ``type``,
    ``description``, ``tags``, ``timestamp``, ``resource``, ``body``,
    ``links``.
    """
    concepts: list[dict] = []
    for md_path in sorted(bundle_dir.rglob("*.md")):
        # Skip by basename.
        if md_path.name.lower() in _SKIP_FILES:
            continue
        # Skip if the file is the viz.html (already covered) or hidden.
        if md_path.name.startswith("."):
            continue

        raw = safe_read_file(md_path)
        if not raw.strip():
            continue

        meta, body = parse_frontmatter(raw)

        # Concept id is the path relative to bundle_dir without .md suffix.
        rel = md_path.relative_to(bundle_dir)
        concept_id = str(rel.with_suffix(""))
        # Normalise Windows-style separators to forward slashes.
        concept_id = concept_id.replace("\\", "/")

        # Extract links from the body.
        links = extract_links(body)

        concepts.append(
            {
                "id": concept_id,
                "file": str(rel).replace("\\", "/"),
                "title": meta.get("title") or concept_id,
                "type": meta.get("type", "Concept"),
                "description": meta.get("description", ""),
                "tags": meta.get("tags", []) or [],
                "timestamp": meta.get("timestamp", ""),
                "resource": meta.get("resource", ""),
                "body": body,
                "links": links,
            }
        )
    return concepts


# ── Graph construction ────────────────────────────────────────────────────


def _resolve_target(target: str, source_id: str, bundle_dir: Path) -> str:
    """Resolve a markdown link target to a concept ID.

    * Absolute internal links (``/foo/bar.md``) → ``foo/bar``
    * Relative internal links (``bar.md``, ``./bar.md``, ``../bar.md``) →
      resolved relative to ``source_id``'s directory then stripped of
      ``.md``.
    * External links (``http://``, ``https://``, ``mailto:``, etc.) → ``""``
      (empty string, meaning "no edge").

    Returns ``""`` for anything that cannot be resolved.
    """
    if not target:
        return ""

    # External links — return empty.
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", target):
        return ""
    if target.startswith(("mailto:", "tel:", "#")):
        return ""

    stripped = target

    # Absolute internal link: starts with "/"
    if stripped.startswith("/"):
        stripped = stripped.lstrip("/")
        if stripped.endswith(".md"):
            stripped = stripped[:-3]
        return stripped.replace("\\", "/")

    # Relative link — resolve against source directory using pure path
    # arithmetic (no filesystem .resolve() so we don't pick up the real
    # filesystem prefix from tmp_path etc.).
    source_dir = Path(source_id).parent
    # Join and normalise without touching the filesystem.
    joined = source_dir / stripped
    # Normalise: remove leading "./" segments and collapse "..".
    parts: list[str] = []
    for part in joined.parts:
        if part == ".":
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    normalised = "/".join(parts)
    # Strip .md suffix.
    if normalised.endswith(".md"):
        normalised = normalised[:-3]
    normalised = normalised.replace("\\", "/")

    # Strip leading "./" if any.
    if normalised.startswith("./"):
        normalised = normalised[2:]

    return normalised


def _build_graph_data(
    concepts: list[dict],
    bundle_dir: Path,
) -> tuple[list[dict], list[dict], dict[str, list[str]]]:
    """Build Cytoscape.js elements (nodes, edges) plus a backlinks map.

    Nodes have keys: ``id``, ``label``, ``type``, ``description``, ``tags``,
    ``timestamp``, ``resource``.

    Edges have keys: ``source``, ``target``.

    Backlinks map ``concept_id -> list[concept_id]`` (the concepts that
    link *to* the key).
    """
    concept_ids = {c["id"] for c in concepts}

    nodes: list[dict] = []
    for c in concepts:
        nodes.append(
            {
                "id": c["id"],
                "label": c["title"],
                "type": c["type"],
                "description": c["description"],
                "tags": c["tags"],
                "timestamp": c["timestamp"],
                "resource": c["resource"],
            }
        )

    edges: list[dict] = []
    backlinks: dict[str, list[str]] = {c["id"]: [] for c in concepts}

    for c in concepts:
        source_id = c["id"]
        seen_targets: set[str] = set()
        for _text, url in c["links"]:
            target_id = _resolve_target(url, source_id, bundle_dir)
            if not target_id or target_id not in concept_ids:
                continue
            if target_id == source_id:
                continue
            if target_id in seen_targets:
                continue
            seen_targets.add(target_id)
            edges.append({"source": source_id, "target": target_id})
            backlinks.setdefault(target_id, []).append(source_id)

    return nodes, edges, backlinks


# ── HTML rendering ─────────────────────────────────────────────────────────


def _render_html(
    name: str,
    nodes: list[dict],
    edges: list[dict],
    backlinks: dict[str, list[str]],
    concepts: list[dict],
) -> str:
    """Render the full self-contained HTML document.

    The template uses :meth:`str.format` to embed JSON data, so every
    literal ``{`` / ``}`` in the CSS/JS body is doubled to ``{{`` / ``}}``.
    """
    # Build the concept lookup map (id -> full concept dict) for the
    # detail panel.
    concept_map = {c["id"]: c for c in concepts}

    # Serialise data as JSON. We use json.dumps with ensure_ascii=False so
    # unicode titles survive. The JSON is embedded in a <script> block.
    # Escape '<' and '>' to prevent XSS via '</script>' injection — replace
    # them with their JSON unicode escape sequences so the browser's HTML
    # parser never sees a literal '</script>' inside the script block.
    def _safe_json(obj):
        return json.dumps(obj, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")

    nodes_json = _safe_json(nodes)
    edges_json = _safe_json(edges)
    backlinks_json = _safe_json(backlinks)
    concepts_json = _safe_json(concept_map)

    return _HTML_TEMPLATE.format(
        name=name,
        cytoscape_cdn=_CYTOSCAPE_CDN,
        nodes_json=nodes_json,
        edges_json=edges_json,
        backlinks_json=backlinks_json,
        concepts_json=concepts_json,
    )


# ── HTML template (uses str.format — all literal braces doubled) ──────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{name} — Knowledge Graph</title>
<style>
  * {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    height: 100vh;
    display: flex;
    overflow: hidden;
  }}
  /* ── Sidebar ───────────────────────────────────── */
  #sidebar {{
    width: 260px;
    background: #16213e;
    padding: 16px;
    overflow-y: auto;
    border-right: 1px solid #0f3460;
    flex-shrink: 0;
  }}
  #sidebar h1 {{
    font-size: 18px;
    margin-bottom: 16px;
    color: #00adb5;
  }}
  .control-group {{
    margin-bottom: 14px;
  }}
  .control-group label {{
    display: block;
    font-size: 12px;
    color: #a0a0a0;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  #search {{
    width: 100%;
    padding: 8px 10px;
    background: #0f3460;
    border: 1px solid #1a1a2e;
    border-radius: 4px;
    color: #e0e0e0;
    font-size: 13px;
  }}
  #search:focus {{
    outline: none;
    border-color: #00adb5;
  }}
  #type-filter, #layout-select {{
    width: 100%;
    padding: 8px 10px;
    background: #0f3460;
    border: 1px solid #1a1a2e;
    border-radius: 4px;
    color: #e0e0e0;
    font-size: 13px;
  }}
  /* ── Graph area ────────────────────────────────── */
  #cy {{
    flex: 1;
    background: #1a1a2e;
  }}
  /* ── Detail panel ───────────────────────────────── */
  #detail {{
    width: 400px;
    background: #16213e;
    padding: 20px;
    overflow-y: auto;
    border-left: 1px solid #0f3460;
    flex-shrink: 0;
  }}
  #detail h2 {{
    color: #00adb5;
    font-size: 18px;
    margin-bottom: 8px;
  }}
  #detail .meta {{
    font-size: 12px;
    color: #a0a0a0;
    margin-bottom: 12px;
  }}
  #detail .meta .badge {{
    display: inline-block;
    padding: 2px 8px;
    background: #0f3460;
    border-radius: 10px;
    margin-right: 4px;
    margin-bottom: 2px;
    font-size: 11px;
  }}
  #detail .body {{
    font-size: 14px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-wrap: break-word;
  }}
  #detail .body a {{
    color: #00adb5;
    cursor: pointer;
  }}
  #detail .backlinks {{
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid #0f3460;
  }}
  #detail .backlinks h3 {{
    font-size: 13px;
    color: #a0a0a0;
    text-transform: uppercase;
    margin-bottom: 8px;
  }}
  #detail .backlinks a {{
    display: block;
    color: #00adb5;
    cursor: pointer;
    padding: 3px 0;
    font-size: 13px;
  }}
  #detail .placeholder {{
    color: #707070;
    font-style: italic;
    text-align: center;
    margin-top: 40px;
  }}
</style>
</head>
<body>
  <div id="sidebar">
    <h1>{name}</h1>
    <div class="control-group">
      <label for="search">Search</label>
      <input type="text" id="search" placeholder="Filter nodes…" />
    </div>
    <div class="control-group">
      <label for="type-filter">Type</label>
      <select id="type-filter">
        <option value="">All types</option>
      </select>
    </div>
    <div class="control-group">
      <label for="layout-select">Layout</label>
      <select id="layout-select">
        <option value="cose">CoSE (force-directed)</option>
        <option value="concentric">Concentric</option>
        <option value="breadthfirst">Breadth-first</option>
        <option value="circle">Circle</option>
        <option value="grid">Grid</option>
      </select>
    </div>
  </div>

  <div id="cy"></div>

  <div id="detail">
    <p class="placeholder">Click a node to view details.</p>
  </div>

<script src="{cytoscape_cdn}"></script>
<script>
  (function() {{
    var NODES = {nodes_json};
    var EDGES = {edges_json};
    var BACKLINKS = {backlinks_json};
    var CONCEPTS = {concepts_json};

    // ── Type colours ──────────────────────────────────
    var TYPE_COLORS = {{
      "Concept":       "#00adb5",
      "Entry":         "#e84393",
      "Source":        "#fdcb6e",
      "Reference":     "#6c5ce7",
      "Map of Content": "#00b894",
      "MOC":           "#00b894"
    }};
    function colorFor(type) {{
      return TYPE_COLORS[type] || "#70a0d0";
    }}

    // ── Build Cytoscape elements ───────────────────────
    var elements = [];
    NODES.forEach(function(n) {{
      elements.push({{
        data: {{
          id: n.id,
          label: n.label,
          type: n.type,
          description: n.description,
          tags: n.tags,
          timestamp: n.timestamp,
          resource: n.resource
        }}
      }});
    }});
    EDGES.forEach(function(e) {{
      elements.push({{
        data: {{ source: e.source, target: e.target }}
      }});
    }});

    // ── Init Cytoscape ─────────────────────────────────
    var cy = cytoscape({{
      container: document.getElementById("cy"),
      elements: elements,
      style: [
        {{
          selector: "node",
          style: {{
            "label": "data(label)",
            "text-valign": "center",
            "text-halign": "center",
            "text-wrap": "wrap",
            "text-max-width": "80px",
            "font-size": "11px",
            "color": "#e0e0e0",
            "background-color": function(ele) {{ return colorFor(ele.data("type")); }},
            "width": "40px",
            "height": "40px",
            "border-width": 1,
            "border-color": "#0f3460"
          }}
        }},
        {{
          selector: "edge",
          style: {{
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "arrow-color": "#0f3460",
            "line-color": "#0f3460",
            "width": 1.5,
            "opacity": 0.6
          }}
        }},
        {{
          selector: "node:selected",
          style: {{
            "border-width": 3,
            "border-color": "#00adb5"
          }}
        }}
      ],
      layout: {{ name: "cose", animate: true, padding: 30 }}
    }});

    // ── Populate type filter ──────────────────────────
    var typeFilter = document.getElementById("type-filter");
    var types = {{}};
    NODES.forEach(function(n) {{ types[n.type] = true; }});
    Object.keys(types).sort().forEach(function(t) {{
      var opt = document.createElement("option");
      opt.value = t;
      opt.textContent = t;
      typeFilter.appendChild(opt);
    }});

    // ── Detail panel rendering ─────────────────────────
    var detail = document.getElementById("detail");

    function escapeHtml(text) {{
      if (!text) return "";
      var div = document.createElement("div");
      div.textContent = text;
      return div.innerHTML;
    }}

    function showDetail(nodeId) {{
      var c = CONCEPTS[nodeId];
      if (!c) {{
        detail.innerHTML = '<p class="placeholder">No data for this node.</p>';
        return;
      }}

      var html = "<h2>" + escapeHtml(c.title) + "</h2>";
      html += '<div class="meta">';
      html += '<span class="badge">' + escapeHtml(c.type) + '</span>';
      if (c.timestamp) {{
        html += '<span class="badge">' + escapeHtml(c.timestamp) + '</span>';
      }}
      if (c.description) {{
        html += '<div style="margin-top:8px">' + escapeHtml(c.description) + '</div>';
      }}
      if (c.tags && c.tags.length) {{
        html += '<div style="margin-top:6px">';
        c.tags.forEach(function(tag) {{
          html += '<span class="badge">' + escapeHtml(tag) + '</span>';
        }});
        html += '</div>';
      }}
      if (c.resource) {{
        html += '<div style="margin-top:6px;font-size:11px">Resource: ' +
                escapeHtml(c.resource) + '</div>';
      }}
      html += '</div>';  // close .meta

      // Body — convert internal markdown links to clickable spans.
      var bodyHtml = escapeHtml(c.body);
      // Highlight [text](/path.md) style links as clickable.
      bodyHtml = bodyHtml.replace(
        /\[([^\]]*)\]\(([^)]*)\)/g,
        function(match, text, url) {{
          return '<a data-link="' + encodeURIComponent(url) + '">' +
                 text + '</a>';
        }}
      );
      html += '<div class="body">' + bodyHtml + '</div>';

      // Backlinks
      var bl = BACKLINKS[nodeId] || [];
      if (bl.length) {{
        html += '<div class="backlinks"><h3>Cited by (' + bl.length + ')</h3>';
        bl.forEach(function(sourceId) {{
          var src = CONCEPTS[sourceId];
          var label = src ? src.title : sourceId;
          html += '<a data-link="' + encodeURIComponent(sourceId) + '">' +
                  escapeHtml(label) + '</a>';
        }});
        html += '</div>';
      }}

      detail.innerHTML = html;

      // Wire up internal links.
      detail.querySelectorAll("a[data-link]").forEach(function(a) {{
        a.addEventListener("click", function(e) {{
          e.preventDefault();
          var raw = decodeURIComponent(a.getAttribute("data-link"));
          // Internal links: /foo.md -> foo,  foo.md -> foo
          var targetId = raw;
          if (targetId.charAt(0) === "/") targetId = targetId.substring(1);
          if (targetId.endsWith(".md")) targetId = targetId.substring(0, targetId.length - 3);
          if (CONCEPTS[targetId]) {{
            cy.getElementById(targetId).select();
            cy.animate({{
              fit: {{ eles: cy.getElementById(targetId), padding: 100 }}
            }}, {{ duration: 300 }});
            showDetail(targetId);
          }}
        }});
      }});
    }}

    // ── Node click handler ────────────────────────────
    cy.on("tap", "node", function(evt) {{
      var nodeId = evt.target.id();
      showDetail(nodeId);
    }});

    // ── Search filter ─────────────────────────────────
    var search = document.getElementById("search");
    search.addEventListener("input", function() {{
      var q = search.value.toLowerCase().trim();
      cy.nodes().forEach(function(n) {{
        var label = (n.data("label") || "").toLowerCase();
        var id = (n.id() || "").toLowerCase();
        if (!q || label.indexOf(q) !== -1 || id.indexOf(q) !== -1) {{
          n.style("display", "element");
          n.style("opacity", 1);
        }} else {{
          n.style("opacity", 0.15);
        }}
      }});
    }});

    // ── Type filter ───────────────────────────────────
    typeFilter.addEventListener("change", function() {{
      var selected = typeFilter.value;
      cy.nodes().forEach(function(n) {{
        if (!selected || n.data("type") === selected) {{
          n.style("display", "element");
        }} else {{
          n.style("display", "none");
        }}
      }});
    }});

    // ── Layout switcher ───────────────────────────────
    document.getElementById("layout-select").addEventListener("change", function() {{
      var layoutName = this.value;
      var layoutOpts = {{ name: layoutName, animate: true, padding: 30 }};
      if (layoutName === "cose") {{
        layoutOpts.animated = true;
        layoutOpts.nodeRepulsion = function() {{ return 8000; }};
        layoutOpts.idealEdgeLength = function() {{ return 50; }};
      }}
      cy.layout(layoutOpts).run();
    }});
  }})();
</script>
</body>
</html>
"""
