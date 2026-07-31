"""Public Notion page extractor via its block-tree endpoint.

Published Notion pages often render only their initial viewport through generic
article extractors.  Public pages expose a block tree through Notion's own web
endpoint; this extractor reads that public representation without cookies or a
headless browser, then renders the supported block types to Markdown.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from urllib.parse import urlparse

import httpx

from obsidian_llm_wiki.config import load_config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor
from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS, DEFAULT_TIMEOUT
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs

_PAGE_ID_RE = re.compile(
    r"(?<![0-9a-f])([0-9a-f]{8})-?([0-9a-f]{4})-?([0-9a-f]{4})-?"
    r"([0-9a-f]{4})-?([0-9a-f]{12})(?![0-9a-f])",
    re.IGNORECASE,
)
_MAX_NOTION_CHUNKS = 20


def _is_notion_url(parsed, _raw: str) -> bool:
    host = (parsed.hostname or "").lower()
    return host == "notion.so" or host.endswith(".notion.so") or host.endswith(".notion.site")


@register_extractor(_is_notion_url)
def extract_notion(raw_url: str) -> SourceDoc:
    """Extract a public Notion page's complete block tree as Markdown."""
    parsed = urlparse(raw_url)
    page_id = _page_id_from_url(raw_url)
    if not page_id:
        raise RuntimeError("Notion URL does not contain a public page ID")
    if not parsed.hostname:
        raise RuntimeError("Notion URL has no hostname")

    endpoint = f"https://{parsed.hostname}/api/v3/loadCachedPageChunk"
    max_bytes = load_config().max_html_bytes
    blocks: dict[str, dict] = {}
    cursor: dict[str, object] = {"stack": []}

    for chunk_number in range(_MAX_NOTION_CHUNKS):
        payload = {
            "page": {"id": page_id},
            "chunkNumber": chunk_number,
            "limit": 30 if chunk_number == 0 else 50,
            "cursor": cursor,
            "verticalColumns": False,
        }
        response = _post_bounded_json(endpoint, payload, max_bytes=max_bytes)
        record_map = response.get("recordMap")
        if not isinstance(record_map, dict):
            raise RuntimeError("Notion response did not contain a record map")
        chunk_blocks = record_map.get("block")
        if not isinstance(chunk_blocks, dict):
            raise RuntimeError("Notion response did not contain page blocks")
        blocks.update(
            {key: value for key, value in chunk_blocks.items() if isinstance(value, dict)}
        )

        next_cursor = response.get("cursor")
        if not isinstance(next_cursor, dict) or not next_cursor.get("stack"):
            break
        cursor = next_cursor
    else:
        raise RuntimeError(f"Notion page exceeded {_MAX_NOTION_CHUNKS} block chunks")

    root = _block_value(blocks.get(page_id))
    if root is None:
        raise RuntimeError("Notion response did not include the requested page")
    title = _plain_rich_text(root.get("properties", {}).get("title")) or "Untitled Notion page"
    content = _blocks_to_markdown(root, blocks)
    if len(content.strip()) < 50:
        raise RuntimeError("Notion page contained no substantive public content")
    return SourceDoc(title=title, content=f"# {title}\n\n{content}", url=raw_url)


def _page_id_from_url(url: str) -> str:
    """Return the canonical dashed UUID embedded in a public Notion URL."""
    match = _PAGE_ID_RE.search(url)
    if not match:
        return ""
    return "-".join(match.groups()).lower()


def _post_bounded_json(url: str, payload: dict[str, object], *, max_bytes: int) -> dict:
    """POST JSON while enforcing an I/O-bound response cap."""
    if max_bytes < 1:
        raise RuntimeError("MAX_HTML_BYTES must be at least 1")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": BROWSER_HEADERS["User-Agent"],
    }
    with (
        httpx.Client(
            **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=False),
            headers=headers,
        ) as client,
        client.stream("POST", url, json=payload) as response,
    ):
        response.raise_for_status()
        declared_size = response.headers.get("content-length")
        if declared_size is not None:
            try:
                if int(declared_size) > max_bytes:
                    raise RuntimeError(
                        f"Notion response Content-Length {declared_size} exceeds {max_bytes} bytes"
                    )
            except ValueError:
                pass
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes(chunk_size=min(65_536, max_bytes + 1)):
            total += len(chunk)
            if total > max_bytes:
                raise RuntimeError(f"Notion response exceeded {max_bytes} bytes")
            chunks.append(chunk)
    try:
        data = json.loads(b"".join(chunks))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Notion response was not valid JSON") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Notion response root was not an object")
    return data


def _block_value(record: object) -> dict | None:
    if not isinstance(record, dict):
        return None
    value = record.get("value")
    if not isinstance(value, dict):
        return None
    nested = value.get("value")
    return nested if isinstance(nested, dict) else None


def _rich_text(value: object) -> str:
    """Render Notion's text-fragment property format, retaining links/styles."""
    if not isinstance(value, list):
        return ""
    rendered: list[str] = []
    for fragment in value:
        if not isinstance(fragment, list) or not fragment or not isinstance(fragment[0], str):
            continue
        text = fragment[0]
        annotations = fragment[1] if len(fragment) > 1 and isinstance(fragment[1], list) else []
        link = ""
        styles: set[str] = set()
        for annotation in annotations:
            if not isinstance(annotation, list) or not annotation:
                continue
            kind = annotation[0]
            if kind == "a" and len(annotation) > 1 and isinstance(annotation[1], str):
                link = annotation[1]
            elif isinstance(kind, str):
                styles.add(kind)
        if "c" in styles:
            text = f"`{text}`"
        if "b" in styles:
            text = f"**{text}**"
        if "i" in styles:
            text = f"*{text}*"
        if "s" in styles:
            text = f"~~{text}~~"
        if link:
            text = f"[{text}]({link})"
        rendered.append(text)
    return "".join(rendered).strip()


def _plain_rich_text(value: object) -> str:
    """Read a Notion title without carrying inline display annotations into metadata."""
    if not isinstance(value, list):
        return ""
    return "".join(
        fragment[0]
        for fragment in value
        if isinstance(fragment, list) and fragment and isinstance(fragment[0], str)
    ).strip()


def _blocks_to_markdown(root: dict, blocks: dict[str, dict]) -> str:
    """Render the root's public blocks in stored page order."""
    lines: list[str] = []
    numbered_index = 0
    for block_id in _descendant_ids(root.get("content"), blocks):
        block = _block_value(blocks.get(block_id))
        if block is None or block.get("alive") is False:
            continue
        block_type = block.get("type")
        properties = block.get("properties")
        if not isinstance(properties, dict):
            properties = {}
        text = _rich_text(properties.get("title"))
        if block_type == "header" and text:
            lines.extend((f"## {text}", ""))
        elif block_type == "sub_header" and text:
            lines.extend((f"### {text}", ""))
        elif block_type == "sub_sub_header" and text:
            lines.extend((f"#### {text}", ""))
        elif block_type == "bulleted_list" and text:
            lines.append(f"- {text}")
        elif block_type == "numbered_list" and text:
            numbered_index += 1
            lines.append(f"{numbered_index}. {text}")
        elif block_type == "to_do" and text:
            checked = block.get("properties", {}).get("checked") == [["Yes"]]
            lines.append(f"- [{'x' if checked else ' '}] {text}")
        elif block_type == "quote" and text:
            lines.extend((f"> {text}", ""))
        elif block_type == "code" and text:
            language = block.get("properties", {}).get("language", [[""]])
            language_name = _rich_text(language)
            lines.extend((f"```{language_name}", text, "```", ""))
        elif block_type == "divider":
            lines.extend(("---", ""))
        elif text:
            lines.extend((text, ""))

    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _descendant_ids(child_ids: object, blocks: dict[str, dict]) -> Iterable[str]:
    """Yield descendants depth-first without revisiting malformed block graphs."""
    pending = list(child_ids) if isinstance(child_ids, list) else []
    seen: set[str] = set()
    while pending:
        block_id = pending.pop(0)
        if not isinstance(block_id, str) or block_id in seen:
            continue
        seen.add(block_id)
        yield block_id
        block = _block_value(blocks.get(block_id))
        if block is not None:
            children = block.get("content")
            if isinstance(children, list):
                pending[0:0] = children
