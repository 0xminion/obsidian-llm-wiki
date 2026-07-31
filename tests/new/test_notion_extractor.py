"""Unit contracts for public Notion page extraction."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx

from obsidian_llm_wiki.ingest.extractors import notion


def _record(value: dict) -> dict:
    return {"value": {"value": value}}


def _properties(text: str) -> dict:
    return {"title": [[text]]}


def test_page_id_from_public_notion_url_normalizes_dashes():
    assert notion._page_id_from_url(
        "https://example.notion.site/A-page-2ba458a33fea801c9221f1435376f477"
    ) == "2ba458a3-3fea-801c-9221-f1435376f477"


def test_blocks_to_markdown_preserves_order_headings_lists_and_links():
    root = {"content": ["heading", "paragraph", "list"]}
    blocks = {
        "heading": _record({"type": "header", "properties": _properties("Overview")}),
        "paragraph": _record(
            {
                "type": "text",
                "properties": {
                    "title": [["Read "], ["the paper", [["a", "https://example.com/paper"]]]]
                },
            }
        ),
        "list": _record({"type": "bulleted_list", "properties": _properties("A key point")}),
    }

    markdown = notion._blocks_to_markdown(root, blocks)

    assert markdown == "## Overview\n\nRead [the paper](https://example.com/paper)\n\n- A key point"


def test_extract_notion_uses_public_block_endpoint_and_renders_full_body(monkeypatch):
    page_id = "2ba458a3-3fea-801c-9221-f1435376f477"
    payload = {
        "cursor": {"stack": []},
        "recordMap": {
            "block": {
                page_id: _record(
                    {
                        "type": "page",
                        "properties": _properties("Public Page"),
                        "content": ["body", "heading", "list"],
                    }
                ),
                "body": _record(
                    {"type": "text", "properties": _properties("A complete public page body.")}
                ),
                "heading": _record(
                    {"type": "header", "properties": _properties("Details")}
                ),
                "list": _record(
                    {"type": "bulleted_list", "properties": _properties("A complete list item")}
                ),
            }
        },
    }
    requested: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requested["url"] = str(request.url)
        requested["payload"] = json.loads(request.content)
        return httpx.Response(200, json=payload, request=request)

    real_client = httpx.Client
    monkeypatch.setattr(
        notion.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    monkeypatch.setattr(notion, "load_config", lambda: SimpleNamespace(max_html_bytes=100_000))

    source = notion.extract_notion(
        "https://example.notion.site/Public-Page-2ba458a33fea801c9221f1435376f477"
    )

    assert requested["url"] == "https://example.notion.site/api/v3/loadCachedPageChunk"
    assert requested["payload"] == {
        "page": {"id": page_id},
        "chunkNumber": 0,
        "limit": 30,
        "cursor": {"stack": []},
        "verticalColumns": False,
    }
    assert source.title == "Public Page"
    assert source.content == (
        "# Public Page\n\nA complete public page body.\n\n"
        "## Details\n\n- A complete list item"
    )
