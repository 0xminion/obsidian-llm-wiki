"""Generation navigation must not rewrite literal source/code evidence."""
import pytest

from obsidian_llm_wiki.core.immutable_publication import _qualify_links


@pytest.mark.parametrize("literal", [
    "```text\n[[target]]\n```", "Use the literal `[[target]]` token.",
])
def test_generation_qualification_preserves_literal_wikilink_text(tmp_path, literal):
    (tmp_path / "concepts").mkdir()
    (tmp_path / "concepts/target.md").write_text("# Target\n")
    source = tmp_path / "sources" / "original.md"
    source.parent.mkdir()
    before = ("# Captured source\n\n" + literal + "\n").encode()
    source.write_bytes(before)
    _qualify_links(tmp_path, "synthetic-generation")
    assert source.read_bytes() == before
