"""State files do not confer permission to write outside the concept directory."""
import pytest

from obsidian_llm_wiki.core.backups import list_backups
from obsidian_llm_wiki.core.orphan import orphan_page


@pytest.mark.parametrize("symlink", [False, True])
async def test_orphan_marking_rejects_external_target(tmp_path, symlink):
    concepts = tmp_path / "wiki" / "concepts"
    concepts.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("---\ntitle: Private\n---\nUntouched body.\n")
    if symlink:
        (concepts / "linked.md").symlink_to(outside)
        slug = "linked"
    else:
        slug = "../../outside"
    before = outside.read_bytes()
    assert not orphan_page(concepts, slug, "source deleted")
    assert outside.read_bytes() == before


def test_orphan_marking_keeps_recoverable_backup(tmp_path):
    concepts = tmp_path / "concepts"
    concepts.mkdir()
    page = concepts / "topic.md"
    original = "---\ntitle: Topic\n---\nHuman observations.\n"
    page.write_text(original)
    assert orphan_page(concepts, "topic", "source deleted")
    backups = list_backups(page, tmp_path / ".llmwiki" / "backups")
    assert len(backups) == 1
    assert backups[0].read_text() == original
