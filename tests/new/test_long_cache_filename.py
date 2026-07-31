"""Cache persistence remains safe for filesystem-length source filenames."""

from __future__ import annotations

from pathlib import Path

from obsidian_llm_wiki.core.cache import (
    load_all_cached_syntheses,
    save_synthesis,
    synthesis_cache_path,
)
from obsidian_llm_wiki.core.models import SourceSynthesis


def test_long_source_filename_uses_digest_cache_key_and_round_trips(tmp_path: Path):
    source_file = ("中" * 79) + ".md"  # 240 UTF-8 bytes: legal source, illegal cache+temp name
    synthesis = SourceSynthesis(source_title="Long source", source_summary="Body")

    path = synthesis_cache_path(tmp_path, source_file)
    save_synthesis(synthesis, tmp_path, source_file)

    assert path.exists()
    assert path.name.startswith("sha256-")
    assert len(path.name.encode("utf-8")) < 255
    assert load_all_cached_syntheses(tmp_path)[source_file].source_title == "Long source"
