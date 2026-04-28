"""Semantic compile operations — cross-linking, concept merging, MoC rebuilding."""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.config import Config
from pipeline.utils import frontmatter_list_items as _frontmatter_list_items

log = logging.getLogger(__name__)


@dataclass
class NoteIndex:
    """In-memory index of vault notes for semantic operations."""
    notes: dict[str, dict] = field(default_factory=dict)
    embeddings: dict[str, list[float]] = field(default_factory=dict)

    def load(self, cfg: Config) -> None:
        for note_dir, note_type in [
            (cfg.entries_dir, "entry"),
            (cfg.concepts_dir, "concept"),
            (cfg.mocs_dir, "moc"),
        ]:
            if not note_dir.exists():
                continue
            for md in note_dir.glob("*.md"):
                try:
                    content = md.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                name = md.stem
                fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
                tags: set[str] = set()
                title = name
                if fm_match:
                    fm = fm_match.group(1)
                    t_match = re.search(r"title:\s*[\"']?(.*?)[\"']?\s*$", fm, re.MULTILINE)
                    if t_match:
                        title = t_match.group(1).strip()
                    tags = {tag.lower() for tag in _frontmatter_list_items(fm, "tags") if tag}
                links = set(re.findall(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]", content))
                preview = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)[:500]
                self.notes[name] = {
                    "type": note_type,
                    "title": title,
                    "tags": tags,
                    "links": links,
                    "preview": preview,
                    "path": str(md.relative_to(cfg.vault_path)),
                }

    def embed_all(self, client, skip_local: bool = False) -> None:
        """Embed all notes. If *skip_local* is True the local batch is skipped (QMD handles it)."""
        if not self.notes or skip_local:
            return
        from pipeline.qmd import _get_client
        if _get_client() is not None:
            log.info("QMD enabled — skipping local embed batch; relying on QMD semantic search")
            return
        texts = [f"{n['title']}\n{n['preview']}" for n in self.notes.values()]
        names = list(self.notes.keys())
        batch = client.embed_batch(texts)
        if batch:
            # Lookup by index to avoid dict-key collision on duplicate content.
            for i, name in enumerate(names):
                text = texts[i]
                if text in batch:
                    self.embeddings[name] = batch[text]
            log.info("Embedded %d/%d notes", len(self.embeddings), len(self.notes))
        else:
            log.warning("Embedding batch failed; semantic operations will use heuristics only")

    def similarity(self, name_a: str, name_b: str) -> float:
        emb_a = self.embeddings.get(name_a)
        emb_b = self.embeddings.get(name_b)
        if not emb_a or not emb_b:
            return 0.0
        dot = sum(x * y for x, y in zip(emb_a, emb_b))
        norm_a = math.sqrt(sum(x * x for x in emb_a))
        norm_b = math.sqrt(sum(x * x for x in emb_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


def _add_wikilink(cfg: Config, source_name: str, target_name: str, reason: str) -> bool:
    source_dirs = [cfg.entries_dir, cfg.concepts_dir, cfg.mocs_dir]
    source_path = None
    for d in source_dirs:
        candidate = d / f"{source_name}.md"
        if candidate.exists():
            source_path = candidate
            break
    if not source_path:
        return False

    try:
        content = source_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False

    if f"[[{target_name}]]" in content:
        return False

    sections = {
        "entry": ["Linked concepts", "Links", "关联概念"],
        "concept": ["Links", "Context", "链接"],
        "moc": ["Related MoCs", "Cross-References", "关联图谱"],
    }
    note_type = "entry"
    for d, nt in [(cfg.entries_dir, "entry"), (cfg.concepts_dir, "concept"), (cfg.mocs_dir, "moc")]:
        if (d / f"{source_name}.md").exists():
            note_type = nt
            break

    target_sections = sections.get(note_type, ["Links"])
    lines = content.splitlines()
    insert_idx = len(lines)
    for i, line in enumerate(lines):
        for sec in target_sections:
            if line.strip().startswith(f"## {sec}"):
                insert_idx = i + 1
                while insert_idx < len(lines) and lines[insert_idx].strip() == "":
                    insert_idx += 1
                break
        if insert_idx < len(lines):
            break

    link_line = f"- [[{target_name}]]"
    if reason:
        link_line += f" — {reason}"
    lines.insert(insert_idx, link_line)
    from pipeline.utils import _atomic_write
    _atomic_write(source_path, "\n".join(lines))
    log.debug("Added link: %s -> %s", source_name, target_name)
    return True


def _semantic_crosslink(cfg: Config, client, index: NoteIndex) -> int:
    """Find unlinked note pairs via embedding cosine similarity + shared-tag boost,
    rank the top 30 candidates, then ask the LLM which pairs deserve wikilinks.
    """
    if len(index.notes) < 2:
        return 0

    candidates: list[tuple[str, str, float, set[str]]] = []
    names = list(index.notes.keys())
    for i, name_a in enumerate(names):
        info_a = index.notes[name_a]
        for name_b in names[i + 1:]:
            info_b = index.notes[name_b]
            if name_b in info_a["links"] or name_a in info_b["links"]:
                continue
            sim = index.similarity(name_a, name_b)
            shared_tags = info_a["tags"] & info_b["tags"]
            score = sim + (len(shared_tags) * 0.1)
            if score > 0.5 or len(shared_tags) >= 2:
                candidates.append((name_a, name_b, score, shared_tags))

    if not candidates:
        return 0

    candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = candidates[:30]

    prompt_lines = [
        "You are a knowledge base editor. Review these candidate note pairs and decide which should link to each other.",
        "For each pair, respond with exactly one of these formats (use pipe separators | ):",
        "  LINK <note_a> | <note_b> | <brief reason>",
        "  SKIP <note_a> | <note_b>",
        "",
        "Candidates:",
    ]
    for a, b, score, tags in candidates:
        prompt_lines.append(f"\n--- {a} ↔ {b} (score: {score:.2f}) ---")
        prompt_lines.append(f"{a}: {index.notes[a]['title']} — {index.notes[a]['preview'][:200]}")
        prompt_lines.append(f"{b}: {index.notes[b]['title']} — {index.notes[b]['preview'][:200]}")
        if tags:
            prompt_lines.append(f"shared tags: {', '.join(tags)}")

    prompt = "\n".join(prompt_lines)
    response = client.generate(prompt, timeout=120)

    links_added = 0
    for line in response.splitlines():
        m = re.match(r"LINK\s+(.+?)\s*\|\s*(.+?)\s*\|\s*(.*)", line)
        if m:
            a, b, reason = m.groups()
            a = a.strip().strip('"').strip("'")
            b = b.strip().strip('"').strip("'")
            if _add_wikilink(cfg, a, b, reason.strip()):
                links_added += 1
            if index.notes.get(b, {}).get("type") in ("concept", "moc"):
                if _add_wikilink(cfg, b, a, reason.strip()):
                    links_added += 1

    log.info("Semantic cross-linking: %d links added", links_added)
    return links_added


def _semantic_concept_merge(cfg: Config, client, index: NoteIndex) -> int:
    concepts = {n: info for n, info in index.notes.items() if info["type"] == "concept"}
    if len(concepts) < 2:
        return 0

    candidates: list[tuple[str, str, float]] = []
    names = list(concepts.keys())
    # Pre-filter with embedding threshold to reduce Python O(N^2) work
    for i, name_a in enumerate(names):
        emb_a = index.embeddings.get(name_a)
        for name_b in names[i + 1:]:
            emb_b = index.embeddings.get(name_b)
            sim = 0.0
            if emb_a and emb_b:
                dot = sum(x * y for x, y in zip(emb_a, emb_b))
                norm_a = math.sqrt(sum(x * x for x in emb_a))
                norm_b = math.sqrt(sum(x * x for x in emb_b))
                if norm_a and norm_b:
                    sim = dot / (norm_a * norm_b)
            # Skip if embeddings are present but say "not similar" (<0.5).
            # If no embeddings, fall through to string-overlap heuristic.
            if (emb_a and emb_b) and sim < 0.5:
                continue
            info_a = concepts[name_a]
            info_b = concepts[name_b]
            words_a = set(re.sub(r"[^a-zA-Z0-9一-鿿]", " ", info_a["title"].lower()).split())
            words_b = set(re.sub(r"[^a-zA-Z0-9一-鿿]", " ", info_b["title"].lower()).split())
            overlap = 0.0
            if words_a and words_b:
                overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
            score = max(overlap, sim)
            if score > 0.75:
                candidates.append((name_a, name_b, score))

    if not candidates:
        return 0

    candidates.sort(key=lambda x: x[2], reverse=True)
    max_candidates = getattr(cfg, 'max_merge_candidates', 10)
    candidates = candidates[:max_candidates]

    prompt_lines = [
        "You are a knowledge base editor. Review these concept pairs and decide if they should be merged.",
        "For each pair, respond with exactly one of (use pipe separators | ):",
        "  MERGE <canonical_name> | <duplicate_name> | <reason>",
        "  KEEP_BOTH <name_a> | <name_b> | <reason>",
        "",
        "Rules:",
        "- If two concepts cover the SAME idea (even in different languages), merge them.",
        "- Choose the older/canonical concept as the first name.",
        "- If they overlap only partially, keep both.",
        "",
        "Candidates:",
    ]
    for a, b, score in candidates:
        prompt_lines.append(f"\n--- {a} ↔ {b} (similarity: {score:.2f}) ---")
        prompt_lines.append(f"{a}: {concepts[a]['title']}")
        prompt_lines.append(f"  {concepts[a]['preview'][:250]}")
        prompt_lines.append(f"{b}: {concepts[b]['title']}")
        prompt_lines.append(f"  {concepts[b]['preview'][:250]}")

    prompt = "\n".join(prompt_lines)
    response = client.generate(prompt, timeout=120)

    merged = 0
    for line in response.splitlines():
        m = re.match(r"MERGE\s+(.+?)\s*\|\s*(.+?)\s*\|\s*(.*)", line)
        if m:
            canonical, duplicate, reason = m.groups()
            canonical = canonical.strip()
            duplicate = duplicate.strip()
            if _merge_concepts(cfg, canonical, duplicate, index):
                merged += 1

    log.info("Semantic concept merge: %d concepts merged", merged)
    return merged


def _merge_concepts(cfg: Config, canonical_name: str, duplicate_name: str, index: NoteIndex) -> bool:
    from pipeline.compile.core import _archive_duplicate
    canonical_path = cfg.concepts_dir / f"{canonical_name}.md"
    duplicate_path = cfg.concepts_dir / f"{duplicate_name}.md"
    if not canonical_path.exists() or not duplicate_path.exists():
        return False

    try:
        canonical_content = canonical_path.read_text(encoding="utf-8", errors="replace")
        duplicate_content = duplicate_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False

    duplicate_body = re.sub(r"^---\s*\n.*?\n---\s*\n", "", duplicate_content, flags=re.DOTALL)
    merged_content = canonical_content.rstrip() + f"\n\n## Merged from {duplicate_name}\n\n{duplicate_body.strip()}\n"
    from pipeline.utils import _atomic_write
    _atomic_write(canonical_path, merged_content)

    _archive_duplicate(duplicate_path, cfg)

    all_dirs = [cfg.entries_dir, cfg.concepts_dir, cfg.mocs_dir, cfg.sources_dir]
    for directory in all_dirs:
        if not directory.exists():
            continue
        for note_md in directory.glob("*.md"):
            try:
                text = note_md.read_text(encoding="utf-8", errors="replace")
                original = text
                text = re.sub(
                    rf"\[\[{re.escape(duplicate_name)}(?P<suffix>[|#][^\]]*)?\]\]",
                    lambda m: f"[[{canonical_name}{m.group('suffix') or ''}]]",
                    text,
                )
                if text != original:
                    from pipeline.utils import _atomic_write
                    _atomic_write(note_md, text)
            except OSError:
                continue

    if cfg.edges_file.exists():
        rewritten: list[str] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for line in cfg.edges_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip() or line.startswith("source\t") or line.startswith("#"):
                rewritten.append(line)
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            parts[0] = canonical_name if parts[0] == duplicate_name else parts[0]
            parts[1] = canonical_name if parts[1] == duplicate_name else parts[1]
            if parts[0] == parts[1]:
                continue
            key = (parts[0], parts[1], parts[2])
            if key in seen_edges:
                continue
            seen_edges.add(key)
            rewritten.append("\t".join(parts))
        from pipeline.utils import _atomic_write
        _atomic_write(cfg.edges_file, "\n".join(rewritten).rstrip() + "\n")

    if duplicate_name in index.notes:
        del index.notes[duplicate_name]
    if duplicate_name in index.embeddings:
        del index.embeddings[duplicate_name]

    log.info("Merged concept %s into %s", duplicate_name, canonical_name)
    return True


def _replace_wikilink_in_dir(directory: Path, old_name: str, new_name: str) -> None:
    if not directory.exists():
        return
    for md in directory.glob("*.md"):
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
            if f"[[{old_name}]]" in text:
                text = text.replace(f"[[{old_name}]]", f"[[{new_name}]]")
                md.write_text(text, encoding="utf-8")
        except OSError:
            continue


def _semantic_moc_rebuild(cfg: Config, client, index: NoteIndex) -> int:
    mocs = {n: info for n, info in index.notes.items() if info["type"] == "moc"}
    if not mocs:
        return 0

    updated = 0
    for moc_name, moc_info in mocs.items():
        related: list[tuple[str, float]] = []
        for name, info in index.notes.items():
            if info["type"] == "moc" or name == moc_name:
                continue
            sim = index.similarity(moc_name, name)
            shared_tags = moc_info["tags"] & info["tags"]
            sim += len(shared_tags) * 0.05
            if sim > 0.3 or moc_name.lower() in info["preview"].lower():
                related.append((name, sim))
        related.sort(key=lambda x: x[1], reverse=True)
        related = related[:10]

        if not related:
            continue

        prompt_lines = [
            f"You are updating a Map of Content (MoC) for the topic: {moc_info['title']}.",
            "",
            "Current MoC preview:",
            moc_info["preview"][:400],
            "",
            "Related notes to include:",
        ]
        for name, score in related:
            info = index.notes[name]
            prompt_lines.append(f"- [[{name}]] ({info['type']}): {info['title']} — {info['preview'][:150]}")

        prompt_lines.extend([
            "",
            "Write an updated MoC section (just the body, no frontmatter). Structure:",
            "## Overview / 概述",
            "<2-3 sentence synthesized summary>",
            "",
            "## <Topic Sections>",
            "- [[Note]] — <1-sentence summary>",
            "",
            "## Bridge Concepts",
            "- <concepts connecting subtopics>",
            "",
            "## Cross-References",
            "- <relevant links>",
            "",
            "Use [[wikilinks]] for all internal links. Keep it concise.",
        ])

        prompt = "\n".join(prompt_lines)
        response = client.generate(prompt, timeout=120)
        if not response:
            continue

        moc_path = cfg.mocs_dir / f"{moc_name}.md"
        try:
            current = moc_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        parts = current.split("\n---\n", 1)
        frontmatter = parts[0] + "\n---\n" if len(parts) > 1 else ""
        new_content = frontmatter + f"# {moc_info['title']}\n\n" + response + "\n"
        moc_path.write_text(new_content, encoding="utf-8")
        updated += 1

    log.info("Semantic MoC rebuild: %d MoCs updated", updated)
    return updated


def _run_semantic_compile(cfg: Config, result) -> tuple[bool, str]:
    """Run semantic compile: direct LLM first, Hermes subprocess fallback.

    If the direct LLM path fails (timeout, model unavailable, generation error),
    falls back to the Hermes agent subprocess as a secondary attempt.
    If both fail, reports loud failure with result.error set.
    """
    from pipeline.llm_client import get_llm_client
    from pipeline.compile.core import _run_agent_compile

    client = get_llm_client(cfg)

    def _try_direct() -> tuple[bool, str]:
        """Attempt direct LLM semantic compile. Returns (ok, summary_or_error)."""
        t0 = time.time()
        index = NoteIndex()
        index.load(cfg)
        if len(index.notes) > 0:
            from pipeline.qmd import _get_client
            skip_local = _get_client() is not None
            if skip_local:
                log.info("QMD enabled --- skipping local embed batch")
            index.embed_all(client, skip_local=skip_local)

        all_ok = True
        try:
            result.crosslinks_added = _semantic_crosslink(cfg, client, index)
        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            log.warning("Direct cross-linking failed: %s", e)
            all_ok = False

        try:
            result.concepts_merged = _semantic_concept_merge(cfg, client, index)
        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            log.warning("Direct concept merge failed: %s", e)
            all_ok = False

        try:
            result.mocs_updated = _semantic_moc_rebuild(cfg, client, index)
        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            log.warning("Direct MoC rebuild failed: %s", e)
            all_ok = False

        result.agent_duration_s = time.time() - t0
        result.agent_succeeded = all_ok

        if all_ok:
            summary = (
                f"cross-links added: {result.crosslinks_added}\n"
                f"concepts merged: {result.concepts_merged}\n"
                f"mocs updated: {result.mocs_updated}"
            )
            return True, summary
        return False, "Direct semantic compile failed (see logs)"

    # Try direct first
    direct_ok, direct_output = _try_direct()
    if direct_ok:
        return True, direct_output

    log.warning("Direct semantic compile failed; attempting Hermes subprocess fallback")

    # Fallback: Hermes agent subprocess (legacy 600s path)
    try:
        agent_ok, agent_output = _run_agent_compile(cfg, result)
        if agent_ok:
            result.agent_succeeded = True
            return True, agent_output
    except Exception as e:
        log.error("Hermes fallback also failed: %s", e)

    result.success = False
    result.error = "Semantic compile failed: direct LLM and Hermes fallback both exhausted"
    result.agent_succeeded = False
    log.error(result.error)
    return False, result.error
