"""Stage 2 planning module.

Takes extracted sources from Stage 1, performs semantic dedup and concept
pre-search, then generates creation plans via hermes agent.

Flow:
  plan_sources(manifest, cfg) -> Plans
    ├─ dedup_check()          — content fingerprint dedup against vault
    ├─ concept_search()       — semantic concept matching via qmd
    └─ generate_plans()       — hermes agent plan generation
"""

from __future__ import annotations

import json
import logging
import math
import re
import time

from pipeline.config import Config
from pipeline.lint import _parse_frontmatter
from pipeline.llm_client import get_llm_client
from pipeline.models import (
    ConceptMatch,
    ExtractedSource,
    Language,
    Manifest,
    Plan,
    Plans,
    SourceType,
    Template,
)
from pipeline.qmd import batch_embed
from pipeline.store import ContentStore
from pipeline.language import detect_language, template_for_language
from pipeline.utils import load_prompt
from pipeline.utils import extract_body as _extract_body

log = logging.getLogger(__name__)


# ─── Fingerprint helpers ──────────────────────────────────────────────────────

def _fingerprint(text: str) -> str:
    """Normalize and extract content fingerprint (first 800 chars).

    Lowercases, collapses whitespace, and truncates.
    """
    normalized = re.sub(r"\s+", " ", text.lower().strip())[:800]
    return normalized


def _jaccard_similarity(fp1: str, fp2: str, ngram: int = 3) -> float:
    """Character n-gram Jaccard similarity. O(n) with sets."""
    if not fp1 or not fp2:
        return 0.0
    ng1 = {fp1[i:i + ngram] for i in range(len(fp1) - ngram + 1)}
    ng2 = {fp2[i:i + ngram] for i in range(len(fp2) - ngram + 1)}
    if not ng1 or not ng2:
        return 0.0
    return len(ng1 & ng2) / len(ng1 | ng2)


# ─── Dedup check ──────────────────────────────────────────────────────────────

def dedup_check(manifest: Manifest, cfg: Config) -> Manifest:
    """Check each source against existing vault sources using content fingerprinting.

    Builds fingerprints from existing vault source files, then compares each
    manifest entry using Jaccard similarity on character 3-grams.
    Sources with similarity > 0.85 are considered duplicates.

    Returns a filtered Manifest with duplicates removed.
    """
    sources_dir = cfg.sources_dir

    # Build fingerprint index from existing sources
    existing_fps: list[dict] = []
    if sources_dir.is_dir():
        for fpath in sorted(sources_dir.glob("*.md")):
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                body = _extract_body(content)
                fp = _fingerprint(body)
                if len(fp) > 100:  # Skip empty/stub sources
                    existing_fps.append({"name": fpath.stem, "fp": fp})
            except OSError:
                continue

    # Check each manifest entry against existing sources
    filtered: list[ExtractedSource] = []
    for entry in manifest.entries:
        entry_fp = _fingerprint(entry.content)
        if len(entry_fp) < 100:
            filtered.append(entry)
            continue

        is_dup = False
        for existing in existing_fps:
            sim = _jaccard_similarity(entry_fp, existing["fp"])
            if sim > 0.85:
                log.info(
                    "Dedup: %s matches existing %s (sim=%.3f)",
                    entry.hash,
                    existing["name"],
                    round(sim, 3),
                )
                is_dup = True
                break

        if not is_dup:
            filtered.append(entry)

    return Manifest(entries=filtered)


# ─── Semantic Near-Duplicate Detection (Rec 3) ────────────────────────────────

def _keyword_dedup_fallback(
    sources: list[ExtractedSource],
    cfg: Config,
) -> list[ExtractedSource]:
    """Keyword/filename fallback when QMD is unreachable.

    Simple dedup based on title/filename exact or substring match."""
    existing_names: set[str] = set()
    sources_dir = cfg.sources_dir
    if sources_dir.is_dir():
        for fpath in sources_dir.glob("*.md"):
            existing_names.add(fpath.stem.lower())

    filtered: list[ExtractedSource] = []
    for src in sources:
        title = (src.title or "").lower().strip()
        # Skip if title matches an existing filename
        if title and title in existing_names:
            log.info("Keyword dedup: %s matches existing filename %s", src.hash, title)
            continue
        filtered.append(src)
    return filtered


def _semantic_dedup(
    sources: list[ExtractedSource],
    cfg: Config,
    store: ContentStore,
) -> list[ExtractedSource]:
    """Semantic near-duplicate detection using QMD embeddings.

    - Embeds content[:1000] for each source via QMD (batch when possible).
    - Queries existing embeddings in store for top match with cosine > 0.92.
    - If match found, logs and skips source.
    - Falls back to Jaccard if QMD is unreachable.
    - Falls back to keyword/filename matching if Jaccard also fails.
    """
    if not sources:
        return []

    # Build preview texts
    texts = [src.content[:1000] for src in sources]
    embeddings: dict[str, list[float]] = {}

    # Try QMD batch embed
    try:
        from pipeline.qmd import _get_client
        _client = _get_client()
        emb_map = batch_embed(texts, client=_client) if _client else {}
        for src in sources:
            key = src.content[:1000]
            if key in emb_map:
                embeddings[src.hash] = emb_map[key]
    except (ImportError, ConnectionError, TimeoutError, OSError, ValueError) as e:
        log.debug("QMD batch embed failed for dedup: %s", e)

    if embeddings:
        filtered: list[ExtractedSource] = []
        for src in sources:
            emb = embeddings.get(src.hash)
            if not emb:
                filtered.append(src)
                continue
            # Store embedding for future dedup
            store.embedding_set(src.content_hash, emb)
            match = store.embedding_find_top_match(src.content_hash, min_similarity=0.92)
            if match:
                chash, sim = match
                src.semantic_similarity = float(sim)
                log.info(
                    "Semantic dedup: %s matches existing content %s (sim=%.3f)",
                    src.hash,
                    chash,
                    round(sim, 3),
                )
                continue
            filtered.append(src)
        return filtered

    # Fallback 1: Jaccard against existing vault sources
    log.info("QMD unreachable for semantic dedup; falling back to Jaccard")
    fallback_manifest = Manifest(entries=sources)
    filtered_manifest = dedup_check(fallback_manifest, cfg)
    if len(filtered_manifest.entries) < len(sources):
        return filtered_manifest.entries

    # Fallback 2: keyword/filename matching
    log.info("Jaccard dedup found no duplicates; falling back to keyword/filename")
    return _keyword_dedup_fallback(sources, cfg)


# ─── Concept Merge Queue (Rec 8) ──────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _process_concept_merge_queue(
    plans: Plans,
    cfg: Config,
    store: ContentStore,
) -> None:
    """Check concept_new candidates against existing concepts via embedding.

    If a new concept is semantically similar (>0.88) to an existing one,
    add it to the merge queue instead of creating a new concept.
    """
    all_new: list[str] = []
    for plan in plans.plans:
        all_new.extend(plan.concept_new)

    if not all_new:
        return

    existing_names: list[str] = []
    if cfg.concepts_dir.is_dir():
        existing_names = [f.stem for f in cfg.concepts_dir.glob("*.md")]
    if not existing_names:
        return

    # Batch embed all names to minimize QMD calls
    unique_texts = list(set(all_new + existing_names))
    from pipeline.qmd import _get_client
    _client = _get_client()
    emb_map = batch_embed(unique_texts, client=_client) if _client else {}
    if not emb_map:
        return

    new_embs = {name: emb for name, emb in emb_map.items() if name in all_new}
    existing_embs = {name: emb for name, emb in emb_map.items() if name in existing_names}

    for plan in plans.plans:
        kept: list[str] = []
        for new_name in plan.concept_new:
            new_emb = new_embs.get(new_name)
            if not new_emb:
                kept.append(new_name)
                continue

            matched = False
            for ex_name, ex_emb in existing_embs.items():
                sim = _cosine_similarity(new_emb, ex_emb)
                if sim > 0.88:
                    store.merge_queue_add(new_name, ex_name, sim, max_size=cfg.max_merge_queue_size)
                    log.info("Merge queue: %s -> %s (sim=%.3f)", new_name, ex_name, sim)
                    matched = True
                    break
            if not matched:
                kept.append(new_name)
        plan.concept_new = kept


# ─── QMD concept search ──────────────────────────────────────────────────────

def concept_search(manifest: Manifest, cfg: Config) -> dict[str, list[ConceptMatch]]:
    """Search existing concepts via qmd for each source.

    Builds query from title + content preview + concept names.
    Returns hash -> [ConceptMatch] mapping.
    Uses parallel qmd queries from shared module.
    """
    from pipeline.qmd import run_qmd_concept_search

    queries: dict[str, str] = {}
    for entry in manifest.entries:
        query = f"{entry.title} {entry.content[:500]}".strip()[:800]
        queries[entry.hash] = query

    return run_qmd_concept_search(queries, cfg)


# ─── Deterministic Planning (Rec 3) ──────────────────────────────────────────



def select_template(source_type: SourceType, content: str) -> Template:
    """Select template based on source type and content characteristics."""
    if source_type == SourceType.PODCAST:
        return Template.STANDARD
    if source_type == SourceType.YOUTUBE:
        return Template.STANDARD
    # Technical content indicators
    technical_markers = [
        "methodology", "data analysis", "results indicate",
        "findings suggest", "empirical", "statistical",
        "p-value", "regression", "hypothesis",
    ]
    content_lower = content[:3000].lower()
    if any(m in content_lower for m in technical_markers):
        return Template.TECHNICAL
    return Template.STANDARD



def generate_plan_heuristic(
    entry: ExtractedSource,
    concept_matches: list[ConceptMatch],
) -> Plan:
    """Generate a creation plan using heuristics — no LLM involved.

    Deterministic decisions: title, language, template, tags.
    Concept linking uses qmd semantic search results.
    """
    from pipeline.extractors._shared import extract_title

    title = entry.title or extract_title(entry.content) or entry.url
    language = detect_language(entry.content)
    template = (
        template_for_language(language)
        if language != Language.EN
        else select_template(entry.type, entry.content)
    )
    tags: list[str] = []  # Tags assigned by agent, not heuristics

    # Determine concept actions from qmd matches
    concept_updates = []
    concept_new = []
    if concept_matches:
        # Top match > 0.5 → update existing
        for match in concept_matches[:3]:
            if match.score > 0.5:
                concept_updates.append(match.concept)
            elif match.score > 0.3:
                # Borderline — still link but note as potential new
                concept_updates.append(match.concept)
        # If only weak matches, suggest new concept
        if concept_matches[0].score < 0.3:
            concept_new.append(title[:80])
    else:
        # No concept matches at all → suggest new concept from title
        if title and title != entry.url:
            concept_new.append(title[:80])

    return Plan(
        hash=entry.hash,
        title=title[:120],
        language=language,
        template=template,
        tags=tags,
        concept_updates=concept_updates,
        concept_new=concept_new,
        moc_targets=[],
    )


def generate_plans_deterministic(
    manifest: Manifest,
    concept_matches: dict[str, list[ConceptMatch]],
) -> tuple[Plans, list[ExtractedSource]]:
    """Generate plans deterministically. Returns (plans, uncertain_sources).

    Sources where heuristics are confident get plans.
    Sources with ambiguous language, no title, or no concept matches
    are returned as uncertain for agent fallback.
    """
    plans = []
    uncertain = []

    for entry in manifest.entries:
        matches = concept_matches.get(entry.hash, [])
        plan = generate_plan_heuristic(entry, matches)

        # Confidence check: does this plan need agent help?
        needs_agent = False
        # No title found from content
        if not plan.title or plan.title == entry.url:
            needs_agent = True
        # Very short content — hard to plan deterministically
        if len(entry.content.strip()) < 50:
            needs_agent = True
        # No concept matches AND suggesting new concept
        if not matches and not plan.concept_new:
            needs_agent = True

        if needs_agent:
            uncertain.append(entry)
        else:
            plans.append(plan)

    return Plans(plans=plans), uncertain


# ─── Plan prompt builder ─────────────────────────────────────────────────────

def build_plan_prompt(
    manifest: Manifest,
    concept_matches: dict[str, list[ConceptMatch]],
    cfg: Config,
) -> str:
    """Compose the agent prompt with all extracted data.

    Includes rules for language detection, template selection, tag suggestions,
    and existing concept/MoC context.
    """
    # Load vault-customized common instructions first, packaged defaults second.
    common = ""
    if cfg.prompts_dir.exists():
        common = load_prompt("common-instructions", cfg.prompts_dir)
    if not common:
        common = load_prompt("common-instructions")
    common = common.replace("{VAULT_PATH}", str(cfg.vault_path)) if common else ""

    # Count existing concepts
    concept_count = 0
    if cfg.concepts_dir.is_dir():
        concept_count = len(list(cfg.concepts_dir.glob("*.md")))

    # Extract existing tag vocabulary from vault
    existing_tags = set()
    for tag_dir in (cfg.entries_dir, cfg.concepts_dir):
        if not tag_dir.exists():
            continue
        for md in tag_dir.glob("*.md"):
            try:
                fm = _parse_frontmatter(md.read_text(encoding="utf-8", errors="replace"))
                tags = fm.get("tags", [])
                if isinstance(tags, list):
                    for t in tags:
                        existing_tags.add(str(t).strip().lower())
            except OSError:
                continue
    existing_tags.discard("")
    tag_vocab = sorted(existing_tags)[:50]  # Cap at 50 most relevant

    # Build sources block
    sources_block_parts = []
    for i, entry in enumerate(manifest.entries):
        h = entry.hash
        title = entry.title[:120]
        content_preview = entry.content[:300].replace("\n", " ")
        source_type = entry.type.value if hasattr(entry.type, "value") else str(entry.type)
        author = entry.author or "unknown"
        matches = concept_matches.get(h, [])
        match_dicts = [{"concept": m.concept, "score": m.score} for m in matches]

        sources_block_parts.append(f"""
---
Source {i+1}:
  hash: {h}
  title: {title}
  type: {source_type}
  author: {author}
  content_preview: {content_preview}
  concept_matches: {json.dumps(match_dicts)}
""")
    sources_block = "".join(sources_block_parts)

    common_section = f"{common}\n\n" if common else ""

    prompt = f"""{common_section}You are a planning agent for an Obsidian wiki pipeline. For each extracted source below, output a creation plan as JSON.

VAULT CONCEPTS DIRECTORY: {concept_count} existing concepts
EXISTING TAG VOCABULARY (prefer reuse over minting new): {', '.join(tag_vocab[:30])}

SOURCES TO PLAN:{sources_block}
---

For EACH source, output a JSON object in a JSON array. Schema per source:

{{"hash": "<source hash>", "title": "<ACTUAL content title for filename — NOT URL slug, NOT platform name>", "language": "en" or "zh", "template": "standard" or "technical" or "chinese", "tags": ["topic-specific tags in English"], "concept_updates": ["existing concept names to update"], "concept_new": ["new concept names to create"], "moc_targets": ["MoC names this source belongs to"]}}

RULES:
- title: Use the content REAL title. Tweet → first meaningful topic. Blog → article title. YouTube → video title.
- NEVER use: "Tweet - user - ID", "Blog - slug", "YouTube - VIDEO_ID", URL slugs
- language: Chinese content → "zh". English content → "en". All other languages → translate to English, process in English.
- template: Data/methodology/findings → "technical". Narrative/philosophical → "standard". Chinese → "chinese".
- tags: 3-6 topic-specific tags derived from content:
  * PREFER reusing tags from EXISTING TAG VOCABULARY above — only mint new tags when content genuinely requires them
  * Prioritize specific entities (e.g. "bitcoin", "gpt-4") over broad categories (e.g. "crypto", "ai")
  * Include compound concepts where relevant (e.g. "smart-contracts", "yield-farming", "zero-knowledge")
  * Lowercase, hyphenated if multi-word
  * NO generic tags: source, url, content, video, podcast, article, blog, tweet, post
  * Tags should reflect what the content is ACTUALLY about, not surface-level keywords
- concept_matches are pre-found via semantic search — rank-sorted by relevance, confirm which are real matches vs tangential
- concept_new: only if genuinely new concept
- Be concise. Output ONLY the JSON array, no explanation.

OUTPUT ONLY VALID JSON."""

    return prompt


# ─── Plan generation via hermes agent ─────────────────────────────────────────

def _parse_agent_output(raw: str) -> list[dict]:
    """Parse hermes agent output into a list of plan dicts.

    Handles ANSI escape codes, box-drawing characters, and partial failures.
    Tries fast-path JSON array first, then falls back to object-by-object parsing.
    """
    # Strip ANSI escape codes and box-drawing characters
    raw_clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", raw)
    raw_clean = re.sub(r"[╭╮╰╯│─╮╰╯├┤┬┴┼]", "", raw_clean)

    # Fast path: try to find a JSON array (non-greedy to avoid matching markdown links)
    json_match = re.search(r"\[[\s\S]*?\]", raw_clean, re.DOTALL)
    if json_match:
        try:
            plans = json.loads(json_match.group())
            if isinstance(plans, list):
                return [p for p in plans if isinstance(p, dict) and "hash" in p]
        except json.JSONDecodeError:
            pass  # Fall through to object-by-object parsing

    # Object-by-object parsing with partial failure recovery
    plans = []
    depth = 0
    start = -1
    for i, c in enumerate(raw_clean):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(raw_clean[start:i + 1])
                    if isinstance(obj, dict) and "hash" in obj:
                        plans.append(obj)
                except (json.JSONDecodeError, Exception):
                    log.warning("Failed to parse JSON object at offset %d", start)
                start = -1

    return plans


def generate_plans(
    manifest: Manifest,
    concept_matches: dict[str, list[ConceptMatch]],
    cfg: Config,
) -> Plans:
    """Generate creation plans via direct LLM call with structured output enforcement.

    Builds the planning prompt, calls the LLM via pipeline.llm_client,
    parses the JSON response using structured output validation,
    and validates each plan against the schema.
    """
    prompt = build_plan_prompt(manifest, concept_matches, cfg)

    # Save prompt for debugging
    extract_dir = cfg.resolved_extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = extract_dir / "plan_prompt.md"
    prompt_file.write_text(prompt, encoding="utf-8")
    log.info("Plan prompt size: %d chars", len(prompt))

    from pipeline.models import PlanOutput

    # Call LLM directly via llm_client (Ollama / OpenRouter / Hermes)
    # Call LLM directly via llm_client (Ollama / OpenRouter / Hermes)
    plan_dicts: list[dict] = []
    llm = get_llm_client(cfg)
    for attempt in range(cfg.max_retries):
        try:
            log.info("Plan agent attempt %d/%d", attempt + 1, cfg.max_retries)
            batch_size = len(manifest.entries)
            if batch_size == 1:
                structured = llm.generate_structured(
                    prompt,
                    schema=PlanOutput,
                    timeout=cfg.llm_structured_timeout,
                )
                if structured is not None and isinstance(structured, PlanOutput):
                    plan_dicts = [{
                        "hash": structured.hash,
                        "title": structured.title,
                        "language": structured.language,
                        "template": structured.template,
                        "tags": structured.tags,
                        "concept_updates": structured.concept_updates,
                        "concept_new": structured.concept_new,
                        "moc_targets": structured.moc_targets,
                    }]
                    break
            else:
                from pipeline.models import PlanOutputList
                structured = llm.generate_structured(
                    prompt,
                    schema=PlanOutputList,
                    timeout=cfg.llm_structured_timeout,
                )
                if structured is not None and isinstance(structured, PlanOutputList):
                    plan_dicts = [
                        {
                            "hash": p.hash,
                            "title": p.title,
                            "language": p.language,
                            "template": p.template,
                            "tags": p.tags,
                            "concept_updates": p.concept_updates,
                            "concept_new": p.concept_new,
                            "moc_targets": p.moc_targets,
                        }
                        for p in structured.plans
                    ]
                    break
            log.warning("Plan agent attempt %d produced empty/invalid structured output", attempt + 1)
            if attempt < cfg.max_retries - 1:
                time.sleep(2 ** attempt)
        except (ImportError, ValueError, TypeError, AttributeError) as e:
            log.warning("Plan agent attempt %d failed: %s", attempt + 1, e)
            if attempt < cfg.max_retries - 1:
                time.sleep(2 ** attempt)

    if not plan_dicts:
        # Fallback to raw text + manual parsing for backwards compat
        log.warning("Structured output failed; falling back to raw text parsing")
        for attempt in range(cfg.max_retries):
            try:
                raw = llm.generate(prompt, timeout=cfg.plan_timeout)
                if raw and raw.strip():
                    # Reuse the legacy JSON extractor
                    data = json.loads(raw)
                    if isinstance(data, list):
                        plan_dicts = [p for p in data if isinstance(p, dict) and "hash" in p]
                        break
                    elif isinstance(data, dict):
                        plan_dicts = [data]
                        break
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
            if attempt < cfg.max_retries - 1:
                time.sleep(2 ** attempt)

    if not plan_dicts:
        log.error("Could not parse any plans from agent output")
        return Plans(plans=[])

    # Convert to Plan objects with validation
    plans: list[Plan] = []
    known_hashes = {e.hash for e in manifest.entries}

    for d in plan_dicts:
        try:
            # Validate required fields
            if "hash" not in d or "title" not in d:
                log.warning("Plan missing required fields (hash/title), skipping")
                continue

            # Skip plans for unknown hashes
            if d["hash"] not in known_hashes:
                log.warning("Plan for unknown hash %s, skipping", d.get("hash"))
                continue

            plan = Plan(
                hash=d["hash"],
                title=d["title"][:120],
                language=Language(d.get("language", "en")),
                template=Template(d.get("template", "standard")),
                tags=d.get("tags", []),
                concept_updates=d.get("concept_updates", []),
                concept_new=d.get("concept_new", []),
                moc_targets=d.get("moc_targets", []),
            )
            plans.append(plan)
        except (ValueError, KeyError) as e:
            log.warning("Failed to validate plan: %s", e)
            continue

    log.info("Parsed %d plans from %d outputs", len(plans), len(plan_dicts))

    # Save plans
    plans_collection = Plans(plans=plans)
    plans_collection.save(extract_dir)

    return plans_collection


# ─── Main entry point ─────────────────────────────────────────────────────────

def plan_sources(manifest: Manifest, cfg: Config) -> Plans:
    """Main entry point for Stage 2 planning.

    Step 0: Dedup check — skip sources already in vault
    Step 1: Semantic concept pre-search via qmd
    Step 2: LLM planning for ALL sources (deterministic planner was producing
            empty tags, concept title-copies, and no MoCs — now always LLM)
    Step 3: Save plans

    Returns Plans object (possibly empty on total failure).
    """
    from pipeline.log import set_correlation
    set_correlation(stage="plan")
    log.info("=== Stage 2: Plan Batch (%d sources) ===", len(manifest.entries))

    if not manifest.entries:
        log.info("No sources to plan, returning empty Plans")
        return Plans(plans=[])

    with ContentStore.open(cfg.resolved_extract_dir) as store:

        # Step 0: Semantic dedup
        log.info("Running semantic dedup check against existing sources...")
        deduped_entries = _semantic_dedup(manifest.entries, cfg, store)
        filtered_manifest = Manifest(entries=deduped_entries)
        removed = len(manifest.entries) - len(filtered_manifest.entries)
        if removed > 0:
            log.info("Found %d semantic duplicates — removed from pipeline", removed)
        else:
            log.info("No semantic duplicates found")

        if not filtered_manifest.entries:
            log.info("All sources were duplicates, returning empty Plans")
            return Plans(plans=[])

        # Step 1: Concept pre-search
        log.info("Pre-searching concept matches via qmd (semantic)...")
        concept_matches = concept_search(filtered_manifest, cfg)
        matched_count = sum(len(v) for v in concept_matches.values())
        log.info(
            "Concept matching complete: %d total matches across %d sources",
            matched_count,
            len(filtered_manifest.entries),
        )

        # Step 2: Always use LLM planner — deterministic planner produced
        # empty tags, title-copy concepts, and zero MoCs.
        log.info("Generating plans via LLM for all %d sources...", len(filtered_manifest.entries))
        all_plans = generate_plans(filtered_manifest, concept_matches, cfg)
        _process_concept_merge_queue(all_plans, cfg, store)

    # Save plans
    extract_dir = cfg.resolved_extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    all_plans.save(extract_dir)

    if all_plans.plans:
        log.info("=== Stage 2 complete: %d plans generated ===", len(all_plans.plans))
    else:
        log.error("=== Stage 2 complete: 0 plans generated ===")

    return all_plans
