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
import re
import subprocess
import time

from pipeline.config import Config
from pipeline.models import (
    ConceptMatch, ExtractedSource, Language, Manifest, Plan, Plans, Template, SourceType,
)
from pipeline.utils import _CJK_RE, extract_body as _extract_body
from pipeline.lint import _parse_frontmatter

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
            except Exception:
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



def detect_language(content: str) -> Language:
    """Detect content language from character distribution."""
    if not content:
        return Language.EN
    sample = content[:2000]
    cjk_chars = len(_CJK_RE.findall(sample))
    total_chars = len(re.sub(r"\s", "", sample))
    if total_chars == 0:
        return Language.EN
    # If >20% CJK characters, treat as Chinese
    if cjk_chars / total_chars > 0.2:
        return Language.ZH
    return Language.EN


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

    title = extract_title(entry.content) or entry.title or entry.url
    language = detect_language(entry.content)
    template = (
        Template.CHINESE if language == Language.ZH
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
    # Load common instructions if available
    common_path = cfg.prompts_dir / "common-instructions.prompt"
    common = ""
    if common_path.exists():
        try:
            common = common_path.read_text(encoding="utf-8").strip()
            common = common.replace("{VAULT_PATH}", str(cfg.vault_path))
        except Exception:
            pass

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
- language: Chinese content → "zh", everything else → "en"
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
    """Generate creation plans via hermes agent.

    Builds the planning prompt, calls hermes chat, parses the JSON response,
    and validates each plan against the schema.
    """
    prompt = build_plan_prompt(manifest, concept_matches, cfg)

    # Save prompt for debugging
    extract_dir = cfg.resolved_extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = extract_dir / "plan_prompt.md"
    prompt_file.write_text(prompt, encoding="utf-8")
    log.info("Plan prompt size: %d chars", len(prompt))

    # Call hermes agent
    cmd = [cfg.agent_cmd, "chat", "-q", prompt, "-Q"]
    plan_dicts: list[dict] = []

    for attempt in range(cfg.max_retries):  # configurable retries
        try:
            log.info("Plan agent attempt %d/%d", attempt + 1, cfg.max_retries)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=cfg.plan_timeout,
            )
            if result.returncode == 0 and result.stdout.strip():
                plan_dicts = _parse_agent_output(result.stdout)
                if plan_dicts:
                    break
            log.warning("Plan agent attempt %d failed (exit %d): %s",
                        attempt + 1, result.returncode,
                        result.stderr[:200] if result.stderr else "no stderr")
            if attempt < cfg.max_retries - 1:
                time.sleep(2 ** attempt)
        except subprocess.TimeoutExpired:
            log.warning("Plan agent timeout on attempt %d", attempt + 1)
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

    log.info("Parsed %d plans from %d agent outputs", len(plans), len(plan_dicts))

    # Save plans
    plans_collection = Plans(plans=plans)
    plans_collection.save(extract_dir)

    return plans_collection


# ─── Main entry point ─────────────────────────────────────────────────────────

def plan_sources(manifest: Manifest, cfg: Config) -> Plans:
    """Main entry point for Stage 2 planning.

    Step 0: Dedup check — skip sources already in vault
    Step 1: Semantic concept pre-search via qmd
    Step 2: Deterministic planning (heuristics) — handles ~80% of sources
    Step 3: Agent fallback — only for uncertain sources

    Returns Plans object (possibly empty on total failure).
    """
    log.info("=== Stage 2: Plan Batch (%d sources) ===", len(manifest.entries))

    if not manifest.entries:
        log.info("No sources to plan, returning empty Plans")
        return Plans(plans=[])

    # Step 0: Dedup check
    log.info("Running semantic dedup check against existing sources...")
    filtered_manifest = dedup_check(manifest, cfg)
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

    # Step 2: Deterministic planning
    log.info("Generating plans deterministically...")
    deterministic_plans, uncertain = generate_plans_deterministic(
        filtered_manifest, concept_matches,
    )
    log.info("Deterministic: %d plans, %d uncertain (need agent)",
             len(deterministic_plans.plans), len(uncertain))

    # Step 3: Agent fallback for uncertain sources only
    if uncertain:
        log.info("Spawning planning agent for %d uncertain sources...", len(uncertain))
        uncertain_manifest = Manifest(entries=uncertain)
        uncertain_concept_matches = {
            e.hash: concept_matches.get(e.hash, []) for e in uncertain
        }
        agent_plans = generate_plans(uncertain_manifest, uncertain_concept_matches, cfg)
        # Merge: deterministic plans first, then agent plans
        all_plans = Plans(plans=deterministic_plans.plans + agent_plans.plans)
    else:
        all_plans = deterministic_plans

    # Save plans
    extract_dir = cfg.resolved_extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    all_plans.save(extract_dir)

    if all_plans.plans:
        log.info("=== Stage 2 complete: %d plans generated ===", len(all_plans.plans))
    else:
        log.error("=== Stage 2 complete: 0 plans generated ===")

    return all_plans
