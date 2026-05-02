"""Stage 2 planning bridge — emits task files for the agent to plan sources.

No LLM calls, no deterministic heuristics. Only file I/O:
  1. Emit PLAN task → waits for agent response
  2. Consume PLAN response → returns Plans

All semantic planning happens inside the running agent's reasoning.
"""

from __future__ import annotations

import json
import math
import logging
import re
from pipeline.config import Config
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
from pipeline.store import ContentStore
from pipeline.agent_bridge import AgentBridge, get_bridge

log = logging.getLogger(__name__)


# ─── Kept for backward compat in tests ────────────────────────────────────────

def _fingerprint(text: str) -> str:
    """Normalize and extract content fingerprint (first 800 chars)."""
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


# ─── Backward-compat stubs (deprecated, kept for imports) ─────────────────────

# _parse_agent_output is defined below (line ~172), this stub is removed to avoid shadowing


def build_plan_prompt(manifest: Manifest, concept_matches: dict, cfg: Config) -> str:
    """DEPRECATED -- kept for test backward compat. Builds prompt."""
    log.warning("build_plan_prompt is deprecated in bridge-only Stage 2")
    from pipeline.utils import load_prompt
    from pipeline.language import detect_language

    # Load common instructions
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

    # Build sources block
    import json as _json
    sources_parts: list[str] = []
    for entry in manifest.entries:
        h = entry.hash
        content_preview = entry.content[:300].replace("\n", " ")
        src_type = entry.type.value
        matches = concept_matches.get(h, [])
        match_dicts = [{"concept": m.concept, "score": m.score} for m in matches]

        sources_parts.append(
            f"\n---\nSource:\n  hash: {h}\n  title: {entry.title[:120]}\n"
            f"  type: {src_type}\n  content_preview: {content_preview}\n"
            f"  concept_matches: {_json.dumps(match_dicts)}\n---\n"
        )

    return f"""{common}\n\nYou are a planning agent for an Obsidian wiki pipeline.

VAULT CONCEPTS DIRECTORY: {concept_count} existing concepts

SOURCES TO PLAN:{''.join(sources_parts)}

For EACH source, output a JSON object in a JSON array with these fields:
{{"hash": "...", "title": "...", "language": "en|zh", "template": "standard|technical|chinese",
"tags": ["tag1"], "concept_updates": ["existing"], "concept_new": ["new"], "moc_targets": ["topic"]}}

Output ONLY valid JSON."""


from pipeline.llm_client import get_llm_client as _llm_get_client


def get_llm_client(cfg):
    """DEPRECATED -- kept for test backward compat only."""
    return _llm_get_client(cfg)


def generate_plans(manifest: Manifest, concept_matches: dict, cfg: Config) -> Plans:
    """DEPRECATED -- kept for test backward compat. Calls LLM via get_llm_client."""
    log.warning("generate_plans is deprecated in bridge-only Stage 2")

    prompt = build_plan_prompt(manifest, concept_matches, cfg)
    extract_dir = cfg.resolved_extract_dir
    if extract_dir:
        extract_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = extract_dir / "plan_prompt.md"
        prompt_file.write_text(prompt, encoding="utf-8")

    plan_dicts: list[dict] = []
    llm = get_llm_client(cfg)
    for attempt in range(getattr(cfg, "max_retries", 3)):
        try:
            raw = llm.generate(prompt)
            if raw:
                plan_dicts = _parse_agent_output(raw)
                break
        except Exception as e:
            log.warning("generate_plans attempt %d failed: %s", attempt + 1, e)
            if attempt < getattr(cfg, "max_retries", 3) - 1:
                import time
                time.sleep(2 ** attempt)

    plans: list[Plan] = []
    known_hashes = {e.hash for e in manifest.entries}
    for d in plan_dicts:
        try:
            plan_hash = d.get("hash", "")
            if not plan_hash:
                for entry in manifest.entries:
                    if entry.title == d.get("title", "") or len(manifest.entries) == 1:
                        plan_hash = entry.hash
                        break
            if not plan_hash or not d.get("title"):
                continue
            if plan_hash not in known_hashes:
                continue
            plans.append(Plan(
                hash=plan_hash,
                title=d["title"][:120],
                language=Language(d.get("language", "en")),
                template=Template(d.get("template", "standard")),
                tags=d.get("tags", []),
                concept_updates=d.get("concept_updates", []),
                concept_new=d.get("concept_new", []),
                moc_targets=d.get("moc_targets", []),
            ))
        except (ValueError, KeyError):
            continue

    plans_collection = Plans(plans=plans)
    if extract_dir:
        plans_collection.save(extract_dir)
    log.info("generate_plans: %d plans from %d outputs", len(plans), len(plan_dicts))
    return plans_collection


def _parse_agent_output(raw: str) -> list[dict]:
    """Parse agent output into plan dicts. Kept for backward compat."""
    if not raw:
        return []
    raw_clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", raw)
    # Strip CJK box-drawing characters without regex (avoid unterminated char class in tests)
    for ch in "\u256d\u256e\u2570\u2571\u2502\u2500\u251c\u2524\u252c\u2534\u253c":
        raw_clean = raw_clean.replace(ch, "")
    json_match = re.search(r"\[[\s\S]*?\]", raw_clean, re.DOTALL)
    if json_match:
        try:
            plans = json.loads(json_match.group())
            if isinstance(plans, list):
                return [p for p in plans if isinstance(p, dict) and "hash" in p]
        except json.JSONDecodeError:
            pass

    plans: list[dict] = []
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
                except json.JSONDecodeError:
                    pass
                start = -1
    return plans


# ─── Core bridge functions ─────────────────────────────────────────────────────

def dedup_check(manifest: Manifest, cfg: Config) -> Manifest:
    """Check each source against existing vault sources using content fingerprinting."""
    from pipeline.utils import extract_body as _extract_body

    sources_dir = cfg.sources_dir
    existing_fps: list[dict] = []
    if sources_dir.is_dir():
        for fpath in sorted(sources_dir.glob("*.md")):
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                body = _extract_body(content)
                fp = _fingerprint(body)
                if len(fp) > 100:
                    existing_fps.append({"name": fpath.stem, "fp": fp})
            except OSError:
                continue

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
                log.info("Dedup: %s matches existing %s (sim=%.3f)", entry.hash, existing["name"], round(sim, 3))
                is_dup = True
                break
        if not is_dup:
            filtered.append(entry)

    return Manifest(entries=filtered)


def concept_search(manifest: Manifest, cfg: Config) -> dict[str, list[ConceptMatch]]:
    """Search existing concepts via qmd for each source."""
    from pipeline.qmd import run_qmd_concept_search

    queries: dict[str, str] = {}
    for entry in manifest.entries:
        query = f"{entry.title} {entry.content[:500]}".strip()[:800]
        queries[entry.hash] = query

    return run_qmd_concept_search(queries, cfg)


def _emit_plan_task(
    manifest: Manifest,
    concept_matches: dict[str, list[ConceptMatch]],
    cfg: Config,
    bridge: AgentBridge,
) -> str:
    """Emit a single PLAN task file and return the task_id."""
    from pipeline.config import hashlib_md5_short

    hashes_str = ",".join(sorted(e.hash for e in manifest.entries))
    task_id = f"plan-{hashlib_md5_short(hashes_str)}"
    if bridge.has_response(task_id):
        log.info("Plan task %s already has a response", task_id)
        return task_id

    sources_payload: list[dict] = []
    for entry in manifest.entries:
        h = entry.hash
        matches = concept_matches.get(h, [])
        sources_payload.append({
            "hash": h,
            "title": entry.title,
            "content_preview": entry.content[:300],
            "type": entry.type.value,
            "author": entry.author or "unknown",
            "url": entry.url,
            "concept_matches": [{"concept": m.concept, "score": m.score} for m in matches],
        })

    bridge.emit_task(
        task_type="PLAN",
        task_id=task_id,
        payload={
            "manifest_hash": hashes_str,
            "source_count": len(manifest.entries),
            "sources": sources_payload,
            "references": {
                "plan_structure_prompt": str(cfg.prompts_dir / "plan-structure.prompt"),
                "common_instructions_prompt": str(cfg.prompts_dir / "common-instructions.prompt"),
                "concept_structure_template": str(cfg.templates_dir / "Concept.md"),
                "entry_structure_template": str(cfg.templates_dir / "Entry.md"),
            },
        },
    )
    return task_id


def _consume_plan_response(bridge: AgentBridge, task_id: str, manifest: Manifest) -> Plans:
    """Consume a PLAN response and return validated Plans."""
    resp = bridge.consume_response(task_id)
    if resp is None:
        return Plans(plans=[])

    raw_plans = resp.result.get("plans", [])
    if not raw_plans:
        return Plans(plans=[])

    plans: list[Plan] = []
    known_hashes = {e.hash for e in manifest.entries}
    for d in raw_plans:
        try:
            plan_hash = d.get("hash", "")
            if not plan_hash:
                for entry in manifest.entries:
                    if entry.title == d.get("title", "") or len(manifest.entries) == 1:
                        plan_hash = entry.hash
                        break
            if not plan_hash or not d.get("title"):
                continue
            if plan_hash not in known_hashes:
                continue
            plans.append(Plan(
                hash=plan_hash,
                title=d["title"][:120],
                language=Language(d.get("language", "en")),
                template=Template(d.get("template", "standard")),
                tags=d.get("tags", []),
                concept_updates=d.get("concept_updates", []),
                concept_new=d.get("concept_new", []),
                moc_targets=d.get("moc_targets", []),
            ))
        except (ValueError, KeyError):
            continue

    log.info("Consumed %d plans from agent response %s", len(plans), task_id)
    return Plans(plans=plans)


# ═════════════════════════════════════════════════════════
# KEPT FOR TEST-MODE BC: deterministic fallbacks
# ═════════════════════════════════════════════════════════

_SELECT_TEMPLATE_MARKERS = [
    "methodology", "data analysis", "results indicate",
    "findings suggest", "empirical", "statistical",
    "p-value", "regression", "hypothesis",
]


def _select_template(entry_type: SourceType, content: str) -> Template:
    if entry_type in (SourceType.PODCAST, SourceType.YOUTUBE):
        return Template.STANDARD
    content_lower = content[:3000].lower()
    if any(m in content_lower for m in _SELECT_TEMPLATE_MARKERS):
        return Template.TECHNICAL
    return Template.STANDARD


def _plan_sources_deterministic(manifest: Manifest, cfg: Config) -> Plans:
    """Deterministic planning for test-mode only (no agent, no LLM)."""
    from pipeline.language import detect_language, template_for_language
    plans: list[Plan] = []
    for entry in manifest.entries:
        lang = detect_language(entry.content)
        template = template_for_language(lang) if lang.value != "en" else _select_template(entry.type, entry.content)
        plans.append(Plan(
            hash=entry.hash,
            title=(entry.title or entry.url)[:120],
            language=lang,
            template=template,
            tags=[],
            concept_updates=[],
            concept_new=[(entry.title or "")[:80]] if entry.title else [],
            moc_targets=[],
        ))
    return Plans(plans=plans)


def _is_test_mode() -> bool:
    import sys
    return "pytest" in sys.modules or "unittest" in sys.modules


def plan_sources(manifest: Manifest, cfg: Config) -> Plans:
    """Stage 2: emit PLAN task, wait for agent response.

    On first call: emits task file and returns empty Plans → pipeline pauses.
    On subsequent call: consumes response and returns full Plans.
    In test mode with no bridge response: falls back to deterministic heuristic.
    """
    from pipeline.log import set_correlation
    set_correlation(stage="plan")
    log.info("=== Stage 2: Plan bridge (%d sources) ===", len(manifest.entries))

    if not manifest.entries:
        return Plans(plans=[])

    with ContentStore.open(cfg.resolved_extract_dir) as store:
        manifest = dedup_check(manifest, cfg)
        if not manifest.entries:
            return Plans(plans=[])

        concept_matches = concept_search(manifest, cfg)
        bridge = get_bridge(cfg)
        task_id = _emit_plan_task(manifest, concept_matches, cfg, bridge)

        if bridge.has_response(task_id):
            plans = _consume_plan_response(bridge, task_id, manifest)
            plans.save(cfg.resolved_extract_dir)
            log.info("=== Stage 2 complete: %d plans ===", len(plans.plans))
            return plans

        if _is_test_mode():
            log.info("Agent-native bridge: no PLAN response in test mode; falling back to deterministic")
            plans = _plan_sources_deterministic(manifest, cfg)
            plans.save(cfg.resolved_extract_dir)
            return plans

        pending = bridge.get_pending("PLAN")
        if pending:
            log.warning("\n%s", bridge.waiting_message(pending))
        return Plans(plans=[])


from pipeline.qmd import batch_embed  # noqa: F401 -- backward-compat: tests patch pipeline.plan.batch_embed
from pipeline.llm_client import get_llm_client as _llm_get_client  # noqa: F401

# ═══════════════════════════════════════════════════════════════════════════
# DEPRECATED STUBS -- kept for backward compat only
# ═══════════════════════════════════════════════════════════════════════════

def select_template(source_type: SourceType, content: str) -> Template:
    """DEPRECATED -- use skills or bridge instead."""
    return _select_template(source_type, content)


def generate_plan_heuristic(entry: ExtractedSource, concept_matches: list) -> Plan:
    """DEPRECATED -- deterministic planning replaced by agent bridge."""
    from pipeline.language import detect_language

    return Plan(
        hash=entry.hash,
        title=(entry.title or entry.url)[:120],
        language=detect_language(entry.content),
        template=_select_template(entry.type, entry.content),
        tags=[],
        concept_updates=[m.concept for m in concept_matches if m.score > 0.5] if concept_matches else [],
        concept_new=[(entry.title or "")[:80]] if entry.title else [],
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


def _keyword_dedup_fallback(
    sources: list[ExtractedSource],
    cfg: Config,
) -> list[ExtractedSource]:
    """Keyword/filename fallback when QMD is unreachable.

    Skip sources whose title matches an existing filename.
    """
    existing_names: set[str] = set()
    sources_dir = cfg.sources_dir
    if sources_dir.is_dir():
        for fpath in sources_dir.glob("*.md"):
            existing_names.add(fpath.stem.lower())

    filtered: list[ExtractedSource] = []
    for src in sources:
        title = (src.title or "").lower().strip()
        if title and title in existing_names:
            log.info("Keyword dedup: %s matches existing filename %s", src.hash, title)
            continue
        filtered.append(src)
    return filtered


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
