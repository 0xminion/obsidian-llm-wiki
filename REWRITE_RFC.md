# RFC: obsidian-llm-wiki v2.0 Architecture Rewrite

**Status:** Draft for review  
**Scope:** Rewrite Stages 2–4. Stage 1 (Extract) stays as-is.  
**Inspiration:** `atomicmemory/llm-wiki-compiler` (Node/TS, v0.6.0)  
**Current branch:** `feat/agent-prompt-stages-23` @ 233b756 (852/853 tests passing)  

---

## 1. Current State — What Works vs What Doesn’t

### ✅ Working
| Component | State |
|-----------|-------|
| Stage 1 (Extract) | Deterministic — defuddle / camoufox / curl / podcast / YouTube. URL dedup, clippings bypass, SHA hash tracking. |
| Config / Setup | `Config` dataclass, vault init, logging correlation. |
| Agent bridge | `pipeline/agent_bridge.py` exists — task/response file I/O, idempotency, audit trail. |
| QMD embeddings | Ollama `qwen3-embedding:0.6b` at `localhost:11434`. Lazy-loaded module cache. |
| Tests | 852/853 pass. One unrelated date-tz failure in `test_lint.py`. |

### ❌ Broken / Suboptimal
| Problem | Root Cause | Impact |
|---------|-----------|--------|
| Plan stage produces thin stubs | `_plan_sources_deterministic()` + test-mode fallback replaces semantic reasoning with regex thresholds | Plans lack genuine concept analysis; entries come out thin |
| Concept notes use wrong headings | `templates.py` still generates `## Core concept` / `## Context` / `## Links` (and Chinese equivalents) | Pipeline skill mandates `## Boundary/## Essence/## Role`; persistent structural mismatch |
| Concept content is a stub | `_generate_concept_template()` generates passive fluff: "X is a concept introduced in… As the vault grows, this note should be enriched…" | Notes are 150–400 chars; user requirement is 1000+ chars, mechanistic depth |
| Create stage empty/malformed | `create_file_templates()` does not wire the agent bridge for actual insight generation | Entry `## Summary` + `## Core Insights` are empty or generic |
| Compile semantic crosslinks broken | Regex `LINK\s+(['\"])([^'\"]+)\1` fails on multi-word filenames with Chinese | Crosslink suggestions are silently dropped or cause `IndexError` |
| No revision queue | Files are written to `04-Wiki/` directly on create — no review gate | User can’t inspect before vault mutation |
| No hash-based incrementality | Every run re-plans all sources regardless of change state | Waste of agent reasoning tokens; slow on large vaults |
| No source provenance markers | Paragraphs have no `^[source-md]` citations | User cannot trace claims back to sources |
| No per-source concept budget | LLM receives entire source body when planning; long sources blow context | OOM or truncated planning for PDFs / long posts |
| No schema layer | Page kinds are hard-coded; no `overview`, `entity`, `comparison` | MoCs and concepts look identical regardless of semantic type |
| No lint pass | Quality issues (dangling links, orphans, contradictions) are never surfaced | Vault silently degrades over time |
| No MCP server | Agent must use file bridge — no programmatic query/serve endpoint | Integration friction |

---

## 2. Proposed New Architecture (Stages 2–4)

Keep the existing **agent-native task/response bridge**. The rewrite replaces the *implementation* of Stages 2–4 under that bridge — it does not remove the bridge.

```
Stage 1: Extract  (unchanged)
  → Raw text + metadata → `08-Plan/manifest.json`

Stage 2: Plan Agent-Native (rewrite)
  → Phase A: Concept extraction (LLM, per-source) → `08-Plan/concepts-{hash}.json`
  → Phase B: Planning synthesis (LLM, vault-wide)     → `08-Plan/plans.json`
  → Per-source prompt budget enforcement

Stage 3: Create Agent-Native (rewrite)
  → Emit create tasks per plan → `.agent-bridge/tasks/create-{hash}.json`
  → Agent processes: writes entries, concepts, sources, MoC entries
  → **Review queue gate** → `.candidate/` (not `04-Wiki/`) until approved

Stage 4: Compile (rewrite)
  → Two-phase: extract concepts → generate pages (like llmwiki)
  → Hash-based incremental: only changed sources go to LLM
  → Semantic cross-linking + concept merge (fixed regex)
  → MoC rebuild with schema-aware overview pages
  → Lint pass: broken links, orphans, contradictions, low-confidence
  → Index + embeddings rebuild

Stage 5: Serve (new)
  → MCP server exposing wiki tools (ingest, compile, query, lint)
  → Semantic query with reranked chunk retrieval
```

---

## 3. Module-by-Module Rewrite Plan

### 3.1 Stage 2: `pipeline/plan.py` → Two-Phase Planning

**Current:** single-phase planning; `_emit_plan_task()` sends one task per run; consumes one response.

**llmwiki pattern:** Phase 1 extracts concepts from *all* sources. Phase 2 generates plans. Eliminates order dependence, dedupes shared concepts, catches failures before writing anything.

**New plan.py design:**

```python
# Phase A: Per-source concept extraction (agent task per source)
def _emit_source_concept_task(entry: ExtractedSource, bridge: AgentBridge) -> str:
    """Extract concepts, tags, language, template from one source.
    Enforces per-source prompt budget (default ~50k chars)."""
    ...

def _consume_source_concept_response(bridge, task_id, entry) -> SourceConcepts:
    """Parse LLM-extracted concepts with validation."""
    ...

# Phase B: Vault-wide plan synthesis (single agent task)
def _emit_plan_synthesis_task(
    all_concepts: list[SourceConcepts],
    existing_concepts: list[str],
    bridge: AgentBridge,
) -> str:
    """Merge per-source concepts into unified plans.
    Resolve duplicates, assign update vs new, choose MoC targets."""
    ...

def _consume_plan_synthesis_response(...) -> Plans:
    ...
```

**Why two phases:**
- Phase A parallelizes per-source → faster on batches.
- Phase B is a single merge step that sees the whole picture → eliminates duplicate "Prediction Markets" concepts from 3 different sources.
- Failures in Phase A are caught before Phase B runs — nothing written.

**Per-source prompt budget:**
- `LLMWIKI_PROMPT_BUDGET_CHARS` default = 200,000 (your requirement ~50–100K chars).
- Each contributing source gets fair-share truncation when the cap fires.
- Warning emitted to stderr when truncation kicks in.

**Test-mode fallback:** Kept but clearly scoped:
- Only when `pytest` in `sys.modules` AND no pre-staged bridge responses.
- Falls back to `_plan_sources_deterministic()` wrapped in `try/finally` that restores `cfg.agent_native = True`.
- No change to 853 test expectations.

**Heading enforcement:**
- `plan.json` now carries `concept_structure`: `"boundary_essence_role"` for all plans.
- `templates.py` updated to generate ONLY `## Boundary`, `## Essence`, `## Role`.
- Chinese concept headings use English: `## Boundary`, `## Essence`, `## Role` — prose in Chinese.
- `validate.py` `_REQUIRED_SECTIONS["concept_en"]` = `["## Boundary", "## Essence", "## Role"]`.
- `validate.py` `_REQUIRED_SECTIONS["concept_zh"]` = `["## Boundary", "## Essence", "## Role"]`.

---

### 3.2 Stage 3: `pipeline/create/orchestrator.py` + `templates.py` → Review Queue Gate

**Current:** writes directly to `04-Wiki/`.

**llmwiki pattern:** `compile --review` writes candidate JSON to `.llmwiki/candidates/`; `review approve` promotes to `wiki/`.

**New design:**

```
Stage 3 output flow:
  1. Agent writes `.candidate/{hash}/` (entry.md, source.md, concepts/*.md)
  2. Pipeline pauses for review (if `--review`)
  3. `review list` → shows pending candidates
  4. `review show <id>` → prints candidate preview
  5. `review approve <id>` → moves to vault proper, rebuilds index/embeddings
  6. `review reject <id>` → archives without touching vault
```

**Candidate file structure per source:**
```
.candidate/
  {hash}/
    entry.md          → entry note (summary, insights)
    source.md         → source note (original content + frontmatter)
    concepts/
      {slug}.md       → evergreen concept notes
    moc_entries/
      {moc}.md        → description paragraphs for MoC pages
```

**Lock:** `approve`/`reject` acquire `.agent-bridge/lock` — serializes against concurrent compile.

**Agent task payload (create stage):**
```json
{
  "task_type": "CREATE",
  "task_id": "create-{hash}",
  "payload": {
    "source_content": "...budget-truncated...",
    "plan": {"title": "...", "language": "en", "tags": [...], "concept_new": [...]},
    "template_paths": ["pipeline/assets/templates/Entry.md", "Concept.md", "Source.md"],
    "concept_structure": "boundary_essence_role",
    "min_concept_chars": 1000,
    "concepts_dir": "04-Wiki/concepts"
  }
}
```

**Agent response payload:**
```json
{
  "result": {
    "entry": "---\ntitle: ...\n---\n## Summary\n...",
    "source": "---\ntitle: ...\n---\n## Original content\n...",
    "concepts": {
      "Slug Name": "---\ntitle: ...\n---\n## Boundary\n...\n## Essence\n...\n## Role\n..."
    },
    "moc_entries": {
      "Topic Name": "# Overview\n..."
    },
    "confidence": 0.95,
    "provenance_state": "extracted"
  }
}
```

**Stub prevention (min 1000 chars, enforced in code + validation):**
- `## Boundary`: exactly 1 sentence, declarative boundary-drawing: "This IS X, this is NOT X"
- `## Essence`: 2–4 paragraphs, irreducible mechanics + evidence + tensions
- `## Role`: 1–2 paragraphs, what enables / constrains / sources — structural dependencies

---

### 3.3 Stage 4: `pipeline/compile/` → Two-Phase Compile + Hash-Based Incremental

**Current:** single pass; broken regex crosslinks; re-runs semantic compile on everything.

**llmwiki pattern:** Incremental via SHA-256. Only changed sources hit the LLM. Everything else skipped.

**New compile flow:**

```python
class Compiler:
    def compile(self, cfg: Config) -> CompileResult:
        # Phase 1: Load source hash registry
        source_hashes = self._load_source_hashes(cfg)
        changed = [s for s in sources if s.hash != source_hashes.get(s.url)]

        # Phase 2: Extract concepts from CHANGED sources only
        new_concepts = self._extract_concepts(changed)  # LLM

        # Phase 3: Merge with existing concepts
        merged = self._merge_concepts(existing_concepts, new_concepts)

        # Phase 4: Generate pages
        pages = self._generate_pages(merged)  # LLM per concept

        # Phase 5: Resolve [[wikilinks]]
        self._resolve_links(pages)

        # Phase 6: Lint
        diagnostics = self._lint(pages)

        # Phase 7: Rebuild index + embeddings
        self._rebuild_index(pages)
        self._rebuild_embeddings(pages)

        return CompileResult(pages=pages, diagnostics=diagnostics)
```

**Provenance tracking (every paragraph cites source):**
```markdown
The system uses a two-phase compile pipeline. ^[architecture-notes.md:42-58]
```

- Line-range format supported: `^[filename.md:42-58]` or `^[filename.md#L42-L58]`
- Validation: missing source files, malformed ranges, impossible ranges (line 0, 8→3), ranges past EOF.

**Schema layer (optional `.llmwiki/schema.json`):**
```json
{
  "kinds": {
    "concept": { "minWikilinks": 2 },
    "entity": { "minWikilinks": 1 },
    "comparison": { "minWikilinks": 3 },
    "overview": { "minWikilinks": 4, "seedPages": ["AI Overview", "Crypto Overview"] }
  }
}
```

**Lint rules added (llmwiki-style):**
| Rule | Description |
|------|-------------|
| `low-confidence` | Flags pages with `confidence < threshold` (default 0.7) |
| `contradicted-page` | Flags pages with non-empty `contradictedBy` |
| `excess-inferred-paragraphs` | Flags pages with >20% uncited paragraphs (provenance gap) |
| `broken-wikilink` | `[[Target]]` points to non-existent page |
| `orphan` | Page not linked from index or any other page |
| `empty-page` | Page has <100 chars body after frontmatter |

**Concept merge reconciliation (when multiple sources contribute to same slug):**
- `confidence`: `min()` across sources
- `provenanceState`: `"merged"`
- `contradictedBy`: union, deduped by slug

---

### 3.4 New: `pipeline/mcp/` — MCP Server

**Design:**
```python
# pipeline/mcp/server.py

class WikiMCPServer:
    """Expose wiki tools via Model Context Protocol (stdio transport)."""

    tools = [
        Tool("ingest_source", "Fetch URL or file into 01-Raw/"),
        Tool("compile_wiki", "Run two-phase compile pipeline"),
        Tool("query_wiki", "Two-step grounded answer with optional --save"),
        Tool("search_pages", "Full content of semantically relevant pages"),
        Tool("read_page", "Read single page by slug"),
        Tool("lint_wiki", "Run quality checks; return structured diagnostics"),
        Tool("wiki_status", "Page count, source count, orphans, pending changes (read-only)"),
    ]

    resources = [
        Resource("llmwiki://index", "Full wiki/index.md"),
        Resource("llmwiki://concept/{slug}", "Single concept page"),
        Resource("llmwiki://sources", "List of ingested source files"),
        Resource("llmwiki://state", "Compilation state: hashes, last compile times"),
    ]
```

**Usage:**
```bash
python -m pipeline.mcp.server --vault ~/MyVault
```

Then configure in Claude/Cursor:
```json
{
  "mcpServers": {
    "obsidian-wiki": {
      "command": "python",
      "args": ["-m", "pipeline.mcp.server", "--vault", "/home/linuxuser/MyVault"],
      "env": { "OLLAMA_HOST": "http://localhost:11434" }
    }
  }
}
```

---

## 4. New File Structure

```
pipeline/
  agent_bridge.py          # UNCHANGED — file I/O, task/response
  config.py                # + agent_native default + prompt_budget
  extract.py               # UNCHANGED
  extractors/              # UNCHANGED
  plan.py                  # REWRITE — two-phase, per-source budget
  create/
    templates.py           # REWRITE — Boundary/Essence/Role only, stub prevention
    orchestrator.py      # REWRITE — review queue, candidate staging
    prompts.py           # + new concept-structure.prompt
    validate.py          # + stub patterns, min lengths
  compile/
    core.py              # REWRITE — incremental compile, hash registry
    semantic.py          # REWRITE — fixed crosslink regex, concept merge
    structural.py        # REWRITE — provenance markers, lint integration
    indexgen.py          # NEW — auto-generated index.md
    hasher.py            # NEW — SHA-256 source hash tracking
    provenance.py        # NEW — paragraph citation validation
  mcp/                     # NEW
    server.py
    tools.py
    resources.py
  lint/
    checks.py            # + low-confidence, contradicted, excess-inferred, orphan
    rules.py             # NEW — schema-aware rule engine
  llm_client.py            # + budget enforcement, prompt truncation
  qmd.py                   # UNCHANGED
  store.py               # + candidate staging paths
  vault.py               # + approve/reject candidate operations
  models.py              # + PageKind, Provenance, Candidate dataclasses
```

---

## 5. Data Flow Comparison

### Current Flow (broken)
```
Extract → [plan.json thin stubs] → create writes directly to vault →
compile breaks on regex → vault degrades
```

### Proposed Flow (llmwiki-inspired)
```
Extract → Phase A: concept extraction (per-source, budgeted) →
Phase B: plan synthesis (vault-wide, deduped) →
Review queue (candidate staging) → Agent approves →
Vault update → Phase 1 compile: extract changed-source concepts →
Phase 2 compile: generate pages → resolve links → lint →
rebuild index/embeddings → MCP serve
```

---

## 6. Credit Budget & Template Strategy

### Agent-native task credit estimate
| Task type | Estimated credits | When |
|-----------|------------------| ------|
| Concept extraction (Phase A) | ~5–8K / source | Once per changed source |
| Plan synthesis (Phase B) | ~3–5K / batch | Once per run |
| Create per source | ~8–12K / source | Once per changed source |
| Compile Phase 1 (concept extract) | ~5–8K / changed source | Once per changed source |
| Compile Phase 2 (page gen) | ~5–10K / concept | Once per changed concept |
| Compile semantic crosslinks | ~3–5K / batch | Optional (can skip) |
| MoC rebuild | ~2–4K / MoC | Once per run |

**Batching:**
- Phase A tasks are batched by `max_workers=4` to parallelize.
- Phase B is a single task (all sources summarized in one prompt).

### Templates added
- `pipeline/assets/templates/Candidate.md` — review queue preview
- `pipeline/assets/prompts/concept-extract.prompt` — per-source concept extraction prompt
- `pipeline/assets/prompts/plan-synthesis.prompt` — vault-wide plan synthesis prompt
- `pipeline/assets/prompts/page-generate.prompt` — per-concept page generation prompt

---

## 7. Key Design Decisions

1. **Stage 1 kept intact.** Your extraction works. No changes to `extract.py`, extractors, clippings pipeline.

2. **Deterministic heuristics removed entirely.** No `_generate_plan_heuristic()`, no `_generate_concept_template()` fallback. When the agent runs out of credits or the bridge has no responses, the pipeline pauses with a message — it does NOT silently degrade to regex scaffolds. (The test-mode fallback remains because 853 tests depend on it, but it's explicitly gated behind `pytest` in `sys.modules` AND zero bridge responses.)

3. **Heading enforcement is code-level, not just prompt-level.** The skill mandates `## Boundary/## Essence/## Role`, but the *code* enforces it in `templates.py` AND `validate.py`. Four-layer sync (prompt + template + code + validation) prevents the drift that caused previous failures.

4. **Incremental by default.** Without this, every recompile re-extracts all concepts, destroying the cost efficiency that makes agent-native mode viable.

5. **Schema is optional.** New vaults work without `.llmwiki/schema.json`. Missing/invalid `kind` falls back to `"concept"`.

6. **Review queue is optional.** `pipeline ingest --auto-approve` bypasses review for headless CI. Default behavior requires review.

7. **Obsidian-native.** Output stays Markdown + YAML frontmatter. No JSON output directories. No proprietary database. `[[wikilinks]]` resolve to concept titles. Compatible with your existing `04-Wiki/` + `05-Map/` structure.

---

## 8. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Template-heading sync drift (again) | High | Four-layer sync + `_STUB_PATTERNS` regex rejects misformatted agent output |
| Agent-native task JSON schema mismatch | Medium | Strict Pydantic validation on task/response payloads; clear error messages |
| Ollama timeout on large batches | Medium | Configurable `LLMWIKI_REQUEST_TIMEOUT_MS`; per-concept budget caps; retry with backoff |
| QMD embedding memory growth | Low | Lazy-loaded cache; no persistence to disk; optional `--no-qmd` flag |
| Review queue lock file left behind | Low | `try/finally` + signal handler; auto-reclaim stale locks on startup |
| Existing vault concepts overwritten | Medium | Phase B "concept merge" uses Jaccard 0.85 dedup — only truly new concepts are added |

---

## 9. Implementation Order

1. **New models:** Add `PageKind`, `Provenance`, `Candidate`, `SourceHash` to `models.py`.
2. **`create/` rewrite:** Fix `templates.py` → Boundary/Essence/Role; fix `validate.py`; rewrite `orchestrator.py` → review queue.
3. **`plan.py` rewrite:** Two-phase planning; per-source budget; heading enforcement.
4. **`compile/` rewrite:** Two-phase compile; hash-based incremental; provenance markers.
5. **`lint/` additions:** New rules (low-confidence, contradictions, orphans).
6. **`mcp/` new module:** Server, tools, resources.
7. **Template assets:** New prompt files for concept-extract, plan-synthesis, page-generate.
8. **Config wiring:** `agent_bridge` integration with new task types.
9. **Tests:** 853 existing tests must pass. Add new tests for incremental compile, review queue, provenance markers.
10. **Integration test:** Run full pipeline on 5 known URLs, verify output format matches vault standards.

---

## 10. Questions for Your Review

1. **Auto-approve toggle.** Should `pipeline ingest` default to review queue (`--review`) or auto-approve (`--auto-approve`)? Current behavior writes directly.
2. **Per-concept budget.** Default 200,000 chars (~50K tokens) per concept? Or lower for local models? Current Ollama config uses 30-min timeout but no budget cap.
3. **Concept merge threshold.** Jaccard 0.85 threshold for dedup — keep or adjust? You previously changed from title fuzzy matching to Jaccard similarity.
4. **Schema kinds.** Do you want `overview` (topic hub) auto-generated as seed pages, or manual only?
5. **Provenance markers.** Use `^[source.md]` (llmwiki-style) or stick to frontmatter `sources:` list only?
6. **MCP priority.** Should MCP server be in this rewrite sprint or a follow-up? It's additive — no existing tests depend on it.

---

**Prepared:** Sat May 02 2026  
**Review format:** Inline edits or red-lines welcome. I will iterate until approved, then execute.
