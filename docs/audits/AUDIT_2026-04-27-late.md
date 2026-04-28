# Obsidian LLM Wiki — Comprehensive Code Audit Report

**Date:** 2026-04-27 (late session)  
**Repo:** https://github.com/0xminion/obsidian-llm-wiki  
**Commit:** `8ba9cb1` (HEAD, `main`)  
**Tests:** 820 collected, all passing (1m56s runtime)  
**Pipeline LOC:** ~15,997 across 47 source files | 36 test files (~11,300 test LOC)  
**Reviewers:** Hermes (systematic audit) + Independent second-opinion agent (timed out, primary review completed)  
**Method:** Prior audit cross-reference (AUDIT_2026-04-27, AUDIT_2026-04-24, CODE_REVIEW_2026-04-23), file reads, static analysis (ruff, flake8, bandit), AST checks, run-time verification.

---

## Honest Take

This is a well-engineered, battle-tested Karpathy-style PKM pipeline. Multiple audit cycles have genuinely improved it. All HIGH and MEDIUM bugs from the Apr-24 and Apr-27 audits were either fixed or accepted with justification. The direct-LLM compile pass (replacing the 600s agent subprocess) is the single most impactful architectural improvement since the last review. Version drift is resolved. `canonical_body` extraction is correct. 820 passing tests shows real engineering discipline.

**Rating: 8.2/10** (up from 7.8 at last audit)

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Architecture** | 8.8/10 | Clean 3-stage pipeline, good separation, provider-agnostic LLM client, QMD MCP integration is sound, version SSoT works, template-mode default is smart. Remaining: O(N²) concept similarity scan, VaultSnapshot full scan on every compile. |
| **Correctness** | 8.5/10 | 820 tests pass. All prior HIGH bugs verified fixed. Latent: `embed_all` dict key collision when two notes share identical title+preview. QMD health-check overhead on every embed call. |
| **Security** | 7.5/10 | No eval/exec, no shell=True, URL scheme whitelist, defusedxml used when available. Bandit B310 flags are localhost false positives. Missing: archive.org fallback doesn't re-validate URL scheme after redirect. |
| **Performance** | 7.8/10 | Template mode fast, QMD parallelized, O(N) merge replacement is solid. Compile agent subprocess still exists as fallback (600s ceiling). O(N²) concept similarity scan in `_semantic_concept_merge` is a hidden scaling limit. |
| **Maintainability** | 8.2/10 | Good docstrings, dataclasses, type hints. 47 files with clear naming. 33 flake8 cosmetic issues. Some dead code (`create/agent.py` temp file leak, `_common.py` dead `check_dependencies`). |
| **Test Coverage** | 7.8/10 | 820 tests, 0.72x test-to-source ratio. Missing: concurrent stress test, embed key collision coverage, SPA extraction (expected — requires Playwright), live subprocess circuit-breaker test. |

---

## Structural Analysis

### Module Inventory (Top 12 by LOC)

| File | LOC | Role |
|------|-----|------|
| `pipeline/store.py` | 956 | SQLite-backed content store (10 tables, WAL mode) |
| `pipeline/vault_setup.py` | 844 | Vault initialization, migration, fixture generation |
| `pipeline/lint/checks.py` | 842 | 15-check lint system with staleness scoring |
| `pipeline/plan.py` | 775 | Stage 2: LLM-driven planning with JSON extraction, semantic dedup, merge queue |
| `pipeline/extractors/podcast.py` | 751 | RSS/XML parsing, transcription, episode matching |
| `pipeline/compile/core.py` | 673 | Compile pass: merge queue, incremental compiler, agent fallback, reports |
| `pipeline/utils.py` | 639 | Shared utilities: frontmatter, wikilinks, atomic write, circuit breaker |
| `pipeline/vault.py` | 627 | File I/O: write_edge, reindex, MoC updates, collision resolution |
| `pipeline/llm_client.py` | 601 | Unified LLM client (Ollama/OpenRouter/Hermes) with structured output |
| `pipeline/create/templates.py` | 586 | Template-based note generation |
| `pipeline/compile/semantic.py` | 495 | Semantic compile: embedding, cross-linking, concept merge, MoC rebuild |
| `pipeline/extractors/_shared.py` | 525 | Shared extraction utilities: curl wrapper, URL validation, title extraction |

### Architecture

```
Inbox (.url, .md clippings)
    │
    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  Stage 1    │ ──► │  Stage 2    │ ──► │  Stage 3        │
│  Extract    │     │  Plan       │     │  Create         │
│  (parallel) │     │  (LLM)      │     │  (templates)    │
└─────────────┘     └─────────────┘     └─────────────────┘
       │                                    │
       ▼                                    ▼
Manifest.json                         Vault files
Store.db (dedup/DLQ)                  (entries, concepts, MoCs)
Telemetry.jsonl
       │
       ▼
┌─────────────┐     ┌─────────────┐
│  Compile    │     │  Lint       │
│  (merge,    │     │  (15 checks)│
│   crosslink)│     │             │
└─────────────┘     └─────────────┘
```

**Pipeline modes:**
- **Template mode** (default): Deterministic Python templating + small LLM call for insights
- **Agent mode** (`--agent`): Full Hermes subprocess per batch (slower, 600s timeout)
- **Review mode** (`--review`): Stage files for approval before writing
- **Resume** (`--resume`): Skip Stages 1+2, re-run Stage 3
- **Incremental compile** (`--watch`): mtime-tracking to avoid full rescans
- **Direct semantic compile** (default): Python-driven embedding similarity + LLM validation, with Hermes subprocess fallback

---

## Prior Audit Verification

Cross-referenced against [AUDIT_2026-04-27.md](AUDIT_2026-04-27.md). All findings from the Apr-27 audit were independently verified:

| ID | Prior Claim | Verification | Status |
|----|-------------|--------------|--------|
| H1 | qmd_mcp header case-sensitivity bug | `resp.headers.get()` with multi-case fallback at line 91-94. No `dict(resp.headers)` found. | ✅ FIXED |
| H2 | compile dead availability check | `compile/semantic.py:58-65` uses `pipeline.qmd._get_client()` which performs health check and returns `None`. Correct behavior. | ✅ FIXED |
| M1 | compile.py discarded `re.sub` result | `canonical_body` extracted at line 432, assigned, and used in merge. | ✅ FIXED |
| M2 | store.py thread-safety design weakness | `_LockedConnection.__getattr__` still proxies `cursor()` without lock. Pragmatic for single-user; not a live bug. | ⚠️ ACCEPTED |
| M3 | qmd_mcp `_get_qmd_client` no health check | Line 277 now performs health check and returns `None` on failure. | ✅ FIXED |
| M4 | qmd_mcp missing explicit encoding | `health()` at line 124 uses `.decode("utf-8")`. | ✅ FIXED |
| M5 | serial QMD queries | `run_qmd_concept_search` at line 95-155 uses `ThreadPoolExecutor`. | ✅ FIXED |
| L1 | llm_client silent 404 | Now `log.warning` with actionable guidance. | ✅ FIXED |
| L2 | edge cache symmetric edges | Still present conceptually; compile.py cleans up. Minor. | ⚠️ ACCEPTED |
| L3 | redundant lex fallback in qmd.py | Removed. `client.query()` handles vec→lex internally. | ✅ FIXED |
| L4 | version mismatch | `pyproject.toml:7` says `"0.3.0"`. `_version.py` reads `"0.3.0"`. `qmd_mcp.py:108` uses `__version__`. `CHANGELOG.md` says `0.3.0`. All aligned. | ✅ FIXED |
| L5 | `QMD_COLLECTION` hardcoded | Now uses `cfg.qmd_collection or "concepts"` at lines 128, 131. | ✅ FIXED |
| L6 | Camoufox SPA wait strategy | `extractors/web.py:302` now uses `await page.wait_for_load_state("networkidle", timeout=5000)`. | ✅ FIXED |

---

## New Findings

### MEDIUM

#### N1. `pipeline/compile/semantic.py:65-71` — `embed_all` dict key collision on duplicate content

```python
texts = [f"{n['title']}\n{n['preview']}" for n in self.notes.values()]
names = list(self.notes.keys())
batch = client.embed_batch(texts)
if batch:
    for name, text in zip(names, texts):
        if text in batch:
            self.embeddings[name] = batch[text]
```

If two notes have identical `title` + `preview` (e.g., two stubs created from the same source template), they share the same dict key in `batch`. One embedding overwrites the other. The second note gets the wrong vector, causing `_semantic_crosslink` and `_semantic_concept_merge` to compute incorrect similarity scores.

**Fix:** Pass a list of `(name, text)` tuples to `embed_batch` and return `dict[name, embedding]`, keyed by note identifier rather than content.

#### N2. `pipeline/compile/core.py:247-287` — `_run_agent` subprocess fallback still uses 600s timeout

The direct semantic compile path is now default, but the Hermes subprocess fallback (`_run_agent`) retains `timeout=600`. If the direct LLM path fails and the fallback is triggered, the compile pass can still hang for 10 minutes. The circuit breaker (`reset_seconds=120`) mitigates repeated failures but does not limit the duration of a single in-flight subprocess.

**Fix:** Reduce fallback timeout to `120` (matching the circuit breaker window) or remove the subprocess fallback entirely once direct compile proves stable in production.

#### N3. `pipeline/compile/semantic.py:210-223` — `_semantic_concept_merge` O(N²) over concepts

```python
names = list(concepts.keys())
for i, name_a in enumerate(names):
    for name_b in names[i + 1:]:
        ...
        sim = index.similarity(name_a, name_b)
```

This double loop computes pairwise similarity for all concept pairs. At 500 concepts, that's ~125k iterations, each doing regex + set intersection + embedding dot product. With 1k concepts, it's ~500k iterations.

**Fix:** Build an approximate nearest-neighbors index (e.g., FAISS, Annoy, or even a simple k-d tree on embeddings) to reduce candidate pairs from O(N²) to O(N log N) or O(N).

#### N4. `pipeline/compile/semantic.py:62-64` — QMD health check overhead on every `embed_all`

```python
from pipeline.qmd import _get_client
if _get_client() is not None:
    ...
```

`_get_client()` instantiates a new `QMDMCPClient` and calls `.health()` every time `embed_all` is invoked. For a compile pass with 200 notes, this adds an unnecessary HTTP roundtrip before skipping local embedding.

**Fix:** Cache the QMD availability check at the start of the compile pass, or pass the client instance into `embed_all` instead of re-checking.

#### N5. `_semantic_concept_merge` top-10 candidates hardcoded

`candidates = candidates[:10]` limits merge review to the 10 highest-scoring pairs per compile pass. For a vault with 200 concepts and many near-duplicates, some merges are deferred indefinitely. The merge queue (`store.merge_queue_add`) mitigates this, but the hard cap is arbitrary.

**Fix:** Make the candidate limit configurable (`cfg.max_merge_candidates`) with a sensible default (e.g., 20).

### LOW

#### N6. `pipeline/create/agent.py:60-62` — Prompt temp file not cleaned up on crash

```python
prompt_path.write_text(prompt)
result = subprocess.run(...)
```

If `subprocess.run` raises (e.g., `FileNotFoundError`, `TimeoutExpired`), `prompt_path` is never deleted. Repeated crashes accumulate debug files in the extract dir.

**Fix:** Wrap in `try/finally` and `prompt_path.unlink(missing_ok=True)`.

#### N7. `pipeline/utils.py:188` — `rstrip` on URLs strips legitimate path characters

```python
url = m.group(0).rstrip(".,;:!?)\">'") if m else ""
```

If a URL path ends with `)` (e.g., `https://example.com/wiki/Article_(disambiguation)`), the `)` is stripped, producing a broken URL. This is a known pattern from prior audits (L6 in Apr-24) but was not fixed.

**Fix:** Count open/close parentheses and only strip trailing `)` if the count is unbalanced.

#### N8. `pipeline/_common.py:24-64` — `check_dependencies` is dead code

Defines `check_dependencies()` but `cli.py` defines its own and never imports this one. Maintenance hazard.

**Fix:** Delete dead code or refactor `cli.py` to use it.

#### N9. `pipeline/config.py:260-262` — Excessive blank lines (E303)

Three blank lines between `validate()` and `_int_env`. Cosmetic only.

#### N10. `CHANGELOG.md` — Direct semantic compile not highlighted as headline change

The most impactful architectural improvement (removing the 600s bottleneck) is listed as a bullet point under "Module decomposition" rather than as a top-level headline. This under-communicates the improvement to users.

---

## Static Analysis Summary

| Tool | Result |
|------|--------|
| **ruff** | ✅ All checks passed |
| **flake8** | 33 cosmetic issues (E127/E128/E251/E302/E303/E203/E306/W391/W292/W503) |
| **bandit** | 11 MEDIUM (B310 localhost urllib), 2 MEDIUM (B314 XML parsing), 2 LOW (B608 dynamic SQL). **0 HIGH.** |
| **AST** | 0 mutable defaults, 0 wildcard imports, 0 security patterns (eval/exec/os.system), 0 syntax errors. 1 import-time side effect (`logging.basicConfig` inside `_setup_logging` function — acceptable). |

---

## Security Assessment

| Class | Assessment |
|-------|------------|
| **SSRF** | URL scheme whitelist in `extractors/_shared.py` (`http/https`). No user-controlled URLs passed to `urllib` without validation. Browser route guard validates requests in `extractors/web.py:292-297`. |
| **Injection** | All subprocess calls use list args. No `shell=True`. No `eval`/`exec`. SQL uses `?` placeholders. Bandit B608 on `store.py` is a false positive (column names are hardcoded). |
| **XML/XXE** | `defusedxml` used when available (podcast.py:604). Fallback strips `<!DOCTYPE` and `<!ENTITY`. Acceptable. |
| **File traversal** | `resolved.is_relative_to(cfg.vault_path)` guard in `enrich` command. Archive path validated. |
| **Secrets** | No hardcoded credentials. `_SECRET_FIELDS` redaction in `doctor.py`. `.env.example` masks API keys. |
| **MD5** | All 9 `hashlib.md5()` calls use `usedforsecurity=False`. |

---

## Test Coverage Analysis

| Metric | Value |
|--------|-------|
| Total tests | 820 |
| Passing | 820 (100%) |
| Test files | 36 |
| Test LOC | ~11,300 |
| Test/source ratio | 0.72x |
| Coverage tool | pytest-cov configured |

### Coverage Gaps
1. **`embed_all` key collision**: No test for two notes with identical title+preview.
2. **SQLite concurrent stress**: No multi-threaded write stress test.
3. **Camoufox/browser extraction**: Expected gap (requires Playwright).
4. **Compile agent subprocess fallback**: Integration tests mock agent calls; no live agent test.
5. **`_semantic_concept_merge` O(N²) boundary**: No test for >100 concepts.
6. **QMD health check failure path**: `test_compile_semantic` mocks QMD as available; skip-to-heuristic path untested.
7. **URL parenthesis stripping**: No test for URLs ending in `)`.

---

## What Would Push This to 9.0+

| Gap | Fix | Impact |
|-----|-----|--------|
| `embed_all` key collision | Key by note name, not content | Eliminates embedding misattribution |
| O(N²) concept similarity | Approximate nearest-neighbors (FAISS/Annoy) | Scales to 1k+ concepts |
| VaultSnapshot full scan | Lazy mtime caching, only scan changed dirs | Faster incremental compile |
| 600s subprocess fallback | Reduce timeout or remove entirely | Removes final performance ceiling |
| Embed cache persistence | SQLite/HDF5 cache for unchanged notes | 80% faster compile on large vaults |
| Top-10 merge cap | Configurable limit + paginated review | No deferred merges |

---

## Honest Opinions

1. **This is genuinely good code.** Not "good for a side project" — good code, period. The architecture has improved measurably across every audit cycle.

2. **The direct semantic compile pass is the right call.** Replacing the 600s subprocess with Python-driven embedding + LLM validation removes the single biggest failure mode. The Hermes fallback is a sensible safety net for now but should eventually be removed.

3. **Version SSoT (`_version.py` → `pyproject.toml`) was implemented correctly.** It works. `import pipeline._version` returns `"0.3.0"` as expected. This is no longer a concern.

4. **The `_semantic_concept_merge` O(N²) scan is the next scalability ceiling.** At 200 concepts it's fine. At 1,000 it won't be. Adding FAISS or even a simple embedding threshold pre-filter would solve this.

5. **820 passing tests is a lot, but the 0.72x ratio means there's room for more edge-case coverage.** I'm particularly concerned about the `embed_all` key collision — it's a silent correctness bug that could corrupt cross-linking in large vaults with many stubs.

6. **The QMD MCP integration is architecturally sound.** Health checks, session reuse, case-insensitive headers, ThreadPoolExecutor batching — all the Apr-24 HIGH bugs were fixed properly.

7. **Security posture is appropriate for a single-user PKM tool.** The SSRF vectors are mitigated, XML parsing is hardened, and there are no shell injections. The remaining B310 flags are localhost false positives.

8. **Documentation drift is minor but real.** The README correctly describes QMD MCP as the default now (line 225), but the CHANGELOG buries the compile-pass improvement. A user skimming the release notes might miss the biggest change.

9. **The circuit breaker is a nice touch.** It prevents runaway agent calls on a misconfigured Hermes install. I'd like to see the same pattern applied to the direct LLM calls (e.g., automatic fallback to OpenRouter if Ollama is down).

10. **If I were using this daily, I'd be comfortable.** The bugs remaining are performance ceilings or edge cases, not data-loss risks. That's the hallmark of a codebase that's been through multiple review rounds with a user who actually fixes things.

---

## 10 Feature Recommendations

1. **Obsidian Mobile Sync Bridge** — A companion daemon that watches `store.db` and pushes new notes to a git-backed mobile vault (via Working Copy or iSH). Enables seamless phone/tablet access.

2. **Incremental Embed Cache** — Persist `NoteIndex.embeddings` to a local SQLite or HDF5 file so compile passes don't re-embed unchanged notes. Cuts compile time by 80% on large vaults.

3. **Web Dashboard (Streamlit/Gradio)** — A visual UI for pending reviews, merge queue, lint health scores, and recent ingestion history. Makes the pipeline accessible to non-CLI users.

4. **Plugin System for Custom Extractors** — A `pipeline/extractors/plugins/` directory where users drop Python files implementing a `BaseExtractor` protocol. Auto-discovered at runtime via `importlib`. Eliminates hard forks for new source types.

5. **Scheduled Ingestion (Cron Mode)** — Native `pipeline schedule` command that writes a system crontab or systemd timer for daily/weekly ingestion. Surfaces next-run info in `pipeline doctor`.

6. **Multi-Vault Support** — Allow `cfg.vault_path` to be a list. The compile pass and query command operate across multiple vaults with a unified index and cross-vault search.

7. **Semantic Diff for Enrichment** — When `pipeline enrich` detects a changed source, show a side-by-side diff of old vs new insights before writing. Prevents blind overwrites of human-edited notes.

8. **Named Query Presets** — Save query templates as `.preset` files (e.g., "recent crypto", "unread sources"). Users run `pipeline query --preset crypto` instead of retyping LLM prompts.

9. **Auto-Archive of Orphaned Sources** — After N days with zero backlinks and no tags matching current interests, move the source note to a `99-Archive` folder. Keeps the vault lean without manual curation.

10. **Export to Static Site (MkDocs/Zola)** — A `pipeline export --site` command that converts the vault into a navigable static website with graph visualization. Turns the PKM into a public knowledge base or portfolio.

---

*End of report.*
