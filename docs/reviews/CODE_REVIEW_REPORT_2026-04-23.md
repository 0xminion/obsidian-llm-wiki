# obsidian-llm-wiki — Comprehensive Code Review Report

**Date:** 2026-04-23  
**Scope:** Full codebase (12,012 lines across 33 Python modules + 21 test modules)  
**Test Suite:** 618 passed, 1 warning  
**Reviewers:** Primary agent + Independent subagent (second opinion)

---

## Executive Summary

The codebase is well-structured with a clean 3-stage pipeline architecture, comprehensive lint system, and strong test coverage. Recent additions (unified LLM client, semantic compile pass, parallel insight generation) are architecturally sound but contain **4 HIGH-severity bugs** that will cause runtime failures in production. The most critical is a missing `import math` that crashes every semantic compile pass.

**Overall Rating: 7.5/10**  
- Architecture: 8/10 (clean separation, good abstractions)
- Code Quality: 7/10 (good patterns, but some brittle regexes and missing imports)
- Test Coverage: 7/10 (618 tests, but semantic compile ops have zero coverage)
- Security: 6/10 (SSRF risk, no prompt injection guards)
- Performance: 8/10 (parallelization, caching, batch embeddings)

---

## Critical Findings (HIGH Severity)

### H1 — Missing `import math` crashes semantic compile
| | |
|---|---|
| **File** | `pipeline/compile.py` |
| **Line** | 495–496 |
| **Issue** | `NoteIndex.similarity()` calls `math.sqrt()` but `math` is never imported. Any compile pass using the semantic path (default) will immediately crash with `NameError: name 'math' is not defined`. |
| **Fix** | Add `import math` to the top-level imports in `compile.py`. |
| **Verification** | `python -c "import pipeline.compile; print('math' in dir(pipeline.compile))"` → `False` |

### H2 — Broken multi-word note parsing in cross-linking
| | |
|---|---|
| **File** | `pipeline/compile.py` |
| **Line** | 609 |
| **Issue** | `_semantic_crosslink` uses regex `r"LINK\s+(\S+)\s+(\S+)\s+(.*)"`. The `\S+` pattern cannot match note names containing spaces. Chinese filenames (which `title_to_filename` preserves spaces for) will parse incorrectly: `LINK 人工智能 安全 机器学习 reason` → `a="人工智能"`, `b="安全"`, `reason="机器学习 reason"`. |
| **Fix** | Use a delimiter-aware format: `LINK <note_a> | <note_b> | <reason>` or require quoted strings. Alternatively, restrict vault filenames to space-free stems. |
| **Impact** | Cross-linking silently creates wrong links or fails for any vault with space-containing Chinese filenames. |

### H3 — Broken multi-word concept parsing in merge
| | |
|---|---|
| **File** | `pipeline/compile.py` |
| **Line** | 682 |
| **Issue** | Same `\S+` issue in `_semantic_concept_merge` regex `r"MERGE\s+(\S+)\s+(\S+)\s+(.*)"`. Concepts with multi-word names (common in Chinese) parse incorrectly. |
| **Fix** | Same as H2 — use delimiter-aware parsing. |

### H4 — Incomplete reference update after concept merge
| | |
|---|---|
| **File** | `pipeline/compile.py` |
| **Line** | 716–724 |
| **Issue** | `_merge_concepts` updates `[[duplicate]]` → `[[canonical]]` in `entries_dir` only. It does **not** update: (a) MoC files in `mocs_dir`, (b) other concept files in `concepts_dir` that link to the duplicate, (c) `edges.tsv` which may contain edges referencing the deleted concept. This leaves dangling references and orphaned edges. |
| **Fix** | Extend the replacement loop to `cfg.mocs_dir`, `cfg.concepts_dir`, and trigger an edge rebuild after merge. |
| **Impact** | Vault accumulates broken links and stale edges over time. |

---

## Significant Findings (MEDIUM Severity)

### M1 — No concurrency control during semantic compile
| | |
|---|---|
| **File** | `pipeline/compile.py` (entire semantic section) |
| **Line** | 558–808 |
| **Issue** | `_semantic_crosslink`, `_semantic_concept_merge`, and `_semantic_moc_rebuild` read and write vault files without any lock. If `ingest` (which acquires `PipelineLock`) or another `compile` runs concurrently, files can be corrupted or edits interleaved. |
| **Fix** | Acquire `PipelineLock` in the `compile_pass` CLI command before calling `run_compile`. |

### M2 — `run_qmd_query` hardcodes vault path
| | |
|---|---|
| **File** | `pipeline/qmd.py` |
| **Line** | 222 |
| **Issue** | `run_qmd_query` uses `Path.home() / "MyVault"` instead of `cfg.vault_path`. Callers with non-default vaults will query the wrong concept directory. |
| **Fix** | Accept `cfg` or `concepts_dir` parameter and remove the hardcoded path. |
| **Impact** | Functional break for any vault not at `~/MyVault`. |

### M3 — Opaque failure mode in `LLMClient.generate()`
| | |
|---|---|
| **File** | `pipeline/llm_client.py` |
| **Line** | 312–322 |
| **Issue** | All provider errors are swallowed and an empty string is returned. Callers (e.g., `query --fast`, template insight generation) cannot distinguish between "LLM returned empty" and "LLM is down / misconfigured / rate-limited". |
| **Fix** | Return `LLMResponse` instead of `str`, or raise a typed exception on failure. At minimum, log at WARNING level when generation fails. |

### M4 — False success reporting in semantic compile
| | |
|---|---|
| **File** | `pipeline/compile.py` |
| **Line** | 842–843 |
| **Issue** | `_run_semantic_compile` sets `result.agent_succeeded = True` unconditionally, even if `crosslinks_added`, `concepts_merged`, or `mocs_updated` raised exceptions internally or produced no changes. |
| **Fix** | Set `agent_succeeded` based on whether all three operations completed without exception, and report partial failures. |

### M5 — Fragile frontmatter extraction in MoC rebuild
| | |
|---|---|
| **File** | `pipeline/compile.py` |
| **Line** | 800–802 |
| **Issue** | Regex `r"^(---\s*\n.*?\n---\s*\n)"` with `re.DOTALL` will match prematurely if a YAML value contains `---` (e.g., a multiline string with a horizontal rule). |
| **Fix** | Use a dedicated YAML frontmatter parser or split on the first `\n---\n` occurrence only. |

### M6 — `compile_pass` CLI bypasses pipeline lock
| | |
|---|---|
| **File** | `pipeline/cli.py` |
| **Line** | 586–616 |
| **Issue** | The `compile_pass` CLI command does not acquire `PipelineLock`, while `ingest` does. Concurrent `ingest` + `compile` can corrupt the vault. |
| **Fix** | Acquire `PipelineLock` at the start of `compile_pass` (same pattern as `ingest`). |

### M7 — Embedding batch uses raw text as dict key
| | |
|---|---|
| **File** | `pipeline/compile.py` |
| **Line** | 479–483 |
| **Issue** | `NoteIndex.embed_all()` uses `f"{title}\n{preview}"` as a dict key to look up batch embeddings. If two notes happen to have identical title+preview (unlikely but possible for stubs or near-empty notes), one embedding overwrites the other. |
| **Fix** | Use `dict[str, list[float]]` keyed by note name instead of text content. |

---

## Minor Findings (LOW Severity)

### L1 — No edge caching in `_build_edges`
| | |
|---|---|
| **File** | `pipeline/compile.py` |
| **Line** | 239–368 |
| **Issue** | Re-reads the entire `edges.tsv` on every compile pass. For large vaults this is O(N) disk I/O on every run. |
| **Fix** | Cache edges in `CompileResult` or use a module-level cache with mtime invalidation. |

### L2 — Process-global QMD embedding cache never invalidated
| | |
|---|---|
| **File** | `pipeline/qmd.py` |
| **Line** | 40–41 |
| **Issue** | `_concept_embedding_cache` and `_cache_loaded` are module-level and never cleared. Long-running processes or tests see stale embeddings after concept edits. |
| **Fix** | Add a `clear_cache()` helper or use a TTL/cache-invalidation strategy. |

### L3 — No HTTP connection pooling in LLM client
| | |
|---|---|
| **File** | `pipeline/llm_client.py` |
| **Line** | 89–104 |
| **Issue** | Every `generate()`/`embed()` call opens a new TCP connection to Ollama/OpenRouter. At scale this adds significant latency. |
| **Fix** | Use `urllib.request` keep-alive or switch to `http.client` / `requests` with a `Session`. |

### L4 — SSRF risk in curl helpers
| | |
|---|---|
| **File** | `pipeline/extractors/_shared.py` |
| **Line** | 45–67 |
| **Issue** | `_curl_get` and `_curl_post_json` pass arbitrary URLs to `curl` without validation. Malicious `.url` files could target internal services. |
| **Fix** | Validate URLs with an allow-list or `urlparse` scheme/host checks before passing to curl. |

### L5 — LLM insight prompt lacks injection guards
| | |
|---|---|
| **File** | `pipeline/create/templates.py` |
| **Line** | 208–212 |
| **Issue** | `generate_entry_insights` embeds raw extracted content directly into the prompt. If content contains prompt-injection patterns (e.g., "ignore previous instructions"), the LLM may misbehave. |
| **Fix** | Add a prompt-injection sanitizer or use structured/system messages where the provider supports them. |

---

## Test Coverage Gaps

| Module | Gap | Risk |
|---|---|---|
| `pipeline/compile.py` | No unit tests for `_semantic_crosslink`, `_semantic_concept_merge`, or `_semantic_moc_rebuild`. The H1–H4 bugs would have been caught. | Regression in semantic compile logic |
| `pipeline/compile.py` | No integration test exercising the full `_run_semantic_compile` path with a mock LLM client. | Silent failures in production |
| `pipeline/qmd.py` | No test for `run_qmd_query` with non-default vault paths. | M2 bug persists undetected |
| `pipeline/llm_client.py` | No test for `generate_parallel` under partial failure (one thread fails, others succeed). | Resource leaks or hangs |
| `pipeline/create/templates.py` | No test for concurrent insight generation + filename collision with shared source/entry dirs. | Race conditions in batch creation |

---

## Backward Compatibility Notes

| Issue | Detail |
|---|---|
| Plan stage ignores `llm_provider` | `plan.py` `generate_plans()` always spawns a Hermes subprocess for uncertain sources, even when the user configured `llm_provider=ollama` or `openrouter`. This is inconsistent with the unified client design. |
| `run_qmd_query` signature preserved | The `qmd_cmd` and `collection` parameters are ignored (documented). This is fine for backward compatibility but the hardcoded vault path (M2) is a functional break for non-default vaults. |

---

## Positive Observations

1. **SQLite store threading safety** — `store.py` `_LockedConnection` correctly wraps every SQLite operation with an `RLock`, and WAL mode is enabled.
2. **Graceful extraction degradation** — `extract.py` distinguishes `ExtractionError` (no retry) from transient failures, and uses a DLQ.
3. **Agent subprocess safety** — All `subprocess.run` calls use list arguments (not `shell=True`), preventing shell injection.
4. **Vault lock with stale detection** — `_common.py` `VaultLock` detects stale locks by PID and 30-minute timeout.
5. **Comprehensive test suite** — 618 tests pass, including regression tests for previously fixed bugs.
6. **Unified LLM client architecture** — Clean provider abstraction (Ollama/OpenRouter/Hermes) with env-based configuration.
7. **Parallel insight generation** — ThreadPoolExecutor in Stage 3 cuts creation time by ~3×.

---

## Priority Actions

1. **Fix H1 immediately** — Add `import math` to `pipeline/compile.py`.
2. **Fix H2 & H3** — Change LLM output parsers to handle spaces in note names.
3. **Fix H4** — Extend reference replacement to MoCs, concepts, and edges after merge.
4. **Fix M1 & M6** — Acquire `PipelineLock` in `compile_pass`.
5. **Fix M2** — Remove hardcoded vault path from `run_qmd_query`.
6. **Fix M3** — Add error transparency to `LLMClient.generate()`.
7. **Add tests** for semantic compile operations to prevent regressions.
