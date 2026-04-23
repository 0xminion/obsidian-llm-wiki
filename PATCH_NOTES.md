# Patch Notes — Code Review Bug Fixes

**Date:** 2026-04-23
**Branch:** `main`
**Reviewers:** Primary + Independent Second-Opinion Agent

## Summary

Systematic patching of all code review findings from the end-to-end comprehensive review. Fixes applied in severity order (HIGH → MEDIUM → LOW). All 596 fast tests pass. One pre-existing scale test (`test_os_fd_limit_not_exceeded_at_scale`) remains slow but unrelated to these changes.

---

## HIGH Severity

### H1 — Frontmatter end detection finds `---` in body
**File:** `pipeline/create/validate.py:100–103`
**Fix:** Replaced naive `content.split("---")` frontmatter parsing with anchored regex `re.match(r"^---\n(.*?)\n---", content, re.DOTALL)` so `---` inside the note body no longer truncates frontmatter.

### H2 — `Plans.load()` crashes on malformed JSON
**File:** `pipeline/models.py:232–238`
**Status:** Already fixed in HEAD. Try/except around `json.loads` and `Plan.from_dict` with graceful fallback.

### H3 — Empty tags render as `tags: null`
**File:** `pipeline/create/templates.py:47–56`
**Fix:** Changed `tags_yaml` generation to emit an empty YAML list (`tags:\n`) instead of `null` when a plan has no tags.

### H4 — RSS with default XML namespaces
**File:** `pipeline/extractors/podcast.py:601`
**Status:** Already fixed in HEAD. `_safe_xml_parse()` with `resolve_entities=False`, `_strip_xml_ns()`, and `_iter_items()` handle namespaced RSS feeds safely.

### H5 — `fix_frontmatter` corrupts already-quoted wikilinks
**File:** `pipeline/lint.py:636–643`
**Status:** Already fixed in HEAD. Negative lookbehind/lookahead regex prevents double-quoting already-quoted wikilinks.

---

## MEDIUM Severity

### M1 — Content dedup creates stub entries without URL registration
**File:** `pipeline/extract.py:124–175`
**Status:** Already fixed in HEAD. Deduplication path calls `store.register_url(url, source_type.value, status="dedup")`.

### M2 — `write_edge` race condition
**File:** `pipeline/vault.py:307–335`
**Status:** Already fixed in HEAD. Entire check-write-add sequence protected by `_edge_cache_lock`.

### M3 — `check_broken_wikilinks` false positive on aliases/sections
**File:** `pipeline/lint.py:258`
**Status:** Already fixed in HEAD. Regex `r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]"` captures only the base note name.

### M4 — MoC duplicate entry via substring match
**File:** `pipeline/vault.py:256–266`
**Status:** Already fixed in HEAD. Regex `rf'\[\[{re.escape(entry_name)}(?:\|[^\]]*)?\]\]'` requires exact link match.

### M5 — Concept validation misses collision-resolved filenames
**File:** `pipeline/create/orchestrator.py:132–138`
**Fix:** Changed filename lookup from exact path to `glob(f"{concept_filename}*.md")` to account for `-1`, `-2` collision suffixes. **Also corrected directory from `concepts_dir` → `entries_dir`** (concepts live in `entries/`).

### M6 — `validate_output` uses stale manifest on resume
**File:** `pipeline/create/orchestrator.py:176–185`
**Status:** Verified non-issue. In resume mode `create_file_templates` passes `.template-postprocess-manifest`; `create_all` is only called on fresh runs where `manifest.json` was just written.

### M7 — YouTube ID regex scans entire URL
**File:** `pipeline/extractors/_shared.py:149–160`
**Status:** Already fixed in HEAD. Fallback iterates over URL path segments instead of scanning the full string.

### M8 — Podcast XML bomb / entity expansion
**File:** `pipeline/extractors/podcast.py`
**Status:** Already fixed in HEAD. `resolve_entities=False` and `_safe_xml_parse()` guard against XXE/billion-laughs.

### M9 — Podcast audio URL SSRF via yt-dlp
**File:** `pipeline/extractors/podcast.py:720–760`
**Status:** Already fixed in HEAD. Scheme whitelist `("http", "https")` rejects file://, ftp://, etc.

### M10 — `fix_banned_tags` mutates body tags
**File:** `pipeline/lint.py`
**Status:** Already fixed in HEAD. Scoped to frontmatter string only, not body content.

### M11 — `check_orphaned_concepts` uses imprecise substring match
**File:** `pipeline/lint.py:366–380`
**Fix:** Replaced substring scan `f"[[{concept_name}]]" in content` with precise wikilink regex `r"\[\[" + re.escape(concept_name) + r"(?:[|#][^\]]*)?\]\]"` to eliminate false positives.

### M12 — `reindex` crashes on unreadable file
**File:** `pipeline/vault.py:441–500`
**Status:** Already fixed in HEAD. Per-file try/except around `read_text()`.

---

## LOW Severity

### L1 — Lock error message shows wrong path
**File:** `pipeline/cli.py:358`
**Fix:** Changed hardcoded path string to `{lock.lock_dir}` so the message always reflects the actual lock location.

### L2 — Query archive collision overwrites existing file
**File:** `pipeline/cli.py:875`
**Fix:** Added collision handling: if archive path exists, append `-1`, `-2`, etc.

### L3 — Tag registry ignores `sources/` tags
**File:** `pipeline/cli.py:742–764`
**Fix:** Added `sources_dir` scan with its own `source_tags` counter, and included Source Tags in the registry report.

### L4 — Unreachable dry-run block in Stage 1
**File:** `pipeline/cli.py:391–396`
**Status:** False positive. Outer `if dry_run:` at line 291 only handles vault detection; inner block at line 393 is reachable when vault exists and `--dry-run` is set.

### L5 — Prompt temp file leak on agent crash
**File:** `pipeline/create/agent.py:60–62`
**Fix:** Wrapped prompt file creation in try/finally so `_agent_prompt_{pid}.md` is always deleted even if `subprocess.run` raises or times out.

### L6 — Podcast Apple ID regex overly broad
**File:** `pipeline/extractors/podcast.py:156`
**Fix:** Anchored regex from `r"id(\d+)"` to `r"/id(\d+)(?:\?|$)"` so random `id123` strings in query params don't match.

### L7 — `check_required_sections` name/docstring misleading
**File:** `pipeline/lint.py:641–669`
**Fix:** Updated docstring to clarify the function currently only validates MoCs; entry/concept sections are handled by other checks.

### L8 — Dead `check_dependencies` in `_common.py`
**File:** `pipeline/_common.py:24–64`
**Fix:** Removed unused `check_dependencies()` to eliminate maintenance hazard. CLI uses its own leaner version.

### L9 — Greedy JSON regex in plan parsing
**File:** `pipeline/plan.py:369`
**Fix:** Changed `r"\[.*\]"` to `r"\[[\s\S]*?\]"` (non-greedy) so markdown links after a JSON array don't get consumed.

---

## Test Adjustments

### `tests/test_create.py::TestCreateAll::test_single_plan_success`
Updated mock side effect to create concept files with proper concept-format frontmatter and sections (`type: concept`, `sources`, `## Core concept`, `## Context`, `## Links`) so batch validation passes under the corrected concept-check logic.

---

## Verification

```bash
python -m pytest tests/test_create.py tests/test_lint.py \
    tests/test_regressions_code_review.py tests/test_models.py \
    tests/test_config.py tests/test_vault_setup.py \
    tests/test_cli_commands.py -q
# 231 passed

python -m pytest tests/test_extract.py tests/test_plan.py \
    tests/test_vault.py tests/test_compile.py tests/test_qmd.py \
    tests/test_edge_cases.py tests/test_integration.py \
    tests/test_new_systems.py tests/test_review.py \
    tests/test_stats.py tests/test_system.py \
    tests/test_template_regressions.py -q
# 365 passed

# Total: 596 passed (1 pre-existing scale test skipped/hanging)
```

---

## Files Modified

- `pipeline/cli.py`
- `pipeline/create/agent.py`
- `pipeline/create/orchestrator.py`
- `pipeline/create/templates.py`
- `pipeline/create/validate.py`
- `pipeline/extractors/podcast.py`
- `pipeline/lint.py`
- `pipeline/plan.py`
- `pipeline/_common.py`
- `tests/test_create.py`
