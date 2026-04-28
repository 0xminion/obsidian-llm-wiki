# Patch notes

## 2026-04-28 — Safety, graph diagnostics, migrations, and golden corpus

**Branch:** `main`
**Review model:** primary implementation + independent second-review blocker pass
**Commit:** `6b700b04e17c4f0b8698e985fbd046641a14ec20`

### Summary

This patch turns the repository from a capable local pipeline into a more credible unattended tool. The work focused on filesystem safety, staged review integrity, SSRF/secret handling, graph correctness, semantic truthfulness, migration readiness, and installed-wheel verification.

### Fixed blockers

1. **Filename/path breakout**
   - Added strict safe note stem/path helpers.
   - CJK titles, LLM filename suggestions, MoC names, and generated paths are treated as untrusted input.
   - Writes enforce resolved-path containment.

2. **Review approval arbitrary writes**
   - Review rows are mapped to allowed vault collections.
   - Absolute/out-of-vault destinations are rejected.
   - Approval validates a full plan before writes.

3. **Review partial failure corruption**
   - Temp files are written before final replace.
   - If a later replace fails, earlier replacements are rolled back and rows are not marked approved.

4. **Network SSRF / DNS rebinding**
   - URL validation rejects non-public targets and weird IP encodings.
   - curl DNS pinning fails closed when no safe public pin exists.
   - YouTube URLs are host-validated and canonicalized before downstream tools.

5. **Secret leaks through argv**
   - Secret-bearing curl headers, including AssemblyAI authorization, are passed via stdin config, not process argv.

6. **Semantic false success**
   - Empty/degraded LLM semantic results now surface through `semantic_status` and `semantic_degraded_reason`.
   - Non-fast query mode treats empty stdout as failure and does not archive the query.

7. **Graph/cache correctness**
   - Edge cache invalidates after direct rewrites.
   - Source notes are first-class graph nodes.
   - Duplicate reports are rewritten when clean.

8. **QMD semantic correctness**
   - `USE_QMD_MCP=false` disables QMD client construction.
   - QMD result path conversion handles vault-relative paths.
   - Compile consumes QMD embeddings when available.

### Added features

- `pipeline graph-doctor [VAULT] --json`
- `pipeline migrate [VAULT] --yes --json`
- `pipeline fixture [VAULT] --adversarial --overwrite --json`
- semantic candidate blocking before expensive pair scoring
- adversarial regression suite: `tests/test_audit_hardening_2026_04_28.py`

### Verification

```bash
ruff check .
pyflakes pipeline tests
pytest -q                    # 852 passed
git diff --check
```

Installed-wheel smoke:

```bash
python3 -m pip wheel . -w /tmp/obsidian-llm-wiki-wheel --no-deps
python3 -m venv /tmp/obsidian-llm-wiki-venv
/tmp/obsidian-llm-wiki-venv/bin/pip install /tmp/obsidian-llm-wiki-wheel/*.whl
/tmp/obsidian-llm-wiki-venv/bin/pipeline init /tmp/obsidian-llm-wiki-vault
/tmp/obsidian-llm-wiki-venv/bin/pipeline fixture /tmp/obsidian-llm-wiki-vault --adversarial --overwrite --json
/tmp/obsidian-llm-wiki-venv/bin/pipeline graph-doctor /tmp/obsidian-llm-wiki-vault --json
/tmp/obsidian-llm-wiki-venv/bin/pipeline migrate /tmp/obsidian-llm-wiki-vault --yes --json
```

Expected packaged assets after `init`:

- prompts: `8`
- templates: `9`

## 2026-04-23 — Code review bug fixes

Historical patch notes for the first broad review pass were superseded by the 2026-04-28 hardening pass. See `docs/audits/` and `docs/reviews/` for the original detailed finding lists.
