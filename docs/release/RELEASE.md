# Release process

This project treats installability as part of correctness. A release is not ready because tests pass from a source checkout; it is ready when a built wheel works in a fresh virtualenv and can initialize/diagnose/migrate a vault.

## Pre-release checklist

1. Update [`CHANGELOG.md`](CHANGELOG.md) and [`PATCH_NOTES.md`](PATCH_NOTES.md).
2. Confirm `pyproject.toml` version matches the intended tag.
3. Confirm root docs are aligned:
   - `README.md`
   - `docs/README.md`
   - `docs/architecture/ARCHITECTURE.md`
   - `docs/product/PRD.md`
   - `skills/obsidian-ingest.md`
4. Run source verification:

```bash
ruff check .
pyflakes pipeline tests
pytest -q
git diff --check
```

5. Run installed-wheel verification:

```bash
rm -rf /tmp/obsidian-llm-wiki-wheel /tmp/obsidian-llm-wiki-venv /tmp/obsidian-llm-wiki-vault
python3 -m pip wheel . -w /tmp/obsidian-llm-wiki-wheel --no-deps
python3 -m venv /tmp/obsidian-llm-wiki-venv
/tmp/obsidian-llm-wiki-venv/bin/pip install /tmp/obsidian-llm-wiki-wheel/*.whl
/tmp/obsidian-llm-wiki-venv/bin/pipeline init /tmp/obsidian-llm-wiki-vault
find /tmp/obsidian-llm-wiki-vault/Meta/prompts -type f | wc -l
find /tmp/obsidian-llm-wiki-vault/Meta/Templates -type f | wc -l
/tmp/obsidian-llm-wiki-venv/bin/pipeline doctor /tmp/obsidian-llm-wiki-vault --json
/tmp/obsidian-llm-wiki-venv/bin/pipeline fixture /tmp/obsidian-llm-wiki-vault --adversarial --overwrite --json
/tmp/obsidian-llm-wiki-venv/bin/pipeline graph-doctor /tmp/obsidian-llm-wiki-vault --json
/tmp/obsidian-llm-wiki-venv/bin/pipeline migrate /tmp/obsidian-llm-wiki-vault --yes --json
/tmp/obsidian-llm-wiki-venv/bin/pipeline release-check --json
```

Expected seeded asset counts after `init`:

- prompts: `8`
- templates: `9`

6. Push and wait for GitHub Actions CI to pass.
7. Tag with `vX.Y.Z` only after CI is green.
8. Build/publish from the tag, not a dirty working tree.

## Versioning rule

- Patch: bug fixes, security hardening, docs alignment, packaging fixes, test/CI improvements.
- Minor: new CLI features or schema-compatible pipeline capabilities.
- Major: vault schema migrations that require manual user action.

## Dirty tree rule

Do not tag or publish if `git status --short` is non-empty. Documentation-only changes still need the same gate; stale docs are how users rerun yesterday’s bugs with confidence.
